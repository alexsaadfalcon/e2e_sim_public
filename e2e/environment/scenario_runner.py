# -*- coding: utf-8 -*-
"""
Scenario generation runner: declarative ``Scenario`` -> simulation frames (.pkl).

This is the "scenario generation" half of the simulator (see CLAUDE.md): it turns a
fully declarative :class:`e2e.scenario.Scenario` into the precomputed S-parameter
frames the runtime pipeline (``blocks.py`` / ``simulation.py``) consumes via
:class:`e2e.environment.sionna_iterator.SionnaIterator`.

For each frame it:

1. applies per-frame node/object motion (resolved by the pure ``motion`` module),
2. ray-traces the scene with Sionna RT and computes paths,
3. extracts the channel frequency response (S-parameters) over the FrequencyPlan grid,
4. accumulates per-frame arrays and dumps them to
   ``e2e/environment/sionna_sims/<scenario.name>.pkl``.

Links and output layout
------------------------
A scenario can describe several simultaneous tx->rx **links** (see :func:`enumerate_links`):
each RADAR node is a monostatic link, and every COMM_TX -> COMM_RX/MONITOR pair is a
communication link. Each link is exported independently.

Sionna's ``paths.cfr(...)`` returns
``[num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, num_freqs]``; per link we keep
``cfr[0, :, 0, :, :, :]`` -> ``[num_rx_ant, num_tx_ant, n_time, num_freqs]`` and stack
frames on a new leading axis.

The dumped (and returned) object is always a **self-describing v2 payload** -- a dict
with two top-level keys, ``"meta"`` and ``"links"`` -- regardless of whether the scenario
has one link or several::

    {
      "meta": {
        "version": 1,
        "scenario_name": <Scenario.name>,
        "freq_plan": {"carrier_hz": float, "start_hz": float, "stop_hz": float,
                       "num_freqs": int},
        "links": {
          <link_name>: {
            "tx_node": str, "rx_node": str,
            "rx_array_shape": [num_rows, num_cols],
            "n_tx_ant": int,
            "kind": "radar" | "comm",   # "radar" iff the link is monostatic
            "tx_power_dbm": float | None,
            "physical_scale": bool,    # == tx_power_dbm is not None
          },
          ...
        },
      },
      "links": {
        <link_name>: ndarray[num_frames, num_rx_ant, num_tx_ant, n_time, num_freqs],
        ...
      },
    }

A single-link scenario (e.g. ``munich_radar``) still yields this same shape -- just one
entry in each map. Loading (picking a link, dispatching on ``meta`` for physical-scale
handling, etc.) is :class:`~e2e.environment.sionna_iterator.SionnaIterator`'s job.

Because antenna devices are not scatterers, the channel for a given (tx, rx) pair is
independent of which *other* devices are present. We therefore build a **separate Sionna
scene per link** (each with that link's own tx/rx ``PlanarArray`` and all physical
objects). This is what lets heterogeneous links coexist -- e.g. a 32x32 radar RX and a
4x4 comm RX -- which a single scene (one ``rx_array`` for all receivers) could not.

Dry-run / mock mode
-------------------
Sionna RT, DrJit and LLVM are heavy and not available on every dev machine. ``--dry-run``
exercises *all* scheduling, motion, link-enumeration and serialization logic but
synthesizes random complex S-parameters of the correct per-link shape instead of ray
tracing. Sionna is imported lazily, only in the real path, so this module imports and the
dry-run CLI runs with no Sionna installed.

CLI
---
    python -m e2e.environment.scenario_runner --scenario munich_radar --dry-run
    python -m e2e.environment.scenario_runner --scenario munich_isac --dry-run
    python -m e2e.environment.scenario_runner --scenario path/to/scene.json --frames 10 --dry-run
    python -m e2e.environment.scenario_runner --scenario munich_radar          # real (needs Sionna)

The real Sionna path is validated on a CUDA-12.x driver (see the "GPU / driver /
LLVM" section in the README); the dry-run path needs no GPU.
"""

from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from e2e.scenario import (
    Scenario,
    Node,
    NodeRole,
    SceneObject,
    ArrayConfig,
    Polarization,
    REFERENCE_SCENARIOS,
    SYSTEM_IMPEDANCE_OHMS,
)
from e2e.environment import motion as motion_mod

# Speed of light (m/s), used to turn a link's carrier frequency into a wavelength for
# the dry-run free-space-path-loss mock.
_SPEED_OF_LIGHT = 299_792_458.0


# Directory the existing generators / iterator agree on.
_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
SIONNA_SIMS_DIR = os.path.join(_THIS_DIR, "sionna_sims")


# --------------------------------------------------------------------------------
# Frame schedule -- pure, Sionna-free
# --------------------------------------------------------------------------------
@dataclass
class FrameSchedule:
    """Resolved per-frame trajectories for every node and object in a scenario.

    Both dicts map ``name -> (num_frames, 3)`` world-position arrays. This is computed
    entirely from the declarative scenario via the pure ``motion`` module, so it is
    identical between dry-run and real execution -- the real path just feeds these
    positions into Sionna instead of into the synthetic generator.
    """
    num_frames: int
    node_tracks: Dict[str, np.ndarray]
    object_tracks: Dict[str, np.ndarray]

    def node_position(self, name: str, frame_idx: int):
        return self.node_tracks[name][frame_idx]

    def object_position(self, name: str, frame_idx: int):
        return self.object_tracks[name][frame_idx]


def build_schedule(scenario: Scenario) -> FrameSchedule:
    """Resolve all node/object motion into per-frame position tracks.

    Rotational motion uses the scene centroid (all node + object base positions) as the
    pivot, so ``angular_velocity_deg`` produces the circular sweep the existing car code
    creates. Pure translation ignores the pivot.
    """
    n = scenario.num_frames
    base_positions = [n_.position for n_ in scenario.nodes] + [o.position for o in scenario.objects]
    pivot = motion_mod.scene_centroid(base_positions)

    node_tracks: Dict[str, np.ndarray] = {}
    for node in scenario.nodes:
        node_tracks[node.name] = motion_mod.resolve_motion(
            node.position, node.motion, n, pivot=pivot
        )

    object_tracks: Dict[str, np.ndarray] = {}
    for obj in scenario.objects:
        object_tracks[obj.name] = motion_mod.resolve_motion(
            obj.position, obj.motion, n, pivot=pivot
        )

    return FrameSchedule(num_frames=n, node_tracks=node_tracks, object_tracks=object_tracks)


# --------------------------------------------------------------------------------
# Role -> tx/rx mapping and link enumeration
# --------------------------------------------------------------------------------
def _tx_rx_nodes(scenario: Scenario):
    """Decide which nodes act as transmitters vs receivers from their roles.

    * RADAR   -> contributes both a TX and an RX (monostatic, co-located by default).
    * COMM_TX -> TX only.
    * COMM_RX -> RX only.
    * MONITOR -> RX only (passive).

    Returns ``(tx_nodes, rx_nodes)`` lists. Kept for introspection / summaries;
    :func:`enumerate_links` is what drives generation.
    """
    tx_nodes: List[Node] = []
    rx_nodes: List[Node] = []
    for node in scenario.nodes:
        if node.role == NodeRole.RADAR:
            tx_nodes.append(node)
            rx_nodes.append(node)
        elif node.role == NodeRole.COMM_TX:
            tx_nodes.append(node)
        elif node.role in (NodeRole.COMM_RX, NodeRole.MONITOR):
            rx_nodes.append(node)
    return tx_nodes, rx_nodes


def tx_array_config(node: Node) -> ArrayConfig:
    """Effective *transmit* array for a node.

    A node's ``ArrayConfig`` describes its receive aperture. For a monostatic RADAR the
    illuminator is a single element by default (this reproduces the reference setup in
    ``sionna_simple_channel.py``: a 1x1 ``tx_array`` against a 32x32 ``rx_array``, giving
    per-frame S-parameters of shape ``[1024, 1, 1, num_freqs]``). A radar can opt into a
    full transmit aperture via ``params["tx_array"] = "full"``. A COMM_TX transmits with
    its own configured array.
    """
    if node.role == NodeRole.RADAR and node.params.get("tx_array") != "full":
        return ArrayConfig(num_rows=1, num_cols=1,
                           pattern=node.array.pattern, polarization=node.array.polarization)
    return node.array


@dataclass
class LinkSpec:
    """A single tx->rx propagation link to export."""
    name: str
    tx_node: Node
    rx_node: Node
    tx_array: ArrayConfig
    rx_array: ArrayConfig

    @property
    def n_tx_ant(self) -> int:
        return self.tx_array.num_elements

    @property
    def n_rx_ant(self) -> int:
        return self.rx_array.num_elements


def _assert_single_pol(array: ArrayConfig, context: str) -> None:
    """Defensive, Sionna-free guard against the dual-pol port-count mismatch.

    Scenario.validate() already rejects VH/CROSS at the scenario level, so this should
    be unreachable from ScenarioRunner's normal entry point (its __init__ calls
    validate() first). This is a second line of defense for callers that build/mutate a
    Scenario without going through validate() -- raising here keeps the failure at setup
    (and in dry-run, which never touches Sionna) rather than silently generating a frame
    whose antenna axis doesn't match the recorded ArrayConfig.num_elements.
    """
    if array.polarization in (Polarization.VH, Polarization.CROSS):
        raise ValueError(
            f"{context}: dual-polarization ('{array.polarization.value}') would make "
            f"Sionna report 2x the antenna ports ({2 * array.num_elements}) vs "
            f"ArrayConfig.num_elements ({array.num_elements}); use V or H (single-pol)."
        )


def enumerate_links(scenario: Scenario) -> List[LinkSpec]:
    """Enumerate the tx->rx links to export, from node roles.

    * each RADAR -> a monostatic link (tx and rx are the same node), named after the node.
    * each COMM_TX -> each COMM_RX / MONITOR -> a communication link named ``"<tx>__<rx>"``.

    A scenario that yields a single link reproduces the legacy single-array export; a
    scenario that yields several links (e.g. a radar plus a comm link) exports one
    frame-stack per link.
    """
    links: List[LinkSpec] = []
    for r in scenario.nodes_by_role(NodeRole.RADAR):
        links.append(LinkSpec(name=r.name, tx_node=r, rx_node=r,
                              tx_array=tx_array_config(r), rx_array=r.array))
    tx_comm = scenario.nodes_by_role(NodeRole.COMM_TX)
    rx_comm = scenario.nodes_by_role(NodeRole.COMM_RX) + scenario.nodes_by_role(NodeRole.MONITOR)
    for t in tx_comm:
        for r in rx_comm:
            links.append(LinkSpec(name=f"{t.name}__{r.name}", tx_node=t, rx_node=r,
                                  tx_array=tx_array_config(t), rx_array=r.array))
    if not links:
        raise ValueError("scenario defines no tx->rx link (need a RADAR, or a COMM_TX + COMM_RX)")
    return links


# --------------------------------------------------------------------------------
# Physical transmit-power scaling (see e2e.scenario.Node.tx_power_dbm docstring).
#
# ``tx_power_dbm is None`` is the legacy contract: no absolute power scale is applied,
# and the real path keeps Sionna's unit-average-energy normalization. Setting a physical
# power switches both the real and dry-run paths onto an absolute voltage scale so the
# dumped S-parameters correspond to a real received power in watts.
# --------------------------------------------------------------------------------
def _cfr_should_normalize(tx_power_dbm: Optional[float]) -> bool:
    """Whether the real Sionna path should request unit-average-energy normalization.

    ``normalize=True`` is Sionna's "unit average energy" mode (legacy, ``tx_power_dbm``
    unset). ``normalize=False`` is Sionna's own default: the CFR then carries physical
    free-space-path-loss / antenna-pattern / multipath scaling, which
    :func:`_tx_power_amplitude_scale` puts on an absolute voltage scale.
    """
    return tx_power_dbm is None


def _tx_power_amplitude_scale(
    tx_power_dbm: Optional[float], num_freqs: int, n_tx_ant: int = 1
) -> float:
    """Per-(tx-element) scalar amplitude applied to a physically-scaled frame (1.0 if legacy).

    A = sqrt(N * P_tx * Z0 / n_tx_ant), where N = num_freqs, P_tx = transmit power in
    watts, Z0 = the system reference impedance, and n_tx_ant = the transmit aperture's
    element count. Derivation: torch/np ``ifft`` uses the 1/N convention, so
    mean|ifft(A*H)|^2 = A^2 * mean|H|^2 / N. Choosing A = sqrt(N * P_tx * Z0) (n_tx_ant=1)
    makes the time-domain mean power = P_tx * Z0 * mean|H|^2, i.e. V_rms = sqrt(P_rx * Z0)
    with P_rx = P_tx * mean|H|^2 the received power -- independent of the frequency-grid
    size N (which would otherwise leak into the absolute voltage scale purely as an
    artifact of how finely the CFR is sampled).

    tx_power_dbm is the TOTAL power radiated by the tx aperture (see
    e2e.scenario.Node.tx_power_dbm), split uniformly across its n_tx_ant elements: each
    element carries P_tx / n_tx_ant, so its amplitude scale divides by sqrt(n_tx_ant).
    Without this, every element radiated the FULL P_tx and a wider tx aperture (e.g. a
    32x32 opt-in full tx array vs the default 1x1) silently inflated summed channel
    power by a factor of n_tx_ant (+10*log10(n_tx_ant) dB EIRP) at the same configured
    tx_power_dbm. n_tx_ant=1 (the default / all pre-existing single-element tx arrays)
    divides by sqrt(1) == 1, i.e. no change from the previous formula.
    """
    if tx_power_dbm is None:
        return 1.0
    p_tx_w = 10.0 ** ((tx_power_dbm - 30.0) / 10.0)
    return float(np.sqrt(num_freqs * p_tx_w * SYSTEM_IMPEDANCE_OHMS / n_tx_ant))


# --------------------------------------------------------------------------------
# The runner
# --------------------------------------------------------------------------------
class ScenarioRunner:
    """Drives frame generation for a single :class:`Scenario`.

    Use :meth:`run` for offline batch generation. The per-frame work is factored into
    :meth:`step` (returns one frame *per link*) so a future live/interactive mode can
    drive generation incrementally (see the hooks at the bottom of the class).
    """

    def __init__(self, scenario: Scenario, dry_run: bool = False, seed: int = 41):
        problems = scenario.validate()
        if problems:
            raise ValueError(f"invalid scenario {scenario.name!r}: {problems}")
        self.scenario = scenario
        self.dry_run = dry_run
        self.seed = seed

        self.schedule = build_schedule(scenario)
        self.links = enumerate_links(scenario)
        for link in self.links:
            _assert_single_pol(link.tx_array, f"link {link.name!r} tx array")
            _assert_single_pol(link.rx_array, f"link {link.name!r} rx array")
        self.primary_link = self.links[0]
        # kept for introspection / describe()
        self.tx_nodes, self.rx_nodes = _tx_rx_nodes(scenario)

        self.freqs = scenario.frequency.linspace()
        self.num_freqs = int(self.freqs.shape[0])
        self.n_time = 1  # single time step / chirp, matching the runtime pipeline assertions

        # Per-link frame shapes (cfr[0, :, 0, :, :, :] layout).
        self.frame_shapes: Dict[str, tuple] = {
            link.name: (link.n_rx_ant, link.n_tx_ant, self.n_time, self.num_freqs)
            for link in self.links
        }
        # Primary-link convenience attributes (back-compat for single-link callers/tests).
        self.frame_shape = self.frame_shapes[self.primary_link.name]
        self.n_rx_ant = self.primary_link.n_rx_ant
        self.n_tx_ant = self.primary_link.n_tx_ant
        self.tx_array = self.primary_link.tx_array

        # Per-link RNGs: each link gets its own independent stream seeded from the base
        # seed plus the link's index. A shared RNG would couple links -- a link's draws
        # (and thus its synthesized frames) would depend on how many elements the *other*
        # links consumed before it, so changing one link's array size would silently
        # change another link's data. Independent per-link streams make each link's
        # dry-run frames a function of (seed, link index, frame index, that link's shape)
        # only, so a link is reproducible regardless of what other links exist.
        self._link_rngs: Dict[str, np.random.Generator] = {
            link.name: np.random.default_rng(seed + idx)
            for idx, link in enumerate(self.links)
        }

        # Sionna handles -- populated lazily only in the real path.
        self._sionna = None

    @property
    def is_multilink(self) -> bool:
        return len(self.links) > 1

    # ---- summary ---------------------------------------------------------------
    def describe(self) -> str:
        sc = self.scenario
        lines = [
            f"scenario:   {sc.name}  (base_scene={sc.base_scene})",
            f"frames:     {sc.num_frames}",
            f"frequency:  {sc.frequency.start_hz/1e9:.3f}-{sc.frequency.stop_hz/1e9:.3f} GHz, "
            f"{self.num_freqs} bins (carrier {sc.frequency.carrier_hz/1e9:.3f} GHz)",
            f"objects:    {[o.name for o in self.scenario.objects]}",
            f"links ({len(self.links)}):",
        ]
        for link in self.links:
            kind = "radar" if link.tx_node is link.rx_node else "comm"
            lines.append(
                f"  - {link.name} [{kind}]: tx={link.tx_node.name} "
                f"(n_tx_ant={link.n_tx_ant}) -> rx={link.rx_node.name} "
                f"(n_rx_ant={link.n_rx_ant}); frame {self.frame_shapes[link.name]}"
            )
        dump = f"v2 payload: meta + {len(self.links)} link array(s)"
        lines.append(f"dumped:     {dump}")
        lines.append(f"mode:       {'DRY-RUN (synthetic, no Sionna)' if self.dry_run else 'REAL (Sionna RT)'}")
        moving = [n.name for n in sc.nodes if not n.motion.is_static]
        moving += [o.name for o in sc.objects if not o.motion.is_static]
        lines.append(f"moving:     {moving or 'none (all static)'}")
        return "\n".join(lines)

    # ---- v2 payload meta ---------------------------------------------------------
    def _build_meta(self) -> dict:
        """Build the ``meta`` half of the v2 payload (see the module docstring)."""
        fp = self.scenario.frequency
        links_meta = {}
        for link in self.links:
            tx_power_dbm = link.tx_node.tx_power_dbm
            links_meta[link.name] = {
                "tx_node": link.tx_node.name,
                "rx_node": link.rx_node.name,
                "rx_array_shape": [link.rx_array.num_rows, link.rx_array.num_cols],
                "n_tx_ant": link.n_tx_ant,
                "kind": "radar" if link.tx_node is link.rx_node else "comm",
                "tx_power_dbm": tx_power_dbm,
                "physical_scale": tx_power_dbm is not None,
            }
        return {
            "version": 1,
            "scenario_name": self.scenario.name,
            "freq_plan": {
                "carrier_hz": float(fp.carrier_hz),
                "start_hz": float(fp.start_hz),
                "stop_hz": float(fp.stop_hz),
                "num_freqs": int(fp.num_freqs),
            },
            "links": links_meta,
        }

    # ---- batch generation ------------------------------------------------------
    def run(self, out_path: Optional[str] = None, verbose: bool = True) -> Dict[str, dict]:
        """Generate every frame for every link and dump the result to ``out_path``.

        Always returns (and dumps) the self-describing v2 payload ``{"meta": ..., "links":
        {link_name: stacked_array}}`` -- single-link scenarios included, as a one-entry
        map (see the module docstring's "Links and output layout" section).
        """
        if not self.dry_run:
            self._setup_sionna()

        per_link: Dict[str, List[np.ndarray]] = {link.name: [] for link in self.links}
        n = self.scenario.num_frames
        for frame_idx in range(n):
            frame = self.step(frame_idx)  # dict {link_name: array}
            for name, arr in frame.items():
                per_link[name].append(arr)
            if verbose and (frame_idx % max(1, n // 10) == 0 or frame_idx == n - 1):
                print(f"  frame {frame_idx + 1}/{n}")

        stacks = {name: np.stack(frames, axis=0) for name, frames in per_link.items()}
        payload: Dict[str, dict] = {"meta": self._build_meta(), "links": stacks}

        if out_path is None:
            out_path = default_out_path(self.scenario.name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if verbose:
            print(f"dumping to file {out_path}")
        with open(out_path, "wb") as f:
            pickle.dump(payload, f)
        if verbose:
            shapes = {k: v.shape for k, v in stacks.items()}
            print(f"done dumping  {len(stacks)} links {shapes}")
        return payload

    # ---- single frame (live-mode hook) ----------------------------------------
    def step(self, frame_idx: int) -> Dict[str, np.ndarray]:
        """Produce one S-parameter array *per link* for a single frame.

        Returns ``{link_name: frame_shape_array}``. This is the unit of work a future
        live/interactive UI would call as the user scrubs a time slider. In dry-run mode
        it never touches Sionna.
        """
        if self.dry_run:
            return {link.name: self._mock_frame(link, frame_idx) for link in self.links}
        return self._real_frame(frame_idx)

    # ---- dry-run path (FULLY EXERCISED on this machine) ------------------------
    def _mock_frame(self, link: LinkSpec, frame_idx: int) -> np.ndarray:
        """Synthesize a plausible complex channel frequency response for one link.

        The values are not physical (legacy 1/dist taper), unless the link's tx node has
        a physical ``tx_power_dbm`` set, in which case the mock instead synthesizes a
        free-space-path-loss-scaled channel (see e2e.scenario.Node.tx_power_dbm) so the
        mock's absolute level is representative of a real link budget. Shape, dtype and
        frame-to-frame variation (fresh RNG each frame, plus a distance taper from the
        link's tx/rx tracks) match what the real path produces either way, so the whole
        scheduling + serialization chain is validated end to end.
        """
        tx_pos = self.schedule.node_position(link.tx_node.name, frame_idx)
        rx_pos = self.schedule.node_position(link.rx_node.name, frame_idx)
        dist = float(np.linalg.norm(np.asarray(rx_pos) - np.asarray(tx_pos))) + 1.0

        shape = self.frame_shapes[link.name]
        rng = self._link_rngs[link.name]
        real = rng.standard_normal(shape)
        imag = rng.standard_normal(shape)
        cfr = (real + 1j * imag).astype(np.complex64)

        tx_power_dbm = link.tx_node.tx_power_dbm
        if tx_power_dbm is None:
            cfr *= np.complex64(1.0 / dist)
        else:
            # H = (lambda_c / (4*pi*d)) * (randn + 1j*randn) / sqrt(2), so
            # E|H|^2 = (lambda_c / (4*pi*d))^2 -- the one-way free-space-path-loss power.
            #
            # HONESTY NOTE for MONOSTATIC RADAR links: tx and rx are the same node, so
            # d is pinned to the 1.0 m floor -- the mock's radar level is a fixed
            # NOMINAL reference (1 m one-way FSPL), NOT target-range physics (a real
            # echo scales ~1/R^4 with target RCS). Comm links (distinct tx/rx) do get
            # a real distance-dependent one-way level. Real ray tracing supplies the
            # actual radar physics; dry-run radar levels are plumbing-representative
            # in SCALE only.
            lambda_c = _SPEED_OF_LIGHT / self.scenario.frequency.carrier_hz
            fspl_amp = lambda_c / (4.0 * np.pi * dist)
            cfr *= np.complex64(fspl_amp / np.sqrt(2.0))
            cfr *= np.complex64(
                _tx_power_amplitude_scale(tx_power_dbm, self.num_freqs, link.n_tx_ant)
            )
        return cfr

    # ---- real Sionna path (needs a CUDA-12.x driver; see driver-incompat note) -
    def _setup_sionna(self):
        """Build one Sionna scene per link (each with its own tx/rx arrays + all objects).

        Per-link scenes are what allow heterogeneous links (e.g. a 32x32 radar RX and a
        4x4 comm RX) to coexist -- Sionna applies a single ``rx_array``/``tx_array`` per
        scene. Antenna devices are not scatterers, so a link's channel is unaffected by
        the other links' devices being absent from its scene.
        """
        import sionna.rt as rt  # lazy import -- only in the real path

        sc = self.scenario
        link_scenes: Dict[str, dict] = {}
        # NOTE (per-link scene cost): the base scene is parsed once PER LINK below, so an
        # N-link scenario loads the (potentially heavy) base scene N times. This is
        # deliberate, not an oversight: Sionna RT 1.2 exposes no Scene clone/copy API, and
        # tx_array / rx_array are *scene-global* single attributes -- there is no safe,
        # well-supported way to give heterogeneous links (e.g. a 32x32 radar RX and a 4x4
        # comm RX) their own arrays within one shared scene without mutating that global
        # state and the receiver/transmitter sets between links every frame, which is
        # fragile and cannot be validated here. We therefore accept the N-load cost in
        # exchange for fully isolated, correct per-link scenes. (The load is one-time setup,
        # not per-frame; the per-frame hot path only moves devices/objects and re-traces.)
        for link in self.links:
            scene = self._load_scene(rt, sc.base_scene)
            scene.frequency = sc.frequency.carrier_hz
            scene.tx_array = self._planar_array(rt, link.tx_array)
            scene.rx_array = self._planar_array(rt, link.rx_array)

            tx = rt.Transmitter(
                name="tx", position=list(link.tx_node.position),
                look_at=list(link.tx_node.look_at) if link.tx_node.look_at else None,
            )
            scene.add(tx)
            rx = rt.Receiver(
                name="rx", position=list(link.rx_node.position),
                look_at=list(link.rx_node.look_at) if link.rx_node.look_at else None,
            )
            scene.add(rx)

            scene_objs = {obj.name: self._add_scene_object(rt, scene, obj)
                          for obj in self.scenario.objects}

            link_scenes[link.name] = dict(
                scene=scene, tx=tx, rx=rx, solver=rt.PathSolver(),
                tx_node=link.tx_node, rx_node=link.rx_node, scene_objs=scene_objs,
            )

        self._sionna = dict(rt=rt, link_scenes=link_scenes)

    @staticmethod
    def _load_scene(rt, scene_name):
        builtin = getattr(rt.scene, scene_name, None)
        if builtin is not None:
            return rt.load_scene(builtin, merge_shapes=False)
        return rt.load_scene(scene_name, merge_shapes=False)

    def _planar_array(self, rt, cfg: ArrayConfig):
        """ArrayConfig -> sionna.rt.PlanarArray."""
        return rt.PlanarArray(
            num_rows=cfg.num_rows,
            num_cols=cfg.num_cols,
            vertical_spacing=cfg.vertical_spacing,
            horizontal_spacing=cfg.horizontal_spacing,
            pattern=cfg.pattern.value,
            polarization=cfg.polarization.value,
        )

    def _add_scene_object(self, rt, scene, obj: SceneObject):
        """Add a SceneObject (sphere/box/mesh) with material/scaling/color.

        Names are namespaced with an ``e2e-`` prefix so scenario scatterers cannot collide
        with objects already present in a loaded base scene (e.g. the Munich city model
        ships with its own vehicles named ``car-0`` etc.).
        """
        material = rt.ITURadioMaterial(
            f"e2e-mat-{obj.name}", obj.material, thickness=0.01,
            color=obj.color if obj.color is not None else (0.8, 0.1, 0.1),
        )
        from e2e.scenario import ObjectKind
        if obj.kind == ObjectKind.SPHERE:
            fname = rt.scene.sphere
        elif obj.kind == ObjectKind.BOX:
            fname = getattr(rt.scene, "box", rt.scene.sphere)
        else:  # MESH
            fname = obj.asset
        so = rt.SceneObject(fname=fname, name=f"e2e-obj-{obj.name}", radio_material=material)
        scene.edit(add=[so])
        so.scaling = obj.scaling
        so.position = [float(c) for c in obj.position]
        return so

    def _real_frame(self, frame_idx: int) -> Dict[str, np.ndarray]:
        """Apply motion, ray-trace and extract S-parameters for every link, one frame."""
        s = self._sionna

        # Mitsuba's Point3f rejects numpy float64 scalars ("Item assignment failed");
        # node_position()/object_position() return ndarrays, so coerce to python floats.
        def _xyz(arr):
            return [float(c) for c in arr]

        out: Dict[str, np.ndarray] = {}
        for link in self.links:
            ls = s["link_scenes"][link.name]
            ls["tx"].position = _xyz(self.schedule.node_position(link.tx_node.name, frame_idx))
            ls["rx"].position = _xyz(self.schedule.node_position(link.rx_node.name, frame_idx))
            # look_at is a static world point (per the Scenario spec: look_at is an
            # Optional[Vec3], a fixed world coordinate -- it cannot name another node).
            # Setting position alone does not re-aim the device, so a node that MOVES but
            # has a look_at target would otherwise keep its frame-0 orientation. Re-apply
            # look_at each frame so a moving *observer* keeps pointing at its (fixed)
            # target; devices without a look_at keep their default orientation.
            #
            # KNOWN LIMITATION (target tracking): because look_at is a fixed world point,
            # a node that aims at the *start* position of another, moving node keeps
            # pointing at that node's frame-0 location, not its current one. E.g. in
            # munich_isac, building_comm_tx.look_at is car_comm_rx's start position, so as
            # car_comm_rx drives away the TX beam no longer follows it. Resolving look_at
            # against a named node's per-frame position would fix this, but the Scenario
            # spec has no node-reference field for look_at (it is a Vec3), so adding one
            # would require changing the shared scenario contract and could not be
            # validated here without Sionna/GPU. Left as documented future work rather
            # than introducing a risky, untested feature.
            if link.tx_node.look_at is not None:
                ls["tx"].look_at(_xyz(link.tx_node.look_at))
            if link.rx_node.look_at is not None:
                ls["rx"].look_at(_xyz(link.rx_node.look_at))
            for name, so in ls["scene_objs"].items():
                so.position = _xyz(self.schedule.object_position(name, frame_idx))

            paths = ls["solver"](
                scene=ls["scene"], max_depth=5, los=True, specular_reflection=True,
                diffuse_reflection=False, refraction=True, synthetic_array=False,
                seed=self.seed,
            )
            out[link.name] = self._extract_s_pars(
                paths, link.tx_node.tx_power_dbm, link.n_tx_ant
            )
        return out

    def _extract_s_pars(
        self, paths, tx_power_dbm: Optional[float] = None, n_tx_ant: int = 1
    ) -> np.ndarray:
        """cfr -> per-frame, single-link S-parameter array (rx 0 / tx 0).

        ``tx_power_dbm`` (from the link's tx node) is None in the legacy contract, which
        keeps ``normalize=True`` (Sionna's unit-average-energy mode) and applies no
        amplitude scale, so output for existing scenarios is unchanged. When set, we
        request ``normalize=False`` (Sionna's own default) so the CFR carries physical
        free-space-path-loss / antenna-pattern / multipath scaling, then apply the
        per-link absolute-voltage scale (see _tx_power_amplitude_scale), which
        ``n_tx_ant`` (the link's transmit element count) splits uniformly across the tx
        aperture so tx_power_dbm stays the aperture's TOTAL radiated power.
        """
        cfr = paths.cfr(
            frequencies=self.freqs, normalize=_cfr_should_normalize(tx_power_dbm),
            normalize_delays=True, out_type="numpy",
        )
        # cfr: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, num_freqs].
        # Each link's scene has exactly one tx and one rx, so indices 0 select them.
        sliced = cfr[0, :, 0, :, :, :]
        # Sionna may return complex128 (or another precision) depending on its dtype
        # config; the dry-run path and the runtime pipeline (torch complex64) expect
        # complex64. Cast defensively so both generation paths honor the same dtype
        # contract and the dumped .pkl matches regardless of Sionna's native precision.
        arr = np.asarray(sliced).astype(np.complex64)
        if tx_power_dbm is not None:
            arr = arr * np.complex64(
                _tx_power_amplitude_scale(tx_power_dbm, self.num_freqs, n_tx_ant)
            )
        return arr

    # ============================================================================
    # LIVE / INTERACTIVE MODE HOOKS (future work)
    # ----------------------------------------------------------------------------
    # The pieces a live UI needs already exist:
    #   * build_schedule() resolves the full trajectory up front (cheap, Sionna-free),
    #     or a UI could recompute a single frame's positions on demand from motion.py.
    #   * step(frame_idx) produces exactly one frame per link; a UI event loop / slider
    #     could call runner.step(i) as the user scrubs, rendering the returned S-pars live.
    #   * _setup_sionna() builds persistent per-link scene/tx/rx/object handles so repeated
    #     step() calls only move objects + re-trace (the expensive part), which is the
    #     natural granularity for an interactive "play / pause / scrub" control.
    # A future StreamingScenarioRunner could expose start()/step()/stop() and push frames
    # into the runtime Simulation without ever writing a .pkl. The offline run() above is
    # just "call step() for every frame, then pickle the stack(s)".
    # ============================================================================


# --------------------------------------------------------------------------------
# Helpers + CLI
# --------------------------------------------------------------------------------
def default_out_path(name: str) -> str:
    return os.path.join(SIONNA_SIMS_DIR, f"{name}.pkl")


def load_scenario(spec: str) -> Scenario:
    """Resolve a CLI ``--scenario`` value into a Scenario.

    Accepts either a key in ``REFERENCE_SCENARIOS`` or a path to a Scenario JSON file.
    """
    if spec in REFERENCE_SCENARIOS:
        return REFERENCE_SCENARIOS[spec]()
    if os.path.isfile(spec):
        return Scenario.load(spec)
    raise ValueError(
        f"unknown scenario {spec!r}: not a reference scenario "
        f"({sorted(REFERENCE_SCENARIOS)}) and not an existing JSON path"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m e2e.environment.scenario_runner",
        description="Generate Sionna RT S-parameter frames from a declarative Scenario.",
    )
    p.add_argument(
        "--scenario", required=True,
        help=f"reference scenario name {sorted(REFERENCE_SCENARIOS)} or path to a Scenario JSON",
    )
    p.add_argument("--frames", type=int, default=None,
                   help="override Scenario.num_frames")
    p.add_argument("--out", default=None,
                   help="output .pkl path (default: sionna_sims/<scenario.name>.pkl)")
    p.add_argument("--dry-run", action="store_true",
                   help="synthesize frames of the correct shape WITHOUT Sionna RT")
    p.add_argument("--seed", type=int, default=41, help="RNG / path solver seed")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    scenario = load_scenario(args.scenario)
    if args.frames is not None:
        scenario.num_frames = args.frames

    runner = ScenarioRunner(scenario, dry_run=args.dry_run, seed=args.seed)

    print("=" * 70)
    print(runner.describe())
    print("=" * 70)

    out_path = args.out or default_out_path(scenario.name)
    payload = runner.run(out_path=out_path)

    print("-" * 70)
    links = payload["links"]
    print(f"Generated {scenario.num_frames} frames x {len(links)} links -> {out_path}")
    for name, arr in links.items():
        print(f"  link {name}: {arr.shape}, dtype {arr.dtype}")
    if args.dry_run:
        print("(dry-run: synthetic data; replace with real generation by dropping --dry-run "
              "on a machine with Sionna RT installed.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
