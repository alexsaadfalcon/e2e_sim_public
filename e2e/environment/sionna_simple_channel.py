"""Re-trace the Munich scenario at Ka-band and write a v2 `{"meta", "links"}` pkl.

Fixes the defect recorded in `notes/ESTABLISHED_FACTS.md` F93: the original script set
`scene.frequency` on a `simple_street_canyon` scene it then discarded (line 66 rebound
`scene` to a freshly loaded, still-3.5-GHz `munich`), so the shipped `munich.pkl` was
traced at Sionna's 3.5 GHz default (4.28 cm array spacing) while everything downstream
labelled it "30 GHz". This module sets `scene.frequency` on the scene it actually solves,
and does so BEFORE building `PlanarArray`s -- Sionna sizes element spacing in
wavelengths at construction time (see F92's note on upstream issue 470), so an array
built before the frequency assignment silently keeps the wrong physical spacing even if
`scene.frequency` is later corrected.

Usage::

    python -m e2e.environment.sionna_simple_channel [--carrier-hz 30e9]
        [--band-hz 28.5e9 31.5e9] [--num-freqs 1000] [--num-frames 100]
        [--out e2e/environment/sionna_sims/munich_ka.pkl] [--seed 41]

Moving the line of sight (2026-09-24, owner directive "the line of sight path change
angle so that even if the rank doesn't change, the direction changes"). Two mutually
compatible ways to make the LoS ARRIVAL ANGLE AT THE ARRAY sweep frame to frame, both
default OFF so the shipped `munich_ka.pkl` recipe is unchanged:

* ``--los-sweep-deg A B`` -- the array PANS: the receiver is re-aimed at the transmitter
  every frame and then yawed by an offset that walks linearly from ``A`` to ``B`` degrees
  over the run (so the LoS sits at a commanded azimuth in the array frame each frame,
  exactly, and `--boresight-offset-deg` is ignored). The ray-traced path SET is untouched
  (paths depend on positions, not on the receiver's attitude), so this rotates the
  principal direction while holding path powers/delays fixed.
* ``--tx-lateral-m T0 T1`` -- the TRANSMITTER moves: its position walks linearly from
  ``T0`` to ``T1`` metres along the horizontal direction perpendicular to the frame-0
  rx->tx line ("across the street"), re-aimed at the array each frame; the receiver's
  attitude is left as the shipped file has it (aimed once at frame 0). This moves the
  real geometry, so the whole multipath set changes with it.
  MEASURED WINDOW (2026-09-24/25, real solves, specular-only probe, this scene, 30 GHz,
  `<scratch>/losweep_probe/tx_{move,edge}.py` + their logs): the direct path is present
  for lateral offsets -100 m .. +32 m at BOTH ends of the receiver's own 30 m travel
  (probed at 10 m steps on the negative side, 2 m steps near the positive edge, and at 9
  coarse interior points); it is BLOCKED from +34 m outward (smallest solved delay
  383-644 ns against a 298-322 ns geometric direct delay, that path's power share
  0.0-0.02 %). The boundary is NOT clean beyond the window: at -110 m the frame-0
  receiver loses it, while at -120 m a weak direct path returns carrying 6.6 %. The
  direct-path power share inside the window is 67-93 % (92 % for the shipped static
  geometry), i.e. the LoS still dominates but the multipath is not held fixed.

`meta["frames"]` records, per frame, the rx/tx positions, the receiver orientation, the
range, the commanded yaw offset, the LoS azimuth in the array frame (`sin_az` and
`los_az_deg`), the solved path count and -- as a LoS-presence receipt -- the smallest
solved delay against the geometric direct delay. `meta["generator_args"]` records the
full parsed argument set. Both are written for EVERY run, sweep or not.

The output is the SAME v2 payload format `e2e.environment.scenario_runner` writes and
`e2e.environment.sionna_iterator.SionnaIterator` reads (`{"meta": {...}, "links":
{name: ndarray}}`), so the frames load through `SionnaEnvironmentBlock` unchanged --
this is a generated artifact (`sionna_sims/` is gitignored), not a committed one.
"""

from __future__ import annotations

import argparse
import datetime
import os
import pickle
import subprocess
import sys

import numpy as np

_C = 299_792_458.0  # m/s

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CARRIER_HZ = 30e9
DEFAULT_BAND_HZ = (28.5e9, 31.5e9)
DEFAULT_NUM_FREQS = 1000  # keeps the runtime frame shape (n_rx, 1, 1, 1000) legacy munich.pkl ships
DEFAULT_NUM_FRAMES = 100
DEFAULT_SEED = 41
DEFAULT_OUT = os.path.join(_THIS_DIR, "sionna_sims", "munich_ka.pkl")
LINK_NAME = "munich"


def _git_head():
    """Best-effort HEAD sha for provenance; None if git is unavailable (never fatal)."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_THIS_DIR,
                                       stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def unambiguous_range_m(num_freqs: int, band_hz) -> float:
    """The USABLE (non-negative-range) unambiguous range window for a band of width
    `B = stop - start` Hz sampled at `num_freqs` complex points.

    An `num_freqs`-point IFFT of the complex CFR gives a range profile with sample
    spacing `c / (2B)` and a FULL symmetric span of `num_freqs * c / (2B)` (from
    `-span/2` to `+span/2`, via `torch.fft.fftshift` -- see `e2e.blocks.RangeAzBlock`).
    Only the non-negative half is physically meaningful (range can't be negative) and is
    what downstream display/cropping (`_nonneg_range` in `webapp/pipeline_runner.py`)
    keeps, so the window a target can actually be unambiguously placed in is HALF the
    full span: `num_freqs * c / (4B)`.

    Verified empirically (`munich_physics` investigation, 2026-09-23): 1000 points over
    3 GHz gives 25.0 m -- this scene's 37 m non-LoS return exceeds that window and is
    cropped, and its 68 m family aliases into the window at 15-18 m; the NAIVE
    `num_freqs*c/(2B)` reading (50.0 m here) is too large to explain either observation
    (37 m would neither crop nor alias within a 50 m window).
    """
    start_hz, stop_hz = band_hz
    bandwidth_hz = float(stop_hz) - float(start_hz)
    return float(num_freqs) * _C / (4.0 * bandwidth_hz)


def _rotation_matrix(alpha, beta, gamma):
    """TR38901 (7.1-4) GCS<-LCS rotation matrix -- same closed form as Sionna's
    `sionna.rt.utils.rotation_matrix`, reimplemented in plain numpy so it is testable
    without Sionna. `R @ local_vector = global_vector`; angles are (yaw about z, pitch
    about y, roll about x) radians."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cc, sc = np.cos(gamma), np.sin(gamma)
    return np.array([
        [ca * cb, ca * sb * sc - sa * cc, ca * sb * cc + sa * sc],
        [sa * cb, sa * sb * sc + ca * cc, sa * sb * cc - ca * sc],
        [-sb, cb * sc, cb * cc],
    ])


def boresight_sin_az(rx_pos, tx_pos, orientation) -> float:
    """`sin(azimuth)` of the direct rx->tx path IN THE RECEIVER'S LOCAL (array) FRAME,
    given the receiver's `orientation = (alpha, beta, gamma)` (radians).

    The array lies in the receiver's local y-z plane (`PlanarArray`), so this is the
    y-component of the rx->tx unit vector after rotating it into that local frame --
    exactly what `--boresight-offset-deg`'s receipt reports (see `build_scene`)."""
    d = np.asarray(tx_pos, dtype=float) - np.asarray(rx_pos, dtype=float)
    v_global = d / np.linalg.norm(d)
    alpha, beta, gamma = orientation
    r = _rotation_matrix(alpha, beta, gamma)
    v_local = r.T @ v_global  # R is orthogonal: local = R^-1 @ global = R^T @ global
    return float(v_local[1])


def los_az_deg_from_sin(sin_az: float) -> float:
    """`sin(azimuth)` -> azimuth in degrees, clipped to the physical [-1, 1] domain.

    `boresight_sin_az` is a projection of a unit vector, so |sin_az| <= 1 up to float
    round-off; clipping keeps `arcsin` from returning nan on a 1+1e-16 input."""
    return float(np.degrees(np.arcsin(np.clip(float(sin_az), -1.0, 1.0))))


def los_sweep_offsets(num_frames: int, sweep_deg) -> np.ndarray:
    """Per-frame commanded boresight yaw offsets (degrees) for `--los-sweep-deg A B`:
    `num_frames` points from A to B inclusive (a single frame sits at A).

    Pure geometry, no Sionna -- the per-frame schedule `generate()` follows, so a test
    can check the sweep without a scene (`tests/test_sionna_simple_channel.py`)."""
    a, b = float(sweep_deg[0]), float(sweep_deg[1])
    if num_frames <= 1:
        return np.array([a])
    return np.linspace(a, b, int(num_frames))


def street_lateral_axis(rx_pos, tx_pos) -> np.ndarray:
    """Unit HORIZONTAL vector perpendicular to the horizontal rx->tx line ("across the
    street" at the transmitter, the direction that changes the arrival azimuth fastest
    and the range slowest). z-component is 0: `--tx-lateral-m` never changes the
    transmitter's height."""
    d = np.asarray(tx_pos, dtype=float) - np.asarray(rx_pos, dtype=float)
    h = np.array([d[0], d[1], 0.0])
    norm = np.linalg.norm(h)
    if norm == 0.0:
        raise ValueError("rx and tx are vertically aligned: no horizontal lateral axis")
    h = h / norm
    return np.array([-h[1], h[0], 0.0])


def tx_sweep_positions(tx_pos, rx_pos, lateral_m, num_frames) -> np.ndarray:
    """`[num_frames, 3]` transmitter positions for `--tx-lateral-m T0 T1`: `tx_pos`
    displaced by a linear walk from T0 to T1 metres along `street_lateral_axis`.

    `rx_pos` is the FRAME-0 receiver position: the lateral axis is fixed once, from the
    initial geometry, so the transmitter travels a straight line (not a curve that
    tracks the moving receiver)."""
    lat = street_lateral_axis(rx_pos, tx_pos)
    t0, t1 = float(lateral_m[0]), float(lateral_m[1])
    ts = np.array([t0]) if num_frames <= 1 else np.linspace(t0, t1, int(num_frames))
    return np.asarray(tx_pos, dtype=float)[None, :] + ts[:, None] * lat[None, :]


def build_frequencies(carrier_hz: float, band_hz, num_freqs: int) -> np.ndarray:
    """`num_freqs` points spanning `band_hz` (absolute Hz), relative to `carrier_hz`.

    Sionna's `paths.cfr(frequencies=...)` wants frequencies RELATIVE to the carrier
    Sionna's array/materials were built at (`scene.frequency`), not absolute Hz.
    """
    start_hz, stop_hz = band_hz
    return np.linspace(start_hz - carrier_hz, stop_hz - carrier_hz, num_freqs)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Re-trace the Sionna 'munich' scene at a given carrier/band.")
    p.add_argument("--carrier-hz", type=float, default=DEFAULT_CARRIER_HZ,
                   help="Carrier frequency in Hz; sets scene.frequency (default 30e9).")
    p.add_argument("--band-hz", type=float, nargs=2, default=list(DEFAULT_BAND_HZ),
                   metavar=("START_HZ", "STOP_HZ"),
                   help="Absolute band edges in Hz (default 28.5e9 31.5e9).")
    p.add_argument("--num-freqs", type=int, default=DEFAULT_NUM_FREQS)
    p.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--diffuse", action="store_true",
                   help="Enable diffuse reflection in the path solver (v1.0 default is "
                        "specular-only, diffuse_reflection=False -- see the module "
                        "docstring's 'diffuse' section for why that under-populates the "
                        "power-delay profile).")
    p.add_argument("--scattering-coefficient", type=float, default=0.0,
                   help="Scattering coefficient S in [0,1] applied to EVERY ITU material "
                        "in the scene when --diffuse is set (Sionna's own default is 0.0, "
                        "so diffuse paths carry no energy without this). Ignored unless "
                        "--diffuse is also given. This is an ASSUMPTION, not a "
                        "measurement -- see the module docstring.")
    p.add_argument("--boresight-offset-deg", type=float, default=0.0,
                   help="Yaw the receiver's orientation this many degrees away from "
                        "boresight-at-transmitter (default 0.0 = rx.look_at(tx), "
                        "unchanged) -- F94: at exact boresight the LoS path carries 92%% "
                        "of the power at broadside, making frames rank-1 by geometry. "
                        "Positive values put the transmitter at POSITIVE sin(azimuth) in "
                        "the receiver's local (array) frame -- see build_scene/"
                        "boresight_sin_az.")
    p.add_argument("--los-sweep-deg", type=float, nargs=2, default=None,
                   metavar=("START_DEG", "STOP_DEG"),
                   help="Sweep the LoS arrival angle by PANNING THE ARRAY: re-aim the "
                        "receiver at the transmitter every frame, then yaw it by an "
                        "offset walking linearly from START_DEG to STOP_DEG over the "
                        "run (overrides --boresight-offset-deg). Default off = the "
                        "shipped static-attitude recipe. See the module docstring.")
    p.add_argument("--tx-lateral-m", type=float, nargs=2, default=None,
                   metavar=("START_M", "STOP_M"),
                   help="Sweep the LoS arrival angle by MOVING THE TRANSMITTER: walk it "
                        "linearly from START_M to STOP_M metres along the horizontal "
                        "axis perpendicular to the frame-0 rx->tx line, re-aimed at the "
                        "array each frame. Default off. The direct path is only present "
                        "for roughly -100..+32 m in this scene (measured -- see the "
                        "module docstring); outside that window frames lose the LoS.")
    return p.parse_args(argv)


# Safety margin under DrJit/Mitsuba's 2^32 (4294967296) per-tensor entry limit --
# `paths.cfr()`'s internal phase tensor is roughly num_paths * n_freqs_chunk * n_rx_ant
# entries; leaving headroom below the hard limit for other same-order-of-magnitude
# intermediates the call allocates internally.
_CFR_TENSOR_BUDGET = 1_500_000_000


def _cfr_chunk_size(num_freqs: int, num_paths: int, n_rx_ant: int) -> int:
    """How many frequencies to synthesize per `paths.cfr()` call so
    `num_paths * chunk * n_rx_ant` stays under `_CFR_TENSOR_BUDGET` -- pulled out of
    `_synthesize_cfr` so the sizing logic is testable without a real `Paths` object."""
    return max(1, min(num_freqs, _CFR_TENSOR_BUDGET // max(1, num_paths * n_rx_ant)))


def _synthesize_cfr(paths, frequencies, n_rx_ant, normalize, normalize_delays):
    """`paths.cfr(frequencies=...)`, chunked over the frequency axis so a path-rich
    solve (e.g. `synthetic_array=True` with diffuse reflection: tens of thousands of
    paths, see `generate`'s "lost paths" fix) never asks DrJit for a single tensor
    bigger than its 2^32-entry limit. Returns `[n_rx_ant, 1, 1, len(frequencies)]`,
    identical in shape/content to one un-chunked `paths.cfr()` call.

    `normalize=True` is NOT forwarded per chunk: Sionna's `cfr()` (see
    `sionna/rt/path_solvers/paths.py`, the `normalize` branch) computes its unit-energy
    scale factor from `mean(|H|**2)` OVER WHATEVER FREQUENCIES ARE PASSED IN THAT CALL --
    doing that once per (much narrower) chunk would independently rescale each chunk to
    its own local energy, corrupting relative power ACROSS chunk boundaries (measured:
    this exact bug, in an earlier version of this function, silently varied a stored
    frame's relative dB by >10 dB depending only on chunk size, at bins with genuine but
    weak content). Instead every chunk is synthesized with `normalize=False`, and the
    SAME formula Sionna uses (`1/sqrt(mean(|H|**2))`) is applied ONCE across the full
    concatenated (unchunked) frequency axis, exactly matching what one un-chunked
    `paths.cfr(normalize=True)` call would have produced.
    """
    num_paths = int(paths.tau.shape[-1])
    chunk = _cfr_chunk_size(len(frequencies), num_paths, n_rx_ant)
    out = []
    for i in range(0, len(frequencies), chunk):
        sub = frequencies[i:i + chunk]
        cfr = paths.cfr(frequencies=sub, normalize=False,
                        normalize_delays=normalize_delays, out_type="numpy")
        # [num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, num_freqs] ->
        # [num_rx_ant, num_tx_ant, num_time_steps, num_freqs] (one rx, one tx node).
        out.append(cfr[0, :, 0, :, :, :])
    s_pars = np.concatenate(out, axis=-1)
    if normalize:
        mean_power = np.mean(np.abs(s_pars) ** 2)
        if mean_power > 0:
            s_pars = s_pars / np.sqrt(mean_power)
    return s_pars


def _node_pos(node) -> np.ndarray:
    """A Sionna scene node's position as a plain `[3]` float array (off the device)."""
    return np.asarray(node.position.numpy(), dtype=float).reshape(3)


def _node_orientation(node) -> tuple:
    """A Sionna scene node's `(alpha, beta, gamma)` orientation in radians, as floats."""
    return tuple(float(c.numpy()[0]) for c in
                 (node.orientation.x, node.orientation.y, node.orientation.z))


def aim_receiver(rx, tx, boresight_offset_deg: float) -> tuple:
    """Point `rx` at `tx` and then yaw it `boresight_offset_deg` degrees off that
    boresight; return `(orientation_rad, sin_az)` for the resulting attitude.

    The yaw SUBTRACTS from alpha, which is the sign that puts the transmitter at POSITIVE
    `sin(azimuth)` in the receiver's local (array) frame -- verified against this scene's
    geometry (`tests/test_sionna_simple_channel.py`). Pulled out of `build_scene` so the
    per-frame pan of `--los-sweep-deg` re-aims with EXACTLY the same convention as the
    one-shot aim of `--boresight-offset-deg`."""
    import mitsuba as mi

    rx.look_at(tx)
    if boresight_offset_deg != 0.0:
        alpha, beta, gamma = rx.orientation.x, rx.orientation.y, rx.orientation.z
        new_alpha = alpha - float(np.radians(boresight_offset_deg))
        rx.orientation = mi.Point3f(new_alpha, beta, gamma)
    orientation_rad = _node_orientation(rx)
    sin_az = boresight_sin_az(_node_pos(rx), _node_pos(tx), orientation_rad)
    return orientation_rad, sin_az


#: Tolerance (ns) for calling a solved path THE direct path: a path whose delay equals
#: the geometric rx-tx distance / c. Measured on this scene (2026-09-24 probe): when the
#: LoS is present the smallest solved delay matches d/c to better than 0.01 ns; when it
#: is blocked the smallest delay is 80-340 ns larger. 0.3 ns = 9 cm of path length.
_LOS_DELAY_TOL_NS = 0.3


def _path_power_share(paths, index) -> float:
    """Fraction of the frame's total solved path power carried by path `index`.

    Best effort: `paths.a` is a device tensor whose exact layout has changed across
    Sionna versions (1.2.2 here returns a `(real, imag)` pair), so this returns None
    rather than failing a multi-hour generation over a receipt. With
    `synthetic_array=True` the antenna axes are singleton, so the pull is small even for
    the ~30k-path diffuse solves."""
    try:
        a = paths.a
        try:
            a_r, a_i = a
            amp = np.asarray(a_r.numpy()) + 1j * np.asarray(a_i.numpy())
        except (TypeError, ValueError):
            amp = np.asarray(a.numpy())
        n_paths = int(paths.tau.shape[-1])
        power = (np.abs(amp) ** 2).reshape(-1, n_paths).sum(axis=0)
        total = float(power.sum())
        if total <= 0.0:
            return None
        return float(power[index] / total)
    except Exception:
        return None


def _los_receipt(paths, rx_pos, tx_pos) -> tuple:
    """`(min_tau_ns, direct_delay_ns, los_present, los_power_share)` for one solved frame.

    A LoS-presence RECEIPT, not an assumption: the transmitter can be occluded at some
    positions of a `--tx-lateral-m` walk (measured: blocked from +34 m outward in this
    scene), and a frame that lost its direct path is not the frame the demo claims. Read
    off `paths.tau` (already-stored delays, no extra solve)."""
    tau_all = np.asarray(paths.tau.numpy(), dtype=float).reshape(-1)
    direct_ns = float(np.linalg.norm(np.asarray(tx_pos, dtype=float)
                                     - np.asarray(rx_pos, dtype=float)) / _C * 1e9)
    valid = tau_all > 0.0
    if not valid.any():
        return None, direct_ns, False, None
    i_min = int(np.argmin(np.where(valid, tau_all, np.inf)))
    min_ns = float(tau_all[i_min] * 1e9)
    los_present = bool(abs(min_ns - direct_ns) < _LOS_DELAY_TOL_NS)
    # The share is reported for the SHORTEST path whether or not it is the direct one:
    # when the LoS is blocked, "the strongest early arrival carries 0.02 %" is itself the
    # diagnostic (measured on the blocked positions of the 2026-09-24 probe).
    return min_ns, direct_ns, los_present, _path_power_share(paths, i_min)


def build_scene(carrier_hz: float, boresight_offset_deg: float = 0.0):
    """Load munich, set `scene.frequency` on the scene that will actually be solved,
    THEN attach the tx/rx `PlanarArray`s and place tx/rx -- see the module docstring for
    why the order matters. Returns `(scene, tx, rx, wavelength, rx_spacing_m, aperture_m)`.

    `boresight_offset_deg` (F94): with plain `rx.look_at(tx)`, the direct path arrives at
    EXACT broadside (sin(az)=0) carrying 92% of the power, so frames are rank-1 by
    geometry. A nonzero value yaws the receiver's orientation away from boresight by
    that many degrees (about the local vertical, applied once after `look_at` -- the
    per-frame motion loop in `generate()` only translates `rx.position`, never
    re-orients it, matching the original script). See `boresight_sin_az` for the sign
    convention and the printed receipt below for the resulting angle.

    Split out from `generate()` so a test can check the scene's own `.frequency` and its
    rx array spacing directly, not just the meta values `generate()` derives from them.
    """
    import sionna.rt
    from sionna.rt import Camera, PlanarArray, Receiver, Transmitter, load_scene

    scene = load_scene(sionna.rt.scene.munich, merge_shapes=True)  # merge -> faster solves
    # Set frequency on THE SCENE THAT IS ACTUALLY SOLVED, before building arrays (see
    # module docstring / F92 / F93).
    scene.frequency = float(carrier_hz)
    # scene.frequency is a DrJit array, not a python float/np.float32 -- .numpy() pulls
    # the scalar off the device before any float()/f-string formatting touches it.
    scene_frequency_hz = float(scene.frequency.numpy()[0])

    wavelength = _C / scene_frequency_hz
    rx_spacing_m = 0.5 * wavelength
    aperture_m = 31 * rx_spacing_m  # 32-element ULA per axis -> 31 inter-element gaps
    print(f"scene.frequency  = {scene_frequency_hz:.6e} Hz")
    print(f"wavelength       = {wavelength:.6e} m")
    print(f"rx element spacing = {rx_spacing_m:.6e} m")
    print(f"rx aperture (per axis, 32 elements) = {aperture_m:.6e} m")

    # Arrays built AFTER the frequency assignment above -- PlanarArray sizes its
    # `vertical_spacing`/`horizontal_spacing` (given in wavelengths) at construction time.
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                 horizontal_spacing=0.5, pattern="tr38901", polarization="V")
    scene.rx_array = PlanarArray(num_rows=32, num_cols=32, vertical_spacing=0.5,
                                 horizontal_spacing=0.5, pattern="iso", polarization="V")

    tx = Transmitter(name="tx", position=[8.5, 21, 27], display_radius=10)
    scene.add(tx)
    rx = Receiver(name="rx", position=[45, 90, 1.5], display_radius=10)
    scene.add(rx)
    tx.look_at(rx)
    # Receipt: F94 found sin(az)=0 (broadside) at plain boresight; report what this run
    # actually achieves, from the SOLVED (post-offset) orientation and positions.
    # `--los-sweep-deg` re-aims per frame in `generate()`; this is the INITIAL attitude.
    _, sin_az = aim_receiver(rx, tx, boresight_offset_deg)
    print(f"boresight_offset_deg = {boresight_offset_deg} -> direct-path sin(az) in "
         f"receiver array frame = {sin_az:.4f}")

    # Kept for parity with the original interactive script (a camera angle used for
    # scene.render()/preview()); this module runs headless (no display), so it is
    # constructed but never rendered.
    Camera(position=[150, 275, 150], look_at=[30, 70, 28])

    return scene, tx, rx, wavelength, rx_spacing_m, aperture_m


def generate(args) -> tuple[np.ndarray, dict]:
    """Ray-trace `args.num_frames` and return `(stacked_s_pars, meta)`.

    Sionna is imported here (via `build_scene`, not at module scope) so `python -m
    e2e.environment.sionna_simple_channel --help` and unit tests that only exercise
    `build_frequencies`/`parse_args`/the pkl writer never need Sionna/DrJit installed.
    """
    import sionna.rt
    from sionna.rt import PathSolver

    scene, tx, rx, wavelength, rx_spacing_m, aperture_m = build_scene(
        args.carrier_hz, boresight_offset_deg=args.boresight_offset_deg)

    # v1.0 solved specular-only (diffuse_reflection=False); its power-delay profile has
    # exactly one tap above -20 dB per frame (vs. 22-26 on the legacy 3.5 GHz file).
    # --diffuse turns diffuse reflection back on; Sionna's own scattering_coefficient
    # default is 0.0 (no energy in the diffuse lobe at all) on every material, so
    # --scattering-coefficient must also be set for --diffuse to change anything.
    if args.diffuse and args.scattering_coefficient > 0.0:
        for mat in scene.radio_materials.values():
            mat.scattering_coefficient = args.scattering_coefficient
        print(f"scattering_coefficient = {args.scattering_coefficient} "
             f"(ASSUMPTION, not measured -- set on {len(scene.radio_materials)} materials)")

    frequencies = build_frequencies(args.carrier_hz, args.band_hz, args.num_freqs)
    p_solver = PathSolver()

    # Physics decisions the validation campaign will revisit -- unchanged from the
    # original script, but now recorded in `meta` (see the module docstring) rather
    # than silently baked into the array.
    normalize = True
    normalize_delays = True

    # PathSolver args in effect (receipt) -- max_depth/los/specular/refraction/seed are
    # fixed; diffuse_reflection and synthetic_array are the two this module varies.
    print(f"PathSolver args: max_depth=5, los=True, specular_reflection=True, "
         f"diffuse_reflection={args.diffuse}, refraction=True, synthetic_array=True, "
         f"seed={args.seed}")

    # --- synthetic_array=True (FIX, see module docstring's "lost paths" section) -------
    # `synthetic_array=False` (the v1/v2 setting) solves candidate paths against each of
    # the array's 1024 individual antenna ELEMENTS as separate ray-tracing targets, and
    # empirically DISCOVERS FAR FEWER valid specular/diffuse candidate sequences than
    # solving once against the array's single phase center (`synthetic_array=True`) --
    # measured on this scene (35 deg offset, diffuse, scattering_coefficient=0.4, frame
    # 1): 10 vs 38 specular-only paths, 662 vs ~33300 with diffuse, and TWO specific
    # non-LoS families (radial excess ~37 m and ~68 m) that read -66/-71 dB (buried in
    # FFT sidelobe noise) under synthetic_array=False recover to roughly -24/-26 dB under
    # synthetic_array=True -- a ~40 dB difference raising `--samples-per-src`/
    # `--max-num-paths-per-src` by 20x did NOT close (path count grew 20x, these two
    # bins' level did not move), so this is a per-element CANDIDATE DISCOVERY limitation,
    # not a sampling-budget one.
    # Plane-wave-approximation validity (why treating the array as a single point for
    # path discovery, then applying the array response ANALYTICALLY, is legitimate here):
    # worst-case path-length difference across the (up to) 0.219 m diagonal aperture is
    # ~4.4 range bins (bin size c/(2B)=0.05 m) at grazing incidence, ~1.7-3 bins at this
    # scene's actual ~35-55 deg incidence -- a few-bin SMEAR, not the ~40 dB DELETION
    # `synthetic_array=False` exhibits above. See boresight_sin_az for the aperture/bin
    # numbers this run actually used (printed below).
    bin_size_m = _C / (2.0 * (args.band_hz[1] - args.band_hz[0]))
    diag_m = aperture_m * (2.0 ** 0.5)
    print(f"plane-wave check: aperture={aperture_m:.4f} m, diagonal={diag_m:.4f} m, "
         f"range bin={bin_size_m:.4f} m -> worst-case smear "
         f"{diag_m / bin_size_m:.1f} bins (grazing incidence)")

    # Path-count receipt (with vs without diffuse), from a cheap probe solve at the
    # FIRST frame's position -- BEFORE the motion loop below moves `rx`.
    probe_no_diffuse = p_solver(scene=scene, max_depth=5, los=True,
                                specular_reflection=True, diffuse_reflection=False,
                                refraction=True, synthetic_array=True, seed=args.seed)
    print(f"num_paths without diffuse (probe) = {probe_no_diffuse.tau.shape[-1]}")

    # --- per-frame LoS-angle schedule (both default off; see the module docstring) ----
    # `offsets[i]`: the commanded yaw of the receiver off boresight-at-transmitter.
    # Without --los-sweep-deg it is the single --boresight-offset-deg value and the
    # receiver is NEVER re-aimed (the shipped recipe: aimed once in build_scene, then
    # translated), so the LoS azimuth drifts only through the receiver's own motion.
    if args.los_sweep_deg is not None:
        offsets = los_sweep_offsets(args.num_frames, args.los_sweep_deg)
        step = (offsets[1] - offsets[0]) if len(offsets) > 1 else 0.0
        print(f"los sweep (ARRAY PAN): re-aim + yaw {offsets[0]:.3f} -> {offsets[-1]:.3f} "
             f"deg over {args.num_frames} frames ({step:.3f} deg/frame); "
             f"--boresight-offset-deg ({args.boresight_offset_deg}) IGNORED")
    else:
        offsets = np.full(int(args.num_frames), float(args.boresight_offset_deg))

    if args.tx_lateral_m is not None:
        # Frame 0 is the position AFTER the first `+= [1,0,0]` step below, matching the
        # original script's motion; the lateral axis is fixed from that geometry.
        rx_frame0 = _node_pos(rx) + np.array([1.0, 0.0, 0.0])
        tx_positions = tx_sweep_positions(_node_pos(tx), rx_frame0,
                                          args.tx_lateral_m, args.num_frames)
        lat_axis = street_lateral_axis(rx_frame0, _node_pos(tx))
        print(f"tx sweep (TRANSMITTER MOVES): {args.tx_lateral_m[0]} -> "
             f"{args.tx_lateral_m[1]} m along {np.round(lat_axis, 4).tolist()} "
             f"({(args.tx_lateral_m[1] - args.tx_lateral_m[0]) / max(1, args.num_frames - 1):.3f} "
             f"m/frame); tx re-aimed at the array every frame")
    else:
        tx_positions = None

    all_s_pars = []
    frames_meta = []
    for frame_idx in range(args.num_frames):
        rx.position += [1, 0, 0]  # same per-frame motion as the original script
        if tx_positions is not None:
            tx.position = tx_positions[frame_idx].tolist()
            tx.look_at(rx)  # the tx pattern is directional (tr38901): keep it on the array
        if args.los_sweep_deg is not None:
            orientation_rad, sin_az = aim_receiver(rx, tx, float(offsets[frame_idx]))
        else:
            orientation_rad = _node_orientation(rx)
            sin_az = boresight_sin_az(_node_pos(rx), _node_pos(tx), orientation_rad)
        paths = p_solver(scene=scene, max_depth=5, los=True, specular_reflection=True,
                         diffuse_reflection=args.diffuse, refraction=True,
                         synthetic_array=True, seed=args.seed)
        if frame_idx == 0:
            # tau is a stored tensor (cheap property access, no extra solve) --
            # [num_rx, num_tx, num_paths] (synthetic_array=True: no antenna axes).
            print(f"num_paths (frame 0, diffuse={args.diffuse}) = {paths.tau.shape[-1]}")
        rx_pos, tx_pos = _node_pos(rx), _node_pos(tx)
        min_tau_ns, direct_ns, los_present, los_share = _los_receipt(paths, rx_pos, tx_pos)
        frames_meta.append({
            "index": frame_idx,
            "rx_pos": rx_pos.tolist(),
            "tx_pos": tx_pos.tolist(),
            "rx_orientation_rad": list(orientation_rad),
            "range_m": float(np.linalg.norm(tx_pos - rx_pos)),
            "boresight_offset_deg": float(offsets[frame_idx]),
            # The LoS arrival angle AT THE ARRAY, in the receiver's local frame: this is
            # the quantity the sweep moves and the subspace tracker sees.
            "sin_az": float(sin_az),
            "los_az_deg": los_az_deg_from_sin(sin_az),
            "num_paths": int(paths.tau.shape[-1]),
            "min_tau_ns": min_tau_ns,
            "direct_delay_ns": direct_ns,
            "los_present": los_present,
            # Share of the frame's total solved path power in that shortest path.
            "los_power_share": los_share,
        })
        print(f"  frame {frame_idx:3d}: rx={np.round(rx_pos, 2).tolist()} "
             f"tx={np.round(tx_pos, 2).tolist()} R={frames_meta[-1]['range_m']:.2f} m "
             f"offset={offsets[frame_idx]:+.2f} deg sin_az={sin_az:+.4f} "
             f"los_az={frames_meta[-1]['los_az_deg']:+.2f} deg "
             f"paths={frames_meta[-1]['num_paths']} los_present={los_present} "
             f"(min_tau {min_tau_ns if min_tau_ns is None else round(min_tau_ns, 2)} ns "
             f"vs d/c {direct_ns:.2f} ns) "
             f"los_share={'None' if los_share is None else format(100 * los_share, '.2f') + '%'}")
        s_pars = _synthesize_cfr(paths, frequencies, n_rx_ant=1024,
                                 normalize=normalize, normalize_delays=normalize_delays)
        all_s_pars.append(s_pars)

    az = [f["los_az_deg"] for f in frames_meta]
    n_no_los = sum(1 for f in frames_meta if not f["los_present"])
    print(f"LoS azimuth at the array: {min(az):+.2f} .. {max(az):+.2f} deg "
         f"(span {max(az) - min(az):.2f} deg); frames without a direct path: {n_no_los}")

    all_s_pars = np.stack(all_s_pars, axis=0).astype(np.complex64)

    try:
        sionna_version = sionna.rt.__version__
    except AttributeError:
        sionna_version = getattr(sys.modules.get("sionna"), "__version__", None)

    unambig_range_m = unambiguous_range_m(args.num_freqs, args.band_hz)
    print(f"unambiguous range = {unambig_range_m:.3f} m "
         f"({args.num_freqs} points over {args.band_hz[1] - args.band_hz[0]:.3e} Hz)")

    meta = {
        "version": 2,
        "scenario_name": "munich",
        "scene": "munich",
        "carrier_hz": float(args.carrier_hz),
        "freq_plan": {
            "carrier_hz": float(args.carrier_hz),
            "start_hz": float(args.band_hz[0]),
            "stop_hz": float(args.band_hz[1]),
            "num_freqs": int(args.num_freqs),
        },
        "unambiguous_range_m": unambig_range_m,
        "rx_spacing_m": float(rx_spacing_m),
        "aperture_m": float(aperture_m),
        "normalize": normalize,
        "normalize_delays": normalize_delays,
        "boresight_offset_deg": float(args.boresight_offset_deg),
        # None unless the corresponding sweep is on (see the module docstring). When
        # `los_sweep_deg` is set, `boresight_offset_deg` above is NOT what was flown --
        # the per-frame commanded value is `frames[i]["boresight_offset_deg"]`.
        "los_sweep_deg": (None if args.los_sweep_deg is None
                          else [float(x) for x in args.los_sweep_deg]),
        "tx_lateral_m": (None if args.tx_lateral_m is None
                         else [float(x) for x in args.tx_lateral_m]),
        "rx_motion_per_frame_m": [1.0, 0.0, 0.0],
        # Per-frame geometry receipt: rx/tx positions, receiver attitude, the LoS arrival
        # azimuth in the array frame, and the direct-path presence check. Written for
        # every run, sweep or not.
        "frames": frames_meta,
        "los_az_deg_first": frames_meta[0]["los_az_deg"],
        "los_az_deg_last": frames_meta[-1]["los_az_deg"],
        "los_az_span_deg": float(max(f["los_az_deg"] for f in frames_meta)
                                 - min(f["los_az_deg"] for f in frames_meta)),
        "frames_without_los": int(sum(1 for f in frames_meta if not f["los_present"])),
        "generator_args": {k: (list(v) if isinstance(v, (list, tuple)) else v)
                           for k, v in vars(args).items()},
        "generator_argv": list(sys.argv[1:]),
        "synthetic_array": True,
        # See generate()'s "lost paths" comment: synthetic_array=False under-discovers
        # specular/diffuse candidates against the 1024 individual antenna elements;
        # synthetic_array=True solves once against the phase center and applies the
        # array response analytically (plane-wave approximation, justified for this
        # aperture -- see the printed "plane-wave check" receipt).
        "diffuse_reflection": bool(args.diffuse),
        # ASSUMPTION, not a measurement, when diffuse_reflection is True: 0.4 is the
        # order-of-magnitude the Sionna scattering tutorial uses for building materials,
        # not something fit to this scene. 0.0 (the no-op default) when --diffuse is off.
        "scattering_coefficient": float(args.scattering_coefficient),
        "scattering_coefficient_is_assumption": bool(args.diffuse
                                                     and args.scattering_coefficient > 0.0),
        "sionna_version": sionna_version,
        "git_head": _git_head(),
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "seed": int(args.seed),
        "links": {
            LINK_NAME: {
                "tx_node": "tx",
                "rx_node": "rx",
                "rx_array_shape": [32, 32],
                "n_tx_ant": 1,
                "kind": "radar",
                "tx_power_dbm": None,
                "physical_scale": False,
            },
        },
    }
    return all_s_pars, meta


def write_payload(all_s_pars: np.ndarray, meta: dict, out_path: str) -> None:
    """Write the `{"meta": ..., "links": {name: ndarray}}` v2 payload `SionnaIterator`
    understands (see `e2e.environment.sionna_iterator` / `scenario_runner`'s writer)."""
    payload = {"meta": meta, "links": {LINK_NAME: all_s_pars}}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"dumping to file {out_path}")
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print("done dumping")


def main(argv=None):
    args = parse_args(argv)
    all_s_pars, meta = generate(args)
    write_payload(all_s_pars, meta, args.out)
    return all_s_pars, meta


if __name__ == "__main__":
    main()
