"""
Radar-ML corpus generation AS A COMPOSED `e2e.simulation.Simulation` RUN.

This is the migration `report/chain_integration_design.html`'s "The result" section
specifies: the corpus generator no longer calls the analytic point-target synthesizer
(`e2e.chain.rd_synth.synthesize_adc`, still available as `e2e.ml.dataset`'s explicitly-
labelled CI/offline fallback) directly. Instead it builds an `e2e.simulation.Simulation`
out of the SAME blocks the runtime pipeline uses, ray-traces via
`e2e.environment.blocks.RTEnvironmentBlock`, and runs it frame by frame:

    RTEnvironmentBlock (ray-traced CFR + labels)
        -> CircuitStage(RFFEBlock)        # RF front end -- ON by default, see below
        -> InterconnectStage(InterconnectBlock)   # ON by default
        -> DechirpBlock                   # crossing: CFR -> RX-time ADC
        -> ImpairmentBlock                # phase noise / leakage / clutter, per-frame
        -> QuantizerBlock                 # ADC digitization
        -> RadarCubeBlock                 # range-Doppler product (downstream)
        -> SinkBlock                      # persists adc + labels + meta (downstream)

RFFE/interconnect are config-gated (`use_rffe`/`use_interconnect`) but default ON --
that is the point of this module: the pre-migration generator realized zero of the
chain's analog stages (see the design notes' "The problem"). `RFFEBlock` is normally
sized for the 1024-element IMAGING array; `build_chain_simulation` forces its `n` to
`cfg.n_rx` (the radar's actual receive-channel count, e.g. 4 for `ti_iwr1443`) unless
the caller overrides `rffe_kwargs["n"]` explicitly -- see that function's docstring.

`generate_chain_corpus` is the CLI/script entry point (mirrors `e2e.ml.dataset.
generate_dataset`'s signature/on-disk contract as closely as the two producers'
different internals allow) and needs real Sionna RT (`RTEnvironmentBlock`); tests build
`Simulation`s directly via `build_chain_simulation(..., environment_block=<stub>)` with a
synthetic CFR-emitting stand-in, so the composition itself is exercised without Sionna.

Written samples share ONE on-disk manifest schema with `e2e.ml.dataset.generate_dataset`
(the "On-disk dataset layout" that module documents): `write_manifest` -- which THIS
module calls, not duplicates -- is the shared tail. `SinkBlock` writes the per-frame
`.npz` (`adc`/`labels`/`meta`, matching field names) directly; this module is the glue
that names files consistently with `write_manifest`'s `sequences` structure.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from e2e.blocks import CircuitStage, InterconnectBlock, InterconnectStage, RFFEBlock
from e2e.chain.dechirp import DechirpBlock
from e2e.chain.receive import (IFHighPassBlock, ImpairmentBlock, QuantizerBlock,
                               RadarCubeBlock)
from e2e.environment.blocks import RTEnvironmentBlock
from e2e.ml.blocks import SinkBlock
from e2e.ml.dataset import DATASETS_DIR
from e2e.simulation import Simulation

DEFAULT_LABEL_CLASSES = ("vehicle", "pedestrian")

#: Interconnect transfer function used for ML corpora. See build_chain_simulation for why
#: the block's own placeholder default is not used here.
DEFAULT_INTERCONNECT_CSV = (Path(__file__).resolve().parent.parent
                            / "data" / "interconnect" / "tessera_case3_s21_77ghz.csv")


# --------------------------------------------------------------------------------
# ImpairmentBlock -> SinkBlock provenance glue
# --------------------------------------------------------------------------------
def default_domain_randomizer(
    *, phase_noise_dbc_hz_range: Tuple[float, float] = (-95.0, -75.0),
    # Re-derived 2026-08-18 for the ABSOLUTE thermal reference (F35 flip). These are now
    # dB ABOVE the thermal floor, not fractions of the cube's own peak, so the old
    # negative ranges would have injected impairments ~60 dB too weak to exist.
    #   leakage: 30-40 dB TX-RX isolation at P_tx 12 dBm over an -85 dBm floor -> 57-67 dB.
    #   clutter: a clutter-to-noise spread around the +10 dB nominal (re-anchored
    #            2026-08-24, plan A15; see ClutterParams.total_relative_db for the
    #            corrected justification -- the level is a difficulty ASSUMPTION
    #            anchored on road-discrete cell CNR of ~10-30 dB with a heavy tail,
    #            +10 at the hot edge). The range is deliberately WIDE (0-18 dB,
    #            median drawn-cell CNR ~24-42 across it): the anchor itself is
    #            uncertain, so the corpus randomizes across the plausible band
    #            rather than pretending one number is known.
    leakage_relative_db_range: Tuple[float, float] = (57.0, 67.0),
    clutter_relative_db_range: Tuple[float, float] = (0.0, 18.0),
) -> Callable[[int, "torch.Generator"], Dict[str, Any]]:
    """Build the `chain_params` callable `ImpairmentBlock` expects (see its docstring):
    `(frame_index, rng) -> {"phase_noise": {...}, "leakage": {...}, "clutter": {...}}`,
    one uniform draw per stage per frame from `rng` (a `torch.Generator` on the ADC's
    device, seeded deterministically by `ImpairmentBlock` from its own `seed` -- see
    that class). This is the per-frame domain randomization requirement 1 of this
    shard's brief: every frame gets its own draw, and the resolved values are recorded
    (via `_ImpairmentStage`) into the written sample.
    """

    def _u(rng: "torch.Generator", lo: float, hi: float) -> float:
        return lo + (hi - lo) * float(torch.rand((), generator=rng, device=rng.device).item())

    def _sample(frame_index: int, rng: "torch.Generator") -> Dict[str, Any]:
        return {
            "phase_noise": {"psd_dbc_hz_at_ref": _u(rng, *phase_noise_dbc_hz_range)},
            "leakage": {"leakage_relative_db": _u(rng, *leakage_relative_db_range)},
            "clutter": {"total_relative_db": _u(rng, *clutter_relative_db_range)},
        }

    return _sample


# --------------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------------
def build_chain_simulation(
    scenario, cfg, out_dir, *, tag: str = "sample",
    use_rffe: bool = True, use_interconnect: bool = True,
    rffe_kwargs: Optional[Dict[str, Any]] = None,
    interconnect_kwargs: Optional[Dict[str, Any]] = None,
    impairment_chain_params=None, impairment_seed: int = 0,
    quant_bits: int = 12, environment_block=None,
    label_grid=None, label_classes: Optional[Sequence[str]] = DEFAULT_LABEL_CLASSES,
    device=None, k: int = 1, base_scene: str = "flat",
    use_transmit_chain: bool = False, tx_pa_config=None,
    coherent_targets: bool = True, antenna_pattern: Optional[str] = None,
    ground_scattering_coefficient: Optional[float] = None,
    samples_per_src: Optional[int] = None,
    use_link_budget: bool = True,
    use_if_hpf: bool = True, if_hpf_kwargs: Optional[Dict[str, Any]] = None,
) -> Simulation:
    """Compose ONE radar-ML `Simulation` run (see module docstring for the block list).

    `coherent_targets` / `antenna_pattern` are the 2026-08-17 target-physics fix and are
    forwarded to `RTEnvironmentBlock` (they are ignored when the caller supplies its own
    `environment_block`). `coherent_targets=True` adds each object's coherent specular
    return -- without it a ray-traced target is pure Monte-Carlo speckle and earns almost
    none of the chain's coherent processing gain (see the "HYBRID RT" banner in
    `e2e.environment.rt_signal_chain`). `antenna_pattern=None` takes
    `rt_scene_build.DEFAULT_ANTENNA_PATTERN` (`"tr38901"`, directive), which is what keeps
    the flat scene's nadir ground bounce from owning the cube peak -- and therefore from
    owning the reference `ImpairmentBlock`'s relative-power stages calibrate against. Pass
    `coherent_targets=False, antenna_pattern="iso"` to reproduce a pre-2026-08-17 corpus.

    `environment_block=None` (default) builds `RTEnvironmentBlock(scenario, cfg,
    device=device, label_grid=label_grid, label_classes=label_classes)` -- real ray
    tracing, needs Sionna. Pass a stand-in (anything exposing `get_S_pars`/`step`/
    `reset`, optionally `get_state_updates`/`array_shape`) to run the same composition
    without Sionna -- this is how the test suite exercises it (no `RUN_SIONNA` gate on
    the composition itself, only on real ray tracing).

    `use_rffe`/`use_interconnect` (default True/True -- ON by default is the point of
    this module, see the module docstring) gate `CircuitStage(RFFEBlock(...))` /
    `InterconnectStage(InterconnectBlock(...))`. `rffe_kwargs` passes through to
    `RFFEBlock`, with `n` forced to `cfg.n_rx` UNLESS the caller already set it --
    `RFFEBlock.apply_circuit` requires `n == s_pars.shape[0]` (the RX axis), and the
    block's own default sizing is for the 1024-element imaging array, not a 4-RX
    automotive radar; passing the imaging-array default here would raise a `view()`
    shape error on the very first frame. `interconnect_kwargs` passes through to
    `InterconnectBlock` unchanged (no array-size dependency -- it broadcasts over every
    leading axis).

    `impairment_chain_params` is `ImpairmentBlock`'s `chain_params` (a fixed dict, or a
    `(frame_index, rng) -> dict` callable for per-frame domain randomization -- see
    `default_domain_randomizer`); `impairment_seed` seeds its per-frame sub-seed
    sequence. The block chain runs through `_ImpairmentStage` (see that class) so its
    JSON-unserializable dataclass provenance still reaches the written sample.

    `k` is `Simulation`'s required subspace-tracking-rank argument; this composition
    has no `subspace_block`, so `k` only sizes the (otherwise-unused) `U_true`/rank
    diagnostic `Simulation.feed_forward`'s frequency-domain branch always computes --
    kept small (default 1) to minimize that overhead.
    """
    env = environment_block if environment_block is not None else RTEnvironmentBlock(
        scenario, cfg, base_scene=base_scene, device=device,
        label_grid=label_grid, label_classes=label_classes,
        coherent_targets=coherent_targets, antenna_pattern=antenna_pattern,
        ground_scattering_coefficient=ground_scattering_coefficient,
        samples_per_src=samples_per_src,
    )

    serial_stages: List[Any] = []

    # The transmit tributary, when enabled: generate the waveform, distort it in the
    # amplifier, then merge its spectrum into the channel response. These come first
    # because everything after them is the receive side. Off by default so the composed
    # chain stays on the long-proven path unless a caller asks for the TX stage.
    if use_transmit_chain:
        from e2e.chain.waveform import ModulateBlock, TxPABlock, WaveformBlock
        from e2e.circuit.tx_pa import TxPA, TxPAConfig
        pa = TxPA(tx_pa_config if tx_pa_config is not None else TxPAConfig())
        serial_stages.append(WaveformBlock(
            kind="fmcw", n_tx=cfg.n_tx, n_chirp=cfg.n_chirps, n_t=cfg.n_samples,
            bw=float(cfg.bandwidth_hz),
        ))
        serial_stages.append(TxPABlock(pa))
        serial_stages.append(ModulateBlock(tx_pa=pa, bandwidth_hz=float(cfg.bandwidth_hz)))
    if use_rffe:
        kwargs = dict(rffe_kwargs or {})
        kwargs.setdefault("n", int(cfg.n_rx))
        serial_stages.append(CircuitStage(RFFEBlock(**kwargs)))
    if use_interconnect:
        # The block's own default is an unnormalised 11-tap boxcar placeholder, which
        # convolves the range profile with an 11-sample boxcar: a point target smears
        # across 11 bins with 0 dB sidelobes, destroying range resolution. Measured, then
        # replaced here with a real simulated interconnect (0.51-0.55 dB passive loss,
        # 0.034 dB ripple in band) that leaves the range response intact -- width 1 bin,
        # sidelobes -75 dB. The classic pipeline's default is deliberately untouched.
        ic_kwargs = dict(interconnect_kwargs) if interconnect_kwargs else {}
        ic_kwargs.setdefault("transfer_csv", str(DEFAULT_INTERCONNECT_CSV))
        ic_kwargs.setdefault("band_hz", (75e9, 81e9))
        serial_stages.append(InterconnectStage(InterconnectBlock(**ic_kwargs)))
    serial_stages.append(DechirpBlock(cfg))
    # The link budget goes BETWEEN dechirp and impairments, and the position is the whole
    # point. Impairments are specified relative to a reference; before this stage runs
    # there is no absolute reference in the chain for them to be relative TO, so they were
    # calibrated against the only thing available -- the cube's own contents, which scale
    # with the target. That is F35's ceiling and F42's missing floor, and both dissolve
    # once the cube is on an absolute scale with a real k*T*B*F floor beneath it.
    if use_link_budget:
        from e2e.chain.link_budget import ThermalNoiseBlock
        serial_stages.append(ThermalNoiseBlock(cfg, seed=impairment_seed))
    serial_stages.append(
        ImpairmentBlock(cfg, impairment_chain_params, seed=impairment_seed)
    )
    # The IF high-pass sits AFTER the impairments (the close-in leakage/bumper tones it
    # exists to suppress must be present) and BEFORE the quantizer (protecting the ADC's
    # dynamic range is its purpose: QuantizerBlock AGCs full scale off the frame peak,
    # which without this filter is the leakage tone, not a target). ON by default --
    # every real FMCW receiver has one, and its absence biased every detection number
    # pessimistically (physics audit entry 9; release-plan A2).
    if use_if_hpf:
        serial_stages.append(IFHighPassBlock(cfg, **(if_hpf_kwargs or {})))
    serial_stages.append(QuantizerBlock(bits=quant_bits))

    downstream_blocks = [RadarCubeBlock(cfg), SinkBlock(out_dir, tag=tag)]

    return Simulation(
        environment_block=env,
        downstream_blocks=downstream_blocks,
        k=k,
        serial_stages=serial_stages,
    )


def _corpus_identity_tag(dataset_dir: Path) -> str:
    """The scene-identity salt for a corpus at `dataset_dir`: its last TWO path
    components (`parent/name`), machine-stable (never an absolute path).

    Two components, not just the leaf (B1 corpus review finding, 2026-08-25): the
    leaf alone is `f"{cfg_name}_{tier}"`, so two corpora under different `--out`
    roots sharing (config, tier, seed) salted identically and drew BIT-IDENTICAL
    scenes -- a 2-frame smoke run collided with the real corpus's train split
    exactly that way, undeclared by the overlap gate's allowlist. NOTE: this
    derivation change alters the drawn scenes of any corpus regenerated at the same
    (config, tier, seed) relative to pre-2026-08-25 runs -- deliberate; the
    manifest's `corpus_tag` records which salt a corpus actually used.
    """
    return f"{dataset_dir.parent.name}/{dataset_dir.name}"


# --------------------------------------------------------------------------------
# Corpus generation entry point
# --------------------------------------------------------------------------------
def generate_chain_corpus(
    cfg_name: str, tier: str, n_scenes: int, out_dir=None, *, seed: int = 0,
    frames_per_scene: int = 1, use_rffe: bool = True, use_interconnect: bool = True,
    rffe_kwargs: Optional[Dict[str, Any]] = None,
    interconnect_kwargs: Optional[Dict[str, Any]] = None,
    quant_bits: int = 12, splits: Tuple[float, ...] = (0.8, 0.1, 0.1),
    range_stride: int = 4, n_azimuth: int = 192, device=None,
    label_classes: Optional[Sequence[str]] = DEFAULT_LABEL_CLASSES,
    randomizer: Optional[Callable[[int, "torch.Generator"], Dict[str, Any]]] = None,
    use_local_assets: bool = True, use_transmit_chain: bool = False,
    coherent_targets: bool = True, antenna_pattern: Optional[str] = None,
    ground_scattering_coefficient: Optional[float] = None,
    samples_per_src: Optional[int] = None,
    allow_unanswerable: bool = False,
    use_if_hpf: bool = True, if_hpf_kwargs: Optional[Dict[str, Any]] = None,
) -> Path:
    """Generate a radar-ML corpus by RUNNING THE COMPOSED CHAIN, one `Simulation` per
    scene (real ray tracing -- needs Sionna; see `build_chain_simulation`).

    Mirrors `e2e.ml.dataset.generate_dataset`'s signature and on-disk contract: `n_scenes`
    independent scenes (`e2e.ml.scenes.sample_scene`, same tier/seed convention), each
    yielding `frames_per_scene` frames of ONE `e2e.simulation.Simulation.run()` call,
    written under `<out_dir>/<cfg_name>_<tier>/` with a `manifest.json` via `e2e.ml.
    dataset.write_manifest` -- so a directory this function writes loads via
    `e2e.ml.dataset.RadarFrameDataset` exactly like one `generate_dataset` wrote.

    Impairments are domain-randomized PER FRAME via `randomizer` (default
    `default_domain_randomizer()`) -- each scene's `ImpairmentBlock` gets a distinct
    seed (`seed + i * frames_per_scene`) so frames across the whole corpus don't repeat
    a randomization draw.

    `coherent_targets` / `antenna_pattern` (defaults `True` / `None` -> directive) are the
    2026-08-17 target-physics fix; see `build_chain_simulation`. **Every corpus generated
    before 2026-08-17 used the broken combination** (speckle-only targets, isotropic
    elements, mirror ground) and its targets sit below their own map background --
    regenerate rather than reuse. Pass `coherent_targets=False, antenna_pattern="iso"`
    only to reproduce one of those deliberately.

    `use_transmit_chain` defaults to **False**, and that default changed in v1.1. It used
    to be True, which silently cancelled the target-physics fix above: MEASURED, with the
    TX tributary on, median target-vs-background is **-1.6 dB**; with it off, **+7.4 dB**.
    `ModulateBlock` multiplies `s_pars` by the transmitted chirp's spectrum, but
    `rt_cfr_frame`'s `s_pars` axis is already a beat/time index, not a conventional
    frequency response -- so enabling it applies a second, incompatible modulation. That
    convention clash is UNRESOLVED; until it is, RT corpora carry no TX-side PA
    distortion, and turning this back on will produce a corpus whose targets do not stand
    out from their own background.
    """
    from e2e.ml.dataset import write_manifest
    from e2e.ml.labels import LabelGrid
    from e2e.radar_config import PRESETS
    from e2e.environment.rt_scenes import RT_DIFFICULTY_TIERS, build_rt_tier_scenario

    if cfg_name not in PRESETS:
        raise ValueError(f"unknown radar config {cfg_name!r}; choices: {sorted(PRESETS)}")
    if tier not in RT_DIFFICULTY_TIERS:
        raise ValueError(f"unknown RT difficulty tier {tier!r}; "
                         f"choices: {sorted(RT_DIFFICULTY_TIERS)}")
    if frames_per_scene < 1:
        raise ValueError(f"frames_per_scene must be >= 1, got {frames_per_scene}")
    cfg = PRESETS[cfg_name]

    # Answerability guard (release-plan A3): refuse to generate a corpus on which the
    # detection benchmark is arithmetically incapable of being answered -- F43 made this
    # class of invalid corpus a discovered fact; this guard makes it an impossible state.
    # `allow_unanswerable=True` is the deliberate escape hatch (ablations, regression
    # reproductions); the resulting corpus must never back a detection benchmark.
    from e2e.radar_config import answerability_problems
    problems = answerability_problems(
        cfg, top_speed_mps=RT_DIFFICULTY_TIERS[tier].speed_mps[1])
    if problems and not allow_unanswerable:
        raise ValueError(
            f"({cfg_name}, {tier}) cannot support a detection benchmark:\n  - "
            + "\n  - ".join(problems)
            + "\nPass allow_unanswerable=True (CLI: --allow-unanswerable) only for a "
              "corpus that will never back a benchmark (see F43 / radar_config."
              "answerability_problems)")

    grid = LabelGrid.for_config(cfg, range_stride=range_stride, n_azimuth=n_azimuth)
    randomize = randomizer if randomizer is not None else default_domain_randomizer()

    out_root = Path(out_dir) if out_dir is not None else DATASETS_DIR
    dataset_dir = out_root / f"{cfg_name}_{tier}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    corpus_tag = _corpus_identity_tag(dataset_dir)

    sequences: List[List[str]] = []
    for i in range(n_scenes):
        # The RAY-TRACED tier builder, not the analytic one: this is what places real
        # decimated vehicle meshes (car/truck/bus/trolley) and, for the city tier, sets
        # the base scene. Sampling from e2e.ml.scenes here would have produced a corpus
        # of spheres with none of the mesh work in it.
        # dt MUST be the real frame period: sampled speeds are physical m/s, and
        # frame_scatterers differences the stored per-frame displacement by this same
        # dt. Omitting it inflated every velocity by frame_rate_hz (0-8 m/s tier ->
        # 0-80 m/s at the solver, past the unambiguous-velocity limit). Fixed 2026-08-16.
        scenario = build_rt_tier_scenario(
            tier, corpus_tag=corpus_tag, frame_idx=i, seed=seed,
            num_frames=frames_per_scene,
            dt=1.0 / float(cfg.frame_rate_hz),
            use_local_assets=use_local_assets,
        )
        tag = f"sample_scene{i:05d}"

        sim = build_chain_simulation(
            scenario, cfg, dataset_dir, tag=tag,
            base_scene=RT_DIFFICULTY_TIERS[tier].base_scene,
            use_transmit_chain=use_transmit_chain,
            use_rffe=use_rffe, use_interconnect=use_interconnect,
            rffe_kwargs=rffe_kwargs, interconnect_kwargs=interconnect_kwargs,
            impairment_chain_params=randomize, impairment_seed=seed + i * frames_per_scene,
            quant_bits=quant_bits, label_grid=grid, label_classes=label_classes, device=device,
            coherent_targets=coherent_targets, antenna_pattern=antenna_pattern,
            ground_scattering_coefficient=ground_scattering_coefficient,
            samples_per_src=samples_per_src,
            use_if_hpf=use_if_hpf, if_hpf_kwargs=if_hpf_kwargs,
        )
        sim.run(n_steps=frames_per_scene)

        scene_files = [f"{tag}_frame_{t:05d}.npz" for t in range(frames_per_scene)]
        sequences.append(scene_files)

    return write_manifest(dataset_dir, cfg, tier, sequences, grid=grid, seed=seed,
                          snr_db=None, frames_per_scene=frames_per_scene, splits=splits,
                          label_classes=label_classes or (), corpus_tag=corpus_tag)


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------
def build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m e2e.ml.chain_generate",
        description="Generate a labeled FMCW radar corpus by running the composed "
                    "block chain (RTEnvironmentBlock -> RFFE -> interconnect -> "
                    "dechirp -> impairments -> quantizer -> radar cube -> sink). "
                    "Needs Sionna RT.",
    )
    p.add_argument("--config", required=True, help="radar config preset name (see e2e.radar_config.PRESETS)")
    p.add_argument("--tier", required=True,
                   help="RT difficulty tier (see e2e.environment.rt_scenes.RT_DIFFICULTY_TIERS)")
    p.add_argument("--n", type=int, required=True, help="number of scenes to generate")
    p.add_argument("--seed", type=int, default=0, help="base RNG seed (scene i uses seed + i)")
    p.add_argument("--out", default=None, help="output root directory (default: e2e/ml/datasets)")
    p.add_argument("--frames-per-scene", type=int, default=1,
                   help="consecutive motion-consistent frames per scene (default: 1)")
    p.add_argument("--no-rffe", action="store_true", help="disable the RFFE front-end stage")
    p.add_argument("--no-interconnect", action="store_true", help="disable the interconnect stage")
    p.add_argument("--quant-bits", type=int, default=12, help="ADC quantizer bit depth")
    # --- target physics (2026-08-17 fix; correct behaviour is the DEFAULT) -------------
    p.add_argument("--no-coherent-targets", action="store_true",
                   help="drop each object's coherent specular return, leaving only the "
                        "ray-traced diffuse speckle. REPRODUCES THE PRE-2026-08-17 BUG "
                        "(targets below their own map background) -- regression use only")
    p.add_argument("--antenna-pattern", default=None,
                   help="antenna ELEMENT pattern (iso|tr38901|dipole|hw_dipole); default "
                        "is rt_scene_build.DEFAULT_ANTENNA_PATTERN (directive). 'iso' "
                        "reproduces the pre-2026-08-17 nadir-ground-bounce behaviour")
    # DEFAULT FLIPPED 2026-08-24 to match generate_chain_corpus's own v1.1 default:
    # the CLI used to pass use_transmit_chain=True unless --no-transmit-chain was
    # given, silently re-enabling the ModulateBlock convention clash the function
    # default was flipped to avoid (targets sink to -1.6 dB median vs +7.4 dB; see
    # generate_chain_corpus's docstring). Caught while preparing the B1 launch
    # command -- a corpus generated through the bare CLI would have been broken.
    p.add_argument("--transmit-chain", action="store_true",
                   help="OPT IN to the TX tributary (waveform -> PA -> modulate). "
                        "OFF by default: MEASURED 2026-08-17, leaving it on costs ~9 dB "
                        "of target-to-background and erases the target-physics fix "
                        "entirely (suspected convention clash in ModulateBlock -- see "
                        "generate_chain_corpus's docstring). Do not enable for corpora")
    p.add_argument("--no-transmit-chain", action="store_true",
                   help="deprecated no-op (the TX tributary is now off by default; "
                        "kept so recorded pre-2026-08-24 command lines still run)")
    p.add_argument("--no-local-assets", action="store_true",
                   help="place only Sionna-bundled meshes (use_local_assets=False). "
                        "REQUIRED for corpora that may back public figures: the local "
                        "mesh pool's licences are unestablished (F21/F51), and frames "
                        "now record their asset provenance either way")
    p.add_argument("--ground-scattering", type=float, default=None,
                   help="ground-plane scattering coefficient for the flat base scene "
                        "(default: rt_scene_build.DEFAULT_GROUND_SCATTERING_COEFFICIENT). "
                        "0.2-0.4 is the physically honest range for asphalt at 3.8 mm but "
                        "costs ~37x the per-frame CFR time and ~8 dB of target margin -- "
                        "read that constant before setting it")
    p.add_argument("--samples-per-src", type=int, default=None,
                   help="Sionna PathSolver Monte-Carlo ray budget (default: Sionna's own "
                        "1e6). The dominant cost knob when the ground scatters diffusely: "
                        "1e5 cut the path count 10x with the target metric unchanged")
    p.add_argument("--no-if-hpf", action="store_true",
                   help="disable the IF high-pass between mixer and ADC (on by "
                        "default -- every real FMCW receiver has one; without it the "
                        "TX-RX leakage tone sets the ADC full scale). Reproduces "
                        "pre-A2 corpora")
    p.add_argument("--if-hpf-corner-range", type=float, default=None,
                   help="IF high-pass corner expressed as a RANGE in metres "
                        "(default 1.0; see IFHighPassBlock)")
    p.add_argument("--allow-unanswerable", action="store_true",
                   help="override the answerability guard (F43): generate even though "
                        "the (config, tier) pair cannot support a detection benchmark "
                        "(targets alias in Doppler, or the azimuth tolerance outresolves "
                        "the array). For ablations/regressions only")
    p.add_argument("--dry-run", action="store_true",
                   help="print the generation plan without ray-tracing/writing anything")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    import sys

    args = build_arg_parser().parse_args(argv)

    from e2e.radar_config import PRESETS

    if args.config not in PRESETS:
        print(f"unknown --config {args.config!r}; choices: {sorted(PRESETS)}", file=sys.stderr)
        return 2
    cfg = PRESETS[args.config]

    # RT_DIFFICULTY_TIERS, not scenes.DIFFICULTY_TIERS: this CLI drives the RAY-TRACED
    # corpus path, whose tier set (D0-D4, incl. the Munich city tier) is what
    # `generate_chain_corpus` itself validates against. Checking the analytic dict here
    # (the pre-2026-08-23 bug) made D4 unreachable from the CLI.
    from e2e.environment.rt_scenes import RT_DIFFICULTY_TIERS

    if args.tier not in RT_DIFFICULTY_TIERS:
        print(f"unknown --tier {args.tier!r}; choices: {sorted(RT_DIFFICULTY_TIERS)}", file=sys.stderr)
        return 2

    if args.dry_run:
        total_frames = args.n * args.frames_per_scene
        print("=" * 70)
        print(f"config:       {args.config}  (mimo={cfg.mimo}, n_rx={cfg.n_rx}, n_tx={cfg.n_tx})")
        print(f"tier:         {args.tier}")
        print(f"scenes:       {args.n}  x  frames_per_scene={args.frames_per_scene}  "
              f"= {total_frames} frames")
        print(f"rffe:         {'off' if args.no_rffe else 'on'}   "
              f"interconnect: {'off' if args.no_interconnect else 'on'}   "
              f"tx chain: {'ON (broken for corpora -- see --transmit-chain help)' if args.transmit_chain else 'off'}   "
              f"if hpf: {'OFF (pre-A2)' if args.no_if_hpf else 'on'}")
        print(f"local assets: {'OFF (Sionna-bundled only; public-figure safe)' if args.no_local_assets else 'ON (licence-unestablished pool -- see F21/F51)'}")
        print(f"seed:         {args.seed}   quant_bits: {args.quant_bits}")
        from e2e.environment.rt_scene_build import (DEFAULT_ANTENNA_PATTERN,
                                                    DEFAULT_GROUND_SCATTERING_COEFFICIENT)
        print(f"coherent targets: {'OFF (pre-2026-08-17 bug)' if args.no_coherent_targets else 'on'}"
              f"   antenna pattern: "
              f"{args.antenna_pattern or DEFAULT_ANTENNA_PATTERN + ' (default)'}"
              f"   ground S: "
              f"{DEFAULT_GROUND_SCATTERING_COEFFICIENT if args.ground_scattering is None else args.ground_scattering}"
              f"   samples_per_src: {args.samples_per_src or 'sionna default'}")
        from e2e.radar_config import answerability_problems
        spec = RT_DIFFICULTY_TIERS.get(args.tier)
        if spec is not None:
            problems = answerability_problems(cfg, top_speed_mps=spec.speed_mps[1])
            if problems:
                print("answerable:   NO -- the real run will REFUSE without "
                      "--allow-unanswerable:")
                for prob in problems:
                    print(f"                - {prob}")
            else:
                print("answerable:   yes (Doppler unaliased, azimuth tolerance within "
                      "the Rayleigh limit)")
        out_root = Path(args.out) if args.out is not None else DATASETS_DIR
        print(f"out:          {out_root / f'{args.config}_{args.tier}'}  (NOT written -- dry-run)")
        print("this path ray-traces with Sionna RT -- see report/chain_integration_design.html")
        print("=" * 70)
        return 0

    manifest_path = generate_chain_corpus(
        args.config, args.tier, args.n, out_dir=args.out, seed=args.seed,
        frames_per_scene=args.frames_per_scene, use_rffe=not args.no_rffe,
        use_interconnect=not args.no_interconnect, quant_bits=args.quant_bits,
        coherent_targets=not args.no_coherent_targets,
        antenna_pattern=args.antenna_pattern,
        ground_scattering_coefficient=args.ground_scattering,
        samples_per_src=args.samples_per_src,
        use_transmit_chain=args.transmit_chain,
        use_local_assets=not args.no_local_assets,
        allow_unanswerable=args.allow_unanswerable,
        use_if_hpf=not args.no_if_hpf,
        if_hpf_kwargs=(None if args.if_hpf_corner_range is None
                       else {"corner_range_m": args.if_hpf_corner_range}),
    )
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
