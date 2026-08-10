"""
Radar-ML corpus generation AS A COMPOSED `e2e.simulation.Simulation` RUN.

This is the migration `report/chain_integration_design.html`'s "The result" section
specifies: the corpus generator no longer calls the analytic point-target synthesizer
(`e2e.ml.rd_synth.synthesize_adc`, still available as `e2e.ml.dataset`'s explicitly-
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
from e2e.chain.receive import ImpairmentBlock, QuantizerBlock, RadarCubeBlock
from e2e.environment.blocks import RTEnvironmentBlock
from e2e.ml.blocks import SinkBlock
from e2e.ml.dataset import DATASETS_DIR
from e2e.simulation import Simulation

DEFAULT_LABEL_CLASSES = ("vehicle", "pedestrian")


# --------------------------------------------------------------------------------
# ImpairmentBlock -> SinkBlock provenance glue
# --------------------------------------------------------------------------------
def default_domain_randomizer(
    *, phase_noise_dbc_hz_range: Tuple[float, float] = (-95.0, -75.0),
    leakage_relative_db_range: Tuple[float, float] = (-10.0, -2.0),
    clutter_relative_db_range: Tuple[float, float] = (-16.0, -4.0),
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
    device=None, k: int = 1,
) -> Simulation:
    """Compose ONE radar-ML `Simulation` run (see module docstring for the block list).

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
        scenario, cfg, device=device, label_grid=label_grid, label_classes=label_classes,
    )

    serial_stages: List[Any] = []
    if use_rffe:
        kwargs = dict(rffe_kwargs or {})
        kwargs.setdefault("n", int(cfg.n_rx))
        serial_stages.append(CircuitStage(RFFEBlock(**kwargs)))
    if use_interconnect:
        serial_stages.append(InterconnectStage(InterconnectBlock(**(interconnect_kwargs or {}))))
    serial_stages.append(DechirpBlock(cfg))
    serial_stages.append(
        ImpairmentBlock(cfg, impairment_chain_params, seed=impairment_seed)
    )
    serial_stages.append(QuantizerBlock(bits=quant_bits))

    downstream_blocks = [RadarCubeBlock(cfg), SinkBlock(out_dir, tag=tag)]

    return Simulation(
        environment_block=env,
        downstream_blocks=downstream_blocks,
        k=k,
        serial_stages=serial_stages,
    )


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
    """
    from e2e.ml.dataset import write_manifest
    from e2e.ml.labels import LabelGrid
    from e2e.ml.radar_config import PRESETS
    from e2e.ml.scenes import DIFFICULTY_TIERS, sample_scene

    if cfg_name not in PRESETS:
        raise ValueError(f"unknown radar config {cfg_name!r}; choices: {sorted(PRESETS)}")
    if tier not in DIFFICULTY_TIERS:
        raise ValueError(f"unknown difficulty tier {tier!r}; choices: {sorted(DIFFICULTY_TIERS)}")
    if frames_per_scene < 1:
        raise ValueError(f"frames_per_scene must be >= 1, got {frames_per_scene}")
    cfg = PRESETS[cfg_name]

    grid = LabelGrid.for_config(cfg, range_stride=range_stride, n_azimuth=n_azimuth)
    randomize = randomizer if randomizer is not None else default_domain_randomizer()

    out_root = Path(out_dir) if out_dir is not None else DATASETS_DIR
    dataset_dir = out_root / f"{cfg_name}_{tier}"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    sequences: List[List[str]] = []
    for i in range(n_scenes):
        rng = np.random.default_rng(seed + i)
        scenario = sample_scene(cfg, tier, rng, n_frames=frames_per_scene)
        tag = f"sample_scene{i:05d}"

        sim = build_chain_simulation(
            scenario, cfg, dataset_dir, tag=tag,
            use_rffe=use_rffe, use_interconnect=use_interconnect,
            rffe_kwargs=rffe_kwargs, interconnect_kwargs=interconnect_kwargs,
            impairment_chain_params=randomize, impairment_seed=seed + i * frames_per_scene,
            quant_bits=quant_bits, label_grid=grid, label_classes=label_classes, device=device,
        )
        sim.run(n_steps=frames_per_scene)

        scene_files = [f"{tag}_frame_{t:05d}.npz" for t in range(frames_per_scene)]
        sequences.append(scene_files)

    return write_manifest(dataset_dir, cfg, tier, sequences, grid=grid, seed=seed,
                          snr_db=None, frames_per_scene=frames_per_scene, splits=splits,
                          label_classes=label_classes or ())


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
    p.add_argument("--config", required=True, help="radar config preset name (see e2e.ml.radar_config.PRESETS)")
    p.add_argument("--tier", required=True, help="difficulty tier (see e2e.ml.scenes.DIFFICULTY_TIERS)")
    p.add_argument("--n", type=int, required=True, help="number of scenes to generate")
    p.add_argument("--seed", type=int, default=0, help="base RNG seed (scene i uses seed + i)")
    p.add_argument("--out", default=None, help="output root directory (default: e2e/ml/datasets)")
    p.add_argument("--frames-per-scene", type=int, default=1,
                   help="consecutive motion-consistent frames per scene (default: 1)")
    p.add_argument("--no-rffe", action="store_true", help="disable the RFFE front-end stage")
    p.add_argument("--no-interconnect", action="store_true", help="disable the interconnect stage")
    p.add_argument("--quant-bits", type=int, default=12, help="ADC quantizer bit depth")
    p.add_argument("--dry-run", action="store_true",
                   help="print the generation plan without ray-tracing/writing anything")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    import sys

    args = build_arg_parser().parse_args(argv)

    from e2e.ml.radar_config import PRESETS

    if args.config not in PRESETS:
        print(f"unknown --config {args.config!r}; choices: {sorted(PRESETS)}", file=sys.stderr)
        return 2
    cfg = PRESETS[args.config]

    from e2e.ml.scenes import DIFFICULTY_TIERS

    if args.tier not in DIFFICULTY_TIERS:
        print(f"unknown --tier {args.tier!r}; choices: {sorted(DIFFICULTY_TIERS)}", file=sys.stderr)
        return 2

    if args.dry_run:
        total_frames = args.n * args.frames_per_scene
        print("=" * 70)
        print(f"config:       {args.config}  (mimo={cfg.mimo}, n_rx={cfg.n_rx}, n_tx={cfg.n_tx})")
        print(f"tier:         {args.tier}")
        print(f"scenes:       {args.n}  x  frames_per_scene={args.frames_per_scene}  "
              f"= {total_frames} frames")
        print(f"rffe:         {'off' if args.no_rffe else 'on'}   "
              f"interconnect: {'off' if args.no_interconnect else 'on'}")
        print(f"seed:         {args.seed}   quant_bits: {args.quant_bits}")
        out_root = Path(args.out) if args.out is not None else DATASETS_DIR
        print(f"out:          {out_root / f'{args.config}_{args.tier}'}  (NOT written -- dry-run)")
        print("this path ray-traces with Sionna RT -- see report/chain_integration_design.html")
        print("=" * 70)
        return 0

    manifest_path = generate_chain_corpus(
        args.config, args.tier, args.n, out_dir=args.out, seed=args.seed,
        frames_per_scene=args.frames_per_scene, use_rffe=not args.no_rffe,
        use_interconnect=not args.no_interconnect, quant_bits=args.quant_bits,
    )
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
