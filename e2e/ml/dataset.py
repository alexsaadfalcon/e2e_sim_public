"""
FMCW radar ML dataset builder: scene sampling -> synthesis -> labeled `.npz` frames.

This module is the top of the `e2e.ml` stack: it wires together `radar_config`
(waveform), `scenes` (synthetic scenario sampling by difficulty tier), `scatterers`
(scenario -> point targets), `rd_synth` (ADC synthesis), `transforms` (ADC -> network
input), and `labels` (targets -> a dense detection-label grid) into a reusable,
on-disk training dataset.

Sample format
-------------
`generate_sample(cfg, scenario, grid, ...)` returns one frame as a dict:

    {
      "input":   float32 CPU tensor, [2*C, R, D] -- see `e2e.ml.transforms.rd_to_input`
                 (real channels then imaginary channels; C/R/D depend on `cfg.mimo`:
                 TDM de-interleaves to the virtual array first, so C = n_virtual and
                 D = n_chirps_per_tx; DDMA/single use the raw ADC, C = n_rx, D = n_chirps.
                 R is always `cfg.n_samples`),
      "labels":  float32 CPU tensor, [3, grid.n_range, grid.n_azimuth] -- see
                 `e2e.ml.labels.encode_detection_labels`,
      "targets": list of (range_m, sin_az, object_class) tuples, one per scene
                 scatterer that falls inside the label grid (`e2e.ml.labels.targets_in_grid`),
      "meta":    small dict of scalar provenance (frame_idx, snr_db, seed, cfg.name,
                 cfg.mimo, radar pose).
    }

On-disk dataset layout
-----------------------
`generate_dataset(cfg_name, tier, n_frames, ...)` draws `n_frames` independent scenes
(one per frame -- each frame is its own single-scene, single-instant sample, not a
multi-frame track) via `e2e.ml.scenes.sample_scene`, synthesizes each with
`generate_sample`, and writes one compressed `.npz` per frame to
`<out_dir>/<cfg_name>_<tier>/frame_?????.npz` with three arrays:

    input : float32 [2*C, R, D]
    labels: float32 [3, grid.n_range, grid.n_azimuth]
    meta  : 0-d unicode array holding a JSON string (frame meta + "targets" +
            "scene" == `e2e.ml.scenes.scene_summary(scenario)`)

...plus a `manifest.json` describing the run and the deterministic train/val/test
split (see `_split_bounds`): `{"config": RadarConfig.to_dict(), "tier": str,
"grid": dataclasses.asdict(LabelGrid), "snr_db": float, "seed": int,
"files": {"train": [...], "val": [...], "test": [...]}}` (filenames only, relative to
the manifest's own directory).

`RadarFrameDataset` is a thin `torch.utils.data.Dataset` over one manifest split,
lazily loading each `.npz` in `__getitem__`.

`e2e.ml.labels` / `e2e.ml.scenes` are imported lazily (inside functions), not at
module scope, so `import e2e.ml.dataset` does not hard-fail if either sibling module
is not yet present in the working tree.

CLI
---
    python -m e2e.ml.dataset --config ti_iwr1443 --tier D1 --n 100 --seed 0 \\
        [--out DIR] [--snr-db 30] [--dry-run]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Gitignored (see .gitignore: `e2e/ml/datasets/`); computed relative to this file so it
# resolves correctly regardless of the caller's working directory.
DATASETS_DIR = Path(__file__).resolve().parent / "datasets"


def _json_default(obj):
    """`json.dump(..., default=_json_default)` fallback for numpy scalars/arrays.

    `targets_in_grid`/`scene_summary` (owned by sibling modules) are not guaranteed to
    return only plain-Python types -- numpy scalars (e.g. `np.float64` from an array
    element) are a common, easy-to-miss source of `TypeError: not JSON serializable`.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


# --------------------------------------------------------------------------------
# Per-frame sample synthesis
# --------------------------------------------------------------------------------
LABEL_CLASSES = ("vehicle", "pedestrian")
"""Object classes that become detection ground truth. Background clutter
(object_class "scatterer") contributes SIGNAL to the synthesized frame but is
deliberately excluded from labels/targets -- a detector must learn to reject
clutter, not report it. Without this filter, D2/D3 scenes were 70-80% clutter
in their own ground truth (adversarial-review finding)."""


def generate_sample(cfg, scenario, grid, *, frame_idx: int = 0, snr_db: Optional[float] = 30.0,
                    seed: Optional[int] = None, device=None,
                    label_classes: Sequence[str] = LABEL_CLASSES) -> Dict[str, Any]:
    """Synthesize one labeled frame: `scenario` @ `frame_idx` -> network input + labels.

    `grid` is an `e2e.ml.labels.LabelGrid` (typically `LabelGrid.for_config(cfg)`).
    Only scatterers whose class is in `label_classes` become ground truth (see
    `LABEL_CLASSES`); pass `None` to label everything, clutter included.
    Returns the sample dict documented in the module docstring.
    """
    from e2e.ml.labels import encode_detection_labels, targets_in_grid
    from e2e.ml.rd_synth import synthesize_adc
    from e2e.ml.scatterers import frame_scatterers, radar_pose
    from e2e.ml.transforms import adc_to_rd, rd_to_input, tdm_deinterleave

    dt = 1.0 / cfg.frame_rate_hz
    scatterers = frame_scatterers(scenario, frame_idx, dt=dt)
    pose = radar_pose(scenario, frame_idx)

    adc = synthesize_adc(cfg, scatterers, pose, snr_db=snr_db, seed=seed, device=device)

    if cfg.mimo == "tdm":
        sub_cfg = dataclasses.replace(cfg, n_tx=1, mimo="single", n_chirps=cfg.n_chirps_per_tx)
        rd = adc_to_rd(sub_cfg, tdm_deinterleave(cfg, adc))
    else:
        rd = adc_to_rd(cfg, adc)
    x = rd_to_input(rd).to("cpu")

    # encode_detection_labels places its output on the library device (cuda if
    # available) with no upstream tensor to inherit a device from -- move to cpu
    # before any numpy conversion, matching the "input"/"labels" cpu-tensor contract.
    labels = torch.as_tensor(encode_detection_labels(grid, scatterers, pose,
                                                     classes=label_classes),
                             dtype=torch.float32).cpu()
    targets = targets_in_grid(grid, scatterers, pose, classes=label_classes)

    meta = {
        "frame_idx": int(frame_idx),
        "snr_db": None if snr_db is None else float(snr_db),
        "seed": None if seed is None else int(seed),
        "config": cfg.name,
        "mimo": cfg.mimo,
        "pose_position": list(pose.position),
        "pose_boresight": list(pose.boresight),
    }
    return {"input": x, "labels": labels, "targets": targets, "meta": meta}


# --------------------------------------------------------------------------------
# Deterministic split
# --------------------------------------------------------------------------------
def _split_bounds(n: int, splits: Tuple[float, ...]) -> List[int]:
    """`n` items, `splits` fractions (need not sum to exactly 1) -> cumulative-floor
    boundaries `[0, b1, b2, ..., n]` such that `files[b_i:b_{i+1}]` is split `i`.

    Cumulative-floor (not per-split floor/round) keeps every split a contiguous,
    order-preserving slice and guarantees the boundaries sum to exactly `n` with no
    remainder-distribution tie-breaking: e.g. n=6, splits=(0.8, 0.1, 0.1) ->
    cumulative fractions (0.8, 0.9, 1.0) -> floor(4.8, 5.4, 6.0) = (4, 5, 6) ->
    bounds [0, 4, 5, 6] -> train=4, val=1, test=1.
    """
    fracs = np.asarray(splits, dtype=float)
    fracs = fracs / fracs.sum()
    cum = np.cumsum(fracs)
    bounds = [0] + [int(np.floor(c * n)) for c in cum]
    bounds[-1] = n  # last boundary always closes out any rounding shortfall
    return bounds


# --------------------------------------------------------------------------------
# Dataset generation
# --------------------------------------------------------------------------------
def generate_dataset(cfg_name: str, tier: str, n_frames: int, out_dir=None, *,
                     seed: int = 0, snr_db: Optional[float] = 30.0, device=None,
                     splits: Tuple[float, ...] = (0.8, 0.1, 0.1),
                     range_stride: int = 4, n_azimuth: int = 192) -> Path:
    """Generate `n_frames` labeled frames for `(cfg_name, tier)` and write a manifest.

    Each frame draws its own scene: `rng = np.random.default_rng(seed + i)`,
    `scenario = sample_scene(cfg, tier, rng)`, `generate_sample(..., seed=seed + i)`.
    Frames are written as `frame_?????.npz` under `<out_dir>/<cfg_name>_<tier>/`; the
    train/val/test split is the deterministic `_split_bounds` slice over frame index
    (not shuffled), so results are exactly reproducible for a given `(seed, n_frames)`.

    Returns the path to the written `manifest.json`.
    """
    from e2e.ml.labels import LabelGrid
    from e2e.ml.radar_config import PRESETS
    from e2e.ml.scenes import DIFFICULTY_TIERS, sample_scene, scene_summary

    if cfg_name not in PRESETS:
        raise ValueError(f"unknown radar config {cfg_name!r}; choices: {sorted(PRESETS)}")
    if tier not in DIFFICULTY_TIERS:
        raise ValueError(f"unknown difficulty tier {tier!r}; choices: {sorted(DIFFICULTY_TIERS)}")
    cfg = PRESETS[cfg_name]

    grid = LabelGrid.for_config(cfg, range_stride=range_stride, n_azimuth=n_azimuth)

    out_root = Path(out_dir) if out_dir is not None else DATASETS_DIR
    dataset_dir = out_root / f"{cfg_name}_{tier}"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    filenames: List[str] = []
    for i in range(n_frames):
        rng = np.random.default_rng(seed + i)
        scenario = sample_scene(cfg, tier, rng)
        sample = generate_sample(cfg, scenario, grid, frame_idx=0, snr_db=snr_db,
                                 seed=seed + i, device=device)

        meta = dict(sample["meta"])
        meta["targets"] = sample["targets"]
        meta["scene"] = scene_summary(scenario)

        fname = f"frame_{i:05d}.npz"
        np.savez_compressed(
            dataset_dir / fname,
            input=sample["input"].numpy(),
            labels=sample["labels"].numpy(),
            meta=np.array(json.dumps(meta, default=_json_default)),
        )
        filenames.append(fname)

    bounds = _split_bounds(n_frames, splits)
    files = {
        "train": filenames[bounds[0]:bounds[1]],
        "val": filenames[bounds[1]:bounds[2]],
        "test": filenames[bounds[2]:bounds[3]],
    }

    manifest = {
        "config": cfg.to_dict(),
        "tier": tier,
        "grid": dataclasses.asdict(grid) if dataclasses.is_dataclass(grid) else {
            "n_range": grid.n_range, "n_azimuth": grid.n_azimuth,
        },
        "snr_db": snr_db,
        "seed": seed,
        "label_classes": list(LABEL_CLASSES),
        "files": files,
    }
    manifest_path = dataset_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=_json_default)
    return manifest_path


# --------------------------------------------------------------------------------
# torch Dataset
# --------------------------------------------------------------------------------
class RadarFrameDataset(torch.utils.data.Dataset):
    """A `torch.utils.data.Dataset` over one split of a `generate_dataset` manifest.

    `__getitem__` lazily loads the frame's `.npz` and returns
    `(input, labels)` as float32 tensors (no augmentation/normalization -- callers
    compose that on top, e.g. via `e2e.ml.transforms.normalize`).
    """

    def __init__(self, manifest_path, split: str = "train"):
        self.manifest_path = Path(manifest_path)
        with open(self.manifest_path) as f:
            self.manifest = json.load(f)
        if split not in self.manifest["files"]:
            raise ValueError(
                f"unknown split {split!r}; choices: {sorted(self.manifest['files'])}"
            )
        self.split = split
        self.files: List[str] = self.manifest["files"][split]
        self.dataset_dir = self.manifest_path.parent

    def __len__(self) -> int:
        return len(self.files)

    def _load(self, idx: int):
        path = self.dataset_dir / self.files[idx]
        with np.load(path) as data:
            return data["input"], data["labels"], json.loads(str(data["meta"].item()))

    def __getitem__(self, idx: int):
        inp, labels, _meta = self._load(idx)
        return torch.from_numpy(inp).to(torch.float32), torch.from_numpy(labels).to(torch.float32)

    def targets(self, idx: int):
        """Decoded target list (`(range_m, sin_az, object_class)` tuples) for frame `idx`."""
        _inp, _labels, meta = self._load(idx)
        return meta["targets"]


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m e2e.ml.dataset",
        description="Generate a labeled FMCW radar range-Doppler dataset.",
    )
    p.add_argument("--config", required=True, help="radar config preset name (see e2e.ml.radar_config.PRESETS)")
    p.add_argument("--tier", required=True, help="difficulty tier (see e2e.ml.scenes.DIFFICULTY_TIERS)")
    p.add_argument("--n", type=int, required=True, help="number of frames to generate")
    p.add_argument("--seed", type=int, default=0, help="base RNG seed (frame i uses seed + i)")
    p.add_argument("--out", default=None, help="output root directory (default: e2e/ml/datasets)")
    p.add_argument("--snr-db", type=float, default=30.0, help="synthesis SNR in dB (see synthesize_adc)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the generation plan without synthesizing/writing anything")
    return p


def main(argv: Optional[List[str]] = None) -> int:
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
    tier_spec = DIFFICULTY_TIERS[args.tier]

    if args.dry_run:
        from e2e.ml.labels import LabelGrid

        grid = LabelGrid.for_config(cfg)
        if cfg.mimo == "tdm":
            c, d = cfg.n_virtual, cfg.n_chirps_per_tx
        else:
            c, d = cfg.n_rx, cfg.n_chirps
        input_shape = (2 * c, cfg.n_samples, d)
        labels_shape = (3, grid.n_range, grid.n_azimuth)
        bytes_per_frame = (
            np.prod(input_shape, dtype=np.int64) * 4 + np.prod(labels_shape, dtype=np.int64) * 4
        )
        print("=" * 70)
        print(f"config:       {args.config}  (mimo={cfg.mimo}, n_virtual={cfg.n_virtual}, "
              f"range_res={cfg.range_resolution_m:.4f}m, max_range={cfg.max_range_m:.2f}m, "
              f"vel_res={cfg.velocity_resolution_mps:.4f}m/s, max_vel={cfg.max_velocity_mps:.2f}m/s)")
        print(f"tier:         {args.tier}  {tier_spec!r}")
        print(f"input shape:  {input_shape} float32")
        print(f"labels shape: {labels_shape} float32")
        print(f"frames:       {args.n}")
        print(f"est. size:    {bytes_per_frame * args.n / 1e6:.2f} MB (uncompressed; .npz is compressed)")
        print(f"seed:         {args.seed}   snr_db: {args.snr_db}")
        out_root = Path(args.out) if args.out is not None else DATASETS_DIR
        print(f"out:          {out_root / f'{args.config}_{args.tier}'}  (NOT written -- dry-run)")
        print("=" * 70)
        return 0

    manifest_path = generate_dataset(
        args.config, args.tier, args.n, out_dir=args.out, seed=args.seed, snr_db=args.snr_db,
    )
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
