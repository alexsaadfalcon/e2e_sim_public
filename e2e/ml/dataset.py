"""
FMCW radar ML dataset builder: scene sampling -> synthesis -> labeled `.npz` frames.

This module is the top of the `e2e.ml` stack: it wires together `radar_config`
(waveform), `scenes` (synthetic scenario sampling by difficulty tier), `scatterers`
(scenario -> point targets), `rd_synth` (ADC synthesis), `transforms` (ADC -> network
input), and `labels` (targets -> a dense detection-label grid) into a reusable,
on-disk training dataset.

NOT THE CORPUS PATH ANYMORE. `generate_sample`/`generate_dataset` below call the
closed-form point-target synthesizer (`rd_synth.synthesize_adc`) directly -- no RFFE,
no interconnect, no dechirp block, none of the chain. They are kept as an explicit,
labelled CI/offline FALLBACK (fast, no GPU, no Sionna) for tests and quick shape/plumbing
checks. The real corpus generation path is `e2e.ml.chain_generate`, which composes the
SAME block chain (`e2e.environment.blocks.RTEnvironmentBlock` -> `e2e.blocks.CircuitStage`/
`InterconnectStage` -> `e2e.chain.dechirp.DechirpBlock` -> impairments/quantizer ->
`e2e.chain.receive.RadarCubeBlock` -> `e2e.ml.blocks.SinkBlock`) as an `e2e.simulation.
Simulation` run -- see `report/chain_integration_design.html`. `write_manifest` below is
the shared manifest-writing tail both paths use, so the two producers agree on one
on-disk schema (see "On-disk dataset layout").

Sample format
-------------
`generate_sample(cfg, scenario, grid, ...)` returns one frame as a dict:

    {
      "adc":     raw complex64 CPU tensor, [n_rx, n_chirps, n_samples] -- the
                 synthesizer's ADC output, UNPROCESSED (no deinterleave/FFT). This is
                 the on-disk source of truth (see "On-disk dataset layout" below);
                 kept in the returned dict too so a caller synthesizing one-off frames
                 doesn't need to re-derive it.
      "input":   float32 CPU tensor, [2*C, R, D] -- see `e2e.ml.transforms.rd_to_input`
                 (real channels then imaginary channels; C/R/D depend on `cfg.mimo`:
                 TDM de-interleaves to the virtual array first, so C = n_virtual and
                 D = n_chirps_per_tx; DDMA/single use the raw ADC, C = n_rx, D = n_chirps.
                 R is always `cfg.n_samples`). Kept for direct/one-off use and back-compat;
                 NOT written to disk by `generate_dataset` (`RadarFrameDataset` re-derives
                 it from "adc" at load time -- see below).
      "labels":  float32 CPU tensor, [3, grid.n_range, grid.n_azimuth] -- see
                 `e2e.ml.labels.encode_detection_labels`,
      "targets": list of (range_m, sin_az, object_class) tuples, one per scene
                 scatterer that falls inside the label grid (`e2e.ml.labels.targets_in_grid`),
      "meta":    small dict of scalar provenance (frame_idx, snr_db, seed, cfg.name,
                 cfg.mimo, radar pose) PLUS "target_extras": a list parallel to
                 "targets" (same order/length) of {"rcs_dbsm", "velocity_mps"} dicts --
                 see `_target_extras`.
    }

On-disk dataset layout (manifest_version 2)
--------------------------------------------
`generate_dataset(cfg_name, tier, n_frames, ...)` draws `n_frames` independent SCENES
via `e2e.ml.scenes.sample_scene`; each scene yields `frames_per_scene` (default 1)
consecutive frames (`frame_idx` 0..frames_per_scene-1) of the SAME scene -- for
`frames_per_scene > 1` the scene's moving objects carry a real motion track (see
`sample_scene`'s `n_frames` parameter), so consecutive frames are physically
consistent, not independent draws. Frames are written to
`<out_dir>/<cfg_name>_<tier>/` as one compressed `.npz` each:

    frame_?????.npz            (frames_per_scene == 1, unchanged naming)
    frame_?????_t??.npz        (frames_per_scene > 1: scene index _ frame-in-scene index)

with three arrays:

    adc   : complex64 [n_rx, n_chirps, n_samples] -- raw ADC, NOT a derived tensor.
            (ADC-native storage: storing the raw signal instead of a precomputed "input"
            makes the transform code, not a disk snapshot, the source of truth for
            "input" -- see `RadarFrameDataset`. On-disk size is essentially unchanged
            from the old "input" npz: both are dense, incompressible float/complex data
            of the same total byte count, just reshaped by the FFT. Written/read through
            `e2e.ml.storage` -- see that module for the measured codec choice/tradeoffs
            -- so a cube that happens to verify as uniformly quantized compresses
            substantially further, exactly losslessly; `meta["codec"]` records which
            representation is actually on disk, and this is invisible to callers of
            `RadarFrameDataset`, which always returns the reconstructed complex64 array.)
    labels: float32 [3, grid.n_range, grid.n_azimuth]
    meta  : 0-d unicode array holding a JSON string (frame meta + "targets" +
            "target_extras" + "scene" == `e2e.ml.scenes.scene_summary(scenario)` +
            "codec"/"codec_meta" -- see `e2e.ml.storage.write_sample_npz`)

...plus a `manifest.json` describing the run and the deterministic train/val/test
split (see `_split_bounds`, applied at the SCENE level so a sequence is never split
across train/val/test): `{"manifest_version": 2, "config": RadarConfig.to_dict(),
"tier": str, "grid": dataclasses.asdict(LabelGrid), "snr_db": float, "seed": int,
"frames_per_scene": int, "files": {"train": [...], "val": [...], "test": [...]},
"sequences": [[frame_?????.npz, ...], ...]}` -- "files" lists are flat (filenames
only, relative to the manifest's own directory, exactly as before); "sequences" groups
them by scene (one inner list per scene, in frame order) for future sequence-aware
consumers -- `RadarFrameDataset` itself stays per-frame; sequence-aware loading is a
documented follow-up, not implemented here.

Deliberately NOT written: a precomputed "input" array (space -- `RadarFrameDataset`
derives it at load time) and an "input_format" manifest field (which derivation to use
is the LOADER's choice, not a property of the stored corpus -- the same on-disk corpus
serves both `input_format="rd"` and `input_format="adc"` consumers).

Back-compat: a manifest/npz pair written before this change ("manifest_version"
absent or 1, npz has "input" not "adc") still loads via `RadarFrameDataset` for
`input_format="rd"` (the array is returned as-is, no re-derivation); `input_format=
"adc"` on such a dataset raises a clear `ValueError` (there is no raw ADC to return --
regenerate the corpus).

`RadarFrameDataset` is a thin `torch.utils.data.Dataset` over one manifest split,
lazily loading each `.npz` in `__getitem__`.

`e2e.ml.labels` / `e2e.ml.scenes` are imported lazily (inside functions), not at
module scope, so `import e2e.ml.dataset` does not hard-fail if either sibling module
is not yet present in the working tree.

CLI
---
    python -m e2e.ml.dataset --config ti_iwr1443 --tier D1 --n 100 --seed 0 \\
        [--out DIR] [--snr-db 30] [--frames-per-scene 1] [--dry-run]
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

from e2e.ml import storage

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


def _target_extras(grid, scatterers, pose, classes) -> List[Dict[str, Any]]:
    """Per-target `{"rcs_dbsm", "velocity_mps"}`, one entry per `targets_in_grid`
    tuple, in the SAME order (so callers can `zip(meta["targets"],
    meta["target_extras"])`).

    Deliberately duplicates (rather than imports) `labels.py`'s private
    `_range_sin_az`/`_in_grid` geometry -- a few lines of plain vector math -- so this
    module does not reach into a sibling module's underscore-prefixed internals. The
    sampled RCS/velocity used to synthesize a frame are otherwise discarded after
    synthesis (baked into the ADC's signal amplitude/phase only, not separably
    recoverable from it), so this is the one place they can be cheaply recorded.
    """
    import numpy as _np

    from e2e.ml.rd_synth import array_axis

    keep = None if classes is None else set(classes)
    origin = _np.asarray(pose.position, dtype=_np.float64)
    axis = array_axis(pose)
    out: List[Dict[str, Any]] = []
    for sc in scatterers:
        if keep is not None and sc.object_class not in keep:
            continue
        los = _np.asarray(sc.position, dtype=_np.float64) - origin
        r = float(_np.linalg.norm(los))
        sin_az = 0.0 if r < 1e-6 else float((los / r) @ axis)
        if not (0.0 <= r < grid.max_range_m and abs(sin_az) < 1.0):
            continue
        out.append({
            "rcs_dbsm": float(sc.rcs_dbsm),
            "velocity_mps": [float(v) for v in sc.velocity],
        })
    return out


def generate_sample(cfg, scenario, grid, *, frame_idx: int = 0, snr_db: Optional[float] = 30.0,
                    seed: Optional[int] = None, device=None,
                    label_classes: Sequence[str] = LABEL_CLASSES) -> Dict[str, Any]:
    """Synthesize one labeled frame: `scenario` @ `frame_idx` -> network input + labels.

    ANALYTIC FALLBACK, NOT THE CORPUS PATH: this calls `rd_synth.synthesize_adc`'s
    closed-form point-target model directly -- it realizes none of the RFFE/
    interconnect/dechirp chain stages (see the module docstring). Kept for CI/offline
    use (fast, no GPU/Sionna needed) and as the reference the chain path's output
    shape/schema must stay compatible with. Real corpus generation is
    `e2e.ml.chain_generate.generate_chain_corpus`.

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
    target_extras = _target_extras(grid, scatterers, pose, label_classes)

    meta = {
        "frame_idx": int(frame_idx),
        "snr_db": None if snr_db is None else float(snr_db),
        "seed": None if seed is None else int(seed),
        "config": cfg.name,
        "mimo": cfg.mimo,
        "pose_position": list(pose.position),
        "pose_boresight": list(pose.boresight),
        "target_extras": target_extras,
    }
    return {"adc": adc.cpu(), "input": x, "labels": labels, "targets": targets, "meta": meta}


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
                     range_stride: int = 4, n_azimuth: int = 192,
                     frames_per_scene: int = 1) -> Path:
    """Generate `n_frames` independent SCENES for `(cfg_name, tier)` and write a manifest.

    ANALYTIC FALLBACK, NOT THE CORPUS PATH -- see `generate_sample`'s docstring and the
    module docstring's "NOT THE CORPUS PATH ANYMORE" note. Real corpus generation is
    `e2e.ml.chain_generate.generate_chain_corpus`, which writes the SAME on-disk schema
    (via `write_manifest`, the manifest-writing tail factored out below) by running the
    composed block chain instead of calling `generate_sample` per frame.

    Each scene draws its own `rng = np.random.default_rng(seed + i)` and
    `scenario = sample_scene(cfg, tier, rng, n_frames=frames_per_scene)`; when
    `frames_per_scene == 1` (default) this is exactly the original one-frame-per-scene
    behavior. When `frames_per_scene > 1`, the SAME scene yields `frames_per_scene`
    consecutive frames (`generate_sample(..., frame_idx=t)` for `t` in
    `range(frames_per_scene)`) -- moving objects carry a real motion track (see
    `sample_scene`), so the frames are a physically consistent sequence, not
    independent draws. Frames are written as `frame_?????.npz` (frames_per_scene == 1)
    or `frame_?????_t??.npz` under `<out_dir>/<cfg_name>_<tier>/`.

    The train/val/test split (`_split_bounds`) is applied at the SCENE level (not
    shuffled) so a sequence's frames always land together in one split -- splitting a
    sequence across train/test would leak the sequence's identity/motion into both.
    Each per-frame synthesis seed is `seed + i * frames_per_scene + t` (distinct per
    frame, so repeated frames of one sequence don't get identical noise realizations),
    so results are exactly reproducible for a given `(seed, n_frames, frames_per_scene)`.

    Returns the path to the written `manifest.json`.
    """
    from e2e.ml.labels import LabelGrid
    from e2e.ml.radar_config import PRESETS
    from e2e.ml.scenes import DIFFICULTY_TIERS, sample_scene, scene_summary

    if cfg_name not in PRESETS:
        raise ValueError(f"unknown radar config {cfg_name!r}; choices: {sorted(PRESETS)}")
    if tier not in DIFFICULTY_TIERS:
        raise ValueError(f"unknown difficulty tier {tier!r}; choices: {sorted(DIFFICULTY_TIERS)}")
    if frames_per_scene < 1:
        raise ValueError(f"frames_per_scene must be >= 1, got {frames_per_scene}")
    cfg = PRESETS[cfg_name]

    grid = LabelGrid.for_config(cfg, range_stride=range_stride, n_azimuth=n_azimuth)

    out_root = Path(out_dir) if out_dir is not None else DATASETS_DIR
    dataset_dir = out_root / f"{cfg_name}_{tier}"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    sequences: List[List[str]] = []
    for i in range(n_frames):
        rng = np.random.default_rng(seed + i)
        scenario = sample_scene(cfg, tier, rng, n_frames=frames_per_scene)
        scene_meta = scene_summary(scenario)

        scene_files: List[str] = []
        for t in range(frames_per_scene):
            frame_seed = seed + i * frames_per_scene + t
            sample = generate_sample(cfg, scenario, grid, frame_idx=t, snr_db=snr_db,
                                     seed=frame_seed, device=device)

            meta = dict(sample["meta"])
            meta["scene_index"] = i
            meta["targets"] = sample["targets"]
            meta["scene"] = scene_meta

            fname = f"frame_{i:05d}.npz" if frames_per_scene == 1 else f"frame_{i:05d}_t{t:02d}.npz"
            # See e2e.ml.storage: compresses "adc" via its measured-best lossless
            # codec when the array happens to verify as uniformly quantized (this
            # analytic-fallback path never runs QuantizerBlock, so in practice this
            # is almost always CODEC_RAW -- unchanged from the old np.savez_compressed
            # call this replaces -- but the reader's contract stays codec-agnostic).
            storage.write_sample_npz(
                dataset_dir / fname,
                {"adc": sample["adc"].numpy(), "labels": sample["labels"].numpy()},
                meta, payload_key="adc", json_default=_json_default,
            )
            scene_files.append(fname)
        sequences.append(scene_files)

    return write_manifest(dataset_dir, cfg, tier, sequences, grid=grid, seed=seed,
                          snr_db=snr_db, frames_per_scene=frames_per_scene, splits=splits)


def write_manifest(dataset_dir, cfg, tier: str, sequences: List[List[str]], *,
                   grid=None, seed: int = 0, snr_db: Optional[float] = None,
                   frames_per_scene: int = 1, splits: Tuple[float, ...] = (0.8, 0.1, 0.1),
                   label_classes: Sequence[str] = LABEL_CLASSES) -> Path:
    """Write a manifest_version-2 `manifest.json` for a corpus already written to
    `dataset_dir` -- the manifest-writing tail factored out of `generate_dataset` so
    OTHER producers of the same on-disk schema (namely `e2e.ml.chain_generate`, which
    writes frames by running the composed block chain instead of calling
    `generate_sample`) share exactly one manifest contract instead of duplicating it.

    `sequences` is a list of per-scene filename lists (frame order within a scene),
    exactly `generate_dataset`'s own `sequences` structure (see the module docstring's
    "On-disk dataset layout") -- filenames relative to `dataset_dir`, files not
    validated/touched here. The split (`_split_bounds`) is applied at the SCENE level,
    same as `generate_dataset`. `grid` is optional (an `e2e.ml.labels.LabelGrid` or
    None if the caller has none to record).
    """
    dataset_dir = Path(dataset_dir)
    bounds = _split_bounds(len(sequences), splits)  # scene-level bounds
    files = {
        "train": [f for scene_files in sequences[bounds[0]:bounds[1]] for f in scene_files],
        "val": [f for scene_files in sequences[bounds[1]:bounds[2]] for f in scene_files],
        "test": [f for scene_files in sequences[bounds[2]:bounds[3]] for f in scene_files],
    }
    if grid is None:
        grid_dict = {}
    elif dataclasses.is_dataclass(grid):
        grid_dict = dataclasses.asdict(grid)
    else:
        grid_dict = {"n_range": grid.n_range, "n_azimuth": grid.n_azimuth}

    manifest = {
        "manifest_version": 2,
        "config": cfg.to_dict(),
        "tier": tier,
        "grid": grid_dict,
        "snr_db": snr_db,
        "seed": seed,
        "frames_per_scene": frames_per_scene,
        "label_classes": list(label_classes) if label_classes is not None else None,
        "files": files,
        "sequences": sequences,
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

    `__getitem__` lazily loads the frame's `.npz` and returns `(input, labels)` as
    float32 tensors (no augmentation/normalization -- callers compose that on top,
    e.g. via `e2e.ml.transforms.normalize`).

    `input_format` selects how the network-input tensor is derived from the on-disk
    frame (a manifest_version-2 corpus stores raw ADC, not a precomputed "input" --
    see the module docstring):

    * `"rd"` (default) -- range-Doppler: `tdm_deinterleave` (only for `cfg.mimo ==
      "tdm"`) -> `adc_to_rd` -> `rd_to_input`, i.e. exactly what `generate_sample`
      used to precompute and store; now derived per-`__getitem__` instead (pure,
      deterministic tensor ops -- no RNG, safe under any `num_workers`). Matches
      `_input_dims`'s `(2*n_virtual, n_samples, n_chirps_per_tx)` / `(2*n_rx,
      n_samples, n_chirps)` shape (TDM / DDMA-or-single respectively).
    * `"adc"` -- raw physical-channel ADC, real/imag-stacked channel-first:
      `[2*n_rx, n_samples, n_chirps]` (note: samples/chirps axis order matches "rd"'s
      `[C, R, D]` convention, but the ADC's native storage order is
      `[n_rx, n_chirps, n_samples]` -- transposed here). Deliberately does NOT run
      `tdm_deinterleave`: that reordering is itself a hand-engineered MIMO-demux step
      (round-robin reorder by known TX index), and feeding it in would defeat the
      "let the model learn the raw-signal structure" premise this format exists to
      serve, the same way `adc_to_rd`'s FFT would. This IS a fork from the upstream
      reference's literal DDMA raw-ADC premise for TDM configs (`ti_iwr1443`):
      adjacent chirps in the raw sequence come from different, non-simultaneous TX
      antennas with an abrupt `n_tx`-chirp periodicity baked in raw -- an open
      research question, not resolved here (see the raw-ADC input-format design
      notes; flagged, not silently smoothed over).

    Per-`input_format` note: RD and raw-ADC have very different per-channel
    statistics (RD has FFT coherent-gain peaks; raw ADC is closer to AWGN + weak beat
    tones) -- normalization constants (`e2e.ml.transforms.input_stats`) must be
    computed/stored SEPARATELY per `input_format`, never shared across the two.

    `in_memory_cache=True` opt-in caches each `__getitem__`'s derived `(input,
    labels)` tensors in a plain `dict[idx -> tensors]` after first access -- avoids
    recomputing the same "rd" derivation (a few ms of CPU FFT) every epoch, at the
    cost of holding the whole accessed split resident in memory. Off by default: this
    is a reference/tutorial dataset, and silently growing memory with dataset size is
    the wrong default for it.

    Back-compat: a manifest_version-1 corpus (npz has "input", not "adc") still loads
    for `input_format="rd"` (the stored array is returned as-is, no re-derivation);
    `input_format="adc"` on such a corpus raises `ValueError` (there is no raw ADC on
    disk to derive from -- regenerate the corpus).
    """

    def __init__(self, manifest_path, split: str = "train", input_format: str = "rd",
                in_memory_cache: bool = False):
        if input_format not in ("rd", "adc"):
            raise ValueError(f"input_format must be 'rd' or 'adc', got {input_format!r}")
        self.manifest_path = Path(manifest_path)
        with open(self.manifest_path) as f:
            self.manifest = json.load(f)
        if split not in self.manifest["files"]:
            raise ValueError(
                f"unknown split {split!r}; choices: {sorted(self.manifest['files'])}"
            )
        self.split = split
        self.input_format = input_format
        self.in_memory_cache = in_memory_cache
        self.files: List[str] = self.manifest["files"][split]
        self.dataset_dir = self.manifest_path.parent
        self._cache: Dict[int, Any] = {}
        self._cfg = None  # lazily built RadarConfig, only needed for input_format="rd"

    def __len__(self) -> int:
        return len(self.files)

    def _radar_config(self):
        if self._cfg is None:
            from e2e.ml.radar_config import RadarConfig

            self._cfg = RadarConfig.from_dict(self.manifest["config"])
        return self._cfg

    def _load_raw(self, idx: int):
        """`(array, is_adc, meta)`: `array` is either raw ADC ("adc" key or, if
        `e2e.ml.storage`-compressed, "adc_code_re"/"adc_code_im" -- see below) or the
        precomputed "input" ("input" key, v1 back-compat); `is_adc` says which.

        Format is detected from which ARRAY KEYS are actually present, not from
        `meta["codec"]` -- `storage.read_payload` already dispatches on that key
        internally (defaulting to `CODEC_RAW` when absent, see its docstring), so
        this only needs to know whether an "adc"-shaped payload is on disk at all
        versus a v1 corpus that only ever wrote "input". Using presence-of-arrays
        (rather than "codec" in meta) also means a stray/copied `meta["codec"]` value
        with no matching array (e.g. a hand-built v1 fixture derived from a v2 file)
        can't be mistaken for an ADC-native corpus.
        """
        path = self.dataset_dir / self.files[idx]
        with np.load(path) as data:
            meta = json.loads(str(data["meta"].item()))
            if "adc" in data or "adc_code_re" in data:
                return storage.read_payload(data, meta, "adc"), True, data["labels"], meta
            return data["input"], False, data["labels"], meta

    def _derive_input(self, array: np.ndarray, is_adc: bool) -> torch.Tensor:
        if not is_adc:
            # v1 corpus: "input" was already the requested "rd" derivation (the only
            # format v1 ever wrote); input_format="adc" is rejected before we get here.
            return torch.from_numpy(array).to(torch.float32)

        adc = torch.from_numpy(array).to(torch.complex64)  # [n_rx, n_chirps, n_samples]
        if self.input_format == "adc":
            # Raw physical channels, no deinterleave (see class docstring): transpose
            # to [n_rx, n_samples, n_chirps] first so the stacked-channel axis order
            # matches "rd"'s [C, R, D] (range-like axis before doppler-like axis).
            adc_rsd = adc.transpose(1, 2)
            return torch.cat([adc_rsd.real, adc_rsd.imag], dim=0).to(torch.float32)

        # input_format == "rd": re-derive exactly what generate_sample used to
        # precompute (deterministic, no RNG -- safe under any num_workers).
        from e2e.ml.transforms import adc_to_rd, rd_to_input, tdm_deinterleave

        cfg = self._radar_config()
        if cfg.mimo == "tdm":
            sub_cfg = dataclasses.replace(cfg, n_tx=1, mimo="single", n_chirps=cfg.n_chirps_per_tx)
            rd = adc_to_rd(sub_cfg, tdm_deinterleave(cfg, adc))
        else:
            rd = adc_to_rd(cfg, adc)
        return rd_to_input(rd).to("cpu")

    def _load(self, idx: int):
        array, is_adc, labels, meta = self._load_raw(idx)
        if not is_adc and self.input_format == "adc":
            raise ValueError(
                f"{self.files[idx]!r} is a manifest_version-1 frame (no raw 'adc' "
                "array on disk) -- input_format='adc' needs a regenerated corpus"
            )
        x = self._derive_input(array, is_adc)
        y = torch.from_numpy(labels).to(torch.float32)
        return x, y, meta

    def __getitem__(self, idx: int):
        if self.in_memory_cache and idx in self._cache:
            return self._cache[idx]
        x, y, _meta = self._load(idx)
        if self.in_memory_cache:
            self._cache[idx] = (x, y)
        return x, y

    def targets(self, idx: int):
        """Decoded target list (`(range_m, sin_az, object_class)` tuples) for frame `idx`.

        Reads only the npz's "meta" entry -- `np.load`'s `NpzFile` decompresses each
        array lazily per-key access, so skipping "adc"/"input"/"labels" here avoids
        the (for "adc") multi-MB decompression `__getitem__` needs but `targets()`
        does not.
        """
        path = self.dataset_dir / self.files[idx]
        with np.load(path) as data:
            meta = json.loads(str(data["meta"].item()))
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
    p.add_argument("--frames-per-scene", type=int, default=1,
                   help="consecutive motion-consistent frames drawn per scene (default: 1, "
                        "independent single-instant scenes; see generate_dataset)")
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
        # ADC-native storage (manifest_version 2): the on-disk array is the raw ADC,
        # not the derived "input" -- see the module docstring. complex64 = 8 bytes.
        adc_shape = (cfg.n_rx, cfg.n_chirps, cfg.n_samples)
        labels_shape = (3, grid.n_range, grid.n_azimuth)
        bytes_per_frame = (
            np.prod(adc_shape, dtype=np.int64) * 8 + np.prod(labels_shape, dtype=np.int64) * 4
        )
        total_frames = args.n * args.frames_per_scene
        print("=" * 70)
        print(f"config:       {args.config}  (mimo={cfg.mimo}, n_virtual={cfg.n_virtual}, "
              f"range_res={cfg.range_resolution_m:.4f}m, max_range={cfg.max_range_m:.2f}m, "
              f"vel_res={cfg.velocity_resolution_mps:.4f}m/s, max_vel={cfg.max_velocity_mps:.2f}m/s)")
        print(f"tier:         {args.tier}  {tier_spec!r}")
        print(f"adc shape:    {adc_shape} complex64  (on-disk array; 'input' is derived at load time)")
        print(f"labels shape: {labels_shape} float32")
        print(f"scenes:       {args.n}  x  frames_per_scene={args.frames_per_scene}  "
              f"= {total_frames} frames")
        print(f"est. size:    {bytes_per_frame * total_frames / 1e6:.2f} MB (uncompressed; .npz is compressed)")
        print(f"seed:         {args.seed}   snr_db: {args.snr_db}")
        out_root = Path(args.out) if args.out is not None else DATASETS_DIR
        print(f"out:          {out_root / f'{args.config}_{args.tier}'}  (NOT written -- dry-run)")
        print("=" * 70)
        return 0

    manifest_path = generate_dataset(
        args.config, args.tier, args.n, out_dir=args.out, seed=args.seed, snr_db=args.snr_db,
        frames_per_scene=args.frames_per_scene,
    )
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
