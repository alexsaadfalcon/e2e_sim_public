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
      "input":   float32 CPU tensor, [2*C, R, D] -- see `e2e.chain.transforms.rd_to_input`
                 (real channels then imaginary channels; C/R/D depend on `cfg.mimo`:
                 TDM de-interleaves to the virtual array first, so C = n_virtual and
                 D = n_chirps_per_tx; DDMA/single use the raw ADC, C = n_rx, D = n_chirps.
                 R is always `cfg.n_samples`). Kept for direct/one-off use and back-compat;
                 NOT written to disk by `generate_dataset` (`RadarFrameDataset` re-derives
                 it from "adc" at load time -- see below).
      "labels":  float32 CPU tensor, [3, grid.n_range, grid.n_azimuth] -- see
                 `e2e.ml.labels.encode_detection_labels`,
      "targets": list of (range_m, sin_az, object_class, surface_range_m,
                 cross_range_half_extent_m) tuples, one per scene scatterer inside the
                 label grid (`e2e.ml.labels.targets_in_grid`; the last element was
                 added 2026-08-27 -- additive, see that function's docstring),
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
import hashlib
import json
import subprocess
import sys
import warnings
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


def _stable_scene_seed(corpus_tag: str, scene_index: int, seed: int) -> int:
    """A `numpy.random.Generator` seed salted with `corpus_tag`, mirroring
    `e2e.environment.rt_scenes._stable_seed` (see that function's docstring for the full
    rationale). Plain `seed + scene_index` (this module's ORIGINAL scheme) has the
    same latent bug the RT path had: two corpora built with the same `seed` draw
    IDENTICAL scenes at every `scene_index`, which is exactly how
    `rt_ablation_txoff`'s val+test split ended up duplicated inside two other
    corpora's train splits. `corpus_tag` is REQUIRED, not defaulted, so a caller
    cannot silently reintroduce the collision by forgetting to pass one.

    Not shared code with `rt_scenes._stable_seed` (different call signature -- this
    one has no `tier`, since `corpus_tag` already encodes it via the
    `f"{cfg_name}_{tier}"` dataset-directory convention `generate_dataset` uses) but
    deliberately the same construction (SHA-256 over a stable string, not Python's
    salted `hash()`) for the same reasons.
    """
    digest = hashlib.sha256(f"{corpus_tag}:{int(scene_index)}:{int(seed)}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _read_git_commit() -> str:
    """Short SHA of HEAD right now, or `"unknown"` if git/the repo is unavailable.

    Provenance-only: a long corpus-generation run must never crash over this, so any
    failure (git not installed, not a repo, detached weirdness, etc.) is swallowed.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent, capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            sha = out.stdout.strip()
            if sha:
                return sha
    except Exception:
        pass
    return "unknown"


#: HEAD captured ONCE at import time -- i.e. (as close as observable to) the code that
#: is actually IN MEMORY doing the generating. The first B1 corpus (2026-08-25) shipped
#: FALSE provenance because the old implementation shelled out at manifest-WRITE time:
#: commits landed in the working tree while the multi-hour background run was still
#: going, so the manifest named a commit whose changes (new meta keys) were provably
#: absent from every frame (caught by the corpus adversarial review). Import-time
#: capture cannot see a mid-run `git commit` either -- nothing can, from inside -- but
#: it reports the state the process STARTED from, which is what a regeneration needs.
#: The residual hazard (editing the working tree during a live run changes lazily
#: imported modules) is a process rule, not detectable here: never commit or edit the
#: repo while a generation job is running.
_GENERATOR_GIT_COMMIT_AT_IMPORT = _read_git_commit()


def _generator_git_commit() -> str:
    """The provenance SHA recorded in manifests: HEAD as of process/module start."""
    return _GENERATOR_GIT_COMMIT_AT_IMPORT


# --------------------------------------------------------------------------------
# Per-frame sample synthesis
# --------------------------------------------------------------------------------
LABEL_CLASSES = ("vehicle", "pedestrian")
"""Object classes that become detection ground truth. Background clutter
(object_class "scatterer") contributes SIGNAL to the synthesized frame but is
deliberately excluded from labels/targets -- a detector must learn to reject
clutter, not report it. Without this filter, D2/D3 scenes were 70-80% clutter
in their own ground truth (adversarial-review finding)."""

# Geometric-correspondence tolerances for `RadarFrameDataset.unlabelled_objects`:
# same values `e2e.ml.detect_viz` already uses (`_SCENE_MATCH_RANGE_M` /
# `_SCENE_MATCH_SIN_AZ`) for tagging the same generator object list as
# labelled/unlabelled, not re-derived independently, so the two call sites agree
# about which objects a corpus's label set actually covers.
_UNLABELLED_MATCH_RANGE_M = 0.75
_UNLABELLED_MATCH_SIN_AZ = 0.02


def _object_surface_range_m(obj: Dict[str, Any], pos: np.ndarray, radar_pos: np.ndarray,
                            centre_range_m: float) -> float:
    """SURFACE range of a generator-placed `obj` (a `scene_provenance` object dict), to
    match the convention `e2e.ml.metrics.match_detections` uses for a bare `(range_m,
    sin_azimuth)` ignore entry -- see `RadarFrameDataset.unlabelled_objects`'s docstring
    for why that convention, not centre range, is the one an ignore entry must carry.

    Uses the same extent/surface-point helpers `e2e.ml.labels.target_geometry` uses for
    real targets (`e2e.environment.geometry.object_extent_m`/`nearest_surface_point`),
    duck-typing `obj` (a plain JSON dict) into the attribute-style object those helpers
    expect. Yaw is taken as 0 rather than resolved through `object_yaw_rad` (which needs
    the full scene plus a per-scenario heading seed to answer for a parked object):
    exact for `ObjectKind.SPHERE`/`ObjectKind.BOX` -- the kinds behind almost every
    unlabelled object (`clutter-box-*`), since both have equal x/y extents, so their
    ellipsoid surface point is yaw-invariant -- and merely an approximation for an
    asymmetric mesh (car/pedestrian) object that happens to have no matching label.
    `centre_range_m` (the pre-fix value) if the object's geometry is unknown, i.e. a
    point target, for which surface == centre exactly as `target_geometry` defines it.
    """
    from types import SimpleNamespace

    from e2e.environment.geometry import nearest_surface_point, object_extent_m

    proxy = SimpleNamespace(kind=obj.get("kind", ""), asset=obj.get("asset"),
                            scaling=obj.get("scaling", 1.0), extent_m=None)
    extent = object_extent_m(proxy)
    if extent is None:
        return centre_range_m
    half = tuple(0.5 * float(e) for e in extent)
    surface = nearest_surface_point(pos, half, radar_pos, yaw_rad=0.0)
    return float(np.linalg.norm(surface - radar_pos))


def _target_extras(grid, scatterers, pose, classes) -> List[Dict[str, Any]]:
    """Per-target `{"rcs_dbsm", "velocity_mps"}`, one entry per `targets_in_grid`
    tuple, in the SAME order (so callers can `zip(meta["targets"],
    meta["target_extras"])`).

    Calls `labels.target_geometry` (the encoder's OWN public geometry helper) rather than
    re-deriving range/azimuth: this list must stay index-for-index parallel to
    `targets_in_grid`, and since 2026-08-17 the in-grid test is made on the target's
    SURFACE point, not its centre -- a re-derivation would silently drift at the grid
    boundary. The sampled RCS/velocity used to synthesize a frame are otherwise discarded
    after synthesis (baked into the ADC's signal amplitude/phase only, not separably
    recoverable from it), so this is the one place they can be cheaply recorded.
    """
    from e2e.ml.labels import target_geometry

    keep = None if classes is None else set(classes)
    out: List[Dict[str, Any]] = []
    for sc in scatterers:
        if keep is not None and sc.object_class not in keep:
            continue
        r_surface, sin_az, _r_centre = target_geometry(sc, pose)
        if not (0.0 <= r_surface < grid.max_range_m and abs(sin_az) < 1.0):
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
    from e2e.environment.scatterers import frame_scatterers, radar_pose
    from e2e.ml.labels import encode_detection_labels, targets_in_grid
    from e2e.chain.rd_synth import synthesize_adc
    from e2e.chain.transforms import adc_to_rd, rd_to_input, tdm_deinterleave

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

    Each scene draws its own `rng = np.random.default_rng(_stable_scene_seed(corpus_tag,
    i, seed))` (`corpus_tag` is the dataset directory name, e.g. `f"{cfg_name}_{tier}"`
    -- see that function's docstring for why plain `seed + i` is not enough) and
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
    so results are exactly reproducible for a given `(seed, n_frames, frames_per_scene)`
    -- the SCENE content additionally depends on `corpus_tag` (`out_dir`'s
    `<cfg_name>_<tier>` directory name), so two calls with the same `seed` but
    different `out_dir`/`cfg_name`/`tier` draw different scenes, by design.

    Returns the path to the written `manifest.json`.
    """
    from e2e.ml.labels import LabelGrid
    from e2e.radar_config import PRESETS
    from e2e.ml.scenes import DIFFICULTY_TIERS, sample_scene, scene_summary

    if cfg_name not in PRESETS:
        raise ValueError(f"unknown radar config {cfg_name!r}; choices: {sorted(PRESETS)}")
    if tier not in DIFFICULTY_TIERS:
        raise ValueError(f"unknown difficulty tier {tier!r}; choices: {sorted(DIFFICULTY_TIERS)}")
    if frames_per_scene < 1:
        raise ValueError(f"frames_per_scene must be >= 1, got {frames_per_scene}")
    cfg = PRESETS[cfg_name]

    # Answerability check (release-plan A3). This is the analytic FALLBACK, used by
    # plumbing tests and quick experiments on deliberately tiny configs, so an
    # unanswerable pair WARNS here instead of refusing the way the real corpus path
    # (`e2e.ml.chain_generate.generate_chain_corpus`) does. The warning is the tripwire:
    # a corpus generated under it must never back a detection benchmark (F43).
    from e2e.radar_config import answerability_problems
    spec = DIFFICULTY_TIERS[tier]
    top_speed = max(spec.vehicle_speed_mps[1], spec.pedestrian_speed_mps[1])
    problems = answerability_problems(cfg, top_speed_mps=top_speed)
    if problems:
        warnings.warn(
            f"({cfg_name}, {tier}) cannot support a detection benchmark (F43): "
            + "; ".join(problems)
            + ". Generating anyway -- this is the analytic fallback path -- but do not "
              "benchmark detection on this corpus.",
            stacklevel=2)

    grid = LabelGrid.for_config(cfg, range_stride=range_stride, n_azimuth=n_azimuth)

    out_root = Path(out_dir) if out_dir is not None else DATASETS_DIR
    dataset_dir = out_root / f"{cfg_name}_{tier}"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    corpus_tag = dataset_dir.name
    sequences: List[List[str]] = []
    for i in range(n_frames):
        rng = np.random.default_rng(_stable_scene_seed(corpus_tag, i, seed))
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
                          snr_db=snr_db, frames_per_scene=frames_per_scene, splits=splits,
                          corpus_tag=corpus_tag)


def resolve_input_scale(manifest: Dict[str, Any], input_format: str, *,
                        where: str = "manifest") -> float:
    """The constant a network input is DIVIDED by for this corpus and format.

    One function, used by `RadarFrameDataset` (training and scoring) AND by
    `e2e.ml.blocks.NeuralDetectorBlock` (the GUI). Until 2026-09-22 the block had its own
    input derivation with no scale at all, so the GUI fed a checkpoint inputs on a
    different scale from the ones it was trained on: on the same benchmark frame the
    objectness peak was 0.12 in the GUI against 0.49 from the scoring path, and the demo's
    ML preset drew zero detections. A constant that lives in two places is two constants.

    Resolution order (unchanged from the dataset's original inline logic):
      * "rad" is self-normalising by construction (dB relative to each frame's own median
        power), so it is pinned at 1.0 -- applying a global constant would be a second,
        contradictory normalisation (F83, and the note in `_derive_input`).
      * `input_scale_by_format[input_format]` when the manifest records per-format scales;
        a format the manifest does not list is refused rather than borrowed from another
        (F80: that rescales the input by orders of magnitude).
      * a legacy single `input_scale` (measured on "rd") is honoured for "rd" only.
      * a corpus predating the constant gets 1.0, the regime it was trained and scored in.
    """
    by_format = manifest.get("input_scale_by_format") or {}
    if input_format == "rad":
        scale = 1.0
    elif by_format:
        if input_format not in by_format:
            raise ValueError(
                f"{where} records input scales for {sorted(by_format)} but not for "
                f"input_format={input_format!r}. Applying another format's constant would "
                "rescale the input by orders of magnitude (see F80); re-run "
                "dataset.finalize_input_scale.")
        scale = float(by_format[input_format])
    elif manifest.get("input_scale"):
        if input_format != "rd":
            raise ValueError(
                f"{where} records a single input_scale measured for input_format='rd', "
                f"but this dataset was opened as {input_format!r}. Re-run "
                "dataset.finalize_input_scale to record both.")
        scale = float(manifest["input_scale"])
    else:
        scale = 1.0
    if scale <= 0.0:
        raise ValueError(f"{where} records a non-positive input_scale ({scale!r}); "
                         "it must be a positive float")
    return scale


def measure_input_scale(manifest_path, *, n_sample: int = 32,
                        split: str = "train", input_format: str = "rd") -> float:
    """The ONE constant every consumer divides the network input by (F63, piece 2).

    MEASURED IN THE RANGE-DOPPLER DOMAIN, through the production load path, because the
    obvious analytic constant is wrong by orders of magnitude and was measured to be so.

    The first version of this function returned `sqrt(thermal_noise_power_w(cfg))` -- the
    ADC-domain thermal RMS -- on the reasoning that dividing by the noise floor expresses
    the cube in units of its own noise. That reasoning is right about the ADC and wrong
    about the NETWORK INPUT, which is the range-Doppler transform of the ADC: two FFTs of
    coherent processing gain, plus the RF front end's own chain gain, sit between them.
    Measured on `b1_bench_v3`, that constant left the network input at std 702 with
    excursions to 8e4, against std 0.107 for the corpus that trained successfully.
    Thirteen epochs of FFTRadNet on it scored val_AP 0.0000 at every epoch.

    So the constant is measured where it is applied. `n_sample` frames of `split` are
    loaded through `RadarFrameDataset` itself -- not a reimplementation of the transform
    chain, so the measured quantity is exactly what the network receives -- and the
    returned scale sets their pooled RMS to 1.

    WHY THIS IS NOT THE DEFECT F63 FIXED. F63 was a PER-FRAME normalisation: each frame
    divided by its own mean, which destroys the relative loudness of frames and hence any
    dependence of SNR on transmit power, range or noise figure. This is ONE constant for a
    whole corpus. Every frame is divided by the same number, so a louder frame arrives at
    the network louder, which is the property `physical_scale=True` exists to preserve.
    The `notes/F63_FIX_PLAN.md` sketch called for a config-derived constant; that is the
    one part of the plan the measurement overruled, and the deviation is deliberate.

    Stored in the manifest at generation time so scoring, visualisation and any later
    re-training resolve the SAME number rather than each recomputing it -- the divergence
    that F52 was.
    """
    ds = RadarFrameDataset(manifest_path, split=split, input_format=input_format)
    if len(ds) == 0:
        raise ValueError(f"{manifest_path} has no frames in split {split!r}")
    if ds.input_scale != 1.0:
        raise ValueError(
            f"{manifest_path} already records input_scale={ds.input_scale!r}; measuring "
            "through a dataset that is already normalising would compound the two. "
            "Measure on a manifest written without the field.")

    total_sq, total_n = 0.0, 0
    for i in range(min(int(n_sample), len(ds))):
        x, _y = ds[i]
        total_sq += float((x.to(torch.float64) ** 2).sum())
        total_n += x.numel()
    rms = float(np.sqrt(total_sq / total_n))
    if not np.isfinite(rms) or rms <= 0.0:
        raise ValueError(
            f"measured input RMS is {rms!r} over {total_n} values -- the corpus is empty, "
            "constant, or corrupt; refusing to write a scale that would divide by it.")
    return rms


def finalize_input_scale(manifest_path, *, n_sample: int = 32) -> float:
    """Measure the corpus's input scale and write it INTO its own manifest.

    Two-phase by necessity: the scale is measured through `RadarFrameDataset`, which needs
    a manifest to exist, so the manifest is written first WITHOUT the field and completed
    here. Callers that generate a corpus should always run this -- a manifest with no
    `input_scale` makes every consumer silently fall back to 1.0 and train on an
    unnormalised cube, which is the failure this whole mechanism exists to prevent.
    """
    manifest_path = Path(manifest_path)
    # One constant PER input_format. The RD cube and the raw ADC differ by two FFTs of
    # coherent processing gain, so a single number cannot serve both -- applying the RD
    # constant to ADC data is the same magnitude error F80 fixed, one path over.
    by_format = {}
    for fmt in ("rd", "adc"):
        try:
            by_format[fmt] = measure_input_scale(manifest_path, n_sample=n_sample,
                                                 input_format=fmt)
        except Exception:
            # A manifest_version-1 corpus has no raw ADC to load; "rd" is all it can offer.
            # Failing to measure a format simply means that format is not available here.
            continue
    if "rd" not in by_format:
        raise ValueError(f"could not measure an input scale for {manifest_path}")
    scale = by_format["rd"]
    manifest = json.loads(manifest_path.read_text())
    manifest["input_scale"] = scale                 # back-compat: the "rd" value
    manifest["input_scale_by_format"] = by_format
    manifest["input_scale_source"] = (
        f"measured RMS of the network input over {n_sample} train frames, PER "
        "input_format (see dataset.measure_input_scale)")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=_json_default)
    return scale


def write_manifest(dataset_dir, cfg, tier: str, sequences: List[List[str]], *,
                   grid=None, seed: int = 0, snr_db: Optional[float] = None,
                   frames_per_scene: int = 1, splits: Tuple[float, ...] = (0.8, 0.1, 0.1),
                   label_classes: Sequence[str] = LABEL_CLASSES,
                   corpus_tag: Optional[str] = None,
                   input_scale: Optional[float] = None) -> Path:
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

    `corpus_tag` records which corpus-identity salt (see `rt_scenes._stable_seed` /
    `_stable_scene_seed`) the frames in `dataset_dir` were actually drawn with;
    defaults to `dataset_dir.name` (both producers already name that directory
    `f"{cfg_name}_{tier}"`, the same string they pass as the seed salt), so callers
    that already follow that convention don't need to pass it explicitly. Recorded
    alongside `generator_git_commit` (short HEAD SHA, `"unknown"` if git is
    unavailable -- see `_generator_git_commit`) so "same seed, different code" is
    DETECTABLE from the manifest instead of silently producing near-duplicate scenes:
    `rt_scenes.py`'s own scene-sampling logic changed between two real generation
    runs that reused a seed, so same-index scenes from those runs were near-duplicates
    with no on-disk record of it.
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
        "corpus_tag": corpus_tag if corpus_tag is not None else dataset_dir.name,
        "generator_git_commit": _generator_git_commit(),
        # F63 piece 2, and ONLY written by a producer that actually put the cube on an
        # absolute scale (chain_generate with the link budget on). The analytic
        # generate_sample path never does, so its manifests carry no input_scale and
        # consumers fall back to 1.0 -- dividing synthetic arbitrary-unit data by a
        # thermal RMS would be a rescale with no physical meaning.
        **({} if input_scale is None else {
            "input_scale": float(input_scale),
            "input_scale_source": "measured -- see dataset.measure_input_scale",
        }),
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
    e.g. via `e2e.chain.transforms.normalize`).

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
    tones) -- normalization constants (`e2e.chain.transforms.input_stats`) must be
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
        if input_format not in ("rd", "adc", "rad"):
            raise ValueError(
                f"input_format must be 'rd', 'adc' or 'rad', got {input_format!r}")
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
        # F63 piece 2. One constant for the whole corpus, resolved by ONE function so
        # every consumer -- this dataset and the GUI's NeuralDetectorBlock -- divides by
        # the same number. See `resolve_input_scale`.
        self.input_scale = resolve_input_scale(self.manifest, input_format,
                                               where=str(self.manifest_path))

    def __len__(self) -> int:
        return len(self.files)

    def _radar_config(self):
        if self._cfg is None:
            from e2e.radar_config import RadarConfig

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

        if self.input_format == "rad":
            # RANGE-AZIMUTH-DOPPLER (added 2026-09-21, F83). The "rd" format hands the
            # network `[2*n_virtual, R, D]`, where azimuth exists ONLY as phase across the
            # virtual-channel axis. Neither shipped head converts that to an angle bin, so
            # both learn a fixed azimuth prior instead: objectness rank-1 energy fraction
            # 0.89/0.76 against 0.31 for ground truth, and azimuth-only AP no better than
            # a constant map. The information is present -- the classical beamformer
            # reaches AP 0.30 from the same ADC -- it is just never made SPATIAL.
            #
            # So reuse the classical arm's own front end, the one already known to be
            # sufficient, and give the network an axis aligned with the label grid.
            # Deliberately the same transform the CFAR baseline runs, so the comparison
            # is "what does learning add on top of the beamformer", not "can a network
            # rediscover beamforming from phase".
            #
            # FRONT-END PARITY, and an earlier version of this comment LIED about it.
            # It claimed "tdm_doppler_comp is left on AUTO to match the classical arm".
            # There is no AUTO at this level: `range_azimuth_power` defaults to
            # `doppler_notch_bins=0, tdm_doppler_comp=False`, and the AUTO logic lives one
            # level up in `classical_detection_map`. Passing neither meant the network was
            # fed a STRICTLY WEAKER front end than the baseline it was being compared to --
            # measured at 0.053 AP of the gap (CFAR scores 0.3006 with both steps and
            # 0.2473 with neither), 72% of it the zero-Doppler notch.
            #
            # So replicate the classical arm's AUTO decisions here rather than hardcoding,
            # keeping the two in step if either default is ever retuned.
            #
            # Separately: the TDM compensation is NOT why this format exists. Measured on
            # 88 real targets it moves target-vs-range-ring contrast by 0.15 dB (16.99 vs
            # 16.85 dB median). That hypothesis was tested and refuted; do not re-open it.
            # The notch is the part that carries weight.
            from e2e.ml.baseline import NOTCH_MIN_VMAX_MPS, range_azimuth_power

            cfg = self._radar_config()
            pw = range_azimuth_power(
                cfg, adc, keep_doppler=True,
                tdm_doppler_comp=(cfg.mimo == "tdm"),
                doppler_notch_bins=(1 if float(cfg.max_velocity_mps) >= NOTCH_MIN_VMAX_MPS
                                    else 0),
            )   # [A, R, D] real power
            # LOG-POWER, PER-FRAME REFERENCED. The linear cube spans ~40 dB frame to frame
            # (measured: per-frame RMS p5 0.243 / p95 5.155 after the single global
            # `input_scale`, crest factor ~130), so a handful of frames dominate every
            # gradient step. CFAR is immune to this by construction -- it is a CONSTANT
            # FALSE ALARM RATE detector, normalised against local background -- and that
            # is part of why it wins. Give the network the same scale-invariance instead
            # of a global constant it cannot adapt to.
            ref = torch.median(pw) + 1e-30
            x = 10.0 * torch.log10(torch.clamp(pw / ref, min=1e-12))
            return x.to(torch.float32)

        if self.input_format == "adc":
            # Raw physical channels, no deinterleave (see class docstring): transpose
            # to [n_rx, n_samples, n_chirps] first so the stacked-channel axis order
            # matches "rd"'s [C, R, D] (range-like axis before doppler-like axis).
            adc_rsd = adc.transpose(1, 2)
            return torch.cat([adc_rsd.real, adc_rsd.imag], dim=0).to(torch.float32)

        # input_format == "rd": re-derive exactly what generate_sample used to
        # precompute (deterministic, no RNG -- safe under any num_workers).
        from e2e.chain.transforms import adc_to_rd, rd_to_input, tdm_deinterleave

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
        x = x / self.input_scale
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
        """Decoded target list for `idx` -- `(range_m, sin_az, object_class,
        surface_range_m, cross_range_half_extent_m)` for a manifest written since
        2026-08-27, `(range_m, sin_az, object_class, surface_range_m)` (or shorter) for
        an older one; see `e2e.ml.labels.targets_in_grid`'s docstring for what each
        element means and which are additive.

        Reads only the npz's "meta" entry -- `np.load`'s `NpzFile` decompresses each
        array lazily per-key access, so skipping "adc"/"input"/"labels" here avoids
        the (for "adc") multi-MB decompression `__getitem__` needs but `targets()`
        does not.
        """
        path = self.dataset_dir / self.files[idx]
        with np.load(path) as data:
            meta = json.loads(str(data["meta"].item()))
        return meta["targets"]

    def unlabelled_objects(self, idx: int) -> List[Tuple[float, float]]:
        """`(range_m, sin_azimuth)` for every generator-PLACED object at frame `idx`
        that has no corresponding entry in `targets(idx)`.

        WHY: `LABEL_CLASSES` (`("vehicle", "pedestrian")`) filters what becomes ground
        truth, but the generator places other objects too -- background clutter
        (`object_class == "scatterer"`, mostly `clutter-box-*`) is roughly 34% of a
        typical scene's placed objects and contributes SIGNAL without ever appearing
        in `targets()`. A detector that correctly fires on one of these real,
        physically-present objects has no way to know it was never labelled, and
        should not be charged a false alarm for it. This method exists so a scorer can
        treat these positions as DON'T-CARE regions (excluded from the false-alarm
        count, not counted as a hit either) -- see `e2e.ml.metrics`.

        Reads `meta['scene_provenance']['scene']['objects']` (every object the
        generator placed) and `['nodes']` (one of which has `role == "radar"`, for its
        `position`) -- both written by `e2e.environment.blocks` (`SinkBlock`'s meta
        allowlist), so this ONLY works for chain-generated corpora (real Sionna RT or
        the analytic chain path), not the pre-2026 `generate_sample` analytic
        fallback, which never wrote scene provenance. Degrades to `[]` -- never raises
        -- for a frame whose meta has no `scene_provenance`, no `scene`, no `objects`,
        or no `radar`-role node: an older/incompatible corpus must still be loadable,
        just with no unlabelled-object information available.

        Coordinate convention, verified empirically against the generator (same one
        `e2e.ml.detect_viz.scene_objects_for_plot` uses, independently re-derived here
        rather than imported -- that module is visualization-layer and owned
        separately, and dataset.py should not depend downward on it): `d =
        object_position - radar_position`; `centre_range_m = norm(d)`; `sin_azimuth =
        d[1] / centre_range_m`. Round-trip-checked against a real frame (test split
        frame 69, `sphere-0` at `(26.97, -11.22, 0.499)`, radar at `(0, 0, 1.5)`): this
        gives `centre_range_m = 29.23`, `sin_azimuth = -0.3839`, matching that frame's
        first labelled target exactly (as it must -- `sphere-0` is a labelled pedestrian
        surrogate in that scene, not one of the objects this method returns). This
        centre range is used ONLY for the "is this the same object as a labelled
        target" identity check below, against `targets()`' own centre range (element 0)
        -- it is NOT what gets returned.

        The RETURNED range is the object's SURFACE range, not its centre range: `e2e.
        ml.metrics.match_detections` treats a bare `(range_m, sin_azimuth)` ignore entry
        as a point target, i.e. it reads `range_m` AS the surface range (`_surface_range`
        falls back to element 0 for any tuple with no 4th element -- see that module's
        `MatchCriterion` docstring and its "WHICH range" section, which is unconditionally
        surface range for every OTHER tuple kind the matcher scores). An unlabelled
        clutter object with real extent (`clutter-box-*`, `ObjectKind.BOX`) has a surface
        several tenths of a metre to a metre closer than its centre; reporting the centre
        range here would silently violate the matcher's own convention and under-cover
        the ignore region by exactly that offset. `_object_surface_range_m` computes it
        via `e2e.environment.geometry.object_extent_m`/`nearest_surface_point`, the same
        helpers `e2e.ml.labels.target_geometry` uses for real targets, with yaw taken as
        0 -- see that function's docstring for why that is exact for the box/sphere
        clutter this method mostly serves.

        An object "corresponds to a labelled target" iff some entry of `targets(idx)`
        is within `0.75` m of it in range AND `0.02` in sin-azimuth -- matched by
        GEOMETRY (position), not by name/class/index, because the label-encoding
        pipeline does not carry the generator's object identity through to the target
        tuple. These tolerances are the same values `e2e.ml.detect_viz` already uses
        for its own "labelled" tagging of the same object list, chosen there to be
        tight enough that two genuinely distinct nearby objects are not merged, loose
        enough to absorb the sub-bin regression/footprint slack in `targets()`'s own
        geometry (see `e2e.ml.labels`).
        """
        path = self.dataset_dir / self.files[idx]
        with np.load(path) as data:
            meta = json.loads(str(data["meta"].item()))

        prov = meta.get("scene_provenance")
        if not prov:
            return []
        scene = prov.get("scene") or {}
        objects = scene.get("objects") or []
        if not objects:
            return []
        nodes = scene.get("nodes") or []
        radar_nodes = [n for n in nodes if n.get("role") == "radar"]
        if not radar_nodes:
            return []
        radar_pos = np.asarray(radar_nodes[0].get("position", (0.0, 0.0, 0.0)), dtype=float)

        targets = meta.get("targets") or []
        target_ranges_sin_az = [(float(t[0]), float(t[1])) for t in targets]

        out: List[Tuple[float, float]] = []
        for obj in objects:
            pos = np.asarray(obj.get("position", (0.0, 0.0, 0.0)), dtype=float)
            d = pos - radar_pos
            centre_range_m = float(np.linalg.norm(d))
            sin_az = float(d[1] / centre_range_m) if centre_range_m > 1e-9 else 0.0
            labelled = any(
                abs(centre_range_m - r) <= _UNLABELLED_MATCH_RANGE_M
                and abs(sin_az - s) <= _UNLABELLED_MATCH_SIN_AZ
                for r, s in target_ranges_sin_az
            )
            if not labelled:
                surface_range_m = _object_surface_range_m(obj, pos, radar_pos, centre_range_m)
                out.append((surface_range_m, sin_az))
        return out


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m e2e.ml.dataset",
        description="Generate a labeled FMCW radar range-Doppler dataset.",
    )
    p.add_argument("--config", required=True, help="radar config preset name (see e2e.radar_config.PRESETS)")
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

    from e2e.radar_config import PRESETS

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
