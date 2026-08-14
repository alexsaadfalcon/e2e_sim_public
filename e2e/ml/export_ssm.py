"""
Export a slice of an RT-generated `e2e.ml` radar corpus for the SSMRadNet authors to
train on THEIR hardware.

STEP 1 (format decision, see this module's docstring for the reasoning): the upstream
`AnuvabSen1/SSMRadNet` repo does not document its expected on-disk (RADIal-style)
format anywhere in THIS repository precisely enough to replicate byte-for-byte -- the
port (`e2e/ml/models/ssm.py` / `ssmradnet.py`) only specifies the network's *tensor*
contract (`[2*C, R, D]` input, `[3, n_range, n_azimuth]` label map, see
`e2e/ml/README.md`'s "Data format" section), not a file layout. Guessing the RADIal
on-disk layout (a real, undocumented-here binary format with its own quirks) would risk
silently shipping something the authors' loader can't actually read. So: export uses
OUR documented layout (`e2e/ml/dataset.py`'s manifest_version-2 on-disk schema -- same
npz key names, same int16-code codec, `e2e/ml/storage.py`) plus a SELF-CONTAINED
`loader_ssm.py` (no `e2e` imports) dropped into the export directory, so the authors can
load it on their own hardware without installing this package. This is a deliberate,
documented choice, not a guess.

What this module does
----------------------
Given a manifest.json written by `e2e.ml.chain_generate.generate_chain_corpus` (the
REAL, ray-traced corpus path -- see that module and `e2e/ml/rt_scenes.py`), select the
first N *train*-split files (recorded verbatim) and, per sample:

  (a) COPY the raw ADC-cube npz byte-for-byte (`shutil.copy2` -- never re-encode, so the
      on-disk int16 codes + `codec_meta["scale"]` are exactly what the corpus wrote);
  (b) write a per-sample labels JSON (class, range/azimuth in several parameterizations,
      radial-frame x/y, footprint extent, label-grid row/col);
  (c) (once) write a top-level `README.md` describing the radar config, coordinate
      frame, file layout, npz key schema, int16 decode formula, provenance and license;
  (d) (once) copy a self-contained `loader_ssm.py` into the export directory.

Scene reconstruction / ground-truth verification
--------------------------------------------------
The RT corpus's stored `meta["targets"]` (written by
`e2e.environment.blocks.RTEnvironmentBlock.get_S_pars`, see that class) comes from
`e2e.ml.labels.targets_in_grid(grid, scats, pose, classes=...)` where `scats`/`pose`
are `e2e.ml.scatterers.frame_scatterers`/`radar_pose` resolved against
`e2e.ml.rt_scenes.build_rt_tier_scenario(tier, frame_idx=<scene index>, seed=<manifest
seed>, num_frames=<frames_per_scene>, use_local_assets=True)` -- see that function's
"Determinism" docstring section: the SAME `(tier, frame_idx, seed)` triple reproduces
the identical `Scenario`, needs no Sionna (`build_rt_tier_scenario`/`frame_scatterers`/
`radar_pose`/`targets_in_grid` are pure numpy/stdlib), and is fast. `use_local_assets`
matters even though asset identity doesn't affect target geometry: it changes the RNG
draw sequence (`_draw_vehicle_asset`'s `rng.integers(0, len(pool))` call consumes a
different number of internal RNG steps depending on the pool it draws from), so
reconstruction MUST pass the same `use_local_assets` the corpus generator used
(`generate_chain_corpus`'s default, `True`) or every subsequent placement draw
desynchronizes -- verified empirically (see this module's report) on `rt_radial_v2`.

Every sample's rebuilt `targets_in_grid(...)` is asserted equal (within a tight float
tolerance) to the npz's own stored `meta["targets"]` before that sample's labels/render
are written -- per this shard's brief, a mismatch STOPS the export (raises) rather than
silently shipping a wrong render.

CLI
---
    python -m e2e.ml.export_ssm --manifest <manifest.json> --n 1000 --out DIR \\
        [--no-render] [--seed-check-only]
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # noqa: E402 -- must precede any pyplot import; headless-safe

import numpy as np  # noqa: E402
import torch  # noqa: E402

from e2e.ml import storage  # noqa: E402
from e2e.ml.labels import LabelGrid, targets_in_grid  # noqa: E402
from e2e.ml.radar_config import RadarConfig  # noqa: E402
from e2e.ml.render_scene import _draw_birdseye, _draw_radar_view, range_azimuth_map  # noqa: E402
from e2e.ml.rt_scenes import VEHICLE_FOOTPRINT_M, build_rt_tier_scenario, vehicle_asset_class  # noqa: E402
from e2e.ml.scatterers import frame_scatterers, radar_pose  # noqa: E402
from e2e.scenario import ObjectKind  # noqa: E402

_FNAME_RE = re.compile(r"^(?P<tag>.+_scene(?P<scene>\d+))_frame_(?P<frame>\d+)\.npz$")

# Pedestrian footprint (length, width) metres -- mirrors `e2e.ml.rt_scenes._FOOTPRINT_M
# ["pedestrian"]` (a private module constant; duplicated here rather than imported, the
# same convention `e2e.ml.dataset._target_extras` uses for a sibling module's private
# geometry). Sphere footprint mirrors that same table's "sphere" entry.
_PEDESTRIAN_FOOTPRINT_M = (0.8, 0.6)
_SPHERE_FOOTPRINT_M = (1.0, 1.0)

_LABEL_CLASSES_DEFAULT = ("vehicle", "pedestrian")


# --------------------------------------------------------------------------------
# Manifest / filename helpers
# --------------------------------------------------------------------------------
def _parse_scene_frame(fname: str) -> Tuple[int, int]:
    """`sample_scene00042_frame_00000.npz` -> `(42, 0)`."""
    m = _FNAME_RE.match(fname)
    if not m:
        raise ValueError(
            f"{fname!r} does not match the expected 'sample_scene<N>_frame_<T>.npz' "
            "naming convention (see e2e.ml.chain_generate.generate_chain_corpus)"
        )
    return int(m.group("scene")), int(m.group("frame"))


def select_train_files(manifest: Dict[str, Any], n: int) -> List[str]:
    """First `n` train-split files, in manifest order (the EXACT list exported)."""
    train_files = manifest["files"]["train"]
    if n > len(train_files):
        raise ValueError(
            f"requested n={n} but the manifest's train split only has "
            f"{len(train_files)} files"
        )
    return list(train_files[:n])


# --------------------------------------------------------------------------------
# Scene reconstruction + GT verification
# --------------------------------------------------------------------------------
def reconstruct_scene(tier: str, seed: int, scene_idx: int, frame_idx: int,
                      frames_per_scene: int, *, use_local_assets: bool = True):
    """`(scenario, scatterers, pose)` for one stored sample -- see the module
    docstring's "Scene reconstruction" section for why these exact arguments
    reproduce the corpus generator's own scene byte-for-byte."""
    scenario = build_rt_tier_scenario(
        tier, frame_idx=scene_idx, seed=seed, num_frames=frames_per_scene,
        use_local_assets=use_local_assets,
    )
    dt = 1.0  # frame_scatterers' own default DEFAULT_DT_S; chain_generate never
    # overrides it (build_chain_simulation has no per-frame dt knob), so this matches.
    scats = frame_scatterers(scenario, frame_idx, dt=dt)
    pose = radar_pose(scenario, frame_idx)
    return scenario, scats, pose


def verify_gt_match(stored_targets: Sequence[Sequence[Any]],
                    rebuilt_targets: Sequence[Tuple[float, float, str]], *,
                    context: str, tol: float = 1e-6) -> None:
    """Raise `AssertionError` (STOP, per the brief) if the reconstructed scene's
    ground truth does not match the npz's own stored `meta["targets"]`."""
    stored = [(float(r), float(s), str(c)) for r, s, c in stored_targets]
    rebuilt = [(float(r), float(s), str(c)) for r, s, c in rebuilt_targets]
    if len(stored) != len(rebuilt):
        raise AssertionError(
            f"GT mismatch for {context}: stored {len(stored)} targets, "
            f"rebuilt {len(rebuilt)} -- scene reconstruction has diverged from the "
            f"corpus generator; STOPPING (see export_ssm.reconstruct_scene docstring)"
        )
    for (r0, s0, c0), (r1, s1, c1) in zip(stored, rebuilt):
        if c0 != c1 or abs(r0 - r1) > tol or abs(s0 - s1) > tol:
            raise AssertionError(
                f"GT mismatch for {context}: stored {(r0, s0, c0)} vs "
                f"rebuilt {(r1, s1, c1)} -- STOPPING (see export_ssm.reconstruct_scene "
                "docstring)"
            )


# --------------------------------------------------------------------------------
# Per-target extent (footprint) -- duplicated, not imported, geometry (see the module
# constants above for why).
# --------------------------------------------------------------------------------
def _target_extent_m(obj) -> Optional[Tuple[float, float]]:
    if obj.kind == ObjectKind.SPHERE:
        return _SPHERE_FOOTPRINT_M
    if obj.object_class == "vehicle":
        return VEHICLE_FOOTPRINT_M.get(vehicle_asset_class(obj.asset), VEHICLE_FOOTPRINT_M["car"])
    if obj.object_class == "pedestrian":
        return _PEDESTRIAN_FOOTPRINT_M
    return None


def _range_sin_az(position, pose) -> Tuple[float, float]:
    """Duplicates `e2e.ml.labels._range_sin_az` (private sibling-module geometry;
    same duplication convention `e2e.ml.dataset._target_extras` uses)."""
    from e2e.ml.rd_synth import array_axis

    origin = np.asarray(pose.position, dtype=np.float64).reshape(3)
    los = np.asarray(position, dtype=np.float64).reshape(3) - origin
    r = float(np.linalg.norm(los))
    if r < 1e-6:
        return r, 0.0
    sin_az = float((los / r) @ array_axis(pose))
    return r, sin_az


def build_target_records(grid: LabelGrid, scenario, scats, pose,
                         label_classes: Sequence[str]) -> List[Dict[str, Any]]:
    """Per-target label record, one per in-grid `label_classes` scatterer, in the
    SAME order `targets_in_grid` returns (`scenario.objects`/`scats` are 1:1, see
    `e2e.ml.scatterers.frame_scatterers`)."""
    keep = set(label_classes)
    range_bin_m = grid.range_bin_m
    az_bin = grid.az_bin
    records: List[Dict[str, Any]] = []
    for obj, sc in zip(scenario.objects, scats):
        if obj.object_class not in keep:
            continue
        r, sin_az = _range_sin_az(sc.position, pose)
        if not (0.0 <= r < grid.max_range_m and abs(sin_az) < 1.0):
            continue
        extent = _target_extent_m(obj)
        ci = min(int(r / range_bin_m), grid.n_range - 1)
        cj = min(int((sin_az + 1.0) / az_bin), grid.n_azimuth - 1)
        cos_az = float(np.sqrt(max(0.0, 1.0 - sin_az ** 2)))
        records.append({
            "class": obj.object_class,
            "range_m": r,
            "sin_azimuth": sin_az,
            "azimuth_deg": float(np.degrees(np.arcsin(np.clip(sin_az, -1.0, 1.0)))),
            "x_m": r * cos_az,
            "y_m": r * sin_az,
            "extent_m": None if extent is None else {"length": extent[0], "width": extent[1]},
            "rcs_dbsm": float(sc.rcs_dbsm),
            "velocity_mps": [float(v) for v in sc.velocity],
            "grid_row": ci,
            "grid_col": cj,
        })
    return records


# --------------------------------------------------------------------------------
# Rendering (bird's-eye + range-azimuth, side by side)
# --------------------------------------------------------------------------------
def render_sample(cfg: RadarConfig, grid: LabelGrid, scats, pose, adc: torch.Tensor,
                  out_path: Path, *, title: str = "", dpi: int = 120) -> Path:
    """One PNG: bird's-eye scene (left, reusing `render_scene._draw_birdseye` exactly
    as the GIF path does) + range-azimuth power map with GT overlay (right, reusing
    `render_scene._draw_radar_view` -- see that function for the `[n_angle, n_range]`
    transpose the imshow extent needs; mirrored here by reuse, not re-derivation, so
    the recurring transpose bug (`tests/test_ml_render.py`) cannot recur here)."""
    import matplotlib.pyplot as plt

    ra_db, sin_az_axis = range_azimuth_map(cfg, adc)
    fig, (ax_bev, ax_rv) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=dpi)
    if title:
        fig.suptitle(title, fontsize=10)
    _draw_birdseye(ax_bev, scats, pose, cfg)
    _draw_radar_view(ax_rv, cfg, grid, ra_db, sin_az_axis, scats, pose)
    fig.tight_layout(rect=(0, 0, 1, 0.94 if title else 1.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------
# README / loader
# --------------------------------------------------------------------------------
_LOADER_SOURCE = '''\
"""
Self-contained, standalone SSMRadNet-corpus loader -- NO `e2e` imports.

Drop this file (and the `npz/`, `labels/` directories it sits next to) anywhere;
`SSMExportDataset` needs only numpy + torch.

On-disk layout (see the top-level README.md in this same directory for the full
schema/provenance):

    npz/sample_?????.npz      -- raw ADC cube, EXACT byte-for-byte copy of the source
                                  simulator's corpus file (int16-quantized codes, see
                                  the decode formula below).
    labels/sample_?????.json  -- per-sample ground truth (see build below).

Usage
-----
    from loader_ssm import SSMExportDataset
    ds = SSMExportDataset("/path/to/export_dir")
    adc_cube, targets = ds[0]   # adc_cube: complex64 [n_rx, n_chirps, n_samples]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def _decode_int16(npz_data, meta) -> np.ndarray:
    """Exact int16-code decode: `value = code.astype(float32) * codec_meta["scale"]`
    (real/imag components decoded separately, then recombined) -- see the top-level
    README's "int16 decode formula" section. Falls back to a raw stored array for a
    sample that was NOT int16-coded (`meta["codec"] == "raw"`)."""
    codec = meta.get("codec", "raw")
    payload_key = meta.get("payload_key", "adc")
    if codec == "raw":
        return np.asarray(npz_data[payload_key])
    scale = np.float32(meta["codec_meta"]["scale"])
    dtype = np.dtype(meta["codec_meta"].get("dtype", "complex64"))
    re = npz_data[f"{payload_key}_code_re"].astype(np.float32) * scale
    im_key = f"{payload_key}_code_im"
    if im_key in npz_data:
        im = npz_data[im_key].astype(np.float32) * scale
        return (re + 1j * im).astype(dtype)
    return re.astype(dtype)


class SSMExportDataset(torch.utils.data.Dataset):
    """One `(adc_cube, targets)` pair per `npz/sample_?????.npz` file.

    `adc_cube` is the raw ADC, complex64, `[n_rx, n_chirps, n_samples]` (native
    storage order -- see the README's "coordinate frame" / "npz key schema"
    sections). NO range-Doppler / range-azimuth transform is applied here -- apply
    your own (e.g. a windowed FFT over the fast-time/slow-time axes) to match
    whatever input convention your model expects; the source simulator's own
    transform is `e2e.ml.transforms.adc_to_rd` (FFT over samples then chirps,
    Hann-windowed, zero-Doppler centred), documented in the README for reference,
    not required.

    `targets` is the sample's `labels/sample_?????.json`'s `"targets"` list,
    loaded as plain Python dicts (schema: `class`, `range_m`, `azimuth_deg`,
    `sin_azimuth`, `x_m`, `y_m`, `extent_m`, `rcs_dbsm`, `velocity_mps`,
    `grid_row`, `grid_col` -- see the README).
    """

    def __init__(self, root):
        self.root = Path(root)
        self.npz_files = sorted((self.root / "npz").glob("sample_*.npz"))
        if not self.npz_files:
            raise FileNotFoundError(f"no sample_*.npz files found under {self.root / 'npz'}")

    def __len__(self) -> int:
        return len(self.npz_files)

    def __getitem__(self, idx: int):
        npz_path = self.npz_files[idx]
        with np.load(npz_path, allow_pickle=True) as data:
            meta = json.loads(str(data["meta"].item()))
            adc_np = _decode_int16(data, meta)
        adc_cube = torch.from_numpy(np.ascontiguousarray(adc_np)).to(torch.complex64)

        labels_path = self.root / "labels" / f"{npz_path.stem}.json"
        with open(labels_path) as f:
            sample_labels = json.load(f)
        return adc_cube, sample_labels["targets"]
'''


def _write_loader(out_dir: Path) -> Path:
    path = out_dir / "loader_ssm.py"
    path.write_text(_LOADER_SOURCE)
    return path


def _radar_config_table(cfg: RadarConfig) -> str:
    rows = list(cfg.to_dict().items()) + [
        ("n_virtual", cfg.n_virtual), ("range_resolution_m", cfg.range_resolution_m),
        ("max_range_m", cfg.max_range_m), ("velocity_resolution_mps", cfg.velocity_resolution_mps),
        ("max_velocity_mps", cfg.max_velocity_mps), ("wavelength_m", cfg.wavelength_m),
    ]
    lines = ["| field | value |", "| --- | --- |"]
    lines += [f"| `{k}` | {v} |" for k, v in rows]
    return "\n".join(lines)


def _write_readme(out_dir: Path, *, cfg: RadarConfig, tier: str, seed: int, n: int,
                  manifest_path: Path, grid: LabelGrid, label_classes: Sequence[str]) -> Path:
    date = datetime.date.today().isoformat()
    readme = f"""\
# SSMRadNet export -- {n} RT-generated radar samples

Generated `{date}` by `e2e/ml/export_ssm.py` from
`{manifest_path}` (source corpus tier `{tier}`, seed block `{seed}`).

## Radar config (`{cfg.name}`)

{_radar_config_table(cfg)}

## Label grid

| field | value |
| --- | --- |
| `n_range` | {grid.n_range} |
| `n_azimuth` | {grid.n_azimuth} |
| `max_range_m` | {grid.max_range_m} |
| `range_bin_m` | {grid.range_bin_m} |
| `az_bin` (sin-azimuth) | {grid.az_bin} |
| labeled classes | {list(label_classes)} |

## Coordinate frame

Radar-centric polar, RADAR AT THE ORIGIN, BORESIGHT = **+x** (every scenario in this
export uses this convention -- see `e2e.ml.scatterers.RadarPose`). `azimuth` is the
angle from boresight, positive toward +y; `sin_azimuth` is the ULA direction cosine
(uniform axis in `[-1, 1)`, NOT linear degrees -- see
`e2e.ml.labels.LabelGrid`). Radial-frame Cartesian per target:
`x_m = range_m * cos(azimuth)`, `y_m = range_m * sin_azimuth` (== `range_m *
sin(azimuth)`), both metres, z ignored (a ULA resolves only the azimuth direction
cosine -- an elevated target is indistinguishable from a coplanar one at the same
cosine).

The detection LABEL GRID (`grid_row`, `grid_col` in each target record) is
`(range, sin(azimuth))`, `grid_row = min(int(range_m / range_bin_m), n_range - 1)`,
`grid_col = min(int((sin_azimuth + 1) / az_bin), n_azimuth - 1)`.

## File layout

    README.md            -- this file
    loader_ssm.py         -- standalone torch Dataset (no `e2e` imports)
    manifest.json          -- this export's own manifest (selected source files, config, seed)
    npz/sample_?????.npz  -- raw ADC cube, byte-for-byte copy of the source corpus file
    labels/sample_?????.json -- per-sample ground truth (see below)
    renders/sample_?????.png -- bird's-eye scene + range-azimuth map, one PNG per sample

## npz key schema

Each `npz/sample_?????.npz` is an UNMODIFIED copy of a source-corpus file (see
`e2e/ml/storage.py` / `e2e/ml/dataset.py`'s "On-disk dataset layout" for the full
contract this mirrors) with these arrays/keys:

  * `adc_code_re`, `adc_code_im` -- int16 codes, `[n_rx, n_chirps, n_samples]`
    (present when `meta["codec"] == "int16"`, which is the case for every sample in
    this export -- see "int16 decode formula" below). A sample stored with
    `meta["codec"] == "raw"` instead has a single complex64 `adc` array under
    `meta["payload_key"]`.
  * `labels` -- float32 `[3, n_range, n_azimuth]` RADIal/FFTRadNet-style detection
    map (channel 0: objectness footprint; channels 1-2: range/azimuth regression
    residuals -- see `e2e.ml.labels`'s module docstring for the exact convention if
    you want to consume this map directly instead of the JSON target list).
  * `meta` -- a 0-d unicode array holding a JSON string: `tag`, `frame_idx`, `domain`,
    `payload_key`, `shape`, `dtype`, `codec`, `codec_meta` (`{{"scale":..., "dtype":...}}`),
    `impairment_params` (per-frame randomized phase-noise/leakage/clutter settings --
    see `e2e.ml.chain_generate.default_domain_randomizer`), `targets` (the SAME
    `(range_m, sin_azimuth, class)` tuples the labels JSON expands with more fields).

## int16 decode formula

    value = code.astype(float32) * codec_meta["scale"]     # per real/imag component
    complex_value = value_re + 1j * value_im

`codec_meta["scale"]` is per-sample (see each npz's own `meta["codec_meta"]["scale"]`);
`loader_ssm.py`'s `_decode_int16` implements exactly this. This is an EXACT (lossless)
decode -- the corpus's own `QuantizerBlock` output was already discretized at this
scale before storage, so the round trip has zero additional error (see
`e2e/ml/storage.py`'s module docstring, "SAFETY").

## Provenance

* Simulator: `e2e_sim_public` (this repo) -- ray-traced (Sionna RT) FMCW MIMO radar
  scenes, difficulty tier `{tier}`, "gentle impairments" domain-randomized per frame
  (phase noise / leakage / clutter -- see each sample's `meta["impairment_params"]`
  and `e2e.ml.chain_generate.default_domain_randomizer`'s ranges).
* Scene determinism: each scene's `Scenario` is a deterministic function of
  `(tier, scene_index, seed_block) = ("{tier}", <scene index>, {seed})` -- a
  SHA-256-derived RNG seed of that exact triple (NOT `{seed} + scene_index`; see
  `e2e.ml.rt_scenes.build_rt_tier_scenario`'s "Determinism" docstring section for the
  precise mechanism, and `reconstruct_scene`/`export()` in `e2e/ml/export_ssm.py` for
  the reconstruction this export's own GT-verification step re-ran and checked against
  every sample's stored `meta["targets"]`). The ORIGINAL `scene_index`/`source_file`
  for every exported sample is recorded in this directory's own `manifest.json`.
* Generation date: `{date}`.

## License

* Mesh pools used to build these scenes are a mix of Apache-2.0 (Sionna-bundled
  car mesh) and CC0 (Kenney fleet, downloaded vehicle assets) -- see
  `e2e/ml/rt_gen.py`'s `ASSET_LICENSES` table for the exact per-asset provenance if
  you need it; no mesh binaries are redistributed in this export (only the
  simulated radar returns and the JSON target geometry they produced).
* Simulator repo: this data was generated by `e2e_sim_public`
  (https://github.com/ -- see the simulator's own top-level README for the current
  repo location) -- code is not redistributed here, only the generated radar data.

## How to load

```python
from loader_ssm import SSMExportDataset
ds = SSMExportDataset(".")   # this export directory
adc_cube, targets = ds[0]    # adc_cube: complex64 [n_rx, n_chirps, n_samples]
```
"""
    path = out_dir / "README.md"
    path.write_text(readme)
    return path


def _write_top_level_manifest(out_dir: Path, *, source_manifest_path: Path, cfg: RadarConfig,
                              tier: str, seed: int, grid: LabelGrid, label_classes: Sequence[str],
                              export_records: List[Dict[str, Any]]) -> Path:
    manifest = {
        "source_manifest": str(source_manifest_path),
        "config": cfg.to_dict(),
        "tier": tier,
        "seed": seed,
        "grid": dataclasses.asdict(grid),
        "label_classes": list(label_classes),
        "n_samples": len(export_records),
        "generated_date": datetime.date.today().isoformat(),
        "samples": export_records,
    }
    path = out_dir / "manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


# --------------------------------------------------------------------------------
# Top-level export
# --------------------------------------------------------------------------------
def export(manifest_path, n: int, out_dir, *, render: bool = True,
          use_local_assets: bool = True, dpi: int = 120,
          progress_every: int = 50) -> Dict[str, Any]:
    """Export `n` samples from `manifest_path`'s train split to `out_dir`. See the
    module docstring. Returns a small summary dict (counts, paths, GT-check result)."""
    manifest_path = Path(manifest_path)
    with open(manifest_path) as f:
        manifest = json.load(f)

    cfg = RadarConfig.from_dict(manifest["config"])
    tier = manifest["tier"]
    seed = int(manifest["seed"])
    frames_per_scene = int(manifest.get("frames_per_scene", 1))
    grid = LabelGrid(**manifest["grid"])
    label_classes = tuple(manifest.get("label_classes") or _LABEL_CLASSES_DEFAULT)

    selected = select_train_files(manifest, n)
    source_dir = manifest_path.parent

    out_dir = Path(out_dir)
    npz_dir = out_dir / "npz"
    labels_dir = out_dir / "labels"
    renders_dir = out_dir / "renders"
    npz_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    if render:
        renders_dir.mkdir(parents=True, exist_ok=True)

    export_records: List[Dict[str, Any]] = []
    for idx, fname in enumerate(selected):
        scene_idx, frame_idx = _parse_scene_frame(fname)
        scenario, scats, pose = reconstruct_scene(
            tier, seed, scene_idx, frame_idx, frames_per_scene,
            use_local_assets=use_local_assets,
        )
        rebuilt_targets = targets_in_grid(grid, scats, pose, classes=label_classes)

        src_path = source_dir / fname
        with np.load(src_path, allow_pickle=True) as data:
            meta = json.loads(str(data["meta"].item()))
            stored_targets = meta.get("targets", [])
            verify_gt_match(stored_targets, rebuilt_targets, context=fname)

            adc_tensor = None
            if render:
                payload_key = meta.get("payload_key", "adc")
                adc_np = storage.read_payload(data, meta, payload_key)
                adc_tensor = torch.from_numpy(np.ascontiguousarray(adc_np)).to(torch.complex64)

        dst_name = f"sample_{idx:05d}"
        shutil.copy2(src_path, npz_dir / f"{dst_name}.npz")

        records = build_target_records(grid, scenario, scats, pose, label_classes)
        labels_payload = {
            "sample_index": idx,
            "source_file": fname,
            "scene_index": scene_idx,
            "frame_idx": frame_idx,
            "tier": tier,
            "config": cfg.name,
            "impairment_params": meta.get("impairment_params"),
            "targets": records,
        }
        with open(labels_dir / f"{dst_name}.json", "w") as f:
            json.dump(labels_payload, f, indent=2)

        if render:
            render_sample(cfg, grid, scats, pose, adc_tensor,
                          renders_dir / f"{dst_name}.png",
                          title=f"{dst_name}  ({cfg.name}, tier {tier}, scene {scene_idx})",
                          dpi=dpi)

        export_records.append({
            "sample_index": idx, "source_file": fname,
            "scene_index": scene_idx, "frame_idx": frame_idx,
        })
        if progress_every and (idx + 1) % progress_every == 0:
            print(f"[export_ssm] {idx + 1}/{len(selected)} samples written", file=sys.stderr)

    _write_top_level_manifest(
        out_dir, source_manifest_path=manifest_path, cfg=cfg, tier=tier, seed=seed,
        grid=grid, label_classes=label_classes, export_records=export_records,
    )
    _write_readme(out_dir, cfg=cfg, tier=tier, seed=seed, n=len(selected),
                  manifest_path=manifest_path, grid=grid, label_classes=label_classes)
    _write_loader(out_dir)

    return {
        "n_samples": len(selected), "out_dir": str(out_dir),
        "gt_verified": True, "selected_files": selected,
    }


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m e2e.ml.export_ssm",
        description="Export a slice of an RT-generated e2e.ml radar corpus, plus "
                    "per-sample renders, for the SSMRadNet authors' own training.",
    )
    p.add_argument("--manifest", required=True, help="path to the source corpus's manifest.json")
    p.add_argument("--n", type=int, default=1000, help="number of train-split samples to export")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--no-render", action="store_true", help="skip per-sample PNG renders")
    p.add_argument("--dpi", type=int, default=120, help="render DPI (default 120)")
    p.add_argument("--no-local-assets", action="store_true",
                   help="reconstruct scenes without the local (unshipped) asset pool -- "
                        "only correct if the source corpus was ALSO generated with "
                        "use_local_assets=False; the default (False here == "
                        "use_local_assets=True) matches generate_chain_corpus's own default")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = export(
        args.manifest, args.n, args.out, render=not args.no_render, dpi=args.dpi,
        use_local_assets=not args.no_local_assets,
    )
    print(f"wrote {result['n_samples']} samples to {result['out_dir']} "
         f"(GT verified: {result['gt_verified']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
