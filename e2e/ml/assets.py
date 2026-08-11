"""
Real downloaded vehicle meshes for the ray-traced ML radar corpus (campaign R3).

The problem this solves: every "car" in every RT scene was one low-poly Sionna geometry
repeated under 17 different names (see `e2e.ml.rt_gen.CAR_ASSET_NAMES`), so a radar model
trained on this data never saw real vehicle-shape diversity. This module owns the three
steps that turn a handful of user-downloaded artist meshes (mixed units, mixed axis
conventions, hundreds of thousands of triangles each) into radar-appropriate scene
assets: **extraction/inventory**, **decimation**, and **scale/orientation normalization**.
`e2e.ml.rt_gen` (asset dispatch + licensing) and `e2e.ml.rt_scenes` (pool composition)
consume the result; neither owns any of this module's logic.

Nothing here ever touches Sionna -- pure numpy/stdlib, so it runs (and degrades
gracefully) with no ray tracer installed.

Cache layout (NEVER inside the repo -- no mesh binary is ever committed):
    <cache_dir()>/raw/<asset>/...        -- extracted archive contents (verbatim)
    <cache_dir()>/processed/<asset>.ply  -- decimated, metre-scaled, ground-aligned PLY

`cache_dir()` defaults to `%LOCALAPPDATA%/e2e_ml_assets` (or `<tempdir>/e2e_ml_assets` if
`LOCALAPPDATA` isn't set), overridable via the `E2E_ML_ASSET_CACHE_DIR` env var. Source
archives are looked for in `downloads_dir()` (default `C:\\Users\\asf3\\Downloads`,
override `E2E_ML_ASSET_DOWNLOADS_DIR`) -- both are workstation-specific by nature (these
are one user's downloaded files); every consumer degrades gracefully (falls back to a
Sionna-bundled mesh) when the cache/archives aren't present, so the repo and CI are
unaffected on any other machine.

Axis normalization -- the load-bearing geometry fix
-----------------------------------------------------
Each source file mixes units (mm/cm/"unitless CAD", even one already in metres) and
axis conventions (which raw axis is the vehicle's long axis, which is "up"). A vehicle
ray-traced lying on its side or scaled 10x too large gives a nonsense radar signature.
For each asset we (a) identify, by inspecting the raw mesh's own geometry (wheel/panel
positions, bounding-box aspect ratios against the real vehicle's known L/W/H), which raw
axis is length and which is height, (b) permute axes so **length -> x, width -> y,
height -> z** (this package's forward/up convention, matching `rt_gen`'s radar boresight
being world +x and z being up), (c) scale so the length axis lands in the vehicle
class's real-world range, and (d) rebase z so the mesh's own bbox rests on z=0 (matches
`rt_gen._load_local_asset`'s ground-rest convention). Two of the five raw files also
bundle non-vehicle geometry (a studio backdrop plane, softbox light-panel props) inside
the same file, identified by inspecting per-object-group bounding boxes (see
`DownloadedAssetSpec.exclude_name_substrings`) and dropped before decimation.

Decimation
----------
Neither `trimesh` nor `open3d` is installed on this machine (checked at write time), and
this module does not add either as a new dependency (see the campaign brief). Instead
`_vertex_cluster_decimate` implements plain **vertex clustering** (Rossignac & Borrel
1993): snap the (already normalized) mesh onto a uniform 3-D grid, collapse every vertex
in a cell to that cell's centroid, drop degenerate (collapsed-to-a-point) and duplicate
triangles. This is intentionally crude next to edge-collapse/QEM decimation, but for a
convex-ish, large-flat-facet target like a vehicle body -- where a radar's RCS is
dominated by big panels and corner reflectors, not paint-detail curvature -- it keeps
the dominant scattering geometry (bounding-box-scale facets/corners) while aggressively
cutting mesh-detail triangle count. `decimate_to_budget` bisects the cell size (in log
space, since triangle count falls roughly monotonically as cell size grows) to land
under `DECIMATE_MAX_TRIS`.

CLI
---
    python -m e2e.ml.assets [--force]
extracts (if needed), decimates, normalizes and caches every `DOWNLOADED_ASSET_SPECS`
entry whose raw file is found, and prints the before/after triangle count + metre bbox +
scale-factor inventory table (the same table this module's report used).
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------------
# Cache / source locations
# --------------------------------------------------------------------------------
CACHE_DIR_ENV = "E2E_ML_ASSET_CACHE_DIR"
DOWNLOADS_DIR_ENV = "E2E_ML_ASSET_DOWNLOADS_DIR"

_DEFAULT_CACHE_DIR = os.path.join(
    os.environ.get("LOCALAPPDATA") or tempfile.gettempdir(), "e2e_ml_assets"
)
_DEFAULT_DOWNLOADS_DIR = r"C:\Users\asf3\Downloads"


def cache_dir() -> str:
    return os.environ.get(CACHE_DIR_ENV) or _DEFAULT_CACHE_DIR


def raw_dir() -> str:
    return os.path.join(cache_dir(), "raw")


def processed_dir() -> str:
    return os.path.join(cache_dir(), "processed")


def downloads_dir() -> str:
    return os.environ.get(DOWNLOADS_DIR_ENV) or _DEFAULT_DOWNLOADS_DIR


# Radar-appropriate decimation budget: RCS is dominated by large facets/corners, not
# fine detail (see module docstring). Target is the bisection's aim point; MAX is the
# hard ceiling `decimate_to_budget` guarantees (with a coarsening safety loop).
DECIMATE_TARGET_TRIS = 12_000
DECIMATE_MAX_TRIS = 20_000


# --------------------------------------------------------------------------------
# Archive -> raw cache extraction (item 1: extract & inventory)
# --------------------------------------------------------------------------------
@dataclass(frozen=True)
class ArchiveSpec:
    archive_filename: str   # inside downloads_dir()
    extract_subdir: str     # inside raw_dir(); 7z extracts the archive's own tree here


# The Mercedes O403 bus (m6f72urgb6yo-o403Onurym.rar, a .max/3ds-Max file) is
# deliberately NOT listed here: not usable without 3ds Max (no converter installed on
# this machine, and none was added as a dependency -- see the module docstring and
# report). Superseded by Type_B_North_American_School_Bus_v2 per an explicit user
# instruction mid-campaign.
ARCHIVE_SPECS: Dict[str, ArchiveSpec] = {
    "delorean": ArchiveSpec("45-delorean.rar", "delorean"),
    "audi_r8": ArchiveSpec("xihu59td0bnk-AudiR8Spyder_2017.rar", "audi_r8"),
    "truck_daf": ArchiveSpec("84-truck_daf.zip", "truck_daf"),
    "trolley": ArchiveSpec(
        "trolley_car_v1_L1.123c0f5c91a9-c941-49f8-8e8c-02d5e7cf05cb.zip", "trolley"
    ),
    "school_bus": ArchiveSpec(
        "Type_B_North_American_School_Bus_v2_L1.184d3df52-7cb6-41ad-9ad2-245a2e472149.zip",
        "bus_schoolbus",
    ),
}


def ensure_extracted(key: str) -> bool:
    """Extract `ARCHIVE_SPECS[key]` into `raw_dir()` via the `7z` CLI if not already
    present. Returns whether the extracted tree now exists (True even if it was already
    there); False if the source archive isn't in `downloads_dir()` or `7z` isn't
    available -- callers degrade gracefully (see `process_asset`)."""
    spec = ARCHIVE_SPECS[key]
    dest = os.path.join(raw_dir(), spec.extract_subdir)
    if os.path.isdir(dest) and os.listdir(dest):
        return True
    archive = os.path.join(downloads_dir(), spec.archive_filename)
    if not os.path.isfile(archive):
        return False
    os.makedirs(dest, exist_ok=True)
    try:
        subprocess.run(["7z", "x", "-y", f"-o{dest}", archive],
                       check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    return os.path.isdir(dest) and os.listdir(dest) != []


def _find_one(root: str, filename: str) -> Optional[str]:
    """First path under `root` (recursive) whose basename matches `filename`
    (case-insensitive) -- archive trees sometimes nest an extra folder level."""
    target = filename.lower()
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.lower() == target:
                return os.path.join(dirpath, f)
    return None


# --------------------------------------------------------------------------------
# Asset registry (items 2/3: decimation + scale/orientation normalization)
# --------------------------------------------------------------------------------
@dataclass(frozen=True)
class DownloadedAssetSpec:
    """One downloaded vehicle mesh: where to find it, and how to turn its raw,
    mixed-unit/mixed-axis geometry into a metre-scaled, ground-resting, length-along-x
    mesh (see the module docstring's "Axis normalization" section for how each of these
    numbers was derived -- by inspecting the raw geometry, not guessed)."""
    name: str
    vehicle_class: str                      # "car" | "truck" | "bus" | "trolley"
    archive_key: str                        # key into ARCHIVE_SPECS / raw_dir() subtree
    filename: str                           # mesh filename to locate under that subtree
    # Permutation of raw axes (0=x,1=y,2=z) placing (length, width, height) at
    # (new x, new y, new z): `new_verts = raw_verts[:, axis_permutation]`.
    axis_permutation: Tuple[int, int, int]
    scale_m: float                          # raw (permuted) units -> metres
    real_length_range_m: Tuple[float, float]
    exclude_name_substrings: Tuple[str, ...] = ()  # OBJ object-group names to drop
    source: str = ""
    derivation: str = ""                    # how scale_m/axes were determined -- audit trail


DOWNLOADED_ASSET_SPECS: Dict[str, DownloadedAssetSpec] = {
    "dl_delorean": DownloadedAssetSpec(
        name="dl_delorean", vehicle_class="car",
        archive_key="delorean", filename="DeLorean.STL",
        axis_permutation=(2, 0, 1), scale_m=4.267 / 72.1558,
        real_length_range_m=(4.2, 5.0),
        source="freestl.com (45-delorean.rar)",
        derivation="raw bbox extents (x=33.07, y=21.15, z=72.16) ratio-match a real "
                   "DMC-12's L:W:H (4.267:1.79:1.14) only as z:x:y (z/x=2.18~=L/W=2.38, "
                   "x/y=1.56~=W/H=1.58) -- so length=z, width=x, height=y. scale so "
                   "length = 4.267 m (real DMC-12 length).",
    ),
    "dl_audi_r8": DownloadedAssetSpec(
        name="dl_audi_r8", vehicle_class="car",
        archive_key="audi_r8", filename="Audi_R8_2017.obj",
        axis_permutation=(2, 0, 1), scale_m=1.0,
        real_length_range_m=(4.2, 5.0),
        exclude_name_substrings=("Ground_Plane", "LightSource"),
        source="freestl.com (xihu59td0bnk-AudiR8Spyder_2017.rar)",
        derivation="OBJ bundles a studio floor plane + two softbox light-panel props as "
                   "extra object groups (huge, name-flagged, excluded before measuring). "
                   "Remaining bbox (x=2.04, y=1.24, z=4.42) matches a real Audi R8's "
                   "W=1.94/H=1.24/L=4.42 almost exactly as x:y:z -> length=z, width=x, "
                   "height=y, raw units already ~metres (scale=1.0).",
    ),
    "dl_truck_daf": DownloadedAssetSpec(
        name="dl_truck_daf", vehicle_class="truck",
        archive_key="truck_daf", filename="truck_daf.obj",
        axis_permutation=(2, 0, 1), scale_m=0.80,
        real_length_range_m=(8.0, 16.0),
        exclude_name_substrings=("Plane",),
        source="freestl.com (84-truck_daf.zip)",
        derivation="OBJ bundles two flat backdrop 'Plane'/'Plane.001' groups (excluded). "
                   "Remaining bbox (x=3.68 width, y=4.89 height, z=19.68 length). Scale "
                   "cross-checked from the 'ban_depan' (front tire) group's own bbox "
                   "diameter (~1.31 raw units) against a real truck tire (~1.05 m) -> "
                   "scale ~0.80; applied length 15.7 m / height 3.9 m / width 2.9 m are "
                   "all independently plausible for a tractor-trailer.",
    ),
    "dl_trolley": DownloadedAssetSpec(
        name="dl_trolley", vehicle_class="trolley",
        archive_key="trolley", filename="trolly.obj",
        axis_permutation=(0, 1, 2), scale_m=12.0 / 986.259,
        real_length_range_m=(10.0, 14.0),
        source="freestl.com (trolley_car_v1_L1...zip)",
        derivation="Single-group OBJ, no backdrop pollution. Raw bbox x=986.26 "
                   "(symmetric -> length), y=293.54 (symmetric -> width), z=414.86 "
                   "(z>=~0, floor-resting -> height, already 'up' in the file). Axes "
                   "already in this package's (length=x, width=y, height=z) convention "
                   "-- identity permutation. Scale so length = 12.0 m (mid-range of a "
                   "10-14 m streetcar).",
    ),
    "dl_school_bus": DownloadedAssetSpec(
        name="dl_school_bus", vehicle_class="bus",
        archive_key="school_bus", filename="18563_Type_B_North_American_School_Bus_v2.obj",
        axis_permutation=(0, 1, 2), scale_m=8.0 / 9.9707,
        real_length_range_m=(7.0, 9.0),
        source="freestl.com (Type_B_North_American_School_Bus_v2_L1...zip); ADDED "
              "mid-campaign, replacing the unusable Mercedes O403 .max file",
        derivation="Single-group OBJ. Raw bbox x=9.97 (asymmetric -> length), "
                   "y=3.70 (symmetric -> width), z=3.41 (z>=0, floor-resting -> "
                   "height). Already length=x/width=y/height=z -- identity "
                   "permutation. Scale so length = 8.0 m (mid-range of a 7-9 m "
                   "Type B school bus).",
    ),
}

DOWNLOADED_CAR_ASSET_NAMES = tuple(
    n for n, s in DOWNLOADED_ASSET_SPECS.items() if s.vehicle_class == "car")
DOWNLOADED_TRUCK_ASSET_NAMES = tuple(
    n for n, s in DOWNLOADED_ASSET_SPECS.items() if s.vehicle_class == "truck")
DOWNLOADED_BUS_ASSET_NAMES = tuple(
    n for n, s in DOWNLOADED_ASSET_SPECS.items() if s.vehicle_class == "bus")
DOWNLOADED_TROLLEY_ASSET_NAMES = tuple(
    n for n, s in DOWNLOADED_ASSET_SPECS.items() if s.vehicle_class == "trolley")


# --------------------------------------------------------------------------------
# Mesh I/O -- group-aware OBJ reader (needs to drop excluded object groups, which the
# simpler per-line readers in `e2e.ml.rt_gen` don't need to) + a binary/ASCII STL reader.
# numpy-backed (these source files run to hundreds of thousands of vertices).
# --------------------------------------------------------------------------------
def _read_stl_np(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """`(verts [N,3], faces [M,3])` float64/int64 arrays from a binary or ASCII STL.
    STL has no shared vertices (3 fresh verts per triangle) -- mirrors
    `rt_gen._read_stl`'s binary-vs-ASCII sniffing, numpy-vectorized for speed."""
    with open(path, "rb") as f:
        header = f.read(80)
        count_bytes = f.read(4)
    is_binary = False
    if len(count_bytes) == 4 and not header.lstrip().lower().startswith(b"solid"):
        is_binary = True
    elif len(count_bytes) == 4:
        n_tri = struct.unpack("<I", count_bytes)[0]
        is_binary = os.path.getsize(path) == 84 + n_tri * 50

    if is_binary:
        with open(path, "rb") as f:
            f.read(80)
            n_tri = struct.unpack("<I", f.read(4))[0]
            # Each record: 12 float32 (normal + 3 verts) + 1 uint16 attr = 50 bytes.
            dtype = np.dtype([("normal", "<f4", 3), ("verts", "<f4", (3, 3)),
                              ("attr", "<u2")])
            data = np.fromfile(f, dtype=dtype, count=n_tri)
        verts = data["verts"].reshape(-1, 3).astype(np.float64)
    else:
        verts_list: List[Tuple[float, float, float]] = []
        with open(path, "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("vertex"):
                    p = line.split()
                    verts_list.append((float(p[1]), float(p[2]), float(p[3])))
        verts = np.asarray(verts_list, dtype=np.float64)

    n_tri = verts.shape[0] // 3
    faces = np.arange(n_tri * 3, dtype=np.int64).reshape(n_tri, 3)
    return verts, faces


def _read_obj_np(path: str, exclude_name_substrings: Sequence[str] = ()
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """`(verts [N,3], faces [M,3])` float64/int64 arrays from a Wavefront OBJ,
    triangulating n-gons by fan and DROPPING any object group (`o <name>` block) whose
    name contains one of `exclude_name_substrings` (case-sensitive substring match) --
    e.g. a bundled studio backdrop plane or light-panel prop that is not part of the
    vehicle (see the module docstring)."""
    raw_verts: List[Tuple[float, float, float]] = []
    vertex_excluded: List[bool] = []
    raw_faces: List[Tuple[int, ...]] = []  # 0-based GLOBAL (pre-filter) indices
    face_excluded: List[bool] = []
    current_excluded = False

    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("o ") or line.startswith("g "):
                name = line[2:].strip()
                current_excluded = any(s in name for s in exclude_name_substrings)
            elif line.startswith("v "):
                p = line.split()
                raw_verts.append((float(p[1]), float(p[2]), float(p[3])))
                vertex_excluded.append(current_excluded)
            elif line.startswith("f "):
                idx = []
                n = len(raw_verts)
                for tok in line.split()[1:]:
                    vi = int(tok.split("/")[0])
                    idx.append(vi - 1 if vi > 0 else n + vi)
                for k in range(1, len(idx) - 1):
                    raw_faces.append((idx[0], idx[k], idx[k + 1]))
                    face_excluded.append(current_excluded)

    verts_all = np.asarray(raw_verts, dtype=np.float64)
    excl = np.asarray(vertex_excluded, dtype=bool)
    keep_mask = ~excl
    new_index = np.full(verts_all.shape[0], -1, dtype=np.int64)
    new_index[keep_mask] = np.arange(int(keep_mask.sum()))

    faces_all = np.asarray(raw_faces, dtype=np.int64) if raw_faces else np.zeros((0, 3), dtype=np.int64)
    f_excl = np.asarray(face_excluded, dtype=bool) if face_excluded else np.zeros((0,), dtype=bool)
    if faces_all.shape[0]:
        any_excluded_vertex = excl[faces_all].any(axis=1)
        face_keep = (~f_excl) & (~any_excluded_vertex)
        faces = new_index[faces_all[face_keep]]
    else:
        faces = faces_all

    return verts_all[keep_mask], faces


def load_raw_mesh(spec: DownloadedAssetSpec) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """`(verts, faces)` for `spec`'s raw source file, or `None` if the archive isn't
    extracted (and can't be, e.g. `7z`/the source archive is missing) -- graceful
    degrade, matching `rt_gen._load_local_asset`'s contract."""
    if not ensure_extracted(spec.archive_key):
        return None
    root = os.path.join(raw_dir(), ARCHIVE_SPECS[spec.archive_key].extract_subdir)
    path = _find_one(root, spec.filename)
    if path is None:
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext == ".stl":
        return _read_stl_np(path)
    if ext == ".obj":
        return _read_obj_np(path, spec.exclude_name_substrings)
    raise ValueError(f"unsupported mesh extension {ext!r} for {spec.name}")


# --------------------------------------------------------------------------------
# Normalization (axis permutation + scale + ground-rest rebase)
# --------------------------------------------------------------------------------
def normalize_mesh(verts: np.ndarray, spec: DownloadedAssetSpec
                   ) -> Tuple[np.ndarray, Dict[str, float]]:
    """Apply `spec.axis_permutation` + `spec.scale_m`, then rebase z so the mesh's own
    bbox min-z is 0 (ground-rest, matching `rt_gen._load_local_asset`'s convention --
    `SceneObject.position` re-centers on the mesh's OWN aabb, so a caller needs this
    self-consistent local frame, not a world placement). Returns `(new_verts, stats)`
    with `stats = {"length_m", "width_m", "height_m"}` (post-normalization bbox, which
    -- thanks to the permutation -- are exactly the x/y/z extents)."""
    perm = list(spec.axis_permutation)
    out = verts[:, perm] * float(spec.scale_m)
    out = out.copy()
    out[:, 2] -= out[:, 2].min()
    extent = out.max(axis=0) - out.min(axis=0)
    stats = {"length_m": float(extent[0]), "width_m": float(extent[1]),
             "height_m": float(extent[2])}
    return out, stats


# --------------------------------------------------------------------------------
# Decimation (vertex clustering -- see module docstring for why, no trimesh/open3d here)
# --------------------------------------------------------------------------------
def _cluster_keys(idx: np.ndarray, base: int = 4096) -> np.ndarray:
    idx = np.clip(idx, 0, base - 1)
    return (idx[:, 0].astype(np.int64) * base + idx[:, 1].astype(np.int64)) * base + idx[:, 2].astype(np.int64)


def _vertex_cluster_pass(verts: np.ndarray, faces: np.ndarray, cell_size: float,
                         bbox_min: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """One vertex-clustering pass at `cell_size` -> `(new_verts, new_faces)` (see
    module docstring: Rossignac & Borrel vertex clustering -- snap to a grid, collapse
    each cell's vertices to their centroid, drop degenerate/duplicate triangles)."""
    idx = np.floor((verts - bbox_min) / max(cell_size, 1e-12)).astype(np.int64)
    keys = _cluster_keys(idx)
    uniq, inverse = np.unique(keys, return_inverse=True)
    n_clusters = uniq.shape[0]

    sums = np.zeros((n_clusters, 3), dtype=np.float64)
    np.add.at(sums, inverse, verts)
    counts = np.bincount(inverse, minlength=n_clusters).astype(np.float64)
    centroids = sums / counts[:, None]

    new_faces = inverse[faces]
    degenerate = ((new_faces[:, 0] == new_faces[:, 1]) |
                 (new_faces[:, 1] == new_faces[:, 2]) |
                 (new_faces[:, 0] == new_faces[:, 2]))
    new_faces = new_faces[~degenerate]
    if new_faces.shape[0] > 0:
        sorted_faces = np.sort(new_faces, axis=1)
        _, dedup_idx = np.unique(sorted_faces, axis=0, return_index=True)
        new_faces = new_faces[np.sort(dedup_idx)]
    return centroids, new_faces


def decimate_to_budget(verts: np.ndarray, faces: np.ndarray,
                       target_tris: int = DECIMATE_TARGET_TRIS,
                       max_tris: int = DECIMATE_MAX_TRIS,
                       iters: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """Vertex-cluster `(verts, faces)` down to at most `max_tris` triangles, aiming for
    `target_tris`. Bisects cell size in log space (triangle count falls roughly
    monotonically as cell size grows) then, as a safety net, keeps coarsening past the
    search bracket if discretization noise left the result over `max_tris`."""
    if faces.shape[0] <= max_tris:
        return verts, faces  # already under budget -- nothing to do

    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    lo, hi = diag * 1e-4, diag * 1.0
    log_lo, log_hi = math.log(lo), math.log(hi)

    best: Optional[Tuple[np.ndarray, np.ndarray, int]] = None
    for _ in range(iters):
        mid = math.exp(0.5 * (log_lo + log_hi))
        v2, f2 = _vertex_cluster_pass(verts, faces, mid, bbox_min)
        n = f2.shape[0]
        if n <= max_tris and (best is None or n > best[2]):
            best = (v2, f2, n)
        if n > target_tris:
            log_lo = math.log(mid)   # too many triangles -- need a coarser (bigger) cell
        else:
            log_hi = math.log(mid)   # at/under target -- can afford to try finer

    if best is None:
        # Bracket didn't reach max_tris (pathological geometry) -- coarsen from hi.
        cell = hi
        for _ in range(10):
            cell *= 1.5
            v2, f2 = _vertex_cluster_pass(verts, faces, cell, bbox_min)
            if f2.shape[0] <= max_tris:
                best = (v2, f2, f2.shape[0])
                break
        if best is None:
            raise RuntimeError("decimate_to_budget: could not reach the triangle budget")
    return best[0], best[1]


# --------------------------------------------------------------------------------
# PLY writer (ASCII, positions + triangle indices -- same minimal format as
# `rt_gen._write_ply`; duplicated rather than imported to keep this module's mesh
# pipeline independent of rt_gen's dispatch/licensing layer).
# --------------------------------------------------------------------------------
def _write_ply(path: str, verts: np.ndarray, faces: np.ndarray) -> None:
    """Write a minimal BINARY little-endian PLY.

    Binary rather than ASCII because the scene is rebuilt every frame, so mesh parsing
    is a per-frame cost paid thousands of times over a corpus -- and Mitsuba logs a
    performance warning for ASCII PLY on every single load, which is how this surfaced.
    Binary also writes exact float32 rather than the 6-decimal text rounding the ASCII
    path applied, so the geometry that reaches the ray tracer is the geometry that came
    out of decimation.
    """
    verts = np.ascontiguousarray(verts, dtype="<f4")
    faces = np.ascontiguousarray(faces, dtype="<i4")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {verts.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {faces.shape[0]}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    # Each face record is a uchar count (always 3) followed by three int32 indices, so
    # the rows are built as a structured array rather than concatenated by hand.
    face_records = np.empty(
        faces.shape[0], dtype=[("n", "u1"), ("v", "<i4", (3,))]
    )
    face_records["n"] = 3
    face_records["v"] = faces
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(verts.tobytes())
        f.write(face_records.tobytes())


# --------------------------------------------------------------------------------
# Top-level pipeline + cache
# --------------------------------------------------------------------------------
@dataclass
class AssetResult:
    name: str
    vehicle_class: str
    ply_path: str
    height_m: float          # z-extent -- what rt_gen.object_local_height_m needs
    length_m: float
    width_m: float
    tris_before: int
    tris_after: int
    scale_m: float


_process_cache: Dict[str, Optional[AssetResult]] = {}


def process_asset(name: str, *, force: bool = False) -> Optional[AssetResult]:
    """Extract (if needed) + decimate + normalize + cache `DOWNLOADED_ASSET_SPECS[name]`,
    returning an `AssetResult`, or `None` if the raw source isn't available on this
    machine (graceful degrade -- callers, e.g. `rt_gen`, fall back to a Sionna mesh).
    Cached both in-process and on disk (`processed_dir()/<name>.ply` + a stats
    sidecar) so repeated calls (every frame of every tier draw) don't re-decimate."""
    if not force and name in _process_cache:
        return _process_cache[name]

    spec = DOWNLOADED_ASSET_SPECS[name]
    ply_path = os.path.join(processed_dir(), f"{name}.ply")
    stats_path = ply_path + ".stats"
    if not force and os.path.isfile(ply_path) and os.path.isfile(stats_path):
        with open(stats_path) as f:
            vals = dict(line.strip().split("=", 1) for line in f if "=" in line)
        result = AssetResult(
            name=name, vehicle_class=spec.vehicle_class, ply_path=ply_path,
            height_m=float(vals["height_m"]), length_m=float(vals["length_m"]),
            width_m=float(vals["width_m"]), tris_before=int(vals["tris_before"]),
            tris_after=int(vals["tris_after"]), scale_m=float(spec.scale_m),
        )
        _process_cache[name] = result
        return result

    loaded = load_raw_mesh(spec)
    if loaded is None:
        _process_cache[name] = None
        return None
    verts, faces = loaded
    tris_before = int(faces.shape[0])

    norm_verts, bbox_stats = normalize_mesh(verts, spec)
    dec_verts, dec_faces = decimate_to_budget(norm_verts, faces)
    # Re-derive bbox from the DECIMATED mesh too (vertex clustering can shave a sliver
    # off the extremes if a corner cluster collapses inward) -- report what actually
    # ships, not the pre-decimation number.
    extent = dec_verts.max(axis=0) - dec_verts.min(axis=0)

    os.makedirs(processed_dir(), exist_ok=True)
    _write_ply(ply_path, dec_verts, dec_faces)
    with open(stats_path, "w") as f:
        f.write(f"length_m={float(extent[0])}\n")
        f.write(f"width_m={float(extent[1])}\n")
        f.write(f"height_m={float(extent[2])}\n")
        f.write(f"tris_before={tris_before}\n")
        f.write(f"tris_after={int(dec_faces.shape[0])}\n")

    result = AssetResult(
        name=name, vehicle_class=spec.vehicle_class, ply_path=ply_path,
        height_m=float(extent[2]), length_m=float(extent[0]), width_m=float(extent[1]),
        tris_before=tris_before, tris_after=int(dec_faces.shape[0]), scale_m=float(spec.scale_m),
    )
    _process_cache[name] = result
    return result


def process_all(*, force: bool = False) -> Dict[str, Optional[AssetResult]]:
    """`process_asset` for every `DOWNLOADED_ASSET_SPECS` entry -- entries whose raw
    source isn't found on this machine map to `None`."""
    return {name: process_asset(name, force=force) for name in DOWNLOADED_ASSET_SPECS}


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m e2e.ml.assets",
        description="Extract, decimate and normalize the downloaded vehicle meshes.",
    )
    p.add_argument("--force", action="store_true", help="reprocess even if cached")
    args = p.parse_args(argv)

    results = process_all(force=args.force)
    header = f"{'name':16s} {'class':8s} {'tris_before':>11s} {'tris_after':>10s} " \
             f"{'L (m)':>7s} {'W (m)':>7s} {'H (m)':>7s} {'scale':>10s}  status"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        spec = DOWNLOADED_ASSET_SPECS[name]
        if r is None:
            print(f"{name:16s} {spec.vehicle_class:8s} {'--':>11s} {'--':>10s} "
                 f"{'--':>7s} {'--':>7s} {'--':>7s} {'--':>10s}  MISSING (no raw source found)")
            continue
        print(f"{r.name:16s} {r.vehicle_class:8s} {r.tris_before:>11d} {r.tris_after:>10d} "
             f"{r.length_m:>7.2f} {r.width_m:>7.2f} {r.height_m:>7.2f} {r.scale_m:>10.5f}  ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
