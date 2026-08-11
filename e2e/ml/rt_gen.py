"""
Ray-traced (Sionna RT) raw-ADC generation for the FMCW MIMO radar ML package.

This is the high-fidelity sibling of `e2e.ml.rd_synth`: instead of evaluating the
closed-form point-target model, it ray-traces the scene with Sionna RT and turns the
resulting channel frequency response into the **same** dechirped ADC cube --
`complex64 [n_rx, n_chirps, n_samples]` -- so `e2e.ml.transforms`, `e2e.ml.labels`
and `e2e.ml.dataset` cannot tell which generator produced a frame. Datasets and
training code stay source-agnostic; only the fidelity/cost trade changes.

Sionna is imported lazily (inside functions), so `import e2e.ml.rt_gen` works on a
machine without Sionna/DrJit -- only the generation calls need it.


The CFR -> beat mapping (the load-bearing derivation)
----------------------------------------------------
An FMCW transmitter emits ``s_t(t) = exp(j2pi(f0 t + S t^2/2))`` (slope
``S = B / T_sweep``). The echo from a scatterer at round-trip delay ``tau`` is
``s_t(t - tau)``, and the receiver *dechirps* it against the transmitted ramp. This
package's convention (see `rd_synth`'s module docstring) is the **positive-exponent**
one, ``s_b(t) = s_t(t) conj(s_t(t-tau))``:

    s_b(t) = exp(j2pi[ f0 tau + S tau t - S tau^2 / 2 ])

Sampling the ADC at ``t = n / fs`` and dropping the residual video phase
``-pi S tau^2`` (rd_synth drops it too, `include_rvp=False`) gives

    b[n] = exp(+j2pi (f0 + S n/fs) tau)                                        (1)

i.e. **the dechirped sample n is the channel evaluated at the instantaneous RF
frequency of the ramp at that sample**, ``f_RF(n) = f0 + S n / fs``.

Sionna's `Paths.cfr(frequencies=f, ...)` returns (paths.py, `cir`/`cfr` docstrings)

    h(f, t) = sum_i a_i exp(-j2pi f_c tau_i) exp(+j2pi f_D,i t) exp(-j2pi f tau_i)
            = sum_i a_i exp(-j2pi (f_c + f) tau_i) exp(+j2pi f_D,i t)          (2)

where ``f_c = scene.frequency`` (the carrier) and ``f`` is a **baseband offset** from
it (Sionna's OFDM helper `subcarrier_frequencies` returns offsets centred on 0).
Comparing (1) and (2): a single path contributes ``exp(-j2pi f_RF tau)`` where (1)
wants ``exp(+j2pi f_RF tau)``. So the beat cube is the **complex conjugate** of the
CFR sampled on the ramp:

    b[c, n] = conj( h( f_bb(n), t = c * T_c ) ),   f_bb(n) = f0 + S n/fs - f_c   (3)

with ``f_c = f0 + B/2`` (the chirp centre; chosen so Sionna's array-element spacing,
which is expressed in wavelengths of ``scene.frequency``, is exactly the
``cfg.wavelength_m / 2`` that `rd_synth` assumes), ``sampling_frequency = 1/T_c`` and
``num_time_steps = n_chirps`` -- Sionna's slow-time axis IS the chirp axis.

The same conjugation fixes the Doppler sign automatically, which is why (3) is
stated as one operation rather than three. Sionna's ``f_D`` is the *physical* Doppler
shift (positive for an approaching target: for a monostatic link and a target with
radial velocity ``v_r``, receding-positive, ``f_D = -2 v_r / lambda``). Conjugating
(2) turns ``exp(+j2pi f_D t)`` into ``exp(+j2pi (2 v_r/lambda) t)`` -- exactly
rd_synth's chirp-to-chirp phase progression (its beat phase ``2pi f0 tau_c`` with
``tau_c = 2(R0 + v_r c T_c)/c`` advances by ``2pi (2 v_r/lambda) T_c`` per chirp).

**Element ordering / array handedness.** With the same conjugation, an element
displaced by ``d`` *towards* the target sees a shorter delay and therefore a beat
phase ``-2pi d sin(theta)/lambda``. rd_synth uses ``+pi * v * sin(theta)`` for
virtual element ``v`` (see its "Spatial phase" comment), with ``sin(theta)`` measured
along ``u = normalise(z_up x boresight)``. The two agree exactly if the element
*index* runs along ``-u`` -- i.e. this package numbers array elements from the
positive-azimuth side towards the negative one. That is a labelling (handedness)
convention, not a physical difference, and we honour it here by **reversing the
antenna index** of the extracted CFR (`_ANTENNA_INDEX_REVERSED`). Reversing the index
of a `PlanarArray` is exact, not approximate: its normalized positions are symmetric
about the array centre (`antenna_array.py`: ``y = d_h*j - (num_cols-1)*d_h/2``), so
index reversal is a mirror about that centre. Verified empirically: without the
reversal a target at ``sin(az) = +0.37`` lands at the mirrored angle-FFT bin.

Deliberate approximations (all shared with, or milder than, rd_synth)
---------------------------------------------------------------------
* **Native Doppler evolution**: one `PathSolver` solve per frame; the geometry
  (delays, angles, amplitudes) is frozen over the CPI and only the per-path Doppler
  phase evolves across chirps. This is Sionna's own ``||v|| << c`` model and the
  classical stop-and-hop assumption. `rt_retrace_reference` re-solves the geometry
  per chirp and `doppler_error_study` quantifies the difference -- run it rather than
  trusting this paragraph.
* **Intra-frame range migration** is absent from the native path (the beat frequency
  is not re-derived per chirp), same order of magnitude as rd_synth's constant-
  amplitude approximation: a 5 m/s target moves 2.4 cm over a 4.8 ms TI CPI, ~0.3 of
  a 7.5 cm range bin.
* **float32 delay phase**: DrJit computes ``2pi f_c tau`` in float32 before wrapping;
  at 78 GHz and 20 m that is ~6.5e4 rad, so the wrapped phase carries ~4e-3 rad of
  rounding noise (~ -48 dBc). Irrelevant for bin-level validation, relevant if you
  ever chase >45 dB phase-coherence numbers out of this path.
* **Ranges are to the reflecting surface**, not to an object's centre: RT reflects
  off real geometry, so a 1 m-radius sphere at 5.4 m peaks ~1 m closer than the
  point-target model would predict. Use small scatterers when comparing bin-for-bin
  against `rd_synth`.
* **TX/RX leakage**: a monostatic link's direct TX-array -> RX-array path is a real
  (huge, near-zero-delay) path. `include_leakage=False` (the default) asks the solver
  for ``los=False``, which removes exactly that path and nothing else -- every target
  return is a reflection of depth >= 1.

Materials: Sionna's defaults make every object a perfect specular mirror
(`RadioMaterial.scattering_coefficient` defaults to 0 and the solver's
`diffuse_reflection` defaults to False), which gives an unrealistic, geometry-
critical RCS lobe for small targets. This module therefore defaults to
``scattering_coefficient=0.3`` with a Lambertian pattern and solves with
``diffuse_reflection=True``; both are exposed as parameters. 0.3 is a plausible
mid-range value for a rough painted/metallic vehicle surface at mmWave -- it is a
modelling choice, not a measured one.

CLI
---
    python -m e2e.ml.rt_gen [--config ti_iwr1443] [--frames 2] [--chirps 16]
                            [--samples 128] [--base-scene flat|free|<sionna scene>]
                            [--target sphere|box] [--no-diffuse]
runs `doppler_error_study` and prints the native-vs-re-trace table. Use
`--target box --no-diffuse` for the deterministic (Monte-Carlo-free) comparison.

Ground-rest placement / local (unshipped) asset library
---------------------------------------------------------
`object_local_height_m` reports each object's unscaled mesh z-extent (bbox height) --
pure constants/cheap file parsing, no Sionna needed -- so `e2e.ml.rt_scenes` can place
every object's CENTER at `0.5 * height * scaling` above the ground. That matters because
`SceneObject.position`'s setter (Sionna) re-centers the mesh's AABB on the given point,
so naively placing every object at world z=0 buries the bottom half of it below the
ground plane. `LOCAL_ASSET_SPECS` registers a small library of higher-fidelity meshes
that live on this workstation only (real Mustang/Charger/semi STLs, an SBR+ pedestrian
OBJ) -- NOT shipped with the repo. Each is loaded from a directory named by an env var
(overriding a machine-specific default constant) and converted to a ground-aligned PLY
in a temp cache dir; if the source file is not found (any other machine, CI), the asset
degrades gracefully to a same-category Sionna-bundled mesh. See `ASSET_LICENSES` --
every local asset is recorded with an explicit UNKNOWN-provenance license status.
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from e2e.ml.assets import DOWNLOADED_ASSET_SPECS, process_asset

# See the module docstring's "Element ordering / array handedness" section.
_ANTENNA_INDEX_REVERSED = True

# Material defaults -- deliberately NOT Sionna's (pure specular mirror); see above.
DEFAULT_SCATTERING_COEFFICIENT = 0.3
DEFAULT_SCATTERING_PATTERN = "lambertian"

# --------------------------------------------------------------------------------
# Real object meshes (campaign R2): Sionna's bundled cars + a procedural pedestrian
# placeholder, replacing sphere-as-car/pedestrian scatterers.
# --------------------------------------------------------------------------------
# Sionna ships 17 car meshes under its own package data, all under the Sionna project's
# Apache-2.0 license (`pip show sionna` -> License: Apache-2.0): the standalone
# `low_poly_car.ply` plus 16 duplicate-geometry, differently-positioned copies used by
# the `simple_street_canyon_with_cars` demo scene (`car-0..7.ply` / `car_1..8.ply` --
# same underlying model baked at 8 different scene slots, shipped twice under both
# naming schemes). `SceneObject.position`'s setter re-centers on the mesh's OWN
# axis-aligned bbox (see `sionna.rt.SceneObject.position`), so these baked offsets do
# not leak into placement -- every name below can be positioned anywhere.
CAR_ASSET_NAMES = (
    ("low_poly_car",)
    + tuple(f"car-{i}" for i in range(8))
    + tuple(f"car_{i}" for i in range(1, 9))
)

# `CAR_ASSET_NAMES` is kept above at its full 17-name inventory (mesh-path resolution,
# license bookkeeping, and any existing scenario that references one of the 16
# scene-slot names by name all still need it) -- but drawing a scene's cars UNIFORMLY
# from all 17 draws the SAME geometry ~85% of the time (see `e2e.ml.rt_scenes`'
# `VEHICLE_CLASS_POOLS`, which uses only this one name plus real downloaded meshes for
# variety; campaign R3).
SIONNA_CAR_REPRESENTATIVE = CAR_ASSET_NAMES[0]  # "low_poly_car"

# No bundled human mesh exists (Sionna ships vehicles, buildings, furniture -- no
# pedestrians). `_pedestrian_mesh_path` procedurally builds a low-poly placeholder
# instead; this sentinel `asset` value selects it in `_object_mesh`.
PEDESTRIAN_ASSET_NAME = "pedestrian_placeholder"

# Human skin at ~77 GHz, order-of-magnitude only (mmWave tissue dielectric tables in
# the Gabriel et al. tradition report skin eps_r roughly 6-8 and conductivity roughly
# 30-40 S/m in this band). NOT a measured or frequency-fitted value for this project --
# a documented approximation until a real tissue model is wired in.
SKIN_RELATIVE_PERMITTIVITY = 6.5
SKIN_CONDUCTIVITY_SPM = 36.0

ASSET_LICENSES = {
    "sionna_cars": {
        "assets": CAR_ASSET_NAMES,
        "count": len(CAR_ASSET_NAMES),
        "source": "sionna-rt PyPI package (sionna/rt/scenes/low_poly_car.ply and "
                  "sionna/rt/scenes/simple_street_canyon_with_cars/meshes/car*.ply)",
        "license": "Apache-2.0",
        "attribution": "(c) The Sionna contributors / NVIDIA; bundled with the "
                       "sionna-rt package, redistributed unmodified.",
    },
    PEDESTRIAN_ASSET_NAME: {
        "source": "procedurally generated at runtime by _pedestrian_mesh_path -- "
                  "no binary is committed to this repository",
        "license": "N/A (no third-party asset; this project's own placeholder geometry)",
        "status": "PLACEHOLDER -- capsule/cylinder primitives, not an artist mesh. "
                  "Slated for replacement by a CC0 human mesh once one is sourced.",
    },
}

# Downloaded vehicle meshes (campaign R3, see `e2e.ml.assets`): real car/truck/bus/
# trolley geometry, decimated + normalized by that module. Every one of these is a
# user-supplied file whose redistribution terms have NOT been checked -- see the
# "license" text below, applied uniformly. No mesh binary is committed to this
# repository (see `e2e.ml.assets`' cache-directory docstring); every consumer degrades
# gracefully to `SIONNA_CAR_REPRESENTATIVE` when the cache/source archive is absent.
for _name, _spec in DOWNLOADED_ASSET_SPECS.items():
    ASSET_LICENSES[_name] = {
        "source": _spec.source,
        "license": "UNKNOWN -- user-supplied; terms not verified; corpus using these is "
                  "INTERNAL-ONLY until cleared",
        "category": _spec.vehicle_class,
        "derivation": _spec.derivation,
    }
del _name, _spec

# --------------------------------------------------------------------------------
# Ground-rest placement: unscaled local mesh z-extents (bbox height), so a caller can
# compute "where must this object's CENTER be so its bbox rests on z=0 after Sionna's
# SceneObject.position setter re-centers it" WITHOUT a ray tracer (see the module
# docstring). Sphere/box are Sionna's own primitives; measured once via
# `sionna.rt.load_mesh(...).bbox()` (see tests/test_ml_rt_meshes.py's gated mesh-scale
# checks for the same numbers from a different angle). Cars share one geometry across
# all of `CAR_ASSET_NAMES` (see that constant's docstring).
# --------------------------------------------------------------------------------
SPHERE_LOCAL_HEIGHT_M = 1.995   # rt.scene.sphere bbox z-extent (unit sphere, diameter ~2)
BOX_LOCAL_HEIGHT_M = 5.0        # Sionna's bundled box mesh (see _box_mesh_path)
CAR_LOCAL_HEIGHT_M = 1.5        # low_poly_car mesh, shared by every CAR_ASSET_NAMES entry
# Procedural pedestrian placeholder: derived exactly from _pedestrian_mesh_path's own
# construction (legs z in [0, 0.85], torso to 1.50, head sphere top at 1.50+0.12=1.62...
# no: head centre 1.50+0.12=1.62, radius 0.12 -> top 1.74) -- see that function.
PEDESTRIAN_LOCAL_HEIGHT_M = 1.74


def object_local_height_m(kind, asset: Optional[str] = None) -> float:
    """Unscaled local z-extent (bbox height, metres) of the mesh `SceneObject.scaling`
    multiplies -- pure constants / cheap local file parsing, no Sionna needed. Callers
    compute a ground-resting placement as `centre_z = 0.5 * object_local_height_m(...) *
    scaling` (world z=0 ground; see the module docstring). Raises `ValueError` for an
    unrecognized (kind, asset) combination.
    """
    from e2e.scenario import ObjectKind

    if kind == ObjectKind.SPHERE:
        return SPHERE_LOCAL_HEIGHT_M
    if kind == ObjectKind.BOX:
        return BOX_LOCAL_HEIGHT_M
    if asset == PEDESTRIAN_ASSET_NAME:
        return PEDESTRIAN_LOCAL_HEIGHT_M
    if asset in CAR_ASSET_NAMES:
        return CAR_LOCAL_HEIGHT_M
    if asset in LOCAL_ASSET_SPECS:
        loaded = _load_local_asset(asset)
        if loaded is not None:
            return loaded[1]
        # Source file absent on this machine: `_object_mesh` will ALSO fall back to a
        # same-category Sionna mesh at load time, so fall back to ITS height here too
        # (keeps placement and the mesh actually loaded consistent).
        return (CAR_LOCAL_HEIGHT_M if LOCAL_ASSET_SPECS[asset].category == "vehicle"
                else PEDESTRIAN_LOCAL_HEIGHT_M)
    if asset in DOWNLOADED_ASSET_SPECS:
        result = process_asset(asset)
        if result is not None:
            return result.height_m
        # Raw source not found on this machine: `_object_mesh` ALSO falls back to
        # SIONNA_CAR_REPRESENTATIVE at load time (see there) -- match its height here.
        return CAR_LOCAL_HEIGHT_M
    raise ValueError(f"no known local height for kind={kind!r} asset={asset!r}")


# --------------------------------------------------------------------------------
# Local (unshipped) asset library: higher-fidelity vehicle/pedestrian meshes that exist
# only on this workstation (see the module docstring). Each entry names an env var
# (falling back to a machine-specific default constant) pointing at the directory that
# holds it; if the file is not found there, every consumer degrades gracefully to a
# Sionna-bundled mesh -- the repo works unmodified for anyone else and in CI.
#
# CRITICAL: none of these binaries are committed to this repository. Provenance/license
# for each is UNKNOWN (see ASSET_LICENSES below) -- a user decision is needed before any
# of them ship in a public dataset or distribution.
# --------------------------------------------------------------------------------
LOCAL_ASSET_DIR_ENV_AVX_MODELS = "E2E_ML_LOCAL_ASSET_DIR_AVX_MODELS"
LOCAL_ASSET_DIR_ENV_AVX_STLS = "E2E_ML_LOCAL_ASSET_DIR_AVX_STLS"
LOCAL_ASSET_DIR_ENV_SBR = "E2E_ML_LOCAL_ASSET_DIR_SBR"

# Machine-specific defaults (this workstation only); env vars above override them.
_DEFAULT_LOCAL_ASSET_DIRS: Dict[str, str] = {
    LOCAL_ASSET_DIR_ENV_AVX_MODELS: r"C:\Users\asf3\workspace\e2e_sim\avx\models",
    LOCAL_ASSET_DIR_ENV_AVX_STLS: r"C:\Users\asf3\Documents\avx\stls",
    LOCAL_ASSET_DIR_ENV_SBR: (r"C:\Users\asf3\Documents\pyaedt_backups\pyaedt_prj_GI6"
                              r"\doppler.pyaedt\sbr_array_32x32"),
}


@dataclass(frozen=True)
class LocalAssetSpec:
    dir_env: str            # env var (falls back to _DEFAULT_LOCAL_ASSET_DIRS[dir_env])
    filename: str            # filename inside that directory
    unit_scale: float        # multiply raw file vertex coordinates by this to get metres
    category: str            # "vehicle" | "pedestrian"


# Inventory (measured empirically -- see the campaign report for the full table of
# every candidate file inspected, including ones NOT wired in here):
#  * mustang-no-wheels.stl: already metres (bbox ~4.77 x 1.81 x 1.24 m -- a real Mustang
#    is ~4.79 m long); unit_scale=1.0.
#  * tractor-trailor.stl: already metres (~16.48 x 3.12 x 4.00 m, a real semi); heavier/
#    longer than a "car" but still a real, distinct vehicle mesh; unit_scale=1.0.
#  * dodgebody_repaired.stl: NOT metres -- raw bbox ~76.9 x 212.6 x 53.25 (unitless CAD
#    units); those numbers only make automotive sense in INCHES (0.0254 m/in ->
#    ~1.95 x 5.40 x 1.35 m, matching a 2013 Dodge Charger's real ~5.03 m length /
#    ~1.9 m width once axis identity is sorted out); unit_scale=0.0254.
#  * person_Unnamed_1.obj (from the SBR+ 32x32 array project): already metres (its
#    scene's bike_wheel.obj bbox is a ~0.62 m diameter, matching a real bicycle wheel,
#    which calibrates the scene's units). Height 1.39 m -- this is a seated/crouched
#    cyclist pose, NOT a standing 1.7 m pedestrian; unit_scale=1.0 (kept physically
#    accurate to the source rather than force-scaled to a standing height, which would
#    distort the pose). Flagged as a caveat, not silently "fixed".
LOCAL_ASSET_SPECS: Dict[str, LocalAssetSpec] = {
    "local_mustang": LocalAssetSpec(LOCAL_ASSET_DIR_ENV_AVX_MODELS,
                                    "mustang-no-wheels.stl", 1.0, "vehicle"),
    "local_tractor_trailer": LocalAssetSpec(LOCAL_ASSET_DIR_ENV_AVX_MODELS,
                                            "tractor-trailor.stl", 1.0, "vehicle"),
    "local_dodge_charger": LocalAssetSpec(LOCAL_ASSET_DIR_ENV_AVX_STLS,
                                          "dodgebody_repaired.stl", 0.0254, "vehicle"),
    "local_pedestrian_rider": LocalAssetSpec(LOCAL_ASSET_DIR_ENV_SBR,
                                             "person_Unnamed_1.obj", 1.0, "pedestrian"),
}
LOCAL_VEHICLE_ASSET_NAMES = tuple(n for n, s in LOCAL_ASSET_SPECS.items()
                                  if s.category == "vehicle")
LOCAL_PEDESTRIAN_ASSET_NAMES = tuple(n for n, s in LOCAL_ASSET_SPECS.items()
                                     if s.category == "pedestrian")

for _name, _spec in LOCAL_ASSET_SPECS.items():
    ASSET_LICENSES[_name] = {
        "source": f"local file {_spec.filename!r} (dir env {_spec.dir_env}, default "
                  f"{_DEFAULT_LOCAL_ASSET_DIRS[_spec.dir_env]!r}) -- NOT bundled with "
                  "this repository, NOT committed",
        "license": "UNKNOWN -- local file, provenance not established; must be cleared "
                  "before any public distribution",
        "category": _spec.category,
        "unit_scale_to_metres": _spec.unit_scale,
    }
del _name, _spec


def _local_asset_dir(dir_env: str) -> Optional[str]:
    """Env var override, else the machine-specific default; `None` if neither is set."""
    return os.environ.get(dir_env) or _DEFAULT_LOCAL_ASSET_DIRS.get(dir_env)


def _local_asset_source_path(name: str) -> Optional[str]:
    """Resolved source file path for a `LOCAL_ASSET_SPECS` entry, or `None` if its
    directory is unknown or the file isn't there (graceful degrade)."""
    spec = LOCAL_ASSET_SPECS.get(name)
    if spec is None:
        return None
    d = _local_asset_dir(spec.dir_env)
    if not d:
        return None
    path = os.path.join(d, spec.filename)
    return path if os.path.isfile(path) else None


def _read_stl(path: str) -> Tuple[List[tuple], List[tuple]]:
    """`(verts, faces)` from an STL file, binary or ASCII. Uses `trimesh` if installed
    (handles more edge cases); otherwise a minimal hand-rolled parser for both variants
    (no third-party mesh library is a hard dependency of this module)."""
    try:
        import trimesh
        mesh = trimesh.load(path, force="mesh", process=False)
        return [tuple(map(float, v)) for v in mesh.vertices], \
               [tuple(map(int, f)) for f in mesh.faces]
    except ImportError:
        pass

    with open(path, "rb") as f:
        header = f.read(80)
        count_bytes = f.read(4)
    is_binary = False
    if len(count_bytes) == 4 and not header.lstrip().lower().startswith(b"solid"):
        is_binary = True
    elif len(count_bytes) == 4:
        # Some binary STLs still start with "solid" (a legal but confusing header) --
        # disambiguate by checking whether the declared triangle count matches the
        # file size exactly (84-byte header/count + 50 bytes/triangle).
        n_tri = struct.unpack("<I", count_bytes)[0]
        is_binary = os.path.getsize(path) == 84 + n_tri * 50
    return _read_stl_binary(path) if is_binary else _read_stl_ascii(path)


def _read_stl_binary(path: str) -> Tuple[List[tuple], List[tuple]]:
    verts: List[tuple] = []
    faces: List[tuple] = []
    with open(path, "rb") as f:
        f.read(80)
        n_tri = struct.unpack("<I", f.read(4))[0]
        for i in range(n_tri):
            data = f.read(50)
            if len(data) < 50:
                break
            vals = struct.unpack("<12f", data[:48])   # normal(3) + 3 vertices(3 each)
            base = len(verts)
            for k in range(3):
                verts.append(tuple(vals[3 + 3 * k: 6 + 3 * k]))
            faces.append((base, base + 1, base + 2))
    return verts, faces


def _read_stl_ascii(path: str) -> Tuple[List[tuple], List[tuple]]:
    verts: List[tuple] = []
    faces: List[tuple] = []
    face_verts: List[tuple] = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("vertex"):
                p = line.split()
                face_verts.append((float(p[1]), float(p[2]), float(p[3])))
                if len(face_verts) == 3:
                    base = len(verts)
                    verts.extend(face_verts)
                    faces.append((base, base + 1, base + 2))
                    face_verts = []
    return verts, faces


def _read_obj(path: str) -> Tuple[List[tuple], List[tuple]]:
    """`(verts, faces)` from a Wavefront OBJ file: vertex positions + triangulated
    faces (fan triangulation for polygons with > 3 vertices). Uses `trimesh` if
    installed, else a minimal parser (positions/faces only -- normals, UVs, materials
    and multiple objects-per-file are ignored, which is fine for our bbox/placement
    purposes)."""
    try:
        import trimesh
        mesh = trimesh.load(path, force="mesh", process=False)
        return [tuple(map(float, v)) for v in mesh.vertices], \
               [tuple(map(int, f)) for f in mesh.faces]
    except ImportError:
        pass

    verts: List[tuple] = []
    faces: List[tuple] = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    vi = int(tok.split("/")[0])
                    idx.append(vi - 1 if vi > 0 else len(verts) + vi)
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))
    return verts, faces


_local_asset_cache: Dict[str, Tuple[str, float]] = {}


def _load_local_asset(name: str) -> Optional[Tuple[str, float]]:
    """Convert (once per process) a `LOCAL_ASSET_SPECS` entry into a ground-aligned,
    metre-scaled PLY cached under a temp dir. Returns `(ply_path, height_m)`, or `None`
    if the source file isn't found on this machine or fails to parse (graceful
    degrade -- callers fall back to a Sionna-bundled mesh)."""
    if name in _local_asset_cache:
        return _local_asset_cache[name]
    spec = LOCAL_ASSET_SPECS.get(name)
    if spec is None:
        return None
    src = _local_asset_source_path(name)
    if src is None:
        return None
    try:
        ext = os.path.splitext(src)[1].lower()
        verts, faces = _read_obj(src) if ext == ".obj" else _read_stl(src)
        if not verts or not faces:
            return None
        s = float(spec.unit_scale)
        zs_scaled = [v[2] * s for v in verts]
        z_min = min(zs_scaled)
        # Rebase so the mesh's OWN local bbox already rests at z=0 (feet/wheels-down) --
        # matches _pedestrian_mesh_path's convention and makes the cached ply's frame
        # intuitive; only the SPAN (not this offset) drives ground-rest placement (see
        # object_local_height_m), but a self-consistent local frame is good hygiene.
        scaled_verts = [(v[0] * s, v[1] * s, v[2] * s - z_min) for v in verts]
        height = max(zs_scaled) - z_min
        d = tempfile.mkdtemp(prefix="e2e-rt-local-asset-")
        ply_path = os.path.join(d, f"{name}.ply")
        _write_ply(ply_path, scaled_verts, faces)
    except (OSError, ValueError, IndexError, struct.error):
        return None
    result = (ply_path, float(height))
    _local_asset_cache[name] = result
    return result


# "flat" base scene: a single large ground rectangle. Mitsuba's built-in `rectangle`
# shape is the [-1,1]^2 unit plane in z=0 with a +z normal, so one scale transform is
# the whole scene -- no mesh file, no asset licensing, loads in milliseconds.
_GROUND_HALF_EXTENT_M = 200.0
_GROUND_MATERIAL = "concrete"   # ITU table entry valid over 1-100 GHz (77 GHz included)

# Note the id spelling: Sionna names the loaded SceneObject after the shape id with the
# "mesh-" prefix stripped, so a bsdf id equal to that stem ("e2e-ground") collides with
# the object and `Scene.add` raises "Name '...' is already used by another item".
_FLAT_SCENE_XML = f"""<scene version="2.1.0">
  <bsdf type="itu-radio-material" id="e2e-ground-mat">
      <string name="type" value="{_GROUND_MATERIAL}"/>
      <float name="thickness" value="0.1"/>
  </bsdf>
  <shape type="rectangle" id="mesh-e2e-ground">
      <transform name="to_world">
          <scale x="{_GROUND_HALF_EXTENT_M}" y="{_GROUND_HALF_EXTENT_M}" z="1"/>
      </transform>
      <ref id="e2e-ground-mat" name="bsdf"/>
  </shape>
</scene>
"""

# "free": no ground, no clutter -- only the scenario's own objects. Useful when
# validating bin placement against the point-target model, which has no ground either.
_FREE_SCENE_XML = """<scene version="2.1.0">
</scene>
"""

_SYNTHETIC_SCENE_XML = {"flat": _FLAT_SCENE_XML, "free": _FREE_SCENE_XML}
_synthetic_scene_paths: Dict[str, str] = {}


def _synthetic_scene_path(name: str) -> str:
    """Write (once per process) and return the path of a built-in synthetic scene XML."""
    path = _synthetic_scene_paths.get(name)
    if path is None:
        d = tempfile.mkdtemp(prefix="e2e-rt-scene-")
        path = os.path.join(d, f"{name}.xml")
        with open(path, "w") as f:
            f.write(_SYNTHETIC_SCENE_XML[name])
        _synthetic_scene_paths[name] = path
    return path


def _resolve_device(dev):
    """`None` -> the library device; anything else -> `torch.device(dev)`."""
    if dev is None:
        from e2e.ml.rd_synth import device as _lib_device

        return _lib_device
    return torch.device(dev)


# --------------------------------------------------------------------------------
# Scene construction
# --------------------------------------------------------------------------------
@dataclass
class RTScene:
    """A built, reusable Sionna RT radar scene.

    `build_rt_scene` returns this rather than the bare `sionna.rt.Scene` so callers
    (and the re-trace / error-study paths) can move the radar and the individual
    scatterers between solves without re-parsing the base scene, which is by far the
    most expensive part of setup. The raw Sionna scene is `.scene`.
    """
    scene: Any                        # sionna.rt.Scene
    tx: Any                           # sionna.rt.Transmitter
    rx: Any                           # sionna.rt.Receiver
    objects: Dict[str, Any]           # scenario object name -> sionna.rt.SceneObject
    cfg: Any                          # RadarConfig
    base_scene: str
    f_center_hz: float
    solver: Any = None                # sionna.rt.PathSolver (created on first solve)
    materials: Dict[str, Any] = field(default_factory=dict)


def _load_base_scene(rt, base_scene: str):
    """`"flat"` / `"free"` / a Sionna built-in name / a path -> a loaded `Scene`.

    City scenes are loaded with `merge_shapes=True`, which is the difference between a
    city tier being affordable and not. Munich ships as ~1150 individually meshed
    shapes; solving it unmerged costs 45-56 s per frame against 0.07 s for the flat
    scene. Merging collapses it to about 11 objects and the same solve takes ~0.07 s
    while finding the IDENTICAL path set -- measured, not assumed. The merge only
    affects the base scene's own static geometry; scenario objects are added afterwards
    via `scene.edit`, so their per-object velocities (and hence Doppler) are untouched,
    which the tests below check rather than take on trust.

    Synthetic scenes stay unmerged: `flat` is a single rectangle, so merging would be a
    no-op, and leaving the call identical keeps every existing flat-scene result
    byte-for-byte unchanged.
    """
    if base_scene in _SYNTHETIC_SCENE_XML:
        return rt.load_scene(_synthetic_scene_path(base_scene), merge_shapes=False)
    builtin = getattr(rt.scene, base_scene, None)
    if builtin is not None:
        return rt.load_scene(builtin, merge_shapes=True)
    return rt.load_scene(base_scene, merge_shapes=True)


def _box_mesh_path(rt) -> str:
    """Path to Sionna's box *mesh*.

    `rt.scene.box` is the path to a box SCENE (`box/box.xml`), not a mesh, so passing
    it to `SceneObject(fname=...)` raises "Invalid mesh type" -- the ply lives one
    level down at `box/meshes/box.ply`. Falls back to the sphere primitive if a future
    Sionna reorganizes the scene package.
    """
    ply = os.path.join(os.path.dirname(rt.scene.box), "meshes", "box.ply")
    return ply if os.path.isfile(ply) else rt.scene.sphere


def _car_mesh_path(rt, name: str) -> str:
    """Resolve one of `CAR_ASSET_NAMES` to its `.ply` path inside the sionna-rt package.

    `rt.scene.sphere`'s directory is the package's scenes root; the street-canyon demo's
    car meshes live one level down, under its own `meshes/` subdirectory.
    """
    scenes_dir = os.path.dirname(rt.scene.sphere)
    if name == "low_poly_car":
        return os.path.join(scenes_dir, "low_poly_car.ply")
    return os.path.join(scenes_dir, "simple_street_canyon_with_cars", "meshes", f"{name}.ply")


# --------------------------------------------------------------------------------
# Procedural pedestrian placeholder (no bundled mesh exists -- see ASSET_LICENSES).
# Pure Python/stdlib: no Sionna needed to build the mesh file itself, only to load it.
# --------------------------------------------------------------------------------
def _uv_sphere_mesh(radius: float, n_lat: int = 6, n_lon: int = 10):
    """`(verts, faces)` for a UV sphere of `radius`, centred at the origin."""
    verts: List[tuple] = []
    for i in range(n_lat + 1):
        theta = math.pi * i / n_lat                       # 0 (north pole) .. pi (south)
        z = radius * math.cos(theta)
        r = radius * math.sin(theta)
        for j in range(n_lon):
            phi = 2.0 * math.pi * j / n_lon
            verts.append((r * math.cos(phi), r * math.sin(phi), z))
    faces: List[tuple] = []
    for i in range(n_lat):
        for j in range(n_lon):
            a = i * n_lon + j
            b = i * n_lon + (j + 1) % n_lon
            c = (i + 1) * n_lon + (j + 1) % n_lon
            d = (i + 1) * n_lon + j
            if i != 0:
                faces.append((a, b, d))
            if i != n_lat - 1:
                faces.append((b, c, d))
    return verts, faces


def _cylinder_mesh(radius: float, length: float, n_seg: int = 8):
    """`(verts, faces)` for a capped cylinder of `radius`/`length` along z, centred at
    the origin (z in `[-length/2, length/2]`)."""
    top, bot = length / 2.0, -length / 2.0
    verts: List[tuple] = []
    for z in (bot, top):
        for j in range(n_seg):
            phi = 2.0 * math.pi * j / n_seg
            verts.append((radius * math.cos(phi), radius * math.sin(phi), z))
    faces: List[tuple] = []
    for j in range(n_seg):
        j2 = (j + 1) % n_seg
        b0, b1, t0, t1 = j, j2, n_seg + j, n_seg + j2
        faces.append((b0, b1, t1))
        faces.append((b0, t1, t0))
    bot_c = len(verts)
    verts.append((0.0, 0.0, bot))
    for j in range(n_seg):
        faces.append((bot_c, (j + 1) % n_seg, j))
    top_c = len(verts)
    verts.append((0.0, 0.0, top))
    for j in range(n_seg):
        faces.append((top_c, n_seg + j, n_seg + (j + 1) % n_seg))
    return verts, faces


def _capsule_mesh(radius: float, cylinder_length: float, n_lat: int = 8, n_lon: int = 10):
    """`(verts, faces)` for a capsule (cylinder + two hemispherical caps) along z,
    centred at the origin. Total end-to-end length is `cylinder_length + 2*radius`.

    Built as a UV sphere split at the equator, with the two hemispheres pulled apart by
    `cylinder_length` -- the duplicated equator ring (one copy per hemisphere, both at
    radius `radius`, offset by `+-cylinder_length/2`) becomes the cylindrical side, so
    no separate cylinder geometry is needed.
    """
    if n_lat % 2:
        n_lat += 1
    half = n_lat // 2
    rings = []  # (z, r)
    for i in range(half + 1):
        theta = math.pi * i / n_lat
        rings.append((radius * math.cos(theta) + cylinder_length / 2.0, radius * math.sin(theta)))
    for i in range(half, n_lat + 1):
        theta = math.pi * i / n_lat
        rings.append((radius * math.cos(theta) - cylinder_length / 2.0, radius * math.sin(theta)))

    verts: List[tuple] = []
    for z, r in rings:
        for j in range(n_lon):
            phi = 2.0 * math.pi * j / n_lon
            verts.append((r * math.cos(phi), r * math.sin(phi), z))
    faces: List[tuple] = []
    for i in range(len(rings) - 1):
        for j in range(n_lon):
            a = i * n_lon + j
            b = i * n_lon + (j + 1) % n_lon
            c = (i + 1) * n_lon + (j + 1) % n_lon
            d = (i + 1) * n_lon + j
            if rings[i][1] > 1e-9:
                faces.append((a, b, d))
            if rings[i + 1][1] > 1e-9:
                faces.append((b, c, d))
    return verts, faces


def _merge_mesh_parts(parts):
    """Merge `[(verts, faces, translate), ...]` into one `(verts, faces)`, index-offset."""
    verts: List[tuple] = []
    faces: List[tuple] = []
    offset = 0
    for v, f, t in parts:
        verts.extend((x + t[0], y + t[1], z + t[2]) for (x, y, z) in v)
        faces.extend((a + offset, b + offset, c + offset) for (a, b, c) in f)
        offset += len(v)
    return verts, faces


def _write_ply(path: str, verts, faces) -> None:
    """Minimal ASCII PLY writer (triangle mesh, vertex positions only)."""
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for x, y, z in verts:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in faces:
            f.write(f"3 {int(a)} {int(b)} {int(c)}\n")


_pedestrian_mesh_cache_path: Optional[str] = None


def _pedestrian_mesh_path() -> str:
    """Build (once per process) and return the path to a PLACEHOLDER pedestrian mesh.

    No bundled human mesh ships with Sionna (see `ASSET_LICENSES`), so this
    procedurally builds a low-poly capsule-torso + head + 4-limb-cylinder human, ~1.74 m
    tall, and writes it to a temp `.ply` (no binary committed to the repo -- regenerated
    every process). This is a crude silhouette, not an anatomically accurate mesh: it
    exists so a pedestrian scatters radar energy at all (a sphere placeholder is at
    least visibly wrong; this is a documented stand-in) -- a CC0 artist-made mesh should
    replace it.
    """
    global _pedestrian_mesh_cache_path
    if _pedestrian_mesh_cache_path is not None:
        return _pedestrian_mesh_cache_path

    parts = []  # (verts, faces, translate)
    for side in (-1.0, 1.0):                                   # legs: z in [0, 0.85]
        v, f = _cylinder_mesh(0.08, 0.85, n_seg=8)
        parts.append((v, f, (0.0, side * 0.10, 0.425)))
    v, f = _capsule_mesh(0.15, 0.35, n_lat=8, n_lon=10)         # torso: z in [0.85, 1.50]
    parts.append((v, f, (0.0, 0.0, 0.85 + 0.15 + 0.35 / 2.0)))
    for side in (-1.0, 1.0):                                   # arms, roughly at the sides
        v, f = _cylinder_mesh(0.045, 0.55, n_seg=6)
        parts.append((v, f, (0.0, side * 0.22, 1.175)))
    v, f = _uv_sphere_mesh(0.12, n_lat=6, n_lon=10)             # head: top at z ~= 1.74
    parts.append((v, f, (0.0, 0.0, 1.50 + 0.12)))

    verts, faces = _merge_mesh_parts(parts)
    d = tempfile.mkdtemp(prefix="e2e-rt-pedestrian-")
    path = os.path.join(d, "pedestrian_placeholder.ply")
    _write_ply(path, verts, faces)
    _pedestrian_mesh_cache_path = path
    return path


def _object_mesh(rt, obj):
    """Mesh source for a scenario `SceneObject` (same dispatch as scenario_runner)."""
    from e2e.scenario import ObjectKind

    if obj.kind == ObjectKind.SPHERE:
        return rt.scene.sphere
    if obj.kind == ObjectKind.BOX:
        return _box_mesh_path(rt)
    if not obj.asset:
        raise ValueError(f"object {obj.name!r} has kind=MESH but no `asset` mesh path")
    if obj.asset == PEDESTRIAN_ASSET_NAME:
        return _pedestrian_mesh_path()
    if obj.asset in CAR_ASSET_NAMES:
        return _car_mesh_path(rt, obj.asset)
    if obj.asset in LOCAL_ASSET_SPECS:
        loaded = _load_local_asset(obj.asset)
        if loaded is not None:
            return loaded[0]
        # Source file not found on this machine: degrade to a same-category
        # Sionna-bundled mesh (see the module docstring / LOCAL_ASSET_SPECS).
        spec = LOCAL_ASSET_SPECS[obj.asset]
        return _pedestrian_mesh_path() if spec.category == "pedestrian" \
            else _car_mesh_path(rt, "low_poly_car")
    if obj.asset in DOWNLOADED_ASSET_SPECS:
        result = process_asset(obj.asset)
        if result is not None:
            return result.ply_path
        # Raw archive/cache not found on this machine (or `7z` unavailable): every
        # DOWNLOADED_ASSET_SPECS entry is a vehicle (no pedestrian among them), so
        # degrade to the one representative Sionna car mesh -- same precedent as
        # LOCAL_ASSET_SPECS above.
        return _car_mesh_path(rt, SIONNA_CAR_REPRESENTATIVE)
    return obj.asset


def build_rt_scene(scenario, cfg, *, base_scene: str = "flat", frame_idx: int = 0,
                   scattering_coefficient: float = DEFAULT_SCATTERING_COEFFICIENT,
                   scattering_pattern: str = DEFAULT_SCATTERING_PATTERN,
                   pattern: str = "iso", polarization: str = "V") -> RTScene:
    """Build a monostatic FMCW-radar Sionna RT scene for `scenario` at `frame_idx`.

    * The scenario's first RADAR node becomes a co-located `Transmitter`/`Receiver`
      pair at `radar_pose(scenario, frame_idx)`, both aimed along the pose boresight.
      Their `PlanarArray`s are 1 x `cfg.n_tx` (spacing `n_rx * lambda/2`) and
      1 x `cfg.n_rx` (spacing `lambda/2`), i.e. the uniform `lambda/2` virtual ULA
      with element index `v = t*n_rx + r` that `rd_synth` assumes.
      `synthetic_array=False` at solve time, so every element is really traced.
    * Every `scenario.objects` entry becomes a `SceneObject` (sphere/box/mesh, same
      dispatch as `scenario_runner._add_scene_object`) with its declared ITU material
      but a NON-zero `scattering_coefficient` (see the module docstring) and its
      per-frame velocity from `frame_scatterers`, so Sionna computes a real per-path
      Doppler shift for it.
    * `scene.frequency` is the chirp centre `f0 + B/2`, which makes Sionna's
      wavelength-normalised element spacing equal `cfg.wavelength_m / 2`.

    `base_scene`: `"flat"` (a 400 m ground plane, the default), `"free"` (no ground --
    only the scenario's objects, matching the point-target model's environment), any
    Sionna built-in scene name (`"munich"`, `"etoile"`, ...), or a path to a scene XML.
    """
    import sionna.rt as rt

    from e2e.ml.scatterers import frame_scatterers, radar_pose

    pose = radar_pose(scenario, frame_idx)
    scats = frame_scatterers(scenario, frame_idx, dt=1.0 / float(cfg.frame_rate_hz))
    if len(scats) != len(scenario.objects):
        raise RuntimeError("frame_scatterers/scenario.objects length mismatch")

    f_center = float(cfg.f0_hz) + float(cfg.bandwidth_hz) / 2.0

    scene = _load_base_scene(rt, base_scene)
    scene.frequency = f_center
    # TX elements are spaced n_rx * lambda/2 so the (tx, rx) pairs tile a uniform
    # lambda/2 virtual ULA -- the geometry rd_synth's `pi * v * sin(theta)` assumes.
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=int(cfg.n_tx),
                                    horizontal_spacing=0.5 * int(cfg.n_rx),
                                    pattern=pattern, polarization=polarization)
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=int(cfg.n_rx),
                                    horizontal_spacing=0.5,
                                    pattern=pattern, polarization=polarization)

    position = [float(c) for c in pose.position]
    # look_at aims the device's local +x at the target point; its local +y is then
    # normalise(z_up x boresight) -- exactly `rd_synth.array_axis`'s ULA axis, which
    # is what makes the PlanarArray's element axis and the analytic model's agree.
    aim = [float(p + b) for p, b in zip(pose.position, pose.boresight)]
    tx = rt.Transmitter(name="e2e-radar-tx", position=position, look_at=aim)
    rx = rt.Receiver(name="e2e-radar-rx", position=position, look_at=aim)
    scene.add(tx)
    scene.add(rx)

    objects: Dict[str, Any] = {}
    materials: Dict[str, Any] = {}
    for obj, sc in zip(scenario.objects, scats):
        if obj.material == "skin":
            # No ITU table entry for tissue -- a plain RadioMaterial with the
            # approximate mmWave skin dielectric constants (see SKIN_* above).
            mat = rt.RadioMaterial(
                f"e2e-rt-mat-{obj.name}", thickness=0.01,
                relative_permittivity=SKIN_RELATIVE_PERMITTIVITY,
                conductivity=SKIN_CONDUCTIVITY_SPM,
                scattering_coefficient=float(scattering_coefficient),
                scattering_pattern=scattering_pattern,
                color=obj.color if obj.color is not None else (0.9, 0.75, 0.65),
            )
        else:
            mat = rt.ITURadioMaterial(
                f"e2e-rt-mat-{obj.name}", obj.material, thickness=0.01,
                scattering_coefficient=float(scattering_coefficient),
                scattering_pattern=scattering_pattern,
                color=obj.color if obj.color is not None else (0.8, 0.1, 0.1),
            )
        so = rt.SceneObject(fname=_object_mesh(rt, obj), name=f"e2e-rt-obj-{obj.name}",
                            radio_material=mat)
        scene.edit(add=[so])
        so.scaling = float(obj.scaling)
        so.position = [float(c) for c in sc.position]
        # Per-object velocity is what `field_calculator._update_doppler_shift` reads at
        # each scattering interaction; without it every path's Doppler is identically 0.
        so.velocity = [float(v) for v in sc.velocity]
        objects[obj.name] = so
        materials[obj.name] = mat

    return RTScene(scene=scene, tx=tx, rx=rx, objects=objects, cfg=cfg,
                   base_scene=base_scene, f_center_hz=f_center, materials=materials)


# --------------------------------------------------------------------------------
# Solve + CFR -> beat cube
# --------------------------------------------------------------------------------
def beat_frequencies(cfg) -> np.ndarray:
    """Baseband CFR frequencies for one chirp: `f0 + S n/fs - (f0 + B/2)`, n < n_samples.

    See the module docstring, equation (3): sampling the CFR on this grid IS sampling
    the dechirped beat signal along the ramp.
    """
    n = np.arange(int(cfg.n_samples), dtype=np.float64)
    f_rf = float(cfg.f0_hz) + float(cfg.ramp_slope_hzps) * n / float(cfg.fs_hz)
    return f_rf - (float(cfg.f0_hz) + float(cfg.bandwidth_hz) / 2.0)


def _solve(rt_scene: RTScene, *, max_depth: int, include_leakage: bool,
           diffuse_reflection: bool, specular_reflection: bool, refraction: bool,
           seed: int, samples_per_src: Optional[int] = None):
    import sionna.rt as rt

    if rt_scene.solver is None:
        rt_scene.solver = rt.PathSolver()
    kwargs = dict(
        scene=rt_scene.scene, max_depth=int(max_depth),
        los=bool(include_leakage),           # the ONLY los path here is TX->RX leakage
        specular_reflection=bool(specular_reflection),
        diffuse_reflection=bool(diffuse_reflection),
        refraction=bool(refraction),
        synthetic_array=False, seed=int(seed),
    )
    if samples_per_src is not None:
        kwargs["samples_per_src"] = int(samples_per_src)
    return rt_scene.solver(**kwargs)


#: DrJit refuses to allocate an array with more than 2**32 entries. `cfr` materialises
#: [rx, rx_ant, tx, tx_ant, num_paths, n_chirps, n_freqs] BEFORE summing over paths, so
#: the safe frequency-chunk size depends on how many paths the solve actually found --
#: which a fixed default cannot know. Real decimated vehicle meshes with diffuse
#: scattering produce ~15k paths where the old sphere scenes produced tens, and the
#: fixed 128-frequency chunk then asked for 4.4e9 entries and failed outright. Budget
#: at half the hard limit so the peak allocation has room around it.
_DRJIT_ELEMENT_BUDGET = 2 ** 31


def _num_paths(paths) -> int:
    """Path count of a solved `Paths`, or 0 if it cannot be determined cheaply."""
    for attr in ("a", "tau"):
        arr = getattr(paths, attr, None)
        if arr is None:
            continue
        shape = getattr(arr, "shape", None)
        if shape:
            # Path axis is the last one for tau ([..., num_paths]) and second-to-last
            # for a; taking the max is a safe over-estimate for budgeting purposes.
            return int(max(shape))
    return 0


def _cfr_freq_chunk(paths, cfg, *, n_chirps: int, requested: int) -> int:
    """Largest frequency chunk that keeps `cfr`'s pre-sum tensor inside DrJit's limit.

    Returns `requested` when the solve is small enough to need no reduction, so ordinary
    scenes keep their previous behaviour (and previous numbers) exactly.
    """
    requested = max(1, int(requested))
    n_paths = _num_paths(paths)
    if n_paths <= 0:
        return requested
    per_freq = max(1, int(cfg.n_rx) * int(cfg.n_tx) * n_paths * int(n_chirps))
    allowed = max(1, _DRJIT_ELEMENT_BUDGET // per_freq)
    if allowed >= requested:
        return requested
    if allowed < 1:  # pragma: no cover -- would need ~1e6 paths
        raise RuntimeError(
            f"a single frequency bin needs {per_freq} elements, over DrJit's limit; "
            f"the solve found {n_paths} paths. Reduce max_depth or scene complexity."
        )
    return int(allowed)


def cfr_from_paths(paths, cfg, *, n_chirps: int, freq_chunk: int = 128) -> np.ndarray:
    """`Paths` -> RAW CFR cube `[n_rx_ant, n_tx_ant, n_chirps, n_samples]`.

    Sionna-specific half of what used to be a single `_beat_from_paths`: samples the
    CFR on the ramp's frequency grid (`beat_frequencies`) but does NOT conjugate or
    antenna-reverse -- that generic tensor mapping is `e2e.chain.dechirp.beat_from_cfr`,
    the ONE implementation both `_beat_from_paths` (below) and `RTEnvironmentBlock`
    delegate to. dtype/scale are whatever `paths.cfr` returns (typically complex64),
    unchanged from before this split.

    `freq_chunk` bounds peak memory: `cfr` materialises a
    `[rx, rx_ant, tx, tx_ant, num_paths, n_chirps, n_freqs]` tensor *before* summing
    over paths, so a full 512-sample / 192-chirp / 50-path call needs hundreds of MB.
    Chunking over frequencies is free -- the expensive ray tracing already happened.
    """
    freqs = beat_frequencies(cfg)
    n_samples = freqs.size
    chunk = _cfr_freq_chunk(paths, cfg, n_chirps=n_chirps, requested=freq_chunk)
    out: List[np.ndarray] = []
    for lo in range(0, n_samples, chunk):
        h = paths.cfr(
            frequencies=freqs[lo:lo + chunk],
            sampling_frequency=1.0 / float(cfg.chirp_period_s),
            num_time_steps=int(n_chirps),
            normalize_delays=False,   # absolute delay IS the range -- never normalize
            normalize=False,          # keep physical amplitudes
            out_type="numpy",
        )
        # h: [num_rx, num_rx_ant, num_tx, num_tx_ant, n_chirps, n_freqs]; one tx/rx
        # device each, so indices 0 select them.
        out.append(np.asarray(h)[0, :, 0, :, :, :])
    return np.ascontiguousarray(np.concatenate(out, axis=-1))


def _beat_from_paths(paths, cfg, *, n_chirps: int, freq_chunk: int = 128) -> np.ndarray:
    """`Paths` -> beat cube `[n_rx_ant, n_tx_ant, n_chirps, n_samples]`, complex64.

    Applies equation (3): CFR on the ramp's frequency grid (`cfr_from_paths`),
    conjugated, with the antenna index reversed (see "Element ordering / array
    handedness") -- via `e2e.chain.dechirp.beat_from_cfr`, so this and
    `RTEnvironmentBlock` share exactly one implementation of that mapping.
    """
    raw = cfr_from_paths(paths, cfg, n_chirps=n_chirps, freq_chunk=freq_chunk)
    from e2e.chain.dechirp import beat_from_cfr

    beat = beat_from_cfr(torch.from_numpy(raw))
    return np.ascontiguousarray(beat.numpy(), dtype=np.complex64)


def mimo_combine(cfg, beat: np.ndarray) -> np.ndarray:
    """Beat cube `[n_rx, n_tx, n_chirps, n_samples]` -> ADC cube `[n_rx, n_chirps, n_samples]`.

    Thin numpy<->torch wrapper around `e2e.chain.dechirp.mimo_combine` -- the ONE
    implementation of the TDM/DDMA combine (see that function's docstring for the
    scheme semantics, which mirror `rd_synth.synthesize_adc`'s per-chirp TX factor).
    """
    from e2e.chain.dechirp import mimo_combine as _mimo_combine_torch

    beat_t = torch.from_numpy(np.ascontiguousarray(beat))
    adc_t = _mimo_combine_torch(cfg, beat_t)
    return np.ascontiguousarray(adc_t.numpy())


# --------------------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------------------
def _peak_reference_amplitude(cfg, adc: np.ndarray, min_range_m: float) -> float:
    """Per-sample amplitude of the strongest target, in `rd_synth`'s SNR convention.

    rd_synth defines `snr_db` as the post-2-D-FFT SNR of the strongest scatterer at
    its peak, with coherent gain `G = n_samples * n_chirps_per_tx` for an *unwindowed*
    2-D FFT of one RX channel using one TX's chirps, and derives the noise power from
    that scatterer's per-sample amplitude `A_max`. Ray tracing gives no such scalar,
    so we invert the same relation: measure the unwindowed 2-D FFT peak `P` and take
    `A_max = P / G`. Identical convention, measured instead of assumed.

    Range bins closer than `min_range_m` (and their negative-frequency mirrors) are
    excluded: a monostatic scene's ground bounce / residual coupling sits at near-zero
    range and would otherwise set the noise floor for a distant target.
    """
    mag = np.abs(np.fft.fft2(_snr_reference_chirps(cfg, adc), axes=(1, 2)))
    n_samples = mag.shape[-1]
    guard = int(np.ceil(float(min_range_m) / float(cfg.range_resolution_m)))
    guard = min(guard, n_samples // 2)
    if guard > 0:
        mag = mag[:, :, guard:n_samples - guard]
    if mag.size == 0:
        return 0.0
    return float(mag.max()) / _coherent_gain(cfg, adc)


def _snr_reference_chirps(cfg, adc):
    """The chirps of a single TX, i.e. what the SNR convention integrates coherently."""
    return adc[:, ::int(cfg.n_tx), :] if str(cfg.mimo).lower() == "tdm" else adc


def _coherent_gain(cfg, adc) -> float:
    """`n_samples * n_chirps_per_tx` of the ACTUAL cube (not of `cfg`).

    Reading the chirp count off the array matters because `rt_retrace_reference` and
    `doppler_error_study` truncate the CPI (`n_chirps_cap`) without necessarily
    replacing `cfg`; a stale `cfg.n_chirps` would mis-scale the noise.
    """
    return float(adc.shape[-1]) * float(_snr_reference_chirps(cfg, adc).shape[1])


def _add_awgn(cfg, adc: torch.Tensor, snr_db, seed, min_range_m: float) -> torch.Tensor:
    """Add calibrated complex AWGN, reusing rd_synth's documented SNR convention."""
    if snr_db is None:
        return adc
    adc_np = adc.cpu().numpy()
    a_max = _peak_reference_amplitude(cfg, adc_np, min_range_m)
    coh_gain = _coherent_gain(cfg, adc_np)
    # An empty (no-path) scene has no reference amplitude; emit unit-variance noise so
    # the frame is still a usable "background only" sample -- same fallback as rd_synth.
    sigma2 = (a_max ** 2 * coh_gain / (10.0 ** (float(snr_db) / 10.0))) if a_max > 0 else 1.0
    gen = torch.Generator(device=adc.device)
    gen.manual_seed(int(seed) if seed is not None
                    else int(torch.randint(0, 2 ** 62, (1,)).item()))
    w = torch.randn(tuple(adc.shape) + (2,), generator=gen, device=adc.device,
                    dtype=torch.float32) * math.sqrt(sigma2 / 2.0)
    return adc + torch.view_as_complex(w.contiguous())


# --------------------------------------------------------------------------------
# Native (single-solve) generation
# --------------------------------------------------------------------------------
def rt_cfr_frame(cfg, scenario, *, frame_idx: int = 0, base_scene: str = "flat",
                 device=None, rt_scene: Optional[RTScene] = None, max_depth: int = 2,
                 include_leakage: bool = False, diffuse_reflection: bool = True,
                 specular_reflection: bool = True, refraction: bool = False,
                 solver_seed: int = 41, freq_chunk: int = 128) -> torch.Tensor:
    """Ray-trace one radar frame and return its RAW channel frequency response.

    `complex64 [n_rx, n_tx, n_chirps, n_samples]` on `device` -- the pipeline's
    DOMAIN_CFR contract (see `e2e.frames`), sampled on the FMCW ramp's beat-frequency
    grid (`beat_frequencies(cfg)`) but NOT YET conjugated, antenna-reversed or
    MIMO-combined. `e2e.chain.dechirp.DechirpBlock(cfg)` is the bridge from here to a
    dechirped ADC cube; `rt_synthesize_adc` (below) and `e2e.environment.blocks.
    RTEnvironmentBlock` both build on this one extraction path so the ray-tracing +
    CFR-sampling logic exists exactly once. See `build_rt_scene` for the scene/`rt_scene`
    parameters and `_solve` for the solver ones (both shared verbatim with
    `rt_synthesize_adc`).
    """
    dev = _resolve_device(device)
    if rt_scene is None:
        rt_scene = build_rt_scene(scenario, cfg, base_scene=base_scene, frame_idx=frame_idx)

    paths = _solve(rt_scene, max_depth=max_depth, include_leakage=include_leakage,
                   diffuse_reflection=diffuse_reflection,
                   specular_reflection=specular_reflection, refraction=refraction,
                   seed=solver_seed)
    raw = cfr_from_paths(paths, cfg, n_chirps=int(cfg.n_chirps), freq_chunk=freq_chunk)
    return torch.as_tensor(raw, dtype=torch.complex64, device=dev)


def rt_synthesize_adc(cfg, scenario, *, frame_idx: int = 0, base_scene: str = "flat",
                      snr_db: Optional[float] = 30.0, seed: Optional[int] = None,
                      device=None, rt_scene: Optional[RTScene] = None,
                      max_depth: int = 2, include_leakage: bool = False,
                      diffuse_reflection: bool = True, specular_reflection: bool = True,
                      refraction: bool = False, solver_seed: int = 41,
                      freq_chunk: int = 128,
                      snr_ref_min_range_m: Optional[float] = None) -> torch.Tensor:
    """Ray-trace one radar frame and return its dechirped ADC cube.

    Drop-in replacement for `e2e.ml.rd_synth.synthesize_adc(cfg, scatterers, pose, ...)`
    at the dataset level: same return contract, `complex64 [n_rx, n_chirps, n_samples]`
    on `device`, consumable by `e2e.ml.transforms` unchanged.

    ONE `PathSolver` solve is performed; the chirp axis comes from Sionna's Doppler
    time-evolution (`cfr(..., sampling_frequency=1/T_c, num_time_steps=n_chirps)`), not
    from re-tracing -- see `rt_retrace_reference` / `doppler_error_study` for the
    ground-truth comparison this trades against.

    Parameters
    ----------
    cfg : RadarConfig
    scenario : e2e.scenario.Scenario   (needs at least one RADAR node)
    frame_idx : int                    frame to resolve motion at
    base_scene : str                   see `build_rt_scene`
    snr_db : float or None             post-2-D-FFT SNR of the strongest target
                                       (`None` disables noise); see `_peak_reference_amplitude`
    seed : int or None                 seeds the AWGN only (the RT solve uses `solver_seed`)
    device : torch device or None      defaults to the library device
    rt_scene : RTScene or None         reuse a scene built by `build_rt_scene`
                                       (skips base-scene parsing); built here if None
    include_leakage : bool             keep the direct TX->RX path (radar TX/RX leakage)
    snr_ref_min_range_m : float or None
        Range guard for the noise-calibration peak search; defaults to
        `3 * cfg.range_resolution_m`.
    """
    dev = _resolve_device(device)
    s_pars = rt_cfr_frame(cfg, scenario, frame_idx=frame_idx, base_scene=base_scene,
                          device=dev, rt_scene=rt_scene, max_depth=max_depth,
                          include_leakage=include_leakage,
                          diffuse_reflection=diffuse_reflection,
                          specular_reflection=specular_reflection, refraction=refraction,
                          solver_seed=solver_seed, freq_chunk=freq_chunk)

    from e2e.chain.dechirp import DechirpBlock

    adc = DechirpBlock(cfg).apply({"s_pars": s_pars})["adc"]

    guard = (3.0 * float(cfg.range_resolution_m) if snr_ref_min_range_m is None
             else float(snr_ref_min_range_m))
    return _add_awgn(cfg, adc, snr_db, seed, guard).to(torch.complex64)


# --------------------------------------------------------------------------------
# Ground truth: re-trace the geometry once per chirp
# --------------------------------------------------------------------------------
def rt_retrace_reference(cfg, scenario, *, frame_idx: int = 0, base_scene: str = "flat",
                         n_chirps_cap: Optional[int] = None, snr_db: Optional[float] = None,
                         seed: Optional[int] = None, device=None,
                         rt_scene: Optional[RTScene] = None, max_depth: int = 2,
                         include_leakage: bool = False, diffuse_reflection: bool = True,
                         specular_reflection: bool = True, refraction: bool = False,
                         solver_seed: int = 41, freq_chunk: int = 128,
                         snr_ref_min_range_m: Optional[float] = None) -> torch.Tensor:
    """Ground-truth ADC cube: re-solve the scene for **every chirp**.

    Chirp `c` is traced with every moving object advanced to `p0 + v * c * T_c` and
    `num_time_steps=1`, so the slow-time phase evolution comes entirely from the
    re-traced geometry (delays, angles, amplitudes all update) instead of from Sionna's
    first-order Doppler phase rotation. Note that at `num_time_steps=1` the per-path
    Doppler factor is `exp(j*2pi*f_D*0) = 1`, so the object velocities do not
    double-count here -- they are consumed purely as the per-chirp displacement.

    The radar itself is held fixed across the CPI, matching `rd_synth` (whose
    `RadarPose` is per-frame) and `frame_scatterers` (whose velocities are per-object).

    Expensive by design: cost is `n_chirps` solves instead of one. `n_chirps_cap`
    truncates the CPI (the returned cube then has `min(n_chirps, cap)` chirps, which
    is what `doppler_error_study` compares against a matching native run). Returns
    `complex64 [n_rx, n_chirps_used, n_samples]` on `device`.
    """
    dev = _resolve_device(device)
    if rt_scene is None:
        rt_scene = build_rt_scene(scenario, cfg, base_scene=base_scene, frame_idx=frame_idx)

    from e2e.ml.scatterers import frame_scatterers

    scats = frame_scatterers(scenario, frame_idx, dt=1.0 / float(cfg.frame_rate_hz))
    base_pos = {obj.name: np.asarray(sc.position, dtype=np.float64)
                for obj, sc in zip(scenario.objects, scats)}
    vel = {obj.name: np.asarray(sc.velocity, dtype=np.float64)
           for obj, sc in zip(scenario.objects, scats)}

    n_chirps = int(cfg.n_chirps) if n_chirps_cap is None else min(int(cfg.n_chirps),
                                                                  int(n_chirps_cap))
    t_c = float(cfg.chirp_period_s)

    frames: List[np.ndarray] = []
    try:
        for c in range(n_chirps):
            for name, so in rt_scene.objects.items():
                so.position = [float(x) for x in (base_pos[name] + vel[name] * (c * t_c))]
            paths = _solve(rt_scene, max_depth=max_depth, include_leakage=include_leakage,
                           diffuse_reflection=diffuse_reflection,
                           specular_reflection=specular_reflection, refraction=refraction,
                           seed=solver_seed)
            # num_time_steps=1 -> [n_rx_ant, n_tx_ant, 1, n_samples]
            beat = _beat_from_paths(paths, cfg, n_chirps=1, freq_chunk=freq_chunk)
            frames.append(beat[:, :, 0, :])
    finally:
        # Leave the scene at its frame-0 geometry so the handle stays reusable.
        for name, so in rt_scene.objects.items():
            so.position = [float(x) for x in base_pos[name]]

    # [n_rx_ant, n_tx_ant, n_chirps, n_samples] -- same TDM/DDMA combine as the native
    # path, via `mimo_combine` (the one implementation; see its docstring), rather than
    # a second hand-rolled copy of the same math.
    beat_cube = np.stack(frames, axis=2)
    adc_np = mimo_combine(cfg, beat_cube)

    adc = torch.as_tensor(np.ascontiguousarray(adc_np), dtype=torch.complex64, device=dev)
    guard = (3.0 * float(cfg.range_resolution_m) if snr_ref_min_range_m is None
             else float(snr_ref_min_range_m))
    return _add_awgn(cfg, adc, snr_db, seed, guard).to(torch.complex64)


# --------------------------------------------------------------------------------
# Doppler error study: native evolution vs per-chirp re-trace
# --------------------------------------------------------------------------------
def _rd_peak_bin(cfg, adc: torch.Tensor):
    """`(range_bin, doppler_bin)` of the strongest cell, through the shipped transforms."""
    import dataclasses as _dc

    from e2e.ml.transforms import adc_to_rd, tdm_deinterleave

    n_chirps = int(adc.shape[1])
    if str(cfg.mimo).lower() == "tdm":
        sub = _dc.replace(cfg, n_tx=1, mimo="single", n_chirps=n_chirps // int(cfg.n_tx))
        rd = adc_to_rd(sub, tdm_deinterleave(_dc.replace(cfg, n_chirps=n_chirps), adc))
    else:
        rd = adc_to_rd(_dc.replace(cfg, n_chirps=n_chirps), adc)
    power = (rd.abs() ** 2).sum(dim=0)                       # [range, doppler]
    flat = int(power.reshape(-1).argmax())
    return (flat // power.shape[1], flat % power.shape[1]), rd


def doppler_error_study(cfg, scenario, *, n_frames: int = 3, base_scene: str = "flat",
                        n_chirps_cap: Optional[int] = 16, device=None,
                        max_depth: int = 2, include_leakage: bool = False,
                        diffuse_reflection: bool = True, solver_seed: int = 41,
                        freq_chunk: int = 128, verbose: bool = False) -> Dict[str, Any]:
    """Quantify the native Doppler-phase model against the per-chirp re-trace.

    For each of the first `n_frames` frames, both paths are run **noise-free** on the
    same scene handle with the same solver seed, truncated to the same `n_chirps_cap`
    chirps, and compared three ways:

    * `per_chirp_rel_err[c]` -- `||native[:,c,:] - retrace[:,c,:]|| / ||retrace[:,c,:]||`
      (chirp 0 must be ~0: identical geometry, Doppler phase `exp(0)=1`);
    * `rel_rmse` -- the same ratio over the whole cube;
    * `peak_bin_native` / `peak_bin_retrace` -- the range/Doppler peak cell through
      `tdm_deinterleave` + `adc_to_rd`, i.e. whether the two agree where it matters.

    Also returns the measured wall-clock cost of each path and their ratio
    (`cost_multiplier`), which is the price of the ground-truth path.

    **`mc_noise_floor` -- read this before interpreting `rel_rmse`.** Sionna's diffuse
    reflections are Monte-Carlo sampled, so re-solving a scene whose geometry moved by
    even a fraction of a millimetre re-randomises which diffuse paths are found. The
    per-chirp re-trace therefore injects sampling noise of its own, and with
    `diffuse_reflection=True` that noise typically DOMINATES the Doppler-model
    difference this study is trying to measure. Each frame is therefore also re-solved
    once with a perturbed solver seed at unchanged geometry; the resulting
    `mc_noise_floor` is the level below which `rel_rmse` carries no information. For a
    clean measurement of the Doppler model alone, run with
    `diffuse_reflection=False` and a specular-friendly (planar-faced) target, where
    the floor is ~0.

    Note `n_chirps_cap` chirps of a TDM config still cover `cap // n_tx` chirps per TX,
    so keep it a multiple of `cfg.n_tx`.
    """
    dev = _resolve_device(device)
    n_chirps = int(cfg.n_chirps) if n_chirps_cap is None else min(int(cfg.n_chirps),
                                                                  int(n_chirps_cap))
    import dataclasses as _dc

    capped = _dc.replace(cfg, n_chirps=n_chirps)
    common = dict(base_scene=base_scene, max_depth=max_depth,
                  include_leakage=include_leakage, diffuse_reflection=diffuse_reflection,
                  solver_seed=solver_seed, freq_chunk=freq_chunk, device=dev, snr_db=None)

    frames: List[Dict[str, Any]] = []
    t_native_total = 0.0
    t_retrace_total = 0.0
    n_solves_native = 0
    n_solves_retrace = 0

    # Untimed warm-up: DrJit compiles (and caches) the trace/CFR kernels on the first
    # call, which would otherwise be charged entirely to the native path and make the
    # measured cost multiplier meaningless (it came out < 1 before this).
    rt_synthesize_adc(capped, scenario, frame_idx=0,
                      rt_scene=build_rt_scene(scenario, cfg, base_scene=base_scene,
                                              frame_idx=0),
                      **common)

    for k in range(int(n_frames)):
        rt_scene = build_rt_scene(scenario, cfg, base_scene=base_scene, frame_idx=k)

        t0 = time.perf_counter()
        native = rt_synthesize_adc(capped, scenario, frame_idx=k, rt_scene=rt_scene,
                                   **common)
        t_native = time.perf_counter() - t0
        n_solves_native += 1

        t0 = time.perf_counter()
        ref = rt_retrace_reference(capped, scenario, frame_idx=k, rt_scene=rt_scene,
                                   n_chirps_cap=n_chirps, **common)
        t_retrace = time.perf_counter() - t0
        n_solves_retrace += n_chirps

        diff = (native - ref)
        den = ref.abs().pow(2).sum().sqrt().item()
        rel_rmse = float(diff.abs().pow(2).sum().sqrt().item() / den) if den > 0 else float("nan")
        per_chirp = []
        for c in range(n_chirps):
            d = float(diff[:, c, :].abs().pow(2).sum().sqrt().item())
            n = float(ref[:, c, :].abs().pow(2).sum().sqrt().item())
            per_chirp.append(d / n if n > 0 else float("nan"))

        # Monte-Carlo floor: same geometry, different solver seed (see the docstring).
        alt = rt_synthesize_adc(capped, scenario, frame_idx=k, rt_scene=rt_scene,
                                **{**common, "solver_seed": solver_seed + 1})
        alt_den = native.abs().pow(2).sum().sqrt().item()
        mc_floor = (float((alt - native).abs().pow(2).sum().sqrt().item() / alt_den)
                    if alt_den > 0 else float("nan"))

        peak_native, rd_native = _rd_peak_bin(capped, native)
        peak_ref, rd_ref = _rd_peak_bin(capped, ref)
        rd_den = rd_ref.abs().pow(2).sum().sqrt().item()
        rd_rel_rmse = float((rd_native - rd_ref).abs().pow(2).sum().sqrt().item() / rd_den) \
            if rd_den > 0 else float("nan")

        t_native_total += t_native
        t_retrace_total += t_retrace
        frames.append({
            "frame_idx": k,
            "rel_rmse": rel_rmse,
            "mc_noise_floor": mc_floor,
            "per_chirp_rel_err": per_chirp,
            "peak_bin_native": tuple(int(v) for v in peak_native),
            "peak_bin_retrace": tuple(int(v) for v in peak_ref),
            "peak_bin_match": tuple(peak_native) == tuple(peak_ref),
            "rd_rel_rmse": rd_rel_rmse,
            "t_native_s": t_native,
            "t_retrace_s": t_retrace,
        })
        if verbose:
            f = frames[-1]
            print(f"  frame {k}: rel_rmse={f['rel_rmse']:.4g} "
                  f"(mc floor {f['mc_noise_floor']:.4g}) "
                  f"rd_rel_rmse={f['rd_rel_rmse']:.4g} "
                  f"peak {f['peak_bin_native']} vs {f['peak_bin_retrace']} "
                  f"({t_native:.2f}s vs {t_retrace:.2f}s)")

    n_ok = sum(1 for f in frames if f["peak_bin_match"])
    return {
        "config": cfg.name,
        "base_scene": base_scene,
        "n_frames": int(n_frames),
        "n_chirps": n_chirps,
        "n_samples": int(cfg.n_samples),
        "frames": frames,
        "rel_rmse_mean": float(np.mean([f["rel_rmse"] for f in frames])) if frames else float("nan"),
        "rel_rmse_max": float(np.max([f["rel_rmse"] for f in frames])) if frames else float("nan"),
        "rd_rel_rmse_mean": float(np.mean([f["rd_rel_rmse"] for f in frames])) if frames else float("nan"),
        "mc_noise_floor_mean": float(np.mean([f["mc_noise_floor"] for f in frames])) if frames else float("nan"),
        "peak_bin_agreement": (n_ok / len(frames)) if frames else float("nan"),
        "t_native_s": t_native_total,
        "t_retrace_s": t_retrace_total,
        "solves_native": n_solves_native,
        "solves_retrace": n_solves_retrace,
        "cost_multiplier": (t_retrace_total / t_native_total) if t_native_total > 0 else float("nan"),
    }


def format_error_study(result: Dict[str, Any]) -> str:
    """Small fixed-width table for `doppler_error_study`'s return value."""
    lines = [
        f"doppler error study: config={result['config']} base_scene={result['base_scene']} "
        f"n_chirps={result['n_chirps']} n_samples={result['n_samples']}",
        f"{'frame':>5} {'rel_rmse':>11} {'mc_floor':>10} {'rd_rel_rmse':>12} "
        f"{'err[c=0]':>10} {'err[c=last]':>12} {'peak(native)':>14} {'peak(retrace)':>15} "
        f"{'t_nat[s]':>9} {'t_ret[s]':>9}",
    ]
    for f in result["frames"]:
        pc = f["per_chirp_rel_err"]
        lines.append(
            f"{f['frame_idx']:>5} {f['rel_rmse']:>11.3e} {f['mc_noise_floor']:>10.2e} "
            f"{f['rd_rel_rmse']:>12.3e} {pc[0]:>10.2e} {pc[-1]:>12.2e} "
            f"{str(f['peak_bin_native']):>14} {str(f['peak_bin_retrace']):>15} "
            f"{f['t_native_s']:>9.2f} {f['t_retrace_s']:>9.2f}"
        )
    lines.append(
        f"mean rel_rmse={result['rel_rmse_mean']:.3e}  max={result['rel_rmse_max']:.3e}  "
        f"rd_rel_rmse={result['rd_rel_rmse_mean']:.3e}  "
        f"mc noise floor={result['mc_noise_floor_mean']:.3e}  "
        f"peak-bin agreement={result['peak_bin_agreement']:.0%}"
    )
    lines.append(
        f"cost: native {result['t_native_s']:.2f}s ({result['solves_native']} solves) vs "
        f"re-trace {result['t_retrace_s']:.2f}s ({result['solves_retrace']} solves) "
        f"-> {result['cost_multiplier']:.1f}x"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------
def _demo_scenario(n_frames: int, cfg=None, target: str = "sphere"):
    """Moving vehicle target(s) in front of a boresight-aimed radar (study scene).

    `Motion.velocity` is a per-FRAME displacement, and `frame_scatterers` converts it
    with `dt = 1/cfg.frame_rate_hz` -- so the m/frame numbers below are chosen to give
    -5 and +3 m/s at the default 10 Hz frame rate, comfortably inside the preset's
    +-12.8 m/s unambiguous Doppler span.

    `target="box"` uses Sionna's box mesh (scaled to a 4x4x2 m slab whose near face is
    perpendicular to the boresight) instead of spheres. That matters: the specular
    path solver finds NO monostatic return off a curved sphere at all (verified -- a
    specular-only solve of a sphere scene returns zero paths), so sphere targets exist
    only through diffuse scattering, which is Monte-Carlo sampled. A planar-faced box
    gives a deterministic specular return and is the right target for measuring the
    Doppler model itself (`--no-diffuse`).
    """
    from e2e.ml.scatterers import vehicle
    from e2e.scenario import Motion, Node, NodeRole, ObjectKind, Scenario, SceneObject

    dt = 1.0 / float(cfg.frame_rate_hz) if cfg is not None else 0.1
    radar = Node(name="radar", role=NodeRole.RADAR, position=(0.0, 0.0, 1.5),
                 look_at=(10.0, 0.0, 1.5))
    if target == "box":
        objects = [SceneObject(name="slab", kind=ObjectKind.BOX, position=(8.0, 0.0, 1.0),
                               scaling=0.4, material="metal", object_class="vehicle",
                               motion=Motion(velocity=(-5.0 * dt, 0.0, 0.0)))]
    else:
        objects = [
            vehicle("car_a", (6.0, 1.5, 1.0), motion=Motion(velocity=(-5.0 * dt, 0.0, 0.0))),
            vehicle("car_b", (4.0, -1.0, 1.0), motion=Motion(velocity=(3.0 * dt, 0.0, 0.0))),
        ]
    return Scenario(
        name="rt_gen_demo",
        base_scene="flat",
        num_frames=max(2, int(n_frames) + 1),   # >= 2 so motion tracks have a step
        nodes=[radar],
        objects=objects,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m e2e.ml.rt_gen",
        description="Ray-traced FMCW ADC generation: native Doppler vs per-chirp re-trace.",
    )
    p.add_argument("--config", default="radial_like",
                   help="radar preset (e2e.ml.radar_config.PRESETS). Defaults to "
                        "radial_like (12 TX x 16 RX = 192 virtual elements): the label "
                        "grid's 192 azimuth bins and the metric's match tolerance are only "
                        "physically answerable at that array size -- see "
                        "e2e.ml.baseline.resolution_report")
    p.add_argument("--frames", type=int, default=2, help="frames to compare")
    p.add_argument("--chirps", type=int, default=16, help="chirps per frame (CPI truncation)")
    p.add_argument("--samples", type=int, default=128, help="ADC samples per chirp")
    p.add_argument("--base-scene", default="flat", help="flat | free | sionna scene name | path")
    p.add_argument("--max-depth", type=int, default=2, help="PathSolver max_depth")
    p.add_argument("--no-diffuse", action="store_true", help="specular-only solve")
    p.add_argument("--target", default="sphere", choices=("sphere", "box"),
                   help="demo target geometry; 'box' is specular-visible (see _demo_scenario)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    import dataclasses as _dc

    args = build_arg_parser().parse_args(argv)

    from e2e.ml.radar_config import PRESETS

    if args.config not in PRESETS:
        print(f"unknown --config {args.config!r}; choices: {sorted(PRESETS)}", file=sys.stderr)
        return 2
    cfg = _dc.replace(PRESETS[args.config], n_chirps=int(args.chirps),
                      n_samples=int(args.samples))
    problems = cfg.validate()
    if problems:
        print(f"invalid derived config: {problems}", file=sys.stderr)
        return 2

    scenario = _demo_scenario(args.frames, cfg, target=args.target)
    result = doppler_error_study(
        cfg, scenario, n_frames=args.frames, base_scene=args.base_scene,
        n_chirps_cap=int(args.chirps), max_depth=args.max_depth,
        diffuse_reflection=not args.no_diffuse, verbose=True,
    )
    print(format_error_study(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
