"""
Ray-traced (Sionna RT) scene / mesh / asset construction for `e2e.ml.rt_gen`.

Builds the Sionna RT `Scene` (ground plane / city tile / free space, plus the
monostatic TX/RX array pair and every scenario object) that `e2e.ml.rt_signal_chain`
solves and turns into an ADC cube. Split out of the original `rt_gen.py` -- pure
scene/mesh/asset plumbing here; the CFR/beat-cube physics live in
`e2e.ml.rt_signal_chain`, the native-vs-re-trace experiment harness + CLI in
`e2e.ml.rt_doppler_study`. `e2e.ml.rt_gen` re-exports this module's public and
private names for backward compatibility.

Sionna is imported lazily (inside functions), so `import e2e.ml.rt_scene_build` works
on a machine without Sionna/DrJit -- only the generation calls need it.

Materials: Sionna's defaults make every object a perfect specular mirror
(`RadioMaterial.scattering_coefficient` defaults to 0 and the solver's
`diffuse_reflection` defaults to False), which gives an unrealistic, geometry-
critical RCS lobe for small targets. This module therefore defaults to
``scattering_coefficient=0.3`` with a Lambertian pattern and solves with
``diffuse_reflection=True``; both are exposed as parameters. 0.3 is a plausible
mid-range value for a rough painted/metallic vehicle surface at mmWave -- it is a
modelling choice, not a measured one.

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

import math
import os
import struct
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from e2e.ml.assets import DOWNLOADED_ASSET_SPECS, process_asset

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
# trolley geometry, decimated + normalized by that module. Most of these are
# user-supplied files whose redistribution terms have NOT been checked -- the generic
# "license" text below applies to those. The Kenney fleet (campaign R3 follow-up) is the
# exception: it carries a VERIFIED `DownloadedAssetSpec.license` (CC0, checked against
# the kit's own License.txt), which takes precedence over the generic text. No mesh
# binary is committed to this repository (see `e2e.ml.assets`' cache-directory
# docstring); every consumer degrades gracefully to `SIONNA_CAR_REPRESENTATIVE` when the
# cache/source archive is absent.
for _name, _spec in DOWNLOADED_ASSET_SPECS.items():
    ASSET_LICENSES[_name] = {
        "source": _spec.source,
        "license": _spec.license if _spec.license is not None else (
            "UNKNOWN -- user-supplied; terms not verified; corpus using these is "
            "INTERNAL-ONLY until cleared"),
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
