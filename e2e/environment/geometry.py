"""
Object geometry: bounding extents, in-plane orientation, and the monostatic SURFACE point.

Why this module exists
----------------------
Two layers need the same answer to the same question -- *where on this object does the
radar's line of sight first touch it?* -- and they must not answer it differently:

* `e2e.ml.rt_signal_chain.coherent_target_cfr` puts each object's coherent point
  scatterer there (its `_specular_point` fallback now delegates here), and
* `e2e.ml.labels.encode_detection_labels` puts each object's objectness footprint there.

A label that disagrees with where the energy was placed is precisely the defect this
module exists to prevent. MEASURED before the fix: a ray-traced car returns from its
nearest surface, ~2.4 m in front of the geometric centre the label marked -- roughly 8
output range bins at the `ti_iwr1443` preset -- so the classification head was trained on
a blind plateau, and `e2e.ml.metrics`' 2.0 m match tolerance turned that offset into a
cliff (oracle AP 1.00 at 1.75 m of offset, 0.21 at 2.00 m, 0.00 from 2.20 m).

The object model
----------------
Each object is a **bounding ellipsoid** with semi-axes equal to its bbox half-extents, in
the object's OWN frame (local +x = length, +y = width, +z = height), rotated by its yaw
about world +z. The surface point is the NEAR intersection of the radar line of sight
with that ellipsoid. Exact for a sphere; for a box-like mesh it is inscribed in the true
bbox, so at intermediate aspect angles it sits slightly INSIDE the real surface (a
4.4 x 1.8 m car viewed at 45 degrees: 1.18 m from the centre, against 1.27 m for an
exact ray/box-slab intersection -- 0.09 m, under a third of a range bin). The two models
agree exactly end-on and broadside. The ellipsoid is used because it is what the ray-
traced signal chain already places its coherent scatterer on, and one shared model beats
two nearly-identical ones.

NOT modelled, deliberately:
  * mesh concavity -- the true nearest visible facet can sit inside the bbox. The RT
    layer prefers the traced mesh vertex (`rt_signal_chain._rt_phase_centres`) when a
    solve is available; the label layer has no solve, so it uses this bound. Measured
    disagreement on the shipped meshes is tens of cm, i.e. well under one range bin.
  * occlusion -- a hidden object still gets a surface point (the RT layer gates on
    visibility separately).

Extents
-------
`object_extent_m` resolves a `e2e.scenario.SceneObject` to `(length, width, height)` in
metres, after `scaling`, following EXACTLY the same asset dispatch (and the same graceful
degradation) as `e2e.ml.rt_scene_build._object_mesh` / `object_local_height_m`: if the
mesh a scene would actually load is the Sionna car (because a downloaded asset's cache is
absent on this machine), the extent reported here is the Sionna car's too. Placement and
labelling therefore cannot disagree about what was put in the scene.

Sionna primitive extents are MEASURED (`sionna.rt.load_mesh(...).bbox()`, sionna-rt
1.2.2), not assumed; downloaded-asset extents are read from `e2e.ml.assets`' own
per-asset stats sidecar -- the bbox of the very mesh that ships. Per-asset (not
per-class) numbers matter: the "truck" class alone spans a 6.5 m delivery van to a
15.7 m semi, i.e. half-lengths 3.25 m to 7.87 m, so a single class-typical figure would
misplace a label by more than the metric's whole 2.0 m match tolerance.

Only numpy + stdlib at module level (asset lookups are imported lazily inside the
functions), so `e2e.ml.labels` stays as cheap to import as it was.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

Vec3 = Tuple[float, float, float]

# --------------------------------------------------------------------------------
# Mesh extents (metres), (length=x, width=y, height=z) in the MESH's own frame.
# MEASURED via sionna.rt.load_mesh(...).bbox() on sionna-rt 1.2.2 -- the z figures agree
# with `rt_scene_build`'s independently recorded SPHERE_LOCAL_HEIGHT_M (1.995),
# BOX_LOCAL_HEIGHT_M (5.0) and CAR_LOCAL_HEIGHT_M (1.5).
# --------------------------------------------------------------------------------
SPHERE_EXTENT_M: Vec3 = (1.9745, 1.9745, 1.9952)   # rt.scene.sphere (unit sphere)
BOX_EXTENT_M: Vec3 = (10.0, 10.0, 5.0)             # Sionna's bundled box mesh
SIONNA_CAR_EXTENT_M: Vec3 = (4.400, 1.800, 1.500)  # low_poly_car.ply, shared by every
                                                   # CAR_ASSET_NAMES entry

#: Procedural pedestrian placeholder, derived from `rt_scene_build._pedestrian_mesh_path`'s
#: own construction: torso capsule radius 0.15 (x half-extent), arms at y = +-0.22 with
#: radius 0.045 (y half-extent 0.265), head top at 1.74.
PEDESTRIAN_EXTENT_M: Vec3 = (0.30, 0.53, 1.74)

#: Local (unshipped, this-workstation-only) assets: bboxes as recorded in
#: `rt_scene_build.LOCAL_ASSET_SPECS`' own measured inventory comment, in the MESH's axis
#: order (these are loaded WITHOUT an axis permutation, unlike `e2e.ml.assets`' downloaded
#: fleet). CAVEAT, inherited not introduced: `local_dodge_charger`'s length runs along its
#: local +y, so yaw -- which rotates the object's local +x onto its heading -- points that
#: one mesh across its direction of travel. That is an asset-normalization gap in the
#: local path; it affects no shipped corpus (these files exist on one machine).
LOCAL_ASSET_EXTENT_M: Dict[str, Vec3] = {
    "local_mustang": (4.77, 1.81, 1.24),
    "local_tractor_trailer": (16.48, 3.12, 4.00),
    "local_dodge_charger": (1.95, 5.40, 1.35),
    "local_pedestrian_rider": (1.80, 0.60, 1.39),   # seated/crouched rider pose
}

#: `(kind, asset)` -> unscaled extent. Keeps the per-object lookup (which can touch the
#: asset stats sidecar on disk) to once per distinct object type per process;
#: `frame_scatterers` calls it for every object of every frame.
_extent_cache: Dict[Tuple[str, Optional[str]], Optional[Vec3]] = {}


def _processed_asset_extent_m(asset: str) -> Optional[Vec3]:
    """Bbox of a downloaded asset's PROCESSED mesh, from `e2e.ml.assets`' stats sidecar.

    That sidecar records the bbox of the exact decimated, metre-normalized mesh
    `rt_scene_build._object_mesh` loads, so a label derived from it lands on the geometry
    that was actually traced. It is only READ here, never built: `process_asset` would
    otherwise extract and decimate a raw archive (seconds to minutes) inside what callers
    treat as a cheap geometry query. `None` means "not processed on this machine" -- and
    a machine without the processed mesh also cannot LOAD it, so `_base_extent_m`
    degrades exactly as `_object_mesh` does, to the Sionna car.
    """
    from e2e.ml.assets import processed_dir

    stats_path = os.path.join(processed_dir(), f"{asset}.ply.stats")
    if not os.path.isfile(stats_path):
        return None
    try:
        with open(stats_path) as f:
            vals = dict(line.strip().split("=", 1) for line in f if "=" in line)
        return (float(vals["length_m"]), float(vals["width_m"]), float(vals["height_m"]))
    except (OSError, KeyError, ValueError):            # pragma: no cover - corrupt sidecar
        return None


def _base_extent_m(kind, asset: Optional[str]) -> Optional[Vec3]:
    """Unscaled `(length, width, height)` for a `(kind, asset)` pair, or None if unknown.

    Dispatch order mirrors `e2e.ml.rt_scene_build._object_mesh` exactly, including its
    graceful degradation to a Sionna-bundled mesh when a local/downloaded asset is not
    present on this machine.
    """
    from e2e.ml.rt_scene_build import (CAR_ASSET_NAMES, LOCAL_ASSET_SPECS,
                                       PEDESTRIAN_ASSET_NAME)
    from e2e.scenario import ObjectKind

    if kind == ObjectKind.SPHERE:
        return SPHERE_EXTENT_M
    if kind == ObjectKind.BOX:
        return BOX_EXTENT_M
    if not asset:
        # kind == MESH with no asset is a scenario error `_object_mesh` raises on; the
        # label layer must not raise, so it is simply "no geometry known" -- a point.
        return None
    if asset == PEDESTRIAN_ASSET_NAME:
        return PEDESTRIAN_EXTENT_M
    if asset in CAR_ASSET_NAMES:
        return SIONNA_CAR_EXTENT_M
    if asset in LOCAL_ASSET_SPECS:
        from e2e.ml.rt_scene_build import _local_asset_source_path

        if _local_asset_source_path(asset) is not None:
            return LOCAL_ASSET_EXTENT_M.get(asset)
        # Source file absent: `_object_mesh` degrades to a same-category Sionna mesh, so
        # report ITS extent (keeps geometry and labels talking about the same object).
        return (PEDESTRIAN_EXTENT_M if LOCAL_ASSET_SPECS[asset].category == "pedestrian"
                else SIONNA_CAR_EXTENT_M)
    from e2e.ml.assets import DOWNLOADED_ASSET_SPECS

    if asset in DOWNLOADED_ASSET_SPECS:
        # `_object_mesh` loads the processed mesh when it exists and the Sionna car when
        # it does not; report whichever of the two a scene would really contain.
        return _processed_asset_extent_m(asset) or SIONNA_CAR_EXTENT_M
    return None


def object_extent_m(obj) -> Optional[Vec3]:
    """`(length, width, height)` of a `e2e.scenario.SceneObject`, metres, after `scaling`.

    `None` when the object's geometry is unknown, which callers must treat as a POINT
    target (surface point == centre, i.e. the pre-2026-08-17 behaviour). An explicit
    `extent_m` attribute on the object, should the scenario layer ever grow one, wins and
    is taken as already-final metres (not multiplied by `scaling`).
    """
    override = getattr(obj, "extent_m", None)
    if override is not None:
        return (float(override[0]), float(override[1]), float(override[2]))

    key = (str(getattr(obj, "kind", "")), getattr(obj, "asset", None))
    if key not in _extent_cache:
        _extent_cache[key] = _base_extent_m(obj.kind, getattr(obj, "asset", None))
    base = _extent_cache[key]
    if base is None:
        return None
    s = float(getattr(obj, "scaling", 1.0) or 1.0)
    return (base[0] * s, base[1] * s, base[2] * s)


# --------------------------------------------------------------------------------
# Orientation
# --------------------------------------------------------------------------------
class _VelocityOnly:
    """Minimal scatterer stand-in for `rt_scene_build.object_yaw_rad`, which reads only
    `.velocity` (see that function and `tests/test_ml_object_yaw.py`)."""

    __slots__ = ("velocity",)

    def __init__(self, velocity):
        self.velocity = velocity


def scene_seed_for(scenario) -> int:
    """A heading seed that is stable across the FRAMES of one scenario.

    Parked objects get a deterministic pseudo-random heading (see
    `rt_scene_build.object_yaw_rad`). Keying that on `frame_idx` -- which both the placer
    and the label layer originally did -- means a PARKED CAR SILENTLY ROTATES from frame
    to frame within a single scene. Placement and labels agreed with each other, so
    nothing failed; the scene was just physically absurd, and any model learning
    frame-to-frame consistency would have been learning noise.

    Keyed on the scenario's name instead: stable across frames, distinct across scenes.
    """
    name = getattr(scenario, "name", None) or ""
    h = 2166136261
    for ch in str(name).encode("utf-8"):
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h


def _placed_object_yaw_rad(velocity, name: str, scene_seed: int) -> Optional[float]:
    """The heading the RT SCENE BUILDER will actually give this object, or None.

    `e2e.ml.rt_scene_build.object_yaw_rad` is what sets `SceneObject.orientation` on the
    real mesh, so it -- not a second opinion computed here -- decides which way an object
    faces. That matters most for a PARKED object: the placer gives it a deterministic
    pseudo-random heading keyed on `(scene_seed, name)`, and a label layer that assumed
    0.0 instead would compute the surface offset off the wrong axis (up to |L-W|/2, i.e.
    1.2 m for a car and 6.4 m for a 15.7 m semi -- comfortably enough to re-open the
    match-tolerance cliff this convention closes).

    Returns None if that function is unavailable, in which case the caller falls back to
    the velocity heading.
    """
    try:
        from e2e.ml.rt_scene_build import object_yaw_rad as _placed
    except Exception:                                  # pragma: no cover - defensive
        return None
    try:
        return float(_placed(_VelocityOnly(tuple(velocity or (0.0, 0.0, 0.0))),
                             name, scene_seed=int(scene_seed)))
    except Exception:                                  # pragma: no cover - defensive
        return None


def object_yaw_rad(obj=None, velocity: Optional[Sequence[float]] = None, *,
                   name: Optional[str] = None, scene_seed: int = 0) -> float:
    """In-plane heading (radians, about world +z) that the object's local +x points along.

    Resolution order:

    1. an explicit `yaw_rad` / `yaw_deg` attribute on `obj` (the scenario layer does not
       carry one today; this is here so that the moment it does, every consumer of this
       function picks it up together);
    2. when `name` is given, whatever `e2e.ml.rt_scene_build.object_yaw_rad` will place
       the mesh at -- the moving object's direction of travel, or a deterministic
       per-object heading if it is parked. Deferring to the placer is the point: a label
       derived from a different heading than the mesh was placed at is wrong by up to
       half the difference between the object's length and its width;
    3. the object's own direction of travel, `atan2(vy, vx)` (`frame_scatterers`
       finite-differences the resolved motion track, so this follows a
       `Motion.angular_velocity_deg` turn as well);
    4. `0.0` -- mesh local +x along world +x. A genuine unknown, not a claim.
    """
    if obj is not None:
        yaw = getattr(obj, "yaw_rad", None)
        if yaw is not None:
            return float(yaw)
        yaw_deg = getattr(obj, "yaw_deg", None)
        if yaw_deg is not None:
            return math.radians(float(yaw_deg))
    if name is not None:
        placed = _placed_object_yaw_rad(velocity, name, scene_seed)
        if placed is not None:
            return placed
    if velocity is not None:
        vx, vy = float(velocity[0]), float(velocity[1])
        if math.hypot(vx, vy) > 1e-9:
            return math.atan2(vy, vx)
    return 0.0


# --------------------------------------------------------------------------------
# Surface point
# --------------------------------------------------------------------------------
def nearest_surface_point(centre, half_extents, observer, *, yaw_rad: float = 0.0,
                          max_fraction: float = 0.99) -> np.ndarray:
    """Near intersection of the `observer` -> `centre` ray with the object's ellipsoid.

    `half_extents` are the ellipsoid's semi-axes in the object's OWN frame
    (`0.5 * (length, width, height)`), and `yaw_rad` rotates that frame about world +z.
    The result lies ON the observer-centre line, so its azimuth and elevation are
    identical to the centre's -- only the RANGE differs. That is what makes the surface/
    centre split cheap for the label encoder: the azimuth residual is unaffected.

    Clamped to `max_fraction * |centre - observer|` so a huge or very close object can
    never place its surface point at or behind the observer.
    """
    centre = np.asarray(centre, dtype=np.float64).reshape(3)
    observer = np.asarray(observer, dtype=np.float64).reshape(3)
    half = np.maximum(np.abs(np.asarray(half_extents, dtype=np.float64).reshape(3)), 1e-6)

    d = centre - observer
    r = float(np.linalg.norm(d))
    if r < 1e-9:
        return centre.copy()
    u = d / r
    if yaw_rad:
        # Express the line of sight in the object's own (yawed) frame: rotating u by
        # -yaw about +z is the same as rotating the ellipsoid by +yaw.
        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
        u_obj = np.array([c * u[0] + s * u[1], -s * u[0] + c * u[1], u[2]])
    else:
        u_obj = u
    t = 1.0 / math.sqrt(float(np.sum((u_obj / half) ** 2)))
    return centre - u * min(t, r * float(max_fraction))


def surface_range_offset_m(centre, half_extents, observer, *, yaw_rad: float = 0.0,
                           max_fraction: float = 0.99) -> float:
    """`delta`: how much CLOSER the surface point is than the centre, metres (>= 0).

    `range_to_surface = range_to_centre - delta`. This is the quantity whose neglect the
    2026-08-17 label-convention change fixes (MEASURED: 2.39 m for a car, 4.17 m bus,
    6.14 m trolley, 7.81 m truck, 0.50 m pedestrian/sphere).
    """
    centre = np.asarray(centre, dtype=np.float64).reshape(3)
    observer = np.asarray(observer, dtype=np.float64).reshape(3)
    p = nearest_surface_point(centre, half_extents, observer, yaw_rad=yaw_rad,
                              max_fraction=max_fraction)
    return float(np.linalg.norm(centre - observer) - np.linalg.norm(p - observer))
