"""
Ray-traced (Sionna RT) difficulty tiers D0-D3 for the radar-ML data campaign.

This is the RT sibling of `e2e.ml.scenes` (which samples scenes for the analytic
point-target synthesizer, `rd_synth`): it draws a `e2e.scenario.Scenario` whose objects
are REAL meshes -- one representative Sionna car plus real downloaded vehicle meshes
(cars/trucks/buses/trolleys, see `VEHICLE_CLASS_POOLS` and `e2e.ml.assets`) and the
procedural pedestrian placeholder -- instead of sphere-as-target scatterers, for
consumption by `e2e.ml.rt_gen.build_rt_scene` / `rt_synthesize_adc` or
`e2e.environment.scenario_runner`.

Difficulty ramps along TWO axes, not just target count:

* **scatterer type**: D0 is bare metal spheres (a specular-visibility sanity check --
  see `rt_gen`'s module docstring on why a curved target needs diffuse scattering to be
  visible at all), D1 promotes to real vehicle meshes, D2/D3 add the pedestrian
  placeholder.
* **environment**: D0/D1 are a bare flat ground plane (`base_scene="flat"`, see
  `rt_gen._FLAT_SCENE_XML`); D2/D3 both add static box "clutter" (parked-container-style
  obstacles) to a flat ground, D3 denser than D2 -- a richer-but-still-cheap scene.
  Clutter boxes are kept clear of the direct radar-target line of sight (see
  `_sample_clutter_offset`) so they add background return without hiding the tier's
  actual targets.

Vehicle variety: `CAR_ASSET_NAMES` (`e2e.ml.rt_gen`) is 17 Sionna mesh NAMES sharing ONE
geometry, so drawing uniformly from it (the pre-campaign-R3 behaviour) put the SAME car
shape in a scene ~85% of the time. `VEHICLE_CLASS_POOLS` instead keeps exactly ONE
representative Sionna name (`SIONNA_CAR_REPRESENTATIVE`) and fills variety from real,
decimated, metre-scaled downloaded meshes (`e2e.ml.assets.DOWNLOADED_ASSET_SPECS`),
grouped by class (car/truck/bus/trolley) so a tier's vehicle draws are a realistic MIX
of vehicle types, not a uniform draw over every available name regardless of size. Every
downloaded mesh degrades gracefully (see `e2e.ml.assets`/`rt_gen`) to
`SIONNA_CAR_REPRESENTATIVE` on a machine without the asset cache -- this module's own
determinism/pool-composition tests hold either way.

  RESOLVED (D4): a Sionna built-in city scene IS usable at 77 GHz -- see
  `e2e.environment.city_scenes`. Kept for the record, the original obstacle was:
  measured empirically, `build_rt_scene`'s `scene.frequency = f0 + B/2` at an automotive
  77 GHz radar preset (`ti_iwr1443`/`radial_like`, both ~77-78 GHz) raises
  `"Properties of ITU material 'marble' are not defined for this frequency"`: munich's
  (and etoile's) baked geometry uses ITU `marble`, whose ITU-R P.2040-3 curve fit is only
  valid 1-60 GHz (`sionna/rt/radio_materials/itu.py`'s `ITU_MATERIALS` table). This is a
  hard incompatibility between these two built-in scenes and any 77 GHz automotive-radar
  config, not something `e2e` code can patch (the materials are baked into the scene
  XML/mesh assignment, not passed through `e2e.scenario`). A real fix needs either a
  77 GHz-valid built-in scene, a re-authored copy of munich/etoile with a marble-free
  material set, or a non-automotive (<=60 GHz) `RadarConfig` preset for D3 specifically.

Determinism: `build_rt_tier_scenario(tier, frame_idx, seed)` draws every random value
from a `numpy.random.Generator` seeded ONLY by `(tier, frame_idx, seed)` (via a stable
SHA-256-derived seed -- NOT Python's salted `hash()`, which is not process-stable), so
the same triple reproduces the identical `Scenario` (byte-for-byte via `to_json`)
regardless of call order or process. `frame_idx` is a sample index here (matching
`e2e.ml.scenes.sample_scene`'s per-draw usage), not a motion-track frame -- the returned
Scenario's OWN `num_frames`/`Motion` fields carry any per-frame motion.

Only numpy + stdlib + `e2e.scenario` / `e2e.ml.rt_gen`'s asset-name constants are
imported here (no torch, no Sionna) so building a tier scenario needs neither GPU nor a
ray tracer -- only *generating frames from it* does.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from e2e.ml.assets import (DOWNLOADED_BUS_ASSET_NAMES, DOWNLOADED_CAR_ASSET_NAMES,
                          DOWNLOADED_TROLLEY_ASSET_NAMES, DOWNLOADED_TRUCK_ASSET_NAMES)
from e2e.ml.rt_gen import (LOCAL_PEDESTRIAN_ASSET_NAMES, LOCAL_VEHICLE_ASSET_NAMES,
                          PEDESTRIAN_ASSET_NAME, SIONNA_CAR_REPRESENTATIVE,
                          object_local_height_m)
from e2e.scenario import Motion, Node, NodeRole, ObjectKind, Scenario, SceneObject

# Placement envelope: targets are scattered in the radar's forward FOV, matching
# e2e.ml.scenes._sample_fov_position's spirit (kept off the extreme edges). Positions
# are computed in a radar-centred, +x-boresight LOCAL frame and then translated (not
# rotated -- boresight is always +x here) by the radar's world position; see
# `build_rt_tier_scenario`.
# Widened from (6, 25) once the separation check exposed the envelope as
# over-subscribed: D3 asks for ~19 objects, and the old span simply did not contain
# that many non-overlapping footprints. 34 m stays inside the ti_iwr1443 preset's
# 38.4 m unambiguous range, so nothing wraps.
_RANGE_M = (6.0, 34.0)
_SIN_AZ_RANGE = (-0.6, 0.6)

# Sionna's bundled box mesh (`rt_gen._box_mesh_path`) is a 10x10x5 m primitive; these
# scalings give ~3-6 m clutter boxes (see the clutter-box loop in
# `build_rt_tier_scenario`).
_CLUTTER_BOX_SCALING = (0.3, 0.6)

# Clutter-box line-of-sight avoidance (fix for the occlusion bug: a box sampled from the
# SAME range/angle envelope as the tier's own vehicle/pedestrian targets could -- and, in
# an earlier draw, did -- sit directly between the radar and a target it shared a scene
# with, hiding it from a monostatic return entirely). `_sample_clutter_offset` rejects a
# candidate whose bearing is within this angular half-width of an already-placed target
# AND whose range doesn't clear that target by the margin below; `_CLUTTER_LOS_MAX_TRIES`
# bounds the resample loop (falls back to the FOV edge -- see that function).
_CLUTTER_LOS_MARGIN_SIN_AZ = 0.06
_CLUTTER_LOS_RANGE_MARGIN_M = 1.5
_CLUTTER_LOS_MAX_TRIES = 12

# --------------------------------------------------------------------------------
# Vehicle mesh pool: ONE representative Sionna car (see rt_gen.SIONNA_CAR_REPRESENTATIVE)
# plus every downloaded mesh of that class -- replaces the old "uniform draw over 17
# duplicate-geometry names" (see module docstring). A class with no downloaded mesh (none
# today) would be an empty tuple; `_draw_vehicle_asset` never lets a class pool go empty
# in practice since every non-"car" class here has >= 1 real mesh.
# --------------------------------------------------------------------------------
VEHICLE_CLASS_POOLS: Dict[str, Tuple[str, ...]] = {
    "car": (SIONNA_CAR_REPRESENTATIVE,) + DOWNLOADED_CAR_ASSET_NAMES,
    "truck": DOWNLOADED_TRUCK_ASSET_NAMES,
    "bus": DOWNLOADED_BUS_ASSET_NAMES,
    "trolley": DOWNLOADED_TROLLEY_ASSET_NAMES,
}

# Realistic road mix for a tier's vehicle draws: mostly cars, with a believable sprinkle
# of larger vehicle types -- NOT a uniform draw across classes (a scene that's 25% buses
# would be as unrealistic as the old 85%-duplicate-car problem this replaces).
VEHICLE_CLASS_WEIGHTS: Dict[str, float] = {"car": 0.70, "truck": 0.12, "bus": 0.10, "trolley": 0.08}
_VEHICLE_CLASSES = tuple(VEHICLE_CLASS_WEIGHTS)
_VEHICLE_CLASS_CUM = np.cumsum([VEHICLE_CLASS_WEIGHTS[c] for c in _VEHICLE_CLASSES])
_VEHICLE_CLASS_CUM = _VEHICLE_CLASS_CUM / _VEHICLE_CLASS_CUM[-1]

# `local_tractor_trailer` (see rt_gen.LOCAL_ASSET_SPECS) is a real semi -- when
# use_local_assets=True it augments the "truck" pool, not "car" like the other two
# local vehicle names (mustang/dodge charger).
_LOCAL_TRUCK_ASSET_NAMES = tuple(n for n in LOCAL_VEHICLE_ASSET_NAMES if "trailer" in n)
_LOCAL_CAR_ASSET_NAMES = tuple(n for n in LOCAL_VEHICLE_ASSET_NAMES if n not in _LOCAL_TRUCK_ASSET_NAMES)


def _draw_vehicle_class(rng: np.random.Generator) -> str:
    r = float(rng.random())
    i = int(np.searchsorted(_VEHICLE_CLASS_CUM, r, side="right"))
    return _VEHICLE_CLASSES[min(i, len(_VEHICLE_CLASSES) - 1)]


def _draw_vehicle_asset(rng: np.random.Generator, use_local_assets: bool) -> str:
    """One asset name, drawn class-weighted (`VEHICLE_CLASS_WEIGHTS`) then
    uniformly within that class's pool (`VEHICLE_CLASS_POOLS`, expanded with the
    matching local assets when `use_local_assets`)."""
    vclass = _draw_vehicle_class(rng)
    pool = VEHICLE_CLASS_POOLS[vclass]
    if use_local_assets:
        if vclass == "car":
            pool = pool + _LOCAL_CAR_ASSET_NAMES
        elif vclass == "truck":
            pool = pool + _LOCAL_TRUCK_ASSET_NAMES
    return pool[int(rng.integers(0, len(pool)))]

# Default (world position, boresight +x via a +x-offset look_at) per base scene: a
# synthetic "flat"/"free" ground plane is uniform, so the origin is as good a spot as
# any. No shipped tier currently uses a Sionna built-in city scene as its base (see the
# module docstring's munich/etoile follow-up note), but a caller building one directly
# (or a future tier) gets a sane default rather than risking the origin landing inside
# a building -- these positions match `e2e.scenario.munich_radar_scenario` /
# `etoile_radar_scenario`, both already validated against the real Sionna path.
_DEFAULT_RADAR_POSITION = (0.0, 0.0, 1.5)
_BASE_SCENE_RADAR_POSITION: Dict[str, Tuple[float, float, float]] = {
    "munich": (45.0, 90.0, 1.5),
    "etoile": (60.0, 0.0, 1.5),
}


@dataclass(frozen=True)
class RTTierSpec:
    """One RT difficulty tier: object mix (count ranges, inclusive) + base scene."""
    name: str
    base_scene: str
    n_spheres: Tuple[int, int]
    n_cars: Tuple[int, int]
    n_pedestrians: Tuple[int, int]
    n_clutter_boxes: Tuple[int, int]
    speed_mps: Tuple[float, float]         # target speed magnitude range
    description: str


RT_DIFFICULTY_TIERS: Dict[str, RTTierSpec] = {
    "D0": RTTierSpec(
        name="D0", base_scene="flat",
        n_spheres=(1, 1), n_cars=(0, 0), n_pedestrians=(0, 0), n_clutter_boxes=(0, 0),
        speed_mps=(0.0, 3.0),
        description="Single metal sphere scatterer on a flat ground plane -- "
                    "specular-visibility sanity tier.",
    ),
    "D1": RTTierSpec(
        name="D1", base_scene="flat",
        n_spheres=(0, 0), n_cars=(1, 3), n_pedestrians=(0, 0), n_clutter_boxes=(0, 0),
        speed_mps=(0.0, 8.0),
        description="A few real (Sionna-bundled) car meshes on a flat ground plane.",
    ),
    "D2": RTTierSpec(
        name="D2", base_scene="flat",
        n_spheres=(0, 2), n_cars=(2, 4), n_pedestrians=(1, 3), n_clutter_boxes=(2, 4),
        speed_mps=(0.0, 8.0),
        description="Cars + pedestrians + static box clutter on a flat ground plane "
                    "-- a richer synthetic environment than D0/D1.",
    ),
    "D3": RTTierSpec(
        name="D3", base_scene="flat",
        n_spheres=(0, 0), n_cars=(3, 6), n_pedestrians=(2, 5), n_clutter_boxes=(4, 8),
        speed_mps=(0.0, 8.0),
        description="Cars + pedestrians + dense static box clutter on a flat ground "
                    "plane -- the richest FLAT-GROUND tier. The city lives in D4.",
    ),
    "D4": RTTierSpec(
        name="D4", base_scene="munich",
        n_spheres=(0, 0), n_cars=(2, 5), n_pedestrians=(2, 4), n_clutter_boxes=(0, 0),
        speed_mps=(0.0, 8.0),
        description="Cars + pedestrians in a REAL ray-traced city (Sionna's munich "
                    "scene) -- the environment axis of the difficulty scale, which "
                    "flat-ground tiers cannot express. Buildings supply their own "
                    "multipath and clutter, so no synthetic clutter boxes are added: "
                    "the city IS the clutter. Loading munich at automotive 77 GHz "
                    "needs out-of-band ITU materials substituted and removed -- see "
                    "e2e.environment.city_scenes, which reports what it swapped so a "
                    "corpus can state what its city was made of.",
    ),
}


def _resolve_tier(tier: Union[str, RTTierSpec]) -> RTTierSpec:
    if isinstance(tier, RTTierSpec):
        return tier
    return RT_DIFFICULTY_TIERS[tier]  # raises KeyError on an unknown tier name


def _stable_seed(tier: str, frame_idx: int, seed: int) -> int:
    """A `numpy.random.Generator` seed derived deterministically from `(tier,
    frame_idx, seed)`, stable across processes/platforms (unlike Python's salted
    `hash()`, which must NOT be used here -- it is randomized per interpreter run
    unless `PYTHONHASHSEED` is fixed)."""
    digest = hashlib.sha256(f"{tier}:{int(frame_idx)}:{int(seed)}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


#: Ground-plane footprint as (length, width) in metres -- a BOX, not a circle. Circles
#: were the first attempt and they let corners overlap: a 3.2 m radius inscribed in a
#: 6x6 m clutter box leaves the corners exposed, so two boxes could clear the distance
#: test and still intersect. Measured after the fix: box overlap fell from 0.57% of
#: pairs to zero.
_FOOTPRINT_M = {
    "pedestrian": (0.8, 0.6),
    "sphere": (1.0, 1.0),
    "scatterer": (6.0, 6.0),      # Sionna's box primitive at our clutter scaling
}
#: Vehicles vary threefold in length, so they are keyed by class rather than lumped.
VEHICLE_FOOTPRINT_M: Dict[str, Tuple[float, float]] = {
    "car": (4.4, 2.0), "truck": (15.7, 2.9), "bus": (8.0, 2.9), "trolley": (11.9, 3.5),
}
_DEFAULT_FOOTPRINT = (4.4, 2.0)
#: Gap left between footprints. Vehicles in traffic sit closer than this, but the point
#: is to guarantee no INTERSECTION, not to model a car park.
_SEPARATION_MARGIN_M = 0.5
_SEPARATION_MAX_TRIES = 60


def _asset_vehicle_class(asset: Optional[str]) -> str:
    """Which class pool an asset name came from ("car" if unknown/None)."""
    if asset:
        for cls, pool in VEHICLE_CLASS_POOLS.items():
            if asset in pool:
                return cls
    return "car"


def _footprint(object_class: str, asset: Optional[str] = None) -> Tuple[float, float]:
    """(length, width) in metres for an object about to be placed."""
    if object_class in _FOOTPRINT_M:
        return _FOOTPRINT_M[object_class]
    if object_class == "vehicle":
        return VEHICLE_FOOTPRINT_M.get(_asset_vehicle_class(asset), _DEFAULT_FOOTPRINT)
    return _DEFAULT_FOOTPRINT


def _footprint_radius(object_class: str, asset: Optional[str] = None) -> float:
    """Half-length, kept for the minimum-range rule and for callers that want a scalar."""
    return _footprint(object_class, asset)[0] / 2.0


def _boxes_clear(ax, ay, a_ext, bx, by, b_ext, margin=_SEPARATION_MARGIN_M) -> bool:
    """True when two axis-aligned footprints do not intersect (with margin).

    Axis-aligned rather than oriented: headings are random, so an AABB sized to the
    object's full length in both axes would be pessimistic, while this is the simple,
    checkable thing the scene actually needs -- objects must not be inside one another.
    """
    return (abs(ax - bx) >= (a_ext[0] + b_ext[0]) / 2.0 + margin
            or abs(ay - by) >= (a_ext[1] + b_ext[1]) / 2.0 + margin)


def _sample_local_offset(rng: np.random.Generator,
                         placed: Sequence[Tuple[float, float, Tuple[float, float]]] = (),
                         object_class: str = "vehicle",
                         asset: Optional[str] = None) -> Tuple[float, float]:
    """`(dx, dy)` in the radar-centred, +x-boresight local frame (ground-level, z=0),
    kept clear of everything already placed.

    Without the separation check objects interpenetrate: a sweep of 40 frames each of
    D2 and D3 put 73 of 5284 object pairs inside one another, the worst being a
    pedestrian 0.12 m from a vehicle -- i.e. standing inside a car. That is not a
    cosmetic problem. Two targets occupying one patch of ground produce a radar return
    no real scene could produce AND two ground-truth labels at the same range and
    bearing, so a model is trained to predict something incoherent.

    Candidates are resampled until they clear every placed object by the sum of the two
    footprint radii. If none does within `_SEPARATION_MAX_TRIES`, this returns **None**
    and the caller SKIPS that object rather than placing it overlapping. That is the
    honest resolution: a tier's object counts are a request, and a scene that cannot
    hold them holds fewer. Taking the least-bad candidate instead -- the earlier
    behaviour -- put vehicles up to 2 m inside one another once footprints became
    length-aware, which is exactly the corruption this check exists to prevent. Labels
    come from the scenario, so a skipped object is simply absent from both the geometry
    and the ground truth; they never disagree.
    """
    own = _footprint(object_class, asset)

    def _draw():
        # Minimum range grows with the object's own half-length so a long vehicle
        # cannot extend backwards through the radar: a 16 m trailer centred at 6 m
        # reaches -2 m, i.e. behind the antenna.
        r = rng.uniform(_RANGE_M[0] + own[0] / 2.0, _RANGE_M[1])
        sin_az = rng.uniform(_SIN_AZ_RANGE[0], _SIN_AZ_RANGE[1])
        cos_az = math.sqrt(max(0.0, 1.0 - sin_az ** 2))
        return (float(r * cos_az), float(r * sin_az))

    if not placed:
        return _draw()

    for _ in range(_SEPARATION_MAX_TRIES):
        dx, dy = _draw()
        if all(_boxes_clear(dx, dy, own, px, py, pext) for px, py, pext in placed):
            return dx, dy
    return None


def _sample_velocity(rng: np.random.Generator, speed_range: Tuple[float, float]) -> Tuple[float, float, float]:
    """In-plane velocity, magnitude in `speed_range`, random heading.

    Numerically m/s == m/frame under this package's `dt = 1` convention (see
    `e2e.ml.scatterers.DEFAULT_DT_S`); consumers with a real `frame_rate_hz` should
    rescale if they need physical timing (this module has no `RadarConfig` to derive
    one from -- callers building a Scenario for `e2e.ml.rt_gen` supply their own `cfg`).
    """
    speed = float(rng.uniform(speed_range[0], speed_range[1]))
    theta = float(rng.uniform(0.0, 2.0 * math.pi))
    return (speed * math.cos(theta), speed * math.sin(theta), 0.0)


def _n_in(rng: np.random.Generator, lo_hi: Tuple[int, int]) -> int:
    lo, hi = lo_hi
    return int(rng.integers(lo, hi + 1))


def _sample_clutter_offset(rng: np.random.Generator,
                           target_offsets: Sequence[Tuple[float, float]],
                           placed: Sequence[Tuple[float, float, Tuple[float, float]]] = ()) -> Tuple[float, float]:
    """`(dx, dy)` for a clutter box, avoiding the direct radar-to-target line of sight
    (item 5 fix -- see `_CLUTTER_LOS_*` above): resamples up to `_CLUTTER_LOS_MAX_TRIES`
    times, rejecting any candidate whose bearing is within `_CLUTTER_LOS_MARGIN_SIN_AZ`
    of an already-placed target's bearing AND whose range doesn't clear that target's
    range by `_CLUTTER_LOS_RANGE_MARGIN_M` (i.e. would sit at or in front of it, blocking
    it). Falls back to a position pinned to the FOV edge -- well clear of the
    `_SIN_AZ_RANGE` span targets are drawn from -- rather than looping unboundedly if a
    scene is too crowded for a clear spot."""
    def _polar(dx: float, dy: float) -> Tuple[float, float]:
        r = math.hypot(dx, dy)
        return r, (dy / r if r > 1e-6 else 0.0)

    target_polar = [_polar(tdx, tdy) for tdx, tdy in target_offsets]
    for _ in range(_CLUTTER_LOS_MAX_TRIES):
        spot = _sample_local_offset(rng, placed, "scatterer")
        if spot is None:
            return None
        dx, dy = spot
        r, sin_az = _polar(dx, dy)
        blocked = any(
            abs(sin_az - t_sin_az) < _CLUTTER_LOS_MARGIN_SIN_AZ
            and r < t_r + _CLUTTER_LOS_RANGE_MARGIN_M
            for t_r, t_sin_az in target_polar
        )
        if not blocked:
            return dx, dy

    # Fallback: the scene is too crowded for a spot that clears every line of sight.
    # Pin to the FOV edge, but still pick the candidate that sits FURTHEST from anything
    # already placed -- an earlier version returned the first edge draw unconditionally,
    # which is where the surviving object overlaps came from.
    own = _footprint("scatterer")
    for _ in range(_SEPARATION_MAX_TRIES):
        r = float(rng.uniform(_RANGE_M[0], _RANGE_M[1]))
        sin_az = math.copysign(0.9, float(rng.uniform(-1.0, 1.0)))
        cos_az = math.sqrt(max(0.0, 1.0 - sin_az ** 2))
        dx, dy = r * cos_az, r * sin_az
        if not placed or all(_boxes_clear(dx, dy, own, px, py, pext)
                             for px, py, pext in placed):
            return dx, dy
    return None


def build_rt_tier_scenario(tier: Union[str, RTTierSpec], *, frame_idx: int = 0, seed: int = 0,
                           num_frames: int = 2,
                           radar_position: Optional[Tuple[float, float, float]] = None,
                           use_local_assets: bool = False) -> Scenario:
    """Draw a deterministic RT `Scenario` at difficulty `tier`.

    `frame_idx`/`seed` together select the draw (see the module docstring's
    "Determinism" section) -- calling this twice with the same `(tier, frame_idx,
    seed)` returns byte-for-byte identical scenarios (`Scenario.to_json()` equal);
    changing either changes the draw. `num_frames` sets the returned Scenario's own
    frame count (each moving object gets a constant-velocity `Motion` track spanning
    it); it is NOT part of the determinism key -- reusing a `(tier, frame_idx, seed)`
    with a different `num_frames` keeps the same object mix/positions/velocities and
    only changes how many motion steps are resolved from them.

    `radar_position` defaults to `_BASE_SCENE_RADAR_POSITION[spec.base_scene]` when the
    tier's base scene has one (currently "munich"), else the origin (`(0, 0, 1.5)`) --
    see that dict's docstring for why a built-in city scene needs a scene-specific
    default. The radar's boresight is always +x (`look_at` = position + (1, 0, 0)) and
    every object's (x, y) position is that local (`dx`, `dy`) offset translated by
    `radar_position`'s (x, y) -- NOT rotated, since boresight never varies. Every
    object's z is `0.5 * object_local_height_m(...) * scaling` (see that function's
    docstring): Sionna's `SceneObject.position` setter re-centers the mesh's own AABB
    on the given point, so this is the centre height that makes the bbox rest ON the
    z=0 ground plane rather than sink half its height below it.

    Vehicles are drawn class-weighted (`VEHICLE_CLASS_WEIGHTS`) then uniformly within
    that class's `VEHICLE_CLASS_POOLS` entry -- see the module docstring's "Vehicle
    variety" section; this is the default (not gated behind `use_local_assets`), since
    the duplicate-Sionna-geometry problem it fixes existed by default.

    `use_local_assets`: when True, vehicles/pedestrians ADDITIONALLY draw from
    `e2e.ml.rt_gen.LOCAL_VEHICLE_ASSET_NAMES` / `LOCAL_PEDESTRIAN_ASSET_NAMES` --
    higher-fidelity meshes that live only on this workstation (see that module's
    docstring), layered on top of the class pools above (`local_tractor_trailer` joins
    "truck", the rest join "car"); on any other machine those names degrade gracefully
    to a Sionna-bundled mesh at `build_rt_scene` time. Defaults to False.

    Raises `KeyError` for an unknown tier name (see `RT_DIFFICULTY_TIERS`).
    """
    spec = _resolve_tier(tier)
    rng = np.random.default_rng(_stable_seed(spec.name, frame_idx, seed))

    if radar_position is None:
        radar_position = _BASE_SCENE_RADAR_POSITION.get(spec.base_scene, _DEFAULT_RADAR_POSITION)
    rx0, ry0, rz0 = radar_position
    radar_look_at = (rx0 + 1.0, ry0, rz0)

    def _ground_pos(dx: float, dy: float, kind, asset: Optional[str], scaling: float) -> Tuple[float, float, float]:
        z = 0.5 * object_local_height_m(kind, asset) * float(scaling)
        return (rx0 + dx, ry0 + dy, z)

    pedestrian_pool = ((PEDESTRIAN_ASSET_NAME,) + LOCAL_PEDESTRIAN_ASSET_NAMES
                       if use_local_assets else (PEDESTRIAN_ASSET_NAME,))

    objects = []
    # (dx, dy) of every vehicle/pedestrian/sphere target placed so far, in the same
    # radar-centred local frame -- feeds `_sample_clutter_offset`'s line-of-sight check.
    target_offsets: List[Tuple[float, float]] = []
    # (dx, dy, footprint_radius) of EVERYTHING placed so far, including clutter, so no
    # two objects end up occupying the same patch of ground -- see _sample_local_offset.
    placed: List[Tuple[float, float, Tuple[float, float]]] = []

    def _motion(vel):
        return Motion(velocity=vel) if num_frames > 1 else Motion()

    for i in range(_n_in(rng, spec.n_spheres)):
        spot = _sample_local_offset(rng, placed, "sphere")
        if spot is None:
            continue          # scene is full; see _sample_local_offset
        dx, dy = spot
        target_offsets.append((dx, dy))
        placed.append((dx, dy, _footprint("sphere")))
        vel = _sample_velocity(rng, spec.speed_mps)
        pos = _ground_pos(dx, dy, ObjectKind.SPHERE, None, 0.5)
        objects.append(SceneObject(
            name=f"sphere-{i}", kind=ObjectKind.SPHERE, position=pos, scaling=0.5,
            material="metal", object_class="vehicle", motion=_motion(vel),
        ))

    for i in range(_n_in(rng, spec.n_cars)):
        asset = _draw_vehicle_asset(rng, use_local_assets)
        spot = _sample_local_offset(rng, placed, "vehicle", asset)
        if spot is None:
            continue          # scene is full; see _sample_local_offset
        dx, dy = spot
        target_offsets.append((dx, dy))
        placed.append((dx, dy, _footprint("vehicle", asset)))
        vel = _sample_velocity(rng, spec.speed_mps)
        pos = _ground_pos(dx, dy, ObjectKind.MESH, asset, 1.0)
        objects.append(SceneObject(
            name=f"vehicle-{i}", kind=ObjectKind.MESH, asset=asset, position=pos,
            scaling=1.0, material="metal", object_class="vehicle", motion=_motion(vel),
        ))

    for i in range(_n_in(rng, spec.n_pedestrians)):
        spot = _sample_local_offset(rng, placed, "pedestrian")
        if spot is None:
            continue          # scene is full; see _sample_local_offset
        dx, dy = spot
        target_offsets.append((dx, dy))
        placed.append((dx, dy, _footprint("pedestrian")))
        # pedestrians are slower than vehicles regardless of the tier's vehicle range.
        vel = _sample_velocity(rng, (0.0, min(2.0, spec.speed_mps[1])))
        asset = pedestrian_pool[int(rng.integers(0, len(pedestrian_pool)))]
        pos = _ground_pos(dx, dy, ObjectKind.MESH, asset, 1.0)
        objects.append(SceneObject(
            name=f"pedestrian-{i}", kind=ObjectKind.MESH, asset=asset,
            position=pos, scaling=1.0, material="skin", object_class="pedestrian",
            motion=_motion(vel),
        ))

    for i in range(_n_in(rng, spec.n_clutter_boxes)):
        # Kept clear of the direct radar-target line of sight (item 5 fix) -- see
        # `_sample_clutter_offset` / `_CLUTTER_LOS_*` above.
        spot = _sample_clutter_offset(rng, target_offsets, placed)
        if spot is None:
            continue          # scene is full; see _sample_local_offset
        dx, dy = spot
        placed.append((dx, dy, _footprint("scatterer")))
        # Sionna's bundled box mesh is a 10x10x5 m primitive (measured from its own
        # bbox) -- scaling by _CLUTTER_BOX_SCALING gives ~3-6 m "parked container"
        # sized clutter, not (unscaled) building-sized blocks.
        scaling = float(rng.uniform(*_CLUTTER_BOX_SCALING))
        pos = _ground_pos(dx, dy, ObjectKind.BOX, None, scaling)
        objects.append(SceneObject(
            name=f"clutter-box-{i}", kind=ObjectKind.BOX, position=pos,
            scaling=scaling, material="concrete",
            object_class="scatterer", motion=Motion(),
        ))

    return Scenario(
        name=f"rt_{spec.name}_{frame_idx}_{seed}",
        base_scene=spec.base_scene,
        num_frames=max(1, int(num_frames)),
        nodes=[Node(name="radar", role=NodeRole.RADAR, position=radar_position,
                    look_at=radar_look_at)],
        objects=objects,
        description=spec.description,
        metadata={"tier": spec.name, "frame_idx": int(frame_idx), "seed": int(seed)},
    )


def vehicle_asset_class(asset: Optional[str]) -> str:
    """Which `VEHICLE_CLASS_POOLS` key `asset` belongs to ("car" for anything not
    otherwise recognized, including the local mustang/dodge-charger names and any raw
    `CAR_ASSET_NAMES` entry -- see `_LOCAL_TRUCK_ASSET_NAMES`'s "trailer" special case)."""
    if asset in _LOCAL_TRUCK_ASSET_NAMES:
        return "truck"
    for vclass, pool in VEHICLE_CLASS_POOLS.items():
        if asset in pool:
            return vclass
    return "car"


def tier_summary(scenario: Scenario) -> dict:
    """Compact per-class object counts for manifests/debugging (mirrors
    `e2e.ml.scenes.scene_summary`'s spirit, RT-specific fields). `n_cars` is the total
    vehicle-mesh count (kept for back-compat with earlier callers/captions); each
    vehicle's actual class (car/truck/bus/trolley) is broken out in
    `n_vehicles_by_class`."""
    vehicles = [o for o in scenario.objects if o.kind == ObjectKind.MESH and o.object_class == "vehicle"]
    by_class: Dict[str, int] = {"car": 0, "truck": 0, "bus": 0, "trolley": 0}
    for o in vehicles:
        by_class[vehicle_asset_class(o.asset)] += 1
    return {
        "tier": scenario.metadata.get("tier"),
        "base_scene": scenario.base_scene,
        "n_spheres": sum(1 for o in scenario.objects if o.kind == ObjectKind.SPHERE),
        "n_cars": len(vehicles),
        "n_vehicles_by_class": by_class,
        "n_pedestrians": sum(1 for o in scenario.objects if o.object_class == "pedestrian"),
        "n_clutter_boxes": sum(1 for o in scenario.objects if o.kind == ObjectKind.BOX),
    }
