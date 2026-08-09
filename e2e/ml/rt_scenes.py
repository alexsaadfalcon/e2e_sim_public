"""
Ray-traced (Sionna RT) difficulty tiers D0-D3 for the radar-ML data campaign.

This is the RT sibling of `e2e.ml.scenes` (which samples scenes for the analytic
point-target synthesizer, `rd_synth`): it draws a `e2e.scenario.Scenario` whose objects
are REAL meshes -- Sionna's bundled car models and the procedural pedestrian
placeholder (see `e2e.ml.rt_gen.CAR_ASSET_NAMES`/`PEDESTRIAN_ASSET_NAME`) -- instead of
sphere-as-target scatterers, for consumption by `e2e.ml.rt_gen.build_rt_scene` /
`rt_synthesize_adc` or `e2e.environment.scenario_runner`.

Difficulty ramps along TWO axes, not just target count:

* **scatterer type**: D0 is bare metal spheres (a specular-visibility sanity check --
  see `rt_gen`'s module docstring on why a curved target needs diffuse scattering to be
  visible at all), D1 promotes to real car meshes, D2/D3 add the pedestrian placeholder.
* **environment**: D0/D1 are a bare flat ground plane (`base_scene="flat"`, see
  `rt_gen._FLAT_SCENE_XML`); D2/D3 both add static box "clutter" (parked-container-style
  obstacles) to a flat ground, D3 denser than D2 -- a richer-but-still-cheap scene.

  FOLLOW-UP, NOT DONE HERE: the task brief's preferred D3 base was a Sionna built-in
  city scene (`"munich"`/`"etoile"`), which WAS tried and does not "load cheaply" --
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
from typing import Dict, Optional, Tuple, Union

import numpy as np

from e2e.ml.rt_gen import (CAR_ASSET_NAMES, LOCAL_PEDESTRIAN_ASSET_NAMES,
                          LOCAL_VEHICLE_ASSET_NAMES, PEDESTRIAN_ASSET_NAME,
                          object_local_height_m)
from e2e.scenario import Motion, Node, NodeRole, ObjectKind, Scenario, SceneObject

# Placement envelope: targets are scattered in the radar's forward FOV, matching
# e2e.ml.scenes._sample_fov_position's spirit (kept off the extreme edges). Positions
# are computed in a radar-centred, +x-boresight LOCAL frame and then translated (not
# rotated -- boresight is always +x here) by the radar's world position; see
# `build_rt_tier_scenario`.
_RANGE_M = (6.0, 25.0)
_SIN_AZ_RANGE = (-0.6, 0.6)

# Sionna's bundled box mesh (`rt_gen._box_mesh_path`) is a 10x10x5 m primitive; these
# scalings give ~3-6 m clutter boxes (see the clutter-box loop in
# `build_rt_tier_scenario`).
_CLUTTER_BOX_SCALING = (0.3, 0.6)

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
                    "plane -- richest tier. FOLLOW-UP: intended to use a Sionna "
                    "built-in city scene (munich/etoile) instead; see the module "
                    "docstring for why that base scene fails outright at automotive "
                    "77 GHz (an ITU-material frequency-validity limit, not something "
                    "e2e code can patch).",
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


def _sample_local_offset(rng: np.random.Generator) -> Tuple[float, float]:
    """`(dx, dy)` in the radar-centred, +x-boresight local frame (ground-level, z=0)."""
    r = rng.uniform(_RANGE_M[0], _RANGE_M[1])
    sin_az = rng.uniform(_SIN_AZ_RANGE[0], _SIN_AZ_RANGE[1])
    cos_az = math.sqrt(max(0.0, 1.0 - sin_az ** 2))
    return (float(r * cos_az), float(r * sin_az))


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

    `use_local_assets`: when True, cars/pedestrians are drawn from an EXPANDED pool
    that also includes `e2e.ml.rt_gen.LOCAL_VEHICLE_ASSET_NAMES` /
    `LOCAL_PEDESTRIAN_ASSET_NAMES` -- higher-fidelity meshes that live only on this
    workstation (see that module's docstring); on any other machine those names
    degrade gracefully to a Sionna-bundled mesh at `build_rt_scene` time. Defaults to
    False so every existing caller/test (including this module's own determinism
    tests) keeps drawing from exactly `CAR_ASSET_NAMES`/`PEDESTRIAN_ASSET_NAME`, byte-
    for-byte unchanged.

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

    car_pool = CAR_ASSET_NAMES + LOCAL_VEHICLE_ASSET_NAMES if use_local_assets else CAR_ASSET_NAMES
    pedestrian_pool = ((PEDESTRIAN_ASSET_NAME,) + LOCAL_PEDESTRIAN_ASSET_NAMES
                       if use_local_assets else (PEDESTRIAN_ASSET_NAME,))

    objects = []

    def _motion(vel):
        return Motion(velocity=vel) if num_frames > 1 else Motion()

    for i in range(_n_in(rng, spec.n_spheres)):
        dx, dy = _sample_local_offset(rng)
        vel = _sample_velocity(rng, spec.speed_mps)
        pos = _ground_pos(dx, dy, ObjectKind.SPHERE, None, 0.5)
        objects.append(SceneObject(
            name=f"sphere-{i}", kind=ObjectKind.SPHERE, position=pos, scaling=0.5,
            material="metal", object_class="vehicle", motion=_motion(vel),
        ))

    for i in range(_n_in(rng, spec.n_cars)):
        dx, dy = _sample_local_offset(rng)
        vel = _sample_velocity(rng, spec.speed_mps)
        asset = car_pool[int(rng.integers(0, len(car_pool)))]
        pos = _ground_pos(dx, dy, ObjectKind.MESH, asset, 1.0)
        objects.append(SceneObject(
            name=f"car-{i}", kind=ObjectKind.MESH, asset=asset, position=pos,
            scaling=1.0, material="metal", object_class="vehicle", motion=_motion(vel),
        ))

    for i in range(_n_in(rng, spec.n_pedestrians)):
        dx, dy = _sample_local_offset(rng)
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
        dx, dy = _sample_local_offset(rng)
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


def tier_summary(scenario: Scenario) -> dict:
    """Compact per-class object counts for manifests/debugging (mirrors
    `e2e.ml.scenes.scene_summary`'s spirit, RT-specific fields)."""
    return {
        "tier": scenario.metadata.get("tier"),
        "base_scene": scenario.base_scene,
        "n_spheres": sum(1 for o in scenario.objects if o.kind == ObjectKind.SPHERE),
        "n_cars": sum(1 for o in scenario.objects
                      if o.kind == ObjectKind.MESH and o.object_class == "vehicle"),
        "n_pedestrians": sum(1 for o in scenario.objects if o.object_class == "pedestrian"),
        "n_clutter_boxes": sum(1 for o in scenario.objects if o.kind == ObjectKind.BOX),
    }
