"""
Random radar scene sampler for the ML radar-dataset package.

Draws a `e2e.scenario.Scenario` -- a radar node plus a mix of vehicle/pedestrian
targets and static low-RCS clutter -- from one of a handful of "difficulty tiers"
(`DIFFICULTY_TIERS`), ranging from a single slow vehicle (D0, a sanity check) up to a
dense, fast, tightly-packed multi-target scene (D3). `dataset.py` calls `sample_scene`
per training example; `scatterers.py`/`rd_synth.py` turn the resulting `Scenario` into
point scatterers and, from there, a range-Doppler cube.

`sample_scene(..., n_frames=1)` defaults to a single-instant scene (the original
behavior); `n_frames>1` gives moving (vehicle/pedestrian) objects a real
constant-velocity `Motion` track spanning `Scenario.num_frames == n_frames`, so
`dataset.py` can synthesize a SEQUENCE of consecutive, motion-consistent frames from
one scene (see `sample_scene`'s docstring). Clutter is always static.

Only numpy + stdlib + `e2e.scenario` / `e2e.ml.scatterers` are imported here (no
torch), matching the rest of this package's torch-free scene-description layer.

Determinism is the *caller's* contract: `sample_scene` draws every random value from
the numpy `Generator` it is given and nothing else (no hidden global RNG, no
wall-clock/uuid tags), so seeding that generator identically reproduces an identical
`Scenario` byte-for-byte (see `Scenario.to_json`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np

from e2e.ml.scatterers import DEFAULT_RCS_DBSM, pedestrian, vehicle
from e2e.scenario import Motion, Node, NodeRole, Scenario, SceneObject

# Fraction of the radar's unambiguous range/angle FOV that targets are placed within
# (kept off the extreme edges, where FFT-based downstream processing is least
# reliable -- range sidelobes near the max-range wraparound, angle grating-lobe
# ambiguity near endfire).
_RANGE_FRAC = (0.1, 0.85)
_SIN_AZ_RANGE = (-0.85, 0.85)

# Fraction of cfg.max_velocity_mps a target's radial velocity is clamped to, so
# training scenes never alias in Doppler (the synthesizer models the target's true
# continuous velocity, not a wrapped one -- see rd_synth.synthesize_adc).
_RADIAL_VELOCITY_FRAC = 0.8

# Clutter RCS range (dBsm): well below the vehicle/pedestrian defaults in
# `DEFAULT_RCS_DBSM`, modeling low-reflectivity static background (poles, curbs,
# foliage) rather than a labeled target class.
_CLUTTER_RCS_DBSM_RANGE = (-20.0, -8.0)

# Cap on reject-sampling attempts per target before giving up on separation.
_MAX_PLACEMENT_ATTEMPTS = 500


@dataclass(frozen=True)
class TierSpec:
    """One difficulty tier of the scene distribution."""
    name: str
    n_vehicles: Tuple[int, int]        # inclusive (min, max)
    n_pedestrians: Tuple[int, int]
    n_clutter: Tuple[int, int]         # static low-RCS point clutter
    vehicle_speed_mps: Tuple[float, float]     # magnitude range
    pedestrian_speed_mps: Tuple[float, float]
    rcs_jitter_db: float               # +- uniform jitter around class default RCS
    min_target_separation_m: float     # reject-sample so targets don't overlap


# Speed ranges shared across tiers that use "full" (unrestricted) target speeds.
# Vehicle: 0 - 30 m/s (~108 km/h, residential-to-arterial urban mix). Pedestrian:
# 0 - 3 m/s (standing still to a jog). Both are order-of-magnitude automotive-radar
# literature figures, not measured; the radial component is clamped downstream (see
# `_RADIAL_VELOCITY_FRAC`) so the exact upper bound does not need to match any single
# real-world distribution precisely.
_FULL_VEHICLE_SPEED = (0.0, 30.0)
_FULL_PEDESTRIAN_SPEED = (0.0, 3.0)

DIFFICULTY_TIERS: Dict[str, TierSpec] = {
    "D0": TierSpec(
        name="D0",
        n_vehicles=(1, 1),
        n_pedestrians=(0, 0),
        n_clutter=(0, 0),
        vehicle_speed_mps=(0.0, 2.0),
        pedestrian_speed_mps=(0.0, 0.0),
        rcs_jitter_db=0.0,
        min_target_separation_m=3.0,
    ),
    "D1": TierSpec(
        name="D1",
        n_vehicles=(1, 2),
        n_pedestrians=(0, 1),
        n_clutter=(0, 3),
        vehicle_speed_mps=(0.0, 10.0),
        pedestrian_speed_mps=(0.0, 1.5),
        rcs_jitter_db=2.0,
        min_target_separation_m=3.0,
    ),
    "D2": TierSpec(
        name="D2",
        n_vehicles=(2, 4),
        n_pedestrians=(1, 3),
        n_clutter=(5, 15),
        vehicle_speed_mps=_FULL_VEHICLE_SPEED,
        pedestrian_speed_mps=_FULL_PEDESTRIAN_SPEED,
        rcs_jitter_db=4.0,
        min_target_separation_m=3.0,
    ),
    "D3": TierSpec(
        name="D3",
        n_vehicles=(3, 6),
        n_pedestrians=(2, 5),
        n_clutter=(20, 40),
        vehicle_speed_mps=_FULL_VEHICLE_SPEED,
        pedestrian_speed_mps=_FULL_PEDESTRIAN_SPEED,
        rcs_jitter_db=6.0,
        min_target_separation_m=2.0,
    ),
}


def _resolve_tier(tier: Union[str, TierSpec]) -> TierSpec:
    if isinstance(tier, TierSpec):
        return tier
    return DIFFICULTY_TIERS[tier]  # raises KeyError on an unknown tier name


def _sample_fov_position(cfg, rng: np.random.Generator) -> np.ndarray:
    """A position in the radar's forward FOV, boresight +x / ULA axis +y.

    range ~ U(0.1*max_range, 0.85*max_range), sin(az) ~ U(-0.85, 0.85); sin_az is the
    direction cosine along the ULA axis (+y), so position = (r*cos_az, r*sin_az, 0)
    with cos_az = sqrt(1 - sin_az^2) (targets stay in front of the array, cos_az > 0).
    """
    max_range = cfg.max_range_m
    r = rng.uniform(_RANGE_FRAC[0] * max_range, _RANGE_FRAC[1] * max_range)
    sin_az = rng.uniform(_SIN_AZ_RANGE[0], _SIN_AZ_RANGE[1])
    cos_az = math.sqrt(1.0 - sin_az ** 2)
    return np.array([r * cos_az, r * sin_az, 0.0])


def _sample_separated_position(cfg, rng: np.random.Generator,
                                placed: List[np.ndarray], min_sep: float) -> Tuple[np.ndarray, int]:
    """`_sample_fov_position`, reject-sampled against `placed` for `min_sep`.

    Returns `(position, attempts_used)` -- `attempts_used` (1-based) is surfaced by
    `sample_scene` into `Scenario.metadata["placement_attempts"]` so a corpus can be
    monitored for how close the reject-sampler is running to `_MAX_PLACEMENT_ATTEMPTS`
    (see the stats-design review: a one-off stress test isn't as convincing as a
    per-corpus signal).
    """
    for attempt in range(1, _MAX_PLACEMENT_ATTEMPTS + 1):
        pos = _sample_fov_position(cfg, rng)
        if all(np.linalg.norm(pos - p) >= min_sep for p in placed):
            return pos, attempt
    raise RuntimeError(
        f"could not place a target with min_target_separation_m={min_sep} after "
        f"{_MAX_PLACEMENT_ATTEMPTS} attempts ({len(placed)} already placed)"
    )


def _sample_velocity(cfg, rng: np.random.Generator, position: np.ndarray,
                      speed_range: Tuple[float, float]) -> np.ndarray:
    """A random in-plane velocity, magnitude in `speed_range`, radial component
    (projection onto the radar->target line of sight) clamped to
    +-`_RADIAL_VELOCITY_FRAC` * cfg.max_velocity_mps so training frames never alias
    in Doppler. Clamping rescales the whole vector (direction is preserved; only the
    magnitude shrinks) rather than truncating just the radial part, so tangential
    velocity remains physically consistent with the (reduced) speed.

    DISTRIBUTION CAVEAT: the alias-free clamp binds whenever the sampled heading is
    mostly radial and the tier's top speed exceeds 0.8 * cfg.max_velocity_mps, so the
    EFFECTIVE speed distribution is config-dependent and can sit well below the tier's
    nominal range -- e.g. TI_IWR1443 (max_vel 12.8 m/s) caps mostly-radial D2/D3
    vehicles near ~10 m/s despite the tier's nominal 0-30 m/s, and DDMA configs
    (max_vel divided by n_tx) are far tighter still. This is the price of guaranteed
    alias-free Doppler labels; tiers state the SAMPLED range, not the post-clamp one.
    """
    speed = rng.uniform(speed_range[0], speed_range[1])
    theta = rng.uniform(0.0, 2.0 * math.pi)
    velocity = speed * np.array([math.cos(theta), math.sin(theta), 0.0])

    r = np.linalg.norm(position)
    e_los = position / r if r > 1e-9 else np.array([1.0, 0.0, 0.0])
    radial = float(np.dot(velocity, e_los))

    cap = _RADIAL_VELOCITY_FRAC * cfg.max_velocity_mps
    if abs(radial) > cap and abs(radial) > 1e-12:
        velocity = velocity * (cap / abs(radial))
    return velocity


def sample_scene(cfg, tier: Union[str, TierSpec], rng: np.random.Generator,
                 *, n_frames: int = 1) -> Scenario:
    """Draw a random radar scene at difficulty `tier`, spanning `n_frames` frames.

    `cfg` (a `RadarConfig`) sets the scene scale (`max_range_m`, `max_velocity_mps`,
    `frame_rate_hz`); `tier` selects a `TierSpec` (either by `DIFFICULTY_TIERS` key or
    directly); `rng` is a caller-seeded `numpy.random.Generator` -- determinism is the
    caller's contract (see module docstring). Raises `KeyError` for an unknown tier
    name and `RuntimeError` if `min_target_separation_m` cannot be satisfied within the
    attempt budget.

    `n_frames=1` (default) is the original single-instant behavior: vehicles/
    pedestrians get a static `Motion` and an explicit `velocity_mps` (used only for
    bookkeeping; there is no second frame to move into). `n_frames>1` gives moving
    (vehicle/pedestrian) objects a real constant-velocity `Motion` track: the sampled
    velocity vector is converted from m/s to the per-frame displacement `Motion.velocity`
    expects (`e2e.scenario.Motion` docstring) via `dt = 1 / cfg.frame_rate_hz`, so
    `resolve_motion`'s frame-indexed track advances the target physically frame to
    frame. Clutter stays static (and un-tracked) regardless of `n_frames`.
    """
    spec = _resolve_tier(tier)

    n_vehicles = int(rng.integers(spec.n_vehicles[0], spec.n_vehicles[1] + 1))
    n_pedestrians = int(rng.integers(spec.n_pedestrians[0], spec.n_pedestrians[1] + 1))
    n_clutter = int(rng.integers(spec.n_clutter[0], spec.n_clutter[1] + 1))

    dt = 1.0 / cfg.frame_rate_hz  # only used when n_frames > 1

    placed: List[np.ndarray] = []
    objects: List[SceneObject] = []
    placement_attempts = 0  # sum of reject-sampling attempts across all placements

    for i in range(n_vehicles):
        pos, attempts = _sample_separated_position(cfg, rng, placed, spec.min_target_separation_m)
        placement_attempts += attempts
        placed.append(pos)
        vel = _sample_velocity(cfg, rng, pos, spec.vehicle_speed_mps)
        rcs = DEFAULT_RCS_DBSM["vehicle"] + rng.uniform(-spec.rcs_jitter_db, spec.rcs_jitter_db)
        motion = Motion(velocity=tuple(float(v * dt) for v in vel)) if n_frames > 1 else None
        objects.append(vehicle(
            f"vehicle-{i}", tuple(float(x) for x in pos),
            velocity=tuple(float(x) for x in vel), motion=motion, rcs_dbsm=float(rcs),
        ))

    for i in range(n_pedestrians):
        pos, attempts = _sample_separated_position(cfg, rng, placed, spec.min_target_separation_m)
        placement_attempts += attempts
        placed.append(pos)
        vel = _sample_velocity(cfg, rng, pos, spec.pedestrian_speed_mps)
        rcs = DEFAULT_RCS_DBSM["pedestrian"] + rng.uniform(-spec.rcs_jitter_db, spec.rcs_jitter_db)
        motion = Motion(velocity=tuple(float(v * dt) for v in vel)) if n_frames > 1 else None
        objects.append(pedestrian(
            f"pedestrian-{i}", tuple(float(x) for x in pos),
            velocity=tuple(float(x) for x in vel), motion=motion, rcs_dbsm=float(rcs),
        ))

    for i in range(n_clutter):
        # Static low-RCS background points; exempt from the separation constraint
        # (real clutter is dense and unstructured, unlike discrete targets).
        pos = _sample_fov_position(cfg, rng)
        rcs = rng.uniform(_CLUTTER_RCS_DBSM_RANGE[0], _CLUTTER_RCS_DBSM_RANGE[1])
        objects.append(SceneObject(
            name=f"clutter-{i}",
            position=tuple(float(x) for x in pos),
            object_class="scatterer",
            rcs_dbsm=float(rcs),
        ))

    tag = f"{int(rng.integers(0, 1_000_000)):06d}"
    return Scenario(
        name=f"scene_{spec.name}_{tag}",
        base_scene="synthetic",  # no Sionna base scene; scatterers are analytic (rd_synth)
        num_frames=n_frames,
        nodes=[
            Node(
                name="radar",
                role=NodeRole.RADAR,
                position=(0.0, 0.0, 0.0),
                look_at=(1.0, 0.0, 0.0),
            ),
        ],
        objects=objects,
        description=f"Randomly sampled {spec.name} scene for {cfg.name}.",
        metadata={
            "tier": spec.name,
            "radar_config": cfg.name,
            "placement_attempts": placement_attempts,
        },
    )


def scene_summary(scenario: Scenario) -> dict:
    """Compact per-class counts + per-clutter position/RCS for manifests/debugging.

    Drops the old flat `classes` list (fully derivable from the counts here and added
    no new information -- see the stats-design review). Adds `clutter` (per-point
    position/RCS -- clutter is signal but not a label, so it is otherwise invisible in
    `meta["targets"]`, which is the only way to check e.g. "does clutter ever land on
    top of a real target's footprint") and `placement_attempts` (the reject-sampler's
    total attempt count for this scene, from `sample_scene`'s `Scenario.metadata`;
    `None` for a hand-built `Scenario` that never went through `sample_scene`).
    """
    clutter = [o for o in scenario.objects if o.object_class == "scatterer"]
    return {
        "n_vehicles": sum(1 for o in scenario.objects if o.object_class == "vehicle"),
        "n_pedestrians": sum(1 for o in scenario.objects if o.object_class == "pedestrian"),
        "n_clutter": len(clutter),
        "clutter": [
            {"position": list(o.position), "rcs_dbsm": o.rcs_dbsm} for o in clutter
        ],
        "placement_attempts": scenario.metadata.get("placement_attempts"),
    }
