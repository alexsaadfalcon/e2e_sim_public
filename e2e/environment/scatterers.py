"""
Scenario -> per-frame point-scatterer bridge.

This is the geometry front-end for the FMCW radar ML-dataset package: it turns a
declarative `e2e.scenario.Scenario` (nodes + objects + motion) into, per frame, a radar
pose and a list of point scatterers (position, velocity, RCS, class). A sibling module
turns those scatterer lists into range-Doppler tensors; this module owns none of that
synthesis.

Only numpy + stdlib + `e2e.scenario` / `e2e.environment.motion` are imported here (no
torch) so it stays usable from both the torch-based synthesizer and pure-Python dataset
manifest/labeling tooling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from e2e.environment.geometry import object_extent_m, object_yaw_rad, scene_seed_for
from e2e.environment.motion import resolve_motion
from e2e.scenario import Motion, NodeRole, Scenario, SceneObject, Vec3

# Coarse per-class radar-cross-section defaults (dBsm), used when a SceneObject doesn't
# set `rcs_dbsm` explicitly. These are typical automotive-radar literature order-of-
# magnitude figures (e.g. a car broadside ~10 dBsm, a pedestrian ~ -5 dBsm; see ETSI TR
# 103 257 automotive radar target modeling and standard radar-handbook RCS tables), not
# measured values -- callers needing accuracy should set `rcs_dbsm` on the object.
DEFAULT_RCS_DBSM = {
    "vehicle": 10.0,
    "pedestrian": -5.0,
    "scatterer": 0.0,
}

#: `Scenario.base_scene` value marking a scene whose objects are POINT targets: there is
#: no ray tracer and no mesh, and `e2e.chain.rd_synth.synthesize_adc` puts each object's
#: entire return at its `position`. `e2e.ml.scenes` (the analytic tier sampler) sets it.
#: Such objects get NO `extent_m`, because giving them one would move the label off the
#: energy -- the exact defect the surface convention exists to remove. Every other
#: `base_scene` ("flat"/"free"/a Sionna scene name) means real geometry is placed and
#: traced, so extents apply.
SYNTHETIC_BASE_SCENE = "synthetic"

# Frame-to-frame time step (seconds) assumed when the caller doesn't supply one.
# `e2e.scenario.Motion.velocity` is documented as "a constant displacement (meters)
# added each frame" -- i.e. neither `Scenario` nor `FrequencyPlan` carries a frame-rate /
# chirp-timing field, and `scenario_runner.py` never converts `frame_idx` to physical
# time (it is a plain integer index into `resolve_motion`'s per-frame track). We adopt
# the same implicit convention already baked into `resolve_motion`/`scenario_runner`:
# one frame step == one second, so a constant `Motion.velocity` (m/frame) is numerically
# identical to m/s. Pass an explicit `dt` to `frame_scatterers` if a real scenario's
# frame rate differs from this.
DEFAULT_DT_S = 1.0


@dataclass
class Scatterer:
    """A single scatterer at one frame: position/velocity in the scene frame.

    `position` is the object's geometric CENTRE. `extent_m`/`yaw_rad` carry just enough
    geometry for a consumer to work out where the object's SURFACE is along a given line
    of sight (`e2e.environment.geometry.nearest_surface_point`) -- which is where a real radar
    return comes from, and therefore where `e2e.ml.labels` puts its objectness footprint.
    Both default to "unknown", which every consumer must treat as a POINT target
    (surface == centre); a hand-built `Scatterer(position, velocity, rcs, cls)` keeps
    exactly its pre-2026-08-17 meaning.
    """
    position: Vec3   # meters, scene frame -- the object's geometric centre
    velocity: Vec3   # m/s
    rcs_dbsm: float
    object_class: str
    #: (length, width, height) in metres, in the object's own frame (local +x = length),
    #: after `SceneObject.scaling`. `None` = unknown geometry -> point target.
    extent_m: Optional[Vec3] = None
    #: Heading of the object's local +x, radians about world +z (see
    #: `e2e.environment.geometry.object_yaw_rad`).
    yaw_rad: float = 0.0


@dataclass
class RadarPose:
    """The (monostatic) radar node's pose at one frame.

    Scene frame convention: right-handed, **+z is world up**, distances in
    metres. `boresight` is the unit vector the array points along; the physical
    ULA is laid out along the lateral axis ``u = normalise(z_up x boresight)``
    (see `e2e.chain.rd_synth.array_axis`), so with the default pose (boresight
    = +x) a target at +y sits at positive azimuth.
    """
    position: Vec3 = (0.0, 0.0, 0.0)
    boresight: Vec3 = (1.0, 0.0, 0.0)  # unit vector


def _unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0.0:
        raise ValueError("cannot normalize a zero-length vector")
    return v / norm


def radar_pose(scenario: Scenario, frame_idx: int) -> RadarPose:
    """Resolve the first RADAR node's position and boresight at `frame_idx`.

    Position comes from the node's `Motion` track, resolved the same way
    `scenario_runner.build_schedule` does (`resolve_motion(node.position, node.motion,
    scenario.num_frames)`). Boresight is the unit vector from that position toward
    `look_at` when the node sets it; otherwise it defaults to +x, matching the
    "translate along +x" convention used by the reference radar scenarios in
    `e2e/scenario.py` (e.g. `munich_radar_scenario`). Note `look_at` is a fixed point,
    not a tracked target, so a moving radar's boresight direction changes over the
    track even though the aim point does not (same limitation scenario_runner documents).
    """
    radars = scenario.nodes_by_role(NodeRole.RADAR)
    if not radars:
        raise ValueError("scenario has no RADAR node")
    node = radars[0]
    track = resolve_motion(node.position, node.motion, scenario.num_frames)
    position = tuple(float(x) for x in track[frame_idx])
    if node.look_at is not None:
        boresight = _unit(np.asarray(node.look_at, dtype=float) - np.asarray(position, dtype=float))
    else:
        boresight = np.array([1.0, 0.0, 0.0])
    return RadarPose(position=position, boresight=tuple(float(x) for x in boresight))


def frame_scatterers(scenario: Scenario, frame_idx: int, dt: float = DEFAULT_DT_S) -> List[Scatterer]:
    """Resolve every `SceneObject` in `scenario` into a `Scatterer` at `frame_idx`.

    - `position`: motion-resolved via `resolve_motion` (same function
      `scenario_runner.build_schedule` uses for `object_tracks`).
    - `velocity`: for objects with non-static `Motion`, a finite difference of the
      resolved track (`(pos(t+1) - pos(t)) / dt`, backward-differenced at the last
      frame so every frame still gets an estimate). See `DEFAULT_DT_S` for the
      frame-duration convention used when `dt` isn't supplied. Static objects use
      `velocity_mps` when set, else zero.
    - `rcs_dbsm`: the object's own field when set, else `DEFAULT_RCS_DBSM[object_class]`
      (unknown classes fall back to the "scatterer" default).
    - `extent_m` / `yaw_rad`: the object's bbox extents and in-plane heading, resolved by
      `e2e.environment.geometry` from the same `(kind, asset, scaling)` dispatch the RT scene
      builder loads its mesh from. These let a consumer find the object's SURFACE along a
      line of sight instead of assuming a point at its centre; see `e2e.ml.labels`.
      `extent_m` is `None` for a `SYNTHETIC_BASE_SCENE` scenario, whose objects genuinely
      ARE points (nothing is meshed and `rd_synth` radiates from `position`).
    """
    n = scenario.num_frames
    if not (0 <= frame_idx < n):
        raise ValueError(f"frame_idx {frame_idx} out of range [0, {n})")

    # Point-target scene -> no extents (see SYNTHETIC_BASE_SCENE).
    meshed = str(getattr(scenario, "base_scene", "")) != SYNTHETIC_BASE_SCENE

    scatterers: List[Scatterer] = []
    for obj in scenario.objects:
        track = resolve_motion(obj.position, obj.motion, n)
        position = tuple(float(x) for x in track[frame_idx])

        if obj.motion.is_static:
            velocity = tuple(obj.velocity_mps) if obj.velocity_mps is not None else (0.0, 0.0, 0.0)
        else:
            if frame_idx < n - 1:
                delta = track[frame_idx + 1] - track[frame_idx]
            else:
                delta = track[frame_idx] - track[frame_idx - 1]
            velocity = tuple(float(x) for x in (delta / dt))

        rcs_dbsm = obj.rcs_dbsm if obj.rcs_dbsm is not None else DEFAULT_RCS_DBSM.get(
            obj.object_class, DEFAULT_RCS_DBSM["scatterer"]
        )
        scatterers.append(Scatterer(
            position=position,
            velocity=velocity,
            rcs_dbsm=rcs_dbsm,
            object_class=obj.object_class,
            extent_m=object_extent_m(obj) if meshed else None,
            # `name`/`scene_seed` make this defer to the heading the RT scene builder
            # will actually place the mesh at (`rt_scene_build.object_yaw_rad`, called
            # there with the SAME scene_seed) -- labels must not assume a different
            # orientation than the geometry was built with.
            yaw_rad=object_yaw_rad(obj, velocity, name=obj.name,
                                   scene_seed=scene_seed_for(scenario)),
        ))
    return scatterers


def vehicle(name: str, position: Vec3, *, velocity: Optional[Vec3] = None,
            motion: Optional[Motion] = None, rcs_dbsm: Optional[float] = None) -> SceneObject:
    """Convenience builder for a "vehicle"-class `SceneObject`."""
    return SceneObject(
        name=name,
        position=position,
        motion=motion if motion is not None else Motion(),
        object_class="vehicle",
        rcs_dbsm=rcs_dbsm,
        velocity_mps=velocity,
    )


def pedestrian(name: str, position: Vec3, *, velocity: Optional[Vec3] = None,
               motion: Optional[Motion] = None, rcs_dbsm: Optional[float] = None) -> SceneObject:
    """Convenience builder for a "pedestrian"-class `SceneObject`."""
    return SceneObject(
        name=name,
        position=position,
        motion=motion if motion is not None else Motion(),
        object_class="pedestrian",
        rcs_dbsm=rcs_dbsm,
        velocity_mps=velocity,
    )
