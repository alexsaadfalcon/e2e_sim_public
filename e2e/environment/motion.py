"""
Pure motion / scheduling helpers for the scenario generation runner.

This module resolves a declarative ``Motion`` (from ``e2e.scenario``) plus a base
position into a per-frame trajectory: an ``(N_frames, 3)`` array of positions and,
where applicable, the matching ``(N_frames,)`` array of cumulative yaw angles.

It is intentionally **pure python + numpy** -- it imports neither Sionna, DrJit, nor
torch -- so the whole scheduling layer is unit-testable on any machine. The Sionna
runner (``scenario_runner.py``) consumes the arrays produced here and only translates
them into Sionna calls inside its (heavy) real path.

Three motion primitives are supported, mirroring ``scenario.Motion``:

* ``velocity``            -- constant displacement (meters) added each frame.
* ``waypoints``           -- positions interpolated across the frames so the entity
                             passes through each waypoint in order (overrides velocity).
* ``angular_velocity_deg`` -- yaw rotation (degrees/frame) about +z, applied around a
                             supplied pivot/centroid.

All functions accept plain tuples/lists and return numpy arrays, so they do not depend
on the dataclasses themselves -- ``resolve_motion`` is the convenience wrapper that
unpacks a ``Motion`` instance.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

Vec3 = Tuple[float, float, float]


# --------------------------------------------------------------------------------
# Translation primitives
# --------------------------------------------------------------------------------
def constant_velocity_track(base_position: Sequence[float],
                            velocity: Sequence[float],
                            num_frames: int) -> np.ndarray:
    """Per-frame positions for constant-velocity translation.

    Frame ``i`` sits at ``base_position + i * velocity`` so frame 0 is the base
    position (matching how the existing scripts step the radar/rx forward by a fixed
    delta each iteration). Returns an ``(num_frames, 3)`` float array.
    """
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    base = np.asarray(base_position, dtype=float).reshape(3)
    vel = np.asarray(velocity, dtype=float).reshape(3)
    steps = np.arange(num_frames, dtype=float).reshape(num_frames, 1)
    return base[None, :] + steps * vel[None, :]


def waypoint_track(base_position: Sequence[float],
                   waypoints: Sequence[Sequence[float]],
                   num_frames: int,
                   include_base: bool = True) -> np.ndarray:
    """Per-frame positions interpolated along an ordered list of waypoints.

    The entity starts at ``base_position`` (when ``include_base`` is True) and passes
    through each waypoint in order, with the frames distributed by piecewise-linear
    arc-length so motion speed is constant between control points. Returns an
    ``(num_frames, 3)`` float array.
    """
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    base = np.asarray(base_position, dtype=float).reshape(3)

    pts: List[np.ndarray] = []
    if include_base:
        pts.append(base)
    for w in waypoints:
        pts.append(np.asarray(w, dtype=float).reshape(3))

    if not pts:
        raise ValueError("waypoint_track requires at least one control point")
    if len(pts) == 1:
        # Nothing to interpolate -- hold the single point for every frame.
        return np.repeat(pts[0][None, :], num_frames, axis=0)

    ctrl = np.stack(pts, axis=0)  # (K, 3)

    # Cumulative arc length as the interpolation parameter.
    seg = np.linalg.norm(np.diff(ctrl, axis=0), axis=1)  # (K-1,)
    cum = np.concatenate([[0.0], np.cumsum(seg)])        # (K,)
    total = cum[-1]

    if total == 0.0:
        # All control points coincide; degenerate to a held position.
        return np.repeat(ctrl[0][None, :], num_frames, axis=0)

    # Parameter values for each frame, spanning the full path inclusive of endpoints.
    if num_frames == 1:
        sample = np.array([0.0])
    else:
        sample = np.linspace(0.0, total, num_frames)

    out = np.empty((num_frames, 3), dtype=float)
    for axis in range(3):
        out[:, axis] = np.interp(sample, cum, ctrl[:, axis])
    return out


# --------------------------------------------------------------------------------
# Rotation primitive
# --------------------------------------------------------------------------------
def yaw_angles(angular_velocity_deg: float, num_frames: int) -> np.ndarray:
    """Cumulative yaw angle (radians) per frame for a constant angular velocity.

    Frame 0 has zero rotation; frame ``i`` is ``i * angular_velocity_deg`` degrees,
    matching the incremental ``theta_step`` accumulation in the existing car stepping.
    """
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    return np.deg2rad(angular_velocity_deg) * np.arange(num_frames, dtype=float)


def rotate_about_z(positions: np.ndarray,
                   pivot: Sequence[float],
                   angles_rad: np.ndarray) -> np.ndarray:
    """Rotate each per-frame position about ``pivot`` around +z by the given angle.

    ``positions`` is ``(N, 3)`` and ``angles_rad`` is ``(N,)``; returns ``(N, 3)``.
    Useful for the circular car motion the existing environment code produces (a point
    offset from a center sweeps around it as the yaw advances).
    """
    positions = np.asarray(positions, dtype=float)
    pivot = np.asarray(pivot, dtype=float).reshape(3)
    angles_rad = np.asarray(angles_rad, dtype=float).reshape(-1)
    if positions.shape[0] != angles_rad.shape[0]:
        raise ValueError("positions and angles_rad must have the same length")

    rel = positions - pivot[None, :]
    cos = np.cos(angles_rad)
    sin = np.sin(angles_rad)
    out = np.empty_like(positions)
    out[:, 0] = rel[:, 0] * cos - rel[:, 1] * sin
    out[:, 1] = rel[:, 0] * sin + rel[:, 1] * cos
    out[:, 2] = rel[:, 2]
    return out + pivot[None, :]


# --------------------------------------------------------------------------------
# High-level resolver
# --------------------------------------------------------------------------------
def resolve_motion(base_position: Sequence[float],
                   motion,
                   num_frames: int,
                   pivot: Optional[Sequence[float]] = None) -> np.ndarray:
    """Resolve a ``scenario.Motion`` (or duck-typed equivalent) into a track.

    Precedence mirrors the ``Motion`` docstring:

    1. ``waypoints`` (if non-empty) drive translation, otherwise ``velocity`` does.
    2. ``angular_velocity_deg`` applies a +z yaw rotation. How it composes with
       translation depends on whether the entity is also translating:

       * **Pure rotation** (no velocity / waypoints): the entity orbits ``pivot``
         (defaulting to its base position) -- the circular sweep the car code expects.
       * **Combined translation + rotation** ("drive forward while turning"): the yaw
         is applied to the *local displacement from the base position* and the base is
         added back, i.e. the rotation is about the entity's OWN origin, not the scene
         pivot. This is the physically sensible composition: the entity stays anchored
         near its starting point and its heading curves over time. Rotating the already
         translated absolute track about a distant fixed pivot (the previous behavior)
         compounds a growing radius with a growing angle and produces an unbounded
         outward SPIRAL, which is not what "drive forward while turning" means.

    ``motion`` only needs ``.velocity``, ``.waypoints`` and ``.angular_velocity_deg``
    attributes, so this works for the dataclass without importing it here.

    Returns an ``(num_frames, 3)`` float array of world positions.
    """
    base = np.asarray(base_position, dtype=float).reshape(3)

    velocity = tuple(getattr(motion, "velocity", (0.0, 0.0, 0.0)))
    waypoints = list(getattr(motion, "waypoints", []) or [])
    ang_vel = float(getattr(motion, "angular_velocity_deg", 0.0))

    has_translation = bool(waypoints) or (tuple(velocity) != (0.0, 0.0, 0.0))

    # 1) translation
    if waypoints:
        track = waypoint_track(base, waypoints, num_frames)
    else:
        track = constant_velocity_track(base, velocity, num_frames)

    # 2) rotation about +z
    if ang_vel != 0.0:
        angles = yaw_angles(ang_vel, num_frames)
        if has_translation:
            # Combined motion: rotate the absolute track about the entity's OWN origin
            # (its base position). rotate_about_z subtracts the pivot, rotates, and adds
            # it back, so passing pivot=base rotates the local displacement (track - base)
            # and re-anchors at base. This curves the heading without the spiral that
            # rotating about a distant fixed pivot produces. ``pivot`` is intentionally
            # ignored here -- for a moving entity the natural turn center is the entity
            # itself, not the scene centroid.
            track = rotate_about_z(track, base, angles)
        else:
            # Pure rotation: orbit the supplied pivot (default: base position).
            if pivot is None:
                pivot = base
            track = rotate_about_z(track, pivot, angles)

    return track


def scene_centroid(positions: Sequence[Sequence[float]]) -> np.ndarray:
    """Centroid of a collection of (x, y, z) positions; ``(3,)`` array.

    Handy as a default rotation pivot when a scenario wants objects to orbit the
    scene center rather than their own base position.
    """
    arr = np.asarray(list(positions), dtype=float).reshape(-1, 3)
    if arr.shape[0] == 0:
        return np.zeros(3)
    return arr.mean(axis=0)


# --------------------------------------------------------------------------------
# Self-test (runs without Sionna)
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    print("motion.py self-test")

    # constant velocity: frame 0 == base, fixed delta each step
    t = constant_velocity_track((45.0, 90.0, 1.5), (1.0, 0.0, 0.0), 100)
    assert t.shape == (100, 3)
    assert np.allclose(t[0], (45.0, 90.0, 1.5))
    assert np.allclose(t[1], (46.0, 90.0, 1.5))
    assert np.allclose(t[99], (144.0, 90.0, 1.5))
    print("  constant_velocity_track OK", t[0], "->", t[-1])

    # static motion (zero velocity) holds position
    s = constant_velocity_track((1.0, 2.0, 3.0), (0.0, 0.0, 0.0), 10)
    assert np.allclose(s, np.array([(1.0, 2.0, 3.0)] * 10))
    print("  static hold OK")

    # waypoints: endpoints hit exactly, midpoint of single straight segment is halfway
    w = waypoint_track((0.0, 0.0, 0.0), [(10.0, 0.0, 0.0)], 11)
    assert np.allclose(w[0], (0.0, 0.0, 0.0))
    assert np.allclose(w[-1], (10.0, 0.0, 0.0))
    assert np.allclose(w[5], (5.0, 0.0, 0.0))
    print("  waypoint_track straight OK", w[0], "->", w[-1])

    # waypoints: multi-leg L-shape, constant-speed arc-length sampling
    w2 = waypoint_track((0.0, 0.0, 0.0), [(10.0, 0.0, 0.0), (10.0, 10.0, 0.0)], 21)
    assert np.allclose(w2[0], (0.0, 0.0, 0.0))
    assert np.allclose(w2[-1], (10.0, 10.0, 0.0))
    assert np.allclose(w2[10], (10.0, 0.0, 0.0))  # the corner at half arc length
    print("  waypoint_track L-shape OK")

    # single coincident control point holds
    w3 = waypoint_track((2.0, 2.0, 2.0), [], 5)
    assert np.allclose(w3, np.array([(2.0, 2.0, 2.0)] * 5))
    print("  waypoint degenerate OK")

    # yaw / rotation: a point at radius r orbits the pivot
    pivot = (0.0, 0.0, 0.0)
    pts = np.array([(1.0, 0.0, 5.0)] * 5)
    ang = yaw_angles(90.0, 5)  # 0, 90, 180, 270, 360 deg
    rot = rotate_about_z(pts, pivot, ang)
    assert np.allclose(rot[0], (1.0, 0.0, 5.0))
    assert np.allclose(rot[1], (0.0, 1.0, 5.0), atol=1e-9)
    assert np.allclose(rot[2], (-1.0, 0.0, 5.0), atol=1e-9)
    assert np.allclose(rot[4], (1.0, 0.0, 5.0), atol=1e-9)
    print("  rotate_about_z OK (z preserved, radius preserved)")

    # resolve_motion end-to-end via a tiny duck-typed Motion
    class _M:
        velocity = (1.0, 0.0, 0.0)
        waypoints: list = []
        angular_velocity_deg = 0.0

    rm = resolve_motion((45.0, 90.0, 1.5), _M(), 100)
    assert np.allclose(rm[0], (45.0, 90.0, 1.5))
    assert np.allclose(rm[-1], (144.0, 90.0, 1.5))
    print("  resolve_motion (velocity) OK")

    class _MR:
        velocity = (0.0, 0.0, 0.0)
        waypoints: list = []
        angular_velocity_deg = 90.0

    rmr = resolve_motion((1.0, 0.0, 0.0), _MR(), 4, pivot=(0.0, 0.0, 0.0))
    assert np.allclose(rmr[1], (0.0, 1.0, 0.0), atol=1e-9)
    print("  resolve_motion (rotation) OK")

    # centroid
    c = scene_centroid([(0.0, 0.0, 0.0), (2.0, 4.0, 6.0)])
    assert np.allclose(c, (1.0, 2.0, 3.0))
    print("  scene_centroid OK", c)

    print("ALL motion.py self-tests passed")
