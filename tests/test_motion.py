"""Tests for the pure motion / scheduling helpers (e2e.environment.motion).

Pure numpy/stdlib -- fast, no Sionna/torch.
"""

import numpy as np
import pytest

from e2e.environment.motion import (
    constant_velocity_track,
    waypoint_track,
    yaw_angles,
    rotate_about_z,
    resolve_motion,
    scene_centroid,
)
from e2e.scenario import Motion


# --------------------------------------------------------------------------- constant velocity

def test_constant_velocity_track_count_and_endpoints():
    base = (45.0, 90.0, 1.5)
    vel = (1.0, 0.0, 0.0)
    n = 10
    t = constant_velocity_track(base, vel, n)
    assert t.shape == (n, 3)
    np.testing.assert_allclose(t[0], base)
    np.testing.assert_allclose(t[1], (46.0, 90.0, 1.5))
    np.testing.assert_allclose(t[-1], (45.0 + (n - 1) * 1.0, 90.0, 1.5))


def test_constant_velocity_static_is_all_identical():
    base = (1.0, 2.0, 3.0)
    t = constant_velocity_track(base, (0.0, 0.0, 0.0), 7)
    assert t.shape == (7, 3)
    np.testing.assert_allclose(t, np.tile(base, (7, 1)))


def test_constant_velocity_rejects_zero_frames():
    with pytest.raises(ValueError):
        constant_velocity_track((0, 0, 0), (1, 0, 0), 0)


# --------------------------------------------------------------------------- waypoints

def test_waypoint_track_passes_through_endpoints():
    base = (0.0, 0.0, 0.0)
    wp = (10.0, 0.0, 0.0)
    t = waypoint_track(base, [wp], 11)
    assert t.shape == (11, 3)
    np.testing.assert_allclose(t[0], base)
    np.testing.assert_allclose(t[-1], wp)
    # straight segment: midpoint is halfway
    np.testing.assert_allclose(t[5], (5.0, 0.0, 0.0))


def test_waypoint_track_multi_leg_hits_corner_and_endpoints():
    base = (0.0, 0.0, 0.0)
    wps = [(10.0, 0.0, 0.0), (10.0, 10.0, 0.0)]
    t = waypoint_track(base, wps, 21)
    np.testing.assert_allclose(t[0], base)
    np.testing.assert_allclose(t[-1], wps[-1])
    # equal-length legs -> the corner sits at the halfway frame
    np.testing.assert_allclose(t[10], (10.0, 0.0, 0.0))


def test_waypoint_track_monotonic_in_arc_length():
    base = (0.0, 0.0, 0.0)
    wps = [(10.0, 0.0, 0.0), (10.0, 10.0, 0.0), (0.0, 10.0, 0.0)]
    n = 50
    t = waypoint_track(base, wps, n)
    seg = np.linalg.norm(np.diff(t, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    # cumulative arc length must be non-decreasing (monotonic in arc length)
    assert np.all(np.diff(cum) >= -1e-9)

    # True path arc length / (n-1) is the per-frame step along a straight leg.
    ctrl = np.array([base] + wps, dtype=float)
    path_len = np.linalg.norm(np.diff(ctrl, axis=0), axis=1).sum()
    nominal = path_len / (n - 1)
    # No frame advances more than the nominal straight-leg step (corner-straddling
    # frames take a shorter chord, so they are <= nominal, never larger).
    assert np.all(seg <= nominal + 1e-6)
    # Most steps equal the nominal step exactly (only corner-straddling frames differ).
    assert np.count_nonzero(np.isclose(seg, nominal)) >= len(seg) - len(wps)


def test_waypoint_track_degenerate_single_point_holds():
    base = (2.0, 2.0, 2.0)
    t = waypoint_track(base, [], 5)
    np.testing.assert_allclose(t, np.tile(base, (5, 1)))


# --------------------------------------------------------------------------- rotation

def test_yaw_angles_accumulate():
    ang = yaw_angles(90.0, 4)
    np.testing.assert_allclose(ang, np.deg2rad([0.0, 90.0, 180.0, 270.0]))


def test_rotate_about_z_quarter_turn():
    pts = np.array([(1.0, 0.0, 5.0)] * 5)
    ang = yaw_angles(90.0, 5)  # 0, 90, 180, 270, 360
    rot = rotate_about_z(pts, (0.0, 0.0, 0.0), ang)
    np.testing.assert_allclose(rot[0], (1.0, 0.0, 5.0), atol=1e-9)
    np.testing.assert_allclose(rot[1], (0.0, 1.0, 5.0), atol=1e-9)
    np.testing.assert_allclose(rot[2], (-1.0, 0.0, 5.0), atol=1e-9)
    np.testing.assert_allclose(rot[3], (0.0, -1.0, 5.0), atol=1e-9)
    np.testing.assert_allclose(rot[4], (1.0, 0.0, 5.0), atol=1e-9)
    # z is preserved
    np.testing.assert_allclose(rot[:, 2], 5.0)


def test_rotate_about_z_preserves_radius_about_pivot():
    pivot = (3.0, -2.0, 0.0)
    pts = np.array([(5.0, 1.0, 0.0)] * 6)
    ang = yaw_angles(37.0, 6)
    rot = rotate_about_z(pts, pivot, ang)
    r0 = np.linalg.norm(pts[0, :2] - np.array(pivot[:2]))
    for p in rot:
        assert np.linalg.norm(p[:2] - np.array(pivot[:2])) == pytest.approx(r0)


def test_rotate_about_z_length_mismatch_raises():
    with pytest.raises(ValueError):
        rotate_about_z(np.zeros((3, 3)), (0, 0, 0), np.zeros(2))


# --------------------------------------------------------------------------- resolve_motion + centroid

def test_resolve_motion_velocity_uses_real_dataclass():
    m = Motion(velocity=(1.0, 0.0, 0.0))
    t = resolve_motion((45.0, 90.0, 1.5), m, 100)
    assert t.shape == (100, 3)
    np.testing.assert_allclose(t[0], (45.0, 90.0, 1.5))
    np.testing.assert_allclose(t[-1], (144.0, 90.0, 1.5))


def test_resolve_motion_static_all_identical():
    m = Motion()  # static
    t = resolve_motion((7.0, 8.0, 9.0), m, 5)
    np.testing.assert_allclose(t, np.tile((7.0, 8.0, 9.0), (5, 1)))


def test_resolve_motion_waypoints_override_velocity():
    # velocity is set but waypoints take precedence
    m = Motion(velocity=(100.0, 0.0, 0.0), waypoints=[(0.0, 10.0, 0.0)])
    t = resolve_motion((0.0, 0.0, 0.0), m, 11)
    np.testing.assert_allclose(t[0], (0.0, 0.0, 0.0))
    np.testing.assert_allclose(t[-1], (0.0, 10.0, 0.0))
    # x stayed at zero -> velocity was ignored
    np.testing.assert_allclose(t[:, 0], 0.0)


def test_resolve_motion_rotation_about_pivot():
    m = Motion(angular_velocity_deg=90.0)
    t = resolve_motion((1.0, 0.0, 0.0), m, 4, pivot=(0.0, 0.0, 0.0))
    np.testing.assert_allclose(t[0], (1.0, 0.0, 0.0), atol=1e-9)
    np.testing.assert_allclose(t[1], (0.0, 1.0, 0.0), atol=1e-9)


def test_resolve_motion_combined_translation_and_rotation_does_not_spiral():
    """Combined linear + angular motion must "drive forward while turning", not spiral.

    Regression test for the spiral bug: the old code built the translated track and then
    rotated each translated point about a distant fixed pivot, so the orbit radius grew
    every frame and the entity flew off in an unbounded outward spiral. The fix rotates
    about the entity's own origin, so the distance from the base position grows only as
    fast as the translation itself (i * |velocity|), never faster.
    """
    base = (10.0, 0.0, 0.0)
    speed = 2.0
    m = Motion(velocity=(speed, 0.0, 0.0), angular_velocity_deg=15.0)
    n = 40
    # A far-away pivot is exactly what triggered the old spiral; it must be ignored now.
    t = resolve_motion(base, m, n, pivot=(1000.0, 1000.0, 0.0))

    assert t.shape == (n, 3)
    np.testing.assert_allclose(t[0], base)  # frame 0 unrotated, unmoved

    base_arr = np.asarray(base)
    dist_from_base = np.linalg.norm(t - base_arr[None, :], axis=1)
    expected = speed * np.arange(n)  # |i * velocity|, the pure-translation distance
    # Distance from base must equal the forward-travel distance (bounded, linear) -- the
    # rotation only curves the heading, it does not inflate the radius.
    np.testing.assert_allclose(dist_from_base, expected, atol=1e-9)

    # Sanity vs. the buggy behavior: an outward spiral's final radius would massively
    # exceed the straight-line travel distance. Here it must not.
    assert dist_from_base[-1] == pytest.approx(speed * (n - 1))

    # The path must actually curve (heading changes), i.e. it is not just the straight
    # translation: the y-coordinate leaves zero once the entity starts turning.
    assert np.any(np.abs(t[:, 1]) > 1e-6)


def test_resolve_motion_pure_rotation_still_orbits_pivot():
    """The pure-rotation (no translation) case must keep orbiting the pivot unchanged."""
    m = Motion(angular_velocity_deg=90.0)
    t = resolve_motion((1.0, 0.0, 0.0), m, 4, pivot=(0.0, 0.0, 0.0))
    np.testing.assert_allclose(t[0], (1.0, 0.0, 0.0), atol=1e-9)
    np.testing.assert_allclose(t[1], (0.0, 1.0, 0.0), atol=1e-9)
    np.testing.assert_allclose(t[2], (-1.0, 0.0, 0.0), atol=1e-9)
    # radius about the pivot is preserved (true orbit)
    r = np.linalg.norm(t[:, :2], axis=1)
    np.testing.assert_allclose(r, 1.0, atol=1e-9)


def test_scene_centroid():
    c = scene_centroid([(0.0, 0.0, 0.0), (2.0, 4.0, 6.0)])
    np.testing.assert_allclose(c, (1.0, 2.0, 3.0))


def test_scene_centroid_empty_is_origin():
    np.testing.assert_allclose(scene_centroid([]), (0.0, 0.0, 0.0))
