"""Tests for scene-object heading (`e2e.ml.rt_scene_build.object_yaw_rad`).

Regression for a realism gap found 2026-08-17: no orientation was applied to any scene
object, so every mesh sat axis-aligned and — because the radar looks down +x — every
vehicle in every generated corpus was nose-on. Two measured consequences: the corpora
contained no aspect-dependent RCS variation at all, and the centre-vs-surface label
offset had near-zero within-class variance (0.03–0.24 m), which made it look like a
per-class constant rather than the genuinely stochastic quantity it becomes once
vehicles can point in arbitrary directions.

These tests need no Sionna: the heading is pure geometry.
"""

import math

import pytest

from e2e.ml.rt_scene_build import _YAW_MOVING_EPS_MPS, object_yaw_rad


class _Scat:
    """Minimal stand-in: `object_yaw_rad` reads only `.velocity`."""

    def __init__(self, velocity):
        self.velocity = velocity


@pytest.mark.parametrize("velocity, expected", [
    ((5.0, 0.0, 0.0), 0.0),
    ((0.0, 5.0, 0.0), math.pi / 2),
    ((-3.0, 0.0, 0.0), math.pi),
    ((0.0, -2.0, 0.0), -math.pi / 2),
    ((1.0, 1.0, 0.0), math.pi / 4),
])
def test_moving_object_faces_its_direction_of_travel(velocity, expected):
    """A vehicle points where it is going. This is the whole reason the heading can be
    derived rather than stored: the per-frame velocity is already there."""
    assert object_yaw_rad(_Scat(velocity), "veh") == pytest.approx(expected, abs=1e-9)


def test_vertical_velocity_does_not_affect_heading():
    """Heading is about +z, so a purely vertical velocity component is irrelevant --
    and must not leak into the atan2."""
    assert object_yaw_rad(_Scat((4.0, 0.0, 9.0)), "veh") == pytest.approx(0.0, abs=1e-9)


def test_parked_object_gets_a_deterministic_heading():
    """`build_rt_tier_scenario` guarantees byte-identical scenarios for a given
    (tier, frame_idx, seed), and that guarantee has to survive headings. A parked
    object therefore cannot draw from a global RNG."""
    a = object_yaw_rad(_Scat((0.0, 0.0, 0.0)), "car0", scene_seed=3)
    b = object_yaw_rad(_Scat((0.0, 0.0, 0.0)), "car0", scene_seed=3)
    assert a == b


def test_parked_objects_differ_from_each_other_and_across_seeds():
    """A deterministic heading that is the SAME for every object would reintroduce the
    bug in a new form: a scene full of parallel cars."""
    same_seed = {object_yaw_rad(_Scat((0.0, 0.0, 0.0)), f"o{i}", scene_seed=0)
                 for i in range(24)}
    assert len(same_seed) == 24, "parked objects share a heading"
    a = object_yaw_rad(_Scat((0.0, 0.0, 0.0)), "car0", scene_seed=0)
    b = object_yaw_rad(_Scat((0.0, 0.0, 0.0)), "car0", scene_seed=1)
    assert a != b, "heading ignores the scene seed"


def test_parked_headings_span_the_full_circle():
    """The point is aspect DIVERSITY. Headings clustered in one quadrant would leave the
    corpora nearly as nose-on as before."""
    ys = [object_yaw_rad(_Scat((0.0, 0.0, 0.0)), f"o{i}") for i in range(400)]
    assert min(ys) >= 0.0 and max(ys) < 2 * math.pi
    # every quadrant populated
    quadrants = {int(y // (math.pi / 2)) for y in ys}
    assert quadrants == {0, 1, 2, 3}, f"headings not spread over the circle: {quadrants}"


def test_the_moving_threshold_is_applied_not_ignored():
    """Just below the threshold the object is parked (atan2 of a near-zero vector is
    meaningless noise); just above it, it faces its travel direction."""
    slow = _Scat((0.0, _YAW_MOVING_EPS_MPS * 0.5, 0.0))
    fast = _Scat((0.0, _YAW_MOVING_EPS_MPS * 2.0, 0.0))
    assert object_yaw_rad(fast, "veh") == pytest.approx(math.pi / 2, abs=1e-9)
    # the parked branch is name-derived, so it will not coincidentally equal pi/2
    assert object_yaw_rad(slow, "veh") != pytest.approx(math.pi / 2, abs=1e-6)
