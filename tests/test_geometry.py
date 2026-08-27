"""
Tests for `e2e.environment.geometry` -- object extents, yaw, and the monostatic surface
point.

The surface point is the single geometric fact the 2026-08-17 label convention rests on:
a radar return comes from an object's nearest face, not from its centre. These tests pin
(a) that the point is exact for a sphere, (b) that it is computed from the object's ACTUAL
orientation rather than assuming an axis-aligned box, (c) that it always lies on the
radar-to-centre line (which is what lets the label layer keep one azimuth for both), and
(d) that extents follow the same asset dispatch -- and the same graceful degradation --
the RT scene builder loads its meshes through.
"""

import math

import numpy as np
import pytest

from e2e.environment.geometry import (BOX_EXTENT_M, PEDESTRIAN_EXTENT_M,
                                      SIONNA_CAR_EXTENT_M, SPHERE_EXTENT_M,
                                      nearest_surface_point, object_extent_m,
                                      object_yaw_rad, surface_range_offset_m)
from e2e.scenario import Motion, ObjectKind, SceneObject


# --------------------------------------------------------------------------------
# nearest_surface_point / surface_range_offset_m
# --------------------------------------------------------------------------------
@pytest.mark.parametrize("direction", [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                                       (0.6, 0.8, 0.0), (0.5, 0.5, math.sqrt(0.5))])
def test_sphere_surface_offset_is_exactly_its_radius(direction):
    """A sphere is the one shape the bounding-ellipsoid model is EXACT for: whatever the
    line of sight, its nearest surface point is one radius nearer than its centre."""
    radius = 1.25
    centre = np.array(direction) * 30.0
    delta = surface_range_offset_m(centre, (radius, radius, radius), np.zeros(3))
    assert delta == pytest.approx(radius, abs=1e-9)


def test_surface_point_lies_on_the_radar_to_centre_line():
    """The label layer gives surface and centre a SINGLE azimuth; that is only legitimate
    because the surface point is on the line of sight. Pinned as a cross product."""
    observer = np.array([1.0, -2.0, 1.5])
    centre = np.array([18.0, 7.0, 0.9])
    p = nearest_surface_point(centre, (2.2, 0.9, 0.75), observer, yaw_rad=0.7)
    u = (centre - observer) / np.linalg.norm(centre - observer)
    v = p - observer
    assert np.linalg.norm(np.cross(u, v)) == pytest.approx(0.0, abs=1e-9)
    assert np.dot(v, u) > 0.0                      # in front of the observer, not behind
    assert np.linalg.norm(v) < np.linalg.norm(centre - observer)


def test_yaw_decides_which_extent_faces_the_radar():
    """NOT axis-aligned: a long, narrow object end-on hides ~half its LENGTH, broadside
    ~half its WIDTH, and the model must follow its actual yaw to tell those apart."""
    centre = np.array([40.0, 0.0, 0.0])            # due +x from the observer
    half = (5.0, 1.0, 0.8)                          # 10 m long, 2 m wide

    end_on = surface_range_offset_m(centre, half, np.zeros(3), yaw_rad=0.0)
    broadside = surface_range_offset_m(centre, half, np.zeros(3), yaw_rad=math.pi / 2)
    oblique = surface_range_offset_m(centre, half, np.zeros(3), yaw_rad=math.pi / 4)

    assert end_on == pytest.approx(5.0, abs=1e-9)
    assert broadside == pytest.approx(1.0, abs=1e-9)
    # 45 degrees: 1/sqrt((cos45/5)^2 + (sin45/1)^2)
    assert oblique == pytest.approx(1.0 / math.hypot(math.cos(math.pi / 4) / 5.0,
                                                     math.sin(math.pi / 4) / 1.0), abs=1e-9)
    assert broadside < oblique < end_on
    # A yaw-blind (axis-aligned) implementation would return the same number for all
    # three; 4 m apart is not a rounding difference.
    assert end_on - broadside > 3.9


def test_yaw_is_periodic_and_symmetric():
    centre = np.array([25.0, 4.0, 0.0])
    half = (3.0, 1.1, 0.7)
    base = surface_range_offset_m(centre, half, np.zeros(3), yaw_rad=0.3)
    assert surface_range_offset_m(centre, half, np.zeros(3),
                                  yaw_rad=0.3 + math.pi) == pytest.approx(base, abs=1e-9)
    assert surface_range_offset_m(centre, half, np.zeros(3),
                                  yaw_rad=0.3 + 2 * math.pi) == pytest.approx(base, abs=1e-9)


def test_surface_point_is_clamped_in_front_of_the_observer():
    """A huge object centred close to the radar must not push its surface point through
    (or behind) the antenna."""
    p = nearest_surface_point((3.0, 0.0, 0.0), (20.0, 20.0, 20.0), np.zeros(3))
    assert 0.0 < float(np.linalg.norm(p)) <= 3.0


def test_degenerate_zero_range_returns_the_centre():
    p = nearest_surface_point((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), np.zeros(3))
    assert np.allclose(p, 0.0)


def test_rt_specular_point_delegates_here():
    """`rt_signal_chain._specular_point` and the label layer MUST be one implementation --
    a label computed differently from where the energy was placed is the defect the
    surface convention exists to remove."""
    pytest.importorskip("torch")
    from e2e.ml.rt_signal_chain import _specular_point

    centre = np.array([12.0, -3.0, 1.0])
    half = np.array([2.2, 0.9, 0.75])
    radar = np.array([0.0, 0.0, 1.5])
    assert np.allclose(_specular_point(centre, half, radar),
                       nearest_surface_point(centre, half, radar))


# --------------------------------------------------------------------------------
# object_yaw_rad
# --------------------------------------------------------------------------------
def test_yaw_from_velocity_heading():
    assert object_yaw_rad(velocity=(0.0, 4.0, 0.0)) == pytest.approx(math.pi / 2)
    assert object_yaw_rad(velocity=(-3.0, 0.0, 0.0)) == pytest.approx(math.pi)
    assert object_yaw_rad(velocity=(1.0, 1.0, 0.0)) == pytest.approx(math.pi / 4)


def test_stationary_object_has_no_declared_heading():
    """0.0 is the documented "unknown" -- a parked car's yaw is not recoverable from a
    Scenario that never states it. Vertical motion is not a heading either."""
    assert object_yaw_rad(velocity=(0.0, 0.0, 0.0)) == 0.0
    assert object_yaw_rad(velocity=(0.0, 0.0, 5.0)) == 0.0
    assert object_yaw_rad() == 0.0


def test_explicit_yaw_attribute_wins_over_velocity():
    """Forward-compatibility with a scenario layer that gains an explicit yaw: whatever
    sets it, labels and the RT scene builder read it through this one function."""
    class _Obj:
        yaw_deg = 90.0

    assert object_yaw_rad(_Obj(), velocity=(1.0, 0.0, 0.0)) == pytest.approx(math.pi / 2)

    class _ObjRad:
        yaw_rad = 0.25

    assert object_yaw_rad(_ObjRad(), velocity=(1.0, 0.0, 0.0)) == pytest.approx(0.25)


# --------------------------------------------------------------------------------
# object_extent_m
# --------------------------------------------------------------------------------
def test_sphere_and_box_primitives_scale():
    sphere = SceneObject(name="s", kind=ObjectKind.SPHERE, scaling=0.5)
    assert object_extent_m(sphere) == pytest.approx(tuple(0.5 * v for v in SPHERE_EXTENT_M))

    box = SceneObject(name="b", kind=ObjectKind.BOX, scaling=0.3)
    assert object_extent_m(box) == pytest.approx(tuple(0.3 * v for v in BOX_EXTENT_M))


def test_sionna_car_and_pedestrian_assets():
    car = SceneObject(name="c", kind=ObjectKind.MESH, asset="low_poly_car",
                      object_class="vehicle")
    assert object_extent_m(car) == pytest.approx(SIONNA_CAR_EXTENT_M)

    ped = SceneObject(name="p", kind=ObjectKind.MESH, asset="pedestrian_placeholder",
                      object_class="pedestrian")
    assert object_extent_m(ped) == pytest.approx(PEDESTRIAN_EXTENT_M)


def test_downloaded_asset_extent_matches_what_would_be_loaded():
    """A downloaded mesh resolves to its processed bbox when the asset cache exists on
    this machine, and to the Sionna car -- which is what `_object_mesh` would actually
    load -- when it does not. Either way the label describes the geometry in the scene."""
    from e2e.ml.assets import DOWNLOADED_ASSET_SPECS
    from e2e.environment.geometry import _processed_asset_extent_m

    name = "kn_sedan"
    assert name in DOWNLOADED_ASSET_SPECS
    obj = SceneObject(name="v", kind=ObjectKind.MESH, asset=name, object_class="vehicle")
    cached = _processed_asset_extent_m(name)
    expected = cached if cached is not None else SIONNA_CAR_EXTENT_M
    assert object_extent_m(obj) == pytest.approx(expected)
    # Whatever the machine, a car-sized answer -- never None, never a point.
    assert 3.0 < object_extent_m(obj)[0] < 7.0


def test_unknown_geometry_is_a_point_target():
    """`None` (not a guess) when nothing is known -- consumers then keep the pre-surface
    behaviour of labelling the centre."""
    mystery = SceneObject(name="?", kind=ObjectKind.MESH, asset="/tmp/nobody.ply",
                          object_class="scatterer")
    assert object_extent_m(mystery) is None


def test_explicit_extent_attribute_wins():
    obj = SceneObject(name="x", kind=ObjectKind.SPHERE, scaling=4.0)
    obj.extent_m = (2.0, 1.0, 0.5)                 # type: ignore[attr-defined]
    assert object_extent_m(obj) == pytest.approx((2.0, 1.0, 0.5))


# --------------------------------------------------------------------------------
# frame_scatterers wiring
# --------------------------------------------------------------------------------
def test_frame_scatterers_carries_extent_and_heading():
    from e2e.environment.scatterers import frame_scatterers
    from e2e.scenario import Node, NodeRole, Scenario

    scenario = Scenario(
        name="t", base_scene="flat", num_frames=2,
        nodes=[Node(name="radar", role=NodeRole.RADAR, position=(0.0, 0.0, 1.5))],
        objects=[SceneObject(name="car", kind=ObjectKind.MESH, asset="low_poly_car",
                             object_class="vehicle", position=(20.0, 0.0, 0.75),
                             motion=Motion(velocity=(0.0, 2.0, 0.0)))],
    )
    sc = frame_scatterers(scenario, 0, dt=1.0)[0]
    assert sc.extent_m == pytest.approx(SIONNA_CAR_EXTENT_M)
    assert sc.yaw_rad == pytest.approx(math.pi / 2)      # driving along +y


def test_analytic_scene_objects_stay_point_targets():
    """`e2e.ml.scenes`' analytic tiers are POINT targets -- `rd_synth` radiates from the
    object's `position`, so giving them an extent would move the label OFF the energy,
    which is the exact failure this convention exists to remove."""
    from e2e.environment.scatterers import SYNTHETIC_BASE_SCENE, frame_scatterers
    from e2e.scenario import Node, NodeRole, Scenario

    scenario = Scenario(
        name="t", base_scene=SYNTHETIC_BASE_SCENE, num_frames=1,
        nodes=[Node(name="radar", role=NodeRole.RADAR)],
        objects=[SceneObject(name="v", position=(10.0, 0.0, 0.0), object_class="vehicle")],
    )
    assert frame_scatterers(scenario, 0)[0].extent_m is None


def test_yaw_defers_to_the_heading_the_scene_builder_places():
    """The label layer must not hold a second opinion about which way an object faces.

    `e2e.ml.rt_scene_build.object_yaw_rad` sets the mesh's real orientation, including a
    deterministic pseudo-random heading for PARKED objects -- where a naive "0.0 if not
    moving" would be wrong by up to |L-W|/2 (1.2 m for a car, 6.4 m for a 15.7 m semi),
    which is more than the metric's whole 2.0 m match tolerance.
    """
    from e2e.ml.rt_scene_build import object_yaw_rad as placed_yaw

    class _Scat:
        velocity = (0.0, 0.0, 0.0)

    for name, seed in (("car0", 0), ("car1", 0), ("car0", 7)):
        assert object_yaw_rad(velocity=(0.0, 0.0, 0.0), name=name,
                              scene_seed=seed) == pytest.approx(
            placed_yaw(_Scat(), name, scene_seed=seed))
    # A parked object is NOT axis-aligned any more, so a bare 0.0 would be a real error.
    assert object_yaw_rad(velocity=(0.0, 0.0, 0.0), name="car0", scene_seed=0) != 0.0


def test_parked_object_labels_use_its_placed_heading():
    """End to end through `frame_scatterers`: a parked vehicle's `yaw_rad` is the scene
    builder's heading, so the surface point is computed off the axis it really presents."""
    from e2e.environment.geometry import scene_seed_for
    from e2e.ml.rt_scene_build import object_yaw_rad as placed_yaw
    from e2e.environment.scatterers import frame_scatterers
    from e2e.scenario import Node, NodeRole, Scenario

    scenario = Scenario(
        name="parked", base_scene="flat", num_frames=1,
        nodes=[Node(name="radar", role=NodeRole.RADAR, position=(0.0, 0.0, 1.5))],
        objects=[SceneObject(name="parked_car", kind=ObjectKind.MESH,
                             asset="low_poly_car", object_class="vehicle",
                             position=(20.0, 0.0, 0.75))],
    )
    sc = frame_scatterers(scenario, 0)[0]

    class _Scat:
        velocity = (0.0, 0.0, 0.0)

    # The seed is scene-stable, NOT frame-keyed: keying a parked object's heading on
    # frame_idx made it rotate between frames of one scene. Labels must key on exactly
    # what the placer keys on, so this asks the placer rather than hardcoding a seed.
    assert sc.yaw_rad == pytest.approx(
        placed_yaw(_Scat(), "parked_car", scene_seed=scene_seed_for(scenario)))
    delta = surface_range_offset_m(sc.position, tuple(0.5 * e for e in sc.extent_m),
                                   (0.0, 0.0, 1.5), yaw_rad=sc.yaw_rad)
    # Between half-width and half-length of the low_poly_car mesh, whatever the heading.
    assert 0.5 * SIONNA_CAR_EXTENT_M[1] - 1e-6 <= delta <= 0.5 * SIONNA_CAR_EXTENT_M[0] + 1e-6


def test_a_parked_object_keeps_one_heading_across_frames():
    """REGRESSION: the heading seed was `frame_idx`, so a PARKED CAR SILENTLY ROTATED
    from frame to frame within a single scene.

    Nothing failed at the time, because the placer and the label layer both used
    frame_idx and therefore agreed with each other -- the scene was simply physically
    absurd, and any model learning frame-to-frame consistency was learning noise. The
    seed is now derived from the scenario, so it is stable across frames and still
    distinct across scenes.
    """
    from e2e.environment.geometry import scene_seed_for
    from e2e.environment.scatterers import frame_scatterers
    from e2e.scenario import Node, NodeRole, Scenario

    def _scenario(name):
        return Scenario(
            name=name, base_scene="flat", num_frames=4,
            nodes=[Node(name="radar", role=NodeRole.RADAR, position=(0.0, 0.0, 1.5))],
            objects=[SceneObject(name="parked_car", kind=ObjectKind.MESH,
                                 asset="low_poly_car", object_class="vehicle",
                                 position=(20.0, 0.0, 0.75))],
        )

    scenario = _scenario("lot_a")
    yaws = [frame_scatterers(scenario, k)[0].yaw_rad for k in range(4)]
    assert len(set(yaws)) == 1, f"a parked car rotated across frames: {yaws}"

    # ...but two different scenes must not share the heading, or every corpus scene
    # would present the same aspect and the diversity this buys would be illusory.
    assert scene_seed_for(_scenario("lot_a")) != scene_seed_for(_scenario("lot_b"))
    other = frame_scatterers(_scenario("lot_b"), 0)[0].yaw_rad
    assert other != pytest.approx(yaws[0])
