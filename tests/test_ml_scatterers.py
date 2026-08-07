"""Tests for e2e.ml.scatterers (Scenario -> per-frame point-scatterer bridge).

Pure numpy/stdlib -- fast, no Sionna/torch.
"""

import numpy as np
import pytest

from e2e.scenario import Motion, Node, NodeRole, Scenario, SceneObject
from e2e.ml.scatterers import (
    DEFAULT_RCS_DBSM,
    RadarPose,
    Scatterer,
    frame_scatterers,
    pedestrian,
    radar_pose,
    vehicle,
)


def _radar_scenario(num_frames=5, look_at=None):
    return Scenario(
        name="radar_only",
        num_frames=num_frames,
        nodes=[
            Node(
                name="radar",
                role=NodeRole.RADAR,
                position=(0.0, 0.0, 1.5),
                look_at=look_at,
                motion=Motion(),
            ),
        ],
    )


# --------------------------------------------------------------------------- frame_scatterers: static

def test_static_object_zero_velocity_and_default_rcs_by_class():
    sc = _radar_scenario()
    sc.objects = [SceneObject(name="o0", position=(10.0, 0.0, 0.0), object_class="vehicle")]
    scat = frame_scatterers(sc, 0)
    assert len(scat) == 1
    s = scat[0]
    assert isinstance(s, Scatterer)
    assert s.position == (10.0, 0.0, 0.0)
    assert s.velocity == (0.0, 0.0, 0.0)
    assert s.rcs_dbsm == DEFAULT_RCS_DBSM["vehicle"]
    assert s.object_class == "vehicle"


def test_unknown_object_class_falls_back_to_scatterer_default():
    sc = _radar_scenario()
    sc.objects = [SceneObject(name="o0", position=(0.0, 0.0, 0.0), object_class="mailbox")]
    s = frame_scatterers(sc, 0)[0]
    assert s.rcs_dbsm == DEFAULT_RCS_DBSM["scatterer"]


def test_static_object_honors_velocity_mps():
    sc = _radar_scenario()
    sc.objects = [
        SceneObject(name="o0", position=(0.0, 0.0, 0.0), velocity_mps=(2.0, -1.0, 0.5)),
    ]
    s = frame_scatterers(sc, 0)[0]
    assert s.velocity == (2.0, -1.0, 0.5)


def test_explicit_rcs_dbsm_wins_over_default():
    sc = _radar_scenario()
    sc.objects = [SceneObject(name="o0", object_class="vehicle", rcs_dbsm=99.0)]
    s = frame_scatterers(sc, 0)[0]
    assert s.rcs_dbsm == 99.0


# --------------------------------------------------------------------------- frame_scatterers: moving

def test_linearly_moving_object_finite_difference_matches_motion_velocity():
    velocity = (3.0, -2.0, 0.0)
    sc = _radar_scenario(num_frames=10)
    sc.objects = [
        SceneObject(name="o0", position=(0.0, 0.0, 0.0), motion=Motion(velocity=velocity)),
    ]
    for frame_idx in range(sc.num_frames):
        s = frame_scatterers(sc, frame_idx)[0]
        assert np.allclose(s.velocity, velocity)


def test_moving_object_last_frame_uses_backward_difference():
    velocity = (1.0, 0.0, 0.0)
    sc = _radar_scenario(num_frames=4)
    sc.objects = [
        SceneObject(name="o0", position=(0.0, 0.0, 0.0), motion=Motion(velocity=velocity)),
    ]
    s = frame_scatterers(sc, sc.num_frames - 1)[0]
    assert np.allclose(s.velocity, velocity)


def test_dt_scales_finite_difference_velocity():
    velocity = (2.0, 0.0, 0.0)
    sc = _radar_scenario(num_frames=5)
    sc.objects = [
        SceneObject(name="o0", position=(0.0, 0.0, 0.0), motion=Motion(velocity=velocity)),
    ]
    s_dt1 = frame_scatterers(sc, 0, dt=1.0)[0]
    s_dt2 = frame_scatterers(sc, 0, dt=2.0)[0]
    assert np.allclose(s_dt1.velocity, (2.0, 0.0, 0.0))
    assert np.allclose(s_dt2.velocity, (1.0, 0.0, 0.0))


def test_frame_scatterers_position_tracks_motion():
    sc = _radar_scenario(num_frames=3)
    sc.objects = [
        SceneObject(name="o0", position=(0.0, 0.0, 0.0), motion=Motion(velocity=(1.0, 0.0, 0.0))),
    ]
    positions = [frame_scatterers(sc, i)[0].position for i in range(3)]
    assert np.allclose(positions[0], (0.0, 0.0, 0.0))
    assert np.allclose(positions[1], (1.0, 0.0, 0.0))
    assert np.allclose(positions[2], (2.0, 0.0, 0.0))


def test_frame_scatterers_rejects_out_of_range_frame_idx():
    sc = _radar_scenario(num_frames=3)
    with pytest.raises(ValueError):
        frame_scatterers(sc, 3)
    with pytest.raises(ValueError):
        frame_scatterers(sc, -1)


def test_frame_scatterers_one_per_object():
    sc = _radar_scenario()
    sc.objects = [
        SceneObject(name="o0", position=(0.0, 0.0, 0.0)),
        SceneObject(name="o1", position=(1.0, 0.0, 0.0)),
        SceneObject(name="o2", position=(2.0, 0.0, 0.0)),
    ]
    scat = frame_scatterers(sc, 0)
    assert len(scat) == 3
    assert [s.position[0] for s in scat] == [0.0, 1.0, 2.0]


# --------------------------------------------------------------------------- radar_pose

def test_radar_pose_default_boresight_is_plus_x():
    sc = _radar_scenario(look_at=None)
    pose = radar_pose(sc, 0)
    assert isinstance(pose, RadarPose)
    assert np.allclose(pose.position, (0.0, 0.0, 1.5))
    assert np.allclose(pose.boresight, (1.0, 0.0, 0.0))


def test_radar_pose_boresight_points_at_look_at_and_is_unit_norm():
    sc = _radar_scenario(look_at=(0.0, 10.0, 1.5))
    pose = radar_pose(sc, 0)
    assert np.allclose(pose.boresight, (0.0, 1.0, 0.0))
    assert np.isclose(np.linalg.norm(pose.boresight), 1.0)


def test_radar_pose_position_follows_node_motion():
    sc = Scenario(
        name="moving_radar",
        num_frames=5,
        nodes=[
            Node(
                name="radar",
                role=NodeRole.RADAR,
                position=(0.0, 0.0, 0.0),
                motion=Motion(velocity=(2.0, 0.0, 0.0)),
            ),
        ],
    )
    assert np.allclose(radar_pose(sc, 0).position, (0.0, 0.0, 0.0))
    assert np.allclose(radar_pose(sc, 3).position, (6.0, 0.0, 0.0))


def test_radar_pose_uses_first_radar_node_when_multiple_present():
    sc = Scenario(
        name="two_radars",
        num_frames=2,
        nodes=[
            Node(name="radar_a", role=NodeRole.RADAR, position=(1.0, 0.0, 0.0)),
            Node(name="radar_b", role=NodeRole.RADAR, position=(9.0, 0.0, 0.0)),
        ],
    )
    pose = radar_pose(sc, 0)
    assert np.allclose(pose.position, (1.0, 0.0, 0.0))


def test_radar_pose_raises_without_radar_node():
    sc = Scenario(name="no_radar", nodes=[Node(name="rx", role=NodeRole.COMM_RX)])
    with pytest.raises(ValueError):
        radar_pose(sc, 0)


# --------------------------------------------------------------------------- builders

def test_vehicle_builder_defaults():
    obj = vehicle("car-0", (1.0, 2.0, 0.0))
    assert obj.object_class == "vehicle"
    assert obj.position == (1.0, 2.0, 0.0)
    assert obj.rcs_dbsm is None
    assert obj.velocity_mps is None
    assert obj.motion.is_static


def test_pedestrian_builder_defaults():
    obj = pedestrian("ped-0", (0.0, 0.0, 0.0))
    assert obj.object_class == "pedestrian"
    assert obj.motion.is_static


def test_builders_honor_optional_overrides():
    obj = vehicle("car-1", (0.0, 0.0, 0.0), velocity=(1.0, 0.0, 0.0), rcs_dbsm=15.0)
    assert obj.velocity_mps == (1.0, 0.0, 0.0)
    assert obj.rcs_dbsm == 15.0

    moving = Motion(velocity=(0.5, 0.0, 0.0))
    obj2 = pedestrian("ped-1", (0.0, 0.0, 0.0), motion=moving)
    assert obj2.motion is moving
    assert not obj2.motion.is_static


def test_builders_produce_valid_scene_objects_embedded_in_scenario():
    sc = _radar_scenario()
    sc.objects = [
        vehicle("car-0", (10.0, 0.0, 0.0)),
        pedestrian("ped-0", (5.0, 0.0, 0.0), velocity=(0.5, 0.0, 0.0)),
    ]
    assert sc.validate() == []


def test_builders_scenario_json_roundtrip():
    sc = _radar_scenario()
    sc.objects = [
        vehicle("car-0", (10.0, 0.0, 0.0), velocity=(1.0, 0.0, 0.0), rcs_dbsm=8.0),
        pedestrian("ped-0", (5.0, 0.0, 0.0), motion=Motion(velocity=(0.3, 0.0, 0.0))),
    ]
    rebuilt = Scenario.from_json(sc.to_json())
    assert rebuilt.to_dict() == sc.to_dict()
    assert rebuilt == sc
    assert rebuilt.objects[0].object_class == "vehicle"
    assert rebuilt.objects[1].object_class == "pedestrian"
