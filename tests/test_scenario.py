"""Tests for the declarative scenario spec (e2e.scenario).

Pure numpy/stdlib -- fast, no Sionna/torch.
"""

import dataclasses

import pytest

from e2e.scenario import (
    Scenario,
    Node,
    SceneObject,
    ArrayConfig,
    Motion,
    FrequencyPlan,
    NodeRole,
    AntennaPattern,
    Polarization,
    ObjectKind,
    REFERENCE_SCENARIOS,
    SYSTEM_IMPEDANCE_OHMS,
    munich_radar_scenario,
    munich_isac_scenario,
    etoile_radar_scenario,
    munich_patrol_scenario,
)


# --------------------------------------------------------------------------- validate

@pytest.mark.parametrize("factory", list(REFERENCE_SCENARIOS.values()))
def test_reference_scenarios_validate_clean(factory):
    assert factory().validate() == []


def test_validate_flags_duplicate_node_names():
    sc = munich_radar_scenario()
    sc.nodes.append(Node(name="radar", role=NodeRole.MONITOR, position=(0, 0, 0)))
    assert any("unique" in p for p in sc.validate())


def test_validate_flags_duplicate_object_names():
    sc = munich_isac_scenario()
    dup = sc.objects[0].name
    sc.objects.append(SceneObject(name=dup, kind=ObjectKind.SPHERE))
    assert any("object names must be unique" in p for p in sc.validate())


def test_validate_flags_comm_tx_without_comm_rx():
    sc = Scenario(
        name="bad_comm",
        nodes=[Node(name="tx", role=NodeRole.COMM_TX, position=(0, 0, 0))],
    )
    problems = sc.validate()
    assert any("comm_tx" in p and "comm_rx" in p for p in problems)


def test_validate_flags_comm_rx_without_comm_tx():
    sc = Scenario(
        name="bad_comm_rx",
        nodes=[Node(name="rx", role=NodeRole.COMM_RX, position=(0, 0, 0))],
    )
    problems = sc.validate()
    assert any("comm_rx" in p and "comm_tx" in p for p in problems)


def test_validate_flags_nonpositive_array_dims():
    sc = munich_radar_scenario()
    sc.nodes[0].array.num_rows = 0
    assert any("num_rows" in p and "radar" in p for p in sc.validate())

    sc = munich_radar_scenario()
    sc.nodes[0].array.num_cols = -1
    assert any("num_cols" in p and "radar" in p for p in sc.validate())


def test_validate_flags_nonpositive_array_spacing():
    sc = munich_radar_scenario()
    sc.nodes[0].array.horizontal_spacing = 0.0
    assert any("spacings" in p and "radar" in p for p in sc.validate())

    sc = munich_radar_scenario()
    sc.nodes[0].array.vertical_spacing = -0.5
    assert any("spacings" in p and "radar" in p for p in sc.validate())


def test_validate_flags_num_freqs_lt_1():
    sc = munich_radar_scenario()
    sc.frequency.num_freqs = 0
    assert any("num_freqs" in p for p in sc.validate())


def test_validate_flags_num_frames_lt_1():
    sc = munich_radar_scenario()
    sc.num_frames = 0
    assert any("num_frames" in p for p in sc.validate())


def test_validate_flags_bad_freq_range():
    sc = munich_radar_scenario()
    sc.frequency.stop_hz = sc.frequency.start_hz  # stop <= start
    assert any("frequency stop" in p for p in sc.validate())

    sc.frequency.stop_hz = sc.frequency.start_hz - 1.0
    assert any("frequency stop" in p for p in sc.validate())


def test_validate_flags_no_nodes():
    sc = Scenario(name="empty", nodes=[])
    assert any("no nodes" in p for p in sc.validate())


@pytest.mark.parametrize("bad_pol", [Polarization.VH, Polarization.CROSS])
def test_validate_rejects_dual_polarization(bad_pol):
    """VH/CROSS double Sionna's antenna port count vs ArrayConfig.num_elements; the
    runtime frame contract (aperture grid, 32x32 view) is not dual-pol aware yet."""
    sc = munich_radar_scenario()
    sc.nodes[0].array.polarization = bad_pol
    problems = sc.validate()
    assert any(
        "dual-polarization" in p and "radar" in p and bad_pol.value in p
        for p in problems
    )


@pytest.mark.parametrize("good_pol", [Polarization.V, Polarization.H])
def test_validate_allows_single_polarization(good_pol):
    sc = munich_radar_scenario()
    sc.nodes[0].array.polarization = good_pol
    assert sc.validate() == []


# --------------------------------------------------------------------------- round-trips

@pytest.mark.parametrize("name,factory", list(REFERENCE_SCENARIOS.items()))
def test_to_dict_from_dict_roundtrip(name, factory):
    sc = factory()
    rebuilt = Scenario.from_dict(sc.to_dict())
    assert rebuilt.to_dict() == sc.to_dict()
    assert rebuilt == sc


@pytest.mark.parametrize("name,factory", list(REFERENCE_SCENARIOS.items()))
def test_to_json_from_json_roundtrip(name, factory):
    sc = factory()
    rebuilt = Scenario.from_json(sc.to_json())
    assert rebuilt.to_dict() == sc.to_dict()
    assert rebuilt == sc


@pytest.mark.parametrize("name,factory", list(REFERENCE_SCENARIOS.items()))
def test_save_load_roundtrip(tmp_path, name, factory):
    sc = factory()
    path = tmp_path / f"{name}.json"
    sc.save(str(path))
    assert path.is_file()
    loaded = Scenario.load(str(path))
    assert loaded.to_dict() == sc.to_dict()
    assert loaded == sc


# --------------------------------------------------------------------------- accessors

def test_is_isac_true_for_munich_isac():
    assert munich_isac_scenario().is_isac is True


def test_is_isac_false_for_munich_radar():
    assert munich_radar_scenario().is_isac is False


def test_nodes_by_role_filters():
    sc = munich_isac_scenario()
    radars = sc.nodes_by_role(NodeRole.RADAR)
    assert [n.name for n in radars] == ["car_radar"]
    assert all(n.role == NodeRole.RADAR for n in radars)

    tx = sc.nodes_by_role(NodeRole.COMM_TX)
    assert [n.name for n in tx] == ["building_comm_tx"]

    rx = sc.nodes_by_role(NodeRole.COMM_RX)
    assert [n.name for n in rx] == ["car_comm_rx"]

    # accepts the raw string value too (NodeRole(role) coercion)
    assert sc.nodes_by_role("radar") == radars

    # no MONITOR nodes in this scenario
    assert sc.nodes_by_role(NodeRole.MONITOR) == []


# --------------------------------------------------------------------------- enums survive serialization

def test_enums_survive_serialization():
    sc = Scenario(
        name="enum_check",
        nodes=[
            Node(
                name="n0",
                role=NodeRole.MONITOR,
                position=(1.0, 2.0, 3.0),
                array=ArrayConfig(
                    num_rows=2,
                    num_cols=2,
                    pattern=AntennaPattern.TR38901,
                    polarization=Polarization.CROSS,
                ),
            ),
        ],
        objects=[
            SceneObject(name="o0", kind=ObjectKind.BOX, position=(0, 0, 0)),
        ],
    )
    rebuilt = Scenario.from_json(sc.to_json())

    node = rebuilt.nodes[0]
    assert isinstance(node.role, NodeRole) and node.role == NodeRole.MONITOR
    assert isinstance(node.array.pattern, AntennaPattern) and node.array.pattern == AntennaPattern.TR38901
    assert isinstance(node.array.polarization, Polarization) and node.array.polarization == Polarization.CROSS

    obj = rebuilt.objects[0]
    assert isinstance(obj.kind, ObjectKind) and obj.kind == ObjectKind.BOX


def test_json_serializes_enum_values_as_strings():
    # asdict() of a (str, Enum) yields the enum members; json must serialize cleanly.
    sc = munich_isac_scenario()
    s = sc.to_json()
    assert '"role"' in s
    # the string value of the enum should appear, not a python repr
    assert '"radar"' in s
    assert "NodeRole" not in s


# --------------------------------------------------------------------------- tuples preserved

def test_position_and_waypoints_become_tuples_after_roundtrip():
    sc = Scenario(
        name="tuple_check",
        nodes=[
            Node(
                name="n0",
                role=NodeRole.RADAR,
                position=(1.0, 2.0, 3.0),
                look_at=(4.0, 5.0, 6.0),
                motion=Motion(velocity=(0.5, 0.0, 0.0), waypoints=[(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)]),
            ),
        ],
    )
    rebuilt = Scenario.from_dict(sc.to_dict())
    n = rebuilt.nodes[0]
    assert n.position == (1.0, 2.0, 3.0)
    assert n.look_at == (4.0, 5.0, 6.0)
    assert n.motion.velocity == (0.5, 0.0, 0.0)
    assert n.motion.waypoints == [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)]
    assert rebuilt == sc


# --------------------------------------------------------------------------- misc dataclass props

def test_array_num_elements():
    assert ArrayConfig(num_rows=32, num_cols=32).num_elements == 1024
    assert ArrayConfig(num_rows=1, num_cols=1).num_elements == 1


def test_motion_is_static():
    assert Motion().is_static is True
    assert Motion(velocity=(1.0, 0.0, 0.0)).is_static is False
    assert Motion(waypoints=[(1.0, 0.0, 0.0)]).is_static is False
    assert Motion(angular_velocity_deg=5.0).is_static is False


def test_frequency_linspace(n_freqs):
    fp = FrequencyPlan(start_hz=1.0, stop_hz=2.0, num_freqs=n_freqs)
    grid = fp.linspace()
    assert grid.shape == (n_freqs,)
    assert grid[0] == pytest.approx(1.0)
    assert grid[-1] == pytest.approx(2.0)


# --------------------------------------------------------------------------- new reference scenarios

def test_etoile_radar_registered_and_validates():
    assert REFERENCE_SCENARIOS["etoile_radar"] is etoile_radar_scenario
    sc = etoile_radar_scenario()
    assert sc.validate() == []
    assert sc.base_scene == "etoile"


def test_etoile_radar_roundtrip():
    sc = etoile_radar_scenario()
    rebuilt = Scenario.from_json(sc.to_json())
    assert rebuilt.to_dict() == sc.to_dict()
    assert rebuilt == sc


def test_munich_patrol_registered_and_validates():
    assert REFERENCE_SCENARIOS["munich_patrol"] is munich_patrol_scenario
    sc = munich_patrol_scenario()
    assert sc.validate() == []
    assert sc.base_scene == "munich"


def test_munich_patrol_roundtrip():
    sc = munich_patrol_scenario()
    rebuilt = Scenario.from_json(sc.to_json())
    assert rebuilt.to_dict() == sc.to_dict()
    assert rebuilt == sc


def test_munich_patrol_radar_has_l_shaped_waypoint_track():
    sc = munich_patrol_scenario()
    radar = sc.nodes_by_role(NodeRole.RADAR)[0]
    assert len(radar.motion.waypoints) >= 3
    assert radar.motion.is_static is False


def test_munich_patrol_objects_exercise_motion_primitives():
    sc = munich_patrol_scenario()
    assert len(sc.objects) >= 3
    # at least one car uses plain velocity, one uses pure rotation, and one combines both
    assert any(o.motion.velocity != (0.0, 0.0, 0.0) and o.motion.angular_velocity_deg == 0.0
               for o in sc.objects)
    assert any(o.motion.velocity == (0.0, 0.0, 0.0) and o.motion.angular_velocity_deg != 0.0
               for o in sc.objects)
    assert any(o.motion.velocity != (0.0, 0.0, 0.0) and o.motion.angular_velocity_deg != 0.0
               for o in sc.objects)


# --------------------------------------------------------------------------- link-budget config

def test_system_impedance_ohms_is_50():
    assert SYSTEM_IMPEDANCE_OHMS == 50.0


def test_tx_power_dbm_defaults_to_none():
    n = Node(name="n0", role=NodeRole.RADAR)
    assert n.tx_power_dbm is None


def test_tx_power_dbm_json_roundtrip_preserves_explicit_value():
    sc = Scenario(
        name="tx_power_check",
        nodes=[Node(name="tx", role=NodeRole.RADAR, tx_power_dbm=15.5)],
    )
    rebuilt = Scenario.from_json(sc.to_json())
    assert rebuilt.nodes[0].tx_power_dbm == 15.5
    assert rebuilt == sc


def test_tx_power_dbm_missing_key_loads_as_none_back_compat():
    sc = munich_radar_scenario()
    d = sc.to_dict()
    del d["nodes"][0]["tx_power_dbm"]
    rebuilt = Scenario.from_dict(d)
    assert rebuilt.nodes[0].tx_power_dbm is None


def test_validate_flags_tx_power_dbm_on_non_transmitting_role():
    sc = Scenario(
        name="bad_tx_power",
        nodes=[
            Node(name="tx", role=NodeRole.COMM_TX, position=(0, 0, 0)),
            Node(name="rx", role=NodeRole.COMM_RX, position=(1, 0, 0), tx_power_dbm=10.0),
        ],
    )
    problems = sc.validate()
    assert any("tx_power_dbm" in p and "rx" in p and "comm_rx" in p for p in problems)


def test_validate_allows_tx_power_dbm_on_radar_and_comm_tx():
    sc = Scenario(
        name="ok_tx_power",
        nodes=[
            Node(name="radar", role=NodeRole.RADAR, position=(0, 0, 0), tx_power_dbm=12.0),
            Node(name="tx", role=NodeRole.COMM_TX, position=(1, 0, 0), tx_power_dbm=12.0),
            Node(name="rx", role=NodeRole.COMM_RX, position=(2, 0, 0)),
        ],
    )
    assert sc.validate() == []


@pytest.mark.parametrize("factory", list(REFERENCE_SCENARIOS.values()))
def test_reference_scenarios_transmitting_nodes_have_tx_power_dbm(factory):
    sc = factory()
    tx_nodes = [n for n in sc.nodes if n.role in (NodeRole.RADAR, NodeRole.COMM_TX)]
    assert tx_nodes, "expected at least one TX-capable node"
    for n in tx_nodes:
        assert n.tx_power_dbm == 12.0
    for n in sc.nodes:
        if n.role == NodeRole.COMM_RX:
            assert n.tx_power_dbm is None
    assert sc.validate() == []
