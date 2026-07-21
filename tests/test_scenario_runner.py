"""Tests for the offline scenario generation runner (e2e.environment.scenario_runner).

Only the dry-run / mock path is exercised by default -- it imports Sionna lazily, so the
module imports and dry-run generation work with no Sionna installed. Any test that would
invoke the real Sionna ray-tracing path is marked @pytest.mark.sionna (auto-skipped
unless RUN_SIONNA=1).

All output goes to tmp_path; we never write into e2e/environment/sionna_sims/.

``ScenarioRunner.run()`` returns (and dumps) the self-describing v2 payload
``{"meta": {...}, "links": {link_name: stacked_array}}`` (see the module docstring's
"Links and output layout" section). These tests inspect that raw dict/pickle directly --
NOT via ``SionnaIterator`` (loading v2 is a separate shard's job).
"""

import pickle
import subprocess
import sys

import numpy as np
import pytest

from e2e.scenario import (
    munich_radar_scenario,
    munich_isac_scenario,
    etoile_radar_scenario,
    munich_patrol_scenario,
    Scenario,
    Node,
    NodeRole,
    ArrayConfig,
    FrequencyPlan,
    SYSTEM_IMPEDANCE_OHMS,
)

_SPEED_OF_LIGHT = 299_792_458.0


# --------------------------------------------------------------------------- import-level

def test_module_imports_without_sionna():
    """Importing the runner must not require Sionna (it imports it lazily)."""
    import importlib

    mod = importlib.import_module("e2e.environment.scenario_runner")
    assert hasattr(mod, "ScenarioRunner")
    assert hasattr(mod, "build_schedule")
    assert hasattr(mod, "main")
    # Sionna must NOT have been pulled in just by importing the runner.
    assert "sionna" not in sys.modules


# --------------------------------------------------------------------------- schedule helper

def test_build_schedule_track_lengths(small_scenario):
    from e2e.environment.scenario_runner import build_schedule

    sched = build_schedule(small_scenario)
    assert sched.num_frames == small_scenario.num_frames
    for node in small_scenario.nodes:
        track = sched.node_tracks[node.name]
        assert track.shape == (small_scenario.num_frames, 3)
    for obj in small_scenario.objects:
        assert sched.object_tracks[obj.name].shape == (small_scenario.num_frames, 3)


def test_build_schedule_isac_all_entities(small_scenario):
    from e2e.environment.scenario_runner import build_schedule

    sc = munich_isac_scenario()
    sc.num_frames = 4
    sched = build_schedule(sc)
    assert set(sched.node_tracks) == {n.name for n in sc.nodes}
    assert set(sched.object_tracks) == {o.name for o in sc.objects}
    for name in sched.node_tracks:
        assert sched.node_tracks[name].shape == (4, 3)


# --------------------------------------------------------------------------- dry-run generation

def _expected_shape(scenario):
    from e2e.environment.scenario_runner import ScenarioRunner

    r = ScenarioRunner(scenario, dry_run=True)
    return (scenario.num_frames,) + r.frame_shape


def test_dry_run_writes_pkl_with_documented_shape(tmp_path, small_scenario):
    from e2e.environment.scenario_runner import ScenarioRunner

    runner = ScenarioRunner(small_scenario, dry_run=True, seed=7)
    out = tmp_path / "small.pkl"
    payload = runner.run(out_path=str(out), verbose=False)

    n_freqs = small_scenario.frequency.num_freqs
    # 32x32 radar -> 1024 rx ant, 1x1 effective tx, 1 time step
    expected = (small_scenario.num_frames, 1024, 1, 1, n_freqs)
    link_name = runner.primary_link.name
    arr = payload["links"][link_name]
    assert arr.shape == expected
    assert arr.dtype == np.complex64
    assert runner.frame_shape == (1024, 1, 1, n_freqs)

    # meta correctness (v2 payload contract)
    meta = payload["meta"]
    assert meta["version"] == 1
    assert meta["scenario_name"] == small_scenario.name
    assert meta["freq_plan"] == {
        "carrier_hz": small_scenario.frequency.carrier_hz,
        "start_hz": small_scenario.frequency.start_hz,
        "stop_hz": small_scenario.frequency.stop_hz,
        "num_freqs": n_freqs,
    }
    link_meta = meta["links"][link_name]
    assert link_meta["rx_array_shape"] == [32, 32]
    assert link_meta["n_tx_ant"] == 1
    assert link_meta["kind"] == "radar"

    # the dumped file matches the returned payload
    assert out.is_file()
    with open(out, "rb") as f:
        on_disk = pickle.load(f)
    assert set(on_disk.keys()) == {"meta", "links"}
    on_disk_arr = on_disk["links"][link_name]
    assert on_disk_arr.shape == expected
    assert on_disk_arr.dtype == np.complex64
    np.testing.assert_array_equal(on_disk_arr, arr)
    assert on_disk["meta"] == meta


def test_dry_run_munich_radar_full_reference(tmp_path):
    from e2e.environment.scenario_runner import ScenarioRunner

    sc = munich_radar_scenario()
    sc.num_frames = 3
    sc.frequency.num_freqs = 16
    runner = ScenarioRunner(sc, dry_run=True)
    out = tmp_path / "munich_radar.pkl"
    payload = runner.run(out_path=str(out), verbose=False)
    arr = payload["links"][runner.primary_link.name]
    assert arr.shape == (3, 1024, 1, 1, 16)
    assert arr.dtype == np.complex64


def test_dry_run_isac_is_multilink(tmp_path):
    """ISAC scenario exports one frame-stack per link (radar + comm), under one dict."""
    from e2e.environment.scenario_runner import ScenarioRunner

    sc = munich_isac_scenario()
    sc.num_frames = 2
    sc.frequency.num_freqs = 8
    runner = ScenarioRunner(sc, dry_run=True)

    assert runner.is_multilink
    assert {link.name for link in runner.links} == {
        "car_radar", "building_comm_tx__car_comm_rx",
    }
    # primary (first) link is the radar: 32x32 rx -> 1024, 1x1 effective tx
    assert runner.n_rx_ant == 1024 and runner.n_tx_ant == 1

    payload = runner.run(out_path=str(tmp_path / "isac.pkl"), verbose=False)
    assert isinstance(payload, dict) and set(payload.keys()) == {"meta", "links"}
    links = payload["links"]
    assert set(links.keys()) == {"car_radar", "building_comm_tx__car_comm_rx"}
    assert links["car_radar"].shape == (2, 1024, 1, 1, 8)              # radar link
    assert links["building_comm_tx__car_comm_rx"].shape == (2, 16, 1, 1, 8)  # 4x4 comm rx
    for arr in links.values():
        assert arr.dtype == np.complex64

    # meta correctness for the mixed radar/comm scenario
    meta = payload["meta"]
    assert set(meta["links"].keys()) == set(links.keys())
    radar_meta = meta["links"]["car_radar"]
    assert radar_meta["kind"] == "radar"
    assert radar_meta["rx_array_shape"] == [32, 32]
    assert radar_meta["n_tx_ant"] == 1
    comm_meta = meta["links"]["building_comm_tx__car_comm_rx"]
    assert comm_meta["kind"] == "comm"
    assert comm_meta["tx_node"] == "building_comm_tx"
    assert comm_meta["rx_node"] == "car_comm_rx"
    assert comm_meta["rx_array_shape"] == [4, 4]


def test_dry_run_isac_payload_link_names_and_shapes(tmp_path):
    """v2 payload's "links" map holds every enumerated link, addressable by name.

    Regression test for the previous SionnaIterator-based link-selection test: link
    selection itself now lives in the reader (SionnaIterator), so here we only pin the
    writer-side contract -- the "links" dict has exactly the enumerated link names, each
    mapping to an array of that link's shape.
    """
    from e2e.environment.scenario_runner import ScenarioRunner

    sc = munich_isac_scenario()
    sc.num_frames = 2
    sc.frequency.num_freqs = 8
    out = tmp_path / "isac.pkl"
    payload = ScenarioRunner(sc, dry_run=True).run(out_path=str(out), verbose=False)

    links = payload["links"]
    assert set(links.keys()) == {"car_radar", "building_comm_tx__car_comm_rx"}
    assert links["car_radar"].shape == (2, 1024, 1, 1, 8)
    assert links["building_comm_tx__car_comm_rx"].shape == (2, 16, 1, 1, 8)
    assert "does_not_exist" not in links


def test_dry_run_payload_reshapes_for_runtime_pipeline(tmp_path, small_scenario):
    from e2e.environment.scenario_runner import ScenarioRunner

    n_freqs = small_scenario.frequency.num_freqs
    out = tmp_path / "small.pkl"
    runner = ScenarioRunner(small_scenario, dry_run=True)
    payload = runner.run(out_path=str(out), verbose=False)

    arr = payload["links"][runner.primary_link.name]
    assert arr.shape[0] == small_scenario.num_frames

    frame = np.asarray(arr[0])
    assert frame.shape == (1024, 1, 1, n_freqs)
    # the runtime pipeline views a single 32x32 radar frame as (32, 32, num_freqs)
    reshaped = frame.reshape(32, 32, n_freqs)
    assert reshaped.shape == (32, 32, n_freqs)


def test_dry_run_frames_differ_when_motion_present(tmp_path, small_scenario):
    from e2e.environment.scenario_runner import ScenarioRunner

    # small_scenario (munich_radar) has a moving radar (velocity (1,0,0)).
    assert any(not n.motion.is_static for n in small_scenario.nodes)

    runner = ScenarioRunner(small_scenario, dry_run=True, seed=3)
    payload = runner.run(out_path=str(tmp_path / "moving.pkl"), verbose=False)
    arr = payload["links"][runner.primary_link.name]
    # frames should not be identical across time
    assert not np.array_equal(arr[0], arr[1])
    assert not np.array_equal(arr[1], arr[2])


def test_dry_run_link_frames_independent_of_other_links(tmp_path):
    """A link's dry-run frames must not depend on the other links' array sizes.

    Regression test for the shared-RNG coupling bug: the runner used one RNG drawn in
    (frame, link) order, so a link's synthesized values depended on how many elements the
    *earlier* links consumed. Changing one link's antenna count therefore silently changed
    a later link's data. With independent per-link RNG streams, a link's frames are a
    function only of (seed, link index, frame index, that link's own shape).
    """
    from e2e.environment.scenario_runner import ScenarioRunner
    from e2e.scenario import ArrayConfig

    sc = munich_isac_scenario()
    sc.num_frames = 3
    sc.frequency.num_freqs = 8
    radar_link = "car_radar"
    comm_link = "building_comm_tx__car_comm_rx"

    base = ScenarioRunner(sc, dry_run=True, seed=99).run(
        out_path=str(tmp_path / "base.pkl"), verbose=False
    )["links"]

    # Mutate ONLY the radar RX aperture (the first link). The comm link is enumerated
    # after the radar, so under the old shared-RNG scheme its values would shift.
    sc2 = munich_isac_scenario()
    sc2.num_frames = 3
    sc2.frequency.num_freqs = 8
    radar_node = next(n for n in sc2.nodes if n.role.name == "RADAR")
    radar_node.array = ArrayConfig(num_rows=8, num_cols=8,
                                   pattern=radar_node.array.pattern,
                                   polarization=radar_node.array.polarization)

    changed = ScenarioRunner(sc2, dry_run=True, seed=99).run(
        out_path=str(tmp_path / "changed.pkl"), verbose=False
    )["links"]

    # The radar link's shape changed (8x8 -> 64 rx) so its data is expected to differ.
    assert base[radar_link].shape[1] != changed[radar_link].shape[1]
    # The comm link, however, must be byte-for-byte identical: independent of link A.
    np.testing.assert_array_equal(base[comm_link], changed[comm_link])


def test_dry_run_is_deterministic_with_seed(tmp_path, small_scenario):
    from e2e.environment.scenario_runner import ScenarioRunner

    a = ScenarioRunner(small_scenario, dry_run=True, seed=123).run(
        out_path=str(tmp_path / "a.pkl"), verbose=False
    )
    b = ScenarioRunner(small_scenario, dry_run=True, seed=123).run(
        out_path=str(tmp_path / "b.pkl"), verbose=False
    )
    assert a["meta"] == b["meta"]
    for name in a["links"]:
        np.testing.assert_array_equal(a["links"][name], b["links"][name])


# --------------------------------------------------------------------------- new reference scenarios

def test_dry_run_etoile_radar(tmp_path):
    from e2e.environment.scenario_runner import ScenarioRunner

    sc = etoile_radar_scenario()
    sc.num_frames = 3
    sc.frequency.num_freqs = 64
    runner = ScenarioRunner(sc, dry_run=True)
    out = tmp_path / "etoile_radar.pkl"
    payload = runner.run(out_path=str(out), verbose=False)
    arr = payload["links"][runner.primary_link.name]
    assert arr.shape == (3, 1024, 1, 1, 64)
    assert arr.dtype == np.complex64


def test_dry_run_munich_patrol(tmp_path):
    from e2e.environment.scenario_runner import ScenarioRunner, build_schedule

    sc = munich_patrol_scenario()
    sc.num_frames = 3
    sc.frequency.num_freqs = 64
    runner = ScenarioRunner(sc, dry_run=True)
    out = tmp_path / "munich_patrol.pkl"
    payload = runner.run(out_path=str(out), verbose=False)
    arr = payload["links"][runner.primary_link.name]
    assert arr.shape == (3, 1024, 1, 1, 64)
    assert arr.dtype == np.complex64

    # the radar's waypoint track actually moves it across frames (not held static)
    sched = build_schedule(sc)
    track = sched.node_tracks["patrol_radar"]
    assert track.shape == (3, 3)
    assert not np.allclose(track[0], track[1])
    assert not np.allclose(track[1], track[2])


@pytest.mark.slow
def test_cli_subprocess_dry_run_etoile_radar(tmp_path):
    """FULL 5000-freq-bin reference scenario via subprocess -- slow, RUN_SLOW=1."""
    out = tmp_path / "etoile_radar.pkl"
    proc = subprocess.run(
        [
            sys.executable, "-m", "e2e.environment.scenario_runner",
            "--scenario", "etoile_radar",
            "--frames", "2",
            "--dry-run",
            "--out", str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.is_file()
    with open(out, "rb") as f:
        payload = pickle.load(f)
    assert set(payload.keys()) == {"meta", "links"}
    arr = next(iter(payload["links"].values()))
    assert arr.shape == (2, 1024, 1, 1, 5000)
    assert arr.dtype == np.complex64


@pytest.mark.slow
def test_cli_subprocess_dry_run_munich_patrol(tmp_path):
    """FULL 5000-freq-bin reference scenario via subprocess -- slow, RUN_SLOW=1."""
    out = tmp_path / "munich_patrol.pkl"
    proc = subprocess.run(
        [
            sys.executable, "-m", "e2e.environment.scenario_runner",
            "--scenario", "munich_patrol",
            "--frames", "2",
            "--dry-run",
            "--out", str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.is_file()
    with open(out, "rb") as f:
        payload = pickle.load(f)
    assert set(payload.keys()) == {"meta", "links"}
    arr = next(iter(payload["links"].values()))
    assert arr.shape == (2, 1024, 1, 1, 5000)
    assert arr.dtype == np.complex64


# --------------------------------------------------------------------------- look_at + motion contract

def test_isac_has_moving_or_aimed_devices_for_real_path_orientation():
    """Pin the precondition for the real-path per-frame look_at re-aim (Finding #3).

    The orientation fix lives in the real Sionna path (_real_frame re-applies each
    node's static look_at every frame so a *moving* device keeps pointing at its
    target). That path cannot be exercised without Sionna/GPU, so here we only assert
    the structural contract the fix relies on: the ISAC reference scenario contains at
    least one node that has a look_at target, and at least one node that is moving --
    i.e. the configuration the per-frame re-aim is meant to handle. The behavioural
    correctness of the re-aim itself needs GPU validation (see @pytest.mark.sionna).
    """
    sc = munich_isac_scenario()
    by_name = {n.name: n for n in sc.nodes}
    assert by_name["building_comm_tx"].look_at is not None
    assert not by_name["car_comm_rx"].motion.is_static
    # at least one node both moves and (for the moving radar) is monostatic; ensure the
    # scenario exercises motion alongside an aimed device.
    assert any(n.look_at is not None for n in sc.nodes)
    assert any(not n.motion.is_static for n in sc.nodes)


# --------------------------------------------------------------------------- physical tx_power_dbm scaling

def test_legacy_none_power_matches_old_1_over_dist_mock(tmp_path):
    """tx_power_dbm=None keeps the legacy 1/dist mock (no amplitude scaling).

    munich_radar_scenario() ships with tx_power_dbm=12.0 on its radar node; strip it
    back to None to reconstruct the pre-physical-scaling contract, then check the
    dry-run statistics match the old mock exactly: cfr = (randn + j*randn)/dist, and for
    a monostatic radar dist = 0 + 1.0 (the co-location floor), so
    E|H| = E|standard complex gaussian magnitude| = sqrt(pi/2) ~= 1.2533.
    """
    from e2e.environment.scenario_runner import ScenarioRunner

    sc = munich_radar_scenario()
    sc.nodes[0].tx_power_dbm = None
    sc.num_frames = 4
    sc.frequency.num_freqs = 64
    runner = ScenarioRunner(sc, dry_run=True, seed=5)
    out = tmp_path / "legacy.pkl"
    payload = runner.run(out_path=str(out), verbose=False)
    arr = payload["links"][runner.primary_link.name]
    assert payload["meta"]["links"][runner.primary_link.name]["physical_scale"] is False

    assert arr.dtype == np.complex64
    mean_mag = float(np.mean(np.abs(arr)))
    # dist == 1.0 for a co-located monostatic radar -> no attenuation beyond the Rayleigh mean.
    np.testing.assert_allclose(mean_mag, np.sqrt(np.pi / 2), rtol=0.1)


def test_legacy_none_power_byte_identical_to_pre_change_formula(tmp_path):
    """Directly pins the legacy mock formula: cfr == (randn + j*randn)/dist, seed-for-seed."""
    from e2e.environment.scenario_runner import ScenarioRunner

    sc = munich_radar_scenario()
    sc.nodes[0].tx_power_dbm = None
    sc.num_frames = 2
    sc.frequency.num_freqs = 8
    runner = ScenarioRunner(sc, dry_run=True, seed=41)
    payload = runner.run(out_path=str(tmp_path / "legacy2.pkl"), verbose=False)
    arr = payload["links"][runner.primary_link.name]

    link = runner.links[0]
    rng = np.random.default_rng(41 + 0)  # link index 0, matches ScenarioRunner's per-link seeding
    shape = runner.frame_shape
    expected_frames = []
    for frame_idx in range(sc.num_frames):
        tx_pos = runner.schedule.node_position(link.tx_node.name, frame_idx)
        rx_pos = runner.schedule.node_position(link.rx_node.name, frame_idx)
        dist = float(np.linalg.norm(np.asarray(rx_pos) - np.asarray(tx_pos))) + 1.0
        real = rng.standard_normal(shape)
        imag = rng.standard_normal(shape)
        cfr = (real + 1j * imag).astype(np.complex64)
        cfr *= np.complex64(1.0 / dist)
        expected_frames.append(cfr)
    expected = np.stack(expected_frames, axis=0)
    np.testing.assert_array_equal(arr, expected)


def test_link_budget_round_trip_munich_radar(tmp_path):
    """THE key validation: dry-run munich_radar's 12 dBm physical power round-trips to
    the expected received power via an independent (test-side) link-budget formula.
    """
    from e2e.environment.scenario_runner import ScenarioRunner

    sc = munich_radar_scenario()
    sc.num_frames = 2
    sc.frequency.num_freqs = 64
    tx_power_dbm = sc.nodes[0].tx_power_dbm
    assert tx_power_dbm is not None

    runner = ScenarioRunner(sc, dry_run=True, seed=41)
    out = tmp_path / "munich_radar_budget.pkl"
    runner.run(out_path=str(out), verbose=False)

    with open(out, "rb") as f:
        payload = pickle.load(f)
    link_meta = payload["meta"]["links"][runner.primary_link.name]
    assert link_meta["physical_scale"] is True
    assert link_meta["tx_power_dbm"] == tx_power_dbm
    frames = payload["links"][runner.primary_link.name]  # (2, 1024, 1, 1, 64)

    # independent re-implementation of the expected link budget (not calling any runner
    # helper): P_rx = P_tx * (lambda_c / (4*pi*d))^2, V_rms^2 = P_rx * Z0.
    p_tx_w = 10.0 ** ((tx_power_dbm - 30.0) / 10.0)
    lambda_c = _SPEED_OF_LIGHT / sc.frequency.carrier_hz
    d = 1.0  # co-located monostatic radar: dist = 0 + the 1.0 floor
    fspl_power = (lambda_c / (4.0 * np.pi * d)) ** 2
    expected_mean_power = p_tx_w * SYSTEM_IMPEDANCE_OHMS * fspl_power

    # time-domain samples via ifft along the frequency axis; average |.|^2 over every
    # element/sample (1024 rx ant x 64 freq bins x 2 frames is a tight Rayleigh average).
    time_domain = np.fft.ifft(frames, axis=-1)
    measured_mean_power = float(np.mean(np.abs(time_domain) ** 2))

    np.testing.assert_allclose(measured_mean_power, expected_mean_power, rtol=0.1)


def test_physical_mock_magnitude_scales_as_1_over_dist(tmp_path):
    """A comm link with a physical tx_power_dbm: mean|frame|^2 matches N*P_tx*Z0*FSPL."""
    from e2e.environment.scenario_runner import ScenarioRunner

    tx_power_dbm = 12.0
    n_freqs = 64
    sc = Scenario(
        name="two_node_comm",
        base_scene="munich",
        frequency=FrequencyPlan(carrier_hz=30e9, start_hz=28.5e9, stop_hz=31.5e9, num_freqs=n_freqs),
        num_frames=3,
        nodes=[
            Node(name="tx", role=NodeRole.COMM_TX, position=(0.0, 0.0, 0.0),
                 array=ArrayConfig(num_rows=1, num_cols=1), tx_power_dbm=tx_power_dbm),
            Node(name="rx", role=NodeRole.COMM_RX, position=(100.0, 0.0, 0.0),
                 array=ArrayConfig(num_rows=2, num_cols=2)),
        ],
    )
    runner = ScenarioRunner(sc, dry_run=True, seed=41)
    payload = runner.run(out_path=str(tmp_path / "two_node_comm.pkl"), verbose=False)
    arr = payload["links"][runner.primary_link.name]
    assert payload["meta"]["links"][runner.primary_link.name]["physical_scale"] is True

    d = 101.0  # norm((100,0,0)) + the 1.0 floor
    lambda_c = _SPEED_OF_LIGHT / sc.frequency.carrier_hz
    fspl_power = (lambda_c / (4.0 * np.pi * d)) ** 2
    p_tx_w = 10.0 ** ((tx_power_dbm - 30.0) / 10.0)
    expected_mean_sq = n_freqs * p_tx_w * SYSTEM_IMPEDANCE_OHMS * fspl_power

    measured_mean_sq = float(np.mean(np.abs(arr) ** 2))
    np.testing.assert_allclose(measured_mean_sq, expected_mean_sq, rtol=0.1)


def test_mixed_scenario_per_link_independence(tmp_path):
    """One link physical, one legacy, in the same multi-link scenario: independent statistics."""
    from e2e.environment.scenario_runner import ScenarioRunner

    n_freqs = 64
    tx_power_dbm = 12.0
    sc = Scenario(
        name="mixed_isac",
        base_scene="munich",
        frequency=FrequencyPlan(carrier_hz=30e9, start_hz=28.5e9, stop_hz=31.5e9, num_freqs=n_freqs),
        num_frames=4,
        nodes=[
            Node(name="radar", role=NodeRole.RADAR, position=(0.0, 0.0, 0.0),
                 array=ArrayConfig(num_rows=4, num_cols=4), tx_power_dbm=None),
            Node(name="comm_tx", role=NodeRole.COMM_TX, position=(0.0, 0.0, 0.0),
                 array=ArrayConfig(num_rows=1, num_cols=1), tx_power_dbm=tx_power_dbm),
            Node(name="comm_rx", role=NodeRole.COMM_RX, position=(100.0, 0.0, 0.0),
                 array=ArrayConfig(num_rows=2, num_cols=2)),
        ],
    )
    runner = ScenarioRunner(sc, dry_run=True, seed=41)
    payload = runner.run(out_path=str(tmp_path / "mixed_isac.pkl"), verbose=False)

    assert runner.is_multilink
    links = payload["links"]
    legacy_arr = links["radar"]
    physical_arr = links["comm_tx__comm_rx"]

    # meta correctness: per-link physical_scale/tx_power_dbm reflect each node's setting
    meta_links = payload["meta"]["links"]
    assert meta_links["radar"]["physical_scale"] is False
    assert meta_links["radar"]["tx_power_dbm"] is None
    assert meta_links["radar"]["kind"] == "radar"
    assert meta_links["comm_tx__comm_rx"]["physical_scale"] is True
    assert meta_links["comm_tx__comm_rx"]["tx_power_dbm"] == tx_power_dbm
    assert meta_links["comm_tx__comm_rx"]["kind"] == "comm"

    # legacy link: mean|H| ~= sqrt(pi/2) / dist, dist == 1.0 (co-located monostatic radar)
    legacy_mean_mag = float(np.mean(np.abs(legacy_arr)))
    np.testing.assert_allclose(legacy_mean_mag, np.sqrt(np.pi / 2), rtol=0.1)

    # physical link: mean|H|^2 matches N*P_tx*Z0*FSPL(d=101)
    d = 101.0
    lambda_c = _SPEED_OF_LIGHT / sc.frequency.carrier_hz
    fspl_power = (lambda_c / (4.0 * np.pi * d)) ** 2
    p_tx_w = 10.0 ** ((tx_power_dbm - 30.0) / 10.0)
    expected_mean_sq = n_freqs * p_tx_w * SYSTEM_IMPEDANCE_OHMS * fspl_power
    physical_mean_sq = float(np.mean(np.abs(physical_arr) ** 2))
    np.testing.assert_allclose(physical_mean_sq, expected_mean_sq, rtol=0.1)


def test_full_tx_array_does_not_inflate_total_radiated_power(tmp_path):
    """DEFECT 2 repro (inverted): tx_power_dbm is the tx aperture's TOTAL radiated
    power. A radar opting into a full (multi-element) tx aperture must produce the same
    summed channel power (summed over the tx-antenna axis) as the default single-element
    tx, at the same tx_power_dbm -- not n_tx_ant times as much (+10*log10(n_tx_ant) dB
    EIRP), which is what the pre-fix formula (each element radiating the FULL P_tx) gave.
    """
    from e2e.environment.scenario_runner import ScenarioRunner

    def _make(tx_array_full: bool) -> Scenario:
        params = {"tx_array": "full"} if tx_array_full else {}
        return Scenario(
            name="tx_power_repro",
            base_scene="munich",
            frequency=FrequencyPlan(carrier_hz=30e9, start_hz=28.5e9, stop_hz=31.5e9,
                                     num_freqs=256),
            num_frames=25,
            nodes=[
                Node(name="radar", role=NodeRole.RADAR, position=(0.0, 0.0, 0.0),
                     array=ArrayConfig(num_rows=4, num_cols=4), tx_power_dbm=12.0,
                     params=params),
            ],
        )

    sc_single = _make(False)  # default radar tx: 1x1 effective aperture
    sc_full = _make(True)     # opts into a full 4x4 = 16-element tx aperture

    r_single = ScenarioRunner(sc_single, dry_run=True, seed=41)
    r_full = ScenarioRunner(sc_full, dry_run=True, seed=41)
    assert r_single.n_tx_ant == 1
    assert r_full.n_tx_ant == 16

    single = r_single.run(out_path=str(tmp_path / "single.pkl"), verbose=False)
    single_arr = single["links"]["radar"]  # (frames, 16 rx_ant, 1 tx_ant, 1, 256)
    full = r_full.run(out_path=str(tmp_path / "full.pkl"), verbose=False)
    full_arr = full["links"]["radar"]      # (frames, 16 rx_ant, 16 tx_ant, 1, 256)

    # total radiated power ~ sum over the tx-antenna axis; average over the rest (rx
    # ant / time / freq / frame) is where each link's independent Rayleigh draws
    # average out.
    single_power = float(np.mean(np.sum(np.abs(single_arr) ** 2, axis=2)))
    full_power = float(np.mean(np.sum(np.abs(full_arr) ** 2, axis=2)))
    np.testing.assert_allclose(full_power, single_power, rtol=0.1)


def test_tx_power_amplitude_scale_divides_by_sqrt_n_tx_ant():
    from e2e.environment.scenario_runner import _tx_power_amplitude_scale

    n_freqs, tx_power_dbm = 1000, 12.0
    p_tx_w = 10.0 ** ((tx_power_dbm - 30.0) / 10.0)
    base = np.sqrt(n_freqs * p_tx_w * SYSTEM_IMPEDANCE_OHMS)

    # n_tx_ant defaults to 1 -- byte-identical to the pre-fix single-element formula.
    np.testing.assert_allclose(
        _tx_power_amplitude_scale(tx_power_dbm, n_freqs), base
    )
    np.testing.assert_allclose(
        _tx_power_amplitude_scale(tx_power_dbm, n_freqs, n_tx_ant=1), base
    )
    np.testing.assert_allclose(
        _tx_power_amplitude_scale(tx_power_dbm, n_freqs, n_tx_ant=16),
        base / np.sqrt(16),
    )
    # legacy (no physical power) is still a no-op regardless of n_tx_ant.
    assert _tx_power_amplitude_scale(None, n_freqs, n_tx_ant=16) == 1.0


def test_cfr_should_normalize_selects_legacy_vs_physical():
    """Pure helper the real Sionna path uses to pick paths.cfr(normalize=...).

    Cannot exercise the real Sionna call here (no Sionna/GPU), so this pins the
    selection logic directly: normalize=True (legacy, unit-average-energy) iff
    tx_power_dbm is None; normalize=False (Sionna's own default, physical scaling)
    when a physical power is set.
    """
    from e2e.environment.scenario_runner import _cfr_should_normalize

    assert _cfr_should_normalize(None) is True
    assert _cfr_should_normalize(12.0) is False
    assert _cfr_should_normalize(0.0) is False


def test_tx_power_amplitude_scale_matches_derivation():
    """A = sqrt(N * P_tx * Z0); 1.0 (no-op) when tx_power_dbm is None."""
    from e2e.environment.scenario_runner import _tx_power_amplitude_scale

    assert _tx_power_amplitude_scale(None, 1024) == 1.0

    n_freqs = 1000
    tx_power_dbm = 12.0
    p_tx_w = 10.0 ** ((tx_power_dbm - 30.0) / 10.0)
    expected = np.sqrt(n_freqs * p_tx_w * SYSTEM_IMPEDANCE_OHMS)
    np.testing.assert_allclose(_tx_power_amplitude_scale(tx_power_dbm, n_freqs), expected)


# --------------------------------------------------------------------------- dual-pol guard

def test_runner_rejects_dual_pol_scenario_in_dry_run():
    """Scenario.validate() rejects VH/CROSS (see test_scenario.py), and ScenarioRunner
    calls validate() before touching anything else -- so this is never reachable via the
    normal entry point. Pin it anyway: dual-pol must fail at setup, in dry-run (no
    Sionna needed), not silently generate a misdescribed frame."""
    from e2e.environment.scenario_runner import ScenarioRunner
    from e2e.scenario import Polarization

    sc = munich_radar_scenario()
    sc.num_frames = 2
    sc.frequency.num_freqs = 8
    sc.nodes[0].array.polarization = Polarization.VH
    with pytest.raises(ValueError, match="invalid scenario"):
        ScenarioRunner(sc, dry_run=True)


def test_assert_single_pol_helper_rejects_dual_pol_directly():
    """Direct unit test of the runner's second, Sionna-free line of defense (used if a
    Scenario is ever mutated/constructed without going through validate())."""
    from e2e.environment.scenario_runner import _assert_single_pol
    from e2e.scenario import ArrayConfig, Polarization

    _assert_single_pol(ArrayConfig(num_rows=2, num_cols=2, polarization=Polarization.V), "ctx")
    _assert_single_pol(ArrayConfig(num_rows=2, num_cols=2, polarization=Polarization.H), "ctx")
    for bad in (Polarization.VH, Polarization.CROSS):
        with pytest.raises(ValueError, match="dual-polarization"):
            _assert_single_pol(ArrayConfig(num_rows=2, num_cols=2, polarization=bad), "ctx")


# --------------------------------------------------------------------------- validation guard

def test_runner_rejects_invalid_scenario():
    from e2e.environment.scenario_runner import ScenarioRunner
    from e2e.scenario import Scenario, Node, NodeRole

    bad = Scenario(name="bad", nodes=[Node(name="tx", role=NodeRole.COMM_TX, position=(0, 0, 0))])
    with pytest.raises(ValueError):
        ScenarioRunner(bad, dry_run=True)


# --------------------------------------------------------------------------- CLI

def test_cli_main_api_dry_run(tmp_path):
    from e2e.environment.scenario_runner import main

    out = tmp_path / "cli.pkl"
    rc = main([
        "--scenario", "munich_radar",
        "--frames", "2",
        "--dry-run",
        "--out", str(out),
    ])
    assert rc == 0
    assert out.is_file()
    with open(out, "rb") as f:
        payload = pickle.load(f)
    assert set(payload.keys()) == {"meta", "links"}
    arr = next(iter(payload["links"].values()))
    assert arr.shape == (2, 1024, 1, 1, 5000)
    assert arr.dtype == np.complex64


def test_cli_subprocess_dry_run(tmp_path):
    """Fast CLI-subprocess plumbing check: a shrunk scenario dumped to JSON and passed
    as ``--scenario <path>`` (num_freqs/num_frames trimmed), instead of a full 5000-bin
    named reference scenario. Exercises the same subprocess/argv/JSON-loading path as
    the full-size scenarios without paying for an ~80 MB array each run; those remain
    covered by the @pytest.mark.slow variants below (RUN_SLOW=1).
    """
    sc = munich_radar_scenario()
    sc.num_frames = 2
    sc.frequency.num_freqs = 8
    spec = tmp_path / "scene.json"
    sc.save(str(spec))

    out = tmp_path / "sub.pkl"
    proc = subprocess.run(
        [
            sys.executable, "-m", "e2e.environment.scenario_runner",
            "--scenario", str(spec),
            "--dry-run",
            "--out", str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.is_file()
    with open(out, "rb") as f:
        payload = pickle.load(f)
    assert set(payload.keys()) == {"meta", "links"}
    arr = next(iter(payload["links"].values()))
    assert arr.shape == (2, 1024, 1, 1, 8)
    assert arr.dtype == np.complex64


@pytest.mark.slow
def test_cli_subprocess_dry_run_full_reference(tmp_path):
    """FULL 5000-freq-bin munich_radar reference scenario via subprocess -- slow, RUN_SLOW=1."""
    out = tmp_path / "sub.pkl"
    proc = subprocess.run(
        [
            sys.executable, "-m", "e2e.environment.scenario_runner",
            "--scenario", "munich_radar",
            "--frames", "2",
            "--dry-run",
            "--out", str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.is_file()
    with open(out, "rb") as f:
        payload = pickle.load(f)
    assert set(payload.keys()) == {"meta", "links"}
    arr = next(iter(payload["links"].values()))
    assert arr.shape == (2, 1024, 1, 1, 5000)
    assert arr.dtype == np.complex64


def test_cli_loads_scenario_from_json_path(tmp_path):
    from e2e.environment.scenario_runner import main

    sc = munich_radar_scenario()
    sc.num_frames = 2
    sc.frequency.num_freqs = 16
    spec = tmp_path / "scene.json"
    sc.save(str(spec))

    out = tmp_path / "fromjson.pkl"
    rc = main(["--scenario", str(spec), "--dry-run", "--out", str(out)])
    assert rc == 0
    with open(out, "rb") as f:
        payload = pickle.load(f)
    assert payload["meta"]["scenario_name"] == sc.name
    arr = next(iter(payload["links"].values()))
    assert arr.shape == (2, 1024, 1, 1, 16)


# --------------------------------------------------------------------------- real Sionna path (gated)

@pytest.mark.sionna
def test_real_sionna_generation(tmp_path):
    """Exercises the REAL Sionna ray-tracing path. Skipped unless RUN_SIONNA=1."""
    from e2e.environment.scenario_runner import ScenarioRunner

    sc = munich_radar_scenario()
    sc.num_frames = 2
    sc.frequency.num_freqs = 16
    runner = ScenarioRunner(sc, dry_run=False)
    out = tmp_path / "real.pkl"
    payload = runner.run(out_path=str(out), verbose=False)
    assert set(payload.keys()) == {"meta", "links"}
    arr = payload["links"][runner.primary_link.name]
    assert arr.shape == (2, 1024, 1, 1, 16)
    assert arr.dtype == np.complex64
    # Finiteness guard: a monostatic link places tx and rx at identical coordinates;
    # if Sionna ever emitted a degenerate zero-length-LOS contribution (1/d
    # singularity), NaN/Inf would corrupt the whole frame silently.
    assert np.isfinite(arr).all()
