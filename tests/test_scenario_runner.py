"""Tests for the offline scenario generation runner (e2e.environment.scenario_runner).

Only the dry-run / mock path is exercised by default -- it imports Sionna lazily, so the
module imports and dry-run generation work with no Sionna installed. Any test that would
invoke the real Sionna ray-tracing path is marked @pytest.mark.sionna (auto-skipped
unless RUN_SIONNA=1).

All output goes to tmp_path; we never write into e2e/environment/sionna_sims/.
"""

import pickle
import subprocess
import sys

import numpy as np
import pytest

from e2e.environment.sionna_iterator import SionnaIterator
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
    arr = runner.run(out_path=str(out), verbose=False)

    n_freqs = small_scenario.frequency.num_freqs
    # 32x32 radar -> 1024 rx ant, 1x1 effective tx, 1 time step
    expected = (small_scenario.num_frames, 1024, 1, 1, n_freqs)
    assert arr.shape == expected
    assert arr.dtype == np.complex64
    assert runner.frame_shape == (1024, 1, 1, n_freqs)

    # the dumped file matches the returned array
    assert out.is_file()
    with open(out, "rb") as f:
        on_disk = pickle.load(f)
    assert on_disk.shape == expected
    assert on_disk.dtype == np.complex64
    np.testing.assert_array_equal(on_disk, arr)


def test_dry_run_munich_radar_full_reference(tmp_path):
    from e2e.environment.scenario_runner import ScenarioRunner

    sc = munich_radar_scenario()
    sc.num_frames = 3
    sc.frequency.num_freqs = 16
    runner = ScenarioRunner(sc, dry_run=True)
    out = tmp_path / "munich_radar.pkl"
    arr = runner.run(out_path=str(out), verbose=False)
    assert arr.shape == (3, 1024, 1, 1, 16)
    assert arr.dtype == np.complex64


def test_dry_run_isac_is_multilink(tmp_path):
    """ISAC scenario exports one frame-stack per link (radar + comm), as a dict."""
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
    assert isinstance(payload, dict)
    assert payload["car_radar"].shape == (2, 1024, 1, 1, 8)              # radar link
    assert payload["building_comm_tx__car_comm_rx"].shape == (2, 16, 1, 1, 8)  # 4x4 comm rx
    for arr in payload.values():
        assert arr.dtype == np.complex64


def test_dry_run_isac_iterator_link_selection(tmp_path):
    """SionnaIterator picks a link from a multi-link dump (default = first)."""
    from e2e.environment.scenario_runner import ScenarioRunner

    sc = munich_isac_scenario()
    sc.num_frames = 2
    sc.frequency.num_freqs = 8
    out = tmp_path / "isac.pkl"
    ScenarioRunner(sc, dry_run=True).run(out_path=str(out), verbose=False)

    it = SionnaIterator(str(out))  # default link = first (radar)
    assert it.links == ["car_radar", "building_comm_tx__car_comm_rx"]
    assert it.link == "car_radar"
    assert np.asarray(it[0]).shape == (1024, 1, 1, 8)

    it_comm = SionnaIterator(str(out), link="building_comm_tx__car_comm_rx")
    assert len(it_comm) == 2
    assert np.asarray(it_comm[0]).shape == (16, 1, 1, 8)

    with pytest.raises(KeyError):
        SionnaIterator(str(out), link="does_not_exist")


def test_dry_run_loads_via_iterator_and_reshapes(tmp_path, small_scenario):
    from e2e.environment.scenario_runner import ScenarioRunner

    n_freqs = small_scenario.frequency.num_freqs
    out = tmp_path / "small.pkl"
    ScenarioRunner(small_scenario, dry_run=True).run(out_path=str(out), verbose=False)

    it = SionnaIterator(str(out))
    assert len(it) == small_scenario.num_frames

    frame = np.asarray(it[0])
    assert frame.shape == (1024, 1, 1, n_freqs)
    # the runtime pipeline views a single 32x32 radar frame as (32, 32, num_freqs)
    reshaped = frame.reshape(32, 32, n_freqs)
    assert reshaped.shape == (32, 32, n_freqs)


def test_dry_run_frames_differ_when_motion_present(tmp_path, small_scenario):
    from e2e.environment.scenario_runner import ScenarioRunner

    # small_scenario (munich_radar) has a moving radar (velocity (1,0,0)).
    assert any(not n.motion.is_static for n in small_scenario.nodes)

    runner = ScenarioRunner(small_scenario, dry_run=True, seed=3)
    arr = runner.run(out_path=str(tmp_path / "moving.pkl"), verbose=False)
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
    )

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
    )

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
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------- new reference scenarios

def test_dry_run_etoile_radar(tmp_path):
    from e2e.environment.scenario_runner import ScenarioRunner

    sc = etoile_radar_scenario()
    sc.num_frames = 3
    sc.frequency.num_freqs = 64
    runner = ScenarioRunner(sc, dry_run=True)
    out = tmp_path / "etoile_radar.pkl"
    arr = runner.run(out_path=str(out), verbose=False)
    assert arr.shape == (3, 1024, 1, 1, 64)
    assert arr.dtype == np.complex64


def test_dry_run_munich_patrol(tmp_path):
    from e2e.environment.scenario_runner import ScenarioRunner, build_schedule

    sc = munich_patrol_scenario()
    sc.num_frames = 3
    sc.frequency.num_freqs = 64
    runner = ScenarioRunner(sc, dry_run=True)
    out = tmp_path / "munich_patrol.pkl"
    arr = runner.run(out_path=str(out), verbose=False)
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
        arr = pickle.load(f)
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
        arr = pickle.load(f)
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
    arr = runner.run(out_path=str(out), verbose=False)

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
    arr = runner.run(out_path=str(tmp_path / "legacy2.pkl"), verbose=False)

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
        frames = pickle.load(f)  # (2, 1024, 1, 1, 64)

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
    arr = runner.run(out_path=str(tmp_path / "two_node_comm.pkl"), verbose=False)

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
    legacy_arr = payload["radar"]
    physical_arr = payload["comm_tx__comm_rx"]

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
        arr = pickle.load(f)
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
        arr = pickle.load(f)
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
        arr = pickle.load(f)
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
        arr = pickle.load(f)
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
    arr = runner.run(out_path=str(out), verbose=False)
    assert arr.shape == (2, 1024, 1, 1, 16)
    assert arr.dtype == np.complex64
    # Finiteness guard: a monostatic link places tx and rx at identical coordinates;
    # if Sionna ever emitted a degenerate zero-length-LOS contribution (1/d
    # singularity), NaN/Inf would corrupt the whole frame silently.
    assert np.isfinite(arr).all()
