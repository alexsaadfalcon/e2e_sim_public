"""Tests for the JSON scenario pack (e2e/environment/scenarios/*.json).

Validates every scenario file in the pack -- both the pre-existing reference scenarios
(munich_radar.json, munich_isac.json) and the new pack members -- loads, validates
clean, and round-trips through JSON. For the new pack members it also exercises a
shrunk dry-run generation (no Sionna needed) to check the generated payload shape/dtype.

Pure numpy/stdlib for the generic pack checks; dry-run generation is numpy-only too
(ScenarioRunner's mock path never imports Sionna or torch).
"""

import glob
import os

import numpy as np
import pytest

from e2e.scenario import Scenario

SCENARIOS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "e2e", "environment", "scenarios",
)

PACK_FILES = sorted(glob.glob(os.path.join(SCENARIOS_DIR, "*.json")))

NEW_PACK_MEMBERS = {"munich_dense_traffic", "munich_two_link_isac", "canyon_radar"}


def _pack_id(path):
    return os.path.splitext(os.path.basename(path))[0]


# --------------------------------------------------------------------------- sanity

def test_pack_directory_is_not_empty():
    assert PACK_FILES, f"no *.json scenario files found in {SCENARIOS_DIR}"


def test_new_pack_members_are_present():
    found = {_pack_id(p) for p in PACK_FILES}
    missing = NEW_PACK_MEMBERS - found
    assert not missing, f"expected new pack members missing from {SCENARIOS_DIR}: {missing}"


# --------------------------------------------------------------------------- generic pack validation

@pytest.mark.parametrize("path", PACK_FILES, ids=_pack_id)
def test_pack_member_loads_and_validates_clean(path):
    sc = Scenario.load(path)
    assert sc.validate() == []


@pytest.mark.parametrize("path", PACK_FILES, ids=_pack_id)
def test_pack_member_name_matches_filename(path):
    sc = Scenario.load(path)
    assert sc.name == _pack_id(path)


@pytest.mark.parametrize("path", PACK_FILES, ids=_pack_id)
def test_pack_member_json_roundtrip_is_stable(path):
    sc = Scenario.load(path)
    rebuilt = Scenario.from_json(sc.to_json())
    assert rebuilt.to_dict() == sc.to_dict()
    assert rebuilt == sc
    # round-trip is stable under repeated (de)serialization, not just a single pass
    rebuilt2 = Scenario.from_json(rebuilt.to_json())
    assert rebuilt2.to_dict() == sc.to_dict()


@pytest.mark.parametrize("path", PACK_FILES, ids=_pack_id)
def test_pack_member_is_valid_json_file(path):
    # Scenario.load already parses JSON via json.loads; this test pins the "no comments,
    # strict JSON" requirement independently of the dataclass loader.
    import json
    with open(path) as f:
        text = f.read()
    parsed = json.loads(text)  # raises if not valid JSON (e.g. trailing commas, comments)
    assert isinstance(parsed, dict)


# --------------------------------------------------------------------------- pack budget (lightweight)

@pytest.mark.parametrize("path", [p for p in PACK_FILES if _pack_id(p) in NEW_PACK_MEMBERS], ids=_pack_id)
def test_new_pack_members_are_lightweight(path):
    sc = Scenario.load(path)
    assert sc.num_frames <= 20, f"{sc.name}: num_frames={sc.num_frames} exceeds the <=20 pack budget"
    assert sc.frequency.num_freqs <= 1024, (
        f"{sc.name}: num_freqs={sc.frequency.num_freqs} exceeds the <=1024 pack budget"
    )


# --------------------------------------------------------------------------- dry-run generation (new members only)

def _dry_run(path, tmp_path, num_frames=2, num_freqs=64):
    from e2e.environment.scenario_runner import ScenarioRunner

    sc = Scenario.load(path)
    sc.num_frames = num_frames
    sc.frequency.num_freqs = num_freqs
    runner = ScenarioRunner(sc, dry_run=True)
    out = tmp_path / f"{sc.name}.pkl"
    payload = runner.run(out_path=str(out), verbose=False)
    return runner, payload, out


def test_dense_traffic_dry_run_generates_single_link(tmp_path):
    path = os.path.join(SCENARIOS_DIR, "munich_dense_traffic.json")
    runner, payload, out = _dry_run(path, tmp_path)

    assert out.is_file()
    assert not runner.is_multilink
    assert isinstance(payload, np.ndarray)
    assert payload.dtype == np.complex64
    # single 32x32 monostatic radar -> 1024 rx ant, 1x1 effective tx, 1 time step
    assert payload.shape == (2, 1024, 1, 1, 64)
    # 9 car objects with varied positions, radar has combined translation + rotation
    assert len(runner.scenario.objects) >= 8
    radar = runner.scenario.nodes[0]
    assert tuple(radar.motion.velocity) != (0.0, 0.0, 0.0)
    assert radar.motion.angular_velocity_deg != 0.0


def test_two_link_isac_dry_run_has_exactly_three_links(tmp_path):
    path = os.path.join(SCENARIOS_DIR, "munich_two_link_isac.json")
    runner, payload, out = _dry_run(path, tmp_path)

    assert out.is_file()
    assert runner.is_multilink
    assert isinstance(payload, dict)
    assert len(payload) == 3
    assert len(runner.links) == 3

    link_names = {link.name for link in runner.links}
    assert link_names == {
        "car_radar",
        "building_comm_tx__car_comm_rx",
        "building_comm_tx__pedestrian_comm_rx",
    }
    for arr in payload.values():
        assert arr.dtype == np.complex64
        assert arr.shape[0] == 2  # num_frames

    # the two comm links differ in rx aperture (4x4 == 16 vs 2x2 == 4 elements)
    assert payload["building_comm_tx__car_comm_rx"].shape[1] == 16
    assert payload["building_comm_tx__pedestrian_comm_rx"].shape[1] == 4


def test_canyon_radar_dry_run_generates_single_link(tmp_path):
    path = os.path.join(SCENARIOS_DIR, "canyon_radar.json")
    runner, payload, out = _dry_run(path, tmp_path)

    assert out.is_file()
    assert not runner.is_multilink
    assert isinstance(payload, np.ndarray)
    assert payload.dtype == np.complex64
    # 16x16 radar -> 256 rx ant, 1x1 effective tx
    assert payload.shape == (2, 256, 1, 1, 64)
    assert runner.scenario.base_scene == "simple_street_canyon"


# --------------------------------------------------------------------------- CLI dry-run for canyon (acceptance check)

def test_canyon_radar_cli_dry_run(tmp_path):
    """Mirrors the acceptance command: CLI dry-run on canyon_radar.json exits 0."""
    from e2e.environment.scenario_runner import main

    path = os.path.join(SCENARIOS_DIR, "canyon_radar.json")
    out = tmp_path / "canyon_cli.pkl"
    rc = main(["--scenario", path, "--dry-run", "--out", str(out)])
    assert rc == 0
    assert out.is_file()
