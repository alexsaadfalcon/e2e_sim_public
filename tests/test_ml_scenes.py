"""Tests for e2e.ml.scenes (random radar scene sampler / difficulty tiers).

Pure numpy/stdlib -- fast, no Sionna/torch.
"""

import numpy as np
import pytest

from e2e.ml.radar_config import PRESETS, RADIAL_LIKE, TI_IWR1443
from e2e.ml.scenes import DIFFICULTY_TIERS, TierSpec, sample_scene, scene_summary
from e2e.scenario import Motion, Scenario

_EPS = 1e-6


def _positions_and_sin_az(scenario):
    """[(range, sin_az), ...] for every object, using the +x boresight / +y ULA
    convention `sample_scene` places targets in."""
    out = []
    for obj in scenario.objects:
        pos = np.asarray(obj.position, dtype=float)
        r = np.linalg.norm(pos)
        sin_az = pos[1] / r if r > 0 else 0.0
        out.append((r, sin_az))
    return out


# --------------------------------------------------------------------------- determinism

def test_sample_scene_deterministic_given_seeded_rng():
    cfg = TI_IWR1443
    sc1 = sample_scene(cfg, "D2", np.random.default_rng(1234))
    sc2 = sample_scene(cfg, "D2", np.random.default_rng(1234))
    assert sc1.to_json() == sc2.to_json()


def test_sample_scene_different_seeds_differ():
    cfg = TI_IWR1443
    sc1 = sample_scene(cfg, "D2", np.random.default_rng(1))
    sc2 = sample_scene(cfg, "D2", np.random.default_rng(2))
    assert sc1.to_json() != sc2.to_json()


# --------------------------------------------------------------------------- tier bounds

def test_d0_always_exactly_one_vehicle_zero_ped_zero_clutter():
    cfg = TI_IWR1443
    for seed in range(30):
        sc = sample_scene(cfg, "D0", np.random.default_rng(seed))
        summary = scene_summary(sc)
        assert summary["n_vehicles"] == 1
        assert summary["n_pedestrians"] == 0
        assert summary["n_clutter"] == 0


def test_d3_counts_within_spec_bounds():
    cfg = TI_IWR1443
    spec = DIFFICULTY_TIERS["D3"]
    for seed in range(30):
        sc = sample_scene(cfg, "D3", np.random.default_rng(seed))
        summary = scene_summary(sc)
        assert spec.n_vehicles[0] <= summary["n_vehicles"] <= spec.n_vehicles[1]
        assert spec.n_pedestrians[0] <= summary["n_pedestrians"] <= spec.n_pedestrians[1]
        assert spec.n_clutter[0] <= summary["n_clutter"] <= spec.n_clutter[1]


@pytest.mark.parametrize("tier", list(DIFFICULTY_TIERS))
def test_all_tiers_produce_valid_round_tripping_scenarios(tier):
    cfg = TI_IWR1443
    for seed in range(5):
        sc = sample_scene(cfg, tier, np.random.default_rng(seed))
        assert sc.validate() == []
        rebuilt = Scenario.from_json(sc.to_json())
        assert rebuilt.to_dict() == sc.to_dict()


# --------------------------------------------------------------------------- FOV bounds

@pytest.mark.parametrize("cfg", [TI_IWR1443, RADIAL_LIKE], ids=lambda c: c.name)
def test_targets_within_fov_bounds(cfg):
    for seed in range(20):
        sc = sample_scene(cfg, "D3", np.random.default_rng(seed))
        for r, sin_az in _positions_and_sin_az(sc):
            assert 0.1 * cfg.max_range_m - _EPS <= r <= 0.85 * cfg.max_range_m + _EPS
            assert -0.85 - _EPS <= sin_az <= 0.85 + _EPS


# --------------------------------------------------------------------------- radial velocity clamp

@pytest.mark.parametrize("cfg", [TI_IWR1443, RADIAL_LIKE], ids=lambda c: c.name)
def test_radial_velocity_clamp_holds(cfg):
    cap = 0.8 * cfg.max_velocity_mps
    for seed in range(20):
        sc = sample_scene(cfg, "D3", np.random.default_rng(seed))
        for obj in sc.objects:
            if obj.object_class not in ("vehicle", "pedestrian"):
                continue
            pos = np.asarray(obj.position, dtype=float)
            vel = np.asarray(obj.velocity_mps, dtype=float)
            r = np.linalg.norm(pos)
            e_los = pos / r if r > 0 else np.array([1.0, 0.0, 0.0])
            radial = float(np.dot(vel, e_los))
            assert abs(radial) <= cap + 1e-6


# --------------------------------------------------------------------------- separation

def test_min_separation_respected_in_d2():
    cfg = TI_IWR1443
    spec = DIFFICULTY_TIERS["D2"]
    for seed in range(30):
        sc = sample_scene(cfg, "D2", np.random.default_rng(seed))
        positions = [
            np.asarray(o.position, dtype=float)
            for o in sc.objects if o.object_class in ("vehicle", "pedestrian")
        ]
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                d = np.linalg.norm(positions[i] - positions[j])
                assert d >= spec.min_target_separation_m - 1e-6


# --------------------------------------------------------------------------- scene_summary

def test_scene_summary_counts_match_objects():
    cfg = TI_IWR1443
    sc = sample_scene(cfg, "D3", np.random.default_rng(7))
    summary = scene_summary(sc)
    classes = [o.object_class for o in sc.objects]
    assert "classes" not in summary  # dropped: redundant with the counts below
    assert summary["n_vehicles"] == classes.count("vehicle")
    assert summary["n_pedestrians"] == classes.count("pedestrian")
    assert summary["n_clutter"] == classes.count("scatterer")
    assert summary["n_vehicles"] + summary["n_pedestrians"] + summary["n_clutter"] == len(classes)


def test_scene_summary_clutter_position_and_rcs():
    cfg = TI_IWR1443
    sc = sample_scene(cfg, "D3", np.random.default_rng(11))
    summary = scene_summary(sc)
    clutter_objs = [o for o in sc.objects if o.object_class == "scatterer"]
    assert len(summary["clutter"]) == summary["n_clutter"] == len(clutter_objs)
    for entry, obj in zip(summary["clutter"], clutter_objs):
        assert entry["position"] == list(obj.position)
        assert entry["rcs_dbsm"] == obj.rcs_dbsm
        assert isinstance(entry["position"], list) and len(entry["position"]) == 3


# --------------------------------------------------------------------------- placement retries

def test_placement_attempts_recorded_and_json_safe():
    cfg = TI_IWR1443
    sc = sample_scene(cfg, "D3", np.random.default_rng(3))
    summary = scene_summary(sc)
    attempts = summary["placement_attempts"]
    n_placed = summary["n_vehicles"] + summary["n_pedestrians"]
    assert isinstance(attempts, int)
    # every placement uses >= 1 attempt (the successful one), so the total is at
    # least the number of vehicles/pedestrians placed under the separation constraint.
    assert attempts >= n_placed
    assert sc.metadata["placement_attempts"] == attempts


def test_placement_attempts_none_for_hand_built_scenario():
    sc = Scenario(name="hand_built")
    assert scene_summary(sc)["placement_attempts"] is None


# --------------------------------------------------------------------------- multi-frame motion

def test_sample_scene_single_frame_default_keeps_static_motion():
    cfg = TI_IWR1443
    sc = sample_scene(cfg, "D1", np.random.default_rng(20))
    assert sc.num_frames == 1
    for obj in sc.objects:
        if obj.object_class in ("vehicle", "pedestrian"):
            assert obj.motion.is_static  # single-instant sample: no Motion track needed


def test_sample_scene_n_frames_gives_moving_objects_a_track():
    cfg = TI_IWR1443
    n_frames = 5
    sc = sample_scene(cfg, "D1", np.random.default_rng(21), n_frames=n_frames)
    assert sc.num_frames == n_frames

    from e2e.environment.motion import resolve_motion

    dt = 1.0 / cfg.frame_rate_hz
    moving_checked = 0
    for obj in sc.objects:
        if obj.object_class not in ("vehicle", "pedestrian"):
            continue
        track = resolve_motion(obj.position, obj.motion, n_frames)
        assert track.shape == (n_frames, 3)
        vel = np.asarray(obj.velocity_mps, dtype=float)
        if np.linalg.norm(vel) < 1e-9:
            continue  # a sampled-zero-speed object legitimately has a static track
        moving_checked += 1
        # constant-velocity track: frame t sits at base + t * (vel * dt).
        expected = track[0] + np.arange(n_frames)[:, None] * (vel * dt)[None, :]
        assert np.allclose(track, expected, atol=1e-6)
    assert moving_checked > 0  # D1 with this seed has at least one genuinely moving object


def test_sample_scene_n_frames_one_matches_default():
    """`n_frames=1` (explicit) must be identical to the default (no motion track)."""
    cfg = TI_IWR1443
    sc1 = sample_scene(cfg, "D2", np.random.default_rng(4))
    sc2 = sample_scene(cfg, "D2", np.random.default_rng(4), n_frames=1)
    assert sc1.to_json() == sc2.to_json()


# --------------------------------------------------------------------------- misc

def test_unknown_tier_raises_key_error():
    with pytest.raises(KeyError):
        sample_scene(TI_IWR1443, "not_a_tier", np.random.default_rng(0))


def test_tier_spec_instance_accepted_directly():
    custom = TierSpec(
        name="custom", n_vehicles=(1, 1), n_pedestrians=(0, 0), n_clutter=(0, 0),
        vehicle_speed_mps=(0.0, 1.0), pedestrian_speed_mps=(0.0, 0.0),
        rcs_jitter_db=0.0, min_target_separation_m=1.0,
    )
    sc = sample_scene(TI_IWR1443, custom, np.random.default_rng(0))
    assert scene_summary(sc)["n_vehicles"] == 1


def test_presets_smoke():
    # sanity: both reference RadarConfig presets are usable scene scales.
    for cfg in PRESETS.values():
        sc = sample_scene(cfg, "D1", np.random.default_rng(0))
        assert sc.validate() == []
