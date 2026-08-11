"""Tests for `e2e.ml.stats_report`: corpus statistical-validation reporting.

`e2e.ml.dataset`/`labels`/`scenes` are sibling shards -- if any hasn't landed in the
working tree yet, this whole module skips cleanly via the `importorskip` calls below
(mirrors `tests/test_ml_dataset.py`'s pattern).
"""
import dataclasses
import json
import math
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
labels = pytest.importorskip("e2e.ml.labels", reason="sibling shard e2e.ml.labels not present")
scenes = pytest.importorskip("e2e.ml.scenes", reason="sibling shard e2e.ml.scenes not present")

from e2e.ml import dataset as ml_dataset
from e2e.ml import stats_report
from e2e.ml.radar_config import PRESETS, TI_IWR1443


@pytest.fixture
def tiny_cfg():
    """A TDM-MIMO config shrunk for fast tests (matches test_ml_dataset.py's pattern), but
    with a larger n_samples than that module's tiny_cfg: D1 places up to 2 vehicles + 1
    pedestrian with min_target_separation_m=3.0, which needs enough max_range_m
    (~n_samples * range_resolution_m) headroom in the [0.1, 0.85]*max_range FOV band to
    reject-sample successfully -- n_samples=64 (test_ml_dataset.py's D0-only value) is
    too small and starves the placement sampler."""
    return dataclasses.replace(TI_IWR1443, name="test_tiny_stats", n_chirps=12, n_samples=256)


@pytest.fixture
def registered_tiny_cfg(tiny_cfg, monkeypatch):
    monkeypatch.setitem(PRESETS, tiny_cfg.name, tiny_cfg)
    return tiny_cfg


def _make_corpus(tmp_path, cfg, tier, n, torch_device, seed=0, **kwargs):
    return ml_dataset.generate_dataset(
        cfg.name, tier, n, out_dir=tmp_path, seed=seed, device=torch_device, **kwargs
    )


# --------------------------------------------------------------------------------
# _radial_velocity geometry sanity
# --------------------------------------------------------------------------------
def test_radial_velocity_dead_ahead_and_broadside():
    # sin_az=0 -> target dead ahead (+x): radial speed is purely vx.
    assert stats_report._radial_velocity(0.0, [5.0, 3.0, 0.0]) == pytest.approx(5.0)
    # sin_az=+-1 -> target at broadside (+-y): radial speed is purely +-vy.
    assert stats_report._radial_velocity(1.0, [5.0, 3.0, 0.0]) == pytest.approx(3.0)
    assert stats_report._radial_velocity(-1.0, [5.0, 3.0, 0.0]) == pytest.approx(-3.0)


# --------------------------------------------------------------------------------
# collect_stats: schema + sane values
# --------------------------------------------------------------------------------
def test_collect_stats_schema_and_sane_values(tmp_path, registered_tiny_cfg, torch_device):
    cfg = registered_tiny_cfg
    manifest_path = _make_corpus(tmp_path, cfg, "D1", 10, torch_device, seed=0)

    stats = stats_report.collect_stats(manifest_path)

    for key in ("manifest_path", "config_name", "tier", "n_frames_total", "grid",
                "max_velocity_mps", "clamp_cap_mps", "splits", "overall", "flags"):
        assert key in stats
    assert stats["config_name"] == cfg.name
    assert stats["tier"] == "D1"
    assert stats["n_frames_total"] == 10
    assert stats["clamp_cap_mps"] == pytest.approx(stats_report.CLAMP_FRAC * cfg.max_velocity_mps)

    assert set(stats["splits"]) == {"train", "val", "test"}
    assert sum(s["frame_count"] for s in stats["splits"].values()) == 10

    overall = stats["overall"]
    for key in ("frame_count", "targets_per_frame", "class_counts", "vehicle_pedestrian_ratio",
                "clutter_per_frame", "range_hist_m", "sin_az_hist", "rcs_dbsm",
                "radial_velocity_mps", "clamp_saturation_rate", "footprint_overlap_fraction",
                "edge_target_fraction", "placement_attempts", "targets_scene_count_mismatches",
                "total_targets"):
        assert key in overall

    # D1 sampling bounds (n_vehicles 1-2, n_pedestrians 0-1) keep targets/frame in [1, 3].
    assert 0 < overall["targets_per_frame"]["mean"] <= 3
    assert overall["targets_per_frame"]["min"] >= 1
    # targets_in_grid count must equal scene's vehicle+pedestrian count for every frame
    # (design doc section 2.5) -- a nonzero mismatch count is a real defect signal.
    assert overall["targets_scene_count_mismatches"] == 0
    # No overlap/edge violations expected at the default grid (design doc sections 2.2/2.3).
    assert 0.0 <= overall["footprint_overlap_fraction"] <= 1.0
    assert math.isnan(overall["edge_target_fraction"]) or 0.0 <= overall["edge_target_fraction"] <= 1.0

    flags = stats["flags"]
    assert "clamp_saturation_flag" in flags
    assert "footprint_overlap_flag" in flags
    assert "zero_class_warnings" in flags
    assert isinstance(flags["zero_class_warnings"], list)

    # RCS values should be roughly near the vehicle/pedestrian defaults +- jitter
    # (D1's rcs_jitter_db=2.0) -- just sanity-bound, not exact.
    if "vehicle" in overall["rcs_dbsm"]:
        assert overall["rcs_dbsm"]["vehicle"]["n"] > 0


def test_collect_stats_zero_class_warning_for_d0(tmp_path, registered_tiny_cfg, torch_device):
    """D0 has 0 pedestrians by construction (design doc section 1a table) -- must be
    flagged, not silently absorbed into an overall average."""
    cfg = registered_tiny_cfg
    manifest_path = _make_corpus(tmp_path, cfg, "D0", 6, torch_device, seed=1)

    stats = stats_report.collect_stats(manifest_path)
    warnings = stats["flags"]["zero_class_warnings"]
    assert any("pedestrian" in w and "overall" in w for w in warnings)
    assert stats["overall"]["class_counts"]["pedestrian"] == 0
    assert stats["overall"]["class_counts"]["vehicle"] > 0
    # No pedestrians at all -> vehicle:pedestrian ratio is the documented "inf" sentinel.
    assert math.isinf(stats["overall"]["vehicle_pedestrian_ratio"])


def test_collect_stats_with_tensor_sample(tmp_path, registered_tiny_cfg, torch_device):
    cfg = registered_tiny_cfg
    manifest_path = _make_corpus(tmp_path, cfg, "D1", 10, torch_device, seed=2)

    stats = stats_report.collect_stats(manifest_path, tensor_sample_per_split=2)

    assert "tensor_sample" in stats
    assert set(stats["tensor_sample"]) == {"train", "val", "test"}
    for split, ts in stats["tensor_sample"].items():
        # n_sampled should be min(2, frames-in-split); each split here has >=1 frame.
        assert ts["n_sampled"] == min(2, stats["splits"][split]["frame_count"])
        if ts["n_sampled"] > 0:
            for section in ("input_magnitude", "dynamic_range_db", "peak_to_median_floor_db"):
                assert section in ts
                for stat_key in ("mean", "min", "max") if section != "input_magnitude" else ("mean_of_frame_means", "min", "max"):
                    assert stat_key in ts[section]
                    assert not math.isnan(ts[section][stat_key])


# --------------------------------------------------------------------------------
# compute_flags: hand-built dict trigger checks
# --------------------------------------------------------------------------------
def test_compute_flags_clamp_rate_triggers():
    below = stats_report.compute_flags({"clamp_saturation_rate": 0.05, "footprint_overlap_fraction": 0.0})
    above = stats_report.compute_flags({"clamp_saturation_rate": 0.20, "footprint_overlap_fraction": 0.0})
    at_threshold = stats_report.compute_flags(
        {"clamp_saturation_rate": stats_report.CLAMP_FLAG_THRESHOLD, "footprint_overlap_fraction": 0.0}
    )
    assert below["clamp_saturation_flag"] is False
    assert above["clamp_saturation_flag"] is True
    assert at_threshold["clamp_saturation_flag"] is False  # strictly-greater-than semantics


def test_compute_flags_overlap_triggers_on_any_nonzero():
    clean = stats_report.compute_flags({"clamp_saturation_rate": 0.0, "footprint_overlap_fraction": 0.0})
    dirty = stats_report.compute_flags({"clamp_saturation_rate": 0.0, "footprint_overlap_fraction": 0.01})
    assert clean["footprint_overlap_flag"] is False
    assert dirty["footprint_overlap_flag"] is True


def test_compute_flags_handles_nan_gracefully():
    flags = stats_report.compute_flags({"clamp_saturation_rate": float("nan"),
                                        "footprint_overlap_fraction": float("nan")})
    assert flags["clamp_saturation_flag"] is False
    assert flags["footprint_overlap_flag"] is False


# --------------------------------------------------------------------------------
# compare_tiers
# --------------------------------------------------------------------------------
def test_compare_tiers_table_shape_and_sorted(tmp_path, registered_tiny_cfg, torch_device):
    cfg = registered_tiny_cfg
    m_d1 = _make_corpus(tmp_path / "d1", cfg, "D1", 8, torch_device, seed=3)
    m_d0 = _make_corpus(tmp_path / "d0", cfg, "D0", 6, torch_device, seed=4)

    stats_d1 = stats_report.collect_stats(m_d1)
    stats_d0 = stats_report.collect_stats(m_d0)

    comparison = stats_report.compare_tiers([stats_d1, stats_d0])  # deliberately out of order
    assert "rows" in comparison and "monotonicity_notes" in comparison
    rows = comparison["rows"]
    assert len(rows) == 2
    assert [r["tier"] for r in rows] == ["D0", "D1"]  # sorted
    for r in rows:
        for key in ("tier", "config", "frame_count", "targets_per_frame_mean",
                    "clutter_per_frame_mean", "vehicle_pedestrian_ratio",
                    "clamp_saturation_rate", "footprint_overlap_fraction"):
            assert key in r
    assert set(comparison["monotonicity_notes"]) == set(stats_report._MONOTONIC_NON_DECREASING_KEYS)


def test_compare_tiers_monotonicity_violation_detected():
    """Hand-built two-tier stats where clutter/frame DECREASES tier to tier -- must be
    flagged as a monotonicity violation (design doc section 4's ladder expectation)."""
    def _stub(tier, targets_mean, clutter_mean):
        return {
            "tier": tier, "config_name": "c",
            "overall": {
                "targets_per_frame": {"mean": targets_mean},
                "clutter_per_frame": {"mean": clutter_mean},
                "vehicle_pedestrian_ratio": 1.0,
                "clamp_saturation_rate": 0.0,
                "footprint_overlap_fraction": 0.0,
                "frame_count": 10,
            },
        }
    comparison = stats_report.compare_tiers([_stub("D0", 1.0, 0.0), _stub("D1", 1.5, 2.0), _stub("D2", 1.2, 5.0)])
    violations = comparison["monotonicity_notes"]["targets_per_frame_mean"]
    assert len(violations) == 1
    assert "D1->D2" in violations[0]
    assert comparison["monotonicity_notes"]["clutter_per_frame_mean"] == []


# --------------------------------------------------------------------------------
# write_report / write_comparison_report
# --------------------------------------------------------------------------------
def test_write_report_creates_markdown_with_flag_section(tmp_path, registered_tiny_cfg, torch_device):
    cfg = registered_tiny_cfg
    manifest_path = _make_corpus(tmp_path, cfg, "D1", 8, torch_device, seed=5)
    stats = stats_report.collect_stats(manifest_path)

    out_path = tmp_path / "report.md"
    returned = stats_report.write_report(stats, out_path)
    assert returned == out_path
    assert out_path.is_file()

    text = out_path.read_text()
    assert "# Corpus Statistical Validation Report" in text
    assert "## Flags" in text
    assert "## Per-split summary" in text
    assert "## RCS distribution" in text
    assert "## Radial velocity / clamp saturation" in text
    assert "## Footprint overlap / edge fraction" in text
    assert cfg.name in text
    assert "D1" in text


def test_write_report_flags_clamp_saturation_visibly(tmp_path):
    """A hand-built stats dict with a saturated clamp rate must render a visible flag
    marker in the written markdown (not just in the returned dict)."""
    stats = {
        "manifest_path": "dummy.json", "config_name": "cfg", "tier": "D2",
        "n_frames_total": 5, "grid": {"n_range": 4, "n_azimuth": 4, "max_range_m": 10.0,
                                       "range_bin_m": 2.5, "az_bin": 0.5},
        "max_velocity_mps": 10.0, "clamp_cap_mps": 8.0,
        "splits": {
            "train": stats_report._aggregate([], 8.0),
        },
        "overall": stats_report._aggregate([], 8.0),
    }
    stats["overall"]["clamp_saturation_rate"] = 0.5  # well above CLAMP_FLAG_THRESHOLD
    stats["flags"] = stats_report.compute_flags(stats["overall"])
    stats["flags"]["zero_class_warnings"] = []

    out_path = tmp_path / "flagged.md"
    stats_report.write_report(stats, out_path)
    text = out_path.read_text()
    assert "[FLAG]" in text
    assert "0.5" in text


def test_write_comparison_report(tmp_path, registered_tiny_cfg, torch_device):
    cfg = registered_tiny_cfg
    m_d1 = _make_corpus(tmp_path / "d1", cfg, "D1", 6, torch_device, seed=6)
    m_d0 = _make_corpus(tmp_path / "d0", cfg, "D0", 6, torch_device, seed=7)
    comparison = stats_report.compare_tiers(
        [stats_report.collect_stats(m_d0), stats_report.collect_stats(m_d1)]
    )

    out_path = tmp_path / "combined.md"
    stats_report.write_comparison_report(comparison, out_path)
    text = out_path.read_text()
    assert "# Cross-Tier Corpus Comparison" in text
    assert "## Monotonicity notes" in text
    assert "D0" in text and "D1" in text


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------
def test_cli_help_exits_zero():
    proc = subprocess.run(
        [sys.executable, "-m", "e2e.ml.stats_report", "--help"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--manifest" in proc.stdout
    assert "--compare" in proc.stdout


def test_cli_missing_args_exits_nonzero():
    proc = subprocess.run(
        [sys.executable, "-m", "e2e.ml.stats_report"],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0


def test_cli_end_to_end_writes_report(tmp_path, registered_tiny_cfg, torch_device):
    cfg = registered_tiny_cfg
    manifest_path = _make_corpus(tmp_path, cfg, "D1", 6, torch_device, seed=8)
    out_path = tmp_path / "cli_report.md"

    rc = stats_report.main(["--manifest", str(manifest_path), "--out", str(out_path)])
    assert rc == 0
    assert out_path.is_file()
    text = out_path.read_text()
    assert "# Corpus Statistical Validation Report" in text


def test_cli_compare_end_to_end_writes_combined_report(tmp_path, registered_tiny_cfg, torch_device):
    cfg = registered_tiny_cfg
    m_d0 = _make_corpus(tmp_path / "d0", cfg, "D0", 6, torch_device, seed=9)
    m_d1 = _make_corpus(tmp_path / "d1", cfg, "D1", 6, torch_device, seed=10)
    out_path = tmp_path / "combined_cli.md"

    rc = stats_report.main(["--compare", str(m_d0), str(m_d1), "--out", str(out_path)])
    assert rc == 0
    assert out_path.is_file()
    text = out_path.read_text()
    assert "# Cross-Tier Corpus Comparison" in text
