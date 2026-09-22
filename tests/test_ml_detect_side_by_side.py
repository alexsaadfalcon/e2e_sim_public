"""The side-by-side figure takes every number from the authority file, never from code."""

import json

import pytest

from e2e.ml import detect_side_by_side as sbs


def _authority(tmp_path):
    doc = {
        "manifest": "some/manifest.json", "split": "test",
        "arms": [
            {"name": "classical CFAR", "AP": 0.301, "max_range_m": 40.0,
             "operating_point": {"score_threshold": 0.661, "fp_per_frame": 6.24}},
            {"name": "raddetnet", "AP": 0.476, "checkpoint": "runs/b7/best.pt",
             "operating_point": {"score_threshold": 0.440, "fp_per_frame": 2.99}},
        ],
    }
    p = tmp_path / "beat_cfar.json"
    p.write_text(json.dumps(doc))
    return p


def test_arms_come_from_the_authority_file(tmp_path):
    arms, doc = sbs.load_arms(_authority(tmp_path), ["classical CFAR", "raddetnet"])
    assert [a["threshold"] for a in arms] == pytest.approx([0.661, 0.440])
    assert arms[0]["checkpoint"] is None and arms[1]["checkpoint"] == "runs/b7/best.pt"
    assert arms[1]["AP"] == 0.476 and doc["split"] == "test"


def test_unknown_arm_is_refused_not_guessed(tmp_path):
    with pytest.raises(SystemExit, match="not in"):
        sbs.load_arms(_authority(tmp_path), ["ssmradnet_rd_b5"])


def test_module_is_outside_the_pipeline_fingerprint():
    """A new file under e2e/ml must not move the fingerprint that certifies checkpoints
    (F84); this module only reads outputs and draws."""
    from e2e.ml.train import INPUT_PIPELINE_SOURCES
    assert not any("detect_side_by_side" in str(s) for s in INPUT_PIPELINE_SOURCES)
