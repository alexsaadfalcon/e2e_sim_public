"""Tests for `e2e.ml.bootstrap_ci` -- paired scene-level CIs on AP differences.

The intervals this module produces exist to answer "is that lead real". A bootstrap that
is subtly wrong does not fail loudly; it reports a confident, narrow, wrong interval. So
the tests below pin the estimator against `e2e.ml.metrics` itself, pin the resampling
UNIT (scene, not detection), and pin the refusal on artifacts that cannot support it.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from e2e.ml.bootstrap_ci import (MissingFrameProvenance, average_precision,
                                 paired_bootstrap, verify_against_stored)
from e2e.ml.labels import LabelGrid
from e2e.ml.metrics import evaluate_dataset


def _frames(n_frames: int, *, hit_prob: float, seed: int, n_per_frame: int = 4):
    """A synthetic split: `n_per_frame` targets per frame, a detector that finds each
    with probability `hit_prob`. Returns (pred_maps, target_lists, grid)."""
    rng = np.random.default_rng(seed)
    grid = LabelGrid(n_range=48, n_azimuth=24, max_range_m=96.0)
    preds, targets = [], []
    for _f in range(n_frames):
        pred = torch.zeros((3, grid.n_range, grid.n_azimuth))
        tl = []
        for k in range(n_per_frame):
            i = int(rng.integers(2, grid.n_range - 2))
            j = int(rng.integers(2, grid.n_azimuth - 2))
            r_m = (i + 0.5) * grid.range_bin_m
            sin_az = -1.0 + (j + 0.5) * grid.az_bin
            tl.append((r_m, sin_az, "vehicle", r_m))
            if rng.random() < hit_prob:
                pred[0, i, j] = float(rng.uniform(0.5, 1.0))
            else:                      # a false alarm somewhere else in this frame
                i2 = (i + 7) % (grid.n_range - 2) + 1
                pred[0, i2, j] = float(rng.uniform(0.1, 0.6))
        preds.append(pred)
        targets.append(tl)
    return preds, targets, grid


def _arm(name, preds, targets, grid):
    m = evaluate_dataset(preds, targets, grid, classes=())
    return {"name": name, "AP": m["AP"], "pr_curve": m["pr_curve"]}, m["gt_per_frame"]


def test_average_precision_matches_metrics_on_the_identity_resample():
    """THE oracle. The bootstrap's own AP must be the same estimator metrics computes --
    including the pessimistic tie-break -- or every interval is around the wrong thing."""
    preds, targets, grid = _frames(30, hit_prob=0.7, seed=0)
    arm, gt_per_frame = _arm("a", preds, targets, grid)
    rows = verify_against_stored([arm], gt_per_frame)
    name, stored, recomputed = rows[0]
    assert recomputed == pytest.approx(stored, abs=1e-12), (name, stored, recomputed)


def test_average_precision_honours_the_pessimistic_tie_break():
    """At equal scores metrics ranks false positives FIRST. A naive sort would rank the
    true positive first and inflate AP, which is exactly the ordering a saturating
    classical detector (many detections scored 1.0) would benefit from."""
    score = np.array([1.0, 1.0])
    optimistic = np.array([True, False])
    ap = average_precision(score, optimistic, n_gt=1.0)
    # TP ranked second => precision 1/2 at its rank, so AP = 0.5, not 1.0.
    assert ap == pytest.approx(0.5)


def test_a_clearly_better_arm_gets_an_interval_excluding_zero():
    preds_a, targets, grid = _frames(60, hit_prob=0.9, seed=1)
    preds_b, _t, _g = _frames(60, hit_prob=0.25, seed=1)
    arm_a, gt = _arm("good", preds_a, targets, grid)
    arm_b, _ = _arm("bad", preds_b, targets, grid)

    res = paired_bootstrap({"arms": [arm_a, arm_b], "gt_per_frame": gt},
                           baseline="good", n_boot=300, seed=0)
    comp = res["comparisons"][0]
    assert comp["arm"] == "bad" and comp["baseline"] == "good"
    assert comp["delta_AP"] < 0.0
    assert comp["ci_high"] < 0.0, f"a clearly worse arm must be significant: {comp}"
    assert comp["excludes_zero"] is True


def test_an_arm_compared_against_itself_straddles_zero():
    """A self-comparison has delta exactly 0 and, because the resample is PAIRED, an
    interval that collapses onto zero. An UNPAIRED bootstrap would give this a spuriously
    wide interval -- so this also pins that the pairing is real."""
    preds, targets, grid = _frames(40, hit_prob=0.6, seed=3)
    arm, gt = _arm("x", preds, targets, grid)
    twin = dict(arm, name="x_copy")

    res = paired_bootstrap({"arms": [arm, twin], "gt_per_frame": gt},
                           n_boot=200, seed=0)
    comp = res["comparisons"][0]
    assert comp["delta_AP"] == pytest.approx(0.0, abs=1e-12)
    assert comp["ci_low"] == pytest.approx(0.0, abs=1e-9)
    assert comp["ci_high"] == pytest.approx(0.0, abs=1e-9)
    assert comp["excludes_zero"] is False


def test_resampling_unit_is_the_scene_not_the_detection():
    """Drawing 4 copies of one frame must reproduce that frame's own AP, not something
    averaged over the split -- the signature of resampling scenes rather than pooled
    detections."""
    from e2e.ml.bootstrap_ci import _ArmResampler

    preds, targets, grid = _frames(5, hit_prob=0.6, seed=5)
    arm, gt = _arm("a", preds, targets, grid)
    r = _ArmResampler(arm, n_frames=len(gt))

    one = np.array([2, 2, 2, 2], dtype=np.int64)
    ap_repeated = r.ap_for(one, n_gt=4.0 * gt[2])
    ap_single = r.ap_for(np.array([2], dtype=np.int64), n_gt=float(gt[2]))
    assert ap_repeated == pytest.approx(ap_single, abs=1e-12)


def test_refuses_a_curve_without_frame_provenance():
    """The exact artifact that caused the withdrawal in 0efb7e4. It must refuse, not
    silently fall back to an iid bootstrap over pooled detections -- that interval would
    be too narrow, and too narrow is the dangerous direction."""
    preds, targets, grid = _frames(10, hit_prob=0.5, seed=7)
    arm, gt = _arm("a", preds, targets, grid)
    legacy = {"name": "legacy", "AP": arm["AP"],
              "pr_curve": {k: v for k, v in arm["pr_curve"].items() if k != "frame"}}
    with pytest.raises(MissingFrameProvenance, match="cannot be resampled by scene"):
        paired_bootstrap({"arms": [arm, legacy], "gt_per_frame": gt}, n_boot=10)


def test_refuses_when_gt_per_frame_is_absent():
    preds, targets, grid = _frames(8, hit_prob=0.5, seed=9)
    arm, _gt = _arm("a", preds, targets, grid)
    with pytest.raises(MissingFrameProvenance, match="gt_per_frame"):
        paired_bootstrap({"arms": [arm, dict(arm, name="b")]}, n_boot=10)


def test_result_is_deterministic_for_a_fixed_seed():
    preds, targets, grid = _frames(25, hit_prob=0.7, seed=11)
    arm_a, gt = _arm("a", preds, targets, grid)
    preds_b, _t, _g = _frames(25, hit_prob=0.4, seed=11)
    arm_b, _ = _arm("b", preds_b, targets, grid)
    payload = {"arms": [arm_a, arm_b], "gt_per_frame": gt}
    r1 = paired_bootstrap(payload, n_boot=120, seed=42)
    r2 = paired_bootstrap(payload, n_boot=120, seed=42)
    assert json.dumps(r1["comparisons"]) == json.dumps(r2["comparisons"])
