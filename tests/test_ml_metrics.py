"""
Tests for `e2e.ml.metrics` (RADIal-style detection evaluation, adapted to our polar
(range, sin-azimuth) label format).

Uses the real `Scatterer`/`RadarPose`/`LabelGrid`/`encode_detection_labels`/
`targets_in_grid` from `e2e.ml.scatterers`/`e2e.ml.labels` (sibling shards' dependency-
free/torch modules), plus a few hand-built label-map tensors where we need exact,
hand-checkable regression values that `encode_detection_labels`' sub-bin-accurate
round trip would otherwise mask (see the RMSE / threshold-sweep tests below).
"""

import math

import pytest

torch = pytest.importorskip("torch")

from e2e.ml.labels import LabelGrid, encode_detection_labels, targets_in_grid
from e2e.ml.metrics import MatchCriterion, evaluate_dataset, evaluate_frame, match_detections
from e2e.ml.scatterers import RadarPose, Scatterer


def _target(r, sin_az, object_class="vehicle"):
    """A Scatterer at the given (range, sin_azimuth) w.r.t. the default RadarPose, z=0."""
    y = r * sin_az
    x = math.sqrt(max(r * r - y * y, 0.0))
    return Scatterer(position=(x, y, 0.0), velocity=(0.0, 0.0, 0.0), rcs_dbsm=0.0,
                      object_class=object_class)


# --------------------------------------------------------------------------------
# match_detections: unit-level behaviour
# --------------------------------------------------------------------------------
def test_boundary_just_inside_matches():
    criterion = MatchCriterion(max_range_err_m=2.0, max_sin_az_err=0.06)
    targets = [(10.0, 0.0, "vehicle")]
    det_in = [(12.0, 0.0, 0.9)]   # dr = 2.0 / 2.0 = 1.0 exactly -> boundary is inclusive

    matches, unmatched_det, unmatched_gt = match_detections(det_in, targets, criterion)
    assert matches == [(0, 0)]
    assert unmatched_det == []
    assert unmatched_gt == []


def test_boundary_just_outside_does_not_match():
    criterion = MatchCriterion(max_range_err_m=2.0, max_sin_az_err=0.06)
    targets = [(10.0, 0.0, "vehicle")]
    det_out = [(12.01, 0.0, 0.9)]   # dr = 2.01 / 2.0 > 1.0

    matches, unmatched_det, unmatched_gt = match_detections(det_out, targets, criterion)
    assert matches == []
    assert unmatched_det == [0]
    assert unmatched_gt == [0]


def test_greedy_matching_prefers_higher_score_over_better_fit():
    """A worse-fit, higher-score detection claims the target before a perfect-fit rival."""
    targets = [(10.0, 0.0, "vehicle")]
    detections = [(10.5, 0.0, 0.99), (10.0, 0.0, 0.5)]   # idx0 worse fit, higher score

    matches, unmatched_det, unmatched_gt = match_detections(detections, targets)
    assert matches == [(0, 0)]
    assert unmatched_det == [1]
    assert unmatched_gt == []


def test_score_ordering_breaks_ties_by_original_order():
    targets = [(10.0, 0.0, "vehicle")]
    # equal scores; original list order should decide who claims the only target
    detections = [(10.0, 0.0, 0.5), (10.05, 0.0, 0.5)]

    matches, unmatched_det, _ = match_detections(detections, targets)
    assert matches == [(0, 0)]
    assert unmatched_det == [1]

    # reversed positions, same scores -> the new index-0 (worse fit) should still win
    detections_rev = [(10.05, 0.0, 0.5), (10.0, 0.0, 0.5)]
    matches_rev, unmatched_det_rev, _ = match_detections(detections_rev, targets)
    assert matches_rev == [(0, 0)]
    assert unmatched_det_rev == [1]


def test_match_detections_empty_gt_and_empty_pred():
    matches, unmatched_det, unmatched_gt = match_detections([], [])
    assert matches == []
    assert unmatched_det == []
    assert unmatched_gt == []


def test_match_detections_empty_gt_one_det_is_all_unmatched():
    matches, unmatched_det, unmatched_gt = match_detections([(10.0, 0.0, 0.9)], [])
    assert matches == []
    assert unmatched_det == [0]
    assert unmatched_gt == []


def test_match_detections_empty_pred_all_gt_unmatched():
    matches, unmatched_det, unmatched_gt = match_detections([], [(10.0, 0.0, "vehicle")])
    assert matches == []
    assert unmatched_det == []
    assert unmatched_gt == [0]


# --------------------------------------------------------------------------------
# evaluate_dataset: perfect predictions
# --------------------------------------------------------------------------------
def test_perfect_predictions_give_ap_ar_one_and_near_zero_rmse():
    grid = LabelGrid(n_range=40, n_azimuth=40, max_range_m=40.0)
    pose = RadarPose()
    cells = [(5, 5), (5, 30), (30, 5), (30, 30)]
    import random
    rng = random.Random(7)
    scatterers = []
    for (ri, ai) in cells:
        r = (ri + rng.uniform(0.1, 0.9)) * grid.range_bin_m
        sin_az = -1.0 + (ai + rng.uniform(0.1, 0.9)) * grid.az_bin
        scatterers.append(_target(r, sin_az))

    pred_map = encode_detection_labels(grid, scatterers, pose)
    targets = targets_in_grid(grid, scatterers, pose)

    result = evaluate_dataset([pred_map], [targets], grid)
    assert result["AP"] == pytest.approx(1.0)
    assert result["AR"] == pytest.approx(1.0)
    assert result["range_rmse_m"] < 1e-3
    assert result["sin_az_rmse"] < 1e-3


def test_missing_target_drops_recall_not_precision():
    grid = LabelGrid(n_range=40, n_azimuth=40, max_range_m=40.0)
    pose = RadarPose()
    cells = [(5, 5), (5, 30), (30, 5), (30, 30)]
    scatterers = [_target((ri + 0.5) * grid.range_bin_m, -1.0 + (ai + 0.5) * grid.az_bin)
                  for (ri, ai) in cells]

    pred_map = encode_detection_labels(grid, scatterers[:-1], pose)   # model misses the 4th target
    targets = targets_in_grid(grid, scatterers, pose)                 # but it's really there

    result = evaluate_dataset([pred_map], [targets], grid)
    assert result["AR"] == pytest.approx(3.0 / 4.0)
    assert result["AP"] == pytest.approx(1.0)


def test_spurious_detection_drops_precision_not_recall():
    grid = LabelGrid(n_range=40, n_azimuth=40, max_range_m=40.0)
    pose = RadarPose()
    cells = [(5, 5), (5, 30), (30, 5), (30, 30)]
    real = [_target((ri + 0.5) * grid.range_bin_m, -1.0 + (ai + 0.5) * grid.az_bin)
            for (ri, ai) in cells]
    phantom = _target((18 + 0.5) * grid.range_bin_m, -1.0 + (18 + 0.5) * grid.az_bin)  # far from all real cells

    pred_map = encode_detection_labels(grid, real + [phantom], pose)
    targets = targets_in_grid(grid, real, pose)   # phantom is not a real target

    result = evaluate_dataset([pred_map], [targets], grid)
    assert result["AP"] == pytest.approx(4.0 / 5.0)
    assert result["AR"] == pytest.approx(1.0)


# --------------------------------------------------------------------------------
# evaluate_dataset / evaluate_frame: empty-GT / empty-pred edge cases
# --------------------------------------------------------------------------------
def test_empty_gt_and_empty_pred_gives_precision_recall_one(torch_device):
    grid = LabelGrid(n_range=10, n_azimuth=10, max_range_m=10.0)
    pred_map = torch.zeros((3, grid.n_range, grid.n_azimuth), dtype=torch.float32, device=torch_device)

    result = evaluate_dataset([pred_map], [[]], grid)
    assert result["AP"] == pytest.approx(1.0)
    assert result["AR"] == pytest.approx(1.0)
    for p in result["precision_per_threshold"].values():
        assert p == pytest.approx(1.0)
    for r in result["recall_per_threshold"].values():
        assert r == pytest.approx(1.0)


def test_empty_gt_one_detection_is_pure_false_positive(torch_device):
    grid = LabelGrid(n_range=10, n_azimuth=10, max_range_m=10.0)
    pred_map = torch.zeros((3, grid.n_range, grid.n_azimuth), dtype=torch.float32, device=torch_device)
    pred_map[0, 5, 5] = 1.0   # one confident detection, no ground truth anywhere

    result = evaluate_dataset([pred_map], [[]], grid)
    assert result["AP"] == pytest.approx(0.0)
    assert result["AR"] == pytest.approx(1.0)   # 0/0 convention: no GT means nothing to miss


def test_evaluate_frame_empty_gt_one_detection_fp_count(torch_device):
    grid = LabelGrid(n_range=10, n_azimuth=10, max_range_m=10.0)
    pred_map = torch.zeros((3, grid.n_range, grid.n_azimuth), dtype=torch.float32, device=torch_device)
    pred_map[0, 5, 5] = 1.0

    result = evaluate_frame(pred_map, [], grid, threshold=0.5)
    assert result["tp"] == 0
    assert result["fp"] == 1
    assert result["fn"] == 0
    assert result["range_errs"] == []
    assert result["sin_az_errs"] == []


# --------------------------------------------------------------------------------
# RMSE: hand-checked on a 2-frame case with known regression residuals
# --------------------------------------------------------------------------------
def test_rmse_hand_checked_two_frames(torch_device):
    grid = LabelGrid(n_range=10, n_azimuth=10, max_range_m=10.0)   # range_bin=1.0, az_bin=0.2
    assert grid.range_bin_m == pytest.approx(1.0)
    assert grid.az_bin == pytest.approx(0.2)

    # Frame 1: single detection at cell (5, 5); centre = (5.5, 0.1). Residuals push the
    # decoded value to r=6.0, sin_az=0.1 (dr=0.5 range-bins, 0 az-bins).
    map1 = torch.zeros((3, grid.n_range, grid.n_azimuth), dtype=torch.float32, device=torch_device)
    map1[0, 5, 5] = 1.0
    map1[1, 5, 5] = 0.5
    map1[2, 5, 5] = 0.0
    target1 = [(5.7, 0.1, "vehicle")]   # true position: range error |6.0 - 5.7| = 0.3

    # Frame 2: single detection at cell (2, 2); centre = (2.5, -0.5). Residuals push the
    # decoded value to r=2.3, sin_az=-0.4.
    map2 = torch.zeros((3, grid.n_range, grid.n_azimuth), dtype=torch.float32, device=torch_device)
    map2[0, 2, 2] = 1.0
    map2[1, 2, 2] = -0.2
    map2[2, 2, 2] = 0.5
    target2 = [(2.5, -0.4, "vehicle")]   # true position: range error |2.3 - 2.5| = 0.2

    result = evaluate_dataset([map1, map2], [target1, target2], grid)

    expected_range_rmse = math.sqrt((0.3 ** 2 + 0.2 ** 2) / 2.0)
    assert result["range_rmse_m"] == pytest.approx(expected_range_rmse, abs=1e-5)
    assert result["sin_az_rmse"] == pytest.approx(0.0, abs=1e-5)
    assert result["AP"] == pytest.approx(1.0)
    assert result["AR"] == pytest.approx(1.0)


# --------------------------------------------------------------------------------
# evaluate_dataset: threshold-sweep monotonicity sanity
# --------------------------------------------------------------------------------
def test_recall_is_non_increasing_as_threshold_rises(torch_device):
    grid = LabelGrid(n_range=20, n_azimuth=20, max_range_m=20.0)   # range_bin=1.0, az_bin=0.1
    pred_map = torch.zeros((3, grid.n_range, grid.n_azimuth), dtype=torch.float32, device=torch_device)

    cells = [(2, 2), (6, 6), (10, 10), (14, 14), (18, 18)]
    scores = [0.15, 0.35, 0.55, 0.75, 0.95]
    targets = []
    for (i, j), score in zip(cells, scores):
        pred_map[0, i, j] = score
        r_center = (i + 0.5) * grid.range_bin_m
        az_center = -1.0 + (j + 0.5) * grid.az_bin
        targets.append((r_center, az_center, "vehicle"))

    thresholds = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    result = evaluate_dataset([pred_map], [targets], grid, thresholds=thresholds)

    recalls = [result["recall_per_threshold"][t] for t in thresholds]
    assert recalls[0] == pytest.approx(1.0)     # threshold 0.1: all 5 detections above it
    assert recalls[-1] == pytest.approx(0.2)    # threshold 0.9: only the 0.95-score one survives
    assert all(recalls[i] >= recalls[i + 1] - 1e-9 for i in range(len(recalls) - 1))


def test_underconfident_model_does_not_collect_vacuous_precision(torch_device):
    """Adversarial-review fix: a model that localizes perfectly but never emits
    confidence above 0.55 must NOT harvest precision=1.0 at thresholds 0.6-0.9
    (where it makes zero detections against existing GT) -- those thresholds have
    UNDEFINED precision (NaN, excluded from AP), so AP reflects only thresholds
    where the model actually detects."""
    grid = LabelGrid(n_range=20, n_azimuth=20, max_range_m=20.0)
    pred_map = torch.zeros((3, grid.n_range, grid.n_azimuth), dtype=torch.float32,
                           device=torch_device)
    pred_map[0, 5, 5] = 0.55                    # one perfect but low-confidence detection
    r = (5 + 0.5) * grid.range_bin_m
    az = -1.0 + (5 + 0.5) * grid.az_bin
    targets = [(r, az, "vehicle")]

    result = evaluate_dataset([pred_map], [targets], grid)
    p = result["precision_per_threshold"]
    assert p[0.5] == pytest.approx(1.0)         # detects, correctly
    assert all(math.isnan(p[t]) for t in (0.6, 0.7, 0.8, 0.9))   # undefined, not 1.0
    assert result["AP"] == pytest.approx(1.0)   # mean over DEFINED thresholds only
    assert result["AR"] == pytest.approx(5 / 9) # recall honestly collapses above 0.55
    # The defined-threshold count exposes how thin the AP sample is (5 of 9 here):
    # a near-silent model's AP can rest on 1-2 points and spike misleadingly.
    assert result["n_defined_precision_thresholds"] == 5
    # And a model that never detects anything at all scores AP = 0, not 1.
    silent = torch.zeros_like(pred_map)
    silent_result = evaluate_dataset([silent], [targets], grid)
    assert silent_result["AP"] == 0.0
    assert silent_result["AR"] == 0.0
