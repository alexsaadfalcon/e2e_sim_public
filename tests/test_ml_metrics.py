"""
Tests for `e2e.ml.metrics` (RADIal-style detection evaluation, adapted to our polar
(range, sin-azimuth) label format).

Uses the real `Scatterer`/`RadarPose`/`LabelGrid`/`encode_detection_labels`/
`targets_in_grid` from `e2e.ml.scatterers`/`e2e.ml.labels` (sibling shards' dependency-
free/torch modules), plus a few hand-built label-map tensors where we need exact,
hand-checkable regression values and exact per-detection scores that
`encode_detection_labels`' all-ones objectness would otherwise mask (see the RMSE and
PR-curve tests below).

`AP` here is all-points interpolated precision-recall average precision and `AR` is
recall at one stated operating point (`score_threshold`, default 0.1) -- NOT the
pre-2026-08-17 mean over absolute thresholds 0.1..0.9. Several tests below exist
specifically to pin the failure modes that definition had.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from e2e.ml.labels import LabelGrid, encode_detection_labels, targets_in_grid
from e2e.ml.metrics import (
    DEFAULT_CLASSES,
    MatchCriterion,
    evaluate_dataset,
    evaluate_frame,
    match_detections,
)
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
# evaluate_dataset: the ORACLE (canonical check, notes/RIGOR_STANDARD.md)
# --------------------------------------------------------------------------------
def test_oracle_ground_truth_as_prediction_scores_exactly_one():
    """ORACLE: feed the ground-truth label map back in as the prediction and the metric
    must return AP and AR of exactly 1.0 -- equality, not approximation.

    This is the canonical check from `notes/RIGOR_STANDARD.md`, and it is a real guard,
    not a formality: an AP implementation that integrates the PR curve as
    `sum((r_k - r_{k-1}) * p_interp[k])` accumulates floating-point error in the
    telescoping recall differences and can land on 0.9999999999999998 for a flawless
    detector. `_interpolated_ap` sums `n_gt` exact 1.0 terms and divides once instead,
    which is why `== 1.0` holds. If this ever fails, the AP definition is wrong.
    """
    grid = LabelGrid(n_range=40, n_azimuth=40, max_range_m=40.0)
    pose = RadarPose()
    # Deliberately not a power of two and not 4: 7 targets makes 1/n_gt a non-terminating
    # binary fraction, which is exactly the case a telescoping-sum AP gets wrong.
    cells = [(3, 3), (3, 20), (3, 36), (20, 3), (20, 36), (36, 3), (36, 36)]
    import random
    rng = random.Random(11)
    scatterers = []
    for (ri, ai) in cells:
        r = (ri + rng.uniform(0.1, 0.9)) * grid.range_bin_m
        sin_az = -1.0 + (ai + rng.uniform(0.1, 0.9)) * grid.az_bin
        scatterers.append(_target(r, sin_az))

    pred_map = encode_detection_labels(grid, scatterers, pose)
    targets = targets_in_grid(grid, scatterers, pose)
    assert len(targets) == 7

    result = evaluate_dataset([pred_map], [targets], grid)
    assert result["AP"] == 1.0
    assert result["AR"] == 1.0
    assert result["fp"] == 0 and result["fn"] == 0


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


def test_missing_target_costs_ap_even_though_precision_stays_perfect():
    """A detector that finds 3 of 4 targets with zero false positives keeps precision =
    1.0, but its AP must be 0.75, not 1.0.

    Guards the headline behavioural difference between interpolated-PR AP and the old
    threshold-mean: AP integrates over the WHOLE recall axis, so the quarter of recall
    this detector never reaches contributes zero area. Under the old definition, which
    averaged precision over absolute score thresholds, a high-precision/low-recall
    detector printed AP = 1.0 and the missed target was invisible in AP -- precisely the
    failure that let a near-silent checkpoint outrank a genuinely better-recalling one.
    """
    grid = LabelGrid(n_range=40, n_azimuth=40, max_range_m=40.0)
    pose = RadarPose()
    cells = [(5, 5), (5, 30), (30, 5), (30, 30)]
    scatterers = [_target((ri + 0.5) * grid.range_bin_m, -1.0 + (ai + 0.5) * grid.az_bin)
                  for (ri, ai) in cells]

    pred_map = encode_detection_labels(grid, scatterers[:-1], pose)   # model misses the 4th target
    targets = targets_in_grid(grid, scatterers, pose)                 # but it's really there

    result = evaluate_dataset([pred_map], [targets], grid)
    assert result["AR"] == pytest.approx(3.0 / 4.0)
    assert result["precision"] == pytest.approx(1.0)   # every detection it made was right
    assert result["AP"] == pytest.approx(3.0 / 4.0)    # ...but a quarter of the recall axis is empty


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
    # Ground-truth-encoded maps score every detection at objectness 1.0, so all five tie.
    # Pessimistic tie-breaking ranks the phantom first, making the whole tied group score
    # its end-of-group precision 4/5 -- see the tie-break test below.
    assert result["AP"] == pytest.approx(4.0 / 5.0)
    assert result["AR"] == pytest.approx(1.0)


def test_tied_scores_are_broken_pessimistically():
    """Equal-scored detections must not let a detector harvest AP from an ordering it
    never produced: within a tie group, false positives rank ahead of true positives.

    Not academic -- `e2e.ml.baseline`'s CFAR objectness is clamped to [0, 1], so a
    saturating classical detector emits many detections scored exactly 1.0, and an
    optimistic tie-break would hand it AP = 1.0 here instead of 0.5 purely on the
    accident of list order.
    """
    grid = LabelGrid(n_range=20, n_azimuth=20, max_range_m=20.0)
    pose = RadarPose()
    real = _target((5 + 0.5) * grid.range_bin_m, -1.0 + (5 + 0.5) * grid.az_bin)
    phantom = _target((15 + 0.5) * grid.range_bin_m, -1.0 + (15 + 0.5) * grid.az_bin)

    targets = targets_in_grid(grid, [real], pose)
    assert len(targets) == 1

    for order in ([real, phantom], [phantom, real]):
        pred_map = encode_detection_labels(grid, order, pose)
        result = evaluate_dataset([pred_map], [targets], grid)
        assert result["n_detections"] == 2
        assert {s for s in result["pr_curve"]["score"]} == {1.0}   # a genuine tie
        assert result["AP"] == pytest.approx(0.5)                  # not 1.0, either way round
        assert result["AR"] == pytest.approx(1.0)


# --------------------------------------------------------------------------------
# evaluate_dataset / evaluate_frame: empty-GT / empty-pred edge cases
# --------------------------------------------------------------------------------
def test_empty_gt_and_empty_pred_gives_precision_recall_one(torch_device):
    """Vacuous split (nothing predicted, nothing to predict): AP = AR = 1.0 by the
    documented 0/0 convention, and `pr_curve` is None rather than an empty-list stub --
    with no ground truth there is no recall axis to integrate, and a plotted "curve" of
    nothing would invite a reader to treat it as a measured result."""
    grid = LabelGrid(n_range=10, n_azimuth=10, max_range_m=10.0)
    pred_map = torch.zeros((3, grid.n_range, grid.n_azimuth), dtype=torch.float32, device=torch_device)

    result = evaluate_dataset([pred_map], [[]], grid)
    assert result["AP"] == pytest.approx(1.0)
    assert result["AR"] == pytest.approx(1.0)
    assert result["pr_curve"] is None
    assert result["n_targets"] == 0 and result["n_detections"] == 0


def test_empty_gt_one_detection_is_pure_false_positive(torch_device):
    grid = LabelGrid(n_range=10, n_azimuth=10, max_range_m=10.0)
    pred_map = torch.zeros((3, grid.n_range, grid.n_azimuth), dtype=torch.float32, device=torch_device)
    pred_map[0, 5, 5] = 1.0   # one confident detection, no ground truth anywhere

    result = evaluate_dataset([pred_map], [[]], grid)
    assert result["AP"] == pytest.approx(0.0)
    assert result["AR"] == pytest.approx(1.0)   # 0/0 convention: no GT means nothing to miss
    assert result["fp"] == 1


def test_empty_predictions_against_real_gt_score_zero_not_vacuously_perfect(torch_device):
    """A silent detector facing real ground truth: AP = AR = 0.0, and `precision` is NaN
    (undefined), never the vacuous 0/0 -> 1.0.

    Guards the "under-confident model harvests free precision" failure. Under the old
    absolute-threshold sweep this was the dominant pathology: every sweep point above a
    detector's score ceiling had no detections, and treating those as precision = 1.0
    let a model that detected essentially nothing print a high AP. AP no longer derives
    from this field at all, but the field itself must still read as undefined.
    """
    grid = LabelGrid(n_range=20, n_azimuth=20, max_range_m=20.0)
    silent = torch.zeros((3, grid.n_range, grid.n_azimuth), dtype=torch.float32,
                         device=torch_device)
    targets = [((5 + 0.5) * grid.range_bin_m, -1.0 + (5 + 0.5) * grid.az_bin, "vehicle")]

    result = evaluate_dataset([silent], [targets], grid)
    assert result["AP"] == 0.0
    assert result["AR"] == 0.0
    assert math.isnan(result["precision"])
    assert result["n_detections"] == 0
    assert result["fn"] == 1
    # Localization error is likewise undefined, not a flattering 0.0 (see `_rmse`).
    assert math.isnan(result["range_rmse_m"])
    assert math.isnan(result["sin_az_rmse"])


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


def test_rmse_is_nan_not_zero_when_nothing_matches(torch_device):
    """REGRESSION (metric audit, 2026-08-16): an empty match set must report localization
    RMSE as NaN, never 0.0 -- 0.0 reads as flawless ranging.

    The original sighting was a detector whose confidence ceiling sat below the old
    RMSE threshold (the sweep point nearest 0.5), so it contributed no matched pairs
    there and printed `range_rmse_m = 0.000` while its recall was correctly nonzero;
    that exact pattern appeared in real SSMRadNet evaluations. Measuring RMSE at the
    operating point instead of at 0.5 removes that particular trap, so this test now
    pins the surviving general case: a detector that fires confidently but matches
    nothing at all.
    """
    grid = LabelGrid(n_range=20, n_azimuth=20, max_range_m=20.0)   # range_bin=1.0, az_bin=0.1
    pred = torch.zeros((3, grid.n_range, grid.n_azimuth), dtype=torch.float32, device=torch_device)
    pred[0, 15, 15] = 0.9      # confident, well above the operating point...
    targets = [((5 + 0.5) * grid.range_bin_m, -1.0 + (5 + 0.5) * grid.az_bin, "vehicle")]
    # ...but nowhere near the only real target, so nothing matches.

    result = evaluate_dataset([pred], [targets], grid)

    assert result["tp"] == 0 and result["fp"] == 1 and result["fn"] == 1
    assert math.isnan(result["range_rmse_m"])
    assert math.isnan(result["sin_az_rmse"])
    assert result["AP"] == 0.0
    assert result["AR"] == 0.0


# --------------------------------------------------------------------------------
# evaluate_dataset: the interpolated PR curve itself
# --------------------------------------------------------------------------------
def _five_detection_pr_case(torch_device):
    """3 ground-truth targets; 5 detections at distinct, descending scores whose
    true/false pattern down the ranking is TP, FP, TP, FP, FP.

    Cells are >= 3 apart in Chebyshev distance so `decode_detections`' 3x3 NMS keeps all
    five, and the false positives sit >= 6 azimuth bins (0.6 in sin-azimuth) from every
    target, far outside the 0.06 match tolerance. The third target is never detected at
    all, so max recall is 2/3.
    """
    grid = LabelGrid(n_range=20, n_azimuth=20, max_range_m=20.0)   # range_bin=1.0, az_bin=0.1
    pred = torch.zeros((3, grid.n_range, grid.n_azimuth), dtype=torch.float32, device=torch_device)

    def _center(ri, ai):
        return ((ri + 0.5) * grid.range_bin_m, -1.0 + (ai + 0.5) * grid.az_bin)

    for (ri, ai), score in (((2, 2), 0.9), ((2, 8), 0.8), ((8, 2), 0.7),
                            ((8, 8), 0.6), ((14, 14), 0.5)):
        pred[0, ri, ai] = score

    targets = [(*_center(2, 2), "vehicle"),     # hit by the 0.9 detection
               (*_center(8, 2), "vehicle"),     # hit by the 0.7 detection
               (*_center(14, 2), "vehicle")]    # never detected
    return grid, pred, targets


def test_interpolated_ap_hand_computed_five_nine(torch_device):
    """Hand-computable AP: the PR case above must score exactly 5/9 = 0.5556.

    Worked by hand down the pooled score ranking (n_gt = 3):

        rank  score  hit  precision  recall  precision_interp
          1    0.9   TP    1/1=1.00   1/3     1.000
          2    0.8   FP    1/2=0.50   1/3     0.667
          3    0.7   TP    2/3=0.667  2/3     0.667
          4    0.6   FP    2/4=0.50   2/3     0.500
          5    0.5   FP    2/5=0.40   2/3     0.400

    AP = (1/n_gt) * sum of precision_interp at the TP ranks = (1.000 + 0.667)/3 = 5/9.
    The un-detected third target contributes no area, which is why AP < AR here.

    This is the arithmetic the whole fix rests on; if it drifts, the AP being reported
    is not the VOC/COCO definition the docstring claims it is.
    """
    grid, pred, targets = _five_detection_pr_case(torch_device)
    result = evaluate_dataset([pred], [targets], grid)

    assert result["n_detections"] == 5
    assert result["tp"] == 2 and result["fp"] == 3 and result["fn"] == 1
    assert result["AP"] == pytest.approx(5.0 / 9.0)
    assert result["AR"] == pytest.approx(2.0 / 3.0)
    assert result["precision"] == pytest.approx(2.0 / 5.0)


def test_pr_curve_is_interpolated_monotone_and_recall_non_decreasing(torch_device):
    """The returned `pr_curve` must expose BOTH the raw and the interpolated precision,
    with recall non-decreasing and `precision_interp` non-increasing down the ranking.

    Guards against the interpolation silently becoming a no-op: raw precision on this
    case genuinely rises at rank 3 (0.50 -> 0.667), so a curve whose `precision_interp`
    equals `precision` would mean the running max was never applied and AP would be the
    area under a saw-toothed curve rather than the standard interpolated one.
    """
    grid, pred, targets = _five_detection_pr_case(torch_device)
    curve = evaluate_dataset([pred], [targets], grid)["pr_curve"]

    assert curve["score"] == pytest.approx([0.9, 0.8, 0.7, 0.6, 0.5], abs=1e-6)
    assert curve["is_tp"] == [True, False, True, False, False]
    assert curve["precision"] == pytest.approx([1.0, 0.5, 2 / 3, 0.5, 0.4])
    assert curve["precision_interp"] == pytest.approx([1.0, 2 / 3, 2 / 3, 0.5, 0.4])
    assert curve["precision"] != pytest.approx(curve["precision_interp"])   # interpolation did something
    assert all(curve["recall"][i] <= curve["recall"][i + 1] + 1e-12
               for i in range(len(curve["recall"]) - 1))
    assert all(curve["precision_interp"][i] >= curve["precision_interp"][i + 1] - 1e-12
               for i in range(len(curve["precision_interp"]) - 1))


def test_score_ceiling_below_old_sweep_is_scored_on_merit(torch_device):
    """THE BUG THIS DEFINITION REPLACED. A detector whose maximum objectness is 0.28 --
    below the old sweep's 0.3..0.9 points, which is where FFTRadNet (max 0.204) and
    SSMRadNet (max 0.292) actually live -- must be scored on its ranking, not punished
    for its calibration.

    Here it localizes all three targets perfectly with no false positives, so AP and AR
    are both exactly 1.0. Under the old absolute-threshold mean, six of the nine sweep
    points were structurally empty: AR was capped at 3/9 = 0.333 no matter how good the
    detector was, and AP rested on the two or three points that had any detections at
    all -- which is how a reported AP could swing 8x between epochs while recall rose
    monotonically. AR here must NOT be 1/3.
    """
    grid = LabelGrid(n_range=20, n_azimuth=20, max_range_m=20.0)
    pred = torch.zeros((3, grid.n_range, grid.n_azimuth), dtype=torch.float32,
                       device=torch_device)
    cells = [(2, 2), (8, 8), (14, 14)]
    scores = [0.12, 0.20, 0.28]     # every one below 0.3; the ceiling is 0.28
    targets = []
    for (ri, ai), score in zip(cells, scores):
        pred[0, ri, ai] = score
        targets.append(((ri + 0.5) * grid.range_bin_m, -1.0 + (ai + 0.5) * grid.az_bin,
                        "vehicle"))

    result = evaluate_dataset([pred], [targets], grid)

    assert max(result["pr_curve"]["score"]) < 0.3      # the premise of the old bug
    assert result["AP"] == 1.0
    assert result["AR"] == 1.0
    assert result["AR"] != pytest.approx(3.0 / 9.0)    # the old structural cap
    assert result["range_rmse_m"] == pytest.approx(0.0, abs=1e-5)


def test_operating_point_is_stated_in_the_result(torch_device):
    """`AR` must never be a number a reader has to guess the meaning of: the result dict
    carries the numeric `score_threshold` it was measured at and a human-readable
    `AR_operating_point`, and both must survive into stored metrics JSON.

    The old `AR` was a mean over nine absolute thresholds with nothing in the dict
    saying so -- readers (and the deck figures) took it for recall. It was not.
    """
    grid, pred, targets = _five_detection_pr_case(torch_device)
    result = evaluate_dataset([pred], [targets], grid, score_threshold=0.55)

    assert result["score_threshold"] == pytest.approx(0.55)
    assert "0.55" in result["AR_operating_point"]
    # The floor genuinely selects the detection set: only the 0.9/0.8/0.7/0.6 detections
    # clear 0.55, so the 0.5-scored false positive is gone.
    assert result["n_detections"] == 4
    assert result["fp"] == 2


# --------------------------------------------------------------------------------
# Per-class AP/AR
# --------------------------------------------------------------------------------
def test_default_classes_is_vehicle_pedestrian():
    assert DEFAULT_CLASSES == ("vehicle", "pedestrian")


def test_all_vehicle_frame_gives_nan_pedestrian_metrics():
    """A frame with only vehicle targets: AP_vehicle/AR_vehicle mirror the pooled
    numbers exactly (filtering to "vehicle" changes nothing when everything already IS
    vehicle), while AP_pedestrian/AR_pedestrian are NaN (nothing of that class exists to
    score), not the vacuous 0/0 -> 1.0 pooled convention."""
    grid = LabelGrid(n_range=40, n_azimuth=40, max_range_m=40.0)
    pose = RadarPose()
    cells = [(5, 5), (30, 30)]
    vehicles = [_target((ri + 0.5) * grid.range_bin_m, -1.0 + (ai + 0.5) * grid.az_bin, "vehicle")
                for (ri, ai) in cells]

    pred_map = encode_detection_labels(grid, vehicles, pose)
    targets = targets_in_grid(grid, vehicles, pose)

    result = evaluate_dataset([pred_map], [targets], grid)
    assert result["AP"] == pytest.approx(1.0)
    assert result["AR"] == pytest.approx(1.0)
    assert result["AP_vehicle"] == pytest.approx(result["AP"])
    assert result["AR_vehicle"] == pytest.approx(result["AR"])
    assert result["n_targets_vehicle"] == 2
    assert result["n_targets_pedestrian"] == 0
    assert math.isnan(result["AP_pedestrian"])
    assert math.isnan(result["AR_pedestrian"])


def test_mixed_frame_per_class_hand_computable():
    """2 vehicle + 2 pedestrian targets, all perfectly detected, plus one spurious
    (unmatched-by-anything) detection far from every real target.

    Pooled: TP=4, FP=1 (the phantom), FN=0 -> AP=4/5=0.8, AR=1.0.
    Per-class (target list filtered to just that class, but ALL 5 detections still
    compete for it -- see metrics.py's "Per-class AP/AR" docstring): each class's own
    2 targets get matched (TP=2), and the OTHER class's 2 correct detections plus the
    phantom (3 detections) all become false positives against that class's restricted
    target list -> AP_vehicle = AP_pedestrian = 2/5 = 0.4, AR_vehicle = AR_pedestrian = 1.0.
    """
    grid = LabelGrid(n_range=40, n_azimuth=40, max_range_m=40.0)
    pose = RadarPose()

    def _at(ri, ai, object_class):
        return _target((ri + 0.5) * grid.range_bin_m, -1.0 + (ai + 0.5) * grid.az_bin,
                        object_class)

    vehicles = [_at(5, 5, "vehicle"), _at(5, 30, "vehicle")]
    pedestrians = [_at(30, 5, "pedestrian"), _at(30, 30, "pedestrian")]
    phantom = _at(18, 18, "vehicle")   # far from all real cells; excluded from targets below

    pred_map = encode_detection_labels(grid, vehicles + pedestrians + [phantom], pose)
    targets = targets_in_grid(grid, vehicles + pedestrians, pose)   # phantom is not real

    result = evaluate_dataset([pred_map], [targets], grid)
    assert result["AP"] == pytest.approx(0.8)
    assert result["AR"] == pytest.approx(1.0)

    assert result["n_targets_vehicle"] == 2
    assert result["n_targets_pedestrian"] == 2
    # Sanity: every real target is accounted for exactly once across the per-class splits.
    assert result["n_targets_vehicle"] + result["n_targets_pedestrian"] == len(targets)

    assert result["AP_vehicle"] == pytest.approx(0.4)
    assert result["AR_vehicle"] == pytest.approx(1.0)
    assert result["AP_pedestrian"] == pytest.approx(0.4)
    assert result["AR_pedestrian"] == pytest.approx(1.0)


def test_classes_empty_tuple_omits_per_class_keys():
    grid = LabelGrid(n_range=10, n_azimuth=10, max_range_m=10.0)
    pose = RadarPose()
    scatterers = [_target((5 + 0.5) * grid.range_bin_m, -1.0 + (5 + 0.5) * grid.az_bin)]
    pred_map = encode_detection_labels(grid, scatterers, pose)
    targets = targets_in_grid(grid, scatterers, pose)

    result = evaluate_dataset([pred_map], [targets], grid, classes=())
    assert "AP_vehicle" not in result
    assert "AR_vehicle" not in result
    assert result["AP"] == pytest.approx(1.0)
