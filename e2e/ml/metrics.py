"""
Detection-evaluation metrics for the FMCW radar detection head, RADIal/FFTRadNet-style.

Adapted from the RADIal repo's evaluation protocol (`utils/metrics.py::GetFullMetrics`,
see the "Evaluation" section of the scout notes) to our polar `(range_m, sin_azimuth)`
label format (`e2e.ml.labels.LabelGrid`/`decode_detections`/`targets_in_grid`) instead of
RADIal's cartesian-box + IoU format.

What we keep from the reference protocol
-----------------------------------------
* A *fixed* matching criterion in place of RADIal's IoU>=0.5 -- our own "close enough in
  (range, sin-azimuth)" criterion (`MatchCriterion`), since our labels have no extent.
* Per-frame greedy matching in descending score order, TP/FP/FN pooled over the whole
  split (not per-frame metrics averaged over frames).

AP/AR definition (CHANGED 2026-08-17 -- see "The absolute-threshold-sweep bug")
-------------------------------------------------------------------------------
* `AP` is **all-points interpolated precision-recall average precision** -- the
  VOC2010+/COCO convention (COCO's `TYPE=1` "area under the interpolated PR curve", not
  VOC2007's 11-point sampled variant, and not RADIal's threshold-mean). All detections
  in the split are pooled and ranked by score descending; precision/recall are
  accumulated down that ranking; precision is made monotonically non-increasing by a
  right-to-left running max; the area is integrated as
  `AP = (1/N_gt) * sum(p_interp[k] for every rank k that is a true positive)`.
  That closed form is exactly the rectangle-rule area, because each true positive
  advances recall by exactly `1/N_gt` and every false positive advances it by zero.
  It is written that way (rather than `sum((r_k - r_{k-1}) * p_interp[k])`) so the
  perfect case sums `N_gt` exact `1.0` terms and returns **exactly** 1.0 in floating
  point -- the oracle check in `notes/RIGOR_STANDARD.md` demands equality, not
  approximation. Recall the detector never reaches contributes zero area, so an
  under-recalling detector is penalized in AP as well as AR (this is the standard
  behaviour and differs from the old threshold-mean, which could not see it).
* `AR` is recall at **one stated operating point**: every detection the decoder emits
  above `score_threshold` (default 0.1). It is reported alongside the human-readable
  `AR_operating_point` string and the numeric `score_threshold`, so no reader has to
  guess what "average recall" was averaged over -- nothing is averaged. `AR` is the
  maximum recall the detector attains at that confidence floor.
* Score **ties are broken pessimistically**: within a group of equal-scored detections,
  false positives are ranked ahead of true positives, so the AP contribution of the
  group is its end-of-group precision. A detector cannot harvest AP from an ordering it
  did not actually produce. This matters here concretely -- `e2e.ml.baseline`'s CFAR
  objectness is clamped to `[0, 1]`, so a saturating classical detector emits many
  detections scored exactly 1.0.

The absolute-threshold-sweep bug (why the definition changed)
--------------------------------------------------------------
Until 2026-08-17 `AP`/`AR` were the mean of precision/recall over the *absolute* score
thresholds 0.1, 0.2, ..., 0.9 -- RADIal's own non-standard `mAP`/`mAR`. That silently
assumed the detector's scores span [0, 1]. They do not: measured maximum objectness was
0.204 (FFTRadNet) and 0.292 (SSMRadNet) on the `rt_kenney_d2_v1` test split, so every
sweep point from 0.3 up was structurally empty. Consequences, all measured, all fixed by
the definition above:

* `AR` was capped at 2/9 = 0.222 for any detector with a sub-0.3 score ceiling -- it was
  not recall, it was a measure of the score ceiling. Re-placing the same 9 thresholds at
  the detectors' own score deciles, which changes not a single detection, moved it to
  0.87/0.97/0.63.
* `AP` rested on one or two sweep points. FFTRadNet's was the mean of P@0.1 = 0.0378 and
  P@0.2 = 0.0000, and that second term came from 7 detections across 60 frames of which
  none matched -- one more match would have moved the reported AP by 4.8x. Epoch to
  epoch the reported `val_AP` swung 8x (0.0224 -> 0.1866 -> 0.0540) while `val_AR` rose
  monotonically; the "best" epoch by AP had worse recall than every epoch after it.
* Localization RMSE inherited the same defect: it was measured at the single sweep point
  nearest 0.5, above every real detector's ceiling, so it was undefined for exactly the
  models being evaluated (see `_rmse`).

Neither `precision_per_threshold`/`recall_per_threshold` nor the
`n_defined_precision_thresholds` caveat-count survive; they existed only to describe and
hedge the threshold-mean. The PR curve backing the new `AP` is returned in full as
`pr_curve` instead.

Where we deliberately diverge
------------------------------
RADIal's shipped `utils/metrics.py` has a confirmed indexing bug (see scout notes,
section 5): its local `RA_to_cartesian_box` returns only cartesian box corners (no
Range/Angle columns), so the printed "Range Error (m)"/"Angle Error (degree)" in the
upstream repo actually index into the box's cartesian corner coordinates, not the
decoded polar (range, angle) values -- they are not faithful range/angle errors despite
the print labels. We do not have that bug: `range_errs`/`sin_az_errs` below are computed
directly from `decode_detections`' own polar output against `targets_in_grid`'s own
polar targets, for the actual matched (detection, ground-truth) pairs.

Matching criterion
------------------
RADIal matches via bounding-box IoU >= 0.5; our labels have no extent, so instead a
detection matches a ground-truth target when both axes are within a bin-count-normalized
tolerance (`MatchCriterion`): `max(|dr|/max_range_err_m, |dsin|/max_sin_az_err) <= 1.0`.
`max_sin_az_err=0.06` is roughly 3.4 degrees of azimuth error near boresight (small-angle
`d(theta) ~= d(sin theta)` there); the corresponding angular tolerance widens off-
boresight since `d(theta) = d(sin theta) / cos(theta)`.

WHICH range (2026-08-17): the range axis of that criterion is the **surface** range --
where the target's nearest reflecting face is, which is what a detector fires on and what
`e2e.ml.labels` now puts the objectness footprint on. Both tuple kinds carry it as an
optional 4th element (`_surface_range`; a 3-tuple is a point target). The RANGE-REGRESSION
error (`range_rmse_m`) is measured separately, against the object CENTRE, from element 0
of the same tuples -- so "did you find it" and "did you size it" stay two different
numbers and neither can hide the other.

The 2.0 m tolerance is NOT to be widened to paper over a labelling offset. MEASURED: at
an 8 m tolerance, a detector fed uniformly RANDOM ranges scores the same recall as the
real one -- i.e. the tolerance buys AP by making the range axis officially unmeasured.

Per-class AP/AR (roadmap: "per-frame / per-class-normalized AP for valid cross-tier
comparison")
------------------------------------------------------------------------------------
`evaluate_dataset` also reports `AP_<class>`/`AR_<class>`/`n_targets_<class>` for each of
`classes` (default `("vehicle", "pedestrian")`) alongside the pooled `AP`/`AR`. The
detection head is class-agnostic -- `decode_detections` scores/positions only, no
predicted class -- so a per-class score is computed by re-running the pooled algorithm
with the target list FILTERED to just that class (all detections still compete for those
targets). This makes `AR_<class>` an honest, well-defined per-class recall (a target of
class C either got matched or it didn't), but `AP_<class>` is a conservative/pessimistic
read on precision: a detection that correctly matched a different-class target in the
pooled evaluation counts as a false positive here, since class C's filtered target list
has nothing left for it to match. If a class has zero targets across the whole dataset,
its AP/AR are NaN (nothing to detect, not vacuously perfect) rather than going through the
0/0 convention. The frames are decoded ONCE and the resulting detections are re-matched
per class, so the per-class breakdown costs matching, not decoding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from e2e.ml.labels import LabelGrid, decode_detections

# (range_m, sin_azimuth, score[, surface_range_m]) -- see `e2e.ml.labels.decode_detections`
Detection = Tuple[float, ...]
# (range_m, sin_azimuth, object_class[, surface_range_m]) -- see `labels.targets_in_grid`
Target = Tuple


def _surface_range(item) -> float:
    """The SURFACE range of a detection/target tuple; its centre range if it has none.

    `e2e.ml.labels` appends a 4th element (the surface range) to both tuple kinds; a
    3-element tuple is a POINT target, whose surface and centre coincide -- so this is
    also what keeps hand-built `(range, sin_az, score)` / `(range, sin_az, class)` tuples
    behaving exactly as they did before 2026-08-17.
    """
    return float(item[3]) if len(item) > 3 else float(item[0])


@dataclass(frozen=True)
class MatchCriterion:
    """Tolerance defining a "close enough" detection/ground-truth match.

    A detection matches a target iff both `|dr| <= max_range_err_m` AND
    `|dsin| <= max_sin_az_err`, expressed as a single normalized distance
    `max(|dr|/max_range_err_m, |dsin|/max_sin_az_err) <= 1.0` (see `match_detections`).
    `dr` is a difference of SURFACE ranges -- see the module docstring's "Matching
    criterion" section, including why widening `max_range_err_m` is not an option.
    """

    max_range_err_m: float = 2.0
    max_sin_az_err: float = 0.06   # ~3.4 deg near boresight; widens off-boresight (see module docstring)


def _normalized_distance(det: Detection, tgt: Target, criterion: MatchCriterion) -> float:
    """Match distance, computed on the SURFACE range (see `_surface_range`)."""
    dr = abs(_surface_range(det) - _surface_range(tgt)) / criterion.max_range_err_m
    ds = abs(det[1] - tgt[1]) / criterion.max_sin_az_err
    return max(dr, ds)


def match_detections(
    detections: Sequence[Detection],
    targets: Sequence[Target],
    criterion: MatchCriterion = None,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Greedily match `detections` to `targets` under `criterion`.

    Processes detections in descending score order (ties keep their original relative
    order -- Python's sort is stable, including under `reverse=True`); each detection
    claims the *nearest* still-unmatched target within the criterion (normalized
    distance <= 1.0, i.e. the boundary itself counts as a match), so once a target is
    claimed it cannot be claimed again by a later (lower-score) detection.

    Returns
    -------
    matches : list of (detection_index, target_index)
    unmatched_det : list of detection indices with no match (false positives)
    unmatched_gt : list of target indices with no match (false negatives)
    All index lists are sorted ascending (i.e. in original list order).
    """
    if criterion is None:
        criterion = MatchCriterion()
    if not detections:
        return [], [], list(range(len(targets)))
    if not targets:
        return [], list(range(len(detections))), []

    order = sorted(range(len(detections)), key=lambda i: detections[i][2], reverse=True)
    matched_gt = set()
    matches: List[Tuple[int, int]] = []
    unmatched_det: List[int] = []

    for di in order:
        best_gi = None
        best_dist = None
        for gi, tgt in enumerate(targets):
            if gi in matched_gt:
                continue
            dist = _normalized_distance(detections[di], tgt, criterion)
            if dist <= 1.0 and (best_dist is None or dist < best_dist):
                best_dist = dist
                best_gi = gi
        if best_gi is None:
            unmatched_det.append(di)
        else:
            matched_gt.add(best_gi)
            matches.append((di, best_gi))

    matches.sort(key=lambda m: m[0])
    unmatched_det.sort()
    unmatched_gt = sorted(set(range(len(targets))) - matched_gt)
    return matches, unmatched_det, unmatched_gt


def evaluate_frame(
    pred_map,
    targets: Sequence[Target],
    grid: LabelGrid,
    *,
    threshold: float,
    criterion: MatchCriterion = None,
) -> Dict:
    """Decode `pred_map` at `threshold`, match against `targets`, score one frame.

    `pred_map` is anything `decode_detections` accepts (a `[3, n_range, n_azimuth]`
    label/prediction tensor). `targets` is the `targets_in_grid`-style list of
    `(range_m, sin_azimuth, object_class[, surface_range_m])` tuples.

    Returns `{"tp", "fp", "fn", "range_errs", "sin_az_errs"}`; the error lists hold one
    entry per matched pair (empty if nothing matched). Matching is on the surface range,
    the returned `range_errs` are centre-vs-centre (element 0 of both tuples) -- see the
    module docstring.
    """
    if criterion is None:
        criterion = MatchCriterion()
    detections = decode_detections(grid, pred_map, threshold=threshold)
    matches, unmatched_det, unmatched_gt = match_detections(detections, targets, criterion)

    range_errs = [abs(detections[di][0] - targets[gi][0]) for di, gi in matches]
    sin_az_errs = [abs(detections[di][1] - targets[gi][1]) for di, gi in matches]

    return {
        "tp": len(matches),
        "fp": len(unmatched_det),
        "fn": len(unmatched_gt),
        "range_errs": range_errs,
        "sin_az_errs": sin_az_errs,
    }


def _rmse(errs: Sequence[float]) -> float:
    """Root-mean-square of `errs`; NaN if `errs` is empty.

    NaN, not 0.0 (the pre-2026-08-16 convention), because an empty match set means the
    localization error is UNDEFINED, and 0.0 reads as "perfect". That misreading was not
    hypothetical: under the old absolute-threshold sweep, RMSE was measured at the sweep
    point nearest 0.5, and a detector whose confidence ceiling sat below 0.5 contributed
    no matched pairs there, so it reported `range_rmse_m = 0.000` -- indistinguishable
    from flawless ranging -- while its recall was correctly nonzero. RMSE is now measured
    at the same operating point as the rest of the metrics, which removes that specific
    trap, but the convention stands: a detector that matches nothing at all still has an
    empty set, and undefined must look undefined.
    """
    if not errs:
        return float("nan")
    return math.sqrt(sum(e * e for e in errs) / len(errs))


def _safe_ratio(numerator: int, denominator: int) -> float:
    """`numerator/denominator`, with the documented 0/0 -> 1.0 convention (see module notes)."""
    if denominator == 0:
        return 1.0
    return numerator / denominator


DEFAULT_CLASSES: Tuple[str, ...] = ("vehicle", "pedestrian")

#: Confidence floor defining the detection set that the PR curve is swept over, and the
#: operating point `AR` is reported at. This is the ONLY absolute score threshold in the
#: metric: everything above it is ranked by score, never re-thresholded. It matches the
#: floor of the old 0.1..0.9 sweep, so the set of detections being scored is unchanged
#: from the pre-2026-08-17 numbers -- only what is computed from them changed.
DEFAULT_SCORE_THRESHOLD: float = 0.1


def _rank_detections(scored_flags: Sequence[Tuple[float, bool]]) -> List[int]:
    """Indices of `scored_flags` ranked by score descending, false positives first on ties.

    `scored_flags` is `[(score, is_true_positive), ...]` pooled over the whole split.
    The tie-break is deliberately pessimistic (see the module docstring): a group of
    equal-scored detections contributes its end-of-group precision, so a detector whose
    scores saturate cannot collect AP from an ordering it never actually produced.
    """
    return sorted(range(len(scored_flags)),
                  key=lambda i: (-scored_flags[i][0], scored_flags[i][1]))


def _pr_curve(scored_flags: Sequence[Tuple[float, bool]], n_gt: int) -> Dict[str, List]:
    """Precision/recall down the pooled score ranking, plus the interpolated precision.

    Returns parallel lists `score`, `is_tp`, `precision`, `recall`, `precision_interp`,
    one entry per detection, in ranked order. `precision_interp[k] = max(precision[k:])`
    (the right-to-left running max), i.e. precision forced monotonically non-increasing
    in recall, which is what the VOC2010+/COCO AP integrates. Requires `n_gt > 0`.
    """
    order = _rank_detections(scored_flags)
    scores: List[float] = []
    is_tp: List[bool] = []
    precision: List[float] = []
    recall: List[float] = []
    tp_cum = 0
    for rank, i in enumerate(order, start=1):
        score, hit = scored_flags[i]
        if hit:
            tp_cum += 1
        scores.append(float(score))
        is_tp.append(bool(hit))
        precision.append(tp_cum / rank)
        recall.append(tp_cum / n_gt)

    interp = list(precision)
    for k in range(len(interp) - 2, -1, -1):
        if interp[k] < interp[k + 1]:
            interp[k] = interp[k + 1]

    return {"score": scores, "is_tp": is_tp, "precision": precision, "recall": recall,
            "precision_interp": interp}


def _interpolated_ap(curve: Dict[str, List], n_gt: int) -> float:
    """Area under the interpolated PR curve, from `_pr_curve`'s output. Requires n_gt > 0.

    `AP = (1/n_gt) * sum(precision_interp[k] for each rank k that is a true positive)`.
    Each true positive advances recall by exactly `1/n_gt` and each false positive by
    zero, so this is the rectangle-rule area under the interpolated curve; the recall
    band the detector never reaches contributes nothing, which is what makes missed
    ground truth cost AP and not only AR. Summing `n_gt` exact 1.0 terms makes the
    oracle case return exactly 1.0 (see the module docstring).
    """
    return sum(p for p, hit in zip(curve["precision_interp"], curve["is_tp"]) if hit) / n_gt


def _score_detections(
    detections_per_frame: Sequence[Sequence[Detection]],
    target_lists: Sequence[Sequence[Target]],
    criterion: MatchCriterion,
) -> Dict:
    """Match already-decoded detections against `target_lists` and score the whole split.

    Split out from `evaluate_dataset` so the per-class breakdown can re-match the SAME
    decoded detections against a filtered target list without re-decoding every frame.

    Returns `{"AP", "AR", "precision", "tp", "fp", "fn", "n_detections", "n_targets",
    "pr_curve", "range_errs", "sin_az_errs"}`. `pr_curve` is `None` when there is no
    ground truth (no recall axis exists to integrate over).
    """
    scored_flags: List[Tuple[float, bool]] = []
    range_errs: List[float] = []
    sin_az_errs: List[float] = []
    tp = fp = fn = 0

    for detections, targets in zip(detections_per_frame, target_lists):
        matches, unmatched_det, unmatched_gt = match_detections(detections, targets, criterion)
        matched_det = {di for di, _gi in matches}
        for di, det in enumerate(detections):
            scored_flags.append((det[2], di in matched_det))
        range_errs.extend(abs(detections[di][0] - targets[gi][0]) for di, gi in matches)
        sin_az_errs.extend(abs(detections[di][1] - targets[gi][1]) for di, gi in matches)
        tp += len(matches)
        fp += len(unmatched_det)
        fn += len(unmatched_gt)

    n_gt = tp + fn
    n_det = len(scored_flags)

    if n_gt == 0:
        # Nothing to detect anywhere in the split. Recall keeps the documented 0/0 -> 1.0
        # convention (there was nothing to miss); AP has no recall axis to integrate, so
        # it is 1.0 only if the detector also stayed silent, and 0.0 if it emitted
        # anything at all (every detection is a false positive).
        curve = None
        ap = 1.0 if n_det == 0 else 0.0
    else:
        curve = _pr_curve(scored_flags, n_gt)
        ap = _interpolated_ap(curve, n_gt)

    # Precision is UNDEFINED, not a vacuous 1.0, when the detector emitted nothing while
    # ground truth existed: 0/0 -> 1.0 there would print "perfect precision" for a silent
    # model. (Under the old threshold sweep this exact confusion inflated AP; AP no
    # longer derives from this number, but the number itself must still read honestly.)
    precision = float("nan") if (tp + fp == 0 and fn > 0) else _safe_ratio(tp, tp + fp)

    return {
        "AP": ap,
        "AR": _safe_ratio(tp, n_gt),
        "precision": precision,
        "tp": tp, "fp": fp, "fn": fn,
        "n_detections": n_det, "n_targets": n_gt,
        "pr_curve": curve,
        "range_errs": range_errs, "sin_az_errs": sin_az_errs,
    }


def evaluate_dataset(
    pred_maps: Sequence,
    target_lists: Sequence[Sequence[Target]],
    grid: LabelGrid,
    *,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    criterion: MatchCriterion = None,
    classes: Sequence[str] = DEFAULT_CLASSES,
) -> Dict:
    """Full-dataset detection evaluation: interpolated-PR AP + recall at one operating point.

    Every frame is decoded ONCE at `score_threshold` (the confidence floor -- the only
    absolute threshold in the metric). Detections are matched to ground truth per frame,
    greedily, in descending score order; the resulting TP/FP flags are pooled across the
    whole split and ranked by score to build the precision-recall curve.

    Returned keys
    -------------
    ``AP``
        All-points interpolated average precision (VOC2010+/COCO convention). See the
        module docstring for the exact integration and the tie-breaking rule.
    ``AR``
        Recall at a single stated operating point: all detections scoring above
        `score_threshold`. Nothing is averaged over thresholds -- that was the bug this
        replaced. ``AR_operating_point`` (human-readable) and ``score_threshold``
        (numeric) name that operating point in the result dict itself, so a stored
        metrics JSON is self-describing.
    ``precision``, ``tp``, ``fp``, ``fn``, ``n_detections``, ``n_targets``
        Raw pooled counts at the same operating point, so `AR` can be re-derived and
        sample size is never hidden.
    ``pr_curve``
        `{"score", "is_tp", "precision", "recall", "precision_interp"}`, parallel lists
        in ranked order -- the actual curve `AP` integrates, kept so the number can be
        audited or re-plotted without re-running the model. `None` when the split has no
        ground truth at all.
    ``range_rmse_m``, ``sin_az_rmse``
        Localization RMSE over all matched pairs at the SAME operating point (NaN if
        nothing matched -- see `_rmse`). `range_rmse_m` scores the REGRESSED object
        CENTRE against the true centre; matching itself happened on the surface (see
        "Matching criterion"), so a detector that finds every target but cannot size it
        keeps its AP/AR and pays here instead. Previously these were measured at the sweep
        point nearest 0.5, which sat above every real detector's score ceiling and so
        was structurally undefined; measuring at the operating point the rest of the
        metrics use removes that trap and the double-counting a multi-threshold sweep
        would otherwise cause.
    ``AP_<class>``, ``AR_<class>``, ``n_targets_<class>``
        Per-class breakdown for each name in `classes` (default `DEFAULT_CLASSES`),
        computed by re-matching the same decoded detections against a class-filtered
        target list. Pass `classes=()` to skip. See the module docstring's "Per-class
        AP/AR" section for the semantics and its pessimistic-precision caveat.

    The 0/0 -> 1.0 convention applies to `AR`/`precision` only where genuinely vacuous
    (no ground truth means nothing to miss; no detections means nothing to be wrong
    about). A silent detector facing real ground truth scores AP = AR = 0.0.
    """
    if criterion is None:
        criterion = MatchCriterion()

    # ONE decode pass over the split; the pooled and per-class scorings all re-match
    # these same detections (the per-class target filter changes what a detection can
    # match, never what the detector emitted).
    detections_per_frame = [decode_detections(grid, pred_map, threshold=score_threshold)
                            for pred_map in pred_maps]

    pooled = _score_detections(detections_per_frame, target_lists, criterion)

    result = {
        "AP": pooled["AP"],
        "AR": pooled["AR"],
        "AR_operating_point": (f"recall over all detections with score > {score_threshold:g} "
                               f"(single operating point, not a threshold average)"),
        "score_threshold": float(score_threshold),
        "precision": pooled["precision"],
        "tp": pooled["tp"],
        "fp": pooled["fp"],
        "fn": pooled["fn"],
        "n_detections": pooled["n_detections"],
        "n_targets": pooled["n_targets"],
        "pr_curve": pooled["pr_curve"],
        "range_rmse_m": _rmse(pooled["range_errs"]),
        "sin_az_rmse": _rmse(pooled["sin_az_errs"]),
    }

    # Per-class AP/AR: see module docstring's "Per-class AP/AR" section. `classes=()`
    # skips this entirely.
    for cls in classes:
        cls_target_lists = [[t for t in targets if t[2] == cls] for targets in target_lists]
        n_cls_targets = sum(len(ts) for ts in cls_target_lists)
        if n_cls_targets == 0:
            # Nothing of this class in the dataset -- NaN (undefined), not the pooled
            # 0/0 -> 1.0 vacuous convention, since there is genuinely nothing to score.
            result[f"AP_{cls}"] = float("nan")
            result[f"AR_{cls}"] = float("nan")
        else:
            cls_result = _score_detections(detections_per_frame, cls_target_lists, criterion)
            result[f"AP_{cls}"] = cls_result["AP"]
            result[f"AR_{cls}"] = cls_result["AR"]
        result[f"n_targets_{cls}"] = n_cls_targets

    return result
