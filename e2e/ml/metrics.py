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

WHICH azimuth (2026-08-27): the surface-range fix above only widened the RANGE axis; the
azimuth axis still compared detection to target CENTRE against the fixed
`max_sin_az_err=0.06`, and the default target model splits an extended object's RCS
across its visible footprint CORNERS -- offset from the centre in azimuth as well as
range. For a 4.4 m car that corner-to-centre offset is 0.238 sin_az at 10 m and 0.119 at
20 m, both far outside 0.06, so a detector that correctly locks onto the brightest corner
was scored as a miss AND a false alarm. Targets optionally carry a 5th element,
`cross_range_half_extent_m` (`_half_extent`, mirroring `_surface_range` -- 0.0, i.e.
today's behaviour, if absent); the effective azimuth tolerance used per-target is
`max(criterion.max_sin_az_err, half_extent_m / max(surface_range_m, eps))`, i.e. the
fixed tolerance widens only as far as the target's own angular half-extent demands, using
its SURFACE range (the same range the corner geometry itself lives at) rather than its
centre range. Like the range axis, this is a WIDENING keyed to the object's own physical
size, not a global loosening: a point target (no known extent) gets exactly the old
0.06 tolerance.

Don't-care / ignore regions (KITTI-style, 2026-08-27)
------------------------------------------------------
The corpus deliberately leaves ~34% of placed objects (clutter) out of the label set, so
a detector that correctly locates one of them was being charged a false alarm for
detecting something real. `match_detections`/`evaluate_frame`/`evaluate_dataset` all take
a keyword-only `ignore` -- a list of `(range_m, sin_azimuth)` positions of real-but-
unlabelled objects (one list per frame in `evaluate_dataset`, aligned with
`target_lists`). A detection that matches no TARGET but falls within the same
`MatchCriterion` distance of an ignore entry is dropped entirely: neither a true positive
nor a false positive, and it does not enter the pooled PR curve (mirroring KITTI's
`DontCare` regions). Targets always win the match first -- a detection close to both a
real target and an ignore entry scores as the true positive it is, never a dropped
don't-care. `ignore=None`/empty reproduces today's numbers bit-for-bit: the "which
detections get scored" set only ever shrinks, and only relative to the ignore-less
baseline.

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
# (range_m, sin_azimuth, object_class[, surface_range_m[, cross_range_half_extent_m]]) --
# see `labels.targets_in_grid`
Target = Tuple


def _surface_range(item) -> float:
    """The SURFACE range of a detection/target tuple; its centre range if it has none.

    `e2e.ml.labels` appends a 4th element (the surface range) to both tuple kinds; a
    3-element tuple is a POINT target, whose surface and centre coincide -- so this is
    also what keeps hand-built `(range, sin_az, score)` / `(range, sin_az, class)` tuples
    behaving exactly as they did before 2026-08-17.
    """
    return float(item[3]) if len(item) > 3 else float(item[0])


def _half_extent(tgt: Target) -> float:
    """The target's CROSS-RANGE half-extent in metres; 0.0 (a point target) if absent.

    Mirrors `_surface_range`'s indexing trick so an optional trailing element cannot break
    callers: `e2e.ml.labels` appends `cross_range_half_extent_m` as a 5th element only for
    targets built from an extended object; a bare `(range, sin_az, class[, surface_range])`
    tuple -- including every hand-built target predating 2026-08-27 -- has no 5th slot and
    is treated as having zero angular extent, i.e. today's behaviour exactly (see
    `_normalized_distance` and the module docstring's "WHICH azimuth" section). Also used
    on `ignore` entries (`(range_m, sin_azimuth)` 2-tuples), which likewise have no extent.
    """
    return float(tgt[4]) if len(tgt) > 4 else 0.0


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


#: Floor for the range used to convert `_half_extent` (metres) into a sin-azimuth
#: tolerance, so a target placed (or hand-built) at r=0 cannot divide by zero.
_MIN_TOLERANCE_RANGE_M = 1e-6


def _normalized_distance(det: Detection, tgt: Target, criterion: MatchCriterion) -> float:
    """Match distance, computed on the SURFACE range (see `_surface_range`).

    The azimuth tolerance widens per-target to `tgt`'s own angular half-extent (see the
    module docstring's "WHICH azimuth" section) -- never narrows, since
    `criterion.max_sin_az_err` is already a floor for point targets.
    """
    dr = abs(_surface_range(det) - _surface_range(tgt)) / criterion.max_range_err_m
    az_tol = max(criterion.max_sin_az_err,
                _half_extent(tgt) / max(_surface_range(tgt), _MIN_TOLERANCE_RANGE_M))
    ds = abs(det[1] - tgt[1]) / az_tol
    return max(dr, ds)


def _drop_ignored(detections: Sequence[Detection], unmatched_det: List[int],
                  ignore: Sequence[Tuple[float, float]], criterion: MatchCriterion) -> List[int]:
    """Remove from `unmatched_det` any detection that falls within `criterion` of an
    ignore entry (see the module docstring's "Don't-care / ignore regions" section).

    An ignore entry is a bare `(range_m, sin_azimuth)` pair -- `_surface_range`/
    `_half_extent` read it as a point target with no known extent, so it gets exactly
    `criterion`'s fixed tolerances, not the widened per-target one.
    """
    return [di for di in unmatched_det
           if not any(_normalized_distance(detections[di], entry, criterion) <= 1.0
                      for entry in ignore)]


def match_detections(
    detections: Sequence[Detection],
    targets: Sequence[Target],
    criterion: MatchCriterion = None,
    *,
    ignore: Sequence[Tuple[float, float]] = None,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Greedily match `detections` to `targets` under `criterion`.

    Processes detections in descending score order (ties keep their original relative
    order -- Python's sort is stable, including under `reverse=True`); each detection
    claims the *nearest* still-unmatched target within the criterion (normalized
    distance <= 1.0, i.e. the boundary itself counts as a match), so once a target is
    claimed it cannot be claimed again by a later (lower-score) detection.

    `ignore`, if given, is a list of `(range_m, sin_azimuth)` real-but-unlabelled
    positions (see the module docstring's "Don't-care / ignore regions" section).
    Targets are matched FIRST, unconditionally -- `ignore` is only ever consulted for a
    detection that already failed to match any target -- and such a detection is dropped
    from `unmatched_det` entirely rather than counted, so it is neither a false positive
    nor (not having matched a target) a true positive. `None`/empty is a no-op: this
    reproduces the pre-2026-08-27 return value bit-for-bit.

    Returns
    -------
    matches : list of (detection_index, target_index)
    unmatched_det : list of detection indices with no match (false positives), MINUS any
        that matched an `ignore` entry instead (dropped, not false positives)
    unmatched_gt : list of target indices with no match (false negatives)
    All index lists are sorted ascending (i.e. in original list order).
    """
    if criterion is None:
        criterion = MatchCriterion()
    if not detections:
        return [], [], list(range(len(targets)))
    if not targets:
        unmatched_det = list(range(len(detections)))
        if ignore:
            unmatched_det = _drop_ignored(detections, unmatched_det, ignore, criterion)
        return [], unmatched_det, []

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
    if ignore:
        unmatched_det = _drop_ignored(detections, unmatched_det, ignore, criterion)
    unmatched_gt = sorted(set(range(len(targets))) - matched_gt)
    return matches, unmatched_det, unmatched_gt


def evaluate_frame(
    pred_map,
    targets: Sequence[Target],
    grid: LabelGrid,
    *,
    threshold: float,
    criterion: MatchCriterion = None,
    ignore: Sequence[Tuple[float, float]] = None,
) -> Dict:
    """Decode `pred_map` at `threshold`, match against `targets`, score one frame.

    `pred_map` is anything `decode_detections` accepts (a `[3, n_range, n_azimuth]`
    label/prediction tensor). `targets` is the `targets_in_grid`-style list of
    `(range_m, sin_azimuth, object_class[, surface_range_m[, cross_range_half_extent_m]])`
    tuples. `ignore` is the don't-care list for this frame (see the module docstring's
    "Don't-care / ignore regions" section); `None`/empty is a no-op.

    Returns `{"tp", "fp", "fn", "range_errs", "sin_az_errs"}`; the error lists hold one
    entry per matched pair (empty if nothing matched). Matching is on the surface range,
    the returned `range_errs` are centre-vs-centre (element 0 of both tuples) -- see the
    module docstring. `fp` already excludes any detection dropped as a don't-care.
    """
    if criterion is None:
        criterion = MatchCriterion()
    detections = decode_detections(grid, pred_map, threshold=threshold)
    matches, unmatched_det, unmatched_gt = match_detections(detections, targets, criterion,
                                                             ignore=ignore)

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


def _pr_curve(scored_flags: Sequence[Tuple[float, bool]], n_gt: int,
              scored_frames: Sequence[int] = None) -> Dict[str, List]:
    """Precision/recall down the pooled score ranking, plus the interpolated precision.

    Returns parallel lists `score`, `is_tp`, `precision`, `recall`, `precision_interp`,
    one entry per detection, in ranked order -- plus `frame` when `scored_frames` is
    given, carrying each detection's source frame THROUGH the ranking permutation so
    `frame[k]` still belongs to `score[k]`. That is what makes a paired, scene-level
    bootstrap possible from a stored curve; without it the pooled curve cannot be
    resampled by scene at all (the reason two README CIs were withdrawn in 0efb7e4).

    `precision_interp[k] = max(precision[k:])`
    (the right-to-left running max), i.e. precision forced monotonically non-increasing
    in recall, which is what the VOC2010+/COCO AP integrates. Requires `n_gt > 0`.
    """
    order = _rank_detections(scored_flags)
    scores: List[float] = []
    is_tp: List[bool] = []
    precision: List[float] = []
    recall: List[float] = []
    frames: List[int] = []
    tp_cum = 0
    for rank, i in enumerate(order, start=1):
        score, hit = scored_flags[i]
        if hit:
            tp_cum += 1
        scores.append(float(score))
        is_tp.append(bool(hit))
        if scored_frames is not None:
            frames.append(int(scored_frames[i]))
        precision.append(tp_cum / rank)
        recall.append(tp_cum / n_gt)

    interp = list(precision)
    for k in range(len(interp) - 2, -1, -1):
        if interp[k] < interp[k + 1]:
            interp[k] = interp[k + 1]

    out = {"score": scores, "is_tp": is_tp, "precision": precision, "recall": recall,
           "precision_interp": interp}
    if scored_frames is not None:
        out["frame"] = frames
    return out


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


def false_alarms_at_recall(pr_curve, n_frames: int, *,
                           target_recall: float = 0.5) -> Dict:
    """False alarms per frame at the score threshold that first reaches `target_recall`.

    Why this metric rather than AP or "Pd at threshold t"
    -----------------------------------------------------
    Both of those let the threshold do the arguing. A detector that fires everywhere
    reports high recall; one that barely fires reports high precision; and AP compresses
    the whole trade into a single number that hides WHERE on the curve a detector is
    usable. Holding every detector at the same recall and counting what it costs is the
    comparison an operator actually faces, and it cannot be gamed: moving the threshold
    moves the recall, and the operating point simply re-anchors somewhere else on the
    same curve.

    Reading the operating point off the curve
    -----------------------------------------
    `pr_curve` is `_pr_curve`'s output -- detections ranked by score descending, with
    `recall` cumulative down that ranking. Recall is non-decreasing, so the FIRST rank
    reaching `target_recall` is also the one with the fewest false positives, and the
    score at that rank is the threshold an operator would set.

    That rank is then extended to the END of its equal-score group, because a real
    threshold at `score[k]` admits every detection scoring exactly `score[k]`, not just
    the ones the ranking happened to place first. `_rank_detections` orders false
    positives ahead of true positives within a tie, so the extension can only add true
    positives: the reported false-alarm count is unchanged and the reported recall is the
    honest one for that threshold.

    Requires a curve that REACHES the target
    ----------------------------------------
    The curve only spans the recall its detections cover, and those detections were
    decoded at some confidence floor upstream. Decode at a permissive floor (0.01, say)
    before asking for a low-recall operating point, or the curve will stop short of it.
    When it does stop short this returns `reached=False` and `fp_per_frame=NaN` rather
    than the false alarms at whatever recall it did manage -- silently answering a
    different question than the one asked is exactly the failure this metric exists to
    avoid. `recall_max` reports how far it got, so the caller can say so.

    Parameters
    ----------
    pr_curve
        `_pr_curve` output, i.e. `evaluate_dataset(...)["pr_curve"]`. `None` (a split
        with no ground truth) yields an all-NaN result with `reached=False`.
    n_frames
        Frames the curve was pooled over -- the denominator. Must be positive.
    target_recall
        The recall to hold every detector at. Must be in (0, 1].

    Returns
    -------
    dict with
        ``reached``           did the curve get to `target_recall`
        ``target_recall``     what was asked for (echoed, so a stored result self-describes)
        ``recall_achieved``   recall at the chosen operating point (NaN if not reached)
        ``recall_max``        the most the curve ever reached
        ``score_threshold``   the score defining that operating point (NaN if not reached)
        ``fp``, ``tp``        pooled counts at it (None if not reached)
        ``fp_per_frame``      `fp / n_frames` -- the headline number (NaN if not reached)
        ``n_frames``          echoed denominator
    """
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames}")
    if not 0.0 < target_recall <= 1.0:
        raise ValueError(f"target_recall must be in (0, 1], got {target_recall}")

    nan = float("nan")
    miss = {"reached": False, "target_recall": float(target_recall),
            "recall_achieved": nan, "recall_max": 0.0, "score_threshold": nan,
            "fp": None, "tp": None, "fp_per_frame": nan, "n_frames": int(n_frames)}

    if not pr_curve or not pr_curve.get("recall"):
        return miss

    recall = pr_curve["recall"]
    is_tp = pr_curve["is_tp"]
    score = pr_curve["score"]
    recall_max = float(recall[-1])          # non-decreasing, so the last entry is the max

    k = next((i for i, r in enumerate(recall) if r >= target_recall), None)
    if k is None:
        return {**miss, "recall_max": recall_max}

    # Extend through the equal-score group: a threshold at score[k] admits all of it.
    j = k
    while j + 1 < len(score) and score[j + 1] == score[k]:
        j += 1

    tp = sum(1 for hit in is_tp[:j + 1] if hit)
    fp = (j + 1) - tp
    return {
        "reached": True,
        "target_recall": float(target_recall),
        "recall_achieved": float(recall[j]),
        "recall_max": recall_max,
        "score_threshold": float(score[j]),
        "fp": int(fp),
        "tp": int(tp),
        "fp_per_frame": fp / float(n_frames),
        "n_frames": int(n_frames),
    }


def _score_detections(
    detections_per_frame: Sequence[Sequence[Detection]],
    target_lists: Sequence[Sequence[Target]],
    criterion: MatchCriterion,
    ignore_per_frame: Sequence[Sequence[Tuple[float, float]]] = None,
) -> Dict:
    """Match already-decoded detections against `target_lists` and score the whole split.

    Split out from `evaluate_dataset` so the per-class breakdown can re-match the SAME
    decoded detections against a filtered target list without re-decoding every frame.
    `ignore_per_frame`, if given, is one don't-care list per frame, aligned with
    `target_lists` -- applied to BOTH the pooled and the per-class re-matching, since a
    real-but-unlabelled object doesn't stop being real just because the target list was
    narrowed to one class.

    Returns `{"AP", "AR", "precision", "tp", "fp", "fn", "n_detections", "n_targets",
    "pr_curve", "range_errs", "sin_az_errs"}`. `pr_curve` is `None` when there is no
    ground truth (no recall axis exists to integrate over). `n_detections`/`fp` exclude
    any detection dropped as a don't-care -- it is neither.
    """
    if ignore_per_frame is None:
        ignore_per_frame = [None] * len(detections_per_frame)
    scored_flags: List[Tuple[float, bool]] = []
    # Parallel to `scored_flags`: the frame each scored detection came from, and the
    # per-frame ground-truth count. Both are what a PAIRED, SCENE-LEVEL bootstrap
    # needs -- resampling frames with replacement and recomputing AP requires knowing
    # which detections and how much ground truth move together. Pooling without this
    # is what made the withdrawn README CIs unreproducible (commit 0efb7e4).
    scored_frames: List[int] = []
    gt_per_frame: List[int] = []
    range_errs: List[float] = []
    sin_az_errs: List[float] = []
    tp = fp = fn = 0

    for frame_idx, (detections, targets, ignore) in enumerate(
            zip(detections_per_frame, target_lists, ignore_per_frame)):
        matches, unmatched_det, unmatched_gt = match_detections(detections, targets, criterion,
                                                                 ignore=ignore)
        matched_det = {di for di, _gi in matches}
        # Score every detection that is either a match or a surviving false positive --
        # NOT a plain enumerate, since a don't-care detection is in neither set and must
        # not enter the pooled PR curve at all (see `match_detections`' docstring).
        for di in sorted(matched_det | set(unmatched_det)):
            scored_flags.append((detections[di][2], di in matched_det))
            scored_frames.append(frame_idx)
        gt_per_frame.append(len(targets))
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
        curve = _pr_curve(scored_flags, n_gt, scored_frames)
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
        # Per-frame ground-truth counts, parallel to the input frame order. A scene-level
        # bootstrap needs these: resampling frames with replacement changes the recall
        # denominator, and AP is not recomputable from the curve alone without them.
        "gt_per_frame": gt_per_frame,
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
    ignore: Sequence[Sequence[Tuple[float, float]]] = None,
    max_range_m: float = None,
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

    `ignore`, if given, is a per-frame sequence of don't-care lists (one per entry of
    `target_lists`, each a list of `(range_m, sin_azimuth)`) -- see the module
    docstring's "Don't-care / ignore regions" section. `None`/empty reproduces the
    pre-2026-08-27 result bit-for-bit.
    """
    if criterion is None:
        criterion = MatchCriterion()
    if ignore is None:
        ignore = [None] * len(pred_maps)

    # ONE decode pass over the split; the pooled and per-class scorings all re-match
    # these same detections (the per-class target filter changes what a detection can
    # match, never what the detector emitted).
    detections_per_frame = [decode_detections(grid, pred_map, threshold=score_threshold)
                            for pred_map in pred_maps]

    # SCORED SWATH. The label grid spans the radar's full unambiguous range, but a corpus
    # need not put targets across all of it -- on benchmark_v1/D2 nothing sits beyond
    # ~34 m of a 102 m grid. Scoring the empty remainder charges every detector for false
    # alarms in a region where a hit is impossible by construction, which is a property of
    # the corpus rather than of the detector. `max_range_m` crops BOTH sides of the
    # comparison (detections and ground truth) at the SURFACE range, which is what
    # matching uses. Default None scores the whole grid, so no stored number moves unless
    # a caller asks.
    if max_range_m is not None:
        limit = float(max_range_m)
        if not limit > 0.0:
            raise ValueError(f"max_range_m must be positive, got {max_range_m!r}")
        detections_per_frame = [[d for d in dets if _surface_range(d) < limit]
                                for dets in detections_per_frame]
        target_lists = [[tgt for tgt in targets if _surface_range(tgt) < limit]
                        for targets in target_lists]
        if ignore is not None:
            ignore = [None if ig is None else [g for g in ig if float(g[0]) < limit]
                      for ig in ignore]

    pooled = _score_detections(detections_per_frame, target_lists, criterion, ignore)

    result = {
        "AP": pooled["AP"],
        "AR": pooled["AR"],
        "AR_operating_point": (f"recall over all detections with score > {score_threshold:g} "
                               f"(single operating point, not a threshold average)"),
        "score_threshold": float(score_threshold),
        # Self-describing, like score_threshold: a stored metrics JSON must say what
        # swath it scored, or two runs are not comparable and nothing on disk says why.
        "max_range_m": None if max_range_m is None else float(max_range_m),
        "precision": pooled["precision"],
        "tp": pooled["tp"],
        "fp": pooled["fp"],
        "fn": pooled["fn"],
        "n_detections": pooled["n_detections"],
        "n_targets": pooled["n_targets"],
        "pr_curve": pooled["pr_curve"],
        # Parallel to the (post-crop) frame order, so a scene-level bootstrap can resample
        # frames and recompute the recall denominator. Together with pr_curve["frame"]
        # this is everything a paired CI needs from a stored artifact.
        "gt_per_frame": pooled["gt_per_frame"],
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
            cls_result = _score_detections(detections_per_frame, cls_target_lists, criterion,
                                           ignore)
            result[f"AP_{cls}"] = cls_result["AP"]
            result[f"AR_{cls}"] = cls_result["AR"]
        result[f"n_targets_{cls}"] = n_cls_targets

    return result
