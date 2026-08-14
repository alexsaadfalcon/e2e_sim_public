"""
Detection-evaluation metrics for the FMCW radar detection head, RADIal/FFTRadNet-style.

Adapted from the RADIal repo's evaluation protocol (`utils/metrics.py::GetFullMetrics`,
see the "Evaluation" section of the scout notes) to our polar `(range_m, sin_azimuth)`
label format (`e2e.ml.labels.LabelGrid`/`decode_detections`/`targets_in_grid`) instead of
RADIal's cartesian-box + IoU format.

What we keep from the reference protocol
-----------------------------------------
* A **confidence-threshold sweep** (default 0.1..0.9 in steps of 0.1, 9 points) at a
  *fixed* matching criterion -- this mirrors RADIal sweeping confidence at a fixed
  IoU=0.5, just with our own "close enough in (range, sin-azimuth)" criterion
  (`MatchCriterion`) standing in for IoU.
* `AP`/`AR` as the simple **mean of precision/recall over the threshold sweep**, not a
  standard interpolated-PR-curve average precision -- this matches RADIal's actual
  (non-standard) `mAP`/`mAR` definition, not the COCO/VOC one.

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
0/0 convention. The ROADMAP also floats a "recall-floor-restricted threshold set" as a
cheaper alternative to full PR-curve interpolation for a fixed-recall operating point;
that variant is not implemented here -- the roadmap entry describes it only as a
direction, not a concrete definition (thresholds computed how, floor per-class or
pooled, etc.), so it is left as a documented follow-up rather than guessed at.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from e2e.ml.labels import LabelGrid, decode_detections

Detection = Tuple[float, float, float]   # (range_m, sin_azimuth, score)
Target = Tuple[float, float, str]        # (range_m, sin_azimuth, object_class)


@dataclass(frozen=True)
class MatchCriterion:
    """Tolerance defining a "close enough" detection/ground-truth match.

    A detection matches a target iff both `|dr| <= max_range_err_m` AND
    `|dsin| <= max_sin_az_err`, expressed as a single normalized distance
    `max(|dr|/max_range_err_m, |dsin|/max_sin_az_err) <= 1.0` (see `match_detections`).
    """

    max_range_err_m: float = 2.0
    max_sin_az_err: float = 0.06   # ~3.4 deg near boresight; widens off-boresight (see module docstring)


def _normalized_distance(det: Detection, tgt: Target, criterion: MatchCriterion) -> float:
    dr = abs(det[0] - tgt[0]) / criterion.max_range_err_m
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
    `(range_m, sin_azimuth, object_class)` tuples.

    Returns `{"tp", "fp", "fn", "range_errs", "sin_az_errs"}`; the error lists hold one
    entry per matched pair (empty if nothing matched).
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
    """Root-mean-square of `errs`; 0.0 (documented convention) if `errs` is empty."""
    if not errs:
        return 0.0
    return math.sqrt(sum(e * e for e in errs) / len(errs))


def _safe_ratio(numerator: int, denominator: int) -> float:
    """`numerator/denominator`, with the documented 0/0 -> 1.0 convention (see module notes)."""
    if denominator == 0:
        return 1.0
    return numerator / denominator


DEFAULT_CLASSES: Tuple[str, ...] = ("vehicle", "pedestrian")


def evaluate_dataset(
    pred_maps: Sequence,
    target_lists: Sequence[Sequence[Target]],
    grid: LabelGrid,
    *,
    thresholds: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    criterion: MatchCriterion = None,
    classes: Sequence[str] = DEFAULT_CLASSES,
) -> Dict:
    """Full-dataset evaluation, mirroring RADIal's `GetFullMetrics` confidence sweep.

    For each threshold in `thresholds`, TP/FP/FN are accumulated as raw counts over
    *all* frames (not per-frame precision averaged over frames), then
    `precision = TP/(TP+FP)`, `recall = TP/(TP+FN)`. The 0/0 -> 1.0 convention applies
    only when it is genuinely vacuous (no detections AND no ground truth); a threshold
    where the model made no detections but ground truth existed has UNDEFINED precision
    (reported as NaN) and is excluded from the AP mean -- otherwise an under-confident
    model would collect free precision=1.0 at every threshold above its confidence
    ceiling and AP would overstate it (this deliberately diverges from a naive reading
    of RADIal's sweep; AR is unaffected and still exposes the missed recall).

    `AP`/`AR` are the plain mean of (defined) precision / recall over the threshold
    sweep (RADIal's own, non-interpolated, definition -- see module docstring).

    `range_rmse_m`/`sin_az_rmse` are computed over all matched pairs at a single
    representative threshold -- the sweep point closest to 0.5 (exactly 0.5 for the
    default `thresholds`) -- rather than across every threshold, since the same
    ground-truth/detection pair would otherwise be double-counted once per threshold at
    which it happens to match. Documented choice, not a RADIal behaviour (upstream does
    not report RMSE at all, only mean error at the default confidence floor).

    `classes` additionally reports `AP_<class>`/`AR_<class>`/`n_targets_<class>` for each
    class (default `DEFAULT_CLASSES = ("vehicle", "pedestrian")`) -- pass `()` to skip
    (e.g. to avoid recursing further from inside a per-class call). See the module
    docstring's "Per-class AP/AR" section for the exact semantics and its caveats.
    """
    if criterion is None:
        criterion = MatchCriterion()
    thresholds = list(thresholds)
    if not thresholds:
        raise ValueError("thresholds must be non-empty")
    mid_idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - 0.5))
    mid_threshold = thresholds[mid_idx]

    precision_per_threshold: Dict[float, float] = {}
    recall_per_threshold: Dict[float, float] = {}
    range_errs_mid: List[float] = []
    sin_az_errs_mid: List[float] = []

    for th in thresholds:
        tp = fp = fn = 0
        for pred_map, targets in zip(pred_maps, target_lists):
            result = evaluate_frame(pred_map, targets, grid, threshold=th, criterion=criterion)
            tp += result["tp"]
            fp += result["fp"]
            fn += result["fn"]
            if th == mid_threshold:
                range_errs_mid.extend(result["range_errs"])
                sin_az_errs_mid.extend(result["sin_az_errs"])
        # Precision is UNDEFINED at a threshold where the model made no detections
        # while ground truth existed (tp+fp == 0, fn > 0): counting it as 1.0 would
        # reward an under-confident model with vacuous perfect precision at every
        # threshold above its confidence ceiling, inflating AP while AR (correctly)
        # collapses. Such thresholds are excluded from the AP mean and reported as
        # NaN per-threshold. The genuinely-vacuous case (no detections AND no ground
        # truth, fn == 0) keeps the documented 0/0 -> 1.0 convention.
        if tp + fp == 0 and fn > 0:
            precision_per_threshold[th] = float("nan")
        else:
            precision_per_threshold[th] = _safe_ratio(tp, tp + fp)
        recall_per_threshold[th] = _safe_ratio(tp, tp + fn)

    defined = [p for p in precision_per_threshold.values() if not math.isnan(p)]
    # A model with no detections at ANY threshold has no defined precision at all;
    # report AP = 0.0 (it detected nothing) rather than dividing by zero.
    ap = sum(defined) / len(defined) if defined else 0.0
    ar = sum(recall_per_threshold.values()) / len(thresholds)

    # SMALL-SAMPLE CAVEAT (results-review finding): when a model makes almost no
    # detections, only 1-2 thresholds have defined precision and AP becomes the mean
    # of that razor-thin sample -- a single lucky confident detection can print
    # AP ~0.5-1.0 while recall is ~0. n_defined_precision_thresholds exposes exactly
    # how many sweep points the AP mean rests on; treat AP with a small count (and
    # always AP alongside AR) as unstable, not as model quality.
    result = {
        "AP": ap,
        "AR": ar,
        "n_defined_precision_thresholds": len(defined),
        "precision_per_threshold": precision_per_threshold,
        "recall_per_threshold": recall_per_threshold,
        "range_rmse_m": _rmse(range_errs_mid),
        "sin_az_rmse": _rmse(sin_az_errs_mid),
    }

    # Per-class AP/AR: see module docstring's "Per-class AP/AR" section. `classes=()`
    # (used for the recursive per-class call itself) skips this entirely.
    for cls in classes:
        cls_target_lists = [[t for t in targets if t[2] == cls] for targets in target_lists]
        n_cls_targets = sum(len(ts) for ts in cls_target_lists)
        if n_cls_targets == 0:
            # Nothing of this class in the dataset -- NaN (undefined), not the pooled
            # 0/0 -> 1.0 vacuous convention, since there is genuinely nothing to score.
            result[f"AP_{cls}"] = float("nan")
            result[f"AR_{cls}"] = float("nan")
        else:
            cls_result = evaluate_dataset(pred_maps, cls_target_lists, grid,
                                           thresholds=thresholds, criterion=criterion,
                                           classes=())
            result[f"AP_{cls}"] = cls_result["AP"]
            result[f"AR_{cls}"] = cls_result["AR"]
        result[f"n_targets_{cls}"] = n_cls_targets

    return result
