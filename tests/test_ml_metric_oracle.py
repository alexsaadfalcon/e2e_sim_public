"""ORACLE TESTS FOR THE DETECTION METRIC ITSELF.

Why this file exists (2026-08-27 post-mortem). Four defects in the detection benchmark
survived every earlier review round -- per-commit inline review, a 5-angle fresh-context
batch review, a cold reader, and a statistics audit. They survived because every one of
those asked some version of "is this code correct?", and none asked:

    IF THE DETECTOR WERE PERFECT, WOULD THIS BENCHMARK SAY SO?

That question is cheap to ask and it finds the defects immediately. A metric is an
instrument, and an uncalibrated instrument makes every number downstream of it
meaningless no matter how correct the code that produced them is. These tests calibrate
the instrument against detectors whose right answer is known a priori:

* `test_metric_scores_a_perfect_detector_perfectly` -- detections placed exactly on the
  labels must score AP 1.0. This is the floor; it passed even when the metric was broken,
  which is precisely why it is not sufficient on its own.
* `test_detection_on_the_object_surface_still_matches` -- THE ONE THAT MATTERED. A real
  detector fires where the ENERGY is, and the simulator's default target model puts an
  object's energy on its visible footprint corners, not its centre. A detection anywhere
  on the object's own footprint is a correct detection of that object and must match it.
  Before the surface-azimuth fix, the metric matched range against the object's surface
  but azimuth against its centre, so a correct detection on a nearby car's corner scored
  as a false alarm AND a miss (0.24 in sin(azimuth) at 10 m, against a 0.06 tolerance).
* `test_unlabelled_real_objects_are_not_charged_as_false_alarms` -- the corpus
  deliberately leaves ~34% of placed objects out of the label set, so a detector was
  charged for correctly detecting a real object.

Keep this file adversarial to the metric, not to the detectors. Anything that measures a
detector belongs in the other metrics tests; what belongs HERE is any invariant of the
form "a detector that did the right thing must be scored as having done the right thing".
"""
from __future__ import annotations

import pytest
import torch

from e2e.ml.labels import LabelGrid
from e2e.ml.metrics import MatchCriterion, evaluate_dataset, match_detections

GRID = LabelGrid(n_range=128, n_azimuth=192, max_range_m=102.4)


def _pred_map(dets, grid=GRID):
    """A `[3, n_range, n_azimuth]` prediction map with one peak per `dets` entry.

    `evaluate_dataset` scores MAPS, not detection tuples -- it decodes each map through
    the same `decode_detections` a real detector's output goes through. Building the map
    (rather than hand-feeding tuples) keeps these oracles honest: they exercise the whole
    decode-then-match path the benchmark actually uses. Regression channels are left at
    zero, so a decoded detection sits at its cell centre -- within a cell of the
    requested position, which is far inside every tolerance asserted here.
    """
    m = torch.zeros((3, grid.n_range, grid.n_azimuth), dtype=torch.float32)
    for d in dets:
        ri = min(max(int(round(d[0] / grid.max_range_m * grid.n_range)), 0), grid.n_range - 1)
        ai = min(max(int(round((d[1] + 1.0) / 2.0 * grid.n_azimuth)), 0), grid.n_azimuth - 1)
        m[0, ri, ai] = float(d[2])
    return m


def _det(range_m, sin_az, score=1.0, surface_range_m=None):
    """Detection tuple `(range_m, sin_azimuth, score, surface_range_m)`."""
    return (range_m, sin_az, score, range_m if surface_range_m is None else surface_range_m)


def _tgt(range_m, sin_az, cls="vehicle", surface_range_m=None, half_extent_m=None):
    """Target tuple, optionally carrying the cross-range half-extent (5th element)."""
    base = (range_m, sin_az, cls, range_m if surface_range_m is None else surface_range_m)
    return base if half_extent_m is None else base + (half_extent_m,)


def test_metric_scores_a_perfect_detector_perfectly():
    """Detections exactly on the labels -> AP 1.0, no false alarms, no misses.

    The trivial calibration. It is here to fail loudly if the matcher, the PR
    integration or the decode floor is ever broken outright -- NOT as evidence the
    instrument is sound, since this passed throughout the period the metric was
    mis-scoring correct detections (see the surface test below).
    """
    targets = [[_tgt(20.0, 0.10), _tgt(35.0, -0.30), _tgt(12.0, 0.55)]]
    dets = [[_det(20.0, 0.10), _det(35.0, -0.30), _det(12.0, 0.55)]]
    maps = [_pred_map(dets[0])]

    pairs, unmatched_det, unmatched_tgt = match_detections(dets[0], targets[0])
    assert len(pairs) == 3
    assert unmatched_det == [] and unmatched_tgt == []

    m = evaluate_dataset(maps, targets, GRID, score_threshold=0.01)
    assert m["AP"] == pytest.approx(1.0)
    assert m["fp"] == 0 and m["fn"] == 0


@pytest.mark.parametrize(
    "range_m, half_extent_m",
    [
        (10.0, 2.38),   # a 4.4 x 1.8 m car broadside at 10 m -> 0.238 sin_az off centre
        (20.0, 2.38),   # ... 0.119 at 20 m, still far outside the 0.06 base tolerance
        (30.0, 2.38),   # ... 0.079 at 30 m
        (8.0, 0.45),    # a pedestrian: extent small enough that the base tolerance rules
    ],
)
def test_detection_on_the_object_surface_still_matches(range_m, half_extent_m):
    """A detection on the object's own footprint is a detection OF that object.

    This is the invariant the benchmark existed to measure and did not. The simulator
    splits an object's radar cross-section across its visible corners, so the brightest
    return from a nearby vehicle is displaced from the labelled centre in azimuth by
    roughly `half_extent_m / range_m`. Scoring that as a false alarm plus a miss
    penalizes the detector for being right, and it bites hardest at close range where
    targets are strongest.

    The target carries its cross-range half-extent, and the azimuth tolerance widens to
    the object's own angular extent when that exceeds the base tolerance.
    """
    offset = half_extent_m / range_m
    target = _tgt(range_m, 0.0, half_extent_m=half_extent_m)
    # A detector firing on the corner: displaced in azimuth, same range gate.
    detection = _det(range_m, offset)

    pairs, unmatched_det, unmatched_tgt = match_detections([detection], [target])
    assert pairs == [(0, 0)], (
        f"a detection {offset:.3f} in sin(az) off centre -- i.e. exactly on the surface "
        f"of an object with a {half_extent_m} m half-extent at {range_m} m -- was not "
        f"matched to it"
    )
    assert unmatched_det == [] and unmatched_tgt == []


def test_surface_tolerance_does_not_make_the_criterion_unbounded():
    """Widening azimuth to the object's extent must not match arbitrarily far detections.

    The fix must be a bounded, physically-motivated widening (the object's own angular
    size), not a loosening. A detection well beyond the footprint still misses, and the
    range tolerance is untouched.
    """
    target = _tgt(10.0, 0.0, half_extent_m=2.38)          # angular half-extent 0.238
    far_in_azimuth = _det(10.0, 0.60)                      # way outside the footprint
    far_in_range = _det(20.0, 0.0)                         # on boresight, wrong gate

    for det in (far_in_azimuth, far_in_range):
        pairs, unmatched_det, _ = match_detections([det], [target])
        assert pairs == [] and unmatched_det == [0]


def test_targets_without_an_extent_keep_the_base_tolerance():
    """3- and 4-element targets must behave exactly as before this change.

    Hand-built tuples and older corpora carry no extent; they must not silently acquire
    a wider criterion.
    """
    crit = MatchCriterion()
    target = _tgt(10.0, 0.0)                 # no half-extent
    just_outside = _det(10.0, crit.max_sin_az_err * 1.5)

    pairs, unmatched_det, _ = match_detections([just_outside], [target])
    assert pairs == [] and unmatched_det == [0]

    just_inside = _det(10.0, crit.max_sin_az_err * 0.5)
    pairs, _, _ = match_detections([just_inside], [target])
    assert pairs == [(0, 0)]


def test_unlabelled_real_objects_are_not_charged_as_false_alarms():
    """A detection on a real object the label set omits is neither a hit nor a miss.

    The corpus deliberately places clutter it does not label -- measured at ~34% of all
    placed objects on `benchmark_v1`. Charging those as false alarms means a detector is
    penalized for working, and it depresses precision at the TOP of the score ranking
    (real objects give strong returns), which is where average precision is most
    sensitive. Don't-care regions are the standard remedy.
    """
    targets = [[_tgt(20.0, 0.10)]]
    ignore = [[(35.0, -0.30)]]               # a real, unlabelled clutter object
    dets = [[_det(20.0, 0.10, score=0.9), _det(35.0, -0.30, score=0.8)]]
    maps = [_pred_map(dets[0])]

    baseline = evaluate_dataset(maps, targets, GRID, score_threshold=0.01)
    assert baseline["fp"] == 1, "precondition: without don't-care the clutter hit is an FA"

    scored = evaluate_dataset(maps, targets, GRID, score_threshold=0.01, ignore=ignore)
    assert scored["fp"] == 0, "a detection on a real unlabelled object must not be an FA"
    assert scored["tp"] == 1, "and it must not become a true positive either"
    assert scored["n_targets"] == 1


def test_ignore_regions_never_steal_a_real_target():
    """When a detection is near both a target and a don't-care region, the target wins."""
    targets = [[_tgt(20.0, 0.10)]]
    ignore = [[(20.2, 0.10)]]                # overlapping the real target
    dets = [[_det(20.0, 0.10)]]
    maps = [_pred_map(dets[0])]

    m = evaluate_dataset(maps, targets, GRID, score_threshold=0.01, ignore=ignore)
    assert m["tp"] == 1 and m["fp"] == 0 and m["fn"] == 0


def test_empty_ignore_reproduces_the_unignored_result():
    """`ignore=None` and `ignore=[[]]` must both be bit-identical to not passing it."""
    targets = [[_tgt(20.0, 0.10), _tgt(35.0, -0.30)]]
    dets = [[_det(20.0, 0.10), _det(35.0, -0.30), _det(50.0, 0.7)]]
    maps = [_pred_map(dets[0])]

    base = evaluate_dataset(maps, targets, GRID, score_threshold=0.01)
    for ig in (None, [[]]):
        other = evaluate_dataset(maps, targets, GRID, score_threshold=0.01, ignore=ig)
        for key in ("AP", "AR", "tp", "fp", "fn", "n_detections", "n_targets"):
            assert other[key] == base[key], f"{key} changed with ignore={ig!r}"
