"""
Tests for `webapp.detector_scoreboard`: the TP/FP/FN scoreboard built from the same
per-frame detection/ground-truth structure `webapp.pipeline_runner.figures_from_outputs`
already has in hand (see that module's detector-figure section, ~1162-1215, and
`webapp.detector_scoreboard`'s own module docstring for the exact field names), plus the
offline PR-curve figure read from `e2e/ml/runs/beat_cfar.json`.

Reuses `e2e.ml.metrics.MatchCriterion`/`match_detections` (frozen, not edited here) for
every match -- these tests never hand-roll a second matcher, they only check that
`score_frames` pools what the real matcher returns.

CPU-only, no torch tensors needed: detections/ground truth here are plain
`(range_m, sin_azimuth, score[, surface_range_m])` tuples, exactly as
`e2e.ml.labels.decode_detections` returns them.
"""

import math

import pytest

from e2e.ml.metrics import MatchCriterion
from webapp import detector_scoreboard as ds

_C = MatchCriterion()  # the real, frozen tolerances -- never hardcoded here (see below)


# --------------------------------------------------------------------------------
# score_frames
# --------------------------------------------------------------------------------
def test_score_frames_on_off_and_outside_tolerance():
    """One frame, one target, three detections: exactly on the boundary (inside),
    comfortably inside, and just outside -- mirrors tests/test_ml_metrics.py's own
    boundary convention so this suite and the frozen matcher's suite agree on what
    'inside' means."""
    target = (10.0, 0.0, "vehicle")
    det_inside = (10.0, 0.0, 0.9, 10.0)             # dead on -> matches
    det_boundary = (10.0 + _C.max_range_err_m, 0.0, 0.8, 10.0 + _C.max_range_err_m)
    det_outside = (10.0 + _C.max_range_err_m + 0.5, 0.0, 0.5,
                   10.0 + _C.max_range_err_m + 0.5)

    # Greedy matching claims the nearest target first; test each detection against its
    # own single-target frame so all three are independently checked, not competing.
    scores = ds.score_frames(
        [[det_inside], [det_boundary], [det_outside]],
        [[target], [target], [target]],
    )
    f0, f1, f2 = scores["frames"]
    assert (f0["tp"], f0["fp"], f0["fn"]) == (1, 0, 0)
    assert (f1["tp"], f1["fp"], f1["fn"]) == (1, 0, 0), "boundary is inclusive"
    assert (f2["tp"], f2["fp"], f2["fn"]) == (0, 1, 1), "outside tolerance is a miss"
    assert scores["n_frames"] == 3 and scores["n_frames_scored"] == 3
    assert scores["cumulative"] == {"tp": 2, "fp": 1, "fn": 1}


def test_score_frames_cumulative_arithmetic_over_several_frames():
    target = (10.0, 0.0, "vehicle")
    hit = (10.0, 0.0, 0.9, 10.0)
    miss_det = (30.0, 0.0, 0.9, 30.0)   # far outside tolerance -> fp, and target -> fn

    detections_per_frame = [[hit], [hit], [miss_det], [hit]]
    gt_per_frame = [[target], [target], [target], [target]]
    scores = ds.score_frames(detections_per_frame, gt_per_frame)

    assert scores["cumulative"] == {"tp": 3, "fp": 1, "fn": 1}
    assert scores["n_frames_scored"] == 4
    assert scores["fa_per_frame"] == pytest.approx(1 / 4)
    assert scores["hit_rate"] == pytest.approx(3 / 4)


def test_score_frames_frame_with_no_ground_truth_at_all_is_unscored():
    """`gt_per_frame=None` (no frame in the run carried labels, e.g. a live-traced run)
    marks every frame unscored rather than inventing a score."""
    scores = ds.score_frames([[(10.0, 0.0, 0.9, 10.0)]], None)
    assert scores["frames"][0]["scored"] is False
    assert scores["n_frames_scored"] == 0
    assert scores["cumulative"] == {"tp": 0, "fp": 0, "fn": 0}
    assert math.isnan(scores["fa_per_frame"])
    assert math.isnan(scores["hit_rate"])


def test_score_frames_frame_with_empty_ground_truth_list_is_scored():
    """A per-frame `[]` (labels present, no target that frame) IS scored: every
    detection on it is a false positive -- distinct from `gt_per_frame=None` above."""
    scores = ds.score_frames([[(10.0, 0.0, 0.9, 10.0)]], [[]])
    f0 = scores["frames"][0]
    assert f0["scored"] is True
    assert (f0["tp"], f0["fp"], f0["fn"]) == (0, 1, 0)
    assert scores["fa_per_frame"] == pytest.approx(1.0)


def test_score_frames_frame_with_no_detections():
    target = (10.0, 0.0, "vehicle")
    scores = ds.score_frames([[]], [[target]])
    f0 = scores["frames"][0]
    assert (f0["tp"], f0["fp"], f0["fn"]) == (0, 0, 1)


def test_score_frames_rejects_misaligned_gt_length():
    with pytest.raises(ValueError, match="frame-aligned"):
        ds.score_frames([[], []], [[]])  # 2 frames of detections, 1 of ground truth


def test_score_frames_threshold_is_echoed_not_reused_for_decoding():
    scores = ds.score_frames([[]], None, threshold=0.42)
    assert scores["threshold"] == 0.42


# --------------------------------------------------------------------------------
# match_rule_text
# --------------------------------------------------------------------------------
def test_match_rule_text_carries_metrics_numbers_not_typed_here():
    text = ds.match_rule_text()
    assert f"{_C.max_range_err_m:g}" in text
    assert f"{_C.max_sin_az_err:g}" in text
    assert "sin(azimuth)" in text and "range" in text


# --------------------------------------------------------------------------------
# scoreboard_figure
# --------------------------------------------------------------------------------
def _table(fig):
    assert len(fig.data) == 1
    table = fig.data[0]
    assert table.type == "table"
    return table


def test_scoreboard_figure_shows_last_frame_and_cumulative_numbers():
    target = (10.0, 0.0, "vehicle")
    hit = (10.0, 0.0, 0.9, 10.0)
    miss = (30.0, 0.0, 0.9, 30.0)
    scores = ds.score_frames([[hit], [miss]], [[target], [target]])
    fig = ds.scoreboard_figure(scores, arm_name="classical CFAR", threshold=0.5,
                               match_rule_text=ds.match_rule_text())
    table = _table(fig)
    labels, values = table.cells.values
    row = dict(zip(labels, values))
    # Last frame (index 1) was a total miss: tp=0, fp=1, fn=1.
    assert row["this frame: TP"] == "0"
    assert row["this frame: FP"] == "1"
    assert row["this frame: FN"] == "1"
    # Cumulative over both frames: 1 hit, 1 false alarm.
    hits_key = next(k for k in row if k.startswith("cumulative hits"))
    assert row[hits_key] == "1"
    assert row["cumulative false alarms"] == "1"
    assert row["FA / frame"] == "0.50"
    assert row["hit rate"] == "0.50"
    assert "classical CFAR" in table.header.values[0]
    assert "0.50" in table.header.values[0]


def test_scoreboard_figure_no_scored_frames_reads_na_not_zero():
    scores = ds.score_frames([[(10.0, 0.0, 0.9, 10.0)]], None)
    fig = ds.scoreboard_figure(scores, arm_name="ML", threshold=0.5,
                               match_rule_text="rule")
    table = _table(fig)
    _labels, values = table.cells.values
    assert values[:3] == ["n/a", "n/a", "n/a"]


def test_scoreboard_figure_fonts_are_legible_at_distance():
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="ML", threshold=0.5,
                               match_rule_text="rule")
    table = _table(fig)
    assert table.header.font.size >= 18
    assert table.cells.font.size >= 18
    # 3 (this-frame) + 4 (cumulative) numbers -- at most 8, per spec.
    _labels, values = table.cells.values
    assert len(values) <= 8


# --------------------------------------------------------------------------------
# stored_pr_figure / arm_name_for_detector -- real beat_cfar.json
# --------------------------------------------------------------------------------
@pytest.fixture
def beat_cfar_data():
    import json
    return json.loads(ds.DEFAULT_BEAT_CFAR_JSON.read_text())


def test_default_beat_cfar_json_exists():
    assert ds.DEFAULT_BEAT_CFAR_JSON.is_file()


def test_stored_pr_figure_one_trace_per_arm(beat_cfar_data):
    fig = ds.stored_pr_figure(highlight_arm="classical CFAR")
    assert len(fig.data) == len(beat_cfar_data["arms"])
    names = [tr.name for tr in fig.data]
    assert any(n.startswith("classical CFAR") for n in names)
    assert "beat_cfar.json" in fig.layout.title.text
    assert str(len(beat_cfar_data["arms"][0]["gt_per_frame"])) in fig.layout.title.text


def test_stored_pr_figure_highlights_bold(beat_cfar_data):
    fig = ds.stored_pr_figure(highlight_arm="classical CFAR")
    widths = {tr.name.split(" (")[0]: tr.line.width for tr in fig.data
             if tr.mode == "lines"}
    assert widths["classical CFAR"] > max(
        w for name, w in widths.items() if name != "classical CFAR")


def test_stored_pr_figure_fonts_are_legible():
    fig = ds.stored_pr_figure()
    assert fig.layout.title.font.size >= 16
    assert fig.layout.font.size >= 16


def test_stored_pr_figure_fallback_when_arm_lacks_pr_curve(tmp_path):
    """An arm with no `pr_curve` falls back to its recall-0.5 operating point, with AP
    in the legend, and the figure says so -- never invented, never silently dropped."""
    import json
    data = {
        "manifest": "e2e/ml/datasets/x/y/manifest.json",
        "arms": [
            {"name": "classical CFAR", "AP": 0.3, "gt_per_frame": [1, 2, 3],
             "pr_curve": {"recall": [0.1, 0.5], "precision": [0.9, 0.5]}},
            {"name": "no curve arm", "AP": 0.2, "gt_per_frame": [1, 2, 3],
             "operating_point": {"reached": True, "recall_achieved": 0.5,
                                 "target_recall": 0.5, "tp": 5, "fp": 5}},
        ],
    }
    path = tmp_path / "beat_cfar.json"
    path.write_text(json.dumps(data))
    fig = ds.stored_pr_figure(path)
    assert len(fig.data) == 2
    marker_trace = next(tr for tr in fig.data if tr.mode == "markers")
    assert "no curve arm" in marker_trace.name
    assert "recall-0.5 pt only" in marker_trace.name
    ann_texts = " ".join(a.text for a in fig.layout.annotations)
    assert "no curve arm" in ann_texts and "no stored PR curve" in ann_texts


def test_stored_pr_figure_raises_on_missing_arms_key(tmp_path):
    import json
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"nope": True}))
    with pytest.raises(ValueError, match="arms"):
        ds.stored_pr_figure(path)


def test_arm_name_for_detector_cfar_mode():
    assert ds.arm_name_for_detector({"mode": "cfar"}) == "classical CFAR"


def test_arm_name_for_detector_ml_mode_maps_checkpoint_parent_dir(beat_cfar_data):
    ckpt_arm = next(a for a in beat_cfar_data["arms"] if a.get("checkpoint"))
    from pathlib import Path
    label = Path(ckpt_arm["checkpoint"]).parent.name
    assert ds.arm_name_for_detector({"mode": "ml", "label": label}) == ckpt_arm["name"]


def test_arm_name_for_detector_ml_mode_unknown_checkpoint_is_none():
    assert ds.arm_name_for_detector({"mode": "ml", "label": "not_a_real_run"}) is None
