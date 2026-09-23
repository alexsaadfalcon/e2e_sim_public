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
    # Renamed (Change 1b, 2026-09-23): "hit rate" alone read as a cross-detector
    # quality ranking even though every detector's threshold is independently
    # calibrated to land near recall 0.5 -- the label now says so in place.
    assert row["hit rate (matched ~0.5 by design)"] == "0.50"
    # Threshold moved out of the header (a long "{arm} -- threshold {thr}" string
    # wrapped to two lines inside the header's declared height and clipped the
    # table's last row, see `_TABLE_HEADER_HEIGHT`'s comment) and into the title.
    assert table.header.values[0] == "classical CFAR"
    assert table.header.values[1] == "count"
    assert "0.50" in fig.layout.title.text


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


def test_scoreboard_figure_rows_never_clip_regardless_of_arm_name_length():
    """The bug this pins: a long header ("CA-CFAR (guard 2, train 6) -- threshold
    0.66") wrapped to two lines inside its declared single-line height, stealing
    room from the bottom of the table and clipping the last ("hit rate") row --
    CFAR and the neural-detector arms run the same code and must show identical
    rows (rehearsal, 2026-09-23). Checked geometrically (the domain the figure's
    own height/margin leaves for the table must fit header + all data rows), since
    a Plotly figure object carries no rendered pixel truth to assert on directly."""
    scores = ds.score_frames([[]], None)
    long_name = "a very long arm name, e.g. an ML checkpoint's parent directory"
    fig = ds.scoreboard_figure(scores, arm_name=long_name, threshold=0.5,
                               match_rule_text="rule")
    table = _table(fig)
    domain_height = fig.layout.height - fig.layout.margin.t - fig.layout.margin.b
    content_height = table.header.height + len(table.cells.values[0]) * table.cells.height
    assert domain_height >= content_height
    # All 7 rows are always present in the underlying data, for every arm -- the
    # rendering bug above was purely geometric, not a difference in what is computed.
    labels, _values = table.cells.values
    assert labels == ["this frame: TP", "this frame: FP", "this frame: FN",
                      "cumulative hits (0/1 frames scored)",
                      "cumulative false alarms", "FA / frame",
                      "hit rate (matched ~0.5 by design)"]


def test_scoreboard_figure_header_count_label_and_threshold_in_title():
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="CA-CFAR (guard 2, train 6)",
                               threshold=0.66, match_rule_text="rule")
    table = _table(fig)
    # The header's second column used to be an empty dark cell.
    assert table.header.values[1] == "count"
    assert "0.66" in fig.layout.title.text


def test_scoreboard_figure_match_rule_is_wrapped_to_fit_the_card():
    scores = ds.score_frames([[]], None)
    long_rule = ds.match_rule_text()
    fig = ds.scoreboard_figure(scores, arm_name="ML", threshold=0.5,
                               match_rule_text=long_rule)
    ann = fig.layout.annotations[0]
    lines = ann.text.split("<br>")
    assert len(lines) >= 2, "the real match rule sentence is too long for one line"
    assert all(len(line) <= 70 for line in lines)
    # Wrapping must not drop or reorder any word.
    assert " ".join(lines) == long_rule


def test_wrap_text_never_exceeds_max_chars_and_preserves_words():
    text = ("a detection counts as a hit within a fixed range and azimuth tolerance "
           "of a ground-truth target; one detection claims at most one target")
    wrapped = ds._wrap_text(text, max_chars=30)
    lines = wrapped.split("<br>")
    assert all(len(line) <= 30 for line in lines)
    assert " ".join(lines) == text


def test_wrap_text_is_a_no_op_for_a_short_string():
    assert ds._wrap_text("short text", max_chars=70) == "short text"


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


# --------------------------------------------------------------------------------
# Change 1a/1c, 2026-09-23 hostile-expert re-read: the scoreboard subline states the
# recall-matching calibration, and the offline block reads AP/FA/stripe/CI off the
# real beat_cfar.json / raddetnet_ci.json -- never a number typed in here.
# --------------------------------------------------------------------------------
def test_scoreboard_subline_states_the_recall_matched_calibration(beat_cfar_data):
    """The bug this fixes: hit rate 0.56 (a weak detector) > 0.50 (CFAR) > 0.47 (the
    strongest detector) reads as a ranking, when every threshold is independently
    that detector's own recall-0.5 point -- the subline must say so, with the real
    recall target and split size, not a hardcoded copy of them."""
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="b7_raddetnet", threshold=0.44,
                               match_rule_text="rule")
    # The sentence is long enough that `_wrap_text` line-breaks it (like every other
    # multi-line subline in this module) -- reassemble before substring-checking so
    # this test doesn't depend on exactly where the wrap falls.
    subline = fig.layout.title.text.replace("<br>", " ")
    target_recall = beat_cfar_data["target_recall"]
    n_frames = next(a["operating_point"]["n_frames"] for a in beat_cfar_data["arms"]
                    if a.get("operating_point"))
    assert f"recall-{target_recall:g}" in subline
    assert f"{n_frames}-frame test split (beat_cfar.json)" in subline
    assert "MATCHED recall" in subline
    assert "compare false alarms, not hits" in subline
    assert "0.44" in subline


def test_scoreboard_subline_falls_back_when_beat_cfar_json_missing(tmp_path):
    """No recall/split numbers to state -> a shorter subline, never an invented one."""
    scores = ds.score_frames([[]], None)
    missing = tmp_path / "no_such_beat_cfar.json"
    fig = ds.scoreboard_figure(scores, arm_name="ML", threshold=0.5,
                               match_rule_text="rule", beat_cfar_json_path=missing)
    assert fig.layout.title.text == "Detector scoreboard<br><sup>threshold 0.50</sup>"


def test_scoreboard_offline_block_reads_ap_fa_and_stripe_for_a_scored_arm(beat_cfar_data):
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="b7_raddetnet", threshold=0.44,
                               match_rule_text="rule", beat_cfar_arm_name="raddetnet")
    table = _table(fig)
    labels, values = table.cells.values
    row = dict(zip(labels, values))
    arm = next(a for a in beat_cfar_data["arms"] if a["name"] == "raddetnet")
    assert row["AP"] == f"{arm['AP']:.3f}"
    fa_label = next(k for k in row if k.startswith("FA/frame at recall"))
    assert row[fa_label] == f"{arm['operating_point']['fp_per_frame']:.2f}"
    stripe = beat_cfar_data["beat_cfar"]["stripe_rank1"]["raddetnet"]
    stripe_gt = beat_cfar_data["beat_cfar"]["stripe_ground_truth"]
    assert row["rank-1 stripe vs ground truth"] == f"{stripe:.3f} vs {stripe_gt:.3f}"
    header_row = next(k for k in row if k.startswith("offline,"))
    assert str(arm["operating_point"]["n_frames"]) in header_row
    # More than the base 7 rows now -- the table's height must have grown to match
    # (see the geometric check below), not silently clipped the new rows.
    assert len(labels) > 7


def test_scoreboard_offline_block_omits_stripe_row_for_classical_cfar(beat_cfar_data):
    """classical CFAR has no rank-1 stripe artifact -- beat_cfar.json's stripe_rank1
    has no entry for it, and the row must be omitted, not shown as 0 or 'n/a'."""
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="CFAR", threshold=0.66,
                               match_rule_text="rule",
                               beat_cfar_arm_name="classical CFAR")
    labels, _values = _table(fig).cells.values
    assert "AP" in labels
    assert "rank-1 stripe vs ground truth" not in labels
    assert "delta AP vs CFAR, 95% CI" not in labels


def test_scoreboard_offline_block_includes_ci_when_raddetnet_ci_json_has_a_row():
    """`raddetnet_ci.json` scores raddetnet against CFAR; the row's numbers are read
    from the file at test time (never a copy pasted into this assertion), so this
    cannot silently drift from what the file stores (CLAUDE.md's provenance rule)."""
    import json
    ci_data = json.loads(ds.DEFAULT_RADDETNET_CI_JSON.read_text())
    comp = next(c for c in ci_data["comparisons"] if c["arm"] == "raddetnet")
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="b7_raddetnet", threshold=0.44,
                               match_rule_text="rule", beat_cfar_arm_name="raddetnet")
    labels, values = _table(fig).cells.values
    row = dict(zip(labels, values))
    expected = (f"{comp['delta_AP']:+.3f} [{comp['ci_low']:+.3f}, {comp['ci_high']:+.3f}]")
    assert row["delta AP vs CFAR, 95% CI"] == expected


def test_scoreboard_offline_block_omits_ci_row_when_ci_file_missing(tmp_path, beat_cfar_data):
    """No `raddetnet_ci.json` for this deployment -> the row is dropped, not filled
    with an invented interval."""
    import json
    path = tmp_path / "beat_cfar.json"
    path.write_text(json.dumps(beat_cfar_data))
    missing_ci = tmp_path / "no_such_ci.json"
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="b7_raddetnet", threshold=0.44,
                               match_rule_text="rule", beat_cfar_json_path=path,
                               beat_cfar_arm_name="raddetnet",
                               raddetnet_ci_json_path=missing_ci)
    labels, _values = _table(fig).cells.values
    assert "delta AP vs CFAR, 95% CI" not in labels


def test_scoreboard_offline_block_absent_by_default():
    """No `beat_cfar_arm_name` -> the table stays at its base 7 rows, exactly the
    pre-existing behaviour every other test in this file exercises."""
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="ML", threshold=0.5,
                               match_rule_text="rule")
    labels, _values = _table(fig).cells.values
    assert len(labels) == 7


def test_scoreboard_offline_block_rows_never_clip_the_table():
    """Same geometric check as
    `test_scoreboard_figure_rows_never_clip_regardless_of_arm_name_length`, extended
    to the offline block: its extra rows must grow the declared table height, not
    just get appended past where the domain ends."""
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="b7_raddetnet", threshold=0.44,
                               match_rule_text="rule", beat_cfar_arm_name="raddetnet")
    table = _table(fig)
    domain_height = fig.layout.height - fig.layout.margin.t - fig.layout.margin.b
    content_height = table.header.height + len(table.cells.values[0]) * table.cells.height
    assert domain_height >= content_height


# --------------------------------------------------------------------------------
# Change 2, 2026-09-23: stored_pr_figure's in-distribution qualifier + the
# highlighted arm's bootstrap CI vs CFAR, read from raddetnet_ci.json.
# --------------------------------------------------------------------------------
def test_stored_pr_figure_states_in_distribution_qualifier():
    fig = ds.stored_pr_figure()
    subline = fig.layout.title.text
    assert "in-distribution: held-out scenes of the training corpus" in subline
    assert "one training seed per curve" in subline


def test_stored_pr_figure_highlighted_arm_legend_carries_the_ci(beat_cfar_data):
    import json
    ci_data = json.loads(ds.DEFAULT_RADDETNET_CI_JSON.read_text())
    comp = next(c for c in ci_data["comparisons"] if c["arm"] == "raddetnet")
    arm = next(a for a in beat_cfar_data["arms"] if a["name"] == "raddetnet")
    fig = ds.stored_pr_figure(highlight_arm="raddetnet")
    trace = next(tr for tr in fig.data if tr.name.startswith("raddetnet"))
    assert f"AP {arm['AP']:.3f}" in trace.name
    assert f"{comp['delta_AP']:+.3f} vs CFAR" in trace.name
    assert f"[{comp['ci_low']:+.3f}, {comp['ci_high']:+.3f}]" in trace.name


def test_stored_pr_figure_non_highlighted_arm_never_gets_a_ci_legend(beat_cfar_data):
    """Only the highlighted (bold) arm's legend gets the CI treatment -- every other
    curve keeps the plain "(AP=...)" legend even though raddetnet_ci.json also scores
    it (as the baseline every OTHER arm is compared against)."""
    fig = ds.stored_pr_figure(highlight_arm="classical CFAR")
    trace = next(tr for tr in fig.data if tr.name.startswith("raddetnet"))
    assert "vs CFAR" not in trace.name
    assert trace.name == "raddetnet (AP=0.476)"


def test_stored_pr_figure_highlighted_arm_omits_ci_when_file_missing(tmp_path):
    """The CI file is absent for this call -> the legend falls back to the plain
    format, never an invented interval."""
    fig = ds.stored_pr_figure(highlight_arm="classical CFAR",
                              raddetnet_ci_json_path=tmp_path / "no_such_ci.json")
    trace = next(tr for tr in fig.data if tr.name.startswith("classical CFAR"))
    assert "vs CFAR" not in trace.name
    assert trace.name.startswith("classical CFAR (AP=")
