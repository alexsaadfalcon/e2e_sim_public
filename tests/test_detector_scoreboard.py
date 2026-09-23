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


def _content_row_height(table) -> int:
    """Total data-row height: `cells.height` is one scalar for the whole table (a
    Plotly Table constraint -- there is no per-row height), raised to the tallest
    row's line count when a multi-line row (the CI-caveat/OOD rows) is present, so
    every row here is that same height."""
    return len(table.cells.values[0]) * table.cells.height


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
    assert row["this frame: unmatched (FP)"] == "1"
    assert row["this frame: FN"] == "1"
    # Cumulative over both frames: 1 hit, 1 false alarm. The "N/N scored" qualifier
    # moved into the value (Change, 2026-09-23 coordinator re-check).
    hits_key = next(k for k in row if k.startswith("cumulative hits"))
    assert row[hits_key] == "1 (2/2 scored)"
    # "cumulative unmatched detections" was dropped (4th hostile-expert read,
    # 2026-09-23) to hold the <=800 px budget while adding the connector/OOD-FA rows
    # -- the rate below now states this run's own frame count instead.
    assert "cumulative unmatched detections" not in row
    # Renamed (finding 1, 2026-09-23 hostile-expert re-read): "false alarm"/"FA" rows
    # count UNMATCHED detections against labels that themselves omit real objects
    # (F83's precision ceiling). Re-renamed again on the 4th read: the label now
    # states THIS RUN'S OWN frame count (2 here) so it cannot be read against the
    # offline block's "FA/frame at recall ..., N frames" row as if the two disagreed
    # rather than differing in sample size -- "(FA)" was dropped to make room.
    assert row["unmatched / frame, these 2 frames"] == "0.50"
    # The frame-count qualifier lives in the VALUE now, not the label (Change,
    # 2026-09-23 coordinator re-check: a long label wrapped to 2 lines and inflated
    # every row in the table to that height -- see `_TABLE_COL_CHARS`). Relabelled
    # from "hit rate (design, not quality)" (hostile-expert read, 2026-09-23, item 5:
    # ambiguous -- "16 hits / 5 frames / an assumed 5 GT/frame" reads as 0.64, not
    # the actual tp/(tp+fn)) to say exactly what it is.
    hit_rate_key = next(k for k in row if k.startswith("recall (hits / GT)"))
    assert row[hit_rate_key].startswith("0.50 ")
    # This run scored 2 frames -- the value states ITS OWN frame count, not the
    # offline split size, so a 5-frame hit rate can't be misread as a stable per-arm
    # number.
    assert "2fr" in row[hit_rate_key]
    assert "GT varies/frame" in row[hit_rate_key]
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
    # 3 (this-frame) + 3 (cumulative) numbers -- at most 8, per spec
    # ("cumulative unmatched detections" dropped, 4th hostile-expert read, 2026-09-23).
    _labels, values = table.cells.values
    assert len(values) <= 8


def test_scoreboard_figure_rows_never_clip_regardless_of_arm_name_length(beat_cfar_data):
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
    content_height = table.header.height + _content_row_height(table)
    assert domain_height >= content_height
    # All 6 base rows are always present in the underlying data, for every arm -- the
    # rendering bug above was purely geometric, not a difference in what is computed.
    # ("cumulative unmatched detections" was dropped, 4th hostile-expert read,
    # 2026-09-23, to hold the <=800 px budget once the connector/OOD-FA rows were
    # added -- see `scoreboard_figure`'s inline comment.)
    # Target recall / split size come from the real beat_cfar.json (never hardcoded
    # here, see CLAUDE.md's provenance rule -- they can drift with the file).
    target_recall = beat_cfar_data["target_recall"]
    n_frames = next(a["operating_point"]["n_frames"] for a in beat_cfar_data["arms"]
                    if a.get("operating_point"))
    # Every label/value is now pre-wrapped to fit its column (see `scoreboard_figure`'s
    # `_TABLE_COL_CHARS` comment) -- reassemble before comparing, same convention the
    # subline tests already use for `_wrap_text`'s output. "cumulative hits"/"recall"
    # moved their qualifier into the VALUE column (Change, 2026-09-23 coordinator
    # re-check) so neither label wraps past one line.
    labels, values = table.cells.values
    labels = [l.replace("<br>", " ") for l in labels]
    values = [v.replace("<br>", " ") for v in values]
    row = dict(zip(labels, values))
    assert labels == ["this frame: TP", "this frame: unmatched (FP)", "this frame: FN",
                      "cumulative hits", "unmatched / frame, these 0 frames",
                      "recall (hits / GT), this run"]
    assert row["cumulative hits"] == "0 (0/1 scored)"
    # target_recall/n_frames (the beat_cfar.json split calibration) no longer appear
    # on this row (item 5, moved out to disambiguate what the number actually is);
    # kept read from the real file above only to document that this row does not
    # depend on it any more, not because it's still used in the assertion below.
    assert target_recall is not None and n_frames is not None
    assert row["recall (hits / GT), this run"] == "n/a (0fr; GT varies/frame)"


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
    # Wrapping must not drop or reorder any word. The annotation now also carries the
    # precision-ceiling caveat (finding 1) appended after the match rule -- split it
    # back off before comparing, since that sentence is this test's own concern
    # (see test_scoreboard_annotation_states_the_precision_ceiling_caveat).
    match_rule_part = ann.text.split("<br>labels omit")[0]
    assert " ".join(match_rule_part.split("<br>")) == long_rule


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


def test_scoreboard_offline_block_reads_ap_fa_for_a_scored_arm(beat_cfar_data):
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="b7_raddetnet", threshold=0.44,
                               match_rule_text="rule", beat_cfar_arm_name="raddetnet")
    table = _table(fig)
    labels, values = table.cells.values
    row = dict(zip(labels, values))
    arm = next(a for a in beat_cfar_data["arms"] if a["name"] == "raddetnet")
    # "offline test split" was merged into the AP row (Change, 4th hostile-expert
    # read, 2026-09-23: freed a row for the connector/OOD-FA rows added this same
    # pass, within the <=800 px budget) -- both the AP number and the split's own
    # frame count now live on one row. RETRACTED (hostile-expert read, 2026-09-23,
    # item 8): this row used to also carry the rank-1 stripe statistic
    # ("AP, split, stripe vs GT" / "stripe 0.617/0.312") -- a bare number pair no
    # visitor could interpret without the presenter's own narration; the row is now
    # AP + split only, and the presenter's card keeps the stripe number.
    assert row["AP, offline test split"] == (
        f"{arm['AP']:.3f}, {arm['operating_point']['n_frames']}fr (beat_cfar.json)")
    fa_label = next(k for k in row if k.startswith("FA/frame at recall"))
    assert row[fa_label] == f"{arm['operating_point']['fp_per_frame']:.2f}"
    # The offline block's own frame count is now stated on this row too (4th
    # hostile-expert read, 2026-09-23) -- it sits directly below the live
    # "unmatched / frame, these N frames" row, and the two numbers must not read as
    # disagreeing without saying they're over different sample sizes.
    assert str(arm["operating_point"]["n_frames"]) in fa_label
    assert any(l.startswith("unmatched / frame, these") for l in row)
    # More than the base 6 rows now -- the table's height must have grown to match
    # (see the geometric check below), not silently clipped the new rows.
    assert len(labels) > 6


def test_scoreboard_offline_block_omits_stripe_row_for_classical_cfar(beat_cfar_data):
    """classical CFAR has no rank-1 stripe artifact -- beat_cfar.json's stripe_rank1
    has no entry for it, and the row must be omitted, not shown as 0 or 'n/a'."""
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="CFAR", threshold=0.66,
                               match_rule_text="rule",
                               beat_cfar_arm_name="classical CFAR")
    labels, _values = _table(fig).cells.values
    assert "AP, offline test split" in labels
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
    """No `beat_cfar_arm_name` -> the table stays at its base 6 rows (7 before the 4th
    hostile-expert read dropped "cumulative unmatched detections", 2026-09-23),
    exactly the pre-existing behaviour every other test in this file exercises."""
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="ML", threshold=0.5,
                               match_rule_text="rule")
    labels, _values = _table(fig).cells.values
    assert len(labels) == 6


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
    content_height = table.header.height + _content_row_height(table)
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


# --------------------------------------------------------------------------------
# Finding 3 (hostile-expert 3rd read, 2026-09-23): the null arm's stored name reads
# as "random INSIDE the ground-truth boxes" (the detector gets the answer); it is
# actually blind to the eval labels entirely -- uniform-random cells inside the
# bounding box of the TRAIN split's own targets (e2e/ml/compare_detectors.py:189-192,
# `score_null`'s docstring). Display-only remap, checked against that exact text.
# --------------------------------------------------------------------------------
def test_display_arm_name_remaps_only_the_null_arm():
    # Shortened (Change, 2026-09-23 coordinator re-check): this was the single
    # longest PR-legend entry, and a right-hand legend sized to it squeezed the plot
    # to a ~80 px sliver -- still states the real definition, just not the full
    # sentence.
    assert (ds._display_arm_name("null (random-in-GT-box)") ==
           "null: random cells in train-label box (chance floor)")
    assert ds._display_arm_name("classical CFAR") == "classical CFAR"
    assert ds._display_arm_name("raddetnet") == "raddetnet"


def test_stored_pr_figure_null_arm_shows_the_real_definition_not_the_stored_name():
    fig = ds.stored_pr_figure()
    names = [tr.name for tr in fig.data]
    assert any(n.startswith("null: random cells in train-label box") for n in names)
    assert not any("random-in-GT-box" in n for n in names)


# --------------------------------------------------------------------------------
# Finding 1 (hostile-expert 3rd read, 2026-09-23): "false positive"/"false alarm"
# rows count UNMATCHED detections against labels that themselves omit real objects
# (F83's precision ceiling) -- renamed above, and the caption below the table states
# the ceiling itself, read off a constant with its own provenance comment, not typed
# into this test as a bare number either.
# --------------------------------------------------------------------------------
def test_scoreboard_annotation_states_the_precision_ceiling_caveat():
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="ML", threshold=0.5,
                               match_rule_text="rule")
    ann = fig.layout.annotations[0]
    text = ann.text.replace("<br>", " ")
    assert f"precision ceiling {ds.PRECISION_CEILING_F83:.2f}" in text
    assert "labels omit ~3 real scatterers per frame inside 40 m" in text
    assert "unmatched is an upper bound on false alarms" in text
    # The match rule sentence must still be present, unmerged/undropped.
    assert "rule" in text
    # No bare "F-ledger"/"F83" tag on screen (hostile-expert read, 2026-09-23, item 8):
    # a visitor cannot look that up.
    assert "F-ledger" not in text and "F83" not in text


# --------------------------------------------------------------------------------
# Item 3 (hostile-expert read, 2026-09-23): four detection crosses on one ground-
# truth box scored as 1 TP + 3 unmatched read as a bug on screen -- it is the same
# 3x3 peak-grouping every detector here is scored under, and the caption must say so.
# --------------------------------------------------------------------------------
def test_scoreboard_annotation_states_the_peak_grouping_caveat():
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="ML", threshold=0.5,
                               match_rule_text="rule")
    ann = fig.layout.annotations[0]
    text = ann.text.replace("<br>", " ")
    assert "3x3" in text
    assert "grouped to local peaks" in text
    assert "same rule for every detector" in text
    assert "wide target can draw extra unmatched hits" in text


def test_scoreboard_annotation_geometry_still_fits_with_two_sentences():
    """The new caveat sentence lengthens the annotation the table's own bottom margin
    must leave room for -- same geometric contract as the row-clipping tests above,
    now covering the annotation block instead of the row block."""
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="ML", threshold=0.5,
                               match_rule_text=ds.match_rule_text())
    assert fig.layout.margin.b >= 5 * ds._TABLE_ANNOTATION_LINE_PX


# --------------------------------------------------------------------------------
# Wave 7 X7 (2026-09-23): neither hit-gate tolerance had a physical-scale reading on
# screen -- `hit_gate_scale_note` states both, computed from the real MatchCriterion
# and the real benchmark_v1 radar config, never a hardcoded copy of either.
# --------------------------------------------------------------------------------
def test_hit_gate_scale_note_states_the_beamwidth_and_native_range_bins():
    from e2e.radar_config import PRESETS as _radar_presets

    text = ds.hit_gate_scale_note()
    assert f"{_C.max_sin_az_err:g}" in text and "32-element array beamwidth" in text
    assert f"2/{ds._ARRAY_ELEMENTS_PER_AXIS}" in text
    native_res_m = _radar_presets[ds._T5_RADAR_PRESET_NAME].range_resolution_m
    n_bins = round(_C.max_range_err_m / native_res_m)
    assert f"{n_bins} native range bins" in text
    assert f"{native_res_m * 100:.0f} cm" in text
    # NOT the Ka-band munich frames' 3 GHz-sweep, ~5 cm resolution (a different
    # corpus/screen, wave 7 X7's own arithmetic mix-up -- see the function's
    # docstring): this corpus (benchmark_v1) is ~0.2 m/bin, not 0.05 m/bin.
    assert "5 cm" not in text and "40 native range bins" not in text


def test_scoreboard_annotation_states_the_hit_gate_scale_caveat():
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="ML", threshold=0.5,
                               match_rule_text="rule")
    ann = fig.layout.annotations[0]
    text = ann.text.replace("<br>", " ")
    assert "32-element array beamwidth" in text
    assert "native range bins" in text


# --------------------------------------------------------------------------------
# Wave 7 X2 (2026-09-23): the matched-recall FA comparison is the ONE thing this
# table can defend across arms; it must be the first row read, not the live block's
# raw (unmatched-recall) cross counts.
# --------------------------------------------------------------------------------
def test_offline_block_leads_with_fa_per_frame_not_ap():
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="b7_raddetnet", threshold=0.44,
                               match_rule_text="rule", beat_cfar_arm_name="raddetnet")
    labels, _values = _table(fig).cells.values
    fa_idx = labels.index(next(l for l in labels if l.startswith("FA/frame at recall")))
    ap_idx = labels.index(next(l for l in labels if l.startswith("AP,")))
    assert fa_idx == 0, "the matched-recall FA row must be the table's first row"
    assert fa_idx < ap_idx


# --------------------------------------------------------------------------------
# Finding 2 (hostile-expert 3rd read, 2026-09-23): the CI row's scene-bootstrap band
# hides the variance that has actually been measured to matter (a second training
# seed), and an out-of-distribution row is added for any arm a SEPARATE OOD-scored
# JSON also covers -- both numbers read from real files at test time, never typed.
# --------------------------------------------------------------------------------
def test_scoreboard_ci_row_states_the_seed_to_seed_spread():
    """The caveat row is a SHORT two-column row right after the CI row (Change,
    2026-09-23 coordinator re-check: the old one-column sentence wrapped to 2+ lines
    and inflated every row in the table to that height -- see `_TABLE_COL_CHARS`).

    4th hostile-expert read (2026-09-23): the row used to say the seed spread was
    merely "comparable to" the CI half-width; it must instead say WHICH is bigger,
    computed from raddetnet_ci.json's own numbers at test time -- never a hardcoded
    "0.040 > 0.032", since the CI file (and so the half-width) can be regenerated."""
    import json
    ci_data = json.loads(ds.DEFAULT_RADDETNET_CI_JSON.read_text())
    comp = next(c for c in ci_data["comparisons"] if c["arm"] == "raddetnet")
    half_width = (comp["ci_high"] - comp["ci_low"]) / 2.0
    op = ">" if ds.SEED_TO_SEED_AP_SPREAD_F86 > half_width else "<="

    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="b7_raddetnet", threshold=0.44,
                               match_rule_text="rule", beat_cfar_arm_name="raddetnet")
    labels, values = _table(fig).cells.values
    row = dict(zip(labels, values))
    ci_idx = labels.index("delta AP vs CFAR, 95% CI")
    caveat_label, caveat_value = labels[ci_idx + 1], values[ci_idx + 1]
    caveat_text = (caveat_label + " " + caveat_value).replace("<br>", " ")
    assert f"seed spread {ds.SEED_TO_SEED_AP_SPREAD_F86:.3f}" in caveat_text
    assert f"{op} CI half-width {half_width:.3f}" in caveat_text
    # No "(F86)" ledger tag on screen (hostile-expert read, 2026-09-23, item 8): a
    # visitor cannot look that up, so the row must stand on its two numbers alone.
    assert "F86" not in caveat_text


def test_default_ood_json_exists():
    assert ds.DEFAULT_OOD_JSON.is_file()


def test_scoreboard_offline_block_includes_ood_row_for_raddetnet():
    """One merged AP-vs-CFAR row (Change, 4th hostile-expert read, 2026-09-23: freed a
    row to make room for the FA row below, within the <=800 px budget) plus a
    dedicated "OOD unmatched/frame" row -- the finding this fixes: AP alone hides that
    the best-AP seed (s42) LOSES to CFAR on false alarms out of distribution, which
    the AP-only rows never showed. The row itself no longer names the corpus by its
    internal tag or the seeds by number (hostile-expert read, 2026-09-23, item 8) --
    it reads "out-of-distribution corpus" / "2 seeds"; this test still reads the real
    numbers from the file, just not the internal identifiers."""
    import json
    ood_arms = {a["name"]: a for a in json.loads(ds.DEFAULT_OOD_JSON.read_text())["arms"]}
    s42, s43 = ood_arms["raddetnet_s42"], ood_arms["raddetnet_s43"]
    cfar = ood_arms["classical CFAR"]

    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="b7_raddetnet", threshold=0.44,
                               match_rule_text="rule", beat_cfar_arm_name="raddetnet")
    labels, values = _table(fig).cells.values
    row = dict(zip(labels, values))
    ood_header = next(l for l in labels if l.startswith("OOD AP,"))
    assert "b1_bench_v2" not in ood_header and "s42" not in ood_header
    assert "out-of-distribution corpus" in ood_header
    ap_row_value = row[ood_header]
    assert "2 seeds" in ap_row_value
    assert f"{s42['AP']:.3f}" in ap_row_value and f"{s43['AP']:.3f}" in ap_row_value
    assert f"{cfar['AP']:.3f}" in ap_row_value

    fa_row_value = row["OOD unmatched/frame"]
    s42_fa = s42["operating_point"]["fp_per_frame"]
    s43_fa = s43["operating_point"]["fp_per_frame"]
    cfar_fa = cfar["operating_point"]["fp_per_frame"]
    assert f"{s42_fa:.1f}" in fa_row_value and f"{s43_fa:.1f}" in fa_row_value
    assert f"{cfar_fa:.1f}" in fa_row_value
    # The finding itself, checked from the real numbers rather than hardcoded: the
    # best-AP seed (s42, higher AP than s43) still has a WORSE (higher) FA/frame than
    # CFAR out of distribution.
    best_seed_ap, best_seed_fa = max((s42["AP"], s42_fa), (s43["AP"], s43_fa))
    assert best_seed_fa > cfar_fa


def test_scoreboard_offline_block_omits_ood_row_when_file_missing(tmp_path, beat_cfar_data):
    import json
    path = tmp_path / "beat_cfar.json"
    path.write_text(json.dumps(beat_cfar_data))
    missing_ood = tmp_path / "no_such_ood.json"

    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="b7_raddetnet", threshold=0.44,
                               match_rule_text="rule", beat_cfar_json_path=path,
                               beat_cfar_arm_name="raddetnet", ood_json_path=missing_ood)
    labels, _values = _table(fig).cells.values
    assert not any(l.startswith("OOD AP,") or l == "OOD unmatched/frame" for l in labels)


# --------------------------------------------------------------------------------
# F87 (5th pass, coordinator reissue 2026-09-23): a THIRD, separately-scored corpus
# (D4/b1_bench_v4, gen_v4_train.json) no checkpoint here trained on -- one more row,
# read from that file at test time, never typed.
# --------------------------------------------------------------------------------
def test_default_third_corpus_json_exists():
    assert ds.DEFAULT_THIRD_CORPUS_JSON.is_file()


def test_scoreboard_offline_block_includes_third_corpus_row_for_raddetnet():
    import json
    arms = {a["name"]: a for a in json.loads(ds.DEFAULT_THIRD_CORPUS_JSON.read_text())["arms"]}
    s42, s43, cfar = arms["raddetnet_s42"], arms["raddetnet_s43"], arms["classical CFAR"]

    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="b7_raddetnet", threshold=0.44,
                               match_rule_text="rule", beat_cfar_arm_name="raddetnet")
    labels, values = _table(fig).cells.values
    row = dict(zip(labels, values))
    label = next(l for l in labels if l.startswith("3rd corpus AP,"))
    assert "D4" in label
    assert str(s42["operating_point"]["n_frames"]) in label
    value = row[label]
    assert f"{s42['AP']:.3f}" in value and f"{s43['AP']:.3f}" in value
    assert f"{cfar['AP']:.3f}" in value
    # F87: every arm here leads CFAR -- checked from the real numbers, not hardcoded.
    assert s42["AP"] > cfar["AP"] and s43["AP"] > cfar["AP"]


def test_scoreboard_offline_block_third_corpus_row_for_classical_cfar_is_its_own_ap():
    """CFAR has no seed-sibling family -- its row is just its own AP, no merge."""
    import json
    cfar = next(a for a in json.loads(ds.DEFAULT_THIRD_CORPUS_JSON.read_text())["arms"]
               if a["name"] == "classical CFAR")
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="CFAR", threshold=0.66,
                               match_rule_text="rule", beat_cfar_arm_name="classical CFAR")
    labels, values = _table(fig).cells.values
    row = dict(zip(labels, values))
    label = next(l for l in labels if l.startswith("3rd corpus AP,"))
    assert row[label] == f"{cfar['AP']:.3f}"


def test_scoreboard_offline_block_omits_third_corpus_row_when_file_missing(tmp_path,
                                                                          beat_cfar_data):
    import json
    path = tmp_path / "beat_cfar.json"
    path.write_text(json.dumps(beat_cfar_data))
    missing = tmp_path / "no_such_third_corpus.json"

    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="b7_raddetnet", threshold=0.44,
                               match_rule_text="rule", beat_cfar_json_path=path,
                               beat_cfar_arm_name="raddetnet", third_corpus_json_path=missing)
    labels, _values = _table(fig).cells.values
    assert not any(l.startswith("3rd corpus AP,") for l in labels)


# --------------------------------------------------------------------------------
# Finding 1, connector row (4th hostile-expert read, 2026-09-23): the live block's
# "unmatched / frame, these N frames" row sits directly above the offline block's
# "FA/frame at recall ..., M frames" row, with two different-looking numbers over two
# very different sample sizes and no row saying so -- a one-line row is inserted
# between the two blocks, only when there is an offline block to point at.
# --------------------------------------------------------------------------------
def test_scoreboard_connector_row_between_live_and_offline_blocks(beat_cfar_data):
    """Order flipped (Change, wave 7 X2, 2026-09-23): the offline, matched-recall
    block now renders FIRST (the comparison the table can defend), then the
    connector, then the live per-frame block -- so a reader hits the fixed-split
    numbers, then the "these vary" note, then this run's own tiny counts, in that
    order (see `scoreboard_figure`'s inline comment)."""
    target = (10.0, 0.0, "vehicle")
    hit = (10.0, 0.0, 0.9, 10.0)
    scores = ds.score_frames([[hit]] * 5, [[target]] * 5)
    fig = ds.scoreboard_figure(scores, arm_name="b7_raddetnet", threshold=0.44,
                               match_rule_text="rule", beat_cfar_arm_name="raddetnet")
    labels, values = _table(fig).cells.values
    row = dict(zip(labels, values))
    n_frames = next(a["operating_point"]["n_frames"] for a in beat_cfar_data["arms"]
                    if a.get("operating_point"))
    live_label = next(l for l in labels if l.startswith("unmatched / frame, these"))
    connector_label = next(l for l in labels if l.endswith("live counts vary"))
    assert "5" in live_label and "5" in connector_label
    assert row[connector_label] == f"{n_frames}-frame numbers are the claim"
    # Must sit BETWEEN the two blocks, not before the offline rows or after the live
    # ones -- so a reader hits the fixed-split numbers, then the "these vary" note,
    # then this run's own tiny live counts, in that order.
    live_idx = labels.index(live_label)
    connector_idx = labels.index(connector_label)
    offline_idx = labels.index(next(l for l in labels if l.startswith("AP,")))
    assert offline_idx < connector_idx < live_idx


def test_scoreboard_no_connector_row_without_an_offline_block():
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="ML", threshold=0.5,
                               match_rule_text="rule")
    labels, _values = _table(fig).cells.values
    assert not any(l.endswith("live counts vary") for l in labels)


# --------------------------------------------------------------------------------
# Finding 5: the detector objectness panel's "labels & scoring stop at X m" line
# reads X from beat_cfar.json (webapp/pipeline_runner.py owns drawing the line
# itself -- see tests/test_webapp_figures_wave4.py).
# --------------------------------------------------------------------------------
def test_scoring_max_range_m_reads_the_real_beat_cfar_json(beat_cfar_data):
    expected = beat_cfar_data["arms"][0]["max_range_m"]
    assert ds.scoring_max_range_m() == pytest.approx(expected)


def test_scoring_max_range_m_none_when_file_missing(tmp_path):
    assert ds.scoring_max_range_m(tmp_path / "no_such.json") is None


# --------------------------------------------------------------------------------
# Coordinator re-check (2026-09-23): the scoreboard table hit ~1500 px because a
# single 4-line row (Plotly's Table `cells.height` is one scalar for the WHOLE
# table, not per-row) forced every other row to that same height. Fixed by keeping
# every row to at most `_TABLE_CELL_MAX_LINES` wrapped lines (content split across
# both columns rather than crammed into one) -- checked here for every arm this
# module knows how to build a full offline block for, not just the happy path.
# --------------------------------------------------------------------------------
def test_scoreboard_no_row_exceeds_the_max_line_cap():
    scores = ds.score_frames([[]], None)
    for arm_name, beat_cfar_arm_name in (
        ("ML", None),
        ("CA-CFAR (guard 2, train 6)", "classical CFAR"),
        ("b7_raddetnet", "raddetnet"),
    ):
        fig = ds.scoreboard_figure(scores, arm_name=arm_name, threshold=0.5,
                                   match_rule_text=ds.match_rule_text(),
                                   beat_cfar_arm_name=beat_cfar_arm_name)
        labels, values = _table(fig).cells.values
        for lbl, val in zip(labels, values):
            n_lines = max(lbl.count("<br>"), val.count("<br>")) + 1
            assert n_lines <= ds._TABLE_CELL_MAX_LINES, (arm_name, lbl, val)


def test_scoreboard_figure_height_fits_a_screen_for_every_arm():
    """~1000 px (raised from the 880 px budget, 2026-09-23, wave 7 X7: the mandatory
    hit-gate scale footnote -- `hit_gate_scale_note`, one 32-element beamwidth / one
    native-range-bins clause -- adds a 4th wrapped annotation sentence to every arm,
    measured +94-98 px here). The worst case (an arm with a CI row, a seed-spread
    caveat, the connector row and a two-row OOD block -- 14 rows total) must still
    fit, with real slack in the browser, not just in this geometric formula (a real
    Playwright render of this exact arm's figure JSON clipped its last row at zero
    slack, 2026-09-23: see `ds._TABLE_RENDER_SAFETY_PX`). Re-verify against a
    rendered rehearsal PNG after any further change to this annotation."""
    scores = ds.score_frames([[]], None)
    for beat_cfar_arm_name in (None, "classical CFAR", "raddetnet"):
        fig = ds.scoreboard_figure(scores, arm_name="b7_raddetnet", threshold=0.44,
                                   match_rule_text=ds.match_rule_text(),
                                   beat_cfar_arm_name=beat_cfar_arm_name)
        assert fig.layout.height <= 1000, (beat_cfar_arm_name, fig.layout.height)


# --------------------------------------------------------------------------------
# Coordinator re-check (2026-09-23): the stored-PR panel's legend, sized to its
# longest entry (the highlighted arm's CI-augmented name), squeezed the plot to a
# ~80 px sliver on the right. Moved below the plot instead -- that only ever costs
# BOTTOM margin, never plot WIDTH, regardless of entry length.
# --------------------------------------------------------------------------------
def test_stored_pr_figure_legend_is_below_the_plot_not_squeezing_it():
    fig = ds.stored_pr_figure(highlight_arm="raddetnet")
    assert fig.layout.legend.orientation == "h"
    assert fig.layout.legend.y < 0, "legend must sit BELOW the plot, not beside it"
    domain = fig.layout.xaxis.domain
    assert domain is not None
    assert (domain[1] - domain[0]) >= 0.6
