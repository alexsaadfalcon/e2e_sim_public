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
from webapp.pipeline_runner import FIGURE_HEIGHT, PANEL_ROW_TABLE, panel_of, panel_text

_C = MatchCriterion()  # the real, frozen tolerances -- never hardcoded here (see below)


def _find(seq, pred, what: str):
    """`next(x for x in seq if pred(x))`, but a miss fails as a named AssertionError
    instead of a `StopIteration` that pytest reports as a teardown `RuntimeError`
    (2026-09-24 layout rewrite: several rows this file used to find in the TABLE moved
    to the panel's Details list, and a bare `next(...)` against the wrong list used to
    surface as an opaque generator crash rather than a clear assertion)."""
    for x in seq:
        if pred(x):
            return x
    raise AssertionError(f"no {what} found in {seq!r}")


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
    # "last frame", not "this frame" (hostile round 11, H3): a Plotly Table cannot
    # animate, so these rows never follow the screen's transport -- and the objectness
    # map beside them now does.
    assert row["last frame: TP"] == "0"
    assert row["last frame: unmatched (FP)"] == "1"
    assert row["last frame: FN"] == "1"
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
    # table's last row, see `_TABLE_HEADER_HEIGHT`'s comment) and into the panel's
    # one-line CAPTION (2026-09-24 redesign: the figure itself carries no title any
    # more -- see `panel_of`/`panel_caption` in webapp.pipeline_runner). The caption
    # is visible without opening anything, same visibility the old title had.
    assert table.header.values[0] == "classical CFAR"
    assert table.header.values[1] == "count"
    assert "0.50" in panel_of(fig)["caption"][0]


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
    # 17, not >= 18 (2026-09-24 layout redesign): 17 px is the new in-figure floor
    # (layout spec section 3) and this table is built to exactly it -- the old >= 18
    # assertion pinned a stricter number than the shipped floor.
    assert table.header.font.size >= 17
    assert table.cells.font.size >= 17
    # 3 (this-frame) + 3 (cumulative) numbers -- at most 8, per spec
    # ("cumulative unmatched detections" dropped, 4th hostile-expert read, 2026-09-23).
    _labels, values = table.cells.values
    assert len(values) <= 8


def test_scoreboard_figure_rows_never_clip_regardless_of_arm_name_length(beat_cfar_data):
    """The bug this pins: a long header ("CA-CFAR (guard 2, train 6) -- threshold
    0.66") wrapped to two lines inside its declared single-line height, stealing
    room from the bottom of the table and clipping the last ("hit rate") row --
    CFAR and the neural-detector arms run the same code and must show identical
    rows (rehearsal, 2026-09-23).

    RETIRED geometric check (2026-09-24 layout redesign): the table's height used to
    be DERIVED from its own row/line count, so "does the content fit the domain" was
    a real question with a real failure mode. It no longer is -- `scoreboard_figure`
    now fixes the figure at `FIGURE_HEIGHT[PANEL_ROW_TABLE]` first and computes
    `row_height` FROM that fixed budget (`(FIGURE_HEIGHT - header) // n_rows`), so
    "does it fit" is true by construction for any n_rows <= 8 and asserting it would
    only be re-deriving the same arithmetic `scoreboard_figure` already ran. What
    survives as a real invariant -- the fixed height and the row content -- is
    checked directly below instead."""
    scores = ds.score_frames([[]], None)
    long_name = "a very long arm name, e.g. an ML checkpoint's parent directory"
    fig = ds.scoreboard_figure(scores, arm_name=long_name, threshold=0.5,
                               match_rule_text="rule")
    table = _table(fig)
    assert fig.layout.height == FIGURE_HEIGHT[PANEL_ROW_TABLE]
    assert fig.layout.margin.t == 0 and fig.layout.margin.b == 0
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
    # No row may wrap at all now (`scoreboard_figure` asserts this itself and fails
    # loudly instead) -- so, unlike before 2026-09-24, no `<br>` reassembly is needed
    # here; a label/value that needed one would already have failed inside the
    # figure builder.
    labels, values = table.cells.values
    row = dict(zip(labels, values))
    assert labels == ["last frame: TP", "last frame: unmatched (FP)", "last frame: FN",
                      "cumulative hits", "unmatched / frame, these 0 frames",
                      "recall (hits / GT), this run"]
    assert row["cumulative hits"] == "0 (0/1 scored)"
    # target_recall/n_frames (the beat_cfar.json split calibration) no longer appear
    # on this row (item 5, moved out to disambiguate what the number actually is);
    # kept read from the real file above only to document that this row does not
    # depend on it any more, not because it's still used in the assertion below.
    assert target_recall is not None and n_frames is not None
    assert row["recall (hits / GT), this run"] == "n/a (0fr; GT varies/frame)"


def test_scoreboard_figure_header_count_label_and_threshold_in_caption():
    """Was "...in_title": the figure itself carries no title any more (2026-09-24
    redesign) -- the threshold now lives in the panel's one-line HTML caption, built
    by `webapp.pipeline_runner.set_panel` (see `panel_of`)."""
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="CA-CFAR (guard 2, train 6)",
                               threshold=0.66, match_rule_text="rule")
    table = _table(fig)
    # The header's second column used to be an empty dark cell.
    assert table.header.values[1] == "count"
    assert panel_of(fig)["title"] == "Detector scoreboard"
    assert "0.66" in panel_of(fig)["caption"][0]


def test_scoreboard_match_rule_is_reachable_verbatim_in_details():
    """Was "...is_wrapped_to_fit_the_card": the match rule used to be a figure
    annotation, wrapped with `_wrap_text` to fit the card's pixel width. 2026-09-24
    redesign: it is now a Details line (HTML, wraps itself in the browser), carried
    VERBATIM -- `_wrap_text` is no longer applied to it at all, so there is nothing
    left to assert about line width; the substance that survives is "the real match
    rule sentence is reachable, unmodified, in one click" (panel_text's whole point)."""
    scores = ds.score_frames([[]], None)
    long_rule = ds.match_rule_text()
    fig = ds.scoreboard_figure(scores, arm_name="ML", threshold=0.5,
                               match_rule_text=long_rule)
    details = panel_of(fig)["details"]
    assert long_rule in details, f"match rule not carried verbatim: {details}"


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
    """The figure carries no title any more (2026-09-24 redesign) -- both clauses this
    used to pin on `fig.layout.title.text` now live in the panel's Details (the file
    name in the "scored offline: ..." line, the frame count in the same line), reached
    via `panel_text`."""
    fig = ds.stored_pr_figure(highlight_arm="classical CFAR")
    assert len(fig.data) == len(beat_cfar_data["arms"])
    names = [tr.name for tr in fig.data]
    assert any(n.startswith("classical CFAR") for n in names)
    text = panel_text(fig)
    assert "beat_cfar.json" in text
    assert str(len(beat_cfar_data["arms"][0]["gt_per_frame"])) in text


def test_stored_pr_figure_highlights_bold(beat_cfar_data):
    fig = ds.stored_pr_figure(highlight_arm="classical CFAR")
    widths = {tr.name.split(" (")[0]: tr.line.width for tr in fig.data
             if tr.mode == "lines"}
    assert widths["classical CFAR"] > max(
        w for name, w in widths.items() if name != "classical CFAR")


def test_stored_pr_figure_fonts_are_legible():
    """The figure title-font check is retired (2026-09-24): the title is now HTML
    (`panel_of(fig)["title"]`), not a figure element, so it has no `fig.layout.title.
    font` to assert on any more -- its own legibility is a page-CSS concern, covered
    by `tests/test_webapp_layout_acceptance.py`'s MIN_PAGE_FONT_PX checks, not this
    module. The in-figure body/legend font floor this test still owns is unchanged."""
    fig = ds.stored_pr_figure()
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
    # The fallback banner used to be a figure annotation; 2026-09-24 redesign moves it
    # to a Details line instead (background info, one click away -- the marker trace's
    # own name above is what stays visible without opening anything).
    details_text = panel_text(fig)
    assert "no curve arm" in details_text and "no stored PR curve" in details_text


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
    # 2026-09-24 redesign: the subline is now the LAST Details line (see
    # `scoreboard_figure`'s docstring), carried verbatim -- Details is HTML and wraps
    # itself, so `_wrap_text` is never applied to it and there is no `<br>` to
    # reassemble any more (unlike the pre-2026-09-24 figure-title subtitle this
    # replaced). `panel_text` is used rather than pulling the exact list index so this
    # test does not also pin the subline's POSITION among the other Details lines.
    subline = panel_text(fig)
    target_recall = beat_cfar_data["target_recall"]
    n_frames = next(a["operating_point"]["n_frames"] for a in beat_cfar_data["arms"]
                    if a.get("operating_point"))
    assert f"recall-{target_recall:g}" in subline
    assert f"{n_frames}-frame test split (beat_cfar.json)" in subline
    assert "MATCHED recall" in subline
    assert "compare false alarms, not hits" in subline
    assert "0.44" in subline


def test_scoreboard_subline_falls_back_when_beat_cfar_json_missing(tmp_path):
    """No recall/split numbers to state -> a shorter subline, never an invented one.

    2026-09-24 redesign: the figure carries no title at all any more -- the panel's
    HTML title is a fixed constant ("Detector scoreboard", never numbers), the
    threshold lives in the caption (visible), and the fallback subline (short,
    because there is nothing to calibrate against) is the last Details line."""
    scores = ds.score_frames([[]], None)
    missing = tmp_path / "no_such_beat_cfar.json"
    fig = ds.scoreboard_figure(scores, arm_name="ML", threshold=0.5,
                               match_rule_text="rule", beat_cfar_json_path=missing)
    panel = panel_of(fig)
    assert panel["title"] == "Detector scoreboard"
    assert panel["caption"][0] == "threshold 0.50"
    assert panel["details"][-1] == "threshold 0.50"


def test_scoreboard_offline_block_reads_ap_fa_for_a_scored_arm(beat_cfar_data):
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="b7_raddetnet", threshold=0.44,
                               match_rule_text="rule", beat_cfar_arm_name="raddetnet")
    table = _table(fig)
    labels, values = table.cells.values
    row = dict(zip(labels, values))
    arm = next(a for a in beat_cfar_data["arms"] if a["name"] == "raddetnet")
    cfar = next(a for a in beat_cfar_data["arms"] if a["name"] == "classical CFAR")
    # "offline test split" was merged into the AP row (Change, 4th hostile-expert
    # read, 2026-09-23: freed a row for the connector/OOD-FA rows added this same
    # pass, within the <=800 px budget) -- both the AP number and the split's own
    # frame count now live on one row. RETRACTED (hostile-expert read, 2026-09-23,
    # item 8): this row used to also carry the rank-1 stripe statistic
    # ("AP, split, stripe vs GT" / "stripe 0.617/0.312") -- a bare number pair no
    # visitor could interpret without the presenter's own narration; the row is now
    # AP + split only, and the presenter's card keeps the stripe number.
    # Item 2 (wave 9 hostile-expert read, 2026-09-23): "raddetnet" is a LEARNED
    # detector's arm, so both the AP and FA/frame rows now also carry CFAR's own
    # number inline (read from this same file, never typed) -- the AP row drops the
    # "{n}fr (beat_cfar.json)" suffix to make room, since the subline above the
    # table already states both.
    assert row["AP, offline test split"] == f"{arm['AP']:.3f} (CFAR {cfar['AP']:.3f})"
    fa_label = next(k for k in row if k.startswith("FA/frame at recall"))
    assert row[fa_label] == (f"{arm['operating_point']['fp_per_frame']:.2f} "
                             f"(CFAR {cfar['operating_point']['fp_per_frame']:.2f})")
    # The offline block's own frame count is now stated on this row too (4th
    # hostile-expert read, 2026-09-23) -- it sits directly below the live
    # "unmatched / frame, these N frames" row, and the two numbers must not read as
    # disagreeing without saying they're over different sample sizes.
    assert str(arm["operating_point"]["n_frames"]) in fa_label
    assert any(l.startswith("unmatched / frame, these") for l in row)
    # More than the base 6 rows now -- the table's height must have grown to match
    # (see the geometric check below), not silently clipped the new rows.
    assert len(labels) > 6


def test_scoreboard_offline_block_cfar_arm_has_no_self_reference(beat_cfar_data):
    """CFAR's own row must never print "CFAR" against itself (item 2 is for a
    LEARNED detector's arm only) -- unchanged format for the classical CFAR row."""
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="CFAR", threshold=0.66,
                               match_rule_text="rule",
                               beat_cfar_arm_name="classical CFAR")
    labels, values = _table(fig).cells.values
    row = dict(zip(labels, values))
    arm = next(a for a in beat_cfar_data["arms"] if a["name"] == "classical CFAR")
    assert row["AP, offline test split"] == (
        f"{arm['AP']:.3f}, {arm['operating_point']['n_frames']}fr (beat_cfar.json)")
    fa_label = next(k for k in row if k.startswith("FA/frame at recall"))
    assert row[fa_label] == f"{arm['operating_point']['fp_per_frame']:.2f}"


def test_scoreboard_offline_block_omits_stripe_row_for_classical_cfar(beat_cfar_data):
    """classical CFAR has no rank-1 stripe artifact -- beat_cfar.json's stripe_rank1
    has no entry for it, and the row must be omitted, not shown as 0 or 'n/a'. The CI
    row is checked in Details too (2026-09-24 redesign moved it there for every arm,
    so "not in the table" alone would be true even for an arm that DID have a CI row
    -- see `test_scoreboard_offline_block_includes_ci_when_raddetnet_ci_json_has_a_row`)."""
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="CFAR", threshold=0.66,
                               match_rule_text="rule",
                               beat_cfar_arm_name="classical CFAR")
    labels, _values = _table(fig).cells.values
    assert "AP, offline test split" in labels
    assert "rank-1 stripe vs ground truth" not in labels
    details = panel_of(fig)["details"]
    assert not any(d.startswith("delta AP vs CFAR, 95% CI:") for d in details)


def test_scoreboard_offline_block_includes_ci_when_raddetnet_ci_json_has_a_row():
    """`raddetnet_ci.json` scores raddetnet against CFAR; the row's numbers are read
    from the file at test time (never a copy pasted into this assertion), so this
    cannot silently drift from what the file stores (CLAUDE.md's provenance rule).

    BACKGROUND, not visible-without-digging (2026-09-24 redesign): the table's <=8
    visible rows are only the FA/frame and AP rows (`scoreboard_figure`'s docstring);
    every other offline row, this CI row included, moved to the panel's Details as a
    "<label>: <value>" line -- reachable in one click, no longer in the table."""
    import json
    ci_data = json.loads(ds.DEFAULT_RADDETNET_CI_JSON.read_text())
    comp = next(c for c in ci_data["comparisons"] if c["arm"] == "raddetnet")
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="b7_raddetnet", threshold=0.44,
                               match_rule_text="rule", beat_cfar_arm_name="raddetnet")
    details = panel_of(fig)["details"]
    line = _find(details, lambda d: d.startswith("delta AP vs CFAR, 95% CI:"),
                what="the delta-AP-vs-CFAR CI Details line")
    expected = (f"delta AP vs CFAR, 95% CI: {comp['delta_AP']:+.3f} "
               f"[{comp['ci_low']:+.3f}, {comp['ci_high']:+.3f}]")
    assert line == expected


def test_scoreboard_offline_block_omits_ci_row_when_ci_file_missing(tmp_path, beat_cfar_data):
    """No `raddetnet_ci.json` for this deployment -> the row is dropped, not filled
    with an invented interval. Checked in Details now (2026-09-24 redesign): the row
    was never in the visible table to begin with (see the test above), so "not in
    `labels`" would pass trivially regardless of whether this file exists."""
    import json
    path = tmp_path / "beat_cfar.json"
    path.write_text(json.dumps(beat_cfar_data))
    missing_ci = tmp_path / "no_such_ci.json"
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="b7_raddetnet", threshold=0.44,
                               match_rule_text="rule", beat_cfar_json_path=path,
                               beat_cfar_arm_name="raddetnet",
                               raddetnet_ci_json_path=missing_ci)
    details = panel_of(fig)["details"]
    assert not any(d.startswith("delta AP vs CFAR, 95% CI:") for d in details)


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
    """RETIRED geometric check (2026-09-24 redesign, see
    `test_scoreboard_figure_rows_never_clip_regardless_of_arm_name_length`'s docstring
    for why): the offline block no longer grows the table past its base row count at
    all -- it contributes at most its 2 promoted rows (FA/frame, AP), never more, so
    the table never exceeds 8 rows and the figure height is the same fixed constant
    regardless of how many offline rows an arm has (the rest go to Details)."""
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="b7_raddetnet", threshold=0.44,
                               match_rule_text="rule", beat_cfar_arm_name="raddetnet")
    table = _table(fig)
    assert fig.layout.height == FIGURE_HEIGHT[PANEL_ROW_TABLE]
    labels, _values = table.cells.values
    assert len(labels) <= 8


# --------------------------------------------------------------------------------
# Change 2, 2026-09-23: stored_pr_figure's in-distribution qualifier + the
# highlighted arm's bootstrap CI vs CFAR, read from raddetnet_ci.json.
# --------------------------------------------------------------------------------
def test_stored_pr_figure_states_in_distribution_qualifier():
    """Was a figure title/subtitle; 2026-09-24 redesign moves it to Details (this
    figure has no title at all any more -- see `stored_pr_figure`'s docstring)."""
    fig = ds.stored_pr_figure()
    text = panel_text(fig)
    assert "in-distribution: held-out scenes of the training corpus" in text
    assert "one training seed per curve" in text


def test_stored_pr_figure_highlighted_arm_carries_its_delta_and_ci(beat_cfar_data):
    """Only the highlighted arm carries its delta-vs-CFAR, and its CONFIDENCE INTERVAL
    is on the panel caption rather than the legend entry (2026-09-24): measured on the
    rendered page, the full 51-character entry overran its half of the two-column
    legend strip and drew straight through the entry beside it. Both numbers are still
    on the panel, computed from raddetnet_ci.json; the caption is the more visible of
    the two places."""
    import json
    from webapp.pipeline_runner import panel_caption
    ci_data = json.loads(ds.DEFAULT_RADDETNET_CI_JSON.read_text())
    comp = next(c for c in ci_data["comparisons"] if c["arm"] == "raddetnet")
    arm = next(a for a in beat_cfar_data["arms"] if a["name"] == "raddetnet")
    fig = ds.stored_pr_figure(highlight_arm="raddetnet")
    trace = next(tr for tr in fig.data if tr.name.startswith("raddetnet"))
    # SHORT form (hostile round 11, D3): "raddetnet 0.476 (+0.175)". At the previous
    # "raddetnet AP 0.476, +0.175 vs CFAR" the entry filled its half of the
    # two-column strip edge to edge and abutted the entry beside it with zero gap, so
    # the two read as one string and the delta looked like it was against THAT arm.
    # "AP" and "vs CFAR" are spelled out in the caption, with the interval.
    assert f"{arm['AP']:.3f}" in trace.name
    assert f"({comp['delta_AP']:+.3f})" in trace.name
    assert len(trace.name) <= 30, trace.name
    caption = panel_caption(fig)
    assert f"[{comp['ci_low']:+.3f}, {comp['ci_high']:+.3f}]" in caption
    # ONE line in a 746 px column at 16 px is ~86 characters; past that the browser
    # clips it, and acceptance check 12 forbids a truncation mark in visible text.
    assert len(caption) <= 86, caption


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
    """The stored JSON name ("null (random-in-GT-box)") reads as "random INSIDE the
    ground-truth boxes", i.e. as if the detector were handed the answer, so it is
    remapped for display. Shortened again 2026-09-24: in the two-column legend a
    52-character entry overran its half of the strip and was CLIPPED at the column
    edge -- and this is the one entry that must never be the one that gets cut,
    because it is the panel's chance floor. The real definition did not go away; it
    moved to Details (see the test below)."""
    assert ds._display_arm_name("null (random-in-GT-box)") == "null (chance floor)"
    assert ds._display_arm_name("classical CFAR") == "classical CFAR"
    assert ds._display_arm_name("raddetnet") == "raddetnet"


def test_stored_pr_details_state_the_identical_on_both_arms_sentence():
    """A panel that is scored OFFLINE and therefore identical on both A/B arms reads as
    a bug (two panels, same numbers) unless it says so. It is a standing Details line
    on every call, because it is true of both arms; `webapp/app.py` additionally puts a
    short "identical on both arms" clause on ARM B's visible caption, so the statement
    is made once per row rather than twice."""
    from webapp.pipeline_runner import panel_text
    text = panel_text(ds.stored_pr_figure(highlight_arm="raddetnet"))
    assert "scored offline; identical on both arms, the knob cannot move it" in text


def test_stored_pr_figure_null_arm_shows_the_real_definition_not_the_stored_name():
    """The misleading stored name never reaches the screen, and the REAL definition is
    still on the panel -- in Details, because the legend entry had to shrink to fit the
    two-column strip (2026-09-24). `panel_text` is title + caption + Details, i.e.
    everything the presenter can reach in one click."""
    from webapp.pipeline_runner import panel_text
    fig = ds.stored_pr_figure()
    names = [tr.name for tr in fig.data]
    assert any(n.startswith("null (chance floor)") for n in names)
    assert not any("random-in-GT-box" in n for n in names)
    text = panel_text(fig)
    assert "random cells in train-label box" in text
    assert "never the eval labels" in text
    assert "random-in-GT-box" not in text


# --------------------------------------------------------------------------------
# Finding 1 (hostile-expert 3rd read, 2026-09-23): "false positive"/"false alarm"
# rows count UNMATCHED detections against labels that themselves omit real objects
# (F83's precision ceiling) -- renamed above, and the caption below the table states
# the ceiling itself, read off a constant with its own provenance comment, not typed
# into this test as a bare number either.
# --------------------------------------------------------------------------------
def test_scoreboard_annotation_states_the_precision_ceiling_caveat():
    """Was a figure annotation (a single merged string with the match rule); 2026-09-24
    redesign moves it to its own Details line, unwrapped and un-merged -- see
    `scoreboard_figure`'s docstring. Background info (a click away), not one of the
    <=8 promoted table rows."""
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="ML", threshold=0.5,
                               match_rule_text="rule")
    text = panel_text(fig)
    assert f"precision ceiling {ds.PRECISION_CEILING_F83:.2f}" in text
    assert "labels omit ~3 real scatterers per frame inside 40 m" in text
    assert "unmatched is an upper bound on false alarms" in text
    # The match rule sentence must still be present, as its own Details line (no
    # longer merged into one annotation with this caveat).
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
    """Was a figure annotation; now its own Details line (see the precision-ceiling
    test above for the same move)."""
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="ML", threshold=0.5,
                               match_rule_text="rule")
    text = panel_text(fig)
    assert "3x3" in text
    assert "grouped to local peaks" in text
    assert "same rule for every detector" in text
    assert "wide target can draw extra unmatched hits" in text


def test_scoreboard_figure_height_is_fixed_regardless_of_caveat_sentence_count():
    """Was "...geometry_still_fits_with_two_sentences", pinning the OLD bottom margin
    growing with the annotation's wrapped-line count (`_TABLE_ANNOTATION_LINE_PX`,
    now retired along with the annotation itself -- 2026-09-24 redesign). The figure
    carries no annotation-based caveats any more (they are Details lines, off the
    figure entirely, see the tests above), so there is nothing left for the bottom
    margin to grow with: it is a fixed, always-zero constant, whatever the real match
    rule's length is."""
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="ML", threshold=0.5,
                               match_rule_text=ds.match_rule_text())
    assert fig.layout.margin.b == 0
    assert fig.layout.height == FIGURE_HEIGHT[PANEL_ROW_TABLE]


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
    """Was a figure annotation; now its own Details line (see the precision-ceiling
    test above for the same move)."""
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="ML", threshold=0.5,
                               match_rule_text="rule")
    text = panel_text(fig)
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
    "0.040 > 0.032", since the CI file (and so the half-width) can be regenerated.

    Both the CI row and this caveat live in Details now (2026-09-24 redesign, see
    `test_scoreboard_offline_block_includes_ci_when_raddetnet_ci_json_has_a_row`),
    immediately adjacent in the same order `_offline_arm_rows` builds them -- checked
    with an explicit search rather than a bare `next(...)`, so a missing row fails as
    an assertion, not a generator error."""
    import json
    ci_data = json.loads(ds.DEFAULT_RADDETNET_CI_JSON.read_text())
    comp = next(c for c in ci_data["comparisons"] if c["arm"] == "raddetnet")
    half_width = (comp["ci_high"] - comp["ci_low"]) / 2.0
    op = ">" if ds.SEED_TO_SEED_AP_SPREAD_F86 > half_width else "<="

    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="b7_raddetnet", threshold=0.44,
                               match_rule_text="rule", beat_cfar_arm_name="raddetnet")
    details = panel_of(fig)["details"]
    ci_idx = next((i for i, d in enumerate(details)
                  if d.startswith("delta AP vs CFAR, 95% CI:")), None)
    assert ci_idx is not None, f"no CI Details line found in {details!r}"
    caveat_text = details[ci_idx + 1]
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
    numbers from the file, just not the internal identifiers.

    BACKGROUND, not visible-without-digging (2026-09-24 redesign): neither OOD row is
    one of the table's <=8 promoted rows (only FA/frame and AP are) -- both are now
    Details lines, found with an explicit search (a bare `next(...)` against the old
    TABLE labels raised a bare `StopIteration` here once these rows moved)."""
    import json
    ood_arms = {a["name"]: a for a in json.loads(ds.DEFAULT_OOD_JSON.read_text())["arms"]}
    s42, s43 = ood_arms["raddetnet_s42"], ood_arms["raddetnet_s43"]
    cfar = ood_arms["classical CFAR"]

    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="b7_raddetnet", threshold=0.44,
                               match_rule_text="rule", beat_cfar_arm_name="raddetnet")
    details = panel_of(fig)["details"]
    ap_line = _find(details, lambda d: d.startswith("OOD AP,"), what="the OOD AP Details line")
    assert "b1_bench_v2" not in ap_line and "s42" not in ap_line
    assert "out-of-distribution corpus" in ap_line
    assert "2 seeds" in ap_line
    assert f"{s42['AP']:.3f}" in ap_line and f"{s43['AP']:.3f}" in ap_line
    assert f"{cfar['AP']:.3f}" in ap_line

    fa_line = _find(details, lambda d: d.startswith("OOD unmatched/frame:"),
                    what="the OOD unmatched/frame Details line")
    s42_fa = s42["operating_point"]["fp_per_frame"]
    s43_fa = s43["operating_point"]["fp_per_frame"]
    cfar_fa = cfar["operating_point"]["fp_per_frame"]
    assert f"{s42_fa:.1f}" in fa_line and f"{s43_fa:.1f}" in fa_line
    assert f"{cfar_fa:.1f}" in fa_line
    # The finding itself, checked from the real numbers rather than hardcoded: the
    # best-AP seed (s42, higher AP than s43) still has a WORSE (higher) FA/frame than
    # CFAR out of distribution.
    best_seed_ap, best_seed_fa = max((s42["AP"], s42_fa), (s43["AP"], s43_fa))
    assert best_seed_fa > cfar_fa


def test_scoreboard_offline_block_omits_ood_row_when_file_missing(tmp_path, beat_cfar_data):
    """Checked in Details now (2026-09-24 redesign): neither OOD row was ever in the
    visible table to begin with (see the test above), so checking `labels` alone
    would pass trivially whether or not this file exists."""
    import json
    path = tmp_path / "beat_cfar.json"
    path.write_text(json.dumps(beat_cfar_data))
    missing_ood = tmp_path / "no_such_ood.json"

    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="b7_raddetnet", threshold=0.44,
                               match_rule_text="rule", beat_cfar_json_path=path,
                               beat_cfar_arm_name="raddetnet", ood_json_path=missing_ood)
    details = panel_of(fig)["details"]
    assert not any(d.startswith("OOD AP,") or d.startswith("OOD unmatched/frame:")
                  for d in details)


# --------------------------------------------------------------------------------
# F87 (5th pass, coordinator reissue 2026-09-23): a THIRD, separately-scored corpus
# (D4/b1_bench_v4, gen_v4_train.json) no checkpoint here trained on -- one more row,
# read from that file at test time, never typed.
# --------------------------------------------------------------------------------
def test_default_third_corpus_json_exists():
    assert ds.DEFAULT_THIRD_CORPUS_JSON.is_file()


def test_scoreboard_offline_block_includes_third_corpus_row_for_raddetnet():
    """BACKGROUND, not visible-without-digging (2026-09-24 redesign): the 3rd-corpus
    row is not one of the table's <=8 promoted rows -- it lives in Details, found with
    an explicit search rather than a bare `next(...)` (which raised a bare
    `StopIteration` against the old TABLE labels once this row moved)."""
    import json
    arms = {a["name"]: a for a in json.loads(ds.DEFAULT_THIRD_CORPUS_JSON.read_text())["arms"]}
    s42, s43, cfar = arms["raddetnet_s42"], arms["raddetnet_s43"], arms["classical CFAR"]

    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="b7_raddetnet", threshold=0.44,
                               match_rule_text="rule", beat_cfar_arm_name="raddetnet")
    details = panel_of(fig)["details"]
    line = _find(details, lambda d: d.startswith("3rd corpus AP,"),
                what="the 3rd-corpus AP Details line")
    assert "D4" in line
    assert str(s42["operating_point"]["n_frames"]) in line
    assert f"{s42['AP']:.3f}" in line and f"{s43['AP']:.3f}" in line
    assert f"{cfar['AP']:.3f}" in line
    # F87: every arm here leads CFAR -- checked from the real numbers, not hardcoded.
    assert s42["AP"] > cfar["AP"] and s43["AP"] > cfar["AP"]


def test_scoreboard_offline_block_third_corpus_row_for_classical_cfar_is_its_own_ap():
    """CFAR has no seed-sibling family -- its row is just its own AP, no merge.
    Details, not the table (see the test above)."""
    import json
    cfar = next(a for a in json.loads(ds.DEFAULT_THIRD_CORPUS_JSON.read_text())["arms"]
               if a["name"] == "classical CFAR")
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="CFAR", threshold=0.66,
                               match_rule_text="rule", beat_cfar_arm_name="classical CFAR")
    details = panel_of(fig)["details"]
    line = _find(details, lambda d: d.startswith("3rd corpus AP,"),
                what="the 3rd-corpus AP Details line")
    # No sibling merge for CFAR (it has no "_s<seed>" family) -- its value is just its
    # own AP, not the "{a}/{b} (2 seeds) CFAR {c}" merged form the sibling case uses.
    assert line.endswith(f": {cfar['AP']:.3f}")


def test_scoreboard_offline_block_omits_third_corpus_row_when_file_missing(tmp_path,
                                                                          beat_cfar_data):
    """Checked in Details now (2026-09-24 redesign): this row was never in the visible
    table to begin with (see the two tests above), so checking `labels` alone would
    pass trivially whether or not this file exists."""
    import json
    path = tmp_path / "beat_cfar.json"
    path.write_text(json.dumps(beat_cfar_data))
    missing = tmp_path / "no_such_third_corpus.json"

    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="b7_raddetnet", threshold=0.44,
                               match_rule_text="rule", beat_cfar_json_path=path,
                               beat_cfar_arm_name="raddetnet", third_corpus_json_path=missing)
    details = panel_of(fig)["details"]
    assert not any(d.startswith("3rd corpus AP,") for d in details)


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
    order (see `scoreboard_figure`'s inline comment).

    2026-09-24 redesign: the connector row moved OFF the table into Details, so the
    original "must sit between the offline and live TABLE rows" ordering claim no
    longer has a table position to sit at -- it is a background line, one click away,
    while the live "unmatched / frame" and offline "AP," rows stay in the visible
    table (checked below). What survives is the row's CONTENT (which frame counts it
    names) and that it only appears when there is an offline block to point at."""
    target = (10.0, 0.0, "vehicle")
    hit = (10.0, 0.0, 0.9, 10.0)
    scores = ds.score_frames([[hit]] * 5, [[target]] * 5)
    fig = ds.scoreboard_figure(scores, arm_name="b7_raddetnet", threshold=0.44,
                               match_rule_text="rule", beat_cfar_arm_name="raddetnet")
    labels, _values = _table(fig).cells.values
    n_frames = next(a["operating_point"]["n_frames"] for a in beat_cfar_data["arms"]
                    if a.get("operating_point"))
    # Still visible, in the table: the live row and the offline AP row.
    live_label = _find(labels, lambda l: l.startswith("unmatched / frame, these"),
                       what="the live 'unmatched / frame' table row")
    assert any(l.startswith("AP,") for l in labels)
    assert "5" in live_label
    # The connector itself: a Details line ("<label>: <value>"), not a table row.
    details = panel_of(fig)["details"]
    connector_line = _find(details, lambda d: "live counts vary:" in d,
                           what="the connector Details line")
    assert "5" in connector_line
    assert connector_line.endswith(f": {n_frames}-frame numbers are the claim")


def test_scoreboard_no_connector_row_without_an_offline_block():
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="ML", threshold=0.5,
                               match_rule_text="rule")
    details = panel_of(fig)["details"]
    assert not any("live counts vary:" in d for d in details)


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
    """RETIRED geometric budget (2026-09-24 layout redesign): the height used to be
    derived from the arm's actual row/annotation-line count (a CI row, a seed-spread
    caveat, the connector row and a two-row OOD block could push it toward a
    ~1000 px cap, and a real Playwright render once clipped the last row at zero
    slack -- the old `ds._TABLE_RENDER_SAFETY_PX` fudge factor this docstring used to
    cite). Under the panel-meta contract every arm variant -- with no offline block,
    with CFAR's (no CI/OOD rows), and with raddetnet's full offline block (CI,
    seed-spread, OOD, 3rd-corpus, all moved to Details) -- gets the exact SAME fixed
    height, because the table caps at <=8 visible rows and everything past that goes
    to Details instead of growing the figure. That equality, not a "<=1000" ceiling,
    is the real invariant now."""
    scores = ds.score_frames([[]], None)
    for beat_cfar_arm_name in (None, "classical CFAR", "raddetnet"):
        fig = ds.scoreboard_figure(scores, arm_name="b7_raddetnet", threshold=0.44,
                                   match_rule_text=ds.match_rule_text(),
                                   beat_cfar_arm_name=beat_cfar_arm_name)
        assert fig.layout.height == FIGURE_HEIGHT[PANEL_ROW_TABLE], \
            (beat_cfar_arm_name, fig.layout.height)


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
