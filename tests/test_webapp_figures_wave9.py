"""Wave 9 (2026-09-24 hostile-expert read of the rendered demo screens, plus one
same-day coordinator addendum) fixes owned by the pipeline_runner/detector_scoreboard/
app shard:

1. (BLOCKER) `detector_scoreboard.stored_pr_figure`'s top margin now comes from the
   title's own "<br>" line count, floored at `_PR_MIN_TITLE_LINES` (4) -- so arm A (a
   3-line title) and arm B (`webapp.app._arm_result` appends a 4th line saying the
   curve is identical on both arms) render at the SAME margin, and neither overprints
   its own y-axis tick.
2. The scoreboard's "FA/frame at recall ..." and "AP, offline test split" rows -- the
   two that carry the demo's headline claim -- now print CFAR's own number inline, for
   a LEARNED detector's arm only, read from the same beat_cfar.json the other rows
   already read (never typed).
3. The tracker panel's "warm-start settled level" reference line no longer collides
   with the right-hand axis when it moves to "top right" to dodge an early-frame
   marker (padded off the edge, `annotation_xshift`); the collision-avoidance choice
   itself (`annotation_position`) is UNCHANGED, so wave 8's own pinned test
   (`test_settled_level_annotation_moves_when_an_early_frame_collides`) still holds.
4. The detector map's "labels & scoring stop at N m" annotation is padded off the
   plot's right border the same way.
5. The range-azimuth/range-elevation heatmaps now draw their peak-median dB
   statistic a second time, LARGE, inside the map's own top-left corner (never a
   second computation -- the same number the subtitle already states), with a
   translucent background. RE-CORRECTED (coordinator re-check, same day): first
   placed top-right, which covered thrust1's real ~115 m streak (range ~93-108 m,
   sin(azimuth) 0.5-1.0 is exactly the top-right corner); moved to top-left
   (range > ~90 m, sin(azimuth) in [-1, -0.5]), empty on every munich preset checked.
6. Run notes (`pipeline_runner.run_pipeline`'s `run_notes`) now render on the Results
   tab, one small line per arm under that arm's own banner -- previously they reached
   only the Block Diagram tab's status line. FOLLOW-UP (coordinator re-check, same
   day): each note is one line, truncated at its first " -- " separator or 160 chars
   (whichever first, `_truncate_note`) -- the live-chain gate's own note ran a
   headline number, " -- ", then a 2-3-line attribution clause, illegible at 11 px on
   every Thrust 5 arm and pushing the figures down; the banner above already carries
   the gate's verdict. The FULL text is unaffected on the Block Diagram status line
   (`_note_for`, untouched). Thrust 4's "scale model x2 (... GHz)" clause has no
   " -- " and sits at the front of its note, so it survives either cut.
7. (coordinator addendum, twice-corrected) The range-azimuth/range-elevation subline
   now also states, per frame, where the brightest VISIBLE (beyond the direct-path
   band) return actually is. The direct-path/leakage cell's own size is stated in
   PHYSICAL units ("one {gate size} m gate ..."), never as a pixel count: a first
   version divided this module's own DECLARED plot-domain constant by the bin
   count, which is not the browser's actual rendered pixel height (measured wrong
   against the PNG) -- a claim this module has no way to verify from inside
   Python, so it no longer makes one.

SECOND hostile-expert read (2026-09-24, same day, against the wave-9-fixed screens
this file already tests -- items numbered "1S..7S" below to avoid colliding with 1-7
above):

1S. (BLOCKER) Item 1's fix did not hold: the margin CODE was fixed, but the actual
    PIXEL requirement was under-measured (the old linear "+24 px/line" extrapolation
    was never re-verified against a real browser). A standalone Playwright
    measurement (`.g-gtitle`/`.bg` bounding rects, not committed to the repo) found
    a real ~11.6 px OVERLAP at the old n_lines=4 margin (108 px) and re-calibrated
    `_PR_MARGIN_T_BASE`/`_PR_MARGIN_T_PER_LINE` (65/55) against measured, positive
    (~20+ px) gaps at both n_lines=3 and 4; the figure's own `height` now grows with
    the margin so the PLOT area does not shrink.
2S. `range_profile`'s own title used a FIXED `t=40` margin that was never wired to
    `_heatmap_margin_t` at all, so its two-line median-floor/direct-path subline
    (added wave 7/8) ran through the plot's own top border and "0" tick on both
    Thrust 4 arms. Now sized (and its title PINNED) the same way `_heatmap` is.
3S. `_heatmap_margin_t`'s old per-line growth (60 px, extrapolated from one
    wave-8 measurement) assumed Plotly auto-positions a title just above the plot;
    measured instead, the title floats somewhere IN BETWEEN, at a position that
    itself scales with `margin.t` -- so a 6-line subline (item 7's own growth)
    left ~145 px of blank space ABOVE the title on thrust1_circuit_knobs.
    `_heatmap` now PINS the title to a small, fixed offset from the card's own top
    (`title.yref="container"`), and `_heatmap_margin_t` is re-calibrated (measured)
    against that pin: 36 px/line, not 60.
4S. "sub-pixel here, not visible" (item 7's own wording) was itself an unverifiable
    -- and, on a different frame, measurably FALSE -- claim: cancel_results.png
    frame 2 shows the direct-path gate as a visible bright stripe. Reworded to
    state only the gate's fixed geometric size and where a stripe would sit if
    bright, never that it is invisible.
5S. Every heat map's colorbar title ("dB rel. peak<br>(clipped at ...)") was two
    lines with no line-count-aware sizing of its own, and its second line
    collided with the colorbar's "0" tick. Now one line.
6S. RE-VERIFIED, not changed: the coordinator's second read reported the tracker
    panel's settled-level annotation still colliding on Thrust 2 arm A / Thrust 3
    arm B under a proposed "always top-left, shift up" redesign. A fresh render +
    magnified crop of BOTH screens (this same session, after item 3's wave-9-first
    `annotation_xshift` fix) found NO collision on either -- the annotation's
    closing ")" clears the right-hand axis with a visible gap in both cases. Left
    UNCHANGED rather than implementing an alternative that could not be reproduced
    and would have broken wave 8's own pinned collision-avoidance test
    (`test_settled_level_annotation_moves_when_an_early_frame_collides`, which
    requires the two cases' `xanchor` to differ -- incompatible with "always
    top-left"). Flagged for the coordinator to confirm against their own PNG.
7S. `stored_pr_figure`'s title names the scene-tier subdirectory only
    ("benchmark_v1_D2"), shared by more than one dataset root (the offline scoring
    corpus AND the live Thrust 5 demo corpus). Now prints "{dataset root}/{tier}"
    (e.g. "b1_bench_v3/benchmark_v1_D2"), read from the same `manifest` path every
    other field on this title already reads.
"""
from __future__ import annotations

import re

import pytest

from webapp import detector_scoreboard as ds
from webapp.pipeline_runner import (
    _DIRECT_PATH_EXCLUSION_M, _SUBSPACE_ERR_SETTLED_LEVEL,
    figures_from_outputs,
)


def _all_text(component) -> str:
    """Every string found anywhere in a Dash component tree, joined -- same helper
    `tests/test_webapp_ab.py` (unowned) defines for itself; duplicated here rather
    than cross-imported so this file stays self-contained."""
    if isinstance(component, str):
        return component
    parts = []
    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        parts.extend(_all_text(c) for c in children if c is not None)
    elif children is not None:
        parts.append(_all_text(children))
    return " ".join(parts)


# --------------------------------------------------------------------------------
# Item 1: stored_pr_figure margin sized from (a floor on) the title's line count
# --------------------------------------------------------------------------------
def test_stored_pr_figure_margin_reserves_the_floor_line_count():
    fig = ds.stored_pr_figure()
    # The base figure's OWN title is 3 lines (main + 2 sup lines) -- fewer than the
    # floor -- so the margin must come from the floor, not the actual count.
    assert fig.layout.title.text.count("<br>") + 1 == 3
    expected = (ds._PR_MARGIN_T_BASE
               + ds._PR_MARGIN_T_PER_LINE * (ds._PR_MIN_TITLE_LINES - 2))
    assert fig.layout.margin.t == expected


def test_pr_figure_margin_is_identical_between_arm_a_and_b(monkeypatch):
    """`webapp.app._arm_result` appends a 4th title line for arm B only; both arms'
    copies of this figure must still share one margin, or their plot axes do not
    line up (item 1's own defect)."""
    import webapp.app as appmod

    def _outputs():
        return {"_axis_meta": {"source": "x", "n_steps_run": 1, "cancelled": False,
                               "detector": {"mode": "cfar", "threshold": 0.5,
                                           "label": "CFAR"}}}

    result_a = appmod._arm_result(1, _outputs(), 1, {}, "", "", arm="a")
    result_b = appmod._arm_result(1, _outputs(), 1, {}, "", "", arm="b")
    pr_a = result_a["figs"]["detector_pr_stored"]
    pr_b = result_b["figs"]["detector_pr_stored"]
    assert pr_b.layout.title.text.count("<br>") == pr_a.layout.title.text.count("<br>") + 1
    assert pr_a.layout.margin.t == pr_b.layout.margin.t


# --------------------------------------------------------------------------------
# Item 2: inline CFAR reference on the FA/frame and AP rows, learned detectors only
# --------------------------------------------------------------------------------
@pytest.fixture
def beat_cfar_data():
    import json
    return json.loads(ds.DEFAULT_BEAT_CFAR_JSON.read_text())


def _table(fig):
    assert len(fig.data) == 1
    table = fig.data[0]
    assert table.type == "table"
    return table


def test_fa_and_ap_rows_carry_cfar_reference_for_a_learned_detector(beat_cfar_data):
    arm = next(a for a in beat_cfar_data["arms"] if a["name"] == "raddetnet")
    cfar = next(a for a in beat_cfar_data["arms"] if a["name"] == "classical CFAR")
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="b7_raddetnet", threshold=0.44,
                               match_rule_text="rule", beat_cfar_arm_name="raddetnet")
    labels, values = _table(fig).cells.values
    row = dict(zip(labels, values))

    fa_label = next(l for l in labels if l.startswith("FA/frame at recall"))
    assert row[fa_label] == (f"{arm['operating_point']['fp_per_frame']:.2f} "
                             f"(CFAR {cfar['operating_point']['fp_per_frame']:.2f})")
    assert row["AP, offline test split"] == f"{arm['AP']:.3f} (CFAR {cfar['AP']:.3f})"


def test_cfar_arm_prints_no_self_reference(beat_cfar_data):
    scores = ds.score_frames([[]], None)
    fig = ds.scoreboard_figure(scores, arm_name="CFAR", threshold=0.66,
                               match_rule_text="rule",
                               beat_cfar_arm_name="classical CFAR")
    labels, values = _table(fig).cells.values
    row = dict(zip(labels, values))
    assert "CFAR" not in row["AP, offline test split"]
    fa_label = next(l for l in labels if l.startswith("FA/frame at recall"))
    assert "CFAR" not in row[fa_label]


# --------------------------------------------------------------------------------
# Item 3/4: annotation padding off the right edge, collision-avoidance choice kept
# --------------------------------------------------------------------------------
def test_settled_level_annotation_is_padded_off_the_edge_when_it_moves_right():
    """Mirrors wave 8's own pinned collision test (xanchor still differs -- the
    choice of side is unchanged), and additionally checks the new padding: a
    colliding run gets a negative xshift so its closing ")" clears the right-hand
    axis tick; a clear run gets none."""
    collide = figures_from_outputs(
        {"subspace_err": [0.5, _SUBSPACE_ERR_SETTLED_LEVEL + 0.01, 0.3]}
    )["subspace_err"]
    clear = figures_from_outputs({"subspace_err": [0.5, 0.4, 0.3]})["subspace_err"]

    def _hline_annotation(fig):
        for ann in fig.layout.annotations:
            if "settled level" in (ann.text or ""):
                return ann
        raise AssertionError("no settled-level annotation found")

    ann_collide = _hline_annotation(collide)
    ann_clear = _hline_annotation(clear)
    assert ann_collide.xanchor != ann_clear.xanchor       # wave 8's own invariant
    assert (ann_collide.xshift or 0) < 0
    assert (ann_clear.xshift or 0) == 0


def test_detector_map_range_line_annotation_is_padded_off_the_right_edge():
    torch = pytest.importorskip("torch")

    det = torch.rand((1, 4, 4))
    outputs = {
        "cfar_detections": [[]], "cfar_detection": [det],
        "_axis_meta": {"detector": {"mode": "cfar", "threshold": 0.5, "label": "CFAR"}},
    }
    fig = figures_from_outputs(outputs)["cfar_detection"]
    ann = next(a for a in fig.layout.annotations if "labels & scoring stop" in a.text)
    assert (ann.xshift or 0) < 0


# --------------------------------------------------------------------------------
# Item 5: the large corner statistic repeats the subtitle's own peak-median number
# --------------------------------------------------------------------------------
def test_range_az_corner_annotation_matches_the_subtitle_stat():
    torch = pytest.importorskip("torch")

    ra = torch.rand((8, 8)).to(torch.complex64)
    fig = figures_from_outputs({"range_az": [ra]})["range_az"]
    subtitle = fig.layout.title.text.replace("<br>", " ")
    m = re.search(r"peak - median, dB: (-?\d+\.\d)", subtitle)
    assert m is not None, subtitle
    stat = m.group(1)

    corner = next(a for a in fig.layout.annotations if a.text.startswith("peak-median"))
    assert corner.text == f"peak-median {stat} dB"
    assert corner.font.size >= 26
    # Anchored to the heatmap's OWN domain, not the whole figure's paper coordinates
    # -- a paper-anchored corner sat under the colorbar (item 5's own defect).
    assert corner.xref == "x domain" and corner.yref == "y domain"
    # TOP-LEFT, not top-right (coordinator re-check, 2026-09-24): top-right covered
    # thrust1's real ~115 m streak (range ~93-108 m, sin(azimuth) 0.5-1.0). A
    # translucent background protects whatever sits under the corner either way.
    assert corner.xanchor == "left" and corner.x < 0.5
    assert corner.bgcolor is not None and "rgba" in corner.bgcolor


def test_range_el_corner_annotation_matches_the_subtitle_stat():
    torch = pytest.importorskip("torch")

    re_ = torch.rand((8, 8)).to(torch.complex64)
    fig = figures_from_outputs({"range_el": [re_]})["range_el"]
    subtitle = fig.layout.title.text.replace("<br>", " ")
    m = re.search(r"peak - median, dB: (-?\d+\.\d)", subtitle)
    assert m is not None, subtitle
    corner = next(a for a in fig.layout.annotations if a.text.startswith("peak-median"))
    assert corner.text == f"peak-median {m.group(1)} dB"


def test_corner_annotation_tracks_the_slider_per_frame():
    """Two frames with visibly different peak-median stats -- the corner text in the
    LAST frame's own layout (what `_heatmap` builds the base figure from) must match
    that frame's own number, not the first frame's."""
    torch = pytest.importorskip("torch")

    flat = torch.ones((8, 8), dtype=torch.complex64)          # peak == median -> 0 dB
    spiky = torch.ones((8, 8), dtype=torch.complex64)
    spiky[0, 0] = 100.0                                        # one hot cell
    fig = figures_from_outputs({"range_az": [flat, spiky]})["range_az"]
    corner = next(a for a in fig.layout.annotations if a.text.startswith("peak-median"))
    assert corner.text != "peak-median 0.0 dB"


# --------------------------------------------------------------------------------
# Item 6: run notes render on the Results tab, per arm, under that arm's own banner
# --------------------------------------------------------------------------------
def test_render_results_shows_each_arms_own_run_notes(monkeypatch):
    import plotly.graph_objects as go

    import webapp.app as appmod
    from webapp.demo_presets import PRESETS_BY_ID, apply_preset

    preset = PRESETS_BY_ID["thrust1_circuit_knobs"]
    state_a = apply_preset(preset)
    calls = []

    def fake_run_pipeline(state, n_steps, should_stop=None):
        idx = len(calls)
        calls.append(state)
        return {"range_az": [], "_axis_meta": {
            "source": "x", "n_steps_run": n_steps, "cancelled": False,
            "notes": [f"arm-{idx} note: evaluated at 14.25-15.75 GHz"],
        }}

    monkeypatch.setattr(appmod, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(appmod, "figures_from_outputs",
                        lambda outputs: {"range_az": go.Figure()})

    data, *_ = appmod._run_pipeline(1, state_a, preset.n_steps, "", None)

    # Each arm carries its OWN note (arms can differ -- item 6's own requirement),
    # ONE (short, unwrapped) note per arm -> unchanged by `_truncate_note` (no " -- "
    # and well under 160 chars).
    assert data["_notes"] == ["arm-0 note: evaluated at 14.25-15.75 GHz"]
    assert data["_previous"]["_notes"] == ["arm-1 note: evaluated at 14.25-15.75 GHz"]

    tree = appmod._render_results(data, "tab-results")
    text = _all_text(tree)
    assert "arm-0 note: evaluated at 14.25-15.75 GHz" in text
    assert "arm-1 note: evaluated at 14.25-15.75 GHz" in text


def test_render_results_single_run_shows_notes(monkeypatch):
    import plotly.graph_objects as go

    import webapp.app as appmod
    from webapp.demo_presets import PRESETS_BY_ID, apply_preset

    state = appmod._with_param(apply_preset(PRESETS_BY_ID["thrust5_detector_cfar"]),
                               "detector", "threshold", 0.9)

    def fake_run_pipeline(state_arg, n_steps, should_stop=None):
        return {"subspace_err": [], "_axis_meta": {
            "source": "x", "n_steps_run": n_steps, "cancelled": False,
            "notes": ["single-run note: evaluated at 14.25-15.75 GHz"],
        }}

    monkeypatch.setattr(appmod, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(appmod, "figures_from_outputs",
                        lambda outputs: {"subspace_err": go.Figure()})

    data, *_ = appmod._run_pipeline(1, state, 5, "", None)
    assert data.get("_notes") == ["single-run note: evaluated at 14.25-15.75 GHz"]
    text = _all_text(appmod._render_results(data, "tab-results"))
    assert "single-run note: evaluated at 14.25-15.75 GHz" in text


def test_render_results_no_notes_key_when_axis_meta_has_none(monkeypatch):
    """No run notes this run -> no stray empty line, same convention as `_note_for`."""
    import plotly.graph_objects as go

    import webapp.app as appmod
    from webapp.demo_presets import PRESETS_BY_ID, apply_preset

    state = appmod._with_param(apply_preset(PRESETS_BY_ID["thrust5_detector_cfar"]),
                               "detector", "threshold", 0.9)

    def fake_run_pipeline(state_arg, n_steps, should_stop=None):
        return {"subspace_err": [], "_axis_meta": {
            "source": "x", "n_steps_run": n_steps, "cancelled": False,
        }}

    monkeypatch.setattr(appmod, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(appmod, "figures_from_outputs",
                        lambda outputs: {"subspace_err": go.Figure()})

    data, *_ = appmod._run_pipeline(1, state, 5, "", None)
    assert "_notes" not in data


# --------------------------------------------------------------------------------
# Item 6 follow-up (2026-09-24): each note truncated at its first " -- " or 160
# chars, one line per note -- the live-chain gate's note otherwise ran 3-5 lines of
# 11 px text on every Thrust 5 arm.
# --------------------------------------------------------------------------------
def test_truncate_note_cuts_at_the_first_double_dash_separator():
    import webapp.app as appmod

    # The real shape of the live-chain gate's own note (pipeline_runner.
    # _StoredADCGateBlock.note): headline number, then " -- ", then an attribution
    # clause long enough on its own to run past 160 chars.
    note = ("live chain vs stored ADC over 5 frame(s): max |diff| = 1 of 8 LSB "
           "(3-bit) (6.994e-05 absolute) -- DIFFERS -- this run's cube is not the "
           "stored one (this run: ADC 3-bit, IF corner 1 m, front end on)")
    truncated = appmod._truncate_note(note)
    assert truncated == (
        "live chain vs stored ADC over 5 frame(s): max |diff| = 1 of 8 LSB "
        "(3-bit) (6.994e-05 absolute) ...")
    assert "the usual answer" not in truncated
    assert len(truncated) < len(note)


def test_truncate_note_cuts_at_160_chars_when_no_separator_is_that_close():
    import webapp.app as appmod

    note = "x" * 200
    truncated = appmod._truncate_note(note)
    assert truncated == "x" * 160 + " ..."


def test_truncate_note_leaves_a_short_note_with_no_separator_unchanged():
    import webapp.app as appmod

    note = "a short note with no separator at all"
    assert appmod._truncate_note(note) == note


def test_truncate_note_preserves_the_thrust4_frequency_disclosure():
    """The exact regression this follow-up must not cause: Thrust 4's interconnect
    note (`InterconnectBlock.describe()`, e2e/blocks.py -- not owned, read only) has
    no " -- " and is a little over 160 chars, but the required clause sits at the
    FRONT of it and must survive whichever cut applies."""
    import webapp.app as appmod

    note = ("Tessera TSV surrogate, scale model x2 (2x geometry, evaluated at "
           "14.25-15.75 GHz): radius 2.5 um, pitch 30 um, height 50 um, liner "
           "0.25 um, 300 K, ring3x3 arrangement")
    assert " -- " not in note
    assert len(note) > appmod._NOTE_TRUNCATE_CHARS
    truncated = appmod._truncate_note(note)
    assert "scale model x2 (2x geometry, evaluated at 14.25-15.75 GHz)" in truncated


def test_notes_line_truncates_every_note_in_the_list():
    import webapp.app as appmod

    axis_meta = {"notes": ["short one", "long " + "x" * 200 + " -- attribution"]}
    lines = appmod._notes_line(axis_meta)
    assert lines[0] == "short one"
    assert lines[1] == appmod._truncate_note(axis_meta["notes"][1])
    assert lines[1].endswith(" ...")


def test_render_results_renders_one_line_per_note_not_one_joined_paragraph():
    """The Results tab used to join every note into one "|"-separated paragraph;
    each note now gets its own line (checked via the rendered tree shape, not just
    substring presence -- a joined paragraph would also contain both substrings)."""
    import webapp.app as appmod

    data = {"_notes": ["note one", "note two"]}
    block = appmod._notes_block(data["_notes"])
    line_texts = [_all_text(child) for child in block.children]
    assert line_texts == ["note one", "note two"]


# --------------------------------------------------------------------------------
# Item 7 (coordinator addendum): direct-path invisibility + brightest-visible-return
# --------------------------------------------------------------------------------
def test_range_az_states_direct_path_is_invisible_and_the_brightest_visible_return():
    torch = pytest.importorskip("torch")
    import numpy as np

    from webapp.pipeline_runner import _nonnegative_range, _range_axis, _range_per_gate_m

    # A frame whose peak sits at range 0 (the direct path) and a single, weaker,
    # genuinely resolved return further out -- so "brightest visible" must name
    # something other than the peak itself. freq_span_hz picked (not the wave-8
    # tests' 3e9) so this tiny 8-bin test axis actually extends past
    # `_DIRECT_PATH_EXCLUSION_M`.
    n_freqs, freq_span_hz, bins = 64, 3e8, 8
    y_full = _range_axis(bins, freq_span_hz, n_freqs)
    keep = _nonnegative_range(y_full)
    y_cropped = y_full[keep]
    orig_rows = np.where(keep)[0]
    zero_row = orig_rows[int(np.argmin(np.abs(y_cropped)))]
    beyond = np.where(y_cropped >= _DIRECT_PATH_EXCLUSION_M)[0]
    assert beyond.size, "test geometry must reach past the exclusion band"
    bright_row = orig_rows[beyond[0]]

    # `_to_numpy_abs_db` transposes before returning (see its own `.T`), so a value
    # placed to land at DISPLAY row `r` after that transpose goes at `z[:, r]`.
    z = np.full((bins, bins), 1e-6, dtype=np.complex64)
    z[0, zero_row] = 1.0     # zero-range gate: the direct-path/leakage peak
    z[3, bright_row] = 0.1   # a genuine return, beyond the exclusion band
    ra = torch.from_numpy(z)
    fig = figures_from_outputs({
        "range_az": [ra],
        "_axis_meta": {"n_freqs": n_freqs, "freq_span_hz": freq_span_hz,
                       "range_az_bins": bins},
    })["range_az"]
    text = fig.layout.title.text.replace("<br>", " ")
    # Physical gate size, never a pixel count (RETRACTED, coordinator re-check,
    # 2026-09-24: a first version divided this module's own declared plot-domain
    # constant by the bin count, which is not the browser's actual rendered pixel
    # height and printed a wrong number against the real PNG).
    gate_m = _range_per_gate_m(bins, freq_span_hz, n_freqs)
    assert f"0 dB cell at range 0 is one {gate_m:.2g} m gate" in text
    # RETRACTED (wave 9, second hostile-expert read, 2026-09-24, item 4): "not
    # visible" was itself unverifiable and measurably false on some frames
    # (cancel_results.png frame 2 shows this gate as a visible bright stripe) --
    # the gate's own BRIGHTNESS is real data that varies frame to frame, unlike its
    # fixed geometric size. Says where a stripe would sit if it IS bright, never
    # that it is invisible.
    assert "may show as a thin stripe at the bottom edge" in text
    assert "not visible" not in text
    assert " px" not in text
    expected_db = 10 * np.log10(0.1 / 1.0)
    expected_range = float(y_cropped[beyond[0]])
    assert (f"brightest visible return: {expected_db:.1f} dB at "
           f"{expected_range:.0f} m") in text


def test_direct_path_note_absent_without_axis_metadata():
    torch = pytest.importorskip("torch")

    ra = torch.rand((8, 8)).to(torch.complex64)
    fig = figures_from_outputs({"range_az": [ra]})["range_az"]
    assert "brightest visible return" not in fig.layout.title.text


def test_direct_path_exclusion_constant_is_used_not_hardcoded_elsewhere():
    """Guards the module's own claim that the exclusion band is a single named
    constant, not a literal repeated in the note-building code."""
    assert _DIRECT_PATH_EXCLUSION_M == 2.0


# --------------------------------------------------------------------------------
# Second hostile-expert read (2026-09-24, same day): items 1S-7S, see module
# docstring. Re-checks/extends the first-read tests above; does not replace them.
# --------------------------------------------------------------------------------
def test_stored_pr_figure_title_includes_the_dataset_root_not_just_the_tier(beat_cfar_data):
    """Item 7S: "benchmark_v1_D2" alone is shared by more than one dataset root."""
    import json
    from pathlib import Path

    manifest = beat_cfar_data["manifest"]
    root, tier = Path(manifest).parent.parent.name, Path(manifest).parent.name
    fig = ds.stored_pr_figure()
    assert f"{root}/{tier}" in fig.layout.title.text
    # Guards against a regression back to the bare, ambiguous tier name: the tier
    # alone must not appear WITHOUT its root immediately before it.
    assert f", {tier}" not in fig.layout.title.text.replace(f"{root}/{tier}", "")


def test_pr_margin_gives_a_positive_measured_gap_not_just_a_formula(beat_cfar_data):
    """Item 1S: the margin must be large enough in absolute terms, not merely
    "computed from the line count" (the BLOCKER regression: the code path was
    already line-count-driven and still overlapped). Pins the RE-CALIBRATED
    constants directly, so a future edit that quietly shrinks them again is
    caught here even without re-running Playwright."""
    assert ds._PR_MARGIN_T_BASE >= 65
    assert ds._PR_MARGIN_T_PER_LINE >= 50
    fig = ds.stored_pr_figure()
    # n_lines is floored at 4 (`_PR_MIN_TITLE_LINES`); this is the exact value a
    # real headless-browser measurement (this session, not committed) found gives
    # a ~20 px clear gap between the title's rendered bottom and the plot's top.
    assert fig.layout.margin.t == pytest.approx(
        ds._PR_MARGIN_T_BASE + ds._PR_MARGIN_T_PER_LINE * (ds._PR_MIN_TITLE_LINES - 2))
    assert fig.layout.margin.t >= 170


def test_pr_figure_height_grows_with_margin_so_the_plot_does_not_shrink():
    """Item 1S: `height` must scale with the re-calibrated (taller) margin, or the
    fix for the overlap would come at the cost of squeezing the PR curves down to
    a sliver."""
    fig = ds.stored_pr_figure()
    assert fig.layout.height == (ds._PR_PLOT_DOMAIN_HEIGHT + fig.layout.margin.t
                                 + ds._PR_MARGIN_B)


def test_range_profile_title_margin_scales_with_its_own_line_count():
    """Item 2S: `range_profile`'s panel used to hardcode `t=40` regardless of its
    title's line count."""
    torch = pytest.importorskip("torch")
    from webapp.pipeline_runner import _heatmap_margin_t

    prof = torch.rand(8, dtype=torch.float32)
    fig = figures_from_outputs({
        "range_profile_agg": [prof],
        "_axis_meta": {"n_freqs": 64, "freq_span_hz": 3e9, "range_profile_bins": 8},
    })["range_profile"]
    assert fig.layout.title.text.count("<br>") + 1 == 2   # main + one median/direct-path line
    assert fig.layout.margin.t == _heatmap_margin_t(fig.layout.title.text)
    assert fig.layout.margin.t > 40, "must have grown past the old flat default"


def test_heatmap_title_is_pinned_to_the_cards_own_top_not_floating():
    """Item 3S: `_heatmap`'s title must be explicitly anchored to the card's own
    top (container-relative), not left at Plotly's default floating position --
    that default is what produced ~145 px of blank space above a 6-line title."""
    torch = pytest.importorskip("torch")

    ra = torch.rand((8, 8)).to(torch.complex64)
    fig = figures_from_outputs({
        "range_az": [ra],
        "_axis_meta": {"n_freqs": 64, "freq_span_hz": 3e9, "range_az_bins": 8},
    })["range_az"]
    assert fig.layout.title.yref == "container"
    assert fig.layout.title.yanchor == "top"
    assert fig.layout.title.y >= 0.9


def test_heatmap_margin_t_per_line_is_smaller_than_the_pre_pin_value():
    """Item 3S: re-calibrated against the PINNED title (see the test above) --
    the old 60 px/line, measured against the unpinned default, would now leave
    the plot needlessly short since the title no longer eats into the margin
    the same way."""
    from webapp.pipeline_runner import _HEATMAP_MARGIN_T_PER_LINE
    assert _HEATMAP_MARGIN_T_PER_LINE < 60
    assert _HEATMAP_MARGIN_T_PER_LINE >= 30   # still enough to clear a real sup line


def test_frame_layout_title_override_keeps_the_same_pin():
    """Item 3S follow-through: a `go.Frame(layout=dict(title=...))` REPLACES the
    whole title object, so the per-frame override used for the slider must repeat
    the pin, or the title would jump to the floating default the moment the
    slider moves off its initial frame."""
    torch = pytest.importorskip("torch")

    ra1 = torch.rand((8, 8)).to(torch.complex64)
    ra2 = torch.rand((8, 8)).to(torch.complex64)
    fig = figures_from_outputs({
        "range_az": [ra1, ra2],
        "_axis_meta": {"n_freqs": 64, "freq_span_hz": 3e9, "range_az_bins": 8},
    })["range_az"]
    assert len(fig.frames) == 2
    for frame in fig.frames:
        ft = frame.layout.title
        assert ft.yref == "container" and ft.yanchor == "top"


def test_heatmap_colorbar_title_is_one_line():
    """Item 5S: a two-line colorbar title collided with the colorbar's own "0"
    tick on every heat map."""
    torch = pytest.importorskip("torch")

    ra = torch.rand((8, 8)).to(torch.complex64)
    fig = figures_from_outputs({"range_az": [ra]})["range_az"]
    assert "<br>" not in fig.data[0].colorbar.title.text
    assert "clipped at" in fig.data[0].colorbar.title.text


def test_settled_level_annotation_clears_the_right_axis_on_both_flagged_screens():
    """Item 6S (re-verify, not a redesign): a fresh render of the two screens the
    second hostile read flagged (Thrust 2 arm A shape, Thrust 3 arm B shape) --
    reproduced here as the same underlying data shape rather than the full preset
    -- must still show the padding fix from item 3 (first read): a colliding
    early frame moves the annotation right AND clears the axis via `xshift`."""
    collide = figures_from_outputs(
        {"subspace_err": [0.5, _SUBSPACE_ERR_SETTLED_LEVEL + 0.01, 0.3, 0.3, 0.3, 0.3]}
    )["subspace_err"]
    ann = next(a for a in collide.layout.annotations if "settled level" in (a.text or ""))
    assert ann.xanchor == "right"
    assert (ann.xshift or 0) < 0


# --------------------------------------------------------------------------------
# Coordinator report, 2026-09-24 (after the second read): the animation slider's
# currentvalue label ("frame N") overlapped the play/pause buttons on Thrust 5's
# narrower (~700 px) cards -- "frame 5" rendered as "rame 5". The buttons occupy a
# FIXED pixel width; the slider's own x is necessarily fractional (no pixel anchor
# in Plotly's slider schema), so the fraction has to clear the worst-case (narrowest
# real) card width, not just the wide single-card layouts this was first tuned
# against.
# --------------------------------------------------------------------------------
def test_slider_buttons_x_extent_was_widened_not_silently_shrunk():
    from webapp.pipeline_runner import _SLIDER_BUTTONS_X_EXTENT, _SLIDER_LEN, _SLIDER_X

    # 0.34 is the value a standalone Playwright measurement (this session, not
    # committed) found clears the ~139 px fixed-width button group on Thrust 5's
    # own ~600-700 px card widths; a regression back toward the old 0.16 would
    # silently reopen the "rame N" defect without any figure-dict test catching it
    # (no unit test here renders in a real browser), so this guards the constant
    # directly rather than only the derived, relative wave-2 check.
    assert _SLIDER_BUTTONS_X_EXTENT >= 0.30
    assert _SLIDER_X == pytest.approx(_SLIDER_BUTTONS_X_EXTENT + 0.08)
    assert _SLIDER_LEN == pytest.approx(0.98 - _SLIDER_X)
    assert _SLIDER_LEN > 0.4, "the slider track itself must stay usably long"
