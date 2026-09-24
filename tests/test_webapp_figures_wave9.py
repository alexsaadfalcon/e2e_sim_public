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

LAYOUT SPEC, 2026-09-24 (retires most of the geometry machinery items 1/1S/2S/3S/5S
tuned above): a figure carries no title, no subtitle and no colour-bar title any
more, and per-figure sliders/updatemenus (item -- the slider-geometry constants
below) are gone -- ONE HTML transport drives every animated figure from the
run-identity row now (`webapp/app.py::_transport_bar` + `assets/results_clock.js`).
Every clause a title/subtitle/annotation used to carry lives in `layout.meta.panel`
instead, reached through `panel_of`/`panel_text` (see
`webapp/pipeline_runner.py`'s "PANEL GEOMETRY AND THE PANEL-META CONTRACT" section).
Consequences for the tests below, spelled out per-test rather than here: some are
repointed onto `panel_text`/the new fixed-geometry constants; some protect a defect
class (a title's line count driving margin/height) that is now structurally
impossible and are deleted with a comment naming what covers the same risk instead
(mostly `tests/test_webapp_layout_acceptance.py`'s checks 4/6/7/10/20, which apply
across every product `figures_from_outputs` builds, including these).

Separately, `webapp/app.py`'s run-notes rendering changed shape (2026-09-24, same
day as the layout spec): `_truncate_note`/`_notes_block`/`_NOTE_TRUNCATE_CHARS` are
gone. Notes render UNTRUNCATED inside each arm's `html.Details` disclosure
(`_details_lines`); the one-line, possibly-shortened text that used to be
`_truncate_note`'s job is now `_note_headline`'s, used ONLY for the arm's one-line
caption (`_arm_caption`), not for what actually reaches the Details body.
"""
from __future__ import annotations

import pytest

from webapp import detector_scoreboard as ds
from webapp.pipeline_runner import (
    _DIRECT_PATH_EXCLUSION_M, _SUBSPACE_ERR_SETTLED_LEVEL,
    figures_from_outputs, panel_caption, panel_text,
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
# Item 1/1S: stored_pr_figure margin -- RETIRED as "sized from (a floor on) the
# title's line count": the figure carries no title at all now (layout spec,
# 2026-09-24), so `_PR_MARGIN_T_BASE`/`_PR_MARGIN_T_PER_LINE`/`_PR_MIN_TITLE_LINES`
# are gone with it -- there is no line count left to floor. What replaces the
# defect these protected (a margin computed from text that can grow) is a FIXED
# margin, same convention as every other panel under the panel-meta contract; the
# generic version of this invariant, for the OTHER figure family
# (`figures_from_outputs`), is `test_webapp_layout_acceptance.py`'s checks 4/20.
# This module builds a DIFFERENT figure (`detector_scoreboard.stored_pr_figure`,
# not covered by that file), so the fixed-geometry check is kept here instead of
# being dropped as a pure duplicate.
# --------------------------------------------------------------------------------
def test_stored_pr_figure_geometry_is_a_fixed_constant_not_derived_from_a_caption():
    """The margin is now `_PR_MARGIN_T`, a plain constant, and `height` is the
    row's own fixed `FIGURE_HEIGHT[PANEL_ROW_PR]` -- neither derives from text any
    more (item 1S's "height must grow with the margin" no longer applies: nothing
    grows). Identical whether the panel's caption (the only text left whose length
    varies by call) is the short "{n} test frames" line or a highlighted arm's
    longer one."""
    from webapp.pipeline_runner import FIGURE_HEIGHT, PANEL_ROW_PR

    short = ds.stored_pr_figure()
    long_ = ds.stored_pr_figure(highlight_arm="raddetnet")
    assert short.layout.margin.t == long_.layout.margin.t == ds._PR_MARGIN_T
    assert short.layout.height == long_.layout.height == FIGURE_HEIGHT[PANEL_ROW_PR]
    assert not (short.layout.title and short.layout.title.text)


def test_pr_figure_geometry_is_identical_between_arm_a_and_b(monkeypatch):
    """The "identical on both arms" clause goes on ARM A's panel caption (hostile
    round 11, C6 -- the heat-map panels put their "same colour scale on both arms"
    clause there, and two conventions for the same kind of statement on one screen
    make the ABSENCE of a clause carry meaning), and it is added at RENDER time by
    `webapp.app._mark_pr_identical_on_arm_a`, when the page knows there IS a second
    arm; `_arm_result` adds it to neither arm. Both arms' copies of this figure must
    still share one geometry, or their plot axes do not line up (item 1's own defect,
    in its new home)."""
    import webapp.app as appmod

    def _outputs():
        return {"_axis_meta": {"source": "x", "n_steps_run": 1, "cancelled": False,
                               "detector": {"mode": "cfar", "threshold": 0.5,
                                           "label": "CFAR"}}}

    result_a = appmod._arm_result(1, _outputs(), 1, {}, "", "", arm="a")
    result_b = appmod._arm_result(1, _outputs(), 1, {}, "", "", arm="b")
    pr_a = result_a["figs"]["detector_pr_stored"]
    pr_b = result_b["figs"]["detector_pr_stored"]
    assert "identical on both arms" not in panel_caption(pr_a)
    assert "identical on both arms" not in panel_caption(pr_b)
    assert pr_a.layout.margin.t == pr_b.layout.margin.t
    assert pr_a.layout.height == pr_b.layout.height
    # The render-time pass is what puts it on arm A, and only when arm B is there.
    figs_a = {"detector_pr_stored": pr_a.to_dict()}
    figs_b = {"detector_pr_stored": pr_b.to_dict()}
    appmod._mark_pr_identical_on_arm_a(figs_a, figs_b)
    assert "identical on both arms" in panel_caption(figs_a["detector_pr_stored"])
    assert "identical on both arms" not in panel_caption(figs_b["detector_pr_stored"])
    # A single-arm screen (the cancel journey) must not claim a comparison it does
    # not show: no arm B, no clause.
    figs_solo = {"detector_pr_stored": pr_a.to_dict()}
    appmod._mark_pr_identical_on_arm_a(figs_solo, {})
    assert "identical on both arms" not in panel_caption(figs_solo["detector_pr_stored"])


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
    """The IN-PLOT tag's wording shortened to " scoring <= N m " (layout spec
    section 4, "stated once per row" -- a short tag at the line's end, not a
    centred sentence) when the six-line subtitle this once lived in was retired;
    the full "labels & scoring stop at N m." sentence survives verbatim in Details
    (acceptance check 15) -- checked here via `panel_text` rather than only assumed."""
    torch = pytest.importorskip("torch")

    det = torch.rand((1, 4, 4))
    outputs = {
        "cfar_detections": [[]], "cfar_detection": [det],
        "_axis_meta": {"detector": {"mode": "cfar", "threshold": 0.5, "label": "CFAR"}},
    }
    fig = figures_from_outputs(outputs)["cfar_detection"]
    ann = next(a for a in fig.layout.annotations if "scoring" in (a.text or ""))
    assert (ann.xshift or 0) < 0
    assert "labels & scoring stop at" in panel_text(fig)


# --------------------------------------------------------------------------------
# Item 5: the large corner statistic repeated the subtitle's own peak-median number,
# top-left, over the plot, with a translucent background so it would not obscure
# whatever real data sat under it.
#
# RETIRED (layout spec, 2026-09-24, check 8 -- the layout redesign's own headline
# defect): drawing a statistic OVER the data at all, translucent background or not,
# is exactly what check 8 exists to forbid -- item 5's "protect whatever sits under
# it either way" was a mitigation for a placement the new spec removes outright. The
# statistic now lives in the reserved strip ABOVE the axes (`_stat_annotations`,
# `name="stat_strip"`), which structurally cannot cover a return; there is also only
# ONE computation of the number now (the strip's own), not a separate subtitle
# figure and a corner figure that could drift apart, so
# `test_range_az_corner_annotation_matches_the_subtitle_stat` /
# `..._range_el_...` protected a consistency bug that is no longer possible to
# create and are deleted rather than repointed. What covers the remaining, still-
# live part of item 5 (>=26 px, axis-domain-anchored, no background pill, never
# over the data) is `test_webapp_layout_acceptance.py`'s
# `test_the_statistic_strip_sits_entirely_above_the_axes` and
# `test_the_statistic_strip_fits_inside_the_reserved_margin`, for every product,
# not just range_az/range_el.
#
# `test_corner_annotation_tracks_the_slider_per_frame` protects a DIFFERENT,
# still-real risk (the headline number must reflect the frame it is shown on, not a
# stale first-frame value) and is repointed onto the strip below rather than deleted.
# --------------------------------------------------------------------------------
def test_stat_strip_tracks_the_frame_it_is_shown_on():
    """Item 5's per-frame half, repointed onto the reserved strip: the BASE figure's
    `stat_strip` annotation must carry the LAST frame's own number, and each
    animation frame's own layout override must carry THAT frame's number -- never a
    value copied from frame 0."""
    torch = pytest.importorskip("torch")

    flat = torch.ones((8, 8), dtype=torch.complex64)          # peak == median -> 0 dB
    spiky = torch.ones((8, 8), dtype=torch.complex64)
    spiky[0, 0] = 100.0                                        # one hot cell
    fig = figures_from_outputs({"range_az": [flat, spiky]})["range_az"]

    def _stat_text(layout):
        for ann in (layout.annotations or ()):
            if ann.name == "stat_strip":
                return ann.text
        raise AssertionError("no stat_strip annotation")

    # Base figure is built from the LAST (spiky) frame -- must not read as flat.
    # The strip is ONE annotation now (hostile round 11, D8): headline in bold, then
    # the per-frame readouts after a separator, at one size.
    assert "<b>0.0 dB peak−median</b>" not in _stat_text(fig.layout)
    # Frame 0's own override must carry frame 0's (flat) number, not the base
    # figure's.
    assert _stat_text(fig.frames[0].layout).startswith("<b>0.0 dB peak−median</b>")


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

    # Each arm carries its OWN note (arms can differ -- item 6's own requirement).
    # `data["_notes"]` is untruncated verbatim (`_truncate_note` is retired,
    # 2026-09-24 -- see the module docstring); this note is short enough that the
    # distinction is not exercised here, but the stored value itself must still be
    # exactly what the pipeline reported.
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
# Item 6 follow-up -- RETRACTED (layout spec section 2.3, 2026-09-24): truncating
# notes for the Results tab is exactly what made the smallest type on the page the
# only text that lost information, and mid-word truncation read as a crash to a
# non-expert (hostile round 10, defect 2.3; acceptance check 12). `_truncate_note`/
# `_notes_block`/`_NOTE_TRUNCATE_CHARS` are gone. The full note now reaches the
# Results tab UNTRUNCATED, inside the arm's `Details` disclosure
# (`_details_lines`); `_note_headline` (up to the first " -- " or
# `_NOTE_HEADLINE_CHARS`=110 chars) survives, but only feeds the arm's one-line
# CAPTION (`_arm_caption`) now, never what actually reaches Details. The four tests
# below repoint onto `_note_headline`'s own contract, same clauses pinned, and a
# fifth checks the untruncated Details rendering directly.
# --------------------------------------------------------------------------------
def test_note_headline_cuts_at_the_first_double_dash_separator():
    import webapp.app as appmod

    # The real shape of the live-chain gate's own note (pipeline_runner.
    # _StoredADCGateBlock.note): headline number, then " -- ", then an attribution
    # clause.
    note = ("live chain vs stored ADC over 5 frame(s): max |diff| = 1 of 8 LSB "
           "(3-bit) (6.994e-05 absolute) -- DIFFERS -- this run's cube is not the "
           "stored one (this run: ADC 3-bit, IF corner 1 m, front end on)")
    headline = appmod._note_headline(note)
    # A cut at " -- " is a complete sentence -- nothing elided, so no truncation
    # mark (acceptance check 12).
    assert headline == (
        "live chain vs stored ADC over 5 frame(s): max |diff| = 1 of 8 LSB "
        "(3-bit) (6.994e-05 absolute)")
    assert "the usual answer" not in headline
    assert "…" not in headline
    assert len(headline) < len(note)


def test_note_headline_cuts_at_a_word_boundary_and_marks_it_when_no_close_separator():
    import webapp.app as appmod

    note = "x" * 200
    headline = appmod._note_headline(note)
    # No spaces to break on -> the plain char-budget cut, marked (it IS elided).
    assert headline == "x" * appmod._NOTE_HEADLINE_CHARS + "…"


def test_note_headline_leaves_a_short_note_with_no_separator_unchanged():
    import webapp.app as appmod

    note = "a short note with no separator at all"
    assert appmod._note_headline(note) == note
    assert "…" not in appmod._note_headline(note)


def test_note_headline_preserves_the_thrust4_frequency_disclosure():
    """The exact regression this follow-up must not cause: Thrust 4's interconnect
    note (`InterconnectBlock.describe()`, e2e/blocks.py -- not owned, read only) has
    no " -- " and runs past the headline char budget, but the required clause sits
    at the FRONT of it and must survive the cut."""
    import webapp.app as appmod

    note = ("Tessera TSV surrogate, scale model x2 (2x geometry, evaluated at "
           "14.25-15.75 GHz): radius 2.5 um, pitch 30 um, height 50 um, liner "
           "0.25 um, 300 K, ring3x3 arrangement")
    assert " -- " not in note
    assert len(note) > appmod._NOTE_HEADLINE_CHARS
    headline = appmod._note_headline(note)
    assert "scale model x2 (2x geometry, evaluated at 14.25-15.75 GHz)" in headline


def test_notes_line_returns_every_note_untruncated():
    """`_notes_line` (feeds the Details body, NOT the caption) no longer shortens
    anything -- repoints the retired "truncates every note" premise onto the new,
    opposite contract."""
    import webapp.app as appmod

    axis_meta = {"notes": ["short one", "long " + "x" * 200 + " -- attribution"]}
    lines = appmod._notes_line(axis_meta)
    assert lines == axis_meta["notes"]


def test_details_lines_render_one_line_per_note_not_one_joined_paragraph():
    """`_notes_block` is gone; notes now render inside the arm's Details disclosure
    via `_details_lines`, one `details-line` Div per note (checked via the rendered
    tree shape, not just substring presence -- a joined paragraph would also
    contain both substrings)."""
    import webapp.app as appmod

    payload = {"_notes": ["note one", "note two"]}
    lines = appmod._details_lines(payload, {}, "")
    assert [_all_text(l) for l in lines] == ["note one", "note two"]


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
    text = panel_text(fig)
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
    assert "brightest visible return" not in panel_text(fig)


def test_direct_path_exclusion_constant_is_used_not_hardcoded_elsewhere():
    """Guards the module's own claim that the exclusion band is a single named
    constant, not a literal repeated in the note-building code."""
    assert _DIRECT_PATH_EXCLUSION_M == 2.0


# --------------------------------------------------------------------------------
# Second hostile-expert read (2026-09-24, same day): items 1S-7S, see module
# docstring. Re-checks/extends the first-read tests above; does not replace them.
# --------------------------------------------------------------------------------
def test_stored_pr_figure_details_includes_the_dataset_root_not_just_the_tier(beat_cfar_data):
    """Item 7S: "benchmark_v1_D2" alone is shared by more than one dataset root.
    The figure carries no title any more; the same clause moved into
    `pipeline_runner.set_panel`'s Details (`stored_pr_figure`'s own docstring),
    reached through `panel_text`."""
    from pathlib import Path

    manifest = beat_cfar_data["manifest"]
    root, tier = Path(manifest).parent.parent.name, Path(manifest).parent.name
    fig = ds.stored_pr_figure()
    text = panel_text(fig)
    assert f"{root}/{tier}" in text
    # Guards against a regression back to the bare, ambiguous tier name: the tier
    # alone must not appear WITHOUT its root immediately before it.
    assert f", {tier}" not in text.replace(f"{root}/{tier}", "")


# Item 1S ("the margin must be large enough in absolute terms, not merely computed
# from the line count") and its height-scaling follow-through are RETIRED
# (2026-09-24): the BLOCKER they protected -- a title overlapping the plot -- is
# now structurally impossible, because the figure carries no title at all to
# overlap anything (layout spec). `_PR_MARGIN_T_BASE`/`_PR_MARGIN_T_PER_LINE`/
# `_PR_MIN_TITLE_LINES`/`_PR_PLOT_DOMAIN_HEIGHT` are gone with the mechanism they
# tuned. What survives of "large enough, not just computed" is
# `test_stored_pr_figure_margin_is_a_fixed_constant_not_derived_from_a_caption`
# above (item 1's own repoint) -- not duplicated here.

# Item 2S ("range_profile's t=40 was never wired to `_heatmap_margin_t`") is
# RETIRED the same way: `_heatmap_margin_t` is now a constant function for every
# product (`test_webapp_layout_acceptance.py::test_heatmap_margin_t_ignores_its_argument`),
# so "is `range_profile` wired to it" is no longer a question with an interesting
# answer -- every product shares the one fixed `_FIG_MARGIN_T`
# (`test_webapp_layout_acceptance.py::test_every_figure_has_the_same_top_margin`,
# which iterates every key `figures_from_outputs` returns, `range_profile`
# included).


# Items 3S and 5S are RETIRED wholesale (2026-09-24): both protected the interaction
# between a per-figure TITLE and the top margin it grew -- `_heatmap`'s title
# pin (`test_heatmap_title_is_pinned_to_the_cards_own_top_not_floating`), the
# per-line margin constant (`test_heatmap_margin_t_per_line_is_smaller_than_the_pre_pin_value`,
# `_HEATMAP_MARGIN_T_PER_LINE`), the per-frame title override needing to repeat that
# pin (`test_frame_layout_title_override_keeps_the_same_pin`), and the colour-bar
# title's own line count (`test_heatmap_colorbar_title_is_one_line`). Figures carry
# no title, no per-frame title override and no colour-bar title at all under the
# layout spec, so none of these properties exist to test any more. What covers the
# same class of risk now, generically, for every product:
#   - no title anywhere, ever, including per-frame --
#     test_webapp_layout_acceptance.py::test_no_figure_carries_a_title_or_a_subtitle
#   - no colour-bar title --
#     test_webapp_layout_acceptance.py::test_no_colorbar_carries_a_title
#   - the margin is one fixed constant, not derived from anything --
#     test_webapp_layout_acceptance.py::test_every_figure_has_the_same_top_margin,
#     ::test_heatmap_margin_t_ignores_its_argument
#   - frames still animate (just via `annotations`, not `title`) --
#     test_webapp_layout_acceptance.py::test_animated_figures_still_carry_their_frames,
#     and this file's own test_stat_strip_tracks_the_frame_it_is_shown_on above.


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
#
# RETIRED (layout spec section 4, "Slider / play controls"; wave 11): per-figure
# sliders and play/pause buttons are gone outright -- `webapp/assets/results_clock.js`
# drives every animated figure from ONE HTML transport in the run-identity row
# (`webapp/app.py::_transport_bar`) instead. `_SLIDER_X`/`_SLIDER_LEN`/
# `_SLIDER_ROW_Y` no longer exist; `_SLIDER_BUTTONS_X_EXTENT`/`_SLIDER_MARGIN_B`
# survive in `pipeline_runner.py` ONLY as unused, documentary constants (their own
# comment there says so), so pinning their value no longer guards any live
# behaviour -- there is nothing left in this module for a "0.16 regression" to
# silently reopen. What replaced the defect (a clipped "frame N" label) is CSS/JS
# pixel geometry in `webapp/assets/`, outside what a figure-dict test can see at
# all -- `test_webapp_layout_acceptance.py`'s own module docstring names
# `python -m webapp.rehearse` as the one tool that can check it, for exactly this
# reason. What IS still this module's job and stays covered:
# `test_webapp_layout_acceptance.py::test_no_figure_carries_its_own_slider_or_play_buttons`
# guards that no per-figure transport (this test's real subject) ever comes back.
