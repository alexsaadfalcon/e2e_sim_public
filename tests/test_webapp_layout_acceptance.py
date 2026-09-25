"""The layout spec's acceptance checks, as far as a figure dict and a Dash tree can
carry them (2026-09-24).

WHAT THIS FILE CAN AND CANNOT DO. A figure dictionary is not the screen: the checks
that are genuinely about PIXELS (plot-area fraction, the y of the first panel, total
page height, rendered ink size, a truncation ellipsis the browser drew) are measured by
`python -m webapp.rehearse`, which dumps a `<preset>_geometry.json` straight out of the
DOM next to each PNG -- see `webapp/rehearse.py::_GEOMETRY_JS`. This file pins what is
knowable WITHOUT a browser, so a regression is caught by `pytest` rather than only by
someone remembering to look:

  * a figure never grows a title or a subtitle back (checks 6, 7);
  * every figure's height is its ROW's fixed height, not a function of its own text
    (checks 4, 20 -- the defect the whole spec exists to remove);
  * no per-figure transport ever comes back (check 10);
  * every panel declares the same plot background (check 13);
  * nothing inside a figure is under 17 px, nothing on the page under 15 px (check 1);
  * the statistic annotation sits ABOVE the axes rectangle, never over the data
    (check 8);
  * and -- the one the spec says must not be skipped -- every clause the old panel
    subtitles carried is still reachable in the panel's Details (check 15).

Synthetic data only; no Sionna, no .pkl frames, no display.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from webapp import pipeline_runner as pr
from webapp.pipeline_runner import (
    FIGURE_HEIGHT,
    PANEL_HEIGHT,
    PANEL_ROW_MAP,
    figures_from_outputs,
    panel_of,
    panel_text,
    share_heatmap_z_limits,
)

#: The spec's own floors (section 3). 15 px anywhere on a results screen; 17 px
#: anywhere inside a figure.
MIN_PAGE_FONT_PX = 15
MIN_FIGURE_FONT_PX = 17


# ----------------------------------------------------------------------------------
# Fixtures: one run's outputs for the munich-style products, built from noise.
# ----------------------------------------------------------------------------------

def _frames(seed: int, n: int = 3, bins: int = 64):
    rng = np.random.default_rng(seed)
    return [torch.tensor(rng.normal(size=(bins, bins))
                         + 1j * rng.normal(size=(bins, bins))) for _ in range(n)]


def _outputs(seed: int = 1):
    frames = _frames(seed)
    return {
        "fft": frames,
        "range_az": frames,
        "range_el": frames,
        "range_profile_agg": [torch.abs(f).mean(1) for f in frames],
        "subspace_err": [0.5, 0.3, 0.2],
        "_axis_meta": {"n_freqs": 128, "freq_span_hz": 3e9,
                       "fft_bins": 64, "range_az_bins": 64, "range_el_bins": 64,
                       "range_profile_bins": 64},
    }


@pytest.fixture(scope="module")
def figs():
    return figures_from_outputs(_outputs(1))


def _detector_outputs():
    """One corpus-replay frame: a range-Doppler cube plus a CFAR objectness map with
    one detection and one ground-truth label."""
    rng = np.random.default_rng(3)
    cube = rng.normal(size=(2, 16, 8)) + 1j * rng.normal(size=(2, 16, 8))
    obj = np.zeros((1, 8, 16), dtype=np.float32)
    return {
        "radar_cube": [cube],
        "cfar_detection": [obj],
        "cfar_detections": [[(0, 0.1, 0.9, 12.0)]],
        "gt_detections": [[(0, -0.2, 1.0, 20.0)]],
        "_axis_meta": {
            "rx": {"grid": {"max_range_m": 40.0}},
            "detector": {"mode": "cfar", "threshold": 0.66,
                         "label": "CA-CFAR (guard 2, train 6)"},
        },
    }


@pytest.fixture(scope="module")
def detector_figs():
    return figures_from_outputs(_detector_outputs())


def _walk_fonts(obj, path="layout"):
    """Every ``font.size`` anywhere in a figure dict, with where it came from."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "font" and isinstance(v, dict) and v.get("size") is not None:
                out.append((f"{path}.{k}", float(v["size"])))
            else:
                out.extend(_walk_fonts(v, f"{path}.{k}"))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.extend(_walk_fonts(v, f"{path}[{i}]"))
    return out


# ----------------------------------------------------------------------------------
# Checks 6 + 7: the figure carries no title and no subtitle, at all, ever.
# ----------------------------------------------------------------------------------

def test_no_figure_carries_a_title_or_a_subtitle(figs):
    """Check 7. The mechanism behind defects 1, 3 and 5 was `_heatmap_margin_t()`
    growing the top margin per wrapped subtitle line; a title that cannot exist cannot
    grow."""
    for key, fig in figs.items():
        title = fig.layout.title
        text = getattr(title, "text", None) if title is not None else None
        assert not text, f"{key} grew a figure title back: {text!r}"
        for frame in (fig.frames or ()):
            f_title = getattr(frame.layout, "title", None)
            f_text = getattr(f_title, "text", None) if f_title is not None else None
            assert not f_text, f"{key} frame grew a title back: {f_text!r}"


def test_no_figure_text_contains_a_sup_subtitle(figs):
    """Check 7, second half: `<sup>` was how every subtitle was drawn."""
    for key, fig in figs.items():
        blob = str(fig.to_dict())
        assert "<sup>" not in blob, f"{key} still draws a <sup> subtitle"


def test_no_colorbar_carries_a_title(figs):
    """Section 4: the colour-bar title was ~310 px wide at 20 px and Plotly bought that
    width by shrinking the plot (defect 5). Units and clip are in the caption."""
    for key, fig in figs.items():
        for trace in fig.data:
            cbar = getattr(trace, "colorbar", None)
            if cbar is None:
                continue
            text = getattr(getattr(cbar, "title", None), "text", None)
            assert not text, f"{key}'s colour bar grew a title back: {text!r}"


# ----------------------------------------------------------------------------------
# Checks 4 + 20: fixed row geometry, identical on every screen.
# ----------------------------------------------------------------------------------

def test_every_figure_height_is_its_rows_fixed_height(figs):
    """Checks 4 and 20. A geometry that depends on how long that day's caption happens
    to be is the defect this whole spec exists to remove."""
    for key, fig in figs.items():
        row = panel_of(fig).get("row")
        assert row in FIGURE_HEIGHT, f"{key} declares no row kind"
        assert fig.layout.height == FIGURE_HEIGHT[row], (
            f"{key} is {fig.layout.height} px, not its row's {FIGURE_HEIGHT[row]}")


def test_every_figure_has_the_same_top_margin(figs):
    """Check 4: the two arms' plot-area top edges must land on the same y. They are the
    same product built by the same code, so this is really "the top margin is a
    constant" -- which `_heatmap_margin_t` now is."""
    tops = {key: fig.layout.margin.t for key, fig in figs.items()}
    assert set(tops.values()) == {pr._FIG_MARGIN_T}, tops


def test_every_figure_has_the_same_left_margin(figs):
    """Check 20 / hostile round 11 D7: the plot ORIGIN has to land on the same x down
    a column, or the left edge staggers from panel to panel (measured on the rendered
    page: 100 px objectness, 107 subspace, 109 PR, 112 maps, 115 range profile).

    `margin.l` is a FLOOR -- Plotly's auto-expansion grows it to whatever each panel's
    own y-axis ticks need -- so the constant has to be wide enough that expansion
    never fires, and every panel kind has to use the SAME constant. The
    stored-PR panel keeps its own copy of the number (it is a different module's
    layout decision); this is what pins the two equal."""
    from webapp import detector_scoreboard as ds

    lefts = {key: fig.layout.margin.l for key, fig in figs.items()}
    assert set(lefts.values()) == {pr._FIG_MARGIN_L}, lefts
    assert ds._PR_MARGIN_L == pr._FIG_MARGIN_L, (
        f"the stored-PR panel starts at {ds._PR_MARGIN_L} px while every other panel "
        f"starts at {pr._FIG_MARGIN_L}")


def test_the_same_product_has_the_same_geometry_on_two_different_runs():
    """Check 20, directly: two runs with different DATA (so different statistics, and
    different caption lengths) must produce byte-identical panel geometry."""
    a = figures_from_outputs(_outputs(1))
    b = figures_from_outputs(_outputs(2))
    assert set(a) == set(b)
    for key in a:
        assert (a[key].layout.height, a[key].layout.margin.t, a[key].layout.margin.b,
                a[key].layout.margin.l, a[key].layout.margin.r) == \
               (b[key].layout.height, b[key].layout.margin.t, b[key].layout.margin.b,
                b[key].layout.margin.l, b[key].layout.margin.r), key


def test_heatmap_margin_t_ignores_its_argument():
    """The constant this used to not be. Passing a six-line title must not move it."""
    assert pr._heatmap_margin_t("") == pr._heatmap_margin_t("a<br>b<br>c<br>d<br>e<br>f")


# ----------------------------------------------------------------------------------
# Check 10: exactly one transport per screen -- so zero per figure.
# ----------------------------------------------------------------------------------

def test_no_figure_carries_its_own_slider_or_play_buttons(figs):
    """Check 10. `assets/results_clock.js` has driven every animated figure from one
    clock since wave 11; the per-figure copies cost 130 px of bottom margin each and
    the presenter never touched them."""
    for key, fig in figs.items():
        assert not (fig.layout.sliders or ()), f"{key} grew a slider back"
        assert not (fig.layout.updatemenus or ()), f"{key} grew play buttons back"


def test_animated_figures_still_carry_their_frames(figs):
    """The transport was removed; the ANIMATION was not. The clock steps frames by
    name, so the names must still be "0".."n-1"."""
    assert [f.name for f in figs["range_az"].frames] == ["0", "1", "2"]


# ----------------------------------------------------------------------------------
# Check 13: one plot background for every panel on a screen.
# ----------------------------------------------------------------------------------

def test_every_panel_declares_the_same_backgrounds(figs):
    """Check 13. Today: white behind maps, `#E5ECF6` behind charts, so two panels in
    the same column read as two different products."""
    papers = {key: fig.layout.paper_bgcolor for key, fig in figs.items()}
    plots = {key: fig.layout.plot_bgcolor for key, fig in figs.items()}
    assert set(papers.values()) == {pr.PAPER_BGCOLOR}, papers
    assert set(plots.values()) == {pr.PLOT_BGCOLOR}, plots


# ----------------------------------------------------------------------------------
# Check 1: minimum legible size.
# ----------------------------------------------------------------------------------

def test_nothing_inside_a_figure_is_under_17px(figs):
    """Check 1, the in-figure half. (The PAGE half is measured in the browser -- see
    this module's docstring -- because a CSS class's computed size is not knowable
    here.)"""
    for key, fig in figs.items():
        for where, size in _walk_fonts(fig.to_dict()):
            assert size >= MIN_FIGURE_FONT_PX, f"{key} {where} is {size} px"


# ----------------------------------------------------------------------------------
# Check 8: the statistic never sits on the data.
# ----------------------------------------------------------------------------------

def test_the_statistic_strip_sits_entirely_above_the_axes(figs):
    """Check 8. The old callout was drawn INSIDE the axes at y-domain 0.94 with a
    translucent pill and hid the whole 105-125 m range band on every map."""
    for key, fig in figs.items():
        for ann in (fig.layout.annotations or ()):
            if not str(getattr(ann, "name", "") or "").startswith(
                    pr._STAT_ANNOTATION_FLAG):
                continue
            assert ann.yref == "y domain", f"{key}: statistic is not axis-anchored"
            assert ann.y > 1.0, f"{key}: statistic at y={ann.y} is over the data"
            assert ann.yanchor == "bottom", f"{key}: statistic anchored downward"
            assert not ann.bgcolor, f"{key}: statistic kept its background pill"


def test_the_statistic_strip_fits_inside_the_reserved_margin(figs):
    """The strip is the figure's TOP MARGIN. An annotation placed past it is drawn
    outside the figure and clipped -- the failure mode the old per-line margin sizing
    existed to avoid, reintroduced at a different anchor."""
    plot_h = FIGURE_HEIGHT[PANEL_ROW_MAP] - pr._FIG_MARGIN_T - pr._FIG_MARGIN_B
    for key, fig in figs.items():
        for ann in (fig.layout.annotations or ()):
            name = str(getattr(ann, "name", "") or "")
            if not name.startswith(pr._STAT_ANNOTATION_FLAG):
                continue
            top_px = (ann.y - 1.0) * plot_h + float(ann.font.size)
            assert top_px <= pr._FIG_MARGIN_T, (
                f"{key} {name} reaches {top_px:.0f} px into a "
                f"{pr._FIG_MARGIN_T} px strip")


def test_the_statistic_takes_the_arms_colour():
    """Section 5: A `#4b6584`, B `#3867d6`. Applied by the app, because the figure
    builder does not know which arm it will be shown as."""
    a = {k: v.to_dict() for k, v in figures_from_outputs(_outputs(1)).items()}
    b = {k: v.to_dict() for k, v in figures_from_outputs(_outputs(2)).items()}
    pr.apply_arm_style(a, "a")
    pr.apply_arm_style(b, "b")

    def _stat_color(fig):
        for ann in (fig["layout"].get("annotations") or []):
            if ann.get("name") == pr._STAT_ANNOTATION_FLAG:
                return ann["font"]["color"]
        return None

    assert _stat_color(a["range_az"]) == pr.ARM_COLORS["a"]
    assert _stat_color(b["range_az"]) == pr.ARM_COLORS["b"]
    # And on every animation frame: a frame's layout REPLACES the annotations list, so
    # a colour applied only to the base figure vanishes the moment the clock steps.
    frame_colors = {ann["font"]["color"]
                    for frame in b["range_az"]["frames"]
                    for ann in (frame["layout"].get("annotations") or [])
                    if ann.get("name") == pr._STAT_ANNOTATION_FLAG}
    assert frame_colors == {pr.ARM_COLORS["b"]}


# ----------------------------------------------------------------------------------
# The panel header: one-line title, one-line caption.
# ----------------------------------------------------------------------------------

def test_every_panel_declares_a_title_and_a_caption(figs):
    for key, fig in figs.items():
        panel = panel_of(fig)
        assert panel.get("title"), f"{key} has no panel title"
        assert panel.get("caption"), f"{key} has no panel caption"


def test_every_caption_is_one_line_and_at_most_110_characters(figs):
    """Section 3. The caption is rendered on ONE line with no wrap, so a caption longer
    than the column is a caption the browser clips -- and check 12 forbids a truncation
    mark in visible text."""
    for key, fig in figs.items():
        caption = pr.panel_caption(fig)
        assert "<br>" not in caption and "\n" not in caption, key
        assert len(caption) <= 110, f"{key} caption is {len(caption)} chars: {caption}"


def test_every_title_is_one_line(figs):
    for key, fig in figs.items():
        title = panel_of(fig)["title"]
        assert "<br>" not in title and "\n" not in title, key
        assert len(title) <= 60, f"{key} title is {len(title)} chars: {title}"


# ----------------------------------------------------------------------------------
# CHECK 15 -- the one that must not be skipped.
# ----------------------------------------------------------------------------------

#: Every clause the pre-2026-09-24 panel subtitles carried, per product, verbatim from
#: `webapp/pipeline_runner.py` as it stood at commit c33bb64 (wave 11) and from the
#: rendered `thrust1_circuit_knobs_results.png` of 2026-09-24 14:27. The honesty
#: content had to be MOVED, never dropped, and this is the pin that says so.
#:
#: SCOPE (CLAUDE.md's provenance rule): these are the strings that were on screen on
#: 2026-09-24. A clause deliberately RETIRED later must be removed from this list in
#: the same commit that retires it, with the reason -- not left here to be worked
#: around.
OLD_SUBTITLE_CLAUSES = {
    "range_az": [
        "non-coherent over elevation",
        "peak - median, dB",
        "0 dB = direct path",
        "(0 = earliest arrival)",
        "not a target",
        "m/gate",
        "native",
        # RETIRED 2026-09-24 (shard 3), with the pipeline they described: "display 0-N m
        # of an M m unambig (neg.-delay half cropped)" was written when each product ran
        # its own range FFT and this module cropped the fftshifted axis at display time.
        # The spine's RangeTransformBlock crops before any product sees the cube, so
        # there is no display-time crop left to disclose. The same two facts -- what is
        # shown and out of what window -- are still on the caption, in the owner's
        # bistatic convention: "0-249 m shown of a 500 m window, bistatic excess path".
        "m shown of a",
        "window, bistatic excess path",
        "0 dB cell at range 0 is one",
        "may show as a thin stripe at the bottom edge",
        "brightest visible return",
        "clip",
    ],
    "range_el": [
        "non-coherent over azimuth",
        "peak - median, dB",
        "0 dB = direct path",
        "(0 = earliest arrival)",
        "not a target",
        "m/gate",
        # Retired with its range_az twin above, same reason, same replacement.
        "m shown of a",
        "brightest visible return",
    ],
    "range_profile": [
        "non-coherent over channels",
        "median floor, dB rel. peak",
        "0 dB = direct path at range 0, not a target",
    ],
    "subspace_err": [
        "unnormalised distance",
        "grows ~sqrt(k), not a fraction",
    ],
    "fft": [
        "non-coherent",
        "over range",
    ],
    # The corpus-replay products. `_detector_outputs` below builds them, so these are
    # checked against the same accessor as the munich panels rather than being left to
    # tests/test_detector_scoreboard.py alone.
    "radar_cube": [
        "clip",
        "median floor + 3 dB",
    ],
    "cfar_detection": [
        "detections at objectness >=",
        # The panel follows the transport now (hostile round 11, H3), so what it
        # states is which frame of how many is on screen -- not that it is pinned.
        "the frame the transport is parked on",
        "hit = cross inside the box",
        "labels & scoring stop at",
    ],
}


@pytest.mark.parametrize("product",
                         sorted(set(OLD_SUBTITLE_CLAUSES)
                                - {"radar_cube", "cfar_detection"}))
def test_every_clause_the_old_subtitle_carried_is_still_reachable(figs, product):
    """CHECK 15. "Opening all of them loses no string that is present in today's
    screens -- the honesty content must be MOVED, never dropped. This check is the one
    that must not be skipped."

    `panel_text` is title + caption + the whole Details body, i.e. everything the
    presenter can reach in one click. WHICH of the three a clause ended up in is a
    layout decision; that it is reachable is not."""
    text = panel_text(figs[product])
    missing = [c for c in OLD_SUBTITLE_CLAUSES[product] if c not in text]
    assert not missing, f"{product} dropped: {missing}\n---\n{text}"


def test_the_shared_colour_limits_clause_survives_into_details():
    """Same check, for the clause the SHARING pass adds at render time -- it used to be
    appended to the title, which no longer exists."""
    a = {k: v.to_dict() for k, v in figures_from_outputs(_outputs(1)).items()}
    b = {k: v.to_dict() for k, v in figures_from_outputs(_outputs(2)).items()}
    share_heatmap_z_limits(a, b)
    a_text = panel_text(a["range_az"])
    b_text = panel_text(b["range_az"])
    assert "colour limits shared with arm B: zmin" in a_text
    assert "colour limits shared with arm A: zmin" in b_text
    # "(was X)" is deliberate provenance for a superseded value (hostile round 10,
    # section 5.3: keep it).
    assert "(was " in a_text


def test_the_shared_scale_sentence_is_stated_once_per_row_not_once_per_panel():
    """Section 4: `. same colour scale on both arms` at the end of ARM A's caption
    only; the exact pair in BOTH arms' Details. Saying it twice invites "why does it
    need saying twice?"."""
    a = {k: v.to_dict() for k, v in figures_from_outputs(_outputs(1)).items()}
    b = {k: v.to_dict() for k, v in figures_from_outputs(_outputs(2)).items()}
    share_heatmap_z_limits(a, b)
    assert pr.SHARED_SCALE_CLAUSE in pr.panel_caption(a["range_az"])
    assert pr.SHARED_SCALE_CLAUSE not in pr.panel_caption(b["range_az"])


def test_sharing_replaces_the_clip_clause_rather_than_printing_two():
    """Two clips printed on one panel read as a bug (wave 11, 2026-09-24)."""
    a = {k: v.to_dict() for k, v in figures_from_outputs(_outputs(1)).items()}
    b = {k: v.to_dict() for k, v in figures_from_outputs(_outputs(2)).items()}
    share_heatmap_z_limits(a, b)
    caption = pr.panel_caption(a["range_az"])
    assert caption.count(pr.CLIP_CLAUSE_PREFIX) == 1, caption


def test_sharing_is_idempotent():
    """`_render_results` runs on every store change; a second pass must not append a
    second clause."""
    a = {k: v.to_dict() for k, v in figures_from_outputs(_outputs(1)).items()}
    b = {k: v.to_dict() for k, v in figures_from_outputs(_outputs(2)).items()}
    share_heatmap_z_limits(a, b)
    first = panel_text(a["range_az"])
    share_heatmap_z_limits(a, b)
    assert panel_text(a["range_az"]) == first


# ----------------------------------------------------------------------------------
# Rule 2 of the brief: the statistic the presenter READS stays visible.
# ----------------------------------------------------------------------------------

def test_the_headline_statistics_are_visible_without_expanding_anything(figs):
    """peak-median (maps), the median floor (range profile) and the brightest visible
    return are all in the reserved strip or the caption -- never only in Details."""
    def _visible(fig):
        anns = " ".join(a.text for a in (fig.layout.annotations or ()))
        return f"{panel_of(fig)['title']} {pr.panel_caption(fig)} {anns}"

    assert "peak−median" in _visible(figs["range_az"])
    assert "peak−median" in _visible(figs["range_el"])
    assert "median floor" in _visible(figs["range_profile"])
    assert "brightest" in _visible(figs["range_az"])


def test_every_animated_panel_says_which_frame_it_is_on(figs):
    """Section 4: "Each animated panel ... so a photograph still identifies the frame".
    It is per-frame, in the strip, so it steps with the clock instead of drifting."""
    for key in ("range_az", "range_el", "fft"):
        fig = figs[key]
        anns = " ".join(a.text for a in (fig.layout.annotations or ()))
        assert "frame 3 of 3" in anns, key
        first = " ".join(a["text"] for a in fig.frames[0].layout.annotations)
        assert "frame 1 of 3" in first, key


def test_the_range_profile_steps_with_the_map_above_it(figs):
    """SUPERSEDED, and the supersession is the point (hostile round 12, item 14).

    Round 10 (section 3.2) found this panel built from the LAST frame while the
    range-azimuth map above it looped, with nothing saying so, and the fix was to SAY
    it: the strip read "last frame of 3 (static)". Round 12 read the result on the
    Thrust 4 screen -- "frame 2 of 3" beside "last frame of 3 (static)", one column,
    one run, two frames -- and the honest label turned out to be a worse answer than
    the animation. Every frame's profile was already computed and discarded; the panel
    now steps on the same clock, which is what makes a column one frame again.

    So: frames, and a strip that names the frame it is on."""
    fig = figs["range_profile"]
    anns = " ".join(a.text for a in (fig.layout.annotations or ()))
    assert "static" not in anns
    assert "frame" in anns
    assert len(fig.frames or ()) == len(figs["range_az"].frames or ())
    assert "same clock" in panel_text(fig)


# ----------------------------------------------------------------------------------
# Check 12: no truncation marks in the text the layout itself produces.
# ----------------------------------------------------------------------------------

def test_no_visible_panel_text_is_truncated(figs):
    """Check 12. Details bodies wrap and may contain an ellipsis a source note itself
    carried; a TITLE or CAPTION may not, because they are rendered `nowrap`."""
    for key, fig in figs.items():
        visible = f"{panel_of(fig)['title']} {pr.panel_caption(fig)}"
        assert "…" not in visible, f"{key} visible text is truncated: {visible}"
        assert " ..." not in visible, f"{key} visible text is truncated: {visible}"


def test_the_arm_caption_cuts_at_a_clause_boundary_without_a_mark():
    """`webapp/app._note_headline`: a note cut at its own " -- " is a complete
    sentence, so nothing is marked. Only a cut past the character budget is."""
    from webapp.app import _note_headline
    assert _note_headline("live vs stored ADC: max |diff| 0 of 4096 LSB -- so the "
                          "live chain IS the chain that wrote them") == \
        "live vs stored ADC: max |diff| 0 of 4096 LSB"
    assert "…" not in _note_headline("short note")
    long_one = "word " * 60
    assert _note_headline(long_one).endswith("…")
    # ... and never mid-word.
    assert not _note_headline(long_one).rstrip("…").endswith("wor")


@pytest.mark.parametrize("product", ["radar_cube", "cfar_detection"])
def test_the_corpus_replay_panels_keep_their_clauses_too(detector_figs, product):
    """CHECK 15 for the Thrust 5 products. Same rule, same accessor: the objectness
    panel's operating point, the frame it is pinned to, the hit rule and the scoring
    crop are all still reachable, and so is the range-Doppler clip's provenance."""
    text = panel_text(detector_figs[product])
    missing = [c for c in OLD_SUBTITLE_CLAUSES[product] if c not in text]
    assert not missing, f"{product} dropped: {missing} -- panel text: {text}"


def test_the_detector_panel_follows_the_same_clock_as_the_cube(detector_figs):
    """Hostile round 11, H3 (supersedes round 10's 3.2, which only asked the pinned
    panel to SAY it was pinned): the objectness panel carries one animation frame per
    stored frame, so the screen's single clock steps it in lockstep with the
    range-Doppler cube above it and the two can no longer read as different frames.

    The single-frame fixture here has nothing to animate -- that is the one case where
    no frames are correct -- so this asserts the per-frame STATISTIC, which exists
    either way, and (on a multi-frame run, see test_webapp_figures_wave9) the frames."""
    anns = " ".join(a.text for a in
                    (detector_figs["cfar_detection"].layout.annotations or ()))
    assert "frame 1 of 1" in anns
    assert "(last)" not in anns


def test_every_panel_declares_the_same_plot_background(tmp_path):
    """ACCEPTANCE CHECK 13: "every panel has the same plot background colour".

    Measured at the figure level and not on the PNG on purpose. `webapp.rehearse`'s
    browser probe reads `.js-plotly-plot .cartesianlayer .bg`, and on these screens that
    selector matches NOTHING (verified on the rendered T1/T2 geometry, 2026-09-24: every
    page reports `plotBg: []`), because a heat-map panel has no cartesian background rect
    and Plotly paints the scatter panels' from the layout instead. So the check the spec
    means -- one background across the deck -- is the one asserted here: every figure any
    screen can show declares `PLOT_BGCOLOR`, and none of them quietly keeps Plotly's
    template default.
    """
    import numpy as np
    import torch
    import webapp.detector_scoreboard as ds
    from webapp.pipeline_runner import PAPER_BGCOLOR, PLOT_BGCOLOR, figures_from_outputs

    # One of every panel kind the presets can put on a Results page, built from the
    # smallest outputs each one accepts (torch tensors, as the real products emit).
    def _m(seed, shape):
        g = torch.Generator().manual_seed(seed)
        return torch.rand(shape, generator=g) + 1e-3

    outputs = {
        "fft": [_m(0, (8, 8))],
        "range_az": [_m(1, (8, 8))],
        "range_el": [_m(2, (8, 8))],
        "range_profile_agg": [_m(3, (16,))],
        "subspace_err": [0.5, 0.2, 0.08],
        "radar_cube": [_m(4, (4, 8, 8))],
        "ber": [0.01, 0.0],
        "evm": [0.03, 0.02],
        "comm_data_eq": [torch.tensor([0.7 + 0.7j, -0.7 - 0.7j], dtype=torch.complex64)],
    }
    figs = dict(figures_from_outputs(outputs))
    figs["scoreboard"] = ds.scoreboard_figure(
        ds.score_frames([[]], [[]]), arm_name="classical CFAR", threshold=0.5,
        match_rule_text=ds.match_rule_text())
    figs["pr"] = ds.stored_pr_figure()

    offenders = []
    for key, fig in figs.items():
        layout = fig.layout if hasattr(fig, "layout") else (fig.get("layout") or {})
        plot_bg = (layout.plot_bgcolor if hasattr(layout, "plot_bgcolor")
                   else layout.get("plot_bgcolor"))
        paper_bg = (layout.paper_bgcolor if hasattr(layout, "paper_bgcolor")
                    else layout.get("paper_bgcolor"))
        if plot_bg != PLOT_BGCOLOR or paper_bg != PAPER_BGCOLOR:
            offenders.append((key, plot_bg, paper_bg))
    assert not offenders, (
        "panels whose background is not the deck's (%s on %s): %s"
        % (PLOT_BGCOLOR, PAPER_BGCOLOR, offenders))


# ----------------------------------------------------------------------------------
# Check 14: total page height <= 2200 px, as far as the DECLARED geometry can carry it
# ----------------------------------------------------------------------------------
#
# The browser measures the real number (`webapp/rehearse.py` dumps it per preset). What
# is knowable here is the SUM the layout commits to: every product row is a fixed
# height by `PANEL_HEIGHT`, the rows are stacked with a fixed gap, and the header band
# above the first panel is measured (303 px on every rendered results screen,
# 2026-09-24/25 rehearsals). That sum is what went over budget on Thrust 5 -- four
# product rows at 540 + 540 + 388 + 552 -- and it is what a fifth product row would blow
# again without anyone opening a browser.

#: Measured on the rendered pages, not declared anywhere in CSS: the distance from the
#: top of the results page to the first panel's top border. 303 px on every results
#: screen of the 2026-09-24 and 2026-09-25 rehearsals (see the `firstPanelTop` /
#: `rootTop` fields of each `<preset>_geometry.json`).
MEASURED_HEADER_BAND_PX = 303
#: `--sp-2` in webapp/assets/demo.css, the `.ab-row` bottom margin.
ROW_GAP_PX = 16
#: Acceptance check 14.
PAGE_HEIGHT_BUDGET_PX = 2200


def _declared_page_height(row_kinds) -> int:
    """What a page of these product rows commits to, top of page to last panel's
    bottom border."""
    rows = list(row_kinds)
    return (MEASURED_HEADER_BAND_PX
            + sum(PANEL_HEIGHT[k] for k in rows)
            + ROW_GAP_PX * max(0, len(rows) - 1))


def test_the_thrust5_product_rows_fit_the_page_budget():
    """The three Thrust 5 screens rendered at 2480 px against the spec's 2200 (hostile
    round 12 / shard 3b's open problem 1), and it was arithmetic: the objectness map,
    the range-Doppler map, the scoreboard table AND the offline precision-recall panel
    are 540 + 540 + 388 + 552 = 2020 px of panel before the 303 px header and the gaps.

    The PR panel is the one of the four that is not this run's product -- scored offline
    on a fixed split, identical on both arms -- so it moved into a closed disclosure
    below the rows (`webapp.app._offline_benchmark_disclosure`) and its numbers stay in
    the scoreboard row set beside it. Three rows fit; four did not."""
    from webapp.pipeline_runner import PANEL_ROW_MAP, PANEL_ROW_PR, PANEL_ROW_TABLE

    four = [PANEL_ROW_MAP, PANEL_ROW_MAP, PANEL_ROW_TABLE, PANEL_ROW_PR]
    three = four[:3]
    assert _declared_page_height(four) > PAGE_HEIGHT_BUDGET_PX, (
        "this test's own premise: four product rows do NOT fit, which is why the PR "
        "panel moved into a disclosure")
    assert _declared_page_height(three) <= PAGE_HEIGHT_BUDGET_PX, (
        _declared_page_height(three))


def test_the_offline_pr_panel_leaves_the_product_rows_for_a_closed_disclosure():
    """Moved, never deleted (acceptance check 15): the panel, its caption and its whole
    Details body are inside a `<details>` that is CLOSED by default, labelled with the
    split size READ from beat_cfar.json, and rendered ONCE for both arms."""
    import webapp.app as appmod
    from webapp import detector_scoreboard as ds
    import plotly.graph_objects as go

    pr_fig = ds.stored_pr_figure().to_dict()
    data = {
        "range_az": go.Figure().to_dict(),
        appmod.PR_PANEL_KEY: pr_fig,
        "_banner": "A: ADC bits 12 bit",
        "_ab": True,
        "_previous": {"range_az": go.Figure().to_dict(),
                      appmod.PR_PANEL_KEY: ds.stored_pr_figure().to_dict(),
                      "_banner": "B: ADC bits 3 bit", "_ab": True},
    }
    tree = appmod._render_results(data, "tab-results")
    grid = next(c for c in tree.children
                if getattr(c, "className", None) == "results-grid")
    rows = [r for r in grid.children if getattr(r, "className", None) == "ab-row"]
    # header row + ONE product row (range_az). The PR panel is not a row any more.
    assert len(rows) == 2, len(rows)

    disclosure = next((c for c in tree.children
                       if "offline-benchmark" in (getattr(c, "className", "") or "")),
                      None)
    assert disclosure is not None, "the PR panel vanished instead of moving"
    assert disclosure.open is False
    label = disclosure.children[0].children
    assert "Offline benchmark" in label
    _, n_frames = ds._load_recall_target_and_n_frames()
    if n_frames:
        assert f"{int(n_frames)}-frame test split" in label
    # ...and exactly one copy of the panel, with its Details text still reachable.
    text = _all_text_of(disclosure)
    assert text.count("Precision") <= 2
    for line in (pr.panel_of(pr_fig).get("details") or []):
        assert line in text, line


def _all_text_of(node) -> str:
    """Every string in a Dash component tree, joined."""
    out = []

    def walk(n):
        if isinstance(n, str):
            out.append(n)
            return
        if isinstance(n, (list, tuple)):
            for c in n:
                walk(c)
            return
        child = getattr(n, "children", None)
        if child is not None:
            walk(child)

    walk(node)
    return " ".join(out)
