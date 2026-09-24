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
        "display 0-",
        "unambig (neg.-delay half cropped)",
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
        "unambig (neg.-delay half cropped)",
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
}


@pytest.mark.parametrize("product", sorted(OLD_SUBTITLE_CLAUSES))
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


def test_the_static_range_profile_says_it_is_static(figs):
    """Hostile round 10, section 3.2: the range-profile panel is built from the LAST
    frame and does not animate while the map above it loops, and nothing said so."""
    anns = " ".join(a.text for a in (figs["range_profile"].layout.annotations or ()))
    assert "static" in anns
    assert not (figs["range_profile"].frames or ())


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
