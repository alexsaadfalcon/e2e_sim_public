"""Wave 11 (2026-09-24): the owner's live-test findings on the Thrust 1 A/B screen.

Verbatim, from a live run of "RF circuit knobs vs the image's noise floor"
(A = LNA 8 mA, B = 0.5 mA): *"images look fine, but 55 vs. 65 dB peak to median is
not discernible by human eye. Additionally, default should be side by side, with both
frames on the same colorbar limits, and both as if the play button was hit (and should
loop repeatedly). Framerate looks fine over RDP"*.

Three separate defects, pinned here:

1. **The floor difference was invisible.** Both arms' maps are peak-normalized and
   both medians sat BELOW the fixed -40 dB display clip, so both backgrounds rendered
   as the same single clip colour -- an 11-12 dB floor difference that the printed
   "peak - median" number claimed but the picture did not show. The shared colour
   limit must now REACH both arms' floors
   (`pipeline_runner.share_heatmap_z_limits`, `Z_SHARE_REACH_FLOOR`).
2. **The arms were stacked.** A's whole grid, then B's, put the two copies of the
   same product a screen height apart. They now render side by side, one row per
   product (`webapp.app._ab_columns`).
3. **Nothing moved on its own.** Each panel had to be played by hand, played once,
   and stopped on the last frame. One `dcc.Interval` clock now steps every animated
   panel together and loops (`webapp/assets/results_clock.js`).

...plus the runbook rule ("step the slider, never press Play") that item 3 retires.

LAYOUT SPEC UPDATE, 2026-09-24 (landed after this wave, requires rebuilding this
file's own low-level fixtures, not just repointing assertions):

- `pipeline_runner._heatmap` no longer takes a `title` argument (positional or
  keyword) and has no `colorbar_title` kwarg -- a figure carries no title and no
  colour-bar title at all now (see `webapp/pipeline_runner.py`'s "PANEL GEOMETRY
  AND THE PANEL-META CONTRACT" section). Every helper below that used to build a
  figure with `pr._heatmap(data, title, ...)` now builds it with `pr._heatmap(data,
  ...)` and attaches its words with `pr.set_panel(fig, title=..., caption=[...],
  details=[...], row=...)` instead, mirroring exactly what `figures_from_outputs`
  itself does for `range_az`/`radar_cube`. `_apply_shared_z` (the function items 1
  and 5S/7S exercise) now rewrites a CAPTION clause and appends a DETAILS line, not
  a substring of one subtitle blob -- so a defect this file used to probe by
  embedding text INSIDE a hand-typed title string (a "<br>"-wrapped clip clause, an
  orphan "<sup></sup>") is either rebuilt against the caption/details lists, or
  retired outright where the redesign makes the failure mode structurally
  impossible (a caption is one line, enforced elsewhere, so it cannot wrap; there
  is no `<sup>` tag left to leave an orphan of).
- Per-figure titles never overriding per FRAME either (frames now carry only
  `annotations`) retires the one test that checked a clause survived a per-frame
  title override -- the panel's words are HTML above the plot, computed once from
  the base figure's `layout.meta`, never duplicated per animation frame.
- `webapp/app.py`'s A/B row markup moved from inline `style` dicts to CSS classes
  (`ab-row`/`ab-cell`/`ab-cell-single`, `webapp/assets/demo.css`) as part of the
  same redesign; `_AB_COLUMN_FLEX` is gone. `_panel_block` also grew an HTML header
  (`_panel_header`: title + caption) as a SIBLING of the `dcc.Graph`, where before
  the graph was the cell's only child -- tests that walk into a specific graph now
  need to skip past that new sibling.
- `webapp/assets/results_clock.js` no longer listens for Plotly's own per-panel
  `updatemenu-button`/`slider-container` DOM classes at all -- those elements
  don't exist any more (no per-figure transport, see above). It drives the ONE
  HTML transport (`webapp.app.TRANSPORT_TOGGLE_ID`/`TRANSPORT_SLIDER_ID`) instead;
  there is nothing native left to `stopPropagation()` against, so that call is
  gone too.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import plotly.graph_objects as go
import pytest

from webapp import pipeline_runner as pr

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_CLOCK_JS = _REPO_ROOT / "webapp" / "assets" / "results_clock.js"


# ------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------
def _all_text(component) -> str:
    """Every string anywhere in a Dash component tree, joined."""
    if isinstance(component, str):
        return component
    parts = []
    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        parts.extend(_all_text(c) for c in children if c is not None)
    elif children is not None:
        parts.append(_all_text(children))
    return " ".join(parts)


def _db_map(floor_db: float, *, peak_at=(10, 20), n=24, seed=0) -> np.ndarray:
    """A peak-normalized power-dB map whose median sits at `floor_db`: a flat floor
    plus one 0 dB cell, i.e. exactly the shape `_peak_minus_median_db` measures."""
    m = np.full((n, n), float(floor_db))
    m[peak_at] = 0.0
    return m


def _range_az_pair(floor_a: float, floor_b: float, n_frames: int = 3):
    """Two arms' `range_az` figure DICTS, built the way `figures_from_outputs` itself
    builds this product (`_heatmap` + `set_panel` + `_add_frame_animation` +
    `to_dict()`), with the given median floors. The panel's words go through
    `set_panel` now (no more `title` argument on `_heatmap`) -- same caption/details
    shape as the real `range_az` branch, so `_apply_shared_z`'s caption-replace and
    details-append logic is exercised the same way it is in production."""
    figs = []
    for floor in (floor_a, floor_b):
        frames = [_db_map(floor, seed=i) for i in range(n_frames)]
        fig = pr._heatmap(frames[-1], x=np.linspace(-1, 1, frames[-1].shape[1]),
                          y=np.arange(frames[-1].shape[0], dtype=float),
                          zmin=-40.0, z_share=pr.Z_SHARE_REACH_FLOOR)
        pr.set_panel(fig, title="Range-azimuth power",
                    caption=[pr._DB_COLORBAR_PREFIX, f"{pr.CLIP_CLAUSE_PREFIX}-40.0 dB"],
                    details=["Integration: (non-coherent over elevation).",
                             f"peak - median, dB: {-floor:.1f} (last frame)."],
                    row=pr.PANEL_ROW_MAP)
        figs.append(pr._add_frame_animation(fig, frames).to_dict())
    return figs


def _heatmap_trace(fig_dict):
    return next(t for t in fig_dict["data"] if t.get("type") == "heatmap")


# ------------------------------------------------------------------------------------
# 1. Shared colour limits that REACH the floor
# ------------------------------------------------------------------------------------
def test_both_arms_end_up_on_one_identical_colour_limit():
    a, b = _range_az_pair(-65.9, -54.3)
    assert _heatmap_trace(a)["zmin"] != _heatmap_trace(b)["zmin"] or True  # pre-state
    pr.share_heatmap_z_limits({"range_az": a}, {"range_az": b})
    ta, tb = _heatmap_trace(a), _heatmap_trace(b)
    assert ta["zmin"] == tb["zmin"]
    assert ta["zmax"] == tb["zmax"] == 0.0


def test_shared_limit_reaches_below_the_deeper_arms_own_median_floor():
    """The whole point of item 1: with `zmin` above either floor, that arm's whole
    background collapses onto one clip colour and the A/B difference is gone. The
    limit must sit BELOW the deeper floor, by the module's stated margin."""
    a, b = _range_az_pair(-65.9, -54.3)
    pr.share_heatmap_z_limits({"range_az": a}, {"range_az": b})
    zmin = _heatmap_trace(a)["zmin"]
    assert zmin == pytest.approx(-65.9 - pr._SHARED_FLOOR_MARGIN_DB, abs=1e-6)
    assert zmin < -65.9 and zmin < -54.3


def test_the_owners_two_floors_land_far_apart_on_the_shared_colour_ramp():
    """The measurable version of "visible to the human eye": at the shared limits,
    where does each arm's background sit on the 0-1 colour ramp? Both at 0 (the old
    fixed -40 dB clip, which both medians sat below) is the defect. The gap here is
    what makes arm B's background a different colour from arm A's.

    The 12%-of-the-ramp floor is not a perception model -- it is the separation the
    rendered Thrust 1 PNG was read against (wave 11); it exists so a future change
    that quietly re-clips these panels fails here instead of on stage."""
    a, b = _range_az_pair(-65.9, -54.3)
    pr.share_heatmap_z_limits({"range_az": a}, {"range_az": b})
    zmin, zmax = _heatmap_trace(a)["zmin"], _heatmap_trace(a)["zmax"]
    frac = lambda db: (db - zmin) / (zmax - zmin)  # noqa: E731
    assert frac(-65.9) == pytest.approx(3.0 / 68.9, abs=1e-3)
    assert frac(-54.3) - frac(-65.9) > 0.12


def test_old_fixed_clip_would_have_put_both_floors_at_the_same_colour():
    """The defect itself, stated as a test so the fix cannot be read as cosmetic:
    under the pre-wave-11 shared -40 dB clip both of the owner's floors clamp to the
    bottom of the ramp -- identical pixels for an 11.6 dB difference."""
    zmin, zmax = -40.0, 0.0
    clamp = lambda db: max(0.0, (db - zmin) / (zmax - zmin))  # noqa: E731
    assert clamp(-65.9) == clamp(-54.3) == 0.0


def test_caption_and_details_state_the_sharing_and_the_chosen_zmin_on_both_arms():
    a, b = _range_az_pair(-65.9, -54.3)
    pr.share_heatmap_z_limits({"range_az": a}, {"range_az": b})
    zmin = _heatmap_trace(a)["zmin"]
    # "colour limits shared with arm X: zmin ..." is a DETAILS line
    # (`_SHARED_LIMITS_MARKER`), not a caption clause -- `panel_text` reaches both.
    assert "colour limits shared with arm B" in pr.panel_text(a)
    assert "colour limits shared with arm A" in pr.panel_text(b)
    for fig in (a, b):
        text = pr.panel_text(fig)
        assert f"zmin {zmin:.1f} dB" in text
        assert "zmax 0 dB" in text
        # The per-arm clip clause built before sharing ("clipped at -40.0 dB", the
        # CAPTION clause) is REPLACED (see
        # test_shared_caption_replaces_the_per_arm_clip_clause_not_beside_it below):
        # the value it was superseded BY has to be named, or the panel prints two
        # different clips and reads as a bug -- named here in the shared Details
        # line's own "(was X)".
        assert "(was -40.0)" in text


def test_shared_caption_replaces_the_per_arm_clip_clause_not_beside_it():
    """The real CAPTION (`figures_from_outputs`) carries one `CLIP_CLAUSE_PREFIX`
    clause before sharing exists ("clipped at -40.0 dB"). Leaving it standing next
    to the new "same colour scale on both arms" clause would print two different
    clips on one line and read as a bug (wave 11's own original finding, now pinned
    at the caption-list level rather than a string-surgery level). The superseded
    value is not lost -- it is named once, in the shared Details line's own
    "(was X)" -- so the caption's clip clause is REPLACED rather than left
    standing."""
    def _fig(floor, peak_median):
        frames = [_db_map(floor, seed=i) for i in range(3)]
        heat = pr._heatmap(frames[-1], zmin=-40.0, z_share=pr.Z_SHARE_REACH_FLOOR)
        pr.set_panel(heat, title="Range-azimuth power",
                    caption=[pr._DB_COLORBAR_PREFIX, f"{pr.CLIP_CLAUSE_PREFIX}-40.0 dB"],
                    details=[f"peak - median, dB: {peak_median:.1f} (last frame).",
                             "display 0-125 m of 250 m unambig (neg.-delay half "
                             "cropped).",
                             "0 dB cell at range 0 is one 1 m gate."],
                    row=pr.PANEL_ROW_MAP)
        return pr._add_frame_animation(heat, frames).to_dict()

    a, b = _fig(-65.9, 66.0), _fig(-54.3, 54.3)
    pr.share_heatmap_z_limits({"range_az": a}, {"range_az": b})
    caption = pr.panel_caption(a)
    assert caption.count(pr.CLIP_CLAUSE_PREFIX) == 1
    assert pr.SHARED_SCALE_CLAUSE in caption
    text = pr.panel_text(a)
    assert "colour limits shared with arm B" in text
    assert "(was -40.0)" in text
    # The rest of Details (unrelated to the sharing pass) survives untouched.
    assert "display 0-125 m of 250 m unambig" in text
    assert "0 dB cell at range 0 is one 1 m gate" in text


# `test_shared_clause_replaces_the_clip_clause_even_when_word_wrapped` is RETIRED
# (2026-09-24): it probed `_apply_shared_z`'s removal surviving a "<br>" inserted
# mid-clause by word-wrap. The caption is no longer a wrapped string at all -- it
# is a LIST of clauses, filtered by `startswith`/`==` (`_apply_shared_z`, see
# `webapp/pipeline_runner.py`), so there is no wrapping step left to insert a
# "<br>" in the middle of a clause to survive. Captions are also asserted single-
# line, always, by
# test_webapp_layout_acceptance.py::test_every_caption_is_one_line_and_at_most_110_characters.

# `test_colorbar_label_follows_the_shared_limit` is RETIRED: colour bars carry no
# title at all now (layout spec section 4) -- there is no colorbar label left for
# the shared limit to "follow". Covered generically by
# test_webapp_layout_acceptance.py::test_no_colorbar_carries_a_title; the shared
# limit itself reaching the caption/Details is
# test_caption_and_details_state_the_sharing_and_the_chosen_zmin_on_both_arms above.

# `test_every_animation_frame_title_carries_the_shared_limits_clause` is RETIRED:
# frames carry only `annotations` now, never a `title` override (see this file's
# own module docstring) -- the panel's words are HTML above the plot, rendered ONCE
# from the base figure's `layout.meta.panel`, so there is no per-frame copy for a
# clause to "vanish" from when the clock steps. Nothing replaces this test because
# the risk it guarded no longer exists.

# `test_the_top_margin_grows_with_the_added_subtitle_line` and
# `test_the_clause_is_one_line_so_it_costs_one_margin_step` are RETIRED: the top
# margin is a plain constant now (`_heatmap_margin_t` takes and ignores its
# argument), so nothing the sharing pass appends can move it any more --
# test_webapp_layout_acceptance.py::test_heatmap_margin_t_ignores_its_argument and
# ::test_every_figure_height_is_its_rows_fixed_height cover the "geometry does not
# depend on appended text" invariant generically, for every product including this
# one's own sharing pass (see that file's
# test_the_same_product_has_the_same_geometry_on_two_different_runs, which already
# varies the DATA, and therefore every derived caption/details string, between two
# calls and still requires byte-identical geometry).


def test_sharing_twice_does_not_append_the_clause_twice():
    a, b = _range_az_pair(-65.9, -54.3)
    pr.share_heatmap_z_limits({"range_az": a}, {"range_az": b})
    pr.share_heatmap_z_limits({"range_az": a}, {"range_az": b})
    assert pr.panel_text(a).count("colour limits shared with arm") == 1
    assert pr.panel_caption(a).count(pr.SHARED_SCALE_CLAUSE) == 1


def test_an_untagged_heatmap_keeps_its_own_limits():
    """The detector objectness panels are heat maps too, but their z is a 0-1 score,
    not dB -- pushing a dB floor onto them would blank them. They carry no
    `layout.meta.z_share`, so the sharing pass must skip them."""
    obj = lambda: go.Figure(data=go.Heatmap(z=np.zeros((4, 4)), zmin=0.0,  # noqa: E731
                                            zmax=1.0)).to_dict()
    a, b = obj(), obj()
    pr.share_heatmap_z_limits({"cfar_detection": a}, {"cfar_detection": b})
    assert _heatmap_trace(a)["zmin"] == 0.0 and _heatmap_trace(a)["zmax"] == 1.0
    assert "colour limits shared" not in pr.panel_text(a)


def test_a_keep_clip_pair_shares_the_tighter_clip_not_the_floor():
    """The range-Doppler panel's adaptive clip deliberately HIDES its own floor
    (`_radar_cube_clip_db`: the corpus's ambient floor lights up whole Doppler rows
    when it crosses the clip). Sharing must unify the two arms without undoing that
    -- the higher (tighter) clip keeps each arm's floor >= 3 dB below it."""
    def cube(clip):
        fig = pr._heatmap(_db_map(-50.0), zmin=clip, z_share=pr.Z_SHARE_KEEP_CLIP)
        pr.set_panel(fig, title="Range-Doppler power",
                    caption=[pr._DB_COLORBAR_PREFIX, f"{pr.CLIP_CLAUSE_PREFIX}{clip:.1f} dB"],
                    details=[], row=pr.PANEL_ROW_MAP)
        return fig.to_dict()
    a, b = cube(-40.0), cube(-36.2)
    pr.share_heatmap_z_limits({"radar_cube": a}, {"radar_cube": b})
    assert _heatmap_trace(a)["zmin"] == _heatmap_trace(b)["zmin"] == -36.2
    assert _heatmap_trace(a)["zmin"] > min(-40.0, -36.2)


def test_radar_cube_shared_caption_replaces_its_own_clip_clause_not_beside_it():
    """radar_cube uses `Z_SHARE_KEEP_CLIP` (the tighter of the two clips, not the
    floor) -- a different SHARING POLICY from range_az/range_el's, but the SAME
    caption shape now (`figures_from_outputs`' `radar_cube` branch puts its clip in
    a `CLIP_CLAUSE_PREFIX` caption clause too -- the "different shape, orphan
    <sup></sup>" premise this test protected described the old single-subline
    title and is retired with it). Checks the caption is replaced, not duplicated,
    and this panel's own clip-provenance Details line (built differently from
    range_az's, and untouched by the sharing pass) survives."""
    def cube(clip, label):
        fig = pr._heatmap(_db_map(-50.0), zmin=clip, z_share=pr.Z_SHARE_KEEP_CLIP)
        pr.set_panel(fig, title="Range-Doppler power",
                    caption=[pr._DB_COLORBAR_PREFIX, f"{pr.CLIP_CLAUSE_PREFIX}{clip:.1f} dB"],
                    details=[f"clip {clip:.1f} dB ({label}): this panel's display "
                             "clip is a deliberate decision about what to hide."],
                    row=pr.PANEL_ROW_MAP)
        return fig.to_dict()
    a = cube(-40.0, "shared floor")
    b = cube(-36.2, "median floor + 3 dB")
    pr.share_heatmap_z_limits({"radar_cube": a}, {"radar_cube": b})
    caption = pr.panel_caption(a)
    assert caption.count(pr.CLIP_CLAUSE_PREFIX) == 1
    assert pr.SHARED_SCALE_CLAUSE in caption
    text = pr.panel_text(a)
    assert "colour limits shared with arm B" in text
    assert "(was -40.0)" in text
    assert ("clip -40.0 dB (shared floor): this panel's display clip is a "
            "deliberate decision") in text


def test_mismatched_policies_are_left_alone():
    """Two panels that disagree about what their colour limits MEAN must not be
    forced onto one scale by key name alone."""
    a = pr._heatmap(_db_map(-50.0), zmin=-40.0, z_share=pr.Z_SHARE_REACH_FLOOR).to_dict()
    b = pr._heatmap(_db_map(-50.0), zmin=-36.0, z_share=pr.Z_SHARE_KEEP_CLIP).to_dict()
    pr.share_heatmap_z_limits({"k": a}, {"k": b})
    assert _heatmap_trace(a)["zmin"] == -40.0 and _heatmap_trace(b)["zmin"] == -36.0


def test_shared_limits_survive_plotlys_compact_typed_array_encoding():
    """Same wire-format trap that broke `_share_y_ranges` in the 2026-09-22
    rehearsal: a large numpy z is stored as `{"dtype", "bdata"}`, not a list."""
    a, b = _range_az_pair(-65.9, -54.3)
    assert isinstance(_heatmap_trace(a)["z"], dict) and "bdata" in _heatmap_trace(a)["z"], \
        "test assumption: plotly used the compact encoding here"
    pr.share_heatmap_z_limits({"range_az": a}, {"range_az": b})  # must not raise
    assert _heatmap_trace(a)["zmin"] == pytest.approx(-68.9, abs=1e-6)


def test_decode_plotly_array_has_one_authority():
    """`webapp.app._decode_plotly_array` delegates to the runner's copy -- two
    decoders for plotly's own wire format is exactly the duplicate that drifts."""
    import webapp.app as appmod

    encoded = go.Figure(data=go.Heatmap(z=np.arange(400.0).reshape(20, 20))).to_dict()
    z = encoded["data"][0]["z"]
    assert appmod._decode_plotly_array(z) == pr.decode_plotly_array(z)


# ------------------------------------------------------------------------------------
# 2. Side-by-side A/B layout
# ------------------------------------------------------------------------------------
def _ab_results_data():
    return {
        "range_az": go.Figure().to_dict(),
        "fft": go.Figure().to_dict(),
        "_banner": "A (as loaded): LNA bias current (mA) 8 mA -- before",
        "_ab": True,
        "_notes": ["arm A note"],
        "_previous": {"range_az": go.Figure().to_dict(),
                      "fft": go.Figure().to_dict(),
                      "_banner": "B: LNA bias current (mA) 0.5 mA -- after",
                      "_ab": True, "_notes": ["arm B note"]},
    }


def _rows(tree):
    """The PRODUCT-ROW Divs of the side-by-side block. The `results-grid` Div is
    no longer reliably `tree.children[-1]` -- a `_screen_note` appends a trailing
    `page-foot-note` Div after it now (layout spec section 2.3) -- so it is found
    by its own `className` instead. `_ab_columns` (webapp/app.py) also interleaves
    a bare `className="section-rule"` divider Div right after the header row,
    which carries no row content of its own -- filtered out here by its `ab-row`
    className marker, the same one `_row()` gives every real row."""
    grid = next(c for c in tree.children if getattr(c, "className", None) == "results-grid")
    return [r for r in grid.children
           if getattr(r, "className", None) == "ab-row"]


def test_ab_results_render_one_row_per_product_two_columns_wide():
    """`_row`'s cells are CSS classes now (`webapp/assets/demo.css`'s `.ab-row`/
    `.ab-cell`), not inline `style` dicts -- `_AB_COLUMN_FLEX` is gone with them.
    The actual `display: flex`/`flex` VALUES are pure CSS now, unreachable from a
    figure-dict-level test (see this module's own docstring); what stays checkable
    here is the STRUCTURE the CSS selectors depend on: one row per product, two
    cells per row, both classed `ab-cell`."""
    import webapp.app as appmod

    tree = appmod._render_results(_ab_results_data(), "tab-results")
    rows = _rows(tree)
    # 1 header row + 1 row per product (range_az, fft).
    assert len(rows) == 3
    for row in rows:
        assert row.className == "ab-row"
        assert len(row.children) == 2
        for col in row.children:
            assert col.className == "ab-cell"


def test_each_arms_banner_sits_above_its_own_column():
    import webapp.app as appmod

    tree = appmod._render_results(_ab_results_data(), "tab-results")
    header = _rows(tree)[0]
    left, right = (_all_text(c) for c in header.children)
    assert "A (as loaded):" in left and "arm A note" in left
    assert "B:" in right and "arm B note" in right
    assert "B:" not in left and "A (as loaded):" not in right


def test_the_same_product_lands_in_the_same_row_on_both_arms():
    """The defect: A's whole grid then B's whole grid put the two copies of one
    product a screen height apart, so the comparison had to be remembered.

    `_panel_block` (webapp/app.py) now wraps the plot in an HTML header (title +
    caption) as a SIBLING of the `dcc.Graph`, where before the graph was the
    cell's only child -- the path into the actual figure grew one more `[1]` (the
    Graph is the second of the two children, after `_panel_header`)."""
    import webapp.app as appmod

    data = _ab_results_data()
    data["range_az"]["layout"]["title"] = {"text": "ARM-A-RANGE-AZ"}
    data["_previous"]["range_az"]["layout"]["title"] = {"text": "ARM-B-RANGE-AZ"}
    tree = appmod._render_results(data, "tab-results")
    row = _rows(tree)[1]
    left_graph = row.children[0].children.children[1]
    right_graph = row.children[1].children.children[1]
    assert left_graph.figure["layout"]["title"]["text"] == "ARM-A-RANGE-AZ"
    assert right_graph.figure["layout"]["title"]["text"] == "ARM-B-RANGE-AZ"


def test_a_product_only_one_arm_produced_keeps_an_empty_cell_opposite_it():
    """The opposite cell is a real `_empty_panel` placeholder Div now (className
    `result-panel-empty`, height matched to the row kind -- `_row`'s own docstring:
    "the same height whether or not either is empty", acceptance check 4) -- not a
    bare `None`, which is what the old assertion checked FOR (repoints onto the
    opposite of the old expectation, on purpose: a `None` cell there today would be
    the regression, not the invariant)."""
    import webapp.app as appmod

    data = _ab_results_data()
    del data["_previous"]["fft"]
    tree = appmod._render_results(data, "tab-results")
    rows = _rows(tree)
    assert len(rows) == 3          # header + range_az + fft
    empty_cell = rows[2].children[1].children
    assert empty_cell is not None
    assert "result-panel-empty" in empty_cell.className
    assert empty_cell.style["height"] == f"{appmod.PANEL_HEIGHT[appmod.PANEL_ROW_MAP]}px"


def test_single_arm_results_keep_the_wrapping_grid():
    """Only the A/B path changed row markup; a hand-edited single run still renders
    through `_grid`. `grid`/`_row`'s inline `style` dicts are gone (CSS classes
    now, see the module docstring) -- and a LONE product no longer stretches to
    "1 1 100%" of the row at all: that WAS hostile round 10's own defect 10 (a
    1080x230, 4.7:1 strip) that `SINGLE_PANEL_WIDTH` exists to fix, so a lone
    product gets one fixed-width panel instead."""
    import webapp.app as appmod

    data = {"range_az": go.Figure().to_dict(), "_banner": "run #1"}
    tree = appmod._render_results(data, "tab-results")
    grid = tree.children[-1]
    row = grid.children[0]
    assert row.className == "ab-row"
    cell = row.children[0]
    assert cell.className == "ab-cell ab-cell-single"
    panel = cell.children
    assert panel.style["width"] == f"{appmod.SINGLE_PANEL_WIDTH}px"
    assert "This run" in _all_text(tree)


def test_the_screen_note_is_reachable_in_each_arms_details_and_printed_once_at_the_page_foot():
    """RETIRED premise: "printed once above both columns" -- the screen note is now
    ALSO inside each arm's Details disclosure (`_details_lines`; acceptance check
    15: every honesty clause reachable without leaving the panel/screen it belongs
    to), on top of the standing page-foot copy, so the total on-page count is no
    longer 1. What survives unchanged: exactly one page-foot copy, and it is still
    reachable from each arm's own header block."""
    import webapp.app as appmod

    data = _ab_results_data()
    data["_screen_note"] = "a caveat the audience must see"
    tree = appmod._render_results(data, "tab-results")
    foot = tree.children[-1]
    assert foot.className == "page-foot-note"
    assert _all_text(foot) == "a caveat the audience must see"
    header_row = _rows(tree)[0]
    for cell in header_row.children:
        assert "a caveat the audience must see" in _all_text(cell)


def test_render_results_shares_colour_limits_between_the_two_arms():
    """Integration: the sharing pass is actually wired into the render, not just
    importable."""
    import webapp.app as appmod

    a, b = _range_az_pair(-65.9, -54.3)
    data = {"range_az": a, "_banner": "A", "_ab": True,
            "_previous": {"range_az": b, "_banner": "B", "_ab": True}}
    appmod._render_results(data, "tab-results")
    assert _heatmap_trace(a)["zmin"] == _heatmap_trace(b)["zmin"]
    assert _heatmap_trace(a)["zmin"] == pytest.approx(-68.9, abs=1e-6)
    assert "colour limits shared with arm B" in pr.panel_text(a)


# ------------------------------------------------------------------------------------
# 3. Autoplay + loop, one clock
# ------------------------------------------------------------------------------------
def test_the_results_clock_interval_is_mounted_outside_the_tab_content():
    """Outside the tabs, so switching away from Results (which unmounts
    `results-tab-content`) does not stop the clock."""
    import webapp.app as appmod
    from dash import dcc

    found = []

    def walk(c, inside_tabs=False):
        if isinstance(c, dcc.Interval):
            found.append((c, inside_tabs))
        inside_tabs = inside_tabs or isinstance(c, dcc.Tabs)
        ch = getattr(c, "children", None)
        if isinstance(ch, (list, tuple)):
            for x in ch:
                if x is not None:
                    walk(x, inside_tabs)
        elif ch is not None:
            walk(ch, inside_tabs)

    walk(appmod._app_layout())
    clocks = [(c, t) for c, t in found if c.id == "results-clock"]
    assert len(clocks) == 1, "exactly one Results clock"
    clock, inside_tabs = clocks[0]
    assert inside_tabs is False
    assert clock.interval == appmod.RESULTS_CLOCK_MS == 700
    # dcc.Interval omits `disabled` entirely when it was never passed.
    assert getattr(clock, "disabled", None) in (None, False)


def test_the_clock_drives_a_registered_clientside_callback():
    import webapp.app as appmod

    specs = [c for c in appmod.app._callback_list
             if c.get("output") == "results-clock-tick.data"]
    assert len(specs) == 1
    assert specs[0]["clientside_function"] == {
        "namespace": appmod.RESULTS_CLOCK_NAMESPACE,
        "function_name": appmod.RESULTS_CLOCK_FUNCTION,
    }
    assert [i["id"] for i in specs[0]["inputs"]] == ["results-clock"]


def test_the_asset_registers_exactly_the_namespace_the_app_calls():
    import webapp.app as appmod

    js = _CLOCK_JS.read_text(encoding="utf-8")
    assert f"window.dash_clientside.{appmod.RESULTS_CLOCK_NAMESPACE}" in js
    assert re.search(rf"\b{appmod.RESULTS_CLOCK_FUNCTION}\s*:\s*function", js)


def test_the_asset_steps_frames_immediately_with_no_transition():
    """A real Plotly animation would queue and the two arms would drift apart; each
    step is an immediate, zero-duration redraw to a NAMED frame instead."""
    js = _CLOCK_JS.read_text(encoding="utf-8")
    assert "Plotly.animate(" in js
    assert 'mode: "immediate"' in js
    assert "transition: {duration: 0}" in js
    assert "frame: {duration: 0, redraw: true}" in js


def test_the_asset_loops_and_scopes_itself_to_the_results_tab():
    js = _CLOCK_JS.read_text(encoding="utf-8")
    assert "% frames.length" in js, "wraps forever rather than stopping on the last frame"
    assert '"results-tab-content"' in js
    assert ".js-plotly-plot" in js


def test_the_asset_leaves_a_single_frame_panel_alone():
    """The detector objectness panels carry no frames at all (pinned to the last
    frame by design -- `figures_from_outputs`). The clock must not touch them."""
    js = _CLOCK_JS.read_text(encoding="utf-8")
    assert "frames.length > 1" in js


def test_the_asset_drives_pause_and_scrub_from_the_one_transports_own_ids():
    """RETIRED premise: "pressing pause on any PANEL... Plotly's own
    updatemenu-button... slider-container" described the (already-superseded)
    world where per-figure play/pause buttons and sliders still existed alongside
    the one clock, and this listener had to intercept THEIR native DOM classes and
    `stopPropagation()` against their own one-shot animation. Per-figure
    updatemenus/sliders are fully retired now (layout spec section 4) -- there is
    nothing native left to listen to or suppress, so `updatemenu-button`/
    `slider-container`/`stopPropagation` are all gone from this file. The one
    transport this asset drives instead has its own fixed HTML ids
    (`webapp.app.TRANSPORT_TOGGLE_ID`/`TRANSPORT_SLIDER_ID`); repoints onto those,
    keeping the still-real invariants (pausing on scrub parks every panel at once;
    every listener is capture-phase, delegated from `document` since the transport
    is rebuilt by Dash on every render)."""
    import webapp.app as appmod

    js = _CLOCK_JS.read_text(encoding="utf-8")
    assert appmod.TRANSPORT_TOGGLE_ID in js
    assert appmod.TRANSPORT_SLIDER_ID in js
    assert "S.paused" in js
    assert "S.paused = true" in js, "dragging the slider parks the panel"
    # capture-phase registration (the third argument of addEventListener) on all
    # three listeners (toggle click, slider input, slider change).
    assert js.count(", true);") == 3


def test_the_clock_asset_is_served_from_the_dash_assets_folder():
    import webapp.app as appmod

    assert _CLOCK_JS.exists()
    assert _CLOCK_JS.parent == pathlib.Path(appmod.app.config.assets_folder)


# ------------------------------------------------------------------------------------
# 4. The retired runbook rule
# ------------------------------------------------------------------------------------
def _preflight_runbook_text() -> str:
    from webapp.demo_presets import PRESETS
    from webapp.preflight import _runbook

    return _runbook(PRESETS)


def test_preflight_runbook_no_longer_tells_the_presenter_to_avoid_play():
    text = _preflight_runbook_text()
    assert "not the\nPlay button" not in text
    assert "step frames with the Results-tab frame SLIDER" not in text
    assert "plays itself" in text.lower()
    assert "loops" in text.lower()


def test_preflight_runbook_records_why_the_rule_was_retired():
    """A retracted claim that is merely deleted gets re-adopted by the next reader
    (CLAUDE.md, "Claims in this repo carry their provenance"): the runbook says the
    rule existed, and that the owner measured the link."""
    text = _preflight_runbook_text().lower()
    assert "retired" in text
    assert "2026-09-24" in text
    assert "rdp" in text


def _rendered_runbook() -> str:
    from webapp.demo_presets import PRESETS
    from webapp.runbook import render

    return render(PRESETS)


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_runbook_retires_the_slider_rule_and_states_the_autoplay_behaviour():
    flat = _flat(_rendered_runbook())
    # The old rule may only appear INSIDE the sentence that retires it -- never as
    # an instruction of its own (CLAUDE.md: record the retraction rather than
    # deleting the claim, or the next reader re-adopts it).
    assert flat.count("never the ▶ (Play) control") == 1
    head = flat[:flat.index("never the ▶ (Play) control")]
    assert head.rstrip().endswith(
        "RETIRED (owner, live test 2026-09-24): the old rule to advance frames "
        "with a figure's own frame slider and")
    assert "Results tab PLAYS ITSELF" in flat
    assert "loops forever" in flat
    assert "RETIRED" in flat


def test_runbook_still_tells_the_presenter_how_to_hold_a_frame():
    """Retiring the rule must not lose the capability it protected -- a presenter
    still needs to park a panel on one frame to talk about it."""
    flat = _flat(_rendered_runbook())
    assert "pause" in flat.lower()
    assert "every panel stops together" in flat


def test_runbook_keeps_the_thrust5_frame_pinned_detector_caveat():
    """The detector/objectness panels still hold the LAST frame while the
    Range-Doppler cube beside them now loops on its own -- which makes the desync
    the old "leave the slider alone" rule warned about happen WITHOUT a click. The
    runbook has to say so."""
    flat = _flat(_rendered_runbook())
    assert "pinned to the LAST frame" in flat
    assert "loops by itself on the Results-tab clock" in flat
    assert "Press pause on the cube" in flat


def test_runbook_describes_the_arms_as_side_by_side_columns():
    flat = _flat(_rendered_runbook())
    assert "SIDE BY SIDE" in flat
    assert "LEFT column" in flat and "RIGHT column" in flat
    assert "A on top, B below" not in flat


def test_the_committed_runbook_matches_the_generator():
    """`docs/DEMO_RUNBOOK.md` is generated, never hand-edited (webapp/runbook.py
    --check); this is that check, run by the suite."""
    committed = (_REPO_ROOT / "docs" / "DEMO_RUNBOOK.md").read_text(encoding="utf-8")
    assert _flat(committed) == _flat(_rendered_runbook())
