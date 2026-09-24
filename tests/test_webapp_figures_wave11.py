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
    """Two arms' `range_az` figure DICTS, built the way the app stores them (through
    `_heatmap`/`_add_frame_animation`/`to_dict`), with the given median floors."""
    figs = []
    for floor in (floor_a, floor_b):
        frames = [_db_map(floor, seed=i) for i in range(n_frames)]
        title = ("Range-azimuth power<br><sup>(non-coherent over elevation); "
                 f"peak - median, dB: {-floor:.1f}</sup>")
        fig = pr._heatmap(frames[-1], title, x=np.linspace(-1, 1, frames[-1].shape[1]),
                          y=np.arange(frames[-1].shape[0], dtype=float),
                          zmin=-40.0, z_share=pr.Z_SHARE_REACH_FLOOR)
        frame_layouts = [dict(title=dict(text=title)) for _ in frames]
        figs.append(pr._add_frame_animation(fig, frames,
                                            frame_layouts=frame_layouts).to_dict())
    return figs


def _heatmap_trace(fig_dict):
    return next(t for t in fig_dict["data"] if t.get("type") == "heatmap")


def _title_text(fig_dict) -> str:
    return fig_dict["layout"]["title"]["text"]


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


def test_subtitle_states_the_sharing_and_the_chosen_zmin_on_both_arms():
    a, b = _range_az_pair(-65.9, -54.3)
    pr.share_heatmap_z_limits({"range_az": a}, {"range_az": b})
    zmin = _heatmap_trace(a)["zmin"]
    assert "colour limits shared with arm B" in _title_text(a)
    assert "colour limits shared with arm A" in _title_text(b)
    for t in (_title_text(a), _title_text(b)):
        assert f"zmin {zmin:.1f} dB" in t
        assert "zmax 0 dB" in t
        # The per-arm clip clause built before sharing ("clip -40.0 dB (shared
        # floor)") is REPLACED (see `test_shared_clause_replaces_the_per_arm_clip_
        # clause_not_beside_it` below): the value it was superseded BY has to be
        # named, or the panel prints two different clips and reads as a bug -- named
        # here in the shared clause's own "(was X)".
        assert "(was -40.0)" in t


def test_shared_clause_replaces_the_per_arm_clip_clause_not_beside_it():
    """The real subtitle (`figures_from_outputs`) embeds a "; clip -40.0 dB (shared
    floor)" clause mid-line before sharing exists. Printing that clause AND the
    "colour limits shared ..." clause side by side reads as two different clips on
    one panel (rendered check, thrust1_circuit_knobs wave 11 PNG, 2026-09-24). The
    superseded value is not lost -- it is named once, in the shared clause's own
    "(was X)" -- so the clip clause is removed rather than left standing."""
    def _fig(floor, peak_median):
        title = ("Range-azimuth power<br><sup>(non-coherent over elevation); "
                 f"peak - median, dB: {peak_median:.1f}; unambig 125 m; "
                 "clip -40.0 dB (shared floor); 0 dB cell at range 0</sup>")
        frames = [_db_map(floor, seed=i) for i in range(3)]
        heat = pr._heatmap(frames[-1], title, zmin=-40.0, z_share=pr.Z_SHARE_REACH_FLOOR)
        return pr._add_frame_animation(
            heat, frames, frame_layouts=[dict(title=dict(text=title)) for _ in frames]).to_dict()

    a, b = _fig(-65.9, 66.0), _fig(-54.3, 54.3)
    pr.share_heatmap_z_limits({"range_az": a}, {"range_az": b})
    text = _title_text(a)
    assert "clip -40.0 dB (shared floor)" not in text
    assert "colour limits shared with arm B" in text
    assert "(was -40.0)" in text
    # The rest of the subline (unrelated content either side of the clip clause)
    # survives untouched.
    assert "unambig 125 m" in text
    assert "0 dB cell at range 0" in text
    for frame in a["frames"]:
        assert "clip -40.0 dB (shared floor)" not in frame["layout"]["title"]["text"]


def test_shared_clause_replaces_the_clip_clause_even_when_word_wrapped():
    """`_wrap_text` may have inserted a "<br>" in place of one of the clip clause's
    own spaces before sharing ever runs; the removal must not depend on the clause
    surviving as one unbroken run of literal spaces."""
    title = ("Range-azimuth power<br><sup>peak - median, dB: 66.0; clip -40.0<br>dB "
             "(shared floor); 0 dB cell at range 0</sup>")
    frames_a = [_db_map(-65.9, seed=i) for i in range(3)]
    heat = pr._heatmap(frames_a[-1], title, zmin=-40.0, z_share=pr.Z_SHARE_REACH_FLOOR)
    a = pr._add_frame_animation(
        heat, frames_a, frame_layouts=[dict(title=dict(text=title)) for _ in frames_a]).to_dict()
    b_title = title.replace("66.0", "54.3")
    frames_b = [_db_map(-54.3, seed=i) for i in range(3)]
    heat_b = pr._heatmap(frames_b[-1], b_title, zmin=-40.0, z_share=pr.Z_SHARE_REACH_FLOOR)
    b = pr._add_frame_animation(
        heat_b, frames_b, frame_layouts=[dict(title=dict(text=b_title)) for _ in frames_b]).to_dict()
    pr.share_heatmap_z_limits({"range_az": a}, {"range_az": b})
    text = _title_text(a)
    assert "clip -40.0" not in text
    assert "shared floor" not in text
    assert "colour limits shared with arm B" in text
    assert "0 dB cell at range 0" in text


def test_colorbar_label_follows_the_shared_limit():
    """The colorbar says what the map is clipped at; leaving the per-arm value there
    after re-clipping would make the panel contradict itself."""
    a, b = _range_az_pair(-65.9, -54.3)
    pr.share_heatmap_z_limits({"range_az": a}, {"range_az": b})
    zmin = _heatmap_trace(a)["zmin"]
    for fig in (a, b):
        text = _heatmap_trace(fig)["colorbar"]["title"]["text"]
        assert text == f"dB rel. peak (clipped at {zmin:.1f})"


def test_every_animation_frame_title_carries_the_shared_limits_clause():
    """A per-frame title override REPLACES the whole title object when the clock
    steps (see `figures_from_outputs`' `frame_layouts`), so a clause only on the base
    title vanishes the moment the panel animates -- which, since wave 11, it does by
    itself."""
    a, b = _range_az_pair(-65.9, -54.3)
    pr.share_heatmap_z_limits({"range_az": a}, {"range_az": b})
    assert a["frames"], "test assumption: the pair is animated"
    for frame in a["frames"]:
        assert "colour limits shared with arm B" in frame["layout"]["title"]["text"]


def test_the_top_margin_grows_with_the_added_subtitle_line():
    """`_heatmap_margin_t` sizes the top margin from the title's own line count; a
    clause appended without re-sizing it overflows DOWN into the plot (the exact
    failure that constant exists for). The figure height moves by the same delta so
    the plot domain is not squeezed instead."""
    a, b = _range_az_pair(-65.9, -54.3)
    t_before, h_before = a["layout"]["margin"]["t"], a["layout"]["height"]
    pr.share_heatmap_z_limits({"range_az": a}, {"range_az": b})
    t_after, h_after = a["layout"]["margin"]["t"], a["layout"]["height"]
    assert t_after == pr._heatmap_margin_t(_title_text(a))
    assert t_after > t_before
    assert h_after - h_before == t_after - t_before


def test_the_clause_is_one_line_so_it_costs_one_margin_step():
    """It is appended to titles that already wrap to six lines on the Thrust 1
    screen, and every line is 45 px of plot height."""
    a, b = _range_az_pair(-65.9, -54.3)
    before = _title_text(a).count("<br>")
    pr.share_heatmap_z_limits({"range_az": a}, {"range_az": b})
    assert _title_text(a).count("<br>") == before + 1
    # ...including the "(was ...)" form, which is the longest one this can produce.
    assert "(was -40.0)" in _title_text(a)


def test_sharing_twice_does_not_append_the_clause_twice():
    a, b = _range_az_pair(-65.9, -54.3)
    pr.share_heatmap_z_limits({"range_az": a}, {"range_az": b})
    pr.share_heatmap_z_limits({"range_az": a}, {"range_az": b})
    assert _title_text(a).count("colour limits shared with arm") == 1
    for frame in a["frames"]:
        assert frame["layout"]["title"]["text"].count("colour limits shared") == 1


def test_an_untagged_heatmap_keeps_its_own_limits():
    """The detector objectness panels are heat maps too, but their z is a 0-1 score,
    not dB -- pushing a dB floor onto them would blank them. They carry no
    `layout.meta.z_share`, so the sharing pass must skip them."""
    obj = lambda: go.Figure(data=go.Heatmap(z=np.zeros((4, 4)), zmin=0.0,  # noqa: E731
                                            zmax=1.0)).to_dict()
    a, b = obj(), obj()
    pr.share_heatmap_z_limits({"cfar_detection": a}, {"cfar_detection": b})
    assert _heatmap_trace(a)["zmin"] == 0.0 and _heatmap_trace(a)["zmax"] == 1.0
    assert "colour limits shared" not in str(a["layout"].get("title") or "")


def test_a_keep_clip_pair_shares_the_tighter_clip_not_the_floor():
    """The range-Doppler panel's adaptive clip deliberately HIDES its own floor
    (`_radar_cube_clip_db`: the corpus's ambient floor lights up whole Doppler rows
    when it crosses the clip). Sharing must unify the two arms without undoing that
    -- the higher (tighter) clip keeps each arm's floor >= 3 dB below it."""
    def cube(clip):
        return pr._heatmap(_db_map(-50.0), "Range-Doppler power<br><sup>x</sup>",
                           zmin=clip, colorbar_title=f"dB rel. peak (clipped at {clip:.1f})",
                           z_share=pr.Z_SHARE_KEEP_CLIP).to_dict()
    a, b = cube(-40.0), cube(-36.2)
    pr.share_heatmap_z_limits({"radar_cube": a}, {"radar_cube": b})
    assert _heatmap_trace(a)["zmin"] == _heatmap_trace(b)["zmin"] == -36.2
    assert _heatmap_trace(a)["zmin"] > min(-40.0, -36.2)


def test_radar_cube_shared_clause_replaces_its_own_clip_clause_not_beside_it():
    """radar_cube's clip clause has a DIFFERENT shape from range_az/range_el's --
    it is the WHOLE subline (`figures_from_outputs`' `radar_cube` branch drops the
    qualifier), not one clause among several. Sharing must still print the shared
    clause ONCE, and must not leave an orphan empty "<sup></sup>" behind where the
    old clause used to be (handoff item, coordinator, 2026-09-24)."""
    def cube(clip, label):
        title = (f"Range-Doppler power<br><sup>clip {clip:.1f} dB "
                 f"({label})</sup>")
        return pr._heatmap(_db_map(-50.0), title, zmin=clip,
                           colorbar_title=f"dB rel. peak (clipped at {clip:.1f})",
                           z_share=pr.Z_SHARE_KEEP_CLIP).to_dict()
    a = cube(-40.0, "shared floor")
    b = cube(-36.2, "median floor + 3 dB")
    pr.share_heatmap_z_limits({"radar_cube": a}, {"radar_cube": b})
    text = _title_text(a)
    assert "clip -40.0 dB (shared floor)" not in text
    assert "<sup></sup>" not in text
    assert text.count("<sup>") == 1
    assert "colour limits shared with arm B" in text
    assert "(was -40.0)" in text


def test_mismatched_policies_are_left_alone():
    """Two panels that disagree about what their colour limits MEAN must not be
    forced onto one scale by key name alone."""
    a = pr._heatmap(_db_map(-50.0), "t<br><sup>s</sup>", zmin=-40.0,
                    z_share=pr.Z_SHARE_REACH_FLOOR).to_dict()
    b = pr._heatmap(_db_map(-50.0), "t<br><sup>s</sup>", zmin=-36.0,
                    z_share=pr.Z_SHARE_KEEP_CLIP).to_dict()
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
    """The row Divs of the side-by-side block (the last child of the Results tree)."""
    return tree.children[-1].children


def test_ab_results_render_one_row_per_product_two_columns_wide():
    import webapp.app as appmod

    tree = appmod._render_results(_ab_results_data(), "tab-results")
    rows = _rows(tree)
    # 1 header row + 1 row per product (range_az, fft).
    assert len(rows) == 3
    for row in rows:
        assert row.style["display"] == "flex"
        assert len(row.children) == 2
        for col in row.children:
            assert col.style["flex"] == appmod._AB_COLUMN_FLEX


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
    product a screen height apart, so the comparison had to be remembered."""
    import webapp.app as appmod

    data = _ab_results_data()
    data["range_az"]["layout"]["title"] = {"text": "ARM-A-RANGE-AZ"}
    data["_previous"]["range_az"]["layout"]["title"] = {"text": "ARM-B-RANGE-AZ"}
    tree = appmod._render_results(data, "tab-results")
    row = _rows(tree)[1]
    left_fig = row.children[0].children.children.figure
    right_fig = row.children[1].children.children.figure
    assert left_fig["layout"]["title"]["text"] == "ARM-A-RANGE-AZ"
    assert right_fig["layout"]["title"]["text"] == "ARM-B-RANGE-AZ"


def test_a_product_only_one_arm_produced_keeps_an_empty_cell_opposite_it():
    import webapp.app as appmod

    data = _ab_results_data()
    del data["_previous"]["fft"]
    tree = appmod._render_results(data, "tab-results")
    rows = _rows(tree)
    assert len(rows) == 3          # header + range_az + fft
    assert rows[2].children[1].children is None


def test_single_arm_results_keep_the_wrapping_grid():
    """Only the A/B path changed; a hand-edited single run still wraps its products
    across the full stage width (and a lone figure still takes the whole row)."""
    import webapp.app as appmod

    data = {"range_az": go.Figure().to_dict(), "_banner": "run #1"}
    tree = appmod._render_results(data, "tab-results")
    grid = tree.children[-1]
    assert grid.style == {"display": "flex", "flexWrap": "wrap"}
    assert grid.children[0].style["flex"] == "1 1 100%"
    assert "This run:" in _all_text(tree)


def test_the_screen_note_is_still_printed_once_above_both_columns():
    import webapp.app as appmod

    data = _ab_results_data()
    data["_screen_note"] = "a caveat the audience must see"
    text = _all_text(appmod._render_results(data, "tab-results"))
    assert text.count("a caveat the audience must see") == 1


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
    assert "colour limits shared with arm B" in _title_text(a)


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


def test_the_asset_pauses_on_plotlys_own_pause_button_and_resumes_on_play():
    """One clock means one pause: pressing pause on any panel stops every panel.
    The listener is capture-phase and stops propagation so plotly's own one-shot
    animation never also runs (two drivers = the drift this replaced)."""
    js = _CLOCK_JS.read_text(encoding="utf-8")
    assert "updatemenu-button" in js
    assert "stopPropagation" in js
    assert "S.paused" in js
    assert "slider-container" in js, "dragging the slider parks the panel"
    # capture-phase registration (the third argument of addEventListener)
    assert js.count("}, true);") >= 2


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
