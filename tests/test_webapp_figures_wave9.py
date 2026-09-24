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
   only the Block Diagram tab's status line.
7. (coordinator addendum, twice-corrected) The range-azimuth/range-elevation subline
   now also states, per frame, where the brightest VISIBLE (beyond the direct-path
   band) return actually is. The direct-path/leakage cell's own size is stated in
   PHYSICAL units ("one {gate size} m gate (sub-pixel here, not visible)"), never as
   a pixel count: a first version divided this module's own DECLARED plot-domain
   constant by the bin count, which is not the browser's actual rendered pixel
   height (measured wrong against the PNG: printed ~2.2 px, the real rendered gate
   was ~1.6 px) -- a claim this module has no way to verify from inside Python, so
   it no longer makes one.
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

    # Each arm carries its OWN note (arms can differ -- item 6's own requirement).
    assert data["_notes"] == "arm-0 note: evaluated at 14.25-15.75 GHz"
    assert data["_previous"]["_notes"] == "arm-1 note: evaluated at 14.25-15.75 GHz"

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
    assert data.get("_notes") == "single-run note: evaluated at 14.25-15.75 GHz"
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
    assert "sub-pixel here, not visible" in text
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
