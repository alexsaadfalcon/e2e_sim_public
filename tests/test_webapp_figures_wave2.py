"""Regressions from reading the rendered wave-1 demo screens (rehearsal, 2026-09-23):

1. Two-card (~600 px) width: the range-azimuth/range-elevation heatmap title ran off
   the right edge, and the frame slider's "frame N" label was drawn behind the
   play/pause buttons once the podium-distance font bump (12 -> 16 px) widened it.
2. The range-Doppler panel's adaptive display clip printed with false precision
   ("-36.233") and didn't say where the number came from.
3. The detector objectness panel drew ground-truth targets as a fixed 18 px circle,
   LARGER than the tolerance the scoreboard actually scores with -- a cross sitting on
   the circle could still be a scored miss.

Items 4/5 (the scoreboard table itself) are covered in tests/test_detector_scoreboard.py,
which owns that module.

REPOINTED for the 2026-09-24 panel-meta layout (see webapp/pipeline_runner.py's "PANEL
GEOMETRY AND THE PANEL-META CONTRACT" section). A figure carries no title/subtitle any
more; the panel's title + one-line caption + full Details body live in
``fig.layout.meta["panel"]`` (accessors ``panel_of``/``panel_caption``/``panel_text``).
Per-figure sliders/play buttons are gone -- one clock now drives every animated figure
(``webapp/assets/results_clock.js``). Every clause below was re-pointed to wherever it
now lives, never dropped; see each test's docstring for exactly where.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from webapp.pipeline_runner import (_add_frame_animation, _radar_cube_clip_db,
                                    figures_from_outputs, panel_caption, panel_of,
                                    panel_text)


# --------------------------------------------------------------------------------
# Item 1a: range-azimuth / range-elevation titles fit the two-card width
# --------------------------------------------------------------------------------
# The character-budget mechanism this used to pin (a plotly title's first `<br>` line
# overflowing a ~600 px panel) cannot recur: a figure carries no title at all now, and
# the panel's HTML title is a short, FIXED string built once (never grown by a wording
# change) -- see set_panel's callers below. That "fits the width" half is now the
# generic job of tests/test_webapp_layout_acceptance.py::test_every_title_is_one_line
# (<=60 chars, no wrap). What's still specific to THIS regression -- not covered
# there -- is that the qualifier ("non-coherent over ...") that used to be baked into
# the overflowing title stays OUT of the short title (it moved to Details), while
# remaining reachable; reachability itself is CHECK 15
# (OLD_SUBTITLE_CLAUSES["range_az"/"range_el"]) in that same module.
def test_range_az_title_is_short_the_qualifier_moved_to_details():
    torch = pytest.importorskip("torch")

    rng = np.random.default_rng(1)
    power = rng.random((8, 8)).astype(np.float32)
    ra = torch.from_numpy(power).to(torch.complex64)
    fig = figures_from_outputs({"range_az": [ra], "_axis_meta": {"range_az_bins": 8}})["range_az"]

    panel = panel_of(fig)
    assert "non-coherent" not in panel["title"]
    assert len(panel["title"]) <= 30
    text = panel_text(fig)
    assert "non-coherent over elevation" in text
    assert "peak - median" in text  # the stat still lives alongside the qualifier


def test_range_el_title_is_short_and_still_carries_its_qualifier():
    torch = pytest.importorskip("torch")

    ra = torch.rand((8, 8)).to(torch.complex64)
    fig = figures_from_outputs({"range_el": [ra], "_axis_meta": {"range_el_bins": 8}})["range_el"]

    panel = panel_of(fig)
    assert "non-coherent" not in panel["title"]
    assert len(panel["title"]) <= 30
    text = panel_text(fig)
    assert "non-coherent over azimuth" in text
    assert "peak - median" in text  # wave 5: range_el prints the statistic too


# --------------------------------------------------------------------------------
# Item 1b: the frame slider's "frame N" label clears the play/pause buttons
# --------------------------------------------------------------------------------
def _animated_heatmap_figure():
    base = go.Figure(data=go.Heatmap(z=[[0, 1], [1, 0]]))
    return _add_frame_animation(base, [np.zeros((2, 2)), np.ones((2, 2))])


def test_no_per_figure_transport_controls():
    """Repointed (layout spec section 4, "Slider / play controls", retired wave 11):
    the geometry this test used to pin -- the slider's `x` clearing the play/pause
    buttons' fixed pixel width, via `_SLIDER_BUTTONS_X_EXTENT` -- no longer applies
    because the per-figure transport itself was removed; one clock now drives every
    animated figure (`webapp/assets/results_clock.js`, wired in
    `webapp/app.py::_transport_bar`). `_SLIDER_BUTTONS_X_EXTENT` is kept in
    pipeline_runner.py only as a comment anchor recording why -- not imported here any
    more, since there is no geometry left to compute from it. What's left to protect:
    `_add_frame_animation` must never grow a slider or play/pause buttons back."""
    fig = _animated_heatmap_figure()
    assert not (fig.layout.sliders or ())
    assert not (fig.layout.updatemenus or ())


def test_animation_is_a_noop_below_two_frames():
    """Repointed: the old assertion was "no slider grows for a single frame"; with no
    slider left to grow at all, the behaviour this protects is `_add_frame_animation`'s
    own documented single-frame no-op -- `fig.frames` stays untouched rather than
    becoming a one-element animation."""
    base = go.Figure(data=go.Heatmap(z=[[0, 1], [1, 0]]))
    fig = _add_frame_animation(base, [np.zeros((2, 2))])
    assert not fig.frames


# --------------------------------------------------------------------------------
# Item 2: range-Doppler adaptive clip label -- rounded, and says what it is
# --------------------------------------------------------------------------------
def test_radar_cube_clip_label_rounds_and_explains_the_adaptive_case():
    """Repointed: the colour bar carries no title any more (layout spec section 4 --
    at 20 px a colour-bar title squeezed the plot to a sliver, defect 5); the rounded
    clip value is now the panel's one-line CAPTION and the "why" sentence is in
    Details. Substance unchanged: rounded, not raw-float precision; provenance
    reachable but not crammed into the narrow caption."""
    torch = pytest.importorskip("torch")

    rng = np.random.default_rng(7)
    cube = (rng.random((4, 8, 6)) + 1j * rng.random((4, 8, 6))).astype(np.complex64)
    cube[:, 3, 2] *= 20  # one bright cell so the map has real dynamic range
    cube_t = torch.from_numpy(cube)
    fig = figures_from_outputs({"radar_cube": [cube_t], "_axis_meta": {}})["radar_cube"]

    p = np.mean(np.abs(cube) ** 2, axis=0)
    db = 10 * np.log10(p / p.max() + 1e-12)
    clip = _radar_cube_clip_db(db)
    assert clip > -40.0, "fixture must exercise the adaptive (median-derived) branch"

    caption = panel_caption(fig)
    # Rounded to one decimal -- not the raw float's full precision.
    assert f"{clip:.1f}" in caption
    assert f"{clip:.6f}" not in panel_text(fig)
    # provenance lives in Details, so the one-line caption stays narrow
    assert "median floor" not in caption
    assert "median floor + 3 dB" in panel_text(fig)


def test_radar_cube_clip_label_plain_when_the_shared_floor_is_used():
    torch = pytest.importorskip("torch")

    # A quiet, low-floor frame: median stays well under -43 dB, so the clip is the
    # plain shared -40 dB, not "3 dB above" anything -- the label must not claim a
    # median-derived origin it doesn't have.
    quiet = np.full((4, 8, 6), 1e-6, dtype=np.complex64)
    quiet[:, 0, 0] = 1.0
    cube_t = torch.from_numpy(quiet)
    fig = figures_from_outputs({"radar_cube": [cube_t], "_axis_meta": {}})["radar_cube"]

    caption = panel_caption(fig)
    assert "clipped at -40.0" in caption
    assert "median floor" not in caption
    assert "shared floor" in panel_text(fig)


# --------------------------------------------------------------------------------
# Item 3: ground truth drawn as its own match-tolerance box, in data coordinates
# --------------------------------------------------------------------------------
def test_ground_truth_drawn_as_match_tolerance_rectangle_sized_by_match_criterion():
    torch = pytest.importorskip("torch")
    from e2e.ml.metrics import MatchCriterion

    crit = MatchCriterion()
    obj = np.zeros((2, 8, 16), dtype=np.float32)
    outputs = {
        "cfar_detection": [obj],
        "cfar_detections": [[(0, 0.1, 0.9, 12.0)]],
        "gt_detections": [[(0, -0.2, 1.0, 20.0)]],
        "_axis_meta": {"rx": {"grid": {"max_range_m": 40.0}},
                       "detector": {"mode": "cfar", "threshold": 0.66, "label": "x"}},
    }
    fig = figures_from_outputs(outputs)["cfar_detection"]

    shapes = fig.layout.shapes
    assert len([sh for sh in shapes if sh.type == "rect"]) == 1  # wave 4 adds a 40 m limit line
    # BY TYPE, not by index: wave 4's 40 m scoring-limit line is shape 0, so `shapes[0]`
    # read the line and every assertion below it failed on a defect that was not there.
    s = next(sh for sh in shapes if sh.type == "rect")
    cx, cy = -0.2, 20.0  # (sin_azimuth, surface_range_m) -- see d[1]/d[3] convention
    assert s.type == "rect"
    assert s.x0 == pytest.approx(cx - crit.max_sin_az_err)
    assert s.x1 == pytest.approx(cx + crit.max_sin_az_err)
    assert s.y0 == pytest.approx(cy - crit.max_range_err_m)
    assert s.y1 == pytest.approx(cy + crit.max_range_err_m)
    # White outline, no fill -- so it reads as a tolerance boundary, not a blob.
    assert s.line.color == "#ffffff"
    assert s.fillcolor in ("rgba(0,0,0,0)", None)

    # Crosses (detections) are untouched.
    # Hostile round 11, H8: detections are split into two traces by whether the
    # SCOREBOARD's own matcher matched them, so the room can count hits by eye. The
    # unmatched ones keep the cross.
    det_trace = next(t for t in fig.data if "unmatched" in (t.name or ""))
    assert det_trace.marker.symbol == "x"
    hit_trace = next(t for t in fig.data if "matched" in (t.name or "")
                     and "unmatched" not in (t.name or ""))
    assert hit_trace.marker.symbol == "diamond"

    gt_trace = next(t for t in fig.data if "ground truth" in (t.name or ""))
    # Repointed: the hit-RULE sentence used to be part of the ground-truth trace's own
    # legend name; it now lives in the panel's one-line CAPTION instead (layout spec
    # section 4, "Detector map" -- as a legend entry it was a ~100-char sentence inside
    # a dark 72 px block). The legend entry itself is back to a plain "n=" count.
    caption = panel_caption(fig)
    assert "hit = cross inside the box" in caption
    assert f"{crit.max_range_err_m:g}" in caption
    assert f"{crit.max_sin_az_err:g}" in caption
    # No longer an oversized fixed-pixel circle standing in for the tolerance.
    assert gt_trace.marker.size < 18


def test_no_ground_truth_no_shapes():
    obj = np.zeros((2, 8, 16), dtype=np.float32)
    outputs = {
        "cfar_detection": [obj],
        "cfar_detections": [[]],
        "_axis_meta": {"detector": {"mode": "cfar", "threshold": 0.5, "label": "x"}},
    }
    fig = figures_from_outputs(outputs)["cfar_detection"]
    assert not [sh for sh in (fig.layout.shapes or ()) if sh.type == "rect"]  # wave 4: only the limit line remains
