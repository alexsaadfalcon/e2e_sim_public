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
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from webapp.pipeline_runner import (_SLIDER_BUTTONS_X_EXTENT, _SLIDER_X,
                                    _add_frame_animation, _radar_cube_clip_db,
                                    figures_from_outputs)


# --------------------------------------------------------------------------------
# Item 1a: range-azimuth / range-elevation titles fit the two-card width
# --------------------------------------------------------------------------------
#: Rough character budget for a single-line title at the podium-distance font size on
#: a two-card (~600 px) panel -- the old single-line "Range-Azimuth power (non-coherent
#: over elevation)" (51 chars) ran off the right edge there (rehearsal screenshot).
_TITLE_MAIN_LINE_MAX_CHARS = 30


def test_range_az_main_title_is_short_the_qualifier_is_a_subline():
    torch = pytest.importorskip("torch")

    rng = np.random.default_rng(1)
    power = rng.random((8, 8)).astype(np.float32)
    ra = torch.from_numpy(power).to(torch.complex64)
    fig = figures_from_outputs({"range_az": [ra], "_axis_meta": {"range_az_bins": 8}})["range_az"]

    title = fig.layout.title.text
    main_line = title.split("<br>")[0]
    assert len(main_line) <= _TITLE_MAIN_LINE_MAX_CHARS, main_line
    assert "non-coherent over elevation" in title
    assert "peak - median" in title  # the stat still lives alongside the qualifier


def test_range_el_main_title_is_short_and_carries_its_qualifier():
    torch = pytest.importorskip("torch")

    ra = torch.rand((8, 8)).to(torch.complex64)
    fig = figures_from_outputs({"range_el": [ra], "_axis_meta": {"range_el_bins": 8}})["range_el"]

    title = fig.layout.title.text
    main_line = title.split("<br>")[0]
    assert len(main_line) <= _TITLE_MAIN_LINE_MAX_CHARS, main_line
    assert "non-coherent over azimuth" in title
    assert "peak - median" not in title


# --------------------------------------------------------------------------------
# Item 1b: the frame slider's "frame N" label clears the play/pause buttons
# --------------------------------------------------------------------------------
def _animated_heatmap_figure():
    base = go.Figure(data=go.Heatmap(z=[[0, 1], [1, 0]]))
    return _add_frame_animation(base, [np.zeros((2, 2)), np.ones((2, 2))])


def test_slider_x_clears_the_play_pause_buttons():
    fig = _animated_heatmap_figure()
    slider = fig.layout.sliders[0]
    menu = fig.layout.updatemenus[0]
    # The label is explicitly left-anchored -- it extends RIGHT from the slider's own
    # x, away from the buttons, rather than growing left into them as the font widens
    # (the regression: right/center-implied growth ate into the button region once the
    # currentvalue font went from 12 -> 16 px).
    assert slider.currentvalue.xanchor == "left"
    assert slider.x >= menu.x + _SLIDER_BUTTONS_X_EXTENT
    # Documents the actual regression this test pins: the old fixed x=0.2 must not
    # silently come back.
    assert slider.x > 0.2


def test_slider_noop_below_two_frames():
    base = go.Figure(data=go.Heatmap(z=[[0, 1], [1, 0]]))
    fig = _add_frame_animation(base, [np.zeros((2, 2))])
    assert not fig.layout.sliders


# --------------------------------------------------------------------------------
# Item 2: range-Doppler adaptive clip label -- rounded, and says what it is
# --------------------------------------------------------------------------------
def test_radar_cube_clip_label_rounds_and_explains_the_adaptive_case():
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

    label = fig.data[0].colorbar.title.text
    # Rounded to one decimal -- not the raw float's full precision.
    assert f"{clip:.1f}" in label
    assert f"{clip:.6f}" not in label
    # provenance lives in the panel subline, so the colorbar stays narrow
    assert "median floor" not in label
    assert "median floor + 3 dB" in fig.layout.title.text


def test_radar_cube_clip_label_plain_when_the_shared_floor_is_used():
    torch = pytest.importorskip("torch")

    # A quiet, low-floor frame: median stays well under -43 dB, so the clip is the
    # plain shared -40 dB, not "3 dB above" anything -- the label must not claim a
    # median-derived origin it doesn't have.
    quiet = np.full((4, 8, 6), 1e-6, dtype=np.complex64)
    quiet[:, 0, 0] = 1.0
    cube_t = torch.from_numpy(quiet)
    fig = figures_from_outputs({"radar_cube": [cube_t], "_axis_meta": {}})["radar_cube"]

    label = fig.data[0].colorbar.title.text
    assert "clipped at -40.0" in label
    assert "median floor" not in label
    assert "shared floor" in fig.layout.title.text


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
    s = shapes[0]
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
    det_trace = next(t for t in fig.data if (t.name or "").startswith("detections"))
    assert det_trace.marker.symbol == "x"

    gt_trace = next(t for t in fig.data if "ground truth" in (t.name or ""))
    assert "hit = cross inside the box" in gt_trace.name
    assert f"{crit.max_range_err_m:g}" in gt_trace.name
    assert f"{crit.max_sin_az_err:g}" in gt_trace.name
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
