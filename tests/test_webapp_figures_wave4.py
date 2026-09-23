"""Regressions from a THIRD hostile-expert read of the rendered demo screens
(2026-09-23), on top of the wave-1/2/3 fixes already pinned in
tests/test_webapp_figures_wave2.py, tests/test_webapp_figures_wave3.py and
tests/test_detector_scoreboard.py:

Finding 5: the detector objectness panel's range axis ran the label grid's full
physical extent (~102 m), but labels AND the offline scoring crop both stop at a much
smaller range (40 m on beat_cfar.json) -- so ~60% of the axis was structurally empty.
Cropped here to 0..50 m with a dashed line at the real scoring crop (read from
beat_cfar.json, never a literal "40").

Findings 1-4 (renamed scoreboard rows, the precision-ceiling caption, the CI-row
seed-spread caveat, the out-of-distribution row, the null-arm display name, the
hit-rate wording) are covered in tests/test_detector_scoreboard.py, which owns
`webapp/detector_scoreboard.py`.

A 4th hostile-expert read (2026-09-23) added two more, both in `webapp/pipeline_runner.py`:

* the range-elevation panel now carries the same peak-median dB statistic the
  range-azimuth panel already prints (previously range_az only -- see
  `tests/test_webapp_ab.py::test_range_el_panel_carries_no_peak_minus_median_statistic`
  and `tests/test_webapp_figures_wave2.py::
  test_range_el_main_title_is_short_and_carries_its_qualifier`, both of which pinned
  the OLD "range_el gets none" behaviour and now need updating by whoever owns those
  files -- this module does not touch them);
* the subspace-error plot's dashed reference line is now labelled "warm-start settled
  level (0.06, reference)" so it does not read as THIS run's own level on a curve that
  sits well above it.
"""

from __future__ import annotations

import numpy as np
import pytest

from webapp import detector_scoreboard
from webapp.pipeline_runner import figures_from_outputs


def _detector_outputs(**axis_meta_extra):
    obj = np.zeros((2, 8, 16), dtype=np.float32)
    return {
        "cfar_detection": [obj],
        "cfar_detections": [[]],
        "_axis_meta": {"detector": {"mode": "cfar", "threshold": 0.5, "label": "x"},
                      **axis_meta_extra},
    }


def test_detector_panel_y_axis_cropped_to_50m_with_real_beat_cfar_json():
    """`beat_cfar.json` is present in this repo (see
    tests/test_detector_scoreboard.py::test_default_beat_cfar_json_exists) -- the
    panel must crop to [0, 50] and draw the line at whatever `max_range_m` that file
    actually stores, not a hardcoded 40."""
    expected = detector_scoreboard.scoring_max_range_m()
    assert expected is not None, "this test needs the real beat_cfar.json present"

    fig = figures_from_outputs(_detector_outputs())["cfar_detection"]
    assert fig.layout.yaxis.range == pytest.approx((0.0, 50.0))
    lines = [s for s in fig.layout.shapes if s.type == "line"]
    assert len(lines) == 1
    assert lines[0].y0 == pytest.approx(expected) and lines[0].y1 == pytest.approx(expected)
    ann_texts = " ".join(a.text or "" for a in fig.layout.annotations)
    assert f"labels & scoring stop at {expected:g} m" in ann_texts


def test_detector_panel_y_axis_uncropped_when_beat_cfar_json_missing(monkeypatch):
    """No scoring crop on record (e.g. this deployment has no beat_cfar.json) -> the
    axis keeps its old, uncropped behaviour rather than inventing a 40 m crop."""
    monkeypatch.setattr(detector_scoreboard, "scoring_max_range_m", lambda *a, **k: None)
    fig = figures_from_outputs(_detector_outputs())["cfar_detection"]
    assert fig.layout.yaxis.range is None
    assert not fig.layout.shapes


def test_detector_panel_crop_line_uses_the_stored_value_not_forty_literal(monkeypatch):
    """Pins the "read from the run, not a literal" requirement directly: a fake
    scoring crop of 17 m must show up as 17, not 40."""
    monkeypatch.setattr(detector_scoreboard, "scoring_max_range_m", lambda *a, **k: 17.0)
    fig = figures_from_outputs(_detector_outputs())["cfar_detection"]
    lines = [s for s in fig.layout.shapes if s.type == "line"]
    assert lines[0].y0 == pytest.approx(17.0)
    ann_texts = " ".join(a.text or "" for a in fig.layout.annotations)
    assert "labels & scoring stop at 17 m" in ann_texts
    assert "40" not in ann_texts


# --------------------------------------------------------------------------------
# The range-Doppler radar_cube panel keeps its full extent (finding 5 explicitly
# excludes it) -- pinned so a future edit doesn't accidentally crop it too.
# --------------------------------------------------------------------------------
def test_radar_cube_panel_is_not_cropped_by_the_detector_scoring_range():
    torch = pytest.importorskip("torch")

    rng = np.random.default_rng(3)
    cube = (rng.random((4, 8, 6)) + 1j * rng.random((4, 8, 6))).astype(np.complex64)
    cube_t = torch.from_numpy(cube)
    fig = figures_from_outputs({"radar_cube": [cube_t], "_axis_meta": {}})["radar_cube"]
    assert fig.layout.yaxis.range is None
    assert not fig.layout.shapes


# --------------------------------------------------------------------------------
# 4th hostile-expert read, finding: range-elevation carried no peak-median statistic
# while range-azimuth did, so a screen note about the elevation cut had no number
# beside it. Both panels must now carry the same `_peak_minus_median_db` statistic,
# computed on the unclipped map exactly as range-azimuth's already is.
# --------------------------------------------------------------------------------
def test_range_az_and_range_el_both_carry_the_peak_minus_median_statistic():
    torch = pytest.importorskip("torch")
    from webapp.pipeline_runner import _peak_minus_median_db, _to_numpy_abs_db

    rng = np.random.default_rng(7)

    def _cplx(shape):
        a = (rng.random(shape) + 1j * rng.random(shape)).astype(np.complex64)
        return torch.from_numpy(a)

    az = _cplx((10, 10))
    el = _cplx((10, 10))
    outputs = {
        "range_az": [az], "range_el": [el],
        "_axis_meta": {"range_az_bins": 10, "range_el_bins": 10},
    }
    figs = figures_from_outputs(outputs)

    expected_az = _peak_minus_median_db(_to_numpy_abs_db(az))
    expected_el = _peak_minus_median_db(_to_numpy_abs_db(el))
    az_title = figs["range_az"].layout.title.text.replace("<br>", " ")
    el_title = figs["range_el"].layout.title.text.replace("<br>", " ")
    assert "peak - median" in az_title and f"{expected_az:.1f}" in az_title
    assert "peak - median" in el_title and f"{expected_el:.1f}" in el_title


# --------------------------------------------------------------------------------
# 4th hostile-expert read, finding: the subspace-error plot's dashed reference line
# read as THIS run's own settled level even on a curve sitting well above it (e.g. a
# cold-start/rank-collapse run reaching ~0.62) -- relabelled to say it is a separate
# warm-start reference case.
# --------------------------------------------------------------------------------
def test_subspace_err_dashed_line_labelled_as_a_reference_not_this_runs_level():
    from webapp.pipeline_runner import _SUBSPACE_ERR_SETTLED_LEVEL, figures_from_outputs as ffo

    fig = ffo({"subspace_err": [0.5, 0.55, 0.62], "_axis_meta": {}})["subspace_err"]
    lines = [s for s in fig.layout.shapes if s.type == "line"]
    assert len(lines) == 1
    ann_texts = " ".join(a.text or "" for a in fig.layout.annotations)
    assert f"warm-start settled level ({_SUBSPACE_ERR_SETTLED_LEVEL:g}, reference)" in ann_texts
