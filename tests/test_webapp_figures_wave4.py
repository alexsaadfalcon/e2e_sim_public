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
