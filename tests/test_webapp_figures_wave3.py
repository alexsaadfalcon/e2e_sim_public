"""Regressions from a SECOND hostile-expert read of the rendered demo screens
(2026-09-23), on top of the wave-1/wave-2 fixes already pinned in
tests/test_webapp_figures_wave2.py and tests/test_detector_scoreboard.py:

1/2. Detector scoreboard subline/rows and the stored-PR legend's in-distribution
   qualifier + CI -- covered in tests/test_detector_scoreboard.py, which owns that
   module.
3. The range-Doppler panel's adaptive-clip subline was clipped mid-sentence at
   two-card (~600 px) width ("clip -36.2 dB = this frame's median floor + "):
   shortened, and pinned here to a hard character budget.
4. The range-azimuth/range-profile panels' range axis is delay-normalised (Sionna's
   normalize_delays=True at generation, see webapp/demo_presets.py), so range 0 is
   the earliest arrival, not "no target" -- the unlabelled bright band there read as
   an unlabelled target. Also: the two panels compress the SAME physical axis but
   rendered different extents (0-22 m vs 0-25 m) because a Heatmap trace's autorange
   pads differently than a Scatter trace's -- pinned here to render identically.

Plus a pixel-measured legibility re-check (coordinator, 2026-09-23): the podium-
distance floor's tick/colorbar sizes rendered smaller in actual ink height than the
Detector scoreboard's table beside them on the same screen; the floor is raised here.
"""

from __future__ import annotations

import numpy as np
import pytest

from webapp.pipeline_runner import (_LEGIBLE_COLORBAR_TICK_SIZE, _LEGIBLE_COLORBAR_TITLE_SIZE,
                                    _LEGIBLE_FONT_SIZE, _LEGIBLE_TICK_SIZE,
                                    _cropped_nonneg_range_axis, _heatmap, _make_legible,
                                    figures_from_outputs)

# --------------------------------------------------------------------------------
# Item 3: range-Doppler adaptive-clip subline fits a two-card panel
# --------------------------------------------------------------------------------
#: The subline ran off a ~600 px card mid-sentence at its old, longer wording
#: ("(non-coherent over channels); clip -36.2 dB = this frame's median floor + 3
#: dB"); the owner's shortened replacement must fit comfortably under that budget.
_RD_SUBLINE_MAX_CHARS = 60


def _rd_subline(fig) -> str:
    title = fig.layout.title.text
    assert "<sup>" in title and "</sup>" in title, title
    return title.split("<sup>")[1].split("</sup>")[0]


def test_radar_cube_clip_subline_fits_when_adaptive():
    torch = pytest.importorskip("torch")

    rng = np.random.default_rng(7)
    cube = (rng.random((4, 8, 6)) + 1j * rng.random((4, 8, 6))).astype(np.complex64)
    cube[:, 3, 2] *= 20  # one bright cell -> real dynamic range -> adaptive clip branch
    cube_t = torch.from_numpy(cube)
    fig = figures_from_outputs({"radar_cube": [cube_t], "_axis_meta": {}})["radar_cube"]

    subline = _rd_subline(fig)
    assert len(subline) <= _RD_SUBLINE_MAX_CHARS, subline
    assert "median floor + 3 dB" in subline
    assert "clip" in subline


def test_radar_cube_clip_subline_fits_when_shared_floor():
    torch = pytest.importorskip("torch")

    quiet = np.full((4, 8, 6), 1e-6, dtype=np.complex64)
    quiet[:, 0, 0] = 1.0
    cube_t = torch.from_numpy(quiet)
    fig = figures_from_outputs({"radar_cube": [cube_t], "_axis_meta": {}})["radar_cube"]

    subline = _rd_subline(fig)
    assert len(subline) <= _RD_SUBLINE_MAX_CHARS, subline
    assert "shared floor" in subline


# --------------------------------------------------------------------------------
# Item 4a: range-azimuth / range-profile y/x-axis label states "0 = earliest arrival"
# --------------------------------------------------------------------------------
def _munich_axis_meta(**bins):
    # n_freqs/freq_span_hz mirror the values a munich .pkl run actually carries
    # through run_pipeline's own axis-metadata stash (see figures_from_outputs'
    # docstring reference); small enough here to keep the fixtures cheap.
    return {"n_freqs": 64, "freq_span_hz": 3e9, **bins}


def test_range_az_subline_states_earliest_arrival_not_bare_range():
    """The claim lives in the SUBLINE, not the (rotated) y-axis title: a rotated
    "range (m; 0 = earliest arrival)" axis title ran into the heatmap's own title
    text at podium font size (fresh-context re-check, 2026-09-23) -- the axis title
    itself stays the short "range (m)"."""
    torch = pytest.importorskip("torch")

    rng = np.random.default_rng(1)
    ra = torch.from_numpy(rng.random((8, 8)).astype(np.float32)).to(torch.complex64)
    fig = figures_from_outputs({
        "range_az": [ra],
        "_axis_meta": _munich_axis_meta(range_az_bins=8),
    })["range_az"]
    assert fig.layout.yaxis.title.text == "range (m)"
    # The subline is long enough here (qualifier + peak-median stat + this note)
    # that `_wrap_text` line-breaks it -- reassemble before substring-checking.
    assert "range 0 = earliest arrival" in fig.layout.title.text.replace("<br>", " ")


def test_range_el_subline_states_earliest_arrival_too():
    """range_el shares the exact same delay-normalised axis (see the loop in
    figures_from_outputs building both from one `_range_axis` call per key) -- the
    caveat is not range-azimuth-specific."""
    torch = pytest.importorskip("torch")

    re_ = torch.rand((8, 8)).to(torch.complex64)
    fig = figures_from_outputs({
        "range_el": [re_],
        "_axis_meta": _munich_axis_meta(range_el_bins=8),
    })["range_el"]
    assert fig.layout.yaxis.title.text == "range (m)"
    assert "range 0 = earliest arrival" in fig.layout.title.text


def test_range_az_ylabel_falls_back_to_bins_without_axis_metadata():
    """No n_freqs/freq_span_hz (e.g. a hand-built outputs dict) -> the bin-index
    fallback label, which carries no "earliest arrival" physical claim to make."""
    torch = pytest.importorskip("torch")

    ra = torch.rand((8, 8)).to(torch.complex64)
    fig = figures_from_outputs({"range_az": [ra],
                                "_axis_meta": {"range_az_bins": 8}})["range_az"]
    assert fig.layout.yaxis.title.text == "range (bins)"
    assert "earliest arrival" not in fig.layout.title.text


def test_range_profile_xlabel_states_earliest_arrival_not_bare_range():
    torch = pytest.importorskip("torch")

    prof = torch.rand(8, dtype=torch.float32)
    fig = figures_from_outputs({
        "range_profile_agg": [prof],
        "_axis_meta": _munich_axis_meta(range_profile_bins=8),
    })["range_profile"]
    assert fig.layout.xaxis.title.text == "range (m; 0 = earliest arrival)"


def test_corpus_replay_detector_panel_range_axis_is_not_relabelled():
    """The corpus-replay detector/radar-cube panels use a DIFFERENT, absolute range
    axis (from the ADC dechirp geometry) -- the delay-normalised caveat above must
    NOT leak onto them (webapp/demo_presets.py's thrust5 scripts already say the
    absolute-range story on screen; a second, contradictory label here would be a
    second, wrong claim about the same axis)."""
    obj = np.zeros((2, 8, 16), dtype=np.float32)
    outputs = {
        "cfar_detection": [obj],
        "cfar_detections": [[]],
        "_axis_meta": {"detector": {"mode": "cfar", "threshold": 0.5, "label": "x"}},
    }
    fig = figures_from_outputs(outputs)["cfar_detection"]
    assert fig.layout.yaxis.title.text == "range (m)"
    assert "earliest arrival" not in fig.layout.yaxis.title.text


# --------------------------------------------------------------------------------
# Item 4b: range-azimuth heatmap and range-profile line plot share one extent
# --------------------------------------------------------------------------------
def test_range_az_and_range_profile_share_the_same_extent_when_both_present():
    torch = pytest.importorskip("torch")

    rng = np.random.default_rng(2)
    ra = torch.from_numpy(rng.random((8, 8)).astype(np.float32)).to(torch.complex64)
    prof = torch.rand(8, dtype=torch.float32)
    outputs = {
        "range_az": [ra],
        "range_profile_agg": [prof],
        "_axis_meta": _munich_axis_meta(range_az_bins=8, range_profile_bins=8),
    }
    figs = figures_from_outputs(outputs)

    expected = float(_cropped_nonneg_range_axis(8, 3e9, 64).max())
    az_range = figs["range_az"].layout.yaxis.range
    profile_range = figs["range_profile"].layout.xaxis.range
    assert az_range is not None and profile_range is not None
    assert az_range == pytest.approx((0.0, expected))
    assert profile_range == pytest.approx((0.0, expected))


def test_range_az_keeps_its_own_autorange_when_no_profile_panel_present():
    """No range_profile panel this run -> nothing to share an extent with, so
    range_az must NOT force a range (falls back to its historical autorange)."""
    torch = pytest.importorskip("torch")

    ra = torch.rand((8, 8)).to(torch.complex64)
    fig = figures_from_outputs({
        "range_az": [ra],
        "_axis_meta": _munich_axis_meta(range_az_bins=8),
    })["range_az"]
    assert fig.layout.yaxis.range is None


# --------------------------------------------------------------------------------
# Pixel-measured legibility re-check (coordinator, 2026-09-23): the first
# podium-distance pass's 14-15 px tick/colorbar sizes measured 8-13 px of actual ink
# height against the SAME 20 px standing threshold, and sat visibly smaller than the
# Detector scoreboard's table (webapp/detector_scoreboard.py, already 18-20 px) on
# the same Thrust 5 screen.
# --------------------------------------------------------------------------------
def test_legible_floor_matches_the_scoreboards_20px_scale():
    assert _LEGIBLE_TICK_SIZE >= 20
    assert _LEGIBLE_COLORBAR_TICK_SIZE >= 20
    assert _LEGIBLE_COLORBAR_TITLE_SIZE >= 20
    assert _LEGIBLE_FONT_SIZE >= 18


def test_make_legible_applies_the_raised_floor_to_ticks_and_colorbar():
    fig = _heatmap(np.zeros((4, 4)), "t", colorbar_title="cbar")
    _make_legible(fig)
    assert fig.layout.xaxis.tickfont.size >= 20
    assert fig.layout.yaxis.tickfont.size >= 20
    assert fig.layout.font.size >= 18
    cbar = fig.data[0].colorbar
    assert cbar.tickfont.size >= 20
    assert cbar.title.font.size >= 20
