"""Wave 7 (KA-band screens, 2026-09-23) fixes owned by the pipeline_runner shard:

X4/X5: the range-azimuth/range-elevation heat maps used a fixed -40 dB clip; on the
Ka-band munich frames (rank-1-by-geometry LoS return, F94) 99%+ of pixels sit below it
and the panel reads as near-black. The clip now follows the frame's own median floor
(+3 dB), exactly like the Thrust 5 range-Doppler panel's `_radar_cube_clip_db`, printed
in the subline as "clip -NN dB (median floor + 3 dB)" (or "(shared floor)" when the
frame's floor is already well under -43 dB, in which case the clip is unchanged).

X6/X7: no panel stated what a display gate is worth in metres, or how far the range
axis can go before it aliases -- a hostile-expert read traced a stale "20-22 m stripe"
quote to exactly this gap. The subline now states both, computed from the frame's own
freq_plan + display bin count (`_range_per_gate_m` / `_native_unambiguous_range_m`),
never hand-typed. This lives in the SUBLINE, not the yaxis title, so the short
"range (m)" axis title (pinned by tests/test_webapp_figures_wave3.py, an unowned file)
is untouched -- a rotated axis-title collided with the heatmap's own title at podium
font size before that fix, which is why the title stays short.

F96 (wave 11, 2026-09-24, notes/ESTABLISHED_FACTS.md): the "unambig N m" wording this
file used to pin read as a physical range CEILING; it is actually HALF of the frame's
own N-point FFT period, the other half cropped as negative delay (a real return
between N/2 and N m is thrown away by the display, not out of range). The subline now
reads "display 0-N m of 2N m unambig (neg.-delay half cropped)", both numbers still
computed from the frame's own freq_plan, never typed -- this file absorbed the pin
update as a handoff item from the pipeline_runner shard (the fix originally landed in
webapp/pipeline_runner.py without touching this then-unowned file).

LAYOUT SPEC, 2026-09-24: a figure no longer carries a title. Every clause this file
used to pin on `fig.layout.title.text` now lives in the panel meta (`layout.meta.panel`)
-- the gate calibration, the unambiguous-range wording and the clip provenance all moved
into the per-arm Details body, reached through `panel_text(fig)` (title + caption + all
of Details, one string -- see `webapp/pipeline_runner.py`'s "PANEL GEOMETRY AND THE
PANEL-META CONTRACT" section). Repointed below rather than dropped: the defects these
tests protect (a hand-typed calibration number, a stale ceiling reading of "unambig",
an unclipped adaptive-clip statistic) are unrelated to where the words are drawn.

The colour-bar-title precision guard (`fig.data[0].colorbar.title.text`) has no
counterpart any more -- colour bars carry no title at all now (layout spec section 4;
pinned once, for every product, by
`test_webapp_layout_acceptance.py::test_no_colorbar_carries_a_title`, not duplicated
here). What IS still this file's job is the other half of that old assertion: the
clip value appears at 1-decimal precision, never a raw float repr -- repointed onto
`panel_text(fig)` below.
"""
from __future__ import annotations

import numpy as np
import pytest

from webapp.pipeline_runner import (
    _native_unambiguous_range_m, _radar_cube_clip_db, _range_per_gate_m, figures_from_outputs,
    panel_text,
)


def _munich_axis_meta(**bins):
    return {"n_freqs": 64, "freq_span_hz": 3e9, **bins}


# --------------------------------------------------------------------------------
# X6/X7: gate calibration is computed, not hand-typed
# --------------------------------------------------------------------------------
def test_range_per_gate_m_matches_direct_computation():
    # n_freqs=64, bins=8 -> per=8, C/(2*3e9)*8
    C = 2.99792458e8
    expected = 8 * C / (2.0 * 3e9)
    assert _range_per_gate_m(8, 3e9, 64) == pytest.approx(expected)


def test_native_unambiguous_range_m_matches_direct_computation():
    C = 2.99792458e8
    expected = (64 // 2) * C / (2.0 * 3e9)
    assert _native_unambiguous_range_m(3e9, 64) == pytest.approx(expected)


def test_range_az_subline_states_metres_per_gate_and_unambiguous_range():
    torch = pytest.importorskip("torch")

    ra = torch.rand((8, 8)).to(torch.complex64)
    fig = figures_from_outputs({
        "range_az": [ra], "_axis_meta": _munich_axis_meta(range_az_bins=8),
    })["range_az"]
    text = panel_text(fig)
    expected_gate = _range_per_gate_m(8, 3e9, 64)
    C = 2.99792458e8
    expected_full = 64 * C / (2.0 * 3e9)      # N * c/(2B): the full FFT period
    expected_half = expected_full / 2.0        # the physical half the display shows
    assert f"{expected_gate:.2f} m/gate" in text
    # F96: "unambig N m" read as a physical ceiling; it is half of the frame's own
    # N-point FFT period, the other half cropped as negative delay.
    assert f"display 0-{expected_half:.0f} m of {expected_full:.0f} m unambig " \
           "(neg.-delay half cropped)" in text


def test_range_el_subline_states_metres_per_gate_too():
    torch = pytest.importorskip("torch")

    re_ = torch.rand((8, 8)).to(torch.complex64)
    fig = figures_from_outputs({
        "range_el": [re_], "_axis_meta": _munich_axis_meta(range_el_bins=8),
    })["range_el"]
    text = panel_text(fig)
    expected_gate = _range_per_gate_m(8, 3e9, 64)
    assert f"{expected_gate:.2f} m/gate" in text


def test_range_az_gate_calibration_absent_without_axis_metadata():
    """No n_freqs/freq_span_hz -> nothing to compute the calibration from; the bin-index
    fallback carries no physical claim (mirrors the "earliest arrival" fallback)."""
    torch = pytest.importorskip("torch")

    ra = torch.rand((8, 8)).to(torch.complex64)
    fig = figures_from_outputs({"range_az": [ra],
                                "_axis_meta": {"range_az_bins": 8}})["range_az"]
    assert "m/gate" not in panel_text(fig)


def test_range_az_yaxis_title_stays_short_the_calibration_lives_in_the_subline():
    """Guards the design choice: the info goes in the (already-mutable) subline, never
    the yaxis title -- test_webapp_figures_wave3.py (an unowned file) pins the title to
    the exact short string "range (m)"; if that ever regresses it is this test's job to
    say why, not that test's."""
    torch = pytest.importorskip("torch")

    ra = torch.rand((8, 8)).to(torch.complex64)
    fig = figures_from_outputs({
        "range_az": [ra], "_axis_meta": _munich_axis_meta(range_az_bins=8),
    })["range_az"]
    assert fig.layout.yaxis.title.text == "range (m)"


# --------------------------------------------------------------------------------
# X4/X5: adaptive display clip, same treatment as the range-Doppler panel
# --------------------------------------------------------------------------------
def test_range_az_clip_tightens_for_a_high_floor_frame_and_says_so():
    """No axis metadata (bin-index fallback) -- deliberately, so the panel does not
    crop to the nonnegative-range half and `frames_db[-1]` (what the clip is computed
    on) is exactly this fixture's own array, matching `_radar_cube_clip_db` computed
    directly on `_to_numpy_abs_db(ra)` with no extra bookkeeping."""
    torch = pytest.importorskip("torch")
    from webapp.pipeline_runner import _to_numpy_abs_db

    # A "busy" map: uniform random power, so the median sits well above -43 dB.
    rng = np.random.default_rng(11)
    power = rng.random((16, 16)).astype(np.float32)
    ra = torch.from_numpy(power).to(torch.complex64)
    fig = figures_from_outputs({"range_az": [ra],
                                "_axis_meta": {"range_az_bins": 16}})["range_az"]

    expected_clip = _radar_cube_clip_db(_to_numpy_abs_db(ra))
    assert expected_clip > -40.0, "fixture must exercise the adaptive branch"
    assert fig.data[0].zmin == pytest.approx(expected_clip)
    text = panel_text(fig)
    assert f"clip {expected_clip:.1f} dB (median floor + 3 dB)" in text
    # The colour bar itself carries no title any more (layout spec section 4; pinned
    # once for every product by
    # test_webapp_layout_acceptance.py::test_no_colorbar_carries_a_title). What this
    # test still owns is the precision guard: the clip is stated at 1 decimal
    # somewhere reachable, never as a raw float repr.
    assert f"{expected_clip:.6f}" not in text


def test_range_el_clip_uses_the_shared_floor_and_says_so_for_a_sparse_frame():
    """A near-rank-1 munich Ka map (F94): one bright cell, everything else near the
    noise floor -- the median sits far below -43 dB, so the clip stays the plain
    shared -40 dB, exactly like the range-Doppler panel's own "quiet frame" case."""
    torch = pytest.importorskip("torch")

    sparse = np.full((16, 16), 1e-6, dtype=np.float32)
    sparse[0, 0] = 1.0
    re_ = torch.from_numpy(sparse).to(torch.complex64)
    fig = figures_from_outputs({"range_el": [re_],
                                "_axis_meta": {"range_el_bins": 16}})["range_el"]

    assert fig.data[0].zmin == pytest.approx(-40.0)
    text = panel_text(fig)
    assert "clip -40.0 dB (shared floor)" in text


def test_range_az_peak_minus_median_statistic_is_unaffected_by_the_new_clip():
    """The peak-median headline the T1/T2 cards quote is computed on the UNCLIPPED map
    (Change 2, pre-existing) -- adding the adaptive clip must not perturb it."""
    torch = pytest.importorskip("torch")
    from webapp.pipeline_runner import _peak_minus_median_db

    rng = np.random.default_rng(3)
    power = rng.random((16, 16)).astype(np.float32)
    ra = torch.from_numpy(power).to(torch.complex64)
    fig = figures_from_outputs({
        "range_az": [ra], "_axis_meta": _munich_axis_meta(range_az_bins=16),
    })["range_az"]

    db = 10 * np.log10(power / power.max() + 1e-12)
    expected = round(float(db.max() - np.median(db)), 1)
    text = panel_text(fig)
    assert f"peak - median, dB: {expected:.1f}" in text
