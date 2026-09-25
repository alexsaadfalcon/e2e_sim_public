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
    _display_range_axis, _radar_cube_clip_db, _range_meta_from_grid, figures_from_outputs,
    panel_text,
)


def _munich_axis_meta(**bins):
    return {"n_freqs": 64, "freq_span_hz": 3e9, **bins}


# --------------------------------------------------------------------------------
# X6/X7: gate calibration is computed, not hand-typed
# --------------------------------------------------------------------------------
# REPOINTED 2026-09-24 (one-chain integration). These tests were written against this
# module's own range calibration (`_range_per_gate_m` / `_native_unambiguous_range_m` /
# `_native_range_resolution_m` / `_range_axis`), which assumed each product ran its own
# range FFT over `n_freqs` samples, fftshifted it and displayed the non-negative half.
# The spine's `RangeTransformBlock` now does that transform once in front of every
# product and crops the negative half itself, so those six helpers are deleted and the
# calibration comes from `e2e.chain.receive.range_axis_m` via `_range_meta_from_grid` /
# `_display_range_axis`. The ASSERTIONS are unchanged in intent -- the panel still states
# metres per gate, the native resolution, the ratio and the window -- but their expected
# values are recomputed from the new (single) authority, and the window is now stated in
# the owner's bistatic excess-path convention (ballot 2B), where every metre is twice the
# v1.0 number.


def test_display_gate_matches_the_spines_own_bin_size():
    """A display gate is `per` of the SPINE's range bins, and the spine's bin is
    `c*tau` at `tau = 1/(N*df)` on the endpoint-inclusive grid (F97d). Written against
    `N*df`, never against a nominal `B`, so the `N/(N-1)` factor cannot be baked in."""
    C = 2.99792458e8
    n_fft, span, bins = 64, 3e9, 8
    df = span / (n_fft - 1)
    m_per_bin = C / (n_fft * df)                    # bistatic: c*tau, not c*tau/2
    rmeta = _range_meta_from_grid(n_fft, span)
    assert rmeta["range_m_per_bin"] == pytest.approx(m_per_bin)
    assert rmeta["range_n_bins"] == n_fft // 2 + 1  # the non-negative half, kept once
    _, gate = _display_range_axis(bins, rmeta)
    per = -(-rmeta["range_n_bins"] // bins)
    assert gate == pytest.approx(per * m_per_bin)


def test_the_window_is_the_full_fft_period_and_the_display_is_its_kept_half():
    """F96/F97d: the window is N bins of the transform (not N/2), and what the panel
    shows is the half the spine keeps -- so the two numbers on the card are a window
    and a display range, not a limit and a ceiling."""
    rmeta = _range_meta_from_grid(64, 3e9)
    assert rmeta["range_window_m"] == pytest.approx(64 * rmeta["range_m_per_bin"])
    assert rmeta["range_displayed_m"] == pytest.approx(
        (rmeta["range_n_bins"] - 1) * rmeta["range_m_per_bin"])


def test_range_az_subline_states_metres_per_gate_and_unambiguous_range():
    torch = pytest.importorskip("torch")

    ra = torch.rand((8, 8)).to(torch.complex64)
    fig = figures_from_outputs({
        "range_az": [ra], "_axis_meta": _munich_axis_meta(range_az_bins=8),
    })["range_az"]
    text = panel_text(fig)
    rmeta = _range_meta_from_grid(64, 3e9)
    _, expected_gate = _display_range_axis(8, rmeta)
    assert f"{expected_gate:.2f} m/gate" in text
    # F96/F97d: the window is the frame's own FULL FFT period, and the panel shows the
    # non-negative half of it -- in the owner's bistatic excess-path convention.
    # `.1f` on the metres since hostile round 12 item 13: at `.0f` this clause printed
    # "0-250 m shown of a 500 m window", which is F97d's RETRACTED nominal-B pair.
    assert (f"0-{rmeta['range_displayed_m']:.1f} m shown of a "
            f"{rmeta['range_window_m']:.1f} m window, bistatic excess path") in text


def test_range_el_subline_states_metres_per_gate_too():
    torch = pytest.importorskip("torch")

    re_ = torch.rand((8, 8)).to(torch.complex64)
    fig = figures_from_outputs({
        "range_el": [re_], "_axis_meta": _munich_axis_meta(range_el_bins=8),
    })["range_el"]
    text = panel_text(fig)
    _, expected_gate = _display_range_axis(8, _range_meta_from_grid(64, 3e9))
    assert f"{expected_gate:.2f} m/gate" in text


def test_range_az_gate_calibration_absent_without_axis_metadata():
    """No n_freqs/freq_span_hz -> nothing to compute the calibration from; the bin-index
    fallback carries no physical claim (mirrors the "earliest arrival" fallback)."""
    torch = pytest.importorskip("torch")

    ra = torch.rand((8, 8)).to(torch.complex64)
    fig = figures_from_outputs({"range_az": [ra],
                                "_axis_meta": {"range_az_bins": 8}})["range_az"]
    assert "m/gate" not in panel_text(fig)


def test_range_az_yaxis_title_names_the_convention_and_the_half_window():
    """SUPERSEDED, deliberately, 2026-09-25 (hostile round 12, item 13). This test used
    to guard the opposite rule -- "the axis title stays SHORT, the calibration lives in
    the caption" -- and the reason it gave (2026-09-23) was that the rotated title ran
    into the heat map's own FIGURE TITLE at podium font size. That mechanism is gone:
    under the layout spec a figure carries no title at all (the title and caption are
    HTML above the plot), so the only thing above the axis is the 52 px statistic strip.
    Meanwhile the caption is at its one-line budget and the fact that the axis shows the
    NON-NEGATIVE HALF of a 499.6 m period was reachable only inside a closed Details
    disclosure. It goes on the axis, which is the thing it is about.

    What stays true, and is what this test now pins: the title names the convention
    first, and the added clause is computed from the frame's own plan."""
    torch = pytest.importorskip("torch")

    ra = torch.rand((8, 8)).to(torch.complex64)
    fig = figures_from_outputs({
        "range_az": [ra], "_axis_meta": _munich_axis_meta(range_az_bins=8),
    })["range_az"]
    rmeta = _range_meta_from_grid(64, 3e9)
    title = fig.layout.yaxis.title.text
    # "excess path (m), c*tau" since round 13's N7 -- every range axis now names its own
    # convention, so the title starts with the vocabulary's own entry rather than a bare
    # "excess path (m)". Read from the module's table, not retyped.
    from webapp.pipeline_runner import _RANGE_AXIS_LABEL
    assert title.startswith(_RANGE_AXIS_LABEL["bistatic_path"])
    assert f"displayed half of {rmeta['range_window_m']:.1f} m" in title
    # ...and it stays ONE short line: ~42 characters is what a 363 px plot height fits
    # at the 18 px figure font.
    assert len(title) <= 46, title


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
