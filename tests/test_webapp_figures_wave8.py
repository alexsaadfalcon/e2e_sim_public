"""Wave 8 (final Ka screens, 2026-09-23) fixes owned by the pipeline_runner shard:

W2: no munich range-azimuth / range-elevation / range-profile panel stated that its own
0 dB reference, at range 0, is the delay-normalised direct-path/leakage band -- not a
target. Added to each panel's subline (kept short: "0 dB = direct path ..., not a
target"); the range-azimuth/range-elevation wording also keeps the exact
"0 = earliest arrival" substring that tests/test_webapp_figures_wave3.py (an unowned
file) already pins in the same subline.

W13: the gate-calibration clause (wave 7, X6/X7) now also states the frame's own
NATIVE (un-binned) range resolution and the display-bin/native ratio -- computed from
the freq_plan (native = c / (2*B); ratio = gate / native), never hand-typed -- while
keeping the "m/gate" substring this module's own wave-7 tests (and webapp/
demo_presets.py's screen notes, an unowned file, via test_demo_presets.py) pin.

F96 (wave 11, notes/ESTABLISHED_FACTS.md): the old "unambig N m" wording read as a
physical range ceiling; it is HALF of the frame's own N-point FFT period, the other
half cropped as negative delay. The clause now reads "display 0-N m of 2N m unambig
(neg.-delay half cropped)", both numbers computed from the frame's own freq_plan
(N*c/(2B) and its half), never typed -- this test file's own pinned substring is
updated to match; tests/test_webapp_figures_wave7.py (unowned) still pins the retired
"unambig N m" wording and needs the same update from its owner.

W3: the Thrust 3 subspace-error panel's right-hand "refinement passes/frame" axis used
to autoscale independently per run (`rangemode="tozero"` only anchors zero); a 5-pass
fixed-effort arm and a 10-pass refine-gate arm rendered at the SAME pixel height. The
axis now gets an explicit, formula-derived range shared in shape across runs (0 to the
larger of 10 and this run's own max + 1), so a real 2x difference in compute reads as a
height difference.

W15 (T3): the left subspace-error axis leaves headroom (relative + a fixed absolute
cushion) above the curve's own max, so a cold-start arm's frame-1 point does not sit
visually on the axis ceiling; the "warm-start settled level" reference-line annotation
moves from its default "top left" to "top right" whenever an early frame's own error
sits close enough to the settled level to put a data marker under the annotation text.

LAYOUT SPEC, 2026-09-24: figures no longer carry a title. Every W2/W13/F96 clause this
file used to pin on `fig.layout.title.text` now lives in the panel meta reached through
`panel_text(fig)` (title + caption + the whole Details body -- see
`webapp/pipeline_runner.py`'s "PANEL GEOMETRY AND THE PANEL-META CONTRACT" section).
Repointed below, not dropped: these clauses are the same honesty content the layout
spec's own acceptance check 15 requires to still be reachable. The `add_hline`
"settled level" annotation (W15) is untouched -- it was never part of the removed
title/subtitle machinery, it is a real plot annotation now and before.
"""
from __future__ import annotations

import numpy as np
import pytest

from webapp.pipeline_runner import (
    _C, _SUBSPACE_ERR_MIN_YMAX, _SUBSPACE_ERR_SETTLED_LEVEL, _REFINE_AXIS_MIN_YMAX,
    _display_range_axis, _range_meta_from_grid,
    figures_from_outputs, panel_text,
)


def _munich_axis_meta(**bins):
    return {"n_freqs": 64, "freq_span_hz": 3e9, **bins}


# --------------------------------------------------------------------------------
# W2: 0 dB at range 0 is stated as the direct-path band, not a target
# --------------------------------------------------------------------------------
def test_range_az_subline_states_direct_path_and_keeps_earliest_arrival_pin():
    torch = pytest.importorskip("torch")

    ra = torch.rand((8, 8)).to(torch.complex64)
    fig = figures_from_outputs({
        "range_az": [ra], "_axis_meta": _munich_axis_meta(range_az_bins=8),
    })["range_az"]
    text = panel_text(fig)
    assert "0 dB = direct path" in text
    assert "not a target" in text
    # The exact substring tests/test_webapp_figures_wave3.py (unowned) pins.
    assert "0 = earliest arrival" in text


def test_range_el_subline_states_direct_path_too():
    torch = pytest.importorskip("torch")

    re_ = torch.rand((8, 8)).to(torch.complex64)
    fig = figures_from_outputs({
        "range_el": [re_], "_axis_meta": _munich_axis_meta(range_el_bins=8),
    })["range_el"]
    text = panel_text(fig)
    assert "0 dB = direct path" in text
    assert "not a target" in text
    assert "0 = earliest arrival" in text


def test_range_profile_subline_states_direct_path():
    torch = pytest.importorskip("torch")

    prof = torch.rand(8, dtype=torch.float32)
    fig = figures_from_outputs({
        "range_profile_agg": [prof],
        "_axis_meta": _munich_axis_meta(range_profile_bins=8),
    })["range_profile"]
    text = panel_text(fig)
    assert "0 dB = direct path at range 0, not a target" in text
    # The xlabel keeps its own, separately-pinned wording (test_webapp_figures_wave3.py).
    assert fig.layout.xaxis.title.text == "excess path (m; 0 = earliest arrival)"


def test_range_profile_no_direct_path_note_without_axis_metadata():
    """No n_freqs/freq_span_hz -> nothing to state a physical claim about (mirrors the
    range-az/range-el gate-calibration and earliest-arrival fallbacks).

    HANDOFF (2026-09-24): the comment this docstring used to carry named
    tests/test_webapp_ab.py::test_range_profile_panel_carries_the_median_floor_statistic
    (unowned) as a test this one "also guards", because that test's regex read the
    floor number off the end of `fig.layout.title.text`. That title no longer exists
    (layout spec, this section) -- the statistic is now `_stat_annotations`'
    `stat_strip` annotation -- so that unowned test now fails outright on this branch
    and needs its own repoint to the new mechanism; not fixed here, out of this
    shard's owned files."""
    torch = pytest.importorskip("torch")

    prof = torch.rand(8, dtype=torch.float32)
    fig = figures_from_outputs({
        "range_profile_agg": [prof], "_axis_meta": {"range_profile_bins": 8},
    })["range_profile"]
    assert "direct path" not in panel_text(fig)


# --------------------------------------------------------------------------------
# W13: gate clause states native resolution + ratio, computed from the freq_plan
# --------------------------------------------------------------------------------
def test_range_az_subline_states_native_resolution_and_ratio():
    torch = pytest.importorskip("torch")

    ra = torch.rand((8, 8)).to(torch.complex64)
    fig = figures_from_outputs({
        "range_az": [ra], "_axis_meta": _munich_axis_meta(range_az_bins=8),
    })["range_az"]
    text = panel_text(fig)

    rmeta = _range_meta_from_grid(64, 3e9)
    _, gate = _display_range_axis(8, rmeta)
    native = float(rmeta["range_m_per_bin"])
    ratio = gate / native

    # "m/gate" substring pinned by this module's own wave-7 tests (and, in webapp/
    # demo_presets.py's screen notes, by test_demo_presets.py -- an unowned file).
    assert f"{gate:.2f} m/gate" in text
    # Wave-8 content: native resolution in cm and the display/native ratio. TWO decimals
    # since hostile round 12 item 13: the native bin is 9.99 cm on the shipped munich
    # grid and one decimal rounds it to "10.0 cm" -- the nominal-B number F97d retracts.
    assert f"{native * 100:.2f} cm native" in text
    assert f"{ratio:.0f}:1" in text
    # F96/F97d: the window is the frame's own FULL FFT period; the panel shows the
    # non-negative half the spine keeps, in the bistatic excess-path convention.
    assert (f"0-{rmeta['range_displayed_m']:.1f} m shown of a "
            f"{rmeta['range_window_m']:.1f} m window, bistatic excess path") in text


def test_the_native_bin_is_c_tau_on_the_endpoint_inclusive_grid():
    """The old `_native_range_resolution_m` was `c/(2B)` on a NOMINAL B. Both halves of
    that are retracted: the convention is bistatic (c*tau, ballot 2B) and the grid is
    endpoint-inclusive (F97d, `B/(N-1)`)."""
    rmeta = _range_meta_from_grid(64, 3e9)
    assert rmeta["range_m_per_bin"] == pytest.approx(_C / (64 * (3e9 / 63)))


# --------------------------------------------------------------------------------
# W3: refinement-passes right axis is pinned to a common range across arms
# --------------------------------------------------------------------------------
def _subspace_outputs(errs, n_refine_used):
    return {"subspace_err": errs, "n_refine_used": n_refine_used}


def test_refine_axis_shares_a_common_range_regardless_of_this_arms_own_max():
    """Arm A (fixed effort, max 5) and arm B (refine gate, at its own ceiling of 10)
    must render the SAME right-axis ceiling -- otherwise a real 2x difference in
    compute sits at the same pixel height on both screens (wave 8, W3). Both are
    within `_REFINE_AXIS_MIN_YMAX`, the shared floor both arms of the shipped Thrust 3
    A/B actually stay under, which is what makes them equal without either figure
    knowing the other's data."""
    fig_a = figures_from_outputs(_subspace_outputs(
        [0.5, 0.4, 0.3], [5, 5, 5]))["subspace_err"]
    fig_b = figures_from_outputs(_subspace_outputs(
        [0.5, 0.2, 0.06], [10, 10, 1]))["subspace_err"]
    assert (fig_a.layout.yaxis2.range[1] == fig_b.layout.yaxis2.range[1]
            == pytest.approx(1.2 * _REFINE_AXIS_MIN_YMAX))
    assert fig_a.layout.yaxis2.range[0] == 0


def test_refine_axis_grows_past_the_floor_for_a_high_pass_count():
    """A run whose own data exceeds the shared floor still gets a visible (not
    clipped) axis -- this necessarily stops matching an arm that stayed under the
    floor, since this function only ever sees one run's own data."""
    fig = figures_from_outputs(_subspace_outputs(
        [0.5, 0.4], [12, 15]))["subspace_err"]
    assert fig.layout.yaxis2.range[1] == pytest.approx(1.2 * 16)
    assert fig.layout.yaxis2.range[1] > _REFINE_AXIS_MIN_YMAX


# --------------------------------------------------------------------------------
# W15: left-axis headroom + annotation collision avoidance
# --------------------------------------------------------------------------------
def test_subspace_err_yaxis_leaves_headroom_above_a_cold_start_frame_one():
    """A cold-start arm whose frame-1 point already sits near the minimum ceiling must
    not have that point land ON `top` -- some margin is always left above the data."""
    fig = figures_from_outputs({"subspace_err": [0.6, 0.3, 0.06]})["subspace_err"]
    top = fig.layout.yaxis.range[1]
    assert top > 0.6
    assert top - 0.6 >= 0.03


def test_settled_level_annotation_moves_when_an_early_frame_collides():
    """An early frame sitting within the collision band of the settled-level line
    (wave 8, W15: "T3-B frame-2 marker inside the annotation") moves the annotation to
    the right end of the line; a run with no early frame near that level keeps the
    original top-left placement."""
    collide = figures_from_outputs(
        {"subspace_err": [0.5, _SUBSPACE_ERR_SETTLED_LEVEL + 0.01, 0.3]})["subspace_err"]
    clear = figures_from_outputs(
        {"subspace_err": [0.5, 0.4, 0.3]})["subspace_err"]

    def _hline_annotation(fig):
        for ann in fig.layout.annotations:
            if "settled level" in (ann.text or ""):
                return ann
        raise AssertionError("no settled-level annotation found")

    assert _hline_annotation(collide).xanchor != _hline_annotation(clear).xanchor
