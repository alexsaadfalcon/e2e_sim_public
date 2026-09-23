"""A/B demo presets (Change 1), the headline dynamic-range/floor annotations (Change
2), the subspace-error floor + reference line (Change 3), and the podium-distance
legibility floor -- all from the 2026-09-22/23 hostile-expert and legibility reviews.

Three of seven demo cards (Thrust 1, 2, 4) claimed a before/after number that no single
screen ever showed. These tests pin: (a) an `ab` preset runs the pipeline twice, the
override applied ONLY to run B; (b) the Results banner names both arms' values; (c) a
before/after pair shares one heatmap colour scale and axis extent, and one y-range
floor for line plots; (d) the range-azimuth/range-profile panels print the exact
statistic notes/tools/demo_thrust1_rescue.py::q defines; (e) the subspace-error plot
has a minimum y-axis bound and a labelled settled-level reference line; (f) every
figure meets the podium-distance font floor.
"""

from __future__ import annotations

import re

import numpy as np
import plotly.graph_objects as go
import pytest

from webapp.demo_presets import PRESETS, PRESETS_BY_ID, apply_preset
from webapp.pipeline_registry import BLOCKS_BY_ID


# ------------------------------------------------------------------------------------
# Change 1: A/B presets run the pipeline twice, override applied only to B
# ------------------------------------------------------------------------------------
@pytest.mark.parametrize("pid", ["thrust1_circuit_knobs", "thrust2_feature_reduction_error",
                                 "thrust4_interconnect_range_profile"])
def test_ab_is_wired_on_the_three_presets_the_review_named(pid):
    p = PRESETS_BY_ID[pid]
    assert p.ab is not None and p.ab_label_a and p.ab_label_b
    bid, key, _value_b = p.ab
    assert key in {ps.key for ps in BLOCKS_BY_ID[bid].params}
    # Run A (as loaded) must NOT already equal run B -- otherwise there is nothing to
    # compare.
    state_a, state_b = apply_preset(p), apply_preset(p, arm="b")
    assert state_a[bid]["params"][key] != state_b[bid]["params"][key]


def test_presets_without_ab_are_unaffected():
    for p in PRESETS:
        if p.id in ("thrust1_circuit_knobs", "thrust2_feature_reduction_error",
                    "thrust4_interconnect_range_profile"):
            continue
        assert p.ab is None


def _fake_runner(monkeypatch, calls, fig_key="range_az", extra_axis_meta=None):
    import webapp.app as appmod

    def fake_run_pipeline(state, n_steps, should_stop=None):
        calls.append(state)
        meta = {"source": "x", "n_steps_run": n_steps, "cancelled": False}
        meta.update(extra_axis_meta or {})
        return {fig_key: [], "_axis_meta": meta}

    monkeypatch.setattr(appmod, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(appmod, "figures_from_outputs", lambda outputs: {fig_key: go.Figure()})
    return appmod


def test_ab_preset_runs_pipeline_twice_override_only_on_b(monkeypatch):
    preset = PRESETS_BY_ID["thrust1_circuit_knobs"]
    state_a = apply_preset(preset)
    calls = []
    appmod = _fake_runner(monkeypatch, calls)

    data, status, tab, _sink = appmod._run_pipeline(1, state_a, preset.n_steps, "", None)

    assert len(calls) == 2, "an ab preset must run the pipeline exactly twice"
    bid, key, value_b = preset.ab
    assert calls[0][bid]["params"][key] == state_a[bid]["params"][key], \
        "run A must be exactly the preset as loaded"
    assert calls[1][bid]["params"][key] == value_b, "run B must carry ONLY the override"
    # Every other block's params/enabled must be identical between the two calls --
    # the override touches nothing else.
    for other_bid in calls[0]:
        if other_bid == bid:
            continue
        assert calls[0][other_bid] == calls[1][other_bid], other_bid
    assert tab == "tab-results"
    assert "_previous" in data


def test_ab_banner_names_both_arms_values(monkeypatch):
    preset = PRESETS_BY_ID["thrust1_circuit_knobs"]
    state_a = apply_preset(preset)
    calls = []
    appmod = _fake_runner(monkeypatch, calls)

    data, *_ = appmod._run_pipeline(1, state_a, preset.n_steps, "", None)

    assert preset.ab_label_a in data["_banner"] and preset.ab_label_b in data["_banner"]
    assert preset.ab_label_a in data["_previous"]["_banner"]
    assert preset.ab_label_b in data["_previous"]["_banner"]
    # Names the knob by its editor label, not the raw param key.
    label = next(ps.label for ps in BLOCKS_BY_ID["rffe"].params if ps.key == "lna_bias_ma")
    assert label in data["_banner"]


def test_ab_preset_skips_run_b_when_a_is_cancelled(monkeypatch):
    preset = PRESETS_BY_ID["thrust1_circuit_knobs"]
    state_a = apply_preset(preset)
    calls = []
    appmod = _fake_runner(monkeypatch, calls, extra_axis_meta={"cancelled": True, "n_steps_run": 1})

    data, status, tab, _sink = appmod._run_pipeline(1, state_a, preset.n_steps, "", None)

    assert len(calls) == 1, "cancelling run A must stop the whole A/B click, not start B"
    assert "did not run" in data["_banner"]
    assert "_previous" not in data


def test_manual_edit_after_loading_ab_preset_falls_back_to_single_run(monkeypatch):
    """The 'turn one knob and run again' path must keep working: once the operator
    hand-edits a param, block_state no longer matches the preset's as-loaded state, and
    Run must revert to the ordinary single-run behaviour (existing before/after via
    `_previous`, not a second automatic run)."""
    import webapp.app as appmod

    preset = PRESETS_BY_ID["thrust1_circuit_knobs"]
    state = apply_preset(preset)
    state = appmod._with_param(state, "rffe", "lna_bias_ma", 4.0)  # a THIRD value
    calls = []
    _fake_runner(monkeypatch, calls)

    data, status, tab, _sink = appmod._run_pipeline(1, state, preset.n_steps, "", None)

    assert len(calls) == 1
    assert "_previous" not in data
    assert "A/B" not in data["_banner"]


def test_matching_ab_preset_none_for_presets_without_ab():
    import webapp.app as appmod

    for pid in ("thrust3_cold_start_acquisition", "thrust5_detector_cfar",
               "thrust5_detector_ml", "thrust5_detector_raddetnet"):
        st = apply_preset(PRESETS_BY_ID[pid])
        assert appmod._matching_ab_preset(st) is None


def test_single_run_path_unchanged_for_non_ab_preset(monkeypatch):
    """Loading/running a preset with no `ab` behaves exactly as before this change:
    one run_pipeline call, no automatic pairing."""
    preset = PRESETS_BY_ID["thrust3_cold_start_acquisition"]
    state = apply_preset(preset)
    calls = []
    appmod = _fake_runner(monkeypatch, calls, fig_key="subspace_err")

    data, status, tab, _sink = appmod._run_pipeline(1, state, preset.n_steps, "", None)

    assert len(calls) == 1
    assert "_previous" not in data
    assert "A/B" not in data["_banner"]


# ------------------------------------------------------------------------------------
# Change 1: shared axes/colour scale (heatmaps) and shared floor (scatter) between an
# A/B pair, via the existing before/after render-time mechanism (_share_y_ranges).
# ------------------------------------------------------------------------------------
def test_share_axes_unifies_heatmap_extent_and_colour_scale():
    import webapp.app as appmod

    cur = {"range_az": {"data": [{"type": "heatmap", "x": [0, 1], "y": [0, 5],
                                  "zmin": -40, "zmax": 0}], "layout": {}}}
    prev = {"range_az": {"data": [{"type": "heatmap", "x": [0, 2], "y": [0, 3],
                                   "zmin": -30, "zmax": 0}], "layout": {}}}
    appmod._share_y_ranges(cur, prev)

    assert cur["range_az"]["layout"]["xaxis"]["range"] == [0, 2]
    assert cur["range_az"]["layout"]["xaxis"]["range"] == prev["range_az"]["layout"]["xaxis"]["range"]
    assert cur["range_az"]["layout"]["yaxis"]["range"] == [0, 5]
    assert cur["range_az"]["layout"]["yaxis"]["range"] == prev["range_az"]["layout"]["yaxis"]["range"]
    assert cur["range_az"]["data"][0]["zmin"] == prev["range_az"]["data"][0]["zmin"] == -40
    assert cur["range_az"]["data"][0]["zmax"] == prev["range_az"]["data"][0]["zmax"] == 0


def test_share_axes_survives_plotlys_compact_typed_array_encoding():
    """Regression (found live in the 2026-09-22 rehearsal): every range_az/range_el/
    radar_cube heatmap's x/y is a numpy array, and `go.Figure.to_dict()` (plotly
    5.20+) encodes large numpy arrays as `{"dtype": ..., "bdata": <base64>}` instead
    of a plain list -- `_share_y_ranges` crashed reading 'dtype' as a coordinate."""
    import webapp.app as appmod

    fig1 = go.Figure(data=go.Heatmap(z=np.zeros((3, 3)), x=np.arange(3),
                                     y=np.arange(3), zmin=-40, zmax=0)).to_dict()
    fig2 = go.Figure(data=go.Heatmap(z=np.zeros((3, 3)), x=np.arange(3) + 1,
                                     y=np.arange(3) + 2, zmin=-30, zmax=0)).to_dict()
    assert isinstance(fig1["data"][0]["x"], dict) and "bdata" in fig1["data"][0]["x"], \
        "test assumption: plotly used the compact encoding here"
    figs, prev = {"range_az": fig1}, {"range_az": fig2}
    appmod._share_y_ranges(figs, prev)  # must not raise
    assert figs["range_az"]["layout"]["xaxis"]["range"] == [0, 3]
    assert figs["range_az"]["layout"]["yaxis"]["range"] == [0, 4]
    assert figs["range_az"]["data"][0]["zmin"] == prev["range_az"]["data"][0]["zmin"] == -40


def test_share_axes_scatter_survives_numpy_array_y_values():
    """Regression: range_profile's Scatter y is a numpy array (10*np.log10(...)), not
    a plain list like subspace_err's -- the same compact-encoding crash as the heatmap
    case above, in the scatter branch."""
    import webapp.app as appmod

    y = go.Figure(data=go.Scatter(x=np.arange(20), y=np.linspace(-40, 0, 20))).to_dict()
    assert isinstance(y["data"][0]["y"], dict) and "bdata" in y["data"][0]["y"]
    prev_y = go.Figure(data=go.Scatter(x=np.arange(20), y=np.linspace(-30, 0, 20))).to_dict()
    figs, prev = {"range_profile": y}, {"range_profile": prev_y}
    appmod._share_y_ranges(figs, prev)  # must not raise


def test_share_axes_does_not_clobber_a_fixed_negative_db_scatter_range():
    """Regression (found live in the 2026-09-23 rehearsal of the T4 A/B pair):
    range_profile always sets a fixed [-60, 2] dB display range; the scatter-sharing
    logic (written for subspace_err's non-negative, zero-anchored curve) blindly
    recomputed [0, top] from the data and blanked both panels."""
    import webapp.app as appmod

    cur = {"range_profile": {"data": [{"type": "scatter", "y": [-2.0, -34.0, -1.0]}],
                             "layout": {"yaxis": {"range": [-60.0, 2.0]}}}}
    prev = {"range_profile": {"data": [{"type": "scatter", "y": [-2.0, -49.0, -1.0]}],
                              "layout": {"yaxis": {"range": [-60.0, 2.0]}}}}
    appmod._share_y_ranges(cur, prev)
    assert cur["range_profile"]["layout"]["yaxis"]["range"] == [-60.0, 2.0]
    assert prev["range_profile"]["layout"]["yaxis"]["range"] == [-60.0, 2.0]


def test_share_axes_scatter_floor_is_not_re_tightened_below_an_existing_bound():
    """The subspace-error plot sets its own minimum y upper bound (Change 3); pairing
    it with a smaller run must not shrink the shared range back below that floor."""
    import webapp.app as appmod

    cur = {"subspace_err": {"data": [{"type": "scatter", "y": [0.04, 0.06]}],
                            "layout": {"yaxis": {"range": [0.0, 0.65]}}}}
    prev = {"subspace_err": {"data": [{"type": "scatter", "y": [0.04, 0.05]}],
                             "layout": {"yaxis": {"range": [0.0, 0.65]}}}}
    appmod._share_y_ranges(cur, prev)
    assert cur["subspace_err"]["layout"]["yaxis"]["range"][1] == 0.65
    assert prev["subspace_err"]["layout"]["yaxis"]["range"][1] == 0.65


def test_share_axes_preexisting_scatter_behaviour_is_unchanged():
    """Pin the original test_webapp_rehearsal.py behaviour (no pre-set floor)."""
    import webapp.app as appmod

    cur = {"subspace_err": {"data": [{"type": "scatter", "y": [0.6, 0.63]}], "layout": {}}}
    prev = {"subspace_err": {"data": [{"type": "scatter", "y": [0.04, 0.06]}], "layout": {}}}
    appmod._share_y_ranges(cur, prev)
    assert cur["subspace_err"]["layout"]["yaxis"]["range"] == prev["subspace_err"]["layout"]["yaxis"]["range"]
    assert cur["subspace_err"]["layout"]["yaxis"]["range"][1] == pytest.approx(0.63 * 1.05)


# ------------------------------------------------------------------------------------
# Change 2: the headline dynamic-range / floor statistic, on screen
# ------------------------------------------------------------------------------------
def test_peak_minus_median_matches_direct_computation_to_1e_minus_6():
    """Reuse notes/tools/demo_thrust1_rescue.py::q's definition exactly: peak (0 dB
    after peak-normalization) minus the median of the whole map."""
    from webapp.pipeline_runner import _peak_minus_median_db

    rng = np.random.default_rng(4)
    power = rng.random((32, 20)).astype(np.float64)
    db = 10 * np.log10(np.maximum(power / power.max(), 1e-12))
    expected = float(db.max() - np.median(db))
    assert abs(_peak_minus_median_db(db) - expected) < 1e-6


def test_range_az_panel_carries_the_peak_minus_median_statistic():
    """The stat lives in the title (a "<br><sup>" subline, like the detector panel),
    not a floating annotation -- an annotation collided with the title text at the
    podium-distance font size (rehearsal, 2026-09-23)."""
    torch = pytest.importorskip("torch")
    from webapp.pipeline_runner import figures_from_outputs

    rng = np.random.default_rng(6)
    power = rng.random((16, 16)).astype(np.float32)
    ra = torch.from_numpy(power).to(torch.complex64)
    fig = figures_from_outputs({"range_az": [ra], "_axis_meta": {"range_az_bins": 16}})["range_az"]

    title = fig.layout.title.text
    assert "peak - median" in title
    measured = float(re.search(r"(-?\d+\.\d+)\s*(?:</sup>)?\s*$", title).group(1))
    db = 10 * np.log10(np.maximum(power / power.max(), 1e-12))
    expected = round(float(db.max() - np.median(db)), 1)
    assert measured == pytest.approx(expected, abs=0.05)
    assert "dB" in title


def test_range_el_panel_carries_no_peak_minus_median_statistic():
    """The statistic is a Thrust-1/2/4 range-AZIMUTH claim; range-elevation gets none."""
    torch = pytest.importorskip("torch")
    from webapp.pipeline_runner import figures_from_outputs

    ra = torch.rand((12, 12)).to(torch.complex64)
    fig = figures_from_outputs({"range_el": [ra], "_axis_meta": {"range_el_bins": 12}})["range_el"]
    assert "peak - median" not in fig.layout.title.text


def test_range_profile_panel_carries_the_median_floor_statistic():
    from webapp.pipeline_runner import figures_from_outputs

    rng = np.random.default_rng(5)
    prof = rng.random(16)
    fig = figures_from_outputs({"range_profile_agg": [prof],
                                "_axis_meta": {"range_profile_bins": 16}})["range_profile"]
    title = fig.layout.title.text
    assert "median floor" in title
    measured = float(re.search(r"(-?\d+\.\d+)\s*(?:</sup>)?\s*$", title).group(1))
    peak = max(float(prof.max()), 1e-12)
    prof_db = 10 * np.log10(prof / peak + 1e-12)
    expected = round(float(np.median(prof_db)), 1)
    assert measured == pytest.approx(expected, abs=0.05)


# ------------------------------------------------------------------------------------
# Range-Doppler panel: adaptive display clip (physics review, 2026-09-23) -- the
# shared -40 dB clip sits too close to this corpus's own ambient floor and lights up
# whole Doppler rows as false smear.
# ------------------------------------------------------------------------------------
def test_radar_cube_clip_never_loosens_below_40_and_tightens_for_a_noisy_frame():
    from webapp.pipeline_runner import _radar_cube_clip_db

    quiet = np.full((4, 4), -50.0)
    quiet[0, 0] = 0.0
    assert _radar_cube_clip_db(quiet) == -40.0  # quiet frame keeps the shared clip

    noisy = np.full((4, 4), -38.0)
    noisy[0, 0] = 0.0
    assert _radar_cube_clip_db(noisy) == pytest.approx(-35.0)  # median(-38) + 3


def test_radar_cube_panel_clip_matches_its_own_colorbar_label():
    torch = pytest.importorskip("torch")
    from webapp.pipeline_runner import _radar_cube_clip_db, figures_from_outputs

    rng = np.random.default_rng(7)
    cube = (rng.random((4, 8, 6)) + 1j * rng.random((4, 8, 6))).astype(np.complex64)
    cube[:, 3, 2] *= 20  # one bright cell so the map has real dynamic range
    cube_t = torch.from_numpy(cube)
    fig = figures_from_outputs({"radar_cube": [cube_t], "_axis_meta": {}})["radar_cube"]

    p = np.mean(np.abs(cube) ** 2, axis=0)
    db = 10 * np.log10(p / p.max() + 1e-12)
    expected_clip = _radar_cube_clip_db(db)

    assert fig.data[0].zmin == pytest.approx(expected_clip)
    assert f"{expected_clip:g}" in fig.data[0].colorbar.title.text


# ------------------------------------------------------------------------------------
# Change 3: subspace-error y-axis floor/bound + labelled settled-level reference line
# ------------------------------------------------------------------------------------
def test_subspace_err_has_a_minimum_upper_bound_and_settled_reference_line():
    from webapp.pipeline_runner import (_SUBSPACE_ERR_MIN_YMAX, _SUBSPACE_ERR_SETTLED_LEVEL,
                                        figures_from_outputs)

    fig = figures_from_outputs({"subspace_err": [0.04, 0.05, 0.06]})["subspace_err"]
    assert fig.layout.yaxis.range[0] == 0.0
    assert fig.layout.yaxis.range[1] >= _SUBSPACE_ERR_MIN_YMAX
    assert "Frobenius" in fig.layout.yaxis.title.text and "unnormalised" in fig.layout.yaxis.title.text
    shapes = fig.layout.shapes or ()
    assert any(abs(float(s.y0) - _SUBSPACE_ERR_SETTLED_LEVEL) < 1e-9 and s.line.dash == "dash"
              for s in shapes), "expected a dashed reference line at the settled level"


def test_subspace_err_bound_grows_for_a_curve_above_the_floor():
    from webapp.pipeline_runner import figures_from_outputs

    fig = figures_from_outputs({"subspace_err": [0.5, 0.6, 0.63]})["subspace_err"]
    assert fig.layout.yaxis.range[1] >= 0.63 * 1.05 - 1e-9


# ------------------------------------------------------------------------------------
# Legibility floor (podium-distance review): every figure figures_from_outputs
# returns meets the font floor; detection markers are enlarged.
# ------------------------------------------------------------------------------------
def test_every_returned_figure_meets_the_legibility_floor(torch_device):
    import torch
    from webapp.pipeline_runner import figures_from_outputs

    bins, n_freqs, span = 16, 64, 3e9
    ra = torch.rand((bins, bins), device=torch_device).to(torch.complex64)
    prof = np.random.default_rng(2).random(bins)
    obj = np.zeros((2, 8, 16), dtype=np.float32)
    comm_data = torch.rand(8, device=torch_device).to(torch.complex64)
    outputs = {
        "range_az": [ra],
        "range_profile_agg": [prof],
        "subspace_err": [0.5, 0.3, 0.1],
        "cfar_detection": [obj],
        "ber": [0.1, 0.01, 0.0],
        "evm": [0.2, 0.1],
        "comm_data_eq": [comm_data],
        "_axis_meta": {"range_az_bins": bins, "range_profile_bins": bins,
                       "n_freqs": n_freqs, "freq_span_hz": span,
                       "detector": {"mode": "cfar", "threshold": 0.5, "label": "x"},
                       "rx": {"grid": {"max_range_m": 40.0}}},
    }
    figs = figures_from_outputs(outputs)
    assert len(figs) >= 6
    for key, fig in figs.items():
        size = fig.layout.font.size if fig.layout.font is not None else None
        assert size is not None and size >= 16, f"{key}: base font {size} below the 16px floor"
        assert fig.layout.xaxis.tickfont.size >= 15, key
        assert fig.layout.yaxis.tickfont.size >= 15, key


# ------------------------------------------------------------------------------------
# Bug hunt (2026-09-23): per-session Cancel, the double-click guard, and a precise
# out-of-range Frames-to-run message.
# ------------------------------------------------------------------------------------
def test_share_axes_stage_blocker_repro_300_point_numpy_y_vs_plain_list():
    """The exact reproduction described in the bug report: a 300-point numpy Range
    Profile y (typed-array encoded) as "this run" against a plain-list previous run."""
    import webapp.app as appmod

    cur = go.Figure(data=go.Scatter(x=np.arange(300), y=np.linspace(-50, 0, 300))).to_dict()
    assert isinstance(cur["data"][0]["y"], dict) and "bdata" in cur["data"][0]["y"]
    prev = go.Figure(data=go.Scatter(x=list(range(300)),
                                     y=list(np.linspace(-45, 0, 300)))).to_dict()
    assert isinstance(prev["data"][0]["y"], list)
    figs, prevs = {"range_profile": cur}, {"range_profile": prev}
    appmod._share_y_ranges(figs, prevs)  # must not raise


def test_cancel_is_scoped_per_session():
    import webapp.app as appmod

    ev_a = appmod._cancel_event("session-a")
    ev_b = appmod._cancel_event("session-b")
    assert ev_a is not ev_b
    ev_a.set()
    assert appmod._cancel_event("session-a").is_set() is True
    assert appmod._cancel_event("session-b").is_set() is False


def test_run_lock_blocks_a_second_concurrent_run_for_the_same_session(monkeypatch):
    """A rapid double-click (or two clicks landing before the button's client-side
    `disabled` state takes effect) must not dispatch two overlapping pipeline runs."""
    import webapp.app as appmod

    session = "dup-click-session"
    calls = []

    def fake_run_pipeline(state, n_steps, should_stop=None):
        # Simulate the SECOND click arriving while this call is still "in flight":
        # the lock is already held for `session` at this point.
        assert not appmod._run_lock(session).acquire(blocking=False), \
            "the lock must already be held while a run for this session is in flight"
        calls.append(state)
        return {"range_az": [], "_axis_meta": {"source": "x", "n_steps_run": n_steps,
                                                "cancelled": False}}

    monkeypatch.setattr(appmod, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(appmod, "figures_from_outputs", lambda outputs: {"range_az": go.Figure()})
    state = apply_preset(PRESETS_BY_ID["thrust3_cold_start_acquisition"])

    data, status, tab, _sink = appmod._run_pipeline(1, state, 3, "", None, None, session)
    assert len(calls) == 1
    assert "_previous" not in data


def test_run_lock_released_after_a_run_so_the_next_click_is_not_blocked(monkeypatch):
    import webapp.app as appmod

    session = "sequential-session"
    monkeypatch.setattr(appmod, "run_pipeline", lambda state, n_steps, should_stop=None: {
        "range_az": [], "_axis_meta": {"source": "x", "n_steps_run": n_steps, "cancelled": False}})
    monkeypatch.setattr(appmod, "figures_from_outputs", lambda outputs: {"range_az": go.Figure()})
    state = apply_preset(PRESETS_BY_ID["thrust3_cold_start_acquisition"])

    appmod._run_pipeline(1, state, 3, "", None, None, session)
    # A second, SEQUENTIAL run for the same session must proceed normally.
    data, status, tab, _sink = appmod._run_pipeline(2, state, 3, "", None, None, session)
    assert tab == "tab-results"
    assert "already in progress" not in str(status)


def test_bad_frame_count_message_distinguishes_blank_from_out_of_range(monkeypatch):
    import webapp.app as appmod

    calls = []
    monkeypatch.setattr(appmod, "run_pipeline", lambda *a, **k: calls.append(1) or {})

    _data, status, _tab, _sink = appmod._run_pipeline(1, None, None, "", None, "", None)
    assert "blank" in str(status)

    _data, status, _tab, _sink = appmod._run_pipeline(1, None, None, "", None, "51", None)
    assert "51" in str(status) and "exceeds the maximum" in str(status)

    _data, status, _tab, _sink = appmod._run_pipeline(1, None, None, "", None, "0", None)
    assert "0" in str(status) and "below the minimum" in str(status)
    assert not calls


def test_detection_markers_are_enlarged_for_podium_distance():
    from webapp.pipeline_runner import figures_from_outputs

    obj = np.zeros((2, 8, 16), dtype=np.float32)
    outputs = {
        "cfar_detection": [obj],
        "cfar_detections": [[(0, 0.1, 0.9, 12.0)]],
        "gt_detections": [[(0, -0.2, 1.0, 20.0)]],
        "_axis_meta": {"rx": {"grid": {"max_range_m": 40.0}},
                       "detector": {"mode": "cfar", "threshold": 0.66, "label": "x"}},
    }
    fig = figures_from_outputs(outputs)["cfar_detection"]
    det_trace = next(t for t in fig.data if (t.name or "").startswith("detections"))
    gt_trace = next(t for t in fig.data if (t.name or "").startswith("ground truth"))
    assert det_trace.marker.size == 14
    assert gt_trace.marker.size == 18
