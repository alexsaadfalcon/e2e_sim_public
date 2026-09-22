"""Regressions from the 2026-09-22 browser rehearsal of the demo presets.

Each test pins a defect that unit tests on block state and figure dicts could not
see, because it lived between the rendered page and the operator: an off-step number
input that wiped a preset value to null, a results page with no provenance, a legend
swatch invisible on its own background, a block diagram lighting two sources at once.
"""

from __future__ import annotations

import numpy as np
import pytest

from webapp.demo_presets import PRESETS, PRESETS_BY_ID, apply_preset
from webapp.pipeline_registry import BLOCKS_BY_ID


# ------------------------------------------------------------------------------------
# Preset values must sit on their input's step grid, or the browser reports null
# ------------------------------------------------------------------------------------
@pytest.mark.parametrize("preset", PRESETS, ids=[p.id for p in PRESETS])
def test_every_numeric_override_is_on_its_step_grid(preset):
    """An HTML number input with step=s flags a value off the grid (base = min, else 0)
    as a stepMismatch and reports null on blur -- even when nothing was typed. The
    Thrust 1 preset's signal_scaling 1e-7 on a step=1e-6 input, and the RADDetNet
    preset's threshold 0.44 on step=0.05, both silently became the registry default
    at run time. A step of None or "any" accepts anything."""
    for bid, ov in preset.overrides.items():
        for key, val in (ov.get("params") or {}).items():
            spec = next(ps for ps in BLOCKS_BY_ID[bid].params if ps.key == key)
            if spec.kind not in ("number", "int") or spec.step in (None, "any"):
                continue
            base = spec.min if spec.min is not None else 0.0
            steps = (float(val) - float(base)) / float(spec.step)
            assert abs(steps - round(steps)) < 1e-6, (
                f"{preset.id}: {bid}.{key}={val} is off the step={spec.step} grid "
                f"(base {base}); the browser will null it on blur")


def test_signal_scaling_accepts_any_step_and_threshold_is_fine_grained():
    rffe = {ps.key: ps for ps in BLOCKS_BY_ID["rffe"].params}
    det = {ps.key: ps for ps in BLOCKS_BY_ID["detector"].params}
    assert rffe["signal_scaling"].step == "any"
    assert det["threshold"].step <= 0.01


# ------------------------------------------------------------------------------------
# The store never takes a null from an input
# ------------------------------------------------------------------------------------
def test_store_keeps_the_last_valid_value_when_an_input_reports_null():
    from dash import no_update

    import webapp.app as appmod

    state = {"rffe": {"enabled": True, "params": {"signal_scaling": 1e-7}}}
    assert appmod._with_param(state, "rffe", "signal_scaling", None) is no_update
    assert state["rffe"]["params"]["signal_scaling"] == 1e-7
    out = appmod._with_param(state, "rffe", "signal_scaling", 2e-7)
    assert out["rffe"]["params"]["signal_scaling"] == 2e-7


# ------------------------------------------------------------------------------------
# The three Thrust 5 presets sit at their recall-0.5 operating points
# ------------------------------------------------------------------------------------
def test_thrust5_presets_share_the_recall_half_operating_point_convention():
    """Cross counts on the three screens are compared against each other; they are
    the false-alarm comparison only if every detector sits at the same recall
    (`beat_cfar.json` operating_point.score_threshold: 0.661 / 0.222 / 0.440)."""
    thr = {pid: apply_preset(PRESETS_BY_ID[pid])["detector"]["params"]["threshold"]
           for pid in ("thrust5_detector_cfar", "thrust5_detector_ml",
                       "thrust5_detector_raddetnet")}
    assert thr == {"thrust5_detector_cfar": pytest.approx(0.66, abs=0.005),
                   "thrust5_detector_ml": pytest.approx(0.22, abs=0.005),
                   "thrust5_detector_raddetnet": pytest.approx(0.44, abs=0.005)}


@pytest.mark.parametrize("pid", ["thrust5_detector_cfar", "thrust5_detector_ml",
                                 "thrust5_detector_raddetnet"])
def test_corpus_presets_do_not_light_the_subspace_tracker(pid):
    """Corpus replay runs no serial stages; a lit AdaOja block fed by dashed edges
    read as 'precomputed frames through a subspace tracker' on the diagram."""
    state = apply_preset(PRESETS_BY_ID[pid])
    assert state["subspace"]["enabled"] is False


# ------------------------------------------------------------------------------------
# Block diagram: one source lit per run
# ------------------------------------------------------------------------------------
def test_diagram_greys_the_pkl_source_when_corpus_replay_feeds_the_run():
    from webapp import block_diagram

    state = apply_preset(PRESETS_BY_ID["thrust5_detector_cfar"])
    elements = block_diagram.build_elements(state)
    env = next(e for e in elements if e["data"].get("id") == "environment")
    assert "disabled" in env["classes"].split()
    corpus = next(e for e in elements if e["data"].get("id") == "corpus_environment")
    assert "disabled" not in corpus["classes"].split()
    env_edges = [e for e in elements if e["data"].get("source") == "environment"]
    assert env_edges and all("inactive" in e["classes"].split() for e in env_edges)


def test_diagram_lights_the_pkl_source_by_default():
    from webapp import block_diagram
    from webapp.pipeline_registry import default_block_state

    elements = block_diagram.build_elements(default_block_state())
    env = next(e for e in elements if e["data"].get("id") == "environment")
    assert "disabled" not in env["classes"].split()


# ------------------------------------------------------------------------------------
# Results carry provenance and keep the previous run for before/after
# ------------------------------------------------------------------------------------
def test_run_stores_a_banner_and_the_previous_run(monkeypatch):
    import plotly.graph_objects as go

    import webapp.app as appmod

    monkeypatch.setattr(appmod, "run_pipeline", lambda *a, **k: {
        "fft": [], "_axis_meta": {"source": "Sionna frames: munich", "n_steps_run": 2,
                                  "cancelled": False}})
    monkeypatch.setattr(appmod, "figures_from_outputs",
                        lambda outputs: {"fft": go.Figure()})

    first, status, _tab, _sink = appmod._run_pipeline(1, None, 2, "", None)
    assert "Sionna frames: munich" in first["_banner"] and "2 of 2 frames" in first["_banner"]
    assert "_previous" not in first
    assert "1 product(s)" in str(status)  # reserved keys are not counted as products

    second, *_ = appmod._run_pipeline(2, None, 2, "", first)
    assert second["_previous"]["_banner"] == first["_banner"]
    assert set(second["_previous"]) == {"fft", "_banner"}


def test_cancelled_run_banner_says_partial(monkeypatch):
    import plotly.graph_objects as go

    import webapp.app as appmod

    monkeypatch.setattr(appmod, "run_pipeline", lambda *a, **k: {
        "fft": [], "_axis_meta": {"n_steps_run": 3, "cancelled": True}})
    monkeypatch.setattr(appmod, "figures_from_outputs",
                        lambda outputs: {"fft": go.Figure()})
    data, *_ = appmod._run_pipeline(1, None, 20, "", None)
    assert "3 of 20 frames" in data["_banner"] and "CANCELLED" in data["_banner"]


def test_results_tab_renders_banner_current_and_previous():
    import plotly.graph_objects as go

    import webapp.app as appmod

    fig = go.Figure().to_dict()
    data = {"fft": fig, "_banner": "run #2  |  now",
            "_previous": {"fft": fig, "_banner": "run #1  |  earlier"}}
    text = str(appmod._render_results(data, "tab-results"))
    assert "This run: run #2" in text and "Previous run" in text and "run #1" in text
    assert text.count("Graph(") == 2


# ------------------------------------------------------------------------------------
# Detector figure: names its operating point, legend readable
# ------------------------------------------------------------------------------------
def test_detector_figure_title_names_detector_and_threshold_and_legend_is_dark():
    from webapp.pipeline_runner import figures_from_outputs

    obj = np.zeros((3, 8, 16), dtype=np.float32)
    outputs = {
        "cfar_detection": [obj],
        "cfar_detections": [[(0, 0.1, 0.9, 12.0)]],
        "gt_detections": [[(0, -0.2, 1.0, 20.0)]],
        "_axis_meta": {"rx": {"grid": {"max_range_m": 40.0}},
                       "detector": {"mode": "cfar", "threshold": 0.66,
                                    "label": "CA-CFAR (guard 2, train 6)"}},
    }
    fig = figures_from_outputs(outputs)["cfar_detection"]
    title = fig.layout.title.text
    assert "CA-CFAR (guard 2, train 6)" in title and ">= 0.66" in title
    assert fig.layout.legend.bgcolor == "#2d3436"
    names = {t.name for t in fig.data}
    assert "detections (n=1)" in names and "ground truth (n=1)" in names


def test_subspace_error_frames_are_integers_from_one():
    from webapp.pipeline_runner import figures_from_outputs

    fig = figures_from_outputs({"subspace_err": [0.5, 0.2, 0.1]})["subspace_err"]
    assert list(fig.data[0].x) == [1, 2, 3]
    assert fig.layout.xaxis.dtick == 1


# ------------------------------------------------------------------------------------
# Loading a preset opens the editor on the knob's block, named as the editor names it
# ------------------------------------------------------------------------------------
@pytest.mark.parametrize("preset", [p for p in PRESETS if p.live_knobs],
                         ids=[p.id for p in PRESETS if p.live_knobs])
def test_load_preset_opens_the_live_knob_block_editor(preset):
    import webapp.app as appmod
    from webapp import block_diagram

    bid, key, _how = preset.live_knobs[0]
    label = next(ps.label for ps in BLOCKS_BY_ID[bid].params if ps.key == key)
    _state, _n, notes, editor, _status, _results = appmod._load_preset(1, preset.id, None)
    assert label in str(editor), f"{preset.id}: editor should show {label!r}"
    assert label in str(block_diagram.preset_notes(preset)), "card names the knob by label"


# ------------------------------------------------------------------------------------
# Operator-flow review (2026-09-22): frame count, zero-frame cancel, failed run
# ------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [None, 0, -1])
def test_invalid_frame_count_refuses_to_run(monkeypatch, bad):
    """The spinner reports None for blank / out-of-range; 0, -1, 500 and blank all ran
    10 frames silently while the field kept showing the typed value."""
    import webapp.app as appmod
    from dash import no_update

    calls = []
    monkeypatch.setattr(appmod, "run_pipeline", lambda *a, **k: calls.append(k) or {})
    data, status, tab, _sink = appmod._run_pipeline(1, None, bad, "", None)
    assert data is no_update and tab is no_update and not calls
    assert "1 to" in str(status)


def test_zero_frame_cancel_stays_on_the_diagram(monkeypatch):
    import webapp.app as appmod
    from dash import no_update

    monkeypatch.setattr(appmod, "run_pipeline", lambda *a, **k: {
        "_axis_meta": {"n_steps_run": 0, "cancelled": True}})
    monkeypatch.setattr(appmod, "figures_from_outputs", lambda outputs: {})
    data, status, tab, _sink = appmod._run_pipeline(1, None, 20, "", None)
    assert data is no_update and tab is no_update
    assert "nothing ran" in str(status)


def test_failed_run_relabels_the_stale_results(monkeypatch):
    import webapp.app as appmod
    from webapp.pipeline_runner import PipelineError

    def boom(*a, **k):
        raise PipelineError("ML checkpoint not found: nope")
    monkeypatch.setattr(appmod, "run_pipeline", boom)
    prev = {"fft": {}, "_banner": "run #3  |  earlier"}
    data, status, tab, _sink = appmod._run_pipeline(4, None, 2, "", prev)
    assert data["_banner"].startswith("NOT this run -- run #4 failed")
    assert "run #3" in data["_banner"] and "fft" in data
    assert "ML checkpoint not found" in str(status)


def test_before_after_pair_shares_one_y_range():
    import webapp.app as appmod

    cur = {"subspace_err": {"data": [{"type": "scatter", "y": [0.6, 0.63]}], "layout": {}}}
    prev = {"subspace_err": {"data": [{"type": "scatter", "y": [0.04, 0.06]}], "layout": {}}}
    appmod._share_y_ranges(cur, prev)
    assert cur["subspace_err"]["layout"]["yaxis"]["range"] == prev["subspace_err"]["layout"]["yaxis"]["range"]
    assert cur["subspace_err"]["layout"]["yaxis"]["range"][1] == pytest.approx(0.63 * 1.05)


def test_single_result_card_takes_the_full_row():
    import plotly.graph_objects as go

    import webapp.app as appmod

    one = str(appmod._render_results({"range_az": go.Figure().to_dict()}, "tab-results"))
    two = str(appmod._render_results({"a": go.Figure().to_dict(), "b": go.Figure().to_dict()},
                                     "tab-results"))
    assert "1 1 100%" in one and "1 1 45%" in two
