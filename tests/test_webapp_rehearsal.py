"""Regressions from the 2026-09-22 browser rehearsal of the demo presets.

Each test pins a defect that unit tests on block state and figure dicts could not
see, because it lived between the rendered page and the operator: an off-step number
input that wiped a preset value to null, a results page with no provenance, a legend
swatch invisible on its own background, a block diagram lighting two sources at once.
"""

from __future__ import annotations

import numpy as np
import pytest

from webapp.demo_presets import PRESETS, PRESETS_BY_ID, ab_key_is_known, apply_preset
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
    at run time. A step of None or "any" accepts anything.

    Some overrides (e.g. subspace.gap_response) are internal tracker params with no
    UI slider at all (demo_presets._INTERNAL_PARAMS) -- there is no step grid to check
    them against. `apply_preset` below still runs the preset module's own validation
    for those (its per-key checker functions), so a typo'd internal value fails loudly
    here too, just not via the step-grid assertion."""
    apply_preset(preset)
    if preset.ab is not None:
        apply_preset(preset, arm="b")
    for bid, ov in preset.overrides.items():
        for key, val in (ov.get("params") or {}).items():
            is_registry_param = any(ps.key == key for ps in BLOCKS_BY_ID[bid].params)
            if not is_registry_param:
                assert ab_key_is_known(bid, key), f"{preset.id}: unknown param {bid}.{key}"
                continue  # internal param, validated by apply_preset above, no step grid
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
    """Cross counts on the three screens are compared against each other; they are the
    false-alarm comparison only if every detector sits at the SAME recall.

    READ FROM THE SCORING FILE, not typed (2026-09-24). The three thresholds moved with
    the Ka switch (owner ballot 4A) -- 0.66/0.22/0.44 at 77 GHz, 0.6155/0.2203/0.4753 at
    Ka -- and a test that pins the digits has to be edited every time the operating point
    is re-derived, which is exactly the edit that would silently let a preset drift off
    its own recall-0.5 point. Pinning the SOURCE instead makes that impossible."""
    import json

    from webapp.detector_scoreboard import DEFAULT_BEAT_CFAR_JSON

    arms = {a["name"]: a for a in
            json.loads(DEFAULT_BEAT_CFAR_JSON.read_text())["arms"]}
    want = {"thrust5_detector_cfar": "classical CFAR",
            "thrust5_detector_ml": "fftradnet_rd_b15",
            "thrust5_detector_raddetnet": "raddetnet"}
    for pid, arm in want.items():
        thr = apply_preset(PRESETS_BY_ID[pid])["detector"]["params"]["threshold"]
        scored = arms[arm]["operating_point"]["score_threshold"]
        assert thr == pytest.approx(scored, abs=0.001), pid
        assert arms[arm]["operating_point"]["target_recall"] == pytest.approx(0.5), arm


@pytest.mark.parametrize("pid", ["thrust5_detector_cfar", "thrust5_detector_ml",
                                 "thrust5_detector_raddetnet"])
def test_corpus_presets_do_not_light_the_subspace_tracker(pid):
    """Corpus replay runs no serial stages; a lit AdaOja block fed by dashed edges
    read as 'precomputed frames through a subspace tracker' on the diagram."""
    state = apply_preset(PRESETS_BY_ID[pid])
    assert state["subspace"]["enabled"] is False


# ------------------------------------------------------------------------------------
# Block diagram: ONE source node, naming the backend this run reads
# ------------------------------------------------------------------------------------
# These two used to assert that the `environment` node went grey and its edges dashed
# while `corpus_environment` lit up -- the honest rendering of a diagram that drew all
# three sources with dotted "alternative source path" edges between them. The one-chain
# diagram draws ONE source node instead (only one source ever feeds a run; the runner
# ignores the others), so what there is to check is that the node NAMES the backend in
# use. Same finding, one node later.
def test_the_source_node_names_the_corpus_backend_when_corpus_replay_feeds_the_run():
    from webapp import block_diagram

    state = apply_preset(PRESETS_BY_ID["thrust5_detector_cfar"])
    assert block_diagram.active_source(state) == "corpus_environment"
    elements = block_diagram.build_elements(state)
    src = next(e for e in elements if e["data"].get("id") == "source")
    assert "disabled" not in src["classes"].split()
    assert "corpus" in src["data"]["label"]
    # ...and the editor opens on the backend actually in use, not on the .pkl source.
    assert src["data"]["block"] == "corpus_environment"
    assert not any(e["data"].get("id") == "environment" for e in elements)


def test_the_source_node_names_the_pkl_backend_by_default():
    from webapp import block_diagram
    from webapp.pipeline_registry import default_block_state

    state = default_block_state()
    assert block_diagram.active_source(state) == "environment"
    elements = block_diagram.build_elements(state)
    src = next(e for e in elements if e["data"].get("id") == "source")
    assert "disabled" not in src["classes"].split()
    assert ".pkl" in src["data"]["label"]


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
    # The store gained "_run_identity" with the 2026-09-24 layout change (the one-line
    # run-identity row that replaced the `Results` H3 and the run half of the banner).
    # The banner itself is NOT gone -- it is the first line of the arm's Details
    # disclosure -- so both keys must be there, and nothing else.
    assert set(second["_previous"]) == {"fft", "_banner", "_run_identity"}
    assert "run #1" in second["_previous"]["_run_identity"]


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
    # Both arms' full banners are still on the page -- as the first line of each arm's
    # Details disclosure (layout spec 2026-09-24, section 2.3). The bold 2-4 line
    # banner that used to sit above each column wrapped across the 13 px A/B gutter
    # and read as one garbled paragraph at podium distance (hostile round 10, defect
    # 2.6); what sits there now is the arm CHIP, which for a run with no A/B preset
    # falls back to "This run" / "Previous run".
    assert "run #2  |  now" in text and "run #1  |  earlier" in text
    assert "This run" in text and "Previous run" in text
    assert text.count("Graph(") == 2


# ------------------------------------------------------------------------------------
# Detector figure: names its operating point, legend readable
# ------------------------------------------------------------------------------------
def test_detector_panel_names_detector_and_threshold_and_its_legend_is_readable():
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
    from webapp.pipeline_runner import panel_of, panel_text
    fig = figures_from_outputs(outputs)["cfar_detection"]
    # The detector and its operating point are still named ON the panel -- as the HTML
    # title and caption, since the figure carries no title any more (layout spec
    # 2026-09-24). `panel_text` is title + caption + Details.
    assert "CA-CFAR (guard 2, train 6)" in panel_of(fig)["title"]
    assert ">= 0.66" in panel_text(fig) or "≥ 0.66" in panel_text(fig)
    # The legend is ONE inline 17 px line on the PANEL background now, not a 72 px
    # dark block (layout spec section 4). The dark block existed so a white open
    # ground-truth circle had a visible swatch; on the white panel that job is done by
    # the marker's own dark outline, which is what this now pins.
    assert fig.layout.legend.bgcolor == "rgba(0,0,0,0)"
    assert fig.layout.legend.font.size == 17
    gt = next(t for t in fig.data if "ground truth" in (t.name or ""))
    assert gt.marker.line.color == "#2d3436" and gt.marker.line.width == 1
    # H8 (hostile round 11): matched and unmatched detections are different glyphs, so
    # the legend now names both counts -- which is what makes the table's TP row
    # checkable against the picture.
    assert any("unmatched" in (t.name or "") for t in fig.data)
    assert any("matched" in (t.name or "") and "unmatched" not in (t.name or "")
               for t in fig.data)
    names = {t.name for t in fig.data}
    # One detection, no ground truth within tolerance of it -> it is the unmatched trace
    # that carries the count. The two counts together are still the panel's detection
    # total, and the statistic strip above the plot prints that total.
    assert any("unmatched (n=1)" in nm for nm in names)
    assert any("ground truth (n=1)" in nm for nm in names)
    # The match RULE moved to the caption: as a legend entry it was a 100-character
    # sentence inside that dark block (layout spec section 4, "Detector map").
    assert "hit = cross inside the box" in panel_text(fig)


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


def test_single_result_card_takes_one_fixed_width_panel_not_the_full_row():
    import plotly.graph_objects as go

    import webapp.app as appmod

    one = str(appmod._render_results({"range_az": go.Figure().to_dict()}, "tab-results"))
    two = str(appmod._render_results({"a": go.Figure().to_dict(), "b": go.Figure().to_dict()},
                                     "tab-results"))
    # NOT the full row any more (layout spec 2026-09-24, section 2.1, "single-arm
    # rule"): at 1523 px the lone thrust-1 map rendered 1080x230, a 4.7:1 strip with
    # ~700 px of white beside its wrapped title (hostile round 10, defect 10). One
    # product -> ONE 1008 px panel; two or more -> the same two-up 746 px grid the A/B
    # case uses, so the same product has the same rectangle on every screen.
    assert f"{appmod.SINGLE_PANEL_WIDTH}px" in one and one.count("Graph(") == 1
    # (The single-ARM header row is itself an `ab-cell-single` cell in both cases, so
    # that class is not what distinguishes them -- the panel WIDTH is.)
    assert f"{appmod.SINGLE_PANEL_WIDTH}px" not in two and two.count("Graph(") == 2


# ------------------------------------------------------------------------------------
# Owner decision 1A (2026-09-22): range figures show the physical half of the axis
# ------------------------------------------------------------------------------------
def test_range_figures_show_only_nonnegative_range():
    """Still the property; the CROP is somewhere else now.

    It used to be this module's job: each product ran its own range FFT over all
    `n_freqs` samples, the axis was fftshifted and negated, and `_nonnegative_range`
    threw the negative-delay half away at display time -- which is why the axis came
    back SHORTER than the map's own bin count. Under the one chain the spine's
    `RangeTransformBlock` crops before any product sees the cube, so the axis is
    ascending from zero and exactly as long as the data. What is pinned is the invariant
    the old assertion was really about: no negative range reaches the screen, and the
    axis matches the map it labels."""
    import numpy as np

    from webapp.pipeline_runner import figures_from_outputs

    torch = pytest.importorskip("torch")
    bins, n_freqs, span = 16, 64, 3e9
    ra = torch.rand((bins, bins)).to(torch.complex64)   # products are [bins, bins]
    prof = np.random.default_rng(1).random(bins)
    figs = figures_from_outputs({
        "range_az": [ra], "range_profile_agg": [prof],
        "_axis_meta": {"range_az_bins": bins, "range_profile_bins": bins,
                       "n_freqs": n_freqs, "freq_span_hz": span},
    })
    y = np.asarray(figs["range_az"].data[0].y)
    assert y.min() >= 0, "no negative range may reach the screen"
    assert np.all(np.diff(y) > 0), "the axis ascends from zero excess delay"
    assert figs["range_az"].data[0].z.shape[0] == len(y)
    x = np.asarray(figs["range_profile"].data[0].x)
    assert x.min() >= 0 and len(x) == prof.shape[0]


def test_bin_index_range_axis_is_kept_whole():
    import numpy as np

    from webapp.pipeline_runner import figures_from_outputs

    torch = pytest.importorskip("torch")
    ra = torch.ones((12, 12), dtype=torch.complex64)
    figs = figures_from_outputs({"range_az": [ra], "_axis_meta": {"range_az_bins": 12}})
    assert len(figs["range_az"].data[0].y) == 12
