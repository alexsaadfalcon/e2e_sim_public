"""The demo presets, the frame-count ceiling, and Cancel (webapp/demo_presets.py).

A preset is the demo. If one silently disagrees with the registry -- a renamed param, a
choice that no longer exists, a knob outside its bounds -- the operator finds out on
stage. `apply_preset` validates; these tests make sure it keeps validating and that the
presets keep the promises notes/DEMO_DEFENSE.md extracted from the adversarial review.
"""

import json

import pytest

from webapp import demo_presets
from webapp.demo_presets import (
    MAX_PRESET_N_STEPS, ML_THRESHOLD, PRESETS, PRESETS_BY_ID, DemoPreset, PresetError,
    apply_preset, validate_all,
)
from webapp.pipeline_registry import BLOCKS_BY_ID, MAX_N_STEPS, default_block_state


# ------------------------------------------------------------------------------------
# Every preset fits the registry
# ------------------------------------------------------------------------------------
def test_every_preset_validates():
    validate_all()


def test_preset_ids_unique_and_cover_all_five_thrusts():
    ids = [p.id for p in PRESETS]
    assert len(ids) == len(set(ids))
    assert {p.thrust for p in PRESETS} == {1, 2, 3, 4, 5}


@pytest.mark.parametrize("preset", PRESETS, ids=[p.id for p in PRESETS])
def test_preset_state_is_json_round_trippable_and_complete(preset):
    """dcc.Store carries the state as JSON; every block must be present with both keys."""
    state = apply_preset(preset)
    assert json.loads(json.dumps(state)) == state
    assert set(state) == set(BLOCKS_BY_ID)
    for bid, st in state.items():
        assert set(st) == {"enabled", "params"}, bid


@pytest.mark.parametrize("preset", PRESETS, ids=[p.id for p in PRESETS])
def test_preset_respects_the_frame_ceiling(preset):
    """DO-NOT-SHOW #9: runs longer than ~20 frames (cost, and a divergence spike)."""
    assert 1 <= preset.n_steps <= MAX_PRESET_N_STEPS <= MAX_N_STEPS


@pytest.mark.parametrize("preset", PRESETS, ids=[p.id for p in PRESETS])
def test_preset_enables_at_least_one_product_and_one_source(preset):
    state = apply_preset(preset)
    products = [b for b in BLOCKS_BY_ID.values() if b.category == "product"]
    assert any(state[b.id]["enabled"] for b in products), preset.id
    sources = [b for b in BLOCKS_BY_ID.values() if b.category == "source"]
    assert any(state[b.id]["enabled"] for b in sources), preset.id


@pytest.mark.parametrize("preset", PRESETS, ids=[p.id for p in PRESETS])
def test_preset_never_enables_two_alternative_sources(preset):
    state = apply_preset(preset)
    assert not (state["rt_environment"]["enabled"] and state["corpus_environment"]["enabled"])


@pytest.mark.parametrize("preset", PRESETS, ids=[p.id for p in PRESETS])
def test_preset_carries_operator_notes(preset):
    """The words are part of the deliverable: every preset says what to say and what not."""
    assert preset.blurb and preset.say and preset.do_not_say
    for bid, key, _how in preset.live_knobs:
        assert key in {p.key for p in BLOCKS_BY_ID[bid].params}, (preset.id, bid, key)


# ------------------------------------------------------------------------------------
# The specific promises the review extracted
# ------------------------------------------------------------------------------------
def test_thrust1_sits_at_the_weak_signal_operating_point():
    st = apply_preset(PRESETS_BY_ID["thrust1_circuit_knobs"])
    assert st["rffe"]["params"]["scale_mode"] == "legacy"
    assert st["rffe"]["params"]["signal_scaling"] == pytest.approx(1e-7)
    assert st["rffe"]["params"]["lna_bias_ma"] == 8.0 and st["rffe"]["params"]["if_bw_mhz"] == 15.0


def test_thrust2_hides_the_az_el_panel_that_contradicts_the_story():
    """DO-NOT-SHOW #8."""
    st = apply_preset(PRESETS_BY_ID["thrust2_feature_reduction_error"])
    assert st["fft"]["enabled"] is False and st["range_el"]["enabled"] is False
    assert st["subspace_err"]["enabled"] and st["range_az"]["enabled"]


def test_thrust3_is_a_cold_start_at_2_to_1():
    st = apply_preset(PRESETS_BY_ID["thrust3_cold_start_acquisition"])
    assert st["subspace"]["params"]["warm_start"] == "cold"
    assert st["subspace"]["params"]["k"] == 8 and st["afe"]["enabled"]


def test_thrust4_synthetic_filter_is_normalized_and_labelled():
    p = PRESETS_BY_ID["thrust4_interconnect_range_profile"]
    st = apply_preset(p)
    assert st["interconnect"]["enabled"] and st["interconnect"]["params"]["normalize_gain"] is True
    assert st["interconnect"]["params"]["case"] == "default"
    assert st["range_profile"]["enabled"]
    assert "synthetic" in p.blurb.lower()


def test_thrust5_presets_replay_the_test_split_and_disable_the_frequency_chain():
    for pid in ("thrust5_detector_cfar", "thrust5_detector_ml", "thrust5_detector_raddetnet"):
        st = apply_preset(PRESETS_BY_ID[pid])
        assert st["corpus_environment"]["enabled"]
        assert st["corpus_environment"]["params"]["split"] == "test"
        assert st["radar_cube"]["enabled"] and st["detector"]["enabled"]
        for bid in ("fft", "range_az", "range_el", "range_profile", "subspace_err", "comms"):
            assert st[bid]["enabled"] is False, (pid, bid)


def test_thrust5_ml_threshold_is_pinned_below_the_blank_figure_point():
    """The demo landmine: at the registry default (0.5) the shipped checkpoint draws
    zero detections (measured on the real frames, 2026-09-22)."""
    st = apply_preset(PRESETS_BY_ID["thrust5_detector_ml"])
    assert st["detector"]["params"]["mode"] == "ml"
    assert st["detector"]["params"]["threshold"] == ML_THRESHOLD < 0.5
    assert st["detector"]["params"]["checkpoint"].endswith("best.pt")


# ------------------------------------------------------------------------------------
# apply_preset refuses what the registry cannot represent
# ------------------------------------------------------------------------------------
def _preset(**overrides):
    return DemoPreset(id="x", label="x", thrust=1, n_steps=1, overrides=overrides,
                      blurb="b", say=["s"], do_not_say=["d"])


@pytest.mark.parametrize("bad", [
    {"nope": {"enabled": True}},
    {"rffe": {"params": {"nope": 1}}},
    {"interconnect": {"params": {"case": "tessera"}}},
    {"rffe": {"params": {"lna_bias_ma": 99.0}}},
    {"subspace": {"params": {"k": 2.5}}},
    {"detector": {"params": {"checkpoint": 3}}},
    {"environment": {"enabled": False}},
])
def test_apply_preset_rejects_misfits(bad):
    with pytest.raises(PresetError):
        apply_preset(_preset(**bad))


def test_apply_preset_rejects_too_many_frames():
    p = DemoPreset(id="x", label="x", thrust=1, n_steps=MAX_PRESET_N_STEPS + 1, overrides={},
                   blurb="b")
    with pytest.raises(PresetError):
        apply_preset(p)


def test_apply_preset_does_not_mutate_the_default_state():
    before = default_block_state()
    apply_preset(PRESETS[0])
    assert default_block_state() == before


# ------------------------------------------------------------------------------------
# UI wiring
# ------------------------------------------------------------------------------------
def _components(root):
    """Every Dash component in a layout tree (pattern-matched dict ids included)."""
    stack = [root]
    while stack:
        c = stack.pop()
        if not hasattr(c, "children") and not hasattr(c, "id"):
            continue
        yield c
        ch = getattr(c, "children", None)
        if isinstance(ch, (list, tuple)):
            stack.extend(x for x in ch if x is not None)
        elif ch is not None and not isinstance(ch, (str, int, float)):
            stack.append(ch)


def test_layout_has_preset_controls_and_cancel_and_bounded_spinner():
    from webapp import block_diagram

    comps = list(_components(block_diagram.layout()))
    ids = {c.id for c in comps if isinstance(getattr(c, "id", None), str)}
    assert {"preset-select", "preset-load", "preset-notes", "cancel-button",
            "run-nsteps"} <= ids
    spinner = next(c for c in comps if getattr(c, "id", None) == "run-nsteps")
    assert spinner.max == MAX_N_STEPS and spinner.min == 1
    cancel = next(c for c in comps if getattr(c, "id", None) == "cancel-button")
    assert cancel.disabled is True, "Cancel is enabled only while a run is in progress"


def test_preset_notes_render_for_every_preset():
    from webapp import block_diagram
    for p in PRESETS:
        assert block_diagram.preset_notes(p) is not None


def test_load_preset_callback_returns_state_and_frames():
    import webapp.app as appmod
    state, n_steps, notes, editor, status = appmod._load_preset(1, PRESETS[0].id, None)
    assert state == apply_preset(PRESETS[0]) and n_steps == PRESETS[0].n_steps
    assert notes is not None and editor is not None and status is not None


def test_load_preset_callback_reports_unknown_preset_without_touching_state():
    import webapp.app as appmod
    from dash import no_update
    state, n_steps, notes, _editor, _status = appmod._load_preset(1, "nope", None)
    assert state is no_update and n_steps is no_update and notes is not None


# ------------------------------------------------------------------------------------
# Frame ceiling and Cancel reach the simulation
# ------------------------------------------------------------------------------------
def test_run_pipeline_refuses_more_than_max_n_steps():
    pytest.importorskip("torch")
    from webapp.pipeline_runner import PipelineError, run_pipeline
    with pytest.raises(PipelineError, match="ceiling"):
        run_pipeline(default_block_state(), n_steps=MAX_N_STEPS + 1)


def test_simulation_run_stops_when_asked(make_env_block):
    pytest.importorskip("torch")
    from e2e.blocks import RangeProfileBlock
    from e2e.simulation import Simulation

    env = make_env_block(n_frames=4, n_freqs=16)
    sim = Simulation(env, [RangeProfileBlock(bins=8)], 2)
    calls = {"n": 0}

    def stop_after_two():
        calls["n"] += 1
        return calls["n"] > 2

    out = sim.run(n_steps=4, should_stop=stop_after_two)
    assert sim.cancelled is True and sim.n_steps_run == 2
    assert len(out["range_profile_agg"]) == 2, "partial outputs are the frames that ran"


def test_simulation_run_without_stop_runs_every_frame(make_env_block):
    pytest.importorskip("torch")
    from e2e.blocks import RangeProfileBlock
    from e2e.simulation import Simulation
    env = make_env_block(n_frames=3, n_freqs=16)
    sim = Simulation(env, [RangeProfileBlock(bins=8)], 2)
    sim.run(n_steps=3)
    assert sim.cancelled is False and sim.n_steps_run == 3


def test_run_pipeline_reports_partial_runs_as_cancelled(monkeypatch, make_env_block):
    pytest.importorskip("torch")
    import e2e.blocks as blocks
    from webapp.pipeline_runner import run_pipeline
    env = make_env_block(n_frames=3, n_freqs=16)
    monkeypatch.setattr(blocks, "SionnaEnvironmentBlock", lambda *a, **k: env)
    calls = {"n": 0}

    def stop_after_one():
        calls["n"] += 1
        return calls["n"] > 1

    out = run_pipeline(default_block_state(), n_steps=3, should_stop=stop_after_one)
    assert out["_axis_meta"]["cancelled"] is True and out["_axis_meta"]["n_steps_run"] == 1
    assert len(out["subspace_err"]) == 1


def test_run_pipeline_builds_only_enabled_classic_products(monkeypatch, make_env_block):
    """A product the UI shows as switchable must actually switch (Thrust 2 hides the
    FFT az-el panel this way)."""
    pytest.importorskip("torch")
    import e2e.blocks as blocks
    from webapp.pipeline_runner import figures_from_outputs, run_pipeline
    env = make_env_block(n_frames=2, n_freqs=16)
    monkeypatch.setattr(blocks, "SionnaEnvironmentBlock", lambda *a, **k: env)
    st = default_block_state()
    st["fft"]["enabled"] = False
    st["range_el"]["enabled"] = False
    out = run_pipeline(st, n_steps=1)
    assert "fft" not in out and "range_el" not in out
    assert out.get("range_az") and out.get("subspace_err")
    figs = figures_from_outputs(out)
    assert "fft" not in figs and "range_az" in figs
