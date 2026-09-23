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


def _wc(text: str) -> int:
    return len(text.split())


# ------------------------------------------------------------------------------------
# screen_note: the audience-facing caveat (hostile-expert third read, 2026-09-23) --
# every preset the finding named carries one; the dataclass default keeps it optional.
# ------------------------------------------------------------------------------------
def test_every_preset_carries_a_screen_note():
    for p in PRESETS:
        assert p.screen_note and isinstance(p.screen_note, str), p.id


def test_demo_preset_screen_note_defaults_to_empty_string():
    p = _preset()
    assert p.screen_note == ""


def test_thrust5_screen_notes_share_the_vmax_placeholder():
    """The three Thrust 5 detector presets share one screen note whose velocity clause
    is filled in from the corpus manifest at render time, not typed here."""
    for pid in ("thrust5_detector_cfar", "thrust5_detector_ml", "thrust5_detector_raddetnet"):
        note = PRESETS_BY_ID[pid].screen_note
        assert "{VMAX_CLAUSE}" in note
        assert "40 m" in note
        # the seed clause is RADDetNet's; the losing-network screen drops it on purpose
        assert ("seed 42 of two" in note) == (pid != "thrust5_detector_ml")


def test_thrust5_screen_notes_all_admit_the_frames_are_replayed():
    """Hostile-expert fourth read (2026-09-23): a visitor reading only the Results-tab
    note must see, on all three Thrust 5 presets, that the frames are a stored
    benchmark corpus replay and the RF chain of Thrusts 1-4 is bypassed -- not just
    hear it from the presenter (the `say` list already carried this)."""
    prefix = ("frames: stored ADC corpus (benchmark_v1_D2), replayed -- the RF chain "
              "of Thrusts 1-4 is bypassed;")
    for pid in ("thrust5_detector_cfar", "thrust5_detector_ml", "thrust5_detector_raddetnet"):
        assert PRESETS_BY_ID[pid].screen_note.startswith(prefix), pid


def test_thrust5_ml_label_and_note_admit_it_loses():
    """This preset is deliberately shown as the losing arm; the label and screen note
    must say so, not just the `say` list (hostile-expert fourth read, 2026-09-23)."""
    p = PRESETS_BY_ID["thrust5_detector_ml"]
    assert "loses" in p.label.lower() and "shown on purpose" in p.label.lower()
    assert "loses to cfar" in p.screen_note.lower()
    assert "0.127" in p.screen_note and "0.301" in p.screen_note
    assert "shown on purpose" in p.screen_note.lower()


@pytest.mark.parametrize("pid", ["thrust5_detector_cfar", "thrust5_detector_ml",
                                 "thrust5_detector_raddetnet"])
def test_thrust5_screen_note_resolves_within_a_one_line_character_budget(pid):
    """Proxy for "fits one line at 16 px on the 1600 px results page" (measured
    against the rendered rehearsal PNGs, 2026-09-23: ~7.1 px/char, ~1550 px usable) --
    a unit test cannot render the page, but it can catch a future addition that blows
    the budget again."""
    from webapp import app as appmod

    preset = PRESETS_BY_ID[pid]
    state = apply_preset(preset)
    note = appmod._resolve_screen_note(preset, state)
    assert len(note) <= 230, f"{pid}: {len(note)} chars, likely wraps to a second line"


@pytest.mark.parametrize("preset", PRESETS, ids=[p.id for p in PRESETS])
def test_card_word_count_ceiling(preset):
    """The presenter cannot glance at a card while talking (927-word RADDetNet card,
    found reading the rendered screen 2026-09-23). blurb + say + do_not_say together
    must fit in 450 words for every card."""
    total = (_wc(preset.blurb) + sum(_wc(s) for s in preset.say)
             + sum(_wc(s) for s in preset.do_not_say))
    assert total <= 450, f"{preset.id}: {total} words, over the 450-word card ceiling"


def test_raddetnet_card_meets_the_tighter_per_section_budgets():
    """The RADDetNet card specifically (Defect 3): blurb <= 120 words, at most 6 'say'
    bullets each <= 45 words, at most 5 'do not say' bullets each <= 35 words."""
    p = PRESETS_BY_ID["thrust5_detector_raddetnet"]
    assert _wc(p.blurb) <= 120
    assert len(p.say) <= 6
    for s in p.say:
        assert _wc(s) <= 45, s
    assert len(p.do_not_say) <= 5
    for s in p.do_not_say:
        assert _wc(s) <= 35, s


def test_raddetnet_card_discloses_all_four_screened_arms():
    """Fix (hostile-expert fourth read, 2026-09-23): the card must say that four
    learned arms were screened (three ported architectures plus this one) and that
    none was dropped from beat_cfar.json, so a visitor cannot suspect cherry-picking."""
    p = PRESETS_BY_ID["thrust5_detector_raddetnet"]
    assert any("four learned arms" in s.lower() and "beat_cfar.json" in s
              for s in p.say)


# ------------------------------------------------------------------------------------
# The specific promises the review extracted
# ------------------------------------------------------------------------------------
def test_thrust1_sits_at_the_weak_signal_operating_point():
    st = apply_preset(PRESETS_BY_ID["thrust1_circuit_knobs"])
    assert st["rffe"]["params"]["scale_mode"] == "legacy"
    assert st["rffe"]["params"]["signal_scaling"] == pytest.approx(1e-7)
    assert st["rffe"]["params"]["lna_bias_ma"] == 8.0 and st["rffe"]["params"]["if_bw_mhz"] == 15.0


def test_thrust2_shows_the_range_el_panel_it_used_to_hide():
    """INTEGRITY fix (hostile-expert third read, 2026-09-23): the FFT range-elevation
    panel used to be turned off because it contradicted the "image barely moves"
    story (former DO-NOT-SHOW #8). Hiding a contradicting panel is worse than showing
    it -- range_el is back on, and the card tells the three-number version instead."""
    st = apply_preset(PRESETS_BY_ID["thrust2_feature_reduction_error"])
    assert st["fft"]["enabled"] is False
    assert st["range_el"]["enabled"] and st["subspace_err"]["enabled"] and st["range_az"]["enabled"]
    p = PRESETS_BY_ID["thrust2_feature_reduction_error"]
    assert not any("deliberately off" in d.lower() and "range_el" in d.lower()
                  for d in p.do_not_say)
    assert any("2.7 db" in s.lower() for s in p.say)


def test_thrust2_screen_note_claims_only_what_the_two_panels_show():
    """Hostile-expert fourth read (2026-09-23): the note used to claim the elevation
    cut moves ~2.7 dB, a number no panel on this screen shows (that figure is a
    different, offline, unclipped-dB metric -- see notes/handoff 2026-09-22). The note
    now claims only the on-screen range-az/range-el images and the tracker curve; the
    2.7 dB claim moves to the card (`say`/`blurb`) with its metric named explicitly."""
    p = PRESETS_BY_ID["thrust2_feature_reduction_error"]
    assert "2.7" not in p.screen_note
    assert "range-elevation" in p.screen_note and "range-azimuth" in p.screen_note
    assert "barely move" in p.screen_note
    assert "not the on-screen statistic" in p.blurb
    # Card numbers drift against the live nondeterministic run (hostile-expert fourth
    # read, 2026-09-23): the cross-reference to Thrust 3's cold-start first frame must
    # say "about 0.6", never the stale fixed "0.58".
    assert not any("0.58" in s for s in p.say)
    assert any("about 0.6" in s for s in p.say)


def test_thrust3_is_a_cold_start_at_2_to_1():
    st = apply_preset(PRESETS_BY_ID["thrust3_cold_start_acquisition"])
    assert st["subspace"]["params"]["warm_start"] == "cold"
    assert st["subspace"]["params"]["k"] == 8 and st["afe"]["enabled"]


def test_thrust3_is_an_ab_preset_naming_its_own_knob():
    """Hostile-expert finding: the screen never named its own knob ('cold' with no
    warm curve to compare against). A now names the cold arm, B the warm arm."""
    p = PRESETS_BY_ID["thrust3_cold_start_acquisition"]
    assert p.ab == ("subspace", "warm_start", "warm")
    st_b = apply_preset(p, arm="b")
    assert st_b["subspace"]["params"]["warm_start"] == "warm"
    assert "cold" in p.ab_label_a.lower() and "warm" in p.ab_label_b.lower()


def test_thrust3_say_list_warns_the_numbers_drift_run_to_run():
    """Card numbers drift against the live nondeterministic run (hostile-expert fourth
    read, 2026-09-23: 0.57 rendered as 0.595 in one rehearsal) -- the blurb now quotes
    "about 0.6" instead of a fixed third decimal, and the say list tells the operator
    why, so they never quote the third decimal on stage."""
    p = PRESETS_BY_ID["thrust3_cold_start_acquisition"]
    assert "about 0.6" in p.blurb
    assert "0.57" not in p.blurb
    assert any("nondeterministic" in s.lower() and "third decimal" in s.lower()
              for s in p.say)


def test_thrust4_synthetic_filter_is_normalized_and_labelled():
    """Hostile-expert fourth read (2026-09-23): as-loaded (A) is now passthrough, the
    HEALTHY arm, and the A/B override (B) swaps in the synthetic boxcar -- the
    degraded arm -- so "after"/B is degraded on every A/B screen, matching T1/T2."""
    p = PRESETS_BY_ID["thrust4_interconnect_range_profile"]
    st = apply_preset(p)
    assert st["interconnect"]["enabled"] and st["interconnect"]["params"]["normalize_gain"] is True
    assert st["interconnect"]["params"]["case"] == "passthrough"
    st_b = apply_preset(p, arm="b")
    assert st_b["interconnect"]["params"]["case"] == "default"
    assert st["range_profile"]["enabled"]
    assert "synthetic" in p.blurb.lower()
    # Owner ballot 3A: labelled synthetic wherever it appears -- including the A/B
    # banner (the concrete finding: it used to read "Case default (boxcar)"). It now
    # lives on ab_label_b, the arm that carries the synthetic boxcar.
    assert "synthetic" in p.ab_label_b.lower()
    assert "passthrough" in p.ab_label_a.lower()


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


def test_bridge_corpora_are_discovered_on_this_machine():
    """The bridge preset only means anything if BOTH regenerated corpora are actually
    on disk and found by the same discovery `corpus_environment.manifest`'s help text
    lists (webapp/corpus_catalog.py) -- otherwise the A/B silently falls back to
    whatever `apply_preset` was handed, with no error until the run itself."""
    from webapp.corpus_catalog import CORPUS_MANIFESTS
    from webapp.demo_presets import BRIDGE_CORPUS_12BIT, BRIDGE_CORPUS_4BIT

    assert BRIDGE_CORPUS_12BIT in CORPUS_MANIFESTS
    assert BRIDGE_CORPUS_4BIT in CORPUS_MANIFESTS
    assert BRIDGE_CORPUS_12BIT != BRIDGE_CORPUS_4BIT


def test_thrust5_bridge_preset_sits_right_after_the_cfar_preset():
    ids = [p.id for p in PRESETS]
    i = ids.index("thrust5_detector_cfar")
    assert ids[i + 1] == "thrust5_bridge_adc_bits_vs_detections"


def test_thrust5_bridge_preset_is_ab_wired_to_the_two_bit_depths():
    """A (as loaded) is the 12-bit corpus, matching thrust5_detector_cfar's CFAR
    settings exactly; B swaps ONLY the corpus manifest to the 4-bit re-generation of
    the same 5 test scenes -- no other param differs between the arms."""
    from webapp.demo_presets import (
        BRIDGE_CORPUS_12BIT, BRIDGE_CORPUS_4BIT, CFAR_THRESHOLD,
    )

    p = PRESETS_BY_ID["thrust5_bridge_adc_bits_vs_detections"]
    assert p.thrust == 5
    assert p.ab == ("corpus_environment", "manifest", BRIDGE_CORPUS_4BIT)
    assert p.ab_label_a and p.ab_label_b
    assert "12-bit" in p.ab_label_a and "4-bit" in p.ab_label_b

    st_a = apply_preset(p)
    assert st_a["corpus_environment"]["params"]["manifest"] == BRIDGE_CORPUS_12BIT
    assert st_a["corpus_environment"]["params"]["split"] == "test"
    assert st_a["detector"]["params"]["mode"] == "cfar"
    assert st_a["detector"]["params"]["threshold"] == CFAR_THRESHOLD
    assert st_a["detector"]["params"]["cfar_guard"] == 2
    assert st_a["detector"]["params"]["cfar_train"] == 6
    for bid in ("rffe", "interconnect", "afe", "subspace"):
        assert st_a[bid]["enabled"] is False

    st_b = apply_preset(p, arm="b")
    assert st_b["corpus_environment"]["params"]["manifest"] == BRIDGE_CORPUS_4BIT
    # Only the manifest differs between the two arms' resolved states.
    st_a_no_manifest = dict(st_a["corpus_environment"]["params"])
    st_b_no_manifest = dict(st_b["corpus_environment"]["params"])
    del st_a_no_manifest["manifest"]
    del st_b_no_manifest["manifest"]
    assert st_a_no_manifest == st_b_no_manifest
    assert st_a["detector"] == st_b["detector"]


def test_thrust5_bridge_preset_replays_the_test_split_and_disables_the_frequency_chain():
    """Same shape as the other Thrust 5 presets: corpus replay into radar_cube +
    detector only, Thrusts 1-4 and every classic frequency-domain product off."""
    st = apply_preset(PRESETS_BY_ID["thrust5_bridge_adc_bits_vs_detections"])
    assert st["corpus_environment"]["enabled"]
    assert st["radar_cube"]["enabled"] and st["detector"]["enabled"]
    for bid in ("fft", "range_az", "range_el", "range_profile", "subspace_err", "comms"):
        assert st[bid]["enabled"] is False


def test_thrust5_bridge_card_reports_the_measured_hit_gap():
    """The card must carry the measured numbers, not a promise to measure later
    (MEASURE FIRST): cumulative hits 19 (12-bit) vs 17 (4-bit) over the 5 test frames,
    reproduced bit-for-bit across two independent runs each arm (2026-09-23)."""
    p = PRESETS_BY_ID["thrust5_bridge_adc_bits_vs_detections"]
    assert "19" in p.blurb and "17" in p.blurb
    assert "same scenes" in p.screen_note or "same 5 scenes" in p.screen_note
    assert "quantizer" in p.screen_note.lower()


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
    state, n_steps, notes, editor, status, results = appmod._load_preset(
        1, PRESETS[0].id, None)
    assert state == apply_preset(PRESETS[0]) and n_steps == PRESETS[0].n_steps
    assert notes is not None and editor is not None and status is not None
    assert results is None, "loading a preset clears the Results tab"


def test_load_preset_callback_reports_unknown_preset_without_touching_state():
    import webapp.app as appmod
    from dash import no_update
    state, n_steps, notes, _editor, _status, _results = appmod._load_preset(1, "nope", None)
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
