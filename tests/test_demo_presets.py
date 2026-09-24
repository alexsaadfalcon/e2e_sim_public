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
    DEMO_CFR_CORPUS, MAX_PRESET_N_STEPS, ML_THRESHOLD, PRESETS, PRESETS_BY_ID,
    DemoPreset, PresetError, apply_preset, validate_all,
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


def test_thrust5_screen_notes_state_the_live_chain_and_scope_the_offline_numbers():
    """Owner wording, 2026-09-23 (the live-chain architecture change): a visitor
    reading only the Results-tab note must see WHAT IS STORED (the ray-traced
    channel), WHAT IS COMPUTED (the ADC chain, live, at the knobs on screen), which
    corpus the offline numbers belong to, and that a live count is shown for
    reference rather than re-measured live. Checked by substring, not an
    exact-prefix pin (2026-09-23 owner course-correction reworded/tightened the
    note to make room for the mandatory corpus-band disclosure -- see the test
    below -- so the exact string is no longer stable; the required CONTENT is).
    Wave 10 (2026-09-24, item 2.12, hostile round 9): "are a demo when live, not a
    re-measurement" was ungrammatical -- reworded to "are shown for reference, not
    re-measured live", so this test no longer pins "demo"/"re-measurement" as
    literal substrings, only the content they used to carry."""
    for pid in ("thrust5_detector_cfar", "thrust5_detector_ml", "thrust5_detector_raddetnet"):
        note = PRESETS_BY_ID[pid].screen_note.lower()
        assert "stored ray-traced channel" in note and "b1_demo_cfr" in note, pid
        assert "adc chain" in note and "live" in note, pid
        assert "beat_cfar.json" in note and "b1_bench_v3" in note, pid
        assert "shown for reference" in note and "not re-measured live" in note, pid
        assert "training distribution" in note, pid


def test_thrust5_screen_notes_lead_with_the_band_disclosure():
    """Wave 10 (2026-09-24, item 4.6, hostile round 9): "the single most
    attackable fact on the screen" was buried sixth of eight clauses; moved to be
    the FIRST clause. The note must still end on the scoring-crop clause (see
    test_resolve_screen_note_drops_vmax_clause_when_manifest_is_unreadable in
    tests/test_webapp_ab.py, which pins the resolved ending)."""
    for pid in ("thrust5_detector_cfar", "thrust5_detector_ml", "thrust5_detector_raddetnet"):
        note = PRESETS_BY_ID[pid].screen_note
        assert note.startswith("corpus traced at 77 GHz"), pid
        assert "scoring crop 40 m" in note, pid


def test_thrust5_screen_notes_disclose_the_corpus_band():
    """Owner course-correction, 2026-09-23: these ML corpora were traced with the
    `benchmark_v1` RadarConfig preset at 77 GHz, a different band from the munich
    frames (Ka, 28.5-31.5 GHz) every other thrust's screen shows, and a re-trace at
    Ka-band is scheduled -- so a visitor must not assume the two match, and the
    disclosure must not read as permanent (it names what is scheduled to change)."""
    for pid in ("thrust5_detector_cfar", "thrust5_detector_ml", "thrust5_detector_raddetnet"):
        note = PRESETS_BY_ID[pid].screen_note
        assert "77 GHz" in note, pid
        assert "Ka-band" in note and "scheduled" in note, pid


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
    # TWO lines since 2026-09-23, not one: the owner's mandated live-chain sentence
    # (what is stored / what runs live / which corpus the offline numbers are from /
    # demonstration not re-measurement) does not fit 230 chars, and the content is
    # not negotiable. 400 is the same ~7.1 px/char proxy applied to two lines, read
    # back on the rendered rehearsal PNG. A THIRD line would push the first figure
    # below the fold, so the ceiling stays.
    assert len(note) <= 400, f"{pid}: {len(note)} chars, likely wraps to a third line"


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


def test_thrust1_label_names_the_noise_floor_not_image_quality():
    """Wave 10 (2026-09-24, item 4.5, hostile round 9): "vs image quality" oversold
    the screen -- both panels are pixel-identical (do-not-change 5.1) and the
    card's own say list concedes the moved statistic isn't target SNR."""
    p = PRESETS_BY_ID["thrust1_circuit_knobs"]
    assert p.label == "Thrust 1 - RF circuit knobs vs the image's noise floor"
    assert "image quality" not in p.label.lower()


def test_thrust1_and_2_multipath_number_matches_the_panel():
    """Wave 10 (2026-09-24, item 1.7, hostile round 9): the panel subtitles print
    "36 m" (1 m display gates rounding the true 37.1 m delay, F94); the cards used
    to tell the presenter to read "37 m" off the screen, a number not printed
    there."""
    for pid in ("thrust1_circuit_knobs", "thrust2_feature_reduction_error"):
        p = PRESETS_BY_ID[pid]
        assert any("~36 m" in s and "37.1 m" in s for s in p.say), pid
        assert not any("near 37 m" in s for s in p.say), pid


def test_thrust1_names_the_operating_point_difference_from_thrust2():
    """Wave 10 (2026-09-24, item 4.1, hostile round 9): peak-median moves 11 dB
    between consecutive slides (T1 ~66 dB, T2 ~77 dB on the same munich frames);
    the cause is T1's deliberate legacy signal_scaling, not a different scene --
    name it before the room asks."""
    p = PRESETS_BY_ID["thrust1_circuit_knobs"]
    assert any("signal_scaling 1e-7" in s and "operating point" in s.lower()
              for s in p.say)


def test_thrust1_stripe_do_not_say_holds_on_every_frame():
    """Wave 10 (2026-09-24, item 2.6, hostile round 9): RETRACTED "it is not
    visible" -- false on frame 2 (a yellow 1-px stripe at range 0 is the brightest
    thing on the map that frame). Reworded to a claim that holds on every frame."""
    p = PRESETS_BY_ID["thrust1_circuit_knobs"]
    assert not any("it is not visible" in s.lower() for s in p.do_not_say)
    assert any("thin stripe" in s.lower() and "when it shows" in s.lower()
              for s in p.do_not_say)


# ------------------------------------------------------------------------------------
# Wave 7 review, KA-band screens re-traced with diffuse scattering (2026-09-23): X4-X8.
# ------------------------------------------------------------------------------------
@pytest.mark.parametrize("pid", ["thrust1_circuit_knobs", "thrust2_feature_reduction_error"])
def test_wave7_cards_carry_the_array_disclosure(pid):
    """X4-X8 (hostile reader): every card that mentions the array states its size,
    spacing/aperture, band, boresight offset and the diffuse-scattering assumption."""
    from webapp.demo_presets import _ARRAY_DISCLOSURE

    p = PRESETS_BY_ID[pid]
    assert _ARRAY_DISCLOSURE in p.screen_note
    assert "32x32" in _ARRAY_DISCLOSURE and "5 mm" in _ARRAY_DISCLOSURE
    assert "15.5 cm" in _ARRAY_DISCLOSURE
    assert "28.5-31.5 ghz" in _ARRAY_DISCLOSURE.lower() or "ka-band" in _ARRAY_DISCLOSURE.lower()
    assert "35 deg" in _ARRAY_DISCLOSURE
    assert "0.4" in _ARRAY_DISCLOSURE and "assumed" in _ARRAY_DISCLOSURE.lower()


@pytest.mark.parametrize("pid", ["thrust1_circuit_knobs", "thrust2_feature_reduction_error"])
def test_wave7_stale_stripe_claim_is_gone(pid):
    """X6: the old '20-22 m' stripe quote was off the true axis (the 1000-point Ka
    sweep aliased a 68 m return down to ~18 m, F94); the fix ships 5000 points (125 m
    unambiguous, no aliasing) and the cards must name the true families instead."""
    p = PRESETS_BY_ID[pid]
    for text in [p.blurb, p.screen_note, *p.say, *p.do_not_say]:
        assert "20-22 m" not in text
    assert any("37" in s and "68" in s for s in p.say), \
        f"{pid}: expected the true multipath ranges (37 m / 68 m family) on the card"


@pytest.mark.parametrize("pid", ["thrust1_circuit_knobs", "thrust2_feature_reduction_error"])
def test_wave7_range_axis_calibration_is_on_the_screen_note(pid):
    """X6/X7: metres-per-gate and the unambiguous range, computed from the frame's own
    freq_plan (pipeline_runner._range_per_gate_m / _native_unambiguous_range_m), not
    hand-typed -- see tests/test_webapp_figures_wave7.py for the panel-level pin."""
    p = PRESETS_BY_ID[pid]
    assert "m/gate" in p.screen_note or "m per gate" in p.screen_note
    assert "125 m" in p.screen_note or "unambiguous" in p.screen_note.lower()


def test_thrust2_tracker_k_repicked_for_the_ka_retrace():
    """X4/X8 (F94): k=8, the pre-retrace default, is degenerate once real multipath is
    restored (effective rank 3-4; arm A spikes hard mid-run). k=2 is the largest k
    measured stable on both arms (2026-09-23, six repeated runs) -- see the preset's
    own `overrides` comment for the numbers."""
    st = apply_preset(PRESETS_BY_ID["thrust2_feature_reduction_error"])
    assert st["subspace"]["params"]["k"] == 2
    p = PRESETS_BY_ID["thrust2_feature_reduction_error"]
    assert any("k=2" in s or "k = 2" in s.lower() for s in [p.blurb] + p.say)
    assert any("degenerate" in s.lower() and "k=8" in s.lower() for s in p.say + p.do_not_say)


def test_thrust2_shows_the_range_el_panel_it_used_to_hide():
    """INTEGRITY fix (hostile-expert third read, 2026-09-23): the FFT range-elevation
    panel used to be turned off because it contradicted the "image barely moves"
    story (former DO-NOT-SHOW #8). Hiding a contradicting panel is worse than showing
    it -- range_el is back on. Wave 9 update #3 (2026-09-24): the card no longer
    quotes a specific dB move for either panel (see
    test_thrust2_never_quotes_a_drifting_dB_pair) -- it tells the operator to read
    both printed peak-median numbers off the screen instead."""
    st = apply_preset(PRESETS_BY_ID["thrust2_feature_reduction_error"])
    assert st["fft"]["enabled"] is False
    assert st["range_el"]["enabled"] and st["subspace_err"]["enabled"] and st["range_az"]["enabled"]
    p = PRESETS_BY_ID["thrust2_feature_reduction_error"]
    assert not any("deliberately off" in d.lower() and "range_el" in d.lower()
                  for d in p.do_not_say)
    assert any("peak-median" in s.lower() for s in p.say)


def test_thrust2_screen_note_claims_only_what_the_two_panels_show():
    """Hostile-expert fourth read (2026-09-23): the note used to claim the elevation
    cut moves ~2.7 dB, a number no panel on this screen shows (that figure was a
    cross-arm mean |dB| difference on the off-screen FFT az-el product). The card
    then went through three more re-quotings of a specific dB pair (0.52 dB mean at
    76.85->76.57; then 76.9->76.6, 0.3 dB; then 76.9->76.5, 0.4 dB, "slightly more
    than" azimuth's 76.7->76.4) -- each one measured true on ITS OWN re-render and
    false on the next, because the pipeline is nondeterministic at ~5e-3 and these
    peak-median statistics drift ~0.1-0.2 dB run to run. Wave 9 update #3
    (2026-09-24, orchestrator course-correction): retracted the whole pattern.
    NO exact pair belongs on the card, ever -- see
    test_thrust2_never_quotes_a_drifting_dB_pair."""
    p = PRESETS_BY_ID["thrust2_feature_reduction_error"]
    assert "2.7" not in p.screen_note
    assert "2.7" not in p.blurb
    assert not any("2.7" in s for s in p.say + p.do_not_say)
    assert "range-elevation" in p.screen_note and "range-azimuth" in p.screen_note
    assert "barely move" in p.screen_note
    # Wave 7 (2026-09-23, F94): the old cross-reference to Thrust 3's cold-start first
    # frame ("about 0.6") compared error values at k=8; this preset now runs at k=2
    # (k=8/k=4 are both degenerate on the Ka retrace -- see the `overrides` comment),
    # and an unnormalized subspace-error distance at a different k is not the same
    # quantity, so the cross-reference is gone rather than silently wrong. The say list
    # says so explicitly instead.
    assert not any("0.58" in s for s in p.say)
    assert not any("about 0.6" in s and "thrust 3" in s.lower() for s in p.say)
    # Wave 8 (W7): Thrust 3 also runs at k=2 (not k=8), so the two ARE comparable --
    # the stale "not comparable" claim is retracted, not just reworded.
    assert any("both run at k=2" in s.lower() for s in p.do_not_say)
    assert not any("not comparable" in s.lower() for s in p.say + p.do_not_say)


def test_thrust2_never_quotes_a_drifting_dB_pair():
    """Wave 9 update #3 (2026-09-24, orchestrator course-correction): a same-day
    re-render printed range-azimuth 76.7->76.5 and range-elevation 76.8->76.6
    (0.2 dB each) -- yet more numbers, none matching any pair this card had quoted
    minutes earlier. The peak-median statistics drift ~0.1-0.2 dB run to run
    (nondeterministic at ~5e-3), so NO exact pair, and no exact "0.3 dB" / "0.4 dB"
    move, belongs on the card: it must instead tell the operator to read the two
    printed numbers off THIS run's own screen."""
    p = PRESETS_BY_ID["thrust2_feature_reduction_error"]
    for text in [p.blurb, p.screen_note, *p.say, *p.do_not_say]:
        assert "76." not in text, text
        assert "0.3 dB" not in text and "0.4 dB" not in text, text
    assert any("run-to-run floor" in s.lower() or "run to run" in s.lower()
              for s in [p.blurb] + p.say + p.do_not_say)


def test_thrust3_is_a_cold_start_at_2_to_1():
    """Owner decision (option A, 2026-09-23): both arms cold-start; k re-picked to 2
    (F94, the largest spike-free rank measured on the current munich_ka.pkl -- see the
    preset's own `overrides` comment). n_refine is NOT pinned in state: it is derived
    from gap_response by pipeline_runner.run_pipeline (see that test below), which is
    what lets the single `ab` switch move both together."""
    st = apply_preset(PRESETS_BY_ID["thrust3_cold_start_acquisition"])
    assert st["subspace"]["params"]["warm_start"] == "cold"
    assert st["subspace"]["params"]["k"] == 2 and st["afe"]["enabled"]
    assert st["subspace"]["params"]["gap_response"] == "none"
    assert "n_refine" not in st["subspace"]["params"]


def test_thrust3_is_an_ab_preset_naming_its_own_knob():
    """Owner decision (option A), round 2 (2026-09-23): an identical-arms screen is a
    null demo. A is now a fixed LOW-effort arm (visible acquisition), B is the shipped
    adaptive gate -- both still cold-start."""
    p = PRESETS_BY_ID["thrust3_cold_start_acquisition"]
    assert p.ab == ("subspace", "gap_response", "refine")
    st_b = apply_preset(p, arm="b")
    assert st_b["subspace"]["params"]["gap_response"] == "refine"
    assert st_b["subspace"]["params"]["warm_start"] == "cold"  # B is still cold-start
    assert "fixed" in p.ab_label_a.lower() and "adaptive" in p.ab_label_b.lower()
    assert "5" in p.ab_label_a and "10" in p.ab_label_b


def test_thrust3_derives_n_refine_from_gap_response(monkeypatch, make_env_block):
    """n_refine has no registry ParamSpec (demo_presets._INTERNAL_PARAMS); its default
    is DERIVED from gap_response in pipeline_runner.run_pipeline: "none" -> 5 (arm A,
    the largest n_refine of {1, 2, 3, 5} measured to still take >=3 frames to reach
    within 1.5x of its own settled level at k=2, real chain, 2026-09-23), anything else
    -> 10 (the original shipped default -- byte-identical to every other preset, which
    never sets gap_response at all). Spies on AdaOjaBlock's constructor directly (a
    synthetic frame's own spectral gap can collapse and make the GATE escalate at
    runtime -- a separate mechanism, not what this test is about)."""
    pytest.importorskip("torch")
    import e2e.blocks as blocks
    from webapp.pipeline_runner import run_pipeline

    real_cls = blocks.AdaOjaBlock
    captured = []

    class _Spy(real_cls):
        def __init__(self, *a, **kw):
            captured.append((kw.get("n_refine"), kw.get("gap_response")))
            super().__init__(*a, **kw)

    monkeypatch.setattr(blocks, "AdaOjaBlock", _Spy)
    env = make_env_block(n_frames=1, n_freqs=16)
    monkeypatch.setattr(blocks, "SionnaEnvironmentBlock", lambda *a, **k: env)
    p = PRESETS_BY_ID["thrust3_cold_start_acquisition"]
    run_pipeline(apply_preset(p), n_steps=1)
    run_pipeline(apply_preset(p, arm="b"), n_steps=1)
    assert captured == [(5, "none"), (10, "refine")]


def test_thrust3_say_list_warns_the_numbers_drift_run_to_run():
    """Card numbers drift against the live nondeterministic run -- the blurb quotes
    "about" rather than a fixed third decimal, and the say list tells the operator
    why. Numbers re-measured 2026-09-23 (round 2, visible-acquisition arms) on the
    current file at k=2 (see the preset's `overrides` comment): arm A about
    0.6 -> 0.31 -> ~0.2 (wave 10, 2026-09-24, item 1.10: "0.19" read ~0.20 on the
    03:0x re-render, off by 2x the card's own ~5e-3 nondeterminism floor -- rounded
    to "about 0.2"), arm B about 0.30 -> 0.09."""
    p = PRESETS_BY_ID["thrust3_cold_start_acquisition"]
    assert "about 0.6" in p.blurb and "about 0.30" in p.blurb and "about 0.2" in p.blurb
    assert "0.57" not in p.blurb and "0.595" not in p.blurb and "0.19" not in p.blurb
    assert any("nondeterministic" in s.lower() and "third decimal" in s.lower()
              for s in p.say)


def test_thrust3_gate_measured_as_a_no_op_at_the_shipped_k():
    """F94's tracker addendum ("sv_gap_norm collapses on every Ka frame... the shipped
    gate... spends 60 power iterations per frame") was measured at k=8 on an EARLIER
    munich_ka.pkl. Re-measured on the current (post-8d1e251) file at the shipped k=2:
    the gate never escalates past its 10-pass baseline (n_refine_used stays flat, 8/8
    frames) -- the card must say so and must not carry the old "always fires" claim or
    the retired frames-23-26 / three-frames talking points."""
    p = PRESETS_BY_ID["thrust3_cold_start_acquisition"]
    assert not any("frames 23-26" in s for s in p.say + p.do_not_say + [p.blurb])
    assert not any("three frames" in s.lower() for s in p.say + p.do_not_say + [p.blurb])
    assert any("never escalate" in s.lower() or "measured false" in s.lower()
              for s in p.say + p.do_not_say + [p.blurb])
    assert any("gap diagnostic" in s.lower() for s in p.say)


def test_thrust3_settled_floor_vs_passes_statistic_is_on_the_card():
    """Round-6 review established the card needs an A/B statistic, both stated and
    rendered (the subspace_err figure's second trace, see tests/test_webapp_ab.py).
    Wave 10 (2026-09-24, item 1.4, hostile round 9): RETRACTED "frames-to-acquire ...
    one frame sooner" -- arm A never reaches B's floor in 8 frames (contradicted the
    blurb's own honest claim on the same card). The statistic is now the settled
    FLOOR at two fixed pass counts (5 vs 10), not a race with an undefined winner."""
    p = PRESETS_BY_ID["thrust3_cold_start_acquisition"]
    assert any("frame 3" in s for s in [p.blurb] + p.say)
    assert any("frame 2" in s for s in [p.blurb] + p.say)
    assert any("passes-per-frame" in s.lower() or "passes/frame" in s.lower()
              for s in p.say)
    assert any("settled floor" in s.lower() and "5 vs 10 passes" in s.lower()
              for s in p.say)
    assert not any("frames-to-acquire" in s.lower() for s in p.say + p.do_not_say + [p.blurb])
    assert not any("one frame sooner" in s.lower() for s in p.say + p.do_not_say + [p.blurb])


def test_thrust3_explains_the_warm_start_reference_line():
    """Wave 10 (2026-09-24, item 4.3, hostile round 9): prepared answer for "why does
    one cold run hit the warm floor and the other never does?" -- both arms are cold
    starts; the dashed line is a WARM-started level."""
    p = PRESETS_BY_ID["thrust3_cold_start_acquisition"]
    assert any("cold start" in s.lower() and "warm" in s.lower() for s in p.say)


def test_thrust4_runs_the_live_tessera_surrogate_ab_on_height():
    """2026-09-23 rebuild: Thrust 4 moved off the synthetic boxcar onto the LIVE
    public Tessera surrogate (InterconnectBlock(source='tessera')). Arm A is the
    canonical geometry (source='tessera' with every knob at its ParamSpec default,
    i.e. the wrapper's shipped design through the scale model); arm B changes ONLY
    the TSV height knob, to the low end of its presented envelope -- the largest
    single-knob mover of the range-profile skirt of the five (measured through the
    same block on the real munich frames, see the preset's own comment)."""
    from webapp.demo_presets import _TESSERA_ARM_B_HEIGHT_UM, _TESSERA_CANONICAL_HEIGHT_UM
    from webapp.pipeline_registry import BLOCKS_BY_ID

    p = PRESETS_BY_ID["thrust4_interconnect_range_profile"]
    st = apply_preset(p)
    assert st["interconnect"]["enabled"]
    assert st["interconnect"]["params"]["source"] == "tessera"
    assert st["interconnect"]["params"]["case"] == "default"  # not passthrough: the
    # surrogate must actually be evaluated, not bypassed
    assert st["interconnect"]["params"]["tessera_height_um"] == _TESSERA_CANONICAL_HEIGHT_UM
    st_b = apply_preset(p, arm="b")
    assert st_b["interconnect"]["params"]["tessera_height_um"] == _TESSERA_ARM_B_HEIGHT_UM
    # Only the height knob differs between the two arms.
    for key in st["interconnect"]["params"]:
        if key != "tessera_height_um":
            assert st_b["interconnect"]["params"][key] == st["interconnect"]["params"][key]
    height_spec = next(ps for ps in BLOCKS_BY_ID["interconnect"].params
                       if ps.key == "tessera_height_um")
    assert _TESSERA_ARM_B_HEIGHT_UM == height_spec.min  # the low end, not an arbitrary drop
    assert st["range_profile"]["enabled"]
    assert "tessera" in p.blurb.lower() and "live" in p.blurb.lower()
    assert "canonical" in p.ab_label_a.lower()
    # The retracted "structurally absent" crosstalk claim must not be ASSERTED as true
    # anywhere the operator would say it (build item 3: F89 gave real, if caveated,
    # crosstalk numbers) -- `do_not_say` is exempt, since flagging the retracted claim
    # BY NAME as something not to say is the whole point of that list.
    for text in [p.blurb, p.screen_note, *p.say]:
        assert "structurally absent" not in text.lower()
    assert any("structurally absent" in s.lower() and "retracted" in s.lower()
              for s in p.do_not_say)
    assert "next" in p.screen_note.lower() and "fext" in p.screen_note.lower()


# ------------------------------------------------------------------------------------
# Wave 7 review, KA-band screens (2026-09-23): X3, X11, X2, X9, X10.
# ------------------------------------------------------------------------------------
def test_thrust4_card_quotes_only_the_on_screen_median_floor_statistic():
    """X3: the card used to quote 'skirt -53.90 -> -57.43 dB' as if it were on the
    rendered panel; that number is an OFFLINE measurement and the panel itself
    prints its own 'median floor' statistic. The offline number may still be named
    (with its own provenance), but the card must say the panel prints something
    else and that the offline move sits below it."""
    p = PRESETS_BY_ID["thrust4_interconnect_range_profile"]
    assert "median floor" in p.blurb.lower()
    assert "sits below" in p.blurb.lower() or "below it" in p.blurb.lower()
    assert "offline" in p.blurb.lower()


def test_thrust4_card_uses_one_height_convention_with_the_model_geometry_said_once():
    """X3: Arm A's own label used to say 'h 100 um' (model geometry) while every
    other reference on the card used the presented 50 um -- one convention now
    (presented), with the model-geometry equivalence computed and stated once."""
    from webapp.demo_presets import (
        _TESSERA_CANONICAL_HEIGHT_MODEL_UM, _TESSERA_CANONICAL_HEIGHT_UM,
    )

    p = PRESETS_BY_ID["thrust4_interconnect_range_profile"]
    assert f"{_TESSERA_CANONICAL_HEIGHT_UM:g} um presented" in p.ab_label_a
    assert "100 um" not in p.ab_label_a
    assert _TESSERA_CANONICAL_HEIGHT_MODEL_UM == _TESSERA_CANONICAL_HEIGHT_UM * 2
    assert (f"{_TESSERA_CANONICAL_HEIGHT_MODEL_UM:g} um model geometry" in p.blurb)


def test_thrust4_card_answers_the_skin_depth_question():
    """X11: the prepared F91 answer to 'skin depth goes as f^-1/2, not f^-1'."""
    p = PRESETS_BY_ID["thrust4_interconnect_range_profile"]
    assert any("skin depth" in s.lower() and "f^-1/2" in s for s in p.say)
    assert any("sqrt(2)" in s and "f91" in s.lower() for s in p.say)


def test_thrust4_card_never_claims_the_median_floor_is_identical_on_both_arms():
    """Wave 10 (2026-09-24, item 1.2, hostile round 9): a re-render printed median
    floor -50.3 (A) vs -50.4 (B) -- the floor is the statistic that moved by ~0.1 dB
    that run, while range-azimuth peak-median printed identically (76.6/76.6). The
    card must not claim a specific panel holds still; it must generalise to
    "either panel"."""
    p = PRESETS_BY_ID["thrust4_interconnect_range_profile"]
    assert "on both arms" not in p.blurb.lower()
    assert "either panel" in p.blurb.lower()
    assert "run-to-run floor" in p.blurb.lower()


def test_thrust4_crosstalk_clause_quotes_presented_pitch_with_model_in_parens():
    """Wave 10 (2026-09-24, item 1.3, hostile round 9): the run-notes line under the
    banner quotes PRESENTED Tessera geometry (pitch 30 um); the crosstalk clause used
    to quote the same shipped pitch in MODEL units (60 um) with no unit noted -- one
    screen, two conventions. Converted to presented-with-model-in-parens, and the
    model->presented conversion must come from the imported Ka scale constant, never
    a hand-typed '/2' or '2'."""
    from webapp.demo_presets import (
        _TESSERA_CROSSTALK_SHIPPED_PITCH_MODEL_UM, _TESSERA_CROSSTALK_SHIPPED_PITCH_UM,
        _TESSERA_CROSSTALK_TRAINING_PITCH_MODEL_UM, _TESSERA_CROSSTALK_TRAINING_PITCH_UM,
        _TESSERA_KA_SCALE,
    )
    assert _TESSERA_CROSSTALK_TRAINING_PITCH_UM == (
        _TESSERA_CROSSTALK_TRAINING_PITCH_MODEL_UM / _TESSERA_KA_SCALE)
    assert _TESSERA_CROSSTALK_SHIPPED_PITCH_UM == (
        _TESSERA_CROSSTALK_SHIPPED_PITCH_MODEL_UM / _TESSERA_KA_SCALE)
    # The shipped pitch, presented, matches the run-notes convention (30 um) --
    # cross-checked against the registry's own presented default, not re-typed.
    from webapp.pipeline_registry import BLOCKS_BY_ID
    pitch_spec = next(ps for ps in BLOCKS_BY_ID["interconnect"].params
                      if ps.key == "tessera_pitch_um")
    assert _TESSERA_CROSSTALK_SHIPPED_PITCH_UM == pitch_spec.default

    p = PRESETS_BY_ID["thrust4_interconnect_range_profile"]
    assert f"{_TESSERA_CROSSTALK_SHIPPED_PITCH_UM:g} um presented" in p.screen_note
    assert f"({_TESSERA_CROSSTALK_SHIPPED_PITCH_MODEL_UM:g} um model" in p.screen_note
    assert f"{_TESSERA_CROSSTALK_TRAINING_PITCH_UM:g} um presented" in p.screen_note
    assert f"({_TESSERA_CROSSTALK_TRAINING_PITCH_MODEL_UM:g} um model" in p.screen_note


def test_thrust4_screen_note_uses_one_crosstalk_vocabulary():
    """Wave 10 (2026-09-24, item 1.11, hostile round 9): the run-notes line says
    'ring3x3 arrangement'; the screen note used to say 'this default single-via
    one' for the same fact -- both correct, two vocabularies. Matched wording."""
    p = PRESETS_BY_ID["thrust4_interconnect_range_profile"]
    assert "single-via" not in p.screen_note.lower()
    assert "ring3x3" in p.screen_note.lower()
    assert "one signal via" in p.screen_note.lower()


def test_thrust4_does_not_claim_50_dB_below_the_noise_floor():
    """Wave 10 (2026-09-24, item 1.8, hostile round 9): the range-profile axis runs
    0 to -60 dB and the offline skirt levels quoted (-53.90 -> -57.43 dB) sit INSIDE
    that range -- not 50 dB below the panel's own -50.3 dB printed floor. Retracted,
    not reworded: the honest sentence right next to it already says the same panel
    prints a different statistic."""
    p = PRESETS_BY_ID["thrust4_interconnect_range_profile"]
    assert "50 db below" not in p.screen_note.lower()
    assert "not the statistic the panel" in p.screen_note.lower()


def test_thrust4_says_what_it_did_establish():
    """Wave 10 (2026-09-24, item 4.2, hostile round 9): a prepared answer for "then
    why is this a thrust?" -- the negative result (six knobs run live, in-band |S21|
    moves <0.03 dB across all of them) stated as a finding, not left implicit."""
    p = PRESETS_BY_ID["thrust4_interconnect_range_profile"]
    assert any("did establish" in s.lower() and "negative result" in s.lower()
              for s in p.say)


def test_thrust5_ml_card_admits_the_shown_frame_scores_zero():
    """X9: the displayed frame (TP = 0, all crosses miss on Arm B) is the mechanism
    on display -- the near-range corner filter removing the returns this checkpoint
    was trained on -- not an accident."""
    p = PRESETS_BY_ID["thrust5_detector_ml"]
    assert any("tp = 0" in s.lower() for s in p.say)
    assert any("not an accident" in s.lower() for s in p.say)


def test_thrust5_cfar_card_explains_unmatched_dropping_at_deeper_quantisation():
    """X10: the prepared answer for 'unmatched dropped, not rose, at 3 bits' --
    quantisation noise raises the CA-CFAR estimate, so fewer weak peaks clear the
    threshold; a loss of sensitivity, not a quality gain."""
    p = PRESETS_BY_ID["thrust5_detector_cfar"]
    assert any("quantisation noise raises the ca-cfar estimate" in s.lower()
              for s in p.say)
    assert any("loss of sensitivity, not a quality gain" in s.lower() for s in p.say)
    # The pre-existing "5 frames cannot resolve" line must still be present.
    assert any("5 frames at one threshold is a demonstration" in s
              for s in p.do_not_say)


def test_thrust5_raddetnet_card_leads_with_the_matched_recall_fa_comparison():
    """X2: the card's opening line must lead with the ONE comparison this table can
    defend -- false alarms at matched recall -- not with the architecture
    description or a hit-count framing that reads as a loss on 5 unmatched-recall
    frames. Numbers themselves are unchanged (`beat_cfar.json`'s own 2.99/6.24)."""
    p = PRESETS_BY_ID["thrust5_detector_raddetnet"]
    opening = p.blurb[:160]  # first sentence -- longer than any decimal-point split
    assert "false alarm" in opening.lower()
    assert "2.99" in opening and "6.24" in opening
    # Exact numbers from beat_cfar.json -- never re-typed independently of the file.
    import json
    from webapp.detector_scoreboard import DEFAULT_BEAT_CFAR_JSON
    data = json.loads(DEFAULT_BEAT_CFAR_JSON.read_text())
    cfar_fa = next(a for a in data["arms"] if a["name"] == "classical CFAR")[
        "operating_point"]["fp_per_frame"]
    raddetnet_fa = next(a for a in data["arms"] if a["name"] == "raddetnet")[
        "operating_point"]["fp_per_frame"]
    assert f"{raddetnet_fa:.2f}" in p.blurb and f"{cfar_fa:.2f}" in p.blurb
    # Wave 10 (2026-09-24, item 1.6, hostile round 9): the number is printed on
    # THIS screen, same row -- do not send the presenter to the CFAR preset for it.
    assert "on the cfar screen" not in p.blurb.lower()


def test_thrust5_raddetnet_drops_the_contradictory_v2_clause():
    """Wave 10 (2026-09-24, item 1.5, hostile round 9): the do_not_say used to add
    "0.487 on v2 is in-distribution, not OOD" -- 0.487 is the JOINT checkpoint's
    number (trained partly on v2), not this seed-42 checkpoint's; the say list
    already states a DIFFERENT number (0.208) for THIS checkpoint on v2. Two
    incompatible statements about "v2" on one card -- dropped, not disambiguated."""
    p = PRESETS_BY_ID["thrust5_detector_raddetnet"]
    assert "0.487" not in p.blurb
    assert not any("0.487" in s for s in p.say + p.do_not_say)


def test_thrust5_raddetnet_names_the_seed_spread_vs_ci_caveat():
    """Wave 10 (2026-09-24, item 4.4, hostile round 9): the scoreboard prints "seed
    spread 0.040 > CI half-width 0.032" under the +0.175 AP lead claim -- a
    statistician finds the CI-understates-uncertainty issue in ten seconds if the
    card has no sentence for it. The claim must rest on the three-seeds-plus-joint-
    arm replication (F86), not the CI alone."""
    p = PRESETS_BY_ID["thrust5_detector_raddetnet"]
    assert any("seed spread" in s.lower() and "ci half-width" in s.lower()
              for s in p.say)
    assert any("not the ci alone" in s.lower() for s in p.say)


def test_thrust5_presets_replay_the_test_split_and_disable_the_frequency_chain():
    for pid in ("thrust5_detector_cfar", "thrust5_detector_ml", "thrust5_detector_raddetnet"):
        st = apply_preset(PRESETS_BY_ID[pid])
        assert st["corpus_environment"]["enabled"]
        assert st["corpus_environment"]["params"]["split"] == "test"
        assert st["radar_cube"]["enabled"] and st["detector"]["enabled"]
        for bid in ("fft", "range_az", "range_el", "range_profile", "subspace_err", "comms"):
            assert st[bid]["enabled"] is False, (pid, bid)


def test_thrust5_all_cards_say_the_detector_panel_is_frame_pinned():
    """Wave 10 (2026-09-24, item 2.2, hostile round 9): on all three T5 screens
    the objectness/scoreboard panel has no frame slider (pinned to the last
    frame) while the Range-Doppler panel does -- the runbook's usual "advance
    with the slider" instruction would desync them if the presenter didn't know."""
    for pid in ("thrust5_detector_cfar", "thrust5_detector_ml", "thrust5_detector_raddetnet"):
        p = PRESETS_BY_ID[pid]
        assert any("slider" in s.lower() and "range-doppler" in s.lower()
                  for s in p.say), pid


def test_thrust5_ml_names_the_pr_legend_alias():
    """Wave 10 (2026-09-24, item 3.7, hostile round 9): the panel title/scoreboard
    say b5_fftradnet_v3, the PR legend says fftradnet_rd_b5 -- one name, stated."""
    p = PRESETS_BY_ID["thrust5_detector_ml"]
    assert any("fftradnet_rd_b5" in s and "b5_fftradnet_v3" in s for s in p.say)


def test_thrust5_ml_and_raddetnet_name_the_seed_spread_caveat():
    """Wave 10 (2026-09-24, item 4.4, hostile round 9): both scoreboards print
    "seed spread ... > CI half-width ..." -- a statistician finds the CI understates
    uncertainty in ten seconds if the card has no sentence for it."""
    for pid in ("thrust5_detector_ml", "thrust5_detector_raddetnet"):
        p = PRESETS_BY_ID[pid]
        assert any("seed spread" in s.lower() and "ci half-width" in s.lower()
                  for s in p.say), pid


def test_thrust5_ml_threshold_is_pinned_below_the_blank_figure_point():
    """The demo landmine: at the registry default (0.5) the shipped checkpoint draws
    zero detections (measured on the real frames, 2026-09-22)."""
    st = apply_preset(PRESETS_BY_ID["thrust5_detector_ml"])
    assert st["detector"]["params"]["mode"] == "ml"
    assert st["detector"]["params"]["threshold"] == ML_THRESHOLD < 0.5
    assert st["detector"]["params"]["checkpoint"].endswith("best.pt")


def test_thrust5_presets_replay_the_stored_channel_through_the_live_chain():
    """The architecture change (owner 2026-09-23): the Thrust 5 screens no longer
    replay a stored ADC cube. They replay the stored RAY-TRACED CHANNEL of the demo
    corpus and re-run the whole chain that produced it -- front end, interconnect,
    dechirp, thermal floor, impairments, IF high-pass, ADC -- so a front-end knob
    reaches the detector. Anything less and the A/B below would be a no-op."""
    for pid in ("thrust5_detector_cfar", "thrust5_detector_ml", "thrust5_detector_raddetnet"):
        st = apply_preset(PRESETS_BY_ID[pid])
        assert st["corpus_environment"]["params"]["domain"] == "cfr", pid
        assert st["corpus_environment"]["params"]["manifest"] == DEMO_CFR_CORPUS, pid
        for bid in ("rffe", "interconnect", "dechirp", "thermal_noise", "impairment",
                    "if_hpf", "quantizer"):
            assert st[bid]["enabled"], (pid, bid)
        # The chain that wrote the corpus: its own data-driven interconnect (what
        # 'default' resolves to on this path, never the boxcar placeholder) and the
        # quantizer's automatic gain (a fixed full scale zeroes a physical cube).
        assert st["interconnect"]["params"]["case"] == "default", pid
        assert st["quantizer"]["params"]["full_scale"] == 0, pid
        assert st["quantizer"]["params"]["bits"] == 12, pid


def test_the_demo_cfr_corpus_is_on_this_machine_with_its_channel_sidecars():
    """The live chain only exists if the corpus actually carries `.cfr.npy` sidecars;
    without them the runner falls back to ADC replay and the screens quietly become
    the old ones. Checked on the manifest's own first test frame, not by globbing."""
    import json
    from pathlib import Path

    from webapp.corpus_catalog import CORPUS_MANIFESTS, REPO_ROOT

    assert DEMO_CFR_CORPUS in CORPUS_MANIFESTS
    manifest_path = REPO_ROOT / DEMO_CFR_CORPUS
    manifest = json.loads(manifest_path.read_text())
    first = manifest["files"]["test"][0]
    assert (manifest_path.parent / first).exists()
    assert (manifest_path.parent / (Path(first).stem + ".cfr.npy")).exists()


def test_thrust5_ab_arms_move_a_front_end_knob_not_the_corpus():
    """Before the live chain, the only Thrust 5 A/B that moved a front-end setting did
    it by swapping to a SECOND pre-generated corpus (the retired bridge preset). Now
    the arms differ by a knob on the same stored frames -- which is the claim the
    screen makes, so the test pins that the A/B never touches the source again."""
    for pid in ("thrust5_detector_cfar", "thrust5_detector_raddetnet"):
        p = PRESETS_BY_ID[pid]
        # 3-bit, not 4 (item 2, hostile-expert read, 2026-09-23): swept {2, 3, 4, 6}
        # bits on both detectors over the same 5 frames; 4-bit made RADDetNet's hit
        # count go UP relative to 12-bit (10 -> 11), reading backwards on screen --
        # 3-bit is the largest depth at which BOTH detectors lose hits.
        assert p.ab == ("quantizer", "bits", 3), pid
        assert "12-bit" in p.ab_label_a and "3-bit" in p.ab_label_b
    ml = PRESETS_BY_ID["thrust5_detector_ml"]
    assert ml.ab == ("if_hpf", "corner_range_m", 25.0)
    for pid in ("thrust5_detector_cfar", "thrust5_detector_ml", "thrust5_detector_raddetnet"):
        p = PRESETS_BY_ID[pid]
        st_a, st_b = apply_preset(p), apply_preset(p, arm="b")
        assert st_a["corpus_environment"] == st_b["corpus_environment"], pid


def test_the_retired_bridge_preset_is_gone():
    """It was replaced by the live bits knob (owner 2026-09-23). Its two corpora may
    stay on disk; the preset must not, or the demo offers two answers to one question."""
    assert "thrust5_bridge_adc_bits_vs_detections" not in PRESETS_BY_ID
    assert not any("bridge" in p.id for p in PRESETS)


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
