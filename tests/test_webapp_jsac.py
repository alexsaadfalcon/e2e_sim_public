"""The waveform CLASSES on the webapp's one chain: fmcw / ofdm / jsac.

These are webapp-level tests: the physics oracles for the classes themselves (O1-O7,
bit-exact FMCW parity, the BER floor, the division-noise closed form) live in
`tests/test_waveform_classes.py` and are not repeated here. What this file pins is the
integration a screen depends on -- that the dropdown offers the owner's three classes,
that the runner builds the spine each class implies, that the products a class cannot
produce are refused BY NAME rather than rendered empty, and that the JSAC preset's
numbers are computed from the frame rather than typed into a card.

Every run here uses the SYNTHETIC frames from `make_env_block` where it can; the two
tests that need the real munich Ka plan say so and are skipped when it is absent.
"""

import pytest

torch = pytest.importorskip("torch")

from webapp.demo_presets import PRESETS_BY_ID, apply_preset          # noqa: E402
from webapp.pipeline_registry import BLOCKS_BY_ID, default_block_state  # noqa: E402


JSAC_PRESET = "thrust6_jsac_resource_split"


def test_the_dropdown_offers_the_owners_three_classes():
    """"Sensing (FMCW), comms (OFDM), and JSAC (hybrid)" -- the owner's words, and the
    three rows of `e2e.comms.ofdm_isac.waveform_chain_spec`. `wideband` stays reachable
    as the legacy random-wideband source but is not one of the three."""
    kinds = next(ps for ps in BLOCKS_BY_ID["waveform"].params if ps.key == "kind").choices
    assert kinds[:3] == ["fmcw", "ofdm", "jsac"], kinds
    assert "wideband" in kinds


def test_every_ofdm_parameter_the_runner_reads_exists_in_the_registry():
    """The runner reads five OFDM/JSAC knobs off the waveform block. A missing one is
    not a soft failure: `_p` would raise deep inside a run, on stage, after the click."""
    keys = {ps.key for ps in BLOCKS_BY_ID["waveform"].params}
    assert {"n_symbols", "pilot_spacing", "bits_per_symbol", "sensing_source",
            "combining"} <= keys


def test_the_jsac_preset_is_the_only_one_whose_mixing_block_is_not_a_dechirp():
    p = PRESETS_BY_ID[JSAC_PRESET]
    state = apply_preset(p)
    assert state["waveform"]["enabled"] and state["waveform"]["params"]["kind"] == "jsac"
    assert not state["dechirp"]["enabled"], (
        "the jsac class brings its own mixing block; the dechirp bridge must be off")
    # The v1.1 scope (JSAC build spec 3.9 option (c)), as state rather than as prose.
    for bid in ("afe", "subspace", "range_profile", "subspace_err", "radar_cube"):
        assert not state[bid]["enabled"], bid


def test_the_ab_knob_is_the_resource_split():
    p = PRESETS_BY_ID[JSAC_PRESET]
    assert p.ab is not None and p.ab[:2] == ("waveform", "pilot_spacing")
    a = apply_preset(p)["waveform"]["params"]
    b = apply_preset(p, arm="b")["waveform"]["params"]
    assert a["sensing_source"] == "pilots_only", (
        "the split only bites when the image is formed from the DATA symbols' comb -- "
        "under 'preamble' symbol 0 is all-pilot at every spacing and both arms render "
        "the same picture")
    assert b["pilot_spacing"] > a["pilot_spacing"]


def test_the_ofdm_class_refuses_the_sensing_products_by_name():
    """`ofdm` is `jsac` with the mixer omitted -- a comms receiver never forms a cube --
    so a sensing product on that class is a configuration error, and saying which one is
    what makes the dropdown a demonstration rather than a trap."""
    from webapp.pipeline_runner import PipelineError, run_pipeline

    state = default_block_state()
    state["waveform"]["enabled"] = True
    state["waveform"]["params"]["kind"] = "ofdm"
    state["range_az"]["enabled"] = True
    with pytest.raises(PipelineError) as e:
        run_pipeline(state, n_steps=1)
    msg = str(e.value)
    assert "range_az" in msg and "jsac" in msg


def test_an_ofdm_run_refuses_a_source_with_no_frequency_plan(monkeypatch,
                                                             synthetic_frames_np):
    """The class places its subcarriers ON the stored channel's own grid, so a source
    with no plan is refused naming the reason -- not silently given an invented one.
    A legacy .pkl (the 3.5 GHz munich trace, F93) is exactly such a source, so this is
    a reachable state and not a hypothetical one."""
    import e2e.blocks as blocks
    from webapp.pipeline_runner import PipelineError, run_pipeline

    class _NoPlanEnv:
        """A stored source with no v2 metadata -- the legacy .pkl case."""
        array_shape = (32, 32)
        freq_plan = None
        physical_scale = None

        def __init__(self, *a, **k):
            self._frames = synthetic_frames_np
            self.i = 0

        def get_S_pars(self):
            return torch.as_tensor(self._frames[0])

        def step(self):
            self.i += 1

        def reset(self):
            self.i = 0

    monkeypatch.setattr(blocks, "SionnaEnvironmentBlock", _NoPlanEnv)
    state = default_block_state()
    state["waveform"]["enabled"] = True
    state["waveform"]["params"]["kind"] = "jsac"
    with pytest.raises(PipelineError) as e:
        run_pipeline(state, n_steps=1)
    assert "freq_plan" in str(e.value) or "frequency grid" in str(e.value)


def test_the_display_symbol_follows_the_sensing_source():
    """MEASURED 2026-09-24, and the reason this function exists: under
    `sensing_source='pilots_only'` the frame keeps symbol 0 as a FULL all-pilot preamble
    and puts the comb on the data symbols, so a screen showing symbol 0 renders the same
    picture at every pilot spacing (both arms measured 72.86-72.88 dB peak-median, to the
    digit). The split lives on symbol 1."""
    from e2e.comms.ofdm_isac import waveform_chain_spec
    from webapp.pipeline_runner import _OFDMTxCfg, _display_symbol_for

    plan = {"carrier_hz": 30e9, "start_hz": 28.5e9, "stop_hz": 31.5e9, "num_freqs": 64}
    pre = waveform_chain_spec("jsac", _OFDMTxCfg(), freq_plan=plan, n_symbols=4,
                              sensing_source="preamble")
    comb = waveform_chain_spec("jsac", _OFDMTxCfg(), freq_plan=plan, n_symbols=4,
                               pilot_spacing=4, sensing_source="pilots_only")
    assert _display_symbol_for(pre) == 0
    assert _display_symbol_for(comb) == 1
    # And the claim the docstring rests on, checked against the frame itself rather
    # than remembered: symbol 0's reference is the whole preamble under BOTH sources.
    assert torch.equal(comb.frame.reference_grid()[0], comb.frame.tx_grid[0])
    assert (comb.frame.reference_grid()[1] == 0).any()


def test_the_run_notes_compute_the_resource_split_from_the_frame():
    """Both halves of the trade, in the run notes, derived from the frame that ran --
    never typed into the preset. The direction is the test: window DOWN, rate UP."""
    from e2e.comms.ofdm_isac import waveform_chain_spec
    from webapp.pipeline_runner import _OFDMTxCfg, _waveform_run_notes

    plan = {"carrier_hz": 30e9, "start_hz": 28.5e9, "stop_hz": 31.5e9, "num_freqs": 64}
    windows, rates = [], []
    for spacing in (2, 8):
        spec = waveform_chain_spec("jsac", _OFDMTxCfg(), freq_plan=plan, n_symbols=4,
                                   pilot_spacing=spacing, sensing_source="pilots_only")
        notes = " ".join(_waveform_run_notes(spec))
        assert "pilot spacing %d" % spacing in notes
        assert "Burst data rate" in notes and "Sensing window" in notes
        windows.append(spec.notes["sensing_window_m"])
        rates.append(spec.notes["data_rate_bps"])
    assert windows[1] < windows[0], "the sensing window must SHRINK as P rises"
    assert rates[1] > rates[0], "the data rate must RISE as P rises"


def test_no_card_claims_ofdm_drives_the_front_end_into_its_clamp():
    """F98 / JSAC build spec 3.7.1, as a test on the shipped text. The claim is false at
    the shipped operating point and false in the wrong direction (measured on munich Ka
    frame 0: FMCW as shipped 36.67 dB of LNA-input PAPR, this JSAC frame 23.68 dB, the
    all-pilot preamble bit-identical to FMCW's 36.67). A card may quote the measurement;
    it may not tell the story the other way round."""
    p = PRESETS_BY_ID[JSAC_PRESET]
    # What the presenter SAYS -- the `do_not_say` list is deliberately excluded, since
    # naming the false claim is its whole job.
    spoken = " ".join([p.blurb] + list(p.say)).lower()
    assert "papr" in spoken, "the measured PAPR pair belongs on this card"
    assert "36.67" in spoken and "23.68" in spoken, (
        "quote both numbers at the preset's own operating point, or neither")
    for claim in ("ofdm's papr drives", "ofdm makes the front end clip",
                  "the ofdm waveform is what makes"):
        assert claim not in spoken
    # And the card must WARN against it, because it is the first thing an RF reader
    # expects to hear and it is false here by 13 dB in the wrong direction.
    assert any("papr" in s.lower() or "clamp" in s.lower() for s in p.do_not_say)


def test_the_screen_note_names_the_symbol_the_runner_actually_shows():
    """The card and the code must not disagree about WHICH symbol is on screen. This is
    the pairing that already went wrong once: the note was written for symbol 0 while the
    runner (correctly) shows symbol 1 under `pilots_only`, and nothing but a reader would
    have caught it."""
    from e2e.comms.ofdm_isac import waveform_chain_spec
    from webapp.pipeline_runner import _OFDMTxCfg, _display_symbol_for

    p = PRESETS_BY_ID[JSAC_PRESET]
    params = apply_preset(p)["waveform"]["params"]
    plan = {"carrier_hz": 30e9, "start_hz": 28.5e9, "stop_hz": 31.5e9, "num_freqs": 64}
    spec = waveform_chain_spec(
        params["kind"], _OFDMTxCfg(), freq_plan=plan,
        n_symbols=params["n_symbols"], pilot_spacing=params["pilot_spacing"],
        bits_per_symbol=params["bits_per_symbol"],
        sensing_source=params["sensing_source"], combining=params["combining"])
    shown = _display_symbol_for(spec)
    assert ("IMAGE SHOWN IS SYMBOL %d" % shown) in p.screen_note, (
        "the screen note must name symbol %d, the one the runner displays" % shown)


def test_the_mixing_node_says_which_mixing_block_this_class_uses():
    """A class is (source, MIXING MODE, products), so the mixing block is not the same
    block on all three -- and a diagram reading "dechirp" on a JSAC run tells the room
    the frame was dechirped when it was DIVIDED by the transmitted grid (read on the
    first JSAC card render, 2026-09-24). `ofdm` has no mixing block at all, and no cube
    with it: the node pair is drawn disabled, which is the same statement the runner
    makes when it refuses a sensing product on that class."""
    from webapp import block_diagram as bd

    def nodes(kind):
        state = apply_preset(PRESETS_BY_ID[JSAC_PRESET])
        state["waveform"]["params"]["kind"] = kind
        els = bd.build_elements(state)
        return {e["data"]["id"]: e for e in els if "position" in e}

    assert "symbol" in nodes("jsac")["dechirp"]["data"]["label"]
    assert "dechirp" in nodes("fmcw")["dechirp"]["data"]["label"]
    ofdm = nodes("ofdm")
    assert "none" in ofdm["dechirp"]["data"]["label"]
    for nid in ("dechirp", "cube"):
        assert "disabled" in ofdm[nid]["classes"], nid
    # ... and the FMCW chain still draws both as structural.
    fmcw = nodes("fmcw")
    for nid in ("dechirp", "cube"):
        assert "disabled" not in fmcw[nid]["classes"], nid
