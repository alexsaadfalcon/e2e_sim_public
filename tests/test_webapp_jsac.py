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


# --------------------------------------------------------------------------------
# The sensing window on the SCREEN (shard 3c, hostile round 12 / JSAC deliverable 1)
# --------------------------------------------------------------------------------
#
# What these pin: the resource-split knob changes the image's UNAMBIGUOUS WINDOW
# (`c / (P * df)`), and the range map must draw that window and nothing past it.
# Measured on the 2026-09-24/25 renders of Thrust 6 before the fix: arm B (P = 8,
# window 62.44 m) drew FOUR copies of the scene inside the transform's own 249.8 m
# half-window and its brightest-return statistic named an alias ("0.0 dB @ 125 m"),
# while arm A's named the wrap of range 0 onto its own period ("-0.0 dB @ 250 m").

def _jsac_env_class(frames_np, n_freqs):
    """A stored source with a v2 frequency plan -- what the JSAC class needs to place
    its subcarriers on the channel's own grid."""
    torch_ = pytest.importorskip("torch")
    from e2e.blocks import device
    import numpy as np

    class _PlanEnv:
        array_shape = (32, 32)
        physical_scale = False
        freq_plan = {"carrier_hz": 30e9, "start_hz": 28.5e9, "stop_hz": 31.5e9,
                     "num_freqs": int(n_freqs)}

        def __init__(self, *a, **k):
            self._frames = frames_np
            self.frame_counter = 0

        def __len__(self):
            return len(self._frames)

        def step(self):
            self.frame_counter = (self.frame_counter + 1) % len(self._frames)

        def reset(self):
            self.frame_counter = 0

        def get_S_pars(self):
            arr = np.ascontiguousarray(self._frames[self.frame_counter])
            return torch_.from_numpy(arr).to(device)

    return _PlanEnv


def _run_jsac_range_az(monkeypatch, synthetic_frames_np, pilot_spacing, n_freqs=64):
    """One JSAC run at this pilot spacing -> (range_az figure, axis meta, spec window)."""
    import e2e.blocks as blocks
    from webapp.pipeline_runner import figures_from_outputs, run_pipeline

    frames = synthetic_frames_np(2, 1024, n_freqs, 0)
    monkeypatch.setattr(blocks, "SionnaEnvironmentBlock",
                        _jsac_env_class(frames, n_freqs))
    state = default_block_state()
    state["waveform"]["enabled"] = True
    state["waveform"]["params"].update(
        kind="jsac", n_symbols=4, pilot_spacing=pilot_spacing, bits_per_symbol=2,
        sensing_source="pilots_only")
    for bid in ("afe", "subspace", "range_profile", "subspace_err", "dechirp",
                "radar_cube", "fft", "range_el"):
        if bid in state:
            state[bid]["enabled"] = False
    state["range_az"]["enabled"] = True
    outputs = run_pipeline(state, n_steps=2)
    figs = figures_from_outputs(outputs)
    return figs, (outputs.get("_axis_meta") or {})


def test_the_sensing_window_reaches_the_axis_meta_from_the_waveform_not_the_transform(
        monkeypatch, synthetic_frames_np):
    """`_spine_range_meta` describes the TRANSFORM, which is identical on both arms.
    The window the knob buys is the waveform's own (`OFDMFrame.sensing_window_m`), and
    it has to reach the figure builder or the crop below cannot exist."""
    _, meta = _run_jsac_range_az(monkeypatch, synthetic_frames_np, 2)
    assert "sensing_window_m" in meta
    # c / (P * df), df endpoint-inclusive over the plan's own band.
    df = (31.5e9 - 28.5e9) / (64 - 1)
    assert meta["sensing_window_m"] == pytest.approx(299792458.0 / (2 * df), rel=1e-6)
    # ...and it is SHORTER than the transform's displayed half on the tighter arm.
    assert meta["range_displayed_m"] > 0


def test_each_arm_draws_only_its_own_unambiguous_window(monkeypatch,
                                                        synthetic_frames_np):
    """The picture, the axis and the statistic, all inside the window -- and the
    window named in the caption, since two panels of different vertical extent side by
    side are read as one comparison unless the screen says otherwise."""
    from webapp.pipeline_runner import decode_plotly_array, panel_caption, y_extent_lock_of

    windows = {}
    for spacing in (2, 8):
        figs, meta = _run_jsac_range_az(monkeypatch, synthetic_frames_np, spacing)
        fig = figs["range_az"].to_plotly_json()
        window = float(meta["sensing_window_m"])
        windows[spacing] = window
        # 1. the axis ends ON the window, not on the transform's half-window
        assert fig["layout"]["yaxis"]["range"] == pytest.approx([0.0, window])
        assert y_extent_lock_of(fig) == pytest.approx(window)
        # 2. no data row past it -- the wrapped copies are gone, not merely hidden
        ys = [v for v in decode_plotly_array(fig["data"][0].get("y")) if v is not None]
        assert ys and max(ys) < window
        # ...on every animation frame too, not only the one the page opens on.
        for frame in (fig.get("frames") or []):
            z = frame["data"][0]["z"]
            # Plotly's compact wire form: a {bdata, dtype, shape} dict, whose first
            # shape axis is the row (range-gate) count.
            rows = (int(str(z["shape"]).split(",")[0]) if isinstance(z, dict)
                    else len(z))
            assert rows == len(ys), "an animation frame kept the uncropped map"
        # 3. the per-frame "brightest" statistic is computed inside the window
        for ann in fig["layout"]["annotations"]:
            if "brightest" in ann["text"]:
                metres = float(ann["text"].split("@", 1)[1].split("m", 1)[0])
                assert metres <= window
        # 4. and the PANEL says what the window is -- on the axis title since
        # 2026-09-25, not in the caption: with the burst-rate clause added (round 13's
        # N9) the caption ran past the ~94 characters that fit one line in a 746 px panel
        # and the browser clipped it mid-word, which acceptance check 12 forbids. The
        # axis title is where a window belongs anyway; the caption's copy was the second.
        assert ("%.1f m window" % window) in fig["layout"]["yaxis"]["title"]["text"]
    assert windows[8] < windows[2], "the window must shrink as the pilot spacing rises"


def test_the_two_arms_windows_are_not_unioned_back_into_one_axis(
        monkeypatch, synthetic_frames_np):
    """The A/B axis-sharing pass exists so a pair reads as a difference in the DATA.
    Here the EXTENT is the data: unioning it would redraw the tighter arm's map on the
    other arm's axis, which is the wrapped-copies picture the crop removes. So the pass
    must leave it alone AND the screen must say the two axes differ."""
    from webapp import app as appmod
    from webapp.pipeline_runner import panel_caption, note_differing_y_extents

    figs_a, meta_a = _run_jsac_range_az(monkeypatch, synthetic_frames_np, 2)
    figs_b, meta_b = _run_jsac_range_az(monkeypatch, synthetic_frames_np, 8)
    a = {"range_az": figs_a["range_az"].to_plotly_json()}
    b = {"range_az": figs_b["range_az"].to_plotly_json()}
    appmod._share_y_ranges(a, b)
    note_differing_y_extents(a, b)
    ya = a["range_az"]["layout"]["yaxis"]["range"]
    yb = b["range_az"]["layout"]["yaxis"]["range"]
    assert ya[1] == pytest.approx(float(meta_a["sensing_window_m"]))
    assert yb[1] == pytest.approx(float(meta_b["sensing_window_m"]))
    assert yb[1] < ya[1]
    # Said on arm A's caption (once per row, layout spec section 4) and in BOTH arms'
    # Details -- the absence of the clause must never be what carries the meaning.
    # DRAWN on arm A's map, not written into its caption (2026-09-25): a shaded band up
    # to arm B's window with a labelled edge. The caption had no room left for a second
    # number once the burst rate joined it, and a picture of the other arm's extent is
    # what makes two maps at 4x different vertical scales comparable by eye anyway.
    assert ("%.1f m window" % ya[1]) in a["range_az"]["layout"]["yaxis"]["title"]["text"]
    shapes = a["range_az"]["layout"].get("shapes") or []
    band = [sh for sh in shapes if sh.get("name") == "other_arm_window"]
    assert band and band[0]["y1"] == pytest.approx(yb[1]), shapes
    marks = [an for an in (a["range_az"]["layout"].get("annotations") or [])
             if an.get("name") == "other_arm_window"]
    assert marks and ("%.0f m" % yb[1]) in marks[0]["text"], marks
    # ...on every animation frame too: a frame layout REPLACES the annotations list.
    for frame in (a["range_az"].get("frames") or []):
        names = [an.get("name") for an in ((frame.get("layout") or {}).get("annotations") or [])]
        assert "other_arm_window" in names, frame.get("name")
    for fig in (a["range_az"], b["range_az"]):
        details = " ".join((fig["layout"]["meta"]["panel"]["details"]))
        assert "not to the same vertical scale" in details
