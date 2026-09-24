"""Unit tests for the optional communications pipeline blocks
(e2e.comms.blocks). These follow the apply(state_dict)->dict convention."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from e2e.comms import blocks as comm_blocks
from e2e.comms.blocks import ModemBlock, BERBlock

device = comm_blocks.device


def _freqs(n=64, start=28.5e9, stop=31.5e9):
    return np.linspace(start, stop, n)


@pytest.fixture
def state_dict(n_freqs):
    """A minimal state_dict shaped like Simulation builds, on the library device.

    Mirrors tests/test_blocks.py: s_pars is [32, 32, 1, F].
    """
    torch.manual_seed(0)
    s_pars = torch.randn(32, 32, 1, n_freqs, dtype=torch.cfloat, device=device)
    return {"s_pars": s_pars}


def test_modem_block_apply_returns_dict_with_expected_keys(state_dict, n_freqs):
    freqs = _freqs(n_freqs)
    block = ModemBlock(freqs, n_symbols=8, fft_size=64, cp_len=16, n_active=52,
                       pilot_spacing=8, bits_per_symbol=2, snr_db=30.0,
                       equalizer="mmse", estimator="ls", seed=0)
    out = block.apply(state_dict)
    assert isinstance(out, dict)
    for key in ("comm_tx_bits", "comm_rx_bits", "comm_data_eq",
                "comm_H_est", "comm_H_true"):
        assert key in out
    # tx/rx bit streams are equal length and on the library device
    assert out["comm_tx_bits"].numel() == out["comm_rx_bits"].numel()
    assert out["comm_tx_bits"].device.type == device.type
    assert torch.all(torch.isfinite(torch.abs(out["comm_data_eq"])))


def test_modem_block_ber_in_valid_range(state_dict, n_freqs):
    """Over the (uncontrolled, random) s_pars channel the block still produces a
    well-formed BER in [0, 1]. End-to-end recovery on a clean channel is checked
    in test_modem_block_uses_precomputed_H_sc."""
    freqs = _freqs(n_freqs)
    block = ModemBlock(freqs, n_symbols=8, snr_db=40.0, equalizer="mmse",
                       estimator="ls", seed=1)
    out = block.apply(state_dict)
    n = out["comm_tx_bits"].numel()
    errs = (out["comm_tx_bits"] != out["comm_rx_bits"]).float().mean().item()
    assert n > 0
    assert 0.0 <= errs <= 1.0


def test_ber_block_emits_ber_and_evm_keys(state_dict, n_freqs):
    """BERBlock follows apply(state_dict)->dict and emits BER + EVM."""
    freqs = _freqs(n_freqs)
    modem = ModemBlock(freqs, n_symbols=8, snr_db=30.0, bits_per_symbol=2, seed=0)
    state = dict(state_dict)
    state.update(modem.apply(state))
    # BERBlock needs the QAM order in the state dict to compute EVM
    state["comm_bits_per_symbol"] = 2

    out = BERBlock().apply(state)
    assert isinstance(out, dict)
    assert "ber" in out
    assert 0.0 <= out["ber"] <= 1.0
    assert "evm" in out
    assert out["evm"] >= 0.0
    assert np.isfinite(out["evm"])


def test_ber_block_ber_only_without_bits_per_symbol(state_dict, n_freqs):
    """With neither a true reference nor comm_bits_per_symbol, BERBlock reports BER only.

    BERBlock computes EVM when it can: against the true transmitted symbols
    (comm_tx_data) if present, else decision-directed if comm_bits_per_symbol is
    present. Drop BOTH to exercise the BER-only fallback. (ModemBlock now advertises
    comm_tx_data and comm_bits_per_symbol, so we remove them explicitly here.)"""
    freqs = _freqs(n_freqs)
    modem = ModemBlock(freqs, n_symbols=8, snr_db=30.0, seed=0)
    state = dict(state_dict)
    state.update(modem.apply(state))
    state.pop("comm_tx_data", None)
    state.pop("comm_bits_per_symbol", None)

    out = BERBlock().apply(state)
    assert "ber" in out
    assert "evm" not in out


def test_comms_blocks_compose_inside_simulation(make_env_block):
    """ModemBlock -> BERBlock run as first-class downstream stages in Simulation.

    Exercises the state-dict chaining: BERBlock consumes the bits ModemBlock emitted
    earlier in the same step. Proves the comms blocks are pipeline-citizens, not just
    standalone helpers.
    """
    from e2e.simulation import Simulation
    from e2e.blocks import AdaOjaBlock, FFTBlock, SubspaceErrorBlock

    n_freqs = 64
    env = make_env_block(n_frames=2, n_freqs=n_freqs)
    freqs = _freqs(n_freqs)
    modem = ModemBlock(freqs, n_symbols=4, fft_size=64, cp_len=16, n_active=52,
                       pilot_spacing=8, bits_per_symbol=2, snr_db=30.0, seed=0)
    # The comms head is a TAP off the frequency-domain side of the spine, so it
    # travels in `comms_head=` rather than in `downstream_blocks` (2026-09-24): the
    # downstream blocks run after the whole spine, by which point the chain is in the
    # cube domain and `s_pars` has been dropped at the crossing. The radar products
    # stay downstream, which is the point -- both heads, one chain, no exclusion rule.
    sim = Simulation(
        env,
        [FFTBlock(), SubspaceErrorBlock()],
        k=16,
        subspace_block=AdaOjaBlock(1024, 16),
        comms_head=[modem, BERBlock()],
    )
    out = sim.run(n_steps=2)
    # comm products accumulated over both steps, alongside the radar products
    assert len(out["ber"]) == 2
    assert "comm_rx_bits" in out and "fft" in out and "subspace_err" in out
    assert all(0.0 <= b <= 1.0 for b in out["ber"])
    # EVM now available because ModemBlock advertises comm_bits_per_symbol downstream
    assert len(out["evm"]) == 2 and all(np.isfinite(e) for e in out["evm"])


def test_cfr_from_state_matches_manual_flatten(state_dict, n_freqs):
    """_cfr_from_state (now delegating to ch.frame_to_cfr) must reproduce the old
    'flatten everything, take row 0, then resample to subcarriers' result."""
    from e2e.comms import channel as ch
    from e2e.comms.blocks import _cfr_from_state

    freqs = _freqs(n_freqs)
    modem = ModemBlock(freqs, n_symbols=2, fft_size=64).modem

    H = _cfr_from_state(state_dict, modem, freqs)

    s_pars = state_dict["s_pars"]
    flat = s_pars.reshape(-1, s_pars.shape[-1])
    cfr_dense = flat[0]                                    # old manual extraction
    carrier = float(np.mean(freqs))
    df = float(freqs[1] - freqs[0])
    expected = ch.cfr_to_subcarriers(cfr_dense, freqs, modem.fft_size, carrier, df)

    assert H.shape == expected.shape
    torch.testing.assert_close(H, expected)


def test_modem_block_uses_precomputed_H_sc(n_freqs):
    """If H_sc is in the state dict, ModemBlock uses it directly (bypasses s_pars)."""
    freqs = _freqs(n_freqs)
    block = ModemBlock(freqs, n_symbols=4, fft_size=64, snr_db=50.0, seed=0)
    H_sc = torch.ones(block.modem.fft_size, dtype=torch.complex64, device=device)
    out = block.apply({"H_sc": H_sc})
    # an all-ones (flat, noiseless-ish) channel should recover bits essentially perfectly
    errs = (out["comm_tx_bits"] != out["comm_rx_bits"]).float().mean().item()
    assert errs == pytest.approx(0.0, abs=1e-9)
    assert torch.allclose(out["comm_H_true"], H_sc)


def test_modem_block_defaults_to_element0_combining(state_dict, n_freqs):
    """`combining` defaults to 'element0' (the historical SISO-tap behavior) and
    never emits 'comm_array_gain_db' -- see test_comms_beamforming.py for the
    'mrc'/'subspace' combining modes."""
    freqs = _freqs(n_freqs)
    block = ModemBlock(freqs, n_symbols=4, fft_size=64, snr_db=20.0, seed=0)
    assert block.combining == "element0"
    out = block.apply(state_dict)
    assert "comm_array_gain_db" not in out


def test_modem_block_rejects_unknown_combining_mode():
    freqs = _freqs(64)
    with pytest.raises(ValueError, match="combining"):
        ModemBlock(freqs, n_symbols=4, fft_size=64, combining="bogus")


def test_modem_block_reset_reproduces_noise_sequence(state_dict, n_freqs):
    """reset() rewinds the per-frame noise counter: without it, repeated runs of
    the same Simulation draw a different AWGN sequence (breaking seed
    reproducibility); after reset() the sequence repeats exactly."""
    freqs = _freqs(n_freqs)
    block = ModemBlock(freqs, n_symbols=4, fft_size=64, snr_db=10.0,
                       bits_per_symbol=2, seed=3)
    out1 = block.apply(dict(state_dict))
    out2 = block.apply(dict(state_dict))
    # frames differ within a run (independent noise per frame)
    assert not torch.equal(out1["comm_rx_bits"], out2["comm_rx_bits"]) or \
        not torch.allclose(out1["comm_data_eq"], out2["comm_data_eq"])
    block.reset()
    out1b = block.apply(dict(state_dict))
    torch.testing.assert_close(out1["comm_data_eq"], out1b["comm_data_eq"])


@pytest.mark.parametrize("combining", ["element0", "subspace"])
def test_modem_block_reads_dict_shaped_freq_plan(state_dict, n_freqs, combining):
    """A `freq_plan` threaded into the state dict has the DICT shape that
    SionnaIterator/the env block expose ({'carrier_hz': ...}). ModemBlock must read
    it as a dict, not via attribute access. Regression: _cfr_from_state /
    _combine_spatial used fp.carrier_hz and would AttributeError the moment any
    freq_plan was present in the state."""
    freqs = _freqs(n_freqs)
    st = dict(state_dict)
    st["freq_plan"] = {"carrier_hz": 30e9, "start_hz": 28.5e9,
                       "stop_hz": 31.5e9, "num_freqs": n_freqs}
    if combining == "subspace":
        st["U"] = torch.linalg.qr(
            torch.randn(1024, 16, dtype=torch.cfloat, device=device))[0]
    block = ModemBlock(freqs, n_symbols=4, fft_size=64, snr_db=15.0,
                       combining=combining)
    out = block.apply(st)   # must not raise AttributeError
    assert "comm_rx_bits" in out
    if combining != "element0":
        assert "comm_array_gain_db" in out


# --------------------------------------------------------------------------------
# The comms head as a CONSUMER of the chain (one-chain contract section 1.4)
# --------------------------------------------------------------------------------
def test_modem_auto_disables_its_own_awgn_when_the_chain_injected_noise(state_dict,
                                                                        n_freqs):
    """ONE noise source per chain. `ModemBlock` used to inject its own AWGN
    unconditionally -- a THIRD floor beside the front end's Friis cascade and the link
    budget's kTBF (the contract's own count). On the one spine that makes every "turn
    this knob and watch both products move" claim false, because half the comms floor
    is a constructor argument that no knob touches.

    `add_noise=None` (the default) reads `state['noise_injected_by']`, the seam the
    front end stamps when it draws. The test is STRUCTURAL rather than a power
    comparison: with the draw off, two applies of the same frame are bit-identical,
    which a quieter-but-still-present noise source would not be.
    """
    modem = ModemBlock(_freqs(n_freqs), n_symbols=4, fft_size=64, snr_db=10.0, seed=0)

    chain_noisy = dict(state_dict)
    chain_noisy["noise_injected_by"] = "frontend"
    assert modem.noise_enabled(chain_noisy) is False
    assert modem.noise_enabled(dict(state_dict)) is True

    modem.reset()
    a = modem.apply(dict(chain_noisy))
    modem.reset()
    b = modem.apply(dict(chain_noisy))
    assert torch.equal(a["comm_data_eq"], b["comm_data_eq"]), (
        "two applies of the same frame differ, so something is still drawing noise")
    assert a["comm_noise_source"] == "frontend"

    # ...and with the chain silent, the block is its own link simulator as before:
    # the per-frame counter makes consecutive frames draw independently.
    modem.reset()
    c = modem.apply(dict(state_dict))
    d = modem.apply(dict(state_dict))
    assert not torch.equal(c["comm_data_eq"], d["comm_data_eq"])
    assert c["comm_noise_source"] == "modem"


def test_modem_measures_the_equaliser_snr_when_the_chain_is_the_noise_source(
        state_dict, n_freqs):
    """The reported SNR must be a MEASUREMENT on that path, not the constructor value.

    When this block draws its own noise it knows the SNR exactly and reports what it
    used. When the CHAIN drew it, nothing in the pipeline knows the post-combining
    SNR, so it is estimated from the pilot residual -- and a number typed into a preset
    must not be able to reach a card as if it had been measured.
    """
    modem = ModemBlock(_freqs(n_freqs), n_symbols=8, fft_size=64, snr_db=7.0, seed=0)
    own = modem.apply(dict(state_dict))
    assert own["comm_snr_db"] == pytest.approx(7.0)

    chain = dict(state_dict)
    chain["noise_injected_by"] = "frontend"
    measured = modem.apply(chain)["comm_snr_db"]
    # A clean chain frame: the measured SNR must NOT come back as the configured 7 dB.
    assert not (measured == pytest.approx(7.0, abs=1e-6)), (
        "the reported SNR is the constructor's, so it was not measured")
