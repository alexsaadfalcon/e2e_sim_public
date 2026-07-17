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
    sim = Simulation(
        env,
        [FFTBlock(), modem, BERBlock(), SubspaceErrorBlock()],
        d=16,
        subspace_block=AdaOjaBlock(1024, 16),
    )
    out = sim.run(n_steps=2)
    # comm products accumulated over both steps, alongside the radar products
    assert len(out["ber"]) == 2
    assert "comm_rx_bits" in out and "fft" in out and "subspace_err" in out
    assert all(0.0 <= b <= 1.0 for b in out["ber"])
    # EVM now available because ModemBlock advertises comm_bits_per_symbol downstream
    assert len(out["evm"]) == 2 and all(np.isfinite(e) for e in out["evm"])


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
