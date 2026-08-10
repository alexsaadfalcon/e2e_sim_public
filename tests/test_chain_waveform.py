"""Tests for e2e.chain.waveform: WaveformBlock / TxPABlock / ModulateBlock."""

import numpy as np
import pytest
import torch

from e2e import frames
from e2e.chain.waveform import ModulateBlock, TxPABlock, WaveformBlock
from e2e.circuit.tx_pa import TxPA


def _cfr_state(synthetic_frame_np, torch_device, n_rx=16, n_freqs=32):
    """A minimal state dict carrying only 's_pars', matching Simulation's shape
    convention [n_rx, n_tx, n_chirp, n_freqs]."""
    arr = synthetic_frame_np(n_rx=n_rx, n_freqs=n_freqs)
    s_pars = torch.from_numpy(arr).to(torch_device)
    return {"s_pars": s_pars}


# --------------------------------------------------------------------------- WaveformBlock

def test_waveform_shape_dtype_device(torch_device):
    wf = WaveformBlock(kind="fmcw", n_tx=2, n_chirp=3, n_t=16, sample_rate=4e9,
                       bw=1e9, chirp_duration=4e-9)
    out = wf.apply({})
    tx_wave = out["tx_wave"]
    assert tx_wave.shape == (2, 3, 16)
    assert tx_wave.dtype == torch.complex64
    assert tx_wave.device.type == torch_device.type


def test_waveform_n_t_derived_from_chirp_duration_and_sample_rate():
    wf = WaveformBlock(kind="fmcw", sample_rate=1e9, chirp_duration=16e-9, n_tx=1, n_chirp=1)
    out = wf.apply({})
    assert out["tx_wave"].shape[-1] == 16


def test_waveform_picks_up_s_pars_device(torch_device):
    wf = WaveformBlock(kind="narrowband", n_t=8)
    state = {"s_pars": torch.zeros(4, 1, 1, 8, dtype=torch.complex64, device=torch_device)}
    out = wf.apply(state)
    assert out["tx_wave"].device.type == torch_device.type


def test_waveform_unknown_kind_raises():
    with pytest.raises(ValueError):
        WaveformBlock(kind="not-a-real-waveform")


# --------------------------------------------------------------------------- TxPABlock

def test_fmcw_constant_envelope_stays_constant_through_txpa(torch_device):
    # FMCWSignal produces a unit-magnitude chirp (product of two unit complex
    # exponentials); TxPA's AM/AM is a function of |x| only, so a constant input
    # envelope maps to a single constant output magnitude everywhere.
    wf = WaveformBlock(kind="fmcw", n_t=256, sample_rate=4e9, bw=1e9, chirp_duration=64e-9)
    tx_wave = wf.apply({}).get("tx_wave")
    tx_wave = tx_wave.to(torch_device)

    in_mag = torch.abs(tx_wave)
    assert torch.allclose(in_mag, torch.ones_like(in_mag), atol=1e-4), \
        "FMCWSignal is expected to be constant-envelope (|x| == 1) before the PA"

    pa_block = TxPABlock()  # default TxPAConfig: 20 dB small-signal gain, a_sat=1.0
    out = pa_block.apply({"tx_wave": tx_wave})
    out_mag = torch.abs(out["tx_wave"])

    # Constant envelope is PRESERVED (AM/AM depends only on |x|, which never varies).
    assert torch.allclose(out_mag, out_mag[0, 0, 0] * torch.ones_like(out_mag), atol=1e-4)
    # ... but COMPRESSED relative to the (deeply-saturating) small-signal linear gain:
    # G = 10**(20/20) = 10, so a linear amp would produce |y| == 10; the Rapp soft
    # limiter's actual output stays close to a_sat=1.0 instead. This is the
    # "documented compression factor" -- the ratio is far below the linear gain.
    small_signal_gain = 10 ** (20.0 / 20.0)
    compression_ratio = float(out_mag[0, 0, 0]) / (small_signal_gain * float(in_mag[0, 0, 0]))
    assert compression_ratio < 0.5, "expected heavy compression at a_sat=1.0, gain=20dB"


def test_two_tone_envelope_varies_through_txpa(torch_device):
    # A two-tone signal's envelope beats between 0 and 2 -- unlike the FMCW chirp,
    # this is NOT constant-envelope, so TxPA (whose AM/AM acts pointwise on |x|)
    # must reproduce that variation in the output magnitude.
    n_t = 256
    t = torch.arange(n_t, dtype=torch.float32, device=torch_device) / 4e9
    two_tone = (torch.exp(2j * np.pi * 50e6 * t) + torch.exp(2j * np.pi * 120e6 * t)).to(torch.complex64)
    tx_wave = two_tone.view(1, 1, -1)

    in_mag = torch.abs(tx_wave)
    assert in_mag.std() > 1e-2, "two-tone envelope should vary substantially before the PA"

    pa_block = TxPABlock()
    out = pa_block.apply({"tx_wave": tx_wave})
    out_mag = torch.abs(out["tx_wave"])
    assert out_mag.std() > 1e-3, "envelope variation should survive (not be flattened) by TxPA"


# --------------------------------------------------------------------------- ModulateBlock

def test_modulate_sets_domain_and_shape(synthetic_frame_np, torch_device):
    state = _cfr_state(synthetic_frame_np, torch_device, n_rx=16, n_freqs=32)
    wf = WaveformBlock(kind="fmcw", n_t=32, sample_rate=4e9, bw=1e9, chirp_duration=8e-9)
    state.update(wf.apply(state))
    state.update(TxPABlock().apply(state))

    mb = ModulateBlock()
    out = mb.apply(state)
    assert out["signal_domain"] == frames.DOMAIN_CFR
    assert out["s_pars"].shape == state["s_pars"].shape
    assert out["s_pars"].dtype == state["s_pars"].dtype
    assert out["s_pars"].device.type == torch_device.type


def test_modulate_applies_pa_ripple(synthetic_frame_np, torch_device):
    state = _cfr_state(synthetic_frame_np, torch_device, n_rx=8, n_freqs=16)
    wf = WaveformBlock(kind="fmcw", n_t=16, sample_rate=4e9, bw=1e9, chirp_duration=4e-9)
    state.update(wf.apply(state))

    tx_pa = TxPA()
    mb_no_ripple = ModulateBlock()
    mb_ripple = ModulateBlock(tx_pa=tx_pa, bandwidth_hz=4e9)

    out_no_ripple = mb_no_ripple.apply(state)["s_pars"]
    out_ripple = mb_ripple.apply(state)["s_pars"]
    assert not torch.allclose(out_no_ripple, out_ripple), \
        "PA frequency ripple should perturb the modulated frame"


# --------------------------------------------------------------------------- fast path

def test_modulate_fast_path_no_tx_wave_is_bit_identical(synthetic_frame_np, torch_device):
    state = _cfr_state(synthetic_frame_np, torch_device)
    original = state["s_pars"].clone()

    mb = ModulateBlock()
    out = mb.apply(state)
    assert "s_pars" not in out, "fast path must not even touch s_pars"
    assert out["signal_domain"] == frames.DOMAIN_CFR
    assert torch.equal(state["s_pars"], original)


def test_modulate_fast_path_ideal_flag_is_bit_identical(synthetic_frame_np, torch_device):
    state = _cfr_state(synthetic_frame_np, torch_device, n_rx=16, n_freqs=32)
    original = state["s_pars"].clone()
    wf = WaveformBlock(kind="fmcw", n_t=32, sample_rate=4e9, bw=1e9, chirp_duration=8e-9)
    state.update(wf.apply(state))  # tx_wave IS present...

    mb = ModulateBlock(ideal=True)  # ...but ideal=True still takes the fast path
    out = mb.apply(state)
    assert "s_pars" not in out
    assert torch.equal(state["s_pars"], original)


# --------------------------------------------------------------------------- capabilities

def test_capability_declarations_match_docstrings():
    wf_caps = WaveformBlock.frame_capabilities
    assert wf_caps.domain == frames.DOMAIN_TX_TIME
    assert wf_caps.emits_domain == frames.DOMAIN_TX_TIME
    assert wf_caps.is_bridge is False  # source, not a bridge (domain == emits_domain)

    pa_caps = TxPABlock.frame_capabilities
    assert pa_caps.domain == frames.DOMAIN_TX_TIME
    assert pa_caps.emits_domain is None
    assert pa_caps.is_bridge is False

    mod_caps = ModulateBlock.frame_capabilities
    assert mod_caps.domain == frames.DOMAIN_TX_TIME
    assert mod_caps.emits_domain == frames.DOMAIN_CFR
    assert mod_caps.is_bridge is True
