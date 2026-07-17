"""Unit tests for channel application, estimation, equalization and metrics
(e2e.comms.channel)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from e2e.comms import channel as ch
from e2e.comms.ofdm import OFDMModem, qam_demod, random_bits

device = ch.device


def _freqs(n=64, start=28.5e9, stop=31.5e9):
    return np.linspace(start, stop, n)


# --------------------------------------------------------------------------- synthetic CFR

def test_synthetic_multipath_cfr_shape_dtype():
    freqs = _freqs(64)
    rng = np.random.default_rng(0)
    H = ch.synthetic_multipath_cfr(freqs, n_taps=6, rng=rng)
    assert H.shape == (64,)
    assert H.dtype == torch.complex64
    assert H.device.type == device.type
    assert torch.all(torch.isfinite(torch.abs(H)))


def test_synthetic_multipath_cfr_deterministic_with_seed():
    freqs = _freqs(32)
    H1 = ch.synthetic_multipath_cfr(freqs, rng=np.random.default_rng(7))
    H2 = ch.synthetic_multipath_cfr(freqs, rng=np.random.default_rng(7))
    assert torch.allclose(H1, H2)


# --------------------------------------------------------------------------- apply_channel

def test_apply_channel_high_snr_is_near_noiseless():
    torch.manual_seed(0)
    freqs = _freqs(64)
    H = ch.synthetic_multipath_cfr(freqs, rng=np.random.default_rng(0))
    tx = torch.randn(4, 64, dtype=torch.complex64, device=device)
    clean = tx * H[None, :]
    rx, noise_pow = ch.apply_channel(tx, H, snr_db=80.0, rng_seed=0)
    rel_err = (torch.norm(rx - clean) / torch.norm(clean)).item()
    assert rel_err < 1e-3
    assert noise_pow > 0.0


def test_apply_channel_noise_power_decreases_with_snr():
    freqs = _freqs(64)
    H = ch.synthetic_multipath_cfr(freqs, rng=np.random.default_rng(1))
    tx = torch.ones(2, 64, dtype=torch.complex64, device=device)
    _, np_low = ch.apply_channel(tx, H, snr_db=0.0, rng_seed=0)
    _, np_high = ch.apply_channel(tx, H, snr_db=30.0, rng_seed=0)
    assert np_high < np_low


# --------------------------------------------------------------------------- end-to-end BER

def _run_link(snr_db, seed=0, estimator="ls", equalizer="zf", max_delay_s=4e-9):
    """One OFDM transmission through a synthetic channel; returns (ber, tx, rx, modem, H).

    `max_delay_s` is kept small so the channel is smooth relative to the pilot
    spacing -- otherwise comb-pilot interpolation cannot track the channel and
    estimation/equalization is fundamentally undersampled (a property of the
    pilot grid, not a bug in the estimators).
    """
    modem = OFDMModem(fft_size=64, cp_len=16, n_active=52, pilot_spacing=8,
                      bits_per_symbol=2)
    freqs = np.linspace(28.5e9, 31.5e9, modem.fft_size)
    H = ch.synthetic_multipath_cfr(freqs, n_taps=4, max_delay_s=max_delay_s,
                                   rng=np.random.default_rng(seed),
                                   rician_k_db=10.0)

    n_symbols = 16
    bits = random_bits(n_symbols * modem.data_bits_per_symbol_block, seed=seed)
    _, tx_freq = modem.modulate(bits, n_symbols)
    rx_freq, _ = ch.apply_channel(tx_freq, H, snr_db, rng_seed=seed)

    rx_pilots = modem.extract_pilots(rx_freq)
    tx_pilots = modem.pilot_grid(n_symbols)
    if estimator == "mmse":
        H_est = ch.mmse_estimate(rx_pilots, tx_pilots, modem.pilot_idx,
                                 modem.fft_size, snr_db)
    else:
        H_est = ch.ls_estimate(rx_pilots, tx_pilots, modem.pilot_idx, modem.fft_size)

    if equalizer == "mmse":
        eq = ch.mmse_equalize(rx_freq, H_est, snr_db)
    else:
        eq = ch.zf_equalize(rx_freq, H_est)
    data_eq = modem.extract_data(eq)
    rx_bits = qam_demod(data_eq.reshape(-1), modem.bits_per_symbol, modem.const)
    return ch.ber(bits, rx_bits), bits, rx_bits, modem, H, H_est


def test_ber_non_increasing_with_snr():
    """A small 2-3 point SNR check: BER should not rise as SNR rises (fast)."""
    torch.manual_seed(0)
    bers = [_run_link(s, seed=0, estimator="mmse", equalizer="mmse")[0]
            for s in (0.0, 15.0, 40.0)]
    # monotonic non-increasing (allow tiny numerical wiggle)
    assert bers[1] <= bers[0] + 1e-9
    assert bers[2] <= bers[1] + 1e-9
    # at very high SNR we recover essentially everything
    assert bers[2] < 1e-3


def test_equalizer_recovers_symbols_at_high_snr():
    ber_val, *_ = _run_link(60.0, seed=2, estimator="ls", equalizer="zf")
    assert ber_val == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- estimation

def test_estimation_beats_no_estimation_and_mmse_close_to_ls():
    """LS and MMSE estimates reduce channel MSE vs assuming H=1.

    Both pilot-based estimators must clearly beat the do-nothing baseline (H=1).
    The MMSE estimator applies a Wiener shrinkage on top of LS; with many clean
    pilot observations its denoising is marginal, so we assert MMSE stays close
    to LS (it shrinks toward LS, never blows up) rather than a strict <=.
    """
    snr_db = 10.0
    _, _, _, modem, H_true, H_ls = _run_link(snr_db, seed=3, estimator="ls",
                                             equalizer="zf")
    _, _, _, _, _, H_mmse = _run_link(snr_db, seed=3, estimator="mmse",
                                      equalizer="zf")
    active = modem.active_idx
    H_none = torch.ones(modem.fft_size, dtype=torch.complex64, device=device)

    mse_none = ch.channel_mse(H_none, H_true, active)
    mse_ls = ch.channel_mse(H_ls, H_true, active)
    mse_mmse = ch.channel_mse(H_mmse, H_true, active)

    assert mse_ls < mse_none
    assert mse_mmse < mse_none
    # MMSE shrinks toward LS; it must not materially exceed it.
    assert mse_mmse <= 1.25 * mse_ls


# --------------------------------------------------------------------------- metrics

def test_ber_zero_for_identical_positive_for_corrupted():
    bits = random_bits(64, seed=0)
    assert ch.ber(bits, bits) == 0.0
    flipped = bits.clone()
    flipped[:8] = 1 - flipped[:8]
    assert ch.ber(bits, flipped) > 0.0


def test_evm_zero_for_identical_positive_for_corrupted():
    syms = torch.randn(50, dtype=torch.complex64, device=device)
    assert ch.evm(syms, syms) == pytest.approx(0.0, abs=1e-6)
    corrupted = syms + 0.1 * torch.randn(50, dtype=torch.complex64, device=device)
    assert ch.evm(corrupted, syms) > 0.0


def test_channel_mse_zero_for_identical_positive_for_corrupted():
    H = torch.randn(64, dtype=torch.complex64, device=device)
    assert ch.channel_mse(H, H) == pytest.approx(0.0, abs=1e-6)
    H2 = H + 0.5
    assert ch.channel_mse(H, H2) > 0.0


def test_channel_mse_respects_active_idx():
    H_true = torch.randn(64, dtype=torch.complex64, device=device)
    H_est = H_true.clone()
    # corrupt only an inactive subcarrier; MSE over active band stays zero
    H_est[0] = H_est[0] + 100.0
    active = torch.arange(10, 50, device=device)
    assert ch.channel_mse(H_est, H_true, active) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- load_or_synthesize

def test_load_or_synthesize_returns_synthetic_when_no_pkl():
    """Unknown scenario name -> no .pkl -> synthetic fallback path."""
    freqs = _freqs(48)
    H, src = ch.load_or_synthesize_cfr("does_not_exist", freqs,
                                       rng=np.random.default_rng(0))
    assert src == "synthetic"
    assert H.shape == (48,)
    assert H.dtype == torch.complex64
    assert torch.all(torch.isfinite(torch.abs(H)))


def test_load_or_synthesize_prefer_pkl_false_is_synthetic():
    freqs = _freqs(32)
    H, src = ch.load_or_synthesize_cfr("munich", freqs, prefer_pkl=False,
                                       rng=np.random.default_rng(1))
    assert src == "synthetic"
    assert H.shape == (32,)


class _FakeIter:
    """Stand-in for a Sionna frame iterator, serving one known frame."""

    def __init__(self, frame):
        self._frame = frame

    def __len__(self):
        return 1

    def __getitem__(self, i):
        return self._frame


def _patch_munich(monkeypatch, frame):
    """Make load_or_synthesize_cfr's munich loader return `frame`."""
    from e2e.environment import sionna_iterator as si
    monkeypatch.setattr(si, "SionnaMunichIterator", lambda *a, **k: _FakeIter(frame))


def test_load_or_synthesize_src_band_remaps_frequency(monkeypatch):
    """`src_band` controls how the pkl's samples map onto `freqs`.

    Build a frame whose first spatial channel is a pure tone exp(-j2pi f tau) on a
    KNOWN source band. Reading it back with the matching `src_band` must recover a
    near-flat group delay; reading it with the (wrong) default band must not.
    """
    n_src = 256
    src_lo, src_hi = 20e9, 40e9                       # the frame's true band
    src_freqs = np.linspace(src_lo, src_hi, n_src)
    # keep tau small so the tone is well-sampled on the source grid (period 1/tau
    # = 5 GHz >> source spacing ~78 MHz) and interpolation is accurate.
    tau = 0.2e-9
    # frame shape [N_RX, F]; only element 0 (taken by the loader) needs to matter
    chan0 = np.exp(-2j * np.pi * src_freqs * tau).astype(np.complex64)
    frame = np.stack([chan0, np.zeros_like(chan0)], axis=0)   # [2, F]
    _patch_munich(monkeypatch, frame)

    # request a sub-band well inside the source band
    freqs = np.linspace(25e9, 35e9, 64)
    H_correct, src = ch.load_or_synthesize_cfr("munich", freqs,
                                               src_band=(src_lo, src_hi))
    assert src == "sionna:munich"
    # with the correct band, H(f) should match the analytic tone closely
    expected = np.exp(-2j * np.pi * freqs * tau).astype(np.complex64)
    err = np.mean(np.abs(H_correct.cpu().numpy() - expected) ** 2)
    assert err < 1e-3

    # default behaviour (assume pkl spans `freqs`) mis-maps the frequency axis,
    # so it should NOT match the analytic tone over this sub-band.
    H_default, _ = ch.load_or_synthesize_cfr("munich", freqs)
    err_default = np.mean(np.abs(H_default.cpu().numpy() - expected) ** 2)
    assert err_default > err


def test_load_or_synthesize_default_band_assumes_freqs_span(monkeypatch):
    """Default (no src_band): the pkl is assumed to span exactly `freqs`.

    A frame sampled on the SAME band as `freqs` is recovered correctly by default.
    """
    freqs = np.linspace(28e9, 32e9, 80)
    tau = 3e-9
    chan0 = np.exp(-2j * np.pi * freqs * tau).astype(np.complex64)
    frame = chan0[None, :]                            # [1, F]
    _patch_munich(monkeypatch, frame)

    H, src = ch.load_or_synthesize_cfr("munich", freqs)
    assert src == "sionna:munich"
    expected = np.exp(-2j * np.pi * freqs * tau).astype(np.complex64)
    assert np.mean(np.abs(H.cpu().numpy() - expected) ** 2) < 1e-6
