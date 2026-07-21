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


def _chanest_trial(snr_db, seed):
    """One pilot-only channel-estimation trial: synthetic multipath channel ->
    OFDM pilots -> LS and MMSE estimate. Returns (mse_ls, mse_mmse) over the
    active band, both vs. the true channel used to generate the pilots."""
    modem = OFDMModem(fft_size=128, cp_len=32, n_active=104, pilot_spacing=4,
                      bits_per_symbol=2)
    freqs = np.linspace(28.5e9, 31.5e9, modem.fft_size)
    H_true = ch.synthetic_multipath_cfr(freqs, n_taps=6,
                                        rng=np.random.default_rng(seed))
    n_symbols = 16
    bits = random_bits(n_symbols * modem.data_bits_per_symbol_block, seed=seed)
    _, tx_freq = modem.modulate(bits, n_symbols)
    rx_freq, _ = ch.apply_channel(tx_freq, H_true, snr_db, rng_seed=seed + 1000)

    rx_pilots = modem.extract_pilots(rx_freq)
    tx_pilots = modem.pilot_grid(n_symbols)
    H_ls = ch.ls_estimate(rx_pilots, tx_pilots, modem.pilot_idx, modem.fft_size)
    H_mmse = ch.mmse_estimate(rx_pilots, tx_pilots, modem.pilot_idx, modem.fft_size, snr_db)

    active = modem.active_idx
    return ch.channel_mse(H_ls, H_true, active), ch.channel_mse(H_mmse, H_true, active)


@pytest.mark.parametrize("snr_db", [0.0, 2.0, 4.0])
def test_mmse_estimate_beats_ls_at_low_snr_monte_carlo(snr_db):
    """Regression for the circular-prior bug: deriving sigma_H^2 from the same
    few noisy pilots it then shrinks was a high-variance, sometimes-negative
    prior that made the 'MMSE' estimator *worse* than plain LS at low SNR. With
    the prior pooled over every pilot and every OFDM symbol in the frame, the
    empirical-Bayes shrinkage must (on average, over >=30 fixed-seed trials)
    beat LS at low SNR."""
    trials = [_chanest_trial(snr_db, seed) for seed in range(30)]
    mse_ls, mse_mmse = zip(*trials)
    assert np.mean(mse_mmse) <= np.mean(mse_ls)


def test_mmse_estimate_converges_to_ls_at_high_snr_monte_carlo():
    """At high SNR the pooled empirical-Bayes prior sees sigma_n2 -> 0, so the
    Wiener gain -> 1 and the shrinkage vanishes: MMSE should track LS closely
    (never materially worse) rather than diverge from it."""
    trials = [_chanest_trial(25.0, seed) for seed in range(30)]
    mse_ls, mse_mmse = zip(*trials)
    mean_ls, mean_mmse = np.mean(mse_ls), np.mean(mse_mmse)
    assert mean_mmse <= mean_ls * 1.05
    assert mean_mmse == pytest.approx(mean_ls, rel=0.05)


# --------------------------------------------------------------------------- equalization bias

def test_mmse_equalize_unbiased_matches_zf_decisions():
    """Regression for the biased-MMSE-vs-hard-slicer bug: the raw scalar-MMSE
    filter shrinks amplitude (E[w H] = b_k < 1), but the demapper slices
    against a unit-power constellation, so uncorrected MMSE decisions differ
    from ZF's even though they shouldn't. Bias-correcting the equalized symbols
    (the default, ``unbiased=True``) must reproduce ZF's bit decisions exactly
    on the same noise realization -- algebraically ``w / b_k == 1/H_k``.
    Meanwhile the RAW (biased) MMSE output must have lower per-symbol MSE than
    ZF vs. the true transmitted symbols -- MMSE's genuine advantage over ZF,
    visible only before the bias correction / hard slicing."""
    modem = OFDMModem(fft_size=64, cp_len=16, n_active=52, pilot_spacing=8,
                      bits_per_symbol=4)                      # 16-QAM
    freqs = np.linspace(28.5e9, 31.5e9, modem.fft_size)
    H_true = ch.synthetic_multipath_cfr(freqs, n_taps=4, max_delay_s=6e-9,
                                        rng=np.random.default_rng(0),
                                        rician_k_db=10.0)
    n_symbols = 32
    snr_db = 12.0
    tx_bits = random_bits(n_symbols * modem.data_bits_per_symbol_block, seed=0)
    _, tx_freq = modem.modulate(tx_bits, n_symbols)
    rx_freq, _ = ch.apply_channel(tx_freq, H_true, snr_db, rng_seed=1000)

    rx_pilots = modem.extract_pilots(rx_freq)
    tx_pilots = modem.pilot_grid(n_symbols)
    H_est = ch.ls_estimate(rx_pilots, tx_pilots, modem.pilot_idx, modem.fft_size)

    eq_zf = modem.extract_data(ch.zf_equalize(rx_freq, H_est))
    eq_mmse_unbiased = modem.extract_data(ch.mmse_equalize(rx_freq, H_est, snr_db))
    eq_mmse_biased = modem.extract_data(
        ch.mmse_equalize(rx_freq, H_est, snr_db, unbiased=False))

    rx_bits_zf = qam_demod(eq_zf.reshape(-1), modem.bits_per_symbol, modem.const)
    rx_bits_mmse = qam_demod(eq_mmse_unbiased.reshape(-1), modem.bits_per_symbol, modem.const)

    # (i) identical decisions on the same noise realization
    assert torch.equal(rx_bits_zf, rx_bits_mmse)
    assert ch.ber(tx_bits, rx_bits_zf) == ch.ber(tx_bits, rx_bits_mmse)

    # (ii) the actual MMSE advantage: lower estimation MSE (before bias
    # correction / hard slicing) than ZF against the true transmitted symbols
    tx_data = modem.extract_data(tx_freq)
    mse_zf = torch.mean(torch.abs(eq_zf - tx_data) ** 2).item()
    mse_mmse_biased = torch.mean(torch.abs(eq_mmse_biased - tx_data) ** 2).item()
    assert mse_mmse_biased < mse_zf


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


# --------------------------------------------------------------------------- frame_to_cfr

def test_frame_to_cfr_element_int_matches_manual_reshape():
    """element=0 must match the old 'flatten everything, take row 0' logic used
    by the pre-refactor load_or_synthesize_cfr / _cfr_from_state."""
    rng = np.random.default_rng(0)
    n_rx, n_tx, chirp, F = 6, 2, 1, 40
    frame = (rng.standard_normal((n_rx, n_tx, chirp, F))
              + 1j * rng.standard_normal((n_rx, n_tx, chirp, F))).astype(np.complex64)
    freqs = _freqs(F)

    H = ch.frame_to_cfr(frame, freqs, element=0)

    flat = frame.reshape(-1, frame.shape[-1])
    expected = flat[0]                        # old manual extraction (rx=0,tx=0,chirp=0)
    assert H.shape == (F,)
    assert H.dtype == torch.complex64
    assert H.device.type == device.type
    np.testing.assert_allclose(H.cpu().numpy(), expected, atol=1e-5)


def test_frame_to_cfr_element_none_returns_all_rows():
    """element=None must return every spatial row, matching the old per-element
    loop used by main_isac._radar_s_pars (tx=0, chirp=0 slice per row)."""
    rng = np.random.default_rng(1)
    n_rx, n_tx, chirp, F = 5, 3, 1, 32
    frame = (rng.standard_normal((n_rx, n_tx, chirp, F))
              + 1j * rng.standard_normal((n_rx, n_tx, chirp, F))).astype(np.complex64)
    freqs = _freqs(F)

    out = ch.frame_to_cfr(frame, freqs, element=None)
    assert out.shape == (n_rx, F)
    assert out.dtype == torch.complex64
    assert out.device.type == device.type

    expected_rows = frame.reshape(n_rx, -1, F)[:, 0, :]   # old manual per-element loop
    np.testing.assert_allclose(out.cpu().numpy(), expected_rows, atol=1e-5)


def test_frame_to_cfr_already_flat_frame():
    """A frame already shaped [n_rx, F] (no tx/chirp axes) is handled as-is."""
    rng = np.random.default_rng(2)
    n_rx, F = 4, 20
    frame = (rng.standard_normal((n_rx, F)) + 1j * rng.standard_normal((n_rx, F))).astype(np.complex64)
    freqs = _freqs(F)
    out = ch.frame_to_cfr(frame, freqs, element=None)
    np.testing.assert_allclose(out.cpu().numpy(), frame, atol=1e-5)


def test_load_or_synthesize_uses_v2_freq_plan_for_auto_band(monkeypatch):
    """When the iterator exposes v2 `freq_plan` metadata and no explicit
    `src_band` is given, the frame's true band (from the metadata) is used
    automatically instead of guessing that the pkl spans `freqs`."""
    n_src = 128
    src_lo, src_hi = 24e9, 36e9                       # the v2-declared true band
    src_freqs = np.linspace(src_lo, src_hi, n_src)
    tau = 0.15e-9
    chan0 = np.exp(-2j * np.pi * src_freqs * tau).astype(np.complex64)
    frame = np.stack([chan0, np.zeros_like(chan0)], axis=0)   # [2, F]

    class _V2FakeIter(_FakeIter):
        @property
        def freq_plan(self):
            return {"carrier_hz": (src_lo + src_hi) / 2, "start_hz": src_lo,
                    "stop_hz": src_hi, "num_freqs": n_src}

    from e2e.environment import sionna_iterator as si
    monkeypatch.setattr(si, "SionnaMunichIterator", lambda *a, **k: _V2FakeIter(frame))

    freqs = np.linspace(27e9, 33e9, 64)
    H, src = ch.load_or_synthesize_cfr("munich", freqs)
    assert src == "sionna:munich"
    expected = np.exp(-2j * np.pi * freqs * tau).astype(np.complex64)
    err = np.mean(np.abs(H.cpu().numpy() - expected) ** 2)
    assert err < 1e-3

    # explicit src_band still wins over the v2 metadata
    H_explicit, _ = ch.load_or_synthesize_cfr("munich", freqs, src_band=(0.0, 1.0))
    err_explicit = np.mean(np.abs(H_explicit.cpu().numpy() - expected) ** 2)
    assert err_explicit > err


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
