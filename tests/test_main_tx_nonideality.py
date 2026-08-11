"""Tests for e2e.main.main_tx_nonideality -- the ideal-vs-non-ideal TX PA A/B.

Fast: tiny FFT size, few symbols, few oversample points, a short backoff list
(never the default 0..12 dB sweep). No figures are written (show=False everywhere).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import matplotlib
matplotlib.use("Agg")            # matches the example's own convention

import e2e.main.main_tx_nonideality as m


# Small, fast config shared by most tests -- deliberately far from the module's
# defaults (fft_size=64/n_symbols=48/oversample=8) to keep the suite quick.
_FAST_KW = dict(fft_size=32, cp_len=8, n_active=20, pilot_spacing=4,
                bits_per_symbol=4, n_symbols=8, oversample=4, snr_db=40.0)


def _run(backoff_db_list, **overrides):
    kw = dict(_FAST_KW)
    kw.update(overrides)
    return m.main(backoff_db_list=backoff_db_list, show=False, **kw)


# --------------------------------------------------------------------------------
# oversample/undersample round trip
# --------------------------------------------------------------------------------
def test_oversample_undersample_round_trip_is_exact():
    """Zero-padding a spectrum then IFFT/FFT-ing at the padded length is a lossless
    sinc interpolation -- undoing it must recover the original grid bit-for-bit
    (up to float rounding), independent of anything downstream in main()."""
    torch.manual_seed(0)
    n_symbols, fft_size, cp_len, L = 5, 32, 8, 4
    tx_freq = (torch.randn(n_symbols, fft_size, dtype=torch.complex64))
    time_up, off = m._oversample_time(tx_freq, fft_size, cp_len, L)
    assert time_up.shape == (n_symbols, (fft_size + cp_len) * L)
    recovered = m._undersample_freq(time_up, fft_size, cp_len, L, off)
    assert torch.allclose(recovered, tx_freq, atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------------
# end-to-end run
# --------------------------------------------------------------------------------
def test_main_runs_end_to_end_and_returns_expected_keys():
    r = _run(backoff_db_list=[0, 6, 12])
    expected_keys = {
        "source", "papr_db", "backoff_db_list", "aggressive_backoff_db",
        "evm_ideal", "evm_nonideal", "acpr_ideal_db", "acpr_nonideal_db",
        "acpr_asymmetry_db", "peak_range_ideal_m", "peak_range_nonideal_m",
        "psl_ideal_db", "psl_nonideal_db",
    }
    assert expected_keys.issubset(r.keys())
    assert r["backoff_db_list"] == [0, 6, 12]
    assert len(r["evm_ideal"]) == 3
    assert len(r["evm_nonideal"]) == 3
    assert np.all(np.isfinite(r["evm_ideal"]))
    assert np.all(np.isfinite(r["evm_nonideal"]))
    assert np.isfinite(r["papr_db"])


def test_main_does_not_touch_disk_when_show_is_false(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "FIG_DIR", str(tmp_path))
    _run(backoff_db_list=[0, 12])
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------------
# physics: this is the actual point of the example
# --------------------------------------------------------------------------------
def test_measured_papr_is_in_the_expected_ofdm_range():
    """Dense-QAM OFDM PAPR is a well-known ~8-12 dB ballpark; a wildly different
    number here would mean the oversampled peak/mean measurement is broken."""
    r = _run(backoff_db_list=[0])
    assert 5.0 < r["papr_db"] < 15.0


def test_nonideal_evm_exceeds_ideal_at_aggressive_backoff():
    """At the most compressed (lowest) backoff, the PA-distorted arm must be
    measurably worse than the distortion-free arm -- the headline claim."""
    r = _run(backoff_db_list=[0, 6, 12])
    assert r["evm_nonideal"][0] > r["evm_ideal"][0] * 2.0


def test_nonideal_evm_improves_monotonically_with_backoff():
    """Backing off further from saturation must not make the non-ideal arm worse
    (allow a hair of floating-point slack, no allowance for a real reversal)."""
    r = _run(backoff_db_list=[0, 3, 6, 9, 12])
    evm = r["evm_nonideal"]
    for prev, nxt in zip(evm[:-1], evm[1:]):
        assert nxt <= prev + 1e-6


def test_ideal_evm_is_near_zero_and_flat_across_backoff():
    """The ideal (no-PA) arm sees the same channel/AWGN at every backoff, and
    equalization cancels the shared drive-level scale factor exactly (see the
    module docstring's algebra) -- so ideal EVM should be small AND essentially
    constant across the whole sweep."""
    r = _run(backoff_db_list=[0, 4, 8, 12], snr_db=45.0)
    evm = np.asarray(r["evm_ideal"])
    assert np.all(evm < 0.05)
    assert (evm.max() - evm.min()) < 1e-3


def test_acpr_asymmetry_is_near_zero_for_the_memoryless_pa():
    """`TxPA` is memoryless, so its spectral regrowth is structurally symmetric
    about the carrier -- the upper/lower ACPR gap should be small, and much
    smaller than the regrowth itself (non-ideal ACPR well above the ideal
    (near-numerical-floor) ACPR)."""
    r = _run(backoff_db_list=[0])
    assert r["acpr_asymmetry_db"] < 2.0
    lo_ideal, hi_ideal = r["acpr_ideal_db"]
    lo_nonideal, hi_nonideal = r["acpr_nonideal_db"]
    assert lo_nonideal > lo_ideal + 30.0
    assert hi_nonideal > hi_ideal + 30.0


def test_radar_side_sensing_cost_is_reported_and_finite():
    r = _run(backoff_db_list=[0])
    assert np.isfinite(r["peak_range_ideal_m"])
    assert np.isfinite(r["peak_range_nonideal_m"])
    assert np.isfinite(r["psl_ideal_db"])
    assert np.isfinite(r["psl_nonideal_db"])


def test_aggressive_backoff_defaults_to_minimum_of_sweep():
    r = _run(backoff_db_list=[2, 5, 9])
    assert r["aggressive_backoff_db"] == 2
