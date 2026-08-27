"""Tests for `e2e.chain.link_budget` -- the absolute power reference for the whole chain.

These constants set the absolute SNR of every target in every generated corpus, so the
tests here are deliberately of the "check it against something that is not this code" kind
rather than the "check it against itself" kind.

The ORACLE is an independent RF review that hand-computed the same budget from the
datasheet and the radar equation, without seeing this implementation:

    thermal floor, B = fs = 25 MHz, NF = 15 dB          -> -85 dBm
    10 dBsm car at 30 m, 0 dBi elements, ADC-sample SNR -> -33 dB
    post-integration (their own TDM caveat applied)     -> ~+12 dB

"Anything within +/-5 dB of that is plausible; tens of dB off means a bug." -- so the
tolerances below are theirs, not ones chosen to make the test pass.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from e2e.chain import link_budget as lb
from e2e.radar_config import PRESETS, RadarConfig

CFG = PRESETS["ti_iwr1443"]


# --------------------------------------------------------------------------------
# Oracle: an independent hand calculation of the same budget
# --------------------------------------------------------------------------------
def test_thermal_floor_matches_independent_hand_calculation():
    """-174 dBm/Hz + 10log10(25e6) + 15 = -85 dBm."""
    floor_dbm = 10.0 * math.log10(lb.thermal_noise_power_w(CFG)) + 30.0
    assert floor_dbm == pytest.approx(-85.0, abs=0.5)


def test_target_snr_matches_independent_hand_calculation():
    """A 10 dBsm car at 30 m with 0 dBi elements: -33 dB at the ADC sample."""
    snr = lb.expected_target_snr_db(CFG, range_m=30.0, rcs_dbsm=10.0, gain_dbi=0.0)
    assert snr == pytest.approx(-33.0, abs=5.0)


def test_post_integration_snr_is_detectable_but_not_absurd():
    """Range + Doppler coherent gain must lift that car above the floor without
    producing a number no real radar achieves. The reviewer's band, after applying their
    own TDM caveat (192 chirps / 3 TX = 64 per virtual channel), is ~+12 dB."""
    snr = (lb.expected_target_snr_db(CFG, 30.0, 10.0, gain_dbi=0.0)
           + lb.coherent_processing_gain_db(CFG))
    assert 5.0 < snr < 30.0, f"post-integration SNR {snr:.1f} dB is outside any plausible range"


# --------------------------------------------------------------------------------
# The physics has to behave like physics
# --------------------------------------------------------------------------------
def test_snr_falls_as_the_fourth_power_of_range():
    """Doubling the range must cost 12.04 dB. This is the one relation that would break
    silently if someone 'fixed' the exponent."""
    near = lb.expected_target_snr_db(CFG, 20.0, 10.0)
    far = lb.expected_target_snr_db(CFG, 40.0, 10.0)
    assert (near - far) == pytest.approx(12.041, abs=0.01)


def test_snr_tracks_rcs_one_for_one():
    a = lb.expected_target_snr_db(CFG, 30.0, 0.0)
    b = lb.expected_target_snr_db(CFG, 30.0, 10.0)
    assert (b - a) == pytest.approx(10.0, abs=0.01)


def test_snr_tracks_tx_power_one_for_one():
    quiet = RadarConfig(**{**CFG.__dict__, "tx_power_dbm": 2.0})
    loud = RadarConfig(**{**CFG.__dict__, "tx_power_dbm": 12.0})
    delta = (lb.expected_target_snr_db(loud, 30.0, 10.0)
             - lb.expected_target_snr_db(quiet, 30.0, 10.0))
    assert delta == pytest.approx(10.0, abs=0.01)


def test_noise_figure_raises_the_floor_one_for_one():
    a = lb.thermal_noise_power_w(CFG, noise_figure_db=10.0)
    b = lb.thermal_noise_power_w(CFG, noise_figure_db=20.0)
    assert 10.0 * math.log10(b / a) == pytest.approx(10.0, abs=0.01)


# --------------------------------------------------------------------------------
# The injected noise must BE the budgeted noise, and must not know about the scene
# --------------------------------------------------------------------------------
def test_injected_noise_power_matches_the_budget(torch_device):
    """What `add_thermal_noise` puts in must equal `k*T*B*F`, measured, not assumed."""
    adc = torch.zeros(4, 32, 512, dtype=torch.complex64, device=torch_device)
    out = lb.add_thermal_noise(adc, CFG, seed=0)
    measured = float(torch.mean(torch.abs(out) ** 2))
    expected = lb.thermal_noise_power_w(CFG)
    assert measured == pytest.approx(expected, rel=0.05)


def test_injected_noise_is_independent_of_the_cube(torch_device):
    """THE F35 TEST, in its most direct form.

    A cube 20 dB stronger must receive the SAME absolute noise. `_add_awgn`'s
    target-relative convention fails this by construction -- its noise scales with the
    peak scatterer, which is exactly the mechanism that pinned target SNR and made every
    physics improvement invisible.
    """
    base = torch.ones(2, 8, 256, dtype=torch.complex64, device=torch_device)
    loud = base * 10.0                     # +20 dB
    n_quiet = float(torch.mean(torch.abs(
        lb.add_thermal_noise(torch.zeros_like(base), CFG, seed=1)) ** 2))
    n_loud = float(torch.mean(torch.abs(
        lb.add_thermal_noise(torch.zeros_like(loud), CFG, seed=1)) ** 2))
    assert n_quiet == pytest.approx(n_loud, rel=1e-6)


def test_target_snr_responds_to_target_strength(torch_device):
    """And the consequence: with an absolute floor, a 20 dB stronger target IS 20 dB more
    detectable. Under the peak-relative regime this delta was 0.00 dB (F35)."""
    ratios = []
    for gain in (1.0, 10.0):
        adc = torch.zeros(2, 8, 256, dtype=torch.complex64, device=torch_device)
        adc[..., 0] = gain * math.sqrt(lb.thermal_noise_power_w(CFG)) * 100.0
        out = lb.add_thermal_noise(adc, CFG, seed=2)
        sig = float(torch.mean(torch.abs(out[..., 0]) ** 2))
        noise = float(torch.mean(torch.abs(out[..., 1:]) ** 2))
        ratios.append(10.0 * math.log10(sig / noise))
    assert (ratios[1] - ratios[0]) == pytest.approx(20.0, abs=1.0)


def test_noise_is_reproducible_from_the_seed(torch_device):
    adc = torch.zeros(2, 4, 64, dtype=torch.complex64, device=torch_device)
    a = lb.add_thermal_noise(adc, CFG, seed=7)
    b = lb.add_thermal_noise(adc, CFG, seed=7)
    c = lb.add_thermal_noise(adc, CFG, seed=8)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


# --------------------------------------------------------------------------------
# The bandwidth choice is load-bearing; pin it so it cannot drift
# --------------------------------------------------------------------------------
def test_noise_bandwidth_is_the_sample_rate_not_the_bin_width():
    """Using `fs / n_samples` would double-count the range FFT's processing gain and make
    every reported SNR optimistic by 10*log10(n_samples) = 27 dB at n_samples=512."""
    assert lb.noise_bandwidth_hz(CFG) == pytest.approx(CFG.fs_hz)
    assert lb.noise_bandwidth_hz(CFG) != pytest.approx(CFG.fs_hz / CFG.n_samples)


def test_config_without_link_budget_fields_falls_back_to_module_defaults():
    """Configs deserialized from pre-link-budget manifests carry no such fields. The
    fallback must be the same value the current default config uses, so old and new
    agree rather than silently diverging."""
    class _Legacy:
        f0_hz, fs_hz, n_samples, n_chirps = CFG.f0_hz, CFG.fs_hz, CFG.n_samples, CFG.n_chirps
    assert lb.thermal_noise_power_w(_Legacy()) == pytest.approx(
        lb.thermal_noise_power_w(CFG), rel=1e-9)
