"""Tests for `e2e.ml.impairments` (FMCW ADC-domain radar impairments)."""

import math

import pytest

torch = pytest.importorskip("torch")

from e2e.ml.impairments import (  # noqa: E402
    ClutterParams,
    LeakageParams,
    PhaseNoiseParams,
    _k_distributed_gain,
    _range_to_bin,
    apply_all,
    apply_clutter,
    apply_leakage,
    apply_phase_noise,
)
from e2e.ml.radar_config import RadarConfig  # noqa: E402


@pytest.fixture
def cfg():
    # Small (fast) but self-consistent radar config: sweep_time = 128/10e6 = 12.8us
    # fits inside chirp_period_s=20e-6.
    c = RadarConfig(
        name="test_small",
        f0_hz=77e9,
        bandwidth_hz=500e6,
        n_tx=1,
        n_rx=8,
        n_chirps=64,
        n_samples=128,
        fs_hz=10e6,
        chirp_period_s=20e-6,
        mimo="single",
    )
    assert not c.validate()
    return c


def _rand_cube(n_rx, n_chirps, n_samples, device, seed=0):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    real = torch.randn((n_rx, n_chirps, n_samples), generator=gen, device=device, dtype=torch.float32)
    imag = torch.randn((n_rx, n_chirps, n_samples), generator=gen, device=device, dtype=torch.float32)
    return torch.view_as_complex(torch.stack([real, imag], dim=-1).contiguous())


def _tone_cube(n_rx, n_chirps, n_samples, k0, amplitude, device):
    """A pure tone at fast-time DFT bin k0, identical on every rx/chirp."""
    n = torch.arange(n_samples, device=device, dtype=torch.float32)
    tone = amplitude * torch.exp(1j * (2.0 * math.pi * k0 * n / n_samples)).to(torch.complex64)
    return tone.view(1, 1, n_samples).expand(n_rx, n_chirps, n_samples).clone()


# --------------------------------------------------------------------------- phase noise

def test_phase_noise_range_correlation(cfg, torch_device):
    adc = _rand_cube(4, cfg.n_chirps, cfg.n_samples, torch_device, seed=1)
    out = apply_phase_noise(adc, cfg, PhaseNoiseParams(), seed=123)

    assert out.shape == adc.shape
    assert out.dtype == adc.dtype
    assert out.device.type == torch_device.type

    x_in = torch.fft.fft(adc, dim=-1)
    x_out = torch.fft.fft(out, dim=-1)
    delta = torch.angle(x_out / x_in)  # [n_rx, n_chirps, n_samples]

    near_rms = delta[:, :, 1].pow(2).mean().sqrt().item()
    far_rms = delta[:, :, cfg.n_samples // 2].pow(2).mean().sqrt().item()
    assert far_rms > 1.5 * near_rms

    # tau(0) == 0 exactly -> the range-correlation factor is exactly zero there.
    assert delta[:, :, 0].abs().max().item() < 1e-4


def test_phase_noise_energy_conserved(cfg, torch_device):
    adc = _rand_cube(4, cfg.n_chirps, cfg.n_samples, torch_device, seed=2)
    out = apply_phase_noise(adc, cfg, PhaseNoiseParams(), seed=5)
    e_in = torch.sum(torch.abs(adc) ** 2).item()
    e_out = torch.sum(torch.abs(out) ** 2).item()
    db = 10.0 * math.log10(e_out / e_in)
    assert abs(db) < 0.5


def test_phase_noise_deterministic(cfg, torch_device):
    adc = _rand_cube(4, cfg.n_chirps, cfg.n_samples, torch_device, seed=3)
    out1 = apply_phase_noise(adc, cfg, PhaseNoiseParams(), seed=42)
    out2 = apply_phase_noise(adc, cfg, PhaseNoiseParams(), seed=42)
    assert torch.allclose(out1, out2, atol=1e-6)


# --------------------------------------------------------------------------- leakage

def test_leakage_bins_and_power(cfg, torch_device):
    n_rx = 6
    a0 = 2.0
    k0 = int(round(_range_to_bin(10.0, cfg))) % cfg.n_samples  # a target far from leak/bumper bins
    params = LeakageParams()  # leakage -5 dB, bumper at 0.2 m, -15 dB
    k_leak = 0
    k_bump = int(round(_range_to_bin(params.bumper_range_m, cfg))) % cfg.n_samples
    assert len({k0, k_leak, k_bump}) == 3  # distinct bins, otherwise the test is ambiguous

    adc = _tone_cube(n_rx, cfg.n_chirps, cfg.n_samples, k0, a0, torch_device)
    out = apply_leakage(adc, cfg, params, seed=7)

    assert out.shape == adc.shape
    assert out.dtype == adc.dtype

    p_ref = (a0 * cfg.n_samples) ** 2
    diff = out - adc
    x_diff = torch.fft.fft(diff, dim=-1)  # [n_rx, n_chirps, n_samples]

    for k, rel_db in ((k_leak, params.leakage_relative_db), (k_bump, params.bumper_relative_db)):
        power = torch.abs(x_diff[:, :, k]) ** 2
        observed_db = 10.0 * torch.log10(power / p_ref)
        assert torch.allclose(observed_db, torch.full_like(observed_db, rel_db), atol=1.0)
        # constant across chirps (all energy in Doppler bin 0)
        assert power.std(dim=1).max().item() / power.mean().item() < 1e-5

    # target bin itself is untouched
    assert torch.abs(x_diff[:, :, k0]).max().item() < 1e-3 * (a0 * cfg.n_samples)


def test_leakage_deterministic(cfg, torch_device):
    adc = _rand_cube(4, cfg.n_chirps, cfg.n_samples, torch_device, seed=9)
    out1 = apply_leakage(adc, cfg, LeakageParams(), seed=11)
    out2 = apply_leakage(adc, cfg, LeakageParams(), seed=11)
    assert torch.allclose(out1, out2, atol=1e-6)


# --------------------------------------------------------------------------- clutter

def test_clutter_kurtosis_decreases_with_nu(torch_device):
    gen = torch.Generator(device=torch_device)
    gen.manual_seed(0)
    n = 20000

    def _excess_kurtosis(nu):
        gen.manual_seed(1234)
        gain = _k_distributed_gain(n, 1, nu, generator=gen, device=torch_device)
        amp = torch.abs(gain).flatten().to(torch.float64)
        mu = amp.mean()
        m2 = ((amp - mu) ** 2).mean()
        m4 = ((amp - mu) ** 4).mean()
        return (m4 / m2 ** 2 - 3.0).item()

    k_heavy = _excess_kurtosis(0.1)
    k_mid = _excess_kurtosis(1.0)
    k_light = _excess_kurtosis(50.0)

    assert k_heavy > k_mid > k_light
    assert k_heavy > k_light + 2.0
    # nu=50 texture barely fluctuates -> amplitude close to Rayleigh (excess kurtosis ~0.1)
    assert k_light < 1.0


def test_clutter_doppler_near_zero_and_shape(cfg, torch_device):
    n_rx = 4
    a0 = 3.0
    k0 = int(round(_range_to_bin(10.0, cfg))) % cfg.n_samples
    adc = _tone_cube(n_rx, cfg.n_chirps, cfg.n_samples, k0, a0, torch_device)

    params = ClutterParams(density=2.0, nu=1.0, doppler_std_mps=0.05, total_relative_db=-10.0)
    out = apply_clutter(adc, cfg, params, seed=17)

    assert out.shape == adc.shape
    assert out.dtype == adc.dtype
    diff = out - adc
    assert torch.abs(diff).max().item() > 0.0

    # Doppler concentration: power (summed over rx, range) should be concentrated
    # within a small window around Doppler bin 0.
    y = torch.fft.fft(diff, dim=1)  # chirp axis -> Doppler
    power_per_bin = torch.sum(torch.abs(y) ** 2, dim=(0, 2))  # [n_chirps]

    sigma_dop_hz = 2.0 * params.doppler_std_mps / cfg.wavelength_m
    df = 1.0 / (cfg.n_chirps * cfg.chirp_period_s)
    sigma_bins = sigma_dop_hz / df
    half_window = max(2, int(math.ceil(6.0 * sigma_bins)) + 2)

    idx = torch.arange(cfg.n_chirps, device=torch_device)
    centered = torch.minimum(idx, cfg.n_chirps - idx)  # distance from bin 0, wrapped
    in_window = centered <= half_window

    frac = (power_per_bin[in_window].sum() / power_per_bin.sum()).item()
    assert frac > 0.7


def test_clutter_deterministic(cfg, torch_device):
    adc = _rand_cube(4, cfg.n_chirps, cfg.n_samples, torch_device, seed=21)
    params = ClutterParams()
    out1 = apply_clutter(adc, cfg, params, seed=23)
    out2 = apply_clutter(adc, cfg, params, seed=23)
    assert torch.allclose(out1, out2, atol=1e-6)


# --------------------------------------------------------------------------- chain

def test_apply_all_defaults_and_skip(cfg, torch_device):
    adc = _rand_cube(4, cfg.n_chirps, cfg.n_samples, torch_device, seed=31)

    out_all = apply_all(adc, cfg, seed=1)
    assert out_all.shape == adc.shape
    assert out_all.dtype == adc.dtype
    assert not torch.allclose(out_all, adc)

    out_skip = apply_all(adc, cfg, {"leakage": None, "clutter": None}, seed=1)
    out_phase_only = apply_phase_noise(adc, cfg, PhaseNoiseParams(), seed=1)
    assert torch.allclose(out_skip, out_phase_only, atol=1e-6)


def test_apply_all_deterministic(cfg, torch_device):
    adc = _rand_cube(4, cfg.n_chirps, cfg.n_samples, torch_device, seed=33)
    out1 = apply_all(adc, cfg, seed=99)
    out2 = apply_all(adc, cfg, seed=99)
    assert torch.allclose(out1, out2, atol=1e-6)
