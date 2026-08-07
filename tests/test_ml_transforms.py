"""
Tests for `e2e.ml.transforms` (raw ADC -> Range-Doppler -> network-input tensors).

These tests build a small local `_RadarConfigStub` rather than importing the real
`RadarConfig` (owned by a sibling shard, may not exist yet / is developed in parallel).
`transforms.py` only duck-types on the attributes it actually reads (`mimo`, `n_tx`,
`n_chirps`, `n_samples`), so the stub only needs to provide those.
"""

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from e2e.ml import transforms


@dataclass
class _RadarConfigStub:
    mimo: str = "single"
    n_tx: int = 1
    n_chirps: int = 8
    n_samples: int = 16


# --------------------------------------------------------------------------------
# adc_to_rd
# --------------------------------------------------------------------------------
def _tone_adc(n_rx, n_chirps, n_samples, k_range=0, k_doppler=0, device=None):
    """Build adc[rx, c, s] = exp(j*2*pi*k_range*s/n_samples) * exp(j*2*pi*k_doppler*c/n_chirps)."""
    s = torch.arange(n_samples, device=device, dtype=torch.float32)
    c = torch.arange(n_chirps, device=device, dtype=torch.float32)
    range_phase = torch.exp(1j * 2 * torch.pi * k_range * s / n_samples).to(torch.complex64)
    doppler_phase = torch.exp(1j * 2 * torch.pi * k_doppler * c / n_chirps).to(torch.complex64)
    tone = doppler_phase[:, None] * range_phase[None, :]     # [n_chirps, n_samples]
    return tone[None, :, :].repeat(n_rx, 1, 1)               # [n_rx, n_chirps, n_samples]


def test_range_bin_location(torch_device):
    n_rx, n_chirps, n_samples, k_range = 2, 8, 16, 5
    adc = _tone_adc(n_rx, n_chirps, n_samples, k_range=k_range, k_doppler=0, device=torch_device)
    cfg = _RadarConfigStub(n_chirps=n_chirps, n_samples=n_samples)
    rd = transforms.adc_to_rd(cfg, adc)
    assert rd.shape == (n_rx, n_samples, n_chirps)
    assert rd.dtype == torch.complex64
    assert rd.device.type == torch_device.type
    power = (rd.abs() ** 2).sum(dim=(0, 2))    # sum over rx, doppler -> per range bin
    assert torch.argmax(power).item() == k_range
    # zero Doppler -> centre bin after fftshift
    doppler_power = (rd.abs() ** 2).sum(dim=(0, 1))
    assert torch.argmax(doppler_power).item() == n_chirps // 2


def test_doppler_bin_location(torch_device):
    n_rx, n_chirps, n_samples = 2, 8, 16
    k_range, k_doppler = 3, 2
    adc = _tone_adc(n_rx, n_chirps, n_samples, k_range=k_range, k_doppler=k_doppler, device=torch_device)
    cfg = _RadarConfigStub(n_chirps=n_chirps, n_samples=n_samples)
    rd = transforms.adc_to_rd(cfg, adc)
    power_range = (rd.abs() ** 2).sum(dim=(0, 2))
    assert torch.argmax(power_range).item() == k_range
    expected_doppler_bin = (k_doppler + n_chirps // 2) % n_chirps
    power_doppler = (rd.abs() ** 2).sum(dim=(0, 1))
    assert torch.argmax(power_doppler).item() == expected_doppler_bin


def test_dc_offset_removed(torch_device):
    n_rx, n_chirps, n_samples, k_range = 2, 8, 16, 4
    weak_tone = _tone_adc(n_rx, n_chirps, n_samples, k_range=k_range, k_doppler=0, device=torch_device)
    dc = torch.full((n_rx, n_chirps, n_samples), 1000.0 + 0j, dtype=torch.complex64, device=torch_device)
    adc = dc + weak_tone
    cfg = _RadarConfigStub(n_chirps=n_chirps, n_samples=n_samples)
    rd = transforms.adc_to_rd(cfg, adc)
    power = (rd.abs() ** 2).sum(dim=(0, 2))
    # a pure DC term is fully removed by per-chirp mean subtraction, so bin 0 must
    # NOT dominate -- the weak tone at k_range should be the peak instead.
    assert torch.argmax(power).item() == k_range
    assert power[0] < power[k_range]


def test_adc_to_rd_wrong_ndim_raises(torch_device):
    bad = torch.zeros(4, 8, dtype=torch.complex64, device=torch_device)
    with pytest.raises(ValueError):
        transforms.adc_to_rd(_RadarConfigStub(), bad)


def test_adc_to_rd_cfg_mismatch_raises(torch_device):
    adc = torch.zeros(2, 8, 16, dtype=torch.complex64, device=torch_device)
    cfg = _RadarConfigStub(n_chirps=99, n_samples=16)
    with pytest.raises(ValueError):
        transforms.adc_to_rd(cfg, adc)


# --------------------------------------------------------------------------------
# tdm_deinterleave
# --------------------------------------------------------------------------------
def test_tdm_deinterleave_shape_and_mapping(torch_device):
    n_rx, n_tx, n_chirps, n_samples = 3, 3, 6, 4
    cfg = _RadarConfigStub(mimo="tdm", n_tx=n_tx, n_chirps=n_chirps, n_samples=n_samples)

    # mark each (rx, chirp) with a distinct value, constant across samples
    rx_idx = torch.arange(n_rx, device=torch_device, dtype=torch.float32)
    c_idx = torch.arange(n_chirps, device=torch_device, dtype=torch.float32)
    marker = rx_idx[:, None] * 1000.0 + c_idx[None, :]           # [n_rx, n_chirps]
    adc = marker[:, :, None].repeat(1, 1, n_samples).to(torch.complex64)

    out = transforms.tdm_deinterleave(cfg, adc)
    n_chirps_per_tx = n_chirps // n_tx
    assert out.shape == (n_tx * n_rx, n_chirps_per_tx, n_samples)
    assert out.dtype == torch.complex64
    assert out.device.type == torch_device.type

    for t in range(n_tx):
        for rx in range(n_rx):
            for j in range(n_chirps_per_tx):
                c = t + j * n_tx     # chirp fired by TX t at slow-time index j
                expected = rx * 1000.0 + c
                got = out[t * n_rx + rx, j, 0]
                assert torch.allclose(got, torch.tensor(expected, dtype=torch.complex64, device=torch_device))


def test_tdm_deinterleave_requires_tdm_mimo(torch_device):
    cfg = _RadarConfigStub(mimo="single", n_tx=1)
    adc = torch.zeros(2, 4, 8, dtype=torch.complex64, device=torch_device)
    with pytest.raises(ValueError):
        transforms.tdm_deinterleave(cfg, adc)


def test_tdm_deinterleave_indivisible_raises(torch_device):
    cfg = _RadarConfigStub(mimo="tdm", n_tx=3)
    adc = torch.zeros(2, 5, 8, dtype=torch.complex64, device=torch_device)   # 5 not divisible by 3
    with pytest.raises(ValueError):
        transforms.tdm_deinterleave(cfg, adc)


# --------------------------------------------------------------------------------
# rd_to_input
# --------------------------------------------------------------------------------
def test_rd_to_input_shape_dtype_and_placement(torch_device):
    c, r, d = 3, 5, 7
    real = torch.randn(c, r, d, device=torch_device)
    imag = torch.randn(c, r, d, device=torch_device)
    rd = torch.complex(real, imag).to(torch.complex64)

    x = transforms.rd_to_input(rd)
    assert x.shape == (2 * c, r, d)
    assert x.dtype == torch.float32
    assert x.device.type == torch_device.type
    assert torch.allclose(x[:c], real, atol=1e-5)
    assert torch.allclose(x[c:], imag, atol=1e-5)


def test_rd_to_input_requires_complex(torch_device):
    x = torch.randn(2, 3, 4, device=torch_device)
    with pytest.raises(ValueError):
        transforms.rd_to_input(x)


# --------------------------------------------------------------------------------
# input_stats / normalize
# --------------------------------------------------------------------------------
def test_normalize_round_trip(torch_device):
    torch.manual_seed(0)
    x = torch.randn(4, 6, 8, device=torch_device) * 5.0 + 3.0
    mean, std = transforms.input_stats(x)
    assert mean.shape == (4,)
    assert std.shape == (4,)
    assert mean.device.type == torch_device.type

    xn = transforms.normalize(x, mean, std)
    assert xn.device.type == torch_device.type
    # per-channel stats of the normalized tensor should be ~0 mean, ~1 std
    n_mean, n_std = transforms.input_stats(xn)
    assert torch.allclose(n_mean, torch.zeros_like(n_mean), atol=1e-4)
    assert torch.allclose(n_std, torch.ones_like(n_std), atol=1e-4)

    # round trip: de-normalize recovers x
    x_back = xn * std.reshape(-1, 1, 1) + mean.reshape(-1, 1, 1)
    assert torch.allclose(x_back, x, atol=1e-4)


# --------------------------------------------------------------------------------
# rd_power_db
# --------------------------------------------------------------------------------
def test_rd_power_db_peak_is_zero_db(torch_device):
    rd = torch.zeros(2, 4, 4, dtype=torch.complex64, device=torch_device)
    rd[0, 1, 1] = 10.0 + 0j
    rd[0, 2, 2] = 1.0 + 0j
    db = transforms.rd_power_db(rd)
    assert db.dtype == torch.float32
    assert db.device.type == torch_device.type
    assert torch.isclose(db[0, 1, 1], torch.tensor(0.0), atol=1e-4)
    assert db[0, 2, 2] < 0.0
