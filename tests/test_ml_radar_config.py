"""Tests for e2e.ml.radar_config -- dependency-free, no torch needed."""

import math

import pytest

from e2e.ml.radar_config import (
    C_MPS,
    PRESETS,
    RADIAL_LIKE,
    TI_IWR1443,
    RadarConfig,
)


def _make(**overrides):
    base = dict(
        name="test",
        f0_hz=77e9,
        bandwidth_hz=1e9,
        n_tx=2,
        n_rx=4,
        n_chirps=64,
        n_samples=128,
        fs_hz=10e6,
        chirp_period_s=50e-6,
        mimo="tdm",
    )
    base.update(overrides)
    return RadarConfig(**base)


# ---- derived-property formulas ------------------------------------------------

def test_n_virtual():
    cfg = _make(n_tx=3, n_rx=4)
    assert cfg.n_virtual == 12


def test_sweep_time_and_slope():
    cfg = _make(n_samples=128, fs_hz=10e6, bandwidth_hz=1e9)
    assert cfg.sweep_time_s == pytest.approx(128 / 10e6)
    assert cfg.ramp_slope_hzps == pytest.approx(1e9 / (128 / 10e6))


def test_wavelength():
    cfg = _make(f0_hz=77e9, bandwidth_hz=1e9)
    f_center = 77e9 + 0.5e9
    assert cfg.wavelength_m == pytest.approx(C_MPS / f_center)


def test_range_resolution_and_max_range_identity():
    cfg = _make(bandwidth_hz=1e9, n_samples=128, fs_hz=10e6)
    expected_res = C_MPS / (2 * 1e9)
    assert cfg.range_resolution_m == pytest.approx(expected_res)
    # max_range_m via fs*c/(2*slope) must algebraically equal n_samples * range_res
    # (fs cancels out -- see the docstring on max_range_m).
    assert cfg.max_range_m == pytest.approx(cfg.n_samples * cfg.range_resolution_m, rel=1e-9)
    # and changing fs_hz alone must not move max_range_m
    cfg2 = _make(bandwidth_hz=1e9, n_samples=128, fs_hz=20e6)
    assert cfg2.max_range_m == pytest.approx(cfg.max_range_m, rel=1e-9)


def test_n_chirps_per_tx_tdm_vs_others():
    cfg_tdm = _make(mimo="tdm", n_tx=4, n_chirps=64)
    assert cfg_tdm.n_chirps_per_tx == 16
    cfg_ddma = _make(mimo="ddma", n_tx=4, n_chirps=64)
    assert cfg_ddma.n_chirps_per_tx == 64
    cfg_single = _make(mimo="single", n_tx=1, n_chirps=64)
    assert cfg_single.n_chirps_per_tx == 64


def test_velocity_resolution_independent_of_mimo():
    # velocity_resolution_mps depends on the full CPI (n_chirps), not on mimo.
    cfg_tdm = _make(mimo="tdm", n_tx=4, n_chirps=64, chirp_period_s=50e-6)
    cfg_ddma = _make(mimo="ddma", n_tx=4, n_chirps=64, chirp_period_s=50e-6)
    assert cfg_tdm.velocity_resolution_mps == pytest.approx(cfg_ddma.velocity_resolution_mps)
    expected = cfg_tdm.wavelength_m / (2 * 64 * 50e-6)
    assert cfg_tdm.velocity_resolution_mps == pytest.approx(expected)


def test_tdm_max_velocity_penalty_vs_single_tx():
    n_tx = 4
    cfg_tdm = _make(mimo="tdm", n_tx=n_tx, n_chirps=64, chirp_period_s=50e-6)
    cfg_single = _make(mimo="single", n_tx=1, n_chirps=64, chirp_period_s=50e-6)
    # TDM's unambiguous velocity is divided by n_tx relative to a single-TX
    # radar at the same chirp rate (the PRF penalty).
    assert cfg_single.max_velocity_mps == pytest.approx(cfg_tdm.max_velocity_mps * n_tx, rel=1e-9)
    expected_tdm = cfg_tdm.wavelength_m / (4 * n_tx * 50e-6)
    assert cfg_tdm.max_velocity_mps == pytest.approx(expected_tdm)


def test_ddma_max_velocity_pays_the_same_n_tx_penalty_as_tdm():
    # DDMA code-divides the Doppler spectrum into n_tx replica sub-bands, so the
    # unambiguous span shrinks by n_tx exactly like TDM's PRF penalty. (An earlier
    # version claimed no penalty; empirically two velocities 1/4 span apart alias
    # to identical replica sets for a 4-TX DDMA config -- adversarial-review fix.)
    n_tx = 4
    cfg_ddma = _make(mimo="ddma", n_tx=n_tx, n_chirps=64, chirp_period_s=50e-6)
    expected = cfg_ddma.wavelength_m / (4 * n_tx * 50e-6)
    assert cfg_ddma.max_velocity_mps == pytest.approx(expected)
    cfg_single = _make(mimo="single", n_tx=1, n_chirps=64, chirp_period_s=50e-6)
    assert cfg_single.max_velocity_mps == pytest.approx(cfg_ddma.max_velocity_mps * n_tx)


def test_mimo_tag_is_case_normalized_at_construction():
    # rd_synth lower-cases its own copy, but the derived properties compare exact
    # strings: an un-normalized "TDM" used to keep synthesizing correctly while
    # n_chirps_per_tx (and with it the noise coherent gain) silently used the
    # non-TDM branch -- a 10*log10(n_tx) dB SNR calibration error.
    cfg = _make(mimo="TDM", n_tx=4, n_chirps=64)
    assert cfg.mimo == "tdm"
    assert cfg.n_chirps_per_tx == 16
    assert cfg.validate() == []


# ---- validate() ----------------------------------------------------------------

def test_validate_clean_config():
    cfg = _make()
    assert cfg.validate() == []


def test_validate_catches_bad_mimo():
    cfg = _make(mimo="bogus")
    problems = cfg.validate()
    assert any("mimo" in p for p in problems)


def test_validate_catches_non_divisible_tdm_chirps():
    cfg = _make(mimo="tdm", n_tx=3, n_chirps=64)  # 64 % 3 != 0
    problems = cfg.validate()
    assert any("divisible" in p for p in problems)


def test_validate_ddma_not_required_to_divide():
    cfg = _make(mimo="ddma", n_tx=3, n_chirps=64)
    assert cfg.validate() == []


def test_validate_catches_non_positive_values():
    cfg = _make(f0_hz=-1.0, bandwidth_hz=0.0, n_tx=0, n_rx=-1, n_chirps=0,
                n_samples=0, fs_hz=0.0, chirp_period_s=-1.0, frame_rate_hz=0.0)
    problems = cfg.validate()
    assert len(problems) >= 8


def test_validate_catches_sweep_time_exceeding_period():
    cfg = _make(n_samples=1000, fs_hz=1e6, chirp_period_s=1e-9)
    problems = cfg.validate()
    assert any("sweep_time" in p for p in problems)


# ---- (de)serialization -----------------------------------------------------

def test_dict_round_trip():
    cfg = _make(name="roundtrip", mimo="ddma")
    d = cfg.to_dict()
    assert isinstance(d, dict)
    restored = RadarConfig.from_dict(d)
    assert restored == cfg


# ---- presets -----------------------------------------------------------------

@pytest.mark.parametrize("cfg", [TI_IWR1443, RADIAL_LIKE])
def test_presets_validate_clean(cfg):
    assert cfg.validate() == []


@pytest.mark.parametrize("name", PRESETS.keys())
def test_presets_dict_in_registry(name):
    assert PRESETS[name].validate() == []


def test_ti_iwr1443_derived_numbers():
    cfg = TI_IWR1443
    assert cfg.n_virtual == 12
    # Mid-range vehicle-scene profile: ~7.5 cm resolution over ~38 m, with a
    # TDM unambiguous velocity that covers pedestrians and urban vehicles.
    assert cfg.range_resolution_m == pytest.approx(0.075, rel=0.05)
    assert cfg.max_range_m == pytest.approx(38.4, rel=0.05)
    assert cfg.max_velocity_mps == pytest.approx(12.8, rel=0.05)
    # Ramp slope must stay inside the device's ~100 MHz/us class.
    assert cfg.ramp_slope_hzps <= 100e6 * 1e6


def test_radial_like_matches_paper_resolutions():
    cfg = RADIAL_LIKE
    assert cfg.n_virtual == 192
    # RADIal paper (Table 5): range res 0.2 m, max range ~103 m, vel res 0.1 m/s.
    assert cfg.range_resolution_m == pytest.approx(0.2, rel=0.05)
    assert cfg.max_range_m == pytest.approx(103.0, rel=0.05)
    assert cfg.velocity_resolution_mps == pytest.approx(0.1, rel=0.05)
