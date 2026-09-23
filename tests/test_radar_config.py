"""Tests for e2e.radar_config -- dependency-free, no torch needed."""

import math

import pytest

from e2e.radar_config import (
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


def test_from_dict_tolerates_keys_a_newer_version_dropped():
    """A manifest written by an older version must still load, with a warning.

    `from_dict` used to be a bare `cls(**d)`, which made every on-disk manifest.json a
    hard constraint on the dataclass: removing any field broke loading every corpus
    generated before the removal, with a TypeError at load time. Measured 2026-08-28
    when `temperature_k` was retired -- both shipped corpora serialize it in `config`,
    so the strict unpack rejected them outright.

    Dropping unknown keys must stay LOUD: a typo in a hand-edited manifest should still
    surface rather than being silently swallowed.
    """
    d = _make(name="legacy").to_dict()
    d["temperature_k"] = 290.0          # the real retired field
    d["a_typo_nobody_meant"] = 1.0

    with pytest.warns(UserWarning, match="unknown key"):
        restored = RadarConfig.from_dict(d)

    assert restored == _make(name="legacy")
    assert not hasattr(restored, "temperature_k")


def test_temperature_k_is_gone_and_nothing_reads_it():
    """Retired 2026-08-28 (owner call): read by nothing -- link_budget and rffe_model
    each hardcode their own T0_KELVIN. Pinned so it is not reintroduced by reflex."""
    from dataclasses import fields as dc_fields
    assert "temperature_k" not in {f.name for f in dc_fields(RadarConfig)}


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


# ---- answerability guard (release-plan A3, F43) --------------------------------
# Explicit max_sin_az_err throughout: the default is read lazily from
# e2e.ml.metrics.MatchCriterion, which imports torch -- and this file stays
# torch-free. The default-tolerance coupling is tested in test_ml_chain_generate.

def test_answerability_benchmark_v1_is_answerable():
    from e2e.radar_config import BENCHMARK_V1, answerability_problems
    assert answerability_problems(BENCHMARK_V1, top_speed_mps=8.0,
                                  max_sin_az_err=0.06) == []


def test_ddma_wide_v1_is_the_answerable_radial_replacement():
    """Release-plan B1's 'radial replacement' (scoping review 2026-08-24): DDMA with
    the aperture on the RX side -- v_max depends only on (n_tx, chirp_period_s), so
    4 x 48 keeps radial_like's full 192-element virtual aperture while reusing
    benchmark_v1's proven chirp timing for a 21% v_max margin over the 8 m/s scene
    ceiling."""
    from e2e.radar_config import DDMA_WIDE_V1, answerability_problems
    cfg = DDMA_WIDE_V1
    assert cfg.n_virtual == 192                    # radial_like aperture parity
    assert cfg.max_velocity_mps == pytest.approx(9.69, rel=0.01)
    assert cfg.range_resolution_m == pytest.approx(0.2, rel=0.05)
    assert cfg.n_chirps % cfg.n_tx == 0            # ddma_demux requirement
    assert cfg.n_samples / cfg.fs_hz < cfg.chirp_period_s  # sweep fits the period
    assert answerability_problems(cfg, top_speed_mps=8.0, max_sin_az_err=0.06) == []


def test_answerability_radial_like_aliases_in_doppler():
    from e2e.radar_config import answerability_problems
    problems = answerability_problems(RADIAL_LIKE, top_speed_mps=8.0,
                                      max_sin_az_err=0.06)
    # Azimuth is fine (192 virtual elements); Doppler is the F43 failure.
    assert len(problems) == 1
    assert "alias in Doppler" in problems[0]


def test_answerability_ti_iwr1443_azimuth_unanswerable():
    from e2e.radar_config import answerability_problems
    # 12 virtual elements -> Rayleigh 0.1667; keep speeds under its 12.8 m/s v_max
    # so only the azimuth failure fires.
    problems = answerability_problems(TI_IWR1443, top_speed_mps=5.0,
                                      max_sin_az_err=0.06)
    assert len(problems) == 1
    assert "Rayleigh" in problems[0]


def test_answerability_both_failures_reported():
    from e2e.radar_config import answerability_problems
    problems = answerability_problems(RADIAL_LIKE, top_speed_mps=8.0,
                                      max_sin_az_err=0.005)  # finer than 2/192
    assert len(problems) == 2


def test_answerability_boundary_is_inclusive():
    from e2e.radar_config import answerability_problems
    from e2e.radar_config import BENCHMARK_V1 as cfg
    # Exactly at v_max and exactly at the Rayleigh limit both pass (>=, not >).
    assert answerability_problems(cfg, top_speed_mps=cfg.max_velocity_mps,
                                  max_sin_az_err=2.0 / cfg.n_virtual) == []


# ---- BENCHMARK_V1_KA (owner decision 2026-09-23, Ka-band re-founding) ---------

def test_benchmark_v1_ka_differs_from_benchmark_v1_only_in_f0_and_derived():
    """`benchmark_v1_ka` must be `benchmark_v1` with ONLY f0_hz moved -- every other
    field byte-for-byte identical, so anything that doesn't depend on carrier
    (range resolution, max range, n_virtual, chirp timing) is unaffected."""
    from e2e.radar_config import BENCHMARK_V1, BENCHMARK_V1_KA
    import dataclasses

    assert BENCHMARK_V1_KA.f0_hz == pytest.approx(30e9)
    assert BENCHMARK_V1_KA.name == "benchmark_v1_ka"
    same_fields = dataclasses.replace(BENCHMARK_V1_KA, f0_hz=BENCHMARK_V1.f0_hz,
                                      name=BENCHMARK_V1.name)
    assert same_fields == BENCHMARK_V1

    # Carrier-independent derived quantities are unchanged.
    assert BENCHMARK_V1_KA.range_resolution_m == pytest.approx(BENCHMARK_V1.range_resolution_m)
    assert BENCHMARK_V1_KA.max_range_m == pytest.approx(BENCHMARK_V1.max_range_m)
    assert BENCHMARK_V1_KA.n_virtual == BENCHMARK_V1.n_virtual

    # Carrier-dependent derived quantities move by the wavelength ratio (77/30 GHz-ish,
    # via f0 + B/2): wavelength grows, so max velocity grows and velocity resolution
    # coarsens by the same factor.
    ratio = BENCHMARK_V1.wavelength_m / BENCHMARK_V1_KA.wavelength_m
    assert ratio < 1.0  # Ka wavelength is longer
    assert BENCHMARK_V1_KA.max_velocity_mps == pytest.approx(
        BENCHMARK_V1.max_velocity_mps / ratio, rel=1e-9)
    assert BENCHMARK_V1_KA.velocity_resolution_mps == pytest.approx(
        BENCHMARK_V1.velocity_resolution_mps / ratio, rel=1e-9)


def test_benchmark_v1_ka_is_answerable():
    from e2e.radar_config import BENCHMARK_V1_KA, answerability_problems
    assert answerability_problems(BENCHMARK_V1_KA, top_speed_mps=8.0,
                                  max_sin_az_err=0.06) == []


def test_benchmark_v1_ka_is_registered_and_legacy_preset_unchanged():
    from e2e.radar_config import BENCHMARK_V1, BENCHMARK_V1_KA, PRESETS
    assert PRESETS["benchmark_v1_ka"] is BENCHMARK_V1_KA
    assert PRESETS["benchmark_v1"] is BENCHMARK_V1
    assert PRESETS["benchmark_v1"].f0_hz == pytest.approx(77e9)
