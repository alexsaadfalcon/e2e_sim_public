"""One-solve Doppler validity law -- pure arithmetic, no Sionna, no GPU.

Lives outside `test_ml_rt_gen.py` deliberately: that module is gated behind the `sionna`
marker (RUN_SIONNA=1), and these checks need neither ray tracing nor a GPU. They are the
guard on a limitation that applies to every corpus this repo generates, so they must run
in CI, not only on a machine with Sionna installed.
"""

import pytest

# --------------------------------------------------------------------------------
# One-solve Doppler validity (measured + derived 2026-08-11; no Sionna needed)
# --------------------------------------------------------------------------------
def test_doppler_validity_scales_inversely_with_speed_and_bandwidth():
    """N(eps) = eps*sqrt(3)*c / (2*pi*B*v*T_c): halving the speed or the bandwidth
    doubles the usable chirp count."""
    from e2e.ml.radar_config import PRESETS
    from e2e.ml.rt_gen import doppler_validity

    cfg = PRESETS["radial_like"]
    slow = doppler_validity(cfg, 5.0)["usable_chirps"]
    fast = doppler_validity(cfg, 10.0)["usable_chirps"]
    assert fast == pytest.approx(slow / 2.0, rel=1e-6)

    import dataclasses
    half_bw = dataclasses.replace(cfg, bandwidth_hz=cfg.bandwidth_hz / 2.0)
    assert doppler_validity(half_bw, 5.0)["usable_chirps"] == pytest.approx(2.0 * slow,
                                                                            rel=1e-6)


def test_doppler_validity_reproduces_the_measured_numbers():
    """Pinned against the stable-path-set measurement (planar target, free space,
    max_depth=1): ~73 chirps at 1 m/s for radial_like, scaling as 1/v."""
    from e2e.ml.radar_config import PRESETS
    from e2e.ml.rt_gen import doppler_validity

    cfg = PRESETS["radial_like"]
    assert doppler_validity(cfg, 1.0)["usable_chirps"] == pytest.approx(72.5, rel=0.02)
    assert doppler_validity(cfg, 20.0)["usable_chirps"] == pytest.approx(3.6, rel=0.05)


def test_doppler_validity_flags_the_shipped_frame_length():
    """The uncomfortable one, pinned deliberately: radial_like's 252-chirp frame is
    OUTSIDE the 5% bound even for a 1 m/s target, so every frame in a corpus generated
    at this preset violates it. Intra-frame range migration is the missing term."""
    from e2e.ml.radar_config import PRESETS
    from e2e.ml.rt_gen import doppler_validity

    assert doppler_validity(PRESETS["radial_like"], 1.0)["within_target"] is False


def test_doppler_validity_static_target_is_exact():
    from e2e.ml.radar_config import PRESETS
    from e2e.ml.rt_gen import doppler_validity

    d = doppler_validity(PRESETS["radial_like"], 0.0)
    assert d["rel_rmse_slope_per_chirp"] == 0.0
    assert d["within_target"] is True


def test_warn_if_doppler_invalid_warns_and_returns_the_verdict():
    import warnings as _w

    from e2e.ml.radar_config import PRESETS
    from e2e.ml.rt_gen import warn_if_doppler_invalid

    with pytest.warns(UserWarning, match="range migration"):
        d = warn_if_doppler_invalid(PRESETS["radial_like"], 20.0)
    assert d["within_target"] is False

    with _w.catch_warnings():
        _w.simplefilter("error")            # a static target must NOT warn
        warn_if_doppler_invalid(PRESETS["radial_like"], 0.0)


# --------------------------------------------------------------------------------
# cfr_sum_over_paths -- the closed form the range-migration fix is built on.
# Pure numpy: no Sionna, no GPU, so these run in CI.
# --------------------------------------------------------------------------------
import numpy as np


def _paths(n_ant=3, n_paths=4, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n_ant, n_paths)) + 1j * rng.standard_normal((n_ant, n_paths))
    tau = rng.uniform(1e-8, 5e-8, size=(n_ant, n_paths))
    dop = rng.uniform(-4e3, 4e3, size=(n_ant, n_paths))
    return a, tau, dop


_KW = dict(f_c=77e9, chirp_period_s=76e-6, n_chirps=6)


def test_cfr_sum_matches_the_closed_form_directly():
    from e2e.ml.rt_gen import cfr_sum_over_paths

    a, tau, dop = _paths()
    freqs = np.linspace(-3.7e8, 3.7e8, 5)
    h = cfr_sum_over_paths(a, tau, dop, freqs, **_KW)

    # Independent, deliberately naive re-derivation with explicit loops.
    want = np.zeros((a.shape[0], _KW["n_chirps"], freqs.size), dtype=np.complex128)
    for ant in range(a.shape[0]):
        for p in range(a.shape[1]):
            for c in range(_KW["n_chirps"]):
                t = c * _KW["chirp_period_s"]
                for k, f in enumerate(freqs):
                    want[ant, c, k] += a[ant, p] * np.exp(
                        -2j * np.pi * (_KW["f_c"] + f) * tau[ant, p]) * np.exp(
                        2j * np.pi * dop[ant, p] * t)
    assert np.allclose(h, want, rtol=1e-8, atol=1e-10)


def test_range_migration_is_a_no_op_for_a_static_target():
    """The drift is proportional to Doppler, so a zero-Doppler path must be untouched --
    EXACTLY, not approximately. This is the guard against the double-counting trap: a
    formulation that put the drift in the carrier term would still alter a static path
    (it would cancel only if the Doppler term were also removed)."""
    from e2e.ml.rt_gen import cfr_sum_over_paths

    a, tau, dop = _paths()
    dop = np.zeros_like(dop)
    freqs = np.linspace(-3.7e8, 3.7e8, 5)

    off = cfr_sum_over_paths(a, tau, dop, freqs, range_migration=False, **_KW)
    on = cfr_sum_over_paths(a, tau, dop, freqs, range_migration=True, **_KW)
    assert np.array_equal(off, on)


def test_range_migration_changes_a_moving_target_and_grows_with_chirp():
    """The correction must do nothing at chirp 0 (no elapsed time, no drift) and grow
    with chirp index -- the linear-in-c behaviour the error study measured."""
    from e2e.ml.rt_gen import cfr_sum_over_paths

    a, tau, dop = _paths()
    freqs = np.linspace(-3.7e8, 3.7e8, 9)
    off = cfr_sum_over_paths(a, tau, dop, freqs, range_migration=False, **_KW)
    on = cfr_sum_over_paths(a, tau, dop, freqs, range_migration=True, **_KW)

    per_chirp = np.linalg.norm((on - off).reshape(off.shape[0], off.shape[1], -1), axis=(0, 2))
    assert per_chirp[0] == pytest.approx(0.0, abs=1e-12)
    assert per_chirp[-1] > per_chirp[1] > 0.0


def test_range_migration_leaves_the_carrier_phase_alone():
    """At f_baseband = 0 the baseband term is exp(0) = 1 regardless of delay, so the
    correction cannot change the DC bin. If it does, the drift leaked into the carrier
    term and the Doppler is being counted twice."""
    from e2e.ml.rt_gen import cfr_sum_over_paths

    a, tau, dop = _paths()
    freqs = np.array([0.0])
    off = cfr_sum_over_paths(a, tau, dop, freqs, range_migration=False, **_KW)
    on = cfr_sum_over_paths(a, tau, dop, freqs, range_migration=True, **_KW)
    assert np.allclose(off, on, rtol=1e-12, atol=1e-12)
