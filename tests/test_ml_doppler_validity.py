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
