"""Tests for the memoryless TX PA model (e2e/circuit/tx_pa.py)."""

import math
from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from e2e.circuit.tx_pa import TxPA, TxPAConfig


def _cfg(**overrides):
    base = dict(
        small_signal_gain_db=20.0,
        a_sat=1.0,
        am_pm_deg_at_sat=8.0,
        rapp_p=2.0,
        ripple_db=1.0,
        ripple_period_hz=50e6,
        ripple_phase_rad=0.3,
    )
    base.update(overrides)
    return TxPAConfig(**base)


def test_config_is_plain_json_serializable():
    """Every field is a plain float except `ripple_model`, which is a str enum-in-a-string
    selecting the frequency-response model -- both are JSON-native, which is the property
    that matters (the config is serialized into dataset manifests)."""
    cfg = _cfg()
    d = asdict(cfg)
    assert isinstance(d.pop("ripple_model"), str)
    assert all(isinstance(v, float) for v in d.values())
    import json
    json.dumps(asdict(cfg))  # must not raise


def test_small_signal_gain_matches_config(torch_device):
    cfg = _cfg()
    pa = TxPA(cfg)
    a_small = 1e-4 * cfg.a_sat
    x = torch.tensor([a_small + 0j], dtype=torch.complex64, device=torch_device)
    y = pa.apply(x)
    gain_db = 20 * torch.log10(torch.abs(y) / torch.abs(x))
    assert torch.allclose(gain_db, torch.tensor(cfg.small_signal_gain_db, device=torch_device), atol=0.1)


def test_large_signal_saturates_at_a_sat(torch_device):
    cfg = _cfg()
    pa = TxPA(cfg)
    x = torch.tensor([1e6 * cfg.a_sat + 0j], dtype=torch.complex64, device=torch_device)
    y = pa.apply(x)
    assert torch.allclose(torch.abs(y), torch.tensor(cfg.a_sat, device=torch_device), atol=1e-2)


def test_am_pm_monotonic_and_approaches_config_near_saturation(torch_device):
    cfg = _cfg()
    pa = TxPA(cfg)
    amps = torch.tensor([0.0, 0.1, 0.5, 1.0, 3.0, 10.0, 100.0], device=torch_device) * cfg.a_sat
    x = amps.to(torch.complex64)  # real input, angle 0 -> output angle == phi directly
    y = pa.apply(x)
    phi_deg = torch.angle(y) * (180.0 / math.pi)
    phi_deg = torch.where(torch.abs(x) == 0, torch.zeros_like(phi_deg), phi_deg)

    diffs = phi_deg[1:] - phi_deg[:-1]
    assert torch.all(diffs >= -1e-4), "AM/PM phase must be monotonically non-decreasing with amplitude"
    # deep in saturation (100x A_sat) the phase should be close to the configured value
    assert torch.allclose(phi_deg[-1], torch.tensor(cfg.am_pm_deg_at_sat, device=torch_device), atol=0.2)
    assert phi_deg[0].item() == pytest.approx(0.0, abs=1e-6)


def test_constant_envelope_in_gives_constant_envelope_out(torch_device):
    cfg = _cfg()
    pa = TxPA(cfg)
    r = 0.7 * cfg.a_sat
    thetas = torch.linspace(0, 2 * math.pi, 32, device=torch_device)
    x = r * torch.exp(1j * thetas.to(torch.complex64))
    y = pa.apply(x)
    mags = torch.abs(y)
    assert torch.allclose(mags, mags[0] * torch.ones_like(mags), atol=1e-4)


def test_zero_input_gives_zero_output_no_nans(torch_device):
    cfg = _cfg()
    pa = TxPA(cfg)
    x = torch.zeros(8, dtype=torch.complex64, device=torch_device)
    y = pa.apply(x)
    assert torch.all(y == 0)
    assert not torch.any(torch.isnan(torch.abs(y)))


def test_apply_on_3d_tensor(torch_device):
    cfg = _cfg()
    pa = TxPA(cfg)
    rng = np.random.default_rng(0)
    arr = (rng.standard_normal((4, 5, 6)) + 1j * rng.standard_normal((4, 5, 6))).astype(np.complex64)
    x = torch.from_numpy(arr).to(torch_device)
    y = pa.apply(x)
    assert y.shape == x.shape
    assert y.dtype == torch.complex64
    assert y.device.type == torch_device.type
    assert not torch.any(torch.isnan(torch.abs(y)))


def test_frequency_response_unit_mean_and_peak_to_peak(torch_device):
    # Legacy magnitude-only model, pinned explicitly now that "mismatch" is the default.
    cfg = _cfg(ripple_model="sinusoid", ripple_db=2.0, ripple_period_hz=1e6,
               ripple_phase_rad=0.0)
    pa = TxPA(cfg)
    # many points across many full periods -> discrete mean approaches the analytic
    # unit-mean-gain normalization, and covers both the peak and the trough.
    freqs = torch.linspace(0, 20e6, 20001, device=torch_device)
    H = pa.frequency_response(freqs)

    assert H.dtype == torch.complex64
    assert H.device.type == torch_device.type
    assert not torch.any(torch.isnan(torch.abs(H)))

    mean_mag = torch.mean(torch.abs(H))
    assert torch.allclose(mean_mag, torch.tensor(1.0, device=torch_device), atol=1e-3)

    db = 20 * torch.log10(torch.abs(H))
    ptp = db.max() - db.min()
    assert torch.allclose(ptp, torch.tensor(2 * cfg.ripple_db, device=torch_device), atol=1e-2)


def test_frequency_response_accepts_numpy_array(torch_device):
    cfg = _cfg()
    pa = TxPA(cfg)
    freqs_np = np.linspace(0, 1e8, 128).astype(np.float32)
    H = pa.frequency_response(freqs_np)
    assert H.shape == (128,)
    assert not torch.any(torch.isnan(torch.abs(H)))


# --------------------------------------------------------------------------------
# "mismatch" frequency response (default since 2026-08-10)
# --------------------------------------------------------------------------------
def test_mismatch_is_the_default_model():
    assert TxPAConfig().ripple_model == "mismatch"


def test_mismatch_period_matches_the_electrical_length(torch_device):
    """Ripple period is c / (2*l*sqrt(eps_eff)) -- the physical round trip, not a free
    parameter. The shipped 8 mm / eps_eff=3 transition gives ~10.8 GHz, i.e. well under
    one cycle across an automotive sweep (the reason the legacy 100 MHz default, which
    implied a 50-150 cm line, was wrong for an on-package 77 GHz PA)."""
    cfg = TxPAConfig()
    assert cfg.mismatch_period_hz == pytest.approx(2.998e8 / (2 * 8e-3 * math.sqrt(3.0)),
                                                   rel=1e-3)
    assert 8e9 < cfg.mismatch_period_hz < 14e9

    # Peaks recur one period apart: |H| at f and f + period must agree.
    pa = TxPA(cfg)
    f0 = torch.tensor([77e9], device=torch_device)
    f1 = f0 + cfg.mismatch_period_hz
    assert torch.allclose(pa.frequency_response(f0).abs(),
                          pa.frequency_response(f1).abs(), atol=1e-5)


def test_mismatch_carries_group_delay_where_the_sinusoid_cannot(torch_device):
    """The whole reason to prefer the mismatch form: amplitude ripple implies phase
    ripple for a causal system, and for an FMCW sweep that phase slope IS a range bias.
    The legacy model sets it to exactly zero by construction."""
    freqs = torch.linspace(76e9, 78e9, 512, device=torch_device)

    h_mis = TxPA(TxPAConfig()).frequency_response(freqs)
    h_sin = TxPA(_cfg(ripple_model="sinusoid")).frequency_response(freqs)

    assert torch.allclose(torch.angle(h_sin), torch.zeros_like(torch.angle(h_sin)),
                          atol=1e-6)                       # legacy: identically zero phase
    assert torch.max(torch.abs(torch.angle(h_mis))) > 1e-3  # physical: it is not


def test_mismatch_reduces_to_the_sinusoid_for_weak_mismatch(torch_device):
    """For |Gs*GL| << 1 the two models agree -- that equivalence is what justifies
    calling the sinusoid a linearization rather than an unrelated ad hoc curve. Compared
    over one full period so both cover peak and trough."""
    gs = gl = 0.02                                     # |Gs*GL| = 4e-4, deep in the limit
    cfg_mis = TxPAConfig(gamma_source=gs, gamma_load=gl)
    # |H| ~= 1 + g*cos(theta) for small g, so 0-to-peak depth is 20*log10(1+g) dB; the
    # sinusoid model's peak-to-peak is 2*ripple_db, matching the mismatch model's 2*that.
    ripple_db = 20.0 * math.log10(1.0 + gs * gl)
    cfg_sin = TxPAConfig(ripple_model="sinusoid", ripple_db=abs(ripple_db),
                         ripple_period_hz=cfg_mis.mismatch_period_hz,
                         ripple_phase_rad=math.pi / 2.0)

    freqs = torch.linspace(0.0, cfg_mis.mismatch_period_hz, 256, device=torch_device)
    db_mis = 20 * torch.log10(TxPA(cfg_mis).frequency_response(freqs).abs())
    db_sin = 20 * torch.log10(TxPA(cfg_sin).frequency_response(freqs).abs())

    ptp_mis = float(db_mis.max() - db_mis.min())
    ptp_sin = float(db_sin.max() - db_sin.min())
    assert ptp_mis == pytest.approx(ptp_sin, abs=5e-3)


def test_mismatch_depth_grows_with_reflection_product(torch_device):
    freqs = torch.linspace(0.0, TxPAConfig().mismatch_period_hz, 256, device=torch_device)

    def depth(g):
        h = TxPA(TxPAConfig(gamma_source=g, gamma_load=g)).frequency_response(freqs)
        db = 20 * torch.log10(h.abs())
        return float(db.max() - db.min())

    assert depth(0.1) < depth(0.3) < depth(0.6)


def test_mismatch_rejects_unknown_model(torch_device):
    with pytest.raises(ValueError, match="ripple_model"):
        TxPA(TxPAConfig(ripple_model="bogus")).frequency_response(
            torch.linspace(0.0, 1e9, 8, device=torch_device))
