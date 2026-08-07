"""
Physics tests for the analytic FMCW raw-ADC synthesizer (`e2e.ml.rd_synth`).

Everything here is checked against closed-form expectations derived from the
waveform parameters, not against golden data:

    range bin      k_R = f_b * n_samples / fs, f_b = 2 R S / c
                       = 2 R B / c          (since S * n_samples/fs = B)
    Doppler bin    k_D = (2 f0 v_r T_eff / c) * n_chirps_per_tx
                   T_eff = n_tx * chirp_period_s for TDM (one TX every n_tx chirps)
    array phase    dphi/d(rx element) = 2 pi (lambda/2) sin(theta) / lambda
                                      = pi sin(theta)

Sign convention asserted here: v_r > 0 means *receding* (range increasing), which
produces a positive Doppler frequency, i.e. a peak in the lower half of the
unshifted FFT. An approaching target lands in the upper half (negative frequency).

The `RadarConfig`/`Scatterer` stubs below are local on purpose: this shard must not
depend on the sibling modules, only on the pinned field contract.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from e2e.ml.rd_synth import C_LIGHT, RadarPose, array_axis, synthesize_adc


# --------------------------------------------------------------------------------
# Minimal local stand-ins for radar_config.RadarConfig / scatterers.Scatterer
# --------------------------------------------------------------------------------
@dataclass(frozen=True)
class CfgStub:
    name: str = "test"
    f0_hz: float = 77e9
    bandwidth_hz: float = 200e6
    n_tx: int = 2
    n_rx: int = 8
    n_chirps: int = 64
    n_samples: int = 128
    fs_hz: float = 10e6
    chirp_period_s: float = 50e-6
    mimo: str = "tdm"
    frame_rate_hz: float = 20.0

    @property
    def n_virtual(self):
        return self.n_tx * self.n_rx

    @property
    def wavelength_m(self):
        return C_LIGHT / (self.f0_hz + 0.5 * self.bandwidth_hz)

    @property
    def ramp_slope_hzps(self):
        return self.bandwidth_hz / (self.n_samples / self.fs_hz)

    @property
    def n_chirps_per_tx(self):
        return self.n_chirps // self.n_tx if self.mimo == "tdm" else self.n_chirps


@dataclass
class ScatStub:
    position: tuple = (10.0, 0.0, 0.0)
    velocity: tuple = (0.0, 0.0, 0.0)
    rcs_dbsm: float = 0.0
    object_class: str = "car"


def single_tx_cfg(**kw):
    """A 1-TX config (mimo='single'), the cleanest case for range/angle checks."""
    base = dict(n_tx=1, mimo="single")
    base.update(kw)
    return CfgStub(**base)


def expected_range_bin(cfg, r_m):
    """k_R = 2 R B / c  (see module docstring)."""
    return 2.0 * r_m * cfg.bandwidth_hz / C_LIGHT


def expected_doppler_bin(cfg, v_r, t_eff):
    """Doppler bin (fractional, unwrapped) for radial velocity `v_r`."""
    f_norm = 2.0 * cfg.f0_hz * v_r * t_eff / C_LIGHT       # cycles per slow-time sample
    return f_norm * cfg.n_chirps_per_tx


def target_at(r_m, theta_deg, v_radial=0.0, rcs_dbsm=0.0):
    """Scatterer at range `r_m`, azimuth `theta_deg` from the default (+x) boresight.

    With the default pose the array axis is +y, so sin(theta) = y / R.
    """
    th = math.radians(theta_deg)
    pos = (r_m * math.cos(th), r_m * math.sin(th), 0.0)
    vel = tuple(v_radial * p / r_m for p in pos)            # purely radial
    return ScatStub(position=pos, velocity=vel, rcs_dbsm=rcs_dbsm)


def range_fft(adc):
    """FFT over fast time -> [..., n_samples] range spectrum."""
    return torch.fft.fft(adc, dim=-1)


# --------------------------------------------------------------------------------
# Basic contract
# --------------------------------------------------------------------------------
def test_shape_dtype_device(torch_device):
    cfg = CfgStub()
    adc = synthesize_adc(cfg, [ScatStub()], RadarPose(), snr_db=20.0, seed=0,
                         device=torch_device)
    assert adc.shape == (cfg.n_rx, cfg.n_chirps, cfg.n_samples)
    assert adc.dtype == torch.complex64
    assert adc.device.type == torch.device(torch_device).type


def test_empty_scene_noise_free_is_zero(torch_device):
    adc = synthesize_adc(CfgStub(), [], RadarPose(), snr_db=None, device=torch_device)
    assert torch.count_nonzero(adc) == 0


def test_empty_scene_with_snr_is_unit_variance_noise(torch_device):
    # No scatterer -> no amplitude reference, so the documented fallback is sigma^2 = 1.
    cfg = CfgStub()
    adc = synthesize_adc(cfg, [], RadarPose(), snr_db=10.0, seed=7, device=torch_device)
    power = float(torch.mean(torch.abs(adc) ** 2))
    assert 0.9 < power < 1.1


def test_single_mode_rejects_multiple_tx(torch_device):
    with pytest.raises(ValueError):
        synthesize_adc(CfgStub(n_tx=2, mimo="single"), [ScatStub()], RadarPose(),
                       snr_db=None, device=torch_device)


def test_array_axis_convention():
    # +z up crossed with +x boresight -> +y lateral axis.
    assert np.allclose(array_axis(RadarPose()), [0.0, 1.0, 0.0])
    assert np.allclose(array_axis(RadarPose(boresight=(0.0, 1.0, 0.0))), [-1.0, 0.0, 0.0])


# --------------------------------------------------------------------------------
# Range
# --------------------------------------------------------------------------------
def test_range_peak_lands_in_expected_bin(torch_device):
    cfg = single_tx_cfg()
    r_m = 12.0
    adc = synthesize_adc(cfg, [target_at(r_m, 0.0)], RadarPose(), snr_db=None,
                         seed=0, device=torch_device)
    spec = torch.abs(range_fft(adc[0, 0])) ** 2
    peak = int(torch.argmax(spec))
    exp = expected_range_bin(cfg, r_m)                      # 12 m -> 16.01 bins
    assert abs(peak - exp) <= 1.0, f"peak {peak}, expected ~{exp:.2f}"


def test_range_scales_linearly(torch_device):
    cfg = single_tx_cfg()
    peaks = []
    for r_m in (6.0, 12.0, 24.0):
        adc = synthesize_adc(cfg, [target_at(r_m, 0.0)], RadarPose(), snr_db=None,
                             seed=1, device=torch_device)
        spec = torch.abs(range_fft(adc[0, 0])) ** 2
        peaks.append(int(torch.argmax(spec)))
        assert abs(peaks[-1] - expected_range_bin(cfg, r_m)) <= 1.0
    assert peaks[0] < peaks[1] < peaks[2]


def test_two_targets_resolve_beyond_range_resolution(torch_device):
    cfg = single_tx_cfg()
    dr = C_LIGHT / (2.0 * cfg.bandwidth_hz)                 # 0.75 m unwindowed resolution
    r1, r2 = 12.0, 12.0 + 5.0 * dr
    tgts = [target_at(r1, 0.0), target_at(r2, 0.0)]
    adc = synthesize_adc(cfg, tgts, RadarPose(), snr_db=None, seed=2, device=torch_device)
    spec = (torch.abs(range_fft(adc[0, 0])) ** 2).cpu().numpy()

    b1 = int(round(expected_range_bin(cfg, r1)))
    b2 = int(round(expected_range_bin(cfg, r2)))
    assert b2 - b1 >= 4
    top = spec.max()
    # both targets present ...
    assert spec[b1 - 1:b1 + 2].max() > 0.25 * top
    assert spec[b2 - 1:b2 + 2].max() > 0.25 * top
    # ... and separated by a dip between them
    mid = (b1 + b2) // 2
    assert spec[mid] < 0.25 * min(spec[b1 - 1:b1 + 2].max(), spec[b2 - 1:b2 + 2].max())


def test_amplitude_follows_r_squared_law(torch_device):
    cfg = single_tx_cfg()
    a = []
    for r_m in (10.0, 20.0):
        adc = synthesize_adc(cfg, [target_at(r_m, 0.0)], RadarPose(), snr_db=None,
                             seed=3, device=torch_device)
        a.append(float(torch.mean(torch.abs(adc))))
    # A = sqrt(sigma)/R^2 -> doubling the range attenuates the amplitude by 4x
    assert a[0] / a[1] == pytest.approx(4.0, rel=1e-3)


# --------------------------------------------------------------------------------
# Doppler
# --------------------------------------------------------------------------------
def _doppler_spectrum(cfg, adc, tx=0):
    """Range-Doppler spectrum of RX 0 using only the chirps of TX `tx`."""
    step = cfg.n_tx if cfg.mimo == "tdm" else 1
    slow = adc[0, tx::step, :]                              # [n_chirps_per_tx, n_samples]
    rd = torch.fft.fft(range_fft(slow), dim=0)              # fast time then slow time
    return (torch.abs(rd) ** 2).cpu().numpy()


@pytest.mark.parametrize("v_r", [3.0, -3.0])
def test_doppler_peak_and_sign(torch_device, v_r):
    cfg = CfgStub()                                         # TDM, 2 TX -> 32 chirps/TX
    r_m = 12.0
    adc = synthesize_adc(cfg, [target_at(r_m, 0.0, v_radial=v_r)], RadarPose(),
                         snr_db=None, seed=4, device=torch_device)
    rd = _doppler_spectrum(cfg, adc)
    k_d, k_r = np.unravel_index(int(np.argmax(rd)), rd.shape)

    assert abs(k_r - expected_range_bin(cfg, r_m)) <= 1.0
    t_eff = cfg.n_tx * cfg.chirp_period_s
    exp = expected_doppler_bin(cfg, v_r, t_eff)             # +-4.93 bins
    exp_wrapped = exp % cfg.n_chirps_per_tx
    err = min(abs(k_d - exp_wrapped), cfg.n_chirps_per_tx - abs(k_d - exp_wrapped))
    assert err <= 1.0, f"doppler bin {k_d}, expected ~{exp_wrapped:.2f}"

    # sign convention: receding -> lower half (positive frequency)
    if v_r > 0:
        assert k_d < cfg.n_chirps_per_tx // 2
    else:
        assert k_d > cfg.n_chirps_per_tx // 2


def test_static_target_is_dc_in_doppler(torch_device):
    cfg = CfgStub()
    adc = synthesize_adc(cfg, [target_at(12.0, 0.0)], RadarPose(), snr_db=None,
                         seed=5, device=torch_device)
    rd = _doppler_spectrum(cfg, adc)
    k_d, _ = np.unravel_index(int(np.argmax(rd)), rd.shape)
    assert k_d == 0


def test_doppler_phase_progression_matches_theory(torch_device):
    """Chirp-to-chirp phase step = 2 pi * 2 f0 v_r T_c / c (single TX, on the raw ADC)."""
    cfg = single_tx_cfg()
    v_r = 2.0
    adc = synthesize_adc(cfg, [target_at(12.0, 0.0, v_radial=v_r)], RadarPose(),
                         snr_db=None, seed=6, device=torch_device)
    spec = range_fft(adc[0])                                # [n_chirps, n_samples]
    k_r = int(torch.argmax(torch.abs(spec[0])))
    ph = torch.angle(spec[1:, k_r] * torch.conj(spec[:-1, k_r])).cpu().numpy()
    exp = 2 * math.pi * 2 * cfg.f0_hz * v_r * cfg.chirp_period_s / C_LIGHT
    exp = (exp + math.pi) % (2 * math.pi) - math.pi
    assert np.allclose(ph, exp, atol=2e-3)


# --------------------------------------------------------------------------------
# Angle / array manifold
# --------------------------------------------------------------------------------
@pytest.mark.parametrize("theta_deg", [-25.0, 0.0, 18.0])
def test_rx_phase_ramp_matches_steering_vector(torch_device, theta_deg):
    cfg = single_tx_cfg(n_rx=8)
    adc = synthesize_adc(cfg, [target_at(15.0, theta_deg)], RadarPose(), snr_db=None,
                         seed=7, device=torch_device)
    # element-to-element phase difference on the raw samples (exact by construction)
    prod = adc[1:, 0, :] * torch.conj(adc[:-1, 0, :])
    step = torch.angle(prod.sum(dim=-1)).cpu().numpy()
    exp = math.pi * math.sin(math.radians(theta_deg))       # 2 pi (d/lambda) sin(theta), d = lambda/2
    assert np.allclose(step, exp, atol=1e-3)


def test_beamforming_peak_at_target_angle(torch_device):
    """A spatial FFT over the RX aperture must peak at the target's direction cosine."""
    cfg = single_tx_cfg(n_rx=16)
    theta_deg = 20.0
    adc = synthesize_adc(cfg, [target_at(15.0, theta_deg)], RadarPose(), snr_db=None,
                         seed=8, device=torch_device)
    n_ang = 256
    spec = range_fft(adc[:, 0, :])                          # [n_rx, n_samples]
    k_r = int(torch.argmax(torch.abs(spec[0])))
    ang = torch.fft.fftshift(torch.fft.fft(spec[:, k_r], n=n_ang))
    k = int(torch.argmax(torch.abs(ang)))
    # exp(+j pi r sin) is a tone at normalised frequency sin(theta)/2, and torch's FFT
    # kernel exp(-j2pi kn/N) puts it at shifted bin k - N/2 = N sin(theta)/2.
    sin_hat = 2.0 * ((k - n_ang // 2) / n_ang)
    assert sin_hat == pytest.approx(math.sin(math.radians(theta_deg)), abs=0.02)


def test_elevation_maps_to_same_direction_cosine(torch_device):
    """A ULA sees only the axis direction cosine: an elevated target with the same
    y/R gives the same spatial phase (documented ambiguity)."""
    cfg = single_tx_cfg()
    r_m, y = 15.0, 5.0
    x = math.sqrt(r_m ** 2 - y ** 2)
    flat = ScatStub(position=(x, y, 0.0))
    z = 3.0
    xy = math.sqrt(r_m ** 2 - y ** 2 - z ** 2)
    high = ScatStub(position=(xy, y, z))
    ph = []
    for s in (flat, high):
        adc = synthesize_adc(cfg, [s], RadarPose(), snr_db=None, seed=9,
                             device=torch_device)
        prod = adc[1:, 0, :] * torch.conj(adc[:-1, 0, :])
        ph.append(torch.angle(prod.sum(dim=-1)).cpu().numpy())
    assert np.allclose(ph[0], ph[1], atol=1e-3)


# --------------------------------------------------------------------------------
# MIMO
# --------------------------------------------------------------------------------
def test_tdm_chirps_carry_per_tx_phase_offset(torch_device):
    """For a static off-boresight target, chirp 1 (TX1) differs from chirp 0 (TX0)
    by exactly the TX spatial phase pi * n_rx * sin(theta)."""
    cfg = CfgStub(n_tx=2, mimo="tdm", n_rx=4)
    theta_deg = 12.0
    adc = synthesize_adc(cfg, [target_at(12.0, theta_deg)], RadarPose(), snr_db=None,
                         seed=10, device=torch_device)
    ratio = torch.angle((adc[:, 1, :] * torch.conj(adc[:, 0, :])).sum())
    exp = math.pi * cfg.n_rx * math.sin(math.radians(theta_deg))
    exp = (exp + math.pi) % (2 * math.pi) - math.pi
    assert float(ratio) == pytest.approx(exp, abs=1e-3)

    # boresight target -> no TX phase difference
    adc0 = synthesize_adc(cfg, [target_at(12.0, 0.0)], RadarPose(), snr_db=None,
                          seed=10, device=torch_device)
    assert float(torch.angle((adc0[:, 1, :] * torch.conj(adc0[:, 0, :])).sum())) == \
        pytest.approx(0.0, abs=1e-3)


def test_tdm_virtual_array_is_uniform(torch_device):
    """Stitching TX0/TX1 chirps must yield one uniform lambda/2 ULA of n_tx*n_rx
    elements: the phase step across the seam equals the step inside a TX block."""
    cfg = CfgStub(n_tx=2, mimo="tdm", n_rx=4)
    theta_deg = 10.0
    adc = synthesize_adc(cfg, [target_at(12.0, theta_deg)], RadarPose(), snr_db=None,
                         seed=11, device=torch_device)
    virt = torch.cat([adc[:, 0, 0], adc[:, 1, 0]])          # v = t*n_rx + r
    step = torch.angle(virt[1:] * torch.conj(virt[:-1])).cpu().numpy()
    exp = math.pi * math.sin(math.radians(theta_deg))
    assert np.allclose(step, exp, atol=1e-3)


def test_ddma_creates_n_tx_doppler_replicas(torch_device):
    """DDMA aliasing signature: one static target -> n_tx Doppler peaks spaced
    n_chirps/n_tx bins apart."""
    cfg = CfgStub(n_tx=2, mimo="ddma", n_rx=4)
    adc = synthesize_adc(cfg, [target_at(12.0, 15.0)], RadarPose(), snr_db=None,
                         seed=12, device=torch_device)
    rd = _doppler_spectrum(cfg, adc)                        # all chirps (ddma)
    k_r = int(np.argmax(rd.sum(axis=0)))
    prof = rd[:, k_r]
    order = np.argsort(prof)[::-1]
    top2 = sorted(order[:2].tolist())
    assert top2 == [0, cfg.n_chirps // cfg.n_tx]
    assert prof[top2[1]] > 0.5 * prof[top2[0]]
    assert prof.max() > 20.0 * np.median(prof)


# --------------------------------------------------------------------------------
# Noise / determinism
# --------------------------------------------------------------------------------
def test_post_fft_snr_matches_requested(torch_device):
    """The documented convention: strongest scatterer's peak in an unwindowed 2-D FFT
    of one RX channel sits `snr_db` above the mean noise floor of that FFT."""
    cfg = single_tx_cfg(n_rx=2, n_chirps=64, n_samples=128)
    k_r = 16
    r_m = k_r * C_LIGHT / (2.0 * cfg.bandwidth_hz)          # exactly on bin -> no straddle
    snr_db = 30.0
    adc = synthesize_adc(cfg, [target_at(r_m, 0.0)], RadarPose(), snr_db=snr_db,
                         seed=13, device=torch_device)
    rd = _doppler_spectrum(cfg, adc)                        # [n_chirps, n_samples]
    peak = rd[0, k_r]
    floor = rd[3:cfg.n_chirps - 2, :].mean()                # exclude the DC Doppler rows
    meas_db = 10.0 * math.log10(peak / floor)
    assert meas_db == pytest.approx(snr_db, abs=1.0)


def test_seeded_output_is_deterministic(torch_device):
    cfg = CfgStub()
    tgts = [target_at(12.0, 10.0, v_radial=2.0)]
    a = synthesize_adc(cfg, tgts, RadarPose(), snr_db=15.0, seed=99, device=torch_device)
    b = synthesize_adc(cfg, tgts, RadarPose(), snr_db=15.0, seed=99, device=torch_device)
    assert torch.equal(a, b)

    c = synthesize_adc(cfg, tgts, RadarPose(), snr_db=15.0, seed=100, device=torch_device)
    assert not torch.equal(a, c)


def test_noise_free_case_is_exactly_reproducible(torch_device):
    cfg = CfgStub()
    tgts = [target_at(9.0, -5.0), target_at(18.0, 20.0, v_radial=-1.0)]
    a = synthesize_adc(cfg, tgts, RadarPose(), snr_db=None, seed=42, device=torch_device)
    b = synthesize_adc(cfg, tgts, RadarPose(), snr_db=None, seed=42, device=torch_device)
    assert torch.equal(a, b)


def test_random_phase_is_a_constant_reflection_phase(torch_device):
    """Random phase must not perturb range/Doppler/angle -- only the global phase."""
    cfg = single_tx_cfg()
    tgts = [target_at(12.0, 8.0, v_radial=1.5)]
    a = synthesize_adc(cfg, tgts, RadarPose(), snr_db=None, seed=5,
                       random_phase=False, device=torch_device)
    b = synthesize_adc(cfg, tgts, RadarPose(), snr_db=None, seed=5,
                       random_phase=True, device=torch_device)
    ratio = b / a
    assert torch.allclose(torch.abs(ratio), torch.ones_like(torch.abs(ratio)), atol=1e-3)
    ph = torch.angle(ratio)
    assert float(torch.std(ph)) < 1e-3


def test_rvp_term_is_a_small_per_chirp_phase(torch_device):
    """Enabling the residual video phase must not move the range bin."""
    cfg = single_tx_cfg()
    tgts = [target_at(12.0, 0.0)]
    a = synthesize_adc(cfg, tgts, RadarPose(), snr_db=None, seed=1,
                       include_rvp=False, device=torch_device)
    b = synthesize_adc(cfg, tgts, RadarPose(), snr_db=None, seed=1,
                       include_rvp=True, device=torch_device)
    ka = int(torch.argmax(torch.abs(range_fft(a[0, 0]))))
    kb = int(torch.argmax(torch.abs(range_fft(b[0, 0]))))
    assert ka == kb
    # -pi S tau^2 at 12 m with S = 1.5625e13 Hz/s -> ~0.02 cycles
    dphi = float(torch.angle((b[0, 0] * torch.conj(a[0, 0])).sum()))
    exp = -math.pi * cfg.ramp_slope_hzps * (2 * 12.0 / C_LIGHT) ** 2
    assert dphi == pytest.approx(exp, abs=1e-3)


def test_superposition_is_linear(torch_device):
    cfg = single_tx_cfg()
    t1 = target_at(9.0, -10.0)
    t2 = target_at(20.0, 15.0, v_radial=2.0)
    kw = dict(snr_db=None, random_phase=False, device=torch_device)
    a = synthesize_adc(cfg, [t1], RadarPose(), **kw)
    b = synthesize_adc(cfg, [t2], RadarPose(), **kw)
    both = synthesize_adc(cfg, [t1, t2], RadarPose(), **kw)
    assert torch.allclose(both, a + b, atol=1e-5 * float(torch.abs(both).max()))


def test_pose_translation_and_rotation(torch_device):
    """Moving/rotating the radar with the target must leave the signal unchanged."""
    cfg = single_tx_cfg()
    kw = dict(snr_db=None, random_phase=False, device=torch_device)
    a = synthesize_adc(cfg, [target_at(12.0, 10.0)], RadarPose(), **kw)

    # same relative geometry, radar translated and looking along +y
    th = math.radians(10.0)
    rel = (12.0 * math.cos(th), 12.0 * math.sin(th), 0.0)
    # boresight +y -> array axis -x, so the lateral offset flips sign
    pose = RadarPose(position=(5.0, -2.0, 1.0), boresight=(0.0, 1.0, 0.0))
    pos = (5.0 - rel[1], -2.0 + rel[0], 1.0)
    b = synthesize_adc(cfg, [ScatStub(position=pos)], pose, **kw)
    assert torch.allclose(a, b, atol=1e-4)
