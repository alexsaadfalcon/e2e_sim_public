"""Tests for `e2e.ml.impairments` (FMCW ADC-domain radar impairments)."""

import math

import os

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
from e2e.ml.radar_config import PRESETS, RadarConfig  # noqa: E402

np = pytest.importorskip("numpy")


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


def _skirt_power(adc, k0, window=3, exclude=1):
    """Mean range-FFT power in bins `exclude < |k - k0| <= window` (wrapped) -- the
    "shoulders" just outside a target's own peak bin, where FMCW phase-noise skirts
    show up."""
    n_samples = adc.shape[-1]
    x = torch.fft.fft(adc, dim=-1)
    p = torch.abs(x) ** 2
    idx = torch.arange(n_samples, device=adc.device)
    dist = torch.minimum((idx - k0) % n_samples, (k0 - idx) % n_samples)
    mask = (dist > exclude) & (dist <= window)
    return p[:, :, mask].mean().item()


# --------------------------------------------------------------------------- phase noise

def test_phase_noise_doppler_axis_range_correlation(cfg, torch_device):
    """Doppler-axis (chirp-to-chirp, STEP B) skirts: far range sees more phase drift
    than near range. Uses single-tone probes rather than a random cube -- now that
    STEP A also perturbs the range axis, a random cube's dense occupancy makes the
    per-bin phase-ratio metric noisy (measured: it flips the old assertion). A tone
    isolates the bin under test so this measures STEP B alone, as intended."""
    n_rx = 4
    a0 = 5.0
    k_near = int(round(_range_to_bin(2.0, cfg))) % cfg.n_samples
    k_far = int(round(_range_to_bin(30.0, cfg))) % cfg.n_samples
    assert k_near != k_far

    def _chirp_phase_rms(k0):
        adc = _tone_cube(n_rx, cfg.n_chirps, cfg.n_samples, k0, a0, torch_device)
        out = apply_phase_noise(adc, cfg, PhaseNoiseParams(), seed=123)
        assert out.shape == adc.shape
        assert out.dtype == adc.dtype
        assert out.device.type == torch_device.type
        x_in = torch.fft.fft(adc, dim=-1)[:, :, k0]
        x_out = torch.fft.fft(out, dim=-1)[:, :, k0]
        delta = torch.angle(x_out / x_in)
        return delta.pow(2).mean().sqrt().item()

    near_rms = _chirp_phase_rms(k_near)
    far_rms = _chirp_phase_rms(k_far)
    # Measured (seed=123, defaults): near_rms ~= 1.0e-3 rad, far_rms ~= 1.4e-2 rad.
    assert far_rms > 1.5 * near_rms


def test_phase_noise_range_axis_skirts_appear(cfg, torch_device):
    """The honesty-debt fix: a strong point target must now show RANGE-axis skirts
    (power leaking into neighboring range-FFT bins), not just Doppler-axis ones.
    Measured (n_rx=4, a0=5.0, far target at 30 m, seed=321, defaults): skirt power
    (mean |X|^2 over bins 2-3 away from the peak) goes from ~3.2e-7 (float noise floor
    of a pure tone) to ~168 -- a ~58 dB increase -- while the peak bin itself is
    ~4.1e5, i.e. skirts sit ~34 dB below the peak, a physically sane FMCW phase-noise
    skirt level."""
    n_rx = 4
    a0 = 5.0
    k0 = int(round(_range_to_bin(30.0, cfg))) % cfg.n_samples
    adc = _tone_cube(n_rx, cfg.n_chirps, cfg.n_samples, k0, a0, torch_device)
    out = apply_phase_noise(adc, cfg, PhaseNoiseParams(), seed=321)

    skirt_in = _skirt_power(adc, k0)
    skirt_out = _skirt_power(out, k0)
    peak_out = torch.abs(torch.fft.fft(out, dim=-1))[:, :, k0].pow(2).mean().item()

    assert skirt_out > 1e4 * max(skirt_in, 1e-30)  # orders of magnitude above the noise-free floor
    assert skirt_out < 0.1 * peak_out  # skirts stay well below the peak (sane relative level)


def test_phase_noise_range_axis_skirts_shrink_with_better_psd(cfg, torch_device):
    """Skirt power must fall as the oscillator improves (lower PSD). Measured (same
    setup as the skirts-appear test): -85 dBc/Hz -> skirt ~168; -110 dBc/Hz (25 dB
    better) -> skirt ~0.54, a ~25 dB drop, tracking the 25 dB PSD improvement."""
    n_rx = 4
    a0 = 5.0
    k0 = int(round(_range_to_bin(30.0, cfg))) % cfg.n_samples
    adc = _tone_cube(n_rx, cfg.n_chirps, cfg.n_samples, k0, a0, torch_device)

    out_default = apply_phase_noise(adc, cfg, PhaseNoiseParams(), seed=321)
    out_better = apply_phase_noise(adc, cfg, PhaseNoiseParams(psd_dbc_hz_at_ref=-110.0), seed=321)

    assert _skirt_power(out_better, k0) < 0.1 * _skirt_power(out_default, k0)


def test_phase_noise_range_axis_correlation(cfg, torch_device):
    """Range correlation must survive on the (newly added) range axis too: a far
    target's skirts exceed a near target's. Measured (n_rx=4, a0=5.0, seed=321,
    defaults): near (2 m) skirt ~0.91, far (30 m) skirt ~168 -- about 180x larger."""
    n_rx = 4
    a0 = 5.0
    k_near = int(round(_range_to_bin(2.0, cfg))) % cfg.n_samples
    k_far = int(round(_range_to_bin(30.0, cfg))) % cfg.n_samples
    assert k_near != k_far

    def _skirt(k0):
        adc = _tone_cube(n_rx, cfg.n_chirps, cfg.n_samples, k0, a0, torch_device)
        out = apply_phase_noise(adc, cfg, PhaseNoiseParams(), seed=321)
        return _skirt_power(out, k0)

    assert _skirt(k_far) > 10.0 * _skirt(k_near)


def test_phase_noise_zero_delay_gate_is_exactly_cancelled(cfg, torch_device):
    """THE range-correlation limit case (2026-08-24 adversarial-review finding): a
    tau = 0 return (the direct TX-RX leakage tone at range-FFT bin 0) sees the SAME
    oscillator on both mixer ports, so its phase-noise residual cancels IDENTICALLY --
    the module's own chain comment calls this out as why direct coupling stays
    coherent in hardware. The old UNIFORM range banding broke exactly this: gate 0
    shared a band with gates [1, 64) and was handed their band-mean delay (~42 ns,
    a ~6 m return's residual), which smeared the +62 dB leakage tone into full-height
    azimuth bands in the leakage+phase attribution figures. Log-spaced banding
    isolates gate 0, so a pure bin-0 tone must pass through BIT-NEAR-UNCHANGED while
    a far tone (same call, same seed) still picks up its skirts."""
    n_rx = 4
    a0 = 5.0
    dc = _tone_cube(n_rx, cfg.n_chirps, cfg.n_samples, 0, a0, torch_device)
    out_dc = apply_phase_noise(dc, cfg, PhaseNoiseParams(), seed=321)
    assert torch.allclose(out_dc, dc, atol=1e-5 * a0), \
        "tau = 0 gate picked up a phase-noise residual that physically cancels"

    # Gate 0's dedicated band exists at EVERY fidelity setting -- the cheapest mode
    # (n_range_bands=1) is two bands, not one, exactly as the docstring now states.
    out_dc_1 = apply_phase_noise(dc, cfg, PhaseNoiseParams(n_range_bands=1), seed=321)
    assert torch.allclose(out_dc_1, dc, atol=1e-5 * a0), \
        "n_range_bands=1 must still isolate the tau = 0 gate"

    k_far = int(round(_range_to_bin(30.0, cfg))) % cfg.n_samples
    far = _tone_cube(n_rx, cfg.n_chirps, cfg.n_samples, k_far, a0, torch_device)
    out_far = apply_phase_noise(far, cfg, PhaseNoiseParams(), seed=321)
    assert _skirt_power(out_far, k_far) > 1e4 * max(_skirt_power(far, k_far), 1e-30), \
        "the far gate must still see phase noise -- the fix must not disable STEP A"


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

    # Against the reference the params DECLARE, not against the cube. The default
    # reference is now the absolute thermal floor (F35 flip), so measuring the tone
    # against the cube's own peak would test a relationship the code no longer asserts --
    # and would quietly start passing again if someone reverted the flip.
    from e2e.ml.impairments import REFERENCE_PEAK, _reference_power
    if params.reference == REFERENCE_PEAK:
        p_ref = (a0 * cfg.n_samples) ** 2
    else:
        p_ref = _reference_power(adc, params.reference, cfg, domain="range_fft")
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
        gain = _k_distributed_gain(n, nu, generator=gen, device=torch_device)
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


# --------------------------------------------------------------- clutter temporal persistence
#
# notes/PHYSICS_JUSTIFICATION_AUDIT.md entry 10: the clutter FIELD (scatterer positions/
# velocities/gains) must be drawn ONCE per seed (frame-independent) and evolve only via a
# deterministic per-scatterer Doppler phase advance keyed to `frame_idx` -- not redrawn
# i.i.d. every frame the way `apply_clutter` used to (old seed was `seed + frame_idx`).

# RETIRED 2026-08-23: `test_clutter_frame0_matches_pre_persistence_behaviour`, the
# bit-identity pin against a frozen pre-A1 reimplementation. Its job was proving the A1
# persistence change altered nothing at frame 0. The F52 array-structure fix (steering +
# MIMO code -- see the tests below) deliberately changed the field's realization, so the
# historical pin no longer holds and is not supposed to. Frame-0 determinism itself is
# still pinned by `test_clutter_deterministic` / `test_clutter_multiframe_determinism`.


def _replay_clutter_draws(cfg, params, seed, device):
    """Replay `apply_clutter`'s field draws to recover what it drew -- positions,
    velocities, azimuths -- for oracle tests. This IS a pin of the draw-order contract
    (positions -> velocities -> azimuths -> texture -> speckle): reordering the draws
    inside `apply_clutter` breaks persistence semantics for stored corpora, and this
    helper makes that break loud here."""
    n_scat = max(1, int(round(float(params.density) * cfg.n_samples)))
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    ranges = torch.rand(n_scat, generator=gen, device=device, dtype=torch.float64) * float(cfg.max_range_m)
    vel = torch.randn(n_scat, generator=gen, device=device, dtype=torch.float64) * float(params.doppler_std_mps)
    sin_az = torch.rand(n_scat, generator=gen, device=device, dtype=torch.float64) * 2.0 - 1.0
    return ranges, vel, sin_az


def _one_scatterer_params():
    # density such that max(1, round(density * n_samples)) == 1 for any n_samples used
    # here -> exactly ONE clutter scatterer, so its azimuth is recoverable.
    return ClutterParams(density=1e-6, nu=50.0, doppler_std_mps=0.0, total_relative_db=0.0)


def test_clutter_azimuth_recovered_through_tdm_deinterleave(torch_device):
    """F52 oracle (TDM): a single clutter scatterer, processed exactly the way the
    detection path processes clutter (deinterleave -> per-TX virtual array -> angle
    FFT), must localize at the azimuth `apply_clutter` drew for it. The pre-F52 model
    (i.i.d. per-RX gains, no code) put a period-n_rx comb here instead."""
    from e2e.ml.transforms import adc_to_rd, tdm_deinterleave
    import dataclasses as _dc

    cfg = RadarConfig(name="f52_tdm", f0_hz=77e9, bandwidth_hz=500e6, n_tx=4, n_rx=4,
                      n_chirps=64, n_samples=128, fs_hz=10e6, chirp_period_s=20e-6,
                      mimo="tdm")
    assert not cfg.validate()
    params = _one_scatterer_params()
    silent = torch.zeros(cfg.n_rx, cfg.n_chirps, cfg.n_samples, dtype=torch.complex64,
                         device=torch_device)
    seed = 314
    clutter = apply_clutter(silent, cfg, params, seed=seed) - silent
    _, _, sin_az = _replay_clutter_draws(cfg, params, seed, torch_device)

    sub_cfg = _dc.replace(cfg, n_tx=1, mimo="single", n_chirps=cfg.n_chirps_per_tx)
    rd = adc_to_rd(sub_cfg, tdm_deinterleave(cfg, clutter))   # [n_virtual, R, D]
    snap = rd.reshape(cfg.n_virtual, -1)
    n_fft = 256
    spec = torch.fft.fftshift(torch.fft.fft(snap, n=n_fft, dim=0), dim=0)
    peak_bin = int(torch.argmax((spec.abs() ** 2).sum(dim=1)))
    sin_est = 2.0 * (peak_bin - n_fft // 2) / n_fft
    rayleigh = 2.0 / cfg.n_virtual
    # CIRCULAR distance (C4 root-cause, 2026-08-25): sin(az) is a 2-periodic axis --
    # +1 and -1 are the same end-fire direction, and a draw within ~rayleigh/2 of +1
    # correctly localizes at the bin that DECODES to -1.0. The old linear abs() scored
    # that correct answer as error ~2.0: a latent ~0.2%-of-seeds flake, device-
    # dependent because CPU and CUDA generators draw different sin_az from one seed.
    d = _wrap_sin_az(sin_est - float(sin_az[0]))
    assert abs(d) <= rayleigh, \
        f"clutter localized at sin(az)={sin_est:.3f}, drawn {float(sin_az[0]):.3f}"


def _wrap_sin_az(d: float) -> float:
    """Project a sin(az) difference onto the fundamental period (-1, 1] -- the
    direction-cosine axis of a half-wavelength array is 2-periodic (see the
    circular-distance comment in the TDM azimuth-recovery oracle)."""
    return d - 2.0 * round(d / 2.0)


def test_clutter_azimuth_recovered_through_ddma_demux(torch_device):
    """F52 oracle (DDMA): same single-scatterer recovery through `ddma_demux` -- the
    code must place the scatterer's replicas so the demux reassembles the full
    n_tx*n_rx virtual aperture at the drawn azimuth (pre-F52: all energy was confined
    to TX-0's sub-band, a 1/n_tx aperture)."""
    from e2e.ml.transforms import adc_to_rd, ddma_demux

    cfg = RadarConfig(name="f52_ddma", f0_hz=77e9, bandwidth_hz=500e6, n_tx=3, n_rx=4,
                      n_chirps=63, n_samples=128, fs_hz=10e6, chirp_period_s=20e-6,
                      mimo="ddma")
    assert not cfg.validate()
    params = _one_scatterer_params()
    silent = torch.zeros(cfg.n_rx, cfg.n_chirps, cfg.n_samples, dtype=torch.complex64,
                         device=torch_device)
    seed = 2718
    clutter = apply_clutter(silent, cfg, params, seed=seed) - silent
    _, _, sin_az = _replay_clutter_draws(cfg, params, seed, torch_device)

    virt = ddma_demux(cfg, adc_to_rd(cfg, clutter))           # [n_tx*n_rx, R, D_sub]
    # Full aperture reassembled: no TX group may be starved of the scatterer's energy.
    per_group = (virt.abs() ** 2).reshape(cfg.n_tx, cfg.n_rx, -1).sum(dim=(1, 2))
    assert float(per_group.min() / per_group.max()) > 0.5, \
        "energy confined to a subset of TX slots -- code missing (pre-F52 behaviour)"

    snap = virt.reshape(cfg.n_virtual, -1)
    n_fft = 256
    spec = torch.fft.fftshift(torch.fft.fft(snap, n=n_fft, dim=0), dim=0)
    peak_bin = int(torch.argmax((spec.abs() ** 2).sum(dim=1)))
    sin_est = 2.0 * (peak_bin - n_fft // 2) / n_fft
    rayleigh = 2.0 / cfg.n_virtual
    # Circular distance -- same C4 wrap-around fix as the TDM oracle above.
    d = _wrap_sin_az(sin_est - float(sin_az[0]))
    assert abs(d) <= rayleigh, \
        f"clutter localized at sin(az)={sin_est:.3f}, drawn {float(sin_az[0]):.3f}"


def test_leakage_tdm_slots_carry_independent_patterns(torch_device):
    """F52 (leakage, TDM): TX t's coupling taps appear only on TX t's chirps, with
    per-(TX, RX) phases -- so the deinterleaved TX slots hold DIFFERENT per-RX
    patterns. The pre-F52 model was chirp-constant, which made every slot identical
    (the exact period-n_rx replication of the comb artifact)."""
    cfg = RadarConfig(name="f52_leak_tdm", f0_hz=77e9, bandwidth_hz=500e6, n_tx=2,
                      n_rx=8, n_chirps=64, n_samples=128, fs_hz=10e6,
                      chirp_period_s=20e-6, mimo="tdm")
    assert not cfg.validate()
    silent = torch.zeros(cfg.n_rx, cfg.n_chirps, cfg.n_samples, dtype=torch.complex64,
                         device=torch_device)
    leak = apply_leakage(silent, cfg, LeakageParams(), seed=99) - silent
    # Per-chirp leakage pattern at the DC (leakage) fast-time bin:
    dc = torch.fft.fft(leak, dim=-1)[:, :, 0]                 # [n_rx, n_chirps]
    slot0, slot1 = dc[:, 0::2], dc[:, 1::2]
    # Within a slot the tap is constant across that TX's chirps...
    assert torch.allclose(slot0, slot0[:, :1].expand_as(slot0), atol=1e-4 * dc.abs().max().item())
    assert torch.allclose(slot1, slot1[:, :1].expand_as(slot1), atol=1e-4 * dc.abs().max().item())
    # ...but the two TX slots must NOT be the same pattern (pre-F52 they were equal).
    assert not torch.allclose(slot0[:, 0], slot1[:, 0],
                              atol=1e-3 * dc.abs().max().item())


def test_leakage_power_calibration_holds_under_ddma(torch_device):
    """The dB knob keeps its meaning under the code: DDMA's per-chirp sum of n_tx
    random-phase taps averages to the SAME per-RX tone power the single-TX model
    injected (the 1/sqrt(n_tx) normalization)."""
    cfg = RadarConfig(name="f52_leak_ddma", f0_hz=77e9, bandwidth_hz=500e6, n_tx=4,
                      n_rx=8, n_chirps=64, n_samples=128, fs_hz=10e6,
                      chirp_period_s=20e-6, mimo="ddma")
    assert not cfg.validate()
    from e2e.ml.impairments import _reference_power
    params = LeakageParams(bumper_relative_db=-300.0)  # isolate the 0 m tone
    silent = torch.zeros(cfg.n_rx, cfg.n_chirps, cfg.n_samples, dtype=torch.complex64,
                         device=torch_device)
    leak = apply_leakage(silent, cfg, params, seed=5) - silent
    p_ref = _reference_power(silent, params.reference, cfg, domain="range_fft")
    power = (torch.fft.fft(leak, dim=-1)[:, :, 0].abs() ** 2).mean().item()
    observed_db = 10.0 * math.log10(power / p_ref)
    assert abs(observed_db - params.leakage_relative_db) < 1.5


def test_clutter_persists_across_frames_and_still_evolves(torch_device):
    """Persistence (b) and evolution (c) together.

    Bound derivation for (b): each scatterer's frame-to-frame phase advance is
    `delta_s = 2*pi*f_dop_s*T_frame`, `f_dop_s = 2*v_s/lambda`, `T_frame =
    1/frame_rate_hz` (`apply_clutter`'s "TEMPORAL BEHAVIOUR"). `v_s ~
    N(0, doppler_std_mps^2)`, so `delta_s ~ N(0, sigma_delta^2)` with
    `sigma_delta = 2*pi*(2*doppler_std_mps/lambda)*(1/frame_rate_hz)`. The clutter
    return is a sum, over many independently-phased scatterers, of unit-magnitude
    phasors that each rotate by `delta_s` from frame 0 to frame 1; summed over (rx,
    chirp, range) samples the cross terms between DIFFERENT scatterers average toward
    zero (their fast/slow-time phases are independent), leaving the same-scatterer
    ("diagonal") term to dominate the frame0-frame1 cross-correlation, whose expected
    value is the Gaussian characteristic function `E[exp(-j*delta_s)] =
    exp(-sigma_delta^2/2)`. Picking `doppler_std_mps`/`frame_rate_hz` so
    `sigma_delta << 1 rad` pins that bound near 1; the test asserts a conservative 0.9x
    of it, comfortably above the ~0 correlation the old i.i.d.-redraw-every-frame
    behaviour produced (see PHYSICS_JUSTIFICATION_AUDIT.md entry 10) yet well short of
    the theoretical bound, leaving slack for the diagonal-dominance approximation.
    """
    cfg = RadarConfig(
        name="persist_cfg", f0_hz=77e9, bandwidth_hz=500e6, n_tx=1, n_rx=4,
        n_chirps=32, n_samples=64, fs_hz=10e6, chirp_period_s=20e-6, mimo="single",
        frame_rate_hz=2000.0,
    )
    assert not cfg.validate()
    params = ClutterParams(density=1.0, nu=1.0, doppler_std_mps=0.01, total_relative_db=0.0)

    sigma_dop = 2.0 * params.doppler_std_mps / cfg.wavelength_m
    t_frame = 1.0 / cfg.frame_rate_hz
    sigma_delta = 2.0 * math.pi * sigma_dop * t_frame
    expected_corr = math.exp(-0.5 * sigma_delta ** 2)
    assert expected_corr > 0.999  # sanity: confirms the chosen regime is near-lossless

    # A constant (zero) input cube fed as both frame 0 and frame 1: the "clutter-only
    # contribution" is then exactly `out - adc`, with no floating-point cancellation.
    adc = torch.zeros(cfg.n_rx, cfg.n_chirps, cfg.n_samples, dtype=torch.complex64, device=torch_device)
    seed = 4242
    out0 = apply_clutter(adc, cfg, params, seed=seed, frame_idx=0)
    out1 = apply_clutter(adc, cfg, params, seed=seed, frame_idx=1)

    c0, c1 = out0 - adc, out1 - adc
    num = torch.sum(c0 * torch.conj(c1))
    den = torch.sqrt(torch.sum(torch.abs(c0) ** 2) * torch.sum(torch.abs(c1) ** 2))
    corr = torch.abs(num / den).item()
    assert corr > 0.9 * expected_corr, (
        f"frame-to-frame clutter correlation {corr:.4f} is far below the "
        f"{expected_corr:.4f} a persistent field implies -- looks i.i.d. redrawn")

    # (c) Evolution is real: frame 1 must NOT equal frame 0 -- the phase advance is
    # nonzero for scatterers with nonzero drawn velocity.
    assert not torch.equal(out0, out1)


def test_clutter_multiframe_determinism(cfg, torch_device):
    """(d) Same (seed, frame_idx) sequence -> bit-identical; different seed -> differs."""
    adc = _rand_cube(4, cfg.n_chirps, cfg.n_samples, torch_device, seed=61)
    params = ClutterParams()

    seq_a = [apply_clutter(adc, cfg, params, seed=100, frame_idx=i) for i in range(3)]
    seq_b = [apply_clutter(adc, cfg, params, seed=100, frame_idx=i) for i in range(3)]
    for a, b in zip(seq_a, seq_b):
        assert torch.equal(a, b)

    seq_c = [apply_clutter(adc, cfg, params, seed=101, frame_idx=i) for i in range(3)]
    assert any(not torch.equal(a, c) for a, c in zip(seq_a, seq_c))


# --------------------------------------------------------------------------- chain

def test_apply_all_defaults_and_skip(cfg, torch_device):
    adc = _rand_cube(4, cfg.n_chirps, cfg.n_samples, torch_device, seed=31)

    out_all = apply_all(adc, cfg, seed=1)
    assert out_all.shape == adc.shape
    assert out_all.dtype == adc.dtype
    assert not torch.allclose(out_all, adc)

    from e2e.ml.impairments import stage_seed
    out_skip = apply_all(adc, cfg, {"leakage": None, "clutter": None}, seed=1)
    out_phase_only = apply_phase_noise(adc, cfg, PhaseNoiseParams(),
                                       seed=stage_seed(1, "phase_noise"))
    assert torch.allclose(out_skip, out_phase_only, atol=1e-6)


def test_apply_all_deterministic(cfg, torch_device):
    adc = _rand_cube(4, cfg.n_chirps, cfg.n_samples, torch_device, seed=33)
    out1 = apply_all(adc, cfg, seed=99)
    out2 = apply_all(adc, cfg, seed=99)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_apply_all_clutter_seed_is_frame_independent_others_are_not(cfg, torch_device):
    """`apply_all`'s `seed` is the BASE seed: clutter keys off it directly (same field
    every frame), while phase_noise/leakage key off `seed + frame_idx` (fresh draw
    every frame) -- see `apply_all`'s docstring and PHYSICS_JUSTIFICATION_AUDIT.md
    entry 10."""
    from e2e.ml.impairments import stage_seed

    silent = torch.zeros(4, cfg.n_chirps, cfg.n_samples, dtype=torch.complex64, device=torch_device)

    # Isolate clutter alone: with leakage/phase_noise skipped, apply_all's output on a
    # silent cube IS the clutter field, directly comparable to a raw apply_clutter call.
    out_frame0 = apply_all(silent, cfg, {"leakage": None, "phase_noise": None}, seed=5, frame_idx=0)
    out_frame3 = apply_all(silent, cfg, {"leakage": None, "phase_noise": None}, seed=5, frame_idx=3)
    direct_frame0 = apply_clutter(silent, cfg, ClutterParams(), seed=stage_seed(5, "clutter"), frame_idx=0)
    direct_frame3 = apply_clutter(silent, cfg, ClutterParams(), seed=stage_seed(5, "clutter"), frame_idx=3)
    assert torch.equal(out_frame0, direct_frame0)
    assert torch.equal(out_frame3, direct_frame3)
    # Same field (same base seed), evolved -- not independently redrawn.
    assert not torch.equal(out_frame0, out_frame3)

    # Phase noise, by contrast, must use seed + frame_idx (a fresh draw every frame) --
    # needs a NON-zero cube, since phase noise multiplies existing signal and a silent
    # cube stays silent regardless of the phase applied to it.
    adc = _rand_cube(4, cfg.n_chirps, cfg.n_samples, torch_device, seed=71)
    out_pn_frame0 = apply_all(adc, cfg, {"leakage": None, "clutter": None}, seed=5, frame_idx=0)
    out_pn_frame3 = apply_all(adc, cfg, {"leakage": None, "clutter": None}, seed=5, frame_idx=3)
    direct_pn_frame0 = apply_phase_noise(adc, cfg, PhaseNoiseParams(), seed=stage_seed(5, "phase_noise"))
    direct_pn_frame3 = apply_phase_noise(adc, cfg, PhaseNoiseParams(), seed=stage_seed(8, "phase_noise"))
    assert torch.allclose(out_pn_frame0, direct_pn_frame0, atol=1e-6)
    assert torch.allclose(out_pn_frame3, direct_pn_frame3, atol=1e-6)


def test_clutter_passes_through_the_oscillator_phase_noise(cfg, torch_device):
    """Leakage and clutter are RETURNS: they reach the mixer with the echoes and must be
    subject to the same phase noise. With phase noise applied first they escaped it
    entirely. Test: inject clutter alone into a silent cube, then check the composite is
    modified by the phase-noise stage rather than passing through untouched."""
    import torch
    from e2e.ml.impairments import (ClutterParams, PhaseNoiseParams, apply_all,
                                    apply_clutter, apply_phase_noise)

    silent = torch.zeros(cfg.n_rx, cfg.n_chirps, cfg.n_samples,
                         dtype=torch.complex64, device=torch_device)
    # Give the cube one strong return so clutter power has something to scale against.
    silent[:, :, 0] = 1.0

    cluttered = apply_clutter(silent, cfg, ClutterParams(), seed=1)
    both = apply_phase_noise(cluttered, cfg, PhaseNoiseParams(), seed=3)
    assert not torch.equal(both, cluttered), "phase noise must act on the clutter too"

    # And the chained order must reach the same arrangement: the last stage applied is
    # phase noise, so the chain output differs from the un-phase-noised composite.
    chained = apply_all(silent, cfg, {"leakage": None}, seed=1)
    assert not torch.equal(chained, apply_clutter(silent, cfg, ClutterParams(), seed=2))


def test_stage_seeds_never_collide_across_frames_or_stages():
    """The corpus advances the frame seed by one per frame. With small per-stage
    offsets that made frame i's leakage seed identical to frame i+1's phase-noise
    seed -- the same noise realization filed under two different labels, which
    quietly correlates a corpus. Sweep frames and stages and assert every sub-seed
    is distinct."""
    from e2e.ml.impairments import stage_seed

    stages = ("phase_noise", "leakage", "clutter")
    seeds = {}
    for frame in range(200):
        for stage in stages:
            s = stage_seed(1000 + frame, stage)
            assert s not in seeds, (
                f"collision: (frame {frame}, {stage}) reuses the seed of {seeds[s]}"
            )
            seeds[s] = (frame, stage)


def test_stage_seed_is_stable_across_processes():
    """Reproducibility of a corpus depends on this being independent of PYTHONHASHSEED
    -- Python's built-in hash() is salted per process and would break it."""
    import subprocess
    import sys

    code = ("from e2e.ml.impairments import stage_seed;"
            "print(stage_seed(7, 'clutter'))")
    runs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={**os.environ, "PYTHONHASHSEED": seed}).stdout.strip()
        for seed in ("0", "1", "12345")
    }
    assert len(runs) == 1, f"stage_seed varies with PYTHONHASHSEED: {runs}"


# --------------------------------------------------------------------------------
# The F35 ceiling: what the calibration reference does to target-to-clutter ratio.
# --------------------------------------------------------------------------------
def _cube_with_noise_floor_and_target(cfg, target_gain_db, *, k=40, n_rx=4, n_chirps=16):
    """A fixed thermal-like noise floor plus a target TONE whose amplitude is swept.

    The noise floor must be present and fixed, and the target must be the only thing that
    moves -- that is the whole experiment. A pure tone with no noise will NOT do: its
    time-domain median and peak are the same number, so it cannot tell the two references
    apart (this test was written after that mistake).
    """
    n_s = int(cfg.n_samples)
    g = torch.Generator().manual_seed(7)
    noise = (torch.randn(n_rx, n_chirps, n_s, generator=g)
             + 1j * torch.randn(n_rx, n_chirps, n_s, generator=g)).to(torch.complex64) * 0.01
    t = torch.arange(n_s, dtype=torch.float32)
    tone = torch.exp(2j * math.pi * k * t / n_s).to(torch.complex64)[None, None, :]
    return noise + tone * (10.0 ** (float(target_gain_db) / 20.0))


def _target_to_background_db(adc, k=40):
    x = adc[0, 0].numpy()
    p = np.abs(np.fft.fft(x))
    lo, hi = max(0, k - 2), k + 3
    return 20.0 * np.log10(p[lo:hi].max() / np.median(np.delete(p, np.arange(lo, hi))))


@pytest.mark.parametrize("reference,expect_responds", [("peak", False), ("noise", True)])
def test_clutter_reference_decides_whether_target_gains_survive(reference, expect_responds):
    """Calibrating clutter against the CUBE PEAK makes target-to-clutter invariant to
    target strength -- a hard ceiling in which no RF/scene physics improvement can ever
    improve detectability (notes/ESTABLISHED_FACTS.md F35). Against the NOISE FLOOR, a
    target improvement shows up 1:1, as it must.

    This is pinned because the ceiling was invisible for months and cost a campaign's
    worth of misattributed measurements.
    """
    cfg = PRESETS["radial_like"]
    ratios = []
    for gain_db in (0.0, 20.0):
        adc = _cube_with_noise_floor_and_target(cfg, gain_db)
        out = apply_clutter(adc, cfg,
                            ClutterParams(total_relative_db=-10.0, reference=reference),
                            seed=1)
        ratios.append(_target_to_background_db(out))
    delta = ratios[1] - ratios[0]
    if expect_responds:
        assert delta > 15.0, (
            f"reference={reference!r}: a 20 dB stronger target moved target-to-clutter by "
            f"only {delta:.2f} dB; the impairment is still tracking the target")
    else:
        assert abs(delta) < 2.0, (
            f"reference={reference!r} is the legacy peak-relative behaviour and must stay "
            f"invariant for corpus reproducibility, but moved {delta:.2f} dB")


def test_unknown_power_reference_fails_loudly():
    """A typo'd reference must raise, not silently pick one -- the two give different
    physics."""
    cfg = PRESETS["radial_like"]
    adc = _cube_with_noise_floor_and_target(cfg, 0.0)
    with pytest.raises(ValueError, match="unknown power reference"):
        apply_clutter(adc, cfg, ClutterParams(reference="Noise"), seed=1)
    # NB "thermal" is a VALID reference since the F35 flip -- it is the default. Use a
    # genuine typo here instead, or this test silently stops testing anything.
    with pytest.raises(ValueError, match="unknown power reference"):
        apply_leakage(adc, cfg, LeakageParams(reference="thermel"), seed=1)


def test_clutter_power_calibration_holds_under_ddma(torch_device):
    """`total_relative_db` keeps its per-RX meaning under the DDMA code (the
    `tx_power = n_tx` divisor): E[|tx_factor|^2] = n_tx is exact only in expectation
    over many scatterers/chirps, so this uses a dense field and a tolerance. Review
    finding 2026-08-23: without this, removing the divisor would ship silently --
    the leakage analogue was tested, the clutter one was not."""
    cfg = RadarConfig(name="f52_clutpow_ddma", f0_hz=77e9, bandwidth_hz=500e6, n_tx=3,
                      n_rx=4, n_chirps=63, n_samples=128, fs_hz=10e6,
                      chirp_period_s=20e-6, mimo="ddma")
    assert not cfg.validate()
    from e2e.ml.impairments import _thermal_reference
    params = ClutterParams(density=8.0, nu=50.0, total_relative_db=10.0)
    silent = torch.zeros(cfg.n_rx, cfg.n_chirps, cfg.n_samples, dtype=torch.complex64,
                         device=torch_device)
    clutter = apply_clutter(silent, cfg, params, seed=7) - silent
    target_total = _thermal_reference(cfg, domain="time") * 10.0 ** (params.total_relative_db / 10.0)
    # Injected TOTAL time-domain power per (rx, chirp, sample) grid, relative to target:
    observed = (clutter.abs() ** 2).mean().item()
    observed_db = 10.0 * math.log10(observed / target_total)
    assert abs(observed_db) < 1.5, f"clutter power off calibration by {observed_db:.2f} dB"
