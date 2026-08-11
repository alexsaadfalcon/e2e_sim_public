"""
FMCW radar impairments applied to dechirped ADC cubes.

Operates on the same contract as `e2e.ml.rt_gen.rt_synthesize_adc` (and
`e2e.ml.rd_synth.synthesize_adc`): a `complex64` tensor `adc[n_rx, n_chirps, n_samples]`
on some torch device, with fast-time sample `n` of chirp `c` holding
`exp(j2pi f_RF(n) tau)` for a target at delay `tau` (see `rt_gen`'s module docstring,
equation (3)) -- i.e. an FFT along the last (fast-time) axis of one chirp gives a range
spectrum whose bin `k` is delay `tau(k) = k / bandwidth_hz` (derived in
`apply_phase_noise` below), and an FFT along the chirp axis gives Doppler.

Three impairments, ranked the top realism gaps by an external audit pass:

1. `apply_phase_noise`   -- range-correlated oscillator phase noise (near-range
   cancellation, far-range less), applied both within a chirp (range-axis skirts)
   and chirp-to-chirp (Doppler-axis skirts).
2. `apply_leakage`       -- TX-RX direct coupling + a short-range bumper/radome
   reflection.
3. `apply_clutter`       -- heavy-tailed (K-distributed) diffuse ground clutter,
   near-zero Doppler.

Each impairment is a pure function `f(adc, cfg, params, *, seed) -> adc` (same shape/
dtype/device in and out) plus a plain-float `@dataclass` params class so the corpus
stage can domain-randomize per frame and serialize the choice to JSON. `apply_all`
chains all three. Everything is torch/numpy only (no Sionna), deterministic given
`seed`, and device-agnostic (never hardcodes `cpu`).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

# Speed of light, m/s (matches `e2e.ml.radar_config.C_MPS`; duplicated to keep this
# module's only cross-package dependency on radar_config limited to attribute access
# on the `cfg` object it's handed, not an import of the constant).
C_MPS = 299_792_458.0


# --------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------
def _range_to_bin(range_m: float, cfg) -> float:
    """Fractional fast-time DFT bin for a scatterer at `range_m`.

    Beat frequency `f_beat = slope * 2r/c` (same mapping `rt_gen.beat_frequencies`
    uses); DFT bin spacing is `fs/n_samples`, so `bin = f_beat * n_samples / fs`.
    """
    f_beat = float(cfg.ramp_slope_hzps) * 2.0 * float(range_m) / C_MPS
    return f_beat * float(cfg.n_samples) / float(cfg.fs_hz)


def _range_fft_peak_power(adc: torch.Tensor) -> float:
    """Max |per-chirp range-FFT|^2 over the whole cube.

    The shared "cube's peak" reference the relative-power (dB) parameters below are
    measured against: an unwindowed FFT along fast-time, per chirp, peak magnitude
    squared over (rx, chirp, range-bin). No Doppler-coherent integration -- keeps the
    reference well-defined even for a cube with only a handful of chirps.
    """
    mag2 = torch.abs(torch.fft.fft(adc, dim=-1)) ** 2
    return float(torch.max(mag2).item())


def _sample_gamma(n: int, shape: float, *, generator: torch.Generator,
                   device: torch.device) -> torch.Tensor:
    """Marsaglia-Tsang Gamma(`shape`, scale=1) sampler, driven by `generator`.

    Implemented by hand (rather than `torch.distributions.Gamma` /
    `torch._standard_gamma`) so the draw is reproducible from a plain
    `torch.Generator` across torch versions/devices -- the public gamma APIs don't
    consistently accept an explicit generator. Standard rejection algorithm (Marsaglia
    & Tsang 2000); for `shape < 1` samples Gamma(shape+1) and applies the usual
    `U^(1/shape)` boost.
    """
    shape = float(shape)
    d = (shape + 1.0 if shape < 1.0 else shape) - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    out = torch.empty(n, device=device, dtype=torch.float64)
    remaining = torch.arange(n, device=device)
    for _ in range(200):  # typical MT acceptance rate ~0.96; this bound is generous
        if remaining.numel() == 0:
            break
        m = remaining.numel()
        x = torch.randn(m, generator=generator, device=device, dtype=torch.float64)
        v = (1.0 + c * x) ** 3
        ok = v > 0
        u = torch.rand(m, generator=generator, device=device, dtype=torch.float64)
        log_u = torch.log(u.clamp_min(1e-300))
        accept = ok & (log_u < 0.5 * x ** 2 + d - d * v + d * torch.log(v.clamp_min(1e-300)))
        acc = remaining[accept]
        out[acc] = d * v[accept]
        remaining = remaining[~accept]
    if remaining.numel() > 0:  # pragma: no cover -- astronomically unlikely
        out[remaining] = d
    if shape < 1.0:
        u2 = torch.rand(n, generator=generator, device=device, dtype=torch.float64).clamp_min(1e-300)
        out = out * u2 ** (1.0 / shape)
    return out


def _k_distributed_gain(n_scat: int, n_rx: int, nu: float, *, generator: torch.Generator,
                         device: torch.device) -> torch.Tensor:
    """K-distributed complex gain `[n_scat, n_rx]`: sqrt(gamma texture) * CN(0,1) speckle.

    Texture ~ Gamma(shape=nu, scale=1/nu) (mean 1, variance 1/nu -- small `nu` means a
    heavier-tailed, more variable RCS, the classic K-distribution clutter model).
    Speckle is drawn independently per (scatterer, rx) -- diffuse clutter decorrelates
    spatially across the array -- but texture is shared across rx: a scatterer's RCS
    fluctuation is a property of the patch of ground, not of which antenna looks at it.
    `E[|gain|^2] = 1`.
    """
    tex = _sample_gamma(n_scat, nu, generator=generator, device=device) / float(nu)
    real = torch.randn((n_scat, n_rx), generator=generator, device=device, dtype=torch.float64)
    imag = torch.randn((n_scat, n_rx), generator=generator, device=device, dtype=torch.float64)
    speckle = (real + 1j * imag) * math.sqrt(0.5)  # E[|speckle|^2] = 1
    return torch.sqrt(tex).unsqueeze(1).to(speckle.dtype) * speckle


# --------------------------------------------------------------------------------
# 1. Range-correlated oscillator phase noise
# --------------------------------------------------------------------------------
@dataclass
class PhaseNoiseParams:
    """Oscillator phase-noise PSD, single-sideband dBc/Hz, -20 dB/decade slope."""

    psd_dbc_hz_at_ref: float = -85.0   # dBc/Hz at ref_offset_hz
    ref_offset_hz: float = 1.0e6       # reference offset, Hz
    n_range_bands: int = 8             # fast-time range-segmentation fidelity/cost knob;
                                        # see `apply_phase_noise` STEP A


def apply_phase_noise(adc: torch.Tensor, cfg, params: PhaseNoiseParams, *,
                       seed: int) -> torch.Tensor:
    """Range-correlated FMCW phase-noise residual -- now smears BOTH range and Doppler.

    An FMCW dechirp mixes the echo (delayed by `tau`) against the live TX ramp, so a
    noisy oscillator's phase `phi(t)` (`t` = FAST time, i.e. within one chirp) appears
    as the *residual* `phi(t) - phi(t - tau)` on the beat signal (this module's
    `s_b(t) = s_t(t) conj(s_t(t-tau))` convention with `phi(t)` added to the TX phase
    reduces exactly to this; `rt_gen`'s eq. (1) is the noise-free special case): near
    ranges (`tau` small) largely cancel, far ranges see nearly the full phase noise --
    the textbook FMCW "range correlation" effect. `phi` varies WITHIN a chirp, so a
    faithful application multiplies the beat SAMPLES by `exp(j(phi(t)-phi(t-tau)))` in
    the TIME domain -- that is what genuinely convolves/smears the range spectrum
    ("skirts" around a target). STEP A below does that; STEP B keeps the original
    per-gate, chirp-to-chirp treatment for Doppler skirts, unchanged.

    STEP A -- within-chirp (fast-time) residual, range-segmented.
    A single time-domain multiply cannot give every range gate its own delay `tau(k)`
    (all gates coexist at every fast-time sample). Compromise adopted here: split the
    `n_samples` range gates into `params.n_range_bands` contiguous bands and give each
    band ONE fast-time residual keyed to its band-mean delay `tau_b`. For band `b`
    covering gates `[k_lo, k_hi)`:
      1. mask the fast-time DFT to just that band's gates and IFFT -> the band's own
         time-domain contribution `y_b[n]` (bands are disjoint in frequency, so this
         is an exact decomposition: `sum_b y_b == adc`, Parseval-orthogonal);
      2. synthesize a real process `dphi_b[chirp, n]` on the ADC fast-time frequency
         axis (0 to `fs_hz/2`) with variance-density
         `S_phi(f) * |1 - exp(-j2pi f tau_b)|^2` -- the SAME oscillator PSD and
         correlation-factor formula STEP B uses, just moved onto the fast-time axis,
         which is actually the MORE faithful axis for a PSD stated as "offset from
         carrier": ADC rates are MHz-scale, close to the usual 1 MHz `ref_offset_hz`,
         whereas STEP B's chirp-rate axis (kHz-scale) is itself an approximation --
         see STEP B's note below;
      3. multiply `y_b` by `exp(j dphi_b)` (broadcast over rx) and re-sum the bands.
    With `n_range_bands=1` this degenerates to the simplest version: one reference
    delay (the whole window's mean tau) applied uniformly to the entire cube.

    STEP B -- per-gate, chirp-to-chirp residual (unchanged from the original model).
    Per-gate delay: FFT bin `k` of one chirp's fast-time samples corresponds to beat
    frequency `f_beat = k * fs / n_samples`; combined with `f_beat = slope * tau`
    (`b[n] = exp(j2pi(f0 + slope*n/fs) tau)`, `rt_gen`'s eq. (1)) and
    `slope = bandwidth / (n_samples/fs)`, this simplifies to `tau(k) = k / bandwidth_hz`
    -- delay grows linearly with gate index, independent of `fs`. `dphi(k, chirp)` is
    synthesized as a colored (chirp-to-chirp correlated) random process per gate, with
    variance-density `S_phi(f) * |1 - exp(-j2pi f tau(k))|^2`, `f` the chirp-to-chirp
    (slow-time) frequency axis and `S_phi(f) = S_phi(ref) * (ref/f)^2` the
    -20 dB/decade oscillator PSD.

    APPROXIMATIONS (documented, not hidden):
    * STEP A is per-BAND, not per-gate: within a band, every gate gets the same
      fast-time residual regardless of its exact delay. Range correlation is
      preserved only at the band's granularity -- coarser than the true continuum,
      finer as `n_range_bands` grows (at proportionally higher synthesis cost).
    * STEP A draws an INDEPENDENT fast-time realization per chirp -- physically the
      same oscillator's phase is continuous across chirps, but the low-frequency
      (chirp-to-chirp-observable) part of that continuity is exactly what STEP B
      already models. Splitting the process into "resolved within one ramp" (A) and
      "resolved across ramps" (B) is a spectral split-of-convenience, not a claim
      that the oscillator literally resets phase every chirp.
    * STEP B remains a **per-gate aggregate**, not per-path: every scatterer (and any
      two-way leakage) landing in gate `k` shares one chirp-to-chirp phase-noise
      draw, rather than each physical path getting its own residual. Correct for one
      dominant scatterer per gate; approximate for overlapping multipath in a gate.
      STEP B's frequency axis is still the CPI's chirp-to-chirp (slow-time) axis
      repurposed as "offset from carrier", because that is the only axis this
      discretized, single-solve ADC cube can resolve chirp-to-chirp correlation on --
      real phase-noise masks extend to much higher offsets than a CPI's slow-time
      Nyquist (PRF/2). STEP A supplies the genuinely-offset-from-carrier fast-time
      axis; STEP B's axis remains an approximation, now clearly scoped to the
      Doppler-visible part of the spectrum. This is the part of the original honesty
      debt that is NARROWED rather than eliminated by this change.
    * The correlation factor (both steps) is applied exactly (`2 - 2cos(2*pi*f*tau)`),
      not just the small-angle approximation quoted in the design note -- but for
      radar-scale delays and frequencies up to each axis's own Nyquist, `f*tau` is
      normally << 1 and the two agree.

    Energy is now only APPROXIMATELY conserved (previously exact): STEP B's
    unit-modulus per-gate multiply is exactly energy-preserving (Parseval), but
    STEP A's per-band phasors make previously-orthogonal (disjoint-frequency) bands'
    time-domain signals interfere -- a second-order-in-phase-noise-variance leakage,
    small for the physically-small `psd_dbc_hz_at_ref` values this module expects
    (verified empirically in the test suite, well under 0.5 dB for defaults).
    """
    n_rx, n_chirps, n_samples = adc.shape
    device, dtype = adc.device, adc.dtype

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    k_idx = torch.arange(n_samples, device=device, dtype=torch.float64)
    tau_gate = k_idx / float(cfg.bandwidth_hz)  # [n_samples], tau(k), see STEP B derivation above

    l_ref_lin = 10.0 ** (float(params.psd_dbc_hz_at_ref) / 10.0)

    # ---- STEP A: within-chirp (fast-time) residual, applied per range band ---------
    n_bands = max(1, min(int(params.n_range_bands), n_samples))
    f_fast = torch.fft.rfftfreq(n_samples, d=1.0 / float(cfg.fs_hz)).to(device=device, dtype=torch.float64)
    n_freq_fast = f_fast.numel()
    f_fast_min = float(cfg.fs_hz) / n_samples  # fundamental fast-time frequency spacing
    f_fast_safe = f_fast.clamp_min(f_fast_min)  # avoid the 1/f^2 singularity at DC
    s_phi_fast = l_ref_lin * (float(params.ref_offset_hz) / f_fast_safe) ** 2  # [n_freq_fast]

    x = torch.fft.fft(adc, dim=-1)  # range-gate domain, per chirp: [n_rx, n_chirps, n_samples]
    y_time = torch.zeros_like(adc)  # accumulates STEP A's band-recombined, time-domain output

    band_edges = [round(b * n_samples / n_bands) for b in range(n_bands + 1)]
    for b in range(n_bands):
        k_lo, k_hi = band_edges[b], band_edges[b + 1]
        if k_hi <= k_lo:
            continue
        mask = torch.zeros(n_samples, dtype=torch.bool, device=device)
        mask[k_lo:k_hi] = True
        y_band = torch.fft.ifft(x * mask.view(1, 1, n_samples), dim=-1)  # this band's time-domain content

        tau_b = float(tau_gate[k_lo:k_hi].mean().item())  # band-representative delay
        corr_b = 2.0 - 2.0 * torch.cos(2.0 * math.pi * f_fast_safe * tau_b)  # [n_freq_fast]
        s_shaped_fast = s_phi_fast * corr_b

        real = torch.randn((n_chirps, n_freq_fast), generator=gen, device=device, dtype=torch.float64)
        imag = torch.randn((n_chirps, n_freq_fast), generator=gen, device=device, dtype=torch.float64)
        # Standard PSD-to-DFT-coefficient-variance synthesis: E[|Y[f]|^2] = S(f)*N/dt.
        scale = torch.sqrt(s_shaped_fast * n_samples * float(cfg.fs_hz) / 2.0)
        yfreq = (real + 1j * imag) * scale
        yfreq[:, 0] = yfreq[:, 0].real  # rfft DC bin of a real signal must be real
        if n_samples % 2 == 0:
            yfreq[:, -1] = yfreq[:, -1].real  # ...and the Nyquist bin, if it exists
        dphi_b = torch.fft.irfft(yfreq, n=n_samples, dim=-1)  # [n_chirps, n_samples], real

        phasor_b = torch.exp(1j * dphi_b.to(torch.float32)).to(dtype)  # [n_chirps, n_samples]
        y_time = y_time + y_band * phasor_b.unsqueeze(0)  # broadcast over rx; shared LO across the array

    x = torch.fft.fft(y_time, dim=-1)  # back to range-gate domain, now fast-time-smeared

    # ---- STEP B: per-gate, chirp-to-chirp (slow-time / Doppler) residual, unchanged --
    n_freq = n_chirps // 2 + 1
    f = torch.fft.rfftfreq(n_chirps, d=float(cfg.chirp_period_s)).to(device=device, dtype=torch.float64)
    f_min = 1.0 / (n_chirps * float(cfg.chirp_period_s))
    f_safe = f.clamp_min(f_min)  # avoid the 1/f^2 singularity at DC

    s_phi = l_ref_lin * (float(params.ref_offset_hz) / f_safe) ** 2  # [n_freq], rad^2/Hz

    phase = 2.0 * math.pi * torch.outer(tau_gate, f_safe)  # [n_samples, n_freq]
    corr = 2.0 - 2.0 * torch.cos(phase)
    s_shaped = s_phi.unsqueeze(0) * corr  # [n_samples, n_freq]

    real = torch.randn((n_samples, n_freq), generator=gen, device=device, dtype=torch.float64)
    imag = torch.randn((n_samples, n_freq), generator=gen, device=device, dtype=torch.float64)
    # Standard PSD-to-DFT-coefficient-variance synthesis: E[|Y[f]|^2] = S(f)*N/dt.
    scale = torch.sqrt(s_shaped * n_chirps / float(cfg.chirp_period_s) / 2.0)
    y = (real + 1j * imag) * scale
    y[:, 0] = y[:, 0].real  # rfft DC bin of a real signal must be real
    if n_chirps % 2 == 0:
        y[:, -1] = y[:, -1].real  # ...and the Nyquist bin, if it exists
    delta_phi = torch.fft.irfft(y, n=n_chirps, dim=-1)  # [n_samples, n_chirps], real

    shift = torch.exp(1j * delta_phi.to(torch.float32)).to(dtype)  # [n_samples, n_chirps]
    x = x * shift.transpose(0, 1).unsqueeze(0)  # broadcast over rx; shared LO across the array

    return torch.fft.ifft(x, dim=-1).to(dtype)


# --------------------------------------------------------------------------------
# 2. TX-RX leakage + bumper/radome reflection
# --------------------------------------------------------------------------------
@dataclass
class LeakageParams:
    """Direct TX-RX coupling and a short-range bumper/radome reflection."""

    leakage_relative_db: float = -5.0    # near-zero-delay coupling tone, dB rel. cube peak
    bumper_range_m: float = 0.2          # bumper/radome reflection range, m
    bumper_relative_db: float = -15.0    # bumper tone power, dB rel. cube peak


def apply_leakage(adc: torch.Tensor, cfg, params: LeakageParams, *, seed: int) -> torch.Tensor:
    """Add a near-zero-delay TX-RX leakage tone plus a short-range bumper reflection.

    Both are modeled as static (chirp-independent) tones, each placed at the nearest
    fast-time DFT bin for its range (`_range_to_bin`, rounded -- both are strong,
    coherent, essentially-fixed-delay returns; snapping to the bin removes spectral
    leakage that would otherwise depend on `n_samples`/`fs` in a way not implied by
    the physical description). Amplitude is per-RX random-phase (`per-RX random
    phase` in the spec) but has the SAME magnitude on every antenna and every chirp --
    a monostatic coupling/short-range reflection has no meaningful per-chirp Doppler
    or, at this fidelity, per-antenna gain variation.

    Power is calibrated against `_range_fft_peak_power(adc)` (the cube's peak,
    measured on the INPUT before this function adds anything), so e.g.
    `leakage_relative_db=-5` places the leakage tone's range-FFT peak power 5 dB below
    the strongest existing return.
    """
    n_rx, n_chirps, n_samples = adc.shape
    device, dtype = adc.device, adc.dtype

    p_ref = _range_fft_peak_power(adc)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    phases = torch.rand((2, n_rx), generator=gen, device=device, dtype=torch.float32) * (2.0 * math.pi)

    n = torch.arange(n_samples, device=device, dtype=torch.float32)

    def _tap(range_m: float, rel_db: float, phase_rx: torch.Tensor) -> torch.Tensor:
        amp = math.sqrt(p_ref * (10.0 ** (float(rel_db) / 10.0))) / n_samples
        k = int(round(_range_to_bin(range_m, cfg))) % n_samples
        tone = amp * torch.exp(1j * (2.0 * math.pi * k * n / n_samples))  # [n_samples]
        tone = tone.to(dtype).view(1, 1, n_samples)
        per_rx = torch.exp(1j * phase_rx.to(torch.float32)).to(dtype).view(n_rx, 1, 1)
        return tone * per_rx  # [n_rx, 1, n_samples], broadcasts over chirps unchanged

    out = adc + _tap(0.0, params.leakage_relative_db, phases[0])
    out = out + _tap(params.bumper_range_m, params.bumper_relative_db, phases[1])
    return out


# --------------------------------------------------------------------------------
# 3. Heavy-tailed diffuse clutter
# --------------------------------------------------------------------------------
@dataclass
class ClutterParams:
    """Diffuse ground clutter: many weak, near-zero-Doppler, K-distributed returns."""

    density: float = 0.5           # scatterers per unambiguous range bin
    nu: float = 1.0                # K-distribution texture shape (small -> heavier tail)
    doppler_std_mps: float = 0.05  # per-scatterer radial-velocity std, m/s
    total_relative_db: float = -10.0  # total clutter power, dB rel. cube's TIME-DOMAIN peak


def apply_clutter(adc: torch.Tensor, cfg, params: ClutterParams, *, seed: int) -> torch.Tensor:
    """Add heavy-tailed diffuse ground clutter.

    `density * n_samples` scatterers are scattered uniformly over the unambiguous
    range window `[0, max_range_m)`, each with a small random radial velocity
    (`N(0, doppler_std_mps^2)`, i.e. clutter sits near Doppler bin 0) and a
    K-distributed complex gain (`_k_distributed_gain`: gamma "texture" times complex
    Gaussian "speckle" -- small `nu` gives a heavy-tailed amplitude distribution, the
    standard sea/ground-clutter model). Total injected power is calibrated to
    `total_relative_db` dB relative to the INPUT cube's time-domain peak power
    (`max(|adc|^2)`) -- unlike `apply_leakage`'s coherent single-bin taps, clutter is a
    sum of many uncorrelated returns, so a time-domain (not range-FFT) peak is the
    simpler well-defined reference; despite the "clutter-to-noise ratio" phrasing in
    the design brief, no explicit noise floor is assumed to be present in `adc`.
    """
    n_rx, n_chirps, n_samples = adc.shape
    device, dtype = adc.device, adc.dtype

    n_scat = max(1, int(round(float(params.density) * n_samples)))
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    ranges = torch.rand(n_scat, generator=gen, device=device, dtype=torch.float64) * float(cfg.max_range_m)
    f_beat = float(cfg.ramp_slope_hzps) * 2.0 * ranges / C_MPS  # [n_scat]

    vel = torch.randn(n_scat, generator=gen, device=device, dtype=torch.float64) * float(params.doppler_std_mps)
    f_dop = 2.0 * vel / float(cfg.wavelength_m)  # [n_scat], near zero

    gain = _k_distributed_gain(n_scat, n_rx, float(params.nu), generator=gen, device=device)  # [n_scat, n_rx]

    peak_power = float(torch.max(torch.abs(adc) ** 2).item())
    target_total = peak_power * (10.0 ** (float(params.total_relative_db) / 10.0))
    mean_power = target_total / n_scat
    gain = gain * math.sqrt(mean_power)

    n = torch.arange(n_samples, device=device, dtype=torch.float64)
    c = torch.arange(n_chirps, device=device, dtype=torch.float64)
    fast_phase = 2.0 * math.pi * torch.outer(f_beat, n) / float(cfg.fs_hz)          # [n_scat, n_samples]
    slow_phase = 2.0 * math.pi * torch.outer(f_dop, c) * float(cfg.chirp_period_s)  # [n_scat, n_chirps]
    fast = torch.exp(1j * fast_phase).to(gain.dtype)
    slow = torch.exp(1j * slow_phase).to(gain.dtype)

    clutter = torch.einsum("sr,sc,sn->rcn", gain, slow, fast)  # [n_rx, n_chirps, n_samples]
    return adc + clutter.to(dtype)


# --------------------------------------------------------------------------------
# Chain
# --------------------------------------------------------------------------------
# ORDER IS PHYSICS, not preference. Leakage and clutter are RETURNS -- they arrive at
# the mixer alongside the target echoes and are therefore subject to the same
# oscillator phase noise. Applying phase noise first (as this chain originally did) let
# them escape it entirely, which matters most for far-range clutter: the module's own
# range-correlation argument says a distant return sees nearly the full phase noise, and
# it was seeing none. Adding the returns first and passing the composite through the
# noisy LO last also gives leakage its correct behaviour for free -- at near-zero delay
# the residual cancels, which is exactly why direct coupling stays coherent in hardware.
_STAGES = (
    ("leakage", LeakageParams, apply_leakage),
    ("clutter", ClutterParams, apply_clutter),
    ("phase_noise", PhaseNoiseParams, apply_phase_noise),
)


def stage_seed(seed: int, stage: str) -> int:
    """The sub-seed a given impairment stage runs with, derived by HASHING the
    (seed, stage) pair rather than adding a small per-stage offset.

    Small offsets are the obvious implementation and they are wrong here. With
    `offset = {phase_noise: 0, leakage: 1, clutter: 2}` and a caller that advances the
    frame seed by one per frame -- which `ImpairmentBlock` does -- frame `i`'s LEAKAGE
    sub-seed (`seed+i+1`) is exactly frame `i+1`'s PHASE-NOISE sub-seed. Identical
    seeds mean identical generator state, so a corpus quietly carries the same noise
    realization under two different labels, one frame apart. An adversarial review
    demonstrated the collision; this hash removes the arithmetic that caused it.

    SHA-256 rather than Python's `hash()`, which is salted per process and would make
    a corpus unreproducible across runs. Stable across reorderings of `_STAGES` too:
    a stage's seed depends on its NAME, so changing the physics order does not silently
    change every realization ever generated.
    """
    digest = hashlib.sha256(f"{int(seed)}:{stage}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2 ** 63)


def apply_all(adc: torch.Tensor, cfg, chain_params: Optional[Dict[str, Any]] = None, *,
              seed: int = 0) -> torch.Tensor:
    """Apply leakage, then clutter, then phase noise -- see `_STAGES` for why.

    Stage order is leakage -> clutter -> phase noise, and that order is physical rather
    than arbitrary: the first two ADD returns, and the last passes the whole composite
    through the oscillator's noisy phase, exactly as a real mixer does. See `_STAGES`.

    `chain_params` maps a subset of `{"phase_noise", "leakage", "clutter"}` to either
    a params dataclass instance, a plain dict of constructor kwargs (the JSON-
    round-trip case), or `None` to explicitly skip that stage. A stage absent from
    `chain_params` (or `chain_params=None` entirely) runs with its default params.
    Each stage gets a distinct, deterministic sub-seed keyed to its NAME (see
    `stage_seed`) so the three never share a realization and a reordering of the chain
    does not change any of them.
    """
    chain_params = dict(chain_params) if chain_params else {}
    out = adc
    for name, cls, fn in _STAGES:
        if name in chain_params and chain_params[name] is None:
            continue
        val = chain_params.get(name, cls())
        p = val if isinstance(val, cls) else cls(**val)
        out = fn(out, cfg, p, seed=stage_seed(seed, name))
    return out
