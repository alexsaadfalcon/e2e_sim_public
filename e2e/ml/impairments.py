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


def _noise_floor_power(adc: torch.Tensor) -> float:
    """Robust, TARGET-INSENSITIVE noise-floor reference: the MEDIAN range-FFT power.

    This exists to replace `_range_fft_peak_power` as the calibration reference for the
    relative-power impairments, and the reason is structural rather than cosmetic.

    A few targets occupy a handful of cells out of tens of thousands, so the median of the
    range-FFT power IS the noise floor, and -- unlike the peak -- it does not move when a
    target gets stronger. Calibrating injected clutter/leakage against the PEAK made the
    target-to-impairment ratio exactly invariant to target strength (MEASURED: 0.00 dB of
    change across a 20 dB change in target power), which put a hard ceiling on the corpus:
    no improvement to the RF front end, antenna pattern, scene geometry or target
    scattering could ever improve detectability. See notes/ESTABLISHED_FACTS.md F35.

    Median rather than mean: the mean is pulled by the same strong returns the peak is.
    Median rather than a low percentile: a low percentile of a range-FFT tracks the
    window's stopband rather than the noise.
    """
    mag2 = torch.abs(torch.fft.fft(adc, dim=-1)) ** 2
    return float(torch.median(mag2).item())


#: Which reference the relative-power impairments are calibrated against.
#:   "noise" -- the cube's noise floor (`_noise_floor_power`). Physically meaningful:
#:             `leakage_relative_db` etc. then read as dB ABOVE THE NOISE FLOOR, the way a
#:             link budget states them, and target improvements show up in target-to-
#:             impairment ratio as they should.
#:   "peak"  -- the cube's own peak (`_range_fft_peak_power`). The pre-2026-08-17 behaviour,
#:             kept ONLY so an existing corpus can be reproduced bit-for-bit. It ties the
#:             injected background to the signal it is supposed to compete with, which is
#:             the F35 ceiling. Do not use it for new corpora.
REFERENCE_NOISE = "noise"
REFERENCE_PEAK = "peak"
#: Absolute k*T*B*F from `e2e.ml.link_budget`, independent of the cube's contents.
#:
#: This is the one that actually breaks F35, and the distinction from REFERENCE_NOISE is
#: worth stating. "noise" MEASURES a floor off the cube (the median range-FFT power); on a
#: ray-traced frame that median is scene structure -- sidelobes, diffuse multipath -- which
#: still scales with the scene, so it moves the ceiling rather than removing it (F42).
#: "thermal" asks the link budget instead and never looks at the cube at all.
REFERENCE_THERMAL = "thermal"
#: FLIPPED 2026-08-18, once the link budget existed to make it meaningful.
#:
#: Under REFERENCE_PEAK the injected impairments were calibrated against the cube's own
#: peak -- which the target sets -- so target-to-clutter was exactly invariant to target
#: strength and no physics improvement could ever move detection (F35). MEASURED on a
#: fresh corpus with the peak reference still active: in 14 of 15 frames a clutter cell
#: out-ranked the real target's own CFAR ratio, by a median of 15 dB, every one of them
#: identifiable by peaking at exactly the zero-Doppler bin `apply_clutter` synthesizes at.
#:
#: The dB values on the params classes below are re-derived accordingly -- they are now
#: levels ABOVE THE THERMAL FLOOR, from the link budget, not fractions of the target.
DEFAULT_POWER_REFERENCE = REFERENCE_THERMAL


def _thermal_reference(cfg, *, domain: str) -> float:
    """Absolute thermal power in the domain an impairment is specified in.

    The two impairments quote their dB against DIFFERENT domains, which is pre-existing
    and easy to get wrong by 10*log10(n_samples) = 27 dB:

      * `apply_leakage` builds a tone whose RANGE-FFT peak power equals `p_ref * 10^(dB/10)`,
        so its reference is a range-FFT bin power. White noise of per-sample power N has
        range-FFT bin power N * n_samples.
      * `apply_clutter` sums per-scatterer mean powers to `p_ref * 10^(dB/10)` in the TIME
        domain, so its reference is the per-sample power N itself.
    """
    from e2e.ml.link_budget import thermal_noise_power_w

    n = thermal_noise_power_w(cfg)
    if domain == "range_fft":
        return n * float(cfg.n_samples)
    if domain == "time":
        return n
    raise ValueError(f"unknown reference domain {domain!r}")


def _reference_power(adc: torch.Tensor, reference: str, cfg=None,
                     *, domain: str = "range_fft") -> float:
    """Dispatch for the two calibration references. Fails loudly on a typo rather than
    silently falling back -- picking the wrong one silently changes the physics."""
    if reference == REFERENCE_THERMAL:
        if cfg is None:
            raise ValueError("the 'thermal' reference needs cfg (it reads the link budget)")
        return _thermal_reference(cfg, domain=domain)
    if reference == REFERENCE_NOISE:
        return _noise_floor_power(adc)
    if reference == REFERENCE_PEAK:
        return _range_fft_peak_power(adc)
    raise ValueError(
        f"unknown power reference {reference!r}; expected {REFERENCE_NOISE!r} "
        f"(dB above the noise floor) or {REFERENCE_PEAK!r} (legacy, dB below the cube "
        f"peak -- see ESTABLISHED_FACTS F35 for why that one is a ceiling)")


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


def _k_distributed_gain(n_scat: int, nu: float, *, generator: torch.Generator,
                         device: torch.device) -> torch.Tensor:
    """K-distributed complex gain `[n_scat]`: sqrt(gamma texture) * CN(0,1) speckle.

    Texture ~ Gamma(shape=nu, scale=1/nu) (mean 1, variance 1/nu -- small `nu` means a
    heavier-tailed, more variable RCS, the classic K-distribution clutter model).
    ONE speckle draw per scatterer -- the scatterer is a coherent patch return whose
    structure across the ARRAY is its steering vector, applied by the caller. (Until
    2026-08-23 speckle was drawn independently per (scatterer, rx) on a
    "decorrelates spatially" argument; that removed all angular structure from the
    clutter and is what let the MIMO demux misassign it -- F52. Aperture
    decorrelation is a property of a patch WIDER than the array resolution, which a
    point-scatterer clutter model does not represent; if that fidelity is ever
    wanted it should come from more, narrower scatterers, not from spatially white
    gains.) `E[|gain|^2] = 1`.
    """
    tex = _sample_gamma(n_scat, nu, generator=generator, device=device) / float(nu)
    real = torch.randn((n_scat,), generator=generator, device=device, dtype=torch.float64)
    imag = torch.randn((n_scat,), generator=generator, device=device, dtype=torch.float64)
    speckle = (real + 1j * imag) * math.sqrt(0.5)  # E[|speckle|^2] = 1
    return torch.sqrt(tex).to(speckle.dtype) * speckle


def _mimo_tx_factor(cfg, sin_az: torch.Tensor, n_chirps: int) -> torch.Tensor:
    """Per-scatterer per-chirp TX factor `[n_scat, n_chirps]` for a return arriving
    from direction `sin_az` on the POST-MIMO-combine ADC cube.

    Mirrors `e2e.ml.rd_synth.synthesize_adc`'s tx_factor EXACTLY (same conventions the
    demux inverts -- see `e2e.ml.transforms.ddma_demux` / `tdm_deinterleave`):

    * `"ddma"`: every TX fires on every chirp, TX `t` at `t * n_rx * lambda/2`
      carrying the code `2*pi*t*c/n_tx`:
      `sum_t exp(j*(pi*n_rx*t*sin_az + 2*pi*t*c/n_tx))`.
    * `"tdm"` / `"single"`: chirp `c` is fired by TX `c % n_tx` alone, so only that
      TX's spatial phase appears: `exp(j*pi*n_rx*(c % n_tx)*sin_az)`.

    Every ADDITIVE return the chain injects after the dechirp/combine must carry this
    structure -- the same TX waveforms that illuminate targets illuminate leakage
    paths and clutter patches. Injecting without it is F52: the demux, whose whole
    job is inverting this code, misassigns codeless energy across the virtual array
    (period-`n_rx` replication under TDM, TX-0-sub-band confinement under DDMA),
    which rendered as the coherent azimuth stripes/comb that invalidated the
    detection figure.

    `E[|factor|^2]` per chirp is `n_tx` for DDMA (a coherent sum of `n_tx` unit
    phasors) and exactly 1 for TDM/single -- callers calibrating injected POWER must
    divide by `cfg.n_tx` for DDMA (see `apply_clutter`/`apply_leakage`).
    """
    device = sin_az.device
    mimo = str(cfg.mimo).lower()
    n_tx = int(cfg.n_tx)
    n_rx = int(cfg.n_rx)
    c_idx = torch.arange(n_chirps, dtype=torch.float64, device=device)
    if mimo == "ddma":
        tx_idx = torch.arange(n_tx, dtype=torch.float64, device=device)
        ph = (math.pi * n_rx * tx_idx[None, :, None] * sin_az[:, None, None]
              + 2.0 * math.pi * tx_idx[None, :, None] * c_idx[None, None, :] / n_tx)
        return torch.polar(torch.ones_like(ph), ph).sum(dim=1)          # [n_scat, C]
    tx_of_chirp = c_idx % max(n_tx, 1)
    ph = math.pi * n_rx * tx_of_chirp[None, :] * sin_az[:, None]        # [n_scat, C]
    return torch.polar(torch.ones_like(ph), ph)


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
    `n_samples` range gates into ~`params.n_range_bands` LOG-spaced contiguous bands
    (edges `round(n_samples^(b/n_bands))`, with gate 0 always a band of its own so the
    `tau = 0` direct-leakage gate keeps its EXACT residual cancellation -- uniform
    bands used to hand it the band-mean delay of a ~6 m return, which painted
    full-height azimuth bands across the leakage+phase attribution figures; found by
    the 2026-08-24 adversarial review) and give each band ONE fast-time residual keyed
    to its band-mean delay `tau_b`. Log spacing bounds the within-band `tau` ratio at
    `~n_samples^(1/n_bands)`, i.e. a few dB of residual-variance granularity, because
    the residual variance goes as `tau^2` while `tau` spans three decades. For band
    `b` covering gates `[k_lo, k_hi)`:
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
    Gate 0's dedicated band exists at EVERY setting (the edge set always seeds
    `{0, 1, n_samples}`), so `n_range_bands=1` gives the cheapest mode that is still
    physically honest at tau = 0: the untouched DC gate plus ONE reference delay (the
    remaining window's mean tau) for everything else -- two bands, not one.

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
      The LOG spacing moves that granularity where it matters: exact at tau = 0,
      but FAR ranges got coarser than under the old uniform scheme (batch physics
      review 2026-08-24: the top log band spans gates [~0.46N, N), ~+-3 dB of
      residual-power granularity where uniform banding gave ~+-0.3 dB there) --
      the right trade against the tau=0 catastrophe it fixed (F54), stated so
      nobody reads "log banding" as strictly finer everywhere.
    * Per-band residual DRAWS are statistically INDEPENDENT (pre-existing, stated
      2026-08-24): physically every range sees the SAME oscillator realization,
      weighted per-tau by the correlation factor, so band residuals should be
      deterministically related, not independent draws. Invisible to single-tone
      oracles; matters in principle whenever returns at two ranges coexist.
      Post-v1.1 follow-up: draw ONE base realization per chirp and shape it
      per band.
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

    # LOG-spaced band edges, gate 0 always isolated (2026-08-24, batch-review finding):
    # the residual variance goes as tau^2 and tau spans [0, n_samples/bandwidth] --
    # three decades -- so UNIFORM bands made band 0's representative delay a pure
    # artifact: gates [0, 64) shared tau_b = 42 ns, handing the tau = 0 direct-leakage
    # tone (whose residual physically cancels EXACTLY -- the range-correlation effect
    # this whole model exists for) the phase noise of a ~6 m return, which painted
    # full-height azimuth bands across the leakage+phase figures. Log spacing bounds
    # the within-band tau ratio at ~n_samples^(1/n_bands) (a few dB of variance
    # granularity) at unchanged cost, and the dedicated [0, 1) band gives the DC gate
    # its exact zero residual.
    edges = {0, 1, n_samples}
    edges.update(int(round(n_samples ** (b / n_bands))) for b in range(1, n_bands))
    band_edges = sorted(edges)
    for b in range(len(band_edges) - 1):
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
        # rfft DC bin of a real signal must be real (and the Nyquist bin, if it
        # exists). Zeroing the imag VIEW in place, not `x = x.real` self-assignment:
        # the latter reads a real view of the same complex storage it writes, which
        # torch's overlap checker rejects on some backends/layouts ("some elements of
        # the input tensor and the written-to tensor refer to a single memory
        # location" -- hit in a webapp end-to-end review probe, 2026-08-24). Same
        # values bit-for-bit, no overlapping read-write.
        yfreq[:, 0].imag.zero_()
        if n_samples % 2 == 0:
            yfreq[:, -1].imag.zero_()
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

    # DERIVED FROM THE LINK BUDGET, not chosen. At P_tx = 12 dBm and a thermal floor of
    # -85 dBm (see `e2e.ml.link_budget`):
    #   leakage: 35 dB TX-RX isolation is typical for an integrated MMIC -> -23 dBm at the
    #            receiver -> +62 dB above the floor. That is enormous, and correctly so:
    #            direct coupling dwarfs every target. It is survivable only because it sits
    #            at ~zero range and is range-gated away, which is exactly what the real
    #            hardware relies on.
    #   bumper:  the two-way radar equation at 0.2 m with a -10 dBsm radome/bumper return
    #            gives +34 dB above the floor.
    # For scale: a 10 dBsm car at 30 m is -33 dB per sample, reaching +12 dB only after
    # 45 dB of coherent integration. The impairments genuinely are ~100 dB stronger than
    # the target per sample; separating them is the receiver's job, not the model's.
    leakage_relative_db: float = 62.0    # dB ABOVE the thermal floor (range-FFT domain)
    bumper_range_m: float = 0.2          # bumper/radome reflection range, m
    bumper_relative_db: float = 34.0     # dB above the thermal floor
    # "peak" (legacy, the F35 ceiling) or "noise" (dB above the noise floor). The dB
    # values above are calibrated for "peak" and are NOT meaningful under "noise" -- a
    # leakage tone 5 dB BELOW the noise floor is invisible. Switching the reference
    # requires re-deriving these numbers from a link budget.
    reference: str = DEFAULT_POWER_REFERENCE


def apply_leakage(adc: torch.Tensor, cfg, params: LeakageParams, *, seed: int) -> torch.Tensor:
    """Add a near-zero-delay TX-RX leakage tone plus a short-range bumper reflection.

    Both are modeled as static tones, each placed at the nearest fast-time DFT bin for
    its range (`_range_to_bin`, rounded -- both are strong, coherent,
    essentially-fixed-delay returns; snapping to the bin removes spectral leakage that
    would otherwise depend on `n_samples`/`fs` in a way not implied by the physical
    description).

    STRUCTURE (since 2026-08-23 / F52): the coupling is per-(TX, RX) PAIR -- each of
    the `n_tx * n_rx` paths gets its own random phase, same magnitude (at this
    fidelity a coupling path has no per-pair gain variation worth modelling), and TX
    `t`'s contribution carries TX `t`'s own MIMO signature: the DDMA code
    `2*pi*t*c/n_tx` on every chirp, or (TDM/single) presence only on the chirps TX `t`
    actually fires. That per-chirp structure is what the downstream demux inverts; the
    pre-2026-08-23 model (one random phase per RX, constant over chirps, no TX
    identity) parked all leakage energy in TX-0's Doppler sub-band, where phase noise
    smeared it into the full-height sin(az) stripes of F52. Note the code phase here
    is the TX's DDMA/TDM signature WITHOUT a spatial `sin(az)` steering term --
    coupling is an internal path, not a far-field arrival, so its per-pair phase is
    simply random.

    Power is calibrated against `params.reference`, measured on the INPUT before this
    function adds anything, and states the summed-over-TX per-RX tone power (DDMA's
    incoherent per-pair sum is divided back out, so the dB number means the same
    thing on every MIMO scheme). Under the legacy `"peak"` reference,
    `leakage_relative_db=-5` places the tone 5 dB below the strongest existing return --
    which ties the injected impairment to the target and is the F35 ceiling. Under
    `"noise"` the same field reads as dB above the noise floor, the way a link budget
    states it, and the numbers must be re-derived accordingly.
    """
    n_rx, n_chirps, n_samples = adc.shape
    device, dtype = adc.device, adc.dtype
    n_tx = max(int(cfg.n_tx), 1)
    mimo = str(cfg.mimo).lower()

    p_ref = _reference_power(adc, params.reference, cfg, domain='range_fft')

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    phases = torch.rand((2, n_tx, n_rx), generator=gen, device=device,
                        dtype=torch.float32) * (2.0 * math.pi)

    n = torch.arange(n_samples, device=device, dtype=torch.float64)
    c_idx = torch.arange(n_chirps, device=device, dtype=torch.float64)

    def _tap(range_m: float, rel_db: float, phase_pair: torch.Tensor) -> torch.Tensor:
        # DDMA sums n_tx random-phase unit taps per chirp (E[power] = n_tx); TDM has
        # exactly one active tap per chirp. Normalize so `rel_db` is the per-RX tone
        # power either way.
        amp = math.sqrt(p_ref * (10.0 ** (float(rel_db) / 10.0))) / n_samples
        if mimo == "ddma":
            amp = amp / math.sqrt(n_tx)
        k = int(round(_range_to_bin(range_m, cfg))) % n_samples
        tone = amp * torch.exp(1j * (2.0 * math.pi * k * n / n_samples))  # [n_samples]
        pair = torch.polar(torch.ones_like(phase_pair),
                           phase_pair).to(torch.complex128)               # [n_tx, n_rx]
        if mimo == "ddma":
            code = torch.exp(2j * math.pi * torch.arange(n_tx, device=device,
                                                         dtype=torch.float64)[:, None]
                             * c_idx[None, :] / n_tx)                     # [n_tx, C]
            rc = torch.einsum("tr,tc->rc", pair, code.to(torch.complex128))
        else:
            tx_of_chirp = (c_idx.long() % n_tx)                           # [C]
            rc = pair[tx_of_chirp, :].T.contiguous()                      # [n_rx, C]
        return (rc[:, :, None] * tone[None, None, :]).to(dtype)

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
    # Clutter-to-noise ratio, dB above the thermal floor (time domain, TOTAL across the
    # whole field). RE-ANCHORED 2026-08-24 (release-plan A15; justification CORRECTED
    # same day by the batch physics review -- the first version claimed the steered
    # model adds "~10*log10(n_virtual) dB per cell at the same total" and that +10
    # "restores the pre-A11 regime": BOTH refuted. Measured map-wide, at a FIXED knob
    # the steered model leaves the upper-percentile cell statistic roughly level with
    # the old azimuth-white model (p99 within ~1 dB) and only lengthens the extreme
    # tail; and "restore the old regime" is not a definable target, because the old
    # model's energy sat at F52-artifact coordinates. See the ledger's retraction
    # table.) What actually anchors this number is the PHYSICAL statement alone:
    # road-clutter discretes present roughly 10-30 dB of per-resolved-cell CNR with a
    # heavy tail that occasionally rivals vehicles. +10 dB total sits at the HOT edge
    # of that band -- measured (notes/tools/a15_clutter_anchor_probe.py): median
    # drawn-cell CNR ~34 dB, p90 ~42 -- chosen so clutter still MATTERS and is
    # rejected in DOPPLER rather than by being weak (which only works on a config
    # whose targets do not alias, see benchmark_v1 / F43). This is a difficulty
    # ASSUMPTION, not a derivation -- there is no datasheet number for road clutter;
    # the corpus randomizer spans the plausible band around it, and a measured
    # re-derivation against real road-clutter statistics remains open.
    total_relative_db: float = 10.0
    # "peak" (legacy) or "noise". Same caveat as LeakageParams: the dB value above is
    # calibrated for "peak" and must be re-derived as a clutter-to-noise ratio to be
    # meaningful under "noise".
    reference: str = DEFAULT_POWER_REFERENCE


def apply_clutter(adc: torch.Tensor, cfg, params: ClutterParams, *, seed: int,
                  frame_idx: int = 0) -> torch.Tensor:
    """Add heavy-tailed diffuse ground clutter.

    `density * n_samples` scatterers are scattered uniformly over the unambiguous
    range window `[0, max_range_m)` AND uniformly in direction cosine
    `sin(az) ~ U[-1, 1)` (since 2026-08-23 / F52 -- isotropic diffuse return, a stated
    approximation; a directive element pattern would taper it). Each has a small
    random radial velocity (`N(0, doppler_std_mps^2)`, i.e. clutter sits near Doppler
    bin 0) and a per-scatterer K-distributed complex gain (`_k_distributed_gain`:
    gamma "texture" times complex Gaussian "speckle" -- small `nu` gives a heavy-tailed
    amplitude distribution, the standard sea/ground-clutter model). Across the ARRAY a
    scatterer contributes its steering vector at its azimuth plus the per-chirp MIMO
    TX code (`_mimo_tx_factor`) -- clutter is illuminated by the same coded waveforms
    as targets, and the demux downstream inverts exactly that structure. The
    pre-2026-08-23 model injected i.i.d. per-RX gains with no steering and no code;
    the demux misassigned that energy across the virtual array (F52: a perfectly
    coherent period-`n_rx` azimuth comb under TDM, a 1/n_tx aperture confinement
    under DDMA). Total injected power is calibrated to `total_relative_db` dB
    relative to the reference (per-RX; DDMA's coherent code gain of `n_tx` is divided
    back out so the number means the same thing on every MIMO scheme).

    TEMPORAL BEHAVIOUR: `seed` draws the clutter FIELD -- scatterer positions,
    velocities, azimuths, K-distribution texture and speckle gains -- and, on its own, is
    independent of `frame_idx`: the road and barriers a real scene's clutter comes
    from stay put, so the caller is expected to hand the SAME `seed` across every
    frame of a scenario (see `ImpairmentBlock`/`apply_all`, which use a base seed for
    this stage rather than the per-frame seed the other two stages get). `frame_idx`
    is what actually evolves the return frame to frame: each scatterer's own drawn
    radial velocity gives it a Doppler `f_dop = 2*v/lambda`, and by frame `frame_idx`
    (at `cfg.frame_rate_hz` frames/s) it has picked up a deterministic extra phase
    `2*pi*f_dop*(frame_idx/frame_rate_hz)` -- the physical model of internal motion
    within an otherwise-static scene. `frame_idx=0` adds zero extra phase: frame 0 IS
    the single-draw field. (Until the 2026-08-23 array-structure fix this was also
    bit-identical to the historical pre-persistence output; the F52 fix deliberately
    changed the field's realization, so that historical pin no longer holds -- the
    frame_idx mechanics are unchanged.)

    TWO STATED APPROXIMATIONS of the persistence model. (a) A scatterer's drawn
    velocity advances its PHASE but never its RANGE -- `f_beat` stays frozen at the
    drawn positions for the life of the field. At the default `doppler_std_mps`
    (0.05 m/s) the range walk is negligible over any realistic frame count, but push
    `frames_per_scene` very high and the inconsistency grows. (b) Persistence holds
    only if the CALLER keeps the field-defining parameters fixed across frames:
    varying `density` or `nu` per frame (no shipped randomizer does) changes the draw
    count / texture shape and silently produces a NEW field each frame while still
    looking deterministic. Vary only power-level parameters per frame.

    HISTORICAL NOTE, because the original choice was deliberate and is worth preserving:
    this docstring used to end "despite the 'clutter-to-noise ratio' phrasing in the
    design brief, no explicit noise floor is assumed to be present in `adc`." That was an
    honest compromise under a premise that has since changed -- the chain now carries a
    physical thermal floor from `e2e.circuit.rffe_model`'s 4kTR model, so a
    clutter-to-NOISE ratio is well defined. Calibrating against the peak instead ties the
    injected clutter to the target that usually sets that peak, making
    target-to-clutter exactly invariant to target strength (F35). `reference="noise"`
    uses the time-domain MEDIAN power, which the targets do not move.
    """
    n_rx, n_chirps, n_samples = adc.shape
    device, dtype = adc.device, adc.dtype

    n_scat = max(1, int(round(float(params.density) * n_samples)))
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    # DRAW ORDER IS A PUBLIC CONTRACT: positions -> velocities -> azimuths -> texture ->
    # speckle, all from this one generator. Stored corpora reproduce their clutter field
    # from (seed, these draws); tests replay the order to recover what was drawn
    # (`tests/test_ml_impairments.py::_replay_clutter_draws`). Inserting or reordering a
    # draw silently changes every seeded field ever generated -- if you must add one,
    # append it AFTER the existing draws and update the replay helper.
    ranges = torch.rand(n_scat, generator=gen, device=device, dtype=torch.float64) * float(cfg.max_range_m)
    f_beat = float(cfg.ramp_slope_hzps) * 2.0 * ranges / C_MPS  # [n_scat]

    vel = torch.randn(n_scat, generator=gen, device=device, dtype=torch.float64) * float(params.doppler_std_mps)
    f_dop = 2.0 * vel / float(cfg.wavelength_m)  # [n_scat], near zero

    # Each patch sits at a real direction: sin(az) ~ U[-1, 1) (a stated approximation --
    # isotropic diffuse ground return over the unambiguous span; a directive antenna
    # pattern would taper this, and is deliberately not modelled here). The steering
    # vector + TX code this implies across the virtual array is the WHOLE fix of F52.
    sin_az = torch.rand(n_scat, generator=gen, device=device, dtype=torch.float64) * 2.0 - 1.0

    gain = _k_distributed_gain(n_scat, float(params.nu), generator=gen, device=device)  # [n_scat]

    if params.reference == REFERENCE_NOISE:
        # NOT the time-domain median. A target is a TONE in the beat signal, so it is
        # present in every fast-time sample and the time-domain median tracks it -- MEASURED,
        # that estimator left target-to-clutter just as invariant as the peak did. The
        # target is sparse only in the RANGE-FFT domain, so estimate the noise there and
        # convert back:
        #   per-bin FFT noise power for white noise of time-power sigma^2 is N*sigma^2, and
        #   |FFT|^2 is exponential, whose MEDIAN is ln(2) times its mean.
        # Hence sigma^2 = median(|FFT|^2) / (N * ln 2).
        peak_power = _noise_floor_power(adc) / (float(n_samples) * math.log(2.0))
    elif params.reference == REFERENCE_THERMAL:
        peak_power = _thermal_reference(cfg, domain="time")
    elif params.reference == REFERENCE_PEAK:
        peak_power = float(torch.max(torch.abs(adc) ** 2).item())
    else:
        raise ValueError(
            f"unknown power reference {params.reference!r}; expected "
            f"{REFERENCE_NOISE!r} or {REFERENCE_PEAK!r} (see ESTABLISHED_FACTS F35)")
    target_total = peak_power * (10.0 ** (float(params.total_relative_db) / 10.0))
    # DDMA's coherent TX code carries E[|tx_factor|^2] = n_tx per chirp (see
    # `_mimo_tx_factor`); divide it back out so `total_relative_db` keeps meaning the
    # same injected per-RX power on every MIMO scheme.
    tx_power = float(cfg.n_tx) if str(cfg.mimo).lower() == "ddma" else 1.0
    mean_power = target_total / (n_scat * tx_power)
    gain = gain * math.sqrt(mean_power)

    if frame_idx:
        # Deterministic phase advance from the scatterer's OWN drawn velocity -- the
        # only thing that is allowed to change the field frame to frame (see docstring
        # "TEMPORAL BEHAVIOUR"). `frame_idx=0` skips this entirely so frame 0 is exactly
        # the single-draw field, bit-for-bit.
        t_frame = float(frame_idx) / float(cfg.frame_rate_hz)
        frame_phase = 2.0 * math.pi * f_dop * t_frame  # [n_scat]
        frame_phasor = torch.exp(1j * frame_phase).to(gain.dtype)
        gain = gain * frame_phasor

    n = torch.arange(n_samples, device=device, dtype=torch.float64)
    c = torch.arange(n_chirps, device=device, dtype=torch.float64)
    fast_phase = 2.0 * math.pi * torch.outer(f_beat, n) / float(cfg.fs_hz)          # [n_scat, n_samples]
    slow_phase = 2.0 * math.pi * torch.outer(f_dop, c) * float(cfg.chirp_period_s)  # [n_scat, n_chirps]
    fast = torch.exp(1j * fast_phase).to(gain.dtype)
    slow = torch.exp(1j * slow_phase).to(gain.dtype)

    # Array structure (the F52 fix): per-RX steering at the patch's azimuth, and the
    # per-chirp MIMO TX factor the demux exists to invert -- RX r at r*lambda/2, so the
    # phase is pi * r * sin_az (`rd_synth.synthesize_adc`'s exact convention).
    rx_idx = torch.arange(n_rx, device=device, dtype=torch.float64)
    e_rx = torch.polar(torch.ones((n_scat, n_rx), dtype=torch.float64, device=device),
                       math.pi * rx_idx[None, :] * sin_az[:, None])
    slow_tx = slow * _mimo_tx_factor(cfg, sin_az, n_chirps)

    clutter = torch.einsum("s,sr,sc,sn->rcn", gain, e_rx, slow_tx, fast)
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
              seed: int = 0, frame_idx: int = 0) -> torch.Tensor:
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

    `seed` is the BASE (frame-independent) seed; `frame_idx` is threaded through
    separately (0-based, default 0). Phase noise and leakage are genuinely new draws
    every frame (a noisy oscillator/coupling has no reason to repeat), so those two
    still key off the per-frame seed `seed + frame_idx`, exactly as before `frame_idx`
    existed as an argument. Clutter is the one exception: it is a persistent SCENE, not
    a fresh draw, so it keys off `seed` alone -- the same field every frame -- and
    `frame_idx` reaches `apply_clutter` directly to evolve that field deterministically
    (see its docstring). At `frame_idx=0` both paths agree (`seed + 0 == seed`), so
    this is a bit-for-bit no-op for single-frame use.
    """
    chain_params = dict(chain_params) if chain_params else {}
    out = adc
    frame_seed = seed + frame_idx  # per-frame seed; phase_noise/leakage only
    for name, cls, fn in _STAGES:
        if name in chain_params and chain_params[name] is None:
            continue
        val = chain_params.get(name, cls())
        p = val if isinstance(val, cls) else cls(**val)
        if name == "clutter":
            out = fn(out, cfg, p, seed=stage_seed(seed, name), frame_idx=frame_idx)
        else:
            out = fn(out, cfg, p, seed=stage_seed(frame_seed, name))
    return out
