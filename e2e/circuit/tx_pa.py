"""
TX power-amplifier (PA) non-idealities: memoryless AM/AM + AM/PM nonlinearity, plus a
frequency-domain ripple response.

Model class: this is a MEMORYLESS, quasi-static bandpass baseband-equivalent model --
the output envelope/phase at time t depends only on the instantaneous input envelope at
t, not on its history. A memory-polynomial / Volterra model (capturing dynamic PA memory
effects such as long-term thermal drift or bias-network dynamics) is deferred; it would
replace `apply()` with a tapped nonlinear filter.

AM/AM follows the Rapp model [C. Rapp, "Effects of HPA-Nonlinearity on a 4-DPSK/OFDM
Signal for a Digital Sound Broadcasting System", ESA Workshop on DSP for Space
Communications, ESA SP-332, 1991], a smooth soft-limiter with a tunable knee sharpness
`p`. AM/PM is a smooth, monotonically-saturating phase term driven by the same envelope.
`frequency_response()` layers a separate sinusoidal gain ripple across frequency (e.g.
from imperfect output matching), to be applied multiplicatively in the CFR/S-parameter
domain -- i.e. it is a *linear*, frequency-selective effect, orthogonal to the
*nonlinear*, memoryless envelope effect in `apply()`.

For a constant-envelope waveform (e.g. an FMCW chirp), `apply()`'s AM/AM compression is
mild because the instantaneous envelope barely varies -- but the AM/PM term still applies
a roughly constant phase rotation, and `frequency_response()` still reshapes the
transfer function across the chirp bandwidth. For envelope-varying waveforms
(OFDM / ISAC), both AM/AM and AM/PM matter far more, since the instantaneous envelope
sweeps a wide dynamic range and drives the operating point across the whole curve.

KNOWN LIMITATIONS (reviewed 2026-08-10; read before using this for comms/ISAC work)
------------------------------------------------------------------------------------
1. **Memoryless is a comms/ISAC-specific weakness, not a general one.** At this repo's
   own bandwidths (750 MHz for the `radial_like` preset, 2 GHz for `ti_iwr1443`,
   `WaveformBlock`'s 1 GHz default) matching-network group delay and bias-network
   dynamics are first-order effects for a wideband modulated signal, so:
     * spectral regrowth / ACPR predicted here is ALWAYS SYMMETRIC about the carrier --
       a structural property of any memoryless polar model, since the output depends only
       on the instantaneous envelope. Real PAs show asymmetric regrowth. Do not use this
       model to support an ACPR-asymmetry or spectral-mask-compliance claim.
     * EVM versus backoff is optimistic near the compression knee.
     * **digital-predistortion studies against this model are near-tautological** -- a
       static memoryless predistorter inverts a static memoryless PA almost exactly, so
       any "DPD improved ACPR by X dB" result here flatters DPD relative to hardware.
   The constant-envelope radar path is genuinely unaffected. ISAC is the exposed case: it
   reuses high-PAPR OFDM (8-12 dB) for both functions, so it inherits the comms weakness
   without the radar excuse. Smallest fix: a memory polynomial (order ~5, depth 2-3 taps).
2. **No gain tilt or roll-off.** `frequency_response()` is normalized to unit mean gain
   over a ripple period, so a monotonic tilt across the band -- plausibly the dominant
   real frequency-domain effect over a 750 MHz-2 GHz sweep -- is absent BY CONSTRUCTION,
   not merely approximated.
3. **The mismatch is not coupled to an actual load.** `gamma_load` is a free parameter;
   nothing ties it to the antenna's real frequency- and scan-angle-dependent VSWR, nor to
   `InterconnectBlock`'s own S11. Changing the interconnect does not change this ripple.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


#: Speed of light, m/s -- used to turn a physical mismatch length into a ripple period.
_C_LIGHT = 299792458.0


@dataclass
class TxPAConfig:
    """Plain-float, JSON-serializable (via dataclasses.asdict) PA configuration."""
    small_signal_gain_db: float = 20.0   # linear (voltage) small-signal gain, dB
    a_sat: float = 1.0                   # output saturation level (linear amplitude units)
    am_pm_deg_at_sat: float = 5.0        # AM/PM phase shift approached deep in saturation, deg
    rapp_p: float = 2.0                  # Rapp smoothness/knee-sharpness parameter

    # --- Output-mismatch frequency response (see TxPA.frequency_response) ---------
    ripple_model: str = "mismatch"       # "mismatch" (physical) | "sinusoid" (legacy)
    ripple_phase_rad: float = 0.0        # ripple starting phase, rad (both models)

    # "mismatch" model: a line of one-way electrical length `mismatch_length_m` in a
    # medium of `eps_eff`, terminated by reflection coefficients gamma_source/gamma_load.
    # Ripple PERIOD = c / (2 * l * sqrt(eps_eff)); ripple DEPTH is set by |Gs*GL|.
    # Defaults describe an 8 mm on-package PA-to-antenna transition on a substrate with
    # eps_eff = 3.0 -> a ~10.8 GHz period, i.e. LESS THAN ONE CYCLE across a 1-2 GHz
    # sweep, which is what an on-package 77 GHz transition actually looks like: a gentle
    # tilt, not a multi-cycle wobble. |Gs*GL| = 0.058 matches the legacy 0.5 dB depth and
    # corresponds to ~15-17 dB return loss per port (VSWR ~1.3-1.4:1), a reasonable match.
    mismatch_length_m: float = 8e-3
    eps_eff: float = 3.0
    gamma_source: float = 0.24           # |Gs| ~ 0.24 -> ~12.4 dB return loss
    gamma_load: float = 0.24             # |GL| ~ 0.24; product 0.058 == legacy 0.5 dB

    # "sinusoid" (legacy) model only.
    ripple_db: float = 0.5               # peak frequency-ripple amplitude, dB (0-to-peak)
    ripple_period_hz: float = 100e6      # ripple period across frequency, Hz

    @property
    def mismatch_period_hz(self) -> float:
        """Ripple period of the `"mismatch"` model, Hz -- `c / (2*l*sqrt(eps_eff))`."""
        return _C_LIGHT / (2.0 * self.mismatch_length_m * math.sqrt(self.eps_eff))


class TxPA:
    """Memoryless AM/AM + AM/PM nonlinearity and frequency-ripple response for a TX PA."""

    def __init__(self, config: TxPAConfig | None = None):
        self.config = config if config is not None else TxPAConfig()

    def _am_am(self, a: torch.Tensor) -> torch.Tensor:
        """Rapp AM/AM curve g(a) = G*a / (1 + (G*a/A_sat)^(2p))^(1/(2p)).

        Evaluated in whichever of two algebraically-equivalent forms keeps the
        exponentiated ratio <= 1 (the other form is discarded via torch.where), so
        large dynamic ranges of `a` never overflow float32 even for large `p`.
        """
        cfg = self.config
        G = 10 ** (cfg.small_signal_gain_db / 20.0)
        p2 = 2.0 * cfg.rapp_p
        r = G * a / cfg.a_sat
        small = r <= 1.0

        r_small = torch.where(small, r, torch.ones_like(r))
        g_small = G * a / (1.0 + r_small ** p2) ** (1.0 / p2)

        r_large = torch.where(small, torch.ones_like(r), r)
        g_large = cfg.a_sat / (1.0 + (1.0 / r_large) ** p2) ** (1.0 / p2)

        return torch.where(small, g_small, g_large)

    def _am_pm(self, a: torch.Tensor) -> torch.Tensor:
        """Smooth AM/PM term: phase (rad) = deg_at_sat * u^2/(1+u^2), u = a/A_sat.

        Monotonically increasing in `a`, 0 at a=0, and approaches (but never exactly
        reaches, per the Rapp-style soft-knee shape) `am_pm_deg_at_sat` well beyond
        saturation.
        """
        cfg = self.config
        u = torch.clamp(a / cfg.a_sat, max=1e15)  # avoid u**2 overflow for pathological a
        u2 = u * u
        phi_deg = cfg.am_pm_deg_at_sat * u2 / (1.0 + u2)
        return phi_deg * (math.pi / 180.0)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Elementwise memoryless AM/AM + AM/PM on a complex tensor of any shape.

        y = g(|x|) * exp(j*(angle(x) + phi(|x|))), built via torch.polar so there is no
        explicit division by |x| (and hence no |x|->0 special-casing): at x=0,
        g(0)=0 and torch.polar(0, ...) == 0 regardless of the (finite) angle term.
        """
        assert torch.is_complex(x), "TxPA.apply expects a complex tensor"
        a = torch.abs(x)
        theta = torch.angle(x)
        g = self._am_am(a)
        phi = self._am_pm(a)
        return torch.polar(g, theta + phi)

    def frequency_response(self, freqs_hz: torch.Tensor) -> torch.Tensor:
        """Complex frequency response of the PA output mismatch, in the CFR domain.

        Two models, selected by `TxPAConfig.ripple_model`:

        `"mismatch"` (default) -- the physical multiple-reflection (Fabry-Perot) response
        of a PA output looking into a mismatched load through a line of one-way electrical
        length `l`:

            H(f) = 1 / (1 - Gs*GL * exp(-j * 4*pi*f*l*sqrt(eps_eff) / c))

        periodic in `f` with period `c / (2*l*sqrt(eps_eff))` and depth set by the product
        of the two reflection coefficients `|Gs*GL|`. This is complex by construction, so
        amplitude ripple comes with the group-delay ripple causality demands -- which is
        the reason to prefer it here: for an FMCW sweep a group-delay error IS a range
        bias, and the magnitude-only model below sets that bias to exactly zero by
        assumption.

        `"sinusoid"` -- the legacy magnitude-only form, retained for reproducibility:
        |H(f)| = exp(A*sin(2*pi*f/period + phase) - log(I0(A))), A = ripple_db*ln(10)/20,
        normalized by the modified Bessel function I0 so E[|H|] == 1 over a full period,
        and phase identically zero. This is the small-|Gs*GL| linearization of the
        mismatch form: expanding `-log(1 - g*exp(-j*theta))` as a geometric series gives
        a leading term `g*cos(theta)` -- a pure sinusoid in dB, with coefficient 1, not
        2. So the two agree closely while the ripple is shallow (at the shipped defaults
        their peak-to-peak depths differ by 0.0017 dB: 0.515 dB for the mismatch model
        against the legacy 0.5 dB nominal) and diverge once it is not, because the true
        response develops sharp resonant notches a sinusoid cannot represent.

        VALIDITY BOUND: the sinusoid model is only physical for |Gs*GL| << 1, i.e. roughly
        `ripple_db` below ~1-2 dB. Beyond that use `"mismatch"`.

        Returned as complex64 for direct multiplication against a complex S-parameter/CFR
        tensor. Both models are normalized to unit mean gain over a ripple period, so this
        function contributes ripple ONLY -- any average gain belongs to
        `small_signal_gain_db`, and note that neither model produces a gain TILT across
        the band (see the class docstring's limitations).
        """
        cfg = self.config
        if not torch.is_tensor(freqs_hz):
            freqs_hz = torch.as_tensor(freqs_hz, dtype=torch.float32)
        freqs_hz = freqs_hz.to(torch.float32)

        if cfg.ripple_model == "sinusoid":
            theta = 2.0 * math.pi * freqs_hz / cfg.ripple_period_hz + cfg.ripple_phase_rad
            A = cfg.ripple_db * math.log(10.0) / 20.0
            log_i0 = torch.special.i0(torch.tensor(A, dtype=torch.float32)).log().item()
            log_mag = A * torch.sin(theta) - log_i0
            mag = torch.exp(log_mag)
            return torch.polar(mag, torch.zeros_like(mag))

        if cfg.ripple_model != "mismatch":
            raise ValueError(
                f"ripple_model must be 'mismatch' or 'sinusoid', got {cfg.ripple_model!r}")

        # Round-trip phase 2*beta*l, with beta = 2*pi*f*sqrt(eps_eff)/c.
        theta = (4.0 * math.pi * freqs_hz * cfg.mismatch_length_m
                 * math.sqrt(cfg.eps_eff) / _C_LIGHT + cfg.ripple_phase_rad)
        gamma_prod = cfg.gamma_source * cfg.gamma_load
        h = 1.0 / (1.0 - gamma_prod * torch.polar(torch.ones_like(theta), -theta))

        # No normalization is applied, and none is needed: expanding the geometric series
        # gives H = sum_n (g*exp(-j*theta))^n, and every n >= 1 term integrates to zero
        # over a full period, so the COMPLEX mean E[H] over a ripple period is exactly 1
        # for |g| < 1. (Note this is E[H] == 1, not E[|H|] == 1 -- the legacy sinusoid
        # normalizes the magnitude instead. The two conventions differ by O(|g|^2), i.e.
        # ~0.3% at the shipped defaults.) Any average gain belongs to small_signal_gain_db.
        return h.to(torch.complex64)
