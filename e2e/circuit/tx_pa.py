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
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class TxPAConfig:
    """Plain-float, JSON-serializable (via dataclasses.asdict) PA configuration."""
    small_signal_gain_db: float = 20.0   # linear (voltage) small-signal gain, dB
    a_sat: float = 1.0                   # output saturation level (linear amplitude units)
    am_pm_deg_at_sat: float = 5.0        # AM/PM phase shift approached deep in saturation, deg
    rapp_p: float = 2.0                  # Rapp smoothness/knee-sharpness parameter
    ripple_db: float = 0.5               # peak frequency-ripple amplitude, dB (0-to-peak)
    ripple_period_hz: float = 100e6      # ripple period across frequency, Hz
    ripple_phase_rad: float = 0.0        # ripple starting phase, rad


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
        """Complex, unit-mean-gain sinusoidal ripple across frequency, in the CFR domain.

        |H(f)| = exp(A*sin(2*pi*f/period + phase) - log(I0(A))), A = ripple_db*ln(10)/20,
        which is the Rapp-independent normalization that gives E[|H|] == 1 exactly when
        averaged over a full ripple period (I0 = modified Bessel function of the first
        kind, order 0 -- the exact mean of exp(A*sin(theta)) over a period). Phase is
        left at 0 (magnitude-only ripple); returned as complex64 for direct
        multiplication against a complex S-parameter/CFR tensor.
        """
        cfg = self.config
        if not torch.is_tensor(freqs_hz):
            freqs_hz = torch.as_tensor(freqs_hz, dtype=torch.float32)
        freqs_hz = freqs_hz.to(torch.float32)

        theta = 2.0 * math.pi * freqs_hz / cfg.ripple_period_hz + cfg.ripple_phase_rad
        A = cfg.ripple_db * math.log(10.0) / 20.0
        log_i0 = torch.special.i0(torch.tensor(A, dtype=torch.float32)).log().item()
        log_mag = A * torch.sin(theta) - log_i0
        mag = torch.exp(log_mag)
        return torch.polar(mag, torch.zeros_like(mag))
