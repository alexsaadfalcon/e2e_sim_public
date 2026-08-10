"""
TX-side signal chain: waveform source -> PA nonlinearity -> modulate-onto-channel bridge.

The chain is symmetric (see `e2e/frames.py`'s "signal domains" section): a transmitted
waveform lives in TX time (`DOMAIN_TX_TIME`, state key `tx_wave`,
`[n_tx, n_chirp, n_t]`), propagation and every analog transfer function live in the
frequency domain (`DOMAIN_CFR`, state key `s_pars`, `[n_rx, n_tx, n_chirp, n_freqs]`),
and (on the RX side, built elsewhere in this package) a dechirped receive waveform
comes back in time. This module builds the three TX-side blocks:

* `WaveformBlock` -- a SOURCE: synthesizes the transmitted complex envelope into
  `tx_wave`. Wraps the existing `e2e.signal_generator.signals` classes
  (`NarrowbandSignal` / `RandomWidebandSignal` / `FMCWSignal`) rather than
  reimplementing their math; see "Wrapping signal_generator" below for the two rough
  edges found doing that.
* `TxPABlock` -- applies `TxPA.apply()`'s memoryless AM/AM + AM/PM envelope
  nonlinearity to `tx_wave`, elementwise, in place in the time domain. This is the
  entire reason a TX time domain exists in the chain: the PA acts on the
  instantaneous envelope, which only exists before the modulate step folds the
  waveform into the linear, frequency-domain channel model.
* `ModulateBlock` -- the BRIDGE from TX time to frequency: multiplies the channel's
  `s_pars` by the transmitted waveform's spectrum (see "Modulate convention" below),
  additionally layering `TxPA.frequency_response()`'s ripple (a linear,
  frequency-selective effect -- orthogonal to the nonlinear envelope effect applied by
  `TxPABlock`, see `e2e/circuit/tx_pa.py`), and hands the result back downstream in
  `s_pars`/`DOMAIN_CFR`.

Modulate convention
--------------------
Physically, a channel's *received* spectrum is its response times the *transmitted*
spectrum: `Y(f) = H(f) * X(f)`. `s_pars` already stores `H(f)` sampled at `n_freqs`
points; `ModulateBlock` needs the SAME `n_freqs`-point grid for `X(f)`. Rather than
resample an arbitrary-length time-domain FFT onto that grid (an interpolation with
edge-clamping ambiguity -- see `InterconnectBlock._resampled_response` for that
approach elsewhere in this codebase), this module makes the two grids coincide by
construction: `X(f)` is computed as an `n_freqs`-point DFT of `tx_wave`
(`torch.fft.fft(tx_wave, n=n_freqs, dim=-1)`, zero-padded/truncated as needed), an
`n`-point DFT with bin spacing `sample_rate / n_freqs`. Bin `k` of `X` then lines up
EXACTLY with bin `k` of `s_pars`'s frequency axis whenever the caller sets
`WaveformBlock.sample_rate` equal to the span the CFR's `n_freqs` bins cover (the
scenario's swept bandwidth) -- the same "sample rate equals swept bandwidth" premise
`RFFEBlock` documents for its own `freq_span_hz`. No interpolation, no edge-clamping:
the two domains are put on one shared `n`-point DFT convention.

The PA's `frequency_response(freqs_hz)` ripple additionally needs an absolute
frequency axis (its `ripple_period_hz` is a physical quantity). Lacking a carrier in
`state`, `ModulateBlock` defaults to a baseband ramp `linspace(0, bandwidth_hz,
n_freqs)`; pass `freqs_hz` explicitly for a physically-anchored ripple phase.

Fast path (exactness requirement)
-----------------------------------
Every simulation run before this module existed assumed an ideal, flat TX -- `s_pars`
IS the received spectrum, unmodified. `ModulateBlock` preserves that exactly:
whenever `tx_wave` is absent from `state` (no `WaveformBlock` ran) OR the block was
constructed with `ideal=True` (an explicit opt-out modelling a flat, distortion-free
transmitter -- `X(f) == 1` and no ripple), `apply()` returns a dict WITHOUT an
`s_pars` key at all, so the caller's existing tensor passes through untouched --
bit-for-bit, not just numerically close (no FFT is even evaluated on that path).

Wrapping signal_generator
--------------------------
`e2e.signal_generator.signals`'s three classes were written standalone (never
wired into the pipeline) and are reused as-is -- their chirp/noise math is not
reimplemented here. Two rough edges, worked around at this module's boundary rather
than by editing that file (out of scope for this change):
* All three classes hardcode `carrier = 1.0` (the up-conversion multiply is present
  in a comment but disabled), so the accepted `fc` metadata key currently has no
  numerical effect -- `WaveformBlock`'s `fc` parameter is threaded through for
  metadata completeness/future use, not because it changes today's output.
* `RandomWidebandSignal.generate` builds its `torch.randn`/`fftfreq` tensors with no
  `device=` (so they land on CPU regardless of the input `t`'s device). `WaveformBlock`
  moves the class's output onto the target device itself after calling `generate`, so
  `tx_wave` is still correctly placed -- this module works around the gap rather than
  editing signal_generator.
"""

import torch

from e2e import frames
from e2e.blocks import device
from e2e.circuit.tx_pa import TxPA
from e2e.frames import FrameCapabilities
from e2e.signal_generator.signals import (
    FMCWSignal,
    NarrowbandSignal,
    RandomWidebandSignal,
)


_WAVEFORM_CLASSES = {
    "narrowband": NarrowbandSignal,
    "wideband": RandomWidebandSignal,
    "fmcw": FMCWSignal,
}


class WaveformBlock:
    """SOURCE: synthesizes the transmitted complex envelope `tx_wave`.

    `kind` selects one of the existing `e2e.signal_generator.signals` classes
    ('narrowband' / 'wideband' / 'fmcw'); `fc`/`bw`/`sample_rate`/`chirp_duration`
    feed that class's `metadata` dict verbatim (see the module docstring's note on
    `fc` currently being inert upstream). `n_t` sizes the time axis directly
    (defaults to `round(chirp_duration * sample_rate)`, mirroring the sample count
    `signal_generator.signals`'s own `__main__` demo derives). The single generated
    1-D waveform is broadcast across `n_tx` TX elements and `n_chirp` chirps -- this
    package's pipeline is currently single-TX/single-chirp (see `e2e/frames.py`), so
    the defaults are `n_tx=n_chirp=1`.

    Emits `tx_wave`, shape `[n_tx, n_chirp, n_t]`, complex64. Not a bridge (declares
    `emits_domain` equal to its own `domain`): it is the first stage in the TX-time
    domain, not a crossing between domains.
    """

    frame_capabilities = FrameCapabilities(
        domain=frames.DOMAIN_TX_TIME, emits_domain=frames.DOMAIN_TX_TIME,
    )

    def __init__(self, kind="fmcw", fc=0.0, bw=1e9, sample_rate=3e9,
                 chirp_duration=1e-6, n_t=None, n_tx=1, n_chirp=1):
        if kind not in _WAVEFORM_CLASSES:
            raise ValueError(
                f"unknown waveform kind {kind!r}; expected one of "
                f"{tuple(_WAVEFORM_CLASSES)}"
            )
        self.kind = kind
        self.metadata = {
            "fc": fc, "bw": bw, "sample_rate": sample_rate,
            "chirp_duration": chirp_duration,
        }
        self._signal = _WAVEFORM_CLASSES[kind](self.metadata)
        self.sample_rate = float(sample_rate)
        self.n_tx = int(n_tx)
        self.n_chirp = int(n_chirp)
        self.n_t = int(n_t) if n_t is not None else max(1, round(chirp_duration * sample_rate))

    def apply(self, state):
        # No real "input" tensor to inherit a device from (this block is a source);
        # take the device of whatever the pipeline already placed in `s_pars` when
        # present, else fall back to the library default (see e2e.blocks.device) --
        # never hardcode cpu.
        s_pars = state.get("s_pars")
        dev = s_pars.device if torch.is_tensor(s_pars) else device

        t = torch.arange(self.n_t, dtype=torch.float32, device=dev) / self.sample_rate
        wave = self._signal.generate(t)
        # Moves the result onto `dev` regardless of what device `generate()` actually
        # computed on (see the module docstring's RandomWidebandSignal note) and
        # normalizes dtype (NarrowbandSignal's constant-carrier path is real-valued).
        wave = wave.to(device=dev, dtype=torch.complex64)
        tx_wave = wave.view(1, 1, -1).expand(self.n_tx, self.n_chirp, self.n_t).clone()
        return {"tx_wave": tx_wave}


class TxPABlock:
    """Applies `TxPA.apply()` (AM/AM + AM/PM memoryless nonlinearity) to `tx_wave`.

    Stays in the TX-time domain (not a bridge): the envelope nonlinearity is exactly
    the effect that only makes sense on a real time-domain envelope, which is the
    whole point of carrying a TX time domain through the chain at all (see
    `e2e/circuit/tx_pa.py`'s module docstring for why constant- vs
    envelope-varying waveforms respond so differently to it).
    """

    frame_capabilities = FrameCapabilities(domain=frames.DOMAIN_TX_TIME)

    def __init__(self, tx_pa=None, config=None):
        self.tx_pa = tx_pa if tx_pa is not None else TxPA(config)

    def apply(self, state):
        return {"tx_wave": self.tx_pa.apply(state["tx_wave"])}


class ModulateBlock:
    """BRIDGE: TX time -> frequency. Multiplies `s_pars` by the transmitted
    waveform's spectrum (plus the PA's frequency ripple); see the module docstring's
    "Modulate convention" and "Fast path" sections for the exact grid convention and
    the bit-exactness guarantee when no `tx_wave` is present.

    `tx_pa`, if given (typically the SAME `TxPA` instance passed to `TxPABlock`, so
    both the nonlinear envelope effect and the linear ripple reflect one physical
    amplifier), layers `tx_pa.frequency_response(freqs_hz)` onto `s_pars`
    multiplicatively. `bandwidth_hz`/`freqs_hz` control the (approximate, baseband)
    frequency axis fed to that ripple only -- NOT the TX-spectrum grid, which is
    always the CFR's own `n_freqs`-point DFT (see the convention note).

    `ideal=True` is an explicit opt-out (flat, distortion-free TX) that keeps `apply`
    on the bit-exact fast path even with a `tx_wave` upstream -- e.g. to disable
    modulation for an A/B comparison without removing `WaveformBlock`/`TxPABlock`
    from the pipeline.
    """

    frame_capabilities = FrameCapabilities(
        domain=frames.DOMAIN_TX_TIME, emits_domain=frames.DOMAIN_CFR,
    )

    def __init__(self, tx_pa=None, bandwidth_hz=3e9, freqs_hz=None, ideal=False):
        self.tx_pa = tx_pa
        self.bandwidth_hz = float(bandwidth_hz)
        self.freqs_hz = freqs_hz
        self.ideal = bool(ideal)

    def apply(self, state):
        if self.ideal or "tx_wave" not in state:
            # Fast path: no `s_pars` key at all, so the caller's existing tensor
            # passes through untouched -- bit-for-bit (see module docstring).
            return {"signal_domain": frames.DOMAIN_CFR}

        s_pars = state["s_pars"]
        tx_wave = state["tx_wave"]
        n_freqs = frames.dims(s_pars).n_freqs

        # TX spectrum on the SAME n_freqs-point DFT grid as s_pars's frequency axis
        # (see "Modulate convention"); [n_tx, n_chirp, n_freqs].
        X = torch.fft.fft(tx_wave.to(torch.complex64), n=n_freqs, dim=-1)
        X = X.to(device=s_pars.device)
        s_pars = s_pars * X.view(1, *X.shape)

        if self.tx_pa is not None:
            freqs_hz = self.freqs_hz
            if freqs_hz is None:
                freqs_hz = torch.linspace(0.0, self.bandwidth_hz, n_freqs, device=s_pars.device)
            H = self.tx_pa.frequency_response(freqs_hz).to(
                device=s_pars.device, dtype=s_pars.dtype
            )
            s_pars = s_pars * H.view(1, 1, 1, -1)

        return {"s_pars": s_pars, "signal_domain": frames.DOMAIN_CFR}
