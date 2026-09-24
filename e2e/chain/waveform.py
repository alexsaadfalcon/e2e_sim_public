"""
TX-side signal chain: waveform source -> PA nonlinearity -> modulate-onto-channel bridge.

The chain is symmetric (see `e2e/frames.py`'s "signal domains" section): a transmitted
waveform lives in TX time (`DOMAIN_TX_TIME`, state key `tx_wave`,
`[n_tx, n_chirp, n_t]`), propagation and every analog transfer function live in the
frequency domain (`DOMAIN_CFR`, state key `s_pars`, `[n_rx, n_tx, n_chirp, n_freqs]`),
and (on the RX side, built elsewhere in this package) a dechirped receive waveform
comes back in time. This module builds the three TX-side blocks:

* `WaveformBlock` -- a SOURCE: synthesizes the transmitted complex envelope into
  `tx_wave` from one of the waveform classes defined below
  (`RandomWidebandSignal` / `FMCWSignal`); see "Waveform classes" below for their
  history and two stated rough edges.
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

Waveform classes (folded in from e2e/signal_generator, release-plan C2)
------------------------------------------------------------------------
`RandomWidebandSignal` and `FMCWSignal` below began life as the standalone
`e2e/signal_generator/signals.py` (pre-pipeline); that package is deleted and the
two real waveforms now live with their only consumer. Their math is unchanged
(bit-compat with every recorded `tx_wave`). The third class, `NarrowbandSignal`,
was an all-ones placeholder -- `torch.ones_like(t)` times a disabled carrier -- and
is DELETED rather than moved: a block that emits ones models nothing, and keeping
it invited "narrowband" runs that were silently flat. Two stated rough edges, kept
as-is for bit-compat:
* Both classes hardcode `carrier = 1.0` (the up-conversion multiply is present in a
  comment but disabled), so the accepted `fc` metadata key currently has no
  numerical effect -- `WaveformBlock`'s `fc` parameter is threaded through for
  metadata completeness/future use, not because it changes today's output.
* `RandomWidebandSignal.generate` builds its `torch.randn`/`fftfreq` tensors with no
  `device=` (so they land on CPU regardless of the input `t`'s device).
  `WaveformBlock` moves the output onto the target device itself after `generate`.
"""

import torch

from e2e import frames
from e2e.blocks import device
from e2e.circuit.tx_pa import TxPA
from e2e.frames import FrameCapabilities


# ================================================================================
# The waveform CLASSES -- the one branch point on the one chain.
# ================================================================================
# The owner's 2026-09-24 ballot made the waveform a three-way choice rather than a
# hardcoded FMCW chirp. The three classes below share ONE contract so the blocks
# downstream of the branch (the dechirp / mixing block, the range transform, the
# products) do not have to know which one ran -- they read the metadata it returns.
#
# THE CONTRACT a waveform class implements:
#
#   .kind                 -- the registry name ("fmcw" | "ofdm" | "jsac").
#   .generate(t)          -> complex64 samples of the transmitted complex envelope on
#                            the time grid `t`. This is the legacy entry point and is
#                            what `WaveformBlock` puts in `tx_wave`.
#   .record_metadata()    -> a dict describing how the RECEIVED record is organised,
#                            which is what the rest of the chain actually needs:
#         "kind"          -- the class name again, so a product can refuse by name;
#         "cube_axes"     -- {"slow": ..., "fast": ...}, the pair
#                            `frames.require_cube_axes` checks. FMCW yields
#                            (chirp, range_bin); OFDM yields (symbol, subcarrier).
#         "n_fast"        -- samples per chirp / subcarriers per symbol;
#         "n_slow"        -- chirps / symbols per frame;
#         "delta_f_hz"    -- the fast-axis spacing the range axis calibrates from,
#                            or None when the class cannot name one;
#         "constant_envelope" -- whether the transmitted envelope is unit modulus.
#                            This is not decoration: it is the precondition under which
#                            the dechirp commutes with the front end's envelope
#                            nonlinearity (`e2e/chain/frontend.py`), and the reason the
#                            TX PA is inert on FMCW and bites on OFDM (PAPR ~10 dB).
#
# See notes/ONE_CHAIN_CONTRACT_2026-09-24.md section 2 for the option table this
# implements, and `notes/JSAC_WAVEFORM_2026-09-24.md` (a research note that did not yet
# exist when this was written) for which hybrid `jsac` is to be.

WAVEFORM_KINDS = ("fmcw", "ofdm", "jsac")


class _WaveformClass:
    """Shared base: holds `metadata` and supplies the default record description.

    Subclasses override `generate` and, where they differ, `record_metadata`.
    """

    kind = None
    constant_envelope = False

    def __init__(self, metadata: dict):
        self.metadata = dict(metadata or {})

    def generate(self, t):                                   # pragma: no cover - abstract
        raise NotImplementedError(
            f"waveform class {type(self).__name__} does not implement generate()")

    def record_metadata(self):
        md = self.metadata
        bw = float(md.get("bw") or 0.0)
        n_fast = md.get("n_fast")
        delta_f = md.get("delta_f_hz")
        if delta_f is None and n_fast:
            delta_f = bw / float(n_fast) if bw else None
        return {
            "kind": self.kind,
            "cube_axes": dict(frames.CUBE_AXES_FMCW),
            "n_fast": n_fast,
            "n_slow": md.get("n_slow", 1),
            "delta_f_hz": delta_f,
            "constant_envelope": self.constant_envelope,
        }


class RandomWidebandSignal(_WaveformClass):
    """Bandlimited complex noise waveform (folded in from signal_generator -- see the
    module docstring's "Waveform classes" section for provenance and rough edges).

    NOT one of the three registered `WAVEFORM_KINDS`: it is a probe waveform with
    no sensing or comms consumer in the repo, kept because existing call sites use it.
    Its envelope is not constant, so nothing that relies on the dechirp/nonlinearity
    commutation may be run on it.
    """

    kind = "wideband"
    constant_envelope = False

    def generate(self, t):
        sample_rate = 1 / (t[1] - t[0])
        n_samples = t.shape[0]
        f = torch.fft.fftfreq(n_samples, 1 / sample_rate)
        bw = self.metadata['bw']

        # randn bandlimited signal
        signal = torch.randn(t.shape)
        signal[(f < -bw / 2) | (f > bw / 2)] = 0
        # modulate up to fc
        signal = torch.fft.ifft(signal)
        # carrier = torch.exp(2j * torch.pi * fc * t)
        carrier = 1.0  # no modulation, keep baseband
        signal = signal * carrier

        return signal


class FMCWSignal(_WaveformClass):
    """Ideal linear-FMCW chirp: constant slope k = bw / chirp_duration. THE DEFAULT.

    The linearity is an APPROXIMATION shared by the whole sensing chain (see
    `e2e.chain.rd_synth`'s scope list): a real PLL/VCO sweep deviates from the ideal
    ramp (chirp nonlinearity), smearing the dechirped beat tone and raising the
    close-in sidelobe floor. Deliberately not modelled.

    `constant_envelope = True` is load-bearing, not a label: it is the precondition
    that makes `|b[n]| = |r(t_n)|` exact and therefore licenses running the front end
    on the beat record (`e2e/chain/frontend.py`). It is also why the TX PA is inert
    here -- a Rapp nonlinearity applied to a unit-modulus signal is a constant.
    """

    kind = "fmcw"
    constant_envelope = True

    def generate(self, t):
        bw = self.metadata['bw']
        chirp_duration = self.metadata['chirp_duration']

        # compute chirp constant
        k = bw / chirp_duration

        # compute baseband chirp
        signal = torch.exp(2j * torch.pi * (k * t ** 2 / 2))
        signal *= torch.exp(-2j * torch.pi * bw / 2 * t)
        # modulate up to fc
        # carrier = torch.exp(2j * torch.pi * fc * t)
        carrier = 1.0  # no modulation, keep baseband
        signal = signal * carrier

        return signal


class OFDMISACSignal(_WaveformClass):
    """OFDM waveform, sensed by SYMBOL DIVISION -- the JSAC-capable option.

    Sensing works differently from FMCW and the difference is the whole point: there is
    no dechirp. The receiver estimates `H_est = Y_q / X_q` per subcarrier (LS on an
    all-pilot symbol, `e2e/comms/channel.py`), and THAT estimate is the frequency-domain
    record the range transform then compresses. So the cube's fast axis counts
    SUBCARRIERS, not range bins, until the transform runs -- which is exactly why
    `cube_axes` travels in state and why a product built for `(chirp, range_bin)`
    refuses this waveform by name instead of drawing a plausible picture of nothing.

    The envelope is NOT constant (PAPR ~10 dB), with two consequences the chain must
    respect: the TX PA's Rapp nonlinearity is live here where it is inert on FMCW, and
    the front end's clamp (`rffe_model.py`'s clamp-at-cubic-peak) engages where an FMCW
    chirp never reaches it. Nothing about `FrontEndBlock`'s placement argument survives
    on this path either: the dechirp/nonlinearity commutation is a unit-modulus
    identity, so on OFDM the front end must act on the actual sampled record.

    IMPLEMENTED AS FAR AS `e2e/comms/ofdm.py` ALLOWS: this class builds the transmitted
    time-domain symbol (pilot-loaded, CP-prefixed) via `OFDMModem`, and reports the
    record layout. The RECEIVE bridge -- `Y/X` per subcarrier into the spine's frequency
    record -- lives in the comms package (`e2e/comms/isac.py` does the CFR version
    today) and is wired by the shard that owns it.

    CP CAVEAT, stated because it is a real constraint and not a default: the cyclic
    prefix must exceed the channel's delay SPREAD, not its fourth-strongest path. F94
    records the munich Ka diffuse families reaching 454 ns, so a CP below ~1/4 symbol
    on that trace produces inter-symbol interference that the range transform will
    faithfully render as clutter. Read it off the PDP; do not assume the default.
    """

    kind = "ofdm"
    constant_envelope = False

    def __init__(self, metadata: dict):
        super().__init__(metadata)
        self.fft_size = int(self.metadata.get("fft_size", 64))
        self.cp_len = int(self.metadata.get("cp_len", self.fft_size // 4))
        self.n_symbols = int(self.metadata.get("n_symbols", 1))
        self.bits_per_symbol = int(self.metadata.get("bits_per_symbol", 2))
        self._modem = None

    def modem(self):
        """The `OFDMModem` this waveform transmits through, built lazily so importing
        the waveform module does not drag in the comms package."""
        if self._modem is None:
            from e2e.comms.ofdm import OFDMModem
            self._modem = OFDMModem(
                fft_size=self.fft_size, cp_len=self.cp_len,
                bits_per_symbol=self.bits_per_symbol,
                **{k: v for k, v in self.metadata.items()
                   if k in ("n_active", "pilot_spacing")})
        return self._modem

    def generate(self, t):
        """The transmitted time-domain OFDM record (CP included), truncated or
        zero-padded to `t`'s length so it drops into `WaveformBlock` unchanged.

        Pilot-loaded random data: the sensing path only needs a KNOWN `X`, and using
        the modem's own pilot/data machinery keeps the transmitted grid identical to
        what the comms head will divide by.
        """
        from e2e.comms.ofdm import random_bits
        modem = self.modem()
        n_bits = modem.data_bits_per_symbol_block * self.n_symbols
        bits = random_bits(n_bits, seed=self.metadata.get("seed", 0))
        # `modulate` returns (tx_time, tx_freq); the transmitted GRID is kept because
        # the sensing path divides by it (`H_est = Y/X`) and the comms head demaps
        # against it -- a receiver that re-guesses `X` is not the same receiver.
        tx_time, tx_freq = modem.modulate(bits, self.n_symbols)
        self.tx_freq = tx_freq
        self.tx_bits = bits
        tx = tx_time.reshape(-1)
        n_t = int(t.shape[0])
        if tx.shape[0] < n_t:
            tx = torch.cat([tx, tx.new_zeros(n_t - tx.shape[0])])
        return tx[:n_t].to(torch.complex64)

    def record_metadata(self):
        md = super().record_metadata()
        md.update({
            "cube_axes": dict(frames.CUBE_AXES_OFDM),
            "n_fast": self.fft_size,
            "n_slow": self.n_symbols,
            "cp_len": self.cp_len,
            "delta_f_hz": (float(self.metadata["bw"]) / self.fft_size
                           if self.metadata.get("bw") else None),
        })
        return md


class JSACSignal(_WaveformClass):
    """The hybrid sensing-and-communication waveform. REGISTERED, NOT IMPLEMENTED.

    It is a registered class rather than an absent one on purpose: the branch point on
    the diagram is real, the contract it must satisfy is written down, and a caller
    asking for it gets a refusal that says what is missing instead of a KeyError that
    says the option does not exist.

    WHICH hybrid is an open research question at the time of writing. The two candidates
    the contract (section 2) costs out:

      * **OFDM-ISAC** -- one OFDM waveform doing both jobs; sensing by symbol division.
        Already half-built as `OFDMISACSignal` above; a JSAC preset on top of it is the
        cheaper path.
      * **FMCW with embedded data** -- chirp-to-chirp PSK on the SLOW axis. Sensing is
        unchanged after code removal (the data is a known unit-modulus factor per
        chirp, so the radar divides it out exactly and the cube is bit-for-bit what it
        would have been); comms decodes the phase at the sync/line-of-sight range bin.
        Multi-chirp frames only -- munich frames are single-chirp, so they carry zero
        bits and this option is corpus-only.

    `notes/JSAC_WAVEFORM_2026-09-24.md` is to decide between them. Implementing it is
    the other shard's task; what THIS class owes it is the contract above
    (`record_metadata` must name the cube axes and whether the envelope is constant),
    plus a `generate` that produces the transmitted record.
    """

    kind = "jsac"
    constant_envelope = False

    def generate(self, t):
        raise NotImplementedError(
            "the JSAC hybrid waveform is registered but not implemented: which hybrid "
            "it is (OFDM-ISAC, or FMCW with chirp-to-chirp PSK on the slow axis) is "
            "still open -- see notes/JSAC_WAVEFORM_2026-09-24.md and this class's "
            "docstring for the contract an implementation must satisfy. Use "
            "kind='fmcw' (or 'ofdm') until it lands."
        )

    def record_metadata(self):
        raise NotImplementedError(
            "JSACSignal.record_metadata: see generate() -- the hybrid is not chosen yet."
        )


#: The waveform registry -- the `kind` switch on `WaveformBlock`. "fmcw" is the
#: default and the only one a v1.1 preset selects; "ofdm" and "jsac" are the v1.2
#: options the contract's section 2 costs out. "wideband" is not one of
#: `WAVEFORM_KINDS`: it is a probe waveform with no consumer (see its docstring).
_WAVEFORM_CLASSES = {
    "wideband": RandomWidebandSignal,
    "fmcw": FMCWSignal,
    "ofdm": OFDMISACSignal,
    "jsac": JSACSignal,
}


def fmcw_plan_from_freq_plan(freq_plan, *, n_rx, n_chirps=1, fs_hz=None,
                             chirp_period_s=None, name="fmcw_from_freq_plan"):
    """Build the FMCW `RadarConfig` that CONSUMES a stored CFR grid exactly.

    The stored channel is a CFR on a uniform grid; sampling a CFR on an FMCW ramp's
    frequency grid and conjugating it IS the dechirped beat record
    (`e2e/environment/rt_signal_chain.py`, eq. 3). That identity holds only when the
    ramp satisfies `S/fs == df` on the grid's OWN spacing, which for a v2 pkl is
    `(stop - start) / (num_freqs - 1)` -- endpoint-inclusive, because
    `sionna_simple_channel.build_frequencies` is an `np.linspace`. Getting this wrong
    by the `N/(N-1)` factor is a silent 0.02% range-scale error, which is why the plan
    is DERIVED here rather than typed into a preset per file (the one exception,
    `radar_config.MUNICH_KA_FMCW`, carries the same arithmetic for the shipped file
    and is checked against this function in tests/test_one_chain_spine.py).

    The sweep time is free: any `(S, fs)` with `S/fs = df` gives the same beat record,
    so `fs_hz`/`chirp_period_s` change no image -- only the sample rate the noise and
    IF filters are referenced to. Defaults put `fs_hz` at `num_freqs * df / 200e-6`
    (a 200 us sweep) unless the caller names one.

    Raises ValueError, naming both numbers, if `freq_plan` cannot describe a grid.
    """
    from e2e.radar_config import RadarConfig

    if not freq_plan:
        raise ValueError(
            "fmcw_plan_from_freq_plan needs a freq_plan dict "
            "{start_hz, stop_hz, num_freqs}; got none. A legacy pkl carries no plan, "
            "so its frames have no metre calibration -- pass a RadarConfig explicitly."
        )
    try:
        start = float(freq_plan["start_hz"])
        stop = float(freq_plan["stop_hz"])
        num = int(freq_plan["num_freqs"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"freq_plan {freq_plan!r} is missing start_hz/stop_hz/num_freqs"
        ) from exc
    if num < 2 or not stop > start:
        raise ValueError(
            f"freq_plan must span a band with >= 2 points, got start={start} "
            f"stop={stop} num_freqs={num}"
        )
    df = (stop - start) / (num - 1)
    # bandwidth_hz is defined over the SAMPLED window (n_samples * df), which is one
    # grid step wider than stop-start precisely because the grid is endpoint-inclusive.
    bandwidth = df * num
    if fs_hz is None:
        # A 200 us sweep: n_samples / T_sweep. Arbitrary and stated as such -- it
        # changes no image, only what `fs` the noise/IF blocks are referenced to.
        fs_hz = num / 200e-6
    slope = bandwidth / (num / float(fs_hz))
    got = slope / float(fs_hz)
    if abs(got - df) > 1e-6 * df:
        raise ValueError(
            f"chirp plan does not sample the stored grid: S/fs = {got:.6f} Hz but the "
            f"stored grid spacing is {df:.6f} Hz"
        )
    return RadarConfig(
        name=name, f0_hz=start, bandwidth_hz=bandwidth, n_tx=1, n_rx=int(n_rx),
        n_chirps=int(n_chirps), n_samples=num, fs_hz=float(fs_hz),
        chirp_period_s=float(chirp_period_s if chirp_period_s is not None
                             else 1.25 * num / float(fs_hz)),
        mimo="single",
    )


class WaveformBlock:
    """SOURCE: synthesizes the transmitted complex envelope `tx_wave`.

    `kind` selects one of the waveform classes above -- 'fmcw' (the default),
    'ofdm', 'jsac' (see WAVEFORM_KINDS), or the unregistered probe 'wideband'.
    'narrowband' was an all-ones placeholder, deleted in the C2 fold-in; asking for
    it raises with this history. `fc`/`bw`/`sample_rate`/`chirp_duration` feed the
    class's `metadata` dict verbatim (see the module docstring's note on `fc`
    currently being inert). `n_t` sizes the time axis directly (defaults to
    `round(chirp_duration * sample_rate)`). The single generated
    1-D waveform is broadcast across `n_tx` TX elements and `n_chirp` chirps -- this
    package's pipeline is currently single-TX/single-chirp (see `e2e/frames.py`), so
    the defaults are `n_tx=n_chirp=1`.

    Emits `tx_wave`, shape `[n_tx, n_chirp, n_t]`, complex64. Not a bridge (declares
    `emits_domain` equal to its own `domain`): it is the first stage in the TX-time
    domain, not a crossing between domains.
    """

    # The transmit path is a TRIBUTARY of the chain, not a segment of it: this block
    # produces `tx_wave` alongside whatever the chain is carrying, and the two merge at
    # ModulateBlock. Declaring DOMAIN_ANY says exactly that -- it consumes no chain
    # payload, so it is legal wherever it is placed before the merge.
    frame_capabilities = FrameCapabilities(
        domain=frames.DOMAIN_ANY,
    )

    def __init__(self, kind="fmcw", fc=0.0, bw=1e9, sample_rate=3e9,
                 chirp_duration=1e-6, n_t=None, n_tx=1, n_chirp=1):
        if kind == "narrowband":
            raise ValueError(
                "waveform kind 'narrowband' was removed (release-plan C2): it was an "
                "all-ones placeholder (constant baseband, disabled carrier) that "
                "modelled nothing. Use 'fmcw' or 'wideband'."
            )
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
        # normalizes dtype.
        wave = wave.to(device=dev, dtype=torch.complex64)
        tx_wave = wave.view(1, 1, -1).expand(self.n_tx, self.n_chirp, self.n_t).clone()
        # The record description travels WITH the waveform, so the dechirp/mixing
        # block and the products read what this waveform actually produced instead of
        # assuming FMCW. `cube_axes` is the pair `frames.require_cube_axes` checks --
        # it is what makes a product refuse an OFDM cube by name.
        record = self._signal.record_metadata()
        return {"tx_wave": tx_wave, "waveform": record,
                "waveform_kind": record["kind"]}


class TxPABlock:
    """Applies `TxPA.apply()` (AM/AM + AM/PM memoryless nonlinearity) to `tx_wave`.

    Stays in the TX-time domain (not a bridge): the envelope nonlinearity is exactly
    the effect that only makes sense on a real time-domain envelope, which is the
    whole point of carrying a TX time domain through the chain at all (see
    `e2e/circuit/tx_pa.py`'s module docstring for why constant- vs
    envelope-varying waveforms respond so differently to it).
    """

    # Also on the transmit tributary: it rewrites `tx_wave` in place and never touches
    # the chain's payload, so it too is domain-agnostic. It does require the waveform to
    # exist, which it checks itself with an actionable error.
    frame_capabilities = FrameCapabilities(domain=frames.DOMAIN_ANY)

    def __init__(self, tx_pa=None, config=None):
        self.tx_pa = tx_pa if tx_pa is not None else TxPA(config)

    def apply(self, state):
        if "tx_wave" not in state:
            raise frames.FrameContractError(
                "TxPABlock needs a transmitted waveform to distort, but none is in "
                "state -- put a WaveformBlock before it."
            )
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

    # The merge point, and therefore NOT a domain bridge: it folds the transmitted
    # spectrum into the channel's frequency response and leaves the chain in the
    # frequency domain it was already in. (An earlier design called this a second
    # crossing, symmetric with the dechirp. That was wrong: the transmit waveform joins
    # the chain by multiplication, it does not carry the chain across.)
    # MIMO and multi-chirp ride along natively: the transmitted spectrum is
    # [n_tx, n_chirp, n_freqs] and broadcasts against the channel's
    # [n_rx, n_tx, n_chirp, n_freqs], which is exactly the per-transmitter pairing a
    # MIMO radar needs. Declaring otherwise blocked the 3-transmit TI preset outright.
    frame_capabilities = FrameCapabilities(
        domain=frames.DOMAIN_CFR,
        accepts_mimo=True,
        chirps=frames.CHIRP_NATIVE,
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

        if "s_pars" not in state:
            raise frames.FrameContractError(
                "ModulateBlock folds the transmitted spectrum into a channel response, "
                "but no 's_pars' frame is in state -- it must run while the chain is in "
                "the cfr domain, after an environment block has supplied a frame."
            )
        s_pars = state["s_pars"]
        tx_wave = state["tx_wave"]
        n_freqs = frames.dims(s_pars).n_freqs

        # TX spectrum on the SAME n_freqs-point grid as s_pars's frequency axis.
        #
        # The fftshift is load-bearing and was missing: torch.fft.fft returns natural
        # DFT order (DC, +f, ..., Nyquist, -f, ..., -df), while EVERY s_pars frequency
        # axis in this repo ascends monotonically from -B/2 to +B/2 (rt_gen's
        # beat_frequencies, and the linspace grids the classic pipeline builds). Without
        # the shift, the chirp's negative-frequency half multiplies the channel's
        # POSITIVE half and vice versa -- an adversarial review measured the resulting
        # misalignment at exactly B/2 (1.5 GHz on a 3 GHz band), which scrambles range
        # and Doppler rather than degrading them subtly.
        X = torch.fft.fft(tx_wave.to(torch.complex64), n=n_freqs, dim=-1)
        X = torch.fft.fftshift(X, dim=-1)
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
