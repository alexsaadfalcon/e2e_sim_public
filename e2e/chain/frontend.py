"""`FrontEndBlock` -- the analog RF front end ON THE SAMPLED BEAT RECORD.

This is the v1.2 placement from `notes/ONE_CHAIN_CONTRACT_2026-09-24.md` (section 1.2
row 5, FULL column), built after the owner's 2026-09-24 ballot moved shard 1 from the
MVC to the FULL contract.

WHAT MOVED, AND WHY IT IS ALLOWED TO
------------------------------------
`e2e.blocks.RFFEBlock` runs the same circuit cascade on `ifft(CFR)` -- the channel
IMPULSE RESPONSE, a signal no amplifier ever sees. That is F96 in
`notes/ESTABLISHED_FACTS.md` (VERIFIED: source + numeric oracle), and it is what the
v1.1 cards had to disclose on stage.

The move is exact for an ideal linear FMCW chirp, and the reason is a commutation
identity worth stating once (contract section 1.1, fact 3):

    with a unit-modulus transmit chirp `s(t)`, the beat record is `b = s* r`, so
    `|b[n]| = |r(t_n)|` EXACTLY, multipath included. A memoryless envelope
    nonlinearity `g(|r|) * r/|r|` therefore dechirps to `g(|b|) * b/|b|`.

`e2e/circuit/rffe_model.py`'s LNA / mixer / baseband cascade is written in exactly that
envelope form (`rffe_model.py:113-149`), so it applies UNCHANGED to beat samples. The
alternative -- an RF-rate record -- would be `T_sweep * B` samples per element, ~4.9 GB
per munich frame, which is not a live demo.

Two approximations are inherited, not introduced: the cubic's intermodulation products
alias at the ADC rate, and the model has NO RF selectivity ahead of the mixer, so
nothing filters them. Both are shared with the v1.0 placement.

WHERE THE LICENCE STOPS (measured 2026-09-24, do not read past this)
--------------------------------------------------------------------
The commutation argument requires the cascade to be a function of the ENVELOPE alone,
i.e. equivariant under a global phase rotation. The LNA and mixer stages are -- both
were deliberately written in baseband-equivalent envelope form. **The baseband stage is
not**: it clamps I and Q SEPARATELY (`rffe_model.py`: `torch.clamp(VBB_I, -Vbias_BB,
Vbias_BB)` and likewise for Q), which is a SQUARE region in the complex plane rather
than a circular one. Once that clamp engages the output depends on the signal's phase
and the commutation stops being exact.

Measured relative error in `g(e^{i0} x) == e^{i0} g(x)` (benchmark_v1_ka config):

    drive 1e-6  ->  1.6e-7   (float32 rounding; structurally exact)
    drive 1e-4  ->  1.3e-5
    drive 1e-2  ->  1.4e-1   <- baseband clamp engaging
    drive >=3e-2 -> 4.7e-1   (saturated)

The demo operating point is far below that (`signal_scaling=1e-7`; nothing clips,
STATE section 5), so the shipped screens are in the exact regime. A preset that drives
the baseband stage into its clamp is NOT covered by this argument. Pinned by
`tests/test_full_chain_frontend.py::test_the_cascade_is_phase_equivariant_below_baseband_clipping`.
Fixing it properly means a circular (envelope) baseband clamp, which would change the
v1.0 numbers and is therefore a separate decision, not a quiet edit.

WHAT ACTUALLY CHANGES NUMERICALLY
---------------------------------
1. The normalisation reference, when `physical_scale=False`. `RFFEBlock` divides by
   `mean|ifft(CFR)|`; this block divides by `mean|beat|`. On an N-point frame those
   differ by ~N, i.e. ~10*log10(N) = 37 dB of peak drive at N = 5000.
2. The noise-bandwidth reference. The legacy placement injects `NBB*BW_IF` per
   frequency bin, pre-divided by `nt` to compensate the caller's unnormalised forward
   FFT. Here there IS no FFT round trip, so the injection is per SAMPLE, band-
   referenced to `min(if_bw, fs)` -- the same number whenever `if_bw <= fs`, which is
   the operating point every shipped preset uses.

NOISE IS INJECTED ONCE (contract section 1.4, FULL)
---------------------------------------------------
When this block is on the chain it is THE thermal injection, with `F` its own Friis
cascade (`FLNA * Fmix * FBB`, computed inside the circuit model). It stamps
`state['noise_injected_by'] = 'frontend'` so `ThermalNoiseBlock` downstream records the
link budget as provenance and adds NO second floor. The chain had two injections
(F81's two mechanisms) and the comms head adds a third; this is the seam that ends the
double-count on the radar path.

`placement="impulse"` is the LEGACY parity flag, and exists for exactly one reason: the
stored ML corpora were generated at the old placement, and `tests/test_ml_store_cfr.py`
/ `tests/test_webapp_live_chain.py` gate live-vs-stored at max |diff| = 0 codes. It is
not an alternative physics; it is a recorded fact about how a file on disk was made.
"""

from __future__ import annotations

import torch

from e2e import frames
from e2e.frames import FrameCapabilities
from e2e.circuit.rffe_model import circuit_model_batch, get_RX_config


#: Where the front end acts.
#:   "beat"    -- on the sampled beat record, after the dechirp. The physics (see the
#:                module docstring). THE DEFAULT.
#:   "impulse" -- on `ifft(CFR)`, the v1.0/F96 placement. Legacy, for corpus parity
#:                only; `e2e.blocks.RFFEBlock` is the block that implements it, and a
#:                `FrontEndBlock` asked for it says so rather than pretending.
PLACEMENTS = ("beat", "impulse")


class FrontEndBlock:
    """Analog RF front-end circuit distortion + thermal noise, on `adc`.

    Consumes `adc [n_rx, n_chirp, n_samples]` (DOMAIN_RX_TIME), rewrites it, and emits
    `PRX`, `amplitude_scale` and `noise_injected_by`.

    Knobs, matching `RFFEBlock`'s so a preset moves across unchanged:

    * `n` -- element count for `get_RX_config`; inferred from the frame when None.
    * `lna_bias_ma` / `if_bw_mhz` -- the two demo knobs (Thrust 1), overriding single
      columns of the per-element config table.
    * `physical_scale` -- True: the record is already in volts at the LNA input, use it
      as is. False: renormalise to `signal_scaling` against `mean|beat|` and stamp
      `amplitude_scale="normalised"` so the F63 guard in `ThermalNoiseBlock` can refuse
      to put an absolute floor under a scale-erased frame.
    * `fs_hz` -- the beat sample rate. Sizes the optional IF boxcar AND caps the noise
      band at `min(if_bw, fs)`. Read from `cfg.fs_hz` when not given; if neither
      answers, the block raises rather than guessing a bandwidth, because guessing one
      is a silent multi-dB error in the floor.
    * `seed` -- as `RFFEBlock`: None draws from the global RNG; an int seeds a
      per-instance generator advanced by frame index.
    * `inject_noise` -- False runs the cascade with NO thermal draw at all (for an
      oracle that wants the nonlinearity alone). It does NOT stamp
      `noise_injected_by`, so a downstream `ThermalFloor` remains the injection.
    """

    frame_capabilities = FrameCapabilities(
        domain=frames.DOMAIN_RX_TIME, chirps=frames.CHIRP_NATIVE, accepts_mimo=True,
    )

    #: Column indices into `get_RX_config`'s `[nRx, 7]` table -- same layout as
    #: `RFFEBlock`'s; named in one more place rather than read positionally.
    RX_CONFIG_IBIAS_LNA = 0     # A
    RX_CONFIG_IF_BW = 6         # Hz

    def __init__(self, cfg=None, *, n=None, signal_scaling=1e-5, if_filter=False,
                 physical_scale=True, lna_bias_ma=None, if_bw_mhz=None, seed=None,
                 fs_hz=None, placement="beat", inject_noise=True, device=None):
        if placement not in PLACEMENTS:
            raise ValueError(
                f"unknown front-end placement {placement!r}; expected one of {PLACEMENTS}"
            )
        if placement == "impulse":
            raise ValueError(
                "FrontEndBlock(placement='impulse') is not implemented here on purpose: "
                "the impulse-domain placement IS `e2e.blocks.RFFEBlock` + `CircuitStage`, "
                "which runs BEFORE the dechirp. Build that instead, and record "
                "placement='impulse' in the chain's own flags so a corpus says how it "
                "was made (contract section 3.5 / 5.4)."
            )
        self.cfg = cfg
        self.placement = placement
        self.n = n
        self.signal_scaling = float(signal_scaling)
        self.if_filter = bool(if_filter)
        self.physical_scale = bool(physical_scale)
        self.lna_bias_ma = lna_bias_ma
        self.if_bw_mhz = if_bw_mhz
        self.inject_noise = bool(inject_noise)
        self.seed = seed
        self._frame_idx = 0
        self._fs_hz = None if fs_hz is None else float(fs_hz)
        self._rx_config = None
        self._device = device
        if lna_bias_ma is not None and not lna_bias_ma > 0:
            raise ValueError(f"lna_bias_ma must be > 0, got {lna_bias_ma!r}")
        if if_bw_mhz is not None and not if_bw_mhz > 0:
            raise ValueError(f"if_bw_mhz must be > 0, got {if_bw_mhz!r}")

    # ------------------------------------------------------------------ config table
    def rx_config(self, n, device):
        """The per-element `[n, 7]` config table, built once and cached.

        Built lazily because `n` is a property of the frame, not of the preset: a
        front end configured for "this chain" should not have to be told how many
        elements the source has before the source has been read.
        """
        if self._rx_config is None or self._rx_config.shape[0] != n:
            table = get_RX_config(n).to(device)
            if self.lna_bias_ma is not None:
                table[:, self.RX_CONFIG_IBIAS_LNA] = float(self.lna_bias_ma) * 1e-3
            if self.if_bw_mhz is not None:
                table[:, self.RX_CONFIG_IF_BW] = float(self.if_bw_mhz) * 1e6
            self._rx_config = table
        return self._rx_config

    def fs_hz(self):
        """The beat sample rate the noise band and the IF boxcar are referenced to."""
        if self._fs_hz is not None:
            return self._fs_hz
        fs = getattr(self.cfg, "fs_hz", None)
        if fs is None:
            raise ValueError(
                "FrontEndBlock needs the beat sample rate to reference its noise "
                "bandwidth and size its IF filter, and neither `fs_hz=` nor `cfg.fs_hz` "
                "supplied one. Guessing it is a silent multi-dB error in the floor "
                "(the band is min(if_bw, fs)), so this refuses instead."
            )
        return float(fs)

    def noise_band_hz(self, rx_config):
        """`min(if_bw, fs)` per element -- the ONE definition of the noise bandwidth
        on this path, and the same definition `if_bw_hz` uses as a low-pass corner. If
        the two halves of that knob disagree, the knob is lying."""
        return torch.clamp(rx_config[:, self.RX_CONFIG_IF_BW], max=self.fs_hz())

    def reset(self):
        """Rewind the per-frame counter (and hence the seed sequence) to frame 0."""
        self._frame_idx = 0

    # -------------------------------------------------------------------------- apply
    def apply(self, state):
        adc = state["adc"]
        if adc.dim() != 3:
            raise frames.FrameContractError(
                f"FrontEndBlock expects adc [n_rx, n_chirp, n_samples], got shape "
                f"{tuple(adc.shape)}"
            )
        n_rx, n_chirp, n_samples = adc.shape
        n = self.n or n_rx
        if n != n_rx:
            raise ValueError(
                f"FrontEndBlock was configured for n={n} elements but the frame has "
                f"n_rx={n_rx}; the per-element config table would not line up."
            )
        rx_config = self.rx_config(n, adc.device)

        frame = adc.view(n_rx, 1, n_chirp, n_samples)
        if not self.physical_scale:
            # Same renormalisation RFFEBlock performs, but against `mean|beat|` rather
            # than `mean|ifft(CFR)|` -- see the module docstring, change (1).
            frame = frame * self.signal_scaling / torch.mean(torch.abs(frame))

        fs = self.fs_hz()
        if self.inject_noise:
            # `circuit_model_batch` flattens [n_rx, 1, n_chirp, n_samples] to
            # [n_rx * n_chirp, n_samples] and expects every per-element config column
            # as a matching [batch, 1]. Replicate the per-element band across chirps in
            # the SAME order, or the bands land on the wrong elements silently.
            band = (self.noise_band_hz(rx_config)
                    .view(n_rx, 1).expand(n_rx, n_chirp).reshape(n_rx * n_chirp, 1))
            kwargs = {"noise_band_hz": band, "noise_divisor": 1.0}
        else:
            # A zero band is a zero variance: the cascade runs, the draw contributes
            # nothing. Cheaper and more obviously correct than a second code path.
            kwargs = {"noise_band_hz": 0.0, "noise_divisor": 1.0}

        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=frame.device)
            generator.manual_seed(int(self.seed) + self._frame_idx)
        self._frame_idx += 1

        out, PRX = circuit_model_batch(rx_config, frame, fs, if_filter=self.if_filter,
                                       generator=generator, **kwargs)
        updates = {
            "adc": out.reshape(adc.shape).to(torch.complex64),
            "PRX": PRX,
            "amplitude_scale": "absolute" if self.physical_scale else "normalised",
            "front_end_placement": self.placement,
        }
        if self.inject_noise:
            # THE seam that makes noise-once true: a downstream ThermalFloor reads this
            # and records the link budget without adding a second floor
            # (contract section 1.4).
            updates["noise_injected_by"] = "frontend"
            updates["noise_band_hz"] = float(
                torch.min(self.noise_band_hz(rx_config)).item())
        return updates
