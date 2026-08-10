"""RX-time-domain processing blocks: ADC impairments, digitization, radar-cube product.

These are the receive-side stages that pick up where a `DechirpBlock` (built
elsewhere; see `e2e/chain/__init__.py`) leaves off: `state['adc']`, a complex64
`[n_rx, n_chirp, n_samples]` cube, `state['signal_domain'] == frames.DOMAIN_RX_TIME`.
Each block below follows the same `apply(state) -> dict of state updates` protocol as
`e2e/blocks.py` and declares a `frames.FrameCapabilities` naming that domain, so
`Simulation._check_frame_contract` raises a named `FrameContractError` if one of these
runs before the chain has actually crossed into RX time (e.g. no DechirpBlock inserted).

Three blocks:

- `ImpairmentBlock`  -- wraps `e2e.ml.impairments.apply_all` (phase noise, TX/RX
  leakage, clutter). Serial stage: rewrites `adc`.
- `QuantizerBlock`   -- ADC digitization (full-scale clip + uniform quantization).
  Serial stage: rewrites `adc`.
- `RadarCubeBlock`   -- range-Doppler product via `e2e.ml.transforms.adc_to_rd`.
  Downstream product block: reads `adc`, emits `radar_cube`, never rewrites `adc`.
"""

import torch

from e2e import frames
from e2e.frames import FrameCapabilities
from e2e.ml.impairments import apply_all, ClutterParams, LeakageParams, PhaseNoiseParams
from e2e.ml.transforms import adc_to_rd, tdm_deinterleave


# Every block here consumes the post-dechirp ADC cube; none handle the chirp/MIMO axes
# of a 4-D S-parameter frame (the ADC cube is 3-D, so those checks are skipped anyway --
# see frames._check_frame_contract -- but the domain declaration is what actually gates
# a mis-ordered chain).
_RX_TIME = FrameCapabilities(domain=frames.DOMAIN_RX_TIME, chirps=frames.CHIRP_NATIVE)

# Mirrors `impairments._STAGES`' name/class pairing (kept local rather than importing
# that module-private tuple) so a chain_params dict/sampler is resolved into concrete,
# recordable dataclass instances the SAME way `apply_all` resolves them internally.
_IMPAIRMENT_STAGES = (
    ("phase_noise", PhaseNoiseParams),
    ("leakage", LeakageParams),
    ("clutter", ClutterParams),
)


def _resolve_impairment_params(chain_params):
    """`chain_params` (dict of name -> dataclass instance / kwargs dict / None, or
    falsy) -> dict of name -> concrete dataclass instance (or None for a skipped
    stage), filling in defaults for any stage the caller didn't mention.

    This is the exact resolution `impairments.apply_all` performs internally; doing
    it here too (cheaply -- no randomness involved) lets `ImpairmentBlock` record the
    ACTUAL per-stage params object it is about to hand to `apply_all`, rather than
    the possibly-partial/possibly-callable input the caller supplied.
    """
    chain_params = dict(chain_params) if chain_params else {}
    resolved = {}
    for name, cls in _IMPAIRMENT_STAGES:
        if name in chain_params and chain_params[name] is None:
            resolved[name] = None
            continue
        val = chain_params.get(name, cls())
        resolved[name] = val if isinstance(val, cls) else cls(**val)
    return resolved


class ImpairmentBlock:
    """FMCW ADC impairments (phase noise, TX/RX leakage, clutter) -- `e2e.ml.impairments
    .apply_all` as a chain stage. Serial stage: rewrites `adc` in place (in the state
    dict, not the tensor).

    `chain_params` (mirrors `apply_all`'s argument) is EITHER:
      * a fixed dict of `{"phase_noise"|"leakage"|"clutter": <dataclass, kwargs dict,
        or None>}` (or `None`/`{}` for all-default), applied identically every frame;
        or
      * a callable `chain_params(frame_index, rng) -> dict` (same value shape as
        above), invoked once per `apply()` call so the corpus stage can domain-
        randomize per frame. `frame_index` is a 0-based counter internal to this
        block (see `reset`); `rng` is a `torch.Generator` on `adc`'s device, seeded
        deterministically from `(seed, frame_index)` -- draw from it (not a fresh
        `torch.rand`) so sampling is reproducible from `seed` alone.

    PROVENANCE: whichever form produced this frame's params, the resolved, concrete
    per-stage dataclass instances (defaults filled in, not the possibly-partial
    input) are recorded into `state['impairment_params']` -- `{"phase_noise": ...,
    "leakage": ..., "clutter": ..., "seed": <int actually passed to apply_all>}` --
    so a corpus sample can always say exactly what was done to it, even when a stage
    ran with its defaults or was skipped (value `None`).

    Determinism: frame `i`'s actual seed is `seed + i` (distinct from any other
    frame's, and from the per-stage sub-seeds `apply_all` derives from it); two
    `ImpairmentBlock`s built with the same `seed` reproduce bit-identically, a
    different `seed` does not.
    """

    frame_capabilities = _RX_TIME

    def __init__(self, cfg, chain_params=None, seed=0):
        self.cfg = cfg
        self.chain_params = chain_params
        self.seed = int(seed)
        self._frame_idx = 0

    def reset(self):
        """Rewind the per-frame counter (and hence the seed sequence) to frame 0."""
        self._frame_idx = 0

    def apply(self, state):
        adc = state["adc"]
        frame_seed = self.seed + self._frame_idx
        raw_params = self.chain_params
        if callable(raw_params):
            gen = torch.Generator(device=adc.device)
            gen.manual_seed(frame_seed)
            raw_params = raw_params(self._frame_idx, gen)
        resolved = _resolve_impairment_params(raw_params)
        out = apply_all(adc, self.cfg, resolved, seed=frame_seed)
        self._frame_idx += 1
        provenance = dict(resolved)
        provenance["seed"] = frame_seed
        return {"adc": out, "impairment_params": provenance}


class QuantizerBlock:
    """ADC digitization: full-scale hard clip (saturation) + UNIFORM quantization.
    Serial stage: rewrites `adc`.

    FULL-SCALE CONVENTION: `full_scale` (default 1.0) is the clip level applied
    INDEPENDENTLY to the real and imaginary parts, in `adc`'s own units -- i.e. the
    ADC's representable range is `Re, Im in [-full_scale, full_scale]`, matching an
    IQ receiver's pair of converters. Values outside are hard-clipped before
    quantization, and that clip is the block's only saturation mechanism.

    QUANTIZATION IS UNIFORM (mid-tread), not floating-point, because that is what an
    ADC does: a `bits`-bit converter spanning +-full_scale has a constant step
    `LSB = full_scale / 2^(bits-1)`, and each sample rounds to the nearest step. The
    consequence matters for radar specifically: uniform quantization lays down a FIXED
    noise floor, so a weak target sitting far below a strong one gets buried exactly as
    it would in hardware. `afe_utils.quantizer_fp` -- the repo's other quantizer -- is
    a FLOATING-point format whose error is roughly constant in RELATIVE terms, which
    would keep weak returns artificially clean and flatter the corpus. That model is
    right for the AFE's compute datapath and wrong here; the two are deliberately
    different and must not be swapped.

    Ideal-ADC SNR follows the textbook `6.02*bits + 1.76` dB for a full-scale sine, so
    the reported figure is checkable against a hand calculation rather than only
    against itself.

    Reports, per frame: `clipped_fraction` (fraction of real/imag samples that hit the
    full-scale clip -- 0 when nothing saturates) and `quant_snr_db` (measured: power of
    the clipped-but-unquantized input over the power of the quantization error added on
    top of it).
    """

    frame_capabilities = _RX_TIME

    def __init__(self, bits=12, full_scale=1.0):
        if int(bits) < 2:
            raise ValueError(f"bits must be >= 2 (got {bits}); one bit is the sign")
        self.bits = int(bits)
        self.full_scale = float(full_scale)

    @property
    def lsb(self):
        """Quantization step: full scale divided by the codes available per sign."""
        return self.full_scale / (2 ** (self.bits - 1))

    def apply(self, state):
        adc = state["adc"]
        fs = self.full_scale
        over_re = adc.real.abs() > fs
        over_im = adc.imag.abs() > fs
        clipped_fraction = float(torch.cat([over_re.flatten(), over_im.flatten()])
                                  .float().mean().item())
        re_clip = adc.real.clamp(-fs, fs)
        im_clip = adc.imag.clamp(-fs, fs)

        # Mid-tread uniform quantization, then clamp the top code so a sample sitting
        # exactly at +full_scale does not round to a code the converter cannot output.
        lsb = self.lsb
        top = 2 ** (self.bits - 1) - 1
        re_q = torch.round(re_clip / lsb).clamp(-top - 1, top) * lsb
        im_q = torch.round(im_clip / lsb).clamp(-top - 1, top) * lsb
        out = torch.complex(re_q, im_q).to(adc.dtype)

        clipped = torch.complex(re_clip, im_clip).to(adc.dtype)
        sig_power = torch.mean(torch.abs(clipped) ** 2)
        noise_power = torch.mean(torch.abs(out - clipped) ** 2)
        eps = torch.finfo(torch.float32).tiny
        quant_snr_db = float(10.0 * torch.log10(sig_power / noise_power.clamp_min(eps)))

        return {"adc": out, "clipped_fraction": clipped_fraction, "quant_snr_db": quant_snr_db}


class RadarCubeBlock:
    """Range-Doppler radar cube -- a downstream PRODUCT block (like `FFTBlock`/
    `RangeAzBlock`): reads `adc` and emits `state['radar_cube']`, never rewriting
    `adc` itself.

    Wraps `e2e.ml.transforms.adc_to_rd`; the returned cube is complex64
    `[n_rx (or n_virtual for TDM), range_bin, doppler_bin]` with `range_bin ==
    cfg.n_samples` and `doppler_bin == cfg.n_chirps` (or `cfg.n_chirps_per_tx` after
    TDM de-interleave -- see below), matching `cfg`'s configured bin counts.

    If `cfg.mimo == 'tdm'`, `adc` is first de-interleaved into the virtual array
    (`transforms.tdm_deinterleave`) so the Doppler axis reflects the correct per-TX
    slow-time sample rate before the Doppler FFT, matching `cfg.n_chirps_per_tx`
    rather than the raw (TDM-interleaved) `cfg.n_chirps`. `transforms.py` does not
    currently offer a range-azimuth transform, so this block emits range-Doppler
    only (see module docstring).
    """

    frame_capabilities = _RX_TIME

    def __init__(self, cfg):
        self.cfg = cfg

    def apply(self, state):
        adc = state["adc"]
        if getattr(self.cfg, "mimo", None) == "tdm":
            adc = tdm_deinterleave(self.cfg, adc)
        radar_cube = adc_to_rd(self.cfg, adc)
        return {"radar_cube": radar_cube}
