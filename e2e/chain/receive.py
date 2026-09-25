"""RX-time-domain processing blocks: ADC impairments, digitization, radar-cube product.

These are the receive-side stages that pick up where a `DechirpBlock` (built
elsewhere; see `e2e/chain/__init__.py`) leaves off: `state['adc']`, a complex64
`[n_rx, n_chirp, n_samples]` cube, `state['signal_domain'] == frames.DOMAIN_RX_TIME`.
Each block below follows the same `apply(state) -> dict of state updates` protocol as
`e2e/blocks.py` and declares a `frames.FrameCapabilities` naming that domain, so
`Simulation._check_frame_contract` raises a named `FrameContractError` if one of these
runs before the chain has actually crossed into RX time (e.g. no DechirpBlock inserted).

Five blocks:

- `RangeTransformBlock` -- THE range FFT of the one chain: `adc` (RX time) ->
  `cube` (DOMAIN_CUBE), the single object every range product reads. Bridge block:
  it is what makes the spine serial instead of branched. See its docstring and
  notes/ONE_CHAIN_CONTRACT_2026-09-24.md section 1.2 row 11.
- `ImpairmentBlock`  -- wraps `e2e.chain.impairments.apply_all` (phase noise, TX/RX
  leakage, clutter). Serial stage: rewrites `adc`.
- `IFHighPassBlock`  -- the IF-chain high-pass every FMCW receiver puts between the
  mixer and the ADC. Serial stage: rewrites `adc`. Sits AFTER `ImpairmentBlock`
  (the close-in tones it exists to suppress must already be present) and BEFORE
  `QuantizerBlock` (protecting the converter's dynamic range is its job).
- `QuantizerBlock`   -- ADC digitization (full-scale clip + uniform quantization).
  Serial stage: rewrites `adc`.
- `RadarCubeBlock`   -- range-Doppler product: reads the spine's `cube` and applies
  only the Doppler half (`e2e.chain.transforms.rd_from_cube`).
  Downstream product block: reads `cube`, emits `radar_cube`, never rewrites `cube`.
"""

import dataclasses
import math

import torch

from e2e import frames
from e2e.frames import FrameCapabilities
from e2e.chain.impairments import apply_all, ClutterParams, LeakageParams, PhaseNoiseParams
from e2e.chain.transforms import rd_from_cube, tdm_deinterleave
from e2e.radar_config import C_MPS


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
    """FMCW ADC impairments (phase noise, TX/RX leakage, clutter) -- `e2e.chain.impairments
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
    "leakage": ..., "clutter": ..., "seed": <legacy per-frame seed, phase_noise/
    leakage only>, "base_seed": <the seed the persistent clutter field was drawn
    from>, "frame_idx": <this frame's 0-based index>}` -- so a corpus sample can
    always say exactly what was done to it, even when a stage ran with its defaults
    or was skipped (value `None`).

    Determinism: frame `i`'s legacy per-frame seed is `seed + i` (distinct from any
    other frame's, and from the per-stage sub-seeds `apply_all` derives from it); two
    `ImpairmentBlock`s built with the same `seed` reproduce bit-identically, a
    different `seed` does not.

    CLUTTER IS THE EXCEPTION to "new draw every frame": real ground clutter is a
    persistent scene (the road/barriers do not get redrawn each frame), so its FIELD
    (scatterer positions/velocities/gains) is drawn once from the block's base `seed`
    -- not `seed + frame_idx` -- and only evolves via a deterministic per-scatterer
    Doppler phase advance keyed to `frame_idx` (see `e2e.chain.impairments.apply_clutter`).
    Phase noise and leakage are unaffected: they still redraw every frame from
    `seed + frame_idx`, because a noisy oscillator/coupling genuinely is a new draw.
    See notes/PHYSICS_JUSTIFICATION_AUDIT.md entry 10.
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
        frame_idx = self._frame_idx
        frame_seed = self.seed + frame_idx  # legacy per-frame seed: phase_noise/leakage only
        raw_params = self.chain_params
        if callable(raw_params):
            gen = torch.Generator(device=adc.device)
            gen.manual_seed(frame_seed)
            raw_params = raw_params(frame_idx, gen)
        resolved = _resolve_impairment_params(raw_params)
        # seed=self.seed (BASE, not frame_seed): the clutter stage draws its field once
        # from this and evolves it via frame_idx instead of redrawing (see class docstring).
        out = apply_all(adc, self.cfg, resolved, seed=self.seed, frame_idx=frame_idx)
        self._frame_idx += 1
        provenance = dict(resolved)
        provenance["seed"] = frame_seed
        provenance["base_seed"] = self.seed
        provenance["frame_idx"] = frame_idx
        return {"adc": out, "impairment_params": provenance}


class IFHighPassBlock:
    """The IF high-pass filter between the FMCW mixer and the ADC. Serial stage:
    rewrites `adc`.

    WHY IT EXISTS (standard automotive receiver practice, and the audit's entry-9
    omission): after dechirp, beat frequency is proportional to range
    (`f_b = 2 * S * R / c`), so the strongest returns a monostatic radar sees -- the
    TX-RX leakage at ~0 m and the bumper/radome reflection tens of cm out -- sit at
    the very bottom of the IF band, tens of dB above every real target. Receivers
    high-pass the IF precisely to suppress them BEFORE digitization; without the
    filter those tones (a) set the ADC's full scale, wasting converter bits on a
    signal nobody wants (this chain's `QuantizerBlock` AGCs off the frame peak, so
    the coupling is modeled), and (b) lay their range sidelobes across the whole
    profile. This block is the filter.

    MODEL: an order-`order` analog Butterworth high-pass, applied as a LINEAR
    convolution along fast time. The complex analog response `H(j*2*pi*f)`
    (`|H| = 1/sqrt(1 + (fc/f)^(2*order))`, `H(0) = 0` exactly) is evaluated on a
    zero-padded DFT grid over the full `[0, fs)` beat span (the positive-exponent
    beat convention -- see `e2e.chain.rd_synth`'s derivation -- is one-sided, so the
    kernel is deliberately NOT conjugate-symmetric: bins near `fs` are far RANGES
    here, not negative frequencies, and must pass), and the record is filtered by
    zero-padded FFT multiplication -- i.e. genuine linear convolution with the
    filter's causal impulse response, NOT a per-bin weighting of the record's own
    `n_samples`-point DFT.

    WHY LINEAR AND NOT A BIN WEIGHTING (release-plan A12, found in batch review):
    a per-bin `|H(f_k)|` weighting on the `n_samples` grid is a CIRCULAR
    convolution -- it filters the record's periodic extension, whose wrap-around
    seam re-injects an off-bin tone's full spectral-leakage skirt at every range.
    Measured: a 0.25 m off-bin tone's 25-50 m skirt changed by -0.0 dB through the
    old weighting. In hardware the filter acts on the continuous tone BEFORE the
    ADC's finite record truncates it, so the tone -- and therefore the truncation
    skirt the range FFT later draws from it -- is attenuated by `|H(f_tone)|` at
    its true (continuous, off-bin) frequency. Linear convolution reproduces that;
    the decisive skirt-suppression oracle lives in tests/test_chain_receive.py.

    BAND-EDGE TAPER: the one-sided convention above puts `H(0) = 0` and
    `H(fs-) ~ 1` at the two ends of one periodic spectrum -- a discontinuity at the
    wrap, which would give the discrete kernel slowly decaying `1/n` Gibbs tails in
    BOTH time directions and let every record edge bleed junk across the whole
    window. The raised-cosine taper from `_BAND_EDGE_START * fs` to zero at
    `_BAND_EDGE_STOP * fs` exists to make the kernel's periodic spectrum continuous
    at that wrap, so its tails decay fast and edge artifacts stay local. It
    RESEMBLES the band-edge roll-off a real IF chain's band-limiting gives, but do
    not over-read that (batch physics review, 2026-08-24): this block runs after
    the modelled ADC, and 0.92/0.98 trace to kernel-locality engineering, not a
    receiver spec. REAL COST: ranges above `_BAND_EDGE_START * max_range` are
    attenuated and ranges past `~0.98 * max_range` are NULLED -- every shipped
    scene generator stays below `0.85 * max_range`, but nothing enforces that here;
    a scene reaching past ~92% of the unambiguous range loses those returns
    silently.

    SETTLING / CAUSALITY, stated honestly (corrected by the same review): the
    Butterworth factor carries its causal analog phase, but the taper is
    ZERO-PHASE, so the realized kernel is NOT strictly causal -- ~1.4% of its
    energy sits in negative time (|h[-1]| ~ -24 dB of peak), which is also why the
    record is extended with `n_samples` of edge replication on BOTH sides (`x[0]` /
    `x[-1]` held constant) before filtering, extensions discarded. The constant
    hold continues the DC leakage tone -- the strongest signal in the chain at
    +62 dB -- EXACTLY, so it is nulled with no turn-on transient (measured -79 dB
    at N=512). THE REAL RESIDUAL MECHANISM for everything else: a constant hold is
    a STEP against any nonzero-frequency component, so each return picks up
    record-edge bursts whose spectral footprint spans the whole range profile --
    measured, an on-bin PASSBAND tone (which the old per-bin weighting passed with
    exactly zero error) acquires a 25-50 m mean skirt of ~-62 dB relative to its
    own peak. This is an intra-profile dynamic-range ceiling of roughly -62 dB per
    return: invisible in every shipped regime (corpus targets sit +16..30 dB over
    the floor, so their bursts land >=30 dB UNDER it, and leakage/bumper sit
    near-DC where the hold is near-exact), but LATENT for a future very-strong
    off-DC return. The physically right extension is periodic/tone continuation of
    the beat record rather than a constant hold -- filed as release-plan A19, not
    shipped here. Kernel time-aliasing itself (the `>= 4*n_samples` grid) sits at
    -74 to -85 dB and is not the dominant term.

    CORNER: `corner_range_m` (default 1.0 m) states the corner where a spec sheet
    states it implicitly -- as the range below which returns are suppressed -- and is
    converted per config via `fc = 2 * S * corner_range_m / c`, so the SAME setting
    means the same thing on every preset regardless of slope. Pass `corner_hz` to
    override with an explicit IF frequency (then `corner_range_m` is ignored). At the
    defaults (order 2, 1.0 m): the 0 m leakage tone is nulled exactly, a 0.2 m bumper
    return takes ~28 dB of attenuation, and a target at 2 m is within ~0.3 dB of
    untouched. Those dB figures depend only on RANGE ratios, so they hold on every
    preset -- what varies with the preset is how many range BINS the suppressed
    region spans (corner range over range resolution).

    Reports, per frame: `if_hpf_corner_hz` (the resolved corner) and `if_hpf_order`.
    """

    frame_capabilities = _RX_TIME

    #: Anti-alias band-edge taper (see the class docstring): the response is unity up
    #: to `_BAND_EDGE_START * fs`, rolls off raised-cosine, and is exactly zero from
    #: `_BAND_EDGE_STOP * fs` -- which also makes the kernel's periodic spectrum
    #: continuous at the 0/fs wrap.
    _BAND_EDGE_START = 0.92
    _BAND_EDGE_STOP = 0.98

    def __init__(self, cfg, *, corner_range_m=1.0, corner_hz=None, order=2):
        if corner_hz is None:
            if float(corner_range_m) <= 0.0:
                raise ValueError(f"corner_range_m must be > 0, got {corner_range_m!r}")
            corner_hz = 2.0 * float(cfg.ramp_slope_hzps) * float(corner_range_m) / C_MPS
        if float(corner_hz) <= 0.0:
            raise ValueError(f"corner_hz must be > 0, got {corner_hz!r}")
        if int(order) < 1:
            raise ValueError(f"order must be >= 1, got {order!r}")
        self.cfg = cfg
        self.corner_hz = float(corner_hz)
        self.order = int(order)

    def analog_response(self, f):
        """Complex analog Butterworth high-pass `H(j*2*pi*f)` at frequencies `f`
        (float64 tensor, Hz; entries at exactly 0 get `H = 0`). complex128, same shape.

        Built factor-wise from the low-pass prototype poles
        `p_k = exp(j*pi*(2k + order + 1) / (2*order))` through the LP->HP mapping
        `s -> omega_c / s`, i.e. `H(j*2*pi*f) = prod_k (-p_k) / (a - p_k)` with
        `a = -j * fc / f`. Factor-wise (running product of bounded quotients) rather
        than as `(f/fc)^n / ...` powers: the power form overflows to `inf/inf = nan`
        at steep orders, and one nan bin poisons the entire output frame through the
        `ifft` (found in review of the original bin-weighting block; the regression
        test survives this rewrite). Each factor's worst case is `|a| -> inf` at
        `f -> 0`, where the quotient underflows to a clean 0.0.
        """
        h = torch.zeros(f.shape, dtype=torch.complex128, device=f.device)
        nz = f != 0.0
        a = -1j * (self.corner_hz / f[nz])
        acc = torch.ones_like(a)
        n = self.order
        for k in range(n):
            ang = math.pi * (2 * k + n + 1) / (2 * n)
            pole = complex(math.cos(ang), math.sin(ang))
            acc = acc * (-pole) / (a - pole)
        h[nz] = acc
        return h

    def _band_edge_taper(self, f):
        """Raised-cosine anti-alias taper (float64, same shape as `f`): 1 below
        `_BAND_EDGE_START * fs`, 0 from `_BAND_EDGE_STOP * fs` -- see the class
        docstring's BAND-EDGE TAPER section."""
        fs = float(self.cfg.fs_hz)
        f0, f1 = self._BAND_EDGE_START * fs, self._BAND_EDGE_STOP * fs
        t = ((f - f0) / (f1 - f0)).clamp(0.0, 1.0)
        return 0.5 * (1.0 + torch.cos(math.pi * t))

    def response(self, n_samples, device=None):
        """|H| on the `n_samples`-point fast-time DFT grid, float32 `[n_samples]`.

        Reporting/inspection view of the full response `apply` filters with
        (Butterworth high-pass x anti-alias band-edge taper; `apply` itself works on
        a finer, zero-padded grid -- see the class docstring).
        """
        f = torch.arange(int(n_samples), dtype=torch.float64, device=device) \
            * (float(self.cfg.fs_hz) / float(n_samples))
        return (self.analog_response(f).abs() * self._band_edge_taper(f)).to(torch.float32)

    def apply(self, state):
        adc = state["adc"]
        n_samples = adc.shape[-1]
        # Edge-replicated settling prefix AND suffix (see the class docstring's
        # SETTLING note), then zero-pad to >= 4*n_samples so the FFT product is a
        # LINEAR convolution with the causal impulse response; the band-edge taper
        # keeps that kernel's tails fast-decaying, so both record edges and the
        # circular wrap stay out of the kept window.
        edge_shape = (*adc.shape[:-1], n_samples)
        padded = torch.cat([adc[..., :1].expand(edge_shape), adc,
                            adc[..., -1:].expand(edge_shape)], dim=-1)
        m = 1 << max(4 * n_samples - 1, 1).bit_length()  # next pow2 >= 4*n_samples
        f = torch.arange(m, dtype=torch.float64, device=adc.device) \
            * (float(self.cfg.fs_hz) / float(m))
        h = (self.analog_response(f) * self._band_edge_taper(f)).to(adc.dtype)
        spec = torch.fft.fft(padded, n=m, dim=-1)
        out = torch.fft.ifft(spec * h, dim=-1)[..., n_samples:2 * n_samples].to(adc.dtype)
        return {"adc": out, "if_hpf_corner_hz": self.corner_hz,
                "if_hpf_order": self.order}


class QuantizerBlock:
    """ADC digitization: full-scale hard clip (saturation) + UNIFORM quantization.
    Serial stage: rewrites `adc`.

    FULL-SCALE CONVENTION: `full_scale` is the clip level applied
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

    **`quant_snr_db` IS NOT A CONVERTER-QUALITY SCORE, AND MUST BE READ NEXT TO
    `clipped_fraction`.** It is the SNR of the QUANTIZER, referenced to the signal that
    survives clipping -- clipping distortion is deliberately not in its numerator or its
    denominator, because that is reported separately. The consequence is that the number
    goes UP as the converter is driven harder into saturation: measured 2026-08-28, a
    frame driven to `clipped_fraction = 1.0000` (every single sample railed) still reports
    **69.28 dB**, because what is left after clipping is a near-constant envelope that
    quantizes beautifully. A reader who takes `quant_snr_db` alone as "how good is the
    ADC" gets exactly the wrong answer in the regime that matters. Neither field is wrong;
    the pair is the measurement.

    `quant_snr_db` is `None` -- not `-inf` -- when the input carries no signal at all, so
    the value stays JSON-serializable. `float("-inf")` serializes as the bare token
    `-Infinity`, which `json.dumps` emits happily and a STRICT JSON parser rejects, so an
    all-zero frame used to be able to poison a whole results file.

    FULL SCALE DEFAULTS TO AUTOMATIC GAIN, and that default matters more than it looks.
    Ray-traced cubes carry PHYSICAL amplitudes -- `rt_gen` deliberately does not
    normalize -- and at 77 GHz over tens of metres a return lands around 1e-7..1e-6,
    while a 12-bit converter spanning +-1.0 has an LSB near 2.4e-4. A fixed
    `full_scale=1.0` therefore quantizes a realistic cube to EXACTLY ZERO, silently,
    with `clipped_fraction` reporting 0.0 as if all were well. An adversarial review
    caught this. Passing `full_scale=None` (the default) instead sets the range from the
    frame's own peak with `headroom_db` of margin -- a crude AGC, which is what real
    receivers use for the same reason. Pass a float only when the absolute scale is
    genuinely known and fixed.
    """

    frame_capabilities = _RX_TIME

    def __init__(self, bits=12, full_scale=None, headroom_db=6.0):
        if int(bits) < 2:
            raise ValueError(f"bits must be >= 2 (got {bits}); one bit is the sign")
        self.bits = int(bits)
        self.full_scale = None if full_scale is None else float(full_scale)
        self.headroom_db = float(headroom_db)
        self._last_full_scale = self.full_scale

    def _resolve_full_scale(self, adc):
        """The converter's range for this frame: fixed if given, else AGC from the peak."""
        if self.full_scale is not None:
            return self.full_scale
        peak = torch.max(torch.stack([adc.real.abs().max(), adc.imag.abs().max()]))
        peak = float(peak.item())
        if peak <= 0.0:
            return 1.0
        return peak * (10.0 ** (self.headroom_db / 20.0))

    @property
    def lsb(self):
        """Quantization step of the LAST frame: full scale over the codes per sign."""
        fs = self._last_full_scale if self._last_full_scale is not None else 1.0
        return fs / (2 ** (self.bits - 1))

    def apply(self, state):
        adc = state["adc"]
        fs = self._resolve_full_scale(adc)
        self._last_full_scale = fs
        over_re = adc.real.abs() > fs
        over_im = adc.imag.abs() > fs
        clipped_fraction = float(torch.cat([over_re.flatten(), over_im.flatten()])
                                  .float().mean().item())
        re_clip = adc.real.clamp(-fs, fs)
        im_clip = adc.imag.clamp(-fs, fs)

        # Mid-tread uniform quantization, then clamp the top code so a sample sitting
        # exactly at +full_scale does not round to a code the converter cannot output.
        lsb = fs / (2 ** (self.bits - 1))
        top = 2 ** (self.bits - 1) - 1
        re_q = torch.round(re_clip / lsb).clamp(-top - 1, top) * lsb
        im_q = torch.round(im_clip / lsb).clamp(-top - 1, top) * lsb
        out = torch.complex(re_q, im_q).to(adc.dtype)

        clipped = torch.complex(re_clip, im_clip).to(adc.dtype)
        sig_power = torch.mean(torch.abs(clipped) ** 2)
        noise_power = torch.mean(torch.abs(out - clipped) ** 2)
        eps = torch.finfo(torch.float32).tiny
        if float(sig_power) <= 0.0:
            # No signal at all: quantization SNR is undefined, not "-inf". Reporting
            # -inf serializes as the bare token -Infinity, which json.dumps writes and
            # a strict parser rejects -- one all-zero frame could invalidate an entire
            # results file. None serializes as null.
            quant_snr_db = None
        else:
            quant_snr_db = float(10.0 * torch.log10(sig_power / noise_power.clamp_min(eps)))

        return {"adc": out, "clipped_fraction": clipped_fraction,
                "quant_snr_db": quant_snr_db, "adc_full_scale": fs}


# ================================================================================
# The range transform -- THE bridge from RX time to the one range-compressed cube.
# ================================================================================
# Every product downstream of it (range-azimuth / range-elevation / az-el / range
# profile / the subspace tracker's snapshots) reads THIS cube. Before the one-chain
# consolidation each of those products ran its own range FFT over the frequency axis
# of `s_pars` (`e2e/blocks.py` at HEAD c33bb64, lines 1090/1164/1204) while the ML
# chain ran a second, differently-conventioned one inside `transforms.adc_to_rd` --
# the two pipelines the owner's 2026-09-24 directive retires. See
# notes/ONE_CHAIN_CONTRACT_2026-09-24.md sections 1.2 (row 11) and 5.1.

#: The two metre conventions the stored bistatic munich frames admit. They are not a
#: labelling choice: the frames are a TX->RX-array link with `normalize_delays=True`,
#: so a bin's delay tau is an EXCESS delay over the line of sight, and the metre it is
#: quoted in changes every number on every card (contract section 3.6, ballot Q2).
#:   "bistatic_path"  -- c*tau, the excess PATH LENGTH. THE DEFAULT since the owner's
#:                       2026-09-24 ballot answer ("if bistatic, math should be
#:                       correct"): the munich link is TX at [8.5,21,27] to an RX array
#:                       at [45,90,1.5], ~82 m apart, traced with
#:                       `normalize_delays=True`, so a bin's delay is an EXCESS delay
#:                       over the line of sight and c*tau is exactly what it measures.
#:                       On the munich Ka plan: 9.99 cm per bin, a 499.55 m window.
#:   "monostatic_c2"  -- c*tau/2, the range a MONOSTATIC radar would report for the same
#:                       delay. The v1.0 cards' convention, kept as the alternative
#:                       (5 cm/bin, 249.78 m) -- it is neither a range nor a path length
#:                       for this geometry, which is why it is no longer the default.
#: Switching convention is a single scalar on every metre the chain quotes; nothing
#: about the cube itself changes, which is why the axis is computed once, here.
RANGE_CONVENTIONS = ("bistatic_path", "monostatic_c2")

#: Metres per second of delay, per convention.
_CONVENTION_SCALE = {"monostatic_c2": C_MPS / 2.0, "bistatic_path": C_MPS}


def delta_f_from_freq_plan(freq_plan):
    """Frequency-grid spacing (Hz) of a stored v2 CFR, from its `freq_plan` meta.

    `e2e/environment/sionna_simple_channel.py:build_frequencies` is
    `np.linspace(start, stop, num_freqs)` -- ENDPOINT-INCLUSIVE -- so the spacing is
    `(stop - start) / (num_freqs - 1)`, NOT `B / num_freqs`. For the shipped munich Ka
    file (28.5-31.5 GHz, 5000 points) that is 600_120.02 Hz against 600_000.0 Hz: a
    0.02% error, worth ~0.5 m at the far end of a 2500-bin window. Measured
    2026-09-24 against the pickle header; holds for any v2 file written by that script.

    Returns None when `freq_plan` is absent or unusable (legacy pkls), which is the
    caller's cue that the cube has no metre axis -- bins only.
    """
    if not freq_plan:
        return None
    try:
        start = float(freq_plan["start_hz"])
        stop = float(freq_plan["stop_hz"])
        num = int(freq_plan["num_freqs"])
    except (KeyError, TypeError, ValueError):
        return None
    if num < 2 or not stop > start:
        return None
    return (stop - start) / (num - 1)


def delta_f_from_cfg(cfg):
    """Frequency-grid spacing (Hz) of an FMCW beat record described by `cfg`.

    `e2e/environment/rt_signal_chain.py:beat_frequencies` samples at `f0 + S*n/fs` for
    `n < n_samples` -- NOT endpoint-inclusive -- so the spacing is `S/fs`, i.e.
    `bandwidth_hz / n_samples`. This is deliberately a DIFFERENT formula from
    `delta_f_from_freq_plan`'s, because the two grids really are built differently;
    the known-delay oracle in tests/test_one_chain_spine.py checks both.
    """
    if cfg is None:
        return None
    try:
        return float(cfg.bandwidth_hz) / float(cfg.n_samples)
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        return None


def range_axis_m(n_bins, delta_f_hz, n_fft, convention="bistatic_path"):
    """Physical range (m) of cube bins `0 .. n_bins-1` as a float64 tensor, or None
    when the chain carries no frequency plan to calibrate against.

    Bin `k` of a forward FFT of `n_fft` beat samples spaced `delta_f_hz` apart is delay
    `tau_k = k / (n_fft * delta_f_hz)`; the metre is `tau_k` times
    `_CONVENTION_SCALE[convention]`. NOTE it is written against `n_fft * delta_f_hz`
    and never against a nominal bandwidth `B`: for an endpoint-inclusive grid those two
    differ by `N/(N-1)`, and baking `B` in is exactly the off-by-one the contract
    (section 3.6) says the oracle must not be able to hide.

    `n_bins` may be smaller than `n_fft` (a cropped cube); the calibration of bin `k`
    does not depend on how many bins were kept.
    """
    if convention not in RANGE_CONVENTIONS:
        raise ValueError(
            f"unknown range convention {convention!r}; expected one of {RANGE_CONVENTIONS}"
        )
    if delta_f_hz is None or not float(delta_f_hz) > 0:
        return None
    step = _CONVENTION_SCALE[convention] / (float(n_fft) * float(delta_f_hz))
    return torch.arange(int(n_bins), dtype=torch.float64) * step


class RangeTransformBlock:
    """RX-time beat samples -> THE range-compressed cube. The one range FFT in the
    simulator.

    `adc [n_rx, n_chirp, n_samples]` (DOMAIN_RX_TIME) in; `cube [n_rx, n_chirp,
    n_range]` (DOMAIN_CUBE) out, plus `range_axis` (metres, or None when the chain
    carries no frequency plan), `range_axis_m_per_bin`, `cube_axes` and
    `range_convention`.

    Chain, in order -- the range half of `transforms.adc_to_rd`, which keeps its own
    copy while `e2e/chain/transforms.py` is under the F84 training freeze; the parity
    is asserted by `tests/test_one_chain_spine.py::test_adc_to_rd_range_half_parity`,
    and the refactor that makes THIS the single implementation is the follow-on:

      1. optional per-(rx, chirp) DC removal (mean over fast time);
      2. optional window on the fast axis;
      3. forward FFT along fast time. NO fftshift, NO negation.

    Conventions, each of which used to live in a display helper and now lives here:

    * **Bin k is delay `k / (n_samples * delta_f)`, k in [0, n_samples).** With IQ
      sampling the whole FFT period is physical delay (F96); there is no negative-delay
      half to fold away. What the v1.0 SCREENS showed as "negative range" was an
      artifact of fftshifting and then negating an axis with no negative half -- the
      upper bins are real delay, cropped (contract section 3.6).
    * **`crop_negative_delay`** (default True) keeps bins `0 .. n_samples//2`, which is
      EXACTLY the half the v1.0 display kept (`webapp/pipeline_runner.py:_range_axis` /
      `_nonnegative_range` at HEAD c33bb64: gates with `axis >= 0` are the gates at or
      below `zero_gate`, which map back to native bins `0 .. n_freqs//2`). Default True
      so nothing on screen moves without a decision; set False for the full unambiguous
      window (250 m on munich Ka under c/2).
    * **`dc_removal`** defaults to True, matching `adc_to_rd`'s scored ML protocol.
      The imaging spine passes False: the munich trace is generated with
      `normalize_delays=True`, so the line-of-sight path sits AT bin 0 and removing the
      mean would delete the brightest feature on the screen. (Subtracting the fast-time
      mean zeroes FFT bin 0 exactly and leaves every other bin untouched -- so this flag
      is a one-bin decision, not a filter.)
    * **`window`** defaults to 'hann' (the scored protocol); the imaging/tracker presets
      pass 'none', which is the point at which this block is the identity with the v1.0
      products (contract section 5.4).

    `delta_f_hz` resolution order: the explicit kwarg, else `state['freq_plan']`
    (endpoint-inclusive: `delta_f_from_freq_plan`), else `cfg` (`delta_f_from_cfg`).
    If none of them answer, `range_axis` is None and the cube is calibrated in bins --
    said out loud rather than guessed.
    """

    frame_capabilities = FrameCapabilities(
        domain=frames.DOMAIN_RX_TIME, emits_domain=frames.DOMAIN_CUBE,
        accepts_mimo=True, chirps=frames.CHIRP_NATIVE,
    )

    #: Windows this block accepts on the FAST (range) axis. 'none' is the v1.0 identity
    #: point; 'hann' is what `adc_to_rd` applies and what the ML corpora were scored on.
    WINDOWS = ("none", "hann", "hamming")

    def __init__(self, cfg=None, *, window="hann", dc_removal=True,
                 convention="bistatic_path", crop_negative_delay=True,
                 delta_f_hz=None):
        window = "none" if window is None else str(window)
        if window not in self.WINDOWS:
            raise ValueError(
                f"unknown range window {window!r}; expected one of {self.WINDOWS}"
            )
        if convention not in RANGE_CONVENTIONS:
            raise ValueError(
                f"unknown range convention {convention!r}; expected one of "
                f"{RANGE_CONVENTIONS}"
            )
        self.cfg = cfg
        self.window = window
        self.dc_removal = bool(dc_removal)
        self.convention = convention
        self.crop_negative_delay = bool(crop_negative_delay)
        self.delta_f_hz = None if delta_f_hz is None else float(delta_f_hz)

    def _fast_window(self, n, device):
        if self.window == "none":
            return None
        if self.window == "hann":
            return torch.hann_window(n, periodic=False, dtype=torch.float32, device=device)
        return torch.hamming_window(n, periodic=False, dtype=torch.float32, device=device)

    def resolve_delta_f(self, state=None):
        """The grid spacing this block calibrates with, or None. Public so a caller
        (a webapp panel, a card test) can read the SAME number the cube was built with
        instead of re-deriving it from a literal."""
        if self.delta_f_hz is not None:
            return self.delta_f_hz
        plan = (state or {}).get("freq_plan")
        df = delta_f_from_freq_plan(plan)
        if df is not None:
            return df
        return delta_f_from_cfg(self.cfg)

    def apply(self, state):
        adc = state["adc"]
        if adc.dim() != 3:
            raise frames.FrameContractError(
                f"RangeTransformBlock expects adc [n_rx, n_chirp, n_samples], got "
                f"shape {tuple(adc.shape)}"
            )
        n_samples = adc.shape[-1]
        x = adc
        if self.dc_removal:
            x = x - x.mean(dim=-1, keepdim=True)
        w = self._fast_window(n_samples, x.device)
        if w is not None:
            x = x * w.to(x.dtype)
        cube = torch.fft.fft(x, n=n_samples, dim=-1)
        n_keep = n_samples // 2 + 1 if self.crop_negative_delay else n_samples
        cube = cube[..., :n_keep].contiguous().to(torch.complex64)

        delta_f = self.resolve_delta_f(state)
        axis = range_axis_m(n_keep, delta_f, n_samples, self.convention)
        per_bin = float(axis[1] - axis[0]) if axis is not None and n_keep > 1 else None
        # The fast axis is range bins BY CONSTRUCTION -- that is what this block does --
        # but WHAT THE SLOW AXIS COUNTS is the waveform's business, not this block's, so
        # it is carried through from whatever mixing block ran upstream (chirps after a
        # dechirp, symbols after a symbol division). Hardcoding "chirp" here would have
        # made every JSAC cube claim to be a chirp cube and `RadarCubeBlock` would have
        # computed a Doppler transform over OFDM symbols without complaint.
        slow = (state.get("cube_axes") or frames.CUBE_AXES_FMCW).get("slow", "chirp")
        return {
            "cube": cube,
            "signal_domain": frames.DOMAIN_CUBE,
            "cube_axes": {"slow": slow, "fast": "range_bin"},
            "range_axis": axis,
            "range_axis_m_per_bin": per_bin,
            "range_convention": self.convention,
            "range_n_fft": n_samples,
            "range_delta_f_hz": delta_f,
            "range_cropped": self.crop_negative_delay,
            # The protocol the cube was built under, travelling in state. Added
            # 2026-09-24 (shard 2) so a downstream product that is only valid under
            # ONE protocol can refuse the others BY NAME instead of computing a
            # plausible wrong answer: `RadarCubeBlock`/`transforms.adc_to_rd` are the
            # scored ML protocol (hann / DC removal / uncropped, F85 and F95 read
            # bit-parity against corpora made that way), and the imaging spine runs
            # the identity point (`none` / no DC removal) instead.
            "range_window": self.window,
            "range_dc_removal": self.dc_removal,
        }


class RadarCubeBlock:
    """Range-Doppler radar cube -- a downstream PRODUCT block (like `FFTBlock`/
    `RangeAzBlock`): reads `cube` and emits `state['radar_cube']`, never rewriting
    `cube` itself.

    Consumes the spine's range-compressed `cube` (`RangeTransformBlock`, the ONE range
    FFT -- contract section 1.2 row 11) and applies `transforms.rd_from_cube`, the
    Doppler half of `adc_to_rd`. It no longer computes a range FFT of its own; the
    returned cube is complex64
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

    frame_capabilities = FrameCapabilities(
        domain=frames.DOMAIN_CUBE, accepts_mimo=True, chirps=frames.CHIRP_NATIVE,
    )

    def __init__(self, cfg):
        self.cfg = cfg

    def _check_protocol(self, state):
        """Refuse a cube that was not range-compressed on the SCORED protocol.

        This block used to read `adc` and run `transforms.adc_to_rd` itself -- a
        second range FFT, which is exactly what the one-chain contract deletes. Now
        it reads the spine's `cube` and applies only the Doppler half, which makes
        the range protocol somebody else's decision and therefore something this
        block has to check: `RangeTransformBlock`'s own defaults match (hann, DC
        removal), but the IMAGING spine deliberately builds the identity point
        (`window="none"`, `dc_removal=False`) and crops to the non-negative half.
        Handing that cube to the Doppler half produces a range-Doppler map on a
        different, unlabelled protocol -- half the range bins and no window -- which
        F85/F95's stored-vs-live gates would read as a detector difference.

        Missing keys mean the cube did not come from a `RangeTransformBlock` (a test
        parking one in state by hand); that is allowed and unchecked.
        """
        from e2e.chain.transforms import RD_RANGE_PROTOCOL
        want = RD_RANGE_PROTOCOL
        got = {"window": state.get("range_window"),
               "dc_removal": state.get("range_dc_removal"),
               "crop_negative_delay": state.get("range_cropped")}
        if all(v is None for v in got.values()):
            return
        bad = {k: got[k] for k in want if got[k] is not None and got[k] != want[k]}
        if bad:
            raise frames.FrameContractError(
                f"{frames.component_name(self)} consumes the SCORED range protocol "
                f"{want} (it applies only the Doppler half of "
                f"`transforms.adc_to_rd`), but the chain's cube was built with "
                f"{bad}. Build the chain's RangeTransformBlock with "
                f"`**transforms.RD_RANGE_PROTOCOL` -- or, for the imaging identity "
                f"point, read a range-angle product instead of range-Doppler."
            )

    def apply(self, state):
        self._check_protocol(state)
        frames.require_cube_axes(state, self, slow="chirp", fast="range_bin")
        cube = state["cube"]
        cfg = self.cfg
        if getattr(cfg, "mimo", None) == "tdm":
            # The de-interleave is a pure reorganisation of the SLOW axis, so it
            # commutes exactly with the fast-axis range FFT that has already run:
            # applying it to the cube gives the same tensor as applying it to `adc`
            # and range-compressing afterwards.
            cube = tdm_deinterleave(cfg, cube)
            # De-interleaving consumes the transmit multiplexing: the cube now has one
            # virtual array and cfg.n_chirps_per_tx slow-time samples. Handing the
            # ORIGINAL cfg to rd_from_cube then fails its own shape check, which blocked
            # the ti_iwr1443 preset outright. Describe
            # the de-interleaved cube instead. Same pattern dataset.py already uses.
            cfg = dataclasses.replace(
                cfg, n_tx=1, mimo="single", n_chirps=cfg.n_chirps_per_tx
            )
        radar_cube = rd_from_cube(cfg, cube)
        return {"radar_cube": radar_cube}
