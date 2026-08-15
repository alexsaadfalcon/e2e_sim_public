"""
Ray-traced (Sionna RT) path-to-signal physics for `e2e.ml.rt_gen`.

Turns a solved Sionna `Paths` object (see `e2e.ml.rt_scene_build.build_rt_scene`) into
the **same** dechirped ADC cube -- `complex64 [n_rx, n_chirps, n_samples]` -- that
`e2e.ml.rd_synth` produces from its closed-form point-target model, so
`e2e.ml.transforms`, `e2e.ml.labels` and `e2e.ml.dataset` cannot tell which generator
produced a frame. Split out of the original `rt_gen.py`; scene/mesh/asset construction
lives in `e2e.ml.rt_scene_build`, the native-vs-re-trace experiment harness + CLI in
`e2e.ml.rt_doppler_study`. `e2e.ml.rt_gen` re-exports this module's public and private
names for backward compatibility.

Sionna is imported lazily (inside functions), so `import e2e.ml.rt_signal_chain` works
on a machine without Sionna/DrJit -- only the generation calls need it.


The CFR -> beat mapping (the load-bearing derivation)
----------------------------------------------------
An FMCW transmitter emits ``s_t(t) = exp(j2pi(f0 t + S t^2/2))`` (slope
``S = B / T_sweep``). The echo from a scatterer at round-trip delay ``tau`` is
``s_t(t - tau)``, and the receiver *dechirps* it against the transmitted ramp. This
package's convention (see `rd_synth`'s module docstring) is the **positive-exponent**
one, ``s_b(t) = s_t(t) conj(s_t(t-tau))``:

    s_b(t) = exp(j2pi[ f0 tau + S tau t - S tau^2 / 2 ])

Sampling the ADC at ``t = n / fs`` and dropping the residual video phase
``-pi S tau^2`` (rd_synth drops it too, `include_rvp=False`) gives

    b[n] = exp(+j2pi (f0 + S n/fs) tau)                                        (1)

i.e. **the dechirped sample n is the channel evaluated at the instantaneous RF
frequency of the ramp at that sample**, ``f_RF(n) = f0 + S n / fs``.

Sionna's `Paths.cfr(frequencies=f, ...)` returns (paths.py, `cir`/`cfr` docstrings)

    h(f, t) = sum_i a_i exp(-j2pi f_c tau_i) exp(+j2pi f_D,i t) exp(-j2pi f tau_i)
            = sum_i a_i exp(-j2pi (f_c + f) tau_i) exp(+j2pi f_D,i t)          (2)

where ``f_c = scene.frequency`` (the carrier) and ``f`` is a **baseband offset** from
it (Sionna's OFDM helper `subcarrier_frequencies` returns offsets centred on 0).
Comparing (1) and (2): a single path contributes ``exp(-j2pi f_RF tau)`` where (1)
wants ``exp(+j2pi f_RF tau)``. So the beat cube is the **complex conjugate** of the
CFR sampled on the ramp:

    b[c, n] = conj( h( f_bb(n), t = c * T_c ) ),   f_bb(n) = f0 + S n/fs - f_c   (3)

with ``f_c = f0 + B/2`` (the chirp centre; chosen so Sionna's array-element spacing,
which is expressed in wavelengths of ``scene.frequency``, is exactly the
``cfg.wavelength_m / 2`` that `rd_synth` assumes), ``sampling_frequency = 1/T_c`` and
``num_time_steps = n_chirps`` -- Sionna's slow-time axis IS the chirp axis.

The same conjugation fixes the Doppler sign automatically, which is why (3) is
stated as one operation rather than three. Sionna's ``f_D`` is the *physical* Doppler
shift (positive for an approaching target: for a monostatic link and a target with
radial velocity ``v_r``, receding-positive, ``f_D = -2 v_r / lambda``). Conjugating
(2) turns ``exp(+j2pi f_D t)`` into ``exp(+j2pi (2 v_r/lambda) t)`` -- exactly
rd_synth's chirp-to-chirp phase progression (its beat phase ``2pi f0 tau_c`` with
``tau_c = 2(R0 + v_r c T_c)/c`` advances by ``2pi (2 v_r/lambda) T_c`` per chirp).

**Element ordering / array handedness.** With the same conjugation, an element
displaced by ``d`` *towards* the target sees a shorter delay and therefore a beat
phase ``-2pi d sin(theta)/lambda``. rd_synth uses ``+pi * v * sin(theta)`` for
virtual element ``v`` (see its "Spatial phase" comment), with ``sin(theta)`` measured
along ``u = normalise(z_up x boresight)``. The two agree exactly if the element
*index* runs along ``-u`` -- i.e. this package numbers array elements from the
positive-azimuth side towards the negative one. That is a labelling (handedness)
convention, not a physical difference, and we honour it here by **reversing the
antenna index** of the extracted CFR (`_ANTENNA_INDEX_REVERSED`). Reversing the index
of a `PlanarArray` is exact, not approximate: its normalized positions are symmetric
about the array centre (`antenna_array.py`: ``y = d_h*j - (num_cols-1)*d_h/2``), so
index reversal is a mirror about that centre. Verified empirically: without the
reversal a target at ``sin(az) = +0.37`` lands at the mirrored angle-FFT bin.

Deliberate approximations (all shared with, or milder than, rd_synth)
---------------------------------------------------------------------
* **Native Doppler evolution**: one `PathSolver` solve per frame; the geometry
  (delays, angles, amplitudes) is frozen over the CPI and only the per-path Doppler
  phase evolves across chirps. This is Sionna's own ``||v|| << c`` model and the
  classical stop-and-hop assumption. `rt_retrace_reference` re-solves the geometry
  per chirp and `doppler_error_study` quantifies the difference -- run it rather than
  trusting this paragraph.
* **Intra-frame range migration** is absent from the native path (the beat frequency
  is not re-derived per chirp), same order of magnitude as rd_synth's constant-
  amplitude approximation: a 5 m/s target moves 2.4 cm over a 4.8 ms TI CPI, ~0.3 of
  a 7.5 cm range bin.
* **float32 delay phase**: DrJit computes ``2pi f_c tau`` in float32 before wrapping;
  at 78 GHz and 20 m that is ~6.5e4 rad, so the wrapped phase carries ~4e-3 rad of
  rounding noise (~ -48 dBc). Irrelevant for bin-level validation, relevant if you
  ever chase >45 dB phase-coherence numbers out of this path.
* **Ranges are to the reflecting surface**, not to an object's centre: RT reflects
  off real geometry, so a 1 m-radius sphere at 5.4 m peaks ~1 m closer than the
  point-target model would predict. Use small scatterers when comparing bin-for-bin
  against `rd_synth`.
* **TX/RX leakage**: a monostatic link's direct TX-array -> RX-array path is a real
  (huge, near-zero-delay) path. `include_leakage=False` (the default) asks the solver
  for ``los=False``, which removes exactly that path and nothing else -- every target
  return is a reflection of depth >= 1.
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional

import numpy as np
import torch

from e2e.ml.rt_scene_build import RTScene, build_rt_scene

# See the module docstring's "Element ordering / array handedness" section.
_ANTENNA_INDEX_REVERSED = True


def _resolve_device(dev):
    """`None` -> the library device; anything else -> `torch.device(dev)`."""
    if dev is None:
        from e2e.ml.rd_synth import device as _lib_device

        return _lib_device
    return torch.device(dev)


# --------------------------------------------------------------------------------
# Solve + CFR -> beat cube
# --------------------------------------------------------------------------------
#: Speed of light, m/s -- fixes the range/delay/Doppler conversions below.
_C_LIGHT_MPS = 299792458.0


def doppler_validity(cfg, radial_speed_mps: float, *, rel_rmse_target: float = 0.05) -> dict:
    """How many chirps the ONE-SOLVE Doppler model is good for at a given target speed.

    The generator ray-traces once per frame and evolves slow time with Sionna's
    first-order per-path Doppler phase (see `rt_cfr_frame`). That gets the CARRIER phase
    advance right but holds each path's delay `tau_0` fixed inside the BASEBAND term, so
    intra-frame range migration is missing. The residual phase error at chirp `c` is
    ~`2*pi*f_baseband*dtau(c)` with `dtau(c) = 2*v_r*c*T_c / c_light`, giving an RMS
    relative error that grows LINEARLY in chirp index at a (dimensionless) rate

        slope = 2*pi*B*v_r*T_c / (sqrt(3)*c_light)   per chirp

    and therefore a usable chirp count `N(eps) = eps / slope`, i.e.

        N(eps) = eps*sqrt(3)*c_light / (2*pi*B*v_r*T_c).

    MEASURED 2026-08-11 on a stable-path-set scene (single planar target, free space,
    max_depth=1, specular only, so the re-trace path set provably cannot change): the
    per-chirp error is smooth and monotonic with a fitted power-law exponent of
    0.93 +/- 0.07 (R^2 = 0.998, i.e. linear), and the measured slope agreed with the
    formula above to 2.3%. The erratic, non-monotonic curves seen on sphere targets are
    a RE-TRACE artifact -- a triangulated sphere's specular return flickers as facets
    turn under sub-millimetre motion, and Sionna marks the path invalid -- not evidence
    about this model. Use a planar target when measuring Doppler fidelity.

    Returns the slope, the usable chirp count, the config's own `n_chirps`, and whether
    the frame stays inside `rel_rmse_target`. Cheap and Sionna-free: call it before
    generating a corpus rather than discovering the problem in the data.
    """
    v = abs(float(radial_speed_mps))
    b = float(cfg.bandwidth_hz)
    t_c = float(cfg.chirp_period_s)
    eps = float(rel_rmse_target)
    # Per-chirp slope is DIMENSIONLESS: the delay advances by dtau = 2*v*T_c/c each
    # chirp, and the RMS of 2*pi*f_bb*dtau over a uniform baseband spanning +-B/2
    # contributes the 1/sqrt(3). The T_c belongs here -- an earlier version omitted it
    # and reported a slope in Hz, ~1/T_c (13158x) too large.
    slope = 2.0 * math.pi * b * v * t_c / (math.sqrt(3.0) * _C_LIGHT_MPS)
    # usable_chirps is just eps/slope; deriving it rather than repeating the algebra
    # keeps the two from drifting apart (they had).
    n_ok = float("inf") if slope == 0.0 else eps / slope
    return {
        "radial_speed_mps": v,
        "rel_rmse_target": eps,
        "rel_rmse_slope_per_chirp": slope,
        "usable_chirps": n_ok,
        "n_chirps": int(cfg.n_chirps),
        "within_target": n_ok >= float(cfg.n_chirps),
        "rel_rmse_at_frame_end": slope * float(cfg.n_chirps),
    }


def warn_if_doppler_invalid(cfg, radial_speed_mps: float, *, rel_rmse_target: float = 0.05):
    """Emit a UserWarning when `doppler_validity` says the frame outruns the model.

    Deliberately a warning and not an error: the approximation is still the right
    default (it is ~40x cheaper than re-tracing per chirp), and plenty of useful
    scenarios sit inside it. What is not acceptable is generating a corpus that silently
    violates it -- at `radial_like`'s 749.5 MHz bandwidth a 20 m/s closing target
    outruns 5% RMS in under 4 chirps of a 252-chirp frame.
    """
    v = doppler_validity(cfg, radial_speed_mps, rel_rmse_target=rel_rmse_target)
    if not v["within_target"]:
        warnings.warn(
            f"one-solve Doppler model: at {v['radial_speed_mps']:.1f} m/s radial speed "
            f"this config is good for ~{v['usable_chirps']:.0f} chirps at "
            f"{v['rel_rmse_target']:.0%} RMS, but the frame has {v['n_chirps']}. "
            f"Intra-frame range migration is not modelled; expect range-Doppler peak "
            f"error. Shorten the frame, slow the targets, or re-trace per chirp "
            f"(see doppler_error_study).",
            UserWarning,
            stacklevel=2,
        )
    return v


def beat_frequencies(cfg) -> np.ndarray:
    """Baseband CFR frequencies for one chirp: `f0 + S n/fs - (f0 + B/2)`, n < n_samples.

    See the module docstring, equation (3): sampling the CFR on this grid IS sampling
    the dechirped beat signal along the ramp.
    """
    n = np.arange(int(cfg.n_samples), dtype=np.float64)
    f_rf = float(cfg.f0_hz) + float(cfg.ramp_slope_hzps) * n / float(cfg.fs_hz)
    return f_rf - (float(cfg.f0_hz) + float(cfg.bandwidth_hz) / 2.0)


def _solve(rt_scene: RTScene, *, max_depth: int, include_leakage: bool,
           diffuse_reflection: bool, specular_reflection: bool, refraction: bool,
           seed: int, samples_per_src: Optional[int] = None):
    import sionna.rt as rt

    if rt_scene.solver is None:
        rt_scene.solver = rt.PathSolver()
    kwargs = dict(
        scene=rt_scene.scene, max_depth=int(max_depth),
        los=bool(include_leakage),           # the ONLY los path here is TX->RX leakage
        specular_reflection=bool(specular_reflection),
        diffuse_reflection=bool(diffuse_reflection),
        refraction=bool(refraction),
        synthetic_array=False, seed=int(seed),
    )
    if samples_per_src is not None:
        kwargs["samples_per_src"] = int(samples_per_src)
    return rt_scene.solver(**kwargs)


#: DrJit refuses to allocate an array with more than 2**32 entries. `cfr` materialises
#: [rx, rx_ant, tx, tx_ant, num_paths, n_chirps, n_freqs] BEFORE summing over paths, so
#: the safe frequency-chunk size depends on how many paths the solve actually found --
#: which a fixed default cannot know. Real decimated vehicle meshes with diffuse
#: scattering produce ~15k paths where the old sphere scenes produced tens, and the
#: fixed 128-frequency chunk then asked for 4.4e9 entries and failed outright. Budget
#: at half the hard limit so the peak allocation has room around it.
_DRJIT_ELEMENT_BUDGET = 2 ** 31


def _num_paths(paths) -> int:
    """Path count of a solved `Paths`, or 0 if it cannot be determined cheaply."""
    for attr in ("a", "tau"):
        arr = getattr(paths, attr, None)
        if arr is None:
            continue
        shape = getattr(arr, "shape", None)
        if shape:
            # Path axis is the last one for tau ([..., num_paths]) and second-to-last
            # for a; taking the max is a safe over-estimate for budgeting purposes.
            return int(max(shape))
    return 0


def _cfr_freq_chunk(paths, cfg, *, n_chirps: int, requested: int) -> int:
    """Largest frequency chunk that keeps `cfr`'s pre-sum tensor inside DrJit's limit.

    Returns `requested` when the solve is small enough to need no reduction, so ordinary
    scenes keep their previous behaviour (and previous numbers) exactly.
    """
    requested = max(1, int(requested))
    n_paths = _num_paths(paths)
    if n_paths <= 0:
        return requested
    per_freq = max(1, int(cfg.n_rx) * int(cfg.n_tx) * n_paths * int(n_chirps))
    allowed = max(1, _DRJIT_ELEMENT_BUDGET // per_freq)
    if allowed >= requested:
        return requested
    if allowed < 1:  # pragma: no cover -- would need ~1e6 paths
        raise RuntimeError(
            f"a single frequency bin needs {per_freq} elements, over DrJit's limit; "
            f"the solve found {n_paths} paths. Reduce max_depth or scene complexity."
        )
    return int(allowed)


def cfr_sum_over_paths(a, tau, doppler, freqs, *, f_c: float, chirp_period_s: float,
                       n_chirps: int, range_migration: bool = False) -> np.ndarray:
    """Closed-form CFR from per-path `(a, tau, doppler)`. Pure numpy, no Sionna.

    Reproduces what `Paths.cfr` computes:

        h(f, t) = sum_i a_i * exp(-j2pi (f_c + f) tau_i) * exp(j2pi f_D,i t)

    VERIFIED against Sionna's own `cfr()` at 2.3e-4 relative error (float32 rounding);
    the variant WITHOUT the carrier in the delay term is wrong by 1.44, so `paths.a`
    definitively does not carry it. That check is what licenses summing over paths here
    instead of calling `cfr()` -- which we must do, because the correction below is
    per-path and `cfr()` sums internally.

    `range_migration=True` adds the term the one-solve model is missing. Each path's
    delay actually drifts, `tau_i(t) = tau_i - (f_D,i / f_c) t`, and the native model
    freezes it, so intra-frame range migration is absent (see `doppler_validity`).

    THE TRAP, and why this is not a one-line substitution: putting `tau_i(t)` into the
    FULL `(f_c + f)` term expands to
    `exp(-j2pi f_c tau_i) * exp(+j2pi f_D,i t) * (baseband)` -- reproducing the explicit
    Doppler factor a SECOND time. The carrier's share of the drift IS the Doppler term.
    So the drift is applied to the BASEBAND term only, and the carrier term keeps the
    frozen delay. A static target (`doppler == 0`) is therefore unchanged, exactly.

    Shapes: `a`/`tau`/`doppler` are `[..., n_paths]` (Sionna resolves all three per
    antenna pair); `freqs` is `[n_freqs]` of baseband offsets. Returns
    `[..., n_chirps, n_freqs]` with the path axis summed away.

    VALIDATED END-TO-END 2026-08-11 against a per-chirp re-trace (planar box target,
    free space, max_depth=1, radial_like config, 48 chirps, 5 m/s radial): native
    whole-cube rel-RMSE 8.52e-2 -> 2.30e-3 with `range_migration=True` (37x), and the
    residual growth rate matches `doppler_validity`'s analytic coefficient to 0.3% at
    the noise-floor-free end of the frame. One nuance from that measurement: on a DDMA
    config the range-Doppler argmax can land on a different CODE REPLICA than the
    re-trace's (replicas sit n_chirps/n_tx Doppler bins apart and are near-equal in
    power at 2.3e-3 cube agreement) -- the range bin is identical, so compare cubes,
    not argmaxes.

    FLIPPED 2026-08-14: `cfr_from_paths` (the corpus-generation call site) now defaults
    to `range_migration=True`, on the strength of the 37x measurement above -- corpora
    generated before this date used the uncorrected (`range_migration=False`) model; see
    CHANGELOG.md. THIS function's own default stays False deliberately: it is a
    general-purpose closed-form utility (also exercised directly, with both values, by
    `tests/test_ml_doppler_validity.py`), and `cfr_from_paths` always passes the flag
    explicitly rather than relying on this default.
    """
    a = np.asarray(a)
    tau = np.asarray(tau)
    doppler = np.asarray(doppler)
    freqs = np.asarray(freqs, dtype=np.float64)

    a_b = a[..., :, None, None]
    tau_b = tau[..., :, None, None].astype(np.float64)
    dop_b = doppler[..., :, None, None].astype(np.float64)
    f_b = freqs.reshape((1,) * (a.ndim - 1) + (1, 1, freqs.size))
    t_b = (np.arange(int(n_chirps), dtype=np.float64) * float(chirp_period_s)).reshape(
        (1,) * (a.ndim - 1) + (1, int(n_chirps), 1))

    tau_baseband = tau_b - (dop_b / float(f_c)) * t_b if range_migration else tau_b
    phase = (np.exp(-2j * np.pi * float(f_c) * tau_b)          # carrier: frozen delay --
             * np.exp(2j * np.pi * dop_b * t_b)                # its drift IS this term
             * np.exp(-2j * np.pi * f_b * tau_baseband))       # baseband: drifting delay
    return (a_b * phase).sum(axis=-3)


def cfr_from_paths(paths, cfg, *, n_chirps: int, freq_chunk: int = 128,
                   range_migration: bool = True) -> np.ndarray:
    """`Paths` -> RAW CFR cube `[n_rx_ant, n_tx_ant, n_chirps, n_samples]`.

    Sionna-specific half of what used to be a single `_beat_from_paths`: samples the
    CFR on the ramp's frequency grid (`beat_frequencies`) but does NOT conjugate or
    antenna-reverse -- that generic tensor mapping is `e2e.chain.dechirp.beat_from_cfr`,
    the ONE implementation both `_beat_from_paths` (below) and `RTEnvironmentBlock`
    delegate to. dtype/scale are whatever the CFR path returns (typically complex64),
    unchanged from before this split.

    `range_migration` (default True, FLIPPED 2026-08-14 -- was False): use
    `cfr_sum_over_paths`, the closed-form per-path CFR with the intra-frame delay-drift
    correction, instead of Sionna's own `Paths.cfr()` (which freezes each path's delay
    across the CPI). `cfr_sum_over_paths` is VERIFIED against `Paths.cfr()` itself to
    2.3e-4 relative error at `range_migration=False` (see that function's docstring), so
    this substitution changes nothing except adding the delay-drift term; MEASURED
    whole-cube rel-RMSE 8.52e-2 -> 2.30e-3 (37x) with the correction on, on a moving
    planar target (2026-08-11). `paths.a`/`paths.tau`/`paths.doppler` all carry the full
    `[num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]` shape here because `_solve`
    always requests `synthetic_array=False` (real per-element ray tracing, not an
    analytic array-response shortcut) -- the same shape `Paths.cfr()` itself broadcasts
    against, so no reshaping is needed before handing them to `cfr_sum_over_paths`.
    `range_migration=False` keeps the old `Paths.cfr()` call, byte-for-byte, for anyone
    who needs to reproduce a pre-flip corpus deliberately.

    `freq_chunk` bounds peak memory: both paths materialise a
    `[rx, rx_ant, tx, tx_ant, num_paths, n_chirps, n_freqs]`-shaped array before summing
    over paths, so a full 512-sample / 192-chirp / 50-path call needs hundreds of MB.
    Chunking over frequencies is free -- the expensive ray tracing already happened.
    """
    freqs = beat_frequencies(cfg)
    n_samples = freqs.size
    chunk = _cfr_freq_chunk(paths, cfg, n_chirps=n_chirps, requested=freq_chunk)
    out: List[np.ndarray] = []
    if range_migration:
        # f_c = f0 + B/2, the chirp centre -- see the module docstring, eq. (3), and
        # `beat_frequencies`, which baseband-references against the same value.
        f_c = float(cfg.f0_hz) + float(cfg.bandwidth_hz) / 2.0
        a_re, a_im = paths.a
        a = np.asarray(a_re.numpy()) + 1j * np.asarray(a_im.numpy())
        tau = np.asarray(paths.tau.numpy())
        doppler = np.asarray(paths.doppler.numpy())
    for lo in range(0, n_samples, chunk):
        if range_migration:
            h = cfr_sum_over_paths(
                a, tau, doppler, freqs[lo:lo + chunk],
                f_c=f_c, chirp_period_s=float(cfg.chirp_period_s),
                n_chirps=int(n_chirps), range_migration=True,
            )
        else:
            h = paths.cfr(
                frequencies=freqs[lo:lo + chunk],
                sampling_frequency=1.0 / float(cfg.chirp_period_s),
                num_time_steps=int(n_chirps),
                normalize_delays=False,   # absolute delay IS the range -- never normalize
                normalize=False,          # keep physical amplitudes
                out_type="numpy",
            )
        # h: [num_rx, num_rx_ant, num_tx, num_tx_ant, n_chirps, n_freqs]; one tx/rx
        # device each, so indices 0 select them.
        out.append(np.asarray(h)[0, :, 0, :, :, :])
    return np.ascontiguousarray(np.concatenate(out, axis=-1))


def _beat_from_paths(paths, cfg, *, n_chirps: int, freq_chunk: int = 128,
                     range_migration: bool = True) -> np.ndarray:
    """`Paths` -> beat cube `[n_rx_ant, n_tx_ant, n_chirps, n_samples]`, complex64.

    Applies equation (3): CFR on the ramp's frequency grid (`cfr_from_paths`),
    conjugated, with the antenna index reversed (see "Element ordering / array
    handedness") -- via `e2e.chain.dechirp.beat_from_cfr`, so this and
    `RTEnvironmentBlock` share exactly one implementation of that mapping.
    """
    raw = cfr_from_paths(paths, cfg, n_chirps=n_chirps, freq_chunk=freq_chunk,
                         range_migration=range_migration)
    from e2e.chain.dechirp import beat_from_cfr

    beat = beat_from_cfr(torch.from_numpy(raw))
    return np.ascontiguousarray(beat.numpy(), dtype=np.complex64)


def mimo_combine(cfg, beat: np.ndarray) -> np.ndarray:
    """Beat cube `[n_rx, n_tx, n_chirps, n_samples]` -> ADC cube `[n_rx, n_chirps, n_samples]`.

    Thin numpy<->torch wrapper around `e2e.chain.dechirp.mimo_combine` -- the ONE
    implementation of the TDM/DDMA combine (see that function's docstring for the
    scheme semantics, which mirror `rd_synth.synthesize_adc`'s per-chirp TX factor).
    """
    from e2e.chain.dechirp import mimo_combine as _mimo_combine_torch

    beat_t = torch.from_numpy(np.ascontiguousarray(beat))
    adc_t = _mimo_combine_torch(cfg, beat_t)
    return np.ascontiguousarray(adc_t.numpy())


# --------------------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------------------
def _peak_reference_amplitude(cfg, adc: np.ndarray, min_range_m: float) -> float:
    """Per-sample amplitude of the strongest target, in `rd_synth`'s SNR convention.

    rd_synth defines `snr_db` as the post-2-D-FFT SNR of the strongest scatterer at
    its peak, with coherent gain `G = n_samples * n_chirps_per_tx` for an *unwindowed*
    2-D FFT of one RX channel using one TX's chirps, and derives the noise power from
    that scatterer's per-sample amplitude `A_max`. Ray tracing gives no such scalar,
    so we invert the same relation: measure the unwindowed 2-D FFT peak `P` and take
    `A_max = P / G`. Identical convention, measured instead of assumed.

    Range bins closer than `min_range_m` (and their negative-frequency mirrors) are
    excluded: a monostatic scene's ground bounce / residual coupling sits at near-zero
    range and would otherwise set the noise floor for a distant target.
    """
    mag = np.abs(np.fft.fft2(_snr_reference_chirps(cfg, adc), axes=(1, 2)))
    n_samples = mag.shape[-1]
    guard = int(np.ceil(float(min_range_m) / float(cfg.range_resolution_m)))
    guard = min(guard, n_samples // 2)
    if guard > 0:
        mag = mag[:, :, guard:n_samples - guard]
    if mag.size == 0:
        return 0.0
    return float(mag.max()) / _coherent_gain(cfg, adc)


def _snr_reference_chirps(cfg, adc):
    """The chirps of a single TX, i.e. what the SNR convention integrates coherently."""
    return adc[:, ::int(cfg.n_tx), :] if str(cfg.mimo).lower() == "tdm" else adc


def _coherent_gain(cfg, adc) -> float:
    """`n_samples * n_chirps_per_tx` of the ACTUAL cube (not of `cfg`).

    Reading the chirp count off the array matters because `rt_retrace_reference` and
    `doppler_error_study` truncate the CPI (`n_chirps_cap`) without necessarily
    replacing `cfg`; a stale `cfg.n_chirps` would mis-scale the noise.
    """
    return float(adc.shape[-1]) * float(_snr_reference_chirps(cfg, adc).shape[1])


def _add_awgn(cfg, adc: torch.Tensor, snr_db, seed, min_range_m: float) -> torch.Tensor:
    """Add calibrated complex AWGN, reusing rd_synth's documented SNR convention."""
    if snr_db is None:
        return adc
    adc_np = adc.cpu().numpy()
    a_max = _peak_reference_amplitude(cfg, adc_np, min_range_m)
    coh_gain = _coherent_gain(cfg, adc_np)
    # An empty (no-path) scene has no reference amplitude; emit unit-variance noise so
    # the frame is still a usable "background only" sample -- same fallback as rd_synth.
    sigma2 = (a_max ** 2 * coh_gain / (10.0 ** (float(snr_db) / 10.0))) if a_max > 0 else 1.0
    gen = torch.Generator(device=adc.device)
    gen.manual_seed(int(seed) if seed is not None
                    else int(torch.randint(0, 2 ** 62, (1,)).item()))
    w = torch.randn(tuple(adc.shape) + (2,), generator=gen, device=adc.device,
                    dtype=torch.float32) * math.sqrt(sigma2 / 2.0)
    return adc + torch.view_as_complex(w.contiguous())


# --------------------------------------------------------------------------------
# Native (single-solve) generation
# --------------------------------------------------------------------------------
def rt_cfr_frame(cfg, scenario, *, frame_idx: int = 0, base_scene: str = "flat",
                 device=None, rt_scene: Optional[RTScene] = None, max_depth: int = 2,
                 include_leakage: bool = False, diffuse_reflection: bool = True,
                 specular_reflection: bool = True, refraction: bool = False,
                 solver_seed: int = 41, freq_chunk: int = 128,
                 range_migration: bool = True) -> torch.Tensor:
    """Ray-trace one radar frame and return its RAW channel frequency response.

    `complex64 [n_rx, n_tx, n_chirps, n_samples]` on `device` -- the pipeline's
    DOMAIN_CFR contract (see `e2e.frames`), sampled on the FMCW ramp's beat-frequency
    grid (`beat_frequencies(cfg)`) but NOT YET conjugated, antenna-reversed or
    MIMO-combined. `e2e.chain.dechirp.DechirpBlock(cfg)` is the bridge from here to a
    dechirped ADC cube; `rt_synthesize_adc` (below) and `e2e.environment.blocks.
    RTEnvironmentBlock` both build on this one extraction path so the ray-tracing +
    CFR-sampling logic exists exactly once. See `build_rt_scene` for the scene/`rt_scene`
    parameters and `_solve` for the solver ones (both shared verbatim with
    `rt_synthesize_adc`). `range_migration` (default True, see `cfr_from_paths`) selects
    the intra-frame delay-drift correction; pass False to reproduce a pre-2026-08-14
    corpus deliberately.
    """
    dev = _resolve_device(device)
    if rt_scene is None:
        rt_scene = build_rt_scene(scenario, cfg, base_scene=base_scene, frame_idx=frame_idx)

    paths = _solve(rt_scene, max_depth=max_depth, include_leakage=include_leakage,
                   diffuse_reflection=diffuse_reflection,
                   specular_reflection=specular_reflection, refraction=refraction,
                   seed=solver_seed)
    raw = cfr_from_paths(paths, cfg, n_chirps=int(cfg.n_chirps), freq_chunk=freq_chunk,
                         range_migration=range_migration)
    return torch.as_tensor(raw, dtype=torch.complex64, device=dev)


def rt_synthesize_adc(cfg, scenario, *, frame_idx: int = 0, base_scene: str = "flat",
                      snr_db: Optional[float] = 30.0, seed: Optional[int] = None,
                      device=None, rt_scene: Optional[RTScene] = None,
                      max_depth: int = 2, include_leakage: bool = False,
                      diffuse_reflection: bool = True, specular_reflection: bool = True,
                      refraction: bool = False, solver_seed: int = 41,
                      freq_chunk: int = 128,
                      snr_ref_min_range_m: Optional[float] = None,
                      range_migration: bool = True) -> torch.Tensor:
    """Ray-trace one radar frame and return its dechirped ADC cube.

    Drop-in replacement for `e2e.ml.rd_synth.synthesize_adc(cfg, scatterers, pose, ...)`
    at the dataset level: same return contract, `complex64 [n_rx, n_chirps, n_samples]`
    on `device`, consumable by `e2e.ml.transforms` unchanged.

    ONE `PathSolver` solve is performed; the chirp axis comes from Sionna's Doppler
    time-evolution (native evolution, `range_migration` correcting the delay it freezes
    -- see `cfr_from_paths`), not from re-tracing -- see `rt_retrace_reference` /
    `doppler_error_study` for the ground-truth comparison this trades against.

    Parameters
    ----------
    cfg : RadarConfig
    scenario : e2e.scenario.Scenario   (needs at least one RADAR node)
    frame_idx : int                    frame to resolve motion at
    base_scene : str                   see `build_rt_scene`
    snr_db : float or None             post-2-D-FFT SNR of the strongest target
                                       (`None` disables noise); see `_peak_reference_amplitude`
    seed : int or None                 seeds the AWGN only (the RT solve uses `solver_seed`)
    range_migration : bool             intra-frame delay-drift correction (default True,
                                       see `cfr_from_paths`); False reproduces a
                                       pre-2026-08-14 corpus deliberately.
    device : torch device or None      defaults to the library device
    rt_scene : RTScene or None         reuse a scene built by `build_rt_scene`
                                       (skips base-scene parsing); built here if None
    include_leakage : bool             keep the direct TX->RX path (radar TX/RX leakage)
    snr_ref_min_range_m : float or None
        Range guard for the noise-calibration peak search; defaults to
        `3 * cfg.range_resolution_m`.
    """
    dev = _resolve_device(device)
    s_pars = rt_cfr_frame(cfg, scenario, frame_idx=frame_idx, base_scene=base_scene,
                          device=dev, rt_scene=rt_scene, max_depth=max_depth,
                          include_leakage=include_leakage,
                          diffuse_reflection=diffuse_reflection,
                          specular_reflection=specular_reflection, refraction=refraction,
                          solver_seed=solver_seed, freq_chunk=freq_chunk,
                          range_migration=range_migration)

    from e2e.chain.dechirp import DechirpBlock

    adc = DechirpBlock(cfg).apply({"s_pars": s_pars})["adc"]

    guard = (3.0 * float(cfg.range_resolution_m) if snr_ref_min_range_m is None
             else float(snr_ref_min_range_m))
    return _add_awgn(cfg, adc, snr_db, seed, guard).to(torch.complex64)


# --------------------------------------------------------------------------------
# Ground truth: re-trace the geometry once per chirp
# --------------------------------------------------------------------------------
def rt_retrace_reference(cfg, scenario, *, frame_idx: int = 0, base_scene: str = "flat",
                         n_chirps_cap: Optional[int] = None, snr_db: Optional[float] = None,
                         seed: Optional[int] = None, device=None,
                         rt_scene: Optional[RTScene] = None, max_depth: int = 2,
                         include_leakage: bool = False, diffuse_reflection: bool = True,
                         specular_reflection: bool = True, refraction: bool = False,
                         solver_seed: int = 41, freq_chunk: int = 128,
                         snr_ref_min_range_m: Optional[float] = None) -> torch.Tensor:
    """Ground-truth ADC cube: re-solve the scene for **every chirp**.

    Chirp `c` is traced with every moving object advanced to `p0 + v * c * T_c` and
    `num_time_steps=1`, so the slow-time phase evolution comes entirely from the
    re-traced geometry (delays, angles, amplitudes all update) instead of from Sionna's
    first-order Doppler phase rotation. Note that at `num_time_steps=1` the per-path
    Doppler factor is `exp(j*2pi*f_D*0) = 1`, so the object velocities do not
    double-count here -- they are consumed purely as the per-chirp displacement.

    The radar itself is held fixed across the CPI, matching `rd_synth` (whose
    `RadarPose` is per-frame) and `frame_scatterers` (whose velocities are per-object).

    Expensive by design: cost is `n_chirps` solves instead of one. `n_chirps_cap`
    truncates the CPI (the returned cube then has `min(n_chirps, cap)` chirps, which
    is what `doppler_error_study` compares against a matching native run). Returns
    `complex64 [n_rx, n_chirps_used, n_samples]` on `device`.
    """
    dev = _resolve_device(device)
    if rt_scene is None:
        rt_scene = build_rt_scene(scenario, cfg, base_scene=base_scene, frame_idx=frame_idx)

    from e2e.ml.scatterers import frame_scatterers

    scats = frame_scatterers(scenario, frame_idx, dt=1.0 / float(cfg.frame_rate_hz))
    base_pos = {obj.name: np.asarray(sc.position, dtype=np.float64)
                for obj, sc in zip(scenario.objects, scats)}
    vel = {obj.name: np.asarray(sc.velocity, dtype=np.float64)
           for obj, sc in zip(scenario.objects, scats)}

    n_chirps = int(cfg.n_chirps) if n_chirps_cap is None else min(int(cfg.n_chirps),
                                                                  int(n_chirps_cap))
    t_c = float(cfg.chirp_period_s)

    frames: List[np.ndarray] = []
    try:
        for c in range(n_chirps):
            for name, so in rt_scene.objects.items():
                so.position = [float(x) for x in (base_pos[name] + vel[name] * (c * t_c))]
            paths = _solve(rt_scene, max_depth=max_depth, include_leakage=include_leakage,
                           diffuse_reflection=diffuse_reflection,
                           specular_reflection=specular_reflection, refraction=refraction,
                           seed=solver_seed)
            # num_time_steps=1 -> [n_rx_ant, n_tx_ant, 1, n_samples]
            beat = _beat_from_paths(paths, cfg, n_chirps=1, freq_chunk=freq_chunk)
            frames.append(beat[:, :, 0, :])
    finally:
        # Leave the scene at its frame-0 geometry so the handle stays reusable.
        for name, so in rt_scene.objects.items():
            so.position = [float(x) for x in base_pos[name]]

    # [n_rx_ant, n_tx_ant, n_chirps, n_samples] -- same TDM/DDMA combine as the native
    # path, via `mimo_combine` (the one implementation; see its docstring), rather than
    # a second hand-rolled copy of the same math.
    beat_cube = np.stack(frames, axis=2)
    adc_np = mimo_combine(cfg, beat_cube)

    adc = torch.as_tensor(np.ascontiguousarray(adc_np), dtype=torch.complex64, device=dev)
    guard = (3.0 * float(cfg.range_resolution_m) if snr_ref_min_range_m is None
             else float(snr_ref_min_range_m))
    return _add_awgn(cfg, adc, snr_db, seed, guard).to(torch.complex64)


