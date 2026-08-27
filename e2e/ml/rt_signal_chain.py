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
from typing import List, Optional, Tuple

import numpy as np
import torch

from e2e.environment.geometry import nearest_surface_point
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


def cfr_sum_over_paths_budgeted(a, tau, doppler, freqs, *, f_c: float,
                                chirp_period_s: float, n_chirps: int,
                                range_migration: bool = False, device=None,
                                byte_budget: int = 2 ** 28) -> np.ndarray:
    """Memory-bounded `cfr_sum_over_paths`: same math, never materialises the full
    `[..., n_paths, n_chirps, n_freqs]` broadcast.

    WHY THIS EXISTS (2026-08-15): the naive broadcast above is fine for the reference
    role it plays (tests, error studies, tens of paths) but is the exact tensor
    `_cfr_freq_chunk`'s ELEMENT budget was never calibrated for -- that budget bounds
    DrJit's lazy index width, while numpy genuinely ALLOCATES: at a realistic solve
    (~800-3500 paths, 252 chirps, 192 antenna pairs) one complex128 temporary is
    ~34 GB and the expression builds several, which swap-thrashed a 32 GB box without
    ever finishing scene 0 of a corpus run. Measured, not speculated -- see
    report/rt_ml/rt_kenney_d2_v1_gen.md.

    Implementation: torch (CUDA when available -- pass `device` to override), chunked
    over the path and frequency axes so no float64 temporary exceeds `byte_budget`
    bytes. The three exponentials collapse into ONE phase argument accumulated in
    float64 (the carrier term alone is ~1e6 rad, far beyond float32), wrapped modulo
    2pi, and only then evaluated in float32 -- so the result is complex64, matching
    what the ADC-cube consumers store anyway, with phase error ~1e-7 rad. Agreement
    with the float64 reference is pinned by tests at 1e-5 relative; the
    range_migration=False/True static-target bit-equality guarantee is preserved
    because a zero Doppler makes `tau_baseband` bitwise equal to `tau`.

    Same shape contract as `cfr_sum_over_paths`; returns complex64.
    """
    a = np.asarray(a)
    tau = np.asarray(tau)
    dop = np.asarray(doppler)
    freqs = np.asarray(freqs, dtype=np.float64)
    lead = a.shape[:-1]
    n_paths = int(a.shape[-1])
    n_chirps = int(n_chirps)
    n_freqs = int(freqs.size)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    lead_n = int(np.prod(lead)) if lead else 1
    a_t = torch.from_numpy(np.ascontiguousarray(a.reshape(lead_n, n_paths))).to(
        dev, torch.complex64)
    tau_t = torch.from_numpy(np.ascontiguousarray(
        tau.reshape(lead_n, n_paths).astype(np.float64))).to(dev)
    dop_t = torch.from_numpy(np.ascontiguousarray(
        dop.reshape(lead_n, n_paths).astype(np.float64))).to(dev)
    f_t = torch.from_numpy(freqs).to(dev)                                   # [F]
    t_t = (torch.arange(n_chirps, dtype=torch.float64, device=dev)
           * float(chirp_period_s))                                         # [C]

    out = torch.zeros(lead_n, n_chirps, n_freqs, dtype=torch.complex64, device=dev)
    two_pi = 2.0 * np.pi

    # Chunk (paths x freqs) so one [lead, p_blk, n_chirps, f_blk] float64 tensor stays
    # under byte_budget; a handful of same-shaped successors (wrap, cos/sin, product)
    # live at once, so the true peak is a small multiple of it.
    unit = lead_n * n_chirps                       # elements per (path, freq) pair
    budget_pf = max(1, (max(1, int(byte_budget)) // 8) // unit)
    if budget_pf >= n_paths and n_paths > 0:
        p_blk, f_blk = n_paths, max(1, min(n_freqs, budget_pf // n_paths))
    else:
        p_blk, f_blk = budget_pf, 1

    for p0 in range(0, n_paths, p_blk):
        p1 = min(n_paths, p0 + p_blk)
        tau_b = tau_t[:, p0:p1, None, None]                                 # [L,Pb,1,1]
        dop_b = dop_t[:, p0:p1, None, None]
        a_b = a_t[:, p0:p1, None, None]
        # [L,Pb,C,1]; with doppler == 0 this is bitwise tau_b -- the static-target
        # no-op guarantee rides on that.
        tau_bb = tau_b - (dop_b / float(f_c)) * t_t.view(1, 1, -1, 1) \
            if range_migration else tau_b
        base = -two_pi * float(f_c) * tau_b + two_pi * dop_b * t_t.view(1, 1, -1, 1)
        for f0 in range(0, n_freqs, f_blk):
            f1 = min(n_freqs, f0 + f_blk)
            arg = base - two_pi * f_t[f0:f1].view(1, 1, 1, -1) * tau_bb     # [L,Pb,C,Fb]
            arg = torch.remainder(arg, two_pi).to(torch.float32)
            phase = torch.complex(torch.cos(arg), torch.sin(arg))
            out[:, :, f0:f1] += (a_b * phase).sum(dim=1)
    return out.cpu().numpy().reshape(*lead, n_chirps, n_freqs)


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
    if range_migration:
        # f_c = f0 + B/2, the chirp centre -- see the module docstring, eq. (3), and
        # `beat_frequencies`, which baseband-references against the same value.
        f_c = float(cfg.f0_hz) + float(cfg.bandwidth_hz) / 2.0
        a_re, a_im = paths.a
        a = np.asarray(a_re.numpy()) + 1j * np.asarray(a_im.numpy())
        tau = np.asarray(paths.tau.numpy())
        doppler = np.asarray(paths.doppler.numpy())
        # The budgeted closed form does its own (path x freq) memory chunking --
        # `_cfr_freq_chunk`'s element budget is a DrJit index bound, NOT an
        # allocation bound, and applying it here let realistic solves (~1e3 paths)
        # ask numpy for ~34 GB temporaries (see cfr_sum_over_paths_budgeted).
        h = cfr_sum_over_paths_budgeted(
            a, tau, doppler, freqs,
            f_c=f_c, chirp_period_s=float(cfg.chirp_period_s),
            n_chirps=int(n_chirps), range_migration=True,
        )
        # h: [num_rx, num_rx_ant, num_tx, num_tx_ant, n_chirps, n_freqs]; one tx/rx
        # device each, so indices 0 select them.
        return np.ascontiguousarray(np.asarray(h)[0, :, 0, :, :, :])
    chunk = _cfr_freq_chunk(paths, cfg, n_chirps=n_chirps, requested=freq_chunk)
    out: List[np.ndarray] = []
    for lo in range(0, n_samples, chunk):
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
# HYBRID RT: coherent multi-centre return per object  (default on; v1.1 = 5 centres)
# --------------------------------------------------------------------------------
# WHY THIS EXISTS -- the defect it repairs, measured on the D0 single-sphere scene:
#
#   `_solve(..., specular_reflection=True, diffuse_reflection=False)` finds **ZERO**
#   paths off a sphere, a low_poly_car, or any other curved/irregular target, at EVERY
#   range tested (0.6 m .. 14.2 m) and at Sionna's own 15,872-facet sphere tessellation.
#   That is not a tuning problem, it is geometry: Sionna's specular search is the image
#   method on planar facets, which requires the facet PLANE's specular point to fall
#   INSIDE the facet. On a convex surface of radius `a` tessellated at facet size `L`,
#   a facet's normal is off the true surface normal by up to ~L/(2a), so the plane's
#   monostatic specular point is displaced by ~R*L/(2a); demanding that be < L/2 gives
#   `R < a`, INDEPENDENT of L. Refining the mesh does not help -- a faceted convex body
#   simply has no monostatic image-method specular path at any useful range.
#
#   The generator's response was to make objects visible through DIFFUSE scattering
#   (`rt_scene_build.DEFAULT_SCATTERING_COEFFICIENT = 0.3`). That gets the ENERGY
#   roughly right -- MEASURED effective RCS of the D0 sphere (a = 0.5 m, true optical
#   pi*a^2 = -1.05 dBsm): incoherent sum over paths = -7.1 dBsm, i.e. exactly the S^2 =
#   0.09 (-10.5 dB) fraction the Degli-Esposti model sends into the diffuse lobe -- but
#   it destroys the COHERENCE. The return arrives as ~300 Monte-Carlo speckle rays with
#   random phases, spread over 6 range bins and decorrelated across the aperture
#   (MEASURED: 20.6 dB element-to-element amplitude spread, 1.73 rad RMS phase residual
#   against the ideal pi*sin(az) ramp, versus 0.03 rad for the analytic point target).
#   A radar detects targets by COHERENT integration; speckle earns none of it.
#
# WHAT THIS ADDS: the coherent half of the return that Sionna structurally cannot find.
# Each object contributes `n_centers` deterministic scattering centres (v1.1 default 5:
# the traced phase centre / specular point plus the bbox's four vertical-edge midpoints
# -- see `coherent_target_cfr`'s docstring and physics audit entry 4 for the measured
# case; `n_centers=1` is the pre-v1.1 point model), splitting an RCS of
# `(1 - S^2) * sigma_object` -- the energy-conserving complement of the diffuse
# lobe Sionna already computes, using the RCS the scenario layer already carries
# (`Scatterer.rcs_dbsm` / `scatterers.DEFAULT_RCS_DBSM`). Ray tracing keeps doing
# everything it is good at: geometry, occlusion, ground bounce, multipath, per-object
# Doppler. Nothing here changes the traced path set; the coherent term is ADDED to the
# CFR in Sionna's own convention (`h = sum a_i exp(-j2pi (f_c + f) tau_i) exp(j2pi f_D t)`,
# see `cfr_sum_over_paths`), so every downstream stage is untouched.
#
# STATUS: DEFAULT ON since 2026-08-17 (`coherent_targets=True` on `rt_cfr_frame` /
# `rt_synthesize_adc` / `RTEnvironmentBlock` / `chain_generate`). Pass
# `coherent_targets=False` to reproduce a pre-2026-08-17 corpus, which is what the
# regression tests do. Known approximations, all deliberate and listed:
#   * bounding-ellipsoid specular point: exact for a sphere, approximate for a car mesh
#     (the phase centre can be off by tens of cm, i.e. a few range bins, on a long
#     vehicle). A real fix would use the mesh's own nearest visible facet.
#   * no occlusion test: an object hidden behind another still gets its coherent return.
#     `paths.objects` carries per-interaction object ids and is the intended gate.
#   * flat-plate/dihedral aspect dependence is not modelled: sigma is the scenario's
#     scalar RCS, not an angle-dependent pattern.
_C_LIGHT = _C_LIGHT_MPS


def _array_element_positions(position, boresight, n_elem: int, spacing_wl: float,
                             wavelength_m: float) -> np.ndarray:
    """World-frame positions of a `1 x n_elem` `PlanarArray`'s elements, `[n_elem, 3]`.

    Mirrors what `build_rt_scene` sets up: `look_at` aims the device's local +x along
    `boresight`, so its local +y is `normalise(z_up x boresight)` -- the same ULA axis
    `rd_synth.array_axis` uses -- and Sionna's `PlanarArray` lays elements out at
    `y_j = d*j - (num_cols-1)*d/2` with `d` in wavelengths.
    """
    p = np.asarray(position, dtype=np.float64)
    u = np.asarray(boresight, dtype=np.float64)
    u = u / np.linalg.norm(u)
    y = np.cross(np.array([0.0, 0.0, 1.0]), u)
    ny = np.linalg.norm(y)
    if ny < 1e-12:                    # boresight straight up/down: pick any perpendicular
        y = np.array([0.0, 1.0, 0.0])
        ny = 1.0
    y = y / ny
    j = np.arange(int(n_elem), dtype=np.float64) - (int(n_elem) - 1) / 2.0
    return p[None, :] + (j * float(spacing_wl) * float(wavelength_m))[:, None] * y[None, :]


def _object_bbox(so) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """World-space `(centre, half_extents)` of a placed `SceneObject`, or None.

    Reads the Mitsuba mesh's own AABB (`SceneObject.mi_mesh.bbox()`), so scaling,
    position and mesh shape are all already baked in -- no mesh re-parsing, no
    per-asset table to keep in sync.
    """
    try:
        bb = so.mi_mesh.bbox()
        lo = np.array([float(bb.min[i]) for i in range(3)], dtype=np.float64)
        hi = np.array([float(bb.max[i]) for i in range(3)], dtype=np.float64)
    except Exception:                                    # pragma: no cover
        return None
    return 0.5 * (lo + hi), np.maximum(0.5 * (hi - lo), 1e-6)


def _specular_point(centre: np.ndarray, half: np.ndarray, radar_pos: np.ndarray) -> np.ndarray:
    """Monostatic specular point: the near intersection of the radar LOS with the
    object's bounding ellipsoid. Exact for a sphere; a documented approximation for
    anything else (see the section banner).

    Delegates to `e2e.environment.geometry.nearest_surface_point`, which is the SAME function
    `e2e.ml.labels` puts its objectness footprint on -- the label and the energy must not
    be computed by two implementations that can drift apart. No yaw is passed: `half`
    here comes from the Mitsuba mesh's WORLD-space AABB (`_object_bbox`), which already
    has the object's placed orientation baked in.
    """
    return nearest_surface_point(centre, half, radar_pos)


def _visible_corner_centers(position, extent_m, yaw_rad: float,
                            radar_pos: np.ndarray) -> np.ndarray:
    """Vertical-edge midpoints of the object's own (yawed, body-frame) footprint that
    FACE the radar, `[n_visible, 3]` at the object's mid-height.

    These are the secondary scattering centres of the multi-centre target model (see
    `coherent_target_cfr`): the corners of a vehicle body are where the strongest
    persistent scattering centres of real automotive targets measure (wheel wells,
    body corners). Corners live at `(+-L/2, +-W/2, 0)` in the object's OWN frame
    (local +x = length, per `e2e.environment.geometry`), rotated by `yaw_rad` about world +z
    around `position` (the geometric centre) -- NOT the world-AABB corners the
    pre-2026-08-24 model used, which sat up to ~3 m off-body at oblique yaw because
    an axis-aligned box inflates around a rotated body, with 2 of its 4 corners
    landing BEHIND the target (batch-review finding A13).

    FAR-SIDE CULL: a corner is kept iff at least one of its two adjacent vertical
    faces has an outward normal with a positive component toward the radar -- the
    fully shadowed corner(s) of a convex body do not backscatter monostatically.
    Generic aspect keeps 3 corners; exact broadside/end-on keeps 2; an empty result
    happens only in the degenerate no-direction case (the radar's horizontal
    position exactly on the centre), where the caller carries the primary point
    alone. Coincident corners -- possible only through an explicit degenerate
    `extent_m` override with a zero length or width, which collapses a corner pair
    -- are deduplicated so a collapsed pair does not silently carry double its
    sigma share (review finding 2026-08-24).
    """
    cx, cy, cz = (float(v) for v in position)
    hl, hw = 0.5 * float(extent_m[0]), 0.5 * float(extent_m[1])
    c, s = math.cos(float(yaw_rad)), math.sin(float(yaw_rad))
    # Radar LOS in the body frame (rotate the world offset by -yaw), normalized so
    # the visibility comparison below has a scale-free epsilon: at EXACT broadside
    # the along-body component is analytically zero but floats leave ~1e-16 of it,
    # which without the epsilon would resurrect a shadowed far corner.
    ux_w, uy_w = float(radar_pos[0]) - cx, float(radar_pos[1]) - cy
    norm = math.hypot(ux_w, uy_w)
    if norm <= 0.0:
        return np.zeros((0, 3), dtype=np.float64)   # radar above the centre: degenerate
    ux, uy = (c * ux_w + s * uy_w) / norm, (-s * ux_w + c * uy_w) / norm
    pts = []
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            if not (sx * ux > 1e-9 or sy * uy > 1e-9):
                continue                       # both adjacent faces look away
            bx, by = sx * hl, sy * hw
            pts.append([cx + c * bx - s * by, cy + s * bx + c * by, cz])
    arr = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    # Exact duplicates only arise from a zero half-extent (see docstring); order is
    # irrelevant downstream (equal power split), so np.unique's sort is harmless.
    return np.unique(arr, axis=0) if arr.size else arr


def _rt_phase_centres(paths, rt_scene) -> dict:
    """`{object name: (nearest visible surface point, visible?)}` read off the solve.

    For each object, look at every traced path whose FIRST interaction is with that
    object, and take the interaction vertex of the one with the shortest delay: that IS
    the nearest visible point of the object's real mesh, from the radar, WITH occlusion
    already applied by the ray tracer. Far better than a bounding-ellipsoid guess for an
    extended target (a 4.5 m car's bbox nearest point can be ~2 m from where the mesh
    actually reflects), and it costs nothing -- the solve already happened.

    Returns `{}` if this Sionna build does not expose `objects`/`vertices`/`tau`.
    """
    try:
        obj_ids = np.asarray(paths.objects.numpy())     # [depth, rx, rxa, tx, txa, P]
        verts = np.asarray(paths.vertices.numpy())      # [depth, rx, rxa, tx, txa, P, 3]
        tau = np.asarray(paths.tau.numpy())             # [rx, rxa, tx, txa, P]
    except Exception:                                   # pragma: no cover
        return {}
    if obj_ids.ndim < 6 or tau.ndim < 5:                # pragma: no cover
        return {}
    first_obj = obj_ids[0, 0, 0, 0, 0]                  # [P], one antenna pair
    first_v = verts[0, 0, 0, 0, 0]                      # [P, 3]
    t0 = tau[0, 0, 0, 0]                                # [P]
    ok = np.isfinite(t0) & (t0 > 0)
    out = {}
    for name, so in rt_scene.objects.items():
        try:
            oid = int(so.object_id)
        except Exception:                               # pragma: no cover
            continue
        sel = ok & (first_obj == oid)
        if not sel.any():
            out[name] = (None, False)
            continue
        k = int(np.argmin(np.where(sel, t0, np.inf)))
        out[name] = (first_v[k].astype(np.float64), True)
    return out


def _element_field_amplitude(pattern: str, boresight, direction) -> float:
    """|field| of ONE array element in `direction`, for Sionna's named element `pattern`.

    Gain, not decoration. The traced diffuse return already receives this factor -- Sionna
    applies the element pattern inside the solve -- while `coherent_target_cfr` synthesizes
    its return analytically and so has to apply it explicitly. When it did not, and the
    scene used the shipped `tr38901` element, 91% of every target's RCS (the coherent
    fraction at `S=0.3`) was radiated and received isotropically while the 9% diffuse
    remainder got the full directive gain -- suppressing targets relative to a background
    that was never penalised the same way. See notes/ESTABLISHED_FACTS.md F32.

    Convention, calibrated against Sionna's own pattern functions rather than assumed:
    `v_tr38901_pattern(theta, phi)` returns a complex FIELD whose `|c|^2` is the gain, and
    `(theta=pi/2, phi=0)` is the element's boresight (MEASURED: 8.00 dB there for
    `tr38901`, 0.00 dB for `iso`). `build_rt_scene`'s `look_at` puts the device's local +x
    along `boresight` and its local +y along `normalise(z_up x boresight)`, so a world
    direction maps to Sionna's spherical angles through that frame.

    ONE evaluation per target, shared by every element: across a metre-scale aperture at
    tens of metres the per-element pattern angle varies by well under a degree (the per-
    element PHASE, which does matter, is handled separately by the path-length term).
    """
    if not pattern or str(pattern) == "iso":
        return 1.0
    import drjit as dr
    import mitsuba as mi
    if mi.variant() is None:
        mi.set_variant("cuda_ad_mono_polarized")
    from sionna.rt import antenna_pattern as _ap

    fn = getattr(_ap, f"v_{pattern}_pattern", None)
    if fn is None:
        raise ValueError(
            f"no Sionna element pattern 'v_{pattern}_pattern'; coherent_target_cfr cannot "
            f"apply the element gain the traced path already has. Known: iso, tr38901, "
            f"dipole, hw_dipole.")

    b = np.asarray(boresight, dtype=np.float64)
    b = b / max(float(np.linalg.norm(b)), 1e-12)
    z_up = np.array([0.0, 0.0, 1.0])
    y_loc = np.cross(z_up, b)
    n = float(np.linalg.norm(y_loc))
    if n < 1e-9:                      # boresight straight up/down: pick any transverse axis
        y_loc = np.array([0.0, 1.0, 0.0])
        n = 1.0
    y_loc = y_loc / n
    z_loc = np.cross(b, y_loc)

    d = np.asarray(direction, dtype=np.float64)
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    dx, dy, dz = float(d @ b), float(d @ y_loc), float(d @ z_loc)
    theta = math.acos(max(-1.0, min(1.0, dz)))
    phi = math.atan2(dy, dx)

    c = fn(dr.cuda.ad.Float(theta), dr.cuda.ad.Float(phi))
    re = float(np.asarray(c[0].numpy()).reshape(-1)[0])
    im = float(np.asarray(c[1].numpy()).reshape(-1)[0])
    return math.hypot(re, im)


def coherent_target_cfr(cfg, rt_scene, scenario, *, frame_idx: int = 0,
                        n_chirps: Optional[int] = None,
                        scattering_coefficient: float = None,
                        range_migration: bool = True,
                        rcs_scale_db: float = 0.0,
                        paths=None, apply_doppler: bool = True,
                        n_centers: int = 5) -> np.ndarray:
    """Deterministic multi-scattering-centre CFR for every object, in Sionna's CFR
    convention.

    Returns `[n_rx_ant, n_tx_ant, n_chirps, n_samples]` complex64, directly addable to
    `cfr_from_paths`' output. See the section banner for the physics and the caveats.

    `n_centers` (default 5, the v1.1 model -- owner decision 2026-08-23) selects the
    model family member: `1` = the single primary point (the RT phase centre when a
    solve is supplied, else the yaw-aware body-frame surface point, else the bbox
    specular point); `5` = the FULL multi-centre model -- the primary point plus the
    VISIBLE vertical-edge midpoints of the object's own yawed footprint
    (`_visible_corner_centers`: body-frame corners from the scatterer's
    `extent_m`/`yaw_rad`, far side culled -- so generically 3-4 points total, not a
    literal five; the name is the family label, kept for provenance stability), each
    carrying an equal `sigma / n_pts` share. Only 1 and 5 are valid -- the two
    measured members of the family. An object with no resolvable extent falls back to
    its single primary point. WHY: one centre
    collapses target extent to ~1 cell where the corpus's own returns measure ~6
    azimuth cells (F44 -- measured on the SIMULATOR'S corpus for CFAR guard sizing,
    not a real-automotive figure; citation corrected 2026-08-24. The mismatch behind
    CFAR self-masking), and the 2026-08-20 spike measured
    the 5-centre model monotonically better on every coherence-sensitive axis (phase
    RMS -13%, peak-to-background +2.2 dB, extent restored; physics audit entry 4).
    `n_centers=1` reproduces the pre-v1.1 single-centre model.

    `scattering_coefficient` is the material's `S`; the coherent term carries
    `(1 - S^2)` of the object's RCS so coherent + diffuse conserve energy. Pass the same
    value `build_rt_scene` was given (defaults to `rt_scene_build`'s default).

    THREE STATED APPROXIMATIONS of the multi-centre split (batch reviews, 2026-08-23
    and -24):
    (a) energy conservation across the centres is IN EXPECTATION, not per frame -- the
    centres sit at different ranges/phases, so any one frame's coherent sum can sit a
    couple of dB off `sigma`; (b) the equal `sigma / n_pts` split ignores specular
    dominance -- on a real vehicle the near corner/specular flash typically carries
    far more than its equal share, so per-aspect RCS dynamics are understated;
    (c) the far-side cull is BINARY while the split is equal, so the kept-corner set
    (and with it the centre spread) switches discretely across silhouette
    transitions -- a hair off exact end-on keeps 3 corners spanning the body length
    where exact end-on keeps 2 spanning ~nothing. Measure-zero under random scene
    draws, so no corpus impact, but the exact-broadside branch is a degenerate
    state, not a physical broadside model; the refinement (grade each corner by
    max(0, face_normal . LOS) over its adjacent faces) is a post-v1.1 follow-up.

    `apply_doppler=False` drops the per-object `exp(j2pi f_D t)` factor. `rt_retrace_reference`
    needs that: it re-solves the geometry once per chirp with each object physically
    advanced, and consumes velocity purely as that displacement (its solves use
    `num_time_steps=1`, where Sionna's own Doppler factor is likewise 1). Leaving the
    factor in would double-count the motion.

    IMPLEMENTATION NOTE (the vectorization): per object, the phase separates as
    `exp(-2j pi (f_c + f_b) tau)` (a per-centre [rx, tx, freq] factor) times
    `exp(2j pi f_d (1 + f_b/f_c) t)` (a per-centre [chirp, freq] factor -- the
    `f_b/f_c` term IS range migration, algebraically identical to the
    `tau - (f_d/f_c) t` form), contracted over centres in one einsum. This does the
    big [rx, tx, chirp, freq] materialization once per OBJECT instead of once per
    centre, so 5 centres cost roughly what 1 did.
    """
    from e2e.environment.scatterers import frame_scatterers, radar_pose
    from e2e.ml.rt_scene_build import DEFAULT_SCATTERING_COEFFICIENT

    if scattering_coefficient is None:
        scattering_coefficient = DEFAULT_SCATTERING_COEFFICIENT
    coh_frac = max(0.0, 1.0 - float(scattering_coefficient) ** 2)

    n_chirps = int(cfg.n_chirps) if n_chirps is None else int(n_chirps)
    freqs = beat_frequencies(cfg)                                   # baseband offsets
    f_c = float(cfg.f0_hz) + float(cfg.bandwidth_hz) / 2.0
    lam = _C_LIGHT / f_c
    pose = radar_pose(scenario, frame_idx)
    scats = frame_scatterers(scenario, frame_idx, dt=1.0 / float(cfg.frame_rate_hz))
    radar_pos = np.asarray(pose.position, dtype=np.float64)

    # Element positions: TX spacing is n_rx*lambda/2, RX spacing lambda/2 (build_rt_scene).
    tx_pos = _array_element_positions(radar_pos, pose.boresight, int(cfg.n_tx),
                                      0.5 * int(cfg.n_rx), lam)
    rx_pos = _array_element_positions(radar_pos, pose.boresight, int(cfg.n_rx), 0.5, lam)

    t = np.arange(n_chirps, dtype=np.float64) * float(cfg.chirp_period_s)
    out = np.zeros((int(cfg.n_rx), int(cfg.n_tx), n_chirps, freqs.size), dtype=np.complex128)
    centres = _rt_phase_centres(paths, rt_scene) if paths is not None else {}

    if int(n_centers) not in (1, 5):
        # Only the two MEASURED members of the model family exist: the pre-v1.1 point
        # model and the 5-centre model the spike validated (audit entry 4). 2-4 would
        # take a non-symmetric prefix of the edge set (biased toward the +x corners)
        # and >5 would silently cap -- both are footguns, not options (review finding
        # 2026-08-23). Design a symmetric scheme before widening this.
        raise ValueError(f"n_centers must be 1 or 5, got {n_centers!r}")

    for obj, sc in zip(scenario.objects, scats):
        so = rt_scene.objects.get(obj.name)
        rt_p, visible = centres.get(obj.name, (None, True))
        if not visible:
            continue          # ray tracer found no path to it -- occluded, stay silent
        extent = getattr(sc, "extent_m", None)
        yaw = float(getattr(sc, "yaw_rad", 0.0))
        sc_pos = np.asarray(sc.position, dtype=np.float64)
        if rt_p is not None:
            p0 = np.asarray(rt_p, dtype=np.float64)
        elif extent is not None:
            # Yaw-aware body-frame surface point -- the SAME geometry `e2e.ml.labels.
            # target_geometry` puts the label on, so energy and label cannot drift.
            p0 = nearest_surface_point(sc_pos, 0.5 * np.asarray(extent, dtype=np.float64),
                                       radar_pos, yaw_rad=yaw)
        elif (bb := _object_bbox(so) if so is not None else None) is not None:
            centre, half = bb
            p0 = _specular_point(centre, half, radar_pos)
        else:
            continue
        sigma = coh_frac * 10.0 ** ((float(sc.rcs_dbsm) + float(rcs_scale_db)) / 10.0)
        if sigma <= 0.0:
            continue

        # The centre set: primary point + the VISIBLE body-frame footprint corners
        # (`_visible_corner_centers` -- yaw-aware, far side culled; A13), equal power
        # split (see docstring). No resolvable extent -> the primary point alone,
        # whatever n_centers says: a point target has nowhere deterministic to put
        # extra centres. (A primary point coinciding EXACTLY with a corner would
        # double-count that centre's sigma share; geometrically prevented for the
        # ellipsoid surface point and physically implausible for a traced mesh
        # vertex -- accepted.)
        if extent is not None and int(n_centers) > 1:
            corners = _visible_corner_centers(sc_pos, extent, yaw, radar_pos)
            pts = (np.vstack([p0[None, :], corners]) if corners.size
                   else p0[None, :])
        else:
            pts = p0[None, :]
        n_pts = pts.shape[0]
        sigma_i = sigma / float(n_pts)

        d_t = np.linalg.norm(tx_pos[None, :, :] - pts[:, None, :], axis=2)   # [P, n_tx]
        d_r = np.linalg.norm(rx_pos[None, :, :] - pts[:, None, :], axis=2)   # [P, n_rx]
        tau = (d_r[:, :, None] + d_t[:, None, :]) / _C_LIGHT                 # [P, rx, tx]

        los = pts - radar_pos[None, :]                                       # [P, 3]
        los = los / np.maximum(np.linalg.norm(los, axis=1, keepdims=True), 1e-12)
        # ELEMENT GAIN. The traced diffuse return gets this from Sionna inside the solve;
        # this analytic term has to apply it explicitly or the two halves of the same
        # target are radiated through different antennas (F32). Squared because it applies
        # once on transmit and once on receive -- TX and RX arrays are co-located and share
        # a boresight, so one evaluation serves both. Evaluated per centre (each has its
        # own direction).
        g_elem = np.array([_element_field_amplitude(
            getattr(rt_scene, "antenna_pattern", "iso"), pose.boresight, los[i])
            for i in range(n_pts)], dtype=np.float64)                        # [P]
        # Sionna's own per-path amplitude convention (VERIFIED against its `paths.a`:
        # |a|^2 = sigma lambda^2 / ((4 pi)^3 R_t^2 R_r^2) reproduces the traced sphere's
        # measured effective RCS).
        amp = (g_elem[:, None, None] ** 2 * math.sqrt(sigma_i) * lam
               / ((4.0 * math.pi) ** 1.5 * d_r[:, :, None] * d_t[:, None, :]))  # [P, rx, tx]
        # Physical Doppler: f_D = -2 v_r / lambda, v_r receding-positive (see the module
        # docstring's sign discussion), per centre via its own LOS.
        v_r = los @ np.asarray(sc.velocity, dtype=np.float64)                # [P]
        f_d = (-2.0 * v_r / lam) if apply_doppler else np.zeros(n_pts)

        # Separable phase (see docstring's implementation note):
        #   A[p, rx, tx, f] = amp * exp(-2j pi (f_c + f_b) tau)
        #   B[p, c, f]      = exp( 2j pi f_d (1 + f_b/f_c) t)   [migration folded in]
        a_fac = amp[:, :, :, None] * np.exp(
            -2j * np.pi * (f_c + freqs)[None, None, None, :] * tau[:, :, :, None])
        f_eff = (f_d[:, None, None] * (1.0 + freqs[None, None, :] / f_c)
                 if range_migration else
                 f_d[:, None, None] * np.ones((1, 1, freqs.size)))
        b_fac = np.exp(2j * np.pi * f_eff * t[None, :, None])                # [P, c, f]
        out += np.einsum("prtf,pcf->rtcf", a_fac, b_fac)

    return np.ascontiguousarray(out.astype(np.complex64))


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
                 range_migration: bool = True,
                 coherent_targets: bool = True,
                 scattering_coefficient: Optional[float] = None,
                 ground_scattering_coefficient: Optional[float] = None,
                 samples_per_src: Optional[int] = None) -> torch.Tensor:
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

    `coherent_targets` (default **True** since 2026-08-17) adds the per-object coherent
    specular return that Sionna's image method structurally cannot find -- see the
    "HYBRID RT" banner above for the physics and the measurements. `False` reproduces a
    pre-2026-08-17 corpus byte-for-byte. `scattering_coefficient` must match the material
    `S` the scene was built with, because it sets the coherent/diffuse energy split
    (`1 - S^2` coherent, `S^2` diffuse); `None` takes `rt_scene_build`'s default. When
    this function builds the scene itself it forwards the same value, so the two cannot
    drift apart.
    """
    dev = _resolve_device(device)
    if rt_scene is None:
        build_kwargs = {} if scattering_coefficient is None else {
            "scattering_coefficient": float(scattering_coefficient)}
        rt_scene = build_rt_scene(
            scenario, cfg, base_scene=base_scene, frame_idx=frame_idx,
            ground_scattering_coefficient=ground_scattering_coefficient, **build_kwargs)

    paths = _solve(rt_scene, max_depth=max_depth, include_leakage=include_leakage,
                   diffuse_reflection=diffuse_reflection,
                   specular_reflection=specular_reflection, refraction=refraction,
                   seed=solver_seed, samples_per_src=samples_per_src)
    raw = cfr_from_paths(paths, cfg, n_chirps=int(cfg.n_chirps), freq_chunk=freq_chunk,
                         range_migration=range_migration)
    if coherent_targets:
        # PROTOTYPE (see "HYBRID RT" banner): add the coherent specular return Sionna's
        # image method structurally cannot find on a curved/faceted target.
        raw = raw + coherent_target_cfr(
            cfg, rt_scene, scenario, frame_idx=frame_idx, n_chirps=int(cfg.n_chirps),
            scattering_coefficient=scattering_coefficient,
            range_migration=range_migration, paths=paths)
    return torch.as_tensor(raw, dtype=torch.complex64, device=dev)


def rt_synthesize_adc(cfg, scenario, *, frame_idx: int = 0, base_scene: str = "flat",
                      snr_db: Optional[float] = 30.0, seed: Optional[int] = None,
                      device=None, rt_scene: Optional[RTScene] = None,
                      max_depth: int = 2, include_leakage: bool = False,
                      diffuse_reflection: bool = True, specular_reflection: bool = True,
                      refraction: bool = False, solver_seed: int = 41,
                      freq_chunk: int = 128,
                      snr_ref_min_range_m: Optional[float] = None,
                      range_migration: bool = True,
                      coherent_targets: bool = True,
                      scattering_coefficient: Optional[float] = None,
                      ground_scattering_coefficient: Optional[float] = None,
                      samples_per_src: Optional[int] = None) -> torch.Tensor:
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
    coherent_targets : bool            add the per-object coherent specular return
                                       (default True since 2026-08-17; see the "HYBRID
                                       RT" banner). False reproduces a pre-2026-08-17
                                       corpus deliberately.
    scattering_coefficient : float or None
                                       the material `S` the scene was built with; sets
                                       the coherent/diffuse split. None -> the default.
    ground_scattering_coefficient : float or None
                                       ground roughness of the "flat" base scene; None ->
                                       `rt_scene_build.DEFAULT_GROUND_SCATTERING_COEFFICIENT`
                                       (read it -- a rough ground is ~37x the CFR cost).
                                       Ignored when `rt_scene` is supplied.
    samples_per_src : int or None       solver Monte-Carlo ray budget; None -> Sionna's own
                                       default (1e6). The dominant cost knob once the
                                       ground scatters diffusely: 1e5 cut the D1 path count
                                       10x and the CFR time 9x with the target metric
                                       unchanged to within 1.5 dB (MEASURED).
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
                          range_migration=range_migration,
                          coherent_targets=coherent_targets,
                          scattering_coefficient=scattering_coefficient,
                          ground_scattering_coefficient=ground_scattering_coefficient,
                          samples_per_src=samples_per_src)

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
                         snr_ref_min_range_m: Optional[float] = None,
                         coherent_targets: bool = True,
                         scattering_coefficient: Optional[float] = None) -> torch.Tensor:
    """Ground-truth ADC cube: re-solve the scene for **every chirp**.

    Chirp `c` is traced with every moving object advanced to `p0 + v * c * T_c` and
    `num_time_steps=1`, so the slow-time phase evolution comes entirely from the
    re-traced geometry (delays, angles, amplitudes all update) instead of from Sionna's
    first-order Doppler phase rotation. Note that at `num_time_steps=1` the per-path
    Doppler factor is `exp(j*2pi*f_D*0) = 1`, so the object velocities do not
    double-count here -- they are consumed purely as the per-chirp displacement.

    The radar itself is held fixed across the CPI, matching `rd_synth` (whose
    `RadarPose` is per-frame) and `frame_scatterers` (whose velocities are per-object).

    `coherent_targets` (default True, matching `rt_cfr_frame`) adds the per-object
    coherent specular return -- see the "HYBRID RT" banner. It MUST track the native
    path's setting or this reference stops being a reference: a static scene's re-trace
    and native cubes agree to <1e-4 relative only when both arms make the same choice
    (pinned by `tests/test_ml_rt_gen.py`'s static-scene equality test).

    Expensive by design: cost is `n_chirps` solves instead of one. `n_chirps_cap`
    truncates the CPI (the returned cube then has `min(n_chirps, cap)` chirps, which
    is what `doppler_error_study` compares against a matching native run). Returns
    `complex64 [n_rx, n_chirps_used, n_samples]` on `device`.
    """
    dev = _resolve_device(device)
    if rt_scene is None:
        rt_scene = build_rt_scene(scenario, cfg, base_scene=base_scene, frame_idx=frame_idx)

    from e2e.environment.scatterers import frame_scatterers

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
            raw = cfr_from_paths(paths, cfg, n_chirps=1, freq_chunk=freq_chunk)
            if coherent_targets:
                # Same coherent term the native path adds, but evaluated at THIS chirp's
                # geometry (the phase centres come from this chirp's own solve) and with
                # the Doppler factor off -- the motion is already in the displacement.
                raw = raw + coherent_target_cfr(
                    cfg, rt_scene, scenario, frame_idx=frame_idx, n_chirps=1,
                    scattering_coefficient=scattering_coefficient, paths=paths,
                    apply_doppler=False)
            from e2e.chain.dechirp import beat_from_cfr

            beat = np.ascontiguousarray(
                beat_from_cfr(torch.from_numpy(np.ascontiguousarray(raw))).numpy(),
                dtype=np.complex64)
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


