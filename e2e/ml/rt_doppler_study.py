"""
Native-vs-re-trace Doppler-error study + CLI for `e2e.ml.rt_gen`'s ray-traced ADC
generation.

Quantifies how well `e2e.ml.rt_signal_chain`'s one-solve-per-frame native Doppler
model (see that module's docstring, "The CFR -> beat mapping") tracks a ground-truth
per-chirp re-trace, and reports the price of the ground-truth path. Split out of the
original `rt_gen.py`; scene/mesh/asset construction lives in `e2e.ml.rt_scene_build`,
the CFR/beat-cube physics in `e2e.ml.rt_signal_chain`. `e2e.ml.rt_gen` re-exports this
module's public and private names for backward compatibility, including its CLI entry
point.

CLI
---
    python -m e2e.ml.rt_gen [--config radial_like] [--frames 2] [--chirps 16]
                            [--samples 128] [--base-scene flat|free|<sionna scene>]
                            [--target sphere|box] [--no-diffuse]
runs `doppler_error_study` and prints the native-vs-re-trace table. Use
`--target box --no-diffuse` for the deterministic (Monte-Carlo-free) comparison.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from e2e.ml.rt_scene_build import build_rt_scene
from e2e.ml.rt_signal_chain import _resolve_device, rt_retrace_reference, rt_synthesize_adc

# --------------------------------------------------------------------------------
# Doppler error study: native evolution vs per-chirp re-trace
# --------------------------------------------------------------------------------
def _rd_peak_bin(cfg, adc: torch.Tensor):
    """`(range_bin, doppler_bin)` of the strongest cell, through the shipped transforms."""
    import dataclasses as _dc

    from e2e.ml.transforms import adc_to_rd, tdm_deinterleave

    n_chirps = int(adc.shape[1])
    if str(cfg.mimo).lower() == "tdm":
        sub = _dc.replace(cfg, n_tx=1, mimo="single", n_chirps=n_chirps // int(cfg.n_tx))
        rd = adc_to_rd(sub, tdm_deinterleave(_dc.replace(cfg, n_chirps=n_chirps), adc))
    else:
        rd = adc_to_rd(_dc.replace(cfg, n_chirps=n_chirps), adc)
    power = (rd.abs() ** 2).sum(dim=0)                       # [range, doppler]
    flat = int(power.reshape(-1).argmax())
    return (flat // power.shape[1], flat % power.shape[1]), rd


def doppler_error_study(cfg, scenario, *, n_frames: int = 3, base_scene: str = "flat",
                        n_chirps_cap: Optional[int] = 16, device=None,
                        max_depth: int = 2, include_leakage: bool = False,
                        diffuse_reflection: bool = True, solver_seed: int = 41,
                        freq_chunk: int = 128, verbose: bool = False) -> Dict[str, Any]:
    """Quantify the native Doppler-phase model against the per-chirp re-trace.

    For each of the first `n_frames` frames, both paths are run **noise-free** on the
    same scene handle with the same solver seed, truncated to the same `n_chirps_cap`
    chirps, and compared three ways:

    * `per_chirp_rel_err[c]` -- `||native[:,c,:] - retrace[:,c,:]|| / ||retrace[:,c,:]||`
      (chirp 0 must be ~0: identical geometry, Doppler phase `exp(0)=1`);
    * `rel_rmse` -- the same ratio over the whole cube;
    * `peak_bin_native` / `peak_bin_retrace` -- the range/Doppler peak cell through
      `tdm_deinterleave` + `adc_to_rd`, i.e. whether the two agree where it matters.

    Also returns the measured wall-clock cost of each path and their ratio
    (`cost_multiplier`), which is the price of the ground-truth path.

    **`mc_noise_floor` -- read this before interpreting `rel_rmse`.** Sionna's diffuse
    reflections are Monte-Carlo sampled, so re-solving a scene whose geometry moved by
    even a fraction of a millimetre re-randomises which diffuse paths are found. The
    per-chirp re-trace therefore injects sampling noise of its own, and with
    `diffuse_reflection=True` that noise typically DOMINATES the Doppler-model
    difference this study is trying to measure. Each frame is therefore also re-solved
    once with a perturbed solver seed at unchanged geometry; the resulting
    `mc_noise_floor` is the level below which `rel_rmse` carries no information. For a
    clean measurement of the Doppler model alone, run with
    `diffuse_reflection=False` and a specular-friendly (planar-faced) target, where
    the floor is ~0.

    Note `n_chirps_cap` chirps of a TDM config still cover `cap // n_tx` chirps per TX,
    so keep it a multiple of `cfg.n_tx`.
    """
    dev = _resolve_device(device)
    n_chirps = int(cfg.n_chirps) if n_chirps_cap is None else min(int(cfg.n_chirps),
                                                                  int(n_chirps_cap))
    import dataclasses as _dc

    capped = _dc.replace(cfg, n_chirps=n_chirps)
    common = dict(base_scene=base_scene, max_depth=max_depth,
                  include_leakage=include_leakage, diffuse_reflection=diffuse_reflection,
                  solver_seed=solver_seed, freq_chunk=freq_chunk, device=dev, snr_db=None)

    frames: List[Dict[str, Any]] = []
    t_native_total = 0.0
    t_retrace_total = 0.0
    n_solves_native = 0
    n_solves_retrace = 0

    # Untimed warm-up: DrJit compiles (and caches) the trace/CFR kernels on the first
    # call, which would otherwise be charged entirely to the native path and make the
    # measured cost multiplier meaningless (it came out < 1 before this).
    rt_synthesize_adc(capped, scenario, frame_idx=0,
                      rt_scene=build_rt_scene(scenario, cfg, base_scene=base_scene,
                                              frame_idx=0),
                      **common)

    for k in range(int(n_frames)):
        rt_scene = build_rt_scene(scenario, cfg, base_scene=base_scene, frame_idx=k)

        t0 = time.perf_counter()
        native = rt_synthesize_adc(capped, scenario, frame_idx=k, rt_scene=rt_scene,
                                   **common)
        t_native = time.perf_counter() - t0
        n_solves_native += 1

        t0 = time.perf_counter()
        ref = rt_retrace_reference(capped, scenario, frame_idx=k, rt_scene=rt_scene,
                                   n_chirps_cap=n_chirps, **common)
        t_retrace = time.perf_counter() - t0
        n_solves_retrace += n_chirps

        diff = (native - ref)
        den = ref.abs().pow(2).sum().sqrt().item()
        rel_rmse = float(diff.abs().pow(2).sum().sqrt().item() / den) if den > 0 else float("nan")
        per_chirp = []
        for c in range(n_chirps):
            d = float(diff[:, c, :].abs().pow(2).sum().sqrt().item())
            n = float(ref[:, c, :].abs().pow(2).sum().sqrt().item())
            per_chirp.append(d / n if n > 0 else float("nan"))

        # Monte-Carlo floor: same geometry, different solver seed (see the docstring).
        alt = rt_synthesize_adc(capped, scenario, frame_idx=k, rt_scene=rt_scene,
                                **{**common, "solver_seed": solver_seed + 1})
        alt_den = native.abs().pow(2).sum().sqrt().item()
        mc_floor = (float((alt - native).abs().pow(2).sum().sqrt().item() / alt_den)
                    if alt_den > 0 else float("nan"))

        peak_native, rd_native = _rd_peak_bin(capped, native)
        peak_ref, rd_ref = _rd_peak_bin(capped, ref)
        rd_den = rd_ref.abs().pow(2).sum().sqrt().item()
        rd_rel_rmse = float((rd_native - rd_ref).abs().pow(2).sum().sqrt().item() / rd_den) \
            if rd_den > 0 else float("nan")

        t_native_total += t_native
        t_retrace_total += t_retrace
        frames.append({
            "frame_idx": k,
            "rel_rmse": rel_rmse,
            "mc_noise_floor": mc_floor,
            "per_chirp_rel_err": per_chirp,
            "peak_bin_native": tuple(int(v) for v in peak_native),
            "peak_bin_retrace": tuple(int(v) for v in peak_ref),
            "peak_bin_match": tuple(peak_native) == tuple(peak_ref),
            "rd_rel_rmse": rd_rel_rmse,
            "t_native_s": t_native,
            "t_retrace_s": t_retrace,
        })
        if verbose:
            f = frames[-1]
            print(f"  frame {k}: rel_rmse={f['rel_rmse']:.4g} "
                  f"(mc floor {f['mc_noise_floor']:.4g}) "
                  f"rd_rel_rmse={f['rd_rel_rmse']:.4g} "
                  f"peak {f['peak_bin_native']} vs {f['peak_bin_retrace']} "
                  f"({t_native:.2f}s vs {t_retrace:.2f}s)")

    n_ok = sum(1 for f in frames if f["peak_bin_match"])
    return {
        "config": cfg.name,
        "base_scene": base_scene,
        "n_frames": int(n_frames),
        "n_chirps": n_chirps,
        "n_samples": int(cfg.n_samples),
        "frames": frames,
        "rel_rmse_mean": float(np.mean([f["rel_rmse"] for f in frames])) if frames else float("nan"),
        "rel_rmse_max": float(np.max([f["rel_rmse"] for f in frames])) if frames else float("nan"),
        "rd_rel_rmse_mean": float(np.mean([f["rd_rel_rmse"] for f in frames])) if frames else float("nan"),
        "mc_noise_floor_mean": float(np.mean([f["mc_noise_floor"] for f in frames])) if frames else float("nan"),
        "peak_bin_agreement": (n_ok / len(frames)) if frames else float("nan"),
        "t_native_s": t_native_total,
        "t_retrace_s": t_retrace_total,
        "solves_native": n_solves_native,
        "solves_retrace": n_solves_retrace,
        "cost_multiplier": (t_retrace_total / t_native_total) if t_native_total > 0 else float("nan"),
    }


def format_error_study(result: Dict[str, Any]) -> str:
    """Small fixed-width table for `doppler_error_study`'s return value."""
    lines = [
        f"doppler error study: config={result['config']} base_scene={result['base_scene']} "
        f"n_chirps={result['n_chirps']} n_samples={result['n_samples']}",
        f"{'frame':>5} {'rel_rmse':>11} {'mc_floor':>10} {'rd_rel_rmse':>12} "
        f"{'err[c=0]':>10} {'err[c=last]':>12} {'peak(native)':>14} {'peak(retrace)':>15} "
        f"{'t_nat[s]':>9} {'t_ret[s]':>9}",
    ]
    for f in result["frames"]:
        pc = f["per_chirp_rel_err"]
        lines.append(
            f"{f['frame_idx']:>5} {f['rel_rmse']:>11.3e} {f['mc_noise_floor']:>10.2e} "
            f"{f['rd_rel_rmse']:>12.3e} {pc[0]:>10.2e} {pc[-1]:>12.2e} "
            f"{str(f['peak_bin_native']):>14} {str(f['peak_bin_retrace']):>15} "
            f"{f['t_native_s']:>9.2f} {f['t_retrace_s']:>9.2f}"
        )
    lines.append(
        f"mean rel_rmse={result['rel_rmse_mean']:.3e}  max={result['rel_rmse_max']:.3e}  "
        f"rd_rel_rmse={result['rd_rel_rmse_mean']:.3e}  "
        f"mc noise floor={result['mc_noise_floor_mean']:.3e}  "
        f"peak-bin agreement={result['peak_bin_agreement']:.0%}"
    )
    lines.append(
        f"cost: native {result['t_native_s']:.2f}s ({result['solves_native']} solves) vs "
        f"re-trace {result['t_retrace_s']:.2f}s ({result['solves_retrace']} solves) "
        f"-> {result['cost_multiplier']:.1f}x"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------
def _demo_scenario(n_frames: int, cfg=None, target: str = "sphere"):
    """Moving vehicle target(s) in front of a boresight-aimed radar (study scene).

    `Motion.velocity` is a per-FRAME displacement, and `frame_scatterers` converts it
    with `dt = 1/cfg.frame_rate_hz` -- so the m/frame numbers below are chosen to give
    -5 and +3 m/s at the default 10 Hz frame rate, comfortably inside the preset's
    +-12.8 m/s unambiguous Doppler span.

    `target="box"` uses Sionna's box mesh (scaled to a 4x4x2 m slab whose near face is
    perpendicular to the boresight) instead of spheres. That matters: the specular
    path solver finds NO monostatic return off a curved sphere at all (verified -- a
    specular-only solve of a sphere scene returns zero paths), so sphere targets exist
    only through diffuse scattering, which is Monte-Carlo sampled. A planar-faced box
    gives a deterministic specular return and is the right target for measuring the
    Doppler model itself (`--no-diffuse`).
    """
    from e2e.ml.scatterers import vehicle
    from e2e.scenario import Motion, Node, NodeRole, ObjectKind, Scenario, SceneObject

    dt = 1.0 / float(cfg.frame_rate_hz) if cfg is not None else 0.1
    radar = Node(name="radar", role=NodeRole.RADAR, position=(0.0, 0.0, 1.5),
                 look_at=(10.0, 0.0, 1.5))
    if target == "box":
        objects = [SceneObject(name="slab", kind=ObjectKind.BOX, position=(8.0, 0.0, 1.0),
                               scaling=0.4, material="metal", object_class="vehicle",
                               motion=Motion(velocity=(-5.0 * dt, 0.0, 0.0)))]
    else:
        objects = [
            vehicle("car_a", (6.0, 1.5, 1.0), motion=Motion(velocity=(-5.0 * dt, 0.0, 0.0))),
            vehicle("car_b", (4.0, -1.0, 1.0), motion=Motion(velocity=(3.0 * dt, 0.0, 0.0))),
        ]
    return Scenario(
        name="rt_gen_demo",
        base_scene="flat",
        num_frames=max(2, int(n_frames) + 1),   # >= 2 so motion tracks have a step
        nodes=[radar],
        objects=objects,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m e2e.ml.rt_gen",
        description="Ray-traced FMCW ADC generation: native Doppler vs per-chirp re-trace.",
    )
    p.add_argument("--config", default="radial_like",
                   help="radar preset (e2e.ml.radar_config.PRESETS). Defaults to "
                        "radial_like (12 TX x 16 RX = 192 virtual elements): the label "
                        "grid's 192 azimuth bins and the metric's match tolerance are only "
                        "physically answerable at that array size -- see "
                        "e2e.ml.baseline.resolution_report")
    p.add_argument("--frames", type=int, default=2, help="frames to compare")
    p.add_argument("--chirps", type=int, default=16, help="chirps per frame (CPI truncation)")
    p.add_argument("--samples", type=int, default=128, help="ADC samples per chirp")
    p.add_argument("--base-scene", default="flat", help="flat | free | sionna scene name | path")
    p.add_argument("--max-depth", type=int, default=2, help="PathSolver max_depth")
    p.add_argument("--no-diffuse", action="store_true", help="specular-only solve")
    p.add_argument("--target", default="sphere", choices=("sphere", "box"),
                   help="demo target geometry; 'box' is specular-visible (see _demo_scenario)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    import dataclasses as _dc

    args = build_arg_parser().parse_args(argv)

    from e2e.ml.radar_config import PRESETS

    if args.config not in PRESETS:
        print(f"unknown --config {args.config!r}; choices: {sorted(PRESETS)}", file=sys.stderr)
        return 2
    cfg = _dc.replace(PRESETS[args.config], n_chirps=int(args.chirps),
                      n_samples=int(args.samples))
    problems = cfg.validate()
    if problems:
        print(f"invalid derived config: {problems}", file=sys.stderr)
        return 2

    scenario = _demo_scenario(args.frames, cfg, target=args.target)
    result = doppler_error_study(
        cfg, scenario, n_frames=args.frames, base_scene=args.base_scene,
        n_chirps_cap=int(args.chirps), max_depth=args.max_depth,
        diffuse_reflection=not args.no_diffuse, verbose=True,
    )
    print(format_error_study(result))
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
