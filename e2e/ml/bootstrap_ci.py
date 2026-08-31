"""Paired, scene-level bootstrap confidence intervals for AP differences between arms.

Why this module exists
----------------------
Two 95% CIs were published in the README and then WITHDRAWN (commit `0efb7e4`) because
nothing in the repository could recompute them: `compare_detectors` stored a pooled
precision-recall curve with no frame identity, so the curve could not be resampled BY
SCENE, and the only bootstrap it admitted -- iid over the flat detection list -- is the
wrong one. Detections within a scene are not independent: a scene with an easy target
contributes several correlated true positives, and a cluttered scene several correlated
false alarms. Resampling detections rather than scenes therefore reports an interval that
is too narrow, which is the failure mode that matters here because the intervals exist to
say whether an arm's lead is real.

`e2e.ml.metrics` now carries `pr_curve["frame"]` and `gt_per_frame` for exactly this
purpose, so a single scoring pass produces everything needed and no GPU re-run is
required. This module consumes that artifact.

What "paired" means and why it is not optional
-----------------------------------------------
Every arm is scored on the SAME frames, so their AP estimates are positively correlated:
a resample that happens to draw easy scenes lifts all arms together. An unpaired interval
around the DIFFERENCE ignores that shared variance and is far too wide. Each bootstrap
iteration here draws one scene multiset and recomputes EVERY arm's AP on that same
multiset, so the correlation is preserved and the interval is around the paired
difference.

The AP recomputed per resample is the same all-points interpolated AP `e2e.ml.metrics`
defines, including its pessimistic tie-break (equal scores rank false positives first).
`verify_against_stored` asserts that: on the identity resample this module must reproduce
each arm's stored AP to floating-point tolerance, and the CLI runs that check before
reporting any interval. A bootstrap built on a subtly different AP would produce
confident, wrong intervals.

CLI
---
    python -m e2e.ml.bootstrap_ci --compare e2e/ml/runs/compare.json \\
        [--baseline "classical CFAR"] [--n-boot 2000] [--seed 0] [--alpha 0.05]

With no `--baseline`, every arm is compared against the FIRST arm in the file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class MissingFrameProvenance(RuntimeError):
    """The stored curve predates `pr_curve["frame"]` and cannot be resampled by scene."""


def _arm_arrays(arm: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(score, is_tp, frame)` as numpy arrays for one arm's stored PR curve."""
    curve = arm.get("pr_curve")
    if not curve:
        raise MissingFrameProvenance(
            f"arm {arm.get('name')!r} has no pr_curve; re-score it with a current "
            "compare_detectors")
    if "frame" not in curve:
        raise MissingFrameProvenance(
            f"arm {arm.get('name')!r} stores a pr_curve with no 'frame' list. It was "
            "produced before frame provenance was recorded, so it cannot be resampled by "
            "scene -- and an iid bootstrap over pooled detections is NOT a valid "
            "substitute (see this module's docstring). Re-score the corpus.")
    return (np.asarray(curve["score"], dtype=np.float64),
            np.asarray(curve["is_tp"], dtype=bool),
            np.asarray(curve["frame"], dtype=np.int64))


def average_precision(score: np.ndarray, is_tp: np.ndarray, n_gt: float) -> float:
    """All-points interpolated AP, matching `e2e.ml.metrics` exactly.

    Ties break pessimistically: among equal scores, false positives rank first, so a
    detector cannot harvest AP from an ordering it did not produce. `np.lexsort` sorts by
    the LAST key first, so the primary key (-score) is given last.
    """
    if n_gt <= 0:
        return float("nan")
    if score.size == 0:
        return 0.0
    order = np.lexsort((is_tp, -score))
    hits = is_tp[order]
    ranks = np.arange(1, hits.size + 1, dtype=np.float64)
    precision = np.cumsum(hits) / ranks
    # Right-to-left running max == precision made monotonically non-increasing in recall.
    interp = np.maximum.accumulate(precision[::-1])[::-1]
    return float(interp[hits].sum() / n_gt)


class _ArmResampler:
    """Pre-groups one arm's detections by frame so a resample is a gather, not a scan."""

    def __init__(self, arm: Dict, n_frames: int):
        score, is_tp, frame = _arm_arrays(arm)
        self.name = arm.get("name", "?")
        self.score, self.is_tp = score, is_tp
        order = np.argsort(frame, kind="stable")
        self._sorted_idx = order
        # Start offset of each frame's block within `_sorted_idx`.
        self._starts = np.searchsorted(frame[order], np.arange(n_frames + 1))

    def indices_for(self, frames: np.ndarray) -> np.ndarray:
        counts = self._starts[frames + 1] - self._starts[frames]
        if not counts.any():
            return np.empty(0, dtype=np.int64)
        # Expand each drawn frame's contiguous block; a frame drawn k times contributes
        # its detections k times, which is what makes this a bootstrap over scenes.
        total = int(counts.sum())
        out = np.empty(total, dtype=np.int64)
        pos = 0
        for f, c in zip(frames.tolist(), counts.tolist()):
            if c:
                out[pos:pos + c] = self._sorted_idx[self._starts[f]:self._starts[f] + c]
                pos += c
        return out

    def ap_for(self, frames: np.ndarray, n_gt: float) -> float:
        idx = self.indices_for(frames)
        return average_precision(self.score[idx], self.is_tp[idx], n_gt)


def verify_against_stored(arms: Sequence[Dict], gt_per_frame: Sequence[int],
                          *, tol: float = 1e-9) -> List[Tuple[str, float, float]]:
    """Identity-resample oracle: recomputed AP must equal each arm's stored AP.

    Returns `[(name, stored, recomputed), ...]`; raises if any pair differs by > `tol`.
    This is what licenses every interval below -- without it the bootstrap could be
    measuring a slightly different estimator than the one being reported.
    """
    n_frames = len(gt_per_frame)
    all_frames = np.arange(n_frames, dtype=np.int64)
    n_gt = float(np.sum(gt_per_frame))
    rows, bad = [], []
    for arm in arms:
        r = _ArmResampler(arm, n_frames)
        got = r.ap_for(all_frames, n_gt)
        stored = float(arm["AP"])
        rows.append((r.name, stored, got))
        if not np.isfinite(got) or abs(got - stored) > tol:
            bad.append((r.name, stored, got))
    if bad:
        detail = "; ".join(f"{n}: stored {s!r} vs recomputed {g!r}" for n, s, g in bad)
        raise AssertionError(
            "bootstrap AP does not reproduce the stored AP on the identity resample -- "
            f"the estimator has diverged from e2e.ml.metrics and no interval below would "
            f"be trustworthy. {detail}")
    return rows


def paired_bootstrap(compare: Dict, *, baseline: Optional[str] = None,
                     n_boot: int = 2000, seed: int = 0,
                     alpha: float = 0.05) -> Dict:
    """Paired scene-level bootstrap of `AP(arm) - AP(baseline)` for every other arm."""
    arms = compare["arms"]
    if len(arms) < 2:
        raise ValueError("need at least two arms to form a difference")
    gt_per_frame = compare.get("gt_per_frame") or arms[0].get("gt_per_frame")
    if not gt_per_frame:
        raise MissingFrameProvenance(
            "no gt_per_frame recorded; the recall denominator cannot be resampled. "
            "Re-score with a current compare_detectors.")

    names = [a.get("name", f"arm{i}") for i, a in enumerate(arms)]
    base_i = 0 if baseline is None else names.index(baseline)

    checks = verify_against_stored(arms, gt_per_frame)

    n_frames = len(gt_per_frame)
    gt = np.asarray(gt_per_frame, dtype=np.float64)
    resamplers = [_ArmResampler(a, n_frames) for a in arms]

    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, len(arms)), dtype=np.float64)
    for b in range(n_boot):
        frames = rng.integers(0, n_frames, size=n_frames)
        n_gt_b = float(gt[frames].sum())
        for j, r in enumerate(resamplers):
            draws[b, j] = r.ap_for(frames, n_gt_b)

    lo_q, hi_q = 100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)
    results = []
    for j, name in enumerate(names):
        if j == base_i:
            continue
        diff = draws[:, j] - draws[:, base_i]
        finite = diff[np.isfinite(diff)]
        lo, hi = np.percentile(finite, [lo_q, hi_q])
        results.append({
            "arm": name,
            "baseline": names[base_i],
            "delta_AP": float(arms[j]["AP"]) - float(arms[base_i]["AP"]),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "excludes_zero": bool(lo > 0.0 or hi < 0.0),
            "n_resamples": int(finite.size),
        })
    return {
        "baseline": names[base_i],
        "n_frames": n_frames,
        "n_boot": n_boot,
        "seed": seed,
        "alpha": alpha,
        "unit": "scene (paired across arms)",
        "identity_check": [{"arm": n, "stored_AP": s, "recomputed_AP": g}
                           for n, s, g in checks],
        "comparisons": results,
    }


def format_report(result: Dict) -> str:
    lines = [
        f"paired scene-level bootstrap  |  {result['n_boot']} resamples of "
        f"{result['n_frames']} scenes  |  seed {result['seed']}",
        f"baseline: {result['baseline']}",
        "",
        f"{'arm':<28}{'dAP':>9}{'  ' + str(int((1 - result['alpha']) * 100)) + '% CI':>20}"
        f"{'sig':>6}",
    ]
    for c in result["comparisons"]:
        ci = f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]"
        lines.append(f"{c['arm']:<28}{c['delta_AP']:>+9.4f}{ci:>20}"
                     f"{('yes' if c['excludes_zero'] else 'no'):>6}")
    lines += ["", "'sig' = the interval excludes zero. Resampling unit is the SCENE, "
                  "paired across arms;", "detections within a scene are correlated, so a "
                  "per-detection bootstrap would be too narrow."]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m e2e.ml.bootstrap_ci",
        description="Paired scene-level bootstrap CIs on AP differences, from a stored "
                    "compare_detectors JSON.")
    p.add_argument("--compare", required=True, help="a compare_detectors --out JSON")
    p.add_argument("--baseline", default=None,
                   help="arm name to difference against (default: the first arm)")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.05, help="0.05 -> 95%% CI")
    p.add_argument("--out", default=None, help="write the full result dict as JSON here")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    compare = json.loads(Path(args.compare).read_text())
    try:
        result = paired_bootstrap(compare, baseline=args.baseline, n_boot=args.n_boot,
                                  seed=args.seed, alpha=args.alpha)
    except MissingFrameProvenance as exc:
        print(f"cannot compute intervals: {exc}", file=sys.stderr)
        return 2
    print(format_report(result))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
