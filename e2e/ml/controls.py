"""The controls a reviewer will demand before "beats CFAR" is said out loud.

    python -m e2e.ml.controls --checkpoint raddetnet=e2e/ml/runs/b7_raddetnet/best.pt \
        [--checkpoint fftradnet_rd_b5=e2e/ml/runs/b5_fftradnet_v3/best.pt ...]

Every number in notes/ESTABLISHED_FACTS.md F83 ("the learned detectors never learn
azimuth") came from scratch scripts that no longer exist; when RADDetNet scored 0.476
against CFAR's 0.301 on 2026-09-22 there was no committed way to ask the same questions of
it. This module asks them, on the SAME test frames and the SAME scoring protocol as
`e2e.ml.beat_cfar` (score floor 0.01, 40 m crop), for any checkpoint:

  AP           the headline, under the protocol.
  deranged     AP when each prediction is scored against a DIFFERENT frame's labels
               (a cyclic shift of the target lists). A detector that reads the radar
               data keeps ~10% (chance overlap of targets between frames, F83 measured
               10.5% for CFAR); one that emits a memorised prior keeps ~50% (F83: 48-51%
               for the shipped nets). Reported as retention = AP_deranged / AP.
  az-only      AP under azimuth-only matching (range tolerance infinite). Beside it, the
               same score for a CONSTANT map -- the mean of the model's own predictions
               over the split, identical for every frame. If the model does not beat its
               own average, its azimuth axis carries no frame-specific information (F83:
               0.421 vs 0.423).
  range-only   the mirror image (azimuth tolerance infinite), with the same constant map.
  stripe       median rank-1 energy fraction of the objectness map (ground truth 0.312;
               the shipped nets 0.89 / 0.76). `e2e.ml.beat_cfar._stripe_statistic`.

The calibration that makes this trustworthy: run it on `b5_fftradnet_v3` and compare with
F83's numbers (deranged 48.4%, az-only 0.421 vs 0.423, stripe 0.89). Agreement means the
implementation matches the one those facts came from; disagreement means F83 or this
module is wrong, and either way the next step is to find out which, not to quote.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from e2e.ml.beat_cfar import DECODE_THRESHOLD, MANIFEST, MAX_RANGE_M, SPLIT

_INF = 1e9


def _predictions(checkpoint: str, device, *, manifest=MANIFEST, split=SPLIT):
    from e2e.ml.train import _make_dataset, _predict_split, load_model_for_eval
    model, _man, grid, fmt = load_model_for_eval(manifest, checkpoint, device=device)
    ds = _make_dataset(manifest, split, fmt)
    preds = _predict_split(model, ds, device=device, batch_size=8)
    targets = [ds.targets(i) for i in range(len(ds))]
    del model
    return preds, targets, grid


def _ap(preds, targets, grid, *, criterion=None, max_range_m=MAX_RANGE_M) -> float:
    from e2e.ml.metrics import evaluate_dataset
    kw = dict(score_threshold=DECODE_THRESHOLD, max_range_m=max_range_m)
    if criterion is not None:
        kw["criterion"] = criterion
    return float(evaluate_dataset(preds, targets, grid, **kw)["AP"])


def _train_label_prior(manifest, device):
    """Mean of the TRAIN split's label maps, `[3, R, A]` -- the strongest frame-independent
    prior available: a detector that memorised where targets usually are."""
    import torch
    from e2e.ml.dataset import RadarFrameDataset
    ds = RadarFrameDataset(manifest, split="train")
    if len(ds) == 0:
        raise ValueError("train split is empty")
    acc = None
    for i in range(len(ds)):
        y = ds[i][1].to(torch.float32)
        acc = y.clone() if acc is None else acc + y
    return (acc / len(ds)).cpu()


def _stripe(preds, limit: int = 40) -> float:
    """Median rank-1 energy fraction of the objectness maps (same statistic as
    `beat_cfar._stripe_statistic`, computed on predictions already in hand)."""
    import numpy as np
    vals = []
    for p in preds[:limit]:
        m = p[0].detach().cpu().numpy().astype(np.float64)
        s = np.linalg.svd(m, compute_uv=False)
        vals.append(float(s[0] ** 2 / (s ** 2).sum()))
    return float(np.median(vals))


def controls_for(checkpoint: str, device=None, *, manifest=MANIFEST, split=SPLIT,
                 max_range_m=MAX_RANGE_M) -> Dict[str, float]:
    """All the controls for one checkpoint; `manifest`/`split`/`max_range_m` default to
    the beat_cfar protocol and exist so a tiny fixture can exercise the same code."""
    import torch
    from e2e.ml.metrics import MatchCriterion

    if device is None:
        from e2e.ml.compare_detectors import _default_device
        device = _default_device()
    preds, targets, grid = _predictions(checkpoint, device, manifest=manifest, split=split)
    n = len(preds)
    if n < 2:
        raise ValueError("the deranged-label control needs at least two frames")
    ap = _ap(preds, targets, grid, max_range_m=max_range_m)
    # Deranged: prediction i is scored against frame (i+1) mod n's labels. A single cyclic
    # shift is a derangement for n >= 2, and deterministic.
    deranged = _ap(preds, targets[1:] + targets[:1], grid, max_range_m=max_range_m)
    az_only = MatchCriterion(max_range_err_m=_INF, max_sin_az_err=0.06)
    range_only = MatchCriterion(max_range_err_m=2.0, max_sin_az_err=_INF)
    constant = torch.stack([p for p in preds]).mean(dim=0)
    constant_preds = [constant] * n
    # A STRONGER frame-independent prior than the model's own mean (independent verifier,
    # 2026-09-22): the mean of the TRAIN split's label maps -- what a detector that had
    # memorised where targets usually are would emit. For a peaky model its own mean map
    # is a weak prior and flatters the az-only margin (RADDetNet: 0.657 vs 0.285 against
    # its own mean, 0.657 vs 0.472 against this). Reported beside it, never instead.
    prior_preds = None
    try:
        prior = _train_label_prior(manifest, device)
        prior_preds = [prior] * n
    except Exception as e:      # a fixture manifest may have no train split
        print(f"  (train-label prior unavailable: {e})")
    out = {
        "AP": ap,
        "deranged_AP": deranged,
        "deranged_retention": (deranged / ap) if ap > 0 else float("nan"),
        "az_only_AP": _ap(preds, targets, grid, criterion=az_only, max_range_m=max_range_m),
        "az_only_constant_AP": _ap(constant_preds, targets, grid, criterion=az_only,
                                   max_range_m=max_range_m),
        "range_only_AP": _ap(preds, targets, grid, criterion=range_only,
                             max_range_m=max_range_m),
        "range_only_constant_AP": _ap(constant_preds, targets, grid, criterion=range_only,
                                      max_range_m=max_range_m),
        # Over EVERY frame of the split (beat_cfar's print uses the first 40; the verifier
        # noted the difference, 0.617 vs 0.600 for RADDetNet -- immaterial, but say which).
        "stripe_rank1": _stripe(preds, limit=n),
    }
    if prior_preds is not None:
        out["az_only_train_prior_AP"] = _ap(prior_preds, targets, grid, criterion=az_only,
                                            max_range_m=max_range_m)
        out["range_only_train_prior_AP"] = _ap(prior_preds, targets, grid,
                                               criterion=range_only, max_range_m=max_range_m)
        out["train_prior_AP"] = _ap(prior_preds, targets, grid, max_range_m=max_range_m)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH",
                   help="may repeat")
    p.add_argument("--out", default="e2e/ml/runs/controls.json")
    p.add_argument("--manifest", default=MANIFEST,
                   help="corpus manifest to score on (default: beat_cfar's, b1_bench_v3). "
                        "The 2026-09-22 joint-arm controls were run on v3 only because this "
                        "flag did not exist; the artifact must name its corpus.")
    p.add_argument("--split", default=SPLIT)
    args = p.parse_args(argv)
    if not args.checkpoint:
        p.error("at least one --checkpoint NAME=PATH")
    print(f"protocol: manifest={args.manifest} split={args.split} "
          f"decode={DECODE_THRESHOLD} max_range_m={MAX_RANGE_M}")
    print(f"{'checkpoint':20s} {'AP':>6s} {'derang':>7s} {'keep%':>6s} {'az-only':>8s} "
          f"{'az-const':>8s} {'az-prior':>8s} {'rng-only':>8s} {'rng-const':>9s} {'stripe':>7s}")
    results = {}
    for spec in args.checkpoint:
        name, path = spec.split("=", 1)
        if not Path(path).is_file():
            print(f"{name:20s} missing: {path}")
            continue
        r = controls_for(path, manifest=args.manifest, split=args.split)
        results[name] = {"checkpoint": path, **r}
        prior = r.get("az_only_train_prior_AP", float("nan"))
        print(f"{name:20s} {r['AP']:6.3f} {r['deranged_AP']:7.3f} "
              f"{100 * r['deranged_retention']:5.1f}% {r['az_only_AP']:8.3f} "
              f"{r['az_only_constant_AP']:8.3f} {prior:8.3f} {r['range_only_AP']:8.3f} "
              f"{r['range_only_constant_AP']:9.3f} {r['stripe_rank1']:7.3f}")
    Path(args.out).write_text(json.dumps({
        "protocol": {"manifest": args.manifest, "split": args.split,
                     "decode_threshold": DECODE_THRESHOLD, "max_range_m": MAX_RANGE_M},
        "reference_F83": {"deranged_retention_shipped_nets": "48-51%",
                          "deranged_retention_cfar": "10.5%",
                          "az_only_model_vs_constant": "0.421 vs 0.423",
                          "stripe_ground_truth": 0.312},
        "results": results,
    }, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
