"""Re-certify a checkpoint against the CURRENT pipeline code, and re-stamp its fingerprint.

    python -m e2e.ml.recertify e2e/ml/runs/b8_fftradnet_rad_frontend [--tol 2e-3] [--dry-run]

WHY THIS EXISTS. `e2e.ml.train` records a fingerprint of the input-pipeline sources in
every checkpoint, and `e2e.ml.beat_cfar` refuses to skip retraining when it no longer
matches (F84: two results died of a mid-run edit). That guard is deliberately blind to
WHAT changed: it cannot tell a behaviour-preserving refactor of `dataset.py` from the
front-end change that caused F84, so after either one every existing checkpoint reads as
stale and the next `beat_cfar` would spend 5 GPU-hours retraining models that are fine.

The honest way out is not to guess which edits were inert; it is to MEASURE. A checkpoint
is certified for the current code if, and only if, reloading it and re-running its own
validation split reproduces the `best_val_AP` it recorded, to within `--tol`. That is the
exact test that caught F84 (recorded 0.484, reproduced 0.023) turned into a tool. On
success the checkpoint's `pipeline_fingerprint` is rewritten to the current one, so the
guard stops flagging it; on failure nothing is written and the process exits 1 with both
numbers, because a checkpoint that does not reproduce its own metric is exactly the thing
the guard exists to stop.

Same protocol as training's per-epoch validation: `_evaluate_split` on the manifest's
"val" split with autocast on CUDA (`amp="auto"`), which is what produced the recorded
number. Tolerance defaults to 2e-3 AP -- the pipeline's own run-to-run noise is ~1e-3
under the non-strict path and 0 under `--deterministic`; F84-class breakage is ~0.4.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional


def recertify(run_dir, *, tol: float = 2e-3, dry_run: bool = False, device=None) -> dict:
    """Returns a dict with `recorded`, `reproduced`, `delta`, `passed`, `stamped`."""
    import torch
    from e2e.ml.train import (_default_device, _evaluate_split, _make_dataset,
                              load_model_for_eval, pipeline_fingerprint)

    run_dir = Path(run_dir)
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"no checkpoint at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    recorded = ckpt.get("best_val_AP")
    if recorded is None:
        raise ValueError(f"{ckpt_path} records no best_val_AP; nothing to certify against")
    manifest_path = ckpt.get("manifest")
    if manifest_path is None or not Path(manifest_path).is_file():
        raise FileNotFoundError(f"{ckpt_path} names manifest {manifest_path!r}, which is missing")

    dev = device if device is not None else _default_device()
    model, _manifest, grid, fmt = load_model_for_eval(manifest_path, str(ckpt_path), device=dev)
    ds = _make_dataset(manifest_path, "val", fmt)
    use_amp = dev.type == "cuda"
    metrics = _evaluate_split(model, ds, grid, device=dev, batch_size=8, amp=use_amp)
    reproduced = float(metrics["AP"])
    delta = reproduced - float(recorded)
    passed = abs(delta) <= tol

    current_fp = pipeline_fingerprint()
    stamped = False
    if passed and not dry_run and ckpt.get("pipeline_fingerprint") != current_fp:
        ckpt["pipeline_fingerprint"] = current_fp
        history = ckpt.setdefault("recertified", [])
        history.append({"reproduced_val_AP": reproduced, "recorded_val_AP": float(recorded),
                        "tol": tol, "fingerprint": current_fp})
        tmp = ckpt_path.with_suffix(".pt.tmp")
        torch.save(ckpt, tmp)
        os.replace(tmp, ckpt_path)   # atomic, like train.py's own writes
        stamped = True
    return {"run_dir": str(run_dir), "recorded": float(recorded), "reproduced": reproduced,
            "delta": delta, "tol": tol, "passed": passed, "stamped": stamped,
            "fingerprint": current_fp}


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_dirs", nargs="+", help="run directories holding best.pt")
    p.add_argument("--tol", type=float, default=2e-3, help="max |reproduced - recorded| AP")
    p.add_argument("--dry-run", action="store_true", help="measure, never write")
    args = p.parse_args(argv)
    rc = 0
    for rd in args.run_dirs:
        r = recertify(rd, tol=args.tol, dry_run=args.dry_run)
        verdict = "PASS" if r["passed"] else "FAIL"
        action = ("re-stamped" if r["stamped"] else
                  ("dry run" if args.dry_run else "already current" if r["passed"] else "NOT written"))
        print(f"[{verdict}] {rd}: recorded val_AP {r['recorded']:.4f}, reproduced "
              f"{r['reproduced']:.4f} (delta {r['delta']:+.4f}, tol {r['tol']}) -- {action}")
        if not r["passed"]:
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
