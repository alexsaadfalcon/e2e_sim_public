"""
Loss-balance / hyperparameter sweep driver for `e2e.ml` radar detection models.

Implements the panel's sweep design (see the low-AP diagnosis: `reg_loss` is
normalized per-positive-cell then multiplied by `reg_weight=100` while `cls_loss` is a
raw batch-sum over ~tens of thousands of cells -- the optimizer's cheapest path to lower
total loss is precise regression on the few positive cells while classification
confidence on those same cells stays low, which shows up as a **recall (`val_AR`)
collapse that gets worse with more epochs**, not underfitting). `e2e.ml.train.train()`
now exposes `reg_weight`/`gamma`/`lr` and logs the per-term `train_cls_loss`/
`train_reg_loss` breakdown into `history.json` -- this module sweeps those knobs.

Grid / budget (verbatim from the design doc, FFTRadNet-first)
---------------------------------------------------------------
`batch_size` is held FIXED across every trial -- loss magnitude (a raw batch-sum) is
batch-size-dependent, so varying it would make `reg_weight`/`lr` comparisons across
trials invalid.

  * Stage 1 (`DEFAULT_GRID["reg_weight"] x DEFAULT_GRID["lr"]`, 4x2=8 trials,
    `gamma` held at `DEFAULT_GRID["gamma_default"]=2.0`, the current default):
    `reg_weight in {1, 10, 30, 100}` (100 is the current/upstream default -- primary
    axis, per the diagnosis above), `lr in {1e-4 (current default), 3e-4}` (secondary
    axis: a lower `reg_weight` shrinks the total-loss magnitude and may want a higher
    `lr` to compensate).
  * Stage 2 (2 more trials, run only after stage 1 finishes): `gamma in {0 (BCE
    fallback), 2 (current default)}`, crossed with the **winning stage-1 trial's**
    `reg_weight` AND `lr` (not a full 4x2x2=16 grid) -- "winner" here means the whole
    winning (`reg_weight`, `lr`) pair, not `reg_weight` alone, since that pair is what
    the stage-1 objective actually ranks.
  * 10 trials total for FFTRadNet. Note the `gamma=2.0` stage-2 point exactly repeats
    the stage-1 winner's params -- it is not re-trained, `run_sweep`'s resume/skip
    mechanism (below) picks up the already-written `history.json` for that trial dir.
  * Epochs/trial: 25 default (`--epochs`), matching the design doc's budget (the
    AR-collapse signature is visible by epoch ~10-15 in the reference `fft_long`/
    `ssm_long` runs, well within 25).

Objective + selection rule (design doc S3) -- `pick_best()`
-------------------------------------------------------------
  * Primary objective: mean of `val_AP` over the trailing `min(5, epochs)` epochs
    (`objective_mean_ap_last5` in each trial record) -- NOT single-epoch-best AP, which
    is exactly the noisy quantity that produced spurious spikes in the reference runs.
  * Guard: a trial is `ar_declining=True` if `val_AR` is strictly, monotonically
    decreasing across every consecutive pair in that same trailing window (the
    AR-collapse signature reproducing) -- `pick_best` excludes `ar_declining` trials
    from consideration UNLESS every trial is flagged (all trials reproduce the bug),
    in which case it falls back to ranking the full pool by the objective anyway so
    callers always get an answer, and callers should treat that outcome as "sweep did
    not find a fix", not silently trust it.
  * Tie-break: higher `final_val_AR` among equal-objective trials.
  * NOT implemented here (documented gap, not silently dropped): the design doc's
    epoch-10 early-kill rule ("if val_AR at epoch 10 is already below its epoch-1 value
    AND still falling, kill the trial") needs a mid-training callback/early-stop hook
    inside `e2e.ml.train.train()`, which this shard does not own and `train()` does not
    currently expose. Every trial here runs the full `epochs` requested; this only
    costs wall-clock time, not correctness of the selection above.

Confirmation protocol (design doc S5, informational -- not automated by this module)
----------------------------------------------------------------------------------------
After `pick_best`, the design calls for one full-length (60-80ep) run at the winning
config on a larger tier plus a held-out **test**-split `e2e.ml.train.evaluate()` call
(val was used for selection), and a "ship as new defaults" bar requiring the win to
reproduce on BOTH FFTRadNet (tuned directly) and SSMRadNet (2-3 spot-check transfer
trials) before touching `losses.py`'s shipped `reg_weight=100, gamma=2` defaults --
otherwise document-don't-default. This module only runs the sweep + selection; the
confirmation run and the losses.py default change are separate, deliberate follow-ups.

Artifacts
---------
`run_sweep(...)` writes `<out_dir>/sweep_results.json` (trials sorted descending by
`objective_mean_ap_last5`) and each trial's own `history.json`/`best.pt` under
`<out_dir>/<trial_slug>/` (via `e2e.ml.train.train(..., out_dir=<trial_slug dir>)`),
mirroring `train.py`'s own artifact-layout convention. `trial_slug` is deterministic
from `(reg_weight, lr, gamma)`, so re-running `run_sweep` with the same grid/out_dir
resumes: any trial whose `<slug>/history.json` already exists is loaded from disk
instead of re-trained.

CLI
---
    python -m e2e.ml.sweep --manifest PATH --model fftradnet|ssmradnet [--epochs 25]
        [--batch-size 8] [--seed 0] [--out DIR] [--grid-json '{"reg_weight": [...],
        "lr": [...], "gamma_default": 2.0, "stage2_gamma": [...]}'] [--dry-run]

`--grid-json` overrides `DEFAULT_GRID` wholesale; the JSON object must have
`"reg_weight"`/`"lr"`/`"stage2_gamma"` (lists) and may omit `"gamma_default"` (defaults
to 2.0). `--dry-run` prints the (deterministic) stage-1 trial list, notes stage 2's
trial count (its exact params are only known after stage 1 finishes), and prints a
rough wall-clock estimate -- WITHOUT calling `train()` -- then exits 0.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from e2e.ml.train import train

_MODEL_NAMES = ("fftradnet", "ssmradnet")

# Design doc S2 "Grid (FFTRadNet-first)" -- verbatim numbers.
DEFAULT_GRID: Dict = {
    "reg_weight": [1, 10, 30, 100],
    "lr": [1e-4, 3e-4],
    "gamma_default": 2.0,
    "stage2_gamma": [0.0, 2.0],
}

# Design doc S2 budget: "~6 min/trial" for 25 epochs on the D1 tier -> ~14.4 s/epoch.
# A rough figure quoted in the design doc, not measured on this box; --dry-run's
# estimate is explicitly labeled as such.
_SEC_PER_EPOCH_ESTIMATE = 14.4

# Objective/guard trailing window (design doc S3: "mean of val_AP over the last 5
# epochs").
_OBJECTIVE_WINDOW = 5


# --------------------------------------------------------------------------------
# Grid expansion
# --------------------------------------------------------------------------------
def _validate_grid(grid: Dict) -> Dict:
    required = {"reg_weight", "lr", "stage2_gamma"}
    missing = required - set(grid)
    if missing:
        raise ValueError(f"grid is missing required key(s): {sorted(missing)}")
    grid = dict(grid)
    grid.setdefault("gamma_default", 2.0)
    return grid


def _stage1_trials(grid: Dict) -> List[Dict]:
    """Stage-1 `reg_weight x lr` cartesian product, `gamma` fixed at `gamma_default`.

    Iteration order: outer loop `reg_weight`, inner loop `lr` (matches the design
    doc's own listing order) -- deterministic, so `--dry-run` output and the actual
    training order agree.
    """
    gamma = float(grid.get("gamma_default", 2.0))
    return [
        {"reg_weight": float(rw), "lr": float(lr), "gamma": gamma}
        for rw in grid["reg_weight"]
        for lr in grid["lr"]
    ]


def _stage2_trials(grid: Dict, winner_params: Dict) -> List[Dict]:
    """Stage-2 `gamma` sweep crossed with the stage-1 winner's `reg_weight`/`lr`."""
    return [
        {"reg_weight": winner_params["reg_weight"], "lr": winner_params["lr"], "gamma": float(g)}
        for g in grid["stage2_gamma"]
    ]


def trial_slug(params: Dict) -> str:
    """Deterministic directory-name-safe slug for `(reg_weight, lr, gamma)`.

    Public (not `_`-prefixed) so callers/tests can pre-seed `<out_dir>/<slug>/
    history.json` to exercise the resume-skip path with the exact name `run_sweep`
    itself would use.
    """
    return f"rw{params['reg_weight']:g}_lr{params['lr']:g}_g{params['gamma']:g}"


# --------------------------------------------------------------------------------
# Per-trial metrics
# --------------------------------------------------------------------------------
def _mean_last_n(values: Sequence[float], n: int = _OBJECTIVE_WINDOW) -> float:
    window = list(values[-n:])
    return sum(window) / len(window) if window else float("nan")


def _is_ar_declining(val_ar: Sequence[float], n: int = _OBJECTIVE_WINDOW) -> bool:
    """True iff `val_ar`'s trailing `n`-epoch window is strictly monotonically falling.

    Fewer than 2 points in the window means no trend is determinable -> False.
    """
    window = list(val_ar[-n:])
    if len(window) < 2:
        return False
    return all(window[i] > window[i + 1] for i in range(len(window) - 1))


def _summarize_trial(params: Dict, history: Dict, *, stage: str, out_dir: Path,
                      wall_s: float, resumed: bool) -> Dict:
    val_ap = history.get("val_AP", [])
    val_ar = history.get("val_AR", [])
    cls_hist = history.get("train_cls_loss", [])
    reg_hist = history.get("train_reg_loss", [])
    return {
        "stage": stage,
        "params": {"reg_weight": params["reg_weight"], "lr": params["lr"],
                   "gamma": params["gamma"]},
        "out_dir": str(out_dir),
        "best_val_AP": max(val_ap) if val_ap else float("nan"),
        "final_val_AP": val_ap[-1] if val_ap else float("nan"),
        "final_val_AR": val_ar[-1] if val_ar else float("nan"),
        "final_train_cls_loss": cls_hist[-1] if cls_hist else float("nan"),
        "final_train_reg_loss": reg_hist[-1] if reg_hist else float("nan"),
        "objective_mean_ap_last5": _mean_last_n(val_ap),
        "ar_declining": _is_ar_declining(val_ar),
        "wall_s": wall_s,
        "resumed": resumed,
    }


def _run_trial(manifest_path, model_name: str, params: Dict, *, epochs: int, batch_size: int,
                seed: int, sweep_out_dir: Path, device, stage: str) -> Dict:
    trial_out = sweep_out_dir / trial_slug(params)
    history_path = trial_out / "history.json"

    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        return _summarize_trial(params, history, stage=stage, out_dir=trial_out,
                                 wall_s=0.0, resumed=True)

    trial_out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    history = train(manifest_path, model_name, epochs=epochs, batch_size=batch_size, seed=seed,
                     out_dir=trial_out, device=device, reg_weight=params["reg_weight"],
                     lr=params["lr"], gamma=params["gamma"])
    wall_s = time.perf_counter() - t0

    # Belt-and-suspenders: the real train() already writes this, but re-writing our
    # own copy of its return value means the resume path works even against a
    # train() stand-in that doesn't do its own file I/O (e.g. a test stub).
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return _summarize_trial(params, history, stage=stage, out_dir=trial_out, wall_s=wall_s,
                             resumed=False)


# --------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------
def pick_best(sweep_results: Union[Dict, Sequence[Dict]]) -> Dict:
    """Design doc S3 selection rule over trial records (see module docstring).

    `sweep_results` is either the `run_sweep`-written payload (`dict` with a
    `"trials"` list) or that list directly.
    """
    trials = sweep_results["trials"] if isinstance(sweep_results, dict) else list(sweep_results)
    if not trials:
        raise ValueError("pick_best: no trials to select from")

    candidates = [t for t in trials if not t.get("ar_declining", False)]
    pool = candidates if candidates else trials  # all declining -> still return an answer
    return max(pool, key=lambda t: (t["objective_mean_ap_last5"], t.get("final_val_AR", float("-inf"))))


# --------------------------------------------------------------------------------
# Sweep driver
# --------------------------------------------------------------------------------
def run_sweep(manifest_path, model_name: str, grid: Optional[Dict] = None, *, epochs: int = 25,
              batch_size: int = 8, seed: int = 0, out_dir=None, device=None) -> Path:
    """Run the design doc's staged sweep sequentially; write/return `sweep_results.json`.

    `grid` defaults to `DEFAULT_GRID`. `out_dir` defaults to
    `<manifest's directory>/runs/sweep_<model_name>/`, mirroring `train.py`'s own
    default `out_dir` convention.
    """
    if model_name not in _MODEL_NAMES:
        raise ValueError(f"unknown model {model_name!r}; choices: {_MODEL_NAMES}")
    manifest_path = Path(manifest_path)
    grid = _validate_grid(grid if grid is not None else DEFAULT_GRID)
    out_dir = Path(out_dir) if out_dir is not None else manifest_path.parent / "runs" / f"sweep_{model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []
    for params in _stage1_trials(grid):
        results.append(_run_trial(manifest_path, model_name, params, epochs=epochs,
                                   batch_size=batch_size, seed=seed, sweep_out_dir=out_dir,
                                   device=device, stage="stage1"))

    winner = pick_best(results)
    for params in _stage2_trials(grid, winner["params"]):
        results.append(_run_trial(manifest_path, model_name, params, epochs=epochs,
                                   batch_size=batch_size, seed=seed, sweep_out_dir=out_dir,
                                   device=device, stage="stage2"))

    results.sort(key=lambda r: r["objective_mean_ap_last5"], reverse=True)

    payload = {
        "manifest": str(manifest_path),
        "model": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "grid": grid,
        "trials": results,
    }
    results_path = out_dir / "sweep_results.json"
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)

    _print_leaderboard(results)
    return results_path


# --------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------
def _print_leaderboard(results: Sequence[Dict]) -> None:
    header = (f"{'rank':>4}  {'stage':<6} {'reg_weight':>10} {'lr':>8} {'gamma':>6} "
              f"{'obj(AP last5)':>14} {'final_AR':>9} {'AR_decl':>8} {'resumed':>8}")
    print(header)
    print("-" * len(header))
    for rank, r in enumerate(results, start=1):
        p = r["params"]
        print(f"{rank:>4}  {r['stage']:<6} {p['reg_weight']:>10g} {p['lr']:>8g} {p['gamma']:>6g} "
              f"{r['objective_mean_ap_last5']:>14.4f} {r['final_val_AR']:>9.4f} "
              f"{str(r['ar_declining']):>8} {str(r['resumed']):>8}")


def _print_dry_run(grid: Dict, *, epochs: int, batch_size: int) -> None:
    stage1 = _stage1_trials(grid)
    n_stage2 = len(grid["stage2_gamma"])
    total = len(stage1) + n_stage2

    print(f"Stage 1: {len(stage1)} trials (reg_weight x lr, gamma fixed="
          f"{grid.get('gamma_default', 2.0)}):")
    for i, p in enumerate(stage1, start=1):
        print(f"  [{i}] reg_weight={p['reg_weight']:g} lr={p['lr']:g} gamma={p['gamma']:g}")
    print(f"Stage 2: {n_stage2} trials (gamma in {grid['stage2_gamma']}, crossed with the "
          "stage-1 winner's reg_weight/lr -- exact params known only after stage 1 completes)")
    est_s = total * epochs * _SEC_PER_EPOCH_ESTIMATE
    print(f"Total trials: {total}  (batch_size={batch_size} fixed, {epochs} epochs/trial)")
    print(f"Rough time estimate: {est_s / 60:.1f} min "
          f"(design-doc figure, ~{_SEC_PER_EPOCH_ESTIMATE:.1f} s/epoch on the D1 tier "
          "reference hardware -- NOT measured on this box)")


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m e2e.ml.sweep",
        description="Loss-balance (reg_weight/lr/gamma) sweep for e2e.ml detection models.",
    )
    p.add_argument("--manifest", required=True, help="path to a generate_dataset manifest.json")
    p.add_argument("--model", required=True, choices=_MODEL_NAMES, help="model architecture")
    p.add_argument("--epochs", type=int, default=25, help="epochs/trial (design doc default: 25)")
    p.add_argument("--batch-size", type=int, default=8,
                   help="fixed across every trial (see module docstring)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="sweep output dir (default: <manifest dir>/runs/sweep_<model>)")
    p.add_argument("--grid-json", default=None,
                   help='JSON grid overriding DEFAULT_GRID, e.g. \'{"reg_weight": [1, 100], '
                        '"lr": [1e-4], "stage2_gamma": [0.0, 2.0]}\' (see module docstring)')
    p.add_argument("--dry-run", action="store_true",
                   help="print the trial list + a rough time estimate; do not train")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    grid = _validate_grid(json.loads(args.grid_json)) if args.grid_json else DEFAULT_GRID

    if args.dry_run:
        _print_dry_run(grid, epochs=args.epochs, batch_size=args.batch_size)
        return 0

    run_sweep(args.manifest, args.model, grid, epochs=args.epochs, batch_size=args.batch_size,
              seed=args.seed, out_dir=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
