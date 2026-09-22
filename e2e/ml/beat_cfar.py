"""Reproducible runner for the "beat CFAR" experiment (owner directive, 2026-09-21).

THE GOAL, stated so a cold reader knows what success is
--------------------------------------------------------
Classical CA-CFAR scores **AP 0.301** on `benchmark_v1_D2` test under the protocol below.
The owner's directive is to train a learned detector that beats it. This module trains the
candidate arms and scores every one of them against CFAR and against two controls, in one
command, so the claim is never assembled by hand from separate runs.

WHY A SCRIPT AND NOT A COMMAND IN A DOC
---------------------------------------
Every number this project has had to retract was quoted from a run somebody reconstructed
from memory. The arms here are scored on the SAME frames, at the SAME matched recall, with
the SAME decode floor and range crop, because they are produced by one invocation.

THE TWO CONTROLS ARE NOT OPTIONAL
---------------------------------
* `classical CFAR` -- the full shipped front end (zero-Doppler notch + TDM Doppler
  compensation). This is the number to beat.
* `null (random-in-GT-box)` -- the data-blind chance floor. An AP quoted without it is
  uninterpretable: ground truth occupies ~16% of the map, so chance is ~0.08, not 0.
A third control worth running by hand when a claim is being made (see
`notes/ML_DETECTION_BRIEF.md`): a fixed global threshold on the exact tensor the network
is fed, which scored 0.169. The learning is only worth the difference from THAT.

WHAT "DETERMINISTIC" HONESTLY MEANS HERE
-----------------------------------------
Determinism is ON BY DEFAULT here, because this script's whole purpose is producing a
number somebody will quote. `--seed` (default 42) and the `--deterministic` flag are passed
to `e2e.ml.train`, whose `set_determinism` seeds torch/CUDA/numpy/random AND pins
`cudnn.deterministic`, disables `cudnn.benchmark` autotuning, and calls
`torch.use_deterministic_algorithms(True)`. With that, a rerun on THIS machine and software
stack is bit-identical -- not merely close.

Two honest limits remain, and no API removes them:
  * NOT identical across different GPU architectures, CUDA/cuDNN versions or torch builds
    -- those change kernel selection and floating-point reduction order beneath any seed.
    Record the hardware next to any number you intend to reproduce exactly.
  * `use_deterministic_algorithms(True)` RAISES on an op with no deterministic kernel. That
    is intended: it fails loudly instead of varying quietly. If it fires, use `--no-strict`
    and SAY SO wherever the number is quoted, because the run is then only
    seed-reproducible (same-machine agreement to a few 1e-3 in AP, the scale of this
    pipeline's own noise). Quote AP to three decimals, never more.

WHAT IT COSTS (measured 2026-09-21, not estimated)
---------------------------------------------------
2 epochs of `fftradnet`/`rad`, batch 8, seed 42, one GPU of this box (RTX 2000-class, 8 GiB;
torch+cuDNN as installed in `alex_env`):

    default (seeded only)     987.2 s
    --deterministic          1097.4 s     1.112x  (+11.2%)
    --deterministic (repeat) 1086.3 s     bit-identical train_loss and val_AP -> True

So determinism costs about 11% wall time here, and it does deliver: the two strict runs
agreed exactly, not approximately.

One result worth not over-reading: the seeded-only run ALSO produced the identical loss
curve. On this box, for this model, `torch.manual_seed` alone was already reproducible --
cuDNN autotuning (`benchmark`) is off by default in torch, which is the main thing the flag
disables. The flag therefore buys a GUARANTEE rather than a changed number, and that
guarantee is what matters on a different box, a different driver, or a model that does use
an atomics-based kernel. Do not conclude from one model that the flag is unnecessary.

USAGE
-----
    python -m e2e.ml.beat_cfar                      # both arms, seed 42, deterministic
    python -m e2e.ml.beat_cfar --arms fftradnet_rad # one arm
    python -m e2e.ml.beat_cfar --skip-train         # score existing checkpoints only
    python -m e2e.ml.beat_cfar --no-strict          # faster, only seed-reproducible
    python -m e2e.ml.beat_cfar --force              # retrain even if epochs are recorded

Training is IDEMPOTENT: an arm whose `history.json` already records the requested epoch
count is skipped unless `--force`. That makes re-running this to regenerate the table cheap
and safe, which is the only way a "reproducible" script actually gets re-run.

The skip is NOT taken when the input pipeline has changed since the checkpoint was written
(see `_stale_reason`) -- such an arm is retrained instead. A checkpoint is only
interchangeable with a rerun while the code that built its inputs still exists, and on
2026-09-21 that assumption failed silently and cost two invalid results.

That guard is blind to WHAT changed. After a behaviour-preserving edit to a fingerprinted
file, do not `--force` a 5-hour retrain and do not edit the fingerprint by hand:

    python -m e2e.ml.recertify e2e/ml/runs/<arm_dir>

re-runs the checkpoint's own validation split under the current code and re-stamps the
fingerprint ONLY if the recorded `best_val_AP` reproduces (default tolerance 2e-3 AP).
The F84 checkpoints would fail that test by 0.4; the 2026-09-22 `dataset.py` refactor
passed it with delta +0.0000.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

MANIFEST = "e2e/ml/datasets/b1_bench_v3/benchmark_v1_D2/manifest.json"

#: The scoring protocol. These are the values every recorded number in
#: `notes/ML_DETECTION_BRIEF.md` and `ESTABLISHED_FACTS.md` F83 used; changing one
#: invalidates the comparison against them, so they live here as named constants rather
#: than as flags with defaults scattered across a docs command line.
SPLIT = "test"
TARGET_RECALL = 0.5
DECODE_THRESHOLD = 0.01
MAX_RANGE_M = 40.0

#: The number to beat, measured under exactly the protocol above.
CFAR_AP = 0.301

#: Candidate arms. `input_format="rad"` hands the network the classical beamformer's own
#: range-azimuth-Doppler cube (see `e2e.ml.dataset._derive_input`); "rd" is the legacy
#: layout in which azimuth exists only as virtual-channel phase and is never learned.
ARMS: Dict[str, Dict] = {
    "fftradnet_rad": {
        "model": "fftradnet", "input_format": "rad", "epochs": 30,
        "out": "e2e/ml/runs/b8_fftradnet_rad_frontend",
        "why": "FFTRadNet on the corrected front end. Its decoder builds output azimuth "
               "from the BACKBONE CHANNEL axis, which the rad layout fills with azimuth "
               "-- so it mixes azimuth away in layer one. Expected to improve but to keep "
               "emitting a stripe.",
    },
    "raddetnet": {
        "model": "raddetnet", "input_format": "rad", "epochs": 40,
        "out": "e2e/ml/runs/b7_raddetnet",
        "why": "Doppler as channels, (range, azimuth) as the spatial plane -- the "
               "architecture matched to the representation. This is the arm that should "
               "break the stripe if the F83 diagnosis is right.",
    },
}

#: Checkpoints scored beside the arms but never trained here: the shipped `rd`-format
#: baselines every earlier number was quoted from. Reviewed finding (2026-09-22): the
#: demo's cards quoted 0.127 / 0.123 from a JSON written on 2026-09-01 that this script
#: never regenerated, while DEMO_DEFENSE.md said to quote THIS script's output only --
#: two authorities, and the one named as authoritative did not contain the number. Now it
#: does. A missing file is skipped with a printed line, not an error, so a clean clone
#: (runs/ is gitignored) still produces the table for the arms it trained.
REFERENCE_CHECKPOINTS: Dict[str, str] = {
    "fftradnet_rd_b5": "e2e/ml/runs/b5_fftradnet_v3/best.pt",
    "ssmradnet_rd_b5": "e2e/ml/runs/b5_ssmradnet_v3/best.pt",
}


def _seed_everything(seed: int, strict: bool) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    if strict:
        # Must be set BEFORE the first CUDA context; cuBLAS reads it at init.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _completed_epochs(out_dir: str) -> int:
    """Epochs actually finished, from `history.json`.

    `train.py` writes its artifacts INCREMENTALLY on every validation improvement, so the
    presence of `best.pt`/`history.json` does NOT mean a run finished -- a 120-epoch run
    once lost ~9 GPU-hours to that assumption. Count the epochs instead.
    """
    h = Path(out_dir) / "history.json"
    if not h.exists():
        return 0
    try:
        return len(json.loads(h.read_text())["epoch"])
    except Exception:
        return 0


def _stale_reason(out_dir: str) -> Optional[str]:
    """Why `out_dir/best.pt` is not interchangeable with a rerun, or None if it is.

    A checkpoint trained by code that no longer exists is worse than useless: reloading it
    feeds the network an input distribution it never saw, which reads as a catastrophic
    model failure rather than as the bookkeeping error it is.

    MEASURED, 2026-09-21, the reason this exists: `e2e/ml/dataset.py` was edited at 17:19
    (the `rad` front-end parity fix, commit 45b6f22) while `b7_raddetnet` trained
    16:36-19:36. That run recorded val_AP 0.484; its checkpoint, reloaded against the edited
    dataset, scored 0.023 on the SAME split. `b6_fftradnet_rad` (trained 13:41-15:58) fell
    from a reported test AP 0.229 to 0.054. Both were invalid, and the epoch-count check in
    `_train` would have skipped the retrain and re-scored them silently.

    Two mechanisms, in order of authority:

    1. `pipeline_fingerprint` recorded IN the checkpoint at start-of-run (`e2e.ml.train`).
       Authoritative: it is the content the training process actually imported.
    2. File mtimes, for checkpoints written before that field existed. PARTIAL, and it
       misses exactly the case above -- the final checkpoint write stamps 19:36, later than
       the 17:19 edit, so mtime calls it fresh. It does catch a source edited after a run
       finished, which is how `b6` is caught.
    """
    ck = Path(out_dir) / "best.pt"
    if not ck.exists():
        return None
    try:
        import torch
        from e2e.ml.train import INPUT_PIPELINE_SOURCES, pipeline_fingerprint
    except ImportError:
        return None

    try:
        # `weights_only=True` reads the metadata without unpickling arbitrary objects; this
        # is a 12 MB file being opened to read one string, per arm, per run.
        recorded = torch.load(ck, map_location="cpu",
                              weights_only=True).get("pipeline_fingerprint")
    except Exception:
        try:
            recorded = torch.load(ck, map_location="cpu").get("pipeline_fingerprint")
        except Exception:
            recorded = None

    if recorded is not None:
        now = pipeline_fingerprint()
        if now is not None and now != recorded:
            return (f"pipeline fingerprint differs from the one recorded at training start "
                    f"({recorded[:12]} -> {now[:12]})")
        return None

    # Fallback for pre-fingerprint checkpoints. Say so, so the weaker check is never
    # mistaken for the strong one.
    ck_mtime = ck.stat().st_mtime
    changed = []
    # Against the repo root, NOT the CWD: from any other directory the sources "do not
    # exist" and the fallback silently vouches for everything (found 2026-09-22 when the
    # GUI, launched from %TEMP%, reported no note for an unfingerprinted checkpoint).
    repo_root = Path(__file__).resolve().parents[2]
    for src in INPUT_PIPELINE_SOURCES:
        p = repo_root / src
        if not p.exists():
            continue
        newest = (max((f.stat().st_mtime for f in p.rglob("*.py")), default=0.0)
                  if p.is_dir() else p.stat().st_mtime)
        if newest > ck_mtime:
            changed.append(src)
    if changed:
        return (f"no recorded fingerprint, and {', '.join(changed)} are newer than best.pt")
    return None


def _train(name: str, spec: Dict, seed: int, strict: bool, force: bool) -> bool:
    done = _completed_epochs(spec["out"])
    if done >= spec["epochs"] and not force:
        reason = _stale_reason(spec["out"])
        if reason:
            # Retrain rather than skip. Skipping here would score a checkpoint against a
            # pipeline it was not trained on and report the result as this arm's number.
            print(f"[{name}] STALE -- {done}/{spec['epochs']} epochs recorded in "
                  f"{spec['out']}, but {reason}. Retraining; the recorded metrics "
                  f"describe code that no longer exists.")
        else:
            print(f"[{name}] SKIP -- {done}/{spec['epochs']} epochs already recorded "
                  f"in {spec['out']}/history.json (use --force to retrain)")
            return True
    cmd = [
        sys.executable, "-u", "-m", "e2e.ml.train",
        "--manifest", MANIFEST,
        "--model", spec["model"],
        "--input-format", spec["input_format"],
        "--epochs", str(spec["epochs"]),
        "--batch-size", "8",
        "--seed", str(seed),
        "--out", spec["out"],
    ]
    if strict:
        # Pins cuDNN/cuBLAS inside the CHILD, which is where training actually runs --
        # setting torch flags in this parent process would never reach it.
        cmd.append("--deterministic")
    print(f"\n[{name}] TRAIN ({done}/{spec['epochs']} done) :: {' '.join(cmd)}")
    env = dict(os.environ, MPLBACKEND="Agg")
    if strict:
        # cuBLAS reads this at CUDA init, so it must be in the child's environment before
        # it starts -- the --deterministic flag above cannot set it late enough itself.
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        print(f"[{name}] TRAIN FAILED rc={rc}")
        return False
    return True


def _stripe_statistic(checkpoint: str, limit: int = 40) -> Optional[float]:
    """Median rank-1 energy fraction of the objectness map.

    THE diagnostic for this experiment, and the reason AP alone is not enough. A map that
    is near-separable `f(range) * g(azimuth)` -- a full-field-of-view stripe rather than
    peaks -- scores ~0.85-0.89; ground truth scores **0.312**. AP can rise substantially
    while this does not move, which is exactly what happened going from the `rd` to the
    `rad` input (0.894 -> 0.853 while AP doubled). If a new arm improves AP without moving
    this toward 0.31, azimuth is still not being learned and the headline must say so.
    """
    try:
        import numpy as np
        import torch
        from e2e.ml.compare_detectors import _default_device
        from e2e.ml.train import _make_dataset, _predict_split, load_model_for_eval
    except ImportError as e:
        print(f"  (stripe statistic unavailable: {e})")
        return None

    dev = _default_device()
    model, _man, _grid, fmt = load_model_for_eval(MANIFEST, checkpoint, device=dev)
    ds = _make_dataset(MANIFEST, SPLIT, fmt)
    preds = _predict_split(model, ds, device=dev, batch_size=8)
    vals = []
    for p in preds[:limit]:
        m = p[0].detach().cpu().numpy().astype(np.float64)
        s = np.linalg.svd(m, compute_uv=False)
        vals.append(float(s[0] ** 2 / (s ** 2).sum()))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return float(np.median(vals))


def main(argv: Optional[List[str]] = None) -> int:
    global MANIFEST  # the CLI may point every train/score call at another corpus
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--arms", default=",".join(ARMS),
                   help=f"comma-separated subset of {sorted(ARMS)}")
    p.add_argument("--skip-train", action="store_true", help="score existing checkpoints only")
    p.add_argument("--force", action="store_true", help="retrain even if epochs are recorded")
    p.add_argument("--no-strict", dest="strict", action="store_false",
                   help="skip deterministic kernels (faster, but the run is only "
                        "seed-reproducible, not bit-identical). Default is STRICT, because "
                        "this script exists to produce quotable numbers")
    p.set_defaults(strict=True)
    p.add_argument("--manifest", default=MANIFEST,
                   help="manifest.json of the corpus to train on and score (default: the "
                        "maintainers' b1_bench_v3 path)")
    p.add_argument("--out", default="e2e/ml/runs/beat_cfar.json")
    args = p.parse_args(argv)
    MANIFEST = args.manifest  # every train/score call below reads the module constant

    names = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [n for n in names if n not in ARMS]
    if unknown:
        p.error(f"unknown arm(s) {unknown}; choices: {sorted(ARMS)}")

    _seed_everything(args.seed, args.strict)
    print(f"seed={args.seed}  strict={args.strict}  arms={names}")
    print(f"protocol: split={SPLIT} recall={TARGET_RECALL} decode={DECODE_THRESHOLD} "
          f"max_range_m={MAX_RANGE_M}   target to beat: CFAR AP {CFAR_AP}")

    if not args.skip_train:
        for n in names:
            if not _train(n, ARMS[n], args.seed, args.strict, args.force):
                return 1

    scored = [n for n in names if _completed_epochs(ARMS[n]["out"]) > 0]
    if not scored:
        print("nothing to score -- no arm has a checkpoint")
        return 1

    cmd = [sys.executable, "-m", "e2e.ml.compare_detectors",
           "--manifest", MANIFEST, "--split", SPLIT, "--classical",
           "--recall", str(TARGET_RECALL),
           "--decode-threshold", str(DECODE_THRESHOLD),
           "--max-range-m", str(MAX_RANGE_M),
           "--out", args.out]
    checkpoints = {n: f"{ARMS[n]['out']}/best.pt" for n in scored}
    for n, path in REFERENCE_CHECKPOINTS.items():
        if Path(path).is_file():
            checkpoints[n] = path
        else:
            print(f"[{n}] reference checkpoint not on this machine, skipped: {path}")
    for n, path in checkpoints.items():
        cmd += ["--checkpoint", f"{n}={path}"]
    print(f"\nSCORE :: {' '.join(cmd)}")
    if subprocess.run(cmd, env=dict(os.environ, MPLBACKEND="Agg")).returncode != 0:
        return 1

    # AP alone cannot say whether azimuth was learned -- see _stripe_statistic.
    print("\nSTRIPE STATISTIC (median rank-1 energy fraction; ground truth = 0.312)")
    stripes = {}
    for n, path in checkpoints.items():
        v = _stripe_statistic(path)
        stripes[n] = v
        if v is not None:
            verdict = ("azimuth LEARNED" if v < 0.55 else
                       "still a STRIPE -- azimuth NOT learned")
            print(f"  {n:20s} {v:.3f}   {verdict}")

    # Fold the diagnostic into the same JSON as the AP table so the two can never be
    # quoted from different runs.
    try:
        res = json.loads(Path(args.out).read_text())
        res["beat_cfar"] = {
            "seed": args.seed, "strict": bool(args.strict),
            "cfar_ap_target": CFAR_AP, "stripe_rank1": stripes,
            "stripe_ground_truth": 0.312,
            "protocol": {"split": SPLIT, "target_recall": TARGET_RECALL,
                         "decode_threshold": DECODE_THRESHOLD,
                         "max_range_m": MAX_RANGE_M},
        }
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"\nwrote {args.out}")
        for arm in res.get("arms", []):
            if arm.get("AP", 0) > CFAR_AP:
                print(f"\n*** {arm['name']} AP {arm['AP']:.3f} BEATS CFAR {CFAR_AP} ***")
                print("    Before quoting it: check the stripe statistic above, and see")
                print("    notes/ML_DETECTION_BRIEF.md for the controls a reviewer will ask for.")
    except Exception as e:
        print(f"(could not annotate {args.out}: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
