"""
Compare several detectors on ONE corpus split at a MATCHED RECALL.

Why this exists
---------------
`e2e.ml.train --eval-only` scores one checkpoint and `e2e.ml.baseline` scores the
classical CFAR detector, but neither answers the question a comparison actually asks:
*at the same detection rate, which one cries wolf less often?* AP compresses the whole
precision-recall trade into one number that hides where on the curve a detector is
usable, and "Pd at threshold t" lets the threshold do the arguing -- a detector tuned to
fire more will always look more sensitive and less precise than one tuned to fire less,
which says nothing about either.

So every arm here is pinned to the same recall and reports **false alarms per frame** to
hold it (`e2e.ml.metrics.false_alarms_at_recall`). That number cannot be gamed by moving
a threshold: moving it moves the recall, and the operating point re-anchors elsewhere on
the same curve.

The decode floor is not the operating point
-------------------------------------------
Frames are decoded once at `--decode-threshold`, a deliberately PERMISSIVE confidence
floor whose only job is to make the precision-recall curve span far enough to contain
the operating point. It is not the threshold being reported -- that is read off the
curve, per arm, and printed. Set the floor too high and the curve stops short of the
target recall, which this reports as `did not reach` rather than silently answering at
whatever recall it managed.

Arms
----
* `--classical` -- the CFAR baseline (`e2e.ml.baseline.classical_detection_map`), which
  is the floor any learned detector has to beat to have earned its parameters.
* `--checkpoint NAME=PATH` (repeatable) -- any trained checkpoint. The model
  architecture comes out of the checkpoint, so FFTRadNet and SSMRadNet mix freely, as do
  a before/after pair of the same architecture.

Example
-------
    python -m e2e.ml.compare_detectors \\
        --manifest _scratch/radial_fixed/radial_like_D2/manifest.json \\
        --split test --recall 0.5 \\
        --classical \\
        --checkpoint "FFTRadNet (old corpus)=report/rt_ml/kenney_d2/fftradnet_e30/best.pt" \\
        --checkpoint "FFTRadNet (new corpus)=report/rt_ml/radial_fixed/fftradnet_e30/best.pt" \\
        --out report/rt_ml/compare_fa.json

Caveat that belongs with any before/after produced this way
-----------------------------------------------------------
Two checkpoints trained on two different corpora differ by everything that changed
between those corpora, not only by the one change of interest. Attribute the difference
to "retrained on a corrected corpus" unless the causes were separated by a controlled
run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from e2e.ml.metrics import evaluate_dataset, false_alarms_at_recall

# A confidence floor low enough that the curve reaches essentially any recall the
# detector can attain, but not so low that decoding drowns in noise cells. Detectors
# whose scores saturate near zero need it lower; the CLI exposes it for that reason.
DEFAULT_DECODE_THRESHOLD = 0.01
DEFAULT_TARGET_RECALL = 0.5


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_checkpoint_arg(spec: str) -> Tuple[str, str]:
    """`"NAME=PATH"` -> `(name, path)`; a bare path becomes its own name.

    Names carry spaces and parentheses ("FFTRadNet (old corpus)"), so only the FIRST
    `=` separates -- Windows paths do not contain one, but a name might.
    """
    if "=" in spec:
        name, path = spec.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise ValueError(f"--checkpoint expects NAME=PATH, got {spec!r}")
        return name, path
    return Path(spec).parent.name or spec, spec


def _score_arm(pred_maps, target_lists, grid, *, n_frames: int,
               decode_threshold: float, target_recall: float,
               ignore_lists: Optional[Sequence] = None) -> Dict:
    """`evaluate_dataset` at the decode floor + the matched-recall operating point.

    `ignore_lists`, if given, is `evaluate_dataset`'s per-frame don't-care list (see
    `e2e.ml.metrics`' "Don't-care / ignore regions"); `None` (the default) reproduces
    the pre-existing (ignore-less) score bit-for-bit.
    """
    metrics = evaluate_dataset(pred_maps, target_lists, grid,
                               score_threshold=decode_threshold, ignore=ignore_lists)
    op = false_alarms_at_recall(metrics["pr_curve"], n_frames,
                                target_recall=target_recall)
    return {
        "AP": metrics["AP"],
        "AR_at_decode_floor": metrics["AR"],
        "n_targets": metrics["n_targets"],
        "n_detections": metrics["n_detections"],
        "decode_threshold": decode_threshold,
        "operating_point": op,
        # The FULL curve, kept (B2 review, 2026-08-25): metrics.py's own docstring
        # promises the curve is stored so numbers can be audited or re-plotted
        # without re-running the model, and this harness was dropping exactly that --
        # answering an operating-point question (e.g. where the arms cross) then
        # required a full GPU re-run.
        "pr_curve": metrics["pr_curve"],
    }


def score_classical(manifest_path, split: str, *, device=None,
                    decode_threshold: float = DEFAULT_DECODE_THRESHOLD,
                    target_recall: float = DEFAULT_TARGET_RECALL,
                    limit: Optional[int] = None, use_ignore_regions: bool = False,
                    **kwargs) -> Dict:
    """The CFAR baseline arm. `kwargs` reach `baseline.classical_detection_map`.

    `use_ignore_regions` (default False -- opt-in, see `build_arg_parser`'s
    `--use-ignore-regions`): score with `RadarFrameDataset.unlabelled_objects`' don't-care
    positions passed through to `evaluate_dataset`, so a detection on real-but-unlabelled
    clutter is neither a hit nor a false alarm. False by default so this arm's number does
    not move under anyone who does not ask for it.
    """
    import numpy as np

    from e2e.ml.baseline import classical_detection_map
    from e2e.ml.dataset import RadarFrameDataset
    from e2e.ml.labels import LabelGrid
    from e2e.radar_config import RadarConfig

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    cfg = RadarConfig.from_dict(manifest["config"])
    g = manifest["grid"]
    grid = LabelGrid(n_range=int(g["n_range"]), n_azimuth=int(g["n_azimuth"]),
                     max_range_m=float(g["max_range_m"]))

    files = manifest["files"][split]
    if limit is not None:
        files = files[:limit]
    targets_ds = RadarFrameDataset(manifest_path, split=split)

    pred_maps, target_lists = [], []
    ignore_lists = [] if use_ignore_regions else None
    for i, fn in enumerate(files):
        with np.load(manifest_path.parent / fn, allow_pickle=True) as z:
            if "adc_code_re" not in z.files:
                raise ValueError(
                    f"{fn} has no raw ADC (keys: {sorted(z.files)}); the classical "
                    "baseline needs adc_code_re/adc_code_im to beamform")
            adc = torch.as_tensor(z["adc_code_re"].astype(np.float32)
                                  + 1j * z["adc_code_im"].astype(np.float32))
        if device is not None:
            adc = adc.to(device)
        pred_maps.append(classical_detection_map(cfg, adc, grid, **kwargs).cpu())
        target_lists.append(targets_ds.targets(i))
        if use_ignore_regions:
            ignore_lists.append(targets_ds.unlabelled_objects(i))

    return _score_arm(pred_maps, target_lists, grid, n_frames=len(files),
                      decode_threshold=decode_threshold, target_recall=target_recall,
                      ignore_lists=ignore_lists)


def score_null(manifest_path, split: str, *,
               decode_threshold: float = DEFAULT_DECODE_THRESHOLD,
               target_recall: float = DEFAULT_TARGET_RECALL,
               seed: int = 0, limit: Optional[int] = None,
               use_ignore_regions: bool = False) -> Dict:
    """The DATA-BLIND chance baseline (B4, from the 2026-08-25 B2 adversarial
    review): uniform random scores inside the TRAIN-split ground-truth bounding box
    (range x sin_az), exactly zero outside, never looking at the RF. Fitted on the
    train split only -- no test leakage -- and deterministic from `seed`.

    WHY IT IS A PERMANENT ARM: on this project's corpora, ground truth occupies a
    small fraction of the evaluated map (~16% on benchmark_v1/D2), so an AP quoted
    alone is uninterpretable -- the first B2 comparison shipped a trained model whose
    AP was BELOW this null. Never print an AP table without the chance floor in it.
    """
    import numpy as np

    from e2e.ml.dataset import RadarFrameDataset
    from e2e.ml.labels import LabelGrid

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    g = manifest["grid"]
    grid = LabelGrid(n_range=int(g["n_range"]), n_azimuth=int(g["n_azimuth"]),
                     max_range_m=float(g["max_range_m"]))

    # The box, from TRAIN targets only (centre range, sin_az -- elements 0 and 1 of
    # the target tuples; see e2e.ml.labels.targets_in_grid).
    train_ds = RadarFrameDataset(manifest_path, split="train")
    ranges, sins = [], []
    for i in range(len(train_ds)):
        for t in train_ds.targets(i):
            ranges.append(float(t[0]))
            sins.append(float(t[1]))
    if not ranges:
        raise ValueError("null baseline: the train split holds no targets to fit a box")
    r_lo = max(0, int(min(ranges) / grid.range_bin_m))
    r_hi = min(grid.n_range, int(max(ranges) / grid.range_bin_m) + 1)
    a_lo = max(0, int((min(sins) + 1.0) / grid.az_bin))
    a_hi = min(grid.n_azimuth, int((max(sins) + 1.0) / grid.az_bin) + 1)

    eval_ds = RadarFrameDataset(manifest_path, split=split)
    n = len(eval_ds) if limit is None else min(limit, len(eval_ds))
    pred_maps, target_lists = [], []
    ignore_lists = [] if use_ignore_regions else None
    for i in range(n):
        rng = np.random.default_rng(int(seed) + i)
        # [3, R, A], the detector output format every arm shares: channel 0 = score,
        # regression channels zero (cell-centre decode -- same honest convention as
        # the classical baseline, see classical_detection_map's docstring).
        m = np.zeros((3, grid.n_range, grid.n_azimuth), dtype=np.float32)
        m[0, r_lo:r_hi, a_lo:a_hi] = rng.random((r_hi - r_lo, a_hi - a_lo),
                                                dtype=np.float32)
        pred_maps.append(torch.from_numpy(m))
        target_lists.append(eval_ds.targets(i))
        if use_ignore_regions:
            ignore_lists.append(eval_ds.unlabelled_objects(i))

    res = _score_arm(pred_maps, target_lists, grid, n_frames=n,
                     decode_threshold=decode_threshold, target_recall=target_recall,
                     ignore_lists=ignore_lists)
    res["null_box"] = {"range_bins": [r_lo, r_hi], "az_bins": [a_lo, a_hi],
                       "fit_split": "train", "seed": int(seed)}
    return res


def score_checkpoint(manifest_path, checkpoint_path, split: str, *, device=None,
                     decode_threshold: float = DEFAULT_DECODE_THRESHOLD,
                     target_recall: float = DEFAULT_TARGET_RECALL,
                     batch_size: int = 8, ssm_chunk_size: Optional[int] = None,
                     limit: Optional[int] = None,
                     use_ignore_regions: bool = False) -> Dict:
    """One trained-checkpoint arm, reusing `train.py`'s reload and forward seams.

    `limit` truncates to the first N frames. `RadarFrameDataset` walks
    `manifest["files"][split]` in order and so does the classical arm, so the same
    `limit` selects the SAME frames in both -- which is the only way false alarms per
    frame stay comparable across arms.
    """
    from e2e.ml.train import _make_dataset, _predict_split, load_model_for_eval

    device = device if device is not None else _default_device()
    model, _manifest, grid, input_format = load_model_for_eval(
        manifest_path, checkpoint_path, device=device, ssm_chunk_size=ssm_chunk_size)
    ds = _make_dataset(manifest_path, split, input_format)
    pred_maps = _predict_split(model, ds, device=device, batch_size=batch_size)
    target_lists = [ds.targets(i) for i in range(len(ds))]
    ignore_lists = [ds.unlabelled_objects(i) for i in range(len(ds))] if use_ignore_regions else None
    if limit is not None:
        pred_maps, target_lists = pred_maps[:limit], target_lists[:limit]
        if ignore_lists is not None:
            ignore_lists = ignore_lists[:limit]

    res = _score_arm(pred_maps, target_lists, grid, n_frames=len(pred_maps),
                     decode_threshold=decode_threshold, target_recall=target_recall,
                     ignore_lists=ignore_lists)
    res["model"] = type(model).__name__
    res["checkpoint"] = str(checkpoint_path)
    return res


def compare(manifest_path, *, split: str = "test",
            checkpoints: Sequence[Tuple[str, str]] = (),
            classical: bool = False,
            target_recall: float = DEFAULT_TARGET_RECALL,
            decode_threshold: float = DEFAULT_DECODE_THRESHOLD,
            device=None, batch_size: int = 8,
            ssm_chunk_size: Optional[int] = None,
            limit: Optional[int] = None,
            classical_kwargs: Optional[Dict] = None,
            classical_doppler_reduce: Optional[Sequence[str]] = None,
            null_baseline: bool = True,
            use_ignore_regions: bool = False) -> Dict:
    """Every requested arm, scored on the same split at the same matched recall.

    `null_baseline` (default True -- see `score_null`) appends the data-blind
    chance arm to every comparison; disable only for a run whose output feeds a
    caller that adds its own floor.

    `use_ignore_regions` (default False, opt-in -- see `build_arg_parser`'s
    `--use-ignore-regions`): threads each arm's don't-care positions (`e2e.ml.dataset.
    RadarFrameDataset.unlabelled_objects`) through to `evaluate_dataset`, so a detection
    on real-but-unlabelled clutter is dropped rather than charged as a false alarm.
    False reproduces every arm's pre-existing score bit-for-bit.
    """
    if not classical and not checkpoints:
        raise ValueError("nothing to compare: pass --classical and/or --checkpoint")

    arms: List[Dict] = []
    if classical:
        # One arm per Doppler reduction, so they are scored on IDENTICAL frames. The
        # reduction changes the noise statistics CA-CFAR's threshold depends on, and a
        # noise-only test cannot see what it costs on real targets -- see
        # notes/tools/measure_cfar_calibration.py.
        for reduce in (classical_doppler_reduce or (None,)):
            kw = dict(classical_kwargs or {})
            label = "classical CFAR"
            if reduce is not None:
                kw["doppler_reduce"] = reduce
                label = f"classical CFAR ({reduce})"
            arms.append({"name": label,
                         **score_classical(manifest_path, split, device=device,
                                           decode_threshold=decode_threshold,
                                           target_recall=target_recall, limit=limit,
                                           use_ignore_regions=use_ignore_regions,
                                           **kw)})
    for name, path in checkpoints:
        arms.append({"name": name,
                     **score_checkpoint(manifest_path, path, split, device=device,
                                        decode_threshold=decode_threshold,
                                        target_recall=target_recall,
                                        batch_size=batch_size,
                                        ssm_chunk_size=ssm_chunk_size,
                                        limit=limit,
                                        use_ignore_regions=use_ignore_regions)})
    null_skipped = None
    if null_baseline:
        try:
            arms.append({"name": "null (random-in-GT-box)",
                         **score_null(manifest_path, split,
                                      decode_threshold=decode_threshold,
                                      target_recall=target_recall, limit=limit,
                                      use_ignore_regions=use_ignore_regions)})
        except ValueError as e:
            # A corpus with no train targets cannot fit the box (e.g. a val-only
            # test corpus). Degrade to a RECORDED skip, never a silent one: the
            # stored JSON must say its chance floor is missing and why.
            null_skipped = str(e)

    return {
        **({"null_skipped": null_skipped} if null_skipped else {}),
        "manifest": str(manifest_path),
        "split": split,
        "target_recall": target_recall,
        "decode_threshold": decode_threshold,
        "use_ignore_regions": use_ignore_regions,
        "arms": arms,
    }


def format_table(result: Dict) -> str:
    """A fixed-width table. `did not reach` is spelled out, never filled with a number."""
    r = result["target_recall"]
    lines = [
        f"corpus : {result['manifest']}   split={result['split']}",
        f"held at recall = {r:.2f}   (frames decoded at score > "
        f"{result['decode_threshold']:g}, a floor, not the operating point)",
        "",
        f"{'detector':<34}{'FA/frame':>10}{'recall':>9}{'thresh':>9}{'AP':>8}"
        f"{'max recall':>12}",
        "-" * 82,
    ]
    for arm in result["arms"]:
        op = arm["operating_point"]
        if op["reached"]:
            fa = f"{op['fp_per_frame']:.1f}"
            rec = f"{op['recall_achieved']:.3f}"
            th = f"{op['score_threshold']:.3g}"
        else:
            fa, rec, th = "did not reach", "--", "--"
        ap = arm["AP"]
        lines.append(f"{arm['name'][:33]:<34}{fa:>10}{rec:>9}{th:>9}"
                     f"{('nan' if math.isnan(ap) else f'{ap:.3f}'):>8}"
                     f"{op['recall_max']:>12.3f}")
    lines.append("")
    lines.append("FA/frame is false positives divided by frames scored; lower is better "
                 "at equal recall.")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m e2e.ml.compare_detectors",
        description="Compare detectors on one split at a matched recall, reporting "
                    "false alarms per frame.",
    )
    p.add_argument("--manifest", required=True, help="path to a manifest.json")
    p.add_argument("--split", default="test", help="dataset split (default test)")
    p.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH",
                   help="a trained checkpoint to score; repeatable. The architecture "
                        "comes from the checkpoint, so models may be mixed")
    p.add_argument("--classical", action="store_true",
                   help="also score the CFAR baseline (needs raw ADC in the corpus)")
    p.add_argument("--classical-doppler-reduce", default=None,
                   metavar="max,sum,cfar_first",
                   help="comma-separated Doppler reductions to score the classical "
                        "baseline under, one arm each, on identical frames (see "
                        "e2e.ml.baseline's DOPPLER_* constants). Default: the shipped one")
    # The two halves of the TDM Doppler fix, ON by default in `classical_detection_map`
    # since 2026-08-29 (+28.7% AP together; neither works alone). Exposed so a scored
    # comparison can turn them OFF from the CLI -- which is how the +28.7% is reproduced,
    # and how a pre-2026-08-29 number is regenerated.
    p.add_argument("--classical-no-tdm-doppler-comp", action="store_true",
                   help="disable the classical arm's per-Doppler-bin TDM phase "
                        "compensation (default: on for TDM configs). Only defined for "
                        "TDM, and invalid above the unambiguous velocity")
    p.add_argument("--classical-doppler-notch-bins", type=int, default=None,
                   metavar="N",
                   help="width of the classical arm's zero-Doppler notch, in bins "
                        "(default 1). 0 disables it")
    p.add_argument("--recall", type=float, default=DEFAULT_TARGET_RECALL,
                   help=f"recall to hold every arm at (default {DEFAULT_TARGET_RECALL})")
    p.add_argument("--decode-threshold", type=float, default=DEFAULT_DECODE_THRESHOLD,
                   help="permissive confidence floor for decoding, chosen so the PR "
                        f"curve reaches the target recall (default "
                        f"{DEFAULT_DECODE_THRESHOLD})")
    p.add_argument("--device", default=None, help="torch device (default: cuda if available)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--ssm-chunk-size", type=int, default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="score only the first N frames of the split, in EVERY arm (for "
                        "smoke runs). Arms must share frames for FA/frame to compare")
    p.add_argument("--no-null", action="store_true",
                   help="omit the data-blind random-in-GT-box chance arm (on by "
                        "default -- see score_null: never print an AP table without "
                        "its chance floor)")
    p.add_argument("--use-ignore-regions", action="store_true",
                   help="score with real-but-unlabelled clutter (e2e.ml.dataset."
                        "RadarFrameDataset.unlabelled_objects) as don't-care regions, so "
                        "a detection on one is neither a hit nor a false alarm (see "
                        "e2e.ml.metrics' 'Don't-care / ignore regions'). Default OFF -- "
                        "every arm's score is unchanged unless this is passed")
    p.add_argument("--out", default=None, help="write the full result dict as JSON here")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    device = torch.device(args.device) if args.device else _default_device()
    checkpoints = [parse_checkpoint_arg(spec) for spec in args.checkpoint]

    reductions = ([r.strip() for r in args.classical_doppler_reduce.split(",") if r.strip()]
                  if args.classical_doppler_reduce else None)
    # Only set what was explicitly asked for: an unset flag must leave
    # `classical_detection_map`'s own default in place rather than restating it here,
    # so the default lives in exactly one file.
    classical_kwargs: Dict[str, Any] = {}
    if args.classical_no_tdm_doppler_comp:
        classical_kwargs["tdm_doppler_comp"] = False
    if args.classical_doppler_notch_bins is not None:
        classical_kwargs["doppler_notch_bins"] = int(args.classical_doppler_notch_bins)
    result = compare(args.manifest, split=args.split, checkpoints=checkpoints,
                     classical=args.classical, target_recall=args.recall,
                     decode_threshold=args.decode_threshold, device=device,
                     batch_size=args.batch_size, ssm_chunk_size=args.ssm_chunk_size,
                     limit=args.limit, classical_doppler_reduce=reductions,
                     classical_kwargs=classical_kwargs or None,
                     null_baseline=not args.no_null,
                     use_ignore_regions=args.use_ignore_regions)
    print(format_table(result))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {out}")

    # A run in which no arm reached the target recall answered nothing; say so with an
    # exit code, so an overnight pipeline cannot mistake it for a result.
    if not any(a["operating_point"]["reached"] for a in result["arms"]):
        print("\nNO ARM REACHED THE TARGET RECALL -- this comparison is empty. Lower "
              "--recall, or lower --decode-threshold so the curve spans further.",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
