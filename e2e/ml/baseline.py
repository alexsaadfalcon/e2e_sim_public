"""
Classical (no-learning) radar detector, scored through the same metric as the networks.

WHY THIS EXISTS
---------------
A detection AP has no meaning on its own. Two measurements, both reproducible from the
shipped code, make the case -- and note that they point in OPPOSITE directions, which is
exactly why the reference is worth having:

* On the `ti_iwr1443` corpus, where the evaluation harness demands finer azimuth accuracy
  than the array can resolve (see below), this baseline scored **AP 0.0241 / AR 0.0596**
  while a trained FFTRadNet on the SAME data reached only **0.0084**. The learned
  detector was losing to an FFT. Without a reference point that stayed invisible for the
  whole campaign, and low AP kept being attributed to the data -- impairments, the
  interconnect, ray tracing -- instead of to the harness.
* On a `radial_like` pilot corpus, where the harness IS answerable, the ordering flips:
  baseline **AP 0.0044**, trained FFTRadNet **AP 0.0168**. The model beats classical
  processing by ~3.9x. (400 frames, 12 epochs -- an early number, not a headline result.)

Reproduce the first with
`python -m e2e.ml.baseline --manifest <ti_iwr1443 corpus>/manifest.json --split val`.
Every reported model AP should be quoted next to whichever of these applies.

HISTORICAL NOTE, because an earlier version of this docstring cited it as if it were the
above: a pre-CFAR prototype of this module scored AP 0.0186 / AR 0.6381. That number came
from thresholding a raw dB map with no CFAR stage, so its recall is not comparable to the
shipped detector's -- quoting it would overstate the classical floor's recall by ~10x.

TARGET-COUNT BUG, fixed 2026-08-15: every number in this docstring above this note --
like the pre-CFAR number just above -- was measured under a `score_manifest` that
recovered ground truth by thresholding the DENSE label map's occupancy channel
(`labels[0] > 0.5`) instead of reading the deduplicated per-frame target list.
`e2e.ml.labels.encode_detection_labels` writes a 3x3-cell FOOTPRINT of `1.0`s around
each real target, so that thresholding counted every footprint cell as its own target --
roughly 9x too many (e.g. `rt_radial_v2/radial_like_D1` val: 487 real objects -> 4383
counted). `score_manifest` now scores against `RadarFrameDataset(...).targets(i)`, the
SAME list `train.py`'s `_evaluate_split` uses for the model rows, so ground truth is
counted identically for the classical baseline and every network. Re-measured (same
pred_maps, both target-list versions, isolating the effect):

  corpus (val split)               old AP    old AR    new AP    new AR
  ti_iwr1443_D1                    0.0204    0.0565    0.0124    0.2222
  pilot_radial/radial_like_D1      0.0043    0.0346    0.0029    0.1931
  rt_radial_v2/radial_like_D1      0.0037    0.0291    0.0023    0.1481

AP drops a little (a correctly-sized denominator changes which near-misses count) while
AR roughly quadruples-to-quintuples (recall was never as bad as the inflated denominator
made it look). Full table across all `ti_iwr1443` tiers (D0-D4) plus both `radial_like`
corpora: `report/rt_ml/baseline_rescore_v1/rescore.md`. The two headline numbers above
(AP 0.0241/AR 0.0596 for "the `ti_iwr1443` corpus"; AP 0.0044 for "a `radial_like` pilot
corpus") predate this fix and are kept as HISTORICAL, same as the pre-CFAR number --
their exact tier/split was not recorded precisely enough to reproduce bit-for-bit (see
the rescore doc); do not compare them against any number measured after 2026-08-15.

It also exposes a hard ceiling that no model can pass. The label grid asks for a target's
azimuth to within `MatchCriterion.max_sin_az_err` (0.06 by default), but an array of
`cfg.n_virtual` elements resolves no finer than ~`2 / n_virtual` in sin(azimuth). For the
`radial_like` preset (12 TX x 16 RX = 192 virtual) those are 0.06 vs 0.0104 -- a tolerance
of ~6 resolution cells, which is reasonable. For `ti_iwr1443` (3 TX x 4 RX = 12 virtual)
they are 0.06 vs 0.1667: the harness demands angular precision 2.8x finer than the array
can deliver. `resolution_report` prints exactly this comparison; call it before trusting
any AP number on a new config.

WHAT IT COMPUTES
----------------
Textbook FMCW processing, then a 2-D cell-averaging CFAR:

1. range/Doppler FFTs via `e2e.ml.transforms.adc_to_rd` (TDM inputs are de-interleaved
   into a virtual array first, exactly as the dataset layer does);
2. an angle FFT across the virtual-channel axis, per Doppler bin -- this is the step that
   needs per-channel phase, so it must precede any Doppler collapse;
3. non-coherent collapse over Doppler (a target's Doppler bin is not known a priori);
4. resampling onto the `LabelGrid`, then CA-CFAR: each cell's power divided by the mean of
   a training annulus around it (guard cells excluded), mapped monotonically to `[0, 1]`.

Step 4 is what makes the comparison fair. Because objectness is a monotone function of the
CFAR ratio, the metric's own threshold sweep (0.1 .. 0.9) *is* a CFAR threshold sweep over
`CFAR_MIN_DB` .. `CFAR_MAX_DB` -- so the baseline is scored as a properly operating
detector across its whole ROC, not as one arbitrary threshold.

Deliberately NOT included: angle interpolation/super-resolution (MUSIC, ESPRIT), Doppler-
aware association, or clutter maps. This is the honest classical floor, not the best
achievable classical result -- a model beating it is doing something, a model losing to it
is not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from e2e.ml.labels import LabelGrid
from e2e.ml.metrics import MatchCriterion, evaluate_dataset
from e2e.ml.transforms import adc_to_rd, tdm_deinterleave

# CFAR ratio (dB) mapped onto the [0, 1] objectness range the metric thresholds over.
# 0 dB == "cell equals its local noise estimate" -> objectness 0; 20 dB -> objectness 1.
# The metric's default sweep (0.1 .. 0.9) therefore spans 2 .. 18 dB of CFAR threshold.
CFAR_MIN_DB = 0.0
CFAR_MAX_DB = 20.0

# CA-CFAR window, in label-grid cells: a (2*train+2*guard+1)^2 outer square minus a
# (2*guard+1)^2 guard square. Guard cells keep a target's own energy out of the noise
# estimate it is being tested against.
CFAR_GUARD = 2
CFAR_TRAIN = 6


def resolution_report(cfg, grid: LabelGrid, criterion: Optional[MatchCriterion] = None) -> Dict:
    """Is this config's evaluation harness physically answerable?

    Returns the array's Rayleigh sin(azimuth) resolution alongside the label-grid cell
    size and the match tolerance, plus `tolerance_over_resolution` -- the ratio that
    matters. A value below 1.0 means the metric demands finer azimuth accuracy than the
    array can resolve, so both AP and AR are capped by geometry no matter what model or
    corpus is used.
    """
    criterion = criterion or MatchCriterion()
    n_virtual = int(cfg.n_virtual)
    rayleigh = 2.0 / n_virtual
    return {
        "n_tx": int(cfg.n_tx),
        "n_rx": int(cfg.n_rx),
        "n_virtual": n_virtual,
        "rayleigh_sin_az": rayleigh,
        "grid_cell_sin_az": grid.az_bin,
        "cells_per_beamwidth": rayleigh / grid.az_bin,
        "match_tolerance_sin_az": float(criterion.max_sin_az_err),
        "tolerance_over_resolution": float(criterion.max_sin_az_err) / rayleigh,
        "answerable": float(criterion.max_sin_az_err) >= rayleigh,
    }


def range_azimuth_power(cfg, adc: torch.Tensor, *, n_angle_fft: Optional[int] = None) -> torch.Tensor:
    """Raw ADC `[n_rx, n_chirps, n_samples]` -> real power `[n_angle, n_range]`.

    Steps 1-3 of the module docstring. `n_angle_fft` defaults to the virtual-channel
    count, i.e. no zero-padding: an interpolated angle axis would place peaks between
    array resolution cells without adding information.
    """
    import dataclasses

    if cfg.mimo == "tdm":
        sub_cfg = dataclasses.replace(cfg, n_tx=1, mimo="single", n_chirps=cfg.n_chirps_per_tx)
        rd = adc_to_rd(sub_cfg, tdm_deinterleave(cfg, adc))
    else:
        rd = adc_to_rd(cfg, adc)

    n_channel = rd.shape[0]
    n_fft = int(n_angle_fft) if n_angle_fft is not None else n_channel
    angle = torch.fft.fftshift(torch.fft.fft(rd, n=n_fft, dim=0), dim=0)   # [n_fft, R, D]
    return (angle.abs() ** 2).max(dim=2).values                            # [n_fft, R]


def _to_grid(power: torch.Tensor, cfg, grid: LabelGrid) -> torch.Tensor:
    """Resample `[n_angle, n_range_fine]` power onto `[grid.n_range, grid.n_azimuth]`.

    Nearest-neighbour on both axes. The angle axis of an `n_fft`-point FFT (fftshifted)
    maps to `sin(az) = 2k/n_fft` for `k` in `[-n_fft/2, n_fft/2)`; the fine range axis is
    `i * cfg.range_resolution_m`.
    """
    n_angle, n_range_fine = power.shape
    dev = power.device

    sin_src = 2.0 * (torch.arange(n_angle, device=dev, dtype=torch.float32) - n_angle // 2) / n_angle
    sin_dst = (torch.arange(grid.n_azimuth, device=dev, dtype=torch.float32) + 0.5) * grid.az_bin - 1.0
    ai = torch.bucketize(sin_dst, sin_src).clamp_(0, n_angle - 1)

    r_dst = (torch.arange(grid.n_range, device=dev, dtype=torch.float32) + 0.5) * grid.range_bin_m
    ri = (r_dst / float(cfg.range_resolution_m)).long().clamp_(0, n_range_fine - 1)

    return power[ai][:, ri].T.contiguous()          # [n_range, n_azimuth]


def cfar_objectness(power: torch.Tensor, *, guard: int = CFAR_GUARD, train: int = CFAR_TRAIN,
                    min_db: float = CFAR_MIN_DB, max_db: float = CFAR_MAX_DB) -> torch.Tensor:
    """`[n_range, n_azimuth]` power -> CA-CFAR objectness in `[0, 1]`.

    The noise estimate under each cell is the mean over an annulus: a
    `(2*(guard+train)+1)^2` outer square minus its `(2*guard+1)^2` guard core. Both means
    come from `avg_pool2d` with `count_include_pad=False`, so cells at the map edge average
    only over real neighbours instead of being biased toward zero by padding.
    """
    x = power[None, None]
    outer_k = 2 * (guard + train) + 1
    guard_k = 2 * guard + 1
    outer_sum = F.avg_pool2d(x, outer_k, stride=1, padding=guard + train,
                             count_include_pad=False) * (outer_k ** 2)
    guard_sum = F.avg_pool2d(x, guard_k, stride=1, padding=guard,
                             count_include_pad=False) * (guard_k ** 2)
    n_train = outer_k ** 2 - guard_k ** 2
    noise = ((outer_sum - guard_sum) / n_train).clamp_min(torch.finfo(power.dtype).tiny)

    ratio_db = 10.0 * torch.log10((power[None, None] / noise).clamp_min(1e-12))
    obj = (ratio_db - min_db) / (max_db - min_db)
    return obj.clamp_(0.0, 1.0)[0, 0]


def classical_detection_map(cfg, adc: torch.Tensor, grid: LabelGrid, **kwargs) -> torch.Tensor:
    """Raw ADC -> a `[3, n_range, n_azimuth]` map in the detector's own output format.

    Channel 0 is the CFAR objectness; the two regression channels are zero, so a decoded
    detection sits at its cell centre. That is the honest classical behaviour -- there is
    no sub-cell refinement without an interpolation stage this baseline deliberately omits.
    """
    power = range_azimuth_power(cfg, adc, n_angle_fft=kwargs.pop("n_angle_fft", None))
    obj = cfar_objectness(_to_grid(power, cfg, grid), **kwargs)
    out = torch.zeros((3, grid.n_range, grid.n_azimuth), dtype=torch.float32, device=obj.device)
    out[0] = obj
    return out


# --------------------------------------------------------------------------------
# Scoring a stored corpus
# --------------------------------------------------------------------------------
def score_manifest(manifest_path, split: str = "val", *, limit: Optional[int] = None,
                   device=None, **kwargs) -> Dict:
    """Score the classical baseline over a stored corpus split.

    Returns `evaluate_dataset`'s metrics dict with a `"resolution"` entry attached (see
    `resolution_report`) and `"n_frames"`. Needs a corpus carrying raw ADC
    (`adc_code_re`/`adc_code_im`); an RD-only corpus has no ADC to beamform and raises.

    Ground truth is `RadarFrameDataset(manifest_path, split=split).targets(i)` -- the
    SAME deduplicated per-frame target list `train.py`'s `_evaluate_split` scores
    against -- NOT re-derived from the dense label map's positive cells. A target's
    positive footprint is 3x3 cells (`e2e.ml.labels.encode_detection_labels`), so
    thresholding the label map directly counted every footprint cell as its own target:
    ~9x too many, corrupting AR (recall's ground-truth-count denominator) and, through
    it, AP (bug found post-hoc; see the docstring's HISTORICAL NOTE analogue in this
    module's history -- `report/rt_ml/baseline_rescore_v1/rescore.md` has the old-vs-new
    numbers this produced).
    """
    from e2e.ml.dataset import RadarFrameDataset
    from e2e.ml.radar_config import RadarConfig

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    cfg = RadarConfig.from_dict(manifest["config"])
    g = manifest["grid"]
    grid = LabelGrid(n_range=int(g["n_range"]), n_azimuth=int(g["n_azimuth"]),
                     max_range_m=float(g["max_range_m"]))

    files = manifest["files"][split]
    if limit is not None:
        files = files[:limit]
    # `files` is a prefix slice (or the full list) of manifest["files"][split] in order,
    # so its positions 0..len(files)-1 line up exactly with targets_ds.files' (index i
    # here == index i there).
    targets_ds = RadarFrameDataset(manifest_path, split=split)

    pred_maps, target_lists = [], []
    for i, fn in enumerate(files):
        with np.load(manifest_path.parent / fn, allow_pickle=True) as z:
            if "adc_code_re" not in z.files:
                raise ValueError(
                    f"{fn} has no raw ADC (keys: {sorted(z.files)}); the classical baseline "
                    "needs adc_code_re/adc_code_im to beamform")
            adc = torch.as_tensor(z["adc_code_re"].astype(np.float32)
                                  + 1j * z["adc_code_im"].astype(np.float32))
        if device is not None:
            adc = adc.to(device)
        pred_maps.append(classical_detection_map(cfg, adc, grid, **kwargs).cpu())
        target_lists.append(targets_ds.targets(i))

    res = dict(evaluate_dataset(pred_maps, target_lists, grid))
    res["n_frames"] = len(files)
    res["resolution"] = resolution_report(cfg, grid)
    return res


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m e2e.ml.baseline",
        description="Score the classical (no-learning) CFAR detector on a stored corpus, "
                    "as the reference point every model AP should be quoted against.")
    p.add_argument("--manifest", required=True, help="dataset manifest.json")
    p.add_argument("--split", default="val")
    p.add_argument("--limit", type=int, default=None, help="score only the first N frames")
    p.add_argument("--out", default=None, help="write the metrics dict here as JSON")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    res = score_manifest(args.manifest, args.split, limit=args.limit)

    r = res["resolution"]
    print(f"array: {r['n_tx']} TX x {r['n_rx']} RX = {r['n_virtual']} virtual elements")
    print(f"  Rayleigh sin(az) resolution : {r['rayleigh_sin_az']:.4f}")
    print(f"  label-grid cell             : {r['grid_cell_sin_az']:.4f} "
          f"({r['cells_per_beamwidth']:.1f} cells per beamwidth)")
    print(f"  match tolerance             : {r['match_tolerance_sin_az']:.4f} "
          f"({r['tolerance_over_resolution']:.2f}x the array resolution)")
    if not r["answerable"]:
        print("  ** the metric demands finer azimuth accuracy than this array can resolve;")
        print("     AP/AR are capped by geometry regardless of model or corpus **")
    print(f"\nclassical CFAR baseline over {res['n_frames']} {args.split} frames:")
    print(f"  AP = {res['AP']:.4f}   AR = {res['AR']:.4f}   "
          f"range_rmse = {res['range_rmse_m']:.3f} m")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {k: v for k, v in res.items() if not isinstance(v, dict) or k == "resolution"},
            indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
