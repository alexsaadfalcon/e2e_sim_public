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

1. range/Doppler FFTs via `e2e.chain.transforms.adc_to_rd` (TDM inputs are de-interleaved
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

import math

import numpy as np
import torch
import torch.nn.functional as F

from e2e.ml.labels import LabelGrid
from e2e.ml.metrics import MatchCriterion, evaluate_dataset
from e2e.chain.transforms import adc_to_rd, ddma_demux, tdm_deinterleave

# CFAR ratio (dB) mapped onto the [0, 1] objectness range the metric thresholds over.
# 0 dB == "cell equals its local noise estimate" -> objectness 0; 20 dB -> objectness 1.
# The metric's default sweep (0.1 .. 0.9) therefore spans 2 .. 18 dB of CFAR threshold.
CFAR_MIN_DB = 0.0
CFAR_MAX_DB = 20.0

# CA-CFAR window, in label-grid cells: a (2*train+2*guard+1)^2 outer square minus a
# (2*guard+1)^2 guard square. Guard cells keep a target's own energy out of the noise
# estimate it is being tested against.
#: How the Doppler axis is collapsed before detection. This is a STATISTICS choice, not
#: a taste one, because CA-CFAR's threshold means what it means only when the cell under
#: test and its training cells share a distribution.
#:
#: `DOPPLER_MAX` (the pre-v1.1 default) -- max over K Doppler bins. Square-law noise power
#:   is exponential; the MAXIMUM of K exponentials is not. Its upper tail is Gumbel-like
#:   and far heavier, so a fixed dB threshold buys a false-alarm rate nothing in the CFAR
#:   design predicts. It is, however, the most SENSITIVE reduction for a point target,
#:   costing only ~ln(K) against the noise floor.
#: `DOPPLER_SUM` (the shipped default since v1.1) -- non-coherent integration. Erlang(K),
#:   whose coefficient of variation is 1/sqrt(K), so the noise ratio concentrates and the
#:   threshold behaves. Pays for it by diluting a single-bin target across K bins.
#:   MEASURED better than `max` at matched recall on the two corpora it was tried on at
#:   the time (F48, F50: e.g. classical FA/frame 154.0 -> 119.8 on the radial test split);
#:   default flipped for v1.1 (owner-approved, release-plan A4). SCOPE (2026-09-22, an
#:   independent verifier, F85 addendum): on `b1_bench_v3/benchmark_v1_D2` test under the
#:   beat_cfar protocol `max` scores AP 0.3116 / 4.94 FA/frame against `sum`'s
#:   0.3006 / 6.24, and `cfar_first` with an unclamped score 0.328. The default is kept
#:   because every published number was scored with it; it is not the strongest
#:   classical readout on every corpus, and a claim that it is would be false here.
#: `DOPPLER_CFAR_FIRST` -- CFAR each Doppler slice, then take the max of the OBJECTNESS.
#:   Detect first, collapse second. Every CFAR sees exponential cells, so the threshold is
#:   calibrated, AND a target that lives in one Doppler bin is tested against that bin's
#:   own noise rather than against K bins of it. Costs K CFAR passes.
DOPPLER_MAX = "max"
DOPPLER_SUM = "sum"
DOPPLER_CFAR_FIRST = "cfar_first"
_DOPPLER_REDUCTIONS = (DOPPLER_MAX, DOPPLER_SUM, DOPPLER_CFAR_FIRST)

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


#: Minimum unambiguous velocity (m/s) at which `classical_detection_map` will switch the
#: zero-Doppler notch on by itself. Below this a scene's own targets alias into the notched
#: bin and the notch deletes them rather than the clutter (measured Pd 0.098 -> 0.000 on
#: `radial_like`, v_max 1.06). Automotive scenes in this repo draw 0-30 m/s and clamp to
#: 80% of v_max, so a config at or above this can still alias in principle -- the threshold
#: buys a margin, it does not prove safety, and an explicit `doppler_notch_bins=` overrides
#: it either way. `benchmark_v1` (9.69) and `ti_iwr1443` (12.81) clear it; `radial_like`
#: (1.06) does not.
NOTCH_MIN_VMAX_MPS = 5.0


def range_azimuth_power(cfg, adc: torch.Tensor, *, n_angle_fft: Optional[int] = None,
                        angle_window: bool = True,
                        doppler_notch_bins: int = 0,
                        tdm_doppler_comp: bool = False,
                        keep_doppler: bool = False) -> torch.Tensor:
    """Raw ADC `[n_rx, n_chirps, n_samples]` -> real power `[n_angle, n_range]`.

    `keep_doppler=True` returns the uncollapsed `[n_angle, n_range, n_doppler]` cube
    instead, for callers that want to detect before collapsing (see `DOPPLER_CFAR_FIRST`).

    Steps 1-3 of the module docstring. `n_angle_fft` defaults to the virtual-channel
    count, i.e. no zero-padding: an interpolated angle axis would place peaks between
    array resolution cells without adding information.

    Both MIMO schemes are resolved to the full virtual array before the angle FFT: TDM by
    de-interleaving chirps (separation in time), DDMA by slicing Doppler sub-bands
    (separation in Doppler, so it can only happen after the Doppler FFT). Until
    2026-08-17 the DDMA branch was missing, and this baseline formed its angle FFT over
    the n_rx PHYSICAL receivers -- 16 standing in for `radial_like`'s 192 virtual
    elements. Every `radial_like` classical-CFAR number produced before that date is
    therefore a 16-element result and is not comparable to one produced after.
    """
    import dataclasses

    if cfg.mimo == "tdm":
        sub_cfg = dataclasses.replace(cfg, n_tx=1, mimo="single", n_chirps=cfg.n_chirps_per_tx)
        rd = adc_to_rd(sub_cfg, tdm_deinterleave(cfg, adc))
    elif cfg.mimo == "ddma":
        rd = ddma_demux(cfg, adc_to_rd(cfg, adc))
    else:
        rd = adc_to_rd(cfg, adc)

    # TDM DOPPLER COMPENSATION. TDM fires its transmitters in sequence, so a moving
    # target's phase advances between one TX's chirps and the next. `tdm_deinterleave`
    # deliberately does NOT correct this (see its docstring) because the per-target
    # Doppler "is not known before detection". That is true before the Doppler FFT --
    # but this cube is AFTER it, and every Doppler bin's f_D is then known exactly, so
    # the correction can be applied per bin with no chicken-and-egg at all.
    #
    # Left uncorrected it wrecks the virtual aperture for exactly the targets we care
    # about. MEASURED on one point target through this function, peak azimuth sidelobe:
    #     v =  0 m/s   -31.5 dB      (ideal Hann on a 64-element ULA)
    #     v =  3 m/s   -27.6 dB
    #     v =  5 m/s   -23.1 dB
    #     v =  8 m/s   -18.7 dB      <- and the corpus's spurious CFAR detections sit
    #                                   at a median -17.2 dB below their range's peak
    # With this correction all of those return to -31.4 dB.
    #
    # VALID ONLY BELOW THE UNAMBIGUOUS VELOCITY. Past it the Doppler bin is itself
    # aliased, so this de-rotates by the wrong amount and makes things worse (measured
    # -9.6 dB at 12 m/s on a config whose v_max is 9.7). Hence opt-in, and hence the
    # guard below rather than a silent wrong answer.
    if tdm_doppler_comp:
        if cfg.mimo != "tdm":
            raise ValueError(f"tdm_doppler_comp needs cfg.mimo == 'tdm', got {cfg.mimo!r}")
        n_tx, n_rx_ = int(cfg.n_tx), int(cfg.n_rx)
        n_ch, _, n_dop = rd.shape
        d = torch.arange(n_dop, device=rd.device, dtype=torch.float32) - n_dop // 2
        t = torch.arange(n_tx, device=rd.device, dtype=torch.float32)
        phase = -2.0 * math.pi * d.view(1, n_dop) * t.view(n_tx, 1) / (n_dop * n_tx)
        corr = torch.exp(1j * phase).to(rd.dtype).repeat_interleave(n_rx_, dim=0)
        rd = rd * corr.view(n_ch, 1, n_dop)

    n_channel = rd.shape[0]

    # ANGLE WINDOW. `adc_to_rd` Hann-windows range and Doppler, but the aperture FFT was
    # bare -- a rectangular aperture has a -13.3 dB peak sidelobe, which lands squarely in
    # the 6-14 dB CFAR range where targets live, and aperture sidelobes spread along
    # AZIMUTH at the target's own range. That is exactly the measured signature: ~99% of
    # detections diffuse rather than clustered on targets. MEASURED: the window cuts false
    # alarms 30x at a 10 dB threshold. It also costs ~1.8 dB of coherent gain and widens
    # the mainlobe, so it is not free -- but an unwindowed aperture is not a choice anyone
    # would defend, it was an omission.
    if angle_window:
        w = torch.hann_window(n_channel, periodic=False, device=rd.device,
                              dtype=torch.float32).to(rd.dtype)
        rd = rd * w.view(-1, 1, 1)

    n_fft = int(n_angle_fft) if n_angle_fft is not None else n_channel
    angle = torch.fft.fftshift(torch.fft.fft(rd, n=n_fft, dim=0), dim=0)   # [n_fft, R, D]
    power = angle.abs() ** 2

    # ZERO-DOPPLER NOTCH (MTI). Stationary clutter -- ground, buildings, bumper -- sits at
    # zero Doppler; moving targets do not. This is the canonical automotive rejector and
    # it is why collapsing Doppler before detection throws away the strongest discriminant
    # the sensor has.
    #
    # AUTO-GATED ON THE CONFIG (see `NOTCH_MIN_VMAX_MPS`), and the reason matters: it
    # only works when the targets are NOT aliased. On `radial_like` (DDMA over 12 TX,
    # v_max +-1.06 m/s) targets moving 0-8 m/s wrap around the Doppler axis and land
    # anywhere INCLUDING zero, so notching removes the targets rather than the clutter --
    # MEASURED, Pd 0.098 -> 0.000. It is applied only on a config whose v_max exceeds the
    # scene's speeds (see `benchmark_v1`); an explicit value always wins over the gate.
    if doppler_notch_bins > 0:
        c = power.shape[2] // 2          # adc_to_rd fftshifts Doppler; zero is the centre
        power = power.clone()
        power[:, :, max(0, c - doppler_notch_bins):c + doppler_notch_bins + 1] = 0.0

    if keep_doppler:
        return power                                                       # [n_fft, R, D]
    return power.max(dim=2).values                                         # [n_fft, R]


def _to_grid(power: torch.Tensor, cfg, grid: LabelGrid) -> torch.Tensor:
    """Resample `[n_angle, n_range_fine]` power onto `[grid.n_range, grid.n_azimuth]`.

    Accepts an optional leading Doppler axis (`[D, n_angle, n_range_fine]` ->
    `[D, grid.n_range, grid.n_azimuth]`), so a per-Doppler detector can resample the whole
    cube in one call and keep the resampling identical to the collapsed path's.

    Angle axis: nearest-neighbour (an upsample -- the FFT's `n_fft` bins map to
    `sin(az) = 2k/n_fft`, `k` in `[-n_fft/2, n_fft/2)`; no information is discarded).

    Range axis: PEAK-POOL -- each grid cell takes the MAX over every fine bin whose
    range falls inside it. This axis is a DOWNSAMPLE (typically 4 fine bins per cell),
    and the pre-2026-08-23 nearest-neighbour point-sample here was a measured defect:
    cell centres sampled only fine bins `stride*i + stride//2`, so a point target whose
    energy sat in any of the other `stride-1` fine bins per cell was simply INVISIBLE
    to the detector -- 3 of 4 possible target positions at the default stride (found
    when the A4 default flip exposed it; the old `max` Doppler collapse had masked it
    by riding a range-sidelobe skirt 0.8 dB above the floor). Peak-pooling is the
    "does ANY covered bin hold a target" semantics a detection cell means.

    WHAT PEAK-POOLING COSTS, stated precisely (batch review 2026-08-23; magnitude
    CORRECTED 2026-08-24 after two independent Monte Carlos refuted the review's
    "~2000x increase" figure -- reproduce: notes/tools/a14_pfa_mc.py): pooling makes
    every noise cell a max of `stride` exponentials -- INCLUDING the training
    annulus's cells, whose pooled mean rises by more than the cell-under-test's
    ordering advantage, so at a FIXED threshold the measured false-alarm rate
    actually DROPS (~20x at objectness 0.3 with the shipped CFAR geometry). Either
    way the operative point stands: absolute Pfa-at-threshold numbers from before
    and after this change are NOT comparable. What survives: the max applies
    identically to the cell under test and to its training annulus, and every
    shipped COMPARISON is scored at matched recall (the scorer re-thresholds per
    arm), so relative detector rankings stand.
    Also note the angle axis's nearest-neighbour UPSAMPLE replicates each FFT bin
    into ~3 grid columns at the default sizes, so grid-cell counts overstate the
    number of independent azimuth samples by that factor (variance estimates on the
    grid must divide by it).
    """
    n_angle, n_range_fine = power.shape[-2:]
    dev = power.device

    sin_src = 2.0 * (torch.arange(n_angle, device=dev, dtype=torch.float32) - n_angle // 2) / n_angle
    sin_dst = (torch.arange(grid.n_azimuth, device=dev, dtype=torch.float32) + 0.5) * grid.az_bin - 1.0
    # NEAREST source bin, not `torch.bucketize` (2026-08-27). bucketize returns the
    # INSERTION index -- the first source >= target, i.e. a CEILING -- so every
    # destination cell sampled the angle bin ABOVE it rather than the closest one, a
    # systematic half-bin (1/n_angle) bias toward negative sin_az. The docstring below
    # has always said "nearest-neighbour"; the code did not do it. Because only the
    # CLASSICAL arm resamples through here (learned arms consume the network input
    # directly), the bias was not common-mode -- it tilted every classical-vs-learned
    # comparison against the classical arm. `sin_src` is uniform with spacing
    # 2/n_angle starting at -1, so the nearest index is a rounding, no search needed.
    ai = torch.round((sin_dst - sin_src[0]) * (n_angle / 2.0)).long().clamp_(0, n_angle - 1)

    r_fine = (torch.arange(n_range_fine, device=dev, dtype=torch.float32) + 0.5) \
        * float(cfg.range_resolution_m)
    cell = (r_fine / grid.range_bin_m).long().clamp_(0, grid.n_range - 1)

    a = power[..., ai, :]                            # [..., n_azimuth, n_range_fine]
    out = torch.zeros(a.shape[:-1] + (grid.n_range,), dtype=power.dtype, device=dev)
    out.scatter_reduce_(-1, cell.expand(a.shape), a, reduce="amax", include_self=False)
    return out.transpose(-1, -2).contiguous()        # [..., n_range, n_azimuth]


def cfar_objectness(power: torch.Tensor, *, guard: int = CFAR_GUARD, train: int = CFAR_TRAIN,
                    min_db: float = CFAR_MIN_DB, max_db: float = CFAR_MAX_DB) -> torch.Tensor:
    """`[n_range, n_azimuth]` power -> CA-CFAR objectness in `[0, 1]`.

    Also accepts a batched `[D, n_range, n_azimuth]` and returns the same shape, so each
    Doppler slice can be tested against ITS OWN noise annulus rather than against a
    Doppler-collapsed one (see `DOPPLER_CFAR_FIRST`).

    The noise estimate under each cell is the mean over an annulus: a
    `(2*(guard+train)+1)^2` outer square minus its `(2*guard+1)^2` guard core. Both means
    come from `avg_pool2d` with `count_include_pad=False`, so cells at the map edge average
    only over real neighbours instead of being biased toward zero by padding.
    """
    squeezed = power.dim() == 2
    x = power[None, None] if squeezed else power[:, None]   # -> [B, 1, H, W]
    outer_k = 2 * (guard + train) + 1
    guard_k = 2 * guard + 1
    outer_sum = F.avg_pool2d(x, outer_k, stride=1, padding=guard + train,
                             count_include_pad=False) * (outer_k ** 2)
    guard_sum = F.avg_pool2d(x, guard_k, stride=1, padding=guard,
                             count_include_pad=False) * (guard_k ** 2)
    n_train = outer_k ** 2 - guard_k ** 2
    noise = ((outer_sum - guard_sum) / n_train).clamp_min(torch.finfo(power.dtype).tiny)

    ratio_db = 10.0 * torch.log10((x / noise).clamp_min(1e-12))
    obj = ((ratio_db - min_db) / (max_db - min_db)).clamp_(0.0, 1.0)
    return obj[0, 0] if squeezed else obj[:, 0]


def group_peaks(obj: torch.Tensor, *, radius: int = 1) -> torch.Tensor:
    """Local-maximum suppression: keep a cell only if it is the max of its neighbourhood.

    Without this the detector reports every cell above threshold, and a single target
    lights its whole footprint -- MEASURED at ~9 detections per target, which is exactly
    the 3x3 positive footprint `labels.encode_detection_labels` paints. That invalidates a
    classical-vs-learned comparison in BOTH directions: it inflates the classical detector's
    true-positive count if the matcher accepts duplicates, and destroys its precision if
    the matcher does not.

    Real detectors group peaks before reporting. `radius=1` is a 3x3 neighbourhood, matched
    to the label footprint; ties keep the first occurrence, which is why the comparison is
    `>=` against the pooled max only at the cell that attains it.
    """
    k = 2 * radius + 1
    pooled = F.max_pool2d(obj[None, None], k, stride=1, padding=radius)[0, 0]
    return torch.where(obj >= pooled, obj, torch.zeros_like(obj))


def classical_detection_map(cfg, adc: torch.Tensor, grid: LabelGrid, *,
                            peak_grouping: bool = True, group_radius: int = 1,
                            **kwargs) -> torch.Tensor:
    """Raw ADC -> a `[3, n_range, n_azimuth]` map in the detector's own output format.

    `doppler_reduce` selects how the Doppler axis is collapsed; see `DOPPLER_MAX` /
    `DOPPLER_SUM` / `DOPPLER_CFAR_FIRST` above for what each does to the noise statistics
    the CFAR threshold depends on. The default is `DOPPLER_SUM` since v1.1 (measured
    better at matched recall on both corpora tried, F48/F50); pass
    `doppler_reduce=DOPPLER_MAX` to reproduce pre-v1.1 numbers.

    Channel 0 is the CFAR objectness; the two regression channels are zero, so a decoded
    detection sits at its cell centre. That is the honest classical behaviour -- there is
    no sub-cell refinement without an interpolation stage this baseline deliberately omits.

    Since the 2026-08-17 surface-label convention that also means this baseline reports
    each object's SURFACE as its centre: CFAR fires where the energy is (which is exactly
    what the metric MATCHES on, so AP/AR are unaffected) but has no size model to convert
    that into an object centre, so its `range_rmse_m` carries the full centre-to-surface
    offset -- ~2.2 m for a car. Undoing that would take an extent estimator, not a
    threshold. See `e2e.ml.labels`.
    """
    doppler_reduce = kwargs.pop("doppler_reduce", DOPPLER_SUM)
    if doppler_reduce not in _DOPPLER_REDUCTIONS:
        raise ValueError(f"doppler_reduce must be one of {_DOPPLER_REDUCTIONS}, "
                         f"got {doppler_reduce!r}")

    # DEFAULTS FLIPPED 2026-08-29 (owner ballot). Together these are worth +28.7% AP on
    # the benchmark_v1 test split (0.1340 -> 0.1724, 2.08x -> 2.67x chance); NEITHER half
    # works alone -- compensation by itself is +0.6%. They are defaults HERE, on the
    # detector, and deliberately not on `range_azimuth_power`, which is also the
    # visualization path: notching zero Doppler would blank a static scene in a figure.
    #
    # `tdm_doppler_comp=None` means AUTO: on for TDM, off otherwise. It cannot be a bare
    # True, because the correction is only defined for TDM and `range_azimuth_power`
    # rightly raises on anything else. An explicit True/False is still honoured, and an
    # explicit True on a non-TDM config still raises rather than being quietly ignored.
    tdm_comp = kwargs.pop("tdm_doppler_comp", None)
    if tdm_comp is None:
        tdm_comp = (cfg.mimo == "tdm")
    # Same AUTO treatment for the notch, and for the same reason: a default that is right
    # for one config and destroys detection on another must be a function of the config,
    # not a constant. `None` -> on when the config's unambiguous velocity clears
    # NOTCH_MIN_VMAX_MPS. An explicit 0 or N still wins.
    notch = kwargs.pop("doppler_notch_bins", None)
    if notch is None:
        notch = 1 if float(cfg.max_velocity_mps) >= NOTCH_MIN_VMAX_MPS else 0
    power = range_azimuth_power(
        cfg, adc,
        n_angle_fft=kwargs.pop("n_angle_fft", None),
        angle_window=kwargs.pop("angle_window", True),
        doppler_notch_bins=notch,
        tdm_doppler_comp=tdm_comp,
        keep_doppler=doppler_reduce != DOPPLER_MAX,
    )

    if doppler_reduce == DOPPLER_MAX:
        obj = cfar_objectness(_to_grid(power, cfg, grid), **kwargs)
    elif doppler_reduce == DOPPLER_SUM:
        obj = cfar_objectness(_to_grid(power.sum(dim=2), cfg, grid), **kwargs)
    else:
        # Detect first, collapse second: CFAR every Doppler slice against its own noise
        # annulus, then keep each cell's best evidence across Doppler.
        cube = _to_grid(power.permute(2, 0, 1).contiguous(), cfg, grid)   # [D, R, A]
        obj = cfar_objectness(cube, **kwargs).max(dim=0).values
    if peak_grouping:
        obj = group_peaks(obj, radius=group_radius)
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
    print(f"  AP = {res['AP']:.4f} (interpolated precision-recall)")
    print(f"  AR = {res['AR']:.4f} ({res['AR_operating_point']})")
    # A permissive CFAR floor can recall almost everything by firing everywhere, so AR is
    # never printed without the precision and detection count that put it in context.
    print(f"  precision = {res['precision']:.4f}   "
          f"{res['tp']} TP / {res['n_detections']} detections / {res['n_targets']} targets")
    print(f"  range_rmse = {res['range_rmse_m']:.3f} m")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {k: v for k, v in res.items() if not isinstance(v, dict) or k == "resolution"},
            indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
