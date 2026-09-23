"""CFAR + learned rescoring head -- a two-stage detector that keeps azimuth by construction.

    Stage 1  classical CA-CFAR at a PERMISSIVE floor  -> up to K candidate cells
    Stage 2  a small conv head on a local patch of the Doppler-summed range-azimuth
             power map (and of the CFAR objectness) around each candidate -> one logit
    Output   the standard `{"detection": [B, 3, n_range_out, n_azimuth_out]}` contract,
             with the head's sigmoid scores SCATTERED into channel 0 at the candidate
             cells and zeros everywhere else.

Design from `notes/HANDOFF-2026-09-22.md` §5 (owner's proposal, §4 item 2). It exists
because of F83/F85: the ported nets never learn azimuth, CFAR gets azimuth for free from
the beamformer, and a rescorer can only reorder what stage 1 hands it -- so the floor is
permissive (0.05, against CFAR's normal ~0.3-0.5 operating point) and K is generous (64).
CFAR's max recall at its normal point is 0.888 (F85 table); the ceiling this head can
reach is whatever recall the candidate set carries AFTER `decode_detections`' NMS, which
is why `candidate_recall()` below is a first-class, reportable number rather than an
implementation detail -- and why it reports the post-NMS figure as the ceiling and the
raw candidate figure beside it as the overcount it is.

WHERE THIS FILE LIVES, AND WHY IT IS NOT IN `e2e/ml/models/`
---------------------------------------------------------------------------------
Written 2026-09-22 with two training runs in flight. `train.pipeline_fingerprint` hashes
`INPUT_PIPELINE_SOURCES` (`train.py:120-128`), whose last entry is the DIRECTORY
`e2e/ml/models` -- so adding a file there moves every checkpoint's fingerprint and every
existing run has to be re-certified (`python -m e2e.ml.recertify <run_dir>`). This module
is outside that list, so importing/using it changes no checkpoint's fingerprint.

REGISTRATION ONCE THE FREEZE LIFTS -- three edits, in this order:

1. `e2e/ml/train.py:92` -- `_MODEL_NAMES = ("fftradnet", "ssmradnet", "raddetnet")`
   becomes `(..., "raddetnet", "cfarhead")`. That tuple is also `--model`'s `choices`
   (`train.py:788`), so this is what makes the CLI accept it.
2. `e2e/ml/train.py:308` `build_model` -- add a branch beside the `raddetnet` one
   (`train.py:361-373`)::

       elif name == "cfarhead":
           if input_format != "rad":
               raise ValueError("cfarhead requires input_format='rad'")
           from e2e.ml.cfar_head import CFARHead
           model = CFARHead(in_channels, n_range_in, n_doppler_in,
                            n_range_out, n_azimuth_out,
                            cfg=cfg, grid=_grid_from_manifest(manifest))

   `cfg` is already in scope at that point (`train.py:334`); the grid is
   `manifest["grid"]` -- `build_grid_from_manifest()` in this module does that
   conversion, so the branch can call it and nothing new is needed in `train.py`.
3. `e2e/ml/train.py:640-642` -- the training loop's loss call. See "TRAINING" below;
   this head cannot be trained by `detection_loss`, and the change is one line.

Until then, `register(train_module)` in this module performs edits 1 and 2 at runtime by
monkey-patching, so every EVALUATION path works with a `cfarhead` checkpoint without
editing a fingerprinted file: `train.build_model`, `train.load_model_for_eval`,
`train.evaluate`, `compare_detectors.score_checkpoint`, `beat_cfar`, `e2e.ml.controls`
and the GUI's `ml` mode (`e2e/ml/blocks.py:601`, which also goes through
`train.build_model`). It does NOT do edit 3, so `register()` additionally makes
`train.train(..., "cfarhead")` RAISE rather than train on the wrong loss -- see `register`.

MOVING THIS FILE INTO `e2e/ml/models/` LATER moves the fingerprint of every checkpoint
ever trained. If that is done, run
`python -m e2e.ml.recertify e2e/ml/runs/b7_raddetnet e2e/ml/runs/b8_fftradnet_rad_frontend
e2e/ml/runs/b5_fftradnet_v3 e2e/ml/runs/b5_ssmradnet_v3 ...` immediately afterwards.

WHAT `forward` ACTUALLY GETS, AND THE ONE DEVIATION FROM THE §5 TEXT
---------------------------------------------------------------------------------
§5 says "computing CFAR inside `forward`". It cannot call `baseline.classical_detection_map`
there, and this is not a preference -- that function's signature is
`(cfg, adc, grid)` and takes RAW ADC, while a model's `forward` is handed the dataset's
`input_format="rad"` tensor: `[B, A, R, D]` float32 LOG-power in dB relative to each
frame's own median (`dataset.derive_network_input`, the "rad" branch). The ADC is gone by
then, and asking for it would mean a `dataset.py` change (frozen, and it would make this
model the only one that needs a second input).

What this module does instead reproduces `classical_detection_map(..., doppler_reduce=
DOPPLER_SUM)` to float32 rounding -- not approximately, ALGEBRAICALLY -- and the reason is
worth stating because it is the whole justification for the deviation:

    rad tensor  x   = 10*log10(pw / ref)      ref = median(pw) over the frame's cube
    recovered   p   = 10**(x/10) = pw / ref

    _to_grid is a max-pool over range and a nearest-neighbour gather over azimuth --
    both are positively homogeneous, so _to_grid(c*pw) == c*_to_grid(pw).
    cfar_objectness is a RATIO of a cell to the mean of its training annulus, so the
    per-frame constant `ref` cancels identically.

So `cfar_from_rad(x)` == `classical_detection_map(cfg, adc, grid)[0]` to float32
rounding, and `tests/test_ml_cfar_head.py::test_cfar_from_rad_matches_the_classical_arm`
asserts it on synthetic frames rather than leaving it as an argument.

ONE WRINKLE, found by measuring rather than by reasoning (2026-09-22): the CFAR FIELD
agrees to ~5e-7 always, but `group_peaks` keeps a cell iff `obj >= pooled`, an EXACT
comparison, and `_to_grid`'s azimuth axis is a nearest-neighbour upsample that replicates
each angle-FFT bin into `n_azimuth / n_angle` identical columns. On those plateaus a
last-bit difference between the two computation orders decides which replica wins, so the
GROUPED maps can differ cell-for-cell while the field does not. Measured: at replication 3
(64 angle bins -> 192 columns, which is the real corpus) no cell flips; at replication 12
(a 16-element single-TX config) 24 cells flip, max delta 0.044. In both cases every
flipped cell is below the 0.05 candidate floor, the delta among cells at or above the
floor is <= 4.7e-7, and **the candidate set is identical** -- which is the property stage
1 has to have. Quote the claim at that level; "bit-identical grouped maps" would be false.

The parity also means the head sees the classical arm's front end (zero-Doppler notch + TDM Doppler compensation) because
`derive_network_input` already applies those -- the F85 caveat 3 wording ("a learned head
on the classical front end") is literally true of this model.

The functions `baseline._to_grid`, `baseline.cfar_objectness` and `baseline.group_peaks`
are IMPORTED, not re-implemented; `_to_grid` is private and imported deliberately, since a
second resampler would be a second set of numbers (the same reason this module never
writes a second matcher and calls `metrics.match_detections`).

THE OUTPUT CONTRACT (copied from `e2e.ml.models.raddetnet`, verified against
`e2e.ml.labels.decode_detections`)
---------------------------------------------------------------------------------
`[B, 3, n_range_out, n_azimuth_out]`: channel 0 is objectness ALREADY through a sigmoid
(`decode_detections` thresholds it directly and `metrics.evaluate_dataset` sweeps it over
[0,1]); channels 1 and 2 are RAW regression residuals, in units of one range bin and one
azimuth bin respectively, added to the cell centre by `decode_detections`
(`labels.py:408-412`).

This head leaves channels 1-2 at ZERO, which is the same choice
`classical_detection_map` makes and for the same reason: there is no extent/interpolation
stage here, so a detection sits at its cell centre. Consequence, stated because it is a
real cost: `range_rmse_m` carries the full centre-to-surface offset (~2.2 m for a car, see
`baseline.classical_detection_map`'s docstring). AP/AR are unaffected -- the metric matches
on the SURFACE range, which is the cell centre either way -- and F83 measured that zeroing
both regression channels moves AP by +/-0.002.

TRAINING -- `train.py`'s loss CANNOT train this head; the change is one line
---------------------------------------------------------------------------------
`train.py:640-642` does, unconditionally::

        pred = model(x)["detection"]
        loss, parts = detection_loss(pred, y, gamma=..., reg_weight=..., cls_normalize=...)

Two things break:

1. **Gradient coverage.** `detection_loss` is a focal BCE over ALL R*A cells. Every
   non-candidate cell of this model's output is a literal constant zero with no path to a
   parameter, so those terms have exactly zero gradient while still dominating the value
   and the `cls_normalize="positives"` denominator. The head would be trained by the
   fraction of the map it emits, scaled by a denominator computed from the rest.
2. **The wrong target.** §5 requires BCE on *candidate-vs-ground-truth matching under the
   METRIC's own criterion*. `y` is the dense label map, whose positives are a 3x3
   FOOTPRINT painted around each target's surface cell (`labels.encode_detection_labels`)
   -- cell-wise overlap, not `metrics.MatchCriterion`. Training on `y[0]` would teach the
   head a different notion of "hit" from the one it is scored by, and in particular would
   ignore the per-target azimuth-tolerance widening (`metrics._normalized_distance` widens
   `max_sin_az_err` to the target's own angular half-extent -- roughly 2x for a car at
   20 m, so a candidate the metric counts as a hit would be labelled a miss).

So this module implements `CFARHead.loss(output, y, targets=...)`, returning
`(total, {"cls": float, "reg": float})` -- the exact tuple shape `train.py:668-670`
consumes. The integration is::

        out = model(x)
        pred = out["detection"]
        if hasattr(model, "loss"):
            loss, parts = model.loss(out, y)
        else:
            loss, parts = detection_loss(pred, y, gamma=..., ...)

`parts["reg"]` is always 0.0 (no regression head), so `--reg-weight` is inert for this
model and `history["train_reg_loss"]` will be a column of zeros. Say so in the run notes
rather than letting a reader infer a converged regression term. `--gamma` and
`--cls-normalize` are inert too: the loss here is a plain BCE over a balanced-ish
candidate set, not a focal loss over 1365:1 background.

TRAIN WITH `--amp off`. `--amp auto` turns autocast on for CUDA, and stage 1 is a
CLASSICAL computation with ~60 dB of dynamic range inside it (`10**(x/10)`, a Doppler
sum, a median, then a ratio of means). Half precision there is not a 5% memory
optimisation, it is a change to the detector. MEASURED 2026-09-22 on one synthetic
noise frame (benchmark_v1 shrunk to 32 chirps x 64 samples, CPU): rounding only the INPUT
cube to fp16 and back moves the CFAR objectness by up to **0.127** on its [0,1] scale --
a quarter of the metric's whole threshold sweep, from a rounding that autocast would do
for free. Not measured on the real corpus and not measured under autocast proper; that is
a reason to leave AMP off here, not a characterisation of how bad it is.

**Exact vs approximate targets.** `loss(out, y)` alone recovers the per-frame target list
by running `labels.decode_detections(grid, y[b], threshold=0.5)` on the ground-truth map.
That is the encoder's own inverse and it recovers each target's centre range and
sin-azimuth exactly, but it CANNOT recover `cross_range_half_extent_m` (the 5th tuple
element), which is not encoded in the map at all -- so the matcher falls back to the fixed
`max_sin_az_err=0.06` floor instead of the widened per-target tolerance. That makes the
training criterion STRICTER than the scoring criterion (some scored hits are labelled
negative). Pass `targets=[ds.targets(i) for i in batch]` to get the identical criterion;
doing that from `train.py` needs the DataLoader to carry the target list, which is a
`dataset.py` change and therefore deferred. The approximation is named here so it is not
rediscovered as a bug: it costs recall in training labels, never precision.

SECOND LIMITATION OF THE FALLBACK (review finding, 2026-09-22): `decode_detections` runs
its own greedy NMS, and `e2e/ml/labels.py`'s module docstring records that this "can
suppress a target sandwiched 1-2 cells between two others entirely" when label footprints
overlap. So the fallback can DROP a target's label outright, not merely narrow its
tolerance -- a dropped target turns every candidate on it into a trained-for false alarm.
The corpus generator enforces a minimum target separation that is supposed to keep the
3x3 footprints disjoint, so this is believed rare and was NOT reproduced when looked for;
it is written down because "believed rare" is not "cannot happen", and `targets=` avoids
it entirely.

CONTROLS (§5 item 4) -- `python -m e2e.ml.cfar_head controls --help`
---------------------------------------------------------------------------------
All four are CLI-runnable and all four are scored by `compare_detectors._score_arm`, i.e.
the same AP / matched-recall-FA protocol as every other arm:

  `cfar`            stage 1 alone at its normal operating point (the 0.301 reference row
                    on b1_bench_v3 test, F85 -- `compare_detectors.score_classical`).
  `cfar_random`     the candidate set, rescored with seeded uniform noise. Answers "is the
                    gain the HEAD, or just the permissive floor + K-truncation?" This is
                    the control that would have caught a rescorer that does nothing.
  `head_random_cand` the head applied to K uniformly-random cells instead of CFAR's.
                    Answers "is the gain stage 1's, or stage 2's?"
  `fixed_threshold` no local adaptivity at all: a global monotone map of the same
                    Doppler-summed grid power, peak-grouped. This is the rung below CFAR
                    on the verifier's ladder (F85: "global threshold 0.18-0.22 -> CFAR
                    0.30 -> RADDetNet 0.48"; `ML_DETECTION_BRIEF.md` measured 0.169 for a
                    fixed global threshold against CFAR's 0.247 on the network's own cube,
                    2026-09-20, +0.079 CI [+0.062,+0.095]). Re-measure it here rather than
                    quoting 0.169: that number was measured on a different cube derivation.

These four are this model's OWN controls. The F83/F85 control battery
(`python -m e2e.ml.controls --checkpoint cfarhead=<path>`: deranged-label retention,
azimuth-only vs a constant map, stripe rank-1) needs no work here -- it goes through
`train.load_model_for_eval`, so `register()` is enough, and it was smoke-run against a
`cfarhead` checkpoint on 2026-09-22. Run BOTH before any claim: these say the two stages
each earn their place, those say the model reads the frame.

NOT RUN ON THE REAL CORPUS YET (2026-09-22): both GPUs were training, so every number
this module can produce is still pending. Nothing in this file states a measured result
about detection performance. What WAS run, on CPU, on an 8-frame synthetic corpus, and is
a plumbing check rather than a result: all five control arms execute and are bit-repeatable
across processes; `model.loss` trains the head (BCE 0.726 -> 0.085 in 40 Adam steps) and
the fitted head outscores the CFAR ordering of its own candidate set on those frames
(train-fit AP 0.851 vs 0.727 -- four frames, no validation split, so this says the
gradient flows, nothing more); a saved checkpoint round-trips through
`compare_detectors.score_checkpoint` and `e2e.ml.controls`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from e2e.ml.baseline import CFAR_GUARD, CFAR_MAX_DB, CFAR_MIN_DB, CFAR_TRAIN
from e2e.ml.baseline import _to_grid, cfar_objectness, group_peaks
from e2e.ml.labels import LabelGrid, decode_detections
from e2e.ml.metrics import MatchCriterion, match_detections
# The metric's OWN one-pair distance. Private, imported deliberately: the alternative is a
# second definition of "close enough", which is the one thing this module must not have.
from e2e.ml.metrics import _normalized_distance


def _matches(det, tgt, criterion: MatchCriterion) -> bool:
    """Would `metrics.match_detections` accept this pair, ignoring competition?

    `match_detections` is greedy and one-to-one; this is the underlying predicate it
    applies (normalized distance <= 1.0, the boundary counting as a match). Used only by
    `CFARHead.candidate_recall`'s oracle, where one-to-one is exactly the wrong question.
    """
    return _normalized_distance(det, tgt, criterion) <= 1.0

#: Registry name once `train.py:92` lists it. Chosen so `type(model).__name__.lower()`
#: (`train.evaluate`, `compare_detectors.score_checkpoint`) maps 1:1 onto it, exactly as
#: `FFTRadNet -> fftradnet` does.
MODEL_NAME = "cfarhead"

#: Stage-1 defaults. Both are constructor arguments; these are the §5 values.
DEFAULT_CFAR_FLOOR = 0.05      # CFAR objectness in [0,1]; 0.05 == 1 dB above local noise
DEFAULT_MAX_CANDIDATES = 64
DEFAULT_PATCH = 7              # odd; the head's receptive field in label-grid cells

#: The head's power channel is dB relative to the frame's median grid power, and that is
#: UNBOUNDED above -- a bright target can sit 60+ dB up. Its sibling channel (CFAR
#: objectness) is clamped to [0,1] by construction, so leaving this one unbounded lets a
#: single hot frame dominate the first convolution's activations purely through scale.
#: Clamped to a range wide enough to contain every real return (the repo's own display
#: convention clips at -40 dB; `cfar_objectness` saturates at +20 dB of CFAR ratio, and
#: a peak 60 dB over the frame MEDIAN is already far past any target this corpus carries).
#: This is a feature-conditioning choice, not a detection threshold: stage 1's candidate
#: set is computed from the unclamped power and is unaffected.
FEATURE_DB_MIN = -40.0
FEATURE_DB_MAX = 60.0


def build_grid_from_manifest(manifest: Dict) -> LabelGrid:
    """`manifest["grid"]` -> `LabelGrid`. Same three fields `train._load_grid` reads."""
    g = manifest["grid"]
    return LabelGrid(n_range=int(g["n_range"]), n_azimuth=int(g["n_azimuth"]),
                     max_range_m=float(g["max_range_m"]))


# --------------------------------------------------------------------------------
# Stage 1: the classical front end, recovered from the `rad` tensor
# --------------------------------------------------------------------------------
def grid_power_from_rad(x: Tensor, cfg, grid: LabelGrid) -> Tensor:
    """`rad` tensor `[A, R, D]` (or `[B, A, R, D]`) -> Doppler-summed grid power.

    Returns `[grid.n_range, grid.n_azimuth]` (or `[B, ...]`), in units of the frame's own
    median cube power -- i.e. `classical_detection_map`'s Doppler-summed, grid-resampled
    power divided by a per-frame constant. Every consumer below is either a ratio or a
    dB-relative quantity, so that constant never reaches a number.
    """
    p = torch.pow(10.0, x / 10.0)            # undo dataset.derive_network_input's log
    return _to_grid(p.sum(dim=-1), cfg, grid)


def cfar_from_rad(x: Tensor, cfg, grid: LabelGrid, *, guard: int = CFAR_GUARD,
                  train: int = CFAR_TRAIN, min_db: float = CFAR_MIN_DB,
                  max_db: float = CFAR_MAX_DB, peak_grouping: bool = True,
                  group_radius: int = 1) -> Tensor:
    """`rad` tensor -> CA-CFAR objectness on the label grid, `[R, A]` or `[B, R, A]`.

    SCOPE OF THE PARITY CLAIM (read it before quoting it). Equal to float32 rounding with
    `baseline.classical_detection_map(cfg, adc, grid)[0]` for the ADC that produced `x`,
    **at that function's DEFAULTS** -- `doppler_reduce=DOPPLER_SUM`, auto notch, auto TDM
    compensation -- because those are the front-end decisions
    `dataset.derive_network_input` bakes into the `rad` tensor. It is NOT equal for
    `doppler_reduce=DOPPLER_MAX` or `DOPPLER_CFAR_FIRST`: both need the uncollapsed
    per-Doppler cube through `_to_grid`, and this function sums Doppler first. Nor is it
    equal if a caller overrides `doppler_notch_bins` or `tdm_doppler_comp` on the classical
    side, since the `rad` tensor was built with the auto values.

    VERIFIED (`tests/test_ml_cfar_head.py::test_cfar_from_rad_matches_the_classical_arm`,
    2026-09-22, CPU, synthetic noise frames) on two configs -- `benchmark_v1` shrunk, TDM,
    notch ON; and a single-TX variant with the notch OFF -- at <= 1e-5 absolute on the
    [0,1] objectness scale for the UNGROUPED field, and with an IDENTICAL candidate set in
    both. With `peak_grouping=True` the maps can differ on azimuth plateaus below the
    floor; see the module docstring's "ONE WRINKLE" for the measurement.
    """
    pw = grid_power_from_rad(x, cfg, grid)
    obj = cfar_objectness(pw, guard=guard, train=train, min_db=min_db, max_db=max_db)
    if not peak_grouping:
        return obj
    if obj.dim() == 2:
        return group_peaks(obj, radius=group_radius)
    return torch.stack([group_peaks(o, radius=group_radius) for o in obj.unbind(0)])


def global_threshold_objectness(x: Tensor, cfg, grid: LabelGrid, *,
                                min_db: float = CFAR_MIN_DB, max_db: float = CFAR_MAX_DB,
                                peak_grouping: bool = True,
                                group_radius: int = 1) -> Tensor:
    """The NO-LOCAL-ADAPTIVITY rung: the same power map, thresholded globally.

    Identical to `cfar_from_rad` except the per-cell noise estimate (the CFAR training
    annulus) is replaced by ONE number per frame -- the median of the grid power. The
    objectness is then the same monotone `min_db..max_db` map, so the metric's threshold
    sweep spans the same dB range for both arms and the comparison isolates adaptivity
    and nothing else (`ML_DETECTION_BRIEF.md` §4, 2026-09-20).
    """
    pw = grid_power_from_rad(x, cfg, grid)
    flat = pw.reshape(pw.shape[0], -1) if pw.dim() == 3 else pw.reshape(1, -1)
    ref = flat.median(dim=1).values.clamp_min(torch.finfo(pw.dtype).tiny)
    ref = ref.view(-1, 1, 1) if pw.dim() == 3 else ref.view(1, 1)
    ratio_db = 10.0 * torch.log10((pw / ref).clamp_min(1e-12))
    obj = ((ratio_db - min_db) / (max_db - min_db)).clamp_(0.0, 1.0)
    if not peak_grouping:
        return obj
    if obj.dim() == 2:
        return group_peaks(obj, radius=group_radius)
    return torch.stack([group_peaks(o, radius=group_radius) for o in obj.unbind(0)])


def select_candidates(obj: Tensor, *, floor: float, max_candidates: int
                      ) -> Tuple[Tensor, Tensor, Tensor]:
    """`[R, A]` peak-grouped objectness -> `(rows, cols, scores)` for up to K candidates.

    Deterministic, including under ties: candidates are ordered by objectness descending
    and, at equal objectness, by ROW-MAJOR CELL INDEX ascending (`torch.argsort`'s stable
    mode on a pre-sorted-by-index list), never by an unspecified kernel order. Cells at
    exactly `floor` are kept (`>=`), so "every CFAR detection above the floor" has no
    boundary ambiguity.
    """
    mask = obj >= float(floor)
    idx = mask.nonzero(as_tuple=False)                 # already row-major ascending
    if idx.numel() == 0:
        empty = torch.zeros(0, dtype=torch.long, device=obj.device)
        return empty, empty, torch.zeros(0, dtype=obj.dtype, device=obj.device)
    scores = obj[idx[:, 0], idx[:, 1]]
    order = torch.argsort(scores, descending=True, stable=True)
    keep = order[: int(max_candidates)]
    return idx[keep, 0], idx[keep, 1], scores[keep]


# --------------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------------
class CFARHead(nn.Module):
    """CFAR candidates + a learned patch rescorer. `input_format="rad"`.

    Constructor mirrors `RADDetNet`'s first five positional arguments so
    `train.build_model` can call it the same way, plus the pieces stage 1 needs:

    * `cfg`    -- a `RadarConfig` (only `range_resolution_m` is read, by `_to_grid`).
    * `grid`   -- the `LabelGrid` the output lives on; `n_range_out`/`n_azimuth_out` must
                  agree with it, and the constructor raises if they do not rather than
                  resampling silently.
    * `cfar_floor`, `max_candidates`, `patch` -- the §5 knobs, all constructor arguments.

    Two seams exist for the controls and for tests, both default `None` and both
    documented as such (`self.candidate_fn`, `self.score_fn`); see `controls()`.
    """

    def __init__(self, in_azimuth: int, n_range_in: int, n_doppler: int,
                 n_range_out: int, n_azimuth_out: int, *,
                 cfg, grid: LabelGrid,
                 cfar_floor: float = DEFAULT_CFAR_FLOOR,
                 max_candidates: int = DEFAULT_MAX_CANDIDATES,
                 patch: int = DEFAULT_PATCH,
                 width: int = 32, hidden: int = 64):
        super().__init__()
        if int(patch) % 2 != 1 or int(patch) < 1:
            raise ValueError(f"patch must be a positive odd number of cells, got {patch!r}")
        if (int(n_range_out), int(n_azimuth_out)) != (int(grid.n_range), int(grid.n_azimuth)):
            raise ValueError(
                f"output geometry ({n_range_out}, {n_azimuth_out}) disagrees with the "
                f"LabelGrid ({grid.n_range}, {grid.n_azimuth}); stage 1 detects ON the "
                "label grid, so these cannot differ")
        self.in_azimuth = int(in_azimuth)
        self.n_range_in = int(n_range_in)
        self.n_doppler = int(n_doppler)
        self.n_range_out = int(n_range_out)
        self.n_azimuth_out = int(n_azimuth_out)
        self.cfg = cfg
        self.grid = grid
        self.cfar_floor = float(cfar_floor)
        self.max_candidates = int(max_candidates)
        self.patch = int(patch)

        #: Control/test seam. `candidate_fn(obj, floor, k) -> (rows, cols, scores)`
        #: replaces stage 1's selection (used by the random-candidate control);
        #: `score_fn(patches, centre, rows, cols) -> logits [K]` replaces stage 2
        #: (used by the random-rescoring control and by the oracle test). Neither is a
        #: parameter, neither is saved in the state dict, and both are None in training.
        self.candidate_fn: Optional[Callable] = None
        self.score_fn: Optional[Callable] = None

        w = int(width)
        # Two input channels: the local dB power (frame-median referenced, so
        # scale-invariant exactly as CFAR is) and the CFAR objectness. No BatchNorm: the
        # "batch" here is a variable-length candidate set, so a batch statistic would make
        # one candidate's score depend on which other candidates the frame happened to
        # produce -- and on K, which the floor controls.
        self.body = nn.Sequential(
            nn.Conv2d(2, w, 3, padding=1), nn.SiLU(inplace=True),
            nn.Conv2d(w, w, 3, padding=1), nn.SiLU(inplace=True),
        )
        # + 2: the candidate's OWN dB power and CFAR objectness, handed to the classifier
        # directly so the head never has to rediscover the centre pixel of its patch.
        self.classifier = nn.Sequential(
            nn.Linear(w * self.patch * self.patch + 2, int(hidden)),
            nn.SiLU(inplace=True),
            nn.Linear(int(hidden), 1),
        )

    # ---- stage 1 ----------------------------------------------------------------
    def _stage1(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """`[B, A, R, D]` -> `(pw, pdb, obj, obj_peaks)`, all `[B, R, A]`.

        THE single stage-1 implementation. `forward`, `cfar` and `candidates` all go
        through here; an earlier version inlined it in `forward` and called
        `cfar_from_rad` from `candidates`, which was two copies that happened to agree
        (review finding, 2026-09-22) and nothing held them together.

        `obj` is the raw CFAR field (the head's patch feature); `obj_peaks` is that field
        after `group_peaks`, which is what candidates are drawn from.
        """
        pw = grid_power_from_rad(x, self.cfg, self.grid)
        ref = pw.reshape(pw.shape[0], -1).median(dim=1).values \
                .clamp_min(torch.finfo(pw.dtype).tiny).view(-1, 1, 1)
        pdb = (10.0 * torch.log10((pw / ref).clamp_min(1e-12))
               ).clamp(FEATURE_DB_MIN, FEATURE_DB_MAX)
        obj = cfar_objectness(pw)
        obj_peaks = torch.stack([group_peaks(o, radius=1) for o in obj.unbind(0)])
        return pw, pdb, obj, obj_peaks

    def cfar(self, x: Tensor) -> Tensor:
        """`[B, A, R, D]` rad tensor -> `[B, R, A]` peak-grouped CFAR objectness.

        Equal to `cfar_from_rad(x, cfg, grid)` -- pinned by
        `tests/test_ml_cfar_head.py::test_model_stage1_equals_the_standalone_function`,
        so the standalone function (which the classical-parity test compares against)
        and the one `forward` actually runs cannot drift apart."""
        return self._stage1(x if x.dim() == 4 else x[None])[3]

    def candidates(self, x: Tensor) -> List[Tuple[Tensor, Tensor, Tensor]]:
        """Per-sample `(rows, cols, cfar_scores)`. Public: the controls and the
        candidate-recall report need the same set `forward` scores."""
        obj = self.cfar(x if x.dim() == 4 else x[None])
        pick = self.candidate_fn or (
            lambda o, floor, k: select_candidates(o, floor=floor, max_candidates=k))
        return [pick(o, self.cfar_floor, self.max_candidates) for o in obj.unbind(0)]

    # ---- stage 2 ----------------------------------------------------------------
    def _patches(self, feat: Tensor, rows: Tensor, cols: Tensor) -> Tensor:
        """`[2, R, A]` feature map + K cells -> `[K, 2, P, P]` patches (zero-padded)."""
        half = self.patch // 2
        padded = F.pad(feat[None], (half, half, half, half))[0]
        d = torch.arange(self.patch, device=feat.device)
        ri = (rows.view(-1, 1, 1) + d.view(1, -1, 1))
        ci = (cols.view(-1, 1, 1) + d.view(1, 1, -1))
        return padded[:, ri, ci].permute(1, 0, 2, 3).contiguous()

    def _logits(self, patches: Tensor, centre: Tensor) -> Tensor:
        z = self.body(patches).flatten(1)
        return self.classifier(torch.cat([z, centre], dim=1)).squeeze(1)

    # ---- the contract -----------------------------------------------------------
    def forward(self, x: Tensor) -> Dict[str, object]:
        """`[B, A, R, D]` -> `{"detection": [B, 3, R_out, A_out], ...}`.

        The extra keys (`logits`, `rows`, `cols`, `cfar`) are per-sample lists carrying
        the candidate set and its pre-sigmoid scores; `loss()` reads them. Every existing
        consumer takes `out["detection"]` and is unaffected -- checked against all four:
        `train._predict_split`, `compare_detectors.score_checkpoint`,
        `e2e.ml.blocks.NeuralDetectorBlock` (the GUI) and `e2e.ml.detect_viz`.

        Note which CFAR map goes where: candidates are chosen from the PEAK-GROUPED
        objectness (`group_peaks`, so one cell per local maximum -- the same set
        `classical_detection_map` would report), but the patch handed to the head is the
        UNGROUPED field, because the shape of the local CFAR response around a candidate
        is exactly the evidence the head is there to read and grouping erases it.
        """
        if x.dim() != 4:
            raise ValueError(f"expected [B, A, R, D], got {tuple(x.shape)}")
        b, a, r, d = x.shape
        if (a, r, d) != (self.in_azimuth, self.n_range_in, self.n_doppler):
            raise ValueError(
                f"expected [B, {self.in_azimuth}, {self.n_range_in}, {self.n_doppler}], "
                f"got {tuple(x.shape)}")

        _pw, pdb, obj, obj_peaks = self._stage1(x)                        # each [B, R, A]

        pick = self.candidate_fn or (
            lambda o, floor, k: select_candidates(o, floor=floor, max_candidates=k))

        out = x.new_zeros((b, 3, self.n_range_out, self.n_azimuth_out))
        all_logits, all_rows, all_cols, all_cfar = [], [], [], []
        for i in range(b):
            rows, cols, scores = pick(obj_peaks[i], self.cfar_floor, self.max_candidates)
            all_rows.append(rows)
            all_cols.append(cols)
            all_cfar.append(scores)
            if rows.numel() == 0:
                all_logits.append(x.new_zeros(0))
                continue
            feat = torch.stack([pdb[i], obj[i]], dim=0)                    # [2, R, A]
            patches = self._patches(feat, rows, cols)                      # [K, 2, P, P]
            centre = torch.stack([pdb[i][rows, cols], obj[i][rows, cols]], dim=1)
            logits = (self.score_fn(patches, centre, rows, cols) if self.score_fn
                      else self._logits(patches, centre))
            all_logits.append(logits)
            out[i, 0, rows, cols] = torch.sigmoid(logits).to(out.dtype)
        return {"detection": out, "logits": all_logits, "rows": all_rows,
                "cols": all_cols, "cfar": all_cfar}

    # ---- training ---------------------------------------------------------------
    def detections_for(self, rows: Tensor, cols: Tensor, scores: Sequence[float]
                       ) -> List[Tuple[float, float, float, float]]:
        """Candidate cells -> `metrics`-shaped `(range_m, sin_az, score, surface_range_m)`.

        Identical geometry to `labels.decode_detections` with zero regression channels,
        which is what this model emits -- so a candidate's tuple here and its tuple after
        a round trip through `decode_detections` are the same numbers.
        """
        rb, ab = self.grid.range_bin_m, self.grid.az_bin
        out = []
        for i, j, s in zip(rows.tolist(), cols.tolist(), list(scores)):
            r_center = (i + 0.5) * rb
            out.append((r_center, -1.0 + (j + 0.5) * ab, float(s), r_center))
        return out

    def loss(self, output: Dict, target_map: Optional[Tensor] = None, *,
             targets: Optional[Sequence[Sequence]] = None,
             criterion: Optional[MatchCriterion] = None,
             gt_threshold: float = 0.5, **_ignored
             ) -> Tuple[Tensor, Dict[str, float]]:
        """BCE on candidate-vs-ground-truth matching, under `metrics`' OWN criterion.

        `output` is this model's `forward` dict. Ground truth comes from `targets` (a
        per-sample list of `labels.targets_in_grid` tuples -- the exact criterion) or,
        failing that, from `target_map` decoded with `labels.decode_detections` (the
        approximation; see the module docstring's "Exact vs approximate targets" -- it
        loses each target's cross-range half-extent, so the azimuth tolerance is the
        0.06 floor instead of the widened per-target one).

        Returns `(total, {"cls": float, "reg": float})`, the tuple `train.py:668-670`
        consumes. `"reg"` is always 0.0: this head has no regression output.

        The matcher is `metrics.match_detections` -- the same function
        `evaluate_frame`/`evaluate_dataset` use. There is deliberately no second matcher
        in this module.

        WHICH candidate gets the positive label: candidates are handed to the matcher
        scored by their STAGE-1 CFAR objectness, not by the head's current logit. The
        matcher is greedy in score order, so a target within tolerance of several
        candidates is claimed by the strongest CFAR cell -- a fixed, model-independent
        assignment for a given frame. Scoring them by the head's own logit instead would
        make the training target move with the prediction (a self-fulfilling label), which
        is why it is not done. The tolerance spans ~6 azimuth bins at the 0.06 floor, so
        the positive cell is often NOT the cell nearest the target; that is the metric's
        semantics and `tests/test_ml_cfar_head.py` pins it.
        """
        criterion = criterion or MatchCriterion()
        logits_list = output["logits"]
        rows_list, cols_list, cfar_list = output["rows"], output["cols"], output["cfar"]
        n = len(logits_list)
        if targets is None:
            if target_map is None:
                raise ValueError("loss() needs either `targets` or the dense `target_map`")
            tm = target_map if target_map.dim() == 4 else target_map[None]
            # decode_detections returns (range_m, sin_az, SCORE, surface_range_m) -- slot
            # 2 holds a score where a Target tuple holds a class string. The matcher never
            # reads slot 2 of a target (only [1], _surface_range -> [3] and _half_extent
            # -> [4]), so these are valid Targets; the mismatch is inert and named here so
            # nobody "fixes" it into a class lookup.
            targets = [decode_detections(self.grid, tm[i], threshold=gt_threshold)
                       for i in range(n)]

        flat_logits, flat_labels = [], []
        for i in range(n):
            logits = logits_list[i]
            if logits.numel() == 0:
                continue
            dets = self.detections_for(rows_list[i], cols_list[i], cfar_list[i].tolist())
            matches, _fp, _fn = match_detections(dets, list(targets[i]), criterion)
            lab = torch.zeros(logits.shape, dtype=logits.dtype, device=logits.device)
            for di, _gi in matches:
                lab[di] = 1.0
            flat_logits.append(logits)
            flat_labels.append(lab)

        if not flat_logits:
            # No candidate anywhere in the batch (an all-noise frame at a high floor).
            # Return a parameter-connected zero so `.backward()` is still legal.
            zero = sum(p.sum() for p in self.parameters()) * 0.0
            return zero, {"cls": 0.0, "reg": 0.0}

        z = torch.cat(flat_logits)
        y = torch.cat(flat_labels)
        cls = F.binary_cross_entropy_with_logits(z, y)
        return cls, {"cls": float(cls.detach().item()), "reg": 0.0}

    # ---- reportable diagnostics --------------------------------------------------
    def candidate_recall(self, frames, targets: Sequence[Sequence], *,
                         criterion: Optional[MatchCriterion] = None) -> Dict[str, float]:
        """Stage 2's hard ceiling: the recall the best possible rescoring could reach.

        Reported as TWO numbers, because they differ and the difference is not stage 2's
        to fix (review finding, 2026-09-22):

        * `candidate_recall` -- the ACHIEVABLE ceiling. Candidates are given a perfect
          oracle score (1.0 if they match any target under `criterion`, 0.0 otherwise),
          scattered into a `[3, R, A]` map and put through `labels.decode_detections`,
          exactly as every scored arm is. No rescoring can beat this, because no scoring
          separates matching from non-matching candidates better than perfectly.
        * `candidate_recall_pre_nms` -- the raw candidate set, matched directly. This is
          what a naive reading of "did stage 1 propose it" gives, and it is an OVERCOUNT:
          `decode_detections` suppresses any detection within Chebyshev distance 2 of a
          higher-scoring one (`nms_footprint=3`), a COARSER radius than the candidate
          set's own `group_peaks(radius=1)`. Two candidates 2 cells apart therefore both
          count here and collapse to one when scored. Quoting this as "the ceiling" would
          overstate what the detector can do; it is reported so the gap is visible.

        `frames` is an iterable of per-frame `[A, R, D]` rad tensors (or a batched
        `[B, A, R, D]` tensor) -- iterated one frame at a time, because a whole test
        split of these cubes is gigabytes.
        """
        criterion = criterion or MatchCriterion()
        hit = raw_hit = tot = ncand = nframes = 0
        for x, tgts in zip(frames, targets):
            rows, cols, scores = self.candidates(x[None] if x.dim() == 3 else x)[0]
            tgts = list(tgts)
            dets = self.detections_for(rows, cols, scores.tolist())
            raw_hit += len(match_detections(dets, tgts, criterion)[0])

            # The oracle: 1.0 for any candidate that COULD match a target. Not the greedy
            # one-to-one assignment -- a rescorer is free to promote whichever of several
            # co-located candidates survives NMS, so the ceiling must let it.
            oracle = torch.zeros(rows.numel(), device=rows.device)
            for di, det in enumerate(dets):
                if any(_matches(det, t, criterion) for t in tgts):
                    oracle[di] = 1.0
            omap = torch.zeros((3, self.grid.n_range, self.grid.n_azimuth),
                               device=rows.device)
            if rows.numel():
                omap[0, rows, cols] = oracle
            decoded = decode_detections(self.grid, omap, threshold=0.5)
            hit += len(match_detections(decoded, tgts, criterion)[0])

            tot += len(tgts)
            ncand += len(dets)
            nframes += 1
        return {"candidate_recall": (hit / tot) if tot else float("nan"),
                "candidate_recall_pre_nms": (raw_hit / tot) if tot else float("nan"),
                "n_targets": float(tot), "candidates_per_frame": ncand / max(nframes, 1)}


# --------------------------------------------------------------------------------
# Registration shim (edits 1 and 2 of the module docstring, at runtime)
# --------------------------------------------------------------------------------
def register(train_module=None) -> None:
    """Teach `e2e.ml.train` about `cfarhead` without editing a fingerprinted file.

    Idempotent. Adds `MODEL_NAME` to `train._MODEL_NAMES` (which is also `--model`'s
    `choices`) and wraps `train.build_model` so the new name is handled and every other
    name falls through to the original. Because `train.load_model_for_eval`,
    `compare_detectors.score_checkpoint` and `e2e.ml.blocks.NeuralDetectorBlock` all
    resolve `build_model` through the `train` module at call time, patching it here
    reaches all of them.

    Delete this function when `train.py:92` and `train.py:361` carry the real branch.
    """
    if train_module is None:
        from e2e.ml import train as train_module

    if MODEL_NAME in getattr(train_module, "_MODEL_NAMES", ()):
        return
    train_module._MODEL_NAMES = tuple(train_module._MODEL_NAMES) + (MODEL_NAME,)
    original = train_module.build_model

    def build_model(name: str, manifest: Dict, *, device=None, ssm_chunk_size=None):
        if name != MODEL_NAME:
            return original(name, manifest, device=device, ssm_chunk_size=ssm_chunk_size)
        from e2e.radar_config import RadarConfig

        input_format = manifest.get("input_format", "rd")
        if input_format != "rad":
            raise ValueError(
                f"{MODEL_NAME} requires input_format='rad' (stage 1 is the classical "
                f"beamformer's own cube); got {input_format!r}")
        cfg = RadarConfig.from_dict(manifest["config"])
        grid = build_grid_from_manifest(manifest)
        in_channels, n_range_in, n_doppler_in = train_module._input_dims(cfg, input_format)
        model = CFARHead(in_channels, n_range_in, n_doppler_in,
                         grid.n_range, grid.n_azimuth, cfg=cfg, grid=grid)
        dev = device if device is not None else train_module._default_device()
        return model.to(dev)

    build_model.__doc__ = (original.__doc__ or "") + \
        f"\n\nPatched by e2e.ml.cfar_head.register(): also accepts {MODEL_NAME!r}."
    build_model.__wrapped__ = original
    train_module.build_model = build_model

    # EDIT 3 IS NOT DONE BY THIS SHIM, AND THAT HAS TO FAIL LOUDLY.
    # `train.train`'s loop calls `detection_loss` unconditionally (train.py:640-642); it
    # does not consult `model.loss`. Registering the architecture is therefore enough to
    # make `python -m e2e.ml.train --model cfarhead` RUN TO COMPLETION while optimising
    # the wrong objective on a map that is constant zero outside <= K cells -- the exact
    # failure the module docstring's "TRAINING" section describes. A silent wrong answer
    # is worse than a missing feature (review finding, 2026-09-22), so training this
    # model is refused until a human makes the one-line change.
    original_train = train_module.train

    def train(manifest_path, model_name: str, *args, **kwargs):
        if model_name == MODEL_NAME:
            raise NotImplementedError(
                f"{MODEL_NAME} cannot be trained by train.train() as it stands: the loop "
                "at train.py:640-642 calls detection_loss unconditionally and never "
                "consults model.loss(), so it would optimise a focal BCE over a map that "
                "is zero everywhere outside the candidate cells. Apply edit 3 from "
                "e2e/ml/cfar_head.py's module docstring (dispatch to model.loss when the "
                "model defines one) and delete this guard, or train with your own loop "
                "calling CFARHead.loss(output, y). See also: train with --amp off.")
        return original_train(manifest_path, model_name, *args, **kwargs)

    train.__wrapped__ = original_train
    train_module.train = train


# --------------------------------------------------------------------------------
# Controls (§5 item 4) -- runnable, not run
# --------------------------------------------------------------------------------
def _frames(manifest_path, split: str, limit: Optional[int], device):
    """`(cfg, grid, frames, [target lists])` for a split.

    `frames` is a callable returning a FRESH generator of per-frame `[A, R, D]` rad
    tensors on `device` -- one frame at a time, because a full split of these cubes is
    gigabytes (172 x 64 x 512 x 64 float32 ~ 1.4 GB on b1_bench_v3). Uses
    `RadarFrameDataset(input_format="rad")`, so the tensors are BIT-IDENTICAL to the ones
    training and `compare_detectors` see -- no second derivation.
    """
    from e2e.ml.dataset import RadarFrameDataset
    from e2e.radar_config import RadarConfig

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    cfg = RadarConfig.from_dict(manifest["config"])
    grid = build_grid_from_manifest(manifest)
    ds = RadarFrameDataset(manifest_path, split=split, input_format="rad")
    n = len(ds) if limit is None else min(int(limit), len(ds))
    tgts = [ds.targets(i) for i in range(n)]

    def frames():
        for i in range(n):
            yield ds[i][0].to(device)

    return cfg, grid, frames, tgts


def _score(pred_maps, target_lists, grid, *, decode_threshold: float,
           target_recall: float, max_range_m: Optional[float]) -> Dict:
    from e2e.ml.compare_detectors import _score_arm

    return _score_arm(pred_maps, target_lists, grid, n_frames=len(pred_maps),
                      decode_threshold=decode_threshold, target_recall=target_recall,
                      max_range_m=max_range_m)


def _map_from_candidates(rows, cols, scores, grid, device) -> Tensor:
    out = torch.zeros((3, grid.n_range, grid.n_azimuth), dtype=torch.float32, device=device)
    if rows.numel():
        out[0, rows, cols] = torch.as_tensor(scores, dtype=torch.float32, device=device)
    return out


def controls(manifest_path, *, split: str = "test", device="cpu",
             checkpoint: Optional[str] = None, limit: Optional[int] = None,
             seed: int = 0, decode_threshold: Optional[float] = None,
             target_recall: Optional[float] = None, max_range_m: Optional[float] = 40.0,
             cfar_floor: float = DEFAULT_CFAR_FLOOR,
             max_candidates: int = DEFAULT_MAX_CANDIDATES,
             arms: Sequence[str] = ("cfar", "head", "cfar_random", "head_random_cand",
                                    "fixed_threshold")) -> Dict:
    """Every §5 control, on one split, scored by `compare_detectors._score_arm`.

    `checkpoint` loads trained head weights; without it the head is UNTRAINED and its row
    is meaningless except as a smoke test (say so in any artifact). Defaults reproduce the
    `beat_cfar` protocol (`DECODE_THRESHOLD=0.01`, recall 0.5, 40 m crop) so a row here is
    comparable to a row there.
    """
    # IMPORTED, not re-typed: "the same protocol as beat_cfar" is a claim that two
    # literals stay equal, and this repo has been bitten by exactly that (review finding,
    # 2026-09-22). `max_range_m=40.0` is beat_cfar's own crop, which has no shared
    # constant to import -- `beat_cfar.MAX_RANGE_M` is the authority if that changes.
    from e2e.ml.compare_detectors import DEFAULT_DECODE_THRESHOLD, DEFAULT_TARGET_RECALL
    if decode_threshold is None:
        decode_threshold = DEFAULT_DECODE_THRESHOLD
    if target_recall is None:
        target_recall = DEFAULT_TARGET_RECALL

    dev = torch.device(device)
    cfg, grid, frames, tgts = _frames(manifest_path, split, limit, dev)

    # One generator PER random arm, each seeded from `seed`, so an arm's numbers do not
    # depend on which other arms were selected with --arms (a shared generator made
    # `head_random_cand` move when `cfar_random` was dropped -- caught in the smoke run).
    # Fixed offsets, not `hash()`: Python randomises string hashing per process.
    _ARM_OFFSET = {"cfar_random": 1, "head_random_cand": 2}

    def gen(tag: str) -> torch.Generator:
        return torch.Generator(device="cpu").manual_seed(int(seed) * 1000 + _ARM_OFFSET[tag])

    from e2e.ml import train as train_mod
    in_ch, n_r_in, n_d_in = train_mod._input_dims(cfg, "rad")
    # Seeded even when untrained: an UNTRAINED head is a smoke row, but it must at least
    # be the SAME smoke row twice.
    torch.manual_seed(int(seed))
    model = CFARHead(in_ch, n_r_in, n_d_in, grid.n_range, grid.n_azimuth,
                     cfg=cfg, grid=grid, cfar_floor=cfar_floor,
                     max_candidates=max_candidates).to(dev)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=dev)
        model.load_state_dict(ckpt["model_state"])
    model.eval()

    results: Dict[str, Dict] = {}
    kw = dict(decode_threshold=decode_threshold, target_recall=target_recall,
              max_range_m=max_range_m)

    def _objectness_arm(fn):
        maps = []
        for x in frames():
            m = torch.zeros((3, grid.n_range, grid.n_azimuth), device=dev)
            m[0] = fn(x)
            maps.append(m.cpu())
        return _score(maps, tgts, grid, **kw)

    # --- CFAR alone (stage 1 at its own operating point, no rescoring) -------------
    if "cfar" in arms:
        results["cfar"] = _objectness_arm(lambda x: cfar_from_rad(x, cfg, grid))

    # --- the head itself ------------------------------------------------------------
    if "head" in arms:
        with torch.no_grad():
            maps = [model(x[None])["detection"][0].cpu() for x in frames()]
        results["head"] = _score(maps, tgts, grid, **kw)
        results["head"]["trained"] = bool(checkpoint)

    # --- CFAR candidates + RANDOM rescoring ------------------------------------------
    # Isolates the head: same candidate set, same K, same floor, scores replaced by noise.
    if "cfar_random" in arms:
        g_cr = gen("cfar_random")
        maps = []
        for x in frames():
            rows, cols, _s = model.candidates(x[None])[0]
            rnd = torch.rand(rows.numel(), generator=g_cr)
            maps.append(_map_from_candidates(rows.cpu(), cols.cpu(), rnd, grid, "cpu"))
        results["cfar_random"] = _score(maps, tgts, grid, **kw)

    # --- the head on RANDOM candidates -------------------------------------------------
    # Isolates stage 1: same head, same K, candidate cells drawn uniformly over the grid.
    if "head_random_cand" in arms:
        g_hr = gen("head_random_cand")

        def random_cells(obj, floor, k):
            n = int(min(k, obj.numel()))
            flat = torch.randperm(obj.numel(), generator=g_hr)[:n].to(obj.device)
            rows = torch.div(flat, obj.shape[1], rounding_mode="floor")
            cols = flat % obj.shape[1]
            return rows, cols, obj[rows, cols]
        model.candidate_fn = random_cells
        try:
            with torch.no_grad():
                maps = [model(x[None])["detection"][0].cpu() for x in frames()]
        finally:
            model.candidate_fn = None
        results["head_random_cand"] = _score(maps, tgts, grid, **kw)

    # --- fixed global threshold (no local adaptivity) ------------------------------------
    if "fixed_threshold" in arms:
        results["fixed_threshold"] = _objectness_arm(
            lambda x: global_threshold_objectness(x, cfg, grid))

    # --- the ceiling stage 2 cannot pass --------------------------------------------------
    results["candidate_set"] = model.candidate_recall(frames(), tgts) if tgts else {}
    return {"protocol": {"manifest": str(manifest_path), "split": split,
                         "decode_threshold": decode_threshold,
                         "target_recall": target_recall, "max_range_m": max_range_m,
                         "cfar_floor": cfar_floor, "max_candidates": max_candidates,
                         "seed": seed, "limit": limit, "device": str(device),
                         "checkpoint": checkpoint},
            "results": results}


def _format(res: Dict) -> str:
    rows = ["arm                    AP    AR@floor   FA/frame@recall   n_det",
            "-" * 64]
    for name, r in res["results"].items():
        if name == "candidate_set" or "AP" not in r:
            continue
        op = r.get("operating_point") or {}
        # `metrics.false_alarms_at_recall` reports `fp_per_frame`, and `reached=False`
        # (NaN) when the arm's PR curve never gets to the target recall -- print that as
        # "not reached" rather than a NaN a reader will misread as zero.
        fa = op.get("fp_per_frame") if op.get("reached") else None
        rows.append(f"{name:20s} {r['AP']:6.3f} {r['AR_at_decode_floor']:9.3f} "
                    f"{(f'{fa:15.1f}' if fa is not None else 'not reached'.rjust(15))} "
                    f"{r['n_detections']:7d}")
    cs = res["results"].get("candidate_set") or {}
    if cs:
        rows.append(f"\ncandidate-set recall ceiling: {cs['candidate_recall']:.3f} "
                    f"(post-NMS, i.e. achievable) | "
                    f"{cs['candidate_recall_pre_nms']:.3f} before decode NMS, an "
                    f"overcount -- see candidate_recall()\n"
                    f"  {cs['candidates_per_frame']:.1f} candidates/frame, "
                    f"{int(cs['n_targets'])} targets")
    return "\n".join(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m e2e.ml.cfar_head",
        description="CFAR + learned rescoring head: controls and diagnostics.")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("controls", help="run the §5 controls on a corpus split")
    c.add_argument("--manifest", default="e2e/ml/datasets/b1_bench_v3/benchmark_v1_D2/manifest.json")
    c.add_argument("--split", default="test")
    c.add_argument("--device", default="cpu", help="cpu (default) or cuda:N")
    c.add_argument("--checkpoint", default=None, help="trained cfarhead weights")
    c.add_argument("--limit", type=int, default=None, help="first N frames only")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--decode-threshold", type=float, default=None,
                   help="default: compare_detectors.DEFAULT_DECODE_THRESHOLD")
    c.add_argument("--recall", type=float, default=None,
                   help="default: compare_detectors.DEFAULT_TARGET_RECALL")
    c.add_argument("--max-range-m", type=float, default=40.0)
    c.add_argument("--cfar-floor", type=float, default=DEFAULT_CFAR_FLOOR)
    c.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    c.add_argument("--arms", default="cfar,head,cfar_random,head_random_cand,fixed_threshold")
    c.add_argument("--out", default=None, help="write the full result dict as JSON")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    res = controls(args.manifest, split=args.split, device=args.device,
                   checkpoint=args.checkpoint, limit=args.limit, seed=args.seed,
                   decode_threshold=args.decode_threshold, target_recall=args.recall,
                   max_range_m=args.max_range_m, cfar_floor=args.cfar_floor,
                   max_candidates=args.max_candidates,
                   arms=tuple(a.strip() for a in args.arms.split(",") if a.strip()))
    print(f"protocol: {res['protocol']}")
    print(_format(res))
    if not res["protocol"]["checkpoint"]:
        print("\nNOTE: no --checkpoint, so the `head` row is an UNTRAINED network. "
              "It is a smoke test, not a result.")
    if args.out:
        # pr_curve is large; keep it out of the printed artifact unless asked.
        Path(args.out).write_text(json.dumps(res, indent=2, default=float))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
