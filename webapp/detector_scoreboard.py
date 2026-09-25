"""
Detector SCOREBOARD: turns decoded detections into TP/FP/FN, and puts the offline
precision-recall curves stored in `e2e/ml/runs/beat_cfar.json` on screen next to a live
run's single operating point.

Why this exists (hostile-expert read of the rendered Thrust 5 screens, 2026-09-22):
nothing on the Detector objectness panel computes or shows AP, false alarms per frame,
or the match rule -- a viewer reads the cross/circle overlay as the detector's accuracy,
when it is really demonstrating `e2e.ml.metrics.match_detections`'s association rule at
whatever recall the threshold happens to land on. This module never re-implements that
matcher: every count here comes from `e2e.ml.metrics` (frozen -- two training runs are in
flight against it), imported lazily so this module carries no torch import at load time
(the webapp shell must import cleanly without torch -- see CLAUDE.md).

Per-frame structure this module consumes
-----------------------------------------
From `webapp.pipeline_runner.run_pipeline`'s `outputs` dict, exactly as
`figures_from_outputs` reads it (webapp/pipeline_runner.py, the `for key, title in
(("cfar_detection", ...), ("ml_detection", ...))` loop -- search for that string
rather than trusting a line number here, which has already drifted once):

* `outputs["cfar_detections"]` / `outputs["ml_detections"]` -- one entry per FRAME RUN
  (outer list length == frames run this call), each entry a list of this frame's decoded
  detections as `(range_m, sin_azimuth, score, surface_range_m)` tuples
  (`e2e.ml.labels.decode_detections`'s return convention). These are decoded ONCE, inside
  `CFARDetectorBlock.apply`/`NeuralDetectorBlock.apply` (`e2e/ml/blocks.py`), at the
  Detector block's own `threshold` param -- i.e. already thresholded to what is on
  screen. There is no re-decoding to do here, only matching.
* `outputs.get("gt_detections")` -- same tuple shape, decoded from the frame's stored
  label map (`e2e.ml.blocks.ground_truth_detections`), but present AT ALL only when the
  replayed frame carries stored labels in the first place (a corpus replay does; a
  live-traced frame does not). Because `e2e.simulation.Simulation` collects downstream
  outputs into a `defaultdict(list)` and appends a key only on the frames a block
  actually returned it (`e2e/simulation.py` ~359-367), a run where NO frame carries
  labels never gets a `"gt_detections"` key at all (`outputs.get(...)` returns `None`);
  a run replaying one corpus (today's only source of labels) carries them on every frame,
  so in practice `outputs["gt_detections"]` is either wholly absent or has exactly one
  entry per frame run, aligned by index with the detections list. `score_frames` below
  asserts that alignment rather than silently mis-pairing frame k's detections against
  frame j's ground truth if a future source ever violates it (e.g. a corpus with
  per-frame-optional labels).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import plotly.graph_objects as go

#: (range_m, sin_azimuth, score[, surface_range_m]) -- e2e.ml.labels.decode_detections'
#: return convention; ground truth decoded the same way carries the same shape.
Detection = Tuple[float, ...]

_REPO_ROOT = Path(__file__).resolve().parents[1]
#: The stored offline scoring this module reads by default (e2e.ml.compare_detectors'
#: output format) -- overridable per call, never edited by this module.
#: KA (owner 2026-09-24, ballot answer 4A: "Ka across the board"). The 77 GHz file is
#: still on disk and still readable -- every function here takes the path -- but the
#: DEFAULT is the Ka scoring, because the Thrust 5 screens now replay the Ka corpus and a
#: scoreboard reading the other band's numbers beside them would be the worst kind of
#: wrong: plausible. Both CFAR baselines are arms of this file (shipped default and the
#: val-tuned `cfar_first`), which is what F95's independent pass requires any Ka screen
#: to print.
DEFAULT_BEAT_CFAR_JSON = _REPO_ROOT / "e2e" / "ml" / "runs" / "beat_cfar_ka.json"
#: Bootstrap AP-delta-vs-CFAR confidence intervals (e2e.ml's paired-bootstrap CI tool),
#: keyed by arm name -- optional: a checkpoint scored in beat_cfar.json need not have a
#: CI entry here yet (F85 addendum). Overridable per call; never edited by this module.
DEFAULT_RADDETNET_CI_JSON = _REPO_ROOT / "e2e" / "ml" / "runs" / "raddetnet_ci_ka.json"
#: A SEPARATE offline-scored run against an out-of-distribution corpus (b1_bench_v2,
#: an earlier generator/impairment model than beat_cfar.json's b1_bench_v3) -- F86
#: (notes/ESTABLISHED_FACTS.md, measured 2026-09-22). Optional: an arm not scored here
#: simply gets no OOD row (`_ood_rows_for_arm`). Overridable per call; never edited.
#: NONE AT KA, deliberately. `gen_s43_v2_test.json` is a 77 GHz scoring of a 77 GHz
#: checkpoint against a 77 GHz corpus; printing it under a Ka arm would attribute another
#: band's out-of-distribution result to a network that has never been scored that way.
#: The Ka pair that DOES exist (D2 in-distribution, D4 never-trained-on) is
#: `beat_cfar_ka.json` / `beat_cfar_ka_d4.json`, and D4 is "a different tier of the same
#: generator family" (F95's independent pass), which is not the OOD claim this row makes.
#: So the row is absent until a Ka OOD scoring exists -- `_ood_rows_for_arm` already
#: returns [] for a missing file, which is the behaviour, not a failure.
DEFAULT_OOD_JSON = None
#: A THIRD, separately-scored corpus (D4/b1_bench_v4) no checkpoint here trained on --
#: F87 (notes/ESTABLISHED_FACTS.md, measured 2026-09-22). Optional, same convention as
#: `DEFAULT_OOD_JSON`: an arm not scored here simply gets no 3rd-corpus row
#: (`_third_corpus_rows_for_arm`). Overridable per call; never edited by this module.
#: Same reasoning as `DEFAULT_OOD_JSON`: 77 GHz scoring, no Ka counterpart, no row.
DEFAULT_THIRD_CORPUS_JSON = None

#: Hard precision ceiling for ANY detector that fires on every real object, because
#: ground truth OMITS real objects: ~3.25 real strongly-scattering objects per frame
#: inside 40 m at median 27.1 dB SNR are unlabelled (notes/ESTABLISHED_FACTS.md F83,
#: "still-open defects" #3, measured 2026-09-21). beat_cfar.json itself is scored with
#: use_ignore_regions=false (its own root key) -- i.e. against these incomplete labels
#: -- so every "false positive"/"false alarm" count this module shows is an UPPER
#: BOUND on real false alarms, not a true count. A stored claim, not a current
#: re-measurement -- re-verify against the ledger before reusing this number for
#: anything beyond the caption below (CLAUDE.md's provenance rule).
PRECISION_CEILING_F83 = 0.640

#: Number of elements along one axis of the receive array (CLAUDE.md: "the array is
#: 32x32", N_RX=1024) -- used only by `hit_gate_scale_note` to translate the
#: sin-azimuth match tolerance into a physical beamwidth; not read from any config
#: file (this module has no other dependency on e2e.blocks/e2e.scenario).
_ARRAY_ELEMENTS_PER_AXIS = 32

#: The radar config preset every Thrust 5 live-chain preset's dechirp block replays
#: with (`webapp.demo_presets._T5_LIVE_CHAIN`) -- read by `hit_gate_scale_note` only
#: for its native range resolution, so that clause cannot drift from the actual
#: corpus config. NOT the Ka-band munich frames' 3 GHz-sweep, ~5 cm native
#: resolution Thrusts 1-4 show -- see that function's docstring for the mix-up this
#: avoids.
_T5_RADAR_PRESET_NAME = "benchmark_v1"

#: In-distribution seed-42-vs-seed-43 AP spread for RADDetNet on benchmark_v1_D2/v3,
#: one architecture, two training seeds, identical protocol (notes/ESTABLISHED_FACTS.md
#: F86, measured 2026-09-22) -- the scale a single checkpoint's scene-bootstrap CI
#: cannot speak to, since bootstrapping resamples SCENES of one already-trained
#: network, never a second training run.
SEED_TO_SEED_AP_SPREAD_F86 = 0.040


def _is_raddetnet_arm(arm_name) -> bool:
    """Whether `arm_name` is a RADDetNet arm, i.e. the architecture
    `SEED_TO_SEED_AP_SPREAD_F86` was measured on. Substring, not equality: the arm reads
    "raddetnet" in `beat_cfar_ka.json` and has read "raddetnet_s43"/"raddetnet (joint)"
    in other scorings, and a spread caption that silently stopped firing would be a
    quieter defect than one that fires too widely."""
    return "raddetnet" in str(arm_name or "").lower()

#: The null arm's stored name reads as "random INSIDE the ground-truth boxes" (i.e.
#: the detector is handed the answer) -- it is actually uniform-random scores inside
#: the BOUNDING BOX of every TRAIN-split label (never the eval labels), fit once and
#: applied blind to the RF (e2e.ml.compare_detectors.score_null's docstring,
#: e2e/ml/compare_detectors.py:189-192). Display-only remap; the stored JSON name
#: (and the JSON itself) is never edited.
#: The arm whose legend entry may never be the one that gets clipped -- it is the
#: panel's chance floor -- and which therefore never dims either. Split out of
#: `_ARM_DISPLAY_NAMES` (seat's read of the 2026-09-24 renders, item 1e): that dict is
#: now a display-name map with more than one entry in it, and using its membership as
#: the null-arm test would have silently promoted the val-tuned CFAR baseline to
#: "never dims" the moment it was given a short name.
_NULL_ARM_NAMES = {"null (random-in-GT-box)"}

_ARM_DISPLAY_NAMES = {
    # Shortened (Change, 2026-09-23 coordinator re-check): the original full sentence
    # was the single longest legend entry, and a right-hand vertical legend sized to
    # its longest entry squeezed the PR plot itself to a ~80 px sliver.
    # Shortened AGAIN (measured on the rendered page, 2026-09-24): in the two-column
    # legend a 52-character entry overran its half of the strip and was clipped at the
    # column edge -- and this is the one entry that must never be the one that gets
    # cut, because it is the panel's chance floor. The full sentence is in Details.
    "null (random-in-GT-box)": "null (chance floor)",
    # SEAT'S READ OF THE 2026-09-24 RENDERS, item 1e: at 48 characters
    # "classical CFAR (val-tuned, cfar_first) (AP=0.326)" ran off the right edge of its
    # half of the two-column legend on EVERY Thrust 5 screen -- measured on the rendered
    # PR panels. `cfar_first` is the reduction this baseline was tuned with and it is
    # named in the scoreboard table and in Details; the legend's job is to tell two CFAR
    # curves apart, which "CFAR val-tuned" does in 14. The AP is not typed here -- the
    # trace name appends it from the scoreboard file (`ap` above).
    "classical CFAR (val-tuned, cfar_first)": "CFAR val-tuned",
}

#: The null arm's REAL definition, which its legend entry is now too short to carry.
#: Stated in `stored_pr_figure`'s Details instead -- the stored JSON name reads as
#: "random INSIDE the ground-truth boxes" (i.e. the detector is handed the answer); it
#: is actually uniform-random scores inside the BOUNDING BOX of every TRAIN-split label
#: (never the eval labels), fit once and applied blind to the RF.
NULL_ARM_DEFINITION = (
    "null: random cells in train-label box (chance floor) -- uniform-random scores "
    "inside the bounding box of every TRAIN-split label, never the eval labels, fit "
    "once and applied blind to the RF"
)

def _display_arm_name(name: str) -> str:
    """`name` as it should read on screen -- see `_ARM_DISPLAY_NAMES`. Passthrough for
    every arm this module has no reason to rename."""
    return _ARM_DISPLAY_NAMES.get(name, name)


def match_rule_text(criterion=None) -> str:
    """The match rule in words, with its numbers read off `e2e.ml.metrics.MatchCriterion`
    -- never typed in here, so this text cannot drift from what `score_frames` actually
    enforces (see CLAUDE.md's "claims carry provenance" -- a hardcoded copy of R/S is
    exactly the kind of claim that silently goes stale)."""
    from e2e.ml.metrics import MatchCriterion

    c = criterion or MatchCriterion()
    return (
        f"a detection counts as a hit within ±{c.max_range_err_m:g} m (surface) "
        f"range and ±{c.max_sin_az_err:g} in sin(azimuth) of a ground-truth target; "
        "one detection claims at most one target, nearest first "
        "(e2e.ml.metrics.match_detections)"
    )


def hit_gate_scale_note(criterion=None) -> str:
    """Two clauses translating the match tolerance (see `match_rule_text`) into
    physical units a viewer can sanity-check, computed at call time so neither
    number is a hardcoded copy of something that can drift (CLAUDE.md's provenance
    rule): the sin-azimuth tolerance against the array's own element count (one
    32-element beamwidth, 2/32), and the range tolerance against the native range
    resolution of the radar config every Thrust 5 preset's dechirp block replays
    with (`benchmark_v1`, `e2e.radar_config.PRESETS` -- lazily imported, torch-free,
    same convention as the `e2e.ml.metrics` import above).

    RE-VERIFIED 2026-09-23 (wave 7, X7): the hostile-expert read's own arithmetic
    for this clause ("40 native range bins at 5 cm") was the Ka-band MUNICH frames'
    3 GHz-sweep resolution (Thrusts 1-4's screen, a different corpus from this
    one) -- `benchmark_v1`'s own bandwidth (749.5 MHz) gives ~0.2 m/native bin, not
    0.05 m/bin, so this function reads the real preset rather than repeating that
    figure (CLAUDE.md's "claims carry provenance": a stored number from one screen
    is not evidence for a different screen).
    """
    from e2e.ml.metrics import MatchCriterion
    from e2e.radar_config import PRESETS as _radar_presets

    c = criterion or MatchCriterion()
    beamwidth = 2.0 / _ARRAY_ELEMENTS_PER_AXIS
    native_res_m = _radar_presets[_T5_RADAR_PRESET_NAME].range_resolution_m
    n_bins = c.max_range_err_m / native_res_m
    return (
        f"scale: the {c.max_sin_az_err:g} sin(azimuth) hit gate is about one "
        f"{_ARRAY_ELEMENTS_PER_AXIS}-element array beamwidth "
        f"(2/{_ARRAY_ELEMENTS_PER_AXIS} = {beamwidth:.4g}); the {c.max_range_err_m:g} m "
        f"range hit gate is about {n_bins:.0f} native range bins at this corpus' "
        f"own {native_res_m * 100:.0f} cm resolution ({_T5_RADAR_PRESET_NAME})"
    )


def score_frames(
    detections_per_frame: Sequence[Sequence[Detection]],
    gt_per_frame: Optional[Sequence[Sequence[Detection]]] = None,
    *,
    threshold: Optional[float] = None,
    criterion=None,
) -> Dict[str, Any]:
    """Per-frame TP/FP/FN plus cumulative counts, FA/frame and hit rate over the run.

    Reuses `e2e.ml.metrics.match_detections` (the SAME matcher `e2e.ml.metrics.
    evaluate_dataset`/`compare_detectors` score with) frame by frame -- this function
    never re-implements matching, only pools its per-frame result.

    Parameters
    ----------
    detections_per_frame
        `outputs["cfar_detections"]` or `outputs["ml_detections"]` (see the module
        docstring) -- already decoded/thresholded, one list of tuples per frame run.
    gt_per_frame
        `outputs.get("gt_detections")`. `None` means no frame in this run carried
        ground truth at all (e.g. a live-traced run) -- every frame is then
        `"scored": False` rather than invented. A per-frame empty list `[]` (labels
        present, no target that frame) IS scored: every detection on it is a false
        positive, matching `match_detections`'s own convention for an empty target list.
    threshold
        Not used for decoding (both inputs are already decoded) -- carried through into
        the returned dict purely so a caption can name the operating point without a
        second source of truth for it.

    Returns
    -------
    dict with ``threshold`` (echoed), ``n_frames``, ``n_frames_scored`` (frames with
    known ground truth), ``frames`` (one ``{"tp","fp","fn","n_det","n_gt","scored"}``
    per frame), ``cumulative`` (``{"tp","fp","fn"}`` summed over scored frames only),
    ``fa_per_frame`` (cumulative fp / n_frames_scored, NaN if nothing was scored) and
    ``hit_rate`` (cumulative tp / (tp+fn), NaN if nothing was scored).
    """
    from e2e.ml.metrics import MatchCriterion, match_detections

    if criterion is None:
        criterion = MatchCriterion()
    n_frames = len(detections_per_frame)
    if gt_per_frame is not None and len(gt_per_frame) != n_frames:
        raise ValueError(
            f"gt_per_frame has {len(gt_per_frame)} entries but detections_per_frame has "
            f"{n_frames} -- they must be frame-aligned (see the module docstring's "
            "per-corpus alignment assumption); refusing to silently mis-pair them."
        )

    frame_scores: List[Dict[str, Any]] = []
    cum_tp = cum_fp = cum_fn = 0
    n_scored = 0
    for i, dets in enumerate(detections_per_frame):
        gt = None if gt_per_frame is None else gt_per_frame[i]
        if gt is None:
            frame_scores.append({"tp": 0, "fp": 0, "fn": 0, "n_det": len(dets),
                                 "n_gt": 0, "scored": False})
            continue
        matches, unmatched_det, unmatched_gt = match_detections(dets, gt, criterion)
        tp, fp, fn = len(matches), len(unmatched_det), len(unmatched_gt)
        frame_scores.append({"tp": tp, "fp": fp, "fn": fn, "n_det": len(dets),
                             "n_gt": len(gt), "scored": True})
        cum_tp += tp
        cum_fp += fp
        cum_fn += fn
        n_scored += 1

    fa_per_frame = (cum_fp / n_scored) if n_scored else float("nan")
    denom = cum_tp + cum_fn
    hit_rate = (cum_tp / denom) if denom else float("nan")

    return {
        "threshold": threshold,
        "n_frames": n_frames,
        "n_frames_scored": n_scored,
        "frames": frame_scores,
        "cumulative": {"tp": cum_tp, "fp": cum_fp, "fn": cum_fn},
        "fa_per_frame": fa_per_frame,
        "hit_rate": hit_rate,
    }


#: Table geometry (px), FIXED under the panel-meta contract (2026-09-24 layout
#: redesign -- see `webapp.pipeline_runner`'s "PANEL GEOMETRY AND THE PANEL-META
#: CONTRACT" section): every scoreboard figure now gets the SAME
#: `FIGURE_HEIGHT[PANEL_ROW_TABLE]` (312 px), independent of row count, wrapped-line
#: count or arm-name length. `scoreboard_figure` enforces this by capping the table
#: at <=8 VISIBLE rows (the two highest-value offline rows plus the 6 live-run rows;
#: everything else moves into the panel's Details disclosure) and asserting no row
#: needs to wrap.
#:
#: RETIRED (2026-09-24): the old machinery that grew the table's own `height` from
#: its actual row/line count and the figure title/subtitle/annotation's own wrapped
#: line counts at call time (a variable `_TABLE_ROW_HEIGHT`, `_TABLE_MARGIN_T`,
#: `_TABLE_SUBLINE_LINE_PX`, `_TABLE_ANNOTATION_LINE_PX`, `_TABLE_RENDER_SAFETY_PX`
#: fudge factor) is gone with the figure title/subtitle/annotation it was sized
#: around -- that text now lives in the panel's HTML title/caption/Details
#: (`pipeline_runner.set_panel`), off the figure entirely, so the figure's own
#: height has nothing left to grow with. This is a genuine simplification, not just
#: a rename: the empirically-fudged render-safety margin the old comment here
#: recorded (Plotly's row-position accumulation drifting ~12 px from this module's
#: own arithmetic) is no longer load-bearing, because the table no longer tries to
#: fit its height exactly to computed content -- it just asserts the content fits
#: inside a height that was never derived from that content in the first place.
_TABLE_HEADER_HEIGHT = 40
#: Row height cap (px, layout spec: "no row taller than 40 px"). The ACTUAL row
#: height used per call is computed in `scoreboard_figure` from the real row count,
#: floor-divided so `_TABLE_HEADER_HEIGHT + n_rows * row_height` never exceeds
#: `FIGURE_HEIGHT[PANEL_ROW_TABLE]` -- at the table's max, 8 rows, this divides out
#: to exactly 34 px/row with zero pixels left over (8*34 + 40 == 312).
_TABLE_MAX_ROW_HEIGHT = 40
#: Column widths (px, relative -- Plotly normalises `columnwidth`): 55/45,
#: label-heavy (Change, 2026-09-24 layout redesign: was 50/50 -- the table now
#: carries only its 8 highest-value rows, so the value side can give a little width
#: back to the label side, which still carries full-sentence rows like "FA/frame at
#: recall 0.5, 172 frames").
_TABLE_COL_WIDTHS = [55, 45]
#: Character budget both columns are checked against -- ONLY to feed the no-wrap
#: assertion in `scoreboard_figure` now (Change, 2026-09-24): under the fixed-height
#: contract above, no table row may wrap AT ALL (Plotly's Table `cells.height` is one
#: scalar for the whole table, and every row is held to a fixed <=40 px), so
#: `_wrap_text` below is no longer used to actually insert a "<br>" into a cell --
#: only to detect that one would be needed, so the figure-building code can fail
#: loudly (an `assert`) instead of silently inflating every row past its cap.
#: Calibrated (2026-09-23, still valid at the new 55/45 split and 17 px font -- both
#: give a row slightly MORE room per character than the value this was measured
#: against, so 34 stays conservative) against this module's own longest rows at
#: `_TABLE_COL_WIDTHS`' width.
_TABLE_COL_CHARS = 34
#: Every table row must now fit in exactly one line (asserted, not just hoped for --
#: `scoreboard_figure`'s no-wrap assertion). Retained under its old name in case a
#: caller still checks it; the old cap of 2 (a table where one row could still wrap
#: to a second line, inflating every OTHER row's height to match) is gone with the
#: variable-height table it protected.
_TABLE_CELL_MAX_LINES = 1


def _wrap_text(text: str, max_chars: int = 70) -> str:
    """Greedy word-wrap `text` into `'<br>'`-joined lines no wider than `max_chars`
    characters, so a long sentence doesn't get silently clipped by its container's
    fixed pixel width -- the un-wrapped match-rule annotation ran off the right edge
    of a two-card (~600 px) panel (rehearsal, 2026-09-23: "...+-0.06 in sin(azimu").
    Word boundaries only (never mid-word), so a URL-like token can still exceed
    `max_chars` on its own line -- none of this module's callers pass one.
    """
    words = text.split()
    lines: List[str] = []
    cur = ""
    for w in words:
        candidate = f"{cur} {w}".strip()
        if cur and len(candidate) > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur = candidate
    if cur:
        lines.append(cur)
    return "<br>".join(lines)


def _load_recall_target_and_n_frames(beat_cfar_json_path=DEFAULT_BEAT_CFAR_JSON):
    """`(target_recall, n_frames)` every arm's threshold in beat_cfar.json was picked
    at -- read from the JSON so the scoreboard's threshold subline (see
    `scoreboard_figure`) cannot state a recall target or split size that has drifted
    from what the file actually stores. `(None, None)` if the JSON is missing or
    carries neither field -- the caller then falls back to a shorter, context-free
    subline rather than inventing either number."""
    try:
        data = _load_beat_cfar(beat_cfar_json_path)
    except (FileNotFoundError, ValueError):
        return None, None
    target_recall = data.get("target_recall")
    n_frames = next(
        (a["operating_point"]["n_frames"] for a in data.get("arms", [])
         if a.get("operating_point") and a["operating_point"].get("n_frames")),
        None,
    )
    return target_recall, n_frames


def scoring_max_range_m(beat_cfar_json_path=DEFAULT_BEAT_CFAR_JSON) -> Optional[float]:
    """The range crop every arm in `beat_cfar_json_path` was SCORED under (each arm's
    own stored `max_range_m`, i.e. `e2e.ml.compare_detectors`' `--max-range-m` CLI
    flag) -- read from the file so a "labels & scoring stop at X m" annotation on the
    live detector panel cannot state a crop the file does not actually carry (CLAUDE.md's
    provenance rule). This is NOT the label grid's own physical extent (~102 m on
    benchmark_v1_D2 -- see `LabelGrid.max_range_m`); it is the separate, smaller crop
    applied at scoring time because labels themselves are sparse past it.

    `None` if the file is missing/malformed or no arm carries the field -- the caller
    then skips the annotation rather than inventing a crop.
    """
    try:
        data = _load_beat_cfar(beat_cfar_json_path)
    except (FileNotFoundError, ValueError):
        return None
    for arm in data.get("arms", []):
        if arm.get("max_range_m") is not None:
            return float(arm["max_range_m"])
    return None


def _load_json_arms(path) -> Optional[List[Dict[str, Any]]]:
    """`path`'s `"arms"` list (`e2e.ml.compare_detectors`' output format), or `None`
    if `path` is missing/unreadable/malformed -- never raises, so a caller can treat a
    not-yet-generated offline-scoring artifact as "nothing to show" rather than a hard
    failure of the live run."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text()).get("arms")
    except (OSError, ValueError):
        return None


def _match_corpus_arm(bc_arm: Dict[str, Any], corpus_arms: List[Dict[str, Any]]
                      ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """`(arm, sibling)` -- `bc_arm`'s (one beat_cfar.json arm's) counterpart in
    `corpus_arms` (a separately-scored corpus' `"arms"` list), plus its two-seed
    sibling if one exists in that SAME file. Shared by every "score this beat_cfar.json
    arm on a separate corpus" row-builder (`_ood_rows_for_arm`, `_third_corpus_rows_for_arm`)
    so the matching convention -- checkpoint PARENT DIRECTORY for an ML arm (same
    convention `arm_name_for_detector` uses), NAME for CFAR (no checkpoint), and a
    "_s<seed>" name suffix for sibling detection -- lives in exactly one place.

    `(None, None)` if `bc_arm` has no counterpart in `corpus_arms` at all, or the
    counterpart carries no AP (nothing to report). `sibling` is `None` whenever the
    matched arm's name carries no "_s<seed>" suffix (CFAR, a joint checkpoint with no
    per-seed family, ...) or no such sibling is present in `corpus_arms`.
    """
    ckpt = bc_arm.get("checkpoint")
    if ckpt:
        ckpt_dir = Path(ckpt).parent.name
        arm = next((a for a in corpus_arms if a.get("checkpoint") and
                   Path(a["checkpoint"]).parent.name == ckpt_dir), None)
    else:
        arm = next((a for a in corpus_arms if a.get("name") == bc_arm.get("name")), None)
    if arm is None or arm.get("AP") is None:
        return None, None

    sibling = None
    base, sep, _seed = arm.get("name", "").rpartition("_s")
    if base and sep:
        sibling = next((a for a in corpus_arms if a is not arm and
                        a.get("name", "").startswith(base + "_s") and
                        a.get("AP") is not None), None)
    return arm, sibling


def _ood_rows_for_arm(bc_arm: Dict[str, Any], ood_json_path=DEFAULT_OOD_JSON
                      ) -> List[Tuple[str, str]]:
    """`[(label, value)]` -- one or two SHORT rows (never one long sentence, see the
    comment inline below) stating `bc_arm`'s (one beat_cfar.json arm's) performance on
    a SEPARATE, out-of-distribution corpus -- `[]` if `ood_json_path` is
    absent/malformed or carries no arm matching `bc_arm` (never invented for an arm
    that file never scored).

    Matched by CHECKPOINT PARENT DIRECTORY for an ML arm (the same convention
    `arm_name_for_detector` uses against beat_cfar.json), or by NAME for CFAR (which
    carries no checkpoint).

    For a two-seed RADDetNet arm specifically (a sibling "_s<other seed>" arm of the
    same base name is in the SAME file), also finds the CFAR arm in that file and adds
    an AP-vs-CFAR row (both seeds and CFAR's own AP, one line -- see the row-budget
    comment inline below for why this is one merged row rather than the AP triple plus
    a separate verdict sentence) and, when every arm's operating point carries one, an
    "OOD unmatched/frame" row: on false alarms specifically the BEST seed can lose to
    CFAR out of distribution even though its AP wins (hostile-expert 4th read,
    2026-09-23 -- AP alone hid that disagreement). Both rows are computed from these
    files' own numbers at call time, never typed as a conclusion here.

    RETRACTED (hostile-expert read, 2026-09-23, item 8): the row used to name the
    corpus by its internal tag (e.g. "b1_bench_v2") and the two seeds by number
    (e.g. "s42/s43") -- ledger shorthand nobody outside this project can look up.
    Both display as generic "out-of-distribution corpus" / "2 seeds" now; the
    presenter's own card names the actual corpus and seeds where that matters.
    """
    if ood_json_path is None:
        return []
    ood_arms = _load_json_arms(ood_json_path)
    if not ood_arms:
        return []

    ood_arm, sibling = _match_corpus_arm(bc_arm, ood_arms)
    if ood_arm is None:
        return []

    ap = ood_arm["AP"]
    fa = (ood_arm.get("operating_point") or {}).get("fp_per_frame")
    cfar_arm = next((a for a in ood_arms if a.get("name") == "classical CFAR"), None)

    # SHORT rows (one per column each), not one long sentence crammed into the label
    # column -- a single 4-line row forced EVERY row in the table to that same height
    # (Plotly's Table `cells.height` is one scalar for the whole table, not per-row),
    # ballooning the whole scoreboard card past a screen (hostile-expert re-read,
    # 2026-09-23). Each piece here is short enough to stay one line at the current
    # column width (see `scoreboard_figure`'s wrap-budget comment). The AP triple and
    # CFAR's own AP are one merged row (not AP-row + separate verdict sentence, as a
    # first version had it) specifically to make room for the FA row below within the
    # <=800 px card budget -- see `scoreboard_figure`'s row-count accounting comment;
    # the qualitative "seeds straddle CFAR" reading is still recoverable from the three
    # numbers here, just no longer spelled out in words.
    if sibling is not None and cfar_arm is not None and cfar_arm.get("AP") is not None:
        cfar_ap, sib_ap = cfar_arm["AP"], sibling["AP"]
        rows = [("OOD AP, out-of-distribution corpus",
                 f"{ap:.3f}/{sib_ap:.3f} (2 seeds) CFAR {cfar_ap:.3f}")]
        # FA/frame specifically: AP alone can hide that the BEST-AP seed still loses
        # to CFAR on false alarms out of distribution -- shown only when every arm's
        # operating point has the number (never invented for one that doesn't).
        cfar_fa = (cfar_arm.get("operating_point") or {}).get("fp_per_frame")
        sib_fa = (sibling.get("operating_point") or {}).get("fp_per_frame")
        if fa is not None and sib_fa is not None and cfar_fa is not None:
            rows.append(("OOD unmatched/frame",
                        f"{fa:.1f}/{sib_fa:.1f} (2 seeds) CFAR {cfar_fa:.1f}"))
        return rows
    value = f"{ap:.3f}" + (f", FA {fa:.1f}" if fa is not None else "")
    return [("OOD AP, out-of-distribution corpus", value)]


def _third_corpus_rows_for_arm(bc_arm: Dict[str, Any],
                               third_corpus_json_path=DEFAULT_THIRD_CORPUS_JSON
                               ) -> List[Tuple[str, str]]:
    """`[(label, value)]` -- ONE row stating `bc_arm`'s (one beat_cfar.json arm's)
    performance on a THIRD, separately-scored corpus no checkpoint here trained on --
    `[]` if `third_corpus_json_path` is absent/malformed or carries no arm matching
    `bc_arm` (never invented for an arm that file never scored). Matching is
    `_match_corpus_arm`'s -- the same convention `_ood_rows_for_arm` uses: checkpoint
    PARENT DIRECTORY for an ML arm (so e.g. the two-seed RADDetNet checkpoint and a
    joint checkpoint trained on both prior corpora each match their own row here, by
    directory, same as everywhere else in this module), NAME for CFAR (no checkpoint),
    sibling via a "_s<seed>" name suffix.

    For a two-seed arm, ONE row carries both seeds and CFAR's own AP (mirrors
    `_ood_rows_for_arm`'s merged AP row -- same reason: a second row would not fit the
    <=800 px card budget, see `scoreboard_figure`'s row-count accounting comment);
    otherwise the row carries just the matched arm's own AP. The two seeds display as
    "2 seeds" (hostile-expert read, 2026-09-23, item 8), never their internal numbers.
    """
    if third_corpus_json_path is None:
        return []
    corpus_arms = _load_json_arms(third_corpus_json_path)
    if not corpus_arms:
        return []

    arm, sibling = _match_corpus_arm(bc_arm, corpus_arms)
    if arm is None:
        return []

    try:
        manifest = json.loads(Path(third_corpus_json_path).read_text()).get("manifest", "")
    except (OSError, ValueError):
        manifest = ""
    # e.g. "e2e/ml/datasets/b1_bench_v4/benchmark_v1_D4/manifest.json" -> "D4" (the
    # SCENE TIER this corpus stresses -- what the label names, since "3rd corpus"
    # already says this is a different corpus from the offline test split above).
    tier = Path(manifest).parent.name.rsplit("_", 1)[-1] if manifest else "?"
    n_frames = (arm.get("operating_point") or {}).get("n_frames")
    frames_txt = f"{n_frames} fr" if n_frames is not None else "?"
    label = f"3rd corpus AP, {tier} {frames_txt}"

    ap = arm["AP"]
    cfar_arm = next((a for a in corpus_arms if a.get("name") == "classical CFAR"), None)
    if sibling is not None and cfar_arm is not None and cfar_arm.get("AP") is not None:
        value = (f"{ap:.3f}/{sibling['AP']:.3f} (2 seeds) "
                f"CFAR {cfar_arm['AP']:.3f}")
    else:
        value = f"{ap:.3f}"
    return [(label, value)]


def _offline_arm_rows(beat_cfar_arm_name: str, beat_cfar_json_path=DEFAULT_BEAT_CFAR_JSON,
                      raddetnet_ci_json_path=DEFAULT_RADDETNET_CI_JSON,
                      ood_json_path=DEFAULT_OOD_JSON,
                      third_corpus_json_path=DEFAULT_THIRD_CORPUS_JSON,
                      ) -> List[Tuple[str, str]]:
    """`(label, value)` rows for the offline scoring of ONE beat_cfar.json arm: an
    "FA/frame at recall ..." row FIRST, then "AP, offline test split" naming AP and
    the split together. Appends a bootstrap AP-delta-vs-CFAR row too, but ONLY
    when `raddetnet_ci_json_path` exists AND carries a `comparisons` entry for this
    exact arm -- never invented for an arm the CI file hasn't scored yet -- plus,
    right after it, a caveat row stating the seed-to-seed AP spread the CI's
    scene-bootstrap cannot see. Appends an out-of-distribution row
    (`_ood_rows_for_arm`) when a separate OOD-scored JSON covers this arm, and,
    finally, a third-corpus row (`_third_corpus_rows_for_arm`) when a separate
    third-corpus-scored JSON covers this arm.

    FA/frame FIRST (Change, wave 7 X2, 2026-09-23): a hostile-expert read of the
    KA-BAND screens found this arm's cross count on the live panel (fewer crosses
    AND fewer hits than CFAR on 5 frames) read as a loss before a viewer ever
    reached the one comparison this table can defend -- false alarms AT MATCHED
    recall (e.g. 2.99 vs 6.24/frame for raddetnet vs CFAR). `scoreboard_figure` now
    renders this whole offline block above the live block for the same reason; this
    function additionally puts the FA row ahead of the AP row within the block, so
    the very first row of the table is the matched-recall FA comparison.

    RETRACTED (hostile-expert read, 2026-09-23, item 8): this block used to also
    surface the rank-1 stripe statistic on the AP row ("AP, split, stripe vs GT",
    e.g. "stripe 0.617/0.312") -- a bare, unexplained pair of numbers that only
    means anything with the presenter's own narration (the `say` list already
    states it as "rank-1 energy fraction 0.89/0.76 against 0.31"). Dropped from
    this visitor-visible table; the presenter's card keeps the number.

    Returns `[]` if `beat_cfar_json_path` is missing/malformed or the arm is not one
    of its scored arms -- nothing invented for an arm this file never scored.
    """
    try:
        data = _load_beat_cfar(beat_cfar_json_path)
    except (FileNotFoundError, ValueError):
        return []
    arm = next((a for a in data.get("arms", []) if a.get("name") == beat_cfar_arm_name), None)
    if arm is None:
        return []

    op = arm.get("operating_point") or {}
    n_frames = op.get("n_frames")
    target_recall = op.get("target_recall", data.get("target_recall"))
    # Split across both columns (Change, 2026-09-23 coordinator re-check): the old
    # single-column "offline, {n}-frame test split (beat_cfar.json)" (46+ chars)
    # wrapped to 2 lines and inflated the WHOLE table to that row height (see
    # `_TABLE_COL_CHARS`).
    # Merged into ONE row with AP when there is one (Change, 4th hostile-expert read,
    # 2026-09-23: freed a row to hold the <=800 px budget once the connector row and
    # the OOD FA row -- both added this same pass -- needed the space; see
    # `scoreboard_figure`'s render-safety-margin comment for why that budget matters
    # beyond the naive row-count arithmetic). Kept as its own row, unmerged, only when
    # this arm has no AP to attach it to (never observed in practice, but the header
    # naming the split should not silently disappear if it ever happens).
    header_value = f"{n_frames}fr (beat_cfar.json)" if n_frames is not None else "(beat_cfar.json)"
    ap = arm.get("AP")
    rows: List[Tuple[str, str]] = []

    # The CFAR reference these two rows print inline, for a LEARNED detector's arm
    # only (never CFAR's own row, which would be "CFAR vs itself") -- read from this
    # SAME beat_cfar.json, never typed (wave 9, hostile-expert read, 2026-09-23,
    # item 2): before this, only the OOD/3rd-corpus rows below stated CFAR's number
    # beside a learned detector's; the two rows that actually carry the demo's
    # headline claim ("FA/frame at recall ...", "AP, offline test split") stated
    # none, so a viewer had to scroll to a different arm's table to compare.
    is_learned = beat_cfar_arm_name != "classical CFAR"
    cfar_arm = (next((a for a in data.get("arms", []) if a.get("name") == "classical CFAR"),
                     None)
               if is_learned else None)

    # FA/frame FIRST, ahead of AP (Change, wave 7 X2, 2026-09-23 -- see this
    # function's docstring): states the OFFLINE frame count on this row too
    # (Change, hostile-expert 4th read, 2026-09-23): this number sits directly
    # above the live "unmatched / frame, these N frames" row below, and the two
    # read as contradictory without both stating which sample size they're each
    # over (see the connector row `scoreboard_figure` inserts between the two
    # blocks for the same reason).
    fa_pf = op.get("fp_per_frame")
    if fa_pf is not None:
        label = "FA/frame"
        if target_recall is not None:
            label += f" at recall {target_recall:g}"
        if n_frames is not None:
            label += f", {n_frames} frames"
        fa_value = f"{fa_pf:.2f}"
        cfar_fa = ((cfar_arm.get("operating_point") or {}).get("fp_per_frame")
                  if cfar_arm is not None else None)
        if cfar_fa is not None:
            fa_value += f" (CFAR {cfar_fa:.2f})"
        rows.append((label, fa_value))

    if ap is not None:
        cfar_ap = cfar_arm.get("AP") if cfar_arm is not None else None
        if cfar_ap is not None:
            # Shorter form (drops the "{n}fr (beat_cfar.json)" suffix): the subline
            # above this table already states both the split size and the file
            # (`scoreboard_figure`'s "... test split (beat_cfar.json)" sentence), so
            # this row spends its char budget on the CFAR number instead of a second
            # copy of them.
            ap_split_value = f"{ap:.3f} (CFAR {cfar_ap:.3f})"
        elif n_frames is not None:
            ap_split_value = f"{ap:.3f}, {n_frames}fr (beat_cfar.json)"
        else:
            ap_split_value = f"{ap:.3f} (beat_cfar.json)"
        rows.append(("AP, offline test split", ap_split_value))
    else:
        rows.append(("offline test split", header_value))

    ci_path = Path(raddetnet_ci_json_path)
    if ci_path.is_file():
        try:
            ci_data = json.loads(ci_path.read_text())
        except (OSError, ValueError):
            ci_data = {}
        comp = next((c for c in ci_data.get("comparisons", [])
                    if c.get("arm") == beat_cfar_arm_name), None)
        if comp and all(k in comp for k in ("delta_AP", "ci_low", "ci_high")):
            rows.append(("delta AP vs CFAR, 95% CI",
                        f"{comp['delta_AP']:+.3f} "
                        f"[{comp['ci_low']:+.3f}, {comp['ci_high']:+.3f}]"))
            # The CI band only speaks to SCENE-bootstrap variance of one already-
            # trained checkpoint; it hides the variance that has actually been
            # measured to matter -- a second training seed moves AP by 0.040.
            # Stated as an explicit comparison against THIS arm's own CI half-width,
            # computed at call time (never a hardcoded "comparable to" -- hostile-
            # expert 4th read, 2026-09-23: the seed spread (0.040) actually EXCEEDS
            # the half-width (0.032) here, a stronger and more checkable claim than
            # "comparable"). A SHORT two-column row (not one long sentence crammed
            # into the label column, which forced every row in the table to that same
            # height -- hostile-expert re-read, 2026-09-23) attached right after the
            # CI row rather than left in the caption below the table, where a viewer
            # skimming the CI number alone would miss it. No "(F86)" ledger tag on
            # screen (hostile-expert read, 2026-09-23, item 8): a visitor cannot look
            # that up; the claim stands on the two numbers alone.
            # GATED TO THE RADDETNET ARM (F95 addendum, 2026-09-24). 0.040 is a
            # measurement of ONE architecture's seed-to-seed spread -- RADDetNet, seeds
            # 42 vs 43, benchmark_v1_D2/v3 at 77 GHz (F86). It fired on every arm that
            # happened to have a CI entry, which attributed RADDetNet's training variance
            # to the ported FFTRadNet checkpoint and to CFAR. A fact stated
            # unconditionally is applied unconditionally; this is the condition.
            if _is_raddetnet_arm(beat_cfar_arm_name):
                half_width = (comp["ci_high"] - comp["ci_low"]) / 2.0
                cmp_op = ">" if SEED_TO_SEED_AP_SPREAD_F86 > half_width else "<="
                rows.append((f"bootstrap: seed spread {SEED_TO_SEED_AP_SPREAD_F86:.3f}",
                            f"{cmp_op} CI half-width {half_width:.3f}"))

    rows.extend(_ood_rows_for_arm(arm, ood_json_path))
    rows.extend(_third_corpus_rows_for_arm(arm, third_corpus_json_path))
    return rows


def scoreboard_figure(scores: Dict[str, Any], *, arm_name: str,
                      threshold: Optional[float], match_rule_text: str,
                      beat_cfar_json_path=DEFAULT_BEAT_CFAR_JSON,
                      beat_cfar_arm_name: Optional[str] = None,
                      raddetnet_ci_json_path=DEFAULT_RADDETNET_CI_JSON,
                      ood_json_path=DEFAULT_OOD_JSON,
                      third_corpus_json_path=DEFAULT_THIRD_CORPUS_JSON) -> go.Figure:
    """A compact table: the LAST frame's TP/FP/FN, cumulative hits/false alarms/FA-per-frame/
    hit-rate, and (when `beat_cfar_arm_name` names a scored arm) the two headline
    offline-scoring rows the demo's numeric claims are actually about -- the numbers
    the hostile-expert read (see the module docstring) said were missing from the
    objectness panel entirely.

    `scores` is `score_frames`'s return value. Legible at ~2 m: >=17px table font.

    LAYOUT (2026-09-24 redesign, panel-meta contract -- see `webapp.pipeline_runner`'s
    "PANEL GEOMETRY AND THE PANEL-META CONTRACT" section): the figure itself carries
    ONLY the table, at a FIXED height (`FIGURE_HEIGHT[PANEL_ROW_TABLE]`) that no
    longer depends on row count, wrapped-line count or arm-name length. Everything
    that used to be a figure title, subtitle or below-table annotation -- the
    threshold subline, the match-rule/ceiling/grouping/hit-gate-scale caveats -- is
    attached instead via `pipeline_runner.set_panel` as the panel's HTML title,
    one-line caption and Details disclosure; nothing is dropped, only moved
    (`panel_text()` is the accessor that returns all of it as one string).

    The table itself now shows AT MOST 8 rows: the 2 highest-value offline rows
    (`_offline_arm_rows`'s "FA/frame ..." and "AP, offline test split" rows, when
    `beat_cfar_arm_name` names a scored arm) plus the 6 live-run rows below. Every
    OTHER offline row (CI, seed-spread caveat, OOD, 3rd-corpus) and the connector row
    move into Details as `"<label>: <value>"` lines -- nothing invented, nothing
    dropped, same convention as the caveat sentences below.

    `beat_cfar_arm_name` (typically `arm_name_for_detector(det_meta)`'s return) adds
    the offline-scored rows for that beat_cfar.json arm -- see `_offline_arm_rows`.
    `None` (the default) leaves the table at its base 6 rows: a caller with no mapped
    arm (e.g. a checkpoint outside the scored comparison) gets no offline block rather
    than a table of blanks.

    Why the "hit rate"/"FA per frame" comparison across screens is misleading without
    context (hostile-expert re-read, 2026-09-23): every detector's on-screen threshold
    is ITS OWN recall-0.5 operating point from beat_cfar.json, so hit rate reads
    ~0.5 for every arm by construction -- a viewer comparing 0.56 (a weak detector) >
    0.50 (CFAR) > 0.47 (the strongest detector) is reading threshold-matching noise,
    not detector quality. The subline (now in Details) states the calibration
    explicitly and says what IS comparable at matched recall (false alarms); the row
    label repeats the caveat so it survives being read in isolation.
    """
    from webapp import pipeline_runner as _pr  # lazy: pipeline_runner imports this
                                                # module at load time (circular import)

    frames = scores.get("frames") or []
    last = frames[-1] if frames else None
    cum = scores["cumulative"]

    if last is None:
        this_frame = ["--", "--", "--"]
    elif not last["scored"]:
        this_frame = ["n/a", "n/a", "n/a"]
    else:
        this_frame = [str(last["tp"]), str(last["fp"]), str(last["fn"])]

    fa = scores["fa_per_frame"]
    hr = scores["hit_rate"]
    cum_values = [
        str(cum["tp"]), str(cum["fp"]),
        ("n/a" if fa != fa else f"{fa:.2f}"),
        ("n/a" if hr != hr else f"{hr:.2f}"),
    ]

    thr_txt = "n/a" if threshold is None else f"{threshold:.2f}"
    n_scored = scores["n_frames_scored"]
    n_total = scores["n_frames"]

    target_recall, n_frames_split = _load_recall_target_and_n_frames(beat_cfar_json_path)
    if threshold is not None and target_recall is not None and n_frames_split is not None:
        # Exact wording from the hostile-expert finding this fixes (2026-09-23): says
        # WHERE the threshold came from and what IS comparable across arms at a
        # threshold each one picked independently. Wave 8 (W6): the MATCHED-recall
        # claim holds only on the fixed n_frames_split-frame split -- the live
        # "recall (hits / GT), this run" row moves with the knob, and a viewer must
        # not read that movement as breaking the calibration. Now a Details line
        # (2026-09-24 redesign), not a figure subtitle -- see this function's
        # docstring.
        subline = (
            f"threshold {thr_txt} = this detector's recall-{target_recall:g} operating "
            f"point on the {n_frames_split}-frame test split (beat_cfar.json); "
            "detectors are compared at MATCHED recall, so compare false alarms, not "
            "hits; recall moves live"
        )
    else:
        subline = f"threshold {thr_txt}"

    # Renamed (hostile-expert re-read, 2026-09-23, finding 1): these rows count
    # UNMATCHED detections against labels that themselves omit real objects -- ground
    # truth misses ~3.25 real strongly-scattering objects per frame inside 40 m
    # (F83), so a detector that correctly fires on every real object still racks up
    # "false alarms" here. The row is no longer read as a true false-alarm count on
    # its own; "(FA)" is dropped from the label (Details still states the
    # upper-bound caveat, see `ceiling_caveat` below) to make room for stating THIS
    # RUN'S OWN frame count instead -- a 4th hostile-expert read (2026-09-23) found
    # this row sitting directly above the offline "FA/frame at recall ..." row with
    # no indication the two are different sample sizes (a live run is capped at
    # MAX_N_STEPS frames; beat_cfar.json's split is fixed and far larger), which
    # reads as unexplained disagreement between two numbers that simply differ in N.
    #
    # "cumulative hits"/"hit rate" split their qualifier into the VALUE column
    # (Change, 2026-09-23 coordinator re-check) instead of a long label -- a row
    # that needs 2 lines is no longer just costly, it is NOT ALLOWED (no row may
    # wrap, see `_TABLE_COL_CHARS`'s comment).
    cum_hits_str, cum_unmatched_str, fa_per_frame_str, hit_rate_str = cum_values
    # Relabelled (hostile-expert read, 2026-09-23, item 5): "hit rate (design, not
    # quality) 0.53 (5fr; R0.5/172fr split)" was ambiguous -- reading it as "16 hits
    # / 5 frames / an assumed 5 GT-per-frame" gives 0.64, not the tp/(tp+fn) this row
    # actually shows. The label now says exactly what the value is (a recall over
    # THIS run's own ground truth, whose per-frame count varies); the recall-target/
    # split-size context lives in the `subline` Details line instead, so it is not
    # repeated here.
    hit_rate_value = f"{hit_rate_str} ({n_scored}fr; GT varies/frame)"
    # "cumulative unmatched detections" (the bare running total, dropped 2026-09-23
    # 4th hostile-expert read) stays dropped: the rate right below it, naming this
    # run's own frame count in its label, already says more, and the table's <=8-row
    # budget (2026-09-24 redesign) has even less room to spare now than the old
    # <=800 px budget did.
    # "last frame", not "this frame" (hostile round 11, H3): a Plotly Table is not an
    # animatable trace, so these three rows CANNOT follow the screen's transport the
    # way the objectness map beside them now does -- they are the last frame scored,
    # always. Saying "this frame" beside a panel parked on frame 4 of 5 made the
    # audience count TP/FP/FN against a frame that was not on screen; saying "last
    # frame" is the same number, correctly labelled.
    # NAME THE FRAME (shard 3, 2026-09-24). "last frame" was correct and still left the
    # room doing arithmetic: the objectness panel beside this table steps with the clock,
    # and once its detections are drawn as matched/unmatched glyphs (H8) a viewer can
    # COUNT the hits on the frame in front of them and compare with a row about a
    # different frame. Naming which frame these rows are makes the two numbers stop
    # looking like a contradiction, at the cost of nothing.
    # SEAT'S READ OF THE 2026-09-24 RENDERS, item 1f. "frame 5 of 5: TP" is the same
    # sentence the transport prints ("frame 4 of 5") in the same words, so on a screen
    # where the clock had stepped to any other frame the two read as a contradiction
    # rather than as two different frames. The rows CANNOT follow the clock: the
    # transport steps `Plotly.animate` over a figure's frames, and this panel is a
    # Plotly Table -- so the label says WHICH frame it is and that it is the last one,
    # in words the transport never uses. `test_scoreboard_rows_name_the_last_frame_the
    # _transport_ends_on` pins the index against the transport's own maximum.
    # (The "N of M" spelling costs a character too many against `_TABLE_COL_CHARS`,
    # and it is the transport's own spelling -- which is the confusion being fixed.)
    _last_frame_label = (f"last frame {n_total}/{n_total}" if n_total
                         else "last frame")
    base_labels = [f"{_last_frame_label}: TP", f"{_last_frame_label}: unmatched (FP)",
                   f"{_last_frame_label}: FN",
                  "cumulative hits",
                  f"unmatched / frame, these {n_scored} frames",
                  "recall (hits / GT), this run"]
    base_values = this_frame + [
        f"{cum_hits_str} ({n_scored}/{n_total} scored)",
        fa_per_frame_str, hit_rate_value,
    ]
    offline_rows = (_offline_arm_rows(beat_cfar_arm_name, beat_cfar_json_path,
                                      raddetnet_ci_json_path, ood_json_path,
                                      third_corpus_json_path)
                    if beat_cfar_arm_name else [])
    # Bridges the offline block (the large, fixed test split the demo's numeric
    # CLAIMS are actually about) and the live block (this run's own, tiny,
    # <=MAX_N_STEPS-frame counts, which swing frame to frame) -- without this, a
    # viewer reads the two blocks' visibly different counts (e.g. 2.99 vs 4.00) as
    # disagreement rather than different sample sizes (hostile-expert 4th read,
    # 2026-09-23, finding 1). Only present when there IS an offline block to point
    # at. Moves to Details now (2026-09-24 redesign) rather than the visible table --
    # see the row-budget comment below.
    connector_rows = (
        [(f"{n_scored}-frame live counts vary",
          f"{n_frames_split}-frame numbers are the claim")]
        if offline_rows and n_frames_split is not None else []
    )

    # The table shows AT MOST 8 VISIBLE rows (layout spec, PANEL_ROW_TABLE's fixed
    # 312 px -- see the geometry comment above `_TABLE_HEADER_HEIGHT`): the FA/frame
    # and AP/offline-test-split rows (`_offline_arm_rows` always produces them
    # first, in that order -- see its docstring) stay on the table; every other
    # offline row (CI, seed-spread caveat, OOD, 3rd-corpus) and the connector row
    # move to Details instead. Matched by CONTENT rather than position, so this
    # still finds the right two rows even if a future arm has no FA/frame row
    # (`_offline_arm_rows` omits it when `fp_per_frame` is absent).
    visible_offline: List[Tuple[str, str]] = []
    offline_detail_rows: List[Tuple[str, str]] = []
    got_fa = got_ap = False
    for row in offline_rows:
        label = row[0]
        if not got_fa and label.startswith("FA/frame"):
            visible_offline.append(row)
            got_fa = True
        elif not got_ap and label in ("AP, offline test split", "offline test split"):
            visible_offline.append(row)
            got_ap = True
        else:
            offline_detail_rows.append(row)

    raw_labels = [r[0] for r in visible_offline] + base_labels
    raw_values = [r[1] for r in visible_offline] + base_values
    n_rows = len(raw_labels)
    assert n_rows <= 8, (
        f"scoreboard table grew to {n_rows} visible rows -- the panel's fixed "
        f"height ({_pr.FIGURE_HEIGHT[_pr.PANEL_ROW_TABLE]} px) is budgeted for <=8; "
        "move the new row into Details instead of the visible table"
    )

    # No row may wrap: Plotly's Table `cells.height` is a single scalar for the
    # WHOLE table, not one-per-row, so a multi-line row would force EVERY row past
    # its fixed `_TABLE_MAX_ROW_HEIGHT` cap rather than just itself. `_wrap_text` is
    # kept ONLY to detect that a row would need a second line -- never to actually
    # insert one -- so this fails loudly (an assert) instead of silently blowing the
    # table past its fixed height.
    for lbl, val in zip(raw_labels, raw_values):
        assert "<br>" not in _wrap_text(lbl, max_chars=_TABLE_COL_CHARS), (
            f"scoreboard row label {lbl!r} needs >1 line at {_TABLE_COL_CHARS} chars "
            "-- shorten the VALUE text (not this label, not the font) so the row "
            "still fits the fixed row height"
        )
        assert "<br>" not in _wrap_text(val, max_chars=_TABLE_COL_CHARS), (
            f"scoreboard row value {val!r} (label {lbl!r}) needs >1 line at "
            f"{_TABLE_COL_CHARS} chars -- shorten it, don't let this row wrap"
        )

    row_height = min(_TABLE_MAX_ROW_HEIGHT,
                     (_pr.FIGURE_HEIGHT[_pr.PANEL_ROW_TABLE] - _TABLE_HEADER_HEIGHT)
                     // n_rows)

    fig = go.Figure(data=[go.Table(
        columnwidth=_TABLE_COL_WIDTHS,
        # Second column used to be an empty dark cell -- it labels the counts below it.
        header=dict(values=[arm_name, "count"],
                   fill_color="#2d3436", font=dict(color="white", size=17),
                   height=_TABLE_HEADER_HEIGHT, align="left"),
        cells=dict(
            values=[raw_labels, raw_values],
            fill_color=[["#f5f6fa"] * n_rows, ["#ffffff"] * n_rows],
            font=dict(size=17), height=row_height, align="left",
        ),
    )])
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=_pr.FIGURE_HEIGHT[_pr.PANEL_ROW_TABLE],
        paper_bgcolor=_pr.PAPER_BGCOLOR, plot_bgcolor=_pr.PLOT_BGCOLOR,
    )

    # The match rule, in words with its numbers (see match_rule_text()), plus one more
    # sentence (finding 1) saying the precision ceiling those "unmatched" rows above
    # cannot exceed and what that means for reading them. Moved into Details verbatim,
    # UNWRAPPED (2026-09-24 redesign): Details is HTML and wraps itself, so the old
    # `_wrap_text`-at-70-chars pass (needed only to keep a figure annotation inside a
    # panel's pixel width) is no longer applied to these sentences at all.
    # Wave 8 (W5): this ceiling binds at FULL recall (every real object fired on),
    # not at every point on the PR curve beside this table -- the clarification
    # ("ceiling at full recall") lives in `stored_pr_figure`'s own Details.
    ceiling_caveat = (
        "labels omit ~3 real scatterers per frame inside 40 m (precision ceiling "
        f"{PRECISION_CEILING_F83:.2f}); unmatched is an upper bound on "
        "false alarms"
    )
    # Item 3 (hostile-expert read, 2026-09-23): four detection crosses sitting on ONE
    # ground-truth box scored as 1 TP + 3 unmatched read as a bug on screen. It is the
    # same 3x3 peak-grouping every detector here is scored under (verified against
    # `e2e.ml.baseline.classical_detection_map`'s own default, `peak_grouping=True`,
    # `group_radius=1` -- the on-screen CFAR path and the offline scorer never pass a
    # different value) -- stated so a viewer does not read those extra crosses as an
    # unfair count against one detector.
    grouping_caveat = (
        "detections are grouped to local peaks (3x3) before matching, the same "
        "rule for every detector; a wide target can draw extra unmatched hits"
    )
    # Wave 7 X7: neither hit-gate tolerance had a physical-scale reading anywhere on
    # screen (a viewer sees "0.06" and "2 m" with no sense of whether that is loose
    # or tight against this array/corpus) -- computed by `hit_gate_scale_note` rather
    # than typed here so it cannot drift.
    scale_caveat = hit_gate_scale_note()

    # Everything that isn't one of the table's 8 visible rows, in the same content --
    # nothing dropped, only moved (layout spec's Details contract): the offline rows
    # the table had no room for, the live/offline connector, then the 4 caveat
    # sentences and the threshold subline that used to be the figure's annotation
    # and title/subtitle.
    # Why three rows say "last frame" on a screen whose other panels follow the clock.
    transport_caveat = (
        "the per-frame rows name the LAST frame scored, not the frame the "
        "transport is parked on: a table is not an animatable Plotly trace, so it "
        "cannot step with the clock the objectness map beside it follows"
    )
    details = ([f"{lbl}: {val}" for lbl, val in offline_detail_rows]
              + [f"{lbl}: {val}" for lbl, val in connector_rows]
              + [match_rule_text, ceiling_caveat, grouping_caveat, scale_caveat,
                 transport_caveat, subline])

    # The caption states WHERE the matched-recall comparison holds, and stops
    # instructing one the live rows cannot support (hostile round 11, H5): the two
    # arms' live rows are at recall 0.50 and 0.13 on this run, so "compare false
    # alarms, not hits" read as an instruction to compare the LIVE false-alarm rows,
    # which are not at matched recall at all. Only the offline split is. ("compared
    # at" is dropped from the reviewer's suggested wording purely for width: the
    # caption is ONE 16 px line in a 746 px column, ~86 characters, and the threshold
    # clause shares it.)
    matched_at = (f"matched recall on the {n_frames_split}-frame split; "
                 "live rows are this run"
                 if n_frames_split is not None else
                 "matched recall on the offline split; live rows are this run")
    _pr.set_panel(
        fig, title="Detector scoreboard",
        caption=[f"threshold {thr_txt}", matched_at],
        details=details, row=_pr.PANEL_ROW_TABLE,
    )
    return fig


#: Label prefixes of the scoreboard rows that are OFFLINE constants: identical on
#: both arms of any A/B, because no knob on this screen can move them. Matched by
#: prefix (the labels carry their own recall target / frame count) by
#: `fold_offline_rows_onto_arm_a`.
OFFLINE_ROW_PREFIXES = ("FA/frame", "AP, offline", "offline test split")
#: What arm A's copy of those rows says, so printing them under ONE arm cannot read
#: as "this arm scored that".
OFFLINE_BOTH_ARMS_SUFFIX = ", both arms"


def _table_cells(fig: Dict[str, Any]):
    """`(labels, values)` of a stored scoreboard figure DICT, or `None`."""
    for trace in (fig.get("data") or []):
        if trace.get("type") != "table":
            continue
        values = (trace.get("cells") or {}).get("values")
        if isinstance(values, list) and len(values) == 2:
            return trace, values
    return None


def fold_offline_rows_onto_arm_a(fig_a: Dict[str, Any], fig_b: Dict[str, Any]) -> None:
    """Print the knob-invariant OFFLINE rows ONCE, under arm A, and leave arm B's
    table holding only rows that move (hostile round 11, H9).

    Both scoreboards used to print byte-identical "FA/frame at recall 0.5, 172 frames"
    and "AP, offline test split" rows under two different arm headings, which invites
    exactly the reading the whole screen exists to prevent: "so the knob changed the
    AP?". Arm A's copy now says `, both arms`; arm B's move into arm B's Details, in
    full, labelled -- nothing is deleted, and the rows that DO move (this run's own
    counts) are all that is left beside them.

    Takes the two stored figure dicts for the SAME scoreboard product (arm A's and arm
    B's) and edits them in place. A figure without a table trace is left alone."""
    cells_a, cells_b = _table_cells(fig_a or {}), _table_cells(fig_b or {})
    if not cells_a or not cells_b:
        return
    _, (labels_a, values_a) = cells_a
    trace_b, (labels_b, values_b) = cells_b
    for i, lbl in enumerate(labels_a):
        if (str(lbl).startswith(OFFLINE_ROW_PREFIXES)
                and not str(values_a[i]).endswith(OFFLINE_BOTH_ARMS_SUFFIX)):
            values_a[i] = f"{values_a[i]}{OFFLINE_BOTH_ARMS_SUFFIX}"
    keep = [i for i, lbl in enumerate(labels_b)
            if not str(lbl).startswith(OFFLINE_ROW_PREFIXES)]
    if len(keep) == len(labels_b):
        return
    moved = [(labels_b[i], values_b[i]) for i in range(len(labels_b))
             if i not in keep]
    trace_b["cells"]["values"] = [[labels_b[i] for i in keep],
                                  [values_b[i] for i in keep]]
    fill = (trace_b.get("cells") or {}).get("fill_color")
    if isinstance(fill, list) and len(fill) == 2:
        trace_b["cells"]["fill_color"] = [c[:len(keep)] if isinstance(c, list) else c
                                          for c in fill]
    panel = _pr_panel_dict(fig_b)
    panel["details"] = list(panel.get("details") or []) + [
        f"{lbl}: {val} -- scored offline, identical on both arms; printed once, "
        "under arm A." for lbl, val in moved]


def _pr_panel_dict(fig: Dict[str, Any]) -> Dict[str, Any]:
    """`webapp.pipeline_runner._panel_dict`, imported lazily (that module imports this
    one at load time -- circular import)."""
    from webapp import pipeline_runner as _pr

    return _pr._panel_dict(fig)


def _load_beat_cfar(path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"beat_cfar.json not found: {path}")
    data = json.loads(path.read_text())
    if "arms" not in data:
        raise ValueError(f"{path}: no 'arms' key -- expected e2e.ml.compare_detectors' "
                         "output format (see e2e/ml/runs/beat_cfar.json)")
    return data


def arm_name_for_detector(det_meta: Dict[str, Any],
                          beat_cfar_json_path=DEFAULT_BEAT_CFAR_JSON) -> Optional[str]:
    """The beat_cfar.json arm this on-screen Detector corresponds to, or `None` if it
    is not one of the scored arms (e.g. a checkpoint outside that comparison).

    `det_meta` is `webapp.pipeline_runner._detector_meta`'s return
    (`{"mode", "threshold", "label"}`). CFAR mode always maps to the ``"classical CFAR"``
    arm if the JSON has one; ML mode maps via the checkpoint's PARENT DIRECTORY name --
    which `_detector_meta` already uses as `label` -- matched against each scored
    checkpoint arm's own `Path(arm["checkpoint"]).parent.name`.
    """
    data = _load_beat_cfar(beat_cfar_json_path)
    arms = data["arms"]
    mode = det_meta.get("mode")
    if mode == "cfar":
        return "classical CFAR" if any(a.get("name") == "classical CFAR" for a in arms) else None
    if mode == "ml":
        label = det_meta.get("label")
        for a in arms:
            ckpt = a.get("checkpoint")
            if ckpt and Path(ckpt).parent.name == label:
                return a["name"]
    return None


def _downsample(xs: List[float], ys: List[float], max_points: int = 400):
    """Stride-decimate a pooled PR curve (36k+ points/arm here) to a size a browser
    renders instantly, keeping the first/last point exactly."""
    n = len(xs)
    if n <= max_points:
        return xs, ys
    step = max(1, n // max_points)
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return [xs[i] for i in idx], [ys[i] for i in idx]


def _raddetnet_ci_for_arm(arm_name: str, raddetnet_ci_json_path) -> Optional[Dict[str, float]]:
    """This arm's bootstrap AP-delta-vs-CFAR row from `raddetnet_ci_json_path`
    (`{"delta_AP", "ci_low", "ci_high"}`), or `None` if the file is absent, unreadable,
    or has no `comparisons` entry for this arm -- e.g. a checkpoint scored in
    beat_cfar.json before the CI tool covered it. Never invented: the caller omits the
    interval entirely rather than showing a placeholder."""
    path = Path(raddetnet_ci_json_path)
    if not path.is_file():
        return None
    try:
        ci_data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    comp = next((c for c in ci_data.get("comparisons", []) if c.get("arm") == arm_name), None)
    if comp and all(k in comp for k in ("delta_AP", "ci_low", "ci_high")):
        return comp
    return None


#: PR-panel geometry (px), FIXED under the panel-meta contract (2026-09-24 layout
#: redesign -- see `webapp.pipeline_runner`'s "PANEL GEOMETRY AND THE PANEL-META
#: CONTRACT" section): the figure carries no title/subtitle any more, so nothing
#: here derives from a title's own wrapped-line count.
#:
#: RETIRED (2026-09-24): `_PR_MARGIN_T_BASE`, `_PR_MARGIN_T_PER_LINE` and
#: `_PR_MIN_TITLE_LINES` existed only to grow the top margin with the old
#: multi-line title/subtitle (and to keep arm A/B margins equal while doing it --
#: `_PR_MIN_TITLE_LINES` floored every call to the same line count so
#: `webapp.app._arm_result`'s extra "<br><sup>" line for arm B never shifted its
#: axes relative to arm A's). Every clause that title carried now lives in this
#: function's Details instead (`stored_pr_figure`, off the figure entirely), so the
#: margin is a small FIXED constant like every other panel's -- see that function's
#: docstring for where each clause went, including the arm-B-only sentence, which
#: is now a standing Details line rather than something only arm B's copy carries.
#: 72, matching `pipeline_runner._FIG_MARGIN_L` (hostile round 11, D7): this panel
#: sits in the same COLUMN as the map panels on every Thrust 5 screen, and at 64 its
#: plot origin landed 8 px left of theirs -- the stagger down the left edge the check
#: is about. Kept as a literal rather than an import for the same reason the bottom
#: margin below is: these are this module's own layout decisions, and the two
#: constants are pinned equal by tests/test_webapp_layout_acceptance.py instead.
_PR_MARGIN_L = 72
_PR_MARGIN_R = 16
_PR_MARGIN_T = 12
#: Legend strip below the plot: 2 columns x 3 rows at 17 px (6 arms in beat_cfar.json
#: today -> 3 rows at `entrywidth=0.5`), plus the x-axis title/tick allowance every
#: other panel in this package budgets (`pipeline_runner._FIG_MARGIN_B`) -- kept as
#: a literal here rather than importing that private constant, since this panel's
#: margins are this module's own layout decision, not something that should
#: silently move if that constant's value ever changes for a heatmap's needs.
#: 80, not the spec table's 64 (measured on the rendered page, 2026-09-24): at 64
#: the two-column legend had room for 2.5 of its 3 rows, so TWO arms -- including
#: the null / chance-floor curve, which is the honesty anchor of this panel and
#: must never be the entry that falls off -- were silently cut off the bottom.
#: The plot is still 324 px tall, i.e. 50.8 % of the panel's area, above
#: acceptance check 5's floor.
_PR_LEGEND_STRIP_PX = 80
_PR_AXIS_MARGIN_B = 60
_PR_MARGIN_B = _PR_LEGEND_STRIP_PX + _PR_AXIS_MARGIN_B


def stored_pr_figure(beat_cfar_json_path=DEFAULT_BEAT_CFAR_JSON, *,
                     highlight_arm: Optional[str] = None,
                     raddetnet_ci_json_path=DEFAULT_RADDETNET_CI_JSON) -> go.Figure:
    """The offline-scored PR curve for every arm in `beat_cfar.json`, so a live run's
    single operating point sits next to the whole curve it was read off.

    Falls back, PER ARM, to plotting just its recall-0.5 (or whatever `target_recall`
    the JSON recorded) operating point -- with AP in the legend, and a Details line
    saying so -- for any arm that has no stored `pr_curve`. The JSON this was built
    against (2026-09-22 `e2e/ml/runs/beat_cfar.json`) has a full curve for every arm,
    so this is a documented fallback, not the observed case; re-verify against a
    fresh `beat_cfar.json` before assuming it never fires.

    `highlight_arm`'s legend entry also carries its bootstrap AP-delta-vs-CFAR 95% CI
    from `raddetnet_ci_json_path`, when that file has a row for it (see
    `_raddetnet_ci_for_arm`) -- omitted, never invented, otherwise. Its own curve
    draws at `width=5`/full opacity; every OTHER arm draws dimmed (`width=2`,
    `opacity=0.45`) EXCEPT the null/chance-floor arm, which this panel never lets
    fade: it is the honesty anchor a viewer needs even while looking at a different
    arm's curve, so it always stays at full opacity.

    LAYOUT (2026-09-24 redesign, panel-meta contract -- see
    `webapp.pipeline_runner`'s "PANEL GEOMETRY AND THE PANEL-META CONTRACT" section):
    the figure carries no title/subtitle and no in-plot fallback banner; every
    clause those used to carry (the offline-split/corpus statement, the
    in-distribution/seed qualifier, the precision-ceiling-at-full-recall
    clarification, the "identical on both arms" note that used to live only on arm
    B's copy in `webapp.app._arm_result`, and the no-stored-curve fallback banner)
    is attached instead via `pipeline_runner.set_panel` as Details -- nothing
    dropped, only moved.

    Raises if the JSON is missing `arms`, or `arms` is empty -- there is nothing
    invented in place of a genuinely absent scoring artifact.
    """
    from webapp import pipeline_runner as _pr  # lazy: circular import, see module docstring

    data = _load_beat_cfar(beat_cfar_json_path)
    arms = data["arms"]
    if not arms:
        raise ValueError(f"{beat_cfar_json_path}: 'arms' is empty -- nothing to plot")

    manifest = data.get("manifest", "")
    # DATASET ROOT included (item 7, wave 9 second hostile read, 2026-09-24): the
    # scene-tier subdirectory name alone ("benchmark_v1_D2") is shared by more than
    # one dataset root -- both the offline scoring corpus this figure is captioned
    # after (b1_bench_v3/benchmark_v1_D2) and the live Thrust 5 demo corpus
    # (b1_demo_cfr/benchmark_v1_D2) -- so a bare "benchmark_v1_D2" reads as one
    # corpus when it names two. Read from the SAME `manifest` path every other field
    # here already reads (never typed) -- same convention `run_pipeline`'s
    # `trained_on` string and `_third_corpus_rows_for_arm`'s tier parsing use
    # elsewhere (`Path(manifest).parent.parent.name`, the dataset root).
    corpus_tier = Path(manifest).parent.name if manifest else "?"
    corpus_root = Path(manifest).parent.parent.name if manifest else ""
    corpus_name = f"{corpus_root}/{corpus_tier}" if corpus_root else corpus_tier
    n_frames = None
    for a in arms:
        gpf = a.get("gt_per_frame")
        if gpf:
            n_frames = len(gpf)
            break
    if n_frames is None:
        n_frames = next(
            (a["operating_point"]["n_frames"] for a in arms if a.get("operating_point")),
            "?",
        )

    fig = go.Figure()
    fallback_arms: List[str] = []
    highlight_disp_name: Optional[str] = None
    highlight_ci: Optional[Dict[str, float]] = None
    highlight_ap: Optional[float] = None
    for a in arms:
        name = a.get("name", "?")
        # Display only -- e.g. the stored "null (random-in-GT-box)" reads as "random
        # INSIDE the ground-truth boxes" (cheating); see `_ARM_DISPLAY_NAMES` for the
        # actual definition. `name` (the stored JSON value) is still what every lookup
        # below (bold/CI matching) keys on.
        disp_name = _display_arm_name(name)
        bold = highlight_arm is not None and name == highlight_arm
        # The null/chance-floor arm never dims (see this function's docstring): a
        # viewer looking at some OTHER highlighted arm must still see where "chance"
        # sits -- fading it like every other non-highlighted arm would let the one
        # honesty anchor on this panel visually disappear. `_ARM_DISPLAY_NAMES`'s
        # keys are exactly the arm names this module treats specially, so membership
        # in it is also the null-arm test.
        is_null = name in _NULL_ARM_NAMES
        opacity = 1.0 if (bold or is_null) else 0.45
        ap = a.get("AP", float("nan"))
        if bold:
            highlight_disp_name, highlight_ap = disp_name, ap
        pr = a.get("pr_curve")
        if pr and pr.get("recall") and pr.get("precision") is not None:
            x, y = _downsample(pr["recall"], pr["precision"])
            ci = _raddetnet_ci_for_arm(name, raddetnet_ci_json_path) if bold else None
            if ci is not None:
                # The delta stays ON the highlighted entry; its CONFIDENCE INTERVAL
                # moves to the panel caption (measured on the rendered page,
                # 2026-09-24: the full 51-character entry overran its half of the
                # two-column legend and drew straight through the entry beside it).
                # The caption is the more visible of the two places anyway.
                # SHORTENED AGAIN (hostile round 11, D3): even without the CI, the
                # 34-character "raddetnet AP 0.476, +0.175 vs CFAR" filled its half of
                # the two-column strip edge to edge and abutted "fftradnet_rd_b5
                # (AP=0.127)" beside it with ZERO gap, so the lead screen's one
                # headline entry read as one run-on string and the delta looked like
                # it was against fftradnet. "vs CFAR" is what the caption below
                # spells out, in full, with the interval.
                trace_name = f"{disp_name} {ap:.3f} ({ci['delta_AP']:+.3f})"
                highlight_ci = ci
            else:
                trace_name = f"{disp_name} (AP={ap:.3f})"
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines", name=trace_name, opacity=opacity,
                line=dict(width=5 if bold else 2),
            ))
        else:
            op = a.get("operating_point") or {}
            if op.get("reached"):
                tp, fp = op.get("tp"), op.get("fp")
                prec = (tp / (tp + fp)) if (tp is not None and fp is not None
                                            and (tp + fp) > 0) else None
                fig.add_trace(go.Scatter(
                    x=[op["recall_achieved"]], y=[prec], mode="markers", opacity=opacity,
                    marker=dict(size=18 if bold else 11, symbol="star"),
                    name=f"{disp_name} (AP={ap:.3f}, recall-{op['target_recall']:g} pt only)",
                ))
                fallback_arms.append(disp_name)

    # Explicit, not the implicit full-width default (Change, 2026-09-23 coordinator
    # re-check): this is the property "the plot is not squeezed by the legend" is
    # tested against, so it must be a real, asserted value, not an assumption about
    # what Plotly leaves alone.
    fig.update_xaxes(title=dict(text="recall", font=dict(size=18)), range=[0, 1],
                     domain=[0.0, 1.0])
    fig.update_yaxes(title=dict(text="precision", font=dict(size=18)), range=[0, 1])
    fig.update_layout(
        # Below the plot, 2 columns x ~3 rows (Change, 2026-09-24: was a single
        # centred row; `entrywidth=0.5` packs 2 entries per row, and Plotly wraps to
        # further rows by itself once 6 arms don't fit one -- still never costs plot
        # WIDTH, only the fixed `_PR_MARGIN_B` strip below, regardless of entry
        # length, same reasoning as the 2026-09-23 move below the plot in the first
        # place).
        legend=dict(orientation="h", entrywidthmode="fraction", entrywidth=0.5,
                   y=-0.16, yanchor="top", x=0, xanchor="left", font=dict(size=17)),
        # 18, not 16: the in-figure floor is 17 px (layout spec section 3) and the
        # axis tick labels inherit this (measured on the rendered page, 2026-09-24).
        font=dict(size=18),
        # `autoexpand=False` (measured on the rendered page, 2026-09-24): with
        # Plotly's default auto-expansion the horizontal legend below the plot is
        # taken OUT of the axis domain ON TOP of the bottom margin already
        # reserved for it, so the plot rendered 285 px tall inside a band sized
        # for 340 and the panel sat at 45 % of its own area against the spec's
        # >= 50 % (acceptance check 5). The margins here are hand-computed, so
        # there is nothing for auto-expansion to discover.
        margin=dict(l=_PR_MARGIN_L, r=_PR_MARGIN_R, t=_PR_MARGIN_T,
                    b=_PR_MARGIN_B, autoexpand=False),
        height=_pr.FIGURE_HEIGHT[_pr.PANEL_ROW_PR],
        paper_bgcolor=_pr.PAPER_BGCOLOR, plot_bgcolor=_pr.PLOT_BGCOLOR,
    )

    if highlight_arm is not None and highlight_disp_name is not None:
        # SHORT by construction: the caption renders on ONE line in a 746 px column at
        # 16 px (~86 characters). With the CI clause and arm B's "identical on both
        # arms" clause appended, "<name> highlighted, AP x" was clipped (measured on
        # the rendered page, 2026-09-24) -- the word "highlighted" is what the thick
        # line already says.
        caption = [f"{highlight_disp_name} AP {highlight_ap:.3f}"]
        if highlight_ci is not None:
            caption.append(f"{highlight_ci['delta_AP']:+.3f} vs CFAR "
                           f"[{highlight_ci['ci_low']:+.3f}, "
                           f"{highlight_ci['ci_high']:+.3f}]")
    else:
        caption = [f"{n_frames} test frames, {corpus_name}"]

    # Every clause the old figure title/subtitle and in-plot fallback banner carried,
    # VERBATIM (2026-09-24 redesign) -- same convention as `scoreboard_figure`'s 4
    # caveat sentences: unmodified substrings of the old `title_text`, split at the
    # same "<br>" boundaries that used to separate its lines, not re-punctuated or
    # merged. The "identical on both arms" sentence used to be appended only to arm
    # B's copy of the OLD title text by `webapp.app._arm_result`; it is a fact about
    # this figure, true on every call, so it is a standing Details line here instead
    # -- app.py's title-mutating append is stale against a figure that no longer has
    # a `layout.title` to append to and needs updating by that file's owner (not
    # this module) to stop relying on it.
    details = [
        # The null arm's legend entry is a short "null (chance floor)" so it cannot be
        # the entry that falls off the two-column strip; its real definition is here.
        NULL_ARM_DEFINITION,

        f"scored offline: {n_frames} test frames, {corpus_name}",
        "beat_cfar.json; in-distribution: held-out scenes of the training corpus; "
        "one training seed per curve",
        "the scoreboard's precision ceiling binds at full recall, not near recall 0",
        "scored offline; identical on both arms, the knob cannot move it",
    ]
    if fallback_arms:
        details.append(
            "no stored PR curve for: " + ", ".join(fallback_arms) +
            " -- showing their recall-0.5 operating point only"
        )

    _pr.set_panel(fig, title="Precision-recall, offline test split",
                 caption=caption, details=details, row=_pr.PANEL_ROW_PR)
    return fig
