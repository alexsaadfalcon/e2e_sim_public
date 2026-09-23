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
DEFAULT_BEAT_CFAR_JSON = _REPO_ROOT / "e2e" / "ml" / "runs" / "beat_cfar.json"
#: Bootstrap AP-delta-vs-CFAR confidence intervals (e2e.ml's paired-bootstrap CI tool),
#: keyed by arm name -- optional: a checkpoint scored in beat_cfar.json need not have a
#: CI entry here yet (F85 addendum). Overridable per call; never edited by this module.
DEFAULT_RADDETNET_CI_JSON = _REPO_ROOT / "e2e" / "ml" / "runs" / "raddetnet_ci.json"
#: A SEPARATE offline-scored run against an out-of-distribution corpus (b1_bench_v2,
#: an earlier generator/impairment model than beat_cfar.json's b1_bench_v3) -- F86
#: (notes/ESTABLISHED_FACTS.md, measured 2026-09-22). Optional: an arm not scored here
#: simply gets no OOD row (`_ood_rows_for_arm`). Overridable per call; never edited.
DEFAULT_OOD_JSON = _REPO_ROOT / "e2e" / "ml" / "runs" / "gen_s43_v2_test.json"

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

#: In-distribution seed-42-vs-seed-43 AP spread for RADDetNet on benchmark_v1_D2/v3,
#: one architecture, two training seeds, identical protocol (notes/ESTABLISHED_FACTS.md
#: F86, measured 2026-09-22) -- the scale a single checkpoint's scene-bootstrap CI
#: cannot speak to, since bootstrapping resamples SCENES of one already-trained
#: network, never a second training run.
SEED_TO_SEED_AP_SPREAD_F86 = 0.040

#: The null arm's stored name reads as "random INSIDE the ground-truth boxes" (i.e.
#: the detector is handed the answer) -- it is actually uniform-random scores inside
#: the BOUNDING BOX of every TRAIN-split label (never the eval labels), fit once and
#: applied blind to the RF (e2e.ml.compare_detectors.score_null's docstring,
#: e2e/ml/compare_detectors.py:189-192). Display-only remap; the stored JSON name
#: (and the JSON itself) is never edited.
_ARM_DISPLAY_NAMES = {
    "null (random-in-GT-box)":
        "null: random cells within the train-label bounding box (chance floor)",
}


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


#: Table geometry (px): header + data rows must fit inside the domain the figure's
#: own `height`/`margin` leaves for the table trace, or the LAST rows get silently cut
#: off by the renderer -- not resized, not scrolled (rehearsal, 2026-09-23: a long
#: header ("CA-CFAR (guard 2, train 6) -- threshold 0.66") wrapped to two lines inside
#: its declared single-line height, stealing room from the bottom of the table and
#: clipping the "hit rate" row -- the CFAR and neural-detector arms otherwise run the
#: same code and must show identical rows). Threshold moved out of the header (into
#: the title, below) specifically so the header text stays short enough not to wrap;
#: the extra header/margin slack below is a second, independent guard for arm names
#: this module does not control the length of (e.g. an ML checkpoint's directory name).
#: Row COUNT is no longer fixed (see `scoreboard_figure`'s `beat_cfar_arm_name`): the
#: offline-scored block appends a variable number of rows for an arm found in
#: beat_cfar.json (AP/FA/stripe/CI/OOD, some of them themselves wrapped to more than
#: one line -- see `_TABLE_COL_LABEL_CHARS`), so the table height below is computed
#: from the actual PER-ROW rendered height at call time, never from a hardcoded row
#: count or a uniform row height -- the same clipping bug this comment describes would
#: otherwise recur the moment that block's row count or a row's line count changed.
_TABLE_HEADER_HEIGHT = 40
_TABLE_ROW_HEIGHT = 30
#: Character budget for a long sentence pre-wrapped INTO the (280 px) label column
#: (e.g. the CI-row caveat, the OOD row) rather than left in the annotation below the
#: table -- narrower than the 70-char budget `_wrap_text`'s other callers use for the
#: full ~600 px panel width, calibrated against this column's existing longest
#: unwrapped label ("rank-1 stripe vs ground truth", 30 chars, one line at font 18).
_TABLE_COL_LABEL_CHARS = 34
#: Base top margin: one title line ("Detector scoreboard") + a ONE-line subtitle
#: ("threshold 0.44"). The real subline (see `scoreboard_figure`) is usually longer
#: and wraps to several lines -- each one needs `_TABLE_SUBLINE_LINE_PX` more margin
#: or Plotly overflows the title DOWN into the table's own header row instead of
#: clipping it (measured, thrust5_detector_cfar rehearsal PNG, 2026-09-23 pixel
#: re-check).
_TABLE_MARGIN_T = 90
_TABLE_SUBLINE_LINE_PX = 32
#: Per-wrapped-line px budget for the annotation block below the table (match rule +,
#: since the F83 precision-ceiling caveat below, the caption stating unmatched counts
#: are an upper bound on false alarms) -- calibrated against the ORIGINAL fixed 120 px
#: budget, which fit exactly the match-rule sentence alone at its default 3 wrapped
#: lines (120 / 3 = 40). `scoreboard_figure` now computes the real total from both
#: annotations' actual wrapped line counts instead of assuming that fixed content.
_TABLE_ANNOTATION_LINE_PX = 40


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


def _ood_rows_for_arm(bc_arm: Dict[str, Any], ood_json_path=DEFAULT_OOD_JSON
                      ) -> List[Tuple[str, str]]:
    """`[(label, value)]` stating `bc_arm`'s (one beat_cfar.json arm's) performance on
    a SEPARATE, out-of-distribution corpus (F86, notes/ESTABLISHED_FACTS.md) -- `[]`
    if `ood_json_path` is absent/malformed or carries no arm matching `bc_arm` (never
    invented for an arm that file never scored).

    Matched by CHECKPOINT PARENT DIRECTORY for an ML arm (the same convention
    `arm_name_for_detector` uses against beat_cfar.json), or by NAME for CFAR (which
    carries no checkpoint).

    For a two-seed RADDetNet arm specifically (its OOD name ends "_s<seed>" and a
    sibling "_s<other seed>" arm of the same base name is in the SAME file), also
    finds the CFAR arm in that file and states whether CFAR's AP falls between the two
    seeds' -- F86's finding that the out-of-distribution lead is seed-dependent,
    computed from these three numbers at call time, never typed as a conclusion here.
    """
    ood_arms = _load_json_arms(ood_json_path)
    if not ood_arms:
        return []

    ckpt = bc_arm.get("checkpoint")
    if ckpt:
        ckpt_dir = Path(ckpt).parent.name
        ood_arm = next((a for a in ood_arms if a.get("checkpoint") and
                        Path(a["checkpoint"]).parent.name == ckpt_dir), None)
    else:
        ood_arm = next((a for a in ood_arms if a.get("name") == bc_arm.get("name")), None)
    if ood_arm is None or ood_arm.get("AP") is None:
        return []

    try:
        ood_manifest = json.loads(Path(ood_json_path).read_text()).get("manifest", "")
    except (OSError, ValueError):
        ood_manifest = ""
    # e.g. "e2e/ml/datasets/b1_bench_v2/benchmark_v1_D2/manifest.json" -> "b1_bench_v2"
    # (the GENERATOR/corpus version, not the scene family both beat_cfar.json and this
    # file happen to share -- "benchmark_v1_D2" -- which is what distinguishes them).
    corpus = Path(ood_manifest).parent.parent.name if ood_manifest else "?"
    label = f"out of distribution ({corpus} test)"

    ap = ood_arm["AP"]
    fa = (ood_arm.get("operating_point") or {}).get("fp_per_frame")

    sibling = None
    base, sep, seed_a = ood_arm.get("name", "").rpartition("_s")
    if base and sep:
        sibling = next((a for a in ood_arms if a is not ood_arm
                        and a.get("name", "").startswith(base + "_s")
                        and a.get("AP") is not None), None)
    cfar_arm = next((a for a in ood_arms if a.get("name") == "classical CFAR"), None)

    if sibling is not None and cfar_arm is not None and cfar_arm.get("AP") is not None:
        seed_b = sibling["name"].rpartition("_s")[2]
        cfar_ap, sib_ap = cfar_arm["AP"], sibling["AP"]
        lo, hi = sorted((ap, sib_ap))
        verdict = ("the two seeds straddle CFAR" if lo < cfar_ap < hi else
                   f"both seeds land on the same side of CFAR ({cfar_ap:.3f})")
        value = (f"OOD AP {ap:.3f} (seed {seed_a}) / {sib_ap:.3f} (seed {seed_b}) "
                f"vs CFAR {cfar_ap:.3f}: {verdict}")
    else:
        value = f"OOD AP {ap:.3f}" + (f", FA/frame {fa:.1f}" if fa is not None else "")

    # Long sentence, table column is narrow (see `_TABLE_COL_LABEL_CHARS`) -- pre-wrap
    # into the label column (280 px, twice the value column's width) with the value
    # column left blank, same convention `_offline_arm_rows` already uses for a
    # section-header row. `scoreboard_figure` derives that row's rendered HEIGHT from
    # the "<br>" count left in this string -- see its row-height comment.
    return [(label, ""), (_wrap_text(value, max_chars=_TABLE_COL_LABEL_CHARS), "")]


def _offline_arm_rows(beat_cfar_arm_name: str, beat_cfar_json_path=DEFAULT_BEAT_CFAR_JSON,
                      raddetnet_ci_json_path=DEFAULT_RADDETNET_CI_JSON,
                      ood_json_path=DEFAULT_OOD_JSON
                      ) -> List[Tuple[str, str]]:
    """`(label, value)` rows for the offline scoring of ONE beat_cfar.json arm: a
    section-header row naming the split, then AP, FA/frame at that arm's recall
    target, and -- only when the arm has one (CFAR does not) -- its rank-1 stripe
    statistic against the stored ground-truth reference. Appends a bootstrap AP-delta-
    vs-CFAR row too, but ONLY when `raddetnet_ci_json_path` exists AND carries a
    `comparisons` entry for this exact arm -- never invented for an arm the CI file
    hasn't scored yet -- plus, right after it, a caveat row stating the seed-to-seed
    AP spread the CI's scene-bootstrap cannot see (F86). Finally appends an
    out-of-distribution row (`_ood_rows_for_arm`) when a separate OOD-scored JSON
    covers this arm.

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
    header_label = (f"offline, {n_frames}-frame test split (beat_cfar.json)"
                    if n_frames is not None else "offline (beat_cfar.json)")
    rows: List[Tuple[str, str]] = [(header_label, "")]

    ap = arm.get("AP")
    if ap is not None:
        rows.append(("AP", f"{ap:.3f}"))

    fa_pf = op.get("fp_per_frame")
    if fa_pf is not None:
        label = (f"FA/frame at recall {target_recall:g}" if target_recall is not None
                 else "FA/frame")
        rows.append((label, f"{fa_pf:.2f}"))

    beat_cfar_block = data.get("beat_cfar") or {}
    stripe = (beat_cfar_block.get("stripe_rank1") or {}).get(beat_cfar_arm_name)
    stripe_gt = beat_cfar_block.get("stripe_ground_truth")
    if stripe is not None and stripe_gt is not None:
        rows.append(("rank-1 stripe vs ground truth", f"{stripe:.3f} vs {stripe_gt:.3f}"))

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
            # measured to matter -- a second training seed moves AP by 0.040 (F86),
            # comparable to the CI half-width itself. Attached to this row rather than
            # left in the caption below the table, where a viewer skimming the CI
            # number alone would miss it (hostile-expert re-read, 2026-09-23).
            rows.append((_wrap_text(
                "(scene bootstrap, ONE training seed; seed-to-seed spread "
                f"{SEED_TO_SEED_AP_SPREAD_F86:.3f} AP, F86)",
                max_chars=_TABLE_COL_LABEL_CHARS), ""))

    rows.extend(_ood_rows_for_arm(arm, ood_json_path))
    return rows


def scoreboard_figure(scores: Dict[str, Any], *, arm_name: str,
                      threshold: Optional[float], match_rule_text: str,
                      beat_cfar_json_path=DEFAULT_BEAT_CFAR_JSON,
                      beat_cfar_arm_name: Optional[str] = None,
                      raddetnet_ci_json_path=DEFAULT_RADDETNET_CI_JSON,
                      ood_json_path=DEFAULT_OOD_JSON) -> go.Figure:
    """A compact table: this frame's TP/FP/FN, cumulative hits/false alarms/FA-per-frame/
    hit-rate, and the match rule stated in words -- the numbers the hostile-expert read
    (see the module docstring) said were missing from the objectness panel entirely.

    `scores` is `score_frames`'s return value. Legible at ~2 m: >=18px table font.

    `beat_cfar_arm_name` (typically `arm_name_for_detector(det_meta)`'s return) adds
    an offline-scored block for that beat_cfar.json arm -- see `_offline_arm_rows`.
    `None` (the default) leaves the table at its base 7 rows: a caller with no mapped
    arm (e.g. a checkpoint outside the scored comparison) gets no offline block rather
    than a table of blanks.

    Why the "hit rate"/"FA per frame" comparison across screens is misleading without
    context (hostile-expert re-read, 2026-09-23): every detector's on-screen threshold
    is ITS OWN recall-0.5 operating point from beat_cfar.json, so hit rate reads
    ~0.5 for every arm by construction -- a viewer comparing 0.56 (a weak detector) >
    0.50 (CFAR) > 0.47 (the strongest detector) is reading threshold-matching noise,
    not detector quality. The subline below states the calibration explicitly and
    says what IS comparable at matched recall (false alarms); the row label repeats
    the caveat so it survives being read in isolation.
    """
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
        # threshold each one picked independently.
        subline = (
            f"threshold {thr_txt} = this detector's recall-{target_recall:g} operating "
            f"point on the {n_frames_split}-frame test split (beat_cfar.json); "
            "detectors are compared at MATCHED recall, so compare false alarms, not hits"
        )
    else:
        subline = f"threshold {thr_txt}"

    # Renamed (hostile-expert re-read, 2026-09-23, finding 1): these rows count
    # UNMATCHED detections against labels that themselves omit real objects -- ground
    # truth misses ~3.25 real strongly-scattering objects per frame inside 40 m
    # (F83), so a detector that correctly fires on every real object still racks up
    # "false alarms" here. "FA" survives only in parentheses; the row can no longer be
    # read as a true false-alarm count on its own.
    if target_recall is not None and n_frames_split is not None:
        # Quotes this run's OWN frame count (never the offline split size) -- the
        # finding this fixes is a *5-frame* hit rate of 0.50/0.47/0.56 being read
        # against each other as if they were a stable per-arm quality number.
        hit_rate_label = (f"hit rate ({n_scored} frames; recall {target_recall:g} by "
                          f"design over the {n_frames_split}-frame split)")
    else:
        hit_rate_label = f"hit rate ({n_scored} frames)"
    base_labels = ["this frame: TP", "this frame: unmatched (FP)", "this frame: FN",
                  f"cumulative hits ({n_scored}/{n_total} frames scored)",
                  "cumulative unmatched detections", "unmatched / frame (FA)",
                  hit_rate_label]
    base_values = this_frame + cum_values
    offline_rows = (_offline_arm_rows(beat_cfar_arm_name, beat_cfar_json_path,
                                      raddetnet_ci_json_path, ood_json_path)
                    if beat_cfar_arm_name else [])
    # EVERY label is pre-wrapped at the same budget the CI-caveat/OOD rows already
    # use (`_TABLE_COL_LABEL_CHARS`) -- not just the rows this module knows are long.
    # The bug this fixes (rehearsal, 2026-09-23): Plotly's Table cells word-wrap
    # automatically to fit the column's PIXEL width regardless of whether this module
    # inserted a "<br>" -- the new, longer "hit rate (...)" label (finding 4) auto-
    # wrapped to 2 lines that this function's line-count never knew about, silently
    # under-sizing the table by one row and clipping the actual last row (the OOD
    # row) off the bottom, exactly the failure mode `_TABLE_HEADER_HEIGHT`'s comment
    # already describes for the header. Pre-wrapping every label at a budget well
    # under the column's real auto-wrap threshold (empirically between 48 and 68
    # characters at this column width/font, measured via a standalone Playwright
    # render, 2026-09-23) means Plotly never NEEDS to auto-wrap, so this module's own
    # "<br>" count is always the true rendered line count.
    labels = [_wrap_text(l, max_chars=_TABLE_COL_LABEL_CHARS)
             for l in base_labels + [r[0] for r in offline_rows]]
    values = base_values + [r[1] for r in offline_rows]
    n_rows = len(labels)
    # Plotly's Table `cells.height` is a single scalar, not one-per-row, so a
    # multi-line row (the pre-wrapped labels above) forces every row to the tallest
    # row's height rather than clipping it (see the geometry comment above
    # `_TABLE_HEADER_HEIGHT`). This is a no-op (stays at `_TABLE_ROW_HEIGHT`) only
    # when every label in this particular table happens to be short.
    max_row_lines = max((max(lbl.count("<br>"), val.count("<br>")) + 1
                        for lbl, val in zip(labels, values)), default=1)
    row_height = _TABLE_ROW_HEIGHT * max_row_lines

    fig = go.Figure(data=[go.Table(
        # Widened from [220, 90] (Change 1c): the offline block's longest label
        # ("rank-1 stripe vs ground truth") and value ("+0.176 [+0.145, +0.208]")
        # need more room than the original 4-row table did.
        columnwidth=[280, 160],
        # Second column used to be an empty dark cell -- it labels the counts below it.
        header=dict(values=[arm_name, "count"],
                   fill_color="#2d3436", font=dict(color="white", size=18),
                   height=_TABLE_HEADER_HEIGHT, align="left"),
        cells=dict(
            values=[labels, values],
            fill_color=[["#f5f6fa"] * n_rows, ["#ffffff"] * n_rows],
            font=dict(size=18), height=row_height, align="left",
        ),
    )])
    # Wrapped like the match-rule annotation below (same `_wrap_text`, same 70-char
    # budget): this sentence is far wider than a two-card panel. The wrap's line
    # COUNT then drives the top margin below -- an insufficient one does not clip
    # the title, it overflows it down into the table (see `_TABLE_SUBLINE_LINE_PX`).
    subline_wrapped = _wrap_text(subline)
    n_subline_lines = subline_wrapped.count("<br>") + 1
    margin_t = _TABLE_MARGIN_T + max(0, n_subline_lines - 1) * _TABLE_SUBLINE_LINE_PX
    # The match rule, in words with its numbers (see match_rule_text()), plus one more
    # sentence (finding 1) saying the precision ceiling those "unmatched" rows above
    # cannot exceed and what that means for reading them -- both wrapped the same way
    # (same `_wrap_text`, same 70-char budget: this text is far wider than a two-card
    # panel). One annotation, not two, so a single y-position and a single dynamic
    # margin below cover both -- `_TABLE_ANNOTATION_LINE_PX` replaces the OLD fixed
    # `_TABLE_MARGIN_B=120`, which silently assumed the match-rule sentence's own
    # (then only) wrapped line count.
    ceiling_caveat = (
        "labels omit ~3 real scatterers per frame inside 40 m (precision ceiling "
        f"{PRECISION_CEILING_F83:.2f}, F-ledger); unmatched is an upper bound on "
        "false alarms"
    )
    annotation_text = f"{_wrap_text(match_rule_text)}<br>{_wrap_text(ceiling_caveat)}"
    n_annotation_lines = annotation_text.count("<br>") + 1
    margin_b = _TABLE_ANNOTATION_LINE_PX * n_annotation_lines
    # Height computed from the ACTUAL row count and the ACTUAL (possibly multi-line)
    # row height above (base 7 rows, or 7 + a variable offline block) -- see the
    # geometry comment above `_TABLE_HEADER_HEIGHT` for why this must never go back to
    # a hardcoded row count or an assumed single-line row height.
    table_height = margin_t + _TABLE_HEADER_HEIGHT + n_rows * row_height + margin_b
    fig.update_layout(
        # Threshold moved here (out of the header -- see `_TABLE_HEADER_HEIGHT`'s
        # comment) as a "<br><sup>" subline, the same pattern pipeline_runner.py uses
        # for every other panel's headline statistic.
        title=dict(text=f"Detector scoreboard<br><sup>{subline_wrapped}</sup>",
                  font=dict(size=20)),
        margin=dict(l=10, r=10, t=margin_t, b=margin_b),
        height=table_height,
    )
    fig.add_annotation(
        text=annotation_text, xref="paper", yref="paper", x=0.0, y=-0.14,
        showarrow=False, align="left", font=dict(size=15), xanchor="left", yanchor="top",
    )
    return fig


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


def stored_pr_figure(beat_cfar_json_path=DEFAULT_BEAT_CFAR_JSON, *,
                     highlight_arm: Optional[str] = None,
                     raddetnet_ci_json_path=DEFAULT_RADDETNET_CI_JSON) -> go.Figure:
    """The offline-scored PR curve for every arm in `beat_cfar.json`, so a live run's
    single operating point sits next to the whole curve it was read off.

    Falls back, PER ARM, to plotting just its recall-0.5 (or whatever `target_recall`
    the JSON recorded) operating point -- with AP in the legend, and a banner saying so
    -- for any arm that has no stored `pr_curve`. The JSON this was built against
    (2026-09-22 `e2e/ml/runs/beat_cfar.json`) has a full curve for every arm, so this is
    a documented fallback, not the observed case; re-verify against a fresh
    `beat_cfar.json` before assuming it never fires.

    `highlight_arm`'s legend entry also carries its bootstrap AP-delta-vs-CFAR 95% CI
    from `raddetnet_ci_json_path`, when that file has a row for it (see
    `_raddetnet_ci_for_arm`) -- omitted, never invented, otherwise.

    Raises if the JSON is missing `arms`, or `arms` is empty -- there is nothing
    invented in place of a genuinely absent scoring artifact.
    """
    data = _load_beat_cfar(beat_cfar_json_path)
    arms = data["arms"]
    if not arms:
        raise ValueError(f"{beat_cfar_json_path}: 'arms' is empty -- nothing to plot")

    manifest = data.get("manifest", "")
    corpus_name = Path(manifest).parent.name if manifest else "?"
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
    for a in arms:
        name = a.get("name", "?")
        # Display only -- e.g. the stored "null (random-in-GT-box)" reads as "random
        # INSIDE the ground-truth boxes" (cheating); see `_ARM_DISPLAY_NAMES` for the
        # actual definition. `name` (the stored JSON value) is still what every lookup
        # below (bold/CI matching) keys on.
        disp_name = _display_arm_name(name)
        bold = highlight_arm is not None and name == highlight_arm
        ap = a.get("AP", float("nan"))
        pr = a.get("pr_curve")
        if pr and pr.get("recall") and pr.get("precision") is not None:
            x, y = _downsample(pr["recall"], pr["precision"])
            ci = _raddetnet_ci_for_arm(name, raddetnet_ci_json_path) if bold else None
            if ci is not None:
                trace_name = (f"{disp_name} AP {ap:.3f}, {ci['delta_AP']:+.3f} vs CFAR "
                             f"[{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]")
            else:
                trace_name = f"{disp_name} (AP={ap:.3f})"
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines", name=trace_name,
                line=dict(width=5 if bold else 2),
            ))
        else:
            op = a.get("operating_point") or {}
            if op.get("reached"):
                tp, fp = op.get("tp"), op.get("fp")
                prec = (tp / (tp + fp)) if (tp is not None and fp is not None
                                            and (tp + fp) > 0) else None
                fig.add_trace(go.Scatter(
                    x=[op["recall_achieved"]], y=[prec], mode="markers",
                    marker=dict(size=18 if bold else 11, symbol="star"),
                    name=f"{disp_name} (AP={ap:.3f}, recall-{op['target_recall']:g} pt only)",
                ))
                fallback_arms.append(disp_name)

    fig.update_xaxes(title=dict(text="recall", font=dict(size=16)), range=[0, 1])
    fig.update_yaxes(title=dict(text="precision", font=dict(size=16)), range=[0, 1])
    fig.update_layout(
        # Short enough to fit a two-card (~600 px) panel -- the old single-line
        # "scored offline over the {n} test frames of {corpus} (beat_cfar.json)" ran
        # off the card edge as "... (b..." (rehearsal, 2026-09-23); the source file
        # name moves to a subline, like every other panel's qualifier text. The
        # in-distribution/seed qualifier (owner-volunteered, 2026-09-23 re-read) says
        # what this curve does and does NOT generalize to: held-out SCENES of the
        # training corpus, not a held-out corpus, and one training seed per curve --
        # neither varies run to run the way the CI band on the highlighted arm might
        # suggest.
        title=dict(
            text=f"scored offline: {n_frames} test frames, {corpus_name}"
                 "<br><sup>beat_cfar.json; in-distribution: held-out scenes of the "
                 "training corpus; one training seed per curve</sup>",
            font=dict(size=18),
        ),
        legend=dict(font=dict(size=15)),
        font=dict(size=16),
        margin=dict(l=50, r=20, t=60, b=40),
        height=440,
    )
    if fallback_arms:
        fig.add_annotation(
            text=("no stored PR curve for: " + ", ".join(fallback_arms) +
                  " -- showing their recall-0.5 operating point only"),
            xref="paper", yref="paper", x=0.5, y=1.10, showarrow=False,
            font=dict(size=13, color="#eb3b5a"),
        )
    return fig
