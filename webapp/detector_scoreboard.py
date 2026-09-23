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
(("cfar_detection", ...), ("ml_detection", ...))` loop, ~1162-1215):

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


#: Table geometry (px): header + 7 data rows must fit inside the domain the figure's
#: own `height`/`margin` leaves for the table trace, or the LAST rows get silently cut
#: off by the renderer -- not resized, not scrolled (rehearsal, 2026-09-23: a long
#: header ("CA-CFAR (guard 2, train 6) -- threshold 0.66") wrapped to two lines inside
#: its declared single-line height, stealing room from the bottom of the table and
#: clipping the "hit rate" row -- the CFAR and neural-detector arms otherwise run the
#: same code and must show identical rows). Threshold moved out of the header (into
#: the title, below) specifically so the header text stays short enough not to wrap;
#: the extra header/margin slack below is a second, independent guard for arm names
#: this module does not control the length of (e.g. an ML checkpoint's directory name).
_TABLE_N_ROWS = 7
_TABLE_HEADER_HEIGHT = 40
_TABLE_ROW_HEIGHT = 30
_TABLE_MARGIN_T = 50
#: Room for the (possibly multi-line, see `_wrap_text`) match-rule annotation below
#: the table.
_TABLE_MARGIN_B = 120
_TABLE_HEIGHT = (_TABLE_MARGIN_T + _TABLE_HEADER_HEIGHT
                + _TABLE_N_ROWS * _TABLE_ROW_HEIGHT + _TABLE_MARGIN_B)


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


def scoreboard_figure(scores: Dict[str, Any], *, arm_name: str,
                      threshold: Optional[float], match_rule_text: str) -> go.Figure:
    """A compact table: this frame's TP/FP/FN, cumulative hits/false alarms/FA-per-frame/
    hit-rate, and the match rule stated in words -- the numbers the hostile-expert read
    (see the module docstring) said were missing from the objectness panel entirely.

    `scores` is `score_frames`'s return value. Legible at ~2 m: >=18px table font, 7
    numbers total (<=8, per spec).
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

    fig = go.Figure(data=[go.Table(
        columnwidth=[220, 90],
        # Second column used to be an empty dark cell -- it labels the counts below it.
        header=dict(values=[arm_name, "count"],
                   fill_color="#2d3436", font=dict(color="white", size=18),
                   height=_TABLE_HEADER_HEIGHT, align="left"),
        cells=dict(
            values=[
                ["this frame: TP", "this frame: FP", "this frame: FN",
                 f"cumulative hits ({n_scored}/{n_total} frames scored)",
                 "cumulative false alarms", "FA / frame", "hit rate"],
                this_frame + cum_values,
            ],
            fill_color=[["#f5f6fa"] * _TABLE_N_ROWS, ["#ffffff"] * _TABLE_N_ROWS],
            font=dict(size=18), height=_TABLE_ROW_HEIGHT, align="left",
        ),
    )])
    fig.update_layout(
        # Threshold moved here (out of the header -- see `_TABLE_HEADER_HEIGHT`'s
        # comment) as a "<br><sup>" subline, the same pattern pipeline_runner.py uses
        # for every other panel's headline statistic.
        title=dict(text=f"Detector scoreboard<br><sup>threshold {thr_txt}</sup>",
                  font=dict(size=20)),
        margin=dict(l=10, r=10, t=_TABLE_MARGIN_T, b=_TABLE_MARGIN_B),
        height=_TABLE_HEIGHT,
    )
    # The match rule, in words with its numbers -- see match_rule_text(). Below the
    # table rather than in it: it is a sentence, not one of the counted numbers.
    # Wrapped (see `_wrap_text`): unwrapped, this sentence is far wider than a
    # two-card (~600 px) panel and got clipped at the card edge.
    fig.add_annotation(
        text=_wrap_text(match_rule_text), xref="paper", yref="paper", x=0.0, y=-0.14,
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


def stored_pr_figure(beat_cfar_json_path=DEFAULT_BEAT_CFAR_JSON, *,
                     highlight_arm: Optional[str] = None) -> go.Figure:
    """The offline-scored PR curve for every arm in `beat_cfar.json`, so a live run's
    single operating point sits next to the whole curve it was read off.

    Falls back, PER ARM, to plotting just its recall-0.5 (or whatever `target_recall`
    the JSON recorded) operating point -- with AP in the legend, and a banner saying so
    -- for any arm that has no stored `pr_curve`. The JSON this was built against
    (2026-09-22 `e2e/ml/runs/beat_cfar.json`) has a full curve for every arm, so this is
    a documented fallback, not the observed case; re-verify against a fresh
    `beat_cfar.json` before assuming it never fires.

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
        bold = highlight_arm is not None and name == highlight_arm
        ap = a.get("AP", float("nan"))
        pr = a.get("pr_curve")
        if pr and pr.get("recall") and pr.get("precision") is not None:
            x, y = _downsample(pr["recall"], pr["precision"])
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines", name=f"{name} (AP={ap:.3f})",
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
                    name=f"{name} (AP={ap:.3f}, recall-{op['target_recall']:g} pt only)",
                ))
                fallback_arms.append(name)

    fig.update_xaxes(title=dict(text="recall", font=dict(size=16)), range=[0, 1])
    fig.update_yaxes(title=dict(text="precision", font=dict(size=16)), range=[0, 1])
    fig.update_layout(
        # Short enough to fit a two-card (~600 px) panel -- the old single-line
        # "scored offline over the {n} test frames of {corpus} (beat_cfar.json)" ran
        # off the card edge as "... (b..." (rehearsal, 2026-09-23); the source file
        # name moves to a subline, like every other panel's qualifier text.
        title=dict(
            text=f"scored offline: {n_frames} test frames, {corpus_name}"
                 "<br><sup>beat_cfar.json</sup>",
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
