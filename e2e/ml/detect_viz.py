"""
Detection visualization: what the trained detectors actually detect, on one frame.

The rest of `e2e.ml` can SCORE a checkpoint (`e2e.ml.metrics.evaluate_dataset`) but
cannot DRAW a detection -- there was no seam that decoded a single frame's predictions
into a picture. This module is that seam, built entirely out of existing pieces so a
figure can never disagree with a reported AP/AR number:

* `e2e.ml.train.load_model_for_eval` reloads a checkpoint (FFTRadNet or SSMRadNet)
  exactly as `train.evaluate`/`afe_sweep` do.
* `e2e.ml.labels.decode_detections` is the SAME decode call `e2e.ml.metrics.evaluate_frame`
  uses (peak-pick + greedy NMS + regression-channel sub-cell refinement) -- run here at
  the identical `(grid, pred_map, threshold)`, so a marker on the picture is a marker the
  metric would also count.
* `e2e.ml.baseline.classical_detection_map` supplies the classical CFAR detector's own
  `[3, n_range, n_azimuth]` map, decoded through the same `decode_detections` call.
* `e2e.ml.dataset.RadarFrameDataset` supplies the network input and `.targets(i)` ground
  truth; raw ADC (needed for the classical baseline and the plotted background power map)
  is read directly off disk via `e2e.ml.storage.read_payload` -- the same codec-detection
  `RadarFrameDataset._load_raw` uses internally, not `baseline.score_manifest`'s stricter
  "must already be int16-quantized" assumption (an analytic-fallback, CODEC_RAW corpus
  has no `adc_code_re` key at all).
* `e2e.render_scene.range_azimuth_map` supplies the background range-azimuth power
  (dB) image (angle-FFT then non-coherent Doppler collapse -- the standard recipe this
  package already has one copy of).
* `e2e.viz.imshow_ra` OWNS the range-azimuth orientation/transpose convention (see its
  module docstring: this exact transpose bug has been reintroduced at least 3 times in
  this project's history) -- every RA panel here is drawn through it, never through a
  raw `ax.imshow`.

The single-frame background is display-tapered (`azimuth_window="hann"`, on by default
-- see `frame_background_ra`'s docstring) and the range axis is cropped to a little past
the farthest target/detection actually drawn (`_crop_range_m`); neither ever touches a
detector's own decode, only the picture. `render_perclass_bar_chart` is the one figure
in this module that is NOT single-frame: a corpus-level AP/AR-per-class bar chart built
straight from `eval_perclass.py`-style metrics JSON.

CLI
---
    python -m e2e.ml.detect_viz --manifest <manifest.json> --checkpoint <best.pt> \\
        --split val --frame 12 --out frame12.png
    python -m e2e.ml.detect_viz --manifest <manifest.json> --split val --select median \\
        --checkpoint <best.pt> --out median_frame.png
    python -m e2e.ml.detect_viz --manifest <manifest.json> --split val --select strong \\
        --compare --fftradnet-checkpoint <fft.pt> --ssmradnet-checkpoint <ssm.pt> \\
        --out compare_strong.png

See `build_arg_parser` for the full flag set (`--threshold`, `--ssm-chunk-size`, `--device`,
`--azimuth-window`, `--db-span`).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # noqa: E402 -- must precede pyplot import; headless/CI-safe

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from e2e.ml import storage  # noqa: E402
from e2e.ml.labels import LabelGrid, decode_detections  # noqa: E402
from e2e.ml.metrics import Detection, Target, evaluate_frame  # noqa: E402
from e2e.viz import imshow_ra  # noqa: E402

# GT marker style: matches e2e.render_scene's radar-view panel convention (square
# vehicle / circle pedestrian) so a reader who has seen that GIF recognizes the markers.
_GT_STYLE = {
    "vehicle": dict(marker="s", markeredgecolor="#2d98da"),
    "pedestrian": dict(marker="o", markeredgecolor="#f7b731"),
}
# Detection marker: a bright green '+' -- deliberately distinct in both shape and color
# from either GT marker (open square/circle) and from the inferno background colormap,
# so GT and detections never visually merge even where they coincide.
_DET_COLOR = "#20e070"


# --------------------------------------------------------------------------------
# Decoding: one frame, one detector -> (detections, targets), the SAME decode path
# e2e.ml.metrics scores through.
# --------------------------------------------------------------------------------
@dataclasses.dataclass
class FrameDetections:
    """Decoded detections + ground truth for one frame, from ONE detector.

    `detections`/`targets` are exactly `e2e.ml.metrics.Detection`/`Target` tuples
    (`(range_m, sin_azimuth, score, surface_range_m)` /
    `(range_m, sin_azimuth, object_class, surface_range_m)`), decoded
    through the SAME `e2e.ml.labels.decode_detections` call `evaluate_frame`/
    `evaluate_dataset` use at the same `(grid, threshold)` -- so this object's contents
    cannot disagree with a reported AP/AR number for the same inputs.
    """

    model_name: str
    frame_idx: int
    split: str
    threshold: float
    grid: LabelGrid
    detections: List[Detection]
    targets: List[Target]
    pred_map: torch.Tensor  # [3, n_range, n_azimuth] cpu float32; kept for reuse/debugging


def _grid_from_manifest(manifest: Dict) -> LabelGrid:
    g = manifest["grid"]
    return LabelGrid(n_range=int(g["n_range"]), n_azimuth=int(g["n_azimuth"]),
                     max_range_m=float(g["max_range_m"]))


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _frame_adc(ds, idx: int) -> torch.Tensor:
    """Raw complex64 ADC `[n_rx, n_chirps, n_samples]` for `ds`'s frame `idx`.

    Reads the npz directly (bypassing `RadarFrameDataset.__getitem__`, which only ever
    returns the DERIVED network input, never the raw cube) through
    `e2e.ml.storage.read_payload` -- the same codec dispatch `RadarFrameDataset._load_raw`
    uses, so this works on BOTH a chain-generated (CODEC_INT16) and an analytic-fallback
    (CODEC_RAW) corpus, unlike `e2e.ml.baseline.score_manifest`'s stricter
    "adc_code_re must already exist" check.
    """
    path = ds.dataset_dir / ds.files[idx]
    with np.load(path) as data:
        meta = json.loads(str(data["meta"].item()))
        if "adc" not in data and "adc_code_re" not in data:
            raise ValueError(
                f"{ds.files[idx]!r} has no raw ADC on disk -- the classical baseline and "
                "the background range-azimuth map both need it (manifest_version 1 "
                "corpus, or a v2 corpus that only ever wrote 'input'; regenerate it)"
            )
        arr = storage.read_payload(data, meta, "adc")
    return torch.as_tensor(arr, dtype=torch.complex64)


def decode_model_frame(manifest_path, checkpoint_path, split: str, frame_idx: int, *,
                       threshold: float = 0.5, device=None, ssm_chunk_size=None) -> FrameDetections:
    """Run a trained checkpoint (FFTRadNet or SSMRadNet) on ONE frame and decode it.

    `e2e.ml.train.load_model_for_eval(manifest_path, checkpoint_path, ...)` reloads the
    model exactly as `train.evaluate` does (same `input_format` resolution); the frame's
    input tensor comes from `e2e.ml.dataset.RadarFrameDataset(manifest_path, split=split,
    input_format=input_format)[frame_idx]`, and ground truth from that dataset's own
    `.targets(frame_idx)`.
    """
    from e2e.ml.dataset import RadarFrameDataset
    from e2e.ml.train import load_model_for_eval

    device = device if device is not None else _default_device()
    model, _manifest, grid, input_format = load_model_for_eval(
        manifest_path, checkpoint_path, device=device, ssm_chunk_size=ssm_chunk_size)
    model_name = type(model).__name__.lower()

    ds = RadarFrameDataset(manifest_path, split=split, input_format=input_format)
    x, _y = ds[frame_idx]
    model.eval()
    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(device))["detection"][0].float().cpu()

    detections = decode_detections(grid, pred, threshold=threshold)
    targets = ds.targets(frame_idx)

    return FrameDetections(model_name=model_name, frame_idx=frame_idx, split=split,
                           threshold=threshold, grid=grid, detections=detections,
                           targets=targets, pred_map=pred)


def decode_classical_frame(manifest_path, split: str, frame_idx: int, *,
                           threshold: float = 0.5, device=None, **cfar_kwargs) -> FrameDetections:
    """Classical CFAR baseline detections for ONE frame.

    `e2e.ml.baseline.classical_detection_map` produces the map, decoded through the
    SAME `decode_detections` call the model path above uses -- so the comparison figure
    is genuinely apples-to-apples, not "a CFAR heuristic" vs. "the metric's own decode".
    `**cfar_kwargs` forwards to `classical_detection_map` (e.g. `guard`/`train`/`min_db`/
    `max_db`, see `e2e.ml.baseline.cfar_objectness`).
    """
    from e2e.ml.baseline import classical_detection_map
    from e2e.ml.dataset import RadarFrameDataset
    from e2e.radar_config import RadarConfig

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    cfg = RadarConfig.from_dict(manifest["config"])
    grid = _grid_from_manifest(manifest)

    ds = RadarFrameDataset(manifest_path, split=split)
    adc = _frame_adc(ds, frame_idx)
    if device is not None:
        adc = adc.to(device)

    pred = classical_detection_map(cfg, adc, grid, **cfar_kwargs).cpu()
    detections = decode_detections(grid, pred, threshold=threshold)
    targets = ds.targets(frame_idx)

    return FrameDetections(model_name="cfar_baseline", frame_idx=frame_idx, split=split,
                           threshold=threshold, grid=grid, detections=detections,
                           targets=targets, pred_map=pred)


def frame_background_ra(manifest_path, split: str, frame_idx: int, *, n_angle_fft=None,
                        azimuth_window: Optional[str] = "hann"):
    """`(ra_db [n_angle, n_range], sin_az_axis, range_axis_m, cfg)` for ONE frame.

    Reuses `e2e.render_scene.range_azimuth_map` (angle-FFT then non-coherent Doppler
    collapse -- see that function's docstring) on the frame's own raw ADC, so the plotted
    background is the actual synthesized signal for this exact frame, not a re-derivation
    of the recipe.

    `azimuth_window` (default `"hann"`) tapers the 12-element virtual aperture before
    the angle FFT -- DISPLAY ONLY. With the rectangular window (`azimuth_window=None`)
    the first sidelobe is only ~13 dB down and decays slowly, so any strong nearby
    target smears a bright ridge across every azimuth bin at its range (measured on
    this corpus: a Hann taper drops that ridge by roughly 17 dB). Pass `None` to see
    exactly what the historical, untapered backdrop looked like.

    CAVEAT (non-negotiable to state wherever this backdrop is drawn together with
    classical-CFAR detections): `e2e.ml.baseline.classical_detection_map` thresholds
    its OWN, always-unwindowed power map -- it never sees this tapered backdrop. So a
    Hann-tapered background is no longer pixel-identical to what CFAR thresholded;
    only the (range, sin-azimuth) *locations* of CFAR's dots are meaningful against
    this picture, not a visual read of "how far above the backdrop's local level" a
    detection sits. `render_comparison_figure` (which draws CFAR) states this in its
    subtitle whenever `azimuth_window` is not `None`.
    """
    from e2e.ml.dataset import RadarFrameDataset
    from e2e.radar_config import RadarConfig
    from e2e.render_scene import range_azimuth_map

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    cfg = RadarConfig.from_dict(manifest["config"])

    ds = RadarFrameDataset(manifest_path, split=split)
    adc = _frame_adc(ds, frame_idx)

    ra_db, sin_az_axis = range_azimuth_map(cfg, adc, n_angle_fft=n_angle_fft,
                                           azimuth_window=azimuth_window)
    range_axis_m = np.arange(ra_db.shape[1]) * float(cfg.range_resolution_m)
    return ra_db, sin_az_axis, range_axis_m, cfg


# --------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------
# Backdrop dB span: measured on 30 randomly sampled rt_kenney_d2_v1 val frames with
# `azimuth_window="hann"` (see this module's report for the measurement). Per-frame
# noise floor (1st percentile, peak-referenced) ranged -47 to -56 dB and the darkest
# single pixel bottomed out -51 to -60 dB, while every ground-truth target's local-max
# cell sat between -3.8 and -23.6 dB. A -50 dB span therefore keeps every observed
# target comfortably inside the color scale (>=26 dB of headroom below the weakest
# one seen) while clipping off the flat, uninformative receiver-noise floor below it.
_DEFAULT_DB_SPAN = 50.0


def _crop_range_m(targets, detections, full_max_range_m: float, *,
                  margin_frac: float = 0.15, floor_m: float = 20.0) -> float:
    """Range-axis crop: a little past the farthest thing actually drawn on the panel.

    Cropping to only the farthest GROUND TRUTH target would hide a detection sitting
    beyond it (misrepresenting the detector's false-positive behavior, which the
    task's honesty rules protect via the on-figure detection-count annotation) -- so
    the farthest of `targets` OR `detections` sets the crop, padded by `margin_frac`
    and floored at `floor_m` so a near-empty frame doesn't crop to an illegibly tight
    range. Never exceeds `full_max_range_m` (nothing to crop if the scene already
    fills the swath).
    """
    ranges = [float(r) for r, *_ in targets] + [float(r) for r, *_ in detections]
    if not ranges:
        return float(full_max_range_m)
    cropped = max(ranges) * (1.0 + margin_frac)
    cropped = max(cropped, floor_m)
    return min(cropped, float(full_max_range_m))


def plot_frame_detections(ax, ra_db, sin_az_axis, range_axis_m, targets, detections, *,
                          threshold: float, title: str, max_range_m: Optional[float] = None,
                          full_max_range_m: Optional[float] = None,
                          vmin: float = -_DEFAULT_DB_SPAN, vmax: float = 0.0,
                          note: Optional[str] = None) -> None:
    """Draw one range-azimuth panel on `ax`: background power (dB) + GT + detections.

    The background goes through `e2e.viz.imshow_ra` (owns the RA-map orientation/
    transpose convention -- see that module's docstring for why this must never be a
    raw `ax.imshow` call). GT `targets` are open squares (vehicle) / circles
    (pedestrian); `detections` are green '+' markers, sized (mildly) by score. The
    score `threshold` used to produce `detections` is stamped into the title, per the
    task's non-negotiable honesty requirement -- a reader must never have to guess
    what operating point a picture represents.

    `max_range_m` sets the y-axis limit (default: the full `range_axis_m` swath); when
    it is given AND is smaller than `full_max_range_m` (the un-cropped swath, for
    labeling only -- pass it whenever the caller cropped the axis), the range-axis
    label states the crop so a reader never mistakes a cropped panel for the full
    unambiguous range. `note`, if given, is appended to the title verbatim (used for
    the tapered-backdrop-vs-unwindowed-CFAR caveat).
    """
    imshow_ra(ax, ra_db, sin_az_axis, range_axis_m, cmap="inferno", vmin=vmin, vmax=vmax)

    seen_labels = set()
    for tgt in targets:
        # Tuples are (range_m, sin_azimuth, object_class[, surface_range_m]) -- index
        # rather than unpack, so the optional surface element does not break this.
        r, sin_az, cls = tgt[0], tgt[1], tgt[2]
        style = _GT_STYLE.get(cls, dict(marker="x", markeredgecolor="white"))
        label = f"GT {cls}" if cls not in seen_labels else None
        seen_labels.add(cls)
        ax.plot(sin_az, r, markersize=9, markerfacecolor="none", markeredgewidth=1.6,
                linestyle="none", label=label, **style)

    det_label = f"detection (score>={threshold:.2f})"
    for det in detections:
        r, sin_az, score = det[0], det[1], det[2]
        ax.plot(sin_az, r, marker="+", markersize=10 + 6 * float(score), color=_DET_COLOR,
                markeredgewidth=1.8, linestyle="none",
                label=det_label if det_label not in seen_labels else None)
        seen_labels.add(det_label)

    ylim_max = float(max_range_m) if max_range_m is not None else float(range_axis_m[-1])
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0.0, ylim_max)
    ax.set_xlabel("sin(azimuth)")
    if full_max_range_m is not None and ylim_max < float(full_max_range_m) - 1e-6:
        ax.set_ylabel(f"range (m)  [cropped to 0-{ylim_max:.0f} of {float(full_max_range_m):.0f} m max]")
    else:
        ax.set_ylabel("range (m)")
    title_text = f"{title}\nthreshold={threshold:.2f}  " \
                f"({len(detections)} detections, {len(targets)} GT)"
    if note:
        title_text += f"\n{note}"
    ax.set_title(title_text, fontsize=9)
    if seen_labels:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.7)


def render_detection_figure(manifest_path, checkpoint_path, split: str, frame_idx: int,
                            out_path, *, threshold: float = 0.5, device=None,
                            ssm_chunk_size=None, n_angle_fft=None, dpi: int = 120,
                            azimuth_window: Optional[str] = "hann",
                            db_span: float = _DEFAULT_DB_SPAN) -> Path:
    """One-panel figure: ground truth + one checkpoint's detections over one frame's RA map.

    `azimuth_window`/`db_span` control the display backdrop only (see
    `frame_background_ra`/`plot_frame_detections`); they never touch what the
    checkpoint itself saw or decoded. The range axis is cropped to a little past the
    farthest thing actually drawn (`_crop_range_m`), with the crop stated in the axis
    label.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fd = decode_model_frame(manifest_path, checkpoint_path, split, frame_idx,
                            threshold=threshold, device=device, ssm_chunk_size=ssm_chunk_size)
    ra_db, sin_az_axis, range_axis_m, _cfg = frame_background_ra(
        manifest_path, split, frame_idx, n_angle_fft=n_angle_fft, azimuth_window=azimuth_window)

    full_max_range_m = fd.grid.max_range_m
    crop_m = _crop_range_m(fd.targets, fd.detections, full_max_range_m)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=dpi)
    plot_frame_detections(ax, ra_db, sin_az_axis, range_axis_m, fd.targets, fd.detections,
                          threshold=threshold, title=f"{fd.model_name}  {split} frame {frame_idx}",
                          max_range_m=crop_m, full_max_range_m=full_max_range_m,
                          vmin=-float(db_span))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


#: Stamped on every comparison figure. The numbers a viewer would otherwise assume:
#: "false alarms" here include deliberately-unlabelled clutter objects a correct
#: detector SHOULD fire on, and the map is overwhelmingly ground-truth-free by
#: construction -- so a dot count is not a false-alarm rate (B2 review, 2026-08-25).
_PROTOCOL_FOOTER = ("FA counts include deliberately-unlabelled clutter (~2.9/frame); "
                    "the map is ~83% GT-free by construction -- dot counts are not a "
                    "false-alarm rate and are not comparable across benchmarks")


def operating_points_from_compare(compare_json_path) -> Dict[str, float]:
    """`{arm name: score threshold}` at the matched-recall operating point, read from a
    `compare_detectors` result JSON.

    This is what makes a three-panel overlay honest: each arm's scores live on its own
    scale, so drawing all three at one threshold shows whichever arm happens to be
    calibrated near it and buries the others. On the 2026-08-26 corpus the arms' own
    operating points were 0.657 (classical) against ~0.25 (both learned) -- a uniform
    0.25 inverted the ranking the score table reports, which is the defect this
    function exists to remove (B4, `notes/B4_FIGURE_PROTOCOL.md`).
    """
    payload = json.loads(Path(compare_json_path).read_text())
    points: Dict[str, float] = {}
    for arm in payload.get("arms", []):
        op = arm.get("operating_point") or {}
        if op.get("score_threshold") is not None:
            points[str(arm["name"])] = float(op["score_threshold"])
    return points


def render_comparison_figure(manifest_path, fftradnet_checkpoint, ssmradnet_checkpoint,
                             split: str, frame_idx: int, out_path, *, threshold: float = 0.5,
                             thresholds: Optional[Mapping[str, float]] = None,
                             device=None, ssm_chunk_size=None, n_angle_fft=None,
                             dpi: int = 120, azimuth_window: Optional[str] = "hann",
                             db_span: float = _DEFAULT_DB_SPAN) -> Path:
    """Three side-by-side panels on the IDENTICAL frame: classical CFAR | FFTRadNet |
    SSMRadNet -- same background RA map, same ground truth, so the three approaches
    are visually comparable on identical data.

    `thresholds` maps panel title -> score threshold (get it from
    `operating_points_from_compare`), giving each arm its OWN operating point; the
    protocol used is stamped on the figure either way, so a uniform-threshold picture
    can never travel without saying that it is one. Prefer per-arm points for anything
    shipped: see `operating_points_from_compare`.

    `azimuth_window`/`db_span` control the display backdrop only; the range axis is
    cropped to a little past the farthest target/detection across ALL THREE panels
    (so the shared y-axis stays a fair comparison, not silently tighter for whichever
    detector happened to fire closest-in), with the crop stated in the axis label.
    When `azimuth_window` is not `None` the CFAR panel's tapered-backdrop-vs-unwindowed-
    threshold caveat (see `frame_background_ra`) is stamped into the figure subtitle.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ra_db, sin_az_axis, range_axis_m, _cfg = frame_background_ra(
        manifest_path, split, frame_idx, n_angle_fft=n_angle_fft, azimuth_window=azimuth_window)

    titles = ("classical CFAR", "FFTRadNet", "SSMRadNet")
    thr = {t: float(threshold) for t in titles}
    matched: Dict[str, str] = {}
    if thresholds:
        # Map the compare JSON's arm names ("classical CFAR", "fftradnet") onto panel
        # titles. Longest match wins and each panel binds once: a loose substring rule
        # let a key like "radnet" bleed into BOTH learned panels, silently drawing them
        # at a threshold meant for neither.
        def _norm(s):
            return s.lower().replace(" ", "").replace("_", "").replace("-", "")

        for title in titles:
            cands = [(len(_norm(n)), n, v) for n, v in thresholds.items()
                     if _norm(n) in _norm(title) or _norm(title) in _norm(n)]
            if cands:
                _, name, value = max(cands)
                thr[title] = float(value)
                matched[title] = name

    cfar_fd = decode_classical_frame(manifest_path, split, frame_idx,
                                     threshold=thr["classical CFAR"], device=device)
    fft_fd = decode_model_frame(manifest_path, fftradnet_checkpoint, split, frame_idx,
                                threshold=thr["FFTRadNet"], device=device)
    ssm_fd = decode_model_frame(manifest_path, ssmradnet_checkpoint, split, frame_idx,
                                threshold=thr["SSMRadNet"], device=device,
                                ssm_chunk_size=ssm_chunk_size)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.5), dpi=dpi, sharey=True)
    full_max_range_m = cfar_fd.grid.max_range_m
    crop_m = max(_crop_range_m(fd.targets, fd.detections, full_max_range_m)
                for fd in (cfar_fd, fft_fd, ssm_fd))
    for ax, title, fd in zip(axes, titles, (cfar_fd, fft_fd, ssm_fd)):
        plot_frame_detections(ax, ra_db, sin_az_axis, range_axis_m, fd.targets, fd.detections,
                              threshold=thr[title], title=title, max_range_m=crop_m,
                              full_max_range_m=full_max_range_m, vmin=-float(db_span))
    # State the protocol by what each panel ACTUALLY got, never by whether the numbers
    # happen to differ: a partial mapping (one arm matched, the rest silently on the
    # default) must not be advertised as "per-arm", and three per-arm points that
    # coincide must not be reported as the --threshold default they are not.
    if len(matched) == len(titles):
        protocol = "per-arm operating points"
    elif matched:
        missing = ", ".join(t for t in titles if t not in matched)
        protocol = (f"PARTIAL: {missing} fell back to the shared threshold "
                    f"{threshold:.2f} -- NOT operating-point matched")
    else:
        protocol = (f"UNIFORM threshold {threshold:.2f} for all arms -- NOT "
                    "operating-point matched")
    subtitle = (f"{split} frame {frame_idx}  --  identical input, three detectors  "
                f"[{protocol}]\n{_PROTOCOL_FOOTER}")
    if azimuth_window is not None:
        subtitle += (f"\nbackdrop display-tapered ({azimuth_window!r}) for legibility -- "
                    "classical CFAR thresholded on its own unwindowed power map, so its "
                    "dots are not pixel-identical to what's drawn beneath them")
    fig.suptitle(subtitle, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90 if azimuth_window is not None else 0.94))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------
# Per-class AP/AR bar chart -- the capability headline (which arm wins, on which class)
# --------------------------------------------------------------------------------
# One fixed color per detector everywhere this chart is used, so a viewer who has seen
# it once recognizes the arms in a later slide without re-reading the legend.
_MODEL_COLORS = {
    "classical CFAR": "#8a8f99",
    "FFTRadNet": "#3867d6",
    "SSMRadNet": "#eb3b5a",
}


def load_perclass_metrics(path) -> Dict:
    """Load one `eval_perclass.py`-produced metrics JSON (AP/AR plus per-class
    `AP_<class>`/`AR_<class>`/`n_targets_<class>` keys) -- a thin, testable wrapper so
    callers don't hand-roll `json.loads(Path(...).read_text())` at every call site."""
    return json.loads(Path(path).read_text())


def _panel_limits(panel_values: Dict[str, List[List[float]]], *, floor: float = 0.02,
                  headroom: float = 1.15, unify_below: float = 1.5
                  ) -> Tuple[Dict[str, float], bool]:
    """Decide the y-limit for each panel, and whether they could be made equal.

    Two side-by-side panels autoscaled independently are the classic misleading chart:
    AP topping out at 0.02 beside AR topping out at 0.20 draws bars of similar HEIGHT
    for numbers that differ 10x, and nothing on the figure says so. (A vision review of
    `perclass_ap_ar_bar.png` flagged exactly this, 2026-08-16.)

    Forcing a shared axis is not automatically the fix: at a 10x spread the smaller
    panel's bars collapse to hairlines and become unreadable, which trades a misleading
    figure for an uninformative one. So the rule is conditional --

    * ratio <= `unify_below`: use ONE limit for both panels. The hazard disappears
      outright and no caveat is needed, which is always better than a caveat.
    * ratio >  `unify_below`: keep independent limits so both panels stay legible, and
      return `unified=False` so the caller is obliged to say so ON the figure.

    Returns `({panel_key: y_top}, unified)`. NaNs (a model never scored on a class) are
    ignored rather than propagating and blanking the axis.
    """
    tops: Dict[str, float] = {}
    for key, rows in panel_values.items():
        finite = [v for row in rows for v in row if v == v]  # v != v -> NaN
        tops[key] = max(floor, (max(finite) if finite else 0.0) * headroom)

    lo, hi = min(tops.values()), max(tops.values())
    unified = lo > 0.0 and (hi / lo) <= unify_below
    if unified:
        tops = {key: hi for key in tops}
    return tops, unified


def render_perclass_bar_chart(models: Sequence[Tuple[str, Dict]], out_path, *,
                              classes: Sequence[str] = ("vehicle", "pedestrian"),
                              dpi: int = 150) -> Path:
    """Grouped bar chart: AP (left panel) / AR (right panel), grouped by class, one bar
    per `models` entry -- the corpus-level capability headline (as opposed to every
    other figure in this module, which is one frame).

    `AP` is `e2e.ml.metrics`' all-points interpolated precision-recall average precision
    and `AR` is recall at that module's single stated operating point (the metrics dict
    carries the operating point itself as `AR_operating_point`/`score_threshold`). The
    panel labels say so: before 2026-08-17 both were means over absolute score
    thresholds the detectors never reached, and the right panel's old "Average Recall"
    label invited exactly the misreading that the number was recall.

    `models` is `[(display_name, metrics_dict), ...]`, each `metrics_dict` exactly what
    `report/rt_ml/kenney_d2/eval_perclass.py` writes (== `e2e.ml.baseline.score_manifest`
    / `e2e.ml.train._evaluate_split`'s return, JSON-round-tripped): `AP_<class>`,
    `AR_<class>`, `n_targets_<class>` for every name in `classes`. Missing per-class
    keys for a given model are plotted as 0 with a `nan`-safe guard (`float("nan")` if
    ABSENT entirely -- distinguishing "measured zero" from "not evaluated" -- vs. a
    silently-plotted zero bar for a model that was never scored on that class).

    Ground-truth target counts (`n_targets_<class>`, expected constant across models --
    same test split) are stamped onto the class x-tick labels so sample size is never
    left for a viewer to wonder about, per the task's non-negotiable honesty rule. Bars
    carry value labels. Fonts are sized for a back-of-room read, not a paper figure.

    The two panels are scaled against each other, not autoscaled independently: see
    `_panel_limits`. Either they end up sharing one y-limit, or they do not and the
    figure says so in the panel titles and in a caution band -- a reader must never be
    able to compare bar heights across the panels without being told the scales differ.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_models = len(models)
    width = 0.8 / max(n_models, 1)
    x = np.arange(len(classes))

    # Resolve BOTH panels' data before drawing either, so the y-limits can be chosen
    # against each other rather than autoscaled in isolation. See `_panel_limits`.
    panel_values = {
        metric_key: [[float(metrics.get(f"{metric_key}_{cls}", float("nan"))) for cls in classes]
                     for _name, metrics in models]
        for metric_key in ("AP", "AR")
    }
    tops, unified = _panel_limits(panel_values)

    fig, (ax_ap, ax_ar) = plt.subplots(1, 2, figsize=(14.0, 6.5), dpi=dpi)
    # Panel labels name the actual definitions (`e2e.ml.metrics`): AP is the area under
    # the interpolated precision-recall curve, and AR is recall at ONE stated operating
    # point -- not, as the old label "Average Recall" implied, an average over anything.
    for metric_key, ax, metric_label in (("AP", ax_ap, "Average Precision (interpolated PR)"),
                                         ("AR", ax_ar, "Recall at operating point (AR)")):
        for i, (name, _metrics) in enumerate(models):
            values = panel_values[metric_key][i]
            offset = (i - (n_models - 1) / 2.0) * width
            color = _MODEL_COLORS.get(name, f"C{i}")
            bars = ax.bar(x + offset, values, width=width * 0.92, label=name, color=color,
                          edgecolor="black", linewidth=0.6)
            labels = ["n/a" if v != v else f"{v:.3f}" for v in values]  # v != v -> NaN
            ax.bar_label(bars, labels=labels, padding=2, fontsize=11, fontweight="bold")

        tick_labels = []
        for cls in classes:
            counts = {int(m.get(f"n_targets_{cls}")) for _n, m in models
                     if f"n_targets_{cls}" in m and m.get(f"n_targets_{cls}") is not None}
            n_str = f"n={counts.pop()}" if len(counts) == 1 else (
                "/".join(str(c) for c in sorted(counts)) if counts else "n=?")
            tick_labels.append(f"{cls}\n({n_str} GT)")

        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=14)
        ax.set_ylabel(metric_label, fontsize=15)
        # When the panels do NOT share a scale, the axis range goes in the title, where
        # it is read at the same moment as the bars it governs.
        title = metric_label if unified else f"{metric_label}   [y-axis 0-{tops[metric_key]:.3g}]"
        ax.set_title(title, fontsize=17, fontweight="bold")
        ax.tick_params(axis="y", labelsize=12)
        ax.set_ylim(0.0, tops[metric_key])
        ax.grid(axis="y", alpha=0.3, linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    handles, legend_labels = ax_ap.get_legend_handles_labels()
    fig.suptitle("Per-class detection performance -- rt_kenney_d2_v1 test split", fontsize=16,
                y=0.995)
    fig.legend(handles, legend_labels, loc="upper center", ncol=len(models), fontsize=14,
              frameon=False, bbox_to_anchor=(0.5, 0.96))

    if not unified:
        # Two short lines rather than one long one: at a compressed 540p screenshare a
        # full-width single line is the first thing to become unreadable.
        ratio = max(tops.values()) / min(tops.values())
        fig.text(0.5, 0.845,
                 f"The two panels use DIFFERENT y-scales (AR axis is {ratio:.0f}x the AP axis)\n"
                 "compare the printed numbers, not the bar heights",
                 ha="center", va="center", fontsize=15, fontweight="bold", color="#8c1d1d",
                 linespacing=1.35,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#fdf0f0", edgecolor="#8c1d1d",
                           linewidth=1.4))
        rect_top = 0.79
    else:
        rect_top = 0.90
    fig.tight_layout(rect=(0, 0, 1, rect_top))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------
# Reproducible, non-hand-picked frame selection (task's honesty requirement)
# --------------------------------------------------------------------------------
def rank_frames_by_quality(manifest_path, checkpoint_path, split: str, *, threshold: float = 0.5,
                           device=None, ssm_chunk_size=None, only_with_targets: bool = True
                           ) -> List[Dict]:
    """Per-frame detection-quality ranking for ONE checkpoint over a whole split.

    Score is per-frame F1 = `2*tp / (2*tp + fp + fn)` at `threshold`, computed via
    `e2e.ml.metrics.evaluate_frame` -- the SAME per-frame scoring `evaluate_dataset`
    accumulates into the corpus-level AP/AR sweep, just not aggregated across frames
    here. `only_with_targets=True` (default) drops frames with zero ground-truth objects
    from the ranking: a frame with nothing to detect says nothing about detection
    quality either way, and would otherwise pollute a "median frame" pick with an
    empty scene.

    Returns a list of `{"frame_idx", "tp", "fp", "fn", "f1", "n_targets"}` dicts, one per
    kept frame, sorted ASCENDING by `f1` (ties broken by `frame_idx`, so the result --
    and therefore `select_frame`'s picks -- is reproducible run to run).
    """
    from e2e.ml.dataset import RadarFrameDataset
    from e2e.ml.train import load_model_for_eval

    device = device if device is not None else _default_device()
    model, _manifest, grid, input_format = load_model_for_eval(
        manifest_path, checkpoint_path, device=device, ssm_chunk_size=ssm_chunk_size)
    model.eval()

    ds = RadarFrameDataset(manifest_path, split=split, input_format=input_format)
    rows: List[Dict] = []
    with torch.no_grad():
        for i in range(len(ds)):
            targets = ds.targets(i)
            if only_with_targets and not targets:
                continue
            x, _y = ds[i]
            pred = model(x.unsqueeze(0).to(device))["detection"][0].float().cpu()
            res = evaluate_frame(pred, targets, grid, threshold=threshold)
            tp, fp, fn = res["tp"], res["fp"], res["fn"]
            denom = 2 * tp + fp + fn
            f1 = (2.0 * tp / denom) if denom > 0 else 0.0
            rows.append({"frame_idx": i, "tp": tp, "fp": fp, "fn": fn, "f1": f1,
                        "n_targets": len(targets)})

    rows.sort(key=lambda r: (r["f1"], r["frame_idx"]))
    return rows


def select_frame(ranked_rows: List[Dict], rule: str) -> int:
    """Pick one `frame_idx` out of `rank_frames_by_quality`'s (f1-ascending) output.

    `rule`: `"median"` (the middle element -- `(n-1)//2`, a deterministic single index
    even for an even-length list), `"strong"` (highest f1), or `"weak"` (lowest f1).
    Exposed as a plain function (not just a CLI branch) so a caller building the deck
    figures can print/caption exactly which rule picked which frame -- the task's
    "STATED, reproducible rule" requirement, not a silently hand-picked frame.
    """
    if not ranked_rows:
        raise ValueError("no frames to select from (empty ranking -- e.g. no targets "
                         "with >=1 ground-truth object in this split)")
    if rule == "median":
        return ranked_rows[(len(ranked_rows) - 1) // 2]["frame_idx"]
    if rule == "strong":
        return ranked_rows[-1]["frame_idx"]
    if rule == "weak":
        return ranked_rows[0]["frame_idx"]
    raise ValueError(f"unknown --select rule {rule!r}; choices: median, strong, weak")


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m e2e.ml.detect_viz",
        description="Draw ground-truth + detection overlays on a range-azimuth map for one "
                    "dataset frame -- single-checkpoint mode, or a 3-way classical-CFAR / "
                    "FFTRadNet / SSMRadNet comparison (--compare).",
    )
    p.add_argument("--manifest", default=None,
                   help="dataset manifest.json (required for every mode except --perclass)")
    p.add_argument("--split", default="val", help="dataset split (train/val/test, default val)")
    p.add_argument("--out", required=True, help="output image path (.png)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="detection score threshold (same convention as e2e.ml.metrics). "
                        "For --compare prefer --operating-points, which gives each arm "
                        "its own threshold instead of one shared number")
    p.add_argument("--operating-points", default=None, metavar="COMPARE.JSON",
                   help="--compare only: a compare_detectors result JSON. Each arm is "
                        "drawn at ITS OWN matched-recall operating point read from that "
                        "file, which is what makes the three panels comparable; without "
                        "it every panel uses --threshold and the figure says so on its "
                        "face")
    p.add_argument("--device", default=None, help="torch device (default: cuda if available)")
    p.add_argument("--ssm-chunk-size", type=int, default=None,
                   help="SSMRadNet chunked-scan size (Windows needs e.g. 128 -- see CLAUDE.md)")
    p.add_argument("--n-angle-fft", type=int, default=None,
                   help="background RA-map angle-FFT size (default: no zero-padding)")
    p.add_argument("--azimuth-window", choices=("hann", "none"), default="hann",
                   help="display-only aperture taper for the RA backdrop (default: hann, "
                        "~17 dB sidelobe suppression measured on this corpus; 'none' shows "
                        "the historical untapered/rectangular-window backdrop -- see "
                        "frame_background_ra's docstring for the CFAR caveat)")
    p.add_argument("--db-span", type=float, default=_DEFAULT_DB_SPAN,
                   help=f"backdrop color-scale span below the frame's peak, in dB "
                        f"(default: {_DEFAULT_DB_SPAN:.0f}, see _DEFAULT_DB_SPAN's comment "
                        "for the measurement that picked it)")

    p.add_argument("--perclass", action="append", default=None, metavar="NAME=METRICS.json",
                   help="corpus-level mode: draw the per-class AP/AR bar chart instead of a "
                        "frame overlay. Repeat once per model, e.g. "
                        "--perclass 'classical CFAR=d2_test_classical.json'. Lives here, in "
                        "tracked code, precisely so the figure's recipe cannot be lost with a "
                        "scratch script the way its predecessor was.")
    p.add_argument("--classes", default="vehicle,pedestrian",
                   help="--perclass mode: comma-separated class names (default vehicle,pedestrian)")

    frame_group = p.add_mutually_exclusive_group(required=False)
    frame_group.add_argument("--frame", type=int, default=None, help="explicit frame index")
    frame_group.add_argument("--select", choices=("median", "strong", "weak"), default=None,
                             help="pick a frame by ranked per-frame F1 (see "
                                  "rank_frames_by_quality/select_frame) instead of an "
                                  "explicit index; ranked against --checkpoint (single-model "
                                  "mode) or --fftradnet-checkpoint (--compare mode)")

    p.add_argument("--checkpoint", default=None, help="single-model mode: one checkpoint .pt")
    p.add_argument("--compare", action="store_true",
                   help="3-way comparison mode: classical CFAR | FFTRadNet | SSMRadNet")
    p.add_argument("--fftradnet-checkpoint", default=None,
                   help="--compare mode: FFTRadNet checkpoint .pt")
    p.add_argument("--ssmradnet-checkpoint", default=None,
                   help="--compare mode: SSMRadNet checkpoint .pt")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    device = torch.device(args.device) if args.device else None

    if args.perclass:
        models = []
        for spec in args.perclass:
            name, sep, path = spec.partition("=")
            if not sep or not name.strip() or not path.strip():
                print(f"--perclass expects NAME=METRICS.json, got {spec!r}", file=sys.stderr)
                return 2
            models.append((name.strip(), load_perclass_metrics(path.strip())))
        classes = tuple(c.strip() for c in args.classes.split(",") if c.strip())
        out = render_perclass_bar_chart(models, args.out, classes=classes)
        print(f"wrote {out}")
        return 0

    # Every other mode reads frames, so it needs a dataset and a frame to draw.
    if not args.manifest:
        print("--manifest is required (except in --perclass mode)", file=sys.stderr)
        return 2
    if args.frame is None and args.select is None:
        print("pick a frame: pass --frame INDEX or --select median|strong|weak",
              file=sys.stderr)
        return 2

    if args.compare:
        if not args.fftradnet_checkpoint or not args.ssmradnet_checkpoint:
            print("--compare needs both --fftradnet-checkpoint and --ssmradnet-checkpoint",
                  file=sys.stderr)
            return 2
        rank_checkpoint = args.fftradnet_checkpoint
    else:
        if not args.checkpoint:
            print("single-model mode needs --checkpoint (or pass --compare with the two "
                  "model checkpoints)", file=sys.stderr)
            return 2
        rank_checkpoint = args.checkpoint

    if args.frame is not None:
        frame_idx = args.frame
    else:
        rows = rank_frames_by_quality(args.manifest, rank_checkpoint, args.split,
                                      threshold=args.threshold, device=device,
                                      ssm_chunk_size=args.ssm_chunk_size)
        frame_idx = select_frame(rows, args.select)
        row = next(r for r in rows if r["frame_idx"] == frame_idx)
        print(f"--select {args.select}: frame {frame_idx}  "
              f"(f1={row['f1']:.3f}, tp={row['tp']} fp={row['fp']} fn={row['fn']}, "
              f"{row['n_targets']} GT targets)  "
              f"[ranked over {len(rows)} frames with >=1 GT target in split {args.split!r}]")

    azimuth_window = None if args.azimuth_window == "none" else args.azimuth_window

    if args.compare:
        thresholds = (operating_points_from_compare(args.operating_points)
                      if args.operating_points else None)
        if thresholds:
            print("per-arm operating points: "
                  + ", ".join(f"{k}={v:.3f}" for k, v in sorted(thresholds.items())))
        out_path = render_comparison_figure(
            args.manifest, args.fftradnet_checkpoint, args.ssmradnet_checkpoint, args.split,
            frame_idx, args.out, threshold=args.threshold, thresholds=thresholds,
            device=device, ssm_chunk_size=args.ssm_chunk_size, n_angle_fft=args.n_angle_fft,
            azimuth_window=azimuth_window, db_span=args.db_span)
    else:
        out_path = render_detection_figure(
            args.manifest, args.checkpoint, args.split, frame_idx, args.out,
            threshold=args.threshold, device=device, ssm_chunk_size=args.ssm_chunk_size,
            n_angle_fft=args.n_angle_fft, azimuth_window=azimuth_window, db_span=args.db_span)

    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
