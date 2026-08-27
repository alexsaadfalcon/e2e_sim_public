"""Tests for `e2e.ml.detect_viz` (detection overlay figures + CLI).

Kept fast: one session-scoped tiny dataset (8 frames, shrunk `n_chirps`/`n_samples`,
mirroring `test_ml_train.py`'s `tiny_manifest_path` pattern) plus one 1-epoch FFTRadNet
checkpoint and one 1-epoch SSMRadNet checkpoint, trained ONCE and reused by every test.
Everything runs on CPU (`device="cpu"` throughout) per this repo's GPU etiquette --
inference/training here must not compete with real training jobs on the shared GPUs.
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("e2e.ml.labels", reason="sibling shard e2e.ml.labels not present")
pytest.importorskip("e2e.ml.scenes", reason="sibling shard e2e.ml.scenes not present")
PIL_Image = pytest.importorskip("PIL.Image", reason="Pillow required to read written PNGs")

from e2e.ml import dataset as ml_dataset
from e2e.ml import detect_viz
from e2e.ml import train as train_mod
from e2e.ml.dataset import RadarFrameDataset
from e2e.ml.labels import LabelGrid, decode_detections
from e2e.ml.metrics import evaluate_frame, match_detections
from e2e.radar_config import PRESETS, TI_IWR1443
from e2e.ml.scenes import DIFFICULTY_TIERS

_CPU = torch.device("cpu")
TIER = sorted(DIFFICULTY_TIERS)[0]  # D0: single vehicle/frame, never an empty scene


# --------------------------------------------------------------------------------
# Fixtures: tiny corpus + tiny trained checkpoints (session-scoped, built once)
# --------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def tiny_manifest_path(tmp_path_factory):
    """Shrunk TDM config (12 chirps, 64 samples) x 8 frames, split 4/2/2 -- exactly the
    `test_ml_train.py::tiny_manifest_path` pattern, duplicated (not imported) since a
    session-scoped fixture belongs to its own module."""
    cfg = dataclasses.replace(TI_IWR1443, name="test_detect_viz_tiny", n_chirps=12, n_samples=64)
    PRESETS[cfg.name] = cfg
    try:
        out_dir = tmp_path_factory.mktemp("detect_viz_dataset")
        manifest_path = ml_dataset.generate_dataset(
            cfg.name, TIER, 8, out_dir=out_dir, seed=0, device=_CPU, splits=(0.5, 0.25, 0.25),
        )
        yield manifest_path
    finally:
        PRESETS.pop(cfg.name, None)


@pytest.fixture(scope="session")
def fftradnet_checkpoint(tiny_manifest_path, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("detect_viz_fftradnet")
    train_mod.train(tiny_manifest_path, "fftradnet", epochs=1, batch_size=2,
                    out_dir=out_dir, seed=0, device=_CPU)
    return out_dir / "best.pt"


@pytest.fixture(scope="session")
def ssmradnet_checkpoint(tiny_manifest_path, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("detect_viz_ssmradnet")
    train_mod.train(tiny_manifest_path, "ssmradnet", epochs=1, batch_size=2,
                    out_dir=out_dir, seed=0, device=_CPU)
    return out_dir / "best.pt"


# --------------------------------------------------------------------------------
# decode_model_frame / decode_classical_frame -- structure + agreement with metrics
# --------------------------------------------------------------------------------
def test_decode_model_frame_returns_expected_structure(tiny_manifest_path, fftradnet_checkpoint):
    fd = detect_viz.decode_model_frame(tiny_manifest_path, fftradnet_checkpoint, "val", 0,
                                       threshold=0.1, device=_CPU)

    assert fd.model_name == "fftradnet"
    assert fd.split == "val"
    assert fd.frame_idx == 0
    assert fd.threshold == pytest.approx(0.1)
    assert isinstance(fd.grid, LabelGrid)
    assert fd.pred_map.shape == (3, fd.grid.n_range, fd.grid.n_azimuth)

    ds = RadarFrameDataset(tiny_manifest_path, split="val")
    assert fd.targets == ds.targets(0)
    for det in fd.detections:
        assert len(det) == 4
        r, sin_az, score, surface_r = det
        assert all(isinstance(v, float) for v in (r, sin_az, score, surface_r))
        assert score > 0.1  # threshold used to decode


def test_decode_model_frame_matches_labels_decode_detections_exactly(tiny_manifest_path,
                                                                     fftradnet_checkpoint):
    """The task's hard requirement: the SAME decode path metrics.evaluate_frame uses.
    Re-run `decode_detections` on the returned `pred_map` independently and assert
    bit-identical detections -- this can only pass if `decode_model_frame` did not
    quietly re-implement its own decode logic."""
    fd = detect_viz.decode_model_frame(tiny_manifest_path, fftradnet_checkpoint, "val", 1,
                                       threshold=0.15, device=_CPU)
    expected = decode_detections(fd.grid, fd.pred_map, threshold=0.15)
    assert fd.detections == expected


def test_decode_model_frame_agrees_with_evaluate_frame_tp_fp_fn(tiny_manifest_path,
                                                                fftradnet_checkpoint):
    """Cross-check against `e2e.ml.metrics.evaluate_frame`/`match_detections`: matching
    the returned detections/targets independently must reproduce the same tp/fp/fn a
    reported AP/AR number for this frame would show."""
    fd = detect_viz.decode_model_frame(tiny_manifest_path, fftradnet_checkpoint, "test", 0,
                                       threshold=0.2, device=_CPU)
    expected = evaluate_frame(fd.pred_map, fd.targets, fd.grid, threshold=0.2)
    matches, unmatched_det, unmatched_gt = match_detections(fd.detections, fd.targets)
    assert len(matches) == expected["tp"]
    assert len(unmatched_det) == expected["fp"]
    assert len(unmatched_gt) == expected["fn"]


def test_decode_classical_frame_returns_expected_structure(tiny_manifest_path):
    fd = detect_viz.decode_classical_frame(tiny_manifest_path, "val", 0, threshold=0.3)

    assert fd.model_name == "cfar_baseline"
    assert isinstance(fd.grid, LabelGrid)
    assert fd.pred_map.shape == (3, fd.grid.n_range, fd.grid.n_azimuth)
    assert torch.all(fd.pred_map[1:] == 0.0)  # classical_detection_map: no sub-cell regression

    ds = RadarFrameDataset(tiny_manifest_path, split="val")
    assert fd.targets == ds.targets(0)
    expected = decode_detections(fd.grid, fd.pred_map, threshold=0.3)
    assert fd.detections == expected


def test_frame_adc_missing_raises_clear_error(tiny_manifest_path, tmp_path):
    """A frame npz with no raw ADC array at all (v1-style, 'input' only) must raise a
    clear error, not a bare KeyError from inside numpy."""
    ds = RadarFrameDataset(tiny_manifest_path, split="val")
    src = ds.dataset_dir / ds.files[0]
    stripped_dir = tmp_path / "stripped"
    stripped_dir.mkdir()
    with np.load(src) as z:
        meta = json.loads(str(z["meta"].item()))
        np.savez_compressed(stripped_dir / ds.files[0],
                            input=np.zeros((2, 2), dtype=np.float32),
                            labels=z["labels"], meta=np.array(json.dumps(meta)))

    class _FakeDs:
        dataset_dir = stripped_dir
        files = [ds.files[0]]

    with pytest.raises(ValueError, match="no raw ADC"):
        detect_viz._frame_adc(_FakeDs(), 0)


# --------------------------------------------------------------------------------
# frame_background_ra
# --------------------------------------------------------------------------------
def test_frame_background_ra_shapes(tiny_manifest_path):
    ra_db, sin_az_axis, range_axis_m, cfg = detect_viz.frame_background_ra(
        tiny_manifest_path, "val", 0)
    assert ra_db.ndim == 2
    assert ra_db.shape[0] == sin_az_axis.shape[0]
    assert ra_db.shape[1] == range_axis_m.shape[0]
    assert float(ra_db.max()) == pytest.approx(0.0, abs=1e-4)  # peak-normalized


def test_frame_background_ra_defaults_to_hann_and_differs_from_rectangular(tiny_manifest_path):
    """Task requirement: the display backdrop defaults to the Hann taper (not the
    historical rectangular window), and `azimuth_window=None` still works and produces
    a genuinely different map -- i.e. the parameter is actually wired through to
    `range_azimuth_map`, not silently ignored."""
    hann_db, hann_sin_az, _range_axis, _cfg = detect_viz.frame_background_ra(
        tiny_manifest_path, "val", 0)
    rect_db, rect_sin_az, _range_axis2, _cfg2 = detect_viz.frame_background_ra(
        tiny_manifest_path, "val", 0, azimuth_window=None)

    assert hann_db.shape == rect_db.shape
    np.testing.assert_array_equal(hann_sin_az, rect_sin_az)
    assert not torch.allclose(hann_db, rect_db)
    # both still peak-normalized (0 dB at the map's own peak bin)
    assert float(hann_db.max()) == pytest.approx(0.0, abs=1e-4)
    assert float(rect_db.max()) == pytest.approx(0.0, abs=1e-4)


# --------------------------------------------------------------------------------
# _crop_range_m -- range-axis cropping (Task 2: crop to the populated region, never
# azimuth, and always state the crop)
# --------------------------------------------------------------------------------
def test_crop_range_m_pads_past_farthest_target_or_detection():
    targets = [(30.0, 0.1, "vehicle")]
    detections = [(10.0, 0.0, 0.9)]
    crop = detect_viz._crop_range_m(targets, detections, full_max_range_m=100.0)
    assert 30.0 < crop < 100.0  # padded past the farthest (30 m), well short of the full swath


def test_crop_range_m_uses_farthest_detection_even_beyond_farthest_target():
    """A detection beyond the farthest GT target must not be cropped out of frame --
    that would silently hide a false positive the honesty rules require showing."""
    targets = [(10.0, 0.0, "vehicle")]
    detections = [(50.0, 0.0, 0.9)]
    crop = detect_viz._crop_range_m(targets, detections, full_max_range_m=100.0)
    assert crop > 50.0


def test_crop_range_m_never_exceeds_full_range():
    targets = [(999.0, 0.0, "vehicle")]
    crop = detect_viz._crop_range_m(targets, [], full_max_range_m=100.0)
    assert crop == pytest.approx(100.0)


def test_crop_range_m_empty_scene_shows_full_swath():
    """Nothing to crop TO if the panel has no targets or detections at all -- show the
    full swath rather than guess a floor that could visually suggest "empty out to
    here" for a frame that just has nothing on it."""
    crop = detect_viz._crop_range_m([], [], full_max_range_m=100.0, floor_m=20.0)
    assert crop == pytest.approx(100.0)


def test_crop_range_m_floor_applies_to_a_single_near_target():
    crop = detect_viz._crop_range_m([(1.0, 0.0, "vehicle")], [], full_max_range_m=100.0,
                                    floor_m=20.0)
    assert crop == pytest.approx(20.0)


# --------------------------------------------------------------------------------
# plot_frame_detections -- orientation regression (consistent with
# tests/test_ml_render.py::test_draw_radar_view_imshow_orientation_matches_extent)
# --------------------------------------------------------------------------------
def test_plot_frame_detections_imshow_orientation_matches_extent():
    import matplotlib.pyplot as plt

    n_angle, n_range = 8, 5
    angle_idx, range_idx = 2, 4  # deliberately distinct so a transpose is detectable
    ra_db = torch.full((n_angle, n_range), -40.0)
    ra_db[angle_idx, range_idx] = 0.0
    sin_az_axis = np.linspace(-1.0, 1.0, n_angle)
    range_axis_m = np.arange(n_range) * 1.0

    fig, ax = plt.subplots()
    try:
        detect_viz.plot_frame_detections(ax, ra_db, sin_az_axis, range_axis_m, [], [],
                                         threshold=0.5, title="orientation check")
        images = ax.get_images()
        assert len(images) == 1
        arr = images[0].get_array()

        assert arr.shape == (n_range, n_angle)  # [row=range, col=angle]
        hot_row, hot_col = np.unravel_index(np.argmax(arr), arr.shape)
        assert (hot_row, hot_col) == (range_idx, angle_idx)
    finally:
        plt.close(fig)


def test_plot_frame_detections_draws_gt_and_detection_markers():
    import matplotlib.pyplot as plt

    ra_db = torch.full((6, 6), -40.0)
    sin_az_axis = np.linspace(-1.0, 1.0, 6)
    range_axis_m = np.arange(6) * 1.0
    targets = [(3.0, 0.1, "vehicle"), (4.0, -0.2, "pedestrian")]
    detections = [(3.1, 0.12, 0.9)]

    fig, ax = plt.subplots()
    try:
        detect_viz.plot_frame_detections(ax, ra_db, sin_az_axis, range_axis_m, targets,
                                         detections, threshold=0.5, title="markers")
        lines = ax.get_lines()
        # 2 GT markers + 1 detection marker == 3 plotted Line2D objects.
        assert len(lines) == 3
        labels = {ln.get_label() for ln in lines if not ln.get_label().startswith("_")}
        assert labels == {"GT vehicle", "GT pedestrian", "detection (score>=0.50)"}
        assert "threshold=0.50" in ax.get_title()
    finally:
        plt.close(fig)


def test_plot_frame_detections_states_crop_in_axis_label_when_cropped():
    import matplotlib.pyplot as plt

    ra_db = torch.full((6, 6), -40.0)
    sin_az_axis = np.linspace(-1.0, 1.0, 6)
    range_axis_m = np.arange(6) * 1.0

    fig, ax = plt.subplots()
    try:
        detect_viz.plot_frame_detections(ax, ra_db, sin_az_axis, range_axis_m, [], [],
                                         threshold=0.5, title="cropped", max_range_m=3.0,
                                         full_max_range_m=6.0)
        assert "cropped" in ax.get_ylabel().lower()
        assert ax.get_ylim()[1] == pytest.approx(3.0)
    finally:
        plt.close(fig)


def test_plot_frame_detections_no_crop_note_when_not_cropped():
    import matplotlib.pyplot as plt

    ra_db = torch.full((6, 6), -40.0)
    sin_az_axis = np.linspace(-1.0, 1.0, 6)
    range_axis_m = np.arange(6) * 1.0

    fig, ax = plt.subplots()
    try:
        detect_viz.plot_frame_detections(ax, ra_db, sin_az_axis, range_axis_m, [], [],
                                         threshold=0.5, title="full", max_range_m=6.0,
                                         full_max_range_m=6.0)
        assert ax.get_ylabel() == "range (m)"
    finally:
        plt.close(fig)


def test_plot_frame_detections_note_param_appended_to_title():
    import matplotlib.pyplot as plt

    ra_db = torch.full((6, 6), -40.0)
    sin_az_axis = np.linspace(-1.0, 1.0, 6)
    range_axis_m = np.arange(6) * 1.0

    fig, ax = plt.subplots()
    try:
        detect_viz.plot_frame_detections(ax, ra_db, sin_az_axis, range_axis_m, [], [],
                                         threshold=0.5, title="t", note="a caveat sentence")
        assert "a caveat sentence" in ax.get_title()
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------------
# render_detection_figure / render_comparison_figure -- writes a non-empty file
# --------------------------------------------------------------------------------
def test_render_detection_figure_writes_nonempty_png(tiny_manifest_path, fftradnet_checkpoint,
                                                      tmp_path):
    out_path = tmp_path / "single.png"
    result = detect_viz.render_detection_figure(tiny_manifest_path, fftradnet_checkpoint,
                                                 "val", 0, out_path, threshold=0.1, device=_CPU)
    assert result == out_path
    assert out_path.exists() and out_path.stat().st_size > 0
    with PIL_Image.open(out_path) as im:
        assert im.format == "PNG"


def test_render_comparison_figure_writes_nonempty_png_with_three_panels(
        tiny_manifest_path, fftradnet_checkpoint, ssmradnet_checkpoint, tmp_path):
    out_path = tmp_path / "compare.png"
    result = detect_viz.render_comparison_figure(
        tiny_manifest_path, fftradnet_checkpoint, ssmradnet_checkpoint, "val", 0, out_path,
        threshold=0.1, device=_CPU)
    assert result == out_path
    assert out_path.exists() and out_path.stat().st_size > 0

    import matplotlib.pyplot as plt

    ra_db, sin_az_axis, range_axis_m, _cfg = detect_viz.frame_background_ra(
        tiny_manifest_path, "val", 0)
    fig, axes = plt.subplots(1, 3)
    try:
        cfar_fd = detect_viz.decode_classical_frame(tiny_manifest_path, "val", 0, threshold=0.1)
        for ax in axes:
            detect_viz.plot_frame_detections(ax, ra_db, sin_az_axis, range_axis_m,
                                             cfar_fd.targets, cfar_fd.detections,
                                             threshold=0.1, title="panel")
        assert len(fig.axes) == 3
    finally:
        plt.close(fig)


def test_render_comparison_figure_states_cfar_caveat_when_tapered(
        tiny_manifest_path, fftradnet_checkpoint, ssmradnet_checkpoint, tmp_path, monkeypatch):
    """Task's non-negotiable honesty requirement: when the backdrop is display-tapered
    (the default), the figure's own suptitle must say the classical-CFAR dots were
    thresholded on an unwindowed map; with the taper off, no such caveat applies.

    `render_comparison_figure` closes its figure internally, so this intercepts
    `plt.close` to capture the figure (and read its real suptitle text) before it's
    discarded, rather than re-deriving the expected string independently.
    """
    import matplotlib.pyplot as plt

    captured = []
    real_close = plt.close
    monkeypatch.setattr(detect_viz.plt, "close",
                        lambda fig=None: (captured.append(fig), real_close(fig)))

    detect_viz.render_comparison_figure(
        tiny_manifest_path, fftradnet_checkpoint, ssmradnet_checkpoint, "val", 0,
        tmp_path / "tapered.png", threshold=0.1, device=_CPU, azimuth_window="hann")
    tapered_suptitle = captured[-1]._suptitle.get_text()

    detect_viz.render_comparison_figure(
        tiny_manifest_path, fftradnet_checkpoint, ssmradnet_checkpoint, "val", 0,
        tmp_path / "untapered.png", threshold=0.1, device=_CPU, azimuth_window=None)
    untapered_suptitle = captured[-1]._suptitle.get_text()

    assert "unwindowed" in tapered_suptitle
    assert "unwindowed" not in untapered_suptitle


def test_render_detection_figure_creates_parent_directories(tiny_manifest_path,
                                                             fftradnet_checkpoint, tmp_path):
    out_path = tmp_path / "nested" / "dir" / "single.png"
    detect_viz.render_detection_figure(tiny_manifest_path, fftradnet_checkpoint, "val", 0,
                                       out_path, threshold=0.1, device=_CPU)
    assert out_path.exists()


def test_render_detection_figure_azimuth_window_none_still_writes_png(
        tiny_manifest_path, fftradnet_checkpoint, tmp_path):
    """The taper-off escape hatch (task requirement): passing `azimuth_window=None`
    must still produce a valid figure, not error."""
    out_path = tmp_path / "untapered.png"
    result = detect_viz.render_detection_figure(
        tiny_manifest_path, fftradnet_checkpoint, "val", 0, out_path, threshold=0.1,
        device=_CPU, azimuth_window=None)
    assert result == out_path
    assert out_path.exists() and out_path.stat().st_size > 0


def test_render_detection_figure_range_axis_cropped_to_populated_region(
        tiny_manifest_path, fftradnet_checkpoint, monkeypatch, tmp_path):
    """Task 2: the range axis must be cropped tighter than the full grid swath when
    the frame's targets/detections don't fill it (this tiny fixture's D0 scenes are
    single-vehicle, near-range) -- intercept `plt.close` the same way the CFAR-caveat
    test does, to read the axis back before the figure is discarded."""
    import matplotlib.pyplot as plt

    captured = []
    real_close = plt.close
    monkeypatch.setattr(detect_viz.plt, "close",
                        lambda fig=None: (captured.append(fig), real_close(fig)))

    fd = detect_viz.decode_model_frame(tiny_manifest_path, fftradnet_checkpoint, "val", 0,
                                       threshold=0.1, device=_CPU)
    detect_viz.render_detection_figure(tiny_manifest_path, fftradnet_checkpoint, "val", 0,
                                       tmp_path / "single.png", threshold=0.1, device=_CPU)
    ax = captured[-1].axes[0]
    assert ax.get_ylim()[1] <= fd.grid.max_range_m + 1e-6
    # azimuth must NEVER be cropped
    assert ax.get_xlim() == pytest.approx((-1.0, 1.0))


# --------------------------------------------------------------------------------
# load_perclass_metrics / render_perclass_bar_chart -- the capability headline
# --------------------------------------------------------------------------------
def test_load_perclass_metrics_round_trips_json(tmp_path):
    payload = {"AP_vehicle": 0.01, "AR_vehicle": 0.1, "n_targets_vehicle": 988}
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(payload))
    assert detect_viz.load_perclass_metrics(path) == payload


def _fake_perclass_metrics(ap_vehicle, ar_vehicle, ap_pedestrian, ar_pedestrian):
    return {
        "AP_vehicle": ap_vehicle, "AR_vehicle": ar_vehicle, "n_targets_vehicle": 988,
        "AP_pedestrian": ap_pedestrian, "AR_pedestrian": ar_pedestrian, "n_targets_pedestrian": 503,
    }


def test_render_perclass_bar_chart_writes_nonempty_png_with_two_panels(tmp_path):
    models = [
        ("classical CFAR", _fake_perclass_metrics(0.015, 0.153, 0.003, 0.143)),
        ("SSMRadNet", _fake_perclass_metrics(0.014, 0.200, 0.007, 0.196)),
    ]
    out_path = tmp_path / "bar.png"
    result = detect_viz.render_perclass_bar_chart(models, out_path)
    assert result == out_path
    assert out_path.exists() and out_path.stat().st_size > 0
    with PIL_Image.open(out_path) as im:
        assert im.format == "PNG"


def test_render_perclass_bar_chart_bar_heights_match_input_values(tmp_path, monkeypatch):
    """Read the actual `Rectangle` patch heights back off the figure (before it's
    closed) and check they equal the input AP/AR values -- not just "a PNG got
    written", but that the bars encode the real numbers."""
    import matplotlib.pyplot as plt

    captured = []
    real_close = plt.close
    monkeypatch.setattr(detect_viz.plt, "close",
                        lambda fig=None: (captured.append(fig), real_close(fig)))

    models = [
        ("classical CFAR", _fake_perclass_metrics(0.015, 0.153, 0.003, 0.143)),
        ("SSMRadNet", _fake_perclass_metrics(0.014, 0.200, 0.007, 0.196)),
    ]
    detect_viz.render_perclass_bar_chart(models, tmp_path / "bar.png")
    fig = captured[-1]
    ax_ap, ax_ar = fig.axes[0], fig.axes[1]

    ap_heights = sorted(round(p.get_height(), 4) for p in ax_ap.patches)
    assert ap_heights == sorted([0.015, 0.014, 0.003, 0.007])
    ar_heights = sorted(round(p.get_height(), 4) for p in ax_ar.patches)
    assert ar_heights == sorted([0.153, 0.200, 0.143, 0.196])


def test_render_perclass_bar_chart_annotates_target_counts(tmp_path, monkeypatch):
    """Honesty rule: the ground-truth target count per class must be visible on the
    figure (x-tick labels here), not left for a viewer to wonder about."""
    import matplotlib.pyplot as plt

    captured = []
    real_close = plt.close
    monkeypatch.setattr(detect_viz.plt, "close",
                        lambda fig=None: (captured.append(fig), real_close(fig)))

    models = [("classical CFAR", _fake_perclass_metrics(0.015, 0.153, 0.003, 0.143))]
    detect_viz.render_perclass_bar_chart(models, tmp_path / "bar.png")
    fig = captured[-1]
    tick_text = " ".join(t.get_text() for ax in fig.axes for t in ax.get_xticklabels())
    assert "988" in tick_text
    assert "503" in tick_text


def test_panel_limits_unifies_scales_when_the_spread_is_small():
    """When the two panels are within `unify_below` of each other, they get ONE limit.
    Removing the hazard outright beats annotating it."""
    tops, unified = detect_viz._panel_limits(
        {"AP": [[0.18, 0.16]], "AR": [[0.20, 0.19]]})
    assert unified
    assert tops["AP"] == tops["AR"]


def test_panel_limits_keeps_separate_scales_when_the_spread_is_large():
    """At a 10x spread a shared axis would squash the AP bars to hairlines, so the
    limits stay independent -- and `unified=False` obliges the caller to say so."""
    tops, unified = detect_viz._panel_limits(
        {"AP": [[0.015, 0.003]], "AR": [[0.200, 0.143]]})
    assert not unified
    assert tops["AR"] > tops["AP"]


def test_panel_limits_ignores_nan_from_unscored_classes():
    """A model never scored on a class contributes NaN; it must not blank the axis.

    Values are deliberately far apart so the unify branch does not fire and each
    panel's own limit is observable.
    """
    tops, unified = detect_viz._panel_limits(
        {"AP": [[float("nan"), 0.03]], "AR": [[0.40, float("nan")]]})
    assert not unified
    assert tops["AP"] == pytest.approx(0.03 * 1.15)
    assert tops["AR"] == pytest.approx(0.40 * 1.15)


def test_render_perclass_bar_chart_flags_mismatched_panel_scales(tmp_path, monkeypatch):
    """Regression for the misleading dual-axis chart a vision review caught (2026-08-16):
    `perclass_ap_ar_bar.png` put AP (0-0.02) beside AR (0-0.20) with nothing marking the
    10x difference, so bars of similar HEIGHT stood for numbers an order of magnitude
    apart. Independent autoscaling is fine ONLY if the figure admits it, so when the
    panels don't share a limit the range must appear in both titles and a caution band
    must appear on the figure."""
    import matplotlib.pyplot as plt

    captured = []
    real_close = plt.close
    monkeypatch.setattr(detect_viz.plt, "close",
                        lambda fig=None: (captured.append(fig), real_close(fig)))

    models = [
        ("classical CFAR", _fake_perclass_metrics(0.015, 0.153, 0.003, 0.143)),
        ("SSMRadNet", _fake_perclass_metrics(0.014, 0.200, 0.007, 0.196)),
    ]
    detect_viz.render_perclass_bar_chart(models, tmp_path / "bar.png")
    fig = captured[-1]
    ax_ap, ax_ar = fig.axes[0], fig.axes[1]

    # The scales really do differ here -- that is the premise of the test.
    assert ax_ap.get_ylim()[1] != pytest.approx(ax_ar.get_ylim()[1])
    for ax in (ax_ap, ax_ar):
        assert "y-axis" in ax.get_title(), "mismatched panel hides its axis range"

    figure_text = " ".join(t.get_text() for t in fig.texts)
    assert "DIFFERENT y-scales" in figure_text
    assert "not the bar heights" in figure_text


def test_render_perclass_bar_chart_omits_the_caution_when_scales_match(tmp_path, monkeypatch):
    """The converse: with comparable panels there is nothing to warn about, and a
    spurious warning would train readers to ignore the real one."""
    import matplotlib.pyplot as plt

    captured = []
    real_close = plt.close
    monkeypatch.setattr(detect_viz.plt, "close",
                        lambda fig=None: (captured.append(fig), real_close(fig)))

    models = [("model", _fake_perclass_metrics(0.18, 0.20, 0.16, 0.19))]
    detect_viz.render_perclass_bar_chart(models, tmp_path / "bar.png")
    fig = captured[-1]

    assert fig.axes[0].get_ylim() == fig.axes[1].get_ylim()
    figure_text = " ".join(t.get_text() for t in fig.texts)
    assert "DIFFERENT y-scales" not in figure_text
    for ax in fig.axes[:2]:
        assert "y-axis" not in ax.get_title()


def test_render_perclass_bar_chart_three_models_and_custom_classes(tmp_path):
    models = [
        ("classical CFAR", {"AP_vehicle": 0.01, "AR_vehicle": 0.1, "n_targets_vehicle": 5}),
        ("FFTRadNet", {"AP_vehicle": 0.02, "AR_vehicle": 0.2, "n_targets_vehicle": 5}),
        ("SSMRadNet", {"AP_vehicle": 0.03, "AR_vehicle": 0.3, "n_targets_vehicle": 5}),
    ]
    out_path = tmp_path / "three.png"
    result = detect_viz.render_perclass_bar_chart(models, out_path, classes=("vehicle",))
    assert out_path.exists() and out_path.stat().st_size > 0
    assert result == out_path


# --------------------------------------------------------------------------------
# rank_frames_by_quality / select_frame
# --------------------------------------------------------------------------------
def test_rank_frames_by_quality_sorted_ascending_and_excludes_empty_frames(
        tiny_manifest_path, fftradnet_checkpoint):
    rows = detect_viz.rank_frames_by_quality(tiny_manifest_path, fftradnet_checkpoint, "train",
                                             threshold=0.1, device=_CPU)
    ds = RadarFrameDataset(tiny_manifest_path, split="train")
    n_with_targets = sum(1 for i in range(len(ds)) if ds.targets(i))
    assert len(rows) == n_with_targets
    assert all(r["n_targets"] > 0 for r in rows)
    f1s = [r["f1"] for r in rows]
    assert f1s == sorted(f1s)
    for r in rows:
        assert 0.0 <= r["f1"] <= 1.0
        assert r["tp"] + r["fn"] == r["n_targets"]


def test_rank_frames_by_quality_only_with_targets_false_includes_everything(
        tiny_manifest_path, fftradnet_checkpoint):
    rows_all = detect_viz.rank_frames_by_quality(tiny_manifest_path, fftradnet_checkpoint,
                                                 "train", threshold=0.1, device=_CPU,
                                                 only_with_targets=False)
    ds = RadarFrameDataset(tiny_manifest_path, split="train")
    assert len(rows_all) == len(ds)


def test_select_frame_median_strong_weak():
    rows = [
        {"frame_idx": 5, "f1": 0.0}, {"frame_idx": 1, "f1": 0.2}, {"frame_idx": 3, "f1": 0.5},
        {"frame_idx": 2, "f1": 0.8}, {"frame_idx": 4, "f1": 1.0},
    ]
    assert detect_viz.select_frame(rows, "weak") == 5
    assert detect_viz.select_frame(rows, "strong") == 4
    assert detect_viz.select_frame(rows, "median") == 3  # (5-1)//2 == 2 -> rows[2]


def test_select_frame_empty_raises():
    with pytest.raises(ValueError, match="no frames"):
        detect_viz.select_frame([], "median")


def test_select_frame_unknown_rule_raises():
    with pytest.raises(ValueError, match="unknown --select rule"):
        detect_viz.select_frame([{"frame_idx": 0, "f1": 1.0}], "not_a_rule")


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------
def test_cli_help_exits_zero():
    proc = subprocess.run([sys.executable, "-m", "e2e.ml.detect_viz", "--help"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "detect_viz" in proc.stdout
    assert "--azimuth-window" in proc.stdout
    assert "--db-span" in proc.stdout


def test_cli_azimuth_window_none_and_custom_db_span(tiny_manifest_path, fftradnet_checkpoint,
                                                     tmp_path):
    out_path = tmp_path / "cli_untapered.png"
    proc = subprocess.run(
        [sys.executable, "-m", "e2e.ml.detect_viz",
         "--manifest", str(tiny_manifest_path), "--checkpoint", str(fftradnet_checkpoint),
         "--split", "val", "--frame", "0", "--out", str(out_path),
         "--threshold", "0.1", "--device", "cpu",
         "--azimuth-window", "none", "--db-span", "30"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out_path.exists() and out_path.stat().st_size > 0


def test_cli_single_mode_end_to_end_writes_png(tiny_manifest_path, fftradnet_checkpoint,
                                                tmp_path):
    out_path = tmp_path / "cli_single.png"
    proc = subprocess.run(
        [sys.executable, "-m", "e2e.ml.detect_viz",
         "--manifest", str(tiny_manifest_path), "--checkpoint", str(fftradnet_checkpoint),
         "--split", "val", "--frame", "0", "--out", str(out_path),
         "--threshold", "0.1", "--device", "cpu"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out_path.exists() and out_path.stat().st_size > 0
    assert "wrote" in proc.stdout.lower()


def test_cli_select_mode_prints_chosen_frame(tiny_manifest_path, fftradnet_checkpoint, tmp_path):
    out_path = tmp_path / "cli_select.png"
    proc = subprocess.run(
        [sys.executable, "-m", "e2e.ml.detect_viz",
         "--manifest", str(tiny_manifest_path), "--checkpoint", str(fftradnet_checkpoint),
         "--split", "train", "--select", "median", "--out", str(out_path),
         "--threshold", "0.1", "--device", "cpu"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out_path.exists() and out_path.stat().st_size > 0
    assert "--select median" in proc.stdout


def test_cli_compare_mode_writes_png(tiny_manifest_path, fftradnet_checkpoint,
                                     ssmradnet_checkpoint, tmp_path):
    out_path = tmp_path / "cli_compare.png"
    proc = subprocess.run(
        [sys.executable, "-m", "e2e.ml.detect_viz",
         "--manifest", str(tiny_manifest_path), "--compare",
         "--fftradnet-checkpoint", str(fftradnet_checkpoint),
         "--ssmradnet-checkpoint", str(ssmradnet_checkpoint),
         "--split", "val", "--frame", "0", "--out", str(out_path),
         "--threshold", "0.1", "--device", "cpu"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out_path.exists() and out_path.stat().st_size > 0


def test_cli_compare_mode_missing_checkpoint_exits_nonzero(tiny_manifest_path, tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "e2e.ml.detect_viz",
         "--manifest", str(tiny_manifest_path), "--compare",
         "--fftradnet-checkpoint", "somewhere.pt",
         "--split", "val", "--frame", "0", "--out", str(tmp_path / "x.png")],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0


def test_cli_single_mode_missing_checkpoint_exits_nonzero(tiny_manifest_path, tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "e2e.ml.detect_viz",
         "--manifest", str(tiny_manifest_path),
         "--split", "val", "--frame", "0", "--out", str(tmp_path / "x.png")],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0


def test_cli_perclass_mode_writes_the_bar_chart_without_a_manifest(tmp_path):
    """The bar chart must be regenerable from tracked code alone.

    Its predecessor was drawn by a scratch script that no longer exists, leaving a
    published figure with no reproducible recipe -- the same failure that lost the
    `tracking_refine` generator. `--perclass` takes only the metrics JSONs that the
    evaluation already writes, so no dataset, checkpoint, or GPU is involved.
    """
    metrics_path = tmp_path / "m.json"
    metrics_path.write_text(json.dumps(_fake_perclass_metrics(0.015, 0.153, 0.003, 0.143)))
    out_path = tmp_path / "bar.png"

    proc = subprocess.run(
        [sys.executable, "-m", "e2e.ml.detect_viz",
         "--perclass", f"classical CFAR={metrics_path}", "--out", str(out_path)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out_path.exists() and out_path.stat().st_size > 0


def test_cli_perclass_rejects_a_malformed_spec(tmp_path):
    """NAME=PATH with no '=' is a typo, not a filename -- fail loudly rather than
    drawing an empty chart."""
    proc = subprocess.run(
        [sys.executable, "-m", "e2e.ml.detect_viz",
         "--perclass", "just-a-path.json", "--out", str(tmp_path / "x.png")],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "NAME=METRICS.json" in proc.stderr


def test_cli_without_a_frame_selection_exits_nonzero(tiny_manifest_path, fftradnet_checkpoint,
                                                     tmp_path):
    """--frame/--select stopped being an argparse-required group when --perclass was
    added; the requirement still has to hold for every frame-drawing mode."""
    proc = subprocess.run(
        [sys.executable, "-m", "e2e.ml.detect_viz",
         "--manifest", str(tiny_manifest_path), "--checkpoint", str(fftradnet_checkpoint),
         "--split", "val", "--out", str(tmp_path / "x.png")],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "--frame" in proc.stderr


def test_cli_frame_and_select_are_mutually_exclusive(tiny_manifest_path, fftradnet_checkpoint,
                                                      tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "e2e.ml.detect_viz",
         "--manifest", str(tiny_manifest_path), "--checkpoint", str(fftradnet_checkpoint),
         "--split", "val", "--frame", "0", "--select", "median",
         "--out", str(tmp_path / "x.png")],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
