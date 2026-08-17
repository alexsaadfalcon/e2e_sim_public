"""Tests for `e2e.ml.render_scene` (bird's-eye + radar-view scene GIFs).

Fast/tiny by construction: a shrunk `RadarConfig` (few chirps/samples), a couple of
animation frames, low DPI. `e2e.ml.scenes`/`e2e.ml.labels` are sibling shards -- if
either isn't in the working tree yet this whole module skips cleanly.
"""
import contextlib
import dataclasses
import subprocess
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("e2e.ml.scenes", reason="sibling shard e2e.ml.scenes not present")
pytest.importorskip("e2e.ml.labels", reason="sibling shard e2e.ml.labels not present")
PIL_Image = pytest.importorskip("PIL.Image", reason="Pillow required to read/write GIFs")

from e2e.ml import render_scene
from e2e.ml.radar_config import TI_IWR1443
from e2e.ml.scenes import sample_scene


@pytest.fixture
def tiny_cfg():
    """A TDM-MIMO config shrunk for fast tests (matches the tiny_cfg pattern used by
    test_ml_dataset.py -- see that file for why these particular fields are shrunk)."""
    return dataclasses.replace(TI_IWR1443, name="test_tiny_render", n_chirps=12, n_samples=64)


@pytest.fixture
def tiny_scenario(tiny_cfg):
    return sample_scene(tiny_cfg, "D1", np.random.default_rng(0))


# --------------------------------------------------------------------------------
# render_scene_gif
# --------------------------------------------------------------------------------
def test_render_scene_gif_writes_readable_gif_with_expected_frame_count(tiny_cfg, tiny_scenario, tmp_path):
    out_path = tmp_path / "scene.gif"
    n_frames = 3

    result_path = render_scene.render_scene_gif(
        tiny_cfg, tiny_scenario, out_path, n_frames=n_frames, fps=4, seed=0, dpi=50,
    )

    assert result_path == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0

    with PIL_Image.open(out_path) as im:
        assert im.format == "GIF"
        # Pillow only exposes n_frames once the file is recognised as multi-frame.
        n_seen = getattr(im, "n_frames", 1)
        assert n_seen == n_frames


def test_render_scene_gif_creates_parent_directories(tiny_cfg, tiny_scenario, tmp_path):
    out_path = tmp_path / "nested" / "dir" / "scene.gif"
    render_scene.render_scene_gif(tiny_cfg, tiny_scenario, out_path, n_frames=2, fps=4, dpi=50)
    assert out_path.exists()


def test_render_scene_gif_ideal_panel_default_and_opt_out(tiny_cfg, tiny_scenario, tmp_path):
    """Default renders THREE panels (bird's-eye | ideal front end | non-ideal front
    end) on a 15-inch canvas; `ideal_panel=False` keeps the legacy two-panel 10-inch
    layout. Pinned via the GIF's pixel width (figsize x dpi), the public observable."""
    dpi = 50
    three = tmp_path / "three.gif"
    two = tmp_path / "two.gif"
    render_scene.render_scene_gif(tiny_cfg, tiny_scenario, three, n_frames=2, fps=4, dpi=dpi)
    render_scene.render_scene_gif(tiny_cfg, tiny_scenario, two, n_frames=2, fps=4, dpi=dpi,
                                  ideal_panel=False)
    with PIL_Image.open(three) as im:
        assert im.size[0] == 15 * dpi
    with PIL_Image.open(two) as im:
        assert im.size[0] == 10 * dpi


def test_render_scene_gif_axes_do_not_move_between_frames(monkeypatch, tiny_cfg, tiny_scenario,
                                                          tmp_path):
    """REGRESSION (owner feedback): the subplot sizes visibly shifted over the GIF's
    first frames because tight_layout ran inside the per-frame update and re-settled
    as frame contents (legends, tick extents) changed. Layout is now computed once,
    primed with frame 0 -- so every grabbed frame must see identical axes positions."""
    from matplotlib.animation import PillowWriter

    captured = []
    orig_grab = PillowWriter.grab_frame

    def spy(self, **kwargs):
        captured.append([tuple(np.round(ax.get_position().bounds, 6))
                         for ax in self.fig.axes])
        return orig_grab(self, **kwargs)

    monkeypatch.setattr(PillowWriter, "grab_frame", spy)
    render_scene.render_scene_gif(tiny_cfg, tiny_scenario, tmp_path / "stable.gif",
                                  n_frames=3, fps=4, dpi=50)
    assert len(captured) == 3
    assert captured[0] == captured[1] == captured[2]


def test_range_azimuth_map_norm_peak_shifts_reference(tiny_cfg):
    """`norm_peak` re-references the dB scale: normalizing against 100x the map's own
    peak must shift every bin down by exactly 20 dB (power/10log)."""
    from e2e.ml.rd_synth import synthesize_adc
    from e2e.ml.scatterers import RadarPose

    scat = [render_scene.Scatterer(position=(20.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0),
                                   rcs_dbsm=10.0, object_class="vehicle")]
    pose = RadarPose(position=(0.0, 0.0, 0.0), boresight=(1.0, 0.0, 0.0))
    adc = synthesize_adc(tiny_cfg, scat, pose, snr_db=30.0, seed=0)

    own_db, _ = render_scene.range_azimuth_map(tiny_cfg, adc)
    power, _ = render_scene.range_azimuth_power(tiny_cfg, adc)
    shifted_db, _ = render_scene.range_azimuth_map(tiny_cfg, adc,
                                                  norm_peak=float(power.max()) * 100.0)
    assert float(own_db.max()) == pytest.approx(0.0, abs=1e-5)
    assert torch.allclose(shifted_db, own_db - 20.0, atol=1e-4)


def test_range_azimuth_power_azimuth_window_hann_suppresses_off_target_sidelobe(tiny_cfg):
    """`azimuth_window="hann"` is a DISPLAY-only knob (detect_viz's justification for
    defaulting the figure backdrop to it): a strong, off-boresight target's rectangular-
    window sidelobe skirt at a DIFFERENT azimuth (same range) must be measurably lower
    with the Hann taper than without it."""
    from e2e.ml.rd_synth import synthesize_adc
    from e2e.ml.scatterers import RadarPose

    scat = [render_scene.Scatterer(position=(20.0, 12.0, 0.0), velocity=(0.0, 0.0, 0.0),
                                   rcs_dbsm=20.0, object_class="vehicle")]
    pose = RadarPose(position=(0.0, 0.0, 0.0), boresight=(1.0, 0.0, 0.0))
    adc = synthesize_adc(tiny_cfg, scat, pose, snr_db=40.0, seed=0)

    rect_power, sin_az_axis = render_scene.range_azimuth_power(tiny_cfg, adc)
    hann_power, hann_sin_az = render_scene.range_azimuth_power(tiny_cfg, adc, azimuth_window="hann")
    assert np.allclose(sin_az_axis, hann_sin_az)

    range_bin = int(rect_power.max(dim=0).values.argmax())  # the target's own range
    az_peak = int(rect_power[:, range_bin].argmax())
    # Sample the sidelobe skirt away from the mainlobe (>= 15 bins off peak).
    far_bins = [i for i in range(rect_power.shape[0]) if abs(i - az_peak) >= 15]
    assert far_bins

    rect_db_far = 10.0 * torch.log10(rect_power[far_bins, range_bin] / rect_power.max())
    hann_db_far = 10.0 * torch.log10(hann_power[far_bins, range_bin].clamp_min(1e-30)
                                     / hann_power.max())
    # Hann's sidelobe skirt must sit meaningfully lower on average than rectangular's.
    assert float(hann_db_far.mean()) < float(rect_db_far.mean()) - 5.0


def test_range_azimuth_power_azimuth_window_rejects_unknown_value(tiny_cfg):
    from e2e.ml.rd_synth import synthesize_adc
    from e2e.ml.scatterers import RadarPose

    scat = [render_scene.Scatterer(position=(20.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0),
                                   rcs_dbsm=10.0, object_class="vehicle")]
    pose = RadarPose(position=(0.0, 0.0, 0.0), boresight=(1.0, 0.0, 0.0))
    adc = synthesize_adc(tiny_cfg, scat, pose, snr_db=30.0, seed=0)
    with pytest.raises(ValueError, match="azimuth_window"):
        render_scene.range_azimuth_power(tiny_cfg, adc, azimuth_window="blackman")


def test_render_scene_gif_color_scale_is_global_and_fixed(monkeypatch, tiny_cfg, tiny_scenario,
                                                          tmp_path):
    """OWNER FEEDBACK: the color scale must be one deliberate window for the whole
    animation, not per-frame autoscale. Pins three things at grab time on every frame:
    (a) every radar image's clim is exactly (-db_range, 0); (b) a colorbar axes
    exists; (c) at most ONE frame's radar maps touch 0 dB -- with per-frame peak
    normalization EVERY frame would (that was the bug)."""
    from matplotlib.animation import PillowWriter

    clims, data_maxes, n_axes = [], [], []
    orig_grab = PillowWriter.grab_frame

    def spy(self, **kwargs):
        frame_clims, frame_max = [], -np.inf
        for ax in self.fig.axes:
            for im in ax.get_images():
                frame_clims.append(im.get_clim())
                frame_max = max(frame_max, float(np.max(im.get_array())))
        clims.append(frame_clims)
        data_maxes.append(frame_max)
        n_axes.append(len(self.fig.axes))
        return orig_grab(self, **kwargs)

    # The DEFAULT window is 80 dB, measured against the scenes' own dynamic range (the
    # pedestrians and the noise floor both live below -38 dB; see render_scene_gif).
    # Pinned here so a future "tidy-up" to a conventional 40 dB can't silently re-hide
    # them. This test then passes db_range explicitly to prove the knob works.
    import inspect
    assert inspect.signature(render_scene.render_scene_gif).parameters["db_range"].default == 80.0

    monkeypatch.setattr(PillowWriter, "grab_frame", spy)
    render_scene.render_scene_gif(tiny_cfg, tiny_scenario, tmp_path / "scale.gif",
                                  n_frames=3, fps=4, dpi=50, db_range=40.0)

    for frame_clims in clims:
        assert frame_clims and all(c == (-40.0, 0.0) for c in frame_clims)
    # 3 panels + 1 colorbar axes.
    assert all(n == 4 for n in n_axes)
    # Global reference: only the frame(s) holding the global peak reach 0 dB.
    assert sum(1 for m in data_maxes if m > -1e-4) <= 1


def test_render_scene_gif_ddma_config_also_renders(tiny_scenario, tmp_path):
    """DDMA (no tdm_deinterleave step) is a distinct code path in range_azimuth_map."""
    from e2e.ml.radar_config import RADIAL_LIKE

    cfg = dataclasses.replace(RADIAL_LIKE, name="test_tiny_ddma", n_chirps=24, n_samples=64)
    scenario = sample_scene(cfg, "D0", np.random.default_rng(1))
    out_path = tmp_path / "ddma.gif"

    render_scene.render_scene_gif(cfg, scenario, out_path, n_frames=2, fps=4, dpi=50)

    assert out_path.exists() and out_path.stat().st_size > 0


def _ddma_single_target_adc(n_samples=256, sin_az=0.35, rng_m=30.0):
    """One stationary point target at a known azimuth, on the DDMA `radial_like` config."""
    import math

    from e2e.ml.radar_config import RADIAL_LIKE
    from e2e.ml.rd_synth import synthesize_adc
    from e2e.ml.scatterers import Scatterer

    cfg = dataclasses.replace(RADIAL_LIKE, n_samples=n_samples)
    pos = (rng_m * math.sqrt(1.0 - sin_az ** 2), rng_m * sin_az, 0.0)
    sc = [Scatterer(position=pos, velocity=(0.0, 0.0, 0.0), rcs_dbsm=10.0,
                    object_class="vehicle")]
    return cfg, synthesize_adc(cfg, sc, snr_db=40.0, seed=0, random_phase=False), sin_az


def test_range_azimuth_power_ddma_uses_the_full_virtual_aperture(tiny_cfg):
    """Regression for the DDMA demux defect (found 2026-08-16, fixed 2026-08-17).

    `range_azimuth_power` de-interleaved TDM into the virtual array but had no DDMA
    branch, so for `radial_like` the angle FFT ran over the 16 PHYSICAL receivers rather
    than the 192 virtual elements. It never looked broken: the peak still lands at the
    right azimuth, because 16 elements are enough to LOCATE a lone target. What is lost
    is resolution -- the mainlobe is ~12x too wide, which is what smeared targets into
    horizontal ridges across every DDMA range-azimuth picture.

    So the assertion is on WIDTH, not position. Measured on this fixture: 14 bins of 256
    over the physical array against 1 bin demuxed.
    """
    import torch

    from e2e.ml.transforms import adc_to_rd, ddma_demux

    cfg, adc, sin_az = _ddma_single_target_adc()
    n_fft = 256

    def mainlobe_bins(rd_cube):
        spec = torch.fft.fftshift(torch.fft.fft(rd_cube, n=n_fft, dim=0), dim=0)
        ra = (spec.abs() ** 2).max(dim=2).values
        col = ra[:, int(ra.argmax()) % ra.shape[1]]
        return int((col >= col.max() / 2).sum().item())

    rd = adc_to_rd(cfg, adc)
    physical = mainlobe_bins(rd)
    virtual = mainlobe_bins(ddma_demux(cfg, rd))
    assert virtual * 4 <= physical, (
        f"demuxed mainlobe ({virtual} bins) is not markedly narrower than the "
        f"physical-array one ({physical} bins) -- the demux is not taking effect")

    # And the shipped path must be the demuxed one.
    ra_power, sin_az_axis = render_scene.range_azimuth_power(cfg, adc, n_angle_fft=n_fft)
    peak_row = int(ra_power.argmax()) // ra_power.shape[1]
    assert abs(float(sin_az_axis[peak_row]) - sin_az) < 0.02, "peak azimuth is wrong"
    col = ra_power[:, int(ra_power.argmax()) % ra_power.shape[1]]
    assert int((col >= col.max() / 2).sum().item()) == virtual


# --------------------------------------------------------------------------------
# range_azimuth_map
# --------------------------------------------------------------------------------
def test_range_azimuth_map_peak_matches_known_target(tiny_cfg):
    """A single static target's range-azimuth peak should land within a couple of
    bins of its true (range, sin_azimuth) -- a coarse correctness check, not a tight
    numerical one (see rd_synth's own tests for exact-bin checks)."""
    from e2e.ml.scatterers import frame_scatterers, radar_pose, vehicle
    from e2e.scenario import Node, NodeRole, Scenario

    scenario = Scenario(
        name="single_target",
        nodes=[Node(name="radar", role=NodeRole.RADAR, position=(0.0, 0.0, 0.0),
                    look_at=(1.0, 0.0, 0.0))],
        # tiny_cfg's shrunk n_samples gives a small max_range_m -- place the target
        # well inside it (e.g. at ~40% of max_range) rather than at a fixed metre value.
        objects=[vehicle("car", (0.4 * tiny_cfg.max_range_m, 0.0, 0.0))],  # dead ahead
    )
    scat = frame_scatterers(scenario, 0)
    pose = radar_pose(scenario, 0)

    from e2e.ml.rd_synth import synthesize_adc

    adc = synthesize_adc(tiny_cfg, scat, pose, snr_db=30.0, seed=0)
    ra_db, sin_az_axis = render_scene.range_azimuth_map(tiny_cfg, adc)

    i, j = np.unravel_index(np.argmax(ra_db.numpy()), ra_db.shape)
    peak_sin_az = float(sin_az_axis[i])
    peak_range_m = float(j) * tiny_cfg.range_resolution_m

    assert abs(peak_sin_az) < 0.1  # dead-ahead target -> near sin(az) == 0
    expected_range_m = 0.4 * tiny_cfg.max_range_m
    assert abs(peak_range_m - expected_range_m) < 2.0 * tiny_cfg.range_resolution_m


# --------------------------------------------------------------------------------
# _draw_radar_view orientation (regression: array/extent transpose mismatch)
# --------------------------------------------------------------------------------
def test_draw_radar_view_imshow_orientation_matches_extent(tiny_cfg):
    """`ra_db` is `[n_angle, n_range]` (see `range_azimuth_map`'s docstring) but the
    panel's `extent` puts azimuth on x and range on y -- imshow needs `[n_range,
    n_angle]` to match. Regression for a transpose bug where the raw (untransposed)
    array was passed to imshow: build a map with a single hot cell at a known
    (angle_idx, range_idx) and assert the rendered AxesImage's array has that cell at
    the position implied by the extent (row -> range, column -> azimuth), not the
    other way around.
    """
    import matplotlib.pyplot as plt

    from e2e.ml.labels import LabelGrid
    from e2e.ml.scatterers import RadarPose

    n_angle, n_range = 8, 5
    angle_idx, range_idx = 2, 4  # deliberately distinct so a transpose is detectable
    ra_db = torch.full((n_angle, n_range), -40.0)
    ra_db[angle_idx, range_idx] = 0.0
    sin_az_axis = np.linspace(-1.0, 1.0, n_angle)

    grid = LabelGrid.for_config(tiny_cfg)
    pose = RadarPose(position=(0.0, 0.0, 0.0), boresight=(1.0, 0.0, 0.0))

    fig, ax = plt.subplots()
    try:
        render_scene._draw_radar_view(ax, tiny_cfg, grid, ra_db, sin_az_axis, [], pose)
        images = ax.get_images()
        assert len(images) == 1
        arr = images[0].get_array()

        assert arr.shape == (n_range, n_angle)  # [row=range, col=angle], not the raw ra_db shape
        hot_row, hot_col = np.unravel_index(np.argmax(arr), arr.shape)
        assert (hot_row, hot_col) == (range_idx, angle_idx)
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------
def test_cli_help_exits_zero():
    proc = subprocess.run(
        [sys.executable, "-m", "e2e.ml.render_scene", "--help"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "render_scene" in proc.stdout


def test_cli_end_to_end_writes_gif(tmp_path):
    out_path = tmp_path / "cli_scene.gif"
    proc = subprocess.run(
        [sys.executable, "-m", "e2e.ml.render_scene",
         "--tier", "D0", "--config", "ti_iwr1443", "--out", str(out_path),
         "--frames", "3", "--fps", "4", "--dpi", "50", "--seed", "1"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out_path.exists() and out_path.stat().st_size > 0
    assert "wrote" in proc.stdout.lower()


def test_cli_unknown_config_exits_nonzero(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "e2e.ml.render_scene",
         "--tier", "D0", "--config", "not_a_real_config", "--out", str(tmp_path / "x.gif")],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0


def test_cli_unknown_tier_exits_nonzero(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "e2e.ml.render_scene",
         "--tier", "not_a_real_tier", "--config", "ti_iwr1443", "--out", str(tmp_path / "x.gif")],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0


# --------------------------------------------------------------------------------
# rt_scene_build._default_object_render_color -- per-class RENDER colour fallback
# (owner feedback: "are there plain cubes? ... can't really see pedestrians"). Pure
# dataclass/ObjectKind dispatch, no Sionna needed.
# --------------------------------------------------------------------------------
def test_default_object_render_color_distinguishes_sphere_from_mesh_vehicle():
    """A D0 sphere target and a D1+ mesh vehicle share `object_class="vehicle"` (see
    `e2e.ml.rt_scenes.build_rt_tier_scenario`) but must still get DIFFERENT colours --
    a reviewer needs to tell a bare sphere from a real car mesh by eye."""
    from e2e.ml.rt_scene_build import (_OBJECT_COLOR_SPHERE, _OBJECT_COLOR_VEHICLE,
                                       _default_object_render_color)
    from e2e.scenario import ObjectKind, SceneObject

    sphere = SceneObject(name="s", kind=ObjectKind.SPHERE, object_class="vehicle")
    mesh_vehicle = SceneObject(name="v", kind=ObjectKind.MESH, object_class="vehicle",
                               asset="low_poly_car")

    assert _default_object_render_color(sphere) == _OBJECT_COLOR_SPHERE
    assert _default_object_render_color(mesh_vehicle) == _OBJECT_COLOR_VEHICLE
    assert _OBJECT_COLOR_SPHERE != _OBJECT_COLOR_VEHICLE


def test_default_object_render_color_covers_every_rt_scenes_class():
    """Every object kind/class `e2e.ml.rt_scenes.build_rt_tier_scenario` actually
    produces (sphere, mesh vehicle, mesh pedestrian, box clutter) resolves to a
    distinct colour, and an unrecognized combination falls back to the legacy default
    rather than raising."""
    from e2e.ml.rt_scene_build import (_OBJECT_COLOR_CLUTTER_BOX, _OBJECT_COLOR_DEFAULT,
                                       _OBJECT_COLOR_PEDESTRIAN, _OBJECT_COLOR_SPHERE,
                                       _OBJECT_COLOR_VEHICLE, _default_object_render_color)
    from e2e.scenario import ObjectKind, SceneObject

    pedestrian = SceneObject(name="p", kind=ObjectKind.MESH, object_class="pedestrian",
                             asset="pedestrian_placeholder", material="skin")
    clutter_box = SceneObject(name="b", kind=ObjectKind.BOX, object_class="scatterer")
    unknown = SceneObject(name="u", kind=ObjectKind.MESH, object_class="unrecognized")

    assert _default_object_render_color(pedestrian) == _OBJECT_COLOR_PEDESTRIAN
    assert _default_object_render_color(clutter_box) == _OBJECT_COLOR_CLUTTER_BOX
    assert _default_object_render_color(unknown) == _OBJECT_COLOR_DEFAULT

    all_colors = {_OBJECT_COLOR_VEHICLE, _OBJECT_COLOR_PEDESTRIAN, _OBJECT_COLOR_CLUTTER_BOX,
                 _OBJECT_COLOR_SPHERE, _OBJECT_COLOR_DEFAULT}
    assert len(all_colors) == 5, "every class colour must be visually distinct"


def test_default_object_render_color_never_consulted_when_obj_color_is_set():
    """`build_rt_scene`'s fallback expression is `obj.color if obj.color is not None
    else _default_object_render_color(obj)` -- an explicit scenario-authored colour
    must always win. This only checks the SceneObject side (the ternary itself is
    exercised end-to-end by the gated `test_build_rt_scene_assigns_...` below, which
    needs real Sionna)."""
    from e2e.scenario import ObjectKind, SceneObject

    obj = SceneObject(name="s", kind=ObjectKind.SPHERE, object_class="vehicle",
                      color=(0.42, 0.42, 0.42))
    assert obj.color == (0.42, 0.42, 0.42)


# --------------------------------------------------------------------------------
# _build_camera -- explicit near-vertical (top-down) camera construction. A fake `rt`
# module stand-in (just `Camera`) is enough: the branch logic itself needs no Sionna,
# only the ACTUAL angle/axis calibration (asserted in the gated test further below)
# does.
# --------------------------------------------------------------------------------
class _FakeCamera:
    def __init__(self, *, position, orientation=None, look_at=None):
        self.position = position
        self.orientation = orientation
        self.look_at = look_at


class _FakeRt:
    Camera = _FakeCamera


def test_build_camera_straight_down_uses_explicit_orientation_not_look_at():
    cam_pos = np.array([0.0, 0.0, 60.0])
    centroid = np.array([0.0, 0.0, 0.0])   # forward = (0, 0, -1): exactly vertical

    camera = render_scene._build_camera(cam_pos, centroid, _FakeRt)

    assert camera.look_at is None, "must not go through Camera.look_at() when vertical"
    assert camera.orientation == pytest.approx(
        (render_scene._TOP_DOWN_YAW_RAD, np.pi / 2.0, 0.0))
    assert camera.position == [0.0, 0.0, 60.0]


def test_build_camera_straight_up_uses_explicit_orientation_with_opposite_pitch():
    cam_pos = np.array([0.0, 0.0, 0.0])
    centroid = np.array([0.0, 0.0, 60.0])  # forward = (0, 0, 1): exactly vertical, looking up

    camera = render_scene._build_camera(cam_pos, centroid, _FakeRt)

    assert camera.look_at is None
    assert camera.orientation == pytest.approx(
        (render_scene._TOP_DOWN_YAW_RAD, -np.pi / 2.0, 0.0))


def test_build_camera_oblique_direction_still_uses_look_at():
    """A normal (non-vertical) camera direction is untouched -- goes through Sionna's
    own `Camera.look_at()` exactly as before this change."""
    cam_pos = np.array([10.0, 10.0, 10.0])
    centroid = np.array([0.0, 0.0, 0.0])

    camera = render_scene._build_camera(cam_pos, centroid, _FakeRt)

    assert camera.orientation is None
    assert camera.look_at == [0.0, 0.0, 0.0]


def test_build_camera_default_render_direction_is_oblique_not_near_vertical():
    """`render_rt_tier_png`'s own default `camera_dir=(-1,-1,1.15)` must NOT trip the
    near-vertical branch -- it is a deliberately oblique "behind and above" view, not
    the dedicated top-down mode."""
    forward = -np.asarray((-1.0, -1.0, 1.15), dtype=float)
    forward = forward / np.linalg.norm(forward)
    assert abs(float(forward[2])) < render_scene._NEAR_VERTICAL_DOT


def test_top_down_camera_dir_is_purely_vertical():
    assert render_scene.TOP_DOWN_CAMERA_DIR == (0.0, 0.0, 1.0)


# --------------------------------------------------------------------------------
# _overlay_legend -- colour-key legend overlay (owner feedback: viewers could not tell
# a pedestrian from clutter). Pure Pillow, no Sionna needed.
# --------------------------------------------------------------------------------
def test_overlay_legend_draws_a_swatch_matching_each_entrys_color():
    img = PIL_Image.new("RGB", (200, 200), (128, 128, 128))
    entries = [("vehicle", (0.0, 0.0, 1.0)), ("pedestrian", (0.0, 1.0, 1.0))]

    render_scene._overlay_legend(img, entries, title="test frame 1/3")

    arr = np.asarray(img)
    # The legend box sits in the bottom-left corner (see _overlay_legend); its first
    # swatch is drawn a fixed (pad, pad) offset inside that box -- sample inside the
    # swatch rectangle rather than depending on exact pixel arithmetic.
    box_h = 2 * 8 + 20 * len(entries)
    y0 = img.height - box_h - 8
    swatch_center = (y0 + 8 + 9, 8 + 8 + 7)   # (row, col) inside the first swatch
    r, g, b = arr[swatch_center[0], swatch_center[1]]
    assert (int(r), int(g), int(b)) == (0, 0, 255), "first entry's swatch must be blue"


def test_overlay_legend_title_bar_present_when_requested():
    img = PIL_Image.new("RGB", (200, 200), (200, 200, 200))
    render_scene._overlay_legend(img, [("radar", (1.0, 0.85, 0.0))], title="hello")
    arr = np.asarray(img)
    # Title bar is a near-black translucent strip across the top few rows.
    assert arr[2, 100].sum() < 200 * 3 * 0.5


# --------------------------------------------------------------------------------
# _build_rt_scene_for_render -- the D4 (munich @ 77 GHz) city-scene material fix,
# ported from the scratch probe (see e2e.environment.city_scenes / RTEnvironmentBlock.
# get_S_pars for the pattern this mirrors). Real Sionna rendering can't run here
# (RUN_SIONNA=1 needed, plus this box's DrJit/LLVM backend is broken -- see CLAUDE.md);
# these tests mock `build_rt_scene`/`patched_builtin_loader` so the guarded BRANCH
# logic itself is verified without touching Sionna at all.
# --------------------------------------------------------------------------------
def test_build_rt_scene_for_render_flat_scene_skips_material_patch(monkeypatch, tiny_cfg):
    """`base_scene="flat"`/`"free"` (D0-D3) must be a pure no-op: no
    `patched_builtin_loader` import/call, `build_rt_scene` called with the plain
    (scenario, cfg, base_scene=..., frame_idx=0) signature."""
    from e2e.ml import render_scene

    calls = []
    monkeypatch.setattr("e2e.ml.rt_gen.build_rt_scene",
                        lambda *a, **kw: calls.append(("build_rt_scene", a, kw)) or "SCENE")

    def _boom(*a, **kw):
        raise AssertionError("patched_builtin_loader must not be called for a flat/free scene")

    monkeypatch.setattr("e2e.environment.city_scenes.patched_builtin_loader", _boom)

    scenario = types.SimpleNamespace(base_scene="flat")
    result = render_scene._build_rt_scene_for_render(scenario, tiny_cfg)

    assert result == "SCENE"
    assert len(calls) == 1
    _, args, kwargs = calls[0]
    assert args == (scenario, tiny_cfg)
    assert kwargs == {"base_scene": "flat", "frame_idx": 0}


def test_build_rt_scene_for_render_free_scene_also_skips_material_patch(monkeypatch, tiny_cfg):
    from e2e.ml import render_scene

    monkeypatch.setattr("e2e.ml.rt_gen.build_rt_scene", lambda *a, **kw: "SCENE")
    monkeypatch.setattr("e2e.environment.city_scenes.patched_builtin_loader",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not be called")))

    scenario = types.SimpleNamespace(base_scene="free")
    result = render_scene._build_rt_scene_for_render(scenario, tiny_cfg)
    assert result == "SCENE"


def test_build_rt_scene_for_render_city_scene_wraps_in_patched_loader(monkeypatch, tiny_cfg):
    """`base_scene="munich"` (D4) must build the scene INSIDE `patched_builtin_loader`,
    at the config's centre frequency, with the requested policy -- this is the actual
    fix: the scratch probe showed munich's out-of-band ITU materials (marble/brick)
    hard-raise on `scene.frequency` assignment unless the loader is patched first."""
    from e2e.ml import render_scene

    events = []

    @contextlib.contextmanager
    def fake_patched_loader(frequency_hz, *, policy, stand_in_itu_type, report_sink=None):
        events.append(("enter", frequency_hz, policy, stand_in_itu_type))
        yield
        events.append(("exit",))

    def fake_build_rt_scene(scenario, cfg, *, base_scene, frame_idx):
        assert events and events[-1] == ("enter", pytest.approx(
            float(tiny_cfg.f0_hz) + float(tiny_cfg.bandwidth_hz) / 2.0),
            "extrapolated", "concrete"), "build_rt_scene must run INSIDE the patched loader"
        events.append(("build_rt_scene", base_scene, frame_idx))
        return "SCENE"

    monkeypatch.setattr("e2e.environment.city_scenes.patched_builtin_loader", fake_patched_loader)
    monkeypatch.setattr("e2e.ml.rt_gen.build_rt_scene", fake_build_rt_scene)

    scenario = types.SimpleNamespace(base_scene="munich")
    result = render_scene._build_rt_scene_for_render(scenario, tiny_cfg)

    assert result == "SCENE"
    assert [e[0] for e in events] == ["enter", "build_rt_scene", "exit"]
    assert events[1] == ("build_rt_scene", "munich", 0)


def test_build_rt_scene_for_render_city_scene_passes_through_policy_overrides(monkeypatch, tiny_cfg):
    from e2e.ml import render_scene

    seen_policy = {}

    @contextlib.contextmanager
    def fake_patched_loader(frequency_hz, *, policy, stand_in_itu_type, report_sink=None):
        seen_policy["policy"] = policy
        seen_policy["stand_in_itu_type"] = stand_in_itu_type
        yield

    monkeypatch.setattr("e2e.environment.city_scenes.patched_builtin_loader", fake_patched_loader)
    monkeypatch.setattr("e2e.ml.rt_gen.build_rt_scene", lambda *a, **kw: "SCENE")

    scenario = types.SimpleNamespace(base_scene="etoile")
    render_scene._build_rt_scene_for_render(
        scenario, tiny_cfg, material_policy="stand_in", stand_in_material="brick",
    )

    assert seen_policy == {"policy": "stand_in", "stand_in_itu_type": "brick"}


def test_build_rt_tier_scenario_d4_uses_munich_base_scene_no_sionna_needed():
    """Sanity: the D4 tier that triggers the city-scene branch above really does resolve
    to `base_scene="munich"` -- `build_rt_tier_scenario` itself needs no Sionna (see
    `e2e.ml.rt_scenes`'s module docstring), so this is a real (non-mocked) check."""
    from e2e.ml.rt_scenes import build_rt_tier_scenario

    scenario = build_rt_tier_scenario("D4", frame_idx=0, seed=0, num_frames=1,
                                      use_local_assets=False)
    assert scenario.base_scene == "munich"


# --------------------------------------------------------------------------------
# Real Sionna RT renders: the top-down camera calibration and the per-class colour
# fallback end to end, plus render_rt_topdown_gif's actual motion. Gated behind
# @pytest.mark.sionna (RUN_SIONNA=1) like tests/test_ml_rt_gen.py -- these are plain
# geometry renders (no path solve), so they are comparatively cheap, but they still
# need a working Sionna RT / DrJit install. Verified locally with
# CUDA_VISIBLE_DEVICES=1 (GPU 0 was busy with an unrelated generation job).
# --------------------------------------------------------------------------------
@pytest.mark.sionna
def test_build_camera_topdown_calibration_matches_bird_eye_axes(tmp_path):
    """The actual empirical calibration `_build_camera`'s docstring/`_TOP_DOWN_YAW_RAD`
    claim: with `orientation=(_TOP_DOWN_YAW_RAD, pi/2, 0)`, world +x maps to screen
    RIGHT and world +y maps to screen UP -- matching `_draw_birdseye`'s (x right, y up)
    convention. Two colour-coded spheres pin down the mapping directly from a real
    render, not asserted from the Euler-angle derivation alone."""
    sionna_rt = pytest.importorskip("sionna.rt")
    from PIL import Image

    from e2e.ml.rt_scene_build import _synthetic_scene_path

    scene = sionna_rt.load_scene(_synthetic_scene_path("flat"), merge_shapes=False)
    scene.frequency = 77e9

    mat_x = sionna_rt.ITURadioMaterial("mx", "metal", thickness=0.01, color=(1.0, 0.0, 0.0))
    sx = sionna_rt.SceneObject(fname=sionna_rt.scene.sphere, name="sx", radio_material=mat_x)
    scene.edit(add=[sx])
    sx.scaling = 3.0
    sx.position = [15.0, 0.0, 3.0]

    mat_y = sionna_rt.ITURadioMaterial("my", "metal", thickness=0.01, color=(0.0, 1.0, 0.0))
    sy = sionna_rt.SceneObject(fname=sionna_rt.scene.sphere, name="sy", radio_material=mat_y)
    scene.edit(add=[sy])
    sy.scaling = 3.0
    sy.position = [0.0, 15.0, 3.0]

    cam_pos = np.array([0.0, 0.0, 60.0])
    centroid = np.array([0.0, 0.0, 0.0])
    camera = render_scene._build_camera(cam_pos, centroid, sionna_rt)

    out_path = tmp_path / "calib.png"
    scene.render_to_file(camera=camera, filename=str(out_path), resolution=(400, 400),
                         num_samples=32, fov=50.0)

    arr = np.asarray(Image.open(out_path).convert("RGB")).astype(float)
    red = (arr[:, :, 0] > 150) & (arr[:, :, 1] < 100) & (arr[:, :, 2] < 100)
    green = (arr[:, :, 1] > 150) & (arr[:, :, 0] < 100) & (arr[:, :, 2] < 100)
    ry, rx = np.where(red)
    gy, gx = np.where(green)
    assert ry.size and gy.size, "expected both markers visible in the top-down render"
    cx, cy = arr.shape[1] / 2.0, arr.shape[0] / 2.0

    # +x marker (red): screen RIGHT of centre, vertically centred.
    assert rx.mean() > cx + 50
    assert abs(ry.mean() - cy) < 20
    # +y marker (green): screen UP (smaller row index) of centre, horizontally centred.
    assert gy.mean() < cy - 50
    assert abs(gx.mean() - cx) < 20


@pytest.mark.sionna
def test_build_rt_scene_assigns_distinct_colors_per_class_without_touching_rf_params(tiny_cfg):
    """End-to-end (real Sionna materials) check that `build_rt_scene` (a) colours each
    class distinctly per `_default_object_render_color`, and (b) leaves the RF-visible
    material parameters (relative_permittivity/conductivity/scattering_coefficient)
    IDENTICAL between two same-base-material objects that only differ in colour --
    i.e. colour and RF material are decoupled, verified empirically against the
    installed Sionna materials, not just read off its source."""
    pytest.importorskip("sionna.rt")
    from e2e.ml.rt_gen import build_rt_scene
    from e2e.ml.rt_scene_build import (_OBJECT_COLOR_CLUTTER_BOX, _OBJECT_COLOR_PEDESTRIAN,
                                       _OBJECT_COLOR_SPHERE, _OBJECT_COLOR_VEHICLE)
    from e2e.scenario import Node, NodeRole, ObjectKind, Scenario, SceneObject

    scenario = Scenario(
        name="color_test", base_scene="free", num_frames=1,
        nodes=[Node(name="radar", role=NodeRole.RADAR, position=(0.0, 0.0, 1.5),
                    look_at=(1.0, 0.0, 1.5))],
        objects=[
            SceneObject(name="sphere-0", kind=ObjectKind.SPHERE, position=(10.0, 0.0, 1.0),
                       scaling=0.5, material="metal", object_class="vehicle"),
            SceneObject(name="vehicle-0", kind=ObjectKind.MESH, asset="low_poly_car",
                       position=(15.0, 3.0, 0.75), scaling=1.0, material="metal",
                       object_class="vehicle"),
            SceneObject(name="pedestrian-0", kind=ObjectKind.MESH, asset="pedestrian_placeholder",
                       position=(8.0, -3.0, 0.87), scaling=1.0, material="skin",
                       object_class="pedestrian"),
            SceneObject(name="clutter-box-0", kind=ObjectKind.BOX, position=(20.0, 5.0, 1.25),
                       scaling=0.5, material="concrete", object_class="scatterer"),
        ],
    )

    rt_scene = build_rt_scene(scenario, tiny_cfg, base_scene="free")

    got = {name: tuple(round(c, 3) for c in mat.color)
          for name, mat in rt_scene.materials.items()}
    assert got["sphere-0"] == tuple(round(c, 3) for c in _OBJECT_COLOR_SPHERE)
    assert got["vehicle-0"] == tuple(round(c, 3) for c in _OBJECT_COLOR_VEHICLE)
    assert got["pedestrian-0"] == tuple(round(c, 3) for c in _OBJECT_COLOR_PEDESTRIAN)
    assert got["clutter-box-0"] == tuple(round(c, 3) for c in _OBJECT_COLOR_CLUTTER_BOX)
    assert len(set(got.values())) == 4, "every class must render a distinct colour"

    # Colour vs RF material decoupling: the sphere and the mesh vehicle are both plain
    # "metal" ITU materials with the SAME scattering_coefficient but DIFFERENT colours
    # -- their RF-visible parameters must still match exactly.
    m_sphere = rt_scene.materials["sphere-0"]
    m_vehicle = rt_scene.materials["vehicle-0"]
    assert m_sphere.color != m_vehicle.color
    assert float(m_sphere.relative_permittivity.numpy()[0]) == \
          pytest.approx(float(m_vehicle.relative_permittivity.numpy()[0]))
    assert float(m_sphere.conductivity.numpy()[0]) == \
          pytest.approx(float(m_vehicle.conductivity.numpy()[0]))
    assert float(m_sphere.scattering_coefficient.numpy()[0]) == \
          pytest.approx(float(m_vehicle.scattering_coefficient.numpy()[0]))


@pytest.mark.sionna
def test_render_rt_topdown_gif_objects_move_between_frames(tmp_path):
    """The actual deliverable's core claim: a top-down GIF built from ONE scenario at
    successive MOTION frames shows real pixel-level movement, not a static loop (the
    exact bug this task's brief warned against -- and the one the `dt` fix in commit
    c212076 exists to prevent)."""
    pytest.importorskip("sionna.rt")
    from PIL import Image

    out_path = tmp_path / "topdown_test.gif"
    n_frames = 4
    result_path = render_scene.render_rt_topdown_gif(
        "D1", out_path, cfg=TI_IWR1443, n_frames=n_frames, fps=4, frame_idx=0, seed=0,
        resolution=(200, 150), num_samples=16, use_local_assets=False,
    )

    assert result_path == out_path
    assert out_path.exists() and out_path.stat().st_size > 0

    gif = Image.open(out_path)
    frame_arrays = []
    for i in range(n_frames):
        gif.seek(i)
        frame_arrays.append(np.asarray(gif.convert("RGB")).astype(int))
    assert len(frame_arrays) == n_frames

    diff = np.abs(frame_arrays[0] - frame_arrays[-1])
    assert diff.max() > 0, "objects must visibly move between the first and last frame"
    # More than a handful of stray anti-aliasing/noise pixels changed -- real motion,
    # not render-noise jitter on an otherwise static frame.
    assert int((diff.sum(axis=-1) > 20).sum()) > 20
