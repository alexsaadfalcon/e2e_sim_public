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


def test_render_scene_gif_ddma_config_also_renders(tiny_scenario, tmp_path):
    """DDMA (no tdm_deinterleave step) is a distinct code path in range_azimuth_map."""
    from e2e.ml.radar_config import RADIAL_LIKE

    cfg = dataclasses.replace(RADIAL_LIKE, name="test_tiny_ddma", n_chirps=24, n_samples=64)
    scenario = sample_scene(cfg, "D0", np.random.default_rng(1))
    out_path = tmp_path / "ddma.gif"

    render_scene.render_scene_gif(cfg, scenario, out_path, n_frames=2, fps=4, dpi=50)

    assert out_path.exists() and out_path.stat().st_size > 0


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
