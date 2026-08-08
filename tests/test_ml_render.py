"""Tests for `e2e.ml.render_scene` (bird's-eye + radar-view scene GIFs).

Fast/tiny by construction: a shrunk `RadarConfig` (few chirps/samples), a couple of
animation frames, low DPI. `e2e.ml.scenes`/`e2e.ml.labels` are sibling shards -- if
either isn't in the working tree yet this whole module skips cleanly.
"""
import dataclasses
import subprocess
import sys

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
