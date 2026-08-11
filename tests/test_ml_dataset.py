"""Tests for `e2e.ml.dataset`: sample synthesis, on-disk dataset generation,
`RadarFrameDataset`, and the CLI.

`e2e.ml.labels` and `e2e.ml.scenes` are sibling shards developed in parallel with this
one -- if either hasn't landed in the working tree yet, this whole module skips
cleanly (rather than erroring) via the `importorskip` calls below.
"""
import dataclasses
import json
import math
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
labels = pytest.importorskip("e2e.ml.labels", reason="sibling shard e2e.ml.labels not present")
scenes = pytest.importorskip("e2e.ml.scenes", reason="sibling shard e2e.ml.scenes not present")

from e2e.ml import dataset as ml_dataset
from e2e.ml.radar_config import PRESETS, RADIAL_LIKE, TI_IWR1443

# Any valid tier works for these plumbing tests; we don't assert on tier-specific content.
TIER = sorted(scenes.DIFFICULTY_TIERS)[0]


@pytest.fixture
def tiny_cfg():
    """A TDM-MIMO config shrunk for fast tests (see the sharding brief for the pattern)."""
    return dataclasses.replace(TI_IWR1443, name="test_tiny_tdm", n_chirps=12, n_samples=64)


@pytest.fixture
def registered_tiny_cfg(tiny_cfg, monkeypatch):
    """Register `tiny_cfg` under its own name in `PRESETS` so `generate_dataset`
    (which looks configs up by name) can use it; undone automatically by `monkeypatch`.
    """
    monkeypatch.setitem(PRESETS, tiny_cfg.name, tiny_cfg)
    return tiny_cfg


def _expected_shapes(cfg):
    grid = labels.LabelGrid.for_config(cfg)
    if cfg.mimo == "tdm":
        c, d = cfg.n_virtual, cfg.n_chirps_per_tx
    else:
        c, d = cfg.n_rx, cfg.n_chirps
    return (2 * c, cfg.n_samples, d), (3, grid.n_range, grid.n_azimuth)


def _expected_adc_shape(cfg):
    return (cfg.n_rx, cfg.n_chirps, cfg.n_samples)


def _expected_adc_input_shape(cfg):
    """Shape of the `input_format="adc"` derived tensor: raw physical channels,
    real/imag-stacked, [n_samples, n_chirps] axis order (see `RadarFrameDataset`)."""
    return (2 * cfg.n_rx, cfg.n_samples, cfg.n_chirps)


# --------------------------------------------------------------------------------
# generate_sample
# --------------------------------------------------------------------------------
def test_generate_sample_tdm_shapes_and_dtypes(registered_tiny_cfg, torch_device):
    from e2e.ml.scatterers import vehicle
    from e2e.scenario import Node, NodeRole, Scenario

    cfg = registered_tiny_cfg
    scenario = Scenario(
        name="tiny_tdm_scene",
        nodes=[Node(name="radar", role=NodeRole.RADAR, position=(0.0, 0.0, 0.0),
                    look_at=(10.0, 0.0, 0.0))],
        objects=[vehicle("car", (3.0, 0.5, 0.0))],
    )
    grid = labels.LabelGrid.for_config(cfg)

    sample = ml_dataset.generate_sample(cfg, scenario, grid, seed=0, device=torch_device)

    expected_input_shape, expected_labels_shape = _expected_shapes(cfg)
    assert sample["input"].shape == expected_input_shape
    assert sample["input"].dtype == torch.float32
    assert sample["input"].device.type == "cpu"
    assert sample["adc"].shape == _expected_adc_shape(cfg)
    assert sample["adc"].dtype == torch.complex64
    assert sample["adc"].device.type == "cpu"
    assert sample["labels"].shape == expected_labels_shape
    assert sample["labels"].dtype == torch.float32
    assert sample["labels"].device.type == "cpu"
    assert isinstance(sample["targets"], list)
    for t in sample["targets"]:
        assert len(t) == 3
    assert sample["meta"]["config"] == cfg.name
    assert sample["meta"]["mimo"] == "tdm"
    # target_extras: one entry per target, same order, with rcs/velocity present.
    extras = sample["meta"]["target_extras"]
    assert len(extras) == len(sample["targets"])
    for e in extras:
        assert isinstance(e["rcs_dbsm"], float)
        assert isinstance(e["velocity_mps"], list) and len(e["velocity_mps"]) == 3


def test_generate_sample_ddma_shapes(torch_device):
    from e2e.ml.scatterers import vehicle
    from e2e.scenario import Node, NodeRole, Scenario

    cfg = dataclasses.replace(RADIAL_LIKE, name="test_tiny_ddma", n_tx=4, n_rx=4, n_chirps=16, n_samples=32)
    scenario = Scenario(
        name="tiny_ddma_scene",
        nodes=[Node(name="radar", role=NodeRole.RADAR, position=(0.0, 0.0, 0.0),
                    look_at=(10.0, 0.0, 0.0))],
        objects=[vehicle("car", (5.0, 1.0, 0.0))],
    )
    grid = labels.LabelGrid.for_config(cfg)

    sample = ml_dataset.generate_sample(cfg, scenario, grid, seed=0, device=torch_device)

    expected_input_shape, expected_labels_shape = _expected_shapes(cfg)
    assert sample["input"].shape == expected_input_shape
    assert sample["labels"].shape == expected_labels_shape
    assert sample["adc"].shape == _expected_adc_shape(cfg)


# --------------------------------------------------------------------------------
# generate_dataset
# --------------------------------------------------------------------------------
def test_generate_dataset_manifest_and_splits(tmp_path, registered_tiny_cfg, torch_device):
    cfg = registered_tiny_cfg
    manifest_path = ml_dataset.generate_dataset(
        cfg.name, TIER, 6, out_dir=tmp_path, seed=0, device=torch_device,
    )
    assert manifest_path.is_file()
    assert manifest_path.parent == tmp_path / f"{cfg.name}_{TIER}"

    manifest = json.loads(manifest_path.read_text())
    assert manifest["config"]["name"] == cfg.name
    assert manifest["tier"] == TIER
    assert manifest["seed"] == 0
    assert manifest["manifest_version"] == 2
    assert manifest["frames_per_scene"] == 1
    assert "input_format" not in manifest  # loader choice, not a corpus property

    files = manifest["files"]
    assert len(files["train"]) == 4
    assert len(files["val"]) == 1
    assert len(files["test"]) == 1
    all_files = files["train"] + files["val"] + files["test"]
    assert len(all_files) == 6
    assert len(set(all_files)) == 6  # no overlap between splits

    # frames_per_scene == 1 -> one singleton sequence per frame.
    assert len(manifest["sequences"]) == 6
    assert all(len(seq) == 1 for seq in manifest["sequences"])
    assert sorted(seq[0] for seq in manifest["sequences"]) == sorted(all_files)

    expected_adc_shape = _expected_adc_shape(cfg)
    expected_labels_shape = _expected_shapes(cfg)[1]
    for fname in all_files:
        with np.load(manifest_path.parent / fname) as data:
            assert "input" not in data  # ADC-native: not precomputed/stored
            assert data["adc"].shape == expected_adc_shape
            assert data["adc"].dtype == np.complex64
            assert data["labels"].shape == expected_labels_shape
            assert data["labels"].dtype == np.float32
            meta = json.loads(str(data["meta"].item()))
            assert "targets" in meta
            assert "target_extras" in meta
            assert len(meta["target_extras"]) == len(meta["targets"])
            assert "scene" in meta
            assert "classes" not in meta["scene"]  # dropped: redundant with the counts
            assert "clutter" in meta["scene"]
            assert "placement_attempts" in meta["scene"]


def test_generate_dataset_split_bounds_helper():
    assert ml_dataset._split_bounds(6, (0.8, 0.1, 0.1)) == [0, 4, 5, 6]
    assert ml_dataset._split_bounds(10, (0.8, 0.1, 0.1)) == [0, 8, 9, 10]
    assert ml_dataset._split_bounds(0, (0.8, 0.1, 0.1)) == [0, 0, 0, 0]


def test_generate_dataset_is_deterministic(tmp_path, registered_tiny_cfg, torch_device):
    cfg = registered_tiny_cfg
    m1 = ml_dataset.generate_dataset(cfg.name, TIER, 3, out_dir=tmp_path / "a", seed=7, device=torch_device)
    m2 = ml_dataset.generate_dataset(cfg.name, TIER, 3, out_dir=tmp_path / "b", seed=7, device=torch_device)

    man1 = json.loads(m1.read_text())
    man2 = json.loads(m2.read_text())
    assert man1 == man2

    fname = man1["files"]["train"][0]
    with np.load(m1.parent / fname) as d1, np.load(m2.parent / fname) as d2:
        assert d1["adc"].tobytes() == d2["adc"].tobytes()
        assert d1["labels"].tobytes() == d2["labels"].tobytes()
        assert d1["meta"].item() == d2["meta"].item()


# --------------------------------------------------------------------------------
# generate_dataset: frames_per_scene sequences
# --------------------------------------------------------------------------------
def test_frames_per_scene_sequence_layout(tmp_path, registered_tiny_cfg, torch_device):
    cfg = registered_tiny_cfg
    manifest_path = ml_dataset.generate_dataset(
        cfg.name, TIER, 2, out_dir=tmp_path, seed=1, device=torch_device,
        frames_per_scene=3, splits=(1.0, 0.0, 0.0),
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["frames_per_scene"] == 3
    assert len(manifest["sequences"]) == 2  # 2 scenes
    for scene_idx, seq in enumerate(manifest["sequences"]):
        assert len(seq) == 3
        for t, fname in enumerate(seq):
            assert fname == f"frame_{scene_idx:05d}_t{t:02d}.npz"
    # every file in a sequence lands in the SAME split (no leakage across train/val/test).
    all_files = set(manifest["files"]["train"]) | set(manifest["files"]["val"]) | set(manifest["files"]["test"])
    assert all_files == {f for seq in manifest["sequences"] for f in seq}


def test_frames_per_scene_moving_target_matches_velocity(tmp_path, registered_tiny_cfg, torch_device):
    """A MOVING target's (range, sin_az) position, reconstructed frame to frame, must
    shift consistently with its recorded velocity * dt (see `sample_scene`'s Motion
    tracks and `frame_scatterers`'s finite-difference velocity)."""
    cfg = registered_tiny_cfg
    dt = 1.0 / cfg.frame_rate_hz
    manifest_path = ml_dataset.generate_dataset(
        cfg.name, TIER, 1, out_dir=tmp_path, seed=5, device=torch_device,
        frames_per_scene=4, splits=(1.0, 0.0, 0.0),
    )
    manifest = json.loads(manifest_path.read_text())
    seq = manifest["sequences"][0]
    assert len(seq) == 4

    positions = []
    velocities = []
    for fname in seq:
        with np.load(manifest_path.parent / fname) as data:
            meta = json.loads(str(data["meta"].item()))
        assert len(meta["targets"]) == 1  # TIER's first tier (D0) has exactly one vehicle
        r, sin_az, cls = meta["targets"][0]
        assert cls == "vehicle"
        cos_az = math.sqrt(max(0.0, 1.0 - sin_az ** 2))
        positions.append((r * cos_az, r * sin_az))
        velocities.append(meta["target_extras"][0]["velocity_mps"])

    # Velocity is constant across the sequence (constant-velocity motion track).
    for v in velocities[1:]:
        assert v == pytest.approx(velocities[0], abs=1e-6)
    vx, vy = velocities[0][0], velocities[0][1]

    for t in range(len(positions) - 1):
        dx = positions[t + 1][0] - positions[t][0]
        dy = positions[t + 1][1] - positions[t][1]
        assert dx == pytest.approx(vx * dt, abs=1e-3)
        assert dy == pytest.approx(vy * dt, abs=1e-3)


def test_frames_per_scene_zero_raises():
    with pytest.raises(ValueError):
        ml_dataset.generate_dataset("ti_iwr1443", TIER, 1, frames_per_scene=0, out_dir="unused")


# --------------------------------------------------------------------------------
# RadarFrameDataset: v2 (ADC-native) round trip, both input_formats
# --------------------------------------------------------------------------------
def test_radar_frame_dataset_len_getitem_targets_and_split_filter(
    tmp_path, registered_tiny_cfg, torch_device
):
    cfg = registered_tiny_cfg
    manifest_path = ml_dataset.generate_dataset(
        cfg.name, TIER, 6, out_dir=tmp_path, seed=1, device=torch_device,
    )

    train_ds = ml_dataset.RadarFrameDataset(manifest_path, split="train")
    val_ds = ml_dataset.RadarFrameDataset(manifest_path, split="val")
    test_ds = ml_dataset.RadarFrameDataset(manifest_path, split="test")
    assert len(train_ds) == 4
    assert len(val_ds) == 1
    assert len(test_ds) == 1
    assert isinstance(train_ds, torch.utils.data.Dataset)

    x, y = train_ds[0]
    expected_input_shape, expected_labels_shape = _expected_shapes(cfg)
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
    assert x.shape == expected_input_shape and x.dtype == torch.float32
    assert y.shape == expected_labels_shape and y.dtype == torch.float32

    tlist = train_ds.targets(0)
    assert isinstance(tlist, list)
    for t in tlist:
        assert len(t) == 3

    with pytest.raises(ValueError):
        ml_dataset.RadarFrameDataset(manifest_path, split="bogus")


def test_radar_frame_dataset_rd_matches_generate_sample_input(
    tmp_path, registered_tiny_cfg, torch_device
):
    """`input_format="rd"` (default) must re-derive BYTE-IDENTICAL to what
    `generate_sample` used to precompute -- pure deterministic tensor ops, no RNG."""
    cfg = registered_tiny_cfg
    manifest_path = ml_dataset.generate_dataset(
        cfg.name, TIER, 3, out_dir=tmp_path, seed=2, device=torch_device,
    )
    ds = ml_dataset.RadarFrameDataset(manifest_path, split="train", input_format="rd")
    x, _y = ds[0]

    with np.load(manifest_path.parent / ds.files[0]) as data:
        adc = torch.from_numpy(data["adc"]).to(torch.complex64)
    from e2e.ml.transforms import adc_to_rd, rd_to_input, tdm_deinterleave

    if cfg.mimo == "tdm":
        sub_cfg = dataclasses.replace(cfg, n_tx=1, mimo="single", n_chirps=cfg.n_chirps_per_tx)
        rd = adc_to_rd(sub_cfg, tdm_deinterleave(cfg, adc))
    else:
        rd = adc_to_rd(cfg, adc)
    expected = rd_to_input(rd)
    assert torch.equal(x, expected)


def test_radar_frame_dataset_adc_format_shape_and_no_deinterleave(
    tmp_path, registered_tiny_cfg, torch_device
):
    """`input_format="adc"`: raw physical channels, real/imag stacked, NO TDM
    deinterleave (the virtual-array reorder is exactly what this format skips)."""
    cfg = registered_tiny_cfg
    manifest_path = ml_dataset.generate_dataset(
        cfg.name, TIER, 3, out_dir=tmp_path, seed=3, device=torch_device,
    )
    ds = ml_dataset.RadarFrameDataset(manifest_path, split="train", input_format="adc")
    x, y = ds[0]
    assert x.shape == _expected_adc_input_shape(cfg)
    assert x.dtype == torch.float32
    _, expected_labels_shape = _expected_shapes(cfg)
    assert y.shape == expected_labels_shape

    with np.load(manifest_path.parent / ds.files[0]) as data:
        adc = torch.from_numpy(data["adc"]).to(torch.complex64)  # [n_rx, n_chirps, n_samples]
    adc_t = adc.transpose(1, 2)  # [n_rx, n_samples, n_chirps] -- no deinterleave applied
    expected = torch.cat([adc_t.real, adc_t.imag], dim=0).to(torch.float32)
    assert torch.equal(x, expected)
    # n_rx (physical), not n_virtual (post-deinterleave) channels.
    assert x.shape[0] == 2 * cfg.n_rx


def test_radar_frame_dataset_bad_input_format_raises(tmp_path, registered_tiny_cfg, torch_device):
    cfg = registered_tiny_cfg
    manifest_path = ml_dataset.generate_dataset(
        cfg.name, TIER, 2, out_dir=tmp_path, seed=4, device=torch_device,
    )
    with pytest.raises(ValueError):
        ml_dataset.RadarFrameDataset(manifest_path, split="train", input_format="bogus")


def test_radar_frame_dataset_in_memory_cache(tmp_path, registered_tiny_cfg, torch_device):
    cfg = registered_tiny_cfg
    manifest_path = ml_dataset.generate_dataset(
        cfg.name, TIER, 2, out_dir=tmp_path, seed=6, device=torch_device,
    )
    ds = ml_dataset.RadarFrameDataset(manifest_path, split="train", in_memory_cache=True)
    x1, y1 = ds[0]
    assert 0 in ds._cache
    x2, y2 = ds[0]  # served from cache, not re-derived
    assert torch.equal(x1, x2) and torch.equal(y1, y2)


# --------------------------------------------------------------------------------
# RadarFrameDataset: manifest_version-1 back-compat
# --------------------------------------------------------------------------------
def _write_v1_dataset(tmp_path, cfg, torch_device, n=3, seed=0):
    """Hand-build a manifest_version-1-shaped corpus (npz has "input", not "adc") by
    generating a normal v2 corpus and rewriting each npz -- the cheapest way to get a
    faithful v1 fixture without a second code path."""
    manifest_path = ml_dataset.generate_dataset(
        cfg.name, TIER, n, out_dir=tmp_path, seed=seed, device=torch_device,
    )
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("manifest_version", None)  # v1 predates this field
    all_files = manifest["files"]["train"] + manifest["files"]["val"] + manifest["files"]["test"]

    from e2e.ml.transforms import adc_to_rd, rd_to_input, tdm_deinterleave

    for fname in all_files:
        path = manifest_path.parent / fname
        with np.load(path) as data:
            adc = torch.from_numpy(data["adc"]).to(torch.complex64)
            labels_arr = data["labels"]
            meta_str = data["meta"].item()
        if cfg.mimo == "tdm":
            sub_cfg = dataclasses.replace(cfg, n_tx=1, mimo="single", n_chirps=cfg.n_chirps_per_tx)
            rd = adc_to_rd(sub_cfg, tdm_deinterleave(cfg, adc))
        else:
            rd = adc_to_rd(cfg, adc)
        input_arr = rd_to_input(rd).numpy()
        np.savez_compressed(path, input=input_arr, labels=labels_arr, meta=np.array(meta_str))

    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    return manifest_path


def test_radar_frame_dataset_v1_backcompat_rd_loads(tmp_path, registered_tiny_cfg, torch_device):
    cfg = registered_tiny_cfg
    manifest_path = _write_v1_dataset(tmp_path, cfg, torch_device, n=3, seed=8)

    ds = ml_dataset.RadarFrameDataset(manifest_path, split="train", input_format="rd")
    x, y = ds[0]
    expected_input_shape, expected_labels_shape = _expected_shapes(cfg)
    assert x.shape == expected_input_shape and x.dtype == torch.float32
    assert y.shape == expected_labels_shape

    with np.load(manifest_path.parent / ds.files[0]) as data:
        assert "adc" not in data
        assert "input" in data
        assert torch.equal(x, torch.from_numpy(data["input"]).to(torch.float32))


def test_radar_frame_dataset_v1_backcompat_adc_raises(tmp_path, registered_tiny_cfg, torch_device):
    cfg = registered_tiny_cfg
    manifest_path = _write_v1_dataset(tmp_path, cfg, torch_device, n=2, seed=9)

    ds = ml_dataset.RadarFrameDataset(manifest_path, split="train", input_format="adc")
    with pytest.raises(ValueError):
        ds[0]


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------
def test_cli_dry_run_exits_zero_without_writing(tmp_path):
    out_dir = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, "-m", "e2e.ml.dataset",
         "--config", "ti_iwr1443", "--tier", TIER, "--n", "3",
         "--frames-per-scene", "2", "--dry-run", "--out", str(out_dir)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert not out_dir.exists()
    assert "dry-run" in proc.stdout.lower()
    assert "adc shape" in proc.stdout.lower()  # dry-run size estimate reflects ADC, not "input"
    assert "6 frames" in proc.stdout  # 3 scenes x frames_per_scene=2


def test_cli_unknown_config_exits_nonzero():
    proc = subprocess.run(
        [sys.executable, "-m", "e2e.ml.dataset",
         "--config", "not_a_real_config", "--tier", TIER, "--n", "1", "--dry-run"],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0


def test_cli_unknown_tier_exits_nonzero():
    proc = subprocess.run(
        [sys.executable, "-m", "e2e.ml.dataset",
         "--config", "ti_iwr1443", "--tier", "not_a_real_tier", "--n", "1", "--dry-run"],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
