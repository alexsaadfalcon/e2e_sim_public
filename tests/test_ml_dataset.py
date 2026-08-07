"""Tests for `e2e.ml.dataset`: sample synthesis, on-disk dataset generation,
`RadarFrameDataset`, and the CLI.

`e2e.ml.labels` and `e2e.ml.scenes` are sibling shards developed in parallel with this
one -- if either hasn't landed in the working tree yet, this whole module skips
cleanly (rather than erroring) via the `importorskip` calls below.
"""
import dataclasses
import json
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
    assert sample["labels"].shape == expected_labels_shape
    assert sample["labels"].dtype == torch.float32
    assert sample["labels"].device.type == "cpu"
    assert isinstance(sample["targets"], list)
    for t in sample["targets"]:
        assert len(t) == 3
    assert sample["meta"]["config"] == cfg.name
    assert sample["meta"]["mimo"] == "tdm"


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

    files = manifest["files"]
    assert len(files["train"]) == 4
    assert len(files["val"]) == 1
    assert len(files["test"]) == 1
    all_files = files["train"] + files["val"] + files["test"]
    assert len(all_files) == 6
    assert len(set(all_files)) == 6  # no overlap between splits

    expected_input_shape, expected_labels_shape = _expected_shapes(cfg)
    for fname in all_files:
        with np.load(manifest_path.parent / fname) as data:
            assert data["input"].shape == expected_input_shape
            assert data["input"].dtype == np.float32
            assert data["labels"].shape == expected_labels_shape
            assert data["labels"].dtype == np.float32
            meta = json.loads(str(data["meta"].item()))
            assert "targets" in meta
            assert "scene" in meta


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
        assert d1["input"].tobytes() == d2["input"].tobytes()
        assert d1["labels"].tobytes() == d2["labels"].tobytes()
        assert d1["meta"].item() == d2["meta"].item()


# --------------------------------------------------------------------------------
# RadarFrameDataset
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


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------
def test_cli_dry_run_exits_zero_without_writing(tmp_path):
    out_dir = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, "-m", "e2e.ml.dataset",
         "--config", "ti_iwr1443", "--tier", TIER, "--n", "3",
         "--dry-run", "--out", str(out_dir)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert not out_dir.exists()
    assert "dry-run" in proc.stdout.lower()


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
