"""Tests for `e2e.ml.train`: model construction, the train/eval loop, and the CLI.

`e2e.ml.dataset`/`labels`/`scenes` are sibling shards -- if any hasn't landed yet, this
whole module skips cleanly via the `importorskip` calls below (matching the pattern in
`tests/test_ml_dataset.py`).

Kept fast: one session-scoped tiny dataset (8 frames, tiny `n_chirps`/`n_samples`) is
generated once and reused by every non-slow test.
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("e2e.ml.labels", reason="sibling shard e2e.ml.labels not present")
pytest.importorskip("e2e.ml.scenes", reason="sibling shard e2e.ml.scenes not present")

from e2e.ml import dataset as ml_dataset
from e2e.ml import train as train_mod
from e2e.ml.radar_config import PRESETS, TI_IWR1443
from e2e.ml.scenes import DIFFICULTY_TIERS

# Any valid tier works for these plumbing tests; D0 (single vehicle/frame) keeps every
# frame non-empty, which is a nicer smoke test than an occasionally-empty tier.
TIER = sorted(DIFFICULTY_TIERS)[0]


@pytest.fixture(scope="session")
def tiny_manifest_path(tmp_path_factory):
    """One small on-disk dataset (manifest + .npz frames), generated once per session.

    Shrunk config (12 chirps, 64 samples/chirp -- copy of the `test_ml_dataset.py`
    shrink pattern) x 8 frames x (0.5, 0.25, 0.25) split -> train=4, val=2, test=2.
    """
    from e2e.blocks import device  # library device (cuda if available, else cpu)

    cfg = dataclasses.replace(TI_IWR1443, name="test_train_tiny_tdm", n_chirps=12, n_samples=64)
    PRESETS[cfg.name] = cfg
    try:
        out_dir = tmp_path_factory.mktemp("ml_train_dataset")
        manifest_path = ml_dataset.generate_dataset(
            cfg.name, TIER, 8, out_dir=out_dir, seed=0, device=device,
            splits=(0.5, 0.25, 0.25),
        )
        yield manifest_path
    finally:
        PRESETS.pop(cfg.name, None)


@pytest.fixture
def manifest_dict(tiny_manifest_path):
    return json.loads(Path(tiny_manifest_path).read_text())


# --------------------------------------------------------------------------------
# build_model
# --------------------------------------------------------------------------------
@pytest.mark.parametrize("model_name", ["fftradnet", "ssmradnet"])
def test_build_model_dims_and_forward(tiny_manifest_path, manifest_dict, model_name):
    from e2e.blocks import device

    ds = ml_dataset.RadarFrameDataset(tiny_manifest_path, split="train")
    x, y = ds[0]

    model = train_mod.build_model(model_name, manifest_dict, device=device)
    assert isinstance(model, torch.nn.Module)

    with torch.no_grad():
        out = model(x.unsqueeze(0).to(device))["detection"]

    assert out.shape[0] == 1
    assert out.shape[1] == 3
    assert tuple(out.shape[2:]) == tuple(y.shape[1:])  # matches the label grid geometry
    assert out.device.type == device.type
    assert torch.isfinite(out).all()


def test_build_model_unknown_name_raises(manifest_dict):
    with pytest.raises(ValueError):
        train_mod.build_model("not_a_real_model", manifest_dict)


# --------------------------------------------------------------------------------
# train() / evaluate()
# --------------------------------------------------------------------------------
def test_train_fftradnet_two_epochs_then_evaluate(tiny_manifest_path, tmp_path):
    out_dir = tmp_path / "run_fft"
    history = train_mod.train(tiny_manifest_path, "fftradnet", epochs=2, batch_size=2,
                               out_dir=out_dir, seed=0)

    assert history["epoch"] == [1, 2]
    for key in ("train_loss", "val_AP", "val_AR", "val_range_rmse_m"):
        assert len(history[key]) == 2
        assert all(math.isfinite(v) for v in history[key])

    best_pt = out_dir / "best.pt"
    history_json = out_dir / "history.json"
    assert best_pt.is_file()
    assert history_json.is_file()
    assert json.loads(history_json.read_text()) == history

    checkpoint = torch.load(best_pt, map_location="cpu")
    assert checkpoint["model_name"] == "fftradnet"
    assert checkpoint["manifest"] == str(tiny_manifest_path)
    assert set(checkpoint) == {"model_state", "model_name", "manifest", "history"}

    metrics = train_mod.evaluate(tiny_manifest_path, best_pt, split="test")
    assert math.isfinite(metrics["AP"])
    assert math.isfinite(metrics["AR"])
    assert math.isfinite(metrics["range_rmse_m"])
    assert math.isfinite(metrics["sin_az_rmse"])


def test_train_ssmradnet_one_epoch(tiny_manifest_path, tmp_path):
    out_dir = tmp_path / "run_ssm"
    history = train_mod.train(tiny_manifest_path, "ssmradnet", epochs=1, batch_size=2,
                               out_dir=out_dir, seed=0)

    assert history["epoch"] == [1]
    assert math.isfinite(history["train_loss"][0])
    assert 0.0 <= history["val_AP"][0] <= 1.0
    assert 0.0 <= history["val_AR"][0] <= 1.0
    assert (out_dir / "best.pt").is_file()
    assert (out_dir / "history.json").is_file()


def test_train_default_out_dir(tiny_manifest_path):
    """No `out_dir` -> `<manifest dir>/runs/<model_name>/`."""
    train_mod.train(tiny_manifest_path, "fftradnet", epochs=1, batch_size=2, seed=0)
    default_dir = Path(tiny_manifest_path).parent / "runs" / "fftradnet"
    assert (default_dir / "best.pt").is_file()
    assert (default_dir / "history.json").is_file()


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------
def test_cli_bad_model_exits_nonzero(tiny_manifest_path):
    with pytest.raises(SystemExit) as exc_info:
        train_mod.main(["--manifest", str(tiny_manifest_path), "--model", "not_a_real_model"])
    assert exc_info.value.code != 0


def test_cli_eval_only(tiny_manifest_path, tmp_path):
    out_dir = tmp_path / "run_cli"
    train_mod.train(tiny_manifest_path, "fftradnet", epochs=1, batch_size=2, out_dir=out_dir, seed=0)
    checkpoint = out_dir / "best.pt"

    rc = train_mod.main([
        "--manifest", str(tiny_manifest_path), "--model", "fftradnet",
        "--eval-only", str(checkpoint), "--split", "test",
    ])
    assert rc == 0


def test_cli_train_argv_wiring(tiny_manifest_path, tmp_path):
    out_dir = tmp_path / "run_cli_train"
    rc = train_mod.main([
        "--manifest", str(tiny_manifest_path), "--model", "ssmradnet",
        "--epochs", "1", "--batch-size", "2", "--seed", "0", "--out", str(out_dir),
    ])
    assert rc == 0
    assert (out_dir / "best.pt").is_file()


# --------------------------------------------------------------------------------
# Slow: full-size config, loss should actually decrease over a few epochs.
# --------------------------------------------------------------------------------
@pytest.mark.slow
def test_train_fftradnet_full_size_loss_decreases(tmp_path_factory):
    from e2e.blocks import device

    out_dir = tmp_path_factory.mktemp("ml_train_full")
    manifest_path = ml_dataset.generate_dataset(
        "ti_iwr1443", TIER, 12, out_dir=out_dir, seed=0, device=device,
    )
    history = train_mod.train(manifest_path, "fftradnet", epochs=3, batch_size=4, seed=0)

    assert len(history["train_loss"]) == 3
    assert history["train_loss"][-1] < history["train_loss"][0]
