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
    assert set(checkpoint) == {"model_state", "model_name", "manifest", "history",
                               "input_format"}

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
# accum_steps (gradient accumulation): see train.py's module docstring for why this,
# not a bigger batch_size, is the memory lever for SSMRadNet on an 8 GiB card.
# --------------------------------------------------------------------------------
def test_train_accum_steps_default_matches_unset_step_count(tiny_manifest_path, tmp_path,
                                                             monkeypatch):
    """`accum_steps` unset (default 1) is bit-for-bit the same optimizer-step schedule
    as passing `accum_steps=1` explicitly -- guards against the new arg silently
    changing default behavior. Forced onto CPU: on CUDA, `amp="auto"` routes `step()`
    through `GradScaler.step()`, whose internal found-inf skip logic (needed for AMP,
    irrelevant to accum_steps) makes a naive `Adam.step` call-count an unreliable probe."""
    real_step = torch.optim.Adam.step
    counts = {"n": 0}

    def counting_step(self, *args, **kwargs):
        counts["n"] += 1
        return real_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", counting_step)
    train_mod.train(tiny_manifest_path, "fftradnet", epochs=2, batch_size=2,
                     out_dir=tmp_path / "default", seed=0, device=torch.device("cpu"))
    # train split has 4 frames -> 2 micro-batches/epoch, accum_steps=1 -> 1 step/micro-batch.
    assert counts["n"] == 2 * 2


def test_train_accum_steps_step_count_matches_effective_batch(tiny_manifest_path, manifest_dict,
                                                               tmp_path, monkeypatch):
    """`batch_size=2, accum_steps=2` (4 train frames -> 2 micro-batches/epoch, grouped
    into 1 step) must take exactly as many optimizer steps per epoch as the
    unaccumulated equivalent `batch_size=4, accum_steps=1` (1 micro-batch/epoch, 1
    step) -- both are effective-batch-4. Loss trajectories are not expected to match
    (different micro-batch schedules -> different BatchNorm running stats), so this
    checks step COUNT and that parameters actually moved, not float equality. CPU-forced
    for the same reason as the test above (AMP's GradScaler.step() skip logic)."""
    real_step = torch.optim.Adam.step
    cpu = torch.device("cpu")

    def _run(batch_size, accum_steps, out_dir):
        counts = {"n": 0}

        def counting_step(self, *args, **kwargs):
            counts["n"] += 1
            return real_step(self, *args, **kwargs)

        monkeypatch.setattr(torch.optim.Adam, "step", counting_step)
        try:
            train_mod.train(tiny_manifest_path, "fftradnet", epochs=3, batch_size=batch_size,
                            accum_steps=accum_steps, out_dir=out_dir, seed=0, device=cpu)
        finally:
            monkeypatch.setattr(torch.optim.Adam, "step", real_step)
        return counts["n"]

    steps_unaccum = _run(4, 1, tmp_path / "unaccum")   # 1 micro-batch/epoch, no accumulation
    steps_accum = _run(2, 2, tmp_path / "accum")       # 2 micro-batches/epoch, grouped by 2
    assert steps_unaccum == steps_accum == 3            # 1 step/epoch x 3 epochs, either way

    # Params actually moved from their (seed-reproducible) initial values in both runs.
    torch.manual_seed(0)
    initial = train_mod.build_model("fftradnet", manifest_dict, device=cpu)
    initial_state = initial.state_dict()
    for tag in ("unaccum", "accum"):
        ckpt = torch.load(tmp_path / tag / "best.pt", map_location="cpu")
        trained_state = ckpt["model_state"]
        moved = any(
            not torch.equal(trained_state[k], initial_state[k]) for k in initial_state
        )
        assert moved, f"{tag} run: parameters identical to initialization"


def test_train_accum_steps_rejects_non_positive(tiny_manifest_path):
    with pytest.raises(ValueError, match="accum_steps"):
        train_mod.train(tiny_manifest_path, "fftradnet", epochs=1, batch_size=2,
                        accum_steps=0, seed=0)


def test_cli_accum_steps_default_and_wiring(monkeypatch):
    captured = {}

    def _fake_train(manifest_path, model_name, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(train_mod, "train", _fake_train)
    train_mod.main(["--manifest", "dummy_manifest.json", "--model", "fftradnet"])
    assert captured["accum_steps"] == 1

    train_mod.main(["--manifest", "dummy_manifest.json", "--model", "fftradnet",
                    "--accum-steps", "4"])
    assert captured["accum_steps"] == 4


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


# --------------------------------------------------------------------------------
# W1b additions: input_format="adc" plumbing (build_model / train() / CLI) and the
# reg_weight/gamma loss-part logging. Deliberately decoupled from the on-disk
# `e2e.ml.dataset` generation pipeline (a concurrently-changing sibling shard) via a
# minimal stub `RadarFrameDataset` -- see `_StubRadarFrameDataset`'s docstring.
# --------------------------------------------------------------------------------
class _StubRadarFrameDataset(torch.utils.data.Dataset):
    """A `RadarFrameDataset`-shaped stand-in that needs no on-disk `.npz` corpus.

    Sizes its random tensors from `manifest["config"]`/`manifest["grid"]` alone (via
    `train_mod._input_dims`), matching the real `RadarFrameDataset(manifest_path,
    split=..., input_format=...)` constructor contract that `train._make_dataset`
    calls -- this lets the `train.py`-focused tests below exercise the real
    `input_format` plumbing without depending on `e2e.ml.dataset`'s on-disk generation
    pipeline landing/being in a consistent state.
    """

    _SPLIT_SIZES = {"train": 4, "val": 2, "test": 2}
    _SPLIT_SEEDS = {"train": 0, "val": 1, "test": 2}

    def __init__(self, manifest_path, split: str = "train", input_format: str = "rd"):
        from e2e.ml.radar_config import RadarConfig  # local: not imported at module scope here

        with open(manifest_path) as f:
            manifest = json.load(f)
        cfg = RadarConfig.from_dict(manifest["config"])
        c, r, d = train_mod._input_dims(cfg, input_format)
        grid = manifest["grid"]
        n_range_out, n_azimuth_out = int(grid["n_range"]), int(grid["n_azimuth"])

        n = self._SPLIT_SIZES[split]
        gen = torch.Generator().manual_seed(self._SPLIT_SEEDS[split])
        self._x = torch.randn(n, c, r, d, generator=gen)
        self._y = torch.zeros(n, 3, n_range_out, n_azimuth_out)
        self._y[:, 0, 0, 0] = 1.0  # one positive cell/frame -> the reg term is non-trivial

    def __len__(self) -> int:
        return self._x.shape[0]

    def __getitem__(self, idx: int):
        return self._x[idx], self._y[idx]

    def targets(self, idx: int):
        return []


def _write_stub_manifest(tmp_path, *, input_format: str = "adc", n_chirps: int = 12,
                         n_samples: int = 64) -> Path:
    """A minimal manifest.json (config + grid only) for `_StubRadarFrameDataset` tests."""
    cfg = dataclasses.replace(TI_IWR1443, name=f"test_stub_{input_format}", n_chirps=n_chirps,
                              n_samples=n_samples)
    manifest = {
        "config": cfg.to_dict(),
        "grid": {"n_range": 8, "n_azimuth": 12, "max_range_m": 40.0},
        "input_format": input_format,
        "files": {"train": [], "val": [], "test": []},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


def test_input_dims_adc_format_is_raw_physical_channels():
    cfg = dataclasses.replace(TI_IWR1443, name="test_input_dims_adc", n_chirps=12, n_samples=64)
    c, r, d = train_mod._input_dims(cfg, "adc")
    assert (c, r, d) == (2 * cfg.n_rx, cfg.n_samples, cfg.n_chirps)
    # regression: "rd" formula is untouched (TDM de-interleaves to the virtual array)
    c_rd, r_rd, d_rd = train_mod._input_dims(cfg, "rd")
    assert (c_rd, r_rd, d_rd) == (2 * cfg.n_virtual, cfg.n_samples, cfg.n_chirps_per_tx)


def test_build_model_ssmradnet_adc_format_dims_and_mode():
    cfg = dataclasses.replace(TI_IWR1443, name="test_build_model_adc", n_chirps=12, n_samples=64)
    manifest = {
        "config": cfg.to_dict(),
        "grid": {"n_range": 8, "n_azimuth": 12, "max_range_m": 40.0},
        "input_format": "adc",
    }
    model = train_mod.build_model("ssmradnet", manifest, device=torch.device("cpu"))
    assert model.input_mode == "adc"
    assert (model.in_channels, model.n_range_in, model.n_doppler_in) == (
        2 * cfg.n_rx, cfg.n_samples, cfg.n_chirps,
    )
    assert isinstance(model.doppler_pool, torch.nn.Identity)


def test_build_model_fftradnet_rejects_adc_format():
    cfg = dataclasses.replace(TI_IWR1443, name="test_build_model_adc_fft", n_chirps=12,
                              n_samples=64)
    manifest = {
        "config": cfg.to_dict(),
        "grid": {"n_range": 8, "n_azimuth": 12, "max_range_m": 40.0},
        "input_format": "adc",
    }
    with pytest.raises(ValueError, match="raw-ADC"):
        train_mod.build_model("fftradnet", manifest, device=torch.device("cpu"))


def test_train_one_epoch_input_format_adc_with_stub_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(train_mod, "RadarFrameDataset", _StubRadarFrameDataset)
    manifest_path = _write_stub_manifest(tmp_path, input_format="adc")
    out_dir = tmp_path / "run_adc"

    history = train_mod.train(manifest_path, "ssmradnet", epochs=1, batch_size=2,
                               input_format="adc", out_dir=out_dir, seed=0)

    assert history["epoch"] == [1]
    for key in ("train_loss", "train_cls_loss", "train_reg_loss", "val_AP", "val_AR",
                "val_range_rmse_m"):
        assert len(history[key]) == 1
        assert math.isfinite(history[key][0])

    checkpoint = torch.load(out_dir / "best.pt", map_location="cpu")
    assert checkpoint["model_name"] == "ssmradnet"
    assert checkpoint["input_format"] == "adc"

    history_json = json.loads((out_dir / "history.json").read_text())
    assert "train_cls_loss" in history_json and "train_reg_loss" in history_json


def test_train_threads_reg_weight_and_gamma_into_detection_loss(tmp_path, monkeypatch):
    """`train(..., reg_weight=..., gamma=...)` must reach `detection_loss` unchanged."""
    monkeypatch.setattr(train_mod, "RadarFrameDataset", _StubRadarFrameDataset)
    manifest_path = _write_stub_manifest(tmp_path, input_format="adc")

    calls = []
    real_detection_loss = train_mod.detection_loss

    def _spy(pred, target, **kwargs):
        calls.append(kwargs)
        return real_detection_loss(pred, target, **kwargs)

    monkeypatch.setattr(train_mod, "detection_loss", _spy)
    train_mod.train(manifest_path, "ssmradnet", epochs=1, batch_size=2, input_format="adc",
                    reg_weight=7.5, gamma=1.5, seed=0)

    assert calls, "detection_loss was never called"
    assert all(c["reg_weight"] == 7.5 and c["gamma"] == 1.5 for c in calls)


def test_cli_threads_reg_weight_gamma_and_input_format(monkeypatch):
    """`--reg-weight`/`--gamma`/`--input-format` reach `train()` unchanged (CLI wiring)."""
    captured = {}

    def _fake_train(manifest_path, model_name, **kwargs):
        captured["manifest_path"] = manifest_path
        captured["model_name"] = model_name
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(train_mod, "train", _fake_train)
    rc = train_mod.main([
        "--manifest", "dummy_manifest.json", "--model", "ssmradnet",
        "--reg-weight", "7.5", "--gamma", "1.5", "--input-format", "adc",
    ])

    assert rc == 0
    assert captured["model_name"] == "ssmradnet"
    assert captured["reg_weight"] == 7.5
    assert captured["gamma"] == 1.5
    assert captured["input_format"] == "adc"


def test_cli_input_format_default_is_rd(monkeypatch):
    captured = {}

    def _fake_train(manifest_path, model_name, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(train_mod, "train", _fake_train)
    train_mod.main(["--manifest", "dummy_manifest.json", "--model", "fftradnet"])
    assert captured["input_format"] == "rd"
    assert captured["reg_weight"] == 100.0
    assert captured["gamma"] == 2.0
