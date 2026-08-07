"""
Training + evaluation entry point for `e2e.ml` radar detection models.

Ties together `e2e.ml.dataset` (`RadarFrameDataset` / a `generate_dataset` manifest),
`e2e.ml.models` (`FFTRadNet`, `SSMRadNet`), `e2e.ml.losses` (`detection_loss`), and
`e2e.ml.metrics` (`evaluate_dataset`) into one reference training script. This is a
tutorial/reference implementation, not a training framework: plain SGD-style Adam, no
LR schedule, no checkpoint resumption, no distributed/mixed-precision support -- read
it top to bottom.

Artifact layout
----------------
`train(manifest_path, model_name, ..., out_dir=None)` writes, under `out_dir`
(default: `<manifest's directory>/runs/<model_name>/`):
  * `best.pt`      -- `torch.save({"model_state": <state_dict, cpu>, "model_name": str,
                       "manifest": str(manifest_path), "history": dict})` for the epoch
                       with the highest validation AP.
  * `history.json` -- `{"epoch": [...], "train_loss": [...], "val_AP": [...],
                        "val_AR": [...], "val_range_rmse_m": [...]}` (one entry/epoch).

CLI
---
    python -m e2e.ml.train --manifest PATH --model fftradnet|ssmradnet [--epochs 10]
        [--batch-size 8] [--lr 1e-4] [--seed 0] [--out DIR] [--eval-only CKPT]
        [--split test]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from e2e.ml.dataset import RadarFrameDataset
from e2e.ml.losses import detection_loss
from e2e.ml.metrics import evaluate_dataset
from e2e.ml.radar_config import RadarConfig

_MODEL_NAMES = ("fftradnet", "ssmradnet")


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _input_dims(cfg: RadarConfig):
    """`(in_channels, n_range_in, n_doppler_in)` for `cfg`.

    Matches the exact `[2*C, R, D]` contract `e2e.ml.dataset.generate_sample` /
    `RadarFrameDataset` produce for this config (see `e2e.ml.dataset`'s module
    docstring, and the same formula in its dry-run CLI / `test_ml_dataset.py`'s
    `_expected_shapes`): TDM de-interleaves to the virtual array first
    (`C = n_virtual`, `D = n_chirps_per_tx`); DDMA/single use the raw ADC
    (`C = n_rx`, `D = n_chirps`). `R` is always `cfg.n_samples`. Deriving this from
    the manifest's own `RadarConfig` avoids loading a dataset sample just to read
    off its shape.
    """
    if cfg.mimo == "tdm":
        c, d = cfg.n_virtual, cfg.n_chirps_per_tx
    else:
        c, d = cfg.n_rx, cfg.n_chirps
    return 2 * c, cfg.n_samples, d


# --------------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------------
def build_model(name: str, manifest: Dict, *, device=None) -> nn.Module:
    """Construct an untrained model matching `manifest`'s input/output geometry.

    `manifest` is a parsed `generate_dataset` manifest dict (e.g.
    `RadarFrameDataset(path).manifest` or `json.load`ed directly) -- its `"config"`
    (a `RadarConfig.to_dict()`) and `"grid"` (a `LabelGrid` dict) entries fully
    determine the model's input/output shapes.

    `name` must be `"fftradnet"` or `"ssmradnet"`; for `fftradnet` on a `mimo=="ddma"`
    config, the DDMA MIMO pre-encoder is selected (`mimo_preencoder="ddma"`,
    `n_tx=cfg.n_tx`) -- otherwise the plain conv-stem path is used (appropriate for
    TDM inputs, where a virtual array has already been formed upstream).
    """
    cfg = RadarConfig.from_dict(manifest["config"])
    in_channels, n_range_in, n_doppler_in = _input_dims(cfg)
    grid = manifest["grid"]
    n_range_out, n_azimuth_out = int(grid["n_range"]), int(grid["n_azimuth"])

    if name == "fftradnet":
        from e2e.ml.models import FFTRadNet

        kwargs = {}
        if cfg.mimo == "ddma":
            kwargs["mimo_preencoder"] = "ddma"
            kwargs["n_tx"] = cfg.n_tx
        model: nn.Module = FFTRadNet(in_channels, n_range_in, n_doppler_in,
                                      n_range_out, n_azimuth_out, **kwargs)
    elif name == "ssmradnet":
        from e2e.ml.models import SSMRadNet

        model = SSMRadNet(in_channels, n_range_in, n_doppler_in, n_range_out, n_azimuth_out)
    else:
        raise ValueError(f"unknown model {name!r}; choices: {_MODEL_NAMES}")

    return model.to(device if device is not None else _default_device())


# --------------------------------------------------------------------------------
# Evaluation helper (shared by train()'s per-epoch val pass and evaluate())
# --------------------------------------------------------------------------------
def _predict_split(model: nn.Module, ds: RadarFrameDataset, *, device, batch_size: int = 8
                    ) -> List[torch.Tensor]:
    """Batched `no_grad` forward pass over every frame in `ds`; returns CPU pred maps."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()
    pred_maps: List[torch.Tensor] = []
    with torch.no_grad():
        for x, _y in loader:
            pred = model(x.to(device))["detection"].cpu()
            pred_maps.extend(pred.unbind(0))
    return pred_maps


def _evaluate_split(model: nn.Module, ds: RadarFrameDataset, grid, *, device,
                     batch_size: int = 8) -> Dict:
    """`metrics.evaluate_dataset` over every frame of `ds` (predictions from `model`)."""
    pred_maps = _predict_split(model, ds, device=device, batch_size=batch_size)
    target_lists = [ds.targets(i) for i in range(len(ds))]
    return evaluate_dataset(pred_maps, target_lists, grid)


def _load_grid(manifest: Dict):
    from e2e.ml.labels import LabelGrid

    g = manifest["grid"]
    return LabelGrid(n_range=int(g["n_range"]), n_azimuth=int(g["n_azimuth"]),
                      max_range_m=float(g["max_range_m"]))


# --------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------
def train(manifest_path, model_name: str, *, epochs: int = 10, batch_size: int = 8,
          lr: float = 1e-4, device=None, out_dir=None, seed: int = 0,
          num_workers: int = 0) -> Dict:
    """Train `model_name` on `manifest_path`'s train split, evaluating on val each epoch.

    Returns the `history` dict (also written to `history.json`); see the module
    docstring for the artifact layout. `drop_last=False` throughout, so this still
    runs (last batch just smaller) on a split with as few as 1 sample.
    """
    manifest_path = Path(manifest_path)
    device = device if device is not None else _default_device()
    torch.manual_seed(seed)

    with open(manifest_path) as f:
        manifest = json.load(f)
    grid = _load_grid(manifest)

    train_ds = RadarFrameDataset(manifest_path, split="train")
    val_ds = RadarFrameDataset(manifest_path, split="val")

    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
                               generator=gen, num_workers=num_workers)

    model = build_model(model_name, manifest, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    out_dir = Path(out_dir) if out_dir is not None else manifest_path.parent / "runs" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    history: Dict[str, list] = {"epoch": [], "train_loss": [], "val_AP": [], "val_AR": [],
                                 "val_range_rmse_m": []}
    best_ap = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)["detection"]
            loss, _parts = detection_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().item())
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        val_metrics = _evaluate_split(model, val_ds, grid, device=device, batch_size=batch_size)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_AP"].append(val_metrics["AP"])
        history["val_AR"].append(val_metrics["AR"])
        history["val_range_rmse_m"].append(val_metrics["range_rmse_m"])

        print(f"[{model_name}] epoch {epoch}/{epochs}  loss={train_loss:.4f}  "
              f"val_AP={val_metrics['AP']:.3f}  val_AR={val_metrics['AR']:.3f}")

        # >= not >: on an AP tie (common when val AP saturates early on easy tiers),
        # keep the LATER, more-converged epoch -- AP's coarse threshold sweep cannot
        # distinguish regression quality, but later epochs have lower loss.
        if val_metrics["AP"] >= best_ap:
            best_ap = val_metrics["AP"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:  # epochs == 0 edge case: nothing trained, save the initial weights
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    torch.save(
        {"model_state": best_state, "model_name": model_name, "manifest": str(manifest_path),
         "history": history},
        out_dir / "best.pt",
    )
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return history


# --------------------------------------------------------------------------------
# Evaluation-only entry point
# --------------------------------------------------------------------------------
def evaluate(manifest_path, checkpoint_path, *, split: str = "test", device=None) -> Dict:
    """Rebuild a model from `checkpoint_path` and run `metrics.evaluate_dataset` on `split`."""
    manifest_path = Path(manifest_path)
    device = device if device is not None else _default_device()

    with open(manifest_path) as f:
        manifest = json.load(f)
    grid = _load_grid(manifest)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(checkpoint["model_name"], manifest, device=device)
    model.load_state_dict(checkpoint["model_state"])

    ds = RadarFrameDataset(manifest_path, split=split)
    metrics = _evaluate_split(model, ds, grid, device=device)

    print(f"[{checkpoint['model_name']}] {split}: AP={metrics['AP']:.3f}  "
          f"AR={metrics['AR']:.3f}  range_rmse_m={metrics['range_rmse_m']:.3f}  "
          f"sin_az_rmse={metrics['sin_az_rmse']:.4f}")
    return metrics


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m e2e.ml.train",
        description="Train/evaluate a radar detection model on an e2e.ml.dataset manifest.",
    )
    p.add_argument("--manifest", required=True, help="path to a generate_dataset manifest.json")
    # Not required with --eval-only: the checkpoint records its own model_name.
    p.add_argument("--model", choices=_MODEL_NAMES, help="model architecture")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="output run directory (default: <manifest dir>/runs/<model>)")
    p.add_argument("--eval-only", default=None, metavar="CKPT",
                   help="skip training; evaluate an existing checkpoint instead")
    p.add_argument("--split", default="test", help="split to evaluate with --eval-only")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.eval_only is not None:
        evaluate(args.manifest, args.eval_only, split=args.split)
        return 0

    if args.model is None:
        parser.error("--model is required when training (omit it only with --eval-only)")
    train(args.manifest, args.model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
          seed=args.seed, out_dir=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
