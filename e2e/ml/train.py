"""
Training + evaluation entry point for `e2e.ml` radar detection models.

Ties together `e2e.ml.dataset` (`RadarFrameDataset` / a `generate_dataset` manifest),
`e2e.ml.models` (`FFTRadNet`, `SSMRadNet`), `e2e.ml.losses` (`detection_loss`), and
`e2e.ml.metrics` (`evaluate_dataset`) into one reference training script. This is a
tutorial/reference implementation, not a training framework: plain SGD-style Adam, no
LR schedule, no checkpoint resumption, no distributed support -- read it top to bottom.
Mixed precision is supported (`amp`, default "auto" = on for CUDA). MEASURED, because
the first version of this note overstated it: on an 8 GiB card SSMRadNet peaks at
6.02 GiB with AMP off and 5.74 GiB with it on -- a 4.6% saving, NOT a halving, and not
enough to buy a larger batch. What actually makes SSMRadNet fit is the BATCH SIZE
(2, not 8): batch 4 does not always raise `OutOfMemoryError` on Windows (the driver's
"system memory fallback" silently spills the excess into host RAM instead), but MEASURED
on this box it turns 1 epoch from 187s/6.0 GiB (batch 2) into 1399s/12.0 GiB-worth of
allocations (batch 4) for the *same* 320-frame split -- a 7.5x slowdown from PCIe-speed
paging, not a clean 2x, and effectively as unusable as an outright OOM. Treat AMP as a
modest speed/memory bonus, and treat "did not raise OutOfMemoryError" as insufficient
evidence a batch size is usable on this hardware.

To reach a larger *effective* batch without more memory, use `accum_steps` (default 1 =
current behavior exactly unchanged): the optimizer step only fires every `accum_steps`
micro-batches of size `batch_size`, with the loss scaled by `1/accum_steps` first so the
accumulated gradient matches a single step over `batch_size * accum_steps` samples. E.g.
`batch_size=2, accum_steps=4` reaches the same effective batch of 8 that OOMs outright at
`batch_size=8`, at roughly `batch_size=2`'s memory footprint.

To reach a larger *materialized* batch (real BatchNorm statistics over 8 samples, unlike
`accum_steps`, which never forms one), use `--ssm-chunk` / `ssm_chunk` (default `None` =
today's unchunked scan; `model="ssmradnet"` only, see
`e2e.ml.models.ssm.selective_scan`'s "CHUNKED SCAN" docs). MEASURED on this same 8 GiB
card / 320-frame split: `batch_size=8, ssm_chunk=None` raises `OutOfMemoryError` (peak
allocation reaches 6.83 GiB before the allocator gives up trying to grow past it);
`batch_size=8, ssm_chunk=128` completes 1 epoch at 5.48 GiB peak / 197s -- i.e. it fits
with room to spare on the same card `batch_size=4` (no chunking) could not use at all.
`ssm_chunk=64` also fits (5.61 GiB / 214s, slightly slower: more, smaller checkpointed
chunks); `ssm_chunk=256` (2 chunks of L=512) does not (OOMs at 6.84 GiB, the same
ballpark as no chunking at all) -- the chunk has to be small enough relative to `L` for
the memory saving to bite. `accum_steps` and `ssm_chunk` are independent: use `ssm_chunk`
first to make a `batch_size` fit outright, `accum_steps` on top of that for a still
larger effective batch.

Artifact layout
----------------
`train(manifest_path, model_name, ..., out_dir=None)` writes, under `out_dir`
(default: `<manifest's directory>/runs/<model_name>/`):
  * `best.pt`      -- `torch.save({"model_state": <state_dict, cpu>, "model_name": str,
                       "input_format": str, "manifest": str(manifest_path),
                       "history": dict})` for the epoch with the highest validation AP.
                       `input_format` pins down which stem `evaluate()` must rebuild
                       even if the manifest's own default has since changed.
  * `history.json` -- `{"epoch": [...], "train_loss": [...], "train_cls_loss": [...],
                        "train_reg_loss": [...], "val_AP": [...], "val_AR": [...],
                        "val_range_rmse_m": [...]}` (one entry/epoch; `train_cls_loss`/
                        `train_reg_loss` are the mean per-epoch `detection_loss` term
                        breakdown, see that function's returned `_parts`).

CLI
---
    python -m e2e.ml.train --manifest PATH --model fftradnet|ssmradnet [--epochs 10]
        [--batch-size 8] [--lr 1e-4] [--seed 0] [--reg-weight 100.0] [--gamma 2.0]
        [--input-format rd|adc] [--amp auto|on|off] [--accum-steps 1] [--ssm-chunk N]
        [--out DIR] [--eval-only CKPT] [--split test]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
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


def _autocast(enabled: bool):
    """`torch.cuda.amp.autocast` when enabled, else a no-op context.

    Wrapped rather than used directly so the training loop reads the same whether or not
    mixed precision is on, and so the CPU path never touches a CUDA-only API.
    """
    if enabled:
        return torch.cuda.amp.autocast()
    return contextlib.nullcontext()


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _input_dims(cfg: RadarConfig, input_format: str = "rd"):
    """`(in_channels, n_range_in, n_doppler_in)` for `cfg` / `input_format`.

    `input_format="rd"` (default) matches the exact `[2*C, R, D]` contract
    `e2e.ml.dataset.generate_sample` / `RadarFrameDataset` produce for this config (see
    `e2e.ml.dataset`'s module docstring, and the same formula in its dry-run CLI /
    `test_ml_dataset.py`'s `_expected_shapes`): TDM de-interleaves to the virtual array
    first (`C = n_virtual`, `D = n_chirps_per_tx`); DDMA/single use the raw ADC
    (`C = n_rx`, `D = n_chirps`). `R` is always `cfg.n_samples`.

    `input_format="adc"` is the raw, un-deinterleaved physical-channel ADC cube
    (derived inline by `e2e.ml.dataset.RadarFrameDataset._derive_input`): `(2 * cfg.n_rx, cfg.n_samples, cfg.n_chirps)`
    regardless of `mimo` -- there is no virtual-array formation on this path (see
    `e2e.ml.models.ssmradnet`'s "Raw-ADC input mode" for why deinterleaving is skipped
    deliberately, not just not-yet-implemented).

    Deriving this from the manifest's own `RadarConfig` avoids loading a dataset sample
    just to read off its shape.
    """
    if input_format == "adc":
        return 2 * cfg.n_rx, cfg.n_samples, cfg.n_chirps
    if cfg.mimo == "tdm":
        c, d = cfg.n_virtual, cfg.n_chirps_per_tx
    else:
        c, d = cfg.n_rx, cfg.n_chirps
    return 2 * c, cfg.n_samples, d


# --------------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------------
def build_model(name: str, manifest: Dict, *, device=None, ssm_chunk_size=None) -> nn.Module:
    """Construct an untrained model matching `manifest`'s input/output geometry.

    `manifest` is a parsed `generate_dataset` manifest dict (e.g.
    `RadarFrameDataset(path).manifest` or `json.load`ed directly) -- its `"config"`
    (a `RadarConfig.to_dict()`) and `"grid"` (a `LabelGrid` dict) entries fully
    determine the model's input/output shapes. `manifest.get("input_format", "rd")`
    selects the raw-ADC vs. range-Doppler input contract (see `_input_dims`); callers
    that want an `input_format` other than the manifest's own default (e.g. `train()`
    threading its own `input_format` argument) should pass a shallow copy of
    `manifest` with `"input_format"` overridden, not mutate the caller's dict.

    `name` must be `"fftradnet"` or `"ssmradnet"`; for `fftradnet` on a `mimo=="ddma"`
    config, the DDMA MIMO pre-encoder is selected (`mimo_preencoder="ddma"`,
    `n_tx=cfg.n_tx`) -- otherwise the plain conv-stem path is used (appropriate for
    TDM inputs, where a virtual array has already been formed upstream). `fftradnet`
    has no raw-ADC path (`e2e.ml.models.fftradnet` is RD-only); `input_format=="adc"`
    with `name=="fftradnet"` raises `ValueError` rather than silently building a model
    that will shape-mismatch on the first batch.

    `ssm_chunk_size` (default `None`, i.e. today's unchunked scan) is forwarded to
    `SSMRadNet(ssm_chunk_size=...)` when `name=="ssmradnet"` -- see
    `e2e.ml.models.ssm.selective_scan`'s "CHUNKED SCAN" docs and `train()`'s
    `ssm_chunk` argument. Ignored for `fftradnet` (no SSM in that architecture).
    """
    cfg = RadarConfig.from_dict(manifest["config"])
    input_format = manifest.get("input_format", "rd")
    if input_format not in ("rd", "adc"):
        raise ValueError(f"input_format must be 'rd' or 'adc', got {input_format!r}")
    in_channels, n_range_in, n_doppler_in = _input_dims(cfg, input_format)
    grid = manifest["grid"]
    n_range_out, n_azimuth_out = int(grid["n_range"]), int(grid["n_azimuth"])

    if name == "fftradnet":
        if input_format == "adc":
            raise ValueError(
                "fftradnet has no raw-ADC input path (RD-only model); use "
                "input_format='rd', or model='ssmradnet' for input_format='adc'"
            )
        from e2e.ml.models import FFTRadNet

        kwargs = {}
        if cfg.mimo == "ddma":
            kwargs["mimo_preencoder"] = "ddma"
            kwargs["n_tx"] = cfg.n_tx
        model: nn.Module = FFTRadNet(in_channels, n_range_in, n_doppler_in,
                                      n_range_out, n_azimuth_out, **kwargs)
    elif name == "ssmradnet":
        from e2e.ml.models import SSMRadNet

        model = SSMRadNet(in_channels, n_range_in, n_doppler_in, n_range_out, n_azimuth_out,
                          input_mode=input_format, ssm_chunk_size=ssm_chunk_size)
    else:
        raise ValueError(f"unknown model {name!r}; choices: {_MODEL_NAMES}")

    return model.to(device if device is not None else _default_device())


def _make_dataset(manifest_path, split: str, input_format: str) -> RadarFrameDataset:
    """`RadarFrameDataset(manifest_path, split=split, input_format=input_format)`.

    Falls back to the pre-`input_format` constructor signature (positional/`split`-only)
    when `input_format=="rd"` and the installed `RadarFrameDataset` does not yet accept
    the kwarg -- this keeps `train.py` working against either version of the sibling
    `e2e.ml.dataset` shard while it lands. `input_format=="adc"` against an old
    `RadarFrameDataset` re-raises: there is nothing sensible to fall back to.
    """
    try:
        return RadarFrameDataset(manifest_path, split=split, input_format=input_format)
    except TypeError:
        if input_format != "rd":
            raise
        return RadarFrameDataset(manifest_path, split=split)


# --------------------------------------------------------------------------------
# Evaluation helper (shared by train()'s per-epoch val pass and evaluate())
# --------------------------------------------------------------------------------
def _predict_split(model: nn.Module, ds: RadarFrameDataset, *, device, batch_size: int = 8,
                    amp: bool = False) -> List[torch.Tensor]:
    """Batched `no_grad` forward pass over every frame in `ds`; returns CPU pred maps.

    `amp` mirrors the training pass. Validation runs every epoch at the same batch size,
    so evaluating in fp32 while training autocasts would leave the eval pass as the
    memory high-water mark on a card that needed AMP in the first place. Predictions are
    cast back to float32 before leaving, so downstream metrics see one dtype regardless.
    """
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()
    pred_maps: List[torch.Tensor] = []
    with torch.no_grad():
        for x, _y in loader:
            with _autocast(amp):
                pred = model(x.to(device))["detection"]
            pred_maps.extend(pred.float().cpu().unbind(0))
    return pred_maps


def _evaluate_split(model: nn.Module, ds: RadarFrameDataset, grid, *, device,
                     batch_size: int = 8, amp: bool = False) -> Dict:
    """`metrics.evaluate_dataset` over every frame of `ds` (predictions from `model`)."""
    pred_maps = _predict_split(model, ds, device=device, batch_size=batch_size, amp=amp)
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
def release_gpu_memory() -> None:
    """Drop torch's cached GPU allocation.

    Several models trained back to back in ONE process share torch's caching allocator,
    and the cache is not returned to the driver between runs. An 8 GB card that has just
    finished a convolutional detector can therefore refuse a state-space model that would
    fit on its own -- which is exactly how an overnight sweep lost every SSMRadNet run to
    an out-of-memory error while each model ran fine in isolation. Call this between runs.
    """
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _accum_group_len(i: int, n_batches: int, accum: int) -> int:
    """Micro-batch count of the accumulation group containing batch index `i`.

    Groups are consecutive runs of `accum` batches; the epoch's final group is
    whatever remains (`n_batches % accum`, when nonzero). The train loop divides each
    micro-batch loss by ITS group's length -- dividing the partial tail group by the
    full `accum` would systematically under-weight the epoch's last examples. Pure
    function so that scaling is unit-testable (see test_ml_train.py)."""
    group_start = (i // accum) * accum
    return min(accum, n_batches - group_start)


def train(manifest_path, model_name: str, *, epochs: int = 10, batch_size: int = 8,
          lr: float = 1e-4, device=None, out_dir=None, seed: int = 0,
          input_format: str = "rd", reg_weight: float = 100.0, gamma: float = 2.0,
          cls_normalize: str = "positives", amp="auto", accum_steps: int = 1,
          num_workers: Optional[int] = None, ssm_chunk: Optional[int] = None) -> Dict:
    """Train `model_name` on `manifest_path`'s train split, evaluating on val each epoch.

    `input_format` ("rd" default | "adc") selects the range-Doppler vs. raw-ADC input
    contract (see `_input_dims` / `e2e.ml.models.ssmradnet`'s "Raw-ADC input mode");
    it overrides the manifest's own `input_format` for both the dataset and the model
    built from it (a manifest may hold both formats' derivable inputs -- see
    `e2e.ml.dataset` -- so this is a legitimate per-run choice, not just a mirror of
    the manifest). `reg_weight`/`gamma`/`cls_normalize` are forwarded to
    `detection_loss` (see that function and `e2e.ml.losses`'s module docstring for the
    upstream-inherited defaults `reg_weight=100, gamma=2` this overrides).
    `cls_normalize="none"` reproduces pre-2026-08-10 runs, which collapsed to an
    all-background predictor -- see `detection_loss`'s "WHY THE DEFAULT CHANGED".

    `accum_steps` (default 1, i.e. exactly today's behavior) groups `accum_steps`
    consecutive `batch_size` micro-batches into one optimizer step, loss-scaled by
    `1/accum_steps` so the accumulated gradient matches training at the effective batch
    `batch_size * accum_steps` -- see the module docstring's memory note for why this,
    not a bigger `batch_size`, is the lever for models (SSMRadNet) whose activation
    memory scales with batch size steeply enough that "bigger batch" means "silent
    system-memory-fallback thrashing" on an 8 GiB card, not a clean OOM.

    `ssm_chunk` (default `None`, `model_name=="ssmradnet"` only) is forwarded to
    `build_model(..., ssm_chunk_size=ssm_chunk)` -- see
    `e2e.ml.models.ssm.selective_scan`'s "CHUNKED SCAN" docs. It is an *additional*
    lever alongside `accum_steps`/`batch_size`: `accum_steps` reaches a larger
    *effective* batch at a small `batch_size`'s memory cost by never materializing a
    large batch at all, whereas `ssm_chunk` lets a `batch_size` that would otherwise
    OOM fit by trading the scan's own peak memory for recompute, so a genuinely larger
    *materialized* batch (bigger BatchNorm statistics per step, unlike accumulation)
    becomes affordable. Ignored (but harmless) for `model_name=="fftradnet"`, which has
    no SSM.

    Returns the `history` dict (also written to `history.json`); see the module
    docstring for the artifact layout. `drop_last=False` throughout, so this still
    runs (last batch just smaller) on a split with as few as 1 sample.
    """
    if accum_steps < 1:
        raise ValueError(f"accum_steps must be >= 1, got {accum_steps}")
    release_gpu_memory()
    manifest_path = Path(manifest_path)
    device = device if device is not None else _default_device()
    torch.manual_seed(seed)

    with open(manifest_path) as f:
        manifest = json.load(f)
    grid = _load_grid(manifest)

    train_ds = _make_dataset(manifest_path, "train", input_format)
    val_ds = _make_dataset(manifest_path, "val", input_format)

    if num_workers is None:
        # input_format=="rd" pays a real per-sample CPU FFT (adc_to_rd/tdm_deinterleave)
        # that DataLoader prefetch workers can overlap with the GPU step (benchmarked
        # ~4.6ms/frame, see adc_design notes); "adc" pays no such transform. Windows
        # multiprocessing DataLoader workers use `spawn` and need the *importing*
        # script guarded by `if __name__ == "__main__"` -- true for this module's own
        # CLI entry point, not guaranteed for embedding callers (tests, notebooks) --
        # and add real per-worker startup cost on the small tiers this reference
        # script trains on, so default 0 on Windows regardless of input_format.
        num_workers = 2 if (input_format == "rd" and sys.platform != "win32") else 0

    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
                               generator=gen, num_workers=num_workers)

    manifest_for_model = dict(manifest)
    manifest_for_model["input_format"] = input_format
    model = build_model(model_name, manifest_for_model, device=device, ssm_chunk_size=ssm_chunk)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    out_dir = Path(out_dir) if out_dir is not None else manifest_path.parent / "runs" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mixed precision. "auto" enables it on CUDA only. Measured on this box's 8 GiB
    # cards: SSMRadNet peaks at 6.02 GiB with AMP off, 5.74 GiB with it on -- 4.6%, not
    # the halving the activation-memory argument would predict (its footprint is not
    # activation-dominated), and batch 4 thrashes into Windows' system-memory fallback
    # with or without it (see module docstring). Batch size (with accum_steps to reach a
    # larger effective batch) is the lever that decides whether it fits; AMP is a bonus.
    use_amp = (device.type == "cuda") if amp == "auto" else bool(amp)
    if use_amp and device.type != "cuda":
        raise ValueError("amp=True requires a CUDA device; got " + str(device))
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    history: Dict[str, list] = {"epoch": [], "train_loss": [], "train_cls_loss": [],
                                 "train_reg_loss": [], "val_AP": [], "val_AR": [],
                                 "val_range_rmse_m": []}
    best_ap = -1.0
    best_state = None

    n_train_batches = len(train_loader)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_cls, total_reg, n_batches = 0.0, 0.0, 0.0, 0
        optimizer.zero_grad()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            with _autocast(use_amp):
                pred = model(x)["detection"]
                loss, parts = detection_loss(pred, y, gamma=gamma, reg_weight=reg_weight,
                                             cls_normalize=cls_normalize)
            # Scale by 1/(this group's ACTUAL micro-batch count) so accumulated grads
            # average the group's members (accum_steps=1: no-op, loss unchanged).
            # The epoch's last group can be PARTIAL (n_train_batches % accum_steps
            # micro-batches); dividing those by the full accum_steps would
            # systematically under-weight the tail examples every epoch (caught in
            # pre-merge review with an empirical repro).
            scaled_loss = loss / _accum_group_len(i, n_train_batches, accum_steps)
            if scaler is not None:
                # Half-precision gradients underflow to zero without loss scaling; the
                # scaler also skips the step on any inf/NaN it detects.
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            # Step every accum_steps micro-batches, and on the last (possibly partial)
            # group of an epoch so no accumulated gradient is ever silently dropped.
            is_accum_boundary = (i + 1) % accum_steps == 0 or (i + 1) == n_train_batches
            if is_accum_boundary:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            total_loss += float(loss.detach().item())
            total_cls += parts["cls"]
            total_reg += parts["reg"]
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)
        train_cls_loss = total_cls / max(n_batches, 1)
        train_reg_loss = total_reg / max(n_batches, 1)

        val_metrics = _evaluate_split(model, val_ds, grid, device=device,
                                       batch_size=batch_size, amp=use_amp)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_cls_loss"].append(train_cls_loss)
        history["train_reg_loss"].append(train_reg_loss)
        history["val_AP"].append(val_metrics["AP"])
        history["val_AR"].append(val_metrics["AR"])
        history["val_range_rmse_m"].append(val_metrics["range_rmse_m"])

        print(f"[{model_name}] epoch {epoch}/{epochs}  loss={train_loss:.4f}  "
              f"(cls={train_cls_loss:.4f} reg={train_reg_loss:.4f})  "
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
        {"model_state": best_state, "model_name": model_name, "input_format": input_format,
         "manifest": str(manifest_path), "history": history},
        out_dir / "best.pt",
    )
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return history


# --------------------------------------------------------------------------------
# Evaluation-only entry point
# --------------------------------------------------------------------------------
def evaluate(manifest_path, checkpoint_path, *, split: str = "test", device=None) -> Dict:
    """Rebuild a model from `checkpoint_path` and run `metrics.evaluate_dataset` on `split`.

    Uses `checkpoint.get("input_format", ...)` (falling back to the manifest's own
    default for older checkpoints saved before this key existed) rather than the
    manifest's current `input_format`, so a manifest edited after training still
    reconstructs the model/dataset the checkpoint was actually trained with.
    """
    manifest_path = Path(manifest_path)
    device = device if device is not None else _default_device()

    with open(manifest_path) as f:
        manifest = json.load(f)
    grid = _load_grid(manifest)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    input_format = checkpoint.get("input_format", manifest.get("input_format", "rd"))
    manifest_for_model = dict(manifest)
    manifest_for_model["input_format"] = input_format
    model = build_model(checkpoint["model_name"], manifest_for_model, device=device)
    model.load_state_dict(checkpoint["model_state"])

    ds = _make_dataset(manifest_path, split, input_format)
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
    p.add_argument("--input-format", choices=("rd", "adc"), default="rd",
                   help="network input contract: range-Doppler (default) or raw ADC "
                        "(see e2e.ml.models.ssmradnet's 'Raw-ADC input mode'; fftradnet "
                        "has no adc path)")
    p.add_argument("--reg-weight", type=float, default=100.0,
                   help="detection_loss regression-term weight (see e2e.ml.losses)")
    p.add_argument("--gamma", type=float, default=2.0,
                   help="focal-loss gamma for the classification term (0 == plain BCE)")
    p.add_argument("--cls-normalize", choices=("positives", "none"), default="positives",
                   help="scale the summed focal term by the positive-cell count "
                        "(default) or not at all; 'none' reproduces pre-2026-08-10 runs, "
                        "which collapsed to an all-background predictor (see e2e.ml.losses)")
    p.add_argument("--amp", choices=("auto", "on", "off"), default="auto",
                   help="mixed precision: 'auto' (default) enables it on CUDA. Measured "
                        "saving on SSMRadNet is ~5%% of peak memory (6.02 -> 5.74 GiB), "
                        "not a halving; batch size is what decides whether it fits")
    p.add_argument("--accum-steps", type=int, default=1,
                   help="gradient accumulation: group this many batch_size micro-batches "
                        "into one optimizer step (default 1 = unchanged behavior), reaching "
                        "effective batch batch_size*accum_steps without more peak memory "
                        "(see module docstring; the lever for models like SSMRadNet where "
                        "a bigger batch_size alone risks Windows system-memory-fallback "
                        "thrashing on an 8 GiB card)")
    p.add_argument("--ssm-chunk", type=int, default=None,
                   help="model='ssmradnet' only: chunk size for the selective-scan's "
                        "chunked+checkpointed evaluation order (default None = "
                        "original unchunked scan). Trades scan compute (~2x) for peak "
                        "activation memory (see e2e.ml.models.ssm's 'CHUNKED SCAN' "
                        "docs); lets a larger --batch-size fit that would otherwise "
                        "OOM. Ignored for model='fftradnet'.")
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
          seed=args.seed, out_dir=args.out, input_format=args.input_format,
          reg_weight=args.reg_weight, gamma=args.gamma, cls_normalize=args.cls_normalize,
          amp={"auto": "auto", "on": True, "off": False}[args.amp], accum_steps=args.accum_steps,
          ssm_chunk=args.ssm_chunk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
