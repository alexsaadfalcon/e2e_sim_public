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

BOTH ARTIFACTS ARE WRITTEN INCREMENTALLY -- on every validation improvement, not once
at the end -- so a run killed at epoch N still leaves the best weights it reached (a
120-epoch run that lost ~9 GPU-hours to an empty directory is why). CONSEQUENCE FOR
CALLERS: their PRESENCE does not mean the run finished. `best.pt["epochs_completed"]`,
or `len(history["epoch"])` against the epochs you asked for, is what distinguishes a
completed run from an interrupted one -- `e2e.ml.sweep` resumes on exactly that test.

CLI
---
    python -m e2e.ml.train --manifest PATH --model fftradnet|ssmradnet [--epochs 10]
        [--batch-size 8] [--lr 1e-4] [--seed 0] [--reg-weight 100.0] [--gamma 2.0]
        [--input-format rd|adc|rad] [--amp auto|on|off] [--accum-steps 1] [--ssm-chunk N]
        [--out DIR] [--eval-only CKPT] [--split test]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from e2e.ml.dataset import RadarFrameDataset
from e2e.ml.losses import detection_loss
from e2e.ml.metrics import evaluate_dataset
from e2e.radar_config import RadarConfig

_MODEL_NAMES = ("fftradnet", "ssmradnet", "raddetnet")


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


#: Sources whose BEHAVIOUR decides what a checkpoint was trained on: the input tensor it
#: was fed, the loop that fed it, the architecture, and the metric. Canonical list --
#: readers (`e2e.ml.beat_cfar`) import it rather than keeping their own copy.
#:
#: `baseline.py` and `chain/transforms.py` are here because of a reviewed near-miss
#: (2026-09-21): the incident in `pipeline_fingerprint`'s docstring was CAUSED by
#: `range_azimuth_power`'s front-end arguments, which live in `baseline.py`, and the first
#: version of this list omitted it. It caught that incident only because the call site in
#: `dataset.py` happened to move too. Retune a default inside `range_azimuth_power` and the
#: identical bug would have recurred with an unchanged fingerprint.
INPUT_PIPELINE_SOURCES = (
    "e2e/ml/dataset.py",        # assembles the input tensor
    "e2e/ml/baseline.py",       # range_azimuth_power + the front-end constants it applies
    "e2e/chain/transforms.py",  # adc_to_rd / tdm_deinterleave beneath it
    "e2e/ml/labels.py",         # the targets and the grid they live on
    "e2e/ml/train.py",          # the training loop and its per-epoch validation
    "e2e/ml/models",            # architectures
    "e2e/ml/metrics.py",        # the AP definition
)


def _semantic_source(path: Path) -> bytes:
    """A module's code with comments and docstrings removed, as a stable byte string.

    Hashing raw bytes makes the fingerprint trip on every typo fix in a docstring, and a
    guard that forces a 5-hour retrain to punish a comment gets switched off by the next
    person -- which is the real failure. Parsing to an AST drops comments, and dropping
    docstring nodes leaves only what can change a number.

    Falls back to the raw bytes if the file does not parse, because an unparseable source is
    exactly when you want to be conservative rather than clever.
    """
    import ast

    raw = path.read_bytes()
    try:
        tree = ast.parse(raw)
    except SyntaxError:
        return raw
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree).encode()


def pipeline_fingerprint(root=None) -> Optional[str]:
    """SHA-256 over the CODE of `INPUT_PIPELINE_SOURCES`, or None if none are found.

    Recorded in every checkpoint AT TRAINING START, which is the only moment that describes
    what the process actually imported. A reader comparing this against its own current
    fingerprint learns whether the checkpoint is interchangeable with a rerun. Comments and
    docstrings are excluded (see `_semantic_source`), so prose edits during a long run do
    not invalidate it; anything that can move a number does.

    WHY NOT FILE MTIMES (measured, 2026-09-21): `dataset.py` was edited at 17:19 while a run
    trained 16:36-19:36. The checkpoint's final write stamps 19:36, so it is NEWER than the
    edit and an mtime check calls it fresh -- while the running process still held the code
    from 16:36. That run reported val_AP 0.484 and reloaded at 0.023 against the edited
    file; independently confirmed by re-scoring it with only the two changed front-end
    arguments reverted, which reproduces 0.4843. Content hashed at start-of-run is immune to
    that; mtimes are not.

    WHY NOT THE GIT SHA ALONE: the edit above was uncommitted when the run began. A SHA
    describes the last commit, not the working tree that was imported.

    WHAT IT STILL CANNOT SEE: anything these files import from outside the list, the torch /
    cuDNN / driver versions, and the corpus itself. It answers "was this trained by the code
    that is here now", not "will this reproduce anywhere".
    """
    import hashlib

    base = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    h = hashlib.sha256()
    found = False
    for src in INPUT_PIPELINE_SOURCES:
        p = base / src
        if not p.exists():
            continue
        # Sorted so the digest does not depend on filesystem iteration order.
        files = sorted(p.rglob("*.py")) if p.is_dir() else [p]
        for f in files:
            try:
                h.update(f.relative_to(base).as_posix().encode())
                h.update(_semantic_source(f))
                found = True
            except OSError:
                continue
    return h.hexdigest() if found else None


def set_determinism(seed: int, *, strict: bool = True) -> None:
    """Seed every RNG this training path touches, and pin the kernels.

    `torch.manual_seed` alone is NOT enough for a repeatable run. Three other things vary:
    Python's and numpy's RNGs (used by dataset/label code), cuDNN's autotuner (which
    benchmarks algorithms on first sight of a shape and may pick a different one run to
    run), and the nondeterministic reduction order of some CUDA kernels.

    With `strict=True` (the default) this pins all of them, and a run on the SAME machine
    and software stack is bit-identical. What it still does NOT promise -- and no API
    can -- is identity ACROSS different GPU architectures, CUDA/cuDNN versions, or torch
    builds: those change kernel selection and floating-point reduction order beneath any
    seed. Record the hardware alongside any number you intend to reproduce exactly.

    `torch.use_deterministic_algorithms(True)` RAISES on an op with no deterministic
    implementation. That is the point: it fails loudly rather than varying quietly. Pass
    `strict=False` to fall back to seeding only, and say so wherever the number is quoted.

    NOTE `CUBLAS_WORKSPACE_CONFIG` is read by cuBLAS when it initializes, so it must be set
    before the first CUDA op. Setting it here is early enough for this module's CLI; a
    caller that has already run CUDA work should set it in the environment instead.
    """
    import random as _random

    _random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if not strict:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False      # autotuning is itself a source of variance
    torch.use_deterministic_algorithms(True, warn_only=False)


def _seed_worker(worker_id: int) -> None:
    """Per-DataLoader-worker seeding.

    Each worker process re-seeds numpy/random from torch's per-worker seed; without this
    every worker inherits the parent's RNG state and any numpy randomness in the dataset
    is correlated across workers (and varies with `num_workers`). Harmless on the
    `num_workers=0` Windows default, load-bearing everywhere else.
    """
    import random as _random

    s = torch.initial_seed() % 2 ** 32
    try:
        import numpy as _np
        _np.random.seed(s)
    except ImportError:
        pass
    _random.seed(s)


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
    if input_format == "rad":
        # Range-azimuth-Doppler (F83): the classical beamformer's own output cube,
        # `[n_angle, n_range, n_doppler]`, REAL log-power -- so the channel axis is
        # azimuth and is spatially ordered, not arbitrary virtual-element phase, and it
        # is not doubled for re/im. `range_azimuth_power` defaults its angle FFT to the
        # virtual-channel count (no zero-padding: interpolating the angle axis places
        # peaks between resolution cells without adding information).
        n_angle = cfg.n_virtual if cfg.mimo == "tdm" else cfg.n_rx
        n_dop = cfg.n_chirps_per_tx if cfg.mimo == "tdm" else cfg.n_chirps
        return n_angle, cfg.n_samples, n_dop
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
    if input_format not in ("rd", "adc", "rad"):
        raise ValueError(
            f"input_format must be 'rd', 'adc' or 'rad', got {input_format!r}")
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
    elif name == "raddetnet":
        # Range-azimuth-Doppler detector (F83). Needs input_format="rad": its whole
        # premise is that the spatial plane IS (range, azimuth), which the "rd" and
        # "adc" layouts do not provide.
        if input_format != "rad":
            raise ValueError(
                f"raddetnet requires input_format='rad' (it convolves over the "
                f"range-azimuth plane); got {input_format!r}")
        from e2e.ml.models import RADDetNet

        model = RADDetNet(in_channels, n_range_in, n_doppler_in,
                          n_range_out, n_azimuth_out)
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
          deterministic: bool = False,
          input_format: str = "rd", reg_weight: float = 100.0, gamma: float = 2.0,
          cls_normalize: str = "positives", amp="auto", accum_steps: int = 1,
          num_workers: Optional[int] = None, ssm_chunk: Optional[int] = None,
          weight_decay: float = 0.0, extra_manifests: Optional[List] = None) -> Dict:
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
    # Seeds torch/numpy/random, and with deterministic=True also pins cuDNN and
    # cuBLAS so a same-machine rerun is bit-identical. See set_determinism.
    set_determinism(seed, strict=deterministic)

    # Captured HERE, before any epoch runs, because this is what the process imported. See
    # `pipeline_fingerprint` for why the checkpoint's own mtime cannot stand in for it.
    fingerprint = pipeline_fingerprint()

    with open(manifest_path) as f:
        manifest = json.load(f)
    grid = _load_grid(manifest)

    train_ds = _make_dataset(manifest_path, "train", input_format)
    if extra_manifests:
        # Joint training across corpora (generalisation campaign): the train splits of
        # every extra manifest are concatenated with the primary's. Validation and model
        # selection stay on the PRIMARY manifest's val split, so a joint run is scored on
        # the same frames as a single-corpus run and the two are comparable. Each dataset
        # resolves its own input_scale from its own manifest.
        from torch.utils.data import ConcatDataset
        extras = [_make_dataset(Path(m), "train", input_format) for m in extra_manifests]
        train_ds = ConcatDataset([train_ds, *extras])
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
                               generator=gen, num_workers=num_workers,
                               worker_init_fn=_seed_worker)

    manifest_for_model = dict(manifest)
    manifest_for_model["input_format"] = input_format
    model = build_model(model_name, manifest_for_model, device=device, ssm_chunk_size=ssm_chunk)
    # Generalisation campaign (owner, 2026-09-22, after F85's out-of-distribution caveat):
    # decoupled weight decay is the first regulariser tried. AdamW at weight_decay=0 is
    # NOT bit-identical to Adam in torch, so the default path keeps Adam and every
    # existing checkpoint's provenance intact.
    if weight_decay > 0.0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
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

    def _write_checkpoint(state) -> None:
        """Write `best.pt` + `history.json` for the best epoch SO FAR, atomically.

        Called on every validation improvement and once at the end, so a killed run
        leaves the best weights it had actually reached instead of nothing. Writes
        to a temp file and `os.replace`s it (atomic on Windows and POSIX), so an
        interrupted write can never leave a truncated checkpoint where a complete
        one used to be.
        """
        payload = {
            "model_state": state, "model_name": model_name, "input_format": input_format,
            "manifest": str(manifest_path), "history": history,
            # Full training provenance (B2 review, 2026-08-25: a checkpoint that does
            # not record its own hyperparameters -- epochs, batch size, accumulation,
            # lr -- is reproducible only from shell history, which is nothing).
            # `best_epoch` names WHICH epoch the saved weights come from; without it,
            # a truncated run whose best == last is indistinguishable from a converged
            # one. `epochs_completed` distinguishes a finished run from a killed one.
            "train_config": {
                "epochs": int(epochs), "batch_size": int(batch_size), "lr": float(lr),
                "seed": int(seed), "deterministic": bool(deterministic),
                "reg_weight": float(reg_weight), "gamma": float(gamma),
                "cls_normalize": cls_normalize, "amp": str(amp),
                "accum_steps": int(accum_steps), "ssm_chunk": ssm_chunk,
                "weight_decay": float(weight_decay),
                "extra_manifests": [str(m) for m in (extra_manifests or [])],
            },
            # What the input pipeline WAS when this run started, so a later reader can tell
            # whether reloading these weights reproduces the metrics recorded beside them.
            "pipeline_fingerprint": fingerprint,
            "epochs_completed": len(history["epoch"]),
            "best_epoch": (int(history["epoch"][int(max(range(len(history["val_AP"])),
                                                        key=history["val_AP"].__getitem__))])
                           if history.get("val_AP") else None),
            "best_val_AP": (max(history["val_AP"]) if history.get("val_AP") else None),
        }
        tmp = out_dir / "best.pt.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, out_dir / "best.pt")
        tmp_hist = out_dir / "history.json.tmp"
        with open(tmp_hist, "w") as f:
            json.dump(history, f, indent=2)
        os.replace(tmp_hist, out_dir / "history.json")

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
            # Persist EVERY time the best improves, not once at the end: a long run
            # killed at epoch N used to leave an empty directory and forfeit all of
            # it (measured: a 120-epoch SSM run lost ~9 GPU-hours at epoch 106 when
            # its parent process exited, 2026-08-27). The write is atomic, so a kill
            # mid-write cannot corrupt an already-good checkpoint either.
            _write_checkpoint(best_state)

    if best_state is None:  # epochs == 0 edge case: nothing trained, save the initial weights
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    _write_checkpoint(best_state)  # final write: history through the LAST epoch

    return history


# --------------------------------------------------------------------------------
# Evaluation-only entry point
# --------------------------------------------------------------------------------
def load_model_for_eval(manifest_path, checkpoint_path, *, device=None,
                        ssm_chunk_size=None, manifest_for_model=None):
    """Shared checkpoint-reload seam for evaluation paths.

    Opens `manifest_path`, loads `checkpoint_path`, resolves the trained
    `input_format` (`checkpoint.get("input_format", ...)` falling back to the
    manifest's own default, so a manifest edited after training still reconstructs
    what the checkpoint was trained with), builds the model, and loads its weights.

    `manifest_for_model`: optional callable `(manifest, input_format) -> dict`
    supplying the manifest `build_model` sizes the stem from -- the hook exists for
    callers whose model geometry differs from the corpus (e.g. `e2e.ml.afe_sweep`'s
    no-reconstruct path sizing the stem to `M` compressed channels via
    `_manifest_at_m`). Default: the corpus manifest with `input_format` overridden.

    Returns `(model, manifest, grid, input_format)`. Extracted from `evaluate()` /
    `afe_sweep.evaluate_at_m()`, which previously duplicated this sequence verbatim
    (reviewed finding: two places to update in lockstep on any schema change).
    """
    manifest_path = Path(manifest_path)
    device = device if device is not None else _default_device()
    with open(manifest_path) as f:
        manifest = json.load(f)
    grid = _load_grid(manifest)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    input_format = checkpoint.get("input_format", manifest.get("input_format", "rd"))
    if manifest_for_model is None:
        mfm = dict(manifest)
        mfm["input_format"] = input_format
    else:
        mfm = manifest_for_model(manifest, input_format)
    model = build_model(checkpoint["model_name"], mfm, device=device,
                        ssm_chunk_size=ssm_chunk_size)
    model.load_state_dict(checkpoint["model_state"])
    return model, manifest, grid, input_format


def evaluate(manifest_path, checkpoint_path, *, split: str = "test", device=None) -> Dict:
    """Rebuild a model from `checkpoint_path` and run `metrics.evaluate_dataset` on `split`.

    See `load_model_for_eval` for the input-format resolution rule.
    """
    device = device if device is not None else _default_device()
    model, manifest, grid, input_format = load_model_for_eval(
        manifest_path, checkpoint_path, device=device)
    # Class names map 1:1 onto the registry names (FFTRadNet -> fftradnet), so the
    # summary line needs no second checkpoint read.
    model_name = type(model).__name__.lower()

    ds = _make_dataset(manifest_path, split, input_format)
    metrics = _evaluate_split(model, ds, grid, device=device)

    # AP is interpolated precision-recall; AR is recall at ONE operating point, printed
    # with the precision and detection count that make it readable (see e2e.ml.metrics).
    print(f"[{model_name}] {split}: AP={metrics['AP']:.3f}  "
          f"AR={metrics['AR']:.3f} @ score>{metrics['score_threshold']:g}  "
          f"precision={metrics['precision']:.4f}  "
          f"({metrics['tp']} TP / {metrics['n_detections']} det / {metrics['n_targets']} GT)  "
          f"range_rmse_m={metrics['range_rmse_m']:.3f}  "
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
    p.add_argument("--deterministic", action="store_true",
                   help="pin cuDNN/cuBLAS kernels so a rerun on THIS machine is "
                        "bit-identical (see set_determinism). Raises on any op "
                        "lacking a deterministic implementation, deliberately")
    p.add_argument("--input-format", choices=("rd", "adc", "rad"), default="rd",
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
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="decoupled weight decay (AdamW); 0 keeps plain Adam, bit-identical "
                        "to every run before 2026-09-22")
    p.add_argument("--extra-manifest", action="append", default=[], metavar="MANIFEST",
                   help="train jointly on this manifest's train split as well (repeatable); "
                        "validation and model selection stay on --manifest's val split")
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
          seed=args.seed, deterministic=args.deterministic,
          out_dir=args.out, input_format=args.input_format,
          reg_weight=args.reg_weight, gamma=args.gamma, cls_normalize=args.cls_normalize,
          amp={"auto": "auto", "on": True, "off": False}[args.amp], accum_steps=args.accum_steps,
          ssm_chunk=args.ssm_chunk, weight_decay=args.weight_decay,
          extra_manifests=args.extra_manifest or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
