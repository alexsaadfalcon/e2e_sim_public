"""
AFE-compression degradation sweep: can a model trained on the full physical aperture
still play nicely once the analog front end has compressed it, and by how much?

THE QUESTION
------------
Corpus frames are stored as raw ADC cubes `[n_rx, n_chirps, n_samples]` -- the physical
receive aperture (`radial_like`: 16 RX). An analog feature-extraction (AFE) stage
combines the physical RX channels *before* digitization -- `M < n_rx` measurements
instead of `n_rx` -- exactly the compression `e2e.chain.compress` models
(`combine`/`reconstruct_aperture`/`quantize_weights`, reused here rather than
reimplemented). A model trained on uncompressed (`M == n_rx`) frames has never seen the
row-space-projected, reconstruction-blurred cube that a compressed AFE hands it; running
that same checkpoint on frames degraded at decreasing `M` quantifies exactly that
mismatch. Because a degradation number is meaningless in isolation, the classical
FFT+CFAR baseline (`e2e.ml.baseline`) is scored on the IDENTICAL degraded frames, so
"the model degrades by X" always sits next to "and the classical floor degrades by Y".

DEGRADATION MODEL (`degrade_adc_cube`)
---------------------------------------
Flatten one cube to `x` = `[n_rx, n_chirps*n_samples]`; draw a deterministic `[M, n_rx]`
complex sensing matrix `A` (`sensing_matrix`, unit-norm rows, seeded by `(seed, M)`);
optionally quantize its entries (`e2e.chain.compress.quantize_weights`, the analog
`WEIGHT_UNIFORM` model -- these are attenuator/phase-shifter settings, not a compute
datapath); combine (`y = A @ x`, `e2e.chain.compress.combine`); reconstruct the minimum-
norm least-squares aperture (`x_hat = pinv(A) @ y`, `e2e.chain.compress.
reconstruct_aperture`); reshape back to `[n_rx, n_chirps, n_samples]`.

STATIC, NOT ADAPTIVE. `A` is drawn once per `(seed, M)` and reused for every frame in
the split -- a fixed analog combining network, matching `e2e.chain.compress.
CompressBlock`'s own static-matrix model. The AFE's ADAPTIVE variant
(`e2e.blocks.AFEBlock`/`MeasurementStage`) redraws its combining matrix per frame from
the subspace tracker's running estimate, which needs the tracker in the loop -- that is
a materially different (and, for a subspace-following consumer, likely more forgiving)
degradation and is deliberately OUT OF SCOPE here; this sweep answers "how bad is a
dumb, fixed compression", not "how bad is the AFE we actually ship".

`M == n_rx` is the CONTROL row: `sensing_matrix` returns the identity (no RNG draw, no
quantization -- a wire has no attenuator setting to quantize), so `degrade_adc_cube`
is an exact pass-through and every metric at `M == n_rx` MUST reproduce the checkpoint's
undegraded evaluation (see `test_afe_sweep.py`'s control-row test and
`e2e.ml.train.evaluate`).

RECONSTRUCTION IS LOSSY for `M < n_rx` (minimum-norm least squares, not an inverse --
see `e2e.chain.compress`'s module docstring) -- that lossiness is the entire point: it
is what a real AFE-then-decompress consumer actually sees.

EVAL PLUMBING
-------------
`_DegradedRadarFrameDataset` subclasses `e2e.ml.dataset.RadarFrameDataset` and overrides
only `_load_raw` -- the one seam between "bytes off disk" and "derive the network input
format" -- to splice `degrade_adc_cube` in between. This is a pure in-memory wrapper:
the on-disk corpus is opened read-only and never touched. `evaluate_at_m` builds the
model from a checkpoint exactly as `e2e.ml.train.evaluate` does, evaluates it on that
degraded dataset via `e2e.ml.train._evaluate_split` (reused, not reimplemented), and
separately runs `e2e.ml.baseline.classical_detection_map` over the SAME degraded
dataset's raw ADC (`.raw_adc(i)`) and the SAME `.targets(i)` ground truth -- one dataset
instance, two scorers, no divergence possible between what the model saw and what the
classical baseline saw.

CLI
---
    python -m e2e.ml.afe_sweep --manifest <corpus>/manifest.json --ckpt <run>/best.pt \\
        [--split val] [--m-list 16 12 8 4 2] [--seed 0] [--weight-bits 8] \\
        [--no-quantize] [--batch-size 8] [--limit N] \\
        [--out report/rt_ml/overnight_0811/afe_sweep/results.json] [--fig PATH]

Writes the results JSON, an AP-vs-M figure (matplotlib Agg backend, savefig only), and
prints a markdown table. DOES NOT MUTATE the corpus or checkpoint directories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch

from e2e.chain.compress import combine, quantize_weights, reconstruct_aperture
from e2e.ml import baseline
from e2e.ml import train as train_mod
from e2e.ml.dataset import RadarFrameDataset
from e2e.ml.metrics import evaluate_dataset
from e2e.ml.radar_config import RadarConfig

DEFAULT_M_LIST = (16, 12, 8, 4, 2)


# --------------------------------------------------------------------------------
# Degradation core
# --------------------------------------------------------------------------------
def sensing_matrix(n_elements: int, m: int, *, seed: int = 0) -> torch.Tensor:
    """Deterministic `[M, n_elements]` complex sensing matrix, unit-norm rows.

    `m == n_elements` is the CONTROL: the identity, drawn without RNG (a physical
    short/open, not an attenuator setting -- see the module docstring). Otherwise a
    seeded complex Gaussian matrix, each row independently normalized to unit L2 norm.
    Deliberately UNLIKE `e2e.chain.compress.CompressBlock`'s own `1/sqrt(2M)`
    energy-preserving scaling (which models one specific hardware calibration this
    sweep does not claim to reproduce) -- unit-norm rows keep each measurement's own
    gain fixed at 1 regardless of `M`, the simpler, architecture-agnostic choice for
    "how degraded is a downstream consumer of M measurements".

    Deterministic per `(seed, m)`: a fresh `torch.Generator` is seeded every call, so
    repeated calls with the same `(n_elements, m, seed)` are bit-identical.
    """
    if m < 1 or m > n_elements:
        raise ValueError(f"m must be in [1, n_elements={n_elements}], got {m}")
    if m == n_elements:
        return torch.eye(m, dtype=torch.complex64)
    g = torch.Generator(device="cpu").manual_seed(seed)
    real = torch.randn(m, n_elements, generator=g)
    imag = torch.randn(m, n_elements, generator=g)
    a = torch.complex(real, imag)
    a = a / a.norm(dim=1, keepdim=True)
    return a.to(torch.complex64)


def degrade_adc_cube(adc: torch.Tensor, m: int, *, seed: int = 0,
                     weight_bits: Optional[int] = 8, quantize: bool = True) -> torch.Tensor:
    """One ADC cube `[n_rx, n_chirps, n_samples]` -> the AFE-degraded cube, same shape.

    `m` is the number of compressed measurements (`M <= n_rx`). See the module
    docstring for the full pipeline (`sensing_matrix` -> optional `quantize_weights`
    -> `combine` -> `reconstruct_aperture`). `weight_bits`/`quantize` expose the
    analog combining-weight resolution (`None` or `quantize=False` -- ideal weights);
    `m == n_rx` (the identity control) is never quantized regardless of these flags,
    since a wire has no attenuator setting to quantize.

    Deterministic per `(seed, m)`. Output dtype matches the input's (normally
    complex64); shape is unchanged.
    """
    if adc.dim() != 3:
        raise ValueError(f"adc must be [n_rx, n_chirps, n_samples], got shape {tuple(adc.shape)}")
    n_rx, n_chirps, n_samples = adc.shape

    a = sensing_matrix(n_rx, m, seed=seed)
    if quantize and weight_bits is not None and m != n_rx:
        a = quantize_weights(a, weight_bits)

    x = adc.reshape(n_rx, n_chirps * n_samples).to(torch.complex64)
    y = combine(a, x)
    x_hat = reconstruct_aperture(a, y)
    return x_hat.reshape(n_rx, n_chirps, n_samples).to(adc.dtype)


# --------------------------------------------------------------------------------
# In-memory degraded dataset (the eval-loop seam)
# --------------------------------------------------------------------------------
class _DegradedRadarFrameDataset(RadarFrameDataset):
    """`RadarFrameDataset` with `degrade_adc_cube` spliced between disk load and
    `input_format` derivation.

    Overrides only `_load_raw` (the seam `RadarFrameDataset._derive_input` consumes) so
    every consumer downstream of it -- `__getitem__` (model input), `targets()`
    (ground truth, deliberately NOT degraded), and `.raw_adc()` (below, for the
    classical baseline) -- sees exactly one degraded cube per frame, drawn once from
    disk. Read-only: `np.load` only, nothing is ever written back to the corpus.
    """

    def __init__(self, manifest_path, split: str, input_format: str, *, m: int,
                seed: int = 0, weight_bits: Optional[int] = 8, quantize: bool = True):
        super().__init__(manifest_path, split=split, input_format=input_format)
        self._afe_m = m
        self._afe_seed = seed
        self._afe_weight_bits = weight_bits
        self._afe_quantize = quantize

    def _load_raw(self, idx: int):
        array, is_adc, labels, meta = super()._load_raw(idx)
        if not is_adc:
            # FAIL FAST, here at the load seam -- not only in raw_adc(). Silently
            # passing a non-ADC frame through would let evaluate_at_m() run the whole
            # (expensive) model-eval pass on UNDEGRADED inputs before the classical
            # stage finally noticed (pre-merge review finding): a caller recording
            # model metrics progressively would log an undegraded AP as if it were
            # the M-degraded number.
            raise ValueError(
                f"{self.files[idx]!r} has no raw ADC on disk (manifest_version 1?) -- "
                "the AFE sweep degrades raw ADC cubes and needs a raw-ADC-native corpus"
            )
        adc = torch.from_numpy(array).to(torch.complex64)
        adc = degrade_adc_cube(adc, self._afe_m, seed=self._afe_seed,
                               weight_bits=self._afe_weight_bits,
                               quantize=self._afe_quantize)
        return adc.numpy(), is_adc, labels, meta

    def raw_adc(self, idx: int) -> torch.Tensor:
        """The degraded raw ADC cube for frame `idx`, `[n_rx, n_chirps, n_samples]`
        complex64 -- undecorated by any `input_format` derivation. What the classical
        baseline (`e2e.ml.baseline.classical_detection_map`) needs, and exactly what
        `__getitem__` derived its model input from for the same frame."""
        array, _is_adc, _labels, _meta = self._load_raw(idx)  # raises on non-ADC
        return torch.from_numpy(array).to(torch.complex64)


# --------------------------------------------------------------------------------
# Per-M evaluation: model + classical, on the SAME degraded frames
# --------------------------------------------------------------------------------
def evaluate_at_m(manifest_path, ckpt_path, split: str, m: int, *, seed: int = 0,
                  weight_bits: Optional[int] = 8, quantize: bool = True, device=None,
                  batch_size: int = 8, limit: Optional[int] = None) -> Dict:
    """Score both the checkpoint and the classical baseline at compression level `M`.

    Returns `{"m": M, "model": {"AP", "AR", "range_rmse_m"}, "classical": {...}}`.
    `limit`, if given, scores only the split's first `limit` frames (fast smoke runs).
    """
    manifest_path = Path(manifest_path)
    device = device if device is not None else train_mod._default_device()
    with open(manifest_path) as f:
        manifest = json.load(f)
    grid = train_mod._load_grid(manifest)
    cfg = RadarConfig.from_dict(manifest["config"])

    checkpoint = torch.load(ckpt_path, map_location=device)
    input_format = checkpoint.get("input_format", manifest.get("input_format", "rd"))
    manifest_for_model = dict(manifest)
    manifest_for_model["input_format"] = input_format
    model = train_mod.build_model(checkpoint["model_name"], manifest_for_model, device=device)
    model.load_state_dict(checkpoint["model_state"])

    ds = _DegradedRadarFrameDataset(manifest_path, split, input_format, m=m, seed=seed,
                                    weight_bits=weight_bits, quantize=quantize)
    if limit is not None:
        ds.files = ds.files[:limit]  # plain list slice; no disk touched

    model_metrics = train_mod._evaluate_split(model, ds, grid, device=device,
                                              batch_size=batch_size)

    pred_maps, target_lists = [], []
    for i in range(len(ds)):
        adc = ds.raw_adc(i).to(device)
        pred_maps.append(baseline.classical_detection_map(cfg, adc, grid).cpu())
        target_lists.append(ds.targets(i))
    classical_metrics = evaluate_dataset(pred_maps, target_lists, grid)

    return {
        "m": int(m),
        "model": {"AP": model_metrics["AP"], "AR": model_metrics["AR"],
                  "range_rmse_m": model_metrics["range_rmse_m"]},
        "classical": {"AP": classical_metrics["AP"], "AR": classical_metrics["AR"],
                      "range_rmse_m": classical_metrics["range_rmse_m"]},
    }


# --------------------------------------------------------------------------------
# Sweep driver
# --------------------------------------------------------------------------------
def run_afe_sweep(manifest_path, ckpt_path, *, split: str = "val",
                  m_list: Sequence[int] = DEFAULT_M_LIST, seed: int = 0,
                  weight_bits: Optional[int] = 8, quantize: bool = True, device=None,
                  batch_size: int = 8, limit: Optional[int] = None) -> Dict:
    """Evaluate the checkpoint and the classical baseline at every `M` in `m_list`.

    Returns the JSON-serializable payload written by the CLI (`{"manifest", "ckpt",
    "model_name", "split", "seed", "weight_bits", "quantize", "n_rx", "m_list",
    "results": [per-M dicts from `evaluate_at_m`, in `m_list` order]}`).
    """
    manifest_path = Path(manifest_path)
    with open(manifest_path) as f:
        manifest = json.load(f)
    cfg = RadarConfig.from_dict(manifest["config"])

    for m in m_list:
        if m > cfg.n_rx:
            raise ValueError(f"--m-list value {m} exceeds this corpus's n_rx={cfg.n_rx}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    results = []
    for m in m_list:
        r = evaluate_at_m(manifest_path, ckpt_path, split, m, seed=seed,
                          weight_bits=weight_bits, quantize=quantize, device=device,
                          batch_size=batch_size, limit=limit)
        results.append(r)
        print(f"  M={m:>3}  model AP={r['model']['AP']:.4f} AR={r['model']['AR']:.4f}  |  "
              f"classical AP={r['classical']['AP']:.4f} AR={r['classical']['AR']:.4f}")

    return {
        "manifest": str(manifest_path),
        "ckpt": str(ckpt_path),
        "model_name": checkpoint["model_name"],
        "split": split,
        "seed": seed,
        "weight_bits": weight_bits,
        "quantize": quantize,
        "n_rx": int(cfg.n_rx),
        "m_list": [int(m) for m in m_list],
        "results": results,
    }


# --------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------
def plot_ap_vs_m(payload: Dict, fig_path) -> Path:
    """AP vs M, two lines (model / classical). Agg backend, savefig only -- no display."""
    import matplotlib

    matplotlib.use("Agg")  # noqa: E402 -- must precede pyplot import; headless/CI-safe
    import matplotlib.pyplot as plt  # noqa: E402

    ms = [r["m"] for r in payload["results"]]
    model_ap = [r["model"]["AP"] for r in payload["results"]]
    classical_ap = [r["classical"]["AP"] for r in payload["results"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ms, model_ap, marker="o", label=f"{payload['model_name']} (model)")
    ax.plot(ms, classical_ap, marker="s", label="classical CFAR")
    ax.set_xlabel("M (compressed channels)")
    ax.set_ylabel("AP")
    ax.set_title(f"AFE compression sweep -- {payload['model_name']} / {payload['split']}")
    ax.invert_xaxis()  # M decreases left -> right: reads as increasing degradation

    positive = [v for v in model_ap + classical_ap if v > 0]
    if positive and max(positive) / min(positive) > 20:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


def print_markdown_table(payload: Dict) -> None:
    print(f"\n### AFE compression sweep: {payload['model_name']} vs classical CFAR "
          f"({payload['split']} split, n_rx={payload['n_rx']})\n")
    print("| M | model AP | model AR | model range_rmse_m | classical AP | classical AR "
          "| classical range_rmse_m |")
    print("|---|---|---|---|---|---|---|")
    for r in payload["results"]:
        mm, cc = r["model"], r["classical"]
        print(f"| {r['m']} | {mm['AP']:.4f} | {mm['AR']:.4f} | {mm['range_rmse_m']:.3f} "
              f"| {cc['AP']:.4f} | {cc['AR']:.4f} | {cc['range_rmse_m']:.3f} |")


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m e2e.ml.afe_sweep",
        description="AFE aperture-compression degradation sweep: model vs classical "
                    "CFAR baseline, scored on the SAME degraded frames.",
    )
    p.add_argument("--manifest", required=True, help="dataset manifest.json (raw-ADC-native)")
    p.add_argument("--ckpt", required=True, help="a train.py-written best.pt checkpoint")
    p.add_argument("--model", choices=("fftradnet", "ssmradnet"), default=None,
                   help="informational only -- the checkpoint records its own model_name; "
                        "a mismatch is printed as a warning, not an error")
    p.add_argument("--split", default="val")
    p.add_argument("--m-list", type=int, nargs="+", default=list(DEFAULT_M_LIST),
                   help="compressed-channel counts to sweep, in report order "
                        "(default: 16 12 8 4 2; include n_rx as the control row)")
    p.add_argument("--seed", type=int, default=0, help="sensing-matrix seed")
    p.add_argument("--weight-bits", type=int, default=8,
                   help="analog combining-weight quantization depth (see quantize_weights)")
    p.add_argument("--no-quantize", action="store_true", help="ideal (unquantized) weights")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="score only the first N frames")
    p.add_argument("--device", default=None,
                   help="torch device, e.g. 'cpu' or 'cuda' (default: library device -- "
                        "cuda if available, else cpu; see e2e.ml.train._default_device)")
    p.add_argument("--out", default="report/rt_ml/overnight_0811/afe_sweep/results.json")
    p.add_argument("--fig", default=None, help="default: <out's dir>/afe_sweep_ap.png")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    checkpoint_name = torch.load(args.ckpt, map_location="cpu").get("model_name")
    if args.model is not None and checkpoint_name is not None and args.model != checkpoint_name:
        print(f"warning: --model {args.model!r} does not match checkpoint's own "
              f"model_name {checkpoint_name!r}; using the checkpoint's")

    device = torch.device(args.device) if args.device is not None else None
    payload = run_afe_sweep(args.manifest, args.ckpt, split=args.split, m_list=args.m_list,
                            seed=args.seed, weight_bits=args.weight_bits,
                            quantize=not args.no_quantize, batch_size=args.batch_size,
                            limit=args.limit, device=device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")

    fig_path = Path(args.fig) if args.fig else out_path.parent / "afe_sweep_ap.png"
    plot_ap_vs_m(payload, fig_path)
    print(f"wrote {fig_path}")

    print_markdown_table(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
