"""
Compressed-domain-v1 campaign: can a detector work IN the compressed measurement domain
-- no reconstruction anywhere in the loop -- rather than reconstruct-then-detect?

THE QUESTION
------------
`e2e.ml.afe_sweep` measured that a full-aperture checkpoint, run on frames RECONSTRUCTED
from `M < n_rx` compressed measurements, collapses ~50x by `M=12` (see
`report/rt_ml/overnight_0811/afe_sweep/fftradnet_results.json`: val AP 0.0429 at M=16 ->
0.00083 at M=12). That conflates two different things: (1) minimum-norm reconstruction
is itself lossy for `M < n_rx` (`e2e.chain.compress`'s module docstring), and (2) the
checkpoint was never trained on reconstruction-blurred inputs. This module isolates (1)
by removing reconstruction from the loop entirely: train a NEW model, from scratch, with
a stem sized to consume the `M`-channel measurement cube `y = A x` directly.

The physics argument for why this can work at all is `e2e.blocks.RangeProfileBlock`'s
docstring: compression's sensing matrix `A` mixes only the APERTURE axis (dim 0); it
never touches frequency or slow-time, so a range/Doppler FFT commutes straight through
it -- `range(y[m, :]) = sum_n A[m, n] * range(x[n, :])`, a valid (basis-mixed but
full-resolution) range-Doppler cube per measurement channel, with no reconstruction
needed. What compression genuinely destroys is anything that indexes PHYSICAL elements
(an angle FFT, a beamformer) -- which a range-Doppler-input detector like FFTRadNet never
does; its "channels" axis is just a conv input dimension, agnostic to whether that
dimension counts antennas or linear combinations of them.

KNOWN CAVEAT, stated up front: the pilot corpus (`radial_like`) has only 16 physical RX,
so this M-grid spans modest compression ratios (16x MIMO-virtual down to 4:1 physical);
a real AFE compresses far larger apertures (hundreds to 1024 elements -- see
`e2e.chain.compress`'s module docstring) by much larger factors. Absolute AP numbers on
this pilot corpus are low across every arm compared (see `e2e.ml.baseline`'s own
caveats about this corpus/harness) -- THE SHAPE OF THE CURVE across M is the finding, not
the absolute AP.

THREE ARMS
----------
1. Native-M (`train_native_m` / `run_compressed_domain_grid`): FFTRadNet trained from
   scratch with its stem sized to `M`, on the no-reconstruct AFE-degraded cube (see
   `e2e.ml.afe_sweep`'s `no_reconstruct` mode). The deliverable.
2. Classical, reconstructed (`e2e.ml.afe_sweep.classical_at_m`): the honest classical
   comparator at the SAME `M` -- CFAR needs a physical aperture to beamform, so it always
   reconstructs (no compressed-domain classical detector exists here).
3. Reconstruct-then-detect (read, not recomputed, from
   `report/rt_ml/overnight_0811/afe_sweep/fftradnet_results.json`): the SAME full-aperture
   checkpoint arm 1 replaces, for the same `m_list`, already measured by
   `e2e.ml.afe_sweep.run_afe_sweep` -- the "reconstruct first" baseline this campaign asks
   whether native-M training beats.

ZERO-SHOT PROBE (`zero_shot_identity_probe`) -- the one defensible zero-shot check
------------------------------------------------------------------------------------
A full-aperture checkpoint's stem has `in_channels` baked in at construction
(`e2e.ml.train.build_model` / `_input_dims`); it CANNOT consume an `M < n_rx` input at
all -- that is a shape mismatch, not a degradation to be measured, and no amount of
probing changes it. The ONE case where the checkpoint's own shape still matches is
`M == n_rx`: there `sensing_matrix` returns the identity (see `afe_sweep`'s module
docstring), so the no-reconstruct measurement cube `y = I @ x` is bit-identical to the
untouched aperture, and evaluating the checkpoint there (`no_reconstruct=True`) is a
permutation-free sanity check of the no-reconstruct harness plumbing itself -- NOT
evidence the checkpoint generalizes to any `M < n_rx`, which remains impossible by
construction. Anything below `M == n_rx` is reported as the stated impossibility, not
fudged with e.g. zero-padding or channel-subsetting a checkpoint never trained for it.

CLI
---
    python -m e2e.ml.compressed_domain --manifest <pilot corpus>/manifest.json \\
        --full-aperture-ckpt <fftradnet_pilot400_e60>/best.pt \\
        [--m-list 16 12 8 4] [--epochs 15] [--batch-size 4] [--lr 3e-4] [--seed 0] \\
        [--weight-bits 8] [--no-quantize] [--device cuda] \\
        [--reconstruct-sweep-json report/rt_ml/overnight_0811/afe_sweep/fftradnet_results.json] \\
        [--out report/rt_ml/compressed_domain_v1]

Writes `<out>/results.json`, `<out>/ap_vs_m.png` (three lines), and prints a markdown
summary (also returned so a caller can write it to `<out>/summary.md`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from e2e.ml import train as train_mod
from e2e.ml.afe_sweep import (_DegradedRadarFrameDataset, _manifest_at_m, classical_at_m,
                              evaluate_at_m)
from e2e.ml.losses import detection_loss

DEFAULT_M_LIST = (16, 12, 8, 4)


# --------------------------------------------------------------------------------
# Arm 2: the zero-shot probe (and the stated impossibility it does NOT paper over)
# --------------------------------------------------------------------------------
def zero_shot_identity_probe(manifest_path, ckpt_path, *, split: str = "val", seed: int = 0,
                             device=None, batch_size: int = 8, limit: Optional[int] = None
                             ) -> Dict:
    """The one defensible zero-shot check: `M == n_rx`, `no_reconstruct=True`.

    See the module docstring's "ZERO-SHOT PROBE" section for why `M < n_rx` is a hard
    shape impossibility for a full-aperture checkpoint, not something this probe (or any
    probe) can measure around -- this function only ever evaluates at `M == n_rx`, the
    identity control, and does not accept an `m` argument for that reason.

    Returns `evaluate_at_m`'s payload (`{"m", "model", "classical"}`, `no_reconstruct=
    True`) plus `"undistorted_AP"` (the checkpoint's plain `train.evaluate` AP on the
    same split, no degradation at all) and `"matches_undistorted"` (bool): if the harness
    is honest, the two must agree, since `no_reconstruct` at the identity `M` changes
    nothing about the data the model sees.
    """
    manifest_path = Path(manifest_path)
    with open(manifest_path) as f:
        manifest = json.load(f)
    n_rx = int(manifest["config"]["n_rx"])

    r = evaluate_at_m(manifest_path, ckpt_path, split, n_rx, seed=seed, device=device,
                      batch_size=batch_size, limit=limit, no_reconstruct=True)
    undistorted = train_mod.evaluate(manifest_path, ckpt_path, split=split, device=device)
    r["undistorted_AP"] = undistorted["AP"]
    r["matches_undistorted"] = abs(r["model"]["AP"] - undistorted["AP"]) < 1e-5
    return r


# --------------------------------------------------------------------------------
# Arm 1: native-M training
# --------------------------------------------------------------------------------
def train_native_m(manifest_path, m: int, *, model_name: str = "fftradnet", epochs: int = 15,
                   batch_size: int = 4, lr: float = 3e-4, seed: int = 0,
                   weight_bits: Optional[int] = 8, quantize: bool = True,
                   input_format: str = "rd", reg_weight: float = 100.0, gamma: float = 2.0,
                   cls_normalize: str = "positives", device=None, out_dir=None) -> Dict:
    """Train `model_name` with a stem sized to `M`, on the AFE-degraded no-reconstruct
    cube, evaluating on val each epoch.

    Deliberately NOT a wrapper around `e2e.ml.train.train` (that function reads its own
    manifest from disk with no hook for a `_DegradedRadarFrameDataset`/`n_rx` override);
    a stripped-down re-implementation of its loop instead, sharing its building blocks
    (`build_model`, `_evaluate_split`, `detection_loss`) so results are directly
    comparable to a `train.py` run at the same manifest/model/epochs/batch_size/lr/seed.
    No AMP/accum_steps/LR schedule (see `train.py`'s own "tutorial, not a framework" note
    -- this is smaller still, one grid-cell of one campaign).

    Returns `{"m", "model": {"AP", "AR", "range_rmse_m"} (the best-val-AP epoch's
    metrics), "history", "ckpt"}`. Writes `best.pt`/`history.json` under `out_dir`
    (default `<manifest dir>/runs/<model_name>_nativeM<m>/`), matching `train.py`'s
    artifact layout; `best.pt` additionally records `"m"` and `"no_reconstruct": True` so
    a later `evaluate_at_m`/`build_model` call knows this checkpoint's stem geometry.
    """
    manifest_path = Path(manifest_path)
    device = device if device is not None else train_mod._default_device()
    torch.manual_seed(seed)

    with open(manifest_path) as f:
        manifest = json.load(f)
    grid = train_mod._load_grid(manifest)

    train_ds = _DegradedRadarFrameDataset(manifest_path, "train", input_format, m=m,
                                          seed=seed, weight_bits=weight_bits,
                                          quantize=quantize, no_reconstruct=True)
    val_ds = _DegradedRadarFrameDataset(manifest_path, "val", input_format, m=m,
                                        seed=seed, weight_bits=weight_bits,
                                        quantize=quantize, no_reconstruct=True)

    manifest_for_model = _manifest_at_m(manifest, m, input_format)
    model = train_mod.build_model(model_name, manifest_for_model, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=False, generator=gen, num_workers=0)

    out_dir = Path(out_dir) if out_dir is not None else (
        manifest_path.parent / "runs" / f"{model_name}_nativeM{m}")
    out_dir.mkdir(parents=True, exist_ok=True)

    history: Dict[str, list] = {"epoch": [], "train_loss": [], "val_AP": [], "val_AR": [],
                                "val_range_rmse_m": []}
    best_ap = -1.0
    best_state = None
    best_metrics: Optional[Dict] = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)["detection"]
            loss, _parts = detection_loss(pred, y, gamma=gamma, reg_weight=reg_weight,
                                          cls_normalize=cls_normalize)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item())
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        val_metrics = train_mod._evaluate_split(model, val_ds, grid, device=device,
                                                 batch_size=batch_size)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_AP"].append(val_metrics["AP"])
        history["val_AR"].append(val_metrics["AR"])
        history["val_range_rmse_m"].append(val_metrics["range_rmse_m"])
        print(f"[native-M={m}] epoch {epoch}/{epochs}  loss={train_loss:.4f}  "
              f"val_AP={val_metrics['AP']:.4f}  val_AR={val_metrics['AR']:.4f}")

        # >= not >: same tie-break as train.py -- an AP tie keeps the later, more-
        # converged epoch (AP's coarse threshold sweep can't distinguish regression
        # quality, but later epochs have lower loss).
        if val_metrics["AP"] >= best_ap:
            best_ap = val_metrics["AP"]
            best_metrics = dict(val_metrics)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:  # epochs == 0 edge case: nothing trained
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_metrics = {"AP": 0.0, "AR": 0.0, "range_rmse_m": 0.0}

    torch.save(
        {"model_state": best_state, "model_name": model_name, "input_format": input_format,
         "manifest": str(manifest_path), "m": int(m), "no_reconstruct": True,
         "history": history},
        out_dir / "best.pt",
    )
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return {
        "m": int(m),
        "model": {"AP": best_metrics["AP"], "AR": best_metrics["AR"],
                  "range_rmse_m": best_metrics["range_rmse_m"]},
        "history": history,
        "ckpt": str(out_dir / "best.pt"),
    }


def run_compressed_domain_grid(manifest_path, m_list: Sequence[int] = DEFAULT_M_LIST, *,
                               model_name: str = "fftradnet", epochs: int = 15,
                               batch_size: int = 4, lr: float = 3e-4, seed: int = 0,
                               weight_bits: Optional[int] = 8, quantize: bool = True,
                               device=None, out_root=None, limit: Optional[int] = None
                               ) -> Dict:
    """Train the native-M grid (arm 1) and score the reconstructed classical comparator
    (arm 2's honest baseline, arm-3-adjacent -- see module docstring) at each `M`, on the
    val split. Returns the JSON-serializable payload the CLI writes to `results.json`
    (`native_model`/`classical_reconstructed` are both `{"AP", "AR", "range_rmse_m"}`).
    """
    manifest_path = Path(manifest_path)
    results: List[Dict] = []
    for m in m_list:
        run_out_dir = Path(out_root) / f"m{m}" if out_root is not None else None
        r = train_native_m(manifest_path, m, model_name=model_name, epochs=epochs,
                           batch_size=batch_size, lr=lr, seed=seed, weight_bits=weight_bits,
                           quantize=quantize, device=device, out_dir=run_out_dir)
        classical = classical_at_m(manifest_path, "val", m, seed=seed, weight_bits=weight_bits,
                                   quantize=quantize, device=device, limit=limit)
        results.append({"m": int(m), "native_model": r["model"], "ckpt": r["ckpt"],
                        "classical_reconstructed": classical})
        print(f"  M={m:>3}  native-M AP={r['model']['AP']:.4f}  |  "
              f"classical(reconstructed) AP={classical['AP']:.4f}")

    return {
        "manifest": str(manifest_path),
        "model_name": model_name,
        "m_list": [int(m) for m in m_list],
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "weight_bits": weight_bits,
        "quantize": quantize,
        "results": results,
    }


# --------------------------------------------------------------------------------
# Reporting: 3-line AP-vs-M figure + markdown summary
# --------------------------------------------------------------------------------
def _reconstruct_then_detect_ap(reconstruct_sweep_json, m_list: Sequence[int]) -> Dict[int, float]:
    """`{m: model AP}` for `m_list`, read from an `e2e.ml.afe_sweep.run_afe_sweep`
    results JSON (e.g. `report/rt_ml/overnight_0811/afe_sweep/fftradnet_results.json`) --
    arm 3, NOT recomputed here (that sweep needs the full-aperture checkpoint + GPU time
    already spent producing that file). Missing `m` values are simply absent from the
    returned dict (the plotting/report code below tolerates a shorter series).
    """
    payload = json.loads(Path(reconstruct_sweep_json).read_text())
    by_m = {int(r["m"]): float(r["model"]["AP"]) for r in payload["results"]}
    return {m: by_m[m] for m in m_list if m in by_m}


def plot_ap_vs_m(payload: Dict, fig_path, *, reconstruct_then_detect: Optional[Dict[int, float]] = None):
    """AP vs M: native-M (arm 1), classical-reconstructed (arm 2), and, if given,
    reconstruct-then-detect (arm 3, read from an existing afe_sweep results JSON) --
    up to three lines. Agg backend, savefig only -- no display."""
    import matplotlib

    matplotlib.use("Agg")  # noqa: E402 -- must precede pyplot import; headless/CI-safe
    import matplotlib.pyplot as plt  # noqa: E402

    ms = [r["m"] for r in payload["results"]]
    native_ap = [r["native_model"]["AP"] for r in payload["results"]]
    classical_ap = [r["classical_reconstructed"]["AP"] for r in payload["results"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ms, native_ap, marker="o", label=f"{payload['model_name']} (native-M, no reconstruct)")
    ax.plot(ms, classical_ap, marker="s", label="classical CFAR (reconstructed)")
    if reconstruct_then_detect:
        rms = [m for m in ms if m in reconstruct_then_detect]
        rap = [reconstruct_then_detect[m] for m in rms]
        if rap:
            ax.plot(rms, rap, marker="^", label=f"{payload['model_name']} (reconstruct-then-detect)")
    ax.set_xlabel("M (compressed channels)")
    ax.set_ylabel("AP")
    ax.set_title("Compressed-domain detection: native-M vs. reconstruct-then-detect")
    ax.invert_xaxis()  # M decreases left -> right: reads as increasing compression

    all_ap = native_ap + classical_ap + (list(reconstruct_then_detect.values())
                                         if reconstruct_then_detect else [])
    positive = [v for v in all_ap if v > 0]
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


def markdown_summary(payload: Dict, *, zero_shot: Optional[Dict] = None,
                     reconstruct_then_detect: Optional[Dict[int, float]] = None) -> str:
    """Results table + the arm-2 outcome, as a markdown string (also printed by the CLI)."""
    lines = [
        "### Compressed-domain-v1: native-M detection vs. reconstruct-then-detect",
        "",
        "CAVEAT: pilot corpus has only 16 physical RX -- modest compression ratios; "
        "absolute AP is low across every arm on this corpus/harness (see "
        "`e2e.ml.baseline`'s module docstring). The SHAPE of the curve across M is the "
        "finding, not the absolute AP.",
        "",
        f"model: {payload['model_name']}  epochs={payload['epochs']}  "
        f"batch_size={payload['batch_size']}  lr={payload['lr']}  seed={payload['seed']}",
        "",
        "| M | native-M AP | native-M AR | classical (reconstructed) AP "
        "| classical AR | reconstruct-then-detect AP |",
        "|---|---|---|---|---|---|",
    ]
    for r in payload["results"]:
        m = r["m"]
        nm, cc = r["native_model"], r["classical_reconstructed"]
        rtd = "n/a" if not (reconstruct_then_detect and m in reconstruct_then_detect) \
            else f"{reconstruct_then_detect[m]:.4f}"
        lines.append(f"| {m} | {nm['AP']:.4f} | {nm['AR']:.4f} | {cc['AP']:.4f} | "
                     f"{cc['AR']:.4f} | {rtd} |")

    lines.append("")
    lines.append("### Arm 2: zero-shot probe")
    lines.append("")
    if zero_shot is not None:
        lines.append(
            f"A full-aperture checkpoint's stem has `in_channels` fixed at construction "
            f"and CANNOT consume `M < n_rx` at all (shape mismatch) -- there is no probe "
            f"for that case, only the stated impossibility. The one case where the "
            f"checkpoint's shape still matches is `M == n_rx` (identity compression): "
            f"probed at M={zero_shot['m']}, no-reconstruct AP={zero_shot['model']['AP']:.4f} "
            f"vs. undegraded AP={zero_shot['undistorted_AP']:.4f} "
            f"(match={zero_shot['matches_undistorted']}) -- a permutation-free sanity check "
            f"of the no-reconstruct harness, not evidence of M<n_rx generalization."
        )
    else:
        lines.append(
            "No full-aperture checkpoint was supplied for the zero-shot probe. Stated "
            "plainly: a full-aperture checkpoint cannot consume M < n_rx (shape "
            "mismatch), so no zero-shot number below M == n_rx exists to report."
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m e2e.ml.compressed_domain",
        description="Native-M compressed-domain detection training grid vs. classical "
                    "(reconstructed) and reconstruct-then-detect (read from an existing "
                    "afe_sweep results JSON) comparators.",
    )
    p.add_argument("--manifest", required=True, help="e2e.ml.dataset manifest.json (raw-ADC-native)")
    p.add_argument("--full-aperture-ckpt", default=None,
                   help="a train.py-written best.pt for the zero-shot probe (arm 2); "
                        "omit to skip the probe and state the impossibility instead")
    p.add_argument("--reconstruct-sweep-json", default=None,
                   help="an e2e.ml.afe_sweep results JSON (arm 3, e.g. "
                        "report/rt_ml/overnight_0811/afe_sweep/fftradnet_results.json); "
                        "omit to plot/report only arms 1-2")
    p.add_argument("--model", choices=("fftradnet",), default="fftradnet")
    p.add_argument("--m-list", type=int, nargs="+", default=list(DEFAULT_M_LIST))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--weight-bits", type=int, default=8)
    p.add_argument("--no-quantize", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="score classical on the first N val frames")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="report/rt_ml/compressed_domain_v1")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    device = torch.device(args.device) if args.device is not None else None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.full_aperture_ckpt is not None:
        zero_shot = zero_shot_identity_probe(args.manifest, args.full_aperture_ckpt,
                                             seed=args.seed, device=device)
    else:
        zero_shot = None

    payload = run_compressed_domain_grid(
        args.manifest, m_list=args.m_list, model_name=args.model, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, seed=args.seed, weight_bits=args.weight_bits,
        quantize=not args.no_quantize, device=device, out_root=out_dir / "runs",
        limit=args.limit,
    )
    payload["zero_shot_probe"] = zero_shot

    reconstruct_then_detect = None
    if args.reconstruct_sweep_json is not None:
        reconstruct_then_detect = _reconstruct_then_detect_ap(args.reconstruct_sweep_json,
                                                               args.m_list)
        payload["reconstruct_then_detect_ap"] = reconstruct_then_detect

    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {results_path}")

    fig_path = plot_ap_vs_m(payload, out_dir / "ap_vs_m.png",
                            reconstruct_then_detect=reconstruct_then_detect)
    print(f"wrote {fig_path}")

    summary = markdown_summary(payload, zero_shot=zero_shot,
                               reconstruct_then_detect=reconstruct_then_detect)
    (out_dir / "summary.md").write_text(summary)
    print(f"wrote {out_dir / 'summary.md'}")
    print()
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
