"""Reactive subspace refinement: what the singular-value gap collapsing costs, and recovery.

The story this figure tells, over a real ray-traced Munich run:

1. **Non-degenerate.** The scene's signal subspace has a clear gap at the rank cutoff, the
   tracker follows it, and error stays low.
2. **Degenerate.** The gap collapses -- in a real city scene the scatterers momentarily fail
   to span `k` independent directions -- and the top-`k` subspace stops being well defined.
   A tracker that keeps taking its usual single refinement step per frame diverges here.
3. **Recovery.** The gap reopens and the tracker comes back. Showing the recovery is the
   point: a figure that stops at step 2 says "this breaks", while one that runs through step
   3 says "this breaks, we detect it, and we spend compute to get back" -- which is what the
   reactive gate actually does.

Two arms, identical in every other respect:

* `gap_response="none"` -- the fixed-effort baseline. `n_refine` passes every frame,
  ignoring the spectrum entirely.
* `gap_response="refine"` -- bump to `n_refine_hi` passes *only while* the gap is collapsed
  (`sv_gap_norm < gap_threshold`). Compute goes where the difficulty is.

This script exists because its predecessor did not: the original figure was drawn by a
scratch script that no longer exists, so a published figure had no reproducible recipe.
Anything shipped from here on regenerates with one command.

    python -m e2e.main.main_subspace_refine                     # default 60 frames
    python -m e2e.main.main_subspace_refine --n-steps 100       # the whole munich stack
    python -m e2e.main.main_subspace_refine --out docs/media/tracking_refine.png
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402 -- must precede pyplot; headless/CI-safe, never plt.show()
import matplotlib.pyplot as plt  # noqa: E402

from e2e.blocks import (  # noqa: E402
    AdaOjaBlock,
    AFEBlock,
    InterconnectBlock,
    RFFEBlock,
    SionnaEnvironmentBlock,
    SubspaceErrorBlock,
)
from e2e.simulation import Simulation  # noqa: E402

N_RX = 1024
N_TX = 1
K = 8
M = 512
N_REFINE = 1        # baseline effort per frame; the gate raises this, it is not the gate
N_REFINE_HI = 60
GAP_THRESHOLD = 0.01

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
DEFAULT_OUT = os.path.join(FIG_DIR, "tracking_refine.png")

# Colours chosen so the two arms stay distinguishable under video compression (a talk over
# screen share): a saturated red against a mid blue, not two similar hues.
C_BASE = "#C1362F"
C_REFINE = "#1F6FB2"
C_GATE = "#4A4A4A"


def _run_arm(scenario, n_steps, gap_response):
    """One pipeline run; returns the per-frame diagnostic lists we plot."""
    env = SionnaEnvironmentBlock(scenario)
    sim = Simulation(
        env,
        [SubspaceErrorBlock()],
        K,
        RFFEBlock(n=N_RX * N_TX,
                  physical_scale=bool(getattr(env, "physical_scale", None))),
        InterconnectBlock(case="case3"),
        AFEBlock(),
        AdaOjaBlock(N_RX, K, m=M, n_refine=N_REFINE, gap_response=gap_response,
                    gap_threshold=GAP_THRESHOLD, n_refine_hi=N_REFINE_HI),
    )
    out = sim.run(n_steps=n_steps)
    return {
        # float() copies scalar tensors off the library device; matplotlib cannot read a
        # CUDA tensor and would fail inside its own .numpy() call.
        "subspace_err": [float(x) for x in out["subspace_err"]],
        "sv_gap_norm": [float(x) for x in out["sv_gap_norm"]],
        "n_refine_used": [float(x) for x in out.get("n_refine_used", [])],
    }


def _degenerate_runs(gaps, threshold, min_len=2):
    """Contiguous [lo, hi] runs of frames whose gap is below `threshold`.

    Runs, not one outer span. On a long munich run the gap collapses around frame 22,
    reopens in the low 90s, then collapses again — so a single first-to-last span would
    shade straight over the recovery, which is the most interesting thing in the figure.
    Runs shorter than `min_len` are dropped so single-frame flickers do not become stripes.
    """
    runs, start = [], None
    for i, g in enumerate(gaps):
        degenerate = (g == g) and g < threshold      # g == g -> not NaN
        if degenerate and start is None:
            start = i
        elif not degenerate and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(gaps) - 1))
    return [r for r in runs if r[1] - r[0] + 1 >= min_len]


def _recovered_windows(gaps, threshold, min_len=2):
    """Runs of frames ABOVE threshold that sit between two collapses — the recoveries."""
    runs = _degenerate_runs(gaps, threshold, min_len=1)
    out = []
    for (_, end_prev), (start_next, _) in zip(runs, runs[1:]):
        if start_next - end_prev - 1 >= min_len:
            out.append((end_prev + 1, start_next - 1))
    return out


def build_figure(base, refine, out_path, threshold=GAP_THRESHOLD):
    frames = range(len(refine["subspace_err"]))
    runs = _degenerate_runs(refine["sv_gap_norm"], threshold)
    recoveries = _recovered_windows(refine["sv_gap_norm"], threshold)

    fig, (ax_err, ax_gap, ax_ref) = plt.subplots(
        3, 1, figsize=(9.0, 8.4), sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.35, 1.0], "hspace": 0.16})

    ax_err.plot(frames, base["subspace_err"], color=C_BASE, lw=2.0,
                label=f"fixed effort (gap_response='none', {N_REFINE} pass/frame)")
    ax_err.plot(frames, refine["subspace_err"], color=C_REFINE, lw=2.0,
                label=f"reactive gate (gap_response='refine', up to {N_REFINE_HI})")
    ax_err.set_ylabel("subspace error", fontsize=12)
    ax_err.set_yscale("log")
    ax_err.legend(fontsize=10.5, loc="upper left", framealpha=0.95)
    ax_err.set_title("Reactive refinement through a subspace collapse and back",
                     fontsize=14, fontweight="bold")

    ax_gap.plot(frames, refine["sv_gap_norm"], color=C_GATE, lw=1.8)
    ax_gap.axhline(threshold, color=C_REFINE, ls="--", lw=1.4,
                   label=f"gap_threshold = {threshold}")
    ax_gap.set_ylabel("sv_gap_norm", fontsize=12)
    ax_gap.set_yscale("log")
    ax_gap.legend(fontsize=10.5, loc="lower left", framealpha=0.95)

    if refine["n_refine_used"]:
        ax_ref.step(frames, refine["n_refine_used"], where="mid", color=C_REFINE, lw=1.8)
    ax_ref.set_ylabel("refinement\npasses used", fontsize=12)
    ax_ref.set_xlabel("frame", fontsize=12)

    for ax in (ax_err, ax_gap, ax_ref):
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)
        # Shade every collapsed stretch on every panel, so the three panels read as one
        # event -- and so the unshaded window between them reads as the recovery.
        for lo, hi in runs:
            ax.axvspan(lo - 0.5, hi + 0.5, color="#000000", alpha=0.07, lw=0)
        for lo, hi in recoveries:
            ax.axvspan(lo - 0.5, hi + 0.5, color=C_REFINE, alpha=0.10, lw=0)

    # Both state labels live on the gap panel -- that is the panel where "collapsed" and
    # "recovered" are *defined*, and its upper region is empty because the gap runs low
    # exactly when it is collapsed. (They were on the error panel and collided with its
    # legend, which is wide enough to cover the middle of the run.)
    if runs:
        lo, hi = max(runs, key=lambda r: r[1] - r[0])
        ax_gap.text((lo + hi) / 2.0, 0.94, "gap collapsed",
                    transform=ax_gap.get_xaxis_transform(), ha="center", va="top",
                    fontsize=11.5, color="#333333", fontstyle="italic")
    for lo, hi in recoveries:
        ax_gap.text((lo + hi) / 2.0, 0.82, "recovers",
                    transform=ax_gap.get_xaxis_transform(), ha="center", va="top",
                    fontsize=11.5, color=C_REFINE, fontweight="bold")

    # The quotable number belongs ON the figure, so a caption cannot drift from it.
    stats = summarize(base, refine, threshold)
    drop = stats.get("degenerate_improvement")
    if drop is not None and drop == drop:
        ax_err.text(0.985, 0.06,
                    f"while collapsed: {drop * 100:.0f}% lower tracking error",
                    transform=ax_err.transAxes, ha="right", va="bottom", fontsize=11.5,
                    color=C_REFINE, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFFFF",
                              edgecolor=C_REFINE, linewidth=1.1, alpha=0.95))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def summarize(base, refine, threshold=GAP_THRESHOLD):
    """Numbers to print (and to quote in a caption) rather than eyeball off the figure."""
    gaps = refine["sv_gap_norm"]
    runs = _degenerate_runs(gaps, threshold)
    recoveries = _recovered_windows(gaps, threshold)
    stats = {
        "n_frames": len(refine["subspace_err"]),
        "degenerate_runs": runs,
        "recovery_windows": recoveries,
        "frames_degenerate": sum(1 for g in gaps if g == g and g < threshold),
    }

    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    if runs:
        first_lo = runs[0][0]
        stats["pre_collapse_err_refine"] = _mean(refine["subspace_err"][:first_lo])
        # Compare the two arms only over the collapsed frames -- that is where the gate
        # does anything at all, so averaging over the whole run would dilute the effect
        # with frames on which the arms are identical by construction.
        idx = [i for lo, hi in runs for i in range(lo, hi + 1)]
        b_deg = _mean([base["subspace_err"][i] for i in idx])
        r_deg = _mean([refine["subspace_err"][i] for i in idx])
        stats["degenerate_err_baseline"] = b_deg
        stats["degenerate_err_refine"] = r_deg
        if b_deg and b_deg == b_deg and b_deg > 0:
            stats["degenerate_improvement"] = 1.0 - r_deg / b_deg
    if recoveries:
        idx = [i for lo, hi in recoveries for i in range(lo, hi + 1)]
        stats["recovered_err_baseline"] = _mean([base["subspace_err"][i] for i in idx])
        stats["recovered_err_refine"] = _mean([refine["subspace_err"][i] for i in idx])
    if refine["n_refine_used"]:
        used = refine["n_refine_used"]
        stats["mean_passes_refine"] = _mean(used)
        stats["passes_vs_fixed"] = _mean(used) / max(N_REFINE, 1)
    return stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scenario", default="munich",
                   help="SionnaEnvironmentBlock scenario name (default munich)")
    p.add_argument("--n-steps", type=int, default=60,
                   help="frames to run; needs to be long enough to get PAST the collapse, "
                        "which is the whole point (default 60)")
    p.add_argument("--out", default=DEFAULT_OUT, help="output .png path")
    p.add_argument("--cache", default=None, metavar="RUNS.json",
                   help="reuse the per-frame diagnostics from this file if it exists, else "
                        "write them to it. Two 100-frame pipeline runs take minutes; "
                        "iterating on the FIGURE should not pay that cost twice.")
    args = p.parse_args(argv)

    cache = Path(args.cache) if args.cache else None
    if cache is not None and cache.is_file():
        payload = json.loads(cache.read_text())
        if payload.get("scenario") != args.scenario or payload.get("n_steps") != args.n_steps:
            raise SystemExit(
                f"{cache} holds scenario={payload.get('scenario')!r} "
                f"n_steps={payload.get('n_steps')}, not {args.scenario!r}/{args.n_steps}. "
                "Delete it or pass a different --cache path rather than silently plotting "
                "the wrong run.")
        base, refine = payload["base"], payload["refine"]
        print(f"reusing cached diagnostics from {cache}")
    else:
        print(f"[1/2] baseline arm  (gap_response='none')   {args.n_steps} frames ...")
        base = _run_arm(args.scenario, args.n_steps, "none")
        print(f"[2/2] reactive arm  (gap_response='refine') {args.n_steps} frames ...")
        refine = _run_arm(args.scenario, args.n_steps, "refine")
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"scenario": args.scenario,
                                         "n_steps": args.n_steps,
                                         "base": base, "refine": refine}, indent=1))
            print(f"cached diagnostics to {cache}")

    out = build_figure(base, refine, args.out)
    print(f"wrote {out}")
    for key, value in summarize(base, refine).items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
