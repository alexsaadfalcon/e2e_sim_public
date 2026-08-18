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
C_CONST = "#1B7837"   # constant HIGH effort -- the honesty arm (F36)


class _ReorderedEnvironmentBlock(SionnaEnvironmentBlock):
    """`SionnaEnvironmentBlock` that walks frames in an EXPLICIT order.

    Purpose is illustrative, and the figure says so. On the real munich stack the gap
    collapses at frame 22 and does not reopen until frame 91 -- 69 degenerate frames
    against 22 clean ones, so a figure of the real run is ~70% collapse and the recovery
    is a sliver at the right-hand edge. The EVENT is real; its duration makes it
    unreadable.

    `--frame-order collapse-window` therefore plays the stack forward into the collapse,
    then plays the pre-collapse frames BACKWARD to bring the gap back. The recovery is a
    genuine pipeline response to genuinely non-degenerate frames -- the tracker is not
    told anything -- but the frame ORDER is constructed, not observed, and that must be
    stated wherever the figure appears.
    """

    def __init__(self, scenario_name, order, **kwargs):
        super().__init__(scenario_name, **kwargs)
        n = len(self.sionna_iterator)
        bad = [i for i in order if not 0 <= i < n]
        if bad:
            raise ValueError(f"frame order references frames outside [0, {n}): {bad[:5]}")
        self._order = list(order)
        self._ptr = 0
        self.frame_counter = self._order[0]

    def step(self):
        self._ptr = (self._ptr + 1) % len(self._order)
        self.frame_counter = self._order[self._ptr]

    def reset(self):
        self._ptr = 0
        self.frame_counter = self._order[0]


def collapse_window_order(gaps, threshold, *, n_degenerate=25, n_pre=25, n_recover=25):
    """Frame order giving `n_pre` clean frames, `n_degenerate` collapsed, then `n_recover`
    clean again, built from an observed `gaps` trace.

    The recovery leg REVERSES back through the pre-collapse frames, because on this stack
    the collapse does not reopen for 69 frames. Returns `(order, spans)` where `spans`
    names each leg for the figure's annotations.
    """
    first = next((i for i, g in enumerate(gaps) if g == g and g < threshold), None)
    if first is None:
        raise SystemExit(
            f"no frame in the probe has sv_gap_norm < {threshold}; nothing to build a "
            "collapse window around. Run more probe frames or raise --gap-threshold.")
    if first < 2:
        raise SystemExit(
            f"the gap is already collapsed at frame {first}; there are no clean frames "
            "before it to play into or reverse back through.")

    pre = list(range(max(0, first - n_pre), first))
    deg = list(range(first, first + n_degenerate))
    rec = list(reversed(pre))[:n_recover]
    order = pre + deg + rec
    spans = {"pre": (0, len(pre) - 1),
             "degenerate": (len(pre), len(pre) + len(deg) - 1),
             "recovery": (len(pre) + len(deg), len(order) - 1)}
    return order, spans


def _run_arm(scenario, n_steps, gap_response, env=None, n_refine=None):
    """One pipeline run; returns the per-frame diagnostic lists we plot.

    `n_refine` is the BASELINE effort per frame (the gate raises it; it is not the gate).
    Exposed so the figure can carry a constant-HIGH-effort arm alongside the gated one --
    without it the figure invites the reading that the gate is what makes tracking good,
    when in fact it is the compute (see this module's docstring, and F36).
    """
    env = env if env is not None else SionnaEnvironmentBlock(scenario)
    n_refine = N_REFINE if n_refine is None else int(n_refine)
    sim = Simulation(
        env,
        [SubspaceErrorBlock()],
        K,
        RFFEBlock(n=N_RX * N_TX,
                  physical_scale=bool(getattr(env, "physical_scale", None))),
        InterconnectBlock(case="case3"),
        AFEBlock(),
        AdaOjaBlock(N_RX, K, m=M, n_refine=n_refine, gap_response=gap_response,
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


#: Printed on every figure. Two caveats a reader cannot infer from the axes, and both
#: change what the figure means:
#:  * the gate's input is oracle-sourced (see e2e/simulation.py's rank_diagnostic
#:    "Honesty note" and notes/ESTABLISHED_FACTS.md F25) -- a deployed receiver holds a
#:    rank-k basis, never the full spectrum, so this gate is not implementable as shown;
#:  * with --frame-order collapse-window the frame ORDER is constructed.
_ORACLE_NOTE = ("sv_gap_norm is computed from the frame's FULL singular-value spectrum — "
                "simulator instrumentation. A receiver holding only a rank-k basis cannot "
                "measure it this way; an in-measurement-domain estimator is feasible but "
                "not implemented.")
_ORDER_NOTE = ("Frame order constructed: played forward into the collapse, then BACKWARD "
               "through the pre-collapse frames. On the real stack the gap stays collapsed "
               "for 69 of 100 frames. The tracker's response is real; the ordering is not.")


def build_figure(base, refine, out_path, threshold=GAP_THRESHOLD, constructed_order=False,
                 const_hi=None):
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
    # The arm that keeps the figure honest. Without it a reader concludes the GATE is what
    # makes tracking good; with it they can see that constant high effort is better
    # everywhere, and that what the gate actually buys is COMPUTE, not accuracy. See F36.
    if const_hi is not None:
        ax_err.plot(frames, const_hi["subspace_err"], color=C_CONST, lw=1.8, ls="--",
                    label=f"constant high effort ({N_REFINE_HI} passes EVERY frame)")
    ax_err.set_ylabel("subspace error", fontsize=12)
    ax_err.set_yscale("log")
    # The two-arm legend lives BELOW the panels, not inside ax_err. In-axes it sat upper
    # left, which is exactly where the baseline arm runs while the gap is collapsed
    # (measured 1.26-1.66 there) -- so the legend hid the divergence the figure exists to
    # show, and the curve read as missing data rather than as a covered line.
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
        # State the TRADE, not just the win. "N% lower error while collapsed" is true but
        # compares 1 pass against N_REFINE_HI passes inside the collapsed window -- it
        # measures the value of the compute, not of reacting to the gap, and the same
        # number would appear if the gate fired at random. With the constant-effort arm
        # present the honest claim is the compute saving. See notes F36.
        if const_hi is not None and const_hi.get("n_refine_used"):
            used = refine["n_refine_used"]
            mean_gated = (sum(used) / len(used)) if used else float("nan")
            msg = (f"gate matches constant effort while collapsed, "
                   f"for {mean_gated / max(N_REFINE_HI, 1) * 100:.0f}% of its compute")
        else:
            msg = f"while collapsed: {drop * 100:.0f}% lower error than fixed 1-pass effort"
        ax_err.text(0.985, 0.06, msg,
                    transform=ax_err.transAxes, ha="right", va="bottom", fontsize=11.5,
                    color=C_REFINE, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFFFF",
                              edgecolor=C_REFINE, linewidth=1.1, alpha=0.95))

    handles, labels = ax_err.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.065),
               ncol=2, fontsize=10.5, framealpha=0.95)

    notes = [_ORACLE_NOTE] + ([_ORDER_NOTE] if constructed_order else [])
    fig.text(0.5, -0.005 - 0.030 * (len(notes) - 1), "\n".join(notes), ha="center",
             va="top", fontsize=9.0, color="#555555", fontstyle="italic", wrap=True)

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
    p.add_argument("--frame-order", default="natural",
                   choices=("natural", "collapse-window"),
                   help="'natural' plays the stack in order (the real run: 22 clean "
                        "frames, then 69 collapsed). 'collapse-window' plays forward "
                        "into the collapse and then BACKWARD through the pre-collapse "
                        "frames to bring the gap back, so the recovery is legible. The "
                        "recovery is a real pipeline response to real clean frames; the "
                        "frame ORDER is constructed, and the figure says so.")
    p.add_argument("--window", type=int, nargs=3, default=(25, 25, 25),
                   metavar=("PRE", "DEGENERATE", "RECOVER"),
                   help="frame counts per leg for --frame-order collapse-window")
    p.add_argument("--no-constant-arm", action="store_true",
                   help="omit the constant-high-effort arm. It is ON by default because "
                        "without it the figure reads as 'the gate makes tracking good' "
                        "when the truth is 'the compute does' -- constant effort beats the "
                        "gate everywhere OUTSIDE the collapse (measured 0.037 vs 0.575). "
                        "See notes/ESTABLISHED_FACTS.md F36.")
    p.add_argument("--probe-steps", type=int, default=100,
                   help="frames of the baseline arm used to LOCATE the collapse before "
                        "building the window")
    args = p.parse_args(argv)

    cache = Path(args.cache) if args.cache else None
    if cache is not None and cache.is_file():
        payload = json.loads(cache.read_text())
        # With --frame-order collapse-window the frame count is DERIVED from where the
        # probe finds the collapse, so it is not known here and cannot be the cache key;
        # the (order, window) pair is what actually determines the run.
        if args.frame_order == "collapse-window":
            stale = (payload.get("scenario") != args.scenario
                     or payload.get("frame_order") != args.frame_order
                     or tuple(payload.get("window") or ()) != tuple(args.window))
            want = f"{args.scenario!r}/{args.frame_order}/window={tuple(args.window)}"
            got = (f"{payload.get('scenario')!r}/{payload.get('frame_order')}/"
                   f"window={tuple(payload.get('window') or ())}")
        else:
            stale = (payload.get("scenario") != args.scenario
                     or payload.get("n_steps") != args.n_steps)
            want = f"{args.scenario!r}/{args.n_steps}"
            got = f"{payload.get('scenario')!r}/n_steps={payload.get('n_steps')}"
        if stale:
            raise SystemExit(
                f"{cache} holds {got}, not {want}. Delete it or pass a different "
                "--cache path rather than silently plotting the wrong run.")
        args.n_steps = payload.get("n_steps", args.n_steps)
        base, refine = payload["base"], payload["refine"]
        const_hi = payload.get("const_hi")
        print(f"reusing cached diagnostics from {cache}")
    else:
        n_steps, mk_env, spans = args.n_steps, (lambda: None), None
        if args.frame_order == "collapse-window":
            n_pre, n_deg, n_rec = args.window
            print(f"[probe] locating the collapse over {args.probe_steps} frames ...")
            probe = _run_arm(args.scenario, args.probe_steps, "none")
            order, spans = collapse_window_order(
                probe["sv_gap_norm"], GAP_THRESHOLD,
                n_pre=n_pre, n_degenerate=n_deg, n_recover=n_rec)
            n_steps = len(order)
            print(f"[probe] collapse begins at stack frame {order[spans['degenerate'][0]]}; "
                  f"window = {len(order[:spans['degenerate'][0]])} clean + {n_deg} "
                  f"collapsed + {n_steps - spans['recovery'][0]} reversed "
                  f"= {n_steps} frames")

            def mk_env():
                return _ReorderedEnvironmentBlock(args.scenario, order)

        print(f"[1/3] baseline arm    (1 pass/frame)          {n_steps} frames ...")
        base = _run_arm(args.scenario, n_steps, "none", env=mk_env())
        print(f"[2/3] reactive arm    (gated 1 -> {N_REFINE_HI})         {n_steps} frames ...")
        refine = _run_arm(args.scenario, n_steps, "refine", env=mk_env())
        const_hi = None
        if not args.no_constant_arm:
            print(f"[3/3] constant-effort arm ({N_REFINE_HI} every frame) {n_steps} frames ...")
            const_hi = _run_arm(args.scenario, n_steps, "none", env=mk_env(),
                                n_refine=N_REFINE_HI)
        args.n_steps = n_steps
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"scenario": args.scenario,
                                         "n_steps": n_steps,
                                         "frame_order": args.frame_order,
                                         "window": list(args.window),
                                         "spans": spans,
                                         "base": base, "refine": refine,
                                         "const_hi": const_hi}, indent=1))
            print(f"cached diagnostics to {cache}")

    out = build_figure(base, refine, args.out,
                       constructed_order=(args.frame_order == "collapse-window"),
                       const_hi=const_hi)
    print(f"wrote {out}")
    for key, value in summarize(base, refine).items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
