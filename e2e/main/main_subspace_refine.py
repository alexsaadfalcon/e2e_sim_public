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


def _recovered_windows(gaps, threshold, min_len=2, include_trailing=False):
    """Runs of frames ABOVE threshold that sit between two collapses — the recoveries.

    `include_trailing` also counts the stretch AFTER the final collapse. It is off by
    default because a run that merely ends without collapsing again has not demonstrated
    anything; but the `--frame-order collapse-window` sequence is built as
    pre / degenerate / recovery precisely so that the tail IS the recovery, and with a
    single collapse in the sequence the between-collapses rule finds nothing at all --
    so the figure silently loses the third act it was ordered to show.
    """
    runs = _degenerate_runs(gaps, threshold, min_len=1)
    out = []
    for (_, end_prev), (start_next, _) in zip(runs, runs[1:]):
        if start_next - end_prev - 1 >= min_len:
            out.append((end_prev + 1, start_next - 1))
    if include_trailing and runs:
        last_end = runs[-1][1]
        if len(gaps) - last_end - 1 >= min_len:
            out.append((last_end + 1, len(gaps) - 1))
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


def _median(xs):
    ys = sorted(xs)
    n = len(ys)
    if n == 0:
        return float("nan")
    return ys[n // 2] if n % 2 else 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def _fixed_effort_cost(series, runs, threshold=None, tail_factor=3.0):
    """What a collapse costs ONE fixed-effort arm. Returns a dict, not a mean.

    MEDIANS, not means, and the reason is the finding itself. An adversarial review
    recomputed both on the same 69-frame run and found they disagree by a factor of
    nearly three on the high-effort arm:

        1 pass/frame  : mean 2.43x, median 2.32x   -> agree; the cost is UNIFORM
        60 passes/frame: mean 4.48x, median 1.67x  -> disagree; the cost is a TAIL

    At 60 passes the typical collapsed frame barely notices the collapse (1.67x), but 8
    of 25 collapsed frames blow past three times the healthy median, the worst reaching
    ~20x it. Reporting the mean alone says "degeneracy costs 4.6x at high effort", which
    reads as a uniform degradation and is not what the data shows; reporting the median
    alone says "1.7x, barely anything" and hides the blow-ups that actually break a
    tracker. So this returns both the median ratio AND the size of the tail, and the
    figure prints both.

    Only meaningful for an arm whose effort does NOT change with the gap -- that is the
    whole point (see `build_figure`). Returns NaNs rather than raising when there is no
    collapse in the sequence, so a caller plotting a non-degenerate run still works.
    """
    nan = float("nan")
    empty = {"pre": nan, "deg": nan, "ratio": nan, "n_tail": 0, "n_deg": 0, "worst": nan}
    if not runs:
        return empty
    first_lo = runs[0][0]
    pre = [x for x in series[:first_lo] if x == x]
    idx = [i for lo, hi in runs for i in range(lo, hi + 1)]
    deg = [series[i] for i in idx if series[i] == series[i]]
    if not pre or not deg:
        return empty
    pre_m = _median(pre)
    deg_m = _median(deg)
    cut = tail_factor * pre_m
    return {
        "pre": pre_m,
        "deg": deg_m,
        "ratio": (deg_m / pre_m if pre_m > 0 else nan),
        "n_tail": sum(1 for x in deg if x > cut),
        "n_deg": len(deg),
        "worst": max(deg),
        "worst_ratio": (max(deg) / pre_m if pre_m > 0 else nan),
    }


def build_figure(base, refine, out_path, threshold=GAP_THRESHOLD, constructed_order=False,
                 const_hi=None):
    """Three stacked panels: tracking error, the singular-value gap, and effort spent.

    WHY THIS FIGURE IS LAID OUT THE WAY IT IS. An earlier version drew the reactive-gate
    arm as the visual headline, and a reader looking at it concluded the exact OPPOSITE
    of the finding: the gate's error line DIPS the instant the shaded collapse begins and
    leaps the instant it ends, so the figure appeared to say the tracker does BETTER when
    the scene is rank-deficient. The owner read it that way, which means it read that way.

    It is an artifact of a confound, not a result. `AdaOjaBlock.effective_n_refine` sets
    the gate arm's refinement passes FROM `sv_gap_norm` -- the same quantity that defines
    "degenerate" -- so that arm's compute steps 1 -> 60 -> 1 exactly at the shaded
    boundaries (measured: mean 1.0 outside, exactly 60.0 inside, zero variance). Its error
    curve therefore mixes two simultaneous causes into one line and cannot be read as a
    measurement of what degeneracy costs.

    What degeneracy costs has to be measured at FIXED effort, and both fixed-effort arms
    agree that it costs a lot (measured over 69 munich frames):

        1 pass/frame  : 0.537 pre-collapse -> 1.304 collapsed   (2.4x worse)
        60 passes/frame: 0.037 pre-collapse -> 0.164 collapsed   (4.5x worse)

    and the ground truth itself corroborates it -- the true subspace's frame-to-frame
    chordal drift rises 0.278 -> 0.817 (2.9x MORE rotation) while the gap is collapsed,
    with effective rank falling 76.8 -> 46.3. Degeneracy makes the target both lower-rank
    and faster-moving. So:

      * the two FIXED-EFFORT arms are drawn as solid, full-weight lines -- they are the
        honest measurement, and their shape (low, rising through the collapse, recovering)
        is the finding;
      * the reactive-gate arm is drawn thinner and dotted, and is annotated as bought,
        with the compute panel directly beneath it carrying the same shading;
      * the headline callout states the fixed-effort cost FIRST and the gate's compute
        trade second.
    """
    frames = range(len(refine["subspace_err"]))
    runs = _degenerate_runs(refine["sv_gap_norm"], threshold)
    recoveries = _recovered_windows(refine["sv_gap_norm"], threshold,
                                    include_trailing=constructed_order)

    # figsize 11.4 x 8.7 (was 9.0 x 8.4): a legibility review measured this figure's
    # bottom notes/legend/stats-box text at 6-10 px once scaled into the deck's
    # tracking_refine slide box (5.8 x 5.15 in, aspect 1.13) -- below the ~11 px
    # screen-share floor. The fonts below are raised to compensate. Width goes up more
    # than height: this box is close to SQUARE, so every extra wrapped LINE divides the
    # box-fit scale down (it is the binding, height-bound dimension) -- wider lets the
    # bottom notes wrap to fewer lines and the legend sit in one row instead of two.
    fig, (ax_err, ax_gap, ax_ref) = plt.subplots(
        3, 1, figsize=(10.3, 8.6), sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.35, 1.0], "hspace": 0.20})

    # --- the honest arms: effort held fixed, so the curve measures the SCENE ----------
    ax_err.plot(frames, base["subspace_err"], color=C_BASE, lw=2.2,
                label=f"fixed effort — {N_REFINE} pass/frame")
    if const_hi is not None:
        ax_err.plot(frames, const_hi["subspace_err"], color=C_CONST, lw=2.2,
                    label=f"fixed effort — {N_REFINE_HI} passes/frame")
    # --- the confounded arm: its own effort tracks the gap, so it is drawn as secondary
    ax_err.plot(frames, refine["subspace_err"], color=C_REFINE, lw=1.5, ls=":",
                label=f"reactive gate — {N_REFINE} to {N_REFINE_HI}, self-selected")
    ax_err.set_ylabel("subspace error", fontsize=12)
    ax_err.set_yscale("log")
    # Headroom ABOVE the data for the stats box, rather than moving the box around the
    # panel looking for a gap. On a log axis the curves already occupy nearly two
    # decades, so every in-axes corner is over some arm at some frame; a quarter-decade
    # of empty space at the top is the only placement that is safe by construction.
    # No y-headroom hack here any more. Reserving empty plot area for the stats box cost
    # nearly half the error panel once the box was raised to the 17 pt legibility floor,
    # which squashed the three curves into the bottom third. The box now lives ABOVE the
    # axes (see the fig.text near the end of this function), so it competes with nothing.
    fig.suptitle("Rank degeneracy costs accuracy at low effort, and the tail at high",
                 fontsize=16, fontweight="bold", y=0.995)

    ax_gap.plot(frames, refine["sv_gap_norm"], color=C_GATE, lw=1.8)
    ax_gap.axhline(threshold, color=C_REFINE, ls="--", lw=1.4,
                   label=f"gap_threshold = {threshold}")
    ax_gap.set_ylabel("sv_gap_norm", fontsize=12)
    ax_gap.set_yscale("log")
    # Lower RIGHT: the gap curve dips to its minimum in the left half of the collapsed
    # window, which is exactly where a lower-left legend sat. It is high on the right.
    ax_gap.legend(fontsize=17, loc="lower right", framealpha=0.95)

    if refine["n_refine_used"]:
        ax_ref.step(frames, refine["n_refine_used"], where="mid", color=C_REFINE, lw=1.8)
    ax_ref.set_ylabel("refinement\npasses used", fontsize=12)
    ax_ref.set_xlabel("frame", fontsize=12)

    for ax in (ax_err, ax_gap, ax_ref):
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)
        for lo, hi in runs:
            ax.axvspan(lo - 0.5, hi + 0.5, color="#000000", alpha=0.07, lw=0)
        for lo, hi in recoveries:
            ax.axvspan(lo - 0.5, hi + 0.5, color=C_REFINE, alpha=0.10, lw=0)

    if runs:
        lo, hi = max(runs, key=lambda r: r[1] - r[0])
        ax_gap.text((lo + hi) / 2.0, 0.94, "gap collapsed",
                    transform=ax_gap.get_xaxis_transform(), ha="center", va="top",
                    fontsize=11.5, color="#333333", fontstyle="italic")
    for lo, hi in recoveries:
        ax_gap.text((lo + hi) / 2.0, 0.82, "recovers",
                    transform=ax_gap.get_xaxis_transform(), ha="center", va="top",
                    fontsize=11.5, color=C_REFINE, fontweight="bold")

    # The one line on the compute panel that stops the gate arm being misread. It sits on
    # the panel that CAUSES the dip, directly under the dip.
    if refine["n_refine_used"] and runs:
        lo, hi = max(runs, key=lambda r: r[1] - r[0])
        ax_ref.annotate(f"the gate's dip above is bought here: "
                        f"{N_REFINE} → {N_REFINE_HI} passes",
                        xy=((lo + hi) / 2.0, N_REFINE_HI), xytext=(0.5, 0.42),
                        textcoords="axes fraction", ha="center", va="center",
                        fontsize=10.5, color=C_REFINE, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFFF",
                                  edgecolor=C_REFINE, linewidth=1.0, alpha=0.95))

    # --- the quotable numbers, computed from the SAME arrays that were plotted --------
    lines = []
    b = _fixed_effort_cost(base["subspace_err"], runs)
    if b["ratio"] == b["ratio"]:
        lines.append(f"at {N_REFINE} pass/frame: median error ×{b['ratio']:.1f} while "
                     f"collapsed ({b['pre']:.2f} → {b['deg']:.2f}), every frame")
    if const_hi is not None:
        c = _fixed_effort_cost(const_hi["subspace_err"], runs)
        if c["ratio"] == c["ratio"]:
            lines.append(f"at {N_REFINE_HI} passes/frame: median only ×{c['ratio']:.1f} "
                         f"— but {c['n_tail']} of {c['n_deg']} collapsed frames blow "
                         f"out, worst ×{c['worst_ratio']:.0f}")
    if refine["n_refine_used"] and const_hi is not None:
        used = refine["n_refine_used"]
        mean_gated = sum(used) / len(used) if used else float("nan")
        lines.append(f"the gate reaches the {N_REFINE_HI}-pass floor for "
                     f"{mean_gated / max(N_REFINE_HI, 1) * 100:.0f}% of its compute")
    if lines:
        # ABOVE the axes, not inside them. Every in-axes corner is over some arm at some
        # frame on a log plot spanning two decades, and the one corner that looked free
        # (upper left) is exactly where the 1-pass arm plateaus during the collapse.
        fig.text(0.5, 0.955, "\n".join(lines), ha="center", va="top", fontsize=15,
                 color="#222222",
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFFFF",
                           edgecolor=C_GATE, linewidth=1.1, alpha=0.95))
        # Make room for it: push the panels down by roughly the box's height.
        fig.subplots_adjust(top=0.955 - 0.037 * len(lines))

    handles, labels = ax_err.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.058),
               ncol=2, fontsize=17, framealpha=0.95)

    # fontsize=17 (was 9.0): same measured legibility floor as the legend/stats box
    # above. Room given via the taller figsize, not via any change to the note text.
    notes = [_ORACLE_NOTE] + ([_ORDER_NOTE] if constructed_order else [])
    fig.text(0.5, -0.012 - 0.038 * (len(notes) - 1), "\n".join(notes), ha="center",
             va="top", fontsize=17, color="#555555", fontstyle="italic", wrap=True)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def summarize(base, refine, threshold=GAP_THRESHOLD, include_trailing=False):
    """Numbers to print (and to quote in a caption) rather than eyeball off the figure.

    `include_trailing` is forwarded to `_recovered_windows` and must be passed the same
    way `build_figure` gets it, or these printed diagnostics describe a different set of
    recovery windows than the figure shades.
    """
    gaps = refine["sv_gap_norm"]
    runs = _degenerate_runs(gaps, threshold)
    recoveries = _recovered_windows(gaps, threshold, include_trailing=include_trailing)
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

    # Bound once and shared: the figure and the printed diagnostics must agree about
    # whether the tail after the final collapse counts as a recovery window.
    constructed_order = (args.frame_order == "collapse-window")
    out = build_figure(base, refine, args.out,
                       constructed_order=constructed_order,
                       const_hi=const_hi)
    print(f"wrote {out}")
    for key, value in summarize(base, refine,
                                include_trailing=constructed_order).items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
