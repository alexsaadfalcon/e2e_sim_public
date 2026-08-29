"""Tutorial: a data-driven interconnect model in the pipeline.

The pipeline's ``InterconnectBlock`` filters the aperture frame by an interconnect
frequency response. By default that response is a placeholder 11-tap boxcar (a fixed
shape, no physical units). This example shows how to instead drive it from simulated/
simulated interconnect transfer functions S21(f) that ship as CSV data
(``e2e/data/interconnect/*.csv``), and how they behave over their respective bands --
both as a raw |S21|(f) response and as a radar RANGE PROFILE (what actually reaches a
downstream detector).

The CSVs were produced by an external collaborator's models (not vendored here -- only
the derived data is committed; see the data README). This tutorial needs only the
committed CSVs + numpy/torch, so it is fully reproducible.

Run:  python -m e2e.main.main_interconnect
"""
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from e2e.blocks import (
    InterconnectBlock,
    load_interconnect_transfer,
    TESSERA_INTERCONNECT_CSV,
    device,
)

# NOT created here (unlike e2e.viz.fig_dir, which creates eagerly): this tutorial's
# `main(show=False)` path must never touch the filesystem (see
# tests/test_interconnect.py::test_tutorial_smoke_runs_without_disk), so the
# directory is made lazily below, only when a figure is actually about to be
# written -- kept as a module-level FIG_DIR (rather than e2e.viz.fig_dir(__file__))
# so that contract holds and the constant stays monkeypatchable by that test.
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")

BAND = (28.5e9, 31.5e9)   # the default pipeline FrequencyPlan band
N_FREQS = 512

# The 77 GHz-automotive-band interconnect models: S21(f) resampled from the collaborator's
# HFSS exports, phase reconstructed as minimum-phase (see each CSV's own header).
#
# Case3 was long the only one shipped, on the collaborator's statement that it is the
# WORST-PERFORMING of their six designs -- a conservative validation bound rather than a
# representative one. With all six now derived, that statement is MEASURED rather than
# taken on trust: Case3's median in-band insertion loss is -0.541 dB with 0.113 dB of
# ripple, against -0.249 to -0.291 dB and <=0.030 dB ripple for the other five. It is
# indeed the worst, by roughly a factor of two in loss and four in ripple.
_INTERCONNECT_DATA = Path(__file__).resolve().parent.parent / "data" / "interconnect"


def case_csv(n):
    """Path to the derived transfer function for Tessera CaseN (N = 1..6)."""
    return _INTERCONNECT_DATA / f"tessera_case{int(n)}_s21_77ghz.csv"


#: All six of the collaborator's HFSS-simulated designs. As of 2026-08-19 the owner
#: cleared the derived CSVs for this repository, so the "other five are private, not even
#: their numbers" restriction above no longer holds and the figures show the whole set.
#: Derived by `notes/tools/derive_interconnect_csv.py`, which re-derives Case3 and checks
#: it against the file shipped on 2026-08-10 (max deviation 5.0e-07 dB) before it is
#: trusted on a new case.
CASE_NUMBERS = (1, 2, 3, 4, 5, 6)
CASE3_INTERCONNECT_CSV = case_csv(3)
CASE3_BAND = (76e9, 81e9)   # automotive radar band the Case exports are used over

# Separate from FIG_DIR (ephemeral, e2e/main/figures/): this is committed, README-facing
# media, so it is a distinct path -- kept as a module global (not a bound default
# parameter) for the same monkeypatch-for-tests reason as FIG_DIR above.
RANGE_PROFILE_FIG_PATH = (
    Path(__file__).resolve().parents[2] / "docs" / "media" / "interconnect_range_profiles.png"
)

# The README's top-of-page gallery figure: a compact, docs-facing "before/after" view of
# the same 4 arms as `range_profile_comparison`, distinct from RANGE_PROFILE_FIG_PATH
# (which also carries the raw |S21|(f) sweeps). Kept as its own module global for the
# same monkeypatch-for-tests reason as the paths above.
BEFORE_AFTER_FIG_PATH = (
    Path(__file__).resolve().parents[2] / "docs" / "media" / "interconnect_before_after.png"
)

# Shared per-arm styling for every multi-arm figure in this module. Colors are chosen to
# be distinct, saturated hues that stay distinguishable after video-call compression --
# NOT relying on linestyle alone (legacy_boxcar used to be "tab:gray", which desaturates
# toward black/ideal on a compressed screen share; tab:purple is unambiguous next to
# black/blue/red). Linestyle still varies per arm as a second, redundant cue.
ARM_COLORS = {"ideal": "black", "tessera_tsv": "tab:blue",
              "tessera_case3": "tab:red", "legacy_boxcar": "tab:purple"}
ARM_STYLES = {"ideal": "-", "tessera_tsv": "--", "tessera_case3": ":",
              "legacy_boxcar": "-."}

#: Per-case styling for the all-six figure. Case3 keeps the red it has everywhere else in
#: this module, so a reader moving between figures does not have to re-learn it; the other
#: five take distinct hues. Linestyle is a second, redundant cue for compressed video.
CASE_COLORS = {1: "tab:olive", 2: "tab:orange", 3: "tab:red",
               4: "tab:green", 5: "tab:brown", 6: "tab:cyan"}
CASE_STYLES = {1: "--", 2: "-.", 3: ":", 4: "--", 5: "-.", 6: ":"}


def main(show=True, band=BAND, n_freqs=N_FREQS):
    """Load the shipped interconnect model, show how InterconnectBlock applies it, and
    (when `show`) save a figure. Returns a dict of the computed quantities."""
    # 1) The shipped interconnect model data: S21(f) over the model's full sweep.
    freq, s21 = load_interconnect_transfer(TESSERA_INTERCONNECT_CSV)
    s21_db = 20 * np.log10(np.abs(s21) + 1e-12)

    # 2) How InterconnectBlock consumes it: resample S21 onto the frame's band grid and
    #    apply as `frame * S21(f)`. A flat unit frame isolates the response (frame -> H).
    frame = torch.ones(2, 2, 1, n_freqs, dtype=torch.complex64, device=device)
    ic = InterconnectBlock(transfer_csv=TESSERA_INTERCONNECT_CSV, band_hz=band)
    H = ic.apply_interconnect(frame)[0, 0, 0, :].cpu().numpy()
    band_freqs = np.linspace(band[0], band[1], n_freqs)
    H_db = 20 * np.log10(np.abs(H) + 1e-12)
    result = {"freq": freq, "s21": s21, "band_freqs": band_freqs, "H": H,
              "band_loss_db": (float(H_db.min()), float(H_db.max()))}

    if show:
        print(f"Interconnect model: {len(freq)} pts, "
              f"{freq[0]/1e9:.1f}-{freq[-1]/1e9:.1f} GHz")
        print(f"Insertion loss over the pipeline band {band[0]/1e9:.1f}-"
              f"{band[1]/1e9:.1f} GHz: {H_db.min():.2f} .. {H_db.max():.2f} dB")
        print("\nWire it into the pipeline with:")
        print("  from e2e.blocks import InterconnectBlock, TESSERA_INTERCONNECT_CSV")
        print("  ic = InterconnectBlock(transfer_csv=TESSERA_INTERCONNECT_CSV,")
        print("                         band_hz=(28.5e9, 31.5e9))")
        print("  Simulation(..., interconnect_block=ic)   # replaces the boxcar")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=130)
        ax1.plot(freq / 1e9, s21_db, "-", color="tab:blue")
        ax1.axvspan(band[0] / 1e9, band[1] / 1e9, color="tab:orange", alpha=0.25,
                    label="pipeline band")
        ax1.set_xlabel("frequency (GHz)"); ax1.set_ylabel("|S21| (dB)")
        ax1.set_title("Shipped TSV interconnect model S21(f)")
        ax1.grid(True, alpha=0.3); ax1.legend(loc="lower left")

        ax2.plot(band_freqs / 1e9, H_db, "-", color="tab:orange", lw=2)
        ax2.set_xlabel("frequency (GHz)"); ax2.set_ylabel("applied |S21| (dB)")
        ax2.set_title("Response InterconnectBlock applies over the band\n"
                      "(S21 resampled onto the frame's frequency grid)")
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        os.makedirs(FIG_DIR, exist_ok=True)
        out = os.path.join(FIG_DIR, "interconnect_tessera_model.png")
        fig.savefig(out, bbox_inches="tight")
        print("saved", out)

    return result


def _band_insertion_loss_ripple(freq, s21, band):
    """Mean insertion loss and peak-to-peak ripple (both dB of |S21|) over `band`,
    computed from the CSV's own native sample points inside the band (no
    interpolation) -- the most direct, least-assumption re-derivation of those two
    numbers from the shipped data."""
    mask = (freq >= band[0]) & (freq <= band[1])
    db = 20 * np.log10(np.abs(s21[mask]) + 1e-12)
    return float(db.mean()), float(db.max() - db.min())


def _native_range_profile_db(H):
    """Range profile of a frequency response `H` (numpy, 1-D), matching
    RangeProfileBlock's own convention exactly: a windowless forward FFT over the
    frequency axis, fftshifted, magnitude. Deliberately NATIVE resolution -- no
    zero-padding/interpolation -- so this is exactly the per-bin range profile the
    pipeline itself would produce, not an interpolated estimate (zero-padding a
    profile that includes the legacy boxcar's sharp-edged impulse response produces
    Gibbs-ringing artifacts that don't reflect anything about the pipeline)."""
    H = np.asarray(H)
    n = H.shape[-1]
    prof = np.fft.fftshift(np.fft.fft(H))
    mag = np.abs(prof)
    bins = np.arange(n) - n // 2
    mag_db = 20 * np.log10(mag / mag.max() + 1e-15)
    return bins, mag_db


def _mainlobe_metrics(mag_db):
    """-3 dB mainlobe width (integer range bins) and peak sidelobe (dB) of a
    NATIVE-resolution range profile (see `_native_range_profile_db`).

    The mainlobe is the contiguous run of bins around the peak that are within 3 dB
    of it (tie-tolerant, so a flat-topped response like the legacy boxcar's reads as
    one wide mainlobe rather than being spuriously split into many equal-height
    'sidelobes'). Peak sidelobe is the largest value outside that run.
    """
    peak = int(np.argmax(mag_db))
    above = mag_db >= -3.0
    lo = peak
    while lo - 1 >= 0 and above[lo - 1]:
        lo -= 1
    hi = peak
    while hi + 1 < len(above) and above[hi + 1]:
        hi += 1
    width = hi - lo + 1
    mask = np.ones(len(mag_db), dtype=bool)
    mask[lo:hi + 1] = False
    peak_sidelobe = float(mag_db[mask].max()) if mask.any() else float("-inf")
    return width, peak_sidelobe


def _interconnect_arms(all_cases=False):
    """The arms shared by every multi-arm figure in this module.

    Default is the 4-arm set (ideal, the two measured Tessera datasets, and the legacy
    boxcar placeholder) -- see `range_profile_comparison`'s docstring for what each one is
    and is not. `all_cases=True` swaps the single Case3 arm for all SIX 77 GHz designs,
    which is what the before/after figure wants: the interesting question there is how the
    simulated designs differ from EACH OTHER, and one of them cannot answer it.
    """
    arms = {
        "ideal": ("ideal (no interconnect)",
                  InterconnectBlock(case='case3')),
        "tessera_tsv": ("Tessera TSV (Ka-band, 28.5-31.5 GHz)",
                         InterconnectBlock(transfer_csv=TESSERA_INTERCONNECT_CSV, band_hz=BAND)),
    }
    if all_cases:
        for n in CASE_NUMBERS:
            arms[f"tessera_case{n}"] = (
                f"Tessera Case{n} (77 GHz auto)",
                InterconnectBlock(transfer_csv=case_csv(n), band_hz=CASE3_BAND))
    else:
        arms["tessera_case3"] = (
            "Tessera Case3 (77 GHz auto, worst of 6 measured)",
            InterconnectBlock(transfer_csv=CASE3_INTERCONNECT_CSV, band_hz=CASE3_BAND))
    arms["legacy_boxcar"] = ("legacy 11-tap boxcar (placeholder)", InterconnectBlock())
    return arms


def _arm_color_style(key):
    """(color, linestyle) for an arm key, covering the per-case keys too."""
    if key in ARM_COLORS:
        return ARM_COLORS[key], ARM_STYLES[key]
    n = int(key.rsplit("case", 1)[-1])
    return CASE_COLORS[n], CASE_STYLES[n]


def _arm_profiles_and_metrics(arms, n_freqs):
    """Apply each arm to a flat (all-ones) frame and range-compress at native
    resolution (see `_native_range_profile_db`). Returns (profiles, metrics) dicts
    keyed the same as `arms`."""
    def _ones(n):
        return torch.ones(2, 2, 1, n, dtype=torch.complex64, device=device)

    profiles, metrics = {}, {}
    for key, (label, blk) in arms.items():
        H = blk.apply_interconnect(_ones(n_freqs))[0, 0, 0, :].cpu().numpy()
        bins, mag_db = _native_range_profile_db(H)
        width, sidelobe = _mainlobe_metrics(mag_db)
        profiles[key] = (bins, mag_db)
        metrics[key] = {"label": label, "width_3db_bins": width, "peak_sidelobe_db": sidelobe}
    return profiles, metrics


def range_profile_comparison(show=True, n_freqs=N_FREQS):
    """Compare what different interconnect models do to the radar RANGE PROFILE --
    the product that actually reaches a downstream detector -- rather than only to
    the raw |S21|(f) response.

    Four arms, each `InterconnectBlock.apply_interconnect` applied to a flat
    (all-ones) frame so the output IS the arm's frequency response, then range-
    compressed exactly as `RangeProfileBlock` does (windowless FFT over frequency):

    - **ideal**: `case='case3'` identity pass-through -- no interconnect at all.
    - **Tessera TSV**: the Ka-band (28.5-31.5 GHz) TSV surrogate (`TESSERA_INTERCONNECT_CSV`).
    - **Tessera Case3**: the 76-81 GHz automotive-band export (`CASE3_INTERCONNECT_CSV`)
      -- deliberately the WORST of six HFSS-simulated designs the collaborator supplied
      (see the module-level comment by `CASE3_INTERCONNECT_CSV`); this is a conservative
      validation bound, not a best case, and no other of the six is referenced anywhere.
    - **legacy boxcar**: `InterconnectBlock()`'s default 11-tap placeholder.

    TSV and Case3 model DIFFERENT physical structures in NON-OVERLAPPING bands (Ka-band
    vs. 77 GHz automotive) -- this function deliberately does NOT rank one against the
    other; each is shown only against its own band and against the band-agnostic ideal/
    legacy arms. Range-profile bin metrics (mainlobe width, sidelobe) ARE compared
    directly across all four arms because the range axis is a dimensionless bin index,
    not a physical frequency -- band-agnostic by construction.

    Returns a dict of every computed quantity (insertion loss/ripple, per-arm mainlobe
    width and peak sidelobe). When `show`, also builds and saves a comparison figure to
    `RANGE_PROFILE_FIG_PATH` (a fixed, docs-facing path -- unlike `main()`, this is not
    swept up by `main(show=False)`'s no-filesystem-writes contract, but follows the same
    pattern: with `show=False` this function itself touches no disk).
    """
    freq_tsv, s21_tsv = load_interconnect_transfer(TESSERA_INTERCONNECT_CSV)
    freq_c3, s21_c3 = load_interconnect_transfer(CASE3_INTERCONNECT_CSV)
    il_tsv, ripple_tsv = _band_insertion_loss_ripple(freq_tsv, s21_tsv, BAND)
    il_c3, ripple_c3 = _band_insertion_loss_ripple(freq_c3, s21_c3, CASE3_BAND)

    arms = _interconnect_arms()
    profiles, metrics = _arm_profiles_and_metrics(arms, n_freqs)

    result = {
        "insertion_loss_db": {"tessera_tsv": il_tsv, "tessera_case3": il_c3},
        "ripple_db": {"tessera_tsv": ripple_tsv, "tessera_case3": ripple_c3},
        "metrics": metrics,
    }

    if show:
        print("\nRange-profile comparison (native resolution, "
              f"n_freqs={n_freqs}):")
        for key, (label, _) in arms.items():
            m = metrics[key]
            print(f"  {label}: -3 dB width {m['width_3db_bins']} bin(s), "
                  f"peak sidelobe {m['peak_sidelobe_db']:.1f} dB")
        print(f"Tessera TSV: insertion loss {il_tsv:.2f} dB, ripple {ripple_tsv:.2f} dB p-p "
              f"over {BAND[0]/1e9:.1f}-{BAND[1]/1e9:.1f} GHz")
        print(f"Tessera Case3 (worst of 6 measured): insertion loss {il_c3:.2f} dB, "
              f"ripple {ripple_c3:.2f} dB p-p over {CASE3_BAND[0]/1e9:.1f}-"
              f"{CASE3_BAND[1]/1e9:.1f} GHz")

        fig, ((ax_s21_tsv, ax_s21_c3), (ax_prof, ax_prof_zoom)) = plt.subplots(
            2, 2, figsize=(15, 11), dpi=140)
        fs_title, fs_label, fs_annot, fs_tick = 15, 13, 11.5, 11

        # (a1) TSV |S21|(f) over its full sweep, band shaded, IL/ripple annotated.
        db_tsv_full = 20 * np.log10(np.abs(s21_tsv) + 1e-12)
        ax_s21_tsv.plot(freq_tsv / 1e9, db_tsv_full, color="tab:blue", lw=2)
        ax_s21_tsv.axvspan(BAND[0] / 1e9, BAND[1] / 1e9, color="tab:orange", alpha=0.25,
                            label="Ka-band (pipeline)")
        ax_s21_tsv.set_title("Tessera TSV: |S21| (1-40 GHz sweep)", fontsize=fs_title)
        ax_s21_tsv.set_xlabel("frequency (GHz)", fontsize=fs_label)
        ax_s21_tsv.set_ylabel("|S21| (dB)", fontsize=fs_label)
        ax_s21_tsv.text(0.97, 0.04,
                         f"in-band ({BAND[0]/1e9:.1f}-{BAND[1]/1e9:.1f} GHz):\n"
                         f"insertion loss {il_tsv:.2f} dB\nripple {ripple_tsv:.2f} dB p-p",
                         transform=ax_s21_tsv.transAxes, fontsize=fs_annot, va="bottom",
                         ha="right",
                         bbox=dict(boxstyle="round", fc="white", ec="tab:blue", alpha=0.9))
        ax_s21_tsv.legend(loc="upper right", fontsize=fs_annot)
        ax_s21_tsv.grid(True, alpha=0.3)
        ax_s21_tsv.tick_params(labelsize=fs_tick)

        # (a2) Case3 |S21|(f), band shaded, IL/ripple annotated, worst-of-6 caveat.
        db_c3_full = 20 * np.log10(np.abs(s21_c3) + 1e-12)
        ax_s21_c3.plot(freq_c3 / 1e9, db_c3_full, color="tab:red", lw=2)
        ax_s21_c3.axvspan(CASE3_BAND[0] / 1e9, CASE3_BAND[1] / 1e9, color="tab:orange",
                           alpha=0.25, label="77 GHz auto band (pipeline)")
        ax_s21_c3.set_title("Tessera Case3: |S21| (70-90 GHz sweep)\n"
                             "worst of 6 simulated designs -- conservative pick",
                             fontsize=fs_title)
        ax_s21_c3.set_xlabel("frequency (GHz)", fontsize=fs_label)
        ax_s21_c3.set_ylabel("|S21| (dB)", fontsize=fs_label)
        ax_s21_c3.text(0.03, 0.05,
                        f"in-band ({CASE3_BAND[0]/1e9:.0f}-{CASE3_BAND[1]/1e9:.0f} GHz):\n"
                        f"insertion loss {il_c3:.2f} dB\nripple {ripple_c3:.2f} dB p-p",
                        transform=ax_s21_c3.transAxes, fontsize=fs_annot, va="bottom",
                        bbox=dict(boxstyle="round", fc="white", ec="tab:red", alpha=0.9))
        ax_s21_c3.legend(loc="upper right", fontsize=fs_annot)
        ax_s21_c3.grid(True, alpha=0.3)
        ax_s21_c3.tick_params(labelsize=fs_tick)

        # (b) Range profile, all 4 arms, native resolution -- y-window sized from the
        # data to resolve the mainlobe-width story (legacy's flat ~11-bin shelf vs. the
        # other three's single-bin needle), not wasted on an empty deep-dB region.
        colors, styles = ARM_COLORS, ARM_STYLES
        x_lim = 16
        for key, (label, _) in arms.items():
            bins, mag_db = profiles[key]
            m = metrics[key]
            leg = (f"{label}\n(-3 dB width {m['width_3db_bins']} bin"
                   f"{'s' if m['width_3db_bins'] != 1 else ''}, "
                   f"sidelobe {m['peak_sidelobe_db']:.0f} dB)")
            ax_prof.plot(bins, mag_db, styles[key], color=colors[key], lw=2.2,
                         marker="o", ms=3.5, label=leg)
        ax_prof.set_xlim(-x_lim, x_lim)
        ax_prof.set_ylim(-22, 3)
        ax_prof.set_xlabel("range bin (native, n_freqs={})".format(n_freqs), fontsize=fs_label)
        ax_prof.set_ylabel("range profile (dB, rel. peak)", fontsize=fs_label)
        ax_prof.set_title("Range profile: mainlobe width", fontsize=fs_title)
        ax_prof.grid(True, alpha=0.3)
        # lower-right is empty for every arm here (legacy's shelf is at y=0 and only
        # spans x in [-10, 0]; everything else drops to the dB floor within 1-2 bins)
        ax_prof.legend(loc="lower right", fontsize=fs_annot - 1.5)
        ax_prof.tick_params(labelsize=fs_tick)

        # (c) Same data, zoomed to the fine dB floor beneath the mainlobe -- the only
        # place the two physically-modelled interconnects show ANY visible cost: a
        # sub-mainlobe skirt from their small in-band ripple/tilt. Ideal is an exact
        # discrete delta at this resolution (no skirt at all) and the legacy boxcar's
        # off-shelf bins sit at the float32 noise floor (~-140 dB) -- both plot below
        # this window's floor, which is itself the honest result, not a hidden one.
        for key, (label, _) in arms.items():
            bins, mag_db = profiles[key]
            ax_prof_zoom.plot(bins, mag_db, styles[key], color=colors[key], lw=2.2,
                              marker="o", ms=3.5, label=label)
        ax_prof_zoom.set_xlim(-x_lim, x_lim)
        ax_prof_zoom.set_ylim(-95, -20)
        ax_prof_zoom.set_xlabel("range bin (native, n_freqs={})".format(n_freqs), fontsize=fs_label)
        ax_prof_zoom.set_ylabel("range profile (dB, rel. peak)", fontsize=fs_label)
        ax_prof_zoom.set_title("Same data, zoomed: sub-mainlobe ripple skirt\n"
                                "(ideal has none here; legacy boxcar is below floor)",
                                fontsize=fs_title)
        ax_prof_zoom.grid(True, alpha=0.3)
        ax_prof_zoom.legend(loc="upper right", fontsize=fs_annot)
        ax_prof_zoom.tick_params(labelsize=fs_tick)

        fig.suptitle(
            "Interconnect models vs. radar range profile\n"
            "TSV (Ka-band) and Case3 (77 GHz, the worst of 6 simulated designs) are each "
            "shown only against their OWN band -- not a head-to-head ranking",
            fontsize=fs_annot + 1.5, y=1.04)
        fig.tight_layout()
        out_path = RANGE_PROFILE_FIG_PATH
        os.makedirs(os.path.dirname(str(out_path)), exist_ok=True)
        fig.savefig(str(out_path), bbox_inches="tight")
        plt.close(fig)
        print("saved", out_path)

    return result


def before_after_comparison(show=True, n_freqs=N_FREQS):
    """Compact "before/after" range-profile figure for the README's top-of-page
    gallery (`BEFORE_AFTER_FIG_PATH`): the legacy placeholder against the
    simulation-driven arms, all 4 at once.

    This is deliberately a smaller figure than `range_profile_comparison` (no raw
    |S21|(f) sweeps, just the range-profile story), but it hits the same
    distinguishability problem: 3 of the 4 arms collapse to a near-ideal single-bin
    spike at native resolution, so an overlay with ONE linear y-axis makes them look
    identical. This uses two panels sharing x but not y -- a "broken axis" in effect
    -- so every arm is visible on its own natural scale:
      (top) the mainlobe itself, where the legacy boxcar's 11-bin smear is obviously
        different in WIDTH from the other three's 1-bin needles;
      (bottom) the same data zoomed into the sub-mainlobe skirt, where ideal/TSV/
        Case3 -- indistinguishable up top -- separate by DEPTH (their differing
        in-band ripple sets differing sidelobe floors).
    ALL SIX 77 GHz cases are drawn here, not just Case3. The owner's note was
    "multiple interconnects shown, with y axis scaled to correctly distinguish them --
    use Case1-Case6", and the derived CSVs for the other five landed 2026-08-19. This is
    the figure where that matters: the interesting question is how the simulated designs
    differ from EACH OTHER, and a single case cannot answer it. It also turns the
    collaborator's "Case3 is the worst of the six" from a statement into something the
    reader can see -- Case3's skirt sits visibly above the other five.

    Same caveats as `range_profile_comparison` apply and are stated on the figure: TSV is
    a different, non-overlapping band from the Case set (not a ranking).
    """
    arms = _interconnect_arms(all_cases=True)
    profiles, metrics = _arm_profiles_and_metrics(arms, n_freqs)
    x_lim = 16

    if show:
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8.5, 9), dpi=140,
                                              sharex=True)
        fs_title, fs_label, fs_annot, fs_tick = 14, 12.5, 11, 10.5

        for key, (label, _) in arms.items():
            bins, mag_db = profiles[key]
            m = metrics[key]
            color, style = _arm_color_style(key)
            # Top panel legend is about WIDTH, which is the only thing separating arms
            # there; the sidelobe number belongs with the bottom panel, where depth is
            # what separates them. With eight arms, one combined legend per panel is
            # already at the limit of what fits.
            leg = (f"{label} (-3 dB width {m['width_3db_bins']} bin"
                   f"{'s' if m['width_3db_bins'] != 1 else ''})")
            ax_top.plot(bins, mag_db, style, color=color, lw=2.2,
                        marker="o", ms=3.5, label=leg)
            ax_bot.plot(bins, mag_db, style, color=color, lw=2.2, marker="o", ms=3.5,
                        label=f"{label}  sidelobe {m['peak_sidelobe_db']:.0f} dB")

        ax_top.set_xlim(-x_lim, x_lim)
        ax_top.set_ylim(-22, 3)
        ax_top.set_ylabel("range profile (dB, rel. peak)", fontsize=fs_label)
        ax_top.set_title("BEFORE vs. AFTER: mainlobe width\n"
                          "the legacy placeholder smears 1 bin -> 11; all seven "
                          "simulation-driven arms stay at native resolution",
                          fontsize=fs_title)
        ax_top.grid(True, alpha=0.3)
        # No per-panel legend: with nine arms it filled a third of the axes and sat on the
        # mainlobe. One shared legend below the figure serves both panels (see below).
        ax_top.tick_params(labelsize=fs_tick)

        ax_bot.set_xlim(-x_lim, x_lim)
        # Down to -100: Cases 4/5/6 bottom out at -92 to -93 dB, so a -95 floor clipped
        # the three arms that are hardest to tell apart in the first place.
        ax_bot.set_ylim(-100, -20)
        ax_bot.set_xlabel(f"range bin (native, n_freqs={n_freqs})", fontsize=fs_label)
        ax_bot.set_ylabel("range profile (dB, rel. peak)", fontsize=fs_label)
        ax_bot.set_title("Same data, y-axis broken and zoomed to the skirt\n"
                          "the ONLY place the simulation-driven arms separate -- and "
                          "they order exactly as their in-band ripple does",
                          fontsize=fs_title)
        ax_bot.grid(True, alpha=0.3)
        ax_bot.tick_params(labelsize=fs_tick)

        # ONE shared legend, below the axes. Nine arms cannot go in an in-axes legend
        # without covering the data they describe -- the bottom panel's legend was sitting
        # squarely on the mainlobe. The sidelobe number rides in the label because depth is
        # what this figure is actually comparing.
        handles, labels = ax_bot.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.055),
                   ncol=3, fontsize=fs_annot - 0.5, framealpha=0.95)

        fig.suptitle(
            "Interconnect range response: legacy placeholder vs. the "
            "simulation-driven models\n"
            "All six 77 GHz Tessera designs, plus the Ka-band TSV -- which is a "
            "DIFFERENT band, so it is not ranked against them",
            fontsize=fs_annot + 1.5, y=1.02)
        fig.tight_layout(rect=(0, 0.085, 1, 1))
        out_path = BEFORE_AFTER_FIG_PATH
        os.makedirs(os.path.dirname(str(out_path)), exist_ok=True)
        fig.savefig(str(out_path), bbox_inches="tight")
        plt.close(fig)
        print("saved", out_path)

    return {"metrics": metrics}


def _build_arg_parser():
    """Exists so that `--help` PRINTS HELP instead of running the tutorial.

    Without an argument parser this module's `__main__` ran all three figure builders
    unconditionally, whatever was on the command line -- so `python -m
    e2e.main.main_interconnect --help`, the first thing anyone types to find out what a
    module does, silently spent minutes recomputing and OVERWRITING two files that are
    tracked in git (`docs/media/interconnect_range_profiles.png` and
    `interconnect_before_after.png`), leaving a dirty working tree. `docs/PHYSICS.md`
    points new readers straight at this module, so that was on the onboarding path.

    Running with no arguments behaves exactly as before.
    """
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m e2e.main.main_interconnect",
        description="Tutorial: drive InterconnectBlock from measured S21(f) CSVs and "
                    "compare the resulting range profiles. Writes three figures.",
    )
    p.add_argument("--out-dir", default=None,
                   help="write the figures here instead of the tracked docs/media/ "
                        "paths -- use this if you do not want a dirty working tree")
    p.add_argument("--only", choices=("response", "range-profile", "before-after"),
                   default=None, help="build just one of the three figures")
    return p


if __name__ == "__main__":
    _args = _build_arg_parser().parse_args()
    if _args.out_dir:
        _out = Path(_args.out_dir)
        _out.mkdir(parents=True, exist_ok=True)
        RANGE_PROFILE_FIG_PATH = _out / Path(str(RANGE_PROFILE_FIG_PATH)).name
        BEFORE_AFTER_FIG_PATH = _out / Path(str(BEFORE_AFTER_FIG_PATH)).name
    if _args.only in (None, "response"):
        main()
    if _args.only in (None, "range-profile"):
        range_profile_comparison()
    if _args.only in (None, "before-after"):
        before_after_comparison()
