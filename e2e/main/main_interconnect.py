"""Tutorial: a data-driven interconnect model in the pipeline.

The pipeline's ``InterconnectBlock`` filters the aperture frame by an interconnect
frequency response. By default that response is a placeholder 11-tap boxcar (a fixed
shape, no physical units). This example shows how to instead drive it from a simulated
Through-Silicon-Via (TSV) transfer function S21(f) that ships as CSV data
(``e2e/data/interconnect/tessera_tsv_s21.csv``), and how it behaves over the pipeline's
frequency band.

The CSV was produced by an external physics-informed TSV surrogate model (not vendored
here -- only the derived data is committed; see the data README). This tutorial needs
only the committed CSV + numpy/torch, so it is fully reproducible.

Run:  python -m e2e.main.main_interconnect
"""
import os

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


if __name__ == "__main__":
    main()
