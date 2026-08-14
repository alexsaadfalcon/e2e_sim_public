"""
Pilot-based channel-estimation example.

Sends OFDM pilots through a frequency-domain channel (precomputed Sionna frame if
available, else synthetic multipath), estimates the channel with LS and MMSE from
the pilots, compares estimated vs true channel, and sweeps SNR to show estimation
MSE vs SNR.

The MMSE estimate is a diagonal Wiener shrinkage whose signal/noise-power prior
is pooled over every pilot subcarrier AND every OFDM symbol in the frame (the
maximal averaging available) rather than derived per-pilot from the same noisy
samples it then shrinks -- the latter is a circular, high-variance prior that can
make "MMSE" worse than LS at low SNR. With the pooled prior, MMSE beats LS *on
average* at low SNR (see the Monte-Carlo regression test in
tests/test_comms_channel.py) and converges to LS as SNR grows and the shrinkage
vanishes; a single run like this script's, at any one SNR point, is one noise
realization and can still go either way by a small margin.

Run:
    python -m e2e.main.main_channel_estimation

Outputs (e2e/main/figures/):
    chanest_true_vs_est.png   |H| and phase, true vs LS/MMSE estimate (SNR=20dB)
    chanest_mse_vs_snr.png    channel-estimate MSE vs SNR (LS vs MMSE)
"""

import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from e2e.scenario import munich_radar_scenario
from e2e.comms.ofdm import OFDMModem
from e2e.comms import channel as ch
from e2e.viz import fig_dir


FIG_DIR = fig_dir(__file__)


def main():
    rng = np.random.default_rng(2)

    scenario = munich_radar_scenario()
    freqs = scenario.frequency.linspace()
    carrier = scenario.frequency.carrier_hz

    # denser pilots make the estimate easier to visualise
    modem = OFDMModem(fft_size=128, cp_len=32, n_active=104, pilot_spacing=4,
                      bits_per_symbol=2)
    # narrow comm band (~30 MHz over 128 tones), see note in main_comms_link.py
    subcarrier_spacing = 240e3
    n_symbols = 16

    cfr_dense, source = ch.load_or_synthesize_cfr("munich", freqs, rng=rng)
    print(f"[chanest] channel source: {source}")
    H_true = ch.cfr_to_subcarriers(cfr_dense, freqs, modem.fft_size, carrier, subcarrier_spacing)

    tx_pilots = modem.pilot_grid(n_symbols)
    # build a full TX frame (pilots + random data) so pilot extraction is realistic
    from e2e.comms.ofdm import random_bits
    tx_bits = random_bits(n_symbols * modem.data_bits_per_symbol_block, seed=3)
    _, tx_freq = modem.modulate(tx_bits, n_symbols)

    # ---- SNR sweep -> estimation MSE -------------------------------------
    # NOTE: kept inline rather than using comms.blocks.ModemBlock/BERBlock: this
    # example reports channel-estimation MSE (LS vs MMSE) over an SNR sweep and
    # never demaps/decodes, which the blocks don't expose. Reusing them would not
    # produce the MSE-vs-SNR figure this script exists to make.
    snr_list = list(range(0, 31, 2))
    mse_ls, mse_mmse = [], []
    snapshot = None
    for snr in snr_list:
        rx_freq, _ = ch.apply_channel(tx_freq, H_true, snr, rng_seed=50 + snr)
        rx_pilots = modem.extract_pilots(rx_freq)

        H_ls = ch.ls_estimate(rx_pilots, tx_pilots, modem.pilot_idx, modem.fft_size)
        H_mmse = ch.mmse_estimate(rx_pilots, tx_pilots, modem.pilot_idx, modem.fft_size, snr)

        # MSE only over the active band (where the channel is actually probed)
        mse_ls.append(ch.channel_mse(H_ls, H_true, modem.active_idx))
        mse_mmse.append(ch.channel_mse(H_mmse, H_true, modem.active_idx))
        if snr == 20:
            snapshot = (H_true.cpu().numpy(), H_ls.cpu().numpy(), H_mmse.cpu().numpy(),
                        modem.active_idx.cpu().numpy())

    print("[chanest] SNR(dB)  MSE(LS)     MSE(MMSE)")
    for s, ml, mm in zip(snr_list, mse_ls, mse_mmse):
        print(f"          {s:5d}   {ml:.3e}   {mm:.3e}")
    n_better = sum(1 for ml, mm in zip(mse_ls, mse_mmse) if mm <= ml)
    print(f"[chanest] MMSE MSE <= LS MSE at {n_better}/{len(snr_list)} SNR points this run "
          "(pooled empirical-Bayes prior; MMSE wins on average at low SNR over many trials, "
          "see the Monte-Carlo regression test, and converges to LS as SNR grows).")

    # ---- plots -----------------------------------------------------------
    plt.figure()
    plt.semilogy(snr_list, mse_ls, "o-", label="LS")
    plt.semilogy(snr_list, mse_mmse, "s-", label="MMSE")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Channel-estimate MSE")
    plt.title(f"Channel estimation MSE vs SNR (channel: {source})")
    plt.grid(True, which="both")
    plt.legend()
    mse_path = os.path.join(FIG_DIR, "chanest_mse_vs_snr.png")
    plt.savefig(mse_path, dpi=120, bbox_inches="tight")
    plt.close()

    if snapshot is not None:
        H_true_np, H_ls_np, H_mmse_np, active = snapshot
        band = np.sort(active)
        plt.figure(figsize=(9, 6))
        plt.subplot(2, 1, 1)
        plt.plot(band, np.abs(H_true_np[band]), "k-", label="true")
        plt.plot(band, np.abs(H_ls_np[band]), "C0.", ms=3, label="LS")
        plt.plot(band, np.abs(H_mmse_np[band]), "C1.", ms=3, label="MMSE")
        plt.ylabel("|H|"); plt.title("True vs estimated channel (SNR=20 dB)")
        plt.legend(); plt.grid(True)
        plt.subplot(2, 1, 2)
        plt.plot(band, np.angle(H_true_np[band]), "k-", label="true")
        plt.plot(band, np.angle(H_ls_np[band]), "C0.", ms=3, label="LS")
        plt.xlabel("subcarrier index"); plt.ylabel("phase (rad)")
        plt.legend(); plt.grid(True)
        cmp_path = os.path.join(FIG_DIR, "chanest_true_vs_est.png")
        plt.savefig(cmp_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[chanest] wrote {mse_path}")
        print(f"[chanest] wrote {cmp_path}")


if __name__ == "__main__":
    main()
