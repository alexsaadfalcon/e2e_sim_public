"""
End-to-end OFDM communications link example.

Transmits random bits over an OFDM modem, pushes them through a frequency-domain
channel (a precomputed Sionna frame if available, otherwise a synthetic multipath
fallback), estimates the channel from pilots, equalizes, demaps, and sweeps SNR to
produce a BER-vs-SNR curve plus a received constellation.

Run:
    python -m e2e.main.main_comms_link

Outputs (saved under e2e/main/figures/, no display needed):
    comms_link_ber.png            BER vs SNR
    comms_link_constellation.png  RX constellation before/after equalization
"""

import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")            # headless: write figures to files, no display
import matplotlib.pyplot as plt

from e2e.scenario import munich_radar_scenario
from e2e.comms.ofdm import OFDMModem, random_bits
from e2e.comms import channel as ch


FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    rng = np.random.default_rng(0)

    # frequency grid comes from the scenario's FrequencyPlan
    scenario = munich_radar_scenario()
    freqs = scenario.frequency.linspace()
    carrier = scenario.frequency.carrier_hz

    # OFDM configuration (802.11-like: 64-point FFT, 52 active tones, comb pilots)
    bits_per_symbol = 4          # 16-QAM
    n_symbols = 32
    modem = OFDMModem(fft_size=64, cp_len=16, n_active=52, pilot_spacing=8,
                      bits_per_symbol=bits_per_symbol)
    # A comm link occupies a narrow band (here ~15 MHz over 64 tones), NOT the full
    # 3 GHz radar sweep -- so the channel is smooth across the OFDM band and pilot
    # interpolation is meaningful. (Using the whole radar bandwidth would alias a
    # wildly frequency-selective response onto a handful of tones.)
    subcarrier_spacing = 240e3

    # source the channel: real Sionna frame if present, else synthetic
    cfr_dense, source = ch.load_or_synthesize_cfr("munich", freqs, rng=rng)
    print(f"[comms_link] channel source: {source}")
    H_sc = ch.cfr_to_subcarriers(cfr_dense, freqs, modem.fft_size, carrier, subcarrier_spacing)

    # ---- SNR sweep -> BER ------------------------------------------------
    # NOTE: this loop is intentionally NOT replaced by comms.blocks.ModemBlock /
    # BERBlock. ModemBlock runs a single (estimator, equalizer) at a single SNR
    # with one shared seed, whereas this example sweeps SNR, runs BOTH ZF and MMSE
    # off the *same* LS estimate per point, uses a per-SNR channel noise seed
    # (100+snr), and captures a pre-/post-EQ constellation snapshot at SNR=20 dB.
    # Folding it into the blocks would change the seeds/structure and hence the
    # figures, so the link loop is kept inline here on purpose.
    snr_list = list(range(0, 31, 2))
    ber_zf, ber_mmse = [], []
    n_bits = n_symbols * modem.data_bits_per_symbol_block
    tx_bits = random_bits(n_bits, seed=1)
    tx_time, tx_freq = modem.modulate(tx_bits, n_symbols)

    constellation_snapshot = None
    for snr in snr_list:
        rx_freq, _ = ch.apply_channel(tx_freq, H_sc, snr, rng_seed=100 + snr)

        # pilot LS channel estimate -> equalize (ZF and MMSE)
        rx_pilots = modem.extract_pilots(rx_freq)
        tx_pilots = modem.pilot_grid(n_symbols)
        H_est = ch.ls_estimate(rx_pilots, tx_pilots, modem.pilot_idx, modem.fft_size)

        eq_zf = modem.extract_data(ch.zf_equalize(rx_freq, H_est))
        eq_mmse = modem.extract_data(ch.mmse_equalize(rx_freq, H_est, snr))

        from e2e.comms.ofdm import qam_demod
        rx_bits_zf = qam_demod(eq_zf.reshape(-1), bits_per_symbol, modem.const)
        rx_bits_mmse = qam_demod(eq_mmse.reshape(-1), bits_per_symbol, modem.const)
        ber_zf.append(ch.ber(tx_bits, rx_bits_zf))
        ber_mmse.append(ch.ber(tx_bits, rx_bits_mmse))

        if snr == 20:
            constellation_snapshot = (modem.extract_data(rx_freq).reshape(-1).cpu().numpy(),
                                      eq_mmse.reshape(-1).cpu().numpy())

    print("[comms_link] SNR(dB)  BER(ZF)     BER(MMSE)")
    for s, bz, bm in zip(snr_list, ber_zf, ber_mmse):
        print(f"            {s:5d}   {bz:.3e}   {bm:.3e}")

    # ---- plots -----------------------------------------------------------
    plt.figure()
    plt.semilogy(snr_list, np.clip(ber_zf, 1e-6, 1), "o-", label="ZF")
    plt.semilogy(snr_list, np.clip(ber_mmse, 1e-6, 1), "s-", label="MMSE")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title(f"OFDM 16-QAM BER vs SNR (channel: {source})")
    plt.grid(True, which="both")
    plt.legend()
    ber_path = os.path.join(FIG_DIR, "comms_link_ber.png")
    plt.savefig(ber_path, dpi=120, bbox_inches="tight")
    plt.close()

    if constellation_snapshot is not None:
        raw, eq = constellation_snapshot
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.scatter(raw.real, raw.imag, s=4, alpha=0.4)
        plt.title("RX data (pre-EQ)")
        plt.axis("equal"); plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.scatter(eq.real, eq.imag, s=4, alpha=0.4)
        plt.title("RX data (post-MMSE-EQ, SNR=20dB)")
        plt.axis("equal"); plt.grid(True)
        const_path = os.path.join(FIG_DIR, "comms_link_constellation.png")
        plt.savefig(const_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[comms_link] wrote {ber_path}")
        print(f"[comms_link] wrote {const_path}")


if __name__ == "__main__":
    main()
