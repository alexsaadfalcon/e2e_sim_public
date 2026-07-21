"""
Joint radar + communications (ISAC) example.

Loads the multi-node `munich_isac_scenario()` (a car-mounted radar AND a
building->car comm link in the same scene), splits it into its sensing and comm
sub-problems, and runs both:

* sensing: a range/angle map from the radar node's 32x32 S-parameters
           (precomputed Sionna munich frame if present, else synthetic), plus the
           estimated target range.
* comm:    an OFDM link over the comm-link channel, reporting BER and EVM.

The same OFDM waveform parameters drive both (the shared-waveform idea): the comm
data subcarriers carry bits while the full occupied band is reused by the sensing
range estimator.

Run:
    python -m e2e.main.main_isac

Outputs (e2e/main/figures/):
    isac_range_angle.png   sensing range/angle map
    isac_constellation.png comm RX constellation (post-EQ)
"""

import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from e2e.scenario import munich_isac_scenario
from e2e.comms.ofdm import OFDMModem, qam_demod, random_bits
from e2e.comms import channel as ch
from e2e.comms import isac


FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

N_RX_X = N_RX_Y = 32


def _radar_s_pars(scenario, freqs, rng, src_band=None):
    """Get [N_RX, N_FREQS] S-parameters for the radar node.

    Uses the precomputed munich frame if available (shape [N_RX, TX, chirp, F]),
    else synthesises an independent multipath channel per array element so the
    range/angle map is well-defined. Delegates the reshape/resample to
    ``channel.frame_to_cfr`` with ``element=None`` (all N_RX array elements,
    unlike ``load_or_synthesize_cfr``'s single spatial channel).

    `src_band` : optional ``(f_start_hz, f_stop_hz)`` for the loaded frame's actual
    band; defaults to ``(freqs[0], freqs[-1])`` (assume the pkl spans `freqs`),
    see ``channel.frame_to_cfr``.
    """
    n_rx = N_RX_X * N_RX_Y
    arr = None
    try:
        from e2e.environment import sionna_iterator as si
        it = si.SionnaMunichIterator()             # raises if .pkl missing
        arr = it[0]
    except Exception:
        arr = None                                 # .pkl missing -> synthetic below

    if arr is not None:
        arr_np = np.asarray(arr)
        # The aperture geometry (N_RX_X x N_RX_Y) is fixed by the range/angle map;
        # the loaded frame MUST provide exactly that many elements or the spatial
        # FFT would be fed uninitialised/garbage rows. Fail loudly (clear error)
        # rather than silently using zeros or falling back to synthetic.
        if arr_np.shape[0] != n_rx:
            raise ValueError(
                f"munich frame has {arr_np.shape[0]} array elements but the "
                f"range/angle map expects N_RX_X*N_RX_Y = {n_rx}; cannot map a "
                f"frame with a different antenna count onto this aperture "
                f"(set N_RX_X/N_RX_Y to match, or provide a matching frame).")
        out = ch.frame_to_cfr(arr_np, freqs, src_band=src_band, element=None)
        return out, "sionna:munich"

    # synthetic fallback (no .pkl): shared target delay (~30 m) + per-element
    # random taps so the aperture sees a coherent point target plus clutter.
    c = 299_792_458.0
    target_range = 30.0
    tau0 = 2 * target_range / c
    out = np.zeros((n_rx, len(freqs)), dtype=np.complex64)
    # tau0 is a scalar delay -> steering vector is just exp(-j2pi f tau0), [F].
    # (np.outer would yield a spurious [F,1] and break the per-row assignment.)
    steer = np.exp(-2j * np.pi * np.asarray(freqs) * tau0)         # [F]
    for i in range(n_rx):
        clutter = ch.synthetic_multipath_cfr(freqs, n_taps=4, rng=rng).cpu().numpy()
        phase = np.exp(1j * 2 * np.pi * (i % N_RX_X) * 0.01)       # small spatial taper
        out[i] = (0.7 * steer * phase + 0.3 * clutter).astype(np.complex64)
    return torch.from_numpy(out).to(ch.device), "synthetic"


def main():
    rng = np.random.default_rng(7)
    scenario = munich_isac_scenario()
    freqs = scenario.frequency.linspace()
    carrier = scenario.frequency.carrier_hz

    print(isac.describe_split(scenario))
    split = isac.split_scenario(scenario)
    assert split["is_isac"], "expected an ISAC scenario"

    # ===== SENSING sub-problem =====
    s_pars, sense_src = _radar_s_pars(scenario, freqs, rng)
    print(f"[isac] sensing channel source: {sense_src}")
    ranges, ra_map = isac.range_angle_map(s_pars, freqs, N_RX_X, N_RX_Y,
                                          angle_bins=128, axis="az")
    # collapse over angle to a range profile and report the peak range
    rng_profile = ra_map.sum(axis=1)
    est_range = isac.peak_range(ranges, rng_profile)
    print(f"[isac] sensing: estimated target range = {est_range:.2f} m "
          f"(peak of range profile)")

    # ===== COMM sub-problem =====
    tx_node, rx_node = split["comm_links"][0]
    print(f"[isac] comm link: {tx_node.name} -> {rx_node.name}")
    bits_per_symbol = 4               # 16-QAM
    n_symbols = 32
    snr_db = 20.0
    modem = OFDMModem(fft_size=64, cp_len=16, n_active=52, pilot_spacing=8,
                      bits_per_symbol=bits_per_symbol)
    # narrow comm band (~15 MHz), see note in main_comms_link.py
    subcarrier_spacing = 240e3

    comm_cfr, comm_src = ch.load_or_synthesize_cfr("munich", freqs, frame=1, rng=rng)
    print(f"[isac] comm channel source: {comm_src}")
    H_sc = ch.cfr_to_subcarriers(comm_cfr, freqs, modem.fft_size, carrier, subcarrier_spacing)

    tx_bits = random_bits(n_symbols * modem.data_bits_per_symbol_block, seed=11)
    _, tx_freq = modem.modulate(tx_bits, n_symbols)
    rx_freq, _ = ch.apply_channel(tx_freq, H_sc, snr_db, rng_seed=123)

    rx_pilots = modem.extract_pilots(rx_freq)
    tx_pilots = modem.pilot_grid(n_symbols)
    H_est = ch.mmse_estimate(rx_pilots, tx_pilots, modem.pilot_idx, modem.fft_size, snr_db)
    eq = modem.extract_data(ch.mmse_equalize(rx_freq, H_est, snr_db))
    rx_bits = qam_demod(eq.reshape(-1), bits_per_symbol, modem.const)

    bit_err = ch.ber(tx_bits, rx_bits)
    # EVM against the TRUE transmitted data symbols (a decision-directed reference
    # biases EVM low once symbols cross decision boundaries; same convention as
    # BERBlock and main_isac_multilink).
    ref = modem.extract_data(tx_freq).reshape(-1)
    evm_pct = ch.evm(eq.reshape(-1), ref) * 100.0
    print(f"[isac] comm @ SNR={snr_db:.0f} dB: BER={bit_err:.3e}, EVM={evm_pct:.2f}%")

    # ===== plots =====
    plt.figure()
    # Zoom the range axis to where the energy actually lives so the map is not
    # mostly empty floor (the full grid can span ~250 m while returns sit in a
    # narrow band). Keep a sensible minimum window and a small margin.
    cum = np.cumsum(rng_profile) / (rng_profile.sum() + 1e-30)
    k99 = int(np.searchsorted(cum, 0.99)) + 1
    disp_max_m = max(float(ranges[min(k99, len(ranges) - 1)]) * 1.5, 2.0)
    kmax = int(np.searchsorted(ranges, disp_max_m)) or len(ranges)
    ra_z = ra_map[:kmax]
    # angle bins -> normalized sine-angle u = sin(theta), centered (fftshift order)
    n_a = ra_map.shape[1]
    u = (np.arange(n_a) - n_a // 2) / (n_a // 2)
    ra_db = 10 * np.log10(ra_z / (ra_z.max() + 1e-12) + 1e-12)
    plt.imshow(ra_db, aspect="auto", origin="lower",
               extent=[u[0], u[-1], ranges[0], ranges[kmax - 1]],
               cmap="viridis", vmin=-30, vmax=0)
    plt.colorbar(label="normalized power (dB)")
    plt.xlabel("azimuth  sin(θ)")
    plt.ylabel("range (m)")
    plt.title(f"ISAC sensing range/azimuth map ({sense_src})")
    ra_path = os.path.join(FIG_DIR, "isac_range_angle.png")
    plt.savefig(ra_path, dpi=120, bbox_inches="tight")
    plt.close()

    eq_np = eq.reshape(-1).cpu().numpy()
    plt.figure()
    plt.scatter(eq_np.real, eq_np.imag, s=5, alpha=0.4)
    plt.axis("equal"); plt.grid(True)
    plt.title(f"ISAC comm RX constellation (SNR={snr_db:.0f} dB, EVM={evm_pct:.1f}%)")
    const_path = os.path.join(FIG_DIR, "isac_constellation.png")
    plt.savefig(const_path, dpi=120, bbox_inches="tight")
    plt.close()

    print(f"[isac] wrote {ra_path}")
    print(f"[isac] wrote {const_path}")


if __name__ == "__main__":
    main()
