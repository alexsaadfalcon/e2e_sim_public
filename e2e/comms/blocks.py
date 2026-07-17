"""
Optional communications pipeline blocks.

These follow the same convention as the downstream blocks in the top-level
`e2e/blocks.py`: each exposes ``apply(state_dict) -> dict`` and is side-effect-free
on import, so they can later slot into `Simulation.downstream_blocks`.

They consume the simulator's `s_pars` (channel frequency response) from the state
dict and produce comm products (received constellation, recovered bits, BER/EVM).
The channel response is resampled onto the modem's OFDM subcarriers, used to
transmit a random OFDM frame, and then equalized + demapped.

Importing this module has no side effects beyond defining the classes.
"""

import numpy as np
import torch

from .ofdm import OFDMModem, random_bits
from . import channel as ch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cfr_from_state(state_dict, modem, freqs):
    """Pull a 1-D channel frequency response out of a pipeline state dict.

    The pipeline stores `s_pars` shaped [N_RX, N_TX, chirp, N_FREQS]. For a comm
    link we want a single spatial channel; we take element (0,0,0) and resample it
    onto the modem's subcarrier grid. If a precomputed `cfr`/`H_sc` is already in
    the state dict we use that directly.
    """
    if "H_sc" in state_dict:
        return torch.as_tensor(state_dict["H_sc"], dtype=torch.complex64, device=device)

    s_pars = state_dict["s_pars"]
    s_pars = torch.as_tensor(s_pars, dtype=torch.complex64, device=device)
    # collapse to a single 1-D frequency response
    flat = s_pars.reshape(-1, s_pars.shape[-1])
    cfr_dense = flat[0]                                  # one spatial channel
    fp = state_dict.get("freq_plan", None)
    carrier = fp.carrier_hz if fp is not None else float(np.mean(freqs))
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    sc_spacing = state_dict.get("subcarrier_spacing_hz", df)
    return ch.cfr_to_subcarriers(cfr_dense, freqs, modem.fft_size, carrier, sc_spacing)


class ModemBlock:
    """Transmit a random OFDM frame through the state-dict channel and equalize it.

    Parameters mirror `OFDMModem`. On `apply` it returns the recovered bits,
    equalized data symbols, the (LS) channel estimate and the bits that were sent
    -- everything a downstream `BERBlock` needs.
    """

    def __init__(self, freqs, n_symbols=8, fft_size=64, cp_len=16, n_active=52,
                 pilot_spacing=8, bits_per_symbol=2, snr_db=20.0,
                 equalizer="mmse", estimator="ls", seed=0):
        self.freqs = np.asarray(freqs, dtype=np.float64)
        self.n_symbols = int(n_symbols)
        self.snr_db = float(snr_db)
        self.equalizer = equalizer
        self.estimator = estimator
        self.seed = seed
        self.modem = OFDMModem(fft_size=fft_size, cp_len=cp_len, n_active=n_active,
                               pilot_spacing=pilot_spacing, bits_per_symbol=bits_per_symbol)

        # The transmitted frame is fully deterministic (fixed seed + fixed modem
        # config), so it is identical on every `apply`. Compute it once here and
        # reuse it -- bits, modulated frame and the known pilot grid -- to avoid
        # regenerating it per frame. Results are unchanged.
        n_bits = self.n_symbols * self.modem.data_bits_per_symbol_block
        self._tx_bits = random_bits(n_bits, seed=self.seed)
        _, self._tx_freq = self.modem.modulate(self._tx_bits, self.n_symbols)
        self._tx_pilots = self.modem.pilot_grid(self.n_symbols)
        # The actually-transmitted data symbols (constellation points), used as the
        # true EVM reference downstream instead of a decision-directed one.
        self._tx_data = self.modem.extract_data(self._tx_freq)
        # Per-frame counter so the AWGN realization differs each apply() (otherwise
        # multi-frame BER/EVM averaging draws identical noise every frame).
        self._frame = 0

    def reset(self):
        """Rewind the per-frame noise counter (Simulation.reset calls this), so
        repeated runs of the same Simulation reproduce identical realizations."""
        self._frame = 0

    def apply(self, state_dict):
        modem = self.modem
        H_sc = _cfr_from_state(state_dict, modem, self.freqs)

        # reuse the cached deterministic TX frame (see __init__)
        tx_bits = self._tx_bits
        tx_freq = self._tx_freq

        # channel acts per-subcarrier (frequency domain) + AWGN. Derive the noise
        # seed from the frame index so each frame sees an independent realization
        # (still reproducible from self.seed); the cached TX is unchanged.
        noise_seed = self.seed + 1 + self._frame
        self._frame += 1
        rx_freq_clean, _ = ch.apply_channel(tx_freq, H_sc, self.snr_db, rng_seed=noise_seed)
        # demod is identity here since we already work in the freq grid, but keep it
        # explicit so the block remains correct if a time-domain channel is swapped in
        rx_freq = rx_freq_clean

        # pilot-based channel estimation
        rx_pilots = modem.extract_pilots(rx_freq)
        tx_pilots = self._tx_pilots               # cached known pilot grid
        if self.estimator == "mmse":
            H_est = ch.mmse_estimate(rx_pilots, tx_pilots, modem.pilot_idx,
                                     modem.fft_size, self.snr_db)
        else:
            H_est = ch.ls_estimate(rx_pilots, tx_pilots, modem.pilot_idx, modem.fft_size)

        # equalize + demap
        if self.equalizer == "zf":
            eq = ch.zf_equalize(rx_freq, H_est)
        else:
            eq = ch.mmse_equalize(rx_freq, H_est, self.snr_db)
        data_eq = modem.extract_data(eq)
        rx_bits = _demap(data_eq, modem)

        return {
            "comm_tx_bits": tx_bits,
            "comm_rx_bits": rx_bits,
            "comm_data_eq": data_eq,
            # true transmitted data symbols -> exact EVM reference for BERBlock
            "comm_tx_data": self._tx_data,
            "comm_H_est": H_est,
            "comm_H_true": H_sc,
            # let a downstream BERBlock compute EVM against the ideal constellation
            "comm_bits_per_symbol": modem.bits_per_symbol,
            # share the prebuilt constellation so BERBlock need not rebuild it
            "comm_const": modem.const,
        }


def _demap(data_eq, modem):
    from .ofdm import qam_demod
    return qam_demod(data_eq.reshape(-1), modem.bits_per_symbol, modem.const)


class BERBlock:
    """Compute BER / EVM from a ModemBlock's outputs.

    Reads `comm_tx_bits`, `comm_rx_bits`, `comm_data_eq` from the state dict
    (which a `ModemBlock` will have populated) and returns the metrics.
    """

    def apply(self, state_dict):
        tx_bits = state_dict["comm_tx_bits"]
        rx_bits = state_dict["comm_rx_bits"]
        out = {"ber": ch.ber(tx_bits, rx_bits)}
        if "comm_data_eq" in state_dict:
            modem_bps = state_dict.get("comm_bits_per_symbol", None)
            data_eq = state_dict["comm_data_eq"].reshape(-1)
            if "comm_tx_data" in state_dict:
                # Exact EVM: reference is the actually-transmitted symbols. This is
                # correct even at high BER, unlike a decision-directed reference
                # (nearest constellation point) which biases EVM low when symbols
                # cross decision boundaries.
                ref = state_dict["comm_tx_data"].reshape(-1)
                out["evm"] = ch.evm(data_eq, ref)
            elif modem_bps is not None:
                # Fallback (no true reference available): decision-directed EVM.
                # Reuse the prebuilt constellation if a ModemBlock shared it rather
                # than rebuilding it (O(M^2)); indexing the nearest point equals the
                # old qam_demod->qam_mod round-trip.
                from .ofdm import qam_constellation
                const = state_dict.get("comm_const", None)
                if const is None:
                    const = qam_constellation(modem_bps)
                d = torch.abs(data_eq[:, None] - const[None, :])
                ref = const[torch.argmin(d, dim=1)]
                out["evm"] = ch.evm(data_eq, ref)
        return out
