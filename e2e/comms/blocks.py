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
from . import beamforming as bf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _carrier_hz(freq_plan, freqs):
    """Carrier frequency (Hz) from a v2 ``freq_plan`` -- the DICT shape that
    ``SionnaIterator.freq_plan`` / the env block expose (``{'carrier_hz': ...}``),
    matching every other freq_plan consumer in the codebase. Falls back to the band
    midpoint when no plan is threaded into the state dict."""
    if freq_plan is not None:
        return float(freq_plan["carrier_hz"])
    return float(np.mean(freqs))


def _cfr_from_state(state_dict, modem, freqs):
    """Pull a 1-D channel frequency response out of a pipeline state dict.

    The pipeline stores `s_pars` shaped [N_RX, N_TX, chirp, N_FREQS]. For a comm
    link we want a single spatial channel; `ch.frame_to_cfr` (element 0) extracts
    element (0,0,0), already resampled onto `freqs` (a no-op here since `s_pars`
    is already sampled on that grid), and we then resample onto the modem's
    subcarrier grid. If a precomputed `cfr`/`H_sc` is already in the state dict
    we use that directly.
    """
    if "H_sc" in state_dict:
        return torch.as_tensor(state_dict["H_sc"], dtype=torch.complex64, device=device)

    s_pars = state_dict["s_pars"]
    cfr_dense = ch.frame_to_cfr(s_pars, freqs, element=0)   # one spatial channel
    carrier = _carrier_hz(state_dict.get("freq_plan", None), freqs)
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    sc_spacing = state_dict.get("subcarrier_spacing_hz", df)
    return ch.cfr_to_subcarriers(cfr_dense, freqs, modem.fft_size, carrier, sc_spacing)


class ModemBlock:
    """Transmit a random OFDM frame through the state-dict channel and equalize it.

    Parameters mirror `OFDMModem`. On `apply` it returns the recovered bits,
    equalized data symbols, the (LS) channel estimate and the bits that were sent
    -- everything a downstream `BERBlock` needs.

    `combining` selects how the array's 1024 elements feed the comms head:
      - "element0" (default): the historical SISO tap -- element (0, 0) only,
        via `_cfr_from_state`. Bit-exact with the pre-`combining` `ModemBlock`.
      - "mrc": full-aperture maximum-ratio combining. Builds a per-element
        channel `H` (`beamforming.element_channels`) from `state['s_pars']`,
        injects INDEPENDENT per-element AWGN, and coherently combines
        (`beamforming.mrc_weights` + `combine`) before the existing pilot
        estimation / equalization / demap path runs -- unchanged -- on the
        combined stream (no genie equalization: the modem still estimates the
        *effective* channel from pilots).
      - "subspace": same per-element noise + combine machinery, but the
        (broadband) weight vector is the tracked subspace's dominant direction
        `state['U'][:, 0]` (`beamforming.subspace_weights`) instead of MRC.
        Requires a subspace tracker (`AdaOjaBlock`) in the pipeline so `state`
        carries `'U'`.
    """

    def __init__(self, freqs, n_symbols=8, fft_size=64, cp_len=16, n_active=52,
                 pilot_spacing=8, bits_per_symbol=2, snr_db=20.0,
                 equalizer="mmse", estimator="ls", seed=0, combining="element0"):
        self.freqs = np.asarray(freqs, dtype=np.float64)
        self.n_symbols = int(n_symbols)
        self.snr_db = float(snr_db)
        self.equalizer = equalizer
        self.estimator = estimator
        self.seed = seed
        if combining not in ("element0", "mrc", "subspace"):
            raise ValueError(
                f"unknown combining mode {combining!r} (expected 'element0', "
                "'mrc' or 'subspace')"
            )
        self.combining = combining
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
        # reuse the cached deterministic TX frame (see __init__)
        tx_bits = self._tx_bits
        tx_freq = self._tx_freq

        # channel acts per-subcarrier (frequency domain) + AWGN. Derive the noise
        # seed from the frame index so each frame sees an independent realization
        # (still reproducible from self.seed); the cached TX is unchanged.
        noise_seed = self.seed + 1 + self._frame
        self._frame += 1

        if self.combining == "element0":
            # historical SISO tap (element (0, 0, 0)); bit-exact with the
            # pre-`combining` ModemBlock.
            H_sc = _cfr_from_state(state_dict, modem, self.freqs)
            rx_freq, _ = ch.apply_channel(tx_freq, H_sc, self.snr_db, rng_seed=noise_seed)
            extra = {}
        else:
            # full-aperture spatial combining (mrc / subspace): see
            # `_combine_spatial`. `H_sc` here is the EFFECTIVE (post-combining)
            # channel `w^H H`, reported the same way element0 reports its
            # single-tap channel.
            rx_freq, H_sc, extra = self._combine_spatial(state_dict, tx_freq, noise_seed)

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

        out = {
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
        out.update(extra)   # mrc/subspace add 'comm_array_gain_db'; element0 adds nothing
        return out

    def _combine_spatial(self, state_dict, tx_freq, noise_seed):
        """Full-aperture receive beamforming for `combining in ('mrc', 'subspace')`.

        Builds the per-element channel `H` [N, fft_size] from `state['s_pars']`
        (`beamforming.element_channels`), injects INDEPENDENT complex AWGN per
        element -- noise variance set from `self.snr_db` relative to the MEAN
        per-element signal power on the active subcarriers (averaged over
        elements too: one scalar noise variance shared by every
        element/subcarrier/symbol) -- then coherently combines
        (`beamforming.combine`) before returning to the caller's unchanged pilot
        estimation / equalization / demap path.

        Returns `(rx_freq, H_eff, extra)`: `rx_freq` [n_symbols, fft_size] is the
        combined received frequency grid (drop-in replacement for the element0
        path's `rx_freq`); `H_eff` [fft_size] is the effective post-combining
        channel `w^H H`; `extra` carries `'comm_array_gain_db'`.
        """
        modem = self.modem
        if "s_pars" not in state_dict:
            raise ValueError(
                f"ModemBlock(combining={self.combining!r}) needs 's_pars' in the "
                "state dict to build per-element channels -- a precomputed "
                "scalar 'H_sc' (the element0 shortcut) is not enough for "
                "spatial combining"
            )
        s_pars = state_dict["s_pars"]

        carrier = _carrier_hz(state_dict.get("freq_plan", None), self.freqs)
        df = float(self.freqs[1] - self.freqs[0]) if len(self.freqs) > 1 else 1.0
        sc_spacing = state_dict.get("subcarrier_spacing_hz", df)
        k = np.arange(modem.fft_size)
        sc_freqs = carrier + (k - modem.fft_size / 2.0) * sc_spacing

        H = bf.element_channels(s_pars, self.freqs, sc_freqs)     # [N, fft_size]

        if self.combining == "mrc":
            w = bf.mrc_weights(H)                                 # [N, fft_size]
        else:   # "subspace"
            if "U" not in state_dict:
                raise ValueError(
                    "ModemBlock(combining='subspace') needs the subspace "
                    "tracker's 'U' in the state dict -- add a subspace tracker "
                    "(e.g. AdaOjaBlock as Simulation's subspace_block) to the "
                    "pipeline so it populates 'U'"
                )
            w = bf.subspace_weights(state_dict["U"])              # [N]

        active = modem.active_idx
        rx_clean = tx_freq[:, None, :] * H[None, :, :]            # [n_sym, N, fft]

        sig_pow = torch.mean(torch.abs(rx_clean[:, :, active]) ** 2).item()
        noise_pow = sig_pow / (10 ** (self.snr_db / 10.0)) if sig_pow > 0 else 1e-12

        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(noise_seed))
        shape = rx_clean.shape
        noise = (torch.randn(shape, generator=gen) + 1j * torch.randn(shape, generator=gen))
        noise = noise.to(device) * float(np.sqrt(noise_pow / 2.0))
        rx_full = rx_clean + noise                                # independent per element

        rx_freq = bf.combine(rx_full, w)                          # [n_sym, fft]
        H_eff = bf.combine(H, w)                                  # [fft]

        elem_pow = torch.mean(torch.abs(H[:, active]) ** 2).item()
        comb_pow = torch.mean(torch.abs(H_eff[active]) ** 2).item()
        array_gain_db = 10.0 * np.log10(comb_pow / elem_pow) if elem_pow > 0 else float("nan")

        return rx_freq, H_eff, {"comm_array_gain_db": array_gain_db}


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
