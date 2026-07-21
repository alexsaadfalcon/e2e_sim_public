"""
OFDM modem and QAM mapping for the communications layer.

Pure torch (complex64), matching the rest of the simulator's tensor conventions
(`device = "cuda" if available else "cpu"`, complex64). Nothing here imports
Sionna or touches the precomputed frames.

The modem produces frequency-domain symbols on a set of active subcarriers,
optionally inserts a comb of pilot tones, takes the IFFT to time domain, and
prepends a cyclic prefix. Demodulation inverts that. Channel application,
estimation and equalization live in `channel.py`.

Note: the shipped examples and `ModemBlock` apply the channel as a per-subcarrier
frequency-domain multiply (see `channel.py`) and never call `modulate`'s time-domain
output (`tx_time`) or `demodulate`. `tx_time`/`demodulate` -- and therefore the cyclic
prefix's actual ISI/multipath protection -- are exercised only by this module's own
round-trip tests, not by any end-to-end path in this repo.

Conventions
-----------
* A "frame" is `[n_symbols, fft_size]` of complex frequency-domain samples.
* In time domain a frame is `[n_symbols, fft_size + cp_len]`.
* Subcarrier indexing is plain (DC at index 0); we simply pick a contiguous
  block of `n_active` subcarriers centred on DC via fftshift-style ordering.
"""

import numpy as np
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------------
# QAM constellations
# --------------------------------------------------------------------------------
# Gray-coded square-QAM constellations. `bits_per_symbol` selects the order:
#   2 -> QPSK, 4 -> 16-QAM, 6 -> 64-QAM.
_SUPPORTED_ORDERS = {2: "QPSK", 4: "16-QAM", 6: "64-QAM"}


def _pam_gray_levels(bits_per_axis):
    """Return Gray-ordered PAM amplitude levels for one axis (I or Q)."""
    m = 1 << bits_per_axis                       # levels per axis
    # natural levels: +/-1, +/-3, ... centred on zero
    levels = np.arange(m) * 2 - (m - 1)          # e.g. m=4 -> [-3,-1,1,3]
    # Gray code maps bit pattern -> level index; build the ordered amplitude list
    gray = [i ^ (i >> 1) for i in range(m)]
    ordered = np.empty(m, dtype=np.float64)
    for idx, g in enumerate(gray):
        ordered[g] = levels[idx]
    return ordered


def qam_constellation(bits_per_symbol):
    """Unit-average-power complex QAM constellation as a 1-D torch tensor.

    The returned tensor has 2**bits_per_symbol points indexed by integer symbol.
    """
    if bits_per_symbol not in _SUPPORTED_ORDERS:
        raise ValueError(f"unsupported QAM order: {bits_per_symbol} bits/symbol "
                         f"(supported: {sorted(_SUPPORTED_ORDERS)})")
    bits_per_axis = bits_per_symbol // 2
    levels = _pam_gray_levels(bits_per_axis)     # Gray-ordered per axis
    m = len(levels)
    # symbol index s -> (i_bits, q_bits): high half -> I, low half -> Q
    const = np.zeros(m * m, dtype=np.complex128)
    for s in range(m * m):
        i_idx = s >> bits_per_axis
        q_idx = s & (m - 1)
        const[s] = levels[i_idx] + 1j * levels[q_idx]
    const /= np.sqrt(np.mean(np.abs(const) ** 2))    # normalise to unit avg power
    return torch.tensor(const, dtype=torch.complex64, device=device)


def qam_mod(bits, bits_per_symbol):
    """Map a 1-D bit tensor/array to QAM symbols (unit avg power).

    `bits` length must be a multiple of `bits_per_symbol`.
    Returns (symbols[complex64], constellation[complex64]).
    """
    bits = torch.as_tensor(bits, dtype=torch.int64, device=device).reshape(-1)
    if bits.numel() % bits_per_symbol != 0:
        raise ValueError("number of bits must be a multiple of bits_per_symbol")
    const = qam_constellation(bits_per_symbol)
    groups = bits.reshape(-1, bits_per_symbol)
    # pack each group of bits (MSB first) into an integer symbol index
    weights = (1 << torch.arange(bits_per_symbol - 1, -1, -1, device=device))
    sym_idx = (groups * weights).sum(dim=1)
    return const[sym_idx], const


def qam_demod(symbols, bits_per_symbol, const=None):
    """Hard-decision (nearest-point) demap of QAM symbols back to bits.

    Returns a flat int64 bit tensor.
    """
    symbols = torch.as_tensor(symbols, dtype=torch.complex64, device=device).reshape(-1)
    if const is None:
        const = qam_constellation(bits_per_symbol)
    # nearest constellation point per symbol
    d = torch.abs(symbols[:, None] - const[None, :])
    sym_idx = torch.argmin(d, dim=1)
    # unpack symbol index -> bits (MSB first)
    shifts = torch.arange(bits_per_symbol - 1, -1, -1, device=device)
    bits = (sym_idx[:, None] >> shifts) & 1
    return bits.reshape(-1).to(torch.int64)


# --------------------------------------------------------------------------------
# OFDM modem
# --------------------------------------------------------------------------------
class OFDMModem:
    """A minimal OFDM modem with cyclic prefix and comb pilots.

    Parameters
    ----------
    fft_size : total number of subcarriers (IFFT length).
    cp_len   : cyclic-prefix length in samples.
    n_active : number of data+pilot subcarriers actually used (<= fft_size).
               They are placed as a contiguous block centred on DC.
    pilot_spacing : insert one pilot every `pilot_spacing` active subcarriers
                    (set to 0 to disable pilots).
    bits_per_symbol : QAM order for the data subcarriers.
    pilot_value : known complex value transmitted on every pilot subcarrier.
    """

    def __init__(self, fft_size=64, cp_len=16, n_active=None, pilot_spacing=8,
                 bits_per_symbol=2, pilot_value=1.0 + 0.0j):
        self.fft_size = int(fft_size)
        self.cp_len = int(cp_len)
        self.n_active = int(n_active) if n_active is not None else self.fft_size
        if self.n_active > self.fft_size:
            raise ValueError("n_active cannot exceed fft_size")
        self.pilot_spacing = int(pilot_spacing)
        self.bits_per_symbol = int(bits_per_symbol)
        self.pilot_value = torch.tensor(pilot_value, dtype=torch.complex64, device=device)

        # active subcarrier indices: a contiguous block centred on DC.
        # Using fftshift ordering keeps the used band in the middle of the spectrum.
        start = (self.fft_size - self.n_active) // 2
        self.active_idx = torch.arange(start, start + self.n_active, device=device)

        # split active subcarriers into pilots vs data
        if self.pilot_spacing > 0:
            mask = (torch.arange(self.n_active, device=device) % self.pilot_spacing) == 0
        else:
            mask = torch.zeros(self.n_active, dtype=torch.bool, device=device)
        self.pilot_local = torch.nonzero(mask, as_tuple=False).reshape(-1)   # within active band
        self.data_local = torch.nonzero(~mask, as_tuple=False).reshape(-1)
        self.pilot_idx = self.active_idx[self.pilot_local]                    # within fft grid
        self.data_idx = self.active_idx[self.data_local]
        self.n_pilots = self.pilot_idx.numel()
        self.n_data = self.data_idx.numel()
        self.const = qam_constellation(self.bits_per_symbol)

    # -- info --------------------------------------------------------------
    @property
    def data_bits_per_symbol_block(self):
        """Number of bits carried by one OFDM symbol (across data subcarriers)."""
        return self.n_data * self.bits_per_symbol

    def pilot_grid(self, n_symbols):
        """Known frequency-domain pilot values, shape [n_symbols, n_pilots]."""
        return self.pilot_value * torch.ones(n_symbols, self.n_pilots,
                                              dtype=torch.complex64, device=device)

    # -- modulation --------------------------------------------------------
    def modulate(self, bits, n_symbols):
        """Build `n_symbols` OFDM symbols from a flat bit stream.

        Returns
        -------
        tx_time : [n_symbols, fft_size + cp_len] complex time-domain frame.
        tx_freq : [n_symbols, fft_size] frequency-domain grid (pre-IFFT), for reference.
        """
        bits = torch.as_tensor(bits, dtype=torch.int64, device=device).reshape(-1)
        need = n_symbols * self.data_bits_per_symbol_block
        if bits.numel() != need:
            raise ValueError(f"expected {need} bits for {n_symbols} symbols, got {bits.numel()}")

        data_syms, _ = qam_mod(bits, self.bits_per_symbol)
        data_syms = data_syms.reshape(n_symbols, self.n_data)

        # assemble the frequency grid
        tx_freq = torch.zeros(n_symbols, self.fft_size, dtype=torch.complex64, device=device)
        tx_freq[:, self.data_idx] = data_syms
        if self.n_pilots:
            tx_freq[:, self.pilot_idx] = self.pilot_grid(n_symbols)

        # IFFT (use ifftshift so the centred band maps to the right tones), add CP
        grid = torch.fft.ifftshift(tx_freq, dim=-1)
        tx_time = torch.fft.ifft(grid, dim=-1)
        cp = tx_time[:, -self.cp_len:] if self.cp_len > 0 else tx_time[:, :0]
        tx_time = torch.cat([cp, tx_time], dim=-1)
        return tx_time, tx_freq

    def demodulate(self, rx_time):
        """Inverse of `modulate`: strip CP, FFT, return the frequency grid.

        rx_time : [n_symbols, fft_size + cp_len] complex.
        Returns rx_freq : [n_symbols, fft_size].
        """
        rx_time = torch.as_tensor(rx_time, dtype=torch.complex64, device=device)
        if self.cp_len > 0:
            rx_time = rx_time[:, self.cp_len:]
        grid = torch.fft.fft(rx_time, dim=-1)
        rx_freq = torch.fft.fftshift(grid, dim=-1)
        return rx_freq

    def extract_data(self, freq_grid):
        """Return data-subcarrier symbols, shape [n_symbols, n_data]."""
        return freq_grid[:, self.data_idx]

    def extract_pilots(self, freq_grid):
        """Return pilot-subcarrier symbols, shape [n_symbols, n_pilots]."""
        return freq_grid[:, self.pilot_idx]


def random_bits(n, seed=None):
    """Convenience: a flat int64 tensor of `n` random bits on the active device."""
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(int(seed))
    return torch.randint(0, 2, (n,), generator=g).to(device=device, dtype=torch.int64)
