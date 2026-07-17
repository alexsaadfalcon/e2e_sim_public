"""Unit tests for the OFDM modem and QAM mapping (e2e.comms.ofdm)."""

import pytest

torch = pytest.importorskip("torch")

from e2e.comms import ofdm
from e2e.comms.ofdm import (
    OFDMModem,
    qam_constellation,
    qam_mod,
    qam_demod,
    random_bits,
)

# Derive the device from the module under test so the suite is green on CPU CI
# and on CUDA dev machines alike.
device = ofdm.device


@pytest.mark.parametrize("bits_per_symbol", [2, 4, 6])
def test_constellation_unit_average_power(bits_per_symbol):
    const = qam_constellation(bits_per_symbol)
    assert const.numel() == 2 ** bits_per_symbol
    assert const.device.type == device.type
    avg_pow = torch.mean(torch.abs(const) ** 2).item()
    assert avg_pow == pytest.approx(1.0, rel=1e-4)


@pytest.mark.parametrize("bits_per_symbol", [2, 4, 6])
def test_qam_map_demap_roundtrip_lossless(bits_per_symbol):
    """At the exact constellation points, map->demap recovers the bits exactly."""
    torch.manual_seed(0)
    n_syms = 200
    bits = random_bits(n_syms * bits_per_symbol, seed=1)
    syms, _ = qam_mod(bits, bits_per_symbol)
    rx_bits = qam_demod(syms, bits_per_symbol)
    assert rx_bits.numel() == bits.numel()
    assert torch.equal(rx_bits, bits)


@pytest.mark.parametrize("bits_per_symbol", [2, 4, 6])
def test_qam_demod_every_constellation_point_unique(bits_per_symbol):
    """Each constellation point demaps back to its own integer symbol index."""
    const = qam_constellation(bits_per_symbol)
    rx_bits = qam_demod(const, bits_per_symbol)
    # decode the bit groups back into symbol indices
    groups = rx_bits.reshape(-1, bits_per_symbol)
    weights = (1 << torch.arange(bits_per_symbol - 1, -1, -1, device=device))
    idx = (groups * weights).sum(dim=1)
    assert torch.equal(idx, torch.arange(2 ** bits_per_symbol, device=device))


def test_qam_mod_rejects_bad_bit_count():
    with pytest.raises(ValueError):
        qam_mod(torch.tensor([1, 0, 1], dtype=torch.int64), bits_per_symbol=2)


def test_qam_constellation_rejects_unsupported_order():
    with pytest.raises(ValueError):
        qam_constellation(3)


def test_modem_subcarrier_counts_respected():
    modem = OFDMModem(fft_size=64, cp_len=16, n_active=52, pilot_spacing=8,
                      bits_per_symbol=2)
    # active = pilots + data
    assert modem.n_active == 52
    assert modem.n_pilots + modem.n_data == modem.n_active
    # active subcarrier indices fall inside the fft grid
    assert int(modem.active_idx.min()) >= 0
    assert int(modem.active_idx.max()) < modem.fft_size


def test_modem_pilot_count_matches_spacing():
    """pilot_spacing=8 over 52 active tones -> ceil(52/8) = 7 pilots."""
    modem = OFDMModem(fft_size=64, cp_len=16, n_active=52, pilot_spacing=8,
                      bits_per_symbol=2)
    expected_pilots = (modem.n_active + modem.pilot_spacing - 1) // modem.pilot_spacing
    assert modem.n_pilots == expected_pilots
    assert modem.n_data == modem.n_active - expected_pilots
    # pilot grid has the expected shape
    pg = modem.pilot_grid(5)
    assert pg.shape == (5, modem.n_pilots)


def test_modem_pilots_disabled():
    modem = OFDMModem(fft_size=64, cp_len=16, n_active=52, pilot_spacing=0,
                      bits_per_symbol=2)
    assert modem.n_pilots == 0
    assert modem.n_data == modem.n_active


def test_modem_rejects_active_exceeding_fft():
    with pytest.raises(ValueError):
        OFDMModem(fft_size=32, n_active=64)


def test_modulate_output_shapes_and_cp_length():
    modem = OFDMModem(fft_size=64, cp_len=16, n_active=52, pilot_spacing=8,
                      bits_per_symbol=2)
    n_symbols = 4
    bits = random_bits(n_symbols * modem.data_bits_per_symbol_block, seed=0)
    tx_time, tx_freq = modem.modulate(bits, n_symbols)
    assert tx_freq.shape == (n_symbols, modem.fft_size)
    # cyclic prefix is prepended: time-domain length is fft_size + cp_len
    assert tx_time.shape == (n_symbols, modem.fft_size + modem.cp_len)


def test_cyclic_prefix_is_copy_of_tail():
    """The CP must equal the last cp_len samples of the IFFT body."""
    modem = OFDMModem(fft_size=64, cp_len=16, n_active=52, pilot_spacing=8,
                      bits_per_symbol=2)
    bits = random_bits(modem.data_bits_per_symbol_block, seed=3)
    tx_time, _ = modem.modulate(bits, 1)
    cp = tx_time[:, :modem.cp_len]
    tail = tx_time[:, -modem.cp_len:]
    assert torch.allclose(cp, tail, atol=1e-5)


def test_modulate_rejects_wrong_bit_count():
    modem = OFDMModem(fft_size=64, cp_len=16, n_active=52, pilot_spacing=8,
                      bits_per_symbol=2)
    with pytest.raises(ValueError):
        modem.modulate(random_bits(7, seed=0), 4)


@pytest.mark.parametrize("bits_per_symbol", [2, 4, 6])
def test_ofdm_modulate_demodulate_roundtrip_no_channel(bits_per_symbol):
    """With no channel, modulate->demodulate recovers symbols and bits exactly."""
    modem = OFDMModem(fft_size=64, cp_len=16, n_active=48, pilot_spacing=8,
                      bits_per_symbol=bits_per_symbol)
    n_symbols = 4
    bits = random_bits(n_symbols * modem.data_bits_per_symbol_block, seed=2)
    tx_time, tx_freq = modem.modulate(bits, n_symbols)

    rx_freq = modem.demodulate(tx_time)
    # frequency grid recovered to numerical precision
    assert torch.allclose(rx_freq, tx_freq, atol=1e-4)

    data_eq = modem.extract_data(rx_freq)
    rx_bits = qam_demod(data_eq.reshape(-1), bits_per_symbol, modem.const)
    assert torch.equal(rx_bits, bits)


def test_extract_pilots_returns_known_values():
    modem = OFDMModem(fft_size=64, cp_len=16, n_active=52, pilot_spacing=8,
                      bits_per_symbol=2)
    bits = random_bits(modem.data_bits_per_symbol_block, seed=5)
    _, tx_freq = modem.modulate(bits, 1)
    pilots = modem.extract_pilots(tx_freq)
    assert pilots.shape == (1, modem.n_pilots)
    assert torch.allclose(pilots, modem.pilot_value * torch.ones_like(pilots))
