"""Unit tests for ISAC scenario splitting and sensing estimators
(e2e.comms.isac)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from e2e.comms import isac
from e2e.comms.channel import synthetic_multipath_cfr
from e2e.scenario import munich_isac_scenario, NodeRole

device = isac.device


def _freqs(n=64, start=28.5e9, stop=31.5e9):
    return np.linspace(start, stop, n)


# --------------------------------------------------------------------------- split_scenario

def test_split_scenario_yields_radar_and_comm_link():
    sc = munich_isac_scenario()
    split = isac.split_scenario(sc)

    assert split["is_isac"] is True

    # exactly one radar node
    assert len(split["radar_nodes"]) == 1
    assert split["radar_nodes"][0].role == NodeRole.RADAR

    # exactly one comm tx->rx link
    assert len(split["comm_links"]) == 1
    tx, rx = split["comm_links"][0]
    assert tx.role == NodeRole.COMM_TX
    assert rx.role == NodeRole.COMM_RX


def test_describe_split_mentions_both_roles():
    sc = munich_isac_scenario()
    text = isac.describe_split(sc)
    assert "sensing" in text
    assert "comm" in text


# --------------------------------------------------------------------------- range_profile

def test_range_profile_shape_and_finite():
    freqs = _freqs(64)
    cfr = synthetic_multipath_cfr(freqs, n_taps=4, rng=np.random.default_rng(0))
    ranges, power = isac.range_profile(cfr, freqs)
    assert ranges.shape == (64,)
    assert power.shape == (64,)
    assert np.all(np.isfinite(ranges))
    assert np.all(np.isfinite(power))
    assert np.all(power >= 0.0)


def test_range_profile_zero_padding_changes_bin_count():
    freqs = _freqs(64)
    cfr = synthetic_multipath_cfr(freqs, rng=np.random.default_rng(1))
    ranges, power = isac.range_profile(cfr, freqs, n_bins=128)
    assert ranges.shape == (128,)
    assert power.shape == (128,)


def test_peak_range_is_sensible_scalar():
    """A single dominant tap at a known delay should peak near that range."""
    freqs = _freqs(128)
    c = 299_792_458.0
    tau = 10e-9  # 10 ns one-way delay
    H = np.exp(-2j * np.pi * np.outer(freqs, [tau])).reshape(-1).astype(np.complex64)
    ranges, power = isac.range_profile(torch.from_numpy(H), freqs)
    pk = isac.peak_range(ranges, power)
    assert isinstance(pk, float)
    assert np.isfinite(pk)
    expected = c * tau / 2.0
    # within a couple of range bins of the true delay
    bin_size = ranges[1] - ranges[0]
    assert abs(pk - expected) <= 2 * bin_size


# --------------------------------------------------------------------------- range_angle_map

def test_range_angle_map_shape_and_finite():
    n_rx_x, n_rx_y, n_f = 4, 4, 32
    freqs = _freqs(n_f)
    torch.manual_seed(0)
    s_pars = torch.randn(n_rx_x * n_rx_y, n_f, dtype=torch.complex64, device=device)
    angle_bins, range_bins = 64, 48
    ranges, power = isac.range_angle_map(
        s_pars, freqs, n_rx_x=n_rx_x, n_rx_y=n_rx_y,
        angle_bins=angle_bins, range_bins=range_bins, axis="az")
    assert power.shape == (range_bins, angle_bins)
    assert ranges.shape == (range_bins,)
    assert np.all(np.isfinite(power))
    assert np.all(np.isfinite(ranges))


def test_range_angle_map_el_axis_runs():
    n_rx_x, n_rx_y, n_f = 4, 4, 16
    freqs = _freqs(n_f)
    s_pars = torch.randn(n_rx_x * n_rx_y, n_f, dtype=torch.complex64, device=device)
    ranges, power = isac.range_angle_map(
        s_pars, freqs, n_rx_x=n_rx_x, n_rx_y=n_rx_y,
        angle_bins=32, axis="el")
    assert power.shape == (n_f, 32)
    assert np.all(np.isfinite(power))


def test_range_angle_map_rejects_mismatched_aperture():
    freqs = _freqs(16)
    s_pars = torch.randn(15, 16, dtype=torch.complex64, device=device)
    with pytest.raises(AssertionError):
        isac.range_angle_map(s_pars, freqs, n_rx_x=4, n_rx_y=4)


# --------------------------------------------------------------------------- main_isac._radar_s_pars

class _FakeIter:
    """Stand-in for SionnaMunichIterator serving one known frame."""

    def __init__(self, frame):
        self._frame = frame

    def __len__(self):
        return 1

    def __getitem__(self, i):
        return self._frame


def test_radar_s_pars_rejects_wrong_antenna_count(monkeypatch):
    """A loaded frame with != N_RX_X*N_RX_Y elements must raise a clear error,
    never feed uninitialised rows into the range/angle map (finding #4)."""
    from e2e.main import main_isac
    from e2e.environment import sionna_iterator as si

    n_f = 16
    freqs = _freqs(n_f)
    # frame shaped [N_RX, TX, chirp, F] with the WRONG element count (8, not 1024)
    bad = np.ones((8, 1, 1, n_f), dtype=np.complex64)
    monkeypatch.setattr(si, "SionnaMunichIterator", lambda *a, **k: _FakeIter(bad))

    with pytest.raises(ValueError, match="array elements"):
        main_isac._radar_s_pars(None, freqs, np.random.default_rng(0))


def test_radar_s_pars_fills_all_rows_when_frame_matches(monkeypatch):
    """A correctly sized frame fills every aperture row (no garbage/zeros gaps)."""
    from e2e.main import main_isac
    from e2e.environment import sionna_iterator as si

    n_rx = main_isac.N_RX_X * main_isac.N_RX_Y
    n_f = 8
    freqs = _freqs(n_f)
    # distinct, all-nonzero per-element responses so a missed row would be detectable
    base = (np.arange(1, n_rx + 1)[:, None] *
            np.ones((1, n_f))).astype(np.complex64)
    good = base.reshape(n_rx, 1, 1, n_f)
    monkeypatch.setattr(si, "SionnaMunichIterator", lambda *a, **k: _FakeIter(good))

    s_pars, src = main_isac._radar_s_pars(None, freqs, np.random.default_rng(0))
    assert src == "sionna:munich"
    arr = s_pars.cpu().numpy()
    assert arr.shape == (n_rx, n_f)
    assert np.all(np.isfinite(arr))
    # every row is non-zero (would be all-zero if left uninitialised by np.zeros)
    assert np.all(np.any(arr != 0, axis=1))


def test_radar_s_pars_synthetic_fallback_fills_all_rows(monkeypatch):
    """No .pkl -> synthetic fallback still returns a fully-populated aperture."""
    from e2e.main import main_isac
    from e2e.environment import sionna_iterator as si

    def _boom(*a, **k):
        raise FileNotFoundError("no munich.pkl")

    monkeypatch.setattr(si, "SionnaMunichIterator", _boom)

    n_rx = main_isac.N_RX_X * main_isac.N_RX_Y
    n_f = 8
    freqs = _freqs(n_f)
    s_pars, src = main_isac._radar_s_pars(None, freqs, np.random.default_rng(1))
    assert src == "synthetic"
    arr = s_pars.cpu().numpy()
    assert arr.shape == (n_rx, n_f)
    assert np.all(np.isfinite(arr))
    assert np.all(np.any(arr != 0, axis=1))
