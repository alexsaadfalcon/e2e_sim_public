"""Unit tests for individual pipeline blocks (e2e.blocks)."""

import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from e2e.blocks import (
    InterconnectBlock,
    AFEBlock,
    AdaOjaBlock,
    FFTBlock,
    RangeAzBlock,
    RangeElBlock,
    SubspaceErrorBlock,
)
from e2e.subspace.algorithms import rand_orth_complex
from e2e.blocks import device


@pytest.fixture
def state_dict(n_freqs):
    """A minimal state_dict shaped like Simulation builds, on the library device."""
    s_pars = torch.randn(32, 32, 1, n_freqs, dtype=torch.cfloat, device=device)
    U_true = rand_orth_complex(1024, 16)
    return {
        "s_pars": s_pars,
        "U_true": U_true,
        "U": U_true.clone(),
        "PRX": None,
    }


@pytest.mark.parametrize("block_cls,key", [
    (FFTBlock, "fft"),
    (RangeAzBlock, "range_az"),
    (RangeElBlock, "range_el"),
])
def test_transform_blocks_emit_expected_key(block_cls, key, state_dict):
    out = block_cls(bins=32).apply(state_dict)
    assert key in out
    assert torch.is_tensor(out[key])
    assert torch.all(torch.isfinite(torch.abs(out[key])))
    # Must be a full 2D [bins, bins] map. Regression guard: RangeAz/RangeEl
    # previously passed torch.fft.fft(x, 1) (n=1) which collapsed the range axis
    # to a single bin -> [bins, 1], silently destroying the range dimension.
    assert out[key].shape == (32, 32)
    # Now a real, non-negative power map (coherent FFT + non-coherent power
    # integration over the collapsed axis), not a raw complex amplitude.
    assert out[key].dtype == torch.float32
    assert torch.all(out[key] >= 0)


# --------------------------------------- DEFECT regression: off-broadside / off-range targets

_C = 2.99792458e8  # speed of light (m/s)


def _steering_frame(n_az, n_el, n_freqs, az_deg, el_deg, tau, dev):
    """A synthetic post-GridStage frame [n_az, n_el, 1, n_freqs] (matching
    frames.to_aperture_grid's convention: dim0=az/columns, dim1=el/rows) holding a
    single half-wavelength-spaced planar-array point target: element (m, n)'s
    response is exp(i*pi*m*sin(az)) * exp(i*pi*n*sin(el)) * exp(-i*2*pi*f_k*tau)
    over the stepped-frequency grid, where tau is the round-trip delay."""
    freqs = np.linspace(28.5e9, 31.5e9, n_freqs)
    m = np.arange(n_az).reshape(n_az, 1, 1)
    n = np.arange(n_el).reshape(1, n_el, 1)
    f = freqs.reshape(1, 1, n_freqs)
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    steer = (np.exp(1j * np.pi * m * np.sin(az))
             * np.exp(1j * np.pi * n * np.sin(el))
             * np.exp(-1j * 2 * np.pi * f * tau)).astype(np.complex64)
    s_pars = torch.from_numpy(steer).to(dev).view(n_az, n_el, 1, n_freqs)
    return {"s_pars": s_pars}


def test_range_az_noncoherent_elevation_shows_offbroadside_target():
    """Regression for DEFECT 1: RangeAzBlock used to coherently sum over elevation
    (an implicit un-steered broadside beam), completely nulling a target off
    broadside in ELEVATION (verified pre-fix: peak 65515 at 0 deg -> 0.00 at 3.58
    deg). Here the target sits at az=0 (broadside) but ~10.8 deg off broadside in
    elevation (bin offset k=3 of n=32, chosen to land exactly on an FFT bin); the
    fix integrates elevation non-coherently (power), so the az/range peak survives
    regardless of the elevation angle."""
    n = 32
    k = 3
    el_deg = np.degrees(np.arcsin(2 * k / n))
    state = _steering_frame(n, n, n, az_deg=0.0, el_deg=el_deg, tau=0.0, dev=device)
    out = RangeAzBlock(bins=n).apply(state)
    power = out["range_az"]
    peak = torch.max(power).item()
    az_bin, range_bin = divmod(torch.argmax(power).item(), power.shape[1])
    # By Parseval, non-coherently summing power over the n=32 elevation ELEMENTS
    # equals summing power over elevation's own FFT bins (up to a constant), and at
    # the true (az, range) bin every elevation element contributes the same
    # magnitude (n_az * n_freqs) regardless of the elevation angle -- this is
    # exactly what makes the peak survive off-broadside elevation targets.
    expected_peak = n * (n * n) ** 2
    assert peak == pytest.approx(expected_peak, rel=1e-3)
    assert (az_bin, range_bin) == (n // 2, n // 2)  # broadside az, zero-delay range


def test_fft_block_noncoherent_range_shows_target_at_10m():
    """Regression for DEFECT 2: FFTBlock used to coherently sum raw frequency
    samples before the aperture FFT -- only coherent for a target at range 0
    (verified pre-fix: 63 dB collapse by 5 cm range). Target here is at a ~10 m
    round-trip range (tau = 2R/c) and ~10.8 deg off broadside in both az and el
    (bin offset k=3 of n=32); the fix range-transforms first and integrates range
    non-coherently, so the az/el peak survives regardless of range."""
    n = 32
    k = 3
    off_deg = np.degrees(np.arcsin(2 * k / n))
    tau = 2 * 10.0 / _C  # round-trip delay for a 10 m range
    state = _steering_frame(n, n, n, az_deg=off_deg, el_deg=off_deg, tau=tau, dev=device)
    out = FFTBlock(bins=n).apply(state)
    power = out["fft"]
    peak = torch.max(power).item()
    az_bin, el_bin = divmod(torch.argmax(power).item(), power.shape[1])
    # By Parseval, non-coherent integration over the n=32 range bins gives exactly
    # (n_az * n_el)**2 * n_freqs**2 at the true (az, el) bin -- independent of tau
    # (i.e. independent of range), which is exactly what makes the peak survive a
    # nonzero-range target.
    expected_peak = (n * n) ** 2 * n ** 2
    assert peak == pytest.approx(expected_peak, rel=1e-3)
    assert (az_bin, el_bin) == (n // 2 + k, n // 2 + k)


# ------------------------------------- full-band range compression (decimate, not truncate)


def test_fft_block_integrates_full_band_when_bins_lt_nfreqs():
    """The range transform must span the FULL frequency band, not be truncated to the
    first `bins` samples. With bins < n_freqs, a matched target's range-integrated
    az/el peak is (n_az*n_el)**2 * n_freqs**2 -- all n_freqs samples coherently
    compressed then power-integrated, independent of range (tau). The old code passed
    `bins` as the range-FFT length, trimming the freq axis to its first `bins` samples
    and giving (n_az*n_el)**2 * bins**2, a factor (n_freqs/bins)**2 too small."""
    n_ap, n_freqs, bins = 8, 64, 8
    k = 2
    off = np.degrees(np.arcsin(2 * k / n_ap))
    tau = 2 * 7.5 / _C  # nonzero range; peak must be tau-independent
    state = _steering_frame(n_ap, n_ap, n_freqs, az_deg=off, el_deg=off, tau=tau, dev=device)
    out = FFTBlock(bins=bins).apply(state)
    power = out["fft"]
    assert power.shape == (bins, bins)
    peak = torch.max(power).item()
    expected_full = (n_ap * n_ap) ** 2 * n_freqs ** 2
    assert peak == pytest.approx(expected_full, rel=1e-3)
    # decisively above the freq-truncated value (would be bins**2, not n_freqs**2)
    assert peak > 10 * ((n_ap * n_ap) ** 2 * bins ** 2)
    az_bin, el_bin = divmod(torch.argmax(power).item(), power.shape[1])
    assert (az_bin, el_bin) == (bins // 2 + k, bins // 2 + k)


def test_range_az_full_band_compression_when_bins_lt_nfreqs():
    """RangeAz compresses over the full band then power-bins to `bins` range gates.
    With bins < n_freqs and a zero-delay target, the peak gate holds
    n_el*(n_az*n_freqs)**2 (full-band coherent range gain), a factor (n_freqs/bins)**2
    above the old freq-truncated value."""
    n_ap, n_freqs, bins = 8, 64, 8
    state = _steering_frame(n_ap, n_ap, n_freqs, az_deg=0.0, el_deg=0.0, tau=0.0, dev=device)
    out = RangeAzBlock(bins=bins).apply(state)
    power = out["range_az"]
    assert power.shape == (bins, bins)
    peak = torch.max(power).item()
    expected_full = n_ap * (n_ap * n_freqs) ** 2
    assert peak == pytest.approx(expected_full, rel=1e-3)
    assert peak > 10 * (n_ap * (n_ap * bins) ** 2)
    az_bin, range_bin = divmod(torch.argmax(power).item(), power.shape[1])
    assert (az_bin, range_bin) == (bins // 2, bins // 2)  # broadside az, zero-delay range


def test_aperture_window_reduces_sidelobes():
    """A Hamming aperture taper lowers the peak sidelobe of the az/el response
    relative to the uniform (rectangular) aperture (classic ~-13 dB -> ~-40 dB
    trade for a wider main lobe). Uses a small real aperture zero-padded by the
    aperture FFT (bins > n_ap) so the beampattern is finely sampled."""
    n_ap, n_freqs, bins = 16, 16, 64
    state = _steering_frame(n_ap, n_ap, n_freqs, az_deg=0.0, el_deg=0.0, tau=0.0, dev=device)
    uni = FFTBlock(bins=bins, window=None).apply(state)["fft"]
    ham = FFTBlock(bins=bins, window="hamming").apply(state)["fft"]

    def peak_sidelobe(m):
        row = m[:, bins // 2]                     # azimuth cut at elevation broadside
        peak_idx = int(torch.argmax(row).item())
        peak = row[peak_idx]
        mask = torch.ones_like(row, dtype=torch.bool)
        # Exclude a main-lobe region wide enough to clear the tapered window's
        # broader main lobe (rect first null ~ bins/n_ap = 4; Hamming ~2x wider).
        lo, hi = max(0, peak_idx - 8), min(len(row), peak_idx + 9)
        mask[lo:hi] = False
        return (torch.max(row[mask]) / peak).item()

    assert peak_sidelobe(ham) < peak_sidelobe(uni)


def test_aperture_window_invalid_raises():
    state = _steering_frame(8, 8, 16, az_deg=0.0, el_deg=0.0, tau=0.0, dev=device)
    with pytest.raises(ValueError):
        FFTBlock(bins=8, window="blackman").apply(state)


def test_range_az_nondivisible_nfreqs_zero_gate_and_energy():
    """When n_freqs is NOT a multiple of bins (the production case, n_freqs~5000,
    bins=256), power-binning groups per = ceil(n_freqs/bins) native bins per gate, so
    a zero-delay target's peak lands in gate (n_freqs//2)//per -- NOT bins//2 -- while
    still carrying the full coherent range energy. Guards the _power_bin / _range_axis
    alignment that the exact-multiple tests miss."""
    import math
    n_ap, n_freqs, bins = 4, 100, 8   # 100 % 8 != 0
    state = _steering_frame(n_ap, n_ap, n_freqs, az_deg=0.0, el_deg=0.0, tau=0.0, dev=device)
    out = RangeAzBlock(bins=bins).apply(state)
    power = out["range_az"]
    assert power.shape == (bins, bins)
    per = math.ceil(n_freqs / bins)
    zero_gate = (n_freqs // 2) // per
    assert zero_gate != bins // 2         # the whole point: non-divisible shifts the zero gate
    az_bin, range_bin = divmod(torch.argmax(power).item(), power.shape[1])
    assert (az_bin, range_bin) == (bins // 2, zero_gate)
    peak = torch.max(power).item()
    assert peak == pytest.approx(n_ap * (n_ap * n_freqs) ** 2, rel=1e-3)


def test_subspace_error_zero_for_identical_basis(state_dict):
    out = SubspaceErrorBlock().apply(state_dict)
    # float32 QR leaves a small residual in subspace_dist_frob; 1e-3 is too tight
    # (the value is ~1e-3..1e-2 and varies by RNG/device). Match test_subspace.py's 1e-2.
    assert out["subspace_err"].item() == pytest.approx(0.0, abs=1e-2)


def test_subspace_error_positive_for_perturbed_basis(state_dict):
    state_dict["U"] = rand_orth_complex(1024, 16)
    out = SubspaceErrorBlock().apply(state_dict)
    assert out["subspace_err"].item() > 0.0


def test_interconnect_case3_is_identity():
    frame = torch.randn(32, 32, 1, 64, dtype=torch.cfloat, device=device)
    out = InterconnectBlock(case="case3").apply_interconnect(frame)
    assert torch.allclose(out, frame)


def test_interconnect_default_filters_frame():
    frame = torch.randn(32, 32, 1, 64, dtype=torch.cfloat, device=device)
    out = InterconnectBlock(case="synthetic").apply_interconnect(frame)
    assert out.shape == frame.shape
    # A non-trivial windowing filter should change the frame.
    assert not torch.allclose(out, frame)


def test_afe_block_matmul_and_reconstruct_shapes():
    afe = AFEBlock(exp=5, mantissa=6)
    n, m, F = 64, 32, 16
    V = torch.randn(n, F, dtype=torch.cfloat, device=device)
    A = torch.randn(m, n, dtype=torch.cfloat, device=device)
    Aq, X = afe.apply_mat_mul(A, V)
    assert Aq.shape == A.shape
    assert X.shape == (m, F)
    Xt = afe.reconstruct(Aq, X)
    assert Xt.shape == (n, F)


def test_ada_oja_block_update_runs():
    block = AdaOjaBlock(n=64, d=4)
    block.oja.U = rand_orth_complex(64, 4)
    A = block.gen_A_ada()
    V = torch.randn(64, 8, dtype=torch.cfloat, device=device)
    X = A @ V
    U_before = block.oja.U.clone()
    block.update(X, A)
    assert block.oja.U.shape == U_before.shape


# --------------------------------------------------- SionnaEnvironmentBlock link selection

def _write_multilink_munich(tmp_path, monkeypatch, links, n_frames=3, n_rx=4, n_freqs=8):
    """Write a multi-link dict pkl and point the 'munich' scenario at it."""
    import e2e.environment.sionna_iterator as si

    r = np.random.default_rng(1)
    data = {
        name: (r.standard_normal((n_frames, n_rx, 1, 1, n_freqs))
               + 1j * r.standard_normal((n_frames, n_rx, 1, 1, n_freqs))).astype(np.complex64)
        for name in links
    }
    path = tmp_path / "munich_multilink.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)
    # Factory reads the module-global path at call time, so patching it redirects the block.
    monkeypatch.setattr(si, "SIONNA_MUNICH_PATH", str(path))
    return data


def test_env_block_default_selects_first_link(tmp_path, monkeypatch):
    from e2e.blocks import SionnaEnvironmentBlock

    data = _write_multilink_munich(tmp_path, monkeypatch, ["tx0", "tx1"])
    block = SionnaEnvironmentBlock("munich")
    assert block.sionna_iterator.link == "tx0"
    got = block.get_S_pars().detach().cpu().numpy()
    np.testing.assert_array_equal(got, data["tx0"][0])


def test_env_block_explicit_link_selects_right_one(tmp_path, monkeypatch):
    from e2e.blocks import SionnaEnvironmentBlock

    data = _write_multilink_munich(tmp_path, monkeypatch, ["tx0", "tx1"])
    block = SionnaEnvironmentBlock("munich", link="tx1")
    assert block.link == "tx1"
    assert block.sionna_iterator.link == "tx1"
    got = block.get_S_pars().detach().cpu().numpy()
    np.testing.assert_array_equal(got, data["tx1"][0])


def test_env_block_single_array_pkl_still_works(tmp_path, monkeypatch):
    import e2e.environment.sionna_iterator as si
    from e2e.blocks import SionnaEnvironmentBlock

    r = np.random.default_rng(2)
    arr = (r.standard_normal((3, 4, 1, 1, 8))
           + 1j * r.standard_normal((3, 4, 1, 1, 8))).astype(np.complex64)
    path = tmp_path / "munich_single.pkl"
    with open(path, "wb") as f:
        pickle.dump(arr, f)
    monkeypatch.setattr(si, "SIONNA_MUNICH_PATH", str(path))

    # Default and an explicit (ignored) link both work for legacy single-array pkls.
    block = SionnaEnvironmentBlock("munich")
    assert block.sionna_iterator.links is None
    np.testing.assert_array_equal(block.get_S_pars().detach().cpu().numpy(), arr[0])

    block2 = SionnaEnvironmentBlock("munich", link="anything")
    np.testing.assert_array_equal(block2.get_S_pars().detach().cpu().numpy(), arr[0])

    # Legacy pkl (no meta) -> (32, 32) fallback and no v2 metadata.
    assert block.array_shape == (32, 32)
    assert block.freq_plan is None
    assert block.physical_scale is None


def _write_v2_multilink_munich(tmp_path, monkeypatch, links_shapes, n_frames=2, n_freqs=8):
    """Write a v2 self-describing multi-link pkl ({"meta": ..., "links": ...}) and point
    the 'munich' scenario at it. `links_shapes` maps link name -> (rows, cols) rx shape."""
    import e2e.environment.sionna_iterator as si

    r = np.random.default_rng(3)
    freq_plan = {"carrier_hz": 28e9, "start_hz": 28e9, "stop_hz": 29e9, "num_freqs": n_freqs}
    links_meta = {}
    links_data = {}
    for name, (rows, cols) in links_shapes.items():
        n_rx = rows * cols
        links_meta[name] = {
            "tx_node": "tx0", "rx_node": "rx0", "rx_array_shape": [rows, cols],
            "n_tx_ant": 1, "kind": "radar", "tx_power_dbm": 20.0, "physical_scale": True,
        }
        links_data[name] = (r.standard_normal((n_frames, n_rx, 1, 1, n_freqs))
                             + 1j * r.standard_normal((n_frames, n_rx, 1, 1, n_freqs))).astype(np.complex64)
    data = {
        "meta": {"version": 1, "scenario_name": "munich", "freq_plan": freq_plan, "links": links_meta},
        "links": links_data,
    }
    path = tmp_path / "munich_v2.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)
    monkeypatch.setattr(si, "SIONNA_MUNICH_PATH", str(path))
    return data, freq_plan


def test_env_block_v2_pkl_auto_derives_array_shape_and_meta(tmp_path, monkeypatch):
    from e2e.blocks import SionnaEnvironmentBlock

    data, freq_plan = _write_v2_multilink_munich(tmp_path, monkeypatch, {"tx0": (8, 6)})
    block = SionnaEnvironmentBlock("munich")
    # rx_array_shape metadata is [num_rows, num_cols], but Sionna numbers antennas
    # column-first (row varies fastest along the flat RX axis), so the row-major
    # aperture view needs the slow axis first: (num_cols, num_rows).
    assert block.array_shape == (6, 8)
    assert block.freq_plan == freq_plan
    assert block.physical_scale is True
    got = block.get_S_pars().detach().cpu().numpy()
    np.testing.assert_array_equal(got, data["links"]["tx0"][0])


def test_env_block_explicit_array_shape_overrides_v2_meta(tmp_path, monkeypatch):
    from e2e.blocks import SionnaEnvironmentBlock

    _write_v2_multilink_munich(tmp_path, monkeypatch, {"tx0": (8, 6)})
    block = SionnaEnvironmentBlock("munich", array_shape=(4, 12))
    assert block.array_shape == (4, 12)
    # metadata pass-throughs are independent of the array_shape override.
    assert block.freq_plan is not None
    assert block.physical_scale is True
