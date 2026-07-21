"""Unit tests for full-aperture spatial combining (e2e.comms.beamforming), plus
the ModemBlock combining modes ("mrc" / "subspace") that consume it."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from e2e.comms import beamforming as bf
from e2e.comms.blocks import ModemBlock, BERBlock

device = bf.device


def _freqs(n=64, start=28.5e9, stop=31.5e9):
    return np.linspace(start, stop, n)


# --------------------------------------------------------------------------- element_channels

def test_element_channels_flat_order_matches_measurement_stage_flatten():
    """element_channels must flatten the SAME way MeasurementStage builds V:
    `s_pars.view(-1, s_pars.shape[-1])` -- a plain row-major reshape, no
    fancy per-element selection."""
    torch.manual_seed(0)
    rx_x, rx_y, F = 4, 3, 32
    s_pars = torch.randn(rx_x, rx_y, 1, F, dtype=torch.cfloat, device=device)
    freqs = _freqs(F)

    H = bf.element_channels(s_pars, freqs, freqs)   # target == source grid: near-identity
    expected_flat = s_pars.view(-1, F)

    assert H.shape == (rx_x * rx_y, F)
    # interpolation onto the identical grid should reproduce the source values closely
    np.testing.assert_allclose(H.cpu().numpy(), expected_flat.cpu().numpy(), atol=1e-4)


def test_element_channels_matches_per_row_np_interp():
    """Cross-check the vectorized batch interpolation against a naive per-row loop."""
    rng = np.random.default_rng(0)
    N, F = 10, 40
    freqs = np.linspace(28e9, 32e9, F)
    arr = (rng.standard_normal((N, F)) + 1j * rng.standard_normal((N, F))).astype(np.complex64)
    target = np.linspace(29e9, 31e9, 20)

    H = bf.element_channels(arr, freqs, target)

    expected = np.stack([
        np.interp(target, freqs, arr[i].real) + 1j * np.interp(target, freqs, arr[i].imag)
        for i in range(N)
    ])
    np.testing.assert_allclose(H.cpu().numpy(), expected, atol=1e-4)


# --------------------------------------------------------------------------- mrc_weights / combine

def test_mrc_weights_unit_norm_columns():
    rng = np.random.default_rng(1)
    N, n_sc = 32, 16
    H = torch.tensor(rng.standard_normal((N, n_sc)) + 1j * rng.standard_normal((N, n_sc)),
                     dtype=torch.complex64, device=device)
    w = bf.mrc_weights(H)
    norms = torch.linalg.norm(w, dim=0)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)


def test_mrc_weights_zero_channel_guard():
    N, n_sc = 8, 4
    H = torch.zeros(N, n_sc, dtype=torch.complex64, device=device)
    w = bf.mrc_weights(H)
    assert torch.all(w == 0)
    assert torch.all(torch.isfinite(w.real)) and torch.all(torch.isfinite(w.imag))


def test_subspace_weights_unit_norm():
    torch.manual_seed(0)
    U, _ = torch.linalg.qr(torch.randn(64, 4, dtype=torch.cfloat, device=device))
    u1 = bf.subspace_weights(U)
    assert u1.shape == (64,)
    assert torch.linalg.norm(u1).item() == pytest.approx(1.0, abs=1e-5)


def test_combine_matched_filter_recovers_signal_no_noise():
    """With no noise, combine(H, mrc_weights(H)) should equal ||H[:,k]|| (the
    matched-filter gain), a basic sanity check on the combine/weights algebra."""
    rng = np.random.default_rng(2)
    N, n_sc = 20, 6
    H = torch.tensor(rng.standard_normal((N, n_sc)) + 1j * rng.standard_normal((N, n_sc)),
                     dtype=torch.complex64, device=device)
    w = bf.mrc_weights(H)
    H_eff = bf.combine(H, w)
    expected = torch.linalg.norm(H, dim=0)
    torch.testing.assert_close(torch.abs(H_eff), expected, atol=1e-4, rtol=1e-4)


def test_combine_broadcasts_batched_symbols():
    rng = np.random.default_rng(3)
    n_sym, N, n_sc = 5, 12, 4
    rx = torch.tensor(rng.standard_normal((n_sym, N, n_sc)) + 1j * rng.standard_normal((n_sym, N, n_sc)),
                      dtype=torch.complex64, device=device)
    w = torch.tensor(rng.standard_normal((N, n_sc)) + 1j * rng.standard_normal((N, n_sc)),
                     dtype=torch.complex64, device=device)
    y = bf.combine(rx, w)
    assert y.shape == (n_sym, n_sc)
    # matches an explicit per-symbol loop
    expected = torch.stack([bf.combine(rx[i], w) for i in range(n_sym)])
    torch.testing.assert_close(y, expected)


def test_combine_broadband_weight_broadcasts_over_subcarriers():
    rng = np.random.default_rng(4)
    N, n_sc = 16, 8
    rx = torch.tensor(rng.standard_normal((N, n_sc)) + 1j * rng.standard_normal((N, n_sc)),
                      dtype=torch.complex64, device=device)
    w = torch.tensor(rng.standard_normal(N) + 1j * rng.standard_normal(N),
                     dtype=torch.complex64, device=device)
    y = bf.combine(rx, w)
    expected = torch.stack([bf.combine(rx[:, k:k + 1], w).squeeze(-1) for k in range(n_sc)], dim=0)
    torch.testing.assert_close(y, expected)


# --------------------------------------------------------------------------- array gain (Monte Carlo)

def test_mrc_array_gain_matches_10log10N_monte_carlo():
    """Every element has unit-magnitude, random-phase channel. Injecting i.i.d.
    unit-variance per-element noise and combining with MRC should recover a
    post-combining SNR gain of 10*log10(N) dB (coherent addition of N unit
    signals vs. incoherent unit-variance noise), within ~0.5 dB over many
    noise draws."""
    rng = np.random.default_rng(0)
    N, n_trials = 64, 400
    phases = rng.uniform(0, 2 * np.pi, size=N)
    H = torch.tensor(np.exp(1j * phases), dtype=torch.complex64, device=device)[:, None]  # [N, 1]
    w = bf.mrc_weights(H)   # [N, 1], unit norm

    gen = torch.Generator(device="cpu")
    gen.manual_seed(123)
    sig_pow_sum = 0.0
    noise_pow_sum = 0.0
    for _ in range(n_trials):
        noise = (torch.randn(N, 1, generator=gen) + 1j * torch.randn(N, 1, generator=gen)) / np.sqrt(2.0)
        noise = noise.to(device).to(torch.complex64)   # unit variance per element
        rx = H + noise
        y_sig = bf.combine(H, w)
        y_full = bf.combine(rx, w)
        y_noise = y_full - y_sig
        sig_pow_sum += torch.abs(y_sig).item() ** 2
        noise_pow_sum += torch.abs(y_noise).item() ** 2

    post_combining_snr = sig_pow_sum / noise_pow_sum
    # per-element SNR is 1 (unit signal, unit noise variance) -> gain == post SNR
    gain_db = 10.0 * np.log10(post_combining_snr)
    expected_db = 10.0 * np.log10(N)
    assert gain_db == pytest.approx(expected_db, abs=0.5)


# --------------------------------------------------------------------------- ModemBlock combining

def _state_dict(rx_x=8, rx_y=8, n_freqs=64, seed=0):
    torch.manual_seed(seed)
    s_pars = torch.randn(rx_x, rx_y, 1, n_freqs, dtype=torch.cfloat, device=device)
    return {"s_pars": s_pars}


def test_element0_bit_exact_regression(n_freqs=64):
    """combining='element0' (the default) is bit-exact with the pre-`combining`
    ModemBlock -- same code path, just reorganized."""
    freqs = _freqs(n_freqs)
    state = _state_dict(n_freqs=n_freqs, seed=7)

    block_default = ModemBlock(freqs, n_symbols=8, fft_size=n_freqs, cp_len=16, n_active=52,
                               pilot_spacing=8, bits_per_symbol=2, snr_db=15.0,
                               equalizer="mmse", estimator="ls", seed=2)
    block_explicit = ModemBlock(freqs, n_symbols=8, fft_size=n_freqs, cp_len=16, n_active=52,
                                pilot_spacing=8, bits_per_symbol=2, snr_db=15.0,
                                equalizer="mmse", estimator="ls", seed=2, combining="element0")

    out_default = block_default.apply(dict(state))
    out_explicit = block_explicit.apply(dict(state))

    assert torch.equal(out_default["comm_rx_bits"], out_explicit["comm_rx_bits"])
    torch.testing.assert_close(out_default["comm_data_eq"], out_explicit["comm_data_eq"])
    torch.testing.assert_close(out_default["comm_H_true"], out_explicit["comm_H_true"])
    assert "comm_array_gain_db" not in out_default


def test_mrc_beats_element0_ber_same_snr():
    """MRC combining across a 64-element aperture should give a lower (or equal)
    BER than the single-element tap at the same nominal per-element SNR, on a
    synthetic multipath-per-element frame."""
    n_freqs = 64
    freqs = _freqs(n_freqs)
    rx_x, rx_y = 8, 8   # N = 64
    snr_db = 5.0

    bers = {}
    for mode in ("element0", "mrc"):
        state = _state_dict(rx_x, rx_y, n_freqs, seed=11)
        modem = ModemBlock(freqs, n_symbols=16, fft_size=n_freqs, cp_len=16, n_active=52,
                          pilot_spacing=8, bits_per_symbol=2, snr_db=snr_db,
                          equalizer="mmse", estimator="ls", seed=5, combining=mode)
        out = modem.apply(state)
        ber_out = BERBlock().apply({**state, **out})
        bers[mode] = ber_out["ber"]

    assert bers["mrc"] <= bers["element0"] + 1e-9


def test_mrc_reports_array_gain_db_and_H_eff():
    n_freqs = 64
    freqs = _freqs(n_freqs)
    state = _state_dict(8, 8, n_freqs, seed=1)
    modem = ModemBlock(freqs, n_symbols=4, fft_size=n_freqs, snr_db=10.0, combining="mrc", seed=0)
    out = modem.apply(state)
    assert "comm_array_gain_db" in out
    assert np.isfinite(out["comm_array_gain_db"])
    # some coherent gain over a single element expected on average (not a tight
    # bound, but it should not be negative for MRC, the SNR-optimal combiner)
    assert out["comm_array_gain_db"] > 0.0
    assert out["comm_H_true"].shape == (modem.modem.fft_size,)


def test_subspace_requires_U_in_state():
    n_freqs = 64
    freqs = _freqs(n_freqs)
    state = _state_dict(8, 8, n_freqs, seed=2)
    modem = ModemBlock(freqs, n_symbols=4, fft_size=n_freqs, snr_db=10.0, combining="subspace", seed=0)
    with pytest.raises(ValueError, match="subspace"):
        modem.apply(state)


def test_mrc_requires_s_pars_not_just_H_sc():
    n_freqs = 64
    freqs = _freqs(n_freqs)
    modem = ModemBlock(freqs, n_symbols=4, fft_size=n_freqs, snr_db=10.0, combining="mrc", seed=0)
    H_sc = torch.ones(modem.modem.fft_size, dtype=torch.complex64, device=device)
    with pytest.raises(ValueError, match="s_pars"):
        modem.apply({"H_sc": H_sc})


def test_subspace_rank1_channel_matches_mrc_level_and_beats_element0():
    """Rank-1 channel x = a * s(f) for a random unit spatial vector `a`: setting
    state['U'][:, 0] = a makes subspace combining match MRC's array gain for
    this channel (both reduce to steering along `a`), and both must beat the
    single-element tap."""
    n_freqs = 64
    freqs = _freqs(n_freqs)
    rx_x, rx_y = 8, 8
    N = rx_x * rx_y
    snr_db = 3.0

    rng = np.random.default_rng(42)
    a = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    a = a / np.linalg.norm(a)
    s_f = rng.standard_normal(n_freqs) + 1j * rng.standard_normal(n_freqs)
    frame = np.outer(a, s_f).astype(np.complex64)              # [N, F], rank-1
    s_pars = torch.from_numpy(frame.reshape(rx_x, rx_y, 1, n_freqs)).to(device)

    U = torch.zeros(N, 4, dtype=torch.complex64, device=device)
    U[:, 0] = torch.from_numpy(a.astype(np.complex64)).to(device)
    U, _ = torch.linalg.qr(U)   # keep U[:,0] direction, orthonormalize the rest

    bers = {}
    gains = {}
    for mode in ("element0", "mrc", "subspace"):
        state = {"s_pars": s_pars.clone()}
        if mode == "subspace":
            state["U"] = U
        modem = ModemBlock(freqs, n_symbols=16, fft_size=n_freqs, cp_len=16, n_active=52,
                          pilot_spacing=8, bits_per_symbol=2, snr_db=snr_db,
                          equalizer="mmse", estimator="ls", seed=9, combining=mode)
        out = modem.apply(state)
        ber_out = BERBlock().apply({**state, **out})
        bers[mode] = ber_out["ber"]
        gains[mode] = out.get("comm_array_gain_db")

    assert bers["subspace"] <= bers["element0"] + 1e-9
    assert bers["mrc"] <= bers["element0"] + 1e-9
    # for this exact rank-1 channel, subspace steering along the true direction
    # gives essentially the same array gain as MRC
    assert gains["subspace"] == pytest.approx(gains["mrc"], abs=0.5)


# --------------------------------------------------------------------------- full pipeline integration

def test_full_pipeline_subspace_combining_runs_green(make_env_block):
    """Simulation with make_env_block + AdaOjaBlock + ModemBlock(combining='subspace')
    + BERBlock runs green and emits 'ber' per frame."""
    from e2e.simulation import Simulation
    from e2e.blocks import AdaOjaBlock

    n_freqs = 64
    env = make_env_block(n_frames=2, n_freqs=n_freqs)
    freqs = _freqs(n_freqs)
    modem = ModemBlock(freqs, n_symbols=4, fft_size=n_freqs, cp_len=16, n_active=52,
                       pilot_spacing=8, bits_per_symbol=2, snr_db=15.0, seed=0,
                       combining="subspace")
    sim = Simulation(
        env,
        [modem, BERBlock()],
        d=16,
        subspace_block=AdaOjaBlock(1024, 16),
    )
    out = sim.run(n_steps=2)
    assert len(out["ber"]) == 2
    assert all(0.0 <= b <= 1.0 for b in out["ber"])
    assert "comm_array_gain_db" in out and len(out["comm_array_gain_db"]) == 2
