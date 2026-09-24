"""The THREE waveform classes, and the oracles that say they are one chain.

`notes/JSAC_WAVEFORM_2026-09-24.md` section 3.8 plus its adversarial review. Every
number here is COMPUTED (from a frequency plan, from the repo's own constellation),
never asserted as a literal -- the design note's own instruction, and the reason the
known-delay oracle cannot bake in the endpoint-inclusive off-by-one it exists to catch.

What each oracle is worth is written on it. In particular O2 (bit parity) certifies the
PLUMBING and not the physics: both arms share `delta_f`, the frequency ordering and the
range transform, so it cannot catch an error in any of them. O1 and O5 are the ones that
carry physics.
"""
from __future__ import annotations

import math

import pytest
import torch

from e2e import frames
from e2e.blocks import RangeAzBlock, RangeProfileBlock, device
from e2e.chain.dechirp import DechirpBlock
from e2e.chain.receive import RangeTransformBlock
from e2e.comms import ofdm_isac as oi
from e2e.comms.blocks import BERBlock
from e2e.comms.ofdm import qam_constellation

N_SC = 64
PLAN = {"carrier_hz": 30e9, "start_hz": 28.5e9, "stop_hz": 31.5e9, "num_freqs": N_SC}


class _Cfg:
    """The minimal cfg the mixing blocks read: `mimo` and `n_tx` only."""
    mimo = "single"
    n_tx = 1
    n_chirps = 1


def _cfr(n_rx=16, n_freqs=N_SC, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(n_rx, 1, 1, n_freqs, generator=g)
         + 1j * torch.randn(n_rx, 1, 1, n_freqs, generator=g))
    return x.to(dtype=torch.complex64, device=device)


def _one_tap_cfr(delay_bins, n_rx=4, n_freqs=N_SC, az_sin=0.0):
    """A CFR whose only path is at `delay_bins` cube bins of excess delay.

    Built from the grid's OWN spacing: `H[k] = exp(-j2pi f_k tau)` sampled at
    `f_k = k * df`, so a forward FFT of `conj(H)` peaks at bin `delay_bins` by
    construction, whatever `df` happens to be.
    """
    k = torch.arange(n_freqs, dtype=torch.float64)
    phase = -2.0 * math.pi * k * float(delay_bins) / n_freqs
    h = torch.exp(1j * phase).to(torch.complex64)
    steer = torch.exp(1j * math.pi * torch.arange(n_rx, dtype=torch.float64)
                      * float(az_sin)).to(torch.complex64)
    return (steer.view(n_rx, 1, 1, 1) * h.view(1, 1, 1, n_freqs)).to(device)


def _range_transform():
    """The IDENTITY-point transform: no window, no DC removal, full window. Every
    oracle here runs at it, because a window or a mean subtraction is a change to the
    thing under test rather than a property of it."""
    return RangeTransformBlock(None, window="none", dc_removal=False,
                               crop_negative_delay=False, delta_f_hz=1.0)


def _cube_through(stages, state):
    for stage in stages:
        state.update(stage.apply(state))
    return state


# ================================================================================
# The classes, as a registry
# ================================================================================
def test_three_registered_classes_with_three_distinct_mixing_modes():
    """The owner's directive was three waveform CLASSES -- sensing, comms, hybrid.

    A class here is the triple (source waveform, mixing mode, product set), and this
    test pins that the three rows are genuinely distinct rather than one row with two
    labels. `ofdm` having `mixing=None` is the substantive part: a comms receiver
    equalises the grid and never forms a cube, which is what makes the difference
    between `ofdm` and `jsac` a screenshot rather than a claim.
    """
    from e2e.chain.waveform import WAVEFORM_KINDS, _WAVEFORM_CLASSES

    assert WAVEFORM_KINDS == ("fmcw", "ofdm", "jsac")
    mixing = {k: _WAVEFORM_CLASSES[k].mixing for k in WAVEFORM_KINDS}
    assert mixing == {"fmcw": "dechirp", "ofdm": None, "jsac": "symbol_division"}

    specs = {k: oi.waveform_chain_spec(k, _Cfg(), freq_plan=PLAN, n_symbols=2)
             for k in WAVEFORM_KINDS}
    assert (specs["fmcw"].sensing, specs["fmcw"].comms) == (True, False)
    assert (specs["ofdm"].sensing, specs["ofdm"].comms) == (False, True)
    assert (specs["jsac"].sensing, specs["jsac"].comms) == (True, True)
    # ofdm and jsac transmit THE SAME frame -- that is what makes them comparable.
    assert torch.equal(specs["ofdm"].frame.tx_grid, specs["jsac"].frame.tx_grid)


def test_the_ofdm_grid_is_the_sources_own_grid():
    """No interpolation anywhere: `n_subcarriers == num_freqs` and the spacing is the
    plan's own endpoint-inclusive step. This equality is the premise that makes
    `Y = H*X` an elementwise multiply, so it is established by construction and pinned
    here rather than assumed downstream."""
    frame = oi.frame_from_freq_plan(PLAN, n_symbols=2)
    assert frame.n_subcarriers == PLAN["num_freqs"]
    expected = (PLAN["stop_hz"] - PLAN["start_hz"]) / (PLAN["num_freqs"] - 1)
    assert frame.subcarrier_spacing_hz == pytest.approx(expected, rel=1e-12)
    # NOT B/N -- the off-by-one this repo has been bitten by (F97d).
    naive = (PLAN["stop_hz"] - PLAN["start_hz"]) / PLAN["num_freqs"]
    assert frame.subcarrier_spacing_hz != pytest.approx(naive, rel=1e-9)


def test_an_ofdm_frame_without_a_grid_is_refused_by_name():
    with pytest.raises(ValueError, match="freq_plan"):
        oi.waveform_chain_spec("jsac", _Cfg())


# ================================================================================
# O2 -- FMCW parity, BIT-EXACT
# ================================================================================
def test_O2_all_pilot_symbol_gives_the_fmcw_cube_bit_for_bit():
    """THE one-chain oracle. With `X == 1` the division is the identity, so the JSAC
    tail is literally `beat_from_cfr` + `mimo_combine` -- the same imported functions
    the dechirp calls -- and the cube must be `torch.equal`, not `allclose`.

    If this ever needs a tolerance, the mixing block's tail has drifted away from the
    dechirp's and the two waveforms are no longer one chain. Fix that; do not loosen
    this.

    WHAT IT DOES NOT CERTIFY (review section 2, and worth restating every time this
    test is read): both arms share `delta_f`, the frequency-axis ordering and the range
    transform, so a mistake in any of them passes here untouched. O1 and O5 carry the
    physics.
    """
    cfr = _cfr()
    frame = oi.OFDMFrame(n_subcarriers=N_SC, subcarrier_spacing_hz=1.0, n_symbols=1)
    assert torch.allclose(frame.tx_grid, torch.ones_like(frame.tx_grid)), (
        "the preamble symbol must be all-pilot with pilot_value 1 -- it is the "
        "parity point, the sensing reference and the channel estimate at once")

    fmcw = _cube_through([DechirpBlock(_Cfg()), _range_transform()],
                         {"s_pars": cfr})
    jsac_state = {"s_pars": oi.apply_ofdm_channel(cfr, frame.tx_grid)}
    jsac = _cube_through([oi.SymbolDivisionBlock(_Cfg(), frame), _range_transform()],
                         jsac_state)

    assert jsac["cube"].shape == fmcw["cube"].shape
    assert torch.equal(jsac["adc"], fmcw["adc"]), "the pre-transform records differ"
    assert torch.equal(jsac["cube"], fmcw["cube"]), (
        f"max |diff| = {float((jsac['cube'] - fmcw['cube']).abs().max()):.3e}")


def test_O2_holds_for_every_symbol_of_a_multi_symbol_preamble_frame():
    """The same identity across M symbols: each symbol sees the same H, so each cube
    slab equals the single-chirp FMCW cube. This is also the test that the symbol axis
    is mapped onto the chirp axis the right way round."""
    cfr = _cfr(n_rx=8)
    m = 3
    frame = oi.OFDMFrame(n_subcarriers=N_SC, subcarrier_spacing_hz=1.0, n_symbols=m,
                         pilot_spacing=1)
    fmcw = _cube_through([DechirpBlock(_Cfg()), _range_transform()],
                         {"s_pars": cfr})["cube"]
    jsac = _cube_through(
        [oi.SymbolDivisionBlock(_Cfg(), frame, x_ref=torch.ones(m, N_SC,
                                                               dtype=torch.complex64)),
         _range_transform()],
        {"s_pars": oi.apply_ofdm_channel(cfr, torch.ones(m, N_SC,
                                                         dtype=torch.complex64))})["cube"]
    assert jsac.shape == (cfr.shape[0], m, N_SC)
    for sym in range(m):
        assert torch.equal(jsac[:, sym, :], fmcw[:, 0, :]), f"symbol {sym}"


def test_O2_degrades_only_to_float_noise_on_a_constant_modulus_data_grid():
    """With a real QPSK grid (unit modulus, not all ones) the division is still exact
    in exact arithmetic, so the parity must hold to float precision. This separates
    "the plumbing is right" from "the all-ones case is a special case"."""
    cfr = _cfr(n_rx=8, seed=3)
    frame = oi.OFDMFrame(n_subcarriers=N_SC, subcarrier_spacing_hz=1.0, n_symbols=2,
                         pilot_spacing=1, bits_per_symbol=2)
    x = frame.tx_grid.clone()
    g = torch.Generator(device="cpu").manual_seed(11)
    const = qam_constellation(2).to(x.device)
    x[1] = const[torch.randint(0, const.numel(), (N_SC,), generator=g)].to(x.device)

    fmcw = _cube_through([DechirpBlock(_Cfg()), _range_transform()],
                         {"s_pars": cfr})["cube"]
    jsac = _cube_through(
        [oi.SymbolDivisionBlock(_Cfg(), frame, x_ref=x), _range_transform()],
        {"s_pars": oi.apply_ofdm_channel(cfr, x)})["cube"]
    rel = float((jsac[:, 1, :] - fmcw[:, 0, :]).abs().max()
                / fmcw[:, 0, :].abs().max())
    assert rel < 1e-5, f"constant-modulus division is not exact: rel {rel:.3e}"


# ================================================================================
# O1 -- known delay. THE oracle that carries physics.
# ================================================================================
@pytest.mark.parametrize("delay_bins", [0, 7, 23])
def test_O1_known_delay_lands_at_the_right_cube_bin_on_both_arms(delay_bins):
    """A one-tap channel at `tau` peaks at cube bin `round(tau * N * df)` -- written
    against `N * df` and never against a nominal bandwidth, so an endpoint-inclusive
    grid cannot hide an `N/(N-1)` error inside it.

    Run on BOTH arms in one test, deliberately: O2 says the two cubes are the same
    object and this says that object is in the right place. Neither statement alone is
    enough.
    """
    cfr = _one_tap_cfr(delay_bins)
    frame = oi.OFDMFrame(n_subcarriers=N_SC, subcarrier_spacing_hz=1.0, n_symbols=1)

    fmcw = _cube_through([DechirpBlock(_Cfg()), _range_transform()],
                         {"s_pars": cfr})["cube"]
    jsac = _cube_through([oi.SymbolDivisionBlock(_Cfg(), frame), _range_transform()],
                         {"s_pars": oi.apply_ofdm_channel(cfr, frame.tx_grid)})["cube"]
    for name, cube in (("fmcw", fmcw), ("jsac", jsac)):
        power = (cube[:, 0, :].abs() ** 2).sum(dim=0)
        assert int(power.argmax()) == delay_bins, f"{name} peaked at {int(power.argmax())}"


# ================================================================================
# O5 -- handedness. The A/B screen's failure mode.
# ================================================================================
def test_O5_the_two_arms_agree_on_range_AND_azimuth_handedness():
    """Same one-tap CFR through both arms, through the SHIPPED angle product.

    This exists because the review measured the v1.0 image path and the ADC path
    peaking at MIRRORED range bins: the image path forward-FFT'd the raw, unconjugated
    CFR and fftshifted, while the ADC path conjugated first. A side-by-side A/B screen
    is exactly where two mirrored images are most visible and least explicable, so the
    agreement is asserted on the product a screen actually draws -- not on the cube,
    where a mirror could still be introduced downstream.
    """
    n_rx, az_sin = 16, 0.37
    cfr = _one_tap_cfr(9, n_rx=n_rx, az_sin=az_sin)
    frame = oi.OFDMFrame(n_subcarriers=N_SC, subcarrier_spacing_hz=1.0, n_symbols=1)
    product = RangeAzBlock(bins=32, array_shape=(n_rx, 1))

    def image(stages, state):
        st = _cube_through(stages, state)
        st["aperture_shape"] = (n_rx, 1)
        return product.apply(st)["range_az"]

    a = image([DechirpBlock(_Cfg()), _range_transform()], {"s_pars": cfr})
    b = image([oi.SymbolDivisionBlock(_Cfg(), frame), _range_transform()],
              {"s_pars": oi.apply_ofdm_channel(cfr, frame.tx_grid)})
    assert a.shape == b.shape
    peak_a = divmod(int(a.argmax()), a.shape[1])
    peak_b = divmod(int(b.argmax()), b.shape[1])
    assert peak_a == peak_b, (
        f"the two arms' images are not in the same place: fmcw {peak_a}, jsac "
        f"{peak_b} (azimuth bin, range gate)")


# ================================================================================
# O4 -- the division's noise amplification, in closed form
# ================================================================================
@pytest.mark.parametrize("bits,expected_mean,expected_worst", [
    (2, 0.0, 0.0),        # QPSK: constant modulus, nothing to amplify
    (4, 2.76, 6.99),
    (6, 4.29, 13.22),
])
def test_O4_qam_order_raises_the_cube_floor_by_the_closed_form(bits, expected_mean,
                                                               expected_worst):
    """`Z = Y/X` scales noise by `1/|X|`, so a higher-order constellation buys data rate
    and pays a raised cube floor -- Xiong et al.'s deterministic-random tradeoff, in one
    A/B. The function computes the expectation over the repo's OWN constellation, so it
    cannot drift from the symbols actually transmitted; the literals here are the design
    note's published numbers and exist to catch a change in the constellation's
    normalisation."""
    mean_db, worst_db = oi.qam_division_noise_rise_db(bits)
    assert mean_db == pytest.approx(expected_mean, abs=0.01)
    assert worst_db == pytest.approx(expected_worst, abs=0.01)


def test_O4_is_visible_in_an_actual_cube_floor():
    """The closed form, measured through the chain rather than asserted on paper: a
    noise-only frame divided by a 16-QAM grid has a cube floor `E[1/|X|^2]` above the
    same frame divided by QPSK."""
    g = torch.Generator(device="cpu").manual_seed(5)
    noise = (torch.randn(8, 1, 1, N_SC, generator=g)
             + 1j * torch.randn(8, 1, 1, N_SC, generator=g)).to(
                 dtype=torch.complex64, device=device)

    def floor_db(bits):
        frame = oi.OFDMFrame(n_subcarriers=N_SC, subcarrier_spacing_hz=1.0,
                             n_symbols=2, pilot_spacing=1, bits_per_symbol=bits,
                             seed=1)
        x = frame.tx_grid.clone()
        const = qam_constellation(bits).to(x.device)
        gg = torch.Generator(device="cpu").manual_seed(2)
        x[1] = const[torch.randint(0, const.numel(), (N_SC,), generator=gg)].to(x.device)
        # The received "grid" here is pure noise: no channel, so the only thing the
        # division does is scale it by 1/|X|.
        y = noise.expand(-1, -1, 2, -1)
        st = _cube_through(
            [oi.SymbolDivisionBlock(_Cfg(), frame, x_ref=x), _range_transform()],
            {"s_pars": y.contiguous()})
        return 10.0 * math.log10(float((st["cube"][:, 1, :].abs() ** 2).mean()))

    rise = floor_db(4) - floor_db(2)
    # The ENSEMBLE expectation is +2.76 dB; what this particular 64-subcarrier draw
    # predicts is `10*log10(mean(1/|x|^2))` over the symbols actually transmitted, and
    # that is the number the measurement must match tightly. Asserting the ensemble
    # value to a loose tolerance would hide a real error inside the sampling spread --
    # the draw here predicts ~3.3 dB, which is 0.5 dB off the ensemble figure.
    def drawn_rise_db(bits):
        const = qam_constellation(bits).to(device)
        gg = torch.Generator(device="cpu").manual_seed(2)
        x = const[torch.randint(0, const.numel(), (N_SC,), generator=gg)]
        return 10.0 * math.log10(float((1.0 / x.abs() ** 2).mean()))

    predicted = drawn_rise_db(4) - drawn_rise_db(2)
    assert rise == pytest.approx(predicted, abs=0.05), (
        f"measured floor rise {rise:+.2f} dB, this draw predicts {predicted:+.2f} dB "
        f"(ensemble expectation {oi.qam_division_noise_rise_db(4)[0]:+.2f} dB)")


# ================================================================================
# The resource split -- the knob that makes "hybrid" mean something
# ================================================================================
def test_the_pilot_spacing_knob_trades_sensing_window_against_data_rate():
    """Turn ONE knob and both products move, in opposite directions, on one frame.

    `sensing_source="pilots_only"` samples the channel every P-th subcarrier, so the
    sensing product's unambiguous window is `c / (P * df)` -- it shrinks by P and the
    scene's multipath aliases -- while P-1 of every P subcarriers carry data and the
    rate rises. That is a genuine frequency-domain division of one waveform between the
    two functions, which is what "hybrid" has to mean beyond "two products coexist".
    """
    def frame(p):
        return oi.OFDMFrame(n_subcarriers=1024, subcarrier_spacing_hz=6e5,
                            n_symbols=4, pilot_spacing=p, sensing_source="pilots_only")

    windows = [frame(p).sensing_window_m() for p in (1, 4, 8)]
    rates = [frame(p).data_rate_bps() for p in (1, 4, 8)]
    assert windows[0] > windows[1] > windows[2]
    assert rates[0] < rates[1] < rates[2]
    assert rates[0] == 0.0, "P=1 is all-pilot: zero bits, and it is the parity point"
    # The window is exactly c/(P*df) -- derived, so a change in the convention or the
    # spacing moves it and this test notices.
    assert windows[1] == pytest.approx(windows[0] / 4, rel=1e-9)


def test_pilots_only_zeroes_the_data_subcarriers_rather_than_dividing_by_them():
    frame = oi.OFDMFrame(n_subcarriers=N_SC, subcarrier_spacing_hz=1.0, n_symbols=2,
                         pilot_spacing=4, sensing_source="pilots_only")
    ref = frame.reference_grid()
    assert torch.count_nonzero(ref[0]) == N_SC          # the preamble is all pilots
    assert int(torch.count_nonzero(ref[1])) == N_SC // 4
    # ...and the block must produce zeros there, not NaNs.
    st = _cube_through([oi.SymbolDivisionBlock(_Cfg(), frame)],
                       {"s_pars": oi.apply_ofdm_channel(_cfr(n_rx=4), frame.tx_grid)})
    assert torch.isfinite(st["adc"]).all()


def test_preamble_sensing_zeroes_every_data_symbol():
    frame = oi.OFDMFrame(n_subcarriers=N_SC, subcarrier_spacing_hz=1.0, n_symbols=3,
                         sensing_source="preamble")
    ref = frame.reference_grid()
    assert torch.count_nonzero(ref[0]) == N_SC
    assert torch.count_nonzero(ref[1:]) == 0


# ================================================================================
# The chain contract: M > 1 flows, and the products that must NOT
# ================================================================================
def test_a_jsac_cube_carries_the_symbol_axis_label_and_reaches_the_angle_products():
    """The cube's slow axis says `symbol`, the fast axis says `range_bin`, and the
    angle products accept it because they broadcast over the slow axis. This is the
    half of the axis contract that has to PASS."""
    n_rx = 16
    frame = oi.OFDMFrame(n_subcarriers=N_SC, subcarrier_spacing_hz=1.0, n_symbols=3)
    st = _cube_through(
        [oi.SymbolDivisionBlock(_Cfg(), frame), _range_transform()],
        {"s_pars": oi.apply_ofdm_channel(_cfr(n_rx=n_rx), frame.tx_grid),
         "aperture_shape": (n_rx, 1)})
    assert st["cube_axes"] == {"slow": "symbol", "fast": "range_bin"}
    az = RangeAzBlock(bins=32, array_shape=(n_rx, 1)).apply(st)["range_az"]
    assert az.shape[0] == 3, "CHIRP_BROADCAST must emit one image per symbol"
    prof = RangeProfileBlock(bins=16, slow_index=0).apply(st)["range_profile"]
    assert prof.shape == (n_rx, 16)


def test_the_range_doppler_product_refuses_a_symbol_slow_cube_by_name():
    """The half that has to FAIL, and loudly. A Doppler FFT over OFDM symbols of one
    time-invariant stored channel is a delta at bin 0 dressed up as a velocity
    measurement -- the most plausible-looking wrong picture this chain can draw."""
    from e2e.chain.receive import RadarCubeBlock

    frame = oi.OFDMFrame(n_subcarriers=N_SC, subcarrier_spacing_hz=1.0, n_symbols=4)
    st = _cube_through([oi.SymbolDivisionBlock(_Cfg(), frame),
                        RangeTransformBlock(None, window="hann", dc_removal=True,
                                            crop_negative_delay=False,
                                            delta_f_hz=1.0)],
                       {"s_pars": oi.apply_ofdm_channel(_cfr(n_rx=4), frame.tx_grid)})
    cfg = _Cfg()
    cfg.n_chirps = 4
    with pytest.raises(frames.FrameContractError, match="slow axis"):
        RadarCubeBlock(cfg).apply(st)


def test_symbol_division_refuses_a_mimo_frame_by_name():
    frame = oi.OFDMFrame(n_subcarriers=N_SC, subcarrier_spacing_hz=1.0, n_symbols=1)
    y = oi.apply_ofdm_channel(_cfr(n_rx=4).expand(-1, 2, -1, -1).contiguous(),
                              frame.tx_grid)
    with pytest.raises(frames.FrameContractError, match="symbol axis"):
        oi.SymbolDivisionBlock(_Cfg(), frame).apply({"s_pars": y})


def test_the_channel_apply_refuses_a_grid_off_the_channels_own_sampling():
    frame = oi.OFDMFrame(n_subcarriers=N_SC // 2, subcarrier_spacing_hz=1.0,
                         n_symbols=1)
    with pytest.raises(frames.FrameContractError, match="same grid"):
        oi.apply_ofdm_channel(_cfr(), frame.tx_grid)


# ================================================================================
# O3 -- the comms head, as a consumer of the chain
# ================================================================================
def test_O3_ber_is_exactly_zero_on_a_clean_flat_channel():
    """No noise anywhere in the chain -> BER exactly 0.0, not 'small'. Anything else
    means the receive path is demapping against a grid the transmitter did not send,
    or equalising with an estimate of the wrong channel."""
    n_rx = 8
    spec = oi.waveform_chain_spec("jsac", _Cfg(), freq_plan=PLAN, n_symbols=4,
                                  pilot_spacing=4, combining="mrc")
    h = torch.ones(n_rx, 1, 1, N_SC, dtype=torch.complex64, device=device)
    state = {"s_pars": h}
    state.update(spec.channel_block.apply(state))
    state.update(spec.receive_block.apply(state))
    out = BERBlock().apply(state)
    assert out["ber"] == 0.0, f"BER {out['ber']} on a clean flat channel"
    assert out["evm"] < 1e-5


def test_the_comms_head_works_on_a_frequency_selective_channel():
    """The same, on a random multipath CFR: the equaliser has to actually equalise."""
    n_rx = 8
    spec = oi.waveform_chain_spec("jsac", _Cfg(), freq_plan=PLAN, n_symbols=4,
                                  pilot_spacing=2, combining="mrc")
    state = {"s_pars": _cfr(n_rx=n_rx, seed=9)}
    state.update(spec.channel_block.apply(state))
    state.update(spec.receive_block.apply(state))
    assert BERBlock().apply(state)["ber"] == 0.0


def test_the_comms_head_injects_no_noise_of_its_own():
    """Contract section 1.4: ONE noise source on the chain. Running the same frame
    twice must give bit-identical outputs -- a head with its own AWGN draw would not.
    This is the structural version of the claim; a power test would pass either way."""
    spec = oi.waveform_chain_spec("ofdm", _Cfg(), freq_plan=PLAN, n_symbols=3)
    state = {"s_pars": _cfr(n_rx=4, seed=4)}
    state.update(spec.channel_block.apply(state))
    a = spec.receive_block.apply(dict(state))
    b = spec.receive_block.apply(dict(state))
    assert torch.equal(a["comm_data_eq"], b["comm_data_eq"])
    assert a["comm_noise_source"] == "none"


def test_the_equaliser_snr_is_measured_not_configured():
    """The review's R-new-2, as a test. `OFDMReceiveBlock` takes no `snr_db`: once the
    front end's cascade is the only noise source, nothing in the pipeline knows the
    post-combining SNR, and a number typed into a preset would reach a card as if it
    had been measured. The estimate comes from the pilot residual, and it must TRACK
    the noise actually present.
    """
    import inspect

    from e2e.comms import channel as ch

    assert "snr_db" not in inspect.signature(oi.OFDMReceiveBlock.__init__).parameters

    n_rx, n_sym = 4, 6
    spec = oi.waveform_chain_spec("ofdm", _Cfg(), freq_plan=PLAN, n_symbols=n_sym,
                                  pilot_spacing=2)
    h = torch.ones(n_rx, 1, 1, N_SC, dtype=torch.complex64, device=device)
    state = {"s_pars": h}
    state.update(spec.channel_block.apply(state))

    def measured(snr_db):
        g = torch.Generator(device="cpu").manual_seed(0)
        y = state["s_pars"]
        sigma = math.sqrt(10 ** (-snr_db / 10.0) / 2.0)
        noise = (torch.randn(y.shape, generator=g)
                 + 1j * torch.randn(y.shape, generator=g)).to(y.device) * sigma
        st = dict(state)
        st["s_pars"] = y + noise.to(y.dtype)
        return spec.receive_block.apply(st)["comm_snr_db"]

    lo, hi = measured(5.0), measured(25.0)
    assert hi - lo == pytest.approx(20.0, abs=3.0), (
        f"the measured SNR moved {hi - lo:+.2f} dB for a 20 dB change in the injected "
        f"noise ({lo:.1f} -> {hi:.1f})")
    # One symbol gives no residual to pool across: the estimator must say None rather
    # than invent a figure.
    assert ch.estimate_snr_db(torch.ones(1, 4, dtype=torch.complex64),
                              torch.ones(1, 4, dtype=torch.complex64)) is None


# ================================================================================
# Numerology and the card-bound numbers
# ================================================================================
def test_the_data_rate_is_a_burst_rate_and_the_average_needs_a_frame_rate():
    frame = oi.frame_from_freq_plan(PLAN, n_symbols=4, pilot_spacing=8)
    burst = frame.data_rate_bps()
    assert burst > 0
    # The duty cycle is a separate, stated assumption -- it cannot ride along silently.
    avg = frame.average_rate_bps(frame_rate_hz=10.0)
    assert avg == pytest.approx(burst * frame.frame_duration_s() * 10.0, rel=1e-12)
    assert avg < burst


def test_the_cp_guard_reads_the_pdp_rather_than_a_named_path():
    """The guard is an ENERGY criterion on the measured PDP. A one-tap channel needs no
    CP; a channel with a late tap needs one longer than that tap."""
    flat = torch.ones(4, 1, 1, N_SC, dtype=torch.complex64, device=device)
    ok, idx = oi.cp_guard_ok(flat, cp_len=0)
    assert ok and idx == 0

    late = _one_tap_cfr(N_SC // 2, n_rx=4)
    ok, idx = oi.cp_guard_ok(late, cp_len=4)
    assert not ok and idx == pytest.approx(N_SC // 2, abs=1)


def test_papr_of_an_fmcw_chirp_is_zero_and_of_an_ofdm_symbol_is_not():
    """The TX-side statement, which is true and uncontroversial -- kept separate from
    any claim about the RECEIVE front end's clamp, which runs the other way (see
    `measure_lna_input_papr`)."""
    from e2e.chain.waveform import WaveformBlock

    chirp = WaveformBlock(kind="fmcw", n_t=256, bw=1e9,
                          sample_rate=1e9).apply({})["tx_wave"]
    assert oi.measure_papr_db(chirp) == pytest.approx(0.0, abs=0.01)

    grid = WaveformBlock(kind="ofdm", freq_plan=PLAN, n_t=N_SC,
                         n_symbols=4, pilot_spacing=4).apply({})["tx_wave"]
    assert oi.measure_papr_db(grid) > 3.0


def test_the_lna_input_papr_measurement_runs_the_way_the_review_measured_it():
    """The PAPR question the review left open, pinned as a MEASUREMENT rather than as
    a story. On a channel whose energy is concentrated in one tap, `ifft(H)` is a spike
    and the bare CFR is FAR peakier at the LNA than the same channel carrying an OFDM
    grid -- i.e. the intuition "OFDM is what makes the front end clip" is backwards for
    the configuration that ships (`s_pars = H`, waveform and modulate blocks off).

    A card that turns a drive knob must name which FMCW configuration it is A/B'd
    against and quote both numbers from this function at the preset's own operating
    point. This test pins the DIRECTION so that requirement cannot quietly lapse.
    """
    spike = _one_tap_cfr(0, n_rx=32)                       # all energy in one tap
    frame = oi.OFDMFrame(n_subcarriers=N_SC, subcarrier_spacing_hz=1.0, n_symbols=2,
                         pilot_spacing=4, bits_per_symbol=2)
    y = oi.apply_ofdm_channel(spike, frame.tx_grid)
    bare = oi.measure_lna_input_papr(spike)
    with_grid = oi.measure_lna_input_papr(y[:, :, 1:, :])
    assert bare > with_grid + 5.0, (
        f"the bare CFR measured {bare:.2f} dB at the LNA and the OFDM-carrying frame "
        f"{with_grid:.2f} dB. If this has inverted, re-measure before any card claims "
        f"the OFDM waveform is what drives the front end into its clamp.")
