"""Oracles for the FULL one-chain contract: front end on beat samples, noise once,
the bistatic range convention, and the waveform-class registry.

Contract: `notes/ONE_CHAIN_CONTRACT_2026-09-24.md` sections 1.2 (FULL column), 1.4,
2 and 3.6, as scoped by the owner's 2026-09-24 ballot answer. The MVC oracles live in
`tests/test_one_chain_spine.py` and are NOT superseded by these -- the serial spine is
the common part of both scopes.

Measured numbers quoted in the assertions below were taken on 2026-09-24 on this
machine (CUDA); each says what it measured so a later reader can tell a drift from a
re-derivation.
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from e2e import frames                                                  # noqa: E402
from e2e.blocks import CircuitStage, RFFEBlock, device                  # noqa: E402
from e2e.chain.dechirp import DechirpBlock                              # noqa: E402
from e2e.chain.frontend import PLACEMENTS, FrontEndBlock                # noqa: E402
from e2e.chain.link_budget import (ThermalNoiseBlock, TxPowerStage,     # noqa: E402
                                   thermal_noise_power_w)
from e2e.chain.receive import (RANGE_CONVENTIONS, RangeTransformBlock,  # noqa: E402
                               range_axis_m)
from e2e.chain.waveform import (WAVEFORM_KINDS, JSACSignal,             # noqa: E402
                                OFDMSignal, WaveformBlock)
from e2e.circuit.rffe_model import circuit_model_batch, get_RX_config, noise_cascade
from e2e.radar_config import BENCHMARK_V1_KA, C_MPS, MUNICH_KA_FMCW     # noqa: E402

CFG = BENCHMARK_V1_KA


class _SingleTxCfg:
    mimo = "single"
    n_tx = 1


def _db(x):
    return 10.0 * math.log10(float(x) + 1e-300)


def _zero_adc(n_rx=8, n_chirp=1, n_samples=4096):
    return torch.zeros(n_rx, n_chirp, n_samples, dtype=torch.complex64, device=device)


# ==================================================== 1. the move onto beat samples

def _phase_equivariance_error(amp, theta=0.7231, n_rx=4, n_s=256, seed=0):
    """Relative error in `g(e^{i0} x) == e^{i0} g(x)` at drive level `amp`."""
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(n_rx, 1, n_s, generator=g)
         + 1j * torch.randn(n_rx, 1, n_s, generator=g)).to(torch.complex64).to(device)
    x = x * amp
    fe = FrontEndBlock(CFG, n=n_rx, physical_scale=True, inject_noise=False)
    a = fe.apply({"adc": x})["adc"]
    b = fe.apply({"adc": x * np.exp(1j * theta)})["adc"]
    rotated = a * np.exp(1j * theta)
    return float((b - rotated).abs().max() / rotated.abs().max())


def test_the_cascade_is_phase_equivariant_below_baseband_clipping():
    """THE identity the front-end move rests on -- AND the drive level at which it stops
    holding. This is the most important number in this file.

    The contract's licence (section 1.1 fact 3) is that for a unit-modulus chirp the
    dechirp commutes with a memoryless ENVELOPE nonlinearity: `g(|r|) r/|r|` dechirped
    is `g(|b|) b/|b|`. That is only true if the cascade is a function of the envelope
    ALONE, i.e. equivariant under a global phase rotation.

    The LNA and mixer stages are: both were deliberately written in baseband-equivalent
    envelope form (`rffe_model.py`, "Bandpass cubic nonlinearity, baseband-equivalent
    (envelope) form"). **The BASEBAND stage is not.** It clamps I and Q SEPARATELY
    (`torch.clamp(VBB_I, -Vbias_BB, Vbias_BB)` and the same for Q), which is a square
    region in the complex plane, not a circular one -- so once that clamp engages, the
    result depends on the signal's phase and the commutation is no longer exact.

    Measured 2026-09-24 (this frame, 4 elements x 256 samples, benchmark_v1_ka config,
    relative max error):

        drive 1e-6  ->  1.6e-7     (float32 rounding; structurally exact)
        drive 1e-4  ->  1.3e-5
        drive 1e-2  ->  1.4e-1     <- baseband clamp engaging
        drive >=3e-2 -> 4.7e-1     (saturated)

    CONSEQUENCE, stated rather than discovered later: the front-end move onto beat
    samples is exact only BELOW baseband clipping. The demo operating point is far
    below it (`signal_scaling=1e-7`, nothing clips -- STATE section 5), so the T1-T4
    screens are in the exact regime. A preset that drives the baseband stage into its
    clamp is NOT covered by the commutation argument and would need the RF-rate record
    the contract rules out on cost. This is a limitation of the model's square baseband
    clamp, not of the placement.
    """
    assert _phase_equivariance_error(1e-6) < 1e-6
    assert _phase_equivariance_error(1e-4) < 1e-4
    # and it genuinely breaks once the square baseband clamp engages
    assert _phase_equivariance_error(1e-1) > 0.1


def test_front_end_is_a_real_nonlinearity_not_a_gain():
    """The commutation licence would be vacuous if the block were linear. Drive it hard
    enough to reach the LNA's clamp-at-cubic-peak and show the gain compresses."""
    n_rx, n_s = 2, 64
    base = torch.ones(n_rx, 1, n_s, dtype=torch.complex64, device=device)
    fe = FrontEndBlock(CFG, n=n_rx, physical_scale=True, inject_noise=False)
    small = fe.apply({"adc": base * 1e-6})["adc"].abs().mean() / 1e-6
    large = fe.apply({"adc": base * 1e-1})["adc"].abs().mean() / 1e-1
    assert large < 0.5 * small, (
        f"no compression: small-signal gain {float(small):.3g}, "
        f"large-signal {float(large):.3g}")


def test_placement_impulse_refuses_and_names_the_block_that_implements_it():
    """`placement="impulse"` is a recorded FACT about a stored corpus, not an
    alternative this block implements -- the impulse placement runs BEFORE the dechirp
    and is `RFFEBlock` + `CircuitStage`. Refusing by name beats implementing a second
    copy of it here."""
    assert PLACEMENTS == ("beat", "impulse")
    with pytest.raises(ValueError, match="RFFEBlock"):
        FrontEndBlock(CFG, placement="impulse")
    with pytest.raises(ValueError, match="unknown front-end placement"):
        FrontEndBlock(CFG, placement="rf")


def test_front_end_refuses_to_guess_a_sample_rate():
    """The noise band is `min(if_bw, fs)`. Inventing `fs` is a silent multi-dB error in
    the floor, so the block raises instead."""
    fe = FrontEndBlock(None, n=2)
    with pytest.raises(ValueError, match="beat sample rate"):
        fe.apply({"adc": _zero_adc(n_rx=2, n_samples=32)})


# ================================================================ 2. noise, exactly once

def test_noise_floor_matches_the_analytic_friis_cascade():
    """The floor the front end injects equals `2 * NBB * B` from an independently
    written three-stage Friis cascade (`rffe_model.noise_cascade`), where
    `B = min(if_bw, fs)`.

    Measured 2026-09-24: -0.021 dB. The tolerance is the contract's 0.2 dB; the margin
    is the sample count (4096 samples x 8 elements).
    """
    n_rx, n_s = 8, 4096
    torch.manual_seed(0)
    fe = FrontEndBlock(CFG, n=n_rx, physical_scale=True, seed=3)
    out = fe.apply({"adc": _zero_adc(n_rx, 1, n_s)})["adc"]
    measured = float((out.abs() ** 2).mean())

    cascade = noise_cascade(get_RX_config(n_rx).to(device))
    band = min(float(get_RX_config(n_rx)[0, 6]), CFG.fs_hz)
    expected = 2.0 * float(cascade["n_out_psd"][0]) * band     # both quadratures

    assert abs(_db(measured) - _db(expected)) < 0.2, (
        f"front-end floor {_db(measured):.3f} dB vs analytic Friis "
        f"{_db(expected):.3f} dB")
    # And the cascade is the documented one: 24 dB voltage gain, ~1.9 dB noise figure.
    assert float(cascade["av_total"][0]) == pytest.approx(10 ** (24 / 20), rel=1e-3)
    assert 1.0 < _db(float(cascade["f_total"][0])) < 4.0


def test_noise_floor_is_proportional_to_the_noise_band():
    """Halving the IF bandwidth halves the noise power -- the OTHER half of the
    `if_bw_hz` knob. If the low-pass corner and the noise reference disagreed, the knob
    would be lying about one of them (contract section 3.1)."""
    n_rx, n_s = 8, 4096
    torch.manual_seed(1)
    hi = FrontEndBlock(CFG, n=n_rx, physical_scale=True, seed=5, if_bw_mhz=16.0)
    torch.manual_seed(1)
    lo = FrontEndBlock(CFG, n=n_rx, physical_scale=True, seed=5, if_bw_mhz=4.0)
    p_hi = float((hi.apply({"adc": _zero_adc(n_rx, 1, n_s)})["adc"].abs() ** 2).mean())
    p_lo = float((lo.apply({"adc": _zero_adc(n_rx, 1, n_s)})["adc"].abs() ** 2).mean())
    # 4x the band -> +6.02 dB, before the cascade's own weak dependence on the column.
    assert _db(p_hi) - _db(p_lo) == pytest.approx(6.02, abs=0.25)


def test_noise_band_is_capped_at_the_sample_rate():
    """`min(if_bw, fs)`, not `if_bw`. Beyond Nyquist there is no band to integrate, and
    referencing to the un-capped IF bandwidth is how a 50 MHz knob on a 25 MHz record
    manufactures noise that the converter never saw (contract section 3.1's 14.0 dB vs
    the 17.0 dB the old oracle quoted)."""
    n_rx = 4
    fe = FrontEndBlock(CFG, n=n_rx, if_bw_mhz=200.0, fs_hz=25e6)
    band = fe.noise_band_hz(fe.rx_config(n_rx, device))
    assert float(band.max()) == pytest.approx(25e6)


def test_the_floor_is_injected_once_when_a_front_end_is_present():
    """mode="once": the front end owns the injection, `ThermalNoiseBlock` records the
    budget and adds nothing. The legacy mode adds a second floor, which is F81's
    double-count and is kept ONLY because every stored corpus was made with it."""
    n_rx, n_s = 8, 2048
    torch.manual_seed(2)
    fe_out = FrontEndBlock(CFG, n=n_rx, physical_scale=True, seed=7).apply(
        {"adc": _zero_adc(n_rx, 1, n_s)})
    state = {"adc": fe_out["adc"], "noise_injected_by": fe_out["noise_injected_by"]}

    once = ThermalNoiseBlock(CFG, mode="once").apply(dict(state))
    assert once["link_budget"]["injected_here"] is False
    assert torch.equal(once["adc"], state["adc"])          # nothing added, bit-for-bit
    assert once["link_budget"]["injected_by"] == "frontend"

    legacy = ThermalNoiseBlock(CFG, mode="legacy").apply(dict(state))
    assert legacy["link_budget"]["injected_here"] is True
    assert not torch.equal(legacy["adc"], state["adc"])

    # With no front end, mode="once" IS the injection -- the floor does not vanish.
    alone = ThermalNoiseBlock(CFG, mode="once").apply({"adc": _zero_adc(n_rx, 1, n_s)})
    assert alone["link_budget"]["injected_here"] is True
    assert float((alone["adc"].abs() ** 2).mean()) > 0.0


def test_noise_once_floor_does_not_track_transmit_power():
    """F81's first defect, as a passing test on the NEW composition.

    The old chain multiplied by `sqrt(P_tx)` DOWNSTREAM of the front end, so transmit
    power scaled the receiver's own 4kTR noise with it: +24 dB of P_tx moved the
    noise-only floor +20.8 dB (measured 2026-08-31), and the corresponding assertion in
    tests/test_ml_link_budget.py is still a strict xfail because it runs through
    `e2e/ml/chain_generate.py`, which composes the legacy chain.

    Here the scaling is at the SOURCE (`TxPowerStage`) and the floor is injected once,
    so a zeroed channel's floor cannot move at all.
    """
    import dataclasses
    n_rx, n_s = 8, 2048
    zero_cfr = torch.zeros(n_rx, 1, 1, n_s, dtype=torch.complex64, device=device)

    def floor_db(tx_power_dbm):
        cfg = dataclasses.replace(CFG, tx_power_dbm=float(tx_power_dbm))
        state = {"s_pars": zero_cfr}
        state.update(TxPowerStage(cfg).apply(state))
        state.update(DechirpBlock(_SingleTxCfg()).apply(state))
        torch.manual_seed(11)
        state.update(FrontEndBlock(cfg, n=n_rx, physical_scale=True, seed=9).apply(state))
        state.update(ThermalNoiseBlock(cfg, mode="once").apply(state))
        return _db(float((state["adc"].abs() ** 2).mean()))

    lo, hi = floor_db(0.0), floor_db(24.0)
    assert abs(hi - lo) < 0.5, (
        f"noise-only floor moved {hi - lo:+.2f} dB for +24 dB of transmit power")


def test_noise_once_signal_does_track_transmit_power():
    """The other half, or the test above would pass for a chain that ignores P_tx."""
    import dataclasses
    n_rx, n_s = 4, 256
    torch.manual_seed(3)
    cfr = (torch.randn(n_rx, 1, 1, n_s) + 1j * torch.randn(n_rx, 1, 1, n_s)).to(
        torch.complex64).to(device) * 1e-5

    def power_db(tx_power_dbm):
        cfg = dataclasses.replace(CFG, tx_power_dbm=float(tx_power_dbm))
        state = {"s_pars": cfr}
        state.update(TxPowerStage(cfg).apply(state))
        return _db(float((state["s_pars"].abs() ** 2).mean()))

    assert power_db(24.0) - power_db(0.0) == pytest.approx(24.0, abs=0.1)


def test_noise_figure_moves_the_floor_when_the_front_end_is_absent():
    """F81's second defect, as a passing test on the NEW composition: with no front end
    the `noise_figure_db` dial IS the floor, so +20 dB of NF moves it +20 dB. The old
    chain added kTBF at the front end's OUTPUT, where it sat below the RFFE's own noise
    and the dial moved the floor ~0.06 dB over the same sweep."""
    import dataclasses
    n_rx, n_s = 8, 2048

    def floor_db(nf_db):
        cfg = dataclasses.replace(CFG, noise_figure_db=float(nf_db))
        out = ThermalNoiseBlock(cfg, mode="once").apply({"adc": _zero_adc(n_rx, 1, n_s)})
        return _db(float((out["adc"].abs() ** 2).mean()))

    assert floor_db(25.0) - floor_db(5.0) == pytest.approx(20.0, abs=0.5)


def test_legacy_circuit_model_call_is_bit_identical():
    """The corpus freeze. `circuit_model_batch` grew two noise-placement kwargs; with
    them left at their defaults it must be bit-for-bit what it was, because that is the
    call every stored corpus was generated through."""
    torch.manual_seed(4)
    n, nt = 6, 128
    sig = (torch.randn(n, 1, 1, nt) + 1j * torch.randn(n, 1, 1, nt)).to(
        torch.complex64).to(device) * 1e-5
    cfg_table = get_RX_config(n).to(device)
    g1 = torch.Generator(device=device); g1.manual_seed(123)
    g2 = torch.Generator(device=device); g2.manual_seed(123)
    a, _ = circuit_model_batch(cfg_table, sig, 3e9, generator=g1)
    b, _ = circuit_model_batch(cfg_table, sig, 3e9, generator=g2,
                               noise_band_hz=None, noise_divisor=None)
    assert torch.equal(a, b)


def test_thermal_block_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="unknown ThermalNoiseBlock mode"):
        ThermalNoiseBlock(CFG, mode="twice")


# ============================================ 3. the bistatic range convention (Q2 = B)

def test_bistatic_path_is_the_default_convention():
    """The owner's 2026-09-24 answer: "if bistatic, math should be correct". The munich
    frames are a TX -> RX-array link traced with `normalize_delays=True`, so a bin's
    delay is an EXCESS delay and `c*tau` is exactly what it measures."""
    assert RANGE_CONVENTIONS[0] == "bistatic_path"
    assert RangeTransformBlock().convention == "bistatic_path"
    axis = range_axis_m(4, 1e6, 8)
    assert axis is not None
    assert float(axis[1]) == pytest.approx(C_MPS / (8 * 1e6))     # c*tau, not c*tau/2


@pytest.mark.parametrize("convention,scale", [("bistatic_path", 1.0),
                                              ("monostatic_c2", 0.5)])
def test_known_delay_oracle_under_the_new_default(convention, scale):
    """A one-tap CFR at delay tau lands at bin round(tau*N*df) and reads `scale*c*tau`.
    The DEFAULT arm of this is the `c*tau` one."""
    n, df, k = 512, 749.5e6 / 512.0, 91
    tau = k / (n * df)
    ramp = torch.arange(n, dtype=torch.float64, device=device)
    h = torch.exp(-2j * np.pi * ramp * df * tau).to(torch.complex64).view(1, 1, 1, n)
    adc = DechirpBlock(_SingleTxCfg()).apply({"s_pars": h})["adc"]
    out = RangeTransformBlock(window="none", dc_removal=False, crop_negative_delay=False,
                              delta_f_hz=df, convention=convention).apply({"adc": adc})
    assert int(torch.argmax(out["cube"][0, 0].abs())) == k
    assert float(out["range_axis"][k]) == pytest.approx(scale * C_MPS * tau, rel=1e-9)


def test_the_munich_plan_in_both_conventions():
    """The two numbers every T1-T4 card inherits, computed from the stored grid rather
    than typed: 9.99 cm per bin over a 499.55 m window (bistatic, the default), or
    5.00 cm over 249.78 m (equivalent monostatic).

    NOTE these are NOT the 10 cm / 500 m and 5 cm / 250 m the contract's prose quotes:
    those use a nominal `B = 3 GHz`, and the stored grid is endpoint-inclusive, so its
    spacing is `B/4999`. The 0.45 m is the off-by-one, and it is why the axis is derived
    from `freq_plan` and never from a literal.
    """
    from e2e.chain.receive import delta_f_from_freq_plan
    df = delta_f_from_freq_plan({"start_hz": 28.5e9, "stop_hz": 31.5e9, "num_freqs": 5000})
    bi = range_axis_m(5000, df, 5000, "bistatic_path")
    mono = range_axis_m(5000, df, 5000, "monostatic_c2")
    assert float(bi[1]) == pytest.approx(0.0999, abs=1e-4)
    assert float(bi[1]) * 5000 == pytest.approx(499.55, abs=0.05)
    assert float(mono[1]) == pytest.approx(0.04995, abs=1e-5)
    assert float(mono[1]) * 5000 == pytest.approx(249.78, abs=0.05)
    assert delta_f_from_freq_plan(
        {"start_hz": 28.5e9, "stop_hz": 31.5e9, "num_freqs": 5000}) == pytest.approx(
            MUNICH_KA_FMCW.ramp_slope_hzps / MUNICH_KA_FMCW.fs_hz, rel=1e-9)


# ================================================ 4. the waveform classes (the branch)

def test_the_registry_holds_the_three_kinds():
    assert WAVEFORM_KINDS == ("fmcw", "ofdm", "jsac")


def test_fmcw_is_the_default_and_is_constant_envelope():
    """`constant_envelope` is load-bearing: it is the precondition that licenses the
    front end on beat samples, and the reason the TX PA is inert here."""
    out = WaveformBlock(bw=1e9, sample_rate=3e9, n_t=256).apply({})
    assert out["waveform_kind"] == "fmcw"
    assert out["waveform"]["constant_envelope"] is True
    assert out["waveform"]["cube_axes"] == frames.CUBE_AXES_FMCW
    env = out["tx_wave"].abs()
    assert float(env.max() / env.min()) == pytest.approx(1.0, abs=1e-4)


def test_ofdm_declares_subcarrier_axes_and_a_real_papr():
    """OFDM's cube fast axis counts SUBCARRIERS until the range transform runs, and its
    envelope is not constant -- the two facts a product and a PA respectively need."""
    out = WaveformBlock(kind="ofdm", bw=1e9, sample_rate=3e9, n_t=80,
                        n_subcarriers=64, cp_len=16).apply({})
    assert out["waveform_kind"] == "ofdm"
    assert out["waveform"]["cube_axes"] == frames.CUBE_AXES_OFDM
    assert out["waveform"]["constant_envelope"] is False
    assert out["waveform"]["n_fast"] == 64
    assert out["waveform"]["cp_len"] == 16
    p = out["tx_wave"].abs() ** 2
    papr_db = _db(float(p.max() / p.mean()))
    assert papr_db > 3.0, f"an OFDM symbol with PAPR {papr_db:.1f} dB is not OFDM"


def test_ofdm_keeps_the_transmitted_grid_the_receiver_divides_by():
    """`H_est = Y/X` needs the actual `X`. A receiver that re-guesses it is a different
    receiver, so the waveform hands the grid on rather than regenerating it.

    RENAMED 2026-09-24: the half-built `OFDMISACSignal` became two classes, `OFDMSignal`
    (comms, no mixing block) and `JSACSignal` (the hybrid, symbol-division mixing), over
    one shared `OFDMFrame` -- see `e2e/comms/ofdm_isac.py`. The property under test is
    unchanged and now reads off the frame, which is the object the mixer and the
    demapper both hold.
    """
    sig = OFDMSignal({"n_subcarriers": 64, "subcarrier_spacing_hz": 1e6, "cp_len": 16,
                      "seed": 1, "n_symbols": 1})
    t = torch.arange(80, dtype=torch.float32) / 3e9
    sig.generate(t)
    assert sig.frame().tx_grid.shape == (1, 64)
    # n_symbols=1 is the all-pilot preamble alone: no data symbols, hence no bits.
    assert sig.frame().tx_bits.numel() == 0

    two = OFDMSignal({"n_subcarriers": 64, "subcarrier_spacing_hz": 1e6, "cp_len": 16,
                      "seed": 1, "n_symbols": 2, "pilot_spacing": 8})
    assert two.frame().tx_bits.numel() == two.frame()._data.data_bits_per_symbol_block


def test_jsac_is_implemented_and_differs_from_ofdm_only_in_its_mixing_mode():
    """REPLACES `test_jsac_is_registered_but_refuses_...` (2026-09-24): the class was a
    registered-and-refusing placeholder and is now implemented, so the assertion moves
    from "refuses with the contract in the message" to the property that contract
    described.

    The two OFDM-grid classes transmit the SAME frame and differ in exactly one
    attribute -- `mixing` -- which decides whether a mixing block runs and therefore
    which products the chain can read out. That one line is the whole distinction
    between "comms" and "hybrid", and pinning it here is what stops `jsac` from
    quietly becoming `ofdm` with a second tab.
    """
    assert "jsac" in WAVEFORM_KINDS
    md = {"n_subcarriers": 64, "subcarrier_spacing_hz": 1e6, "cp_len": 16,
          "seed": 1, "n_symbols": 2, "pilot_spacing": 8}
    ofdm, jsac = OFDMSignal(dict(md)), JSACSignal(dict(md))
    assert (ofdm.mixing, jsac.mixing) == (None, "symbol_division")
    assert torch.equal(ofdm.frame().tx_grid, jsac.frame().tx_grid)
    assert jsac.record_metadata()["kind"] == "jsac"
    assert jsac.record_metadata()["cube_axes"] == frames.CUBE_AXES_OFDM


def test_an_unknown_waveform_kind_still_names_the_registry():
    with pytest.raises(ValueError, match="unknown waveform kind"):
        WaveformBlock(kind="otfs")
    with pytest.raises(ValueError, match="narrowband"):
        WaveformBlock(kind="narrowband")


# ============================================ 5. the front end, composed on the spine

def test_the_full_spine_runs_front_end_after_the_dechirp(make_env_block):
    """The FULL block order: source -> TxPower -> dechirp -> front end -> floor ->
    range transform. The front end sits AFTER the dechirp, which is the whole change."""
    from e2e.simulation import Simulation
    from e2e.blocks import RangeProfileBlock

    env = make_env_block(n_frames=2, n_freqs=64)
    stages = [
        TxPowerStage(CFG),
        DechirpBlock(_SingleTxCfg()),
        FrontEndBlock(CFG, n=1024, physical_scale=True, seed=2),
        ThermalNoiseBlock(CFG, mode="once"),
        RangeTransformBlock(window="none", dc_removal=False, delta_f_hz=1e6),
    ]
    sim = Simulation(env, [RangeProfileBlock(bins=8)], 2, serial_stages=stages)
    out = sim.run(n_steps=2)
    assert len(out["range_profile"]) == 2
    assert torch.all(torch.isfinite(out["range_profile"][0]))


def test_the_two_placements_agree_on_the_noise_floor_LEVEL():
    """The contract's own prediction (section 1.1 fact 2), checked: the noise-bandwidth
    reference is `NBB*BW_IF` per frequency bin in the legacy placement and
    `NBB*min(if_bw, fs)` per sample in the beat placement -- THE SAME NUMBER whenever
    `if_bw <= fs`, which every shipped preset satisfies (15 MHz IF, 25 MHz fs).

    The legacy path gets there by a two-seam route: it injects `NBB*BW/nt` per time
    sample and the caller's UNNORMALISED forward FFT multiplies the variance by `nt`.
    The beat path injects `NBB*B` per sample with no FFT at all. That they land on the
    same floor is the whole reason the front end could move without re-measuring T1.

    Measured 2026-09-24 on a zeroed frame (16 elements, 512 samples,
    benchmark_v1_ka): the two floors agree to 8.1e-7 dB.
    """
    n, n_s = 16, 512
    torch.manual_seed(0)
    legacy = RFFEBlock(n=n, seed=5, physical_scale=True).apply_circuit(
        torch.zeros(n, 1, 1, n_s, dtype=torch.complex64, device=device))[0]
    torch.manual_seed(0)
    beat = FrontEndBlock(CFG, n=n, seed=5, physical_scale=True).apply(
        {"adc": _zero_adc(n, 1, n_s)})["adc"]
    p_legacy = float((legacy.abs() ** 2).mean())
    p_beat = float((beat.abs() ** 2).mean())
    assert abs(_db(p_beat) - _db(p_legacy)) < 0.05, (
        f"floors disagree: legacy {_db(p_legacy):.4f} dB, beat {_db(p_beat):.4f} dB")


def test_inject_noise_false_is_a_true_bypass_on_both_blocks():
    """Both front ends can run the nonlinearity with NO thermal draw. This is what the
    placement-parity oracle needs: with the floor in, it would report a difference of
    noise REALIZATIONS as a difference of placements."""
    n, n_s = 4, 128
    quiet_old = RFFEBlock(n=n, seed=1, physical_scale=True, inject_noise=False)
    a = quiet_old.apply_circuit(torch.zeros(n, 1, 1, n_s, dtype=torch.complex64,
                                            device=device))[0]
    assert float(a.abs().max()) == 0.0
    quiet_new = FrontEndBlock(CFG, n=n, seed=1, physical_scale=True, inject_noise=False)
    out = quiet_new.apply({"adc": _zero_adc(n, 1, n_s)})
    assert float(out["adc"].abs().max()) == 0.0
    assert "noise_injected_by" not in out      # a downstream floor is still the injection


def test_the_front_end_on_beat_samples_differs_from_the_impulse_placement(make_env_block):
    """The move is not cosmetic. Same knobs, same seed, same frame: the two placements
    produce different cubes, because the nonlinearity and the normalisation see
    different signals (contract section 1.1 fact 2 -- ~10*log10(N) of peak drive)."""
    env = make_env_block(n_frames=1, n_freqs=128)
    s_pars = env.get_S_pars()

    beat = DechirpBlock(_SingleTxCfg()).apply({"s_pars": s_pars})["adc"]
    torch.manual_seed(0)
    new = FrontEndBlock(CFG, n=1024, physical_scale=False, seed=1,
                        fs_hz=3e9).apply({"adc": beat})["adc"]

    torch.manual_seed(0)
    old_cfr = CircuitStage(RFFEBlock(n=1024, seed=1)).apply({"s_pars": s_pars})["s_pars"]
    old = DechirpBlock(_SingleTxCfg()).apply({"s_pars": old_cfr})["adc"]

    assert new.shape == old.shape
    assert not torch.allclose(new, old, rtol=1e-2, atol=1e-12)


_KA_CORPUS = ("e2e/ml/datasets/b1_demo_cfr_ka/benchmark_v1_ka_D2/"
              "benchmark_v1_ka_D2")


@pytest.mark.skipif(not __import__("os").path.isdir(_KA_CORPUS),
                    reason="needs the generated b1_demo_cfr_ka corpus")
def test_placement_parity_on_the_ka_corpus_is_below_one_lsb():
    """THE decision measurement the FULL contract turns on (section 5.4): does moving
    the front end onto beat samples change a stored corpus's cube by more than one ADC
    LSB at 12 bits?

    Measured 2026-09-24 on 5 `b1_demo_cfr_ka` frames, thermal noise OFF on both arms
    (see `test_inject_noise_false_is_a_true_bypass_on_both_blocks` for why -- with the
    floor in, two independent draws of the SAME-variance noise differ by ~sqrt(2)x the
    floor and the measurement would report that as a placement difference):

        mean rel-RMSE   8.6e-05
        one LSB / |cube| rms   1.76e-04
        -> BELOW one LSB, by a factor of ~2.

    So the SIGNAL path is corpus-safe: the nonlinearity and the normalisation reference
    land within the converter's own resolution. Combined with
    `test_the_two_placements_agree_on_the_noise_floor_LEVEL` (floors identical to
    8.1e-7 dB), the placement move changes NO measurable quantity.

    It still does not survive the live-vs-stored gates, and the distinction matters:
    those read max |diff| = 0 CODES, and a differently-ordered RNG consumption changes
    the noise realization, which fails a zero tolerance at any floor. The legacy flag
    therefore stays for BIT parity while being unnecessary for NUMERICAL fidelity.
    """
    import glob
    import json
    import os

    from e2e.chain.receive import QuantizerBlock
    from e2e.radar_config import RadarConfig

    with open(os.path.join(_KA_CORPUS, "manifest.json")) as fh:
        man = json.load(fh)
    cfg_d = man.get("config") or man.get("cfg") or man.get("radar_config")
    cfg = RadarConfig.from_dict(cfg_d)
    files = sorted(glob.glob(os.path.join(_KA_CORPUS, "*.cfr.npy")))[:5]
    assert files, "corpus directory has no .cfr.npy sidecars"

    rt = RangeTransformBlock(cfg, window="hann", dc_removal=True,
                             crop_negative_delay=False)
    rels, lsbs = [], []
    for i, path in enumerate(files):
        cfr = torch.from_numpy(np.load(path)).to(device)
        if cfr.ndim == 3:
            cfr = cfr.unsqueeze(1)
        n_rx = cfr.shape[0]

        torch.manual_seed(1000 + i)
        a = CircuitStage(RFFEBlock(n=n_rx, seed=7 + i, physical_scale=True,
                                   inject_noise=False)).apply({"s_pars": cfr})
        a.update(DechirpBlock(cfg).apply(a))
        qa = QuantizerBlock(bits=12)
        a.update(qa.apply(a))
        cube_a = rt.apply(a)["cube"]

        torch.manual_seed(1000 + i)
        b = DechirpBlock(cfg).apply({"s_pars": cfr})
        b.update(FrontEndBlock(cfg, n=n_rx, seed=7 + i, physical_scale=True,
                               inject_noise=False).apply(b))
        b.update(QuantizerBlock(bits=12).apply(b))
        cube_b = rt.apply(b)["cube"]

        ref = float(torch.sqrt((cube_a.abs() ** 2).mean()))
        rels.append(float(torch.sqrt(((cube_b - cube_a).abs() ** 2).mean())) / ref)
        lsbs.append(float(qa.lsb) / ref)

    rel = float(np.mean(rels))
    lsb = float(np.mean(lsbs))
    assert rel < lsb, (
        f"placement parity {rel:.3e} rel-RMSE exceeds one LSB ({lsb:.3e}); the legacy "
        f"placement flag is load-bearing for numerical fidelity, not just bit parity")
