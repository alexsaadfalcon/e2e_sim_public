"""The COMPOSITION oracle for the link budget: does the assembled ML chain actually
respond to the radar equation's knobs?

`e2e/chain/link_budget.py` has cited this file since it was written -- "compares it
against what the chain actually produces; the two agreeing is the oracle for this whole
module" -- but **the file did not exist** until 2026-08-28. `tests/test_link_budget.py`
does exist and is good, but every test in it is an isolated unit test of a formula; none
of them constructs a `build_chain_simulation`. So the module claimed a validating
instrument it never had, and the defect pinned below lived through every review.

That is the point of this file. `expected_target_snr_db` being right on paper says
nothing about whether the composed chain honours it, and the two questions had never
been connected by a test.

All tests here are Sionna-free: `build_chain_simulation` accepts a stand-in environment
block, which is how the rest of the suite exercises the composition.
"""
from __future__ import annotations

import dataclasses
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_ml_chain_generate import _CFG, _FakeRTEnvironment  # noqa: E402

from e2e.blocks import RFFEBlock  # noqa: E402
from e2e.chain.frontend import FrontEndBlock  # noqa: E402
from e2e.ml import chain_generate  # noqa: E402
from e2e.ml.labels import LabelGrid  # noqa: E402
from e2e.simulation import CircuitStage  # noqa: E402


def _cube_power_db(tx_power_dbm, *, use_rffe, physical_scale=None, device=None,
                   use_link_budget=True, composition="full"):
    """Mean power of the radar cube out of the fully composed chain, in dB."""
    cfg = dataclasses.replace(_CFG, tx_power_dbm=float(tx_power_dbm))
    grid = LabelGrid.for_config(cfg, range_stride=1, n_azimuth=8)
    env = _FakeRTEnvironment(cfg, grid, n_frames=1, device=device, seed=0)
    rffe_kwargs = None if physical_scale is None else {"physical_scale": physical_scale}
    with tempfile.TemporaryDirectory() as out_dir:
        sim = chain_generate.build_chain_simulation(
            scenario=None, cfg=cfg, out_dir=out_dir, environment_block=env,
            use_rffe=use_rffe, rffe_kwargs=rffe_kwargs, device=device,
            use_link_budget=use_link_budget, composition=composition,
        )
        sim.run(n_steps=1)
        cube = sim.get_outputs()["radar_cube"][0]
    x = cube.detach().cpu().numpy() if hasattr(cube, "detach") else np.asarray(cube)
    return 10.0 * np.log10(float(np.mean(np.abs(x) ** 2)) + 1e-300)


#: Transmit powers (dBm) at which the FULL composition's front end is LINEAR on this
#: fixture's channel. MEASURED 2026-09-24 on `_CFG` + `_FakeRTEnvironment` (see
#: `test_transmit_power_drives_the_front_end_into_compression` for the curve): the cube
#: tracks P_tx to 0.05 dB from -60 to -24 dBm, departs by ~0.7 dB at -12, and is hard
#: against the cascade's clamp by 0 dBm. The pair below is 24 dB apart inside that
#: linear span. NOT a property of the radar -- a property of THIS fixture's amplitude
#: and this front end's drive level; a different channel compresses somewhere else.
_LINEAR_TX_DBM = (-48.0, -24.0)


@pytest.mark.parametrize("use_rffe,physical_scale", [
    (False, None),        # link budget alone, no front end
    (True, True),         # front end on, absolute scale preserved
])
def test_composed_chain_scales_with_transmit_power(use_rffe, physical_scale, torch_device):
    """A 24 dB rise in transmit power must move the composed chain's cube by 24 dB.

    This is the connection `expected_target_snr_db`'s docstring promises and that
    nothing made: the formula is checkable on paper, but only running a frame through
    `build_chain_simulation` says whether the assembled chain honours it.

    OPERATING POINT MOVED 2026-09-24, with the reason, because the physics changed
    under it. This test used to run at 0 -> 24 dBm. Under the v1.0 order that was
    safe for an accidental reason: `sqrt(P_tx)` was applied inside `ThermalNoiseBlock`,
    i.e. AFTER the RF front end, so transmit power could never reach the front end's
    nonlinearity and the cube scaled linearly however hard you transmitted. The FULL
    contract (section 1.4) moves `sqrt(P_tx)` to the SOURCE -- which is the whole point
    of F81 -- and the front end now sees it. At 0 dBm on this fixture the cascade is in
    compression, and 0 -> 24 dBm moves the cube +3.2 dB, not +24. That is the front end
    working, not the link budget failing, and the two are separated by measuring where
    the front end is linear (`_LINEAR_TX_DBM`) and pinning the compression separately
    below.
    """
    lo_dbm, hi_dbm = _LINEAR_TX_DBM
    lo = _cube_power_db(lo_dbm, use_rffe=use_rffe, physical_scale=physical_scale,
                        device=torch_device)
    hi = _cube_power_db(hi_dbm, use_rffe=use_rffe, physical_scale=physical_scale,
                        device=torch_device)
    assert hi - lo == pytest.approx(hi_dbm - lo_dbm, abs=0.5), (
        f"tx_power_dbm {lo_dbm}->{hi_dbm} moved the cube {hi - lo:+.2f} dB, expected "
        f"{hi_dbm - lo_dbm:+.2f}"
    )


def test_transmit_power_drives_the_front_end_into_compression(torch_device):
    """The consequence of moving `sqrt(P_tx)` to the source, pinned so it is a
    DECISION rather than a surprise on a demo screen.

    Under the v1.0 order the transmit power could not reach the receiver's
    nonlinearity (it was applied downstream of it), so "more transmit power" was
    always exactly "more cube", without limit. Under the FULL contract it is applied at
    the source and the LNA/mixer/baseband cascade sees it, so past some drive the cube
    stops tracking -- which is what a real receiver does and what the impairment
    ceiling (`ESTABLISHED_FACTS.md` F35/F62) exists to talk about.

    MEASURED 2026-09-24 on this fixture, `use_rffe=True`, `physical_scale=True`, mean
    cube power in dB: -60 -> -53.09, -48 -> -41.05, -36 -> -29.03, -24 -> -17.04,
    -12 -> -5.28, 0 -> +4.01, +12 -> +6.96, +24 -> +7.24. Linear to 0.05 dB below
    -24 dBm; 16.8 dB of gain lost over the last 24 dB of drive. The numbers are this
    fixture's; the SHAPE is the claim, and it is what is asserted.
    """
    lo_dbm, hi_dbm = _LINEAR_TX_DBM
    linear = (_cube_power_db(hi_dbm, use_rffe=True, physical_scale=True,
                             device=torch_device)
              - _cube_power_db(lo_dbm, use_rffe=True, physical_scale=True,
                               device=torch_device))
    compressed = (_cube_power_db(24.0, use_rffe=True, physical_scale=True,
                                 device=torch_device)
                  - _cube_power_db(0.0, use_rffe=True, physical_scale=True,
                                   device=torch_device))
    assert linear == pytest.approx(24.0, abs=0.5)
    assert compressed < linear - 10.0, (
        f"the front end did not compress: +24 dB of drive from 0 dBm moved the cube "
        f"{compressed:+.2f} dB, the same {linear:+.2f} dB it moves in the linear "
        f"regime -- so `sqrt(P_tx)` is not reaching the nonlinearity, which is the "
        f"v1.0 placement this composition moved away from"
    )


def test_ml_corpus_composition_keeps_the_absolute_scale(torch_device):
    """INVERTED 2026-08-29, as the previous version's docstring instructed.

    Until then this test pinned the DEFECT (F62.2/F63): `RFFEBlock` defaults
    `physical_scale=False`, on which path it runs `frame * signal_scaling /
    mean(abs(frame))` -- a per-frame normalisation to a unit-less constant --
    and `build_chain_simulation` set only `n`, so the ML corpus generator took that
    default and then appended `ThermalNoiseBlock` AFTER it. The chain installed an
    absolute kTBF floor beneath a cube whose absolute level had already been erased, so
    target SNR stopped tracking transmit power, noise figure and range.

    Note what the defect did NOT do: it did not stop the cube scaling. It scaled
    uniformly, which is why the total-power test above passed even then. That is why the
    guard has to be structural rather than a power check.

    `build_chain_simulation` now sets `physical_scale=True`. An explicit override is
    still honoured, and `test_thermal_noise_refuses_a_normalised_cube` covers what
    happens if someone takes it.

    UPDATED 2026-09-24: under the FULL composition the front end is a `FrontEndBlock`
    on the beat record rather than a `CircuitStage(RFFEBlock)` on `ifft(CFR)`. The flag
    it carries is the same flag, translated across by `FrontEndBlock.from_rffe`, and
    the property under test is unchanged -- which is why the legacy arm is checked in
    the same test rather than in a copy of it.
    """
    grid = LabelGrid.for_config(_CFG, range_stride=1, n_azimuth=8)
    env = _FakeRTEnvironment(_CFG, grid, n_frames=1, device=torch_device, seed=0)
    with tempfile.TemporaryDirectory() as out_dir:
        sim = chain_generate.build_chain_simulation(
            scenario=None, cfg=_CFG, out_dir=out_dir, environment_block=env,
        )
        legacy = chain_generate.build_chain_simulation(
            scenario=None, cfg=_CFG, out_dir=out_dir, environment_block=env,
            composition="legacy_impulse",
        )
    front = next(s for s in sim.serial_stages if isinstance(s, FrontEndBlock))
    assert front.physical_scale is True, (
        "the ML corpus composition must keep the absolute amplitude scale -- an "
        "absolute thermal floor beneath a normalised cube is F63"
    )
    stage = next(s for s in legacy.serial_stages if isinstance(s, CircuitStage))
    assert isinstance(stage.rffe_block, RFFEBlock)
    assert stage.rffe_block.physical_scale is True


def test_thermal_noise_refuses_a_normalised_cube(torch_device):
    """The invalid composition must FAIL, not quietly produce a plausible corpus.

    Overriding `physical_scale=False` while the link budget is on rebuilds F63 exactly.
    A silent result here is worse than a crash: the corpus looks fine, trains fine, and
    every impairment dB on it is referenced to a floor that means nothing. The guard is
    on `ThermalNoiseBlock` rather than at assembly so it cannot be bypassed by any other
    route to the same chain.
    """
    grid = LabelGrid.for_config(_CFG, range_stride=1, n_azimuth=8)
    env = _FakeRTEnvironment(_CFG, grid, n_frames=1, device=torch_device, seed=0)
    with tempfile.TemporaryDirectory() as out_dir:
        sim = chain_generate.build_chain_simulation(
            scenario=None, cfg=_CFG, out_dir=out_dir, environment_block=env,
            rffe_kwargs={"physical_scale": False},
        )
        with pytest.raises(ValueError, match="F63"):
            sim.run(n_steps=1)


def test_physical_scale_is_reachable_and_changes_the_output(torch_device):
    """The fix is a one-line default change, so prove the flag is live, not vestigial.

    Measured with the link budget OFF. With it on, `physical_scale=False` is now a
    refused composition (see `test_thermal_noise_refuses_a_normalised_cube`), so the
    comparison has to be made on the one chain where both settings are still legal --
    which is enough, because what is under test is the RFFE flag, not the floor.
    """
    a = _cube_power_db(12.0, use_rffe=True, physical_scale=False, device=torch_device,
                       use_link_budget=False)
    b = _cube_power_db(12.0, use_rffe=True, physical_scale=True, device=torch_device,
                       use_link_budget=False)
    assert abs(b - a) > 1.0, (
        f"physical_scale made no difference ({a:.2f} vs {b:.2f} dB) -- it is supposed "
        f"to be the switch between an arbitrary and an absolute amplitude scale."
    )


def _noise_floor_db(tx_power_dbm=12.0, noise_figure_db=None, *, use_rffe=True,
                    device=None, composition="full"):
    """Mean power of the cube with the CHANNEL ZEROED -- i.e. instrument noise alone."""
    cfg = dataclasses.replace(_CFG, tx_power_dbm=float(tx_power_dbm))
    if noise_figure_db is not None:
        cfg = dataclasses.replace(cfg, noise_figure_db=float(noise_figure_db))
    grid = LabelGrid.for_config(cfg, range_stride=1, n_azimuth=8)
    env = _FakeRTEnvironment(cfg, grid, n_frames=1, device=device, seed=0)
    inner = env.get_S_pars

    def zeroed(*a, **k):
        out = inner(*a, **k)
        if isinstance(out, tuple):
            return (torch.zeros_like(out[0]),) + tuple(out[1:])
        return torch.zeros_like(out)

    env.get_S_pars = zeroed
    with tempfile.TemporaryDirectory() as out_dir:
        sim = chain_generate.build_chain_simulation(
            scenario=None, cfg=cfg, out_dir=out_dir, environment_block=env,
            use_rffe=use_rffe, device=device, composition=composition)
        sim.run(n_steps=1)
        cube = sim.get_outputs()["radar_cube"][0]
    x = cube.detach().cpu().numpy()
    return 10.0 * np.log10(float(np.mean(np.abs(x) ** 2)) + 1e-300)


def test_link_budget_alone_gives_a_transmit_power_independent_noise_floor(torch_device):
    """With no front end, the floor is the kTBF floor and does not move with P_tx."""
    lo = _noise_floor_db(0.0, use_rffe=False, device=torch_device)
    hi = _noise_floor_db(24.0, use_rffe=False, device=torch_device)
    assert abs(hi - lo) < 0.5, (
        f"link-budget-only noise floor moved {hi - lo:+.2f} dB for +24 dB of transmit "
        f"power; receiver noise cannot depend on how hard you transmit")


def test_front_end_noise_floor_does_not_track_transmit_power(torch_device):
    """The claim F63 was landed to deliver, stated as a test that currently fails.

    `tests/test_composed_chain_scales_with_transmit_power` passes today and does NOT
    catch this: it measures TOTAL cube power, which scales uniformly whether the scaling
    is applied at the right node or the wrong one. The inverted defect test in this file
    says as much in its own docstring -- "it did not stop the cube scaling ... that is why
    the guard has to be structural" -- and I shipped a power test anyway. Zeroing the
    channel is what separates the two.

    XFAIL REMOVED 2026-09-24: it passes. WHAT FIXED IT (one-chain contract section
    1.4): `sqrt(P_tx)` moved out of `ThermalNoiseBlock` into `TxPowerStage` at the
    SOURCE, so it multiplies the transmitted signal and nothing else; and the
    receiver's thermal draw happens exactly once, inside the front end, at the antenna
    reference. MEASURED on this fixture 2026-09-24: 0 dBm -> -75.18 dB, 24 dBm ->
    -75.18 dB, delta 0.00 dB. The v1.0 order on the same fixture in the same run:
    -99.11 -> -76.77, i.e. +22.34 dB, which is F81. That arm is still reachable
    (`composition="legacy_impulse"`) and still carries the defect, so the record is
    kept as a test rather than deleted -- see the next one.
    """
    lo = _noise_floor_db(0.0, device=torch_device)
    hi = _noise_floor_db(24.0, device=torch_device)
    assert abs(hi - lo) < 1.0, (
        f"noise-only floor moved {hi - lo:+.2f} dB for +24 dB of transmit power")


def test_the_legacy_composition_still_carries_F81(torch_device):
    """The retracted behaviour, kept reachable and kept MEASURED rather than deleted.

    `composition="legacy_impulse"` exists so the stored corpora's bit-parity gates can
    reproduce the chain those files were generated under. That chain has F81 in it. If
    someone "fixes" the legacy arm, every stored corpus stops reproducing, and this
    test is the tripwire that says so in F81's own words instead of as a mystery diff.
    MEASURED 2026-09-24 on this fixture: +22.3 dB of floor for +24 dB of P_tx.
    """
    lo = _noise_floor_db(0.0, device=torch_device, composition="legacy_impulse")
    hi = _noise_floor_db(24.0, device=torch_device, composition="legacy_impulse")
    assert hi - lo > 10.0, (
        f"the legacy composition's noise floor moved {hi - lo:+.2f} dB for +24 dB of "
        f"transmit power; F81 measured ~+20.8 dB and the stored corpora were generated "
        f"with it. If this is now small, the legacy arm is no longer legacy and every "
        f"bit-parity gate against a stored corpus is about to read a non-zero diff.")


def test_noise_figure_moves_the_corpus_noise_floor(torch_device):
    """XFAIL REMOVED 2026-09-24 -- it passes, with its SCOPE now stated in the test.

    Old xfail reason: "the kTBF floor is added at the front end's OUTPUT rather than
    input-referred, so it sits below the RFFE's own noise and noise_figure_db ... moves
    the floor by ~0.06 dB over a 20 dB sweep." Under the FULL contract there is exactly
    ONE thermal injection in the chain, at the antenna reference, and when no front end
    is configured it is `ThermalNoiseBlock`'s, with `F = cfg.noise_figure_db`. MEASURED
    2026-09-24 on this fixture: 5 dB -> -113.45, 25 dB -> -93.45, delta +20.00 dB.

    THE SCOPE, and it is the whole reason this test says `use_rffe=False`: when a front
    end IS configured, the noise figure that sets the floor is ITS OWN CASCADE's
    (`FLNA*Fmix*FBB`, `rffe_model.py`), not `cfg.noise_figure_db`, and
    `ThermalNoiseBlock(mode="once")` adds nothing at all. That is contract section 1.4,
    not a residual defect -- two noise figures both setting one floor is precisely the
    double-count F81 is about. `test_cfg_noise_figure_is_inert_behind_a_front_end`
    below pins that half, so the dial's real scope is written down in the suite instead
    of being rediscovered from a flat sweep.
    """
    lo = _noise_floor_db(noise_figure_db=5.0, use_rffe=False, device=torch_device)
    hi = _noise_floor_db(noise_figure_db=25.0, use_rffe=False, device=torch_device)
    assert hi - lo > 10.0, (
        f"+20 dB of noise figure moved the floor {hi - lo:+.2f} dB; the dial does nothing")


def test_cfg_noise_figure_is_inert_behind_a_front_end(torch_device):
    """The other half of the scope above, stated as a test so it cannot rot.

    With a `FrontEndBlock` present the floor is its Friis cascade's, referenced to
    `min(if_bw, fs)`, and `cfg.noise_figure_db` is not in that path. MEASURED
    2026-09-24: 5 dB -> -75.18, 25 dB -> -75.14, delta +0.04 dB. A corpus generator
    that wants to sweep difficulty behind a front end must move the front end's OWN
    knobs (`lna_bias_ma`, `if_bw_mhz`), and this test exists so that is read off the
    suite rather than inferred from a sweep that does nothing.
    """
    lo = _noise_floor_db(noise_figure_db=5.0, device=torch_device)
    hi = _noise_floor_db(noise_figure_db=25.0, device=torch_device)
    assert abs(hi - lo) < 1.0, (
        f"cfg.noise_figure_db moved the floor {hi - lo:+.2f} dB behind a front end. "
        f"If that is now a real dial, the chain has TWO noise figures setting one "
        f"floor, which is the double-count contract section 1.4 removed.")
