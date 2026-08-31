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
from e2e.ml import chain_generate  # noqa: E402
from e2e.ml.labels import LabelGrid  # noqa: E402
from e2e.simulation import CircuitStage  # noqa: E402


def _cube_power_db(tx_power_dbm, *, use_rffe, physical_scale=None, device=None,
                   use_link_budget=True):
    """Mean power of the radar cube out of the fully composed chain, in dB."""
    cfg = dataclasses.replace(_CFG, tx_power_dbm=float(tx_power_dbm))
    grid = LabelGrid.for_config(cfg, range_stride=1, n_azimuth=8)
    env = _FakeRTEnvironment(cfg, grid, n_frames=1, device=device, seed=0)
    rffe_kwargs = None if physical_scale is None else {"physical_scale": physical_scale}
    with tempfile.TemporaryDirectory() as out_dir:
        sim = chain_generate.build_chain_simulation(
            scenario=None, cfg=cfg, out_dir=out_dir, environment_block=env,
            use_rffe=use_rffe, rffe_kwargs=rffe_kwargs, device=device,
            use_link_budget=use_link_budget,
        )
        sim.run(n_steps=1)
        cube = sim.get_outputs()["radar_cube"][0]
    x = cube.detach().cpu().numpy() if hasattr(cube, "detach") else np.asarray(cube)
    return 10.0 * np.log10(float(np.mean(np.abs(x) ** 2)) + 1e-300)


@pytest.mark.parametrize("use_rffe,physical_scale", [
    (False, None),        # link budget alone, no front end
    (True, True),         # front end on, absolute scale preserved
])
def test_composed_chain_scales_with_transmit_power(use_rffe, physical_scale, torch_device):
    """A 24 dB rise in transmit power must move the composed chain's cube by 24 dB.

    This is the connection `expected_target_snr_db`'s docstring promises and that
    nothing made: the formula is checkable on paper, but only running a frame through
    `build_chain_simulation` says whether the assembled chain honours it.
    """
    lo = _cube_power_db(0.0, use_rffe=use_rffe, physical_scale=physical_scale,
                        device=torch_device)
    hi = _cube_power_db(24.0, use_rffe=use_rffe, physical_scale=physical_scale,
                        device=torch_device)
    assert hi - lo == pytest.approx(24.0, abs=0.5), (
        f"tx_power_dbm 0->24 moved the cube {hi - lo:+.2f} dB, expected +24.00"
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
    """
    grid = LabelGrid.for_config(_CFG, range_stride=1, n_azimuth=8)
    env = _FakeRTEnvironment(_CFG, grid, n_frames=1, device=torch_device, seed=0)
    with tempfile.TemporaryDirectory() as out_dir:
        sim = chain_generate.build_chain_simulation(
            scenario=None, cfg=_CFG, out_dir=out_dir, environment_block=env,
        )
    stage = next(s for s in sim.serial_stages if isinstance(s, CircuitStage))
    assert isinstance(stage.rffe_block, RFFEBlock)
    assert stage.rffe_block.physical_scale is True, (
        "the ML corpus composition must keep the absolute amplitude scale -- an "
        "absolute thermal floor beneath a normalised cube is F63"
    )


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


def _noise_floor_db(tx_power_dbm=12.0, noise_figure_db=None, *, use_rffe=True, device=None):
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
            use_rffe=use_rffe, device=device)
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


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN DEFECT (F81, measured 2026-08-31). ThermalNoiseBlock applies sqrt(P_tx) "
    "DOWNSTREAM of the RF front end, so it multiplies the front end's own 4kTR noise by "
    "transmit power: the noise-only floor rises ~+20.8 dB for +24 dB of P_tx. Delete the "
    "xfail when the scaling moves to the correct node."))
def test_front_end_noise_floor_does_not_track_transmit_power(torch_device):
    """The claim F63 was landed to deliver, stated as a test that currently fails.

    `tests/test_composed_chain_scales_with_transmit_power` passes today and does NOT
    catch this: it measures TOTAL cube power, which scales uniformly whether the scaling
    is applied at the right node or the wrong one. The inverted defect test in this file
    says as much in its own docstring -- "it did not stop the cube scaling ... that is why
    the guard has to be structural" -- and I shipped a power test anyway. Zeroing the
    channel is what separates the two.
    """
    lo = _noise_floor_db(0.0, device=torch_device)
    hi = _noise_floor_db(24.0, device=torch_device)
    assert abs(hi - lo) < 1.0, (
        f"noise-only floor moved {hi - lo:+.2f} dB for +24 dB of transmit power")


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN DEFECT (F81). The kTBF floor is added at the front end's OUTPUT rather than "
    "input-referred, so it sits below the RFFE's own noise and noise_figure_db -- "
    "documented as 'THE DIFFICULTY DIAL for every generated corpus' -- moves the floor "
    "by ~0.06 dB over a 20 dB sweep."))
def test_noise_figure_moves_the_corpus_noise_floor(torch_device):
    lo = _noise_floor_db(noise_figure_db=5.0, device=torch_device)
    hi = _noise_floor_db(noise_figure_db=25.0, device=torch_device)
    assert hi - lo > 10.0, (
        f"+20 dB of noise figure moved the floor {hi - lo:+.2f} dB; the dial does nothing")
