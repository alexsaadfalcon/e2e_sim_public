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

sys.path.insert(0, str(Path(__file__).parent))
from test_ml_chain_generate import _CFG, _FakeRTEnvironment  # noqa: E402

from e2e.blocks import RFFEBlock  # noqa: E402
from e2e.ml import chain_generate  # noqa: E402
from e2e.ml.labels import LabelGrid  # noqa: E402
from e2e.simulation import CircuitStage  # noqa: E402


def _cube_power_db(tx_power_dbm, *, use_rffe, physical_scale=None, device=None):
    """Mean power of the radar cube out of the fully composed chain, in dB."""
    cfg = dataclasses.replace(_CFG, tx_power_dbm=float(tx_power_dbm))
    grid = LabelGrid.for_config(cfg, range_stride=1, n_azimuth=8)
    env = _FakeRTEnvironment(cfg, grid, n_frames=1, device=device, seed=0)
    rffe_kwargs = None if physical_scale is None else {"physical_scale": physical_scale}
    with tempfile.TemporaryDirectory() as out_dir:
        sim = chain_generate.build_chain_simulation(
            scenario=None, cfg=cfg, out_dir=out_dir, environment_block=env,
            use_rffe=use_rffe, rffe_kwargs=rffe_kwargs, device=device,
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


def test_ml_corpus_composition_normalises_away_the_absolute_scale(torch_device):
    """PINS A KNOWN DEFECT (F62.2, diagnosed 2026-08-28) -- read before "fixing" this.

    `RFFEBlock` defaults `physical_scale=False`, and on that path it runs
    `frame * signal_scaling / mean(abs(frame))` (`e2e/blocks.py:109`) -- a per-frame
    normalisation to a UNIT-LESS constant. `build_chain_simulation` sets only `n` on the
    block, so the ML corpus generator has always taken that default, and it then appends
    `ThermalNoiseBlock` AFTER it under a comment insisting that position "is the whole
    point" because impairments need an absolute reference to be relative to.

    The consequence is not that the cube stops scaling -- it does scale, uniformly, which
    is why a total-power test like the one above passes even here. It is that the SIGNAL's
    level relative to the front end's own absolute noise is fixed by `signal_scaling`
    rather than by the radar equation, so target SNR stops tracking transmit power, noise
    figure and range.

    This test asserts the CURRENT state so the defect is visible in the suite rather than
    only in a report. When the fix lands -- `physical_scale=True` for the ML composition,
    together with a CORPUS-WIDE (never per-frame) input normalisation in the training
    path, which is the part that makes switching it on safe -- this test should be
    inverted, not deleted, and its docstring updated to say so.
    """
    grid = LabelGrid.for_config(_CFG, range_stride=1, n_azimuth=8)
    env = _FakeRTEnvironment(_CFG, grid, n_frames=1, device=torch_device, seed=0)
    with tempfile.TemporaryDirectory() as out_dir:
        sim = chain_generate.build_chain_simulation(
            scenario=None, cfg=_CFG, out_dir=out_dir, environment_block=env,
        )
    stage = next(s for s in sim.serial_stages if isinstance(s, CircuitStage))
    assert isinstance(stage.rffe_block, RFFEBlock)
    assert stage.rffe_block.physical_scale is False, (
        "physical_scale is no longer False for the ML composition -- if that is the "
        "intended fix, invert this test and update its docstring (see F62.2)."
    )


def test_physical_scale_is_reachable_and_changes_the_output(torch_device):
    """The fix is a one-line default change, so prove the flag is live, not vestigial."""
    a = _cube_power_db(12.0, use_rffe=True, physical_scale=False, device=torch_device)
    b = _cube_power_db(12.0, use_rffe=True, physical_scale=True, device=torch_device)
    assert abs(b - a) > 1.0, (
        f"physical_scale made no difference ({a:.2f} vs {b:.2f} dB) -- it is supposed "
        f"to be the switch between an arbitrary and an absolute amplitude scale."
    )
