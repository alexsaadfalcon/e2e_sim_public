"""Tests for `e2e.chain.dechirp.DechirpBlock` -- the frequency -> RX-time bridge.

All tests here are ungated (synthetic `s_pars`, no Sionna/DrJit needed): the block's
math (conjugate, antenna-index reversal, TDM/DDMA MIMO combine) is pure tensor algebra,
independent of where the CFR came from. The gated ray-tracing tests in
`tests/test_ml_rt_gen.py` continue to exercise `_beat_from_paths`/`mimo_combine`, which
this module now delegates to (see `e2e/chain/dechirp.py`'s module docstring).
"""

import dataclasses
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from e2e.chain.dechirp import ANTENNA_INDEX_REVERSED, DechirpBlock, beat_from_cfr, mimo_combine
from e2e.frames import (
    CHIRP_NATIVE,
    DOMAIN_CFR,
    DOMAIN_PAYLOAD_KEY,
    DOMAIN_RX_TIME,
    FrameCapabilities,
    FrameContractError,
    require_domain,
)
from e2e.ml.radar_config import TI_IWR1443


def _cfg(**overrides):
    return dataclasses.replace(TI_IWR1443, **overrides)


def _random_s_pars(n_rx, n_tx, n_chirp, n_freq, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    re = torch.randn(n_rx, n_tx, n_chirp, n_freq, generator=g)
    im = torch.randn(n_rx, n_tx, n_chirp, n_freq, generator=g)
    return torch.complex(re, im).to(torch.complex64).to(device)


# --------------------------------------------------------------------------------
# Bit-exactness against the pre-refactor reference math
# --------------------------------------------------------------------------------
def _reference_beat_and_adc(s_pars_np, cfg):
    """Verbatim transcription of the pre-refactor `e2e.ml.rt_gen._beat_from_paths` /
    `mimo_combine` numpy math (see that module's git history) -- an independent golden
    reference, NOT a call into `e2e.chain.dechirp`, so this test actually guards against
    the two implementations drifting apart rather than comparing code with itself.
    """
    beat = np.conjugate(s_pars_np)
    beat = beat[::-1, ::-1, :, :]
    beat = np.ascontiguousarray(beat, dtype=np.complex64)

    mimo = str(cfg.mimo).lower()
    n_rx, n_tx, n_chirps, n_samples = beat.shape
    if mimo in ("tdm", "single"):
        tx_of_chirp = np.arange(n_chirps) % n_tx
        adc = beat[:, tx_of_chirp, np.arange(n_chirps), :]
    elif mimo == "ddma":
        t_idx = np.arange(n_tx)[None, :, None]
        c_idx = np.arange(n_chirps)[None, None, :]
        code = np.exp(2j * np.pi * t_idx * c_idx / n_tx).astype(np.complex64)
        adc = np.einsum("rtcn,xtc->rcn", beat, code)
    else:
        raise ValueError(mimo)
    return beat, adc


def test_beat_mapping_bit_exact_against_reference(torch_device):
    """Conjugate + antenna-index reversal: pure data movement, no arithmetic reordering
    -- must be EXACTLY equal (torch.equal, not allclose)."""
    s_pars = _random_s_pars(6, 3, 4, 8, torch_device, seed=1)
    ref_beat, _ = _reference_beat_and_adc(s_pars.cpu().numpy(), _cfg())
    new_beat = beat_from_cfr(s_pars)
    assert torch.equal(new_beat.cpu(), torch.from_numpy(ref_beat))


def test_tdm_combine_bit_exact_against_reference(torch_device):
    """TDM/single combine is pure indexing (no arithmetic) -- must be EXACTLY equal."""
    cfg = _cfg(n_tx=3, n_rx=4, n_chirps=9, n_samples=5, mimo="tdm")
    s_pars = _random_s_pars(cfg.n_rx, cfg.n_tx, cfg.n_chirps, cfg.n_samples, torch_device, seed=2)
    ref_beat, ref_adc = _reference_beat_and_adc(s_pars.cpu().numpy(), cfg)

    out = DechirpBlock(cfg).apply({"s_pars": s_pars})["adc"]
    assert out.shape == ref_adc.shape
    assert torch.equal(out.cpu(), torch.from_numpy(ref_adc))


def test_single_scheme_bit_exact_against_reference(torch_device):
    cfg = _cfg(n_tx=1, n_rx=4, n_chirps=5, n_samples=6, mimo="single")
    s_pars = _random_s_pars(cfg.n_rx, cfg.n_tx, cfg.n_chirps, cfg.n_samples, torch_device, seed=3)
    _, ref_adc = _reference_beat_and_adc(s_pars.cpu().numpy(), cfg)
    out = DechirpBlock(cfg).apply({"s_pars": s_pars})["adc"]
    assert torch.equal(out.cpu(), torch.from_numpy(ref_adc))


def test_ddma_combine_matches_reference(torch_device):
    """DDMA sums complex products across the TX axis: exact equality is not guaranteed
    across independently-written summations (the gated `test_ml_rt_gen.py` also only
    checks this scheme with `assert_allclose`, never exact) -- but it must be equal to
    tight numerical precision."""
    cfg = _cfg(n_tx=3, n_rx=4, n_chirps=9, n_samples=5, mimo="ddma")
    s_pars = _random_s_pars(cfg.n_rx, cfg.n_tx, cfg.n_chirps, cfg.n_samples, torch_device, seed=4)
    _, ref_adc = _reference_beat_and_adc(s_pars.cpu().numpy(), cfg)
    out = DechirpBlock(cfg).apply({"s_pars": s_pars})["adc"].cpu().numpy()
    np.testing.assert_allclose(out, ref_adc, rtol=1e-5, atol=1e-6 * np.abs(ref_adc).max())


def test_rt_gen_delegates_to_dechirp_block_and_stays_bit_exact(torch_device):
    """`e2e.ml.rt_gen.mimo_combine` (the historical public entry point, still exercised
    by the gated `test_ml_rt_gen.py`) must be bit-exact with `DechirpBlock`'s combine on
    the same beat cube, proving the delegation is a true no-op refactor."""
    from e2e.ml.rt_gen import mimo_combine as rt_gen_mimo_combine

    cfg = _cfg(n_tx=3, n_rx=4, n_chirps=9, n_samples=5, mimo="tdm")
    s_pars = _random_s_pars(cfg.n_rx, cfg.n_tx, cfg.n_chirps, cfg.n_samples, torch_device, seed=5)
    beat = beat_from_cfr(s_pars)

    old = rt_gen_mimo_combine(cfg, beat.cpu().numpy())
    new = mimo_combine(cfg, beat).cpu().numpy()
    np.testing.assert_array_equal(old, new)


# --------------------------------------------------------------------------------
# Domain / capability declarations
# --------------------------------------------------------------------------------
def test_declares_cfr_in_rx_time_out():
    caps = DechirpBlock.frame_capabilities
    assert caps.domain == DOMAIN_CFR
    assert caps.emits_domain == DOMAIN_RX_TIME
    assert caps.accepts_mimo is True
    assert caps.chirps == CHIRP_NATIVE
    assert caps.is_bridge is True


def test_antenna_index_reversal_flag_is_on_by_default():
    # Locks in the convention e2e.ml.rt_gen's module docstring documents as validated
    # against re-traced ground truth -- see "Element ordering / array handedness".
    assert ANTENNA_INDEX_REVERSED is True


def test_dechirp_names_itself_as_the_bridge_in_a_misordered_chain():
    with pytest.raises(FrameContractError, match=r"expects the cfr domain.*insert a ModulateBlock"):
        require_domain(DOMAIN_RX_TIME, DechirpBlock(_cfg()))


def test_misordered_chain_before_dechirp_names_it_as_the_remedy():
    """A component still expecting DOMAIN_RX_TIME while the chain is in DOMAIN_CFR
    names DechirpBlock as the fix (frames.py's own _DOMAIN_BRIDGE table)."""
    class _RxTimeConsumer:
        frame_capabilities = FrameCapabilities(domain=DOMAIN_RX_TIME, chirps=CHIRP_NATIVE)

    with pytest.raises(FrameContractError, match=r"expects the rx_time domain.*insert a DechirpBlock"):
        require_domain(DOMAIN_CFR, _RxTimeConsumer())


# --------------------------------------------------------------------------------
# signal_domain transition + state contract
# --------------------------------------------------------------------------------
def test_apply_sets_signal_domain_and_adc_payload_key(torch_device):
    cfg = _cfg(n_tx=2, n_rx=4, n_chirps=6, n_samples=8, mimo="tdm")
    s_pars = _random_s_pars(cfg.n_rx, cfg.n_tx, cfg.n_chirps, cfg.n_samples, torch_device, seed=6)
    out = DechirpBlock(cfg).apply({"s_pars": s_pars})
    assert out["signal_domain"] == DOMAIN_RX_TIME
    assert DOMAIN_PAYLOAD_KEY[DOMAIN_RX_TIME] == "adc"
    assert "adc" in out
    assert out["adc"].dtype == torch.complex64


# --------------------------------------------------------------------------------
# ADC shape for a TDM MIMO config
# --------------------------------------------------------------------------------
def test_adc_shape_for_tdm_mimo(torch_device):
    cfg = _cfg(n_tx=3, n_rx=4, n_chirps=9, n_samples=16, mimo="tdm")
    s_pars = _random_s_pars(cfg.n_rx, cfg.n_tx, cfg.n_chirps, cfg.n_samples, torch_device, seed=7)
    adc = DechirpBlock(cfg).apply({"s_pars": s_pars})["adc"]
    assert adc.shape == (cfg.n_rx, cfg.n_chirps, cfg.n_samples)
    assert adc.device.type == s_pars.device.type


def test_adc_shape_for_ddma_mimo(torch_device):
    cfg = _cfg(n_tx=3, n_rx=4, n_chirps=9, n_samples=16, mimo="ddma")
    s_pars = _random_s_pars(cfg.n_rx, cfg.n_tx, cfg.n_chirps, cfg.n_samples, torch_device, seed=8)
    adc = DechirpBlock(cfg).apply({"s_pars": s_pars})["adc"]
    assert adc.shape == (cfg.n_rx, cfg.n_chirps, cfg.n_samples)


def test_unsupported_mimo_scheme_raises():
    cfg = _cfg(n_tx=2, n_rx=4, n_chirps=4, n_samples=4, mimo="unknown")
    s_pars = _random_s_pars(4, 2, 4, 4, "cpu", seed=9)
    with pytest.raises(ValueError, match="unsupported mimo scheme"):
        DechirpBlock(cfg).apply({"s_pars": s_pars})


# --------------------------------------------------------------------------------
# e2e.environment.blocks import boundary (no Sionna/DrJit at module scope)
# --------------------------------------------------------------------------------
def test_rt_environment_block_module_imports_without_sionna():
    """`e2e.environment.blocks` must be importable (and `RTEnvironmentBlock`
    constructible) without ever touching Sionna -- it is imported lazily inside
    `get_S_pars` only. Constructing does no ray tracing, so this needs neither Sionna
    nor a real Scenario/RadarConfig pair beyond simple stand-ins.
    """
    import e2e.environment.blocks as env_blocks

    cfg = _cfg(n_rx=4)
    block = env_blocks.RTEnvironmentBlock(scenario=object(), cfg=cfg)
    assert block.array_shape == (4, 1)
    assert block.frame_counter == 0
    assert block.last_labels is None
