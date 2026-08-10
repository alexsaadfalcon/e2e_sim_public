"""Tests for e2e/chain/receive.py: ImpairmentBlock, QuantizerBlock, RadarCubeBlock.

Synthetic ADC cubes only -- no Sionna, no dependency on a sibling shard's
DechirpBlock. Each test builds the `state` dict `{"adc": <cube>}` directly, matching
the contract these blocks are handed downstream of a dechirp step.
"""

import pytest
import torch

from e2e import frames
from e2e.chain.receive import ImpairmentBlock, QuantizerBlock, RadarCubeBlock
from e2e.ml.impairments import ClutterParams, LeakageParams, PhaseNoiseParams
from e2e.ml.radar_config import RadarConfig


# Tiny single-TX config so tests run fast; RadarCubeBlock's expected output shape is
# then directly (n_rx, n_samples, n_chirps), with no TDM de-interleave to account for.
_CFG = RadarConfig(
    name="test_cfg",
    f0_hz=77e9,
    bandwidth_hz=500e6,
    n_tx=1,
    n_rx=4,
    n_chirps=8,
    n_samples=16,
    fs_hz=5e6,
    chirp_period_s=10e-6,
    mimo="single",
)


@pytest.fixture
def small_adc(torch_device):
    """Factory: a synthetic ADC cube [n_rx, n_chirps, n_samples] complex64, uniform
    in [-scale, scale] on both real and imaginary parts, deterministic given `seed`.
    """
    def _make(scale=1.0, seed=0):
        g = torch.Generator(device=torch_device)
        g.manual_seed(seed)
        shape = (_CFG.n_rx, _CFG.n_chirps, _CFG.n_samples)
        real = (torch.rand(shape, generator=g, device=torch_device) * 2 - 1) * scale
        imag = (torch.rand(shape, generator=g, device=torch_device) * 2 - 1) * scale
        return (real + 1j * imag).to(torch.complex64)
    return _make


# --------------------------------------------------------------------------- domain contract

def test_blocks_declare_rx_time_domain():
    blocks = [ImpairmentBlock(_CFG), QuantizerBlock(), RadarCubeBlock(_CFG)]
    for block in blocks:
        assert frames.capabilities_of(block).domain == frames.DOMAIN_RX_TIME


def test_blocks_reject_frequency_domain():
    blocks = [ImpairmentBlock(_CFG), QuantizerBlock(), RadarCubeBlock(_CFG)]
    for block in blocks:
        with pytest.raises(frames.FrameContractError):
            frames.require_domain(frames.DOMAIN_CFR, block)


# --------------------------------------------------------------------------- ImpairmentBlock

def test_impairment_block_changes_cube_and_records_params(small_adc):
    adc = small_adc()
    block = ImpairmentBlock(_CFG, seed=0)
    out = block.apply({"adc": adc})

    assert out["adc"].shape == adc.shape
    assert out["adc"].dtype == adc.dtype
    assert not torch.equal(out["adc"], adc)

    params = out["impairment_params"]
    assert set(params) == {"phase_noise", "leakage", "clutter", "seed"}
    assert isinstance(params["phase_noise"], PhaseNoiseParams)
    assert isinstance(params["leakage"], LeakageParams)
    assert isinstance(params["clutter"], ClutterParams)
    assert params["seed"] == 0


def test_impairment_block_seed_determinism(small_adc):
    adc = small_adc()

    out1 = ImpairmentBlock(_CFG, seed=42).apply({"adc": adc.clone()})
    out2 = ImpairmentBlock(_CFG, seed=42).apply({"adc": adc.clone()})
    assert torch.equal(out1["adc"], out2["adc"])
    assert out1["impairment_params"]["leakage"] == out2["impairment_params"]["leakage"]

    out3 = ImpairmentBlock(_CFG, seed=43).apply({"adc": adc.clone()})
    assert not torch.equal(out1["adc"], out3["adc"])


def test_impairment_block_sampler_varies_and_records_per_frame(small_adc):
    adc = small_adc()

    def sampler(frame_index, rng):
        val = torch.rand(1, generator=rng, device=adc.device).item()
        return {
            "leakage": LeakageParams(leakage_relative_db=-20.0 * val),
            "clutter": None,  # explicitly skipped this frame
        }

    block = ImpairmentBlock(_CFG, chain_params=sampler, seed=7)
    seen_leakage_db = []
    for _ in range(3):
        out = block.apply({"adc": adc.clone()})
        seen_leakage_db.append(out["impairment_params"]["leakage"].leakage_relative_db)
        assert out["impairment_params"]["clutter"] is None
        assert isinstance(out["impairment_params"]["phase_noise"], PhaseNoiseParams)

    assert len(set(seen_leakage_db)) == 3  # all three frames sampled distinct params


# --------------------------------------------------------------------------- QuantizerBlock

def test_quantizer_block_high_bits_near_lossless(small_adc):
    adc = small_adc(scale=0.5)
    out = QuantizerBlock(bits=16, full_scale=1.0).apply({"adc": adc})
    assert out["clipped_fraction"] == 0.0
    assert out["quant_snr_db"] > 60.0


def test_quantizer_block_low_bits_measurably_coarser(small_adc):
    adc = small_adc(scale=0.5)
    hi = QuantizerBlock(bits=16, full_scale=1.0).apply({"adc": adc})
    lo = QuantizerBlock(bits=4, full_scale=1.0).apply({"adc": adc})
    assert lo["quant_snr_db"] < hi["quant_snr_db"] - 20.0


def test_quantizer_block_clipped_fraction_rises_with_overrange(small_adc):
    block = QuantizerBlock(bits=8, full_scale=1.0)
    within_range = block.apply({"adc": small_adc(scale=0.1)})
    over_range = block.apply({"adc": small_adc(scale=5.0)})
    assert within_range["clipped_fraction"] == 0.0
    assert over_range["clipped_fraction"] > 0.0


# --------------------------------------------------------------------------- RadarCubeBlock

def test_radar_cube_block_shape_and_no_mutation(small_adc):
    adc = small_adc()
    original = adc.clone()
    state = {"adc": adc}

    out = RadarCubeBlock(_CFG).apply(state)

    assert out["radar_cube"].shape == (_CFG.n_rx, _CFG.n_samples, _CFG.n_chirps)
    assert out["radar_cube"].dtype == torch.complex64
    assert "adc" not in out
    assert torch.equal(state["adc"], original)  # the block must not mutate adc


def test_quantizer_is_uniform_and_matches_the_textbook_adc_snr():
    """A uniform ADC's SNR is 6.02*bits + 1.76 dB for a full-scale sine. Checking
    against that closed form -- rather than only against another run of our own code --
    is what distinguishes a real converter model from an arbitrary rounding scheme.
    A floating-point quantizer does NOT satisfy this."""
    from e2e.chain.receive import QuantizerBlock

    n = 200_000
    t = torch.arange(n, dtype=torch.float32)
    # Full-scale complex sine, incommensurate frequency so samples don't land on a
    # repeating subset of quantization codes.
    x = torch.exp(1j * 2 * torch.pi * 0.0123456 * t).to(torch.complex64)

    for bits in (8, 12):
        out = QuantizerBlock(bits=bits, full_scale=1.0).apply({"adc": x})
        expected = 6.02 * bits + 1.76
        assert abs(out["quant_snr_db"] - expected) < 1.5, (
            f"{bits}-bit SNR {out['quant_snr_db']:.1f} dB is not within 1.5 dB of the "
            f"ideal uniform-ADC {expected:.1f} dB"
        )


def test_quantization_error_is_bounded_by_half_an_lsb():
    """The defining property of uniform quantization: every sample INSIDE the
    representable range lands within LSB/2 of its input. This is what lays down a FIXED
    noise floor, so a weak target below a strong one is buried the way hardware would
    bury it.

    The bound is asserted below the top code, not at the rail: a two's-complement
    converter spans codes -2^(b-1) .. 2^(b-1)-1, so the positive rail saturates half an
    LSB earlier than the negative one -- real behaviour, covered separately below.
    """
    from e2e.chain.receive import QuantizerBlock

    blk = QuantizerBlock(bits=10, full_scale=1.0)
    top_value = (2 ** (blk.bits - 1) - 1) * blk.lsb
    torch.manual_seed(0)
    scale = top_value * 0.98
    x = ((torch.rand(4, 3, 512) * 2 - 1) * scale
         + 1j * (torch.rand(4, 3, 512) * 2 - 1) * scale).to(torch.complex64)
    out = blk.apply({"adc": x})
    err = out["adc"] - x
    assert err.real.abs().max().item() <= blk.lsb / 2 + 1e-7
    assert err.imag.abs().max().item() <= blk.lsb / 2 + 1e-7


def test_positive_rail_saturates_at_the_top_code():
    """A sample at +full_scale cannot be represented: the highest positive code is
    2^(b-1)-1, so it pins there rather than wrapping or exceeding the range."""
    from e2e.chain.receive import QuantizerBlock

    blk = QuantizerBlock(bits=10, full_scale=1.0)
    top_value = (2 ** (blk.bits - 1) - 1) * blk.lsb
    x = torch.full((1, 1, 8), 1.0, dtype=torch.float32)
    out = blk.apply({"adc": torch.complex(x, -x).to(torch.complex64)})
    assert torch.allclose(out["adc"].real, torch.full_like(x, top_value))
    assert torch.allclose(out["adc"].imag, torch.full_like(x, -blk.full_scale))


def test_weak_signal_sits_at_the_quantization_floor_not_below_it():
    """A signal far below the LSB must be destroyed by the converter, not preserved.
    A floating-point quantizer would keep it clean -- the exact flattery this model
    exists to avoid."""
    from e2e.chain.receive import QuantizerBlock

    blk = QuantizerBlock(bits=8, full_scale=1.0)
    tiny = (blk.lsb / 100) * torch.ones(2, 2, 64, dtype=torch.complex64)
    out = blk.apply({"adc": tiny})
    assert torch.count_nonzero(out["adc"]) == 0
