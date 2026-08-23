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
    # "base_seed"/"frame_idx" are new fields (PHYSICS_JUSTIFICATION_AUDIT.md entry 10):
    # the clutter field is now drawn from `base_seed` alone and evolved by `frame_idx`,
    # so provenance must record both, not just the legacy per-frame "seed".
    assert set(params) == {"phase_noise", "leakage", "clutter", "seed", "base_seed", "frame_idx"}
    assert isinstance(params["phase_noise"], PhaseNoiseParams)
    assert isinstance(params["leakage"], LeakageParams)
    assert isinstance(params["clutter"], ClutterParams)
    assert params["seed"] == 0
    assert params["base_seed"] == 0
    assert params["frame_idx"] == 0


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


def test_impairment_block_clutter_field_persists_frame_to_frame(small_adc):
    """PHYSICS_JUSTIFICATION_AUDIT.md entry 10: the clutter field must be drawn once
    (from the block's base `seed`) and evolve, not be i.i.d. redrawn every frame the
    way `ImpairmentBlock` used to (old per-frame seed was `seed + frame_idx`, feeding
    every impairment stage including clutter). Isolate clutter by disabling the other
    two stages and feeding the SAME (silent) cube on frames 0 and 1: the clutter-only
    contribution must correlate strongly frame to frame."""
    silent = torch.zeros(_CFG.n_rx, _CFG.n_chirps, _CFG.n_samples, dtype=torch.complex64)
    block = ImpairmentBlock(
        _CFG, chain_params={"leakage": None, "phase_noise": None,
                            "clutter": ClutterParams(doppler_std_mps=0.0002)},
        seed=17,
    )
    out0 = block.apply({"adc": silent.clone()})["adc"]
    out1 = block.apply({"adc": silent.clone()})["adc"]

    assert not torch.equal(out0, out1)  # the field still evolves frame to frame

    num = torch.sum(out0 * torch.conj(out1))
    den = torch.sqrt(torch.sum(torch.abs(out0) ** 2) * torch.sum(torch.abs(out1) ** 2))
    corr = torch.abs(num / den).item()
    # doppler_std_mps=0.0002 m/s at 77 GHz / frame_rate_hz=10 (_CFG default) gives a
    # phase drift per frame of sigma_delta = 2*pi*(2*0.0002/lambda_m)*(1/10) ~= 0.065 rad
    # (see test_ml_impairments.py's persistence test for the same derivation), so the
    # field should stay highly correlated -- the old i.i.d.-redraw-every-frame behaviour
    # gave ~0 correlation here.
    assert corr > 0.9


def test_impairment_block_multiframe_determinism(small_adc):
    """(d) Two blocks with the same seed reproduce a multi-frame sequence bit-
    identically; a different seed diverges somewhere in the sequence."""
    adc = small_adc()

    def _run(seed, n=3):
        block = ImpairmentBlock(_CFG, seed=seed)
        return [block.apply({"adc": adc.clone()})["adc"] for _ in range(n)]

    seq_a = _run(seed=5)
    seq_b = _run(seed=5)
    for a, b in zip(seq_a, seq_b):
        assert torch.equal(a, b)

    seq_c = _run(seed=6)
    assert any(not torch.equal(a, c) for a, c in zip(seq_a, seq_c))


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


def test_default_full_scale_survives_physically_small_ray_traced_amplitudes():
    """Ray-traced cubes carry physical amplitudes around 1e-7..1e-6, far below a
    12-bit converter's LSB on a +-1.0 range. With a FIXED full scale such a frame
    quantizes to exactly zero and nothing reports a problem. The automatic default
    must instead scale to the frame and preserve it."""
    from e2e.chain.receive import QuantizerBlock

    torch.manual_seed(0)
    tiny = ((torch.randn(4, 2, 128) + 1j * torch.randn(4, 2, 128)) * 1e-7).to(torch.complex64)

    fixed = QuantizerBlock(bits=12, full_scale=1.0).apply({"adc": tiny})
    assert torch.count_nonzero(fixed["adc"]) == 0, "precondition: fixed scale destroys it"

    auto = QuantizerBlock(bits=12).apply({"adc": tiny})
    assert torch.count_nonzero(auto["adc"]) > 0
    assert auto["quant_snr_db"] > 50.0
    assert auto["adc_full_scale"] < 1e-5  # scaled to the frame, not to 1.0
    assert auto["clipped_fraction"] == 0.0  # headroom means the peak does not clip


# --------------------------------------------------------------------------- IFHighPassBlock

def _tone(bin_idx, amp=1.0, device="cpu"):
    """[1, 1, n_samples] complex64 beat tone at fast-time DFT bin `bin_idx`."""
    n = torch.arange(_CFG.n_samples, dtype=torch.float64, device=device)
    sig = amp * torch.exp(2j * torch.pi * bin_idx * n / _CFG.n_samples)
    return sig.to(torch.complex64).view(1, 1, -1)


def test_if_hpf_declares_rx_time_domain():
    from e2e.chain.receive import IFHighPassBlock
    assert IFHighPassBlock(_CFG).frame_capabilities.domain == frames.DOMAIN_RX_TIME


def test_if_hpf_nulls_dc_exactly_and_matches_butterworth_oracle(torch_device):
    """Oracle: a pure tone at bin k comes out scaled by the hand-computed Butterworth
    magnitude |H| = (f/fc)^n / sqrt(1 + (f/fc)^(2n)) at f = k*fs/N; the DC (range-0
    leakage) tone is nulled EXACTLY because H(0) = 0."""
    import math
    from e2e.chain.receive import IFHighPassBlock
    blk = IFHighPassBlock(_CFG, corner_range_m=1.0, order=2)

    out_dc = blk.apply({"adc": _tone(0, device=torch_device)})["adc"]
    assert torch.allclose(out_dc, torch.zeros_like(out_dc), atol=1e-6)

    for k in (1, 3, 8):
        f = k * _CFG.fs_hz / _CFG.n_samples
        ratio = (f / blk.corner_hz) ** blk.order
        expected = ratio / math.sqrt(1.0 + ratio ** 2)
        out = blk.apply({"adc": _tone(k, device=torch_device)})["adc"]
        measured = float(out.abs().max().item())
        assert measured == pytest.approx(expected, rel=1e-4), f"bin {k}"


def test_if_hpf_response_is_monotonic_and_transparent_at_far_range():
    from e2e.chain.receive import IFHighPassBlock
    blk = IFHighPassBlock(_CFG, corner_range_m=1.0, order=2)
    h = blk.response(_CFG.n_samples)
    assert torch.all(h[1:] >= h[:-1])
    # The last bin sits at ~max_range; the filter must be transparent there (<0.1 dB).
    assert float(h[-1].item()) > 10 ** (-0.1 / 20)


def test_if_hpf_corner_range_conversion_and_explicit_override():
    from e2e.chain.receive import IFHighPassBlock
    from e2e.ml.radar_config import C_MPS
    blk = IFHighPassBlock(_CFG, corner_range_m=2.0)
    assert blk.corner_hz == pytest.approx(2.0 * _CFG.ramp_slope_hzps * 2.0 / C_MPS)
    assert IFHighPassBlock(_CFG, corner_hz=123e3).corner_hz == 123e3


@pytest.mark.parametrize("kwargs", [
    {"corner_range_m": 0.0}, {"corner_range_m": -1.0},
    {"corner_hz": 0.0}, {"corner_hz": -5.0}, {"order": 0},
])
def test_if_hpf_invalid_params_raise(kwargs):
    from e2e.chain.receive import IFHighPassBlock
    with pytest.raises(ValueError):
        IFHighPassBlock(_CFG, **kwargs)


def test_if_hpf_reports_provenance(small_adc):
    from e2e.chain.receive import IFHighPassBlock
    blk = IFHighPassBlock(_CFG, corner_range_m=1.0, order=2)
    out = blk.apply({"adc": small_adc()})
    assert out["if_hpf_corner_hz"] == pytest.approx(blk.corner_hz)
    assert out["if_hpf_order"] == 2


def test_if_hpf_protects_quantizer_dynamic_range(torch_device):
    """The coupling A2 exists for: a range-0 leakage tone 60 dB above a far target
    sets QuantizerBlock's AGC full scale; the high-pass removes it so full scale is
    set by the target instead (drop of ~the full 60 dB)."""
    from e2e.chain.receive import IFHighPassBlock
    adc = _tone(0, amp=1.0, device=torch_device) + _tone(10, amp=1e-3, device=torch_device)
    quant = QuantizerBlock(bits=12)
    fs_without = quant._resolve_full_scale(adc)
    filtered = IFHighPassBlock(_CFG, corner_range_m=1.0, order=2).apply({"adc": adc})["adc"]
    fs_with = quant._resolve_full_scale(filtered)
    assert fs_without / fs_with > 10 ** (40 / 20), \
        "high-pass should drop the AGC full scale by tens of dB"


def test_if_hpf_steep_order_never_nans(torch_device):
    """Regression (found in review): the naive Butterworth form overflows to
    inf/inf = nan at high bins for a steep order, and one nan bin poisons the whole
    frame through the ifft. The stable form's worst case is a clean 0.0."""
    from e2e.chain.receive import IFHighPassBlock
    blk = IFHighPassBlock(_CFG, corner_range_m=1.0, order=100)
    h = blk.response(512)
    assert not torch.isnan(h).any()
    assert torch.all(h[1:] >= h[:-1])  # still monotonic
    out = blk.apply({"adc": torch.ones(1, 1, 512, dtype=torch.complex64,
                                       device=torch_device)})["adc"]
    assert not torch.isnan(out.real).any() and not torch.isnan(out.imag).any()
