"""Tests for analog aperture compression and the full/reduced dimension contract."""

import pytest

torch = pytest.importorskip("torch")

from e2e import frames
from e2e.chain.compress import CompressBlock, DecompressBlock, quantize_weights
from e2e.frames import (
    DIMENSION_FULL,
    DIMENSION_REDUCED,
    FrameCapabilities,
    FrameContractError,
)


def _frame(n_rx=16, n_tx=1, n_chirp=2, n_freqs=8, device=None, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    re = torch.randn(n_rx, n_tx, n_chirp, n_freqs, generator=g)
    im = torch.randn(n_rx, n_tx, n_chirp, n_freqs, generator=g)
    return torch.complex(re, im).to(device=device)


def _state(s_pars):
    return {"s_pars": s_pars, "signal_domain": frames.DOMAIN_CFR,
            "signal_dimension": DIMENSION_FULL}


# --------------------------------------------------------------------------------
# The contract itself
# --------------------------------------------------------------------------------
def test_dimension_defaults_preserve_the_historical_contract():
    """A block that declares nothing about dimension is full-dimension, so every
    pre-existing block keeps working untouched."""
    caps = FrameCapabilities()
    assert caps.dimension == DIMENSION_FULL
    assert caps.emits_dimension is None
    assert caps.is_dimension_bridge is False


def test_require_dimension_rejects_reduced_data_at_a_full_dimension_block():
    """The failure this contract exists to prevent: an angle FFT handed random
    projections would run happily and image nothing."""
    class AngleFFT:
        frame_capabilities = FrameCapabilities(dimension=DIMENSION_FULL)

    with pytest.raises(FrameContractError, match="DecompressBlock"):
        frames.require_dimension(DIMENSION_REDUCED, AngleFFT())


def test_require_dimension_allows_any():
    class Agnostic:
        frame_capabilities = FrameCapabilities(dimension=frames.DIMENSION_ANY)

    frames.require_dimension(DIMENSION_REDUCED, Agnostic())
    frames.require_dimension(DIMENSION_FULL, Agnostic())


def test_capabilities_reject_unknown_dimension():
    with pytest.raises(ValueError, match="unknown signal dimension"):
        FrameCapabilities(dimension="squashed")
    with pytest.raises(ValueError, match="unknown emitted signal dimension"):
        FrameCapabilities(emits_dimension="squashed")


# --------------------------------------------------------------------------------
# CompressBlock
# --------------------------------------------------------------------------------
def test_compress_reduces_dim0_and_flags_the_dimension(torch_device):
    s = _frame(n_rx=16, device=torch_device)
    out = CompressBlock(n_measurements=4).apply(_state(s))

    assert out["s_pars"].shape == (4, 1, 2, 8)
    assert out["signal_dimension"] == DIMENSION_REDUCED
    assert out["aperture_shape"] == (16, 1)
    assert out["sensing_matrix"].shape == (4, 16)


def test_compress_rejects_expansion(torch_device):
    s = _frame(n_rx=8, device=torch_device)
    with pytest.raises(FrameContractError, match="expansion, not a compression"):
        CompressBlock(n_measurements=32).apply(_state(s))


def test_compress_reuses_one_sensing_matrix_across_frames(torch_device):
    """A static analog combining network has ONE set of weights; redrawing per frame
    would model a different architecture and flatter any subspace tracker."""
    block = CompressBlock(n_measurements=6)
    a1 = block.apply(_state(_frame(seed=1, device=torch_device)))["sensing_matrix"]
    a2 = block.apply(_state(_frame(seed=2, device=torch_device)))["sensing_matrix"]
    assert torch.equal(a1, a2)


def test_compress_is_linear(torch_device):
    """Analog combining is a linear operation -- superposition must hold exactly."""
    block = CompressBlock(n_measurements=5)
    a = _frame(seed=3, device=torch_device)
    b = _frame(seed=4, device=torch_device)
    ya = block.apply(_state(a))["s_pars"]
    yb = block.apply(_state(b))["s_pars"]
    yab = block.apply(_state(a + b))["s_pars"]
    assert torch.allclose(ya + yb, yab, atol=1e-5)


# --------------------------------------------------------------------------------
# Weight quantization -- these are ANALOG control settings, not data
# --------------------------------------------------------------------------------
def test_quantize_weights_is_uniform_and_bounded(torch_device):
    g = torch.Generator(device="cpu").manual_seed(0)
    a = torch.complex(torch.randn(8, 32, generator=g),
                      torch.randn(8, 32, generator=g)).to(torch_device)
    q = quantize_weights(a, bits=6)

    full_scale = torch.max(torch.abs(torch.view_as_real(a)))
    lsb = full_scale / (2 ** 5 - 1)                       # top code == full scale
    err = torch.abs(torch.view_as_real(q - a))
    # EVERY weight within half an LSB, including the extreme -- no clamping, unlike the
    # ADC quantizer where clipping the out-of-range signal is the physical behaviour.
    assert float(err.max()) <= float(lsb) / 2 + 1e-6
    # Uniform, not floating point: the error floor is constant in ABSOLUTE terms, so
    # small weights are hit proportionally hardest. That is the physical behaviour.
    assert float(err.mean()) > 0.0


def test_quantize_weights_rejects_one_bit(torch_device):
    """One bit leaves zero levels once a sign is spent -- reject it rather than emit an
    all-zero matrix that would silently annihilate the signal."""
    a = torch.complex(torch.randn(2, 2), torch.randn(2, 2)).to(torch_device)
    with pytest.raises(ValueError, match="weight_bits"):
        quantize_weights(a, bits=1)


def test_quantize_weights_none_is_exact(torch_device):
    a = torch.complex(torch.randn(4, 4), torch.randn(4, 4)).to(torch_device)
    assert torch.equal(quantize_weights(a, bits=None), a)


def test_compress_weight_bits_changes_the_matrix(torch_device):
    s = _frame(device=torch_device)
    coarse = CompressBlock(n_measurements=4, weight_bits=3).apply(_state(s))["sensing_matrix"]
    ideal = CompressBlock(n_measurements=4, weight_bits=None).apply(_state(s))["sensing_matrix"]
    assert not torch.allclose(coarse, ideal, atol=1e-6)


# --------------------------------------------------------------------------------
# DecompressBlock
# --------------------------------------------------------------------------------
def test_decompress_restores_shape_and_dimension(torch_device):
    s = _frame(n_rx=16, device=torch_device)
    state = _state(s)
    state.update(CompressBlock(n_measurements=6).apply(state))
    out = DecompressBlock().apply(state)

    assert out["s_pars"].shape == s.shape
    assert out["signal_dimension"] == DIMENSION_FULL


def test_decompress_is_exact_when_not_actually_compressing(torch_device):
    """M == N with a well-conditioned matrix makes the pseudo-inverse a true inverse --
    the one case where reconstruction is lossless, worth pinning so the lossy case
    below is unambiguous."""
    s = _frame(n_rx=8, device=torch_device)
    state = _state(s)
    state.update(CompressBlock(n_measurements=8, weight_bits=None).apply(state))
    out = DecompressBlock().apply(state)
    assert torch.allclose(out["s_pars"], s, atol=1e-3)


def test_decompress_is_lossy_when_compressing(torch_device):
    """M < N cannot be inverted: the pseudo-inverse recovers only the row-space
    component. Pinned so nobody mistakes DecompressBlock for a free undo."""
    s = _frame(n_rx=16, device=torch_device)
    state = _state(s)
    state.update(CompressBlock(n_measurements=4, weight_bits=None).apply(state))
    rec = DecompressBlock().apply(state)["s_pars"]

    err = torch.linalg.norm((rec - s).flatten()) / torch.linalg.norm(s.flatten())
    assert float(err) > 0.1


def test_decompress_without_a_sensing_matrix_names_the_cause(torch_device):
    state = _state(_frame(device=torch_device))
    state["signal_dimension"] = DIMENSION_REDUCED
    with pytest.raises(FrameContractError, match="sensing_matrix"):
        DecompressBlock().apply(state)


def test_compress_then_decompress_declares_a_round_trip(torch_device):
    assert CompressBlock(n_measurements=2).frame_capabilities.is_dimension_bridge
    assert DecompressBlock().frame_capabilities.is_dimension_bridge
    assert CompressBlock(n_measurements=2).frame_capabilities.emits_dimension == DIMENSION_REDUCED
    assert DecompressBlock().frame_capabilities.emits_dimension == DIMENSION_FULL


# --------------------------------------------------------------------------------
# End-to-end through a real Simulation -- the contract must fire in the pipeline,
# not merely in isolation.
# --------------------------------------------------------------------------------
def test_simulation_enforces_dimension_on_a_full_dimension_stage(make_env_block):
    """A stage that needs the full aperture, placed after compression, must be stopped
    by Simulation with a named error -- the whole point of the contract."""
    from e2e.simulation import Simulation

    class NeedsFullAperture:
        frame_capabilities = FrameCapabilities(accepts_mimo=True, chirps=frames.CHIRP_NATIVE,
                                                dimension=DIMENSION_FULL)

        def apply(self, state):                      # pragma: no cover -- must not run
            raise AssertionError("ran on compressed data")

    env = make_env_block(n_frames=1, n_freqs=16)
    sim = Simulation(env, [], 2)
    sim.serial_stages = [CompressBlock(n_measurements=4), NeedsFullAperture()]

    with pytest.raises(FrameContractError, match="DecompressBlock"):
        sim.run(n_steps=1)


def test_simulation_round_trips_compress_then_decompress(make_env_block):
    """Compress -> decompress leaves the chain back in full dimension with the original
    frame shape, so downstream full-dimension blocks run normally."""
    from e2e.simulation import Simulation

    seen = {}

    class Recorder:
        frame_capabilities = FrameCapabilities(accepts_mimo=True, chirps=frames.CHIRP_NATIVE,
                                                dimension=DIMENSION_FULL)

        def apply(self, state):
            seen["shape"] = tuple(state["s_pars"].shape)
            seen["dimension"] = state.get("signal_dimension")
            return {}

    env = make_env_block(n_frames=1, n_freqs=16)
    original = tuple(env.get_S_pars().shape)
    sim = Simulation(env, [], 2)
    sim.serial_stages = [CompressBlock(n_measurements=4), DecompressBlock(), Recorder()]
    sim.run(n_steps=1)

    assert seen["dimension"] == DIMENSION_FULL
    assert seen["shape"] == original


def test_simulation_allows_a_reduced_dimension_stage_after_compression(make_env_block):
    """The capability this exists to enable: a block that declares DIMENSION_REDUCED
    runs on the compressed measurements, with no decompression inserted and no error."""
    from e2e.simulation import Simulation

    seen = {}

    class WorksCompressed:
        frame_capabilities = FrameCapabilities(accepts_mimo=True, chirps=frames.CHIRP_NATIVE,
                                                dimension=DIMENSION_REDUCED)

        def apply(self, state):
            seen["shape"] = tuple(state["s_pars"].shape)
            return {}

    env = make_env_block(n_frames=1, n_freqs=16)
    sim = Simulation(env, [], 2)
    sim.serial_stages = [CompressBlock(n_measurements=5), WorksCompressed()]
    sim.run(n_steps=1)

    assert seen["shape"][0] == 5          # saw the measurements, not the antennas


def test_compress_before_dechirp_is_refused_not_silently_wrong():
    """PINS A KNOWN LIMITATION, so it is documented behaviour rather than a surprise.

    The "analog compression saves ADCs" architecture wants CompressBlock upstream of the
    dechirp. It does not work today and must not appear to: `beat_from_cfr` reverses the
    RX/TX antenna axes and `mimo_combine` applies a per-TX code down dim 1, and neither
    means anything once dim 0/1 index linear combinations of the aperture rather than
    antennas. DechirpBlock therefore declares DIMENSION_FULL and this composition raises.

    If someone later teaches the dechirp stage to carry the aperture basis, this test is
    the one to change -- deliberately, with the physics argued -- not to delete.
    """
    from e2e.chain.dechirp import DechirpBlock

    assert DechirpBlock.frame_capabilities.dimension == DIMENSION_FULL
    with pytest.raises(FrameContractError, match="DecompressBlock"):
        frames.require_dimension(DIMENSION_REDUCED, DechirpBlock.__new__(DechirpBlock))
