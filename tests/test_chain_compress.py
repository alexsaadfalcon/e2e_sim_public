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
    """PINS A PHYSICAL ORDERING CONSTRAINT, not an implementation gap.

    `beat_from_cfr` reverses the RX/TX antenna axes to fix array handedness and
    `mimo_combine` applies a per-TX code down dim 1; neither means anything once dim 0/1
    index linear combinations of the aperture. Both are properties of the ARRAY, so they
    must be settled while the antenna axes still exist -- i.e. compress AFTER dechirp and
    TX de-multiplexing, not before. DechirpBlock declares DIMENSION_FULL and this
    composition raises rather than producing a plausible wrong cube.
    """
    from e2e.chain.dechirp import DechirpBlock

    assert DechirpBlock.frame_capabilities.dimension == DIMENSION_FULL
    with pytest.raises(FrameContractError, match="DecompressBlock"):
        frames.require_dimension(DIMENSION_REDUCED, DechirpBlock.__new__(DechirpBlock))


def test_reduced_dimension_still_enforces_the_chirp_axis(torch_device):
    """REGRESSION (found in pre-merge review): `check_capabilities` used to return early
    on reduced-dimension data, dropping the MIMO check (justified -- dim 1 is no longer
    TX) AND the chirp check (not justified -- CompressBlock preserves the chirp axis
    exactly). A chirp-restricted block placed after compression was therefore handed
    multi-chirp frames silently, which is precisely the 'plausible wrong answer' the
    contract exists to prevent."""
    class SingleChirpCompressed:
        frame_capabilities = FrameCapabilities(dimension=DIMENSION_REDUCED,
                                                chirps=frames.CHIRP_SINGLE,
                                                accepts_mimo=True)

    multi_chirp_reduced = _frame(n_rx=5, n_tx=1, n_chirp=3, n_freqs=8, device=torch_device)
    with pytest.raises(FrameContractError, match="multiple chirps"):
        frames.check_capabilities(multi_chirp_reduced, SingleChirpCompressed(),
                                  dimension=DIMENSION_REDUCED)

    # ...while the MIMO check IS still correctly skipped, since dim 1 no longer means TX.
    class MimoRestrictedCompressed:
        frame_capabilities = FrameCapabilities(dimension=DIMENSION_REDUCED,
                                                chirps=frames.CHIRP_NATIVE,
                                                accepts_mimo=False)

    frames.check_capabilities(_frame(n_rx=5, n_tx=4, n_chirp=2, n_freqs=8, device=torch_device),
                              MimoRestrictedCompressed(), dimension=DIMENSION_REDUCED)


# --------------------------------------------------------------------------------
# One compression concept: the AFE is the ADAPTIVE variant, sharing the same math.
# --------------------------------------------------------------------------------
def test_afe_shares_the_compression_math(torch_device):
    """AFEBlock must not carry a second implementation of combine/reconstruct. Pinned by
    behaviour: its outputs equal the shared primitives applied to the same quantized
    matrix."""
    from e2e.blocks import AFEBlock
    from e2e.chain.compress import WEIGHT_FLOAT, combine, quantize_weights, reconstruct_aperture

    afe = AFEBlock(exp=5, mantissa=6)
    g = torch.Generator(device="cpu").manual_seed(0)
    a = torch.complex(torch.randn(6, 20, generator=g), torch.randn(6, 20, generator=g)).to(torch_device)
    v = torch.complex(torch.randn(20, 9, generator=g), torch.randn(20, 9, generator=g)).to(torch_device)

    aq, x = afe.apply_mat_mul(a, v)
    expect_aq = quantize_weights(a, model=WEIGHT_FLOAT, exp=5, mantissa=6)
    assert torch.equal(aq, expect_aq)
    assert torch.equal(x, combine(expect_aq, v))
    assert torch.allclose(afe.reconstruct(aq, x), reconstruct_aperture(aq, x), atol=1e-6)


def test_weight_models_are_distinct_and_named(torch_device):
    """The two models describe different hardware -- uniform for analog control settings
    (constant ABSOLUTE step), float for a digital datapath (constant RELATIVE error).
    They must not silently coincide, or the distinction the docstrings draw is fiction."""
    from e2e.chain.compress import WEIGHT_FLOAT, WEIGHT_UNIFORM, quantize_weights

    g = torch.Generator(device="cpu").manual_seed(1)
    # Wide dynamic range: this is exactly where absolute- and relative-error models part.
    a = torch.complex(torch.randn(4, 64, generator=g), torch.randn(4, 64, generator=g))
    a = (a * torch.logspace(0, -4, 64)).to(torch_device)

    uni = quantize_weights(a, bits=6, model=WEIGHT_UNIFORM)
    flt = quantize_weights(a, model=WEIGHT_FLOAT, exp=5, mantissa=6)
    assert not torch.allclose(uni, flt, atol=1e-9)

    def rel_err(q):
        small = torch.abs(a) < torch.quantile(torch.abs(a), 0.2)
        return float((torch.abs(q - a)[small] / torch.abs(a)[small]).mean())

    # The smallest weights are hit far harder by the uniform model -- the physical point.
    # Uniform relative error SATURATES at 1.0 (those weights quantize to exactly zero, so
    # 100% of them is lost); the float model keeps them at ~22%. Assert both the ordering
    # and that saturation, rather than a ratio the saturation makes impossible.
    assert rel_err(uni) == pytest.approx(1.0, abs=1e-6)
    assert rel_err(flt) < 0.5
    assert rel_err(uni) > 3.0 * rel_err(flt)


def test_quantize_weights_rejects_unknown_model(torch_device):
    from e2e.chain.compress import quantize_weights

    with pytest.raises(ValueError, match="model must be"):
        quantize_weights(torch.complex(torch.randn(2, 2), torch.randn(2, 2)).to(torch_device),
                         bits=6, model="bogus")


def test_measurement_stage_can_stay_in_reduced_dimension(make_env_block):
    """The architectural choice the user asked for: digitizing 1024 elements needs 1024
    converters, combining to M first needs M. reconstruct=False keeps the chain in the
    measurement space and says so, instead of paying a lossy pinv nobody asked for."""
    from e2e.blocks import AFEBlock, MeasurementStage
    from e2e.simulation import Simulation
    from e2e.subspace.algorithms import Oja

    class Tracker:
        def __init__(self, d, k):
            self.oja = Oja(d, k)
            self.n_refine = 1
        def gen_A_ada(self):
            from e2e.subspace.algorithms import gen_A_ada
            return gen_A_ada(self.oja.U, 8)
        def update(self, X, A):
            pass

    env = make_env_block(n_frames=1, n_freqs=16)
    s = env.get_S_pars()
    tracker = Tracker(s.shape[0] * s.shape[1], 2)
    stage = MeasurementStage(AFEBlock(), tracker, reconstruct=False)
    assert stage.frame_capabilities.emits_dimension == DIMENSION_REDUCED

    seen = {}

    class Recorder:
        frame_capabilities = FrameCapabilities(accepts_mimo=True, chirps=frames.CHIRP_NATIVE,
                                                dimension=DIMENSION_REDUCED)
        def apply(self, state):
            seen["shape"] = tuple(state["s_pars"].shape)
            seen["dim"] = state.get("signal_dimension")
            return {}

    sim = Simulation(env, [], 2)
    sim.serial_stages = [stage, Recorder()]
    sim.run(n_steps=1)

    assert seen["dim"] == DIMENSION_REDUCED
    assert seen["shape"][0] == 8            # measurements, not the 1024-ish aperture
