"""Tests for RangeProfileBlock -- the compressed-domain range-profile product.

Covers the physics claim the block exists to demonstrate (compression mixes only
the aperture axis, so a per-channel range FFT survives it) and the frame-contract
claim (DIMENSION_ANY, unlike the angle products which require DIMENSION_FULL).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from e2e import frames
from e2e.blocks import AFEBlock, MeasurementStage, RangeAzBlock, RangeProfileBlock
from e2e.frames import DIMENSION_FULL, DIMENSION_REDUCED, FrameContractError
from e2e.subspace.algorithms import Oja, gen_A_ada


def _delay_frame(n_elements, n_freqs, k0, dev, seed=0):
    """A synthetic full-dimension frame `[n_elements, 1, 1, n_freqs]` where every
    element carries the SAME range delay (a pure complex exponential at bin `k0`)
    but an element-dependent complex amplitude: element `n`'s trace is
    `s[n] * exp(2j*pi*k0*f/n_freqs)`. A per-channel FFT along the frequency axis
    (torch.fft.fft) then puts every channel's peak at the exact same range bin --
    the orthogonality of complex exponentials makes this exact, not approximate.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    s = torch.complex(torch.randn(n_elements, generator=g), torch.randn(n_elements, generator=g))
    f = torch.arange(n_freqs, dtype=torch.float32)
    phase = torch.exp(2j * np.pi * k0 * f / n_freqs)
    v = s.view(n_elements, 1).to(torch.complex64) * phase.view(1, n_freqs).to(torch.complex64)
    return v.to(device=dev).view(n_elements, 1, 1, n_freqs)


class _StubTracker:
    """Minimal subspace-tracker stub -- just enough for
    `MeasurementStage(reconstruct=False)` to draw a measurement matrix and run an
    update; mirrors the stub in tests/test_simulation.py /
    tests/test_chain_compress.py."""

    def __init__(self, d, k, m):
        self.oja = Oja(d, k)
        self.n_refine = 1
        self.m = m

    def gen_A_ada(self):
        return gen_A_ada(self.oja.U, self.m)

    def update(self, X, A):
        pass


def _expected_bin(k0, n_freqs):
    """The bin a pure complex exponential exp(2j*pi*k0*f/n_freqs) lands at after
    torch.fft.fft + torch.fft.fftshift (even n_freqs): fftshift maps raw index k0
    to (k0 + n_freqs // 2) % n_freqs."""
    return (k0 + n_freqs // 2) % n_freqs


# --------------------------------------------------------------------------------
# Full-dimension vs reduced-dimension parity: the physics claim.
# --------------------------------------------------------------------------------
def test_range_profile_peak_bin_matches_full_vs_reduced(torch_device):
    n_elements, n_freqs, k0, k_sub, m = 64, 64, 10, 4, 16
    bins = n_freqs  # >= n_freqs -> _power_bin is identity, no binning offset to track

    full = _delay_frame(n_elements, n_freqs, k0, torch_device)
    expected_bin = _expected_bin(k0, n_freqs)

    # Full-dimension range profile.
    out_full = RangeProfileBlock(bins=bins).apply({"s_pars": full})
    per_channel_full = out_full["range_profile"]
    agg_full = out_full["range_profile_agg"]
    assert per_channel_full.shape == (n_elements, bins)
    assert torch.all(torch.argmax(per_channel_full, dim=1) == expected_bin)
    assert int(torch.argmax(agg_full).item()) == expected_bin

    # Compress via AFEBlock + MeasurementStage(reconstruct=False), the reduced
    # -dimension path a real pipeline takes (e2e/chain/compress.py, e2e/blocks.py).
    tracker = _StubTracker(d=n_elements, k=k_sub, m=m)
    stage = MeasurementStage(AFEBlock(), tracker, reconstruct=False)
    assert stage.frame_capabilities.emits_dimension == DIMENSION_REDUCED
    state = {"s_pars": full}
    state.update(stage.apply(state))
    assert state["s_pars"].shape == (m, 1, 1, n_freqs)

    out_reduced = RangeProfileBlock(bins=bins).apply(state)
    per_channel_reduced = out_reduced["range_profile"]
    agg_reduced = out_reduced["range_profile_agg"]
    assert per_channel_reduced.shape == (m, bins)
    # Every compressed measurement channel is STILL a linear combination of traces
    # that all share the same phase-vs-frequency factor, so its own peak survives at
    # the identical bin -- exactly, regardless of the (quantized) combining weights.
    assert torch.all(torch.argmax(per_channel_reduced, dim=1) == expected_bin)
    assert int(torch.argmax(agg_reduced).item()) == expected_bin


def test_range_profile_reduced_channel_count_differs_from_full():
    """Sanity check on the fixture itself: the compressed channel count is the
    measurement count M, not the element count N, so the parity above is a real
    test of the claim (same range bin, different basis) and not a shape coincidence."""
    n_elements, n_freqs, k0, m = 32, 32, 5, 8
    assert m != n_elements


# --------------------------------------------------------------------------------
# Frame-contract test: DIMENSION_ANY accepts reduced data where an angle
# product (DIMENSION_FULL, implicit) is refused.
# --------------------------------------------------------------------------------
def test_range_profile_declares_dimension_any():
    assert RangeProfileBlock.frame_capabilities.dimension == frames.DIMENSION_ANY


def test_range_profile_accepts_reduced_dimension_where_range_az_is_refused(torch_device):
    reduced = torch.randn(6, 1, 1, 16, dtype=torch.cfloat, device=torch_device)

    # DIMENSION_ANY: no error.
    frames.check_capabilities(reduced, RangeProfileBlock(), dimension=DIMENSION_REDUCED)
    RangeProfileBlock().apply({"s_pars": reduced})  # actually runs, no reconstruction needed

    # RangeAzBlock declares the historical full-dimension contract (default): refused.
    with pytest.raises(FrameContractError, match="DecompressBlock"):
        frames.check_capabilities(reduced, RangeAzBlock(), dimension=DIMENSION_REDUCED)


def test_range_profile_also_runs_on_full_dimension(torch_device):
    full = torch.randn(8, 1, 1, 16, dtype=torch.cfloat, device=torch_device)
    frames.check_capabilities(full, RangeProfileBlock(), dimension=DIMENSION_FULL)
    out = RangeProfileBlock(bins=16).apply({"s_pars": full})
    assert out["range_profile"].shape == (8, 16)
    assert out["range_profile_agg"].shape == (16,)


# --------------------------------------------------------------------------------
# Output shape / dtype / convention checks.
# --------------------------------------------------------------------------------
def test_range_profile_output_keys_and_shapes(torch_device):
    n_elements, n_freqs, bins = 12, 20, 8
    s_pars = torch.randn(n_elements, 1, 1, n_freqs, dtype=torch.cfloat, device=torch_device)
    out = RangeProfileBlock(bins=bins).apply({"s_pars": s_pars})

    assert set(out) == {"range_profile", "range_profile_agg"}
    assert out["range_profile"].shape == (n_elements, bins)
    assert out["range_profile_agg"].shape == (bins,)
    # Real, non-negative power (not raw complex amplitude, not dB -- dB conversion
    # is a display-layer concern, matching RangeAzBlock/RangeElBlock/FFTBlock).
    assert out["range_profile"].dtype == torch.float32
    assert torch.all(out["range_profile"] >= 0)
    assert out["range_profile_agg"].dtype == torch.float32
    assert torch.all(out["range_profile_agg"] >= 0)
    # Aggregate is the non-coherent (power) MEAN over channels.
    assert torch.allclose(out["range_profile_agg"], out["range_profile"].mean(dim=0))


def test_range_profile_reserved_keys_do_not_collide():
    """RangeProfileBlock's output keys must not be in Simulation's reserved set
    (e2e/simulation.py feed_forward), or a real pipeline run would raise."""
    reserved = {"U", "U_true", "s_pars", "PRX", "frame_layout", "signal_domain",
                "signal_dimension", "sensing_matrix", "aperture_shape", "tx_wave", "adc"}
    assert not ({"range_profile", "range_profile_agg"} & reserved)
