"""Unit tests for individual pipeline blocks (e2e.blocks)."""

import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from e2e.blocks import (
    InterconnectBlock,
    AFEBlock,
    AdaOjaBlock,
    FFTBlock,
    RangeAzBlock,
    RangeElBlock,
    SubspaceErrorBlock,
)
from e2e.subspace.algorithms import rand_orth_complex
from e2e.blocks import device


@pytest.fixture
def state_dict(n_freqs):
    """A minimal state_dict shaped like Simulation builds, on the library device."""
    s_pars = torch.randn(32, 32, 1, n_freqs, dtype=torch.cfloat, device=device)
    U_true = rand_orth_complex(1024, 16)
    return {
        "s_pars": s_pars,
        "U_true": U_true,
        "U": U_true.clone(),
        "PRX": None,
    }


@pytest.mark.parametrize("block_cls,key", [
    (FFTBlock, "fft"),
    (RangeAzBlock, "range_az"),
    (RangeElBlock, "range_el"),
])
def test_transform_blocks_emit_expected_key(block_cls, key, state_dict):
    out = block_cls(bins=32).apply(state_dict)
    assert key in out
    assert torch.is_tensor(out[key])
    assert torch.all(torch.isfinite(torch.abs(out[key])))
    # Must be a full 2D [bins, bins] map. Regression guard: RangeAz/RangeEl
    # previously passed torch.fft.fft(x, 1) (n=1) which collapsed the range axis
    # to a single bin -> [bins, 1], silently destroying the range dimension.
    assert out[key].shape == (32, 32)


def test_subspace_error_zero_for_identical_basis(state_dict):
    out = SubspaceErrorBlock().apply(state_dict)
    # float32 QR leaves a small residual in subspace_dist_frob; 1e-3 is too tight
    # (the value is ~1e-3..1e-2 and varies by RNG/device). Match test_subspace.py's 1e-2.
    assert out["subspace_err"].item() == pytest.approx(0.0, abs=1e-2)


def test_subspace_error_positive_for_perturbed_basis(state_dict):
    state_dict["U"] = rand_orth_complex(1024, 16)
    out = SubspaceErrorBlock().apply(state_dict)
    assert out["subspace_err"].item() > 0.0


def test_interconnect_case3_is_identity():
    frame = torch.randn(32, 32, 1, 64, dtype=torch.cfloat, device=device)
    out = InterconnectBlock(case="case3").apply_interconnect(frame)
    assert torch.allclose(out, frame)


def test_interconnect_default_filters_frame():
    frame = torch.randn(32, 32, 1, 64, dtype=torch.cfloat, device=device)
    out = InterconnectBlock(case="synthetic").apply_interconnect(frame)
    assert out.shape == frame.shape
    # A non-trivial windowing filter should change the frame.
    assert not torch.allclose(out, frame)


def test_afe_block_matmul_and_reconstruct_shapes():
    afe = AFEBlock(exp=5, mantissa=6)
    n, m, F = 64, 32, 16
    V = torch.randn(n, F, dtype=torch.cfloat, device=device)
    A = torch.randn(m, n, dtype=torch.cfloat, device=device)
    Aq, X = afe.apply_mat_mul(A, V)
    assert Aq.shape == A.shape
    assert X.shape == (m, F)
    Xt = afe.reconstruct(Aq, X)
    assert Xt.shape == (n, F)


def test_ada_oja_block_update_runs():
    block = AdaOjaBlock(n=64, d=4)
    block.oja.U = rand_orth_complex(64, 4)
    A = block.gen_A_ada()
    V = torch.randn(64, 8, dtype=torch.cfloat, device=device)
    X = A @ V
    U_before = block.oja.U.clone()
    block.update(X, A)
    assert block.oja.U.shape == U_before.shape


# --------------------------------------------------- SionnaEnvironmentBlock link selection

def _write_multilink_munich(tmp_path, monkeypatch, links, n_frames=3, n_rx=4, n_freqs=8):
    """Write a multi-link dict pkl and point the 'munich' scenario at it."""
    import e2e.environment.sionna_iterator as si

    r = np.random.default_rng(1)
    data = {
        name: (r.standard_normal((n_frames, n_rx, 1, 1, n_freqs))
               + 1j * r.standard_normal((n_frames, n_rx, 1, 1, n_freqs))).astype(np.complex64)
        for name in links
    }
    path = tmp_path / "munich_multilink.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)
    # Factory reads the module-global path at call time, so patching it redirects the block.
    monkeypatch.setattr(si, "SIONNA_MUNICH_PATH", str(path))
    return data


def test_env_block_default_selects_first_link(tmp_path, monkeypatch):
    from e2e.blocks import SionnaEnvironmentBlock

    data = _write_multilink_munich(tmp_path, monkeypatch, ["tx0", "tx1"])
    block = SionnaEnvironmentBlock("munich")
    assert block.sionna_iterator.link == "tx0"
    got = block.get_S_pars().detach().cpu().numpy()
    np.testing.assert_array_equal(got, data["tx0"][0])


def test_env_block_explicit_link_selects_right_one(tmp_path, monkeypatch):
    from e2e.blocks import SionnaEnvironmentBlock

    data = _write_multilink_munich(tmp_path, monkeypatch, ["tx0", "tx1"])
    block = SionnaEnvironmentBlock("munich", link="tx1")
    assert block.link == "tx1"
    assert block.sionna_iterator.link == "tx1"
    got = block.get_S_pars().detach().cpu().numpy()
    np.testing.assert_array_equal(got, data["tx1"][0])


def test_env_block_single_array_pkl_still_works(tmp_path, monkeypatch):
    import e2e.environment.sionna_iterator as si
    from e2e.blocks import SionnaEnvironmentBlock

    r = np.random.default_rng(2)
    arr = (r.standard_normal((3, 4, 1, 1, 8))
           + 1j * r.standard_normal((3, 4, 1, 1, 8))).astype(np.complex64)
    path = tmp_path / "munich_single.pkl"
    with open(path, "wb") as f:
        pickle.dump(arr, f)
    monkeypatch.setattr(si, "SIONNA_MUNICH_PATH", str(path))

    # Default and an explicit (ignored) link both work for legacy single-array pkls.
    block = SionnaEnvironmentBlock("munich")
    assert block.sionna_iterator.links is None
    np.testing.assert_array_equal(block.get_S_pars().detach().cpu().numpy(), arr[0])

    block2 = SionnaEnvironmentBlock("munich", link="anything")
    np.testing.assert_array_equal(block2.get_S_pars().detach().cpu().numpy(), arr[0])
