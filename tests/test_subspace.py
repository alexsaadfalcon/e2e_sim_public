"""Tests for subspace utilities and the online Oja tracker (e2e.subspace)."""

import pytest

torch = pytest.importorskip("torch")

from e2e.subspace.subspace_utils import subspace_dist_frob, subspace_dist
from e2e.subspace.algorithms import Oja, rand_orth_complex, gen_A_ada, randn_complex, device


def test_subspace_dist_zero_for_identical_basis():
    U = rand_orth_complex(20, 4)
    # float32 SVD leaves a small residual; loosen accordingly.
    assert subspace_dist_frob(U, U).item() == pytest.approx(0.0, abs=1e-2)
    assert subspace_dist(U, U).item() == pytest.approx(0.0, abs=1e-2)


def test_subspace_dist_positive_for_different_basis():
    U = rand_orth_complex(20, 4)
    V = rand_orth_complex(20, 4)
    assert subspace_dist_frob(U, V).item() > 0.0


def test_subspace_dist_frob_finite_for_edge_basis():
    # A slightly non-orthonormal basis can push k - ||A^H B||^2 marginally
    # negative; the clamp must keep the result finite (no sqrt(nan)).
    torch.manual_seed(0)
    U = rand_orth_complex(20, 4)
    # Perturb so it is no longer exactly orthonormal.
    pert = 1e-3 * (torch.randn_like(U.real) + 1j * torch.randn_like(U.real))
    U_edge = U + pert.to(U.dtype)
    out = subspace_dist_frob(U_edge, U_edge)
    assert torch.isfinite(out)


def test_subspace_dist_frob_stays_on_device():
    U = rand_orth_complex(20, 4)
    V = rand_orth_complex(20, 4)
    out = subspace_dist_frob(U, V)
    # Compare by device type (cuda/cpu), not the exact index: the module-level
    # `device` is "cuda" while tensors land on "cuda:0", so a raw == would spuriously
    # fail on a GPU box even though the result is correctly on the library device.
    assert out.device == U.device
    assert out.device.type == device.type
    # Still a scalar tensor.
    assert out.shape == torch.Size([])


def test_gen_A_ada_shape_and_rows():
    U = rand_orth_complex(30, 5)
    m = 12
    A = gen_A_ada(U, m)
    assert A.shape == (m, 30)
    # First k rows are U^H by construction.
    assert torch.allclose(A[:5, :], U.t().conj(), atol=1e-5)


def test_gen_A_ada_random_rows_match_deterministic_row_scale():
    # Regression guard: unnormalized random rows had norm ~sqrt(n) (measured
    # ~32 for n=1024) against unit-norm deterministic rows (U's columns are
    # orthonormal) -- a huge, unintended scale mismatch in the sensing matrix.
    U = rand_orth_complex(256, 5)
    m = 20
    A = gen_A_ada(U, m)
    det_norms = A[:5, :].norm(dim=1)
    rand_norms = A[5:, :].norm(dim=1)
    assert torch.allclose(det_norms, torch.ones_like(det_norms), atol=1e-4)
    assert torch.allclose(rand_norms, torch.ones_like(rand_norms), atol=1e-4)


def test_randn_complex_variance():
    # Measured E|X|^2 = 2.0 over >=1e6 samples (real/imag parts each unit
    # variance); pins the construction against an accidental doubling.
    x = randn_complex(1000, 1000, device=device)
    assert x.numel() >= 1_000_000
    e_abs2 = (x.abs() ** 2).mean().item()
    assert e_abs2 == pytest.approx(2.0, abs=0.05)


def test_oja_tracks_a_static_subspace():
    """Oja with adaptive sensing should reduce subspace error toward a fixed truth."""
    torch.manual_seed(0)
    d, k = 40, 4
    U_true = rand_orth_complex(d, k)
    oja = Oja(d, k, eta=1e0, fixed_step=True)

    err0 = subspace_dist_frob(oja.U, U_true).item()
    for _ in range(200):
        coeffs = torch.randn(k, 8, dtype=torch.cfloat, device=device)
        V = U_true @ coeffs
        A = gen_A_ada(oja.U.clone(), k * 2)
        X = A @ V
        oja.add_data(X, A)
    err1 = subspace_dist_frob(oja.U, U_true).item()
    assert err1 < err0  # learning made progress


def test_oja_zero_frame_does_not_poison_the_basis():
    """Regression for the NaN-poisoning bug (review wave 1, CONFIRMED): a degenerate
    all-zero measurement frame made grad/||grad|| = 0/0 = NaN, which then corrupted
    U through every subsequent orth() with no exception. The guard must skip the
    uninformative frame (U unchanged, still finite) and later good frames must
    still update normally."""
    import torch

    d, k, m = 16, 2, 8
    oja = Oja(d, k, eta=0.1, fixed_step=True)
    U_before = oja.U.clone()
    A = gen_A_ada(oja.U, m)

    oja.add_data(torch.zeros(m, 4, dtype=torch.cfloat, device=device), A)
    assert torch.isfinite(oja.U).all(), "zero frame produced non-finite basis"
    assert torch.allclose(oja.U, U_before), "zero frame should be a no-op update"

    X = A @ randn_complex(d, 4)
    oja.add_data(X, A)
    assert torch.isfinite(oja.U).all(), "tracker corrupted after the zero frame"
