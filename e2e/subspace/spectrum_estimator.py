"""Measurement-domain estimator of the subspace-tracking gate signal.

`e2e.simulation.rank_diagnostic`'s "Honesty note" (and `AdaOjaBlock.effective_n_refine`'s
docstring) flag that `sv_gap_norm` -- the diagnostic `gap_response` reacts to -- is
ORACLE-sourced: it comes from a full-frame SVD the simulator performs anyway for scoring,
not from anything a deployed receiver actually has. This module prototypes the estimator
those docstrings describe but never implemented: recovering `sv_gap_norm` (and the
boundary singular values it is built from) from the measurement domain alone -- `X = A V`
and the known sensing matrix `A` -- with no access to the full aperture `V`.

FORMULATION
-----------
`gen_A_ada(U, m)` (e2e.subspace.algorithms) builds `A` (m x d) as `k` deterministic
"anchor" rows `U^H` (U the tracker's current d x k orthonormal basis estimate) stacked on
`m - k` random "complement" rows, each unit-norm and explicitly orthogonalized against U
(but NOT against each other). Consequently `A` is not orthonormal: its Gram
`A A^H` is exactly block-diagonal, `[[I_k, 0], [0, G]]`, where `G = B^H B` (B the raw
complement columns) is a generically well-conditioned but non-identity (m-k)x(m-k)
Gram matrix -- the anchor block is an exact isometry onto U's directions, the complement
block is not.

Naively eigendecomposing `X X^H` (the sample covariance of the measurements) would treat
both blocks as if they were isometries, over-weighting the complement rows' contribution
relative to the anchor rows'. The correct comparison is the GENERALIZED eigenproblem
`X X^H w = lambda (A A^H) w`, solved here via Cholesky whitening: `A A^H = L L^H`,
`X_w = L^{-1} X`, so `eig(X_w X_w^H) = eig(L^{-1} X X^H L^{-H})` gives the generalized
eigenvalues directly as ordinary singular values of `X_w`. Cost: forming `A A^H` is
`O(m^2 d)` (the only step touching the full aperture dimension `d`; everything after is
`O(m^2)`-`O(m^2 n_freqs)` in `m` alone) -- with `m` on the order of a few hundred and
`d` ~1e3, this is orders of magnitude cheaper than the `O(d^2 n_freqs)` SVD of the full
aperture matrix `V` that the oracle diagnostic relies on, and critically the receiver
never needs `V` itself.

Whitening alone is not sufficient, though: even after whitening, the complement block's
`m - k` rows only span a RANDOM `(m - k)`-dimensional subspace of the `(d - k)`-dimensional
orthogonal complement of U (`B` is orthogonalized against U, not against V's true
structure). A fixed direction's energy captured by a uniformly random `p`-dimensional
subspace of an `n`-dimensional space concentrates around `p / n` (principal-angle
concentration for random subspaces), so -- unlike the anchor block, which is an EXACT,
undistorted projection whenever U is aligned with the tracked subspace -- the complement
block's raw singular values are attenuated in expectation by `sqrt((m - k) / (d - k))`
relative to the true tail singular values of V. This is `A`'s distortion referenced in
`rank_diagnostic`'s honesty note; we correct for it by rescaling the whitened complement
rows by the inverse factor before the final SVD. (This is a first-order/expectation-based
correction, not an exact deconvolution -- see the module docstring's failure-regime notes
in the test file and the task report for measured accuracy.)

The estimator therefore assumes (matching `gen_A_ada`'s convention, and NOT re-derived
generically from `A` alone): `A`'s first `k` rows are the anchor block `U^H` for some
orthonormal `U`, and rows `k:` are the complement block.
"""

import torch


def estimate_boundary_spectrum(X, A, k):
    """Estimate the top-(k+1) singular values of the (unobserved) aperture matrix V from
    the measurements `X = A V` (m x n_freqs) and the known sensing matrix `A` (m x d),
    without ever forming or decomposing V.

    `A` is assumed to follow `gen_A_ada`'s convention: rows `[:k]` are the deterministic
    anchor block `U^H` (U orthonormal), rows `[k:]` are the random complement block
    (unit-norm columns of B, orthogonalized against U but not against each other).

    Returns a dict:
      - `sv_head`: estimated top-`min(k+1, m, n_freqs)` singular values of V, descending
        (real tensor, on `X`'s device).
      - `sv_gap_norm`: estimated `(sv_head[k-1] - sv_head[k]) / sv_head[0]` -- the same
        quantity `e2e.simulation.rank_diagnostic` computes from the oracle spectrum.
        `float('nan')` if fewer than `k + 1` singular values are available.
      - `correction_factor`: the `sqrt((d - k) / (m - k))` complement-block rescaling
        applied (see module docstring).
      - `sv_full`: the full (whitened + rescaled) singular value spectrum, length
        `min(m, n_freqs)` -- occasionally useful for diagnostics beyond the boundary pair.
    """
    m, d = A.shape
    if not (0 < k < m):
        raise ValueError(f"need 0 < k < m (anchor rows < total rows), got k={k}, m={m}")

    # Gram of A: block-diagonal [I_k, G] for a gen_A_ada-built A (verified, not assumed,
    # by test_spectrum_estimator's anchor-orthonormality checks); computed generically
    # here (no shortcut exploiting that structure) so numerical noise in A doesn't matter.
    Gram_A = A @ A.conj().T
    L = torch.linalg.cholesky(Gram_A)
    # X_w = L^{-1} X: the whitened measurements, i.e. the generalized-eigenproblem change
    # of variables described in the module docstring.
    X_w = torch.linalg.solve_triangular(L, X, upper=False)

    # De-bias the complement block only -- the anchor rows are an exact isometry (U
    # orthonormal) and need no correction.
    corr = ((d - k) / (m - k)) ** 0.5
    weight = torch.ones(m, dtype=torch.float32, device=X.device)
    weight[k:] = corr
    X_c = weight.unsqueeze(1) * X_w

    sv_full = torch.linalg.svdvals(X_c)
    n_head = min(k + 1, sv_full.shape[0])
    sv_head = sv_full[:n_head]

    if n_head >= k + 1 and sv_head[0] > 0:
        sv_gap_norm = float(((sv_head[k - 1] - sv_head[k]) / sv_head[0]).item())
    else:
        sv_gap_norm = float("nan")

    return {
        "sv_head": sv_head,
        "sv_gap_norm": sv_gap_norm,
        "correction_factor": corr,
        "sv_full": sv_full,
    }
