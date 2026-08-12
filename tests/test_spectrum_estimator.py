"""Tests for the measurement-domain gate-signal estimator (e2e.subspace.spectrum_estimator).

Accuracy numbers referenced in the tolerances below were MEASURED (not derived from
theory) with the harness in `_make_frame` at d=1024, m=512, k=8, n_freqs=64, 20 seeds:
anchor (top-k) singular values: max relative error ~1.8e-5 (mean ~1.0e-5); the boundary
(k+1-th) singular value: max relative error ~5.2% (mean ~1.0%) -- see spectrum_estimator's
module docstring for why the boundary value carries strictly more error than the anchor
block (it's a random-subspace-sampling estimate of the tail, debiased only in expectation,
not an exact projection like the anchor).
"""

import pytest

torch = pytest.importorskip("torch")

from e2e.subspace.algorithms import rand_orth_complex, gen_A_ada
from e2e.subspace.spectrum_estimator import estimate_boundary_spectrum


def _make_frame(d, n_freqs, S_vals, device, seed):
    """Build V = U_o diag(S) V_o^H with a hand-set descending spectrum `S_vals`."""
    torch.manual_seed(seed)
    r = len(S_vals)
    U_o = rand_orth_complex(d, r, device=device)
    V_o = rand_orth_complex(n_freqs, r, device=device)
    S = torch.tensor(S_vals, dtype=torch.cfloat, device=device)
    V = U_o @ torch.diag(S) @ V_o.conj().T
    return V, U_o, S


def _add_noise(X, snr_db, seed):
    """Additive complex measurement noise at the given SNR (relative to X's mean power)."""
    torch.manual_seed(seed)
    sig_power = (X.abs() ** 2).mean()
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = (torch.randn_like(X.real) + 1j * torch.randn_like(X.real)) * (noise_power / 2).sqrt()
    return X + noise


# ----------------------------------------------------------------------- (1) exactness

def test_exactness_noiseless_known_spectrum(torch_device):
    """With U anchored exactly on V's true top-k subspace (the tracker's converged
    operating point) and no measurement noise, the anchor block recovers S[:k] almost
    exactly, and the debiased complement block recovers S[k] (the boundary value) to
    within the measured tolerance -- see module docstring for the numbers."""
    d, n_freqs, k, m = 1024, 64, 8, 512
    S_vals = [10.0] * 7 + [3.0, 0.5] + [0.3] * 6
    V, U_o, S = _make_frame(d, n_freqs, S_vals, torch_device, seed=0)
    U = U_o[:, :k]
    A = gen_A_ada(U, m)
    X = A @ V

    out = estimate_boundary_spectrum(X, A, k)
    assert out["sv_head"].shape == (k + 1,)
    assert out["sv_head"].device.type == torch_device.type

    rel_err = (out["sv_head"] - S[: k + 1].real).abs() / S[: k + 1].real
    # Anchor block (indices 0..k-1): an exact projection when U is aligned with the
    # true subspace -- measured max ~1.8e-5 over 20 seeds; generous 1e-3 margin.
    assert rel_err[:k].max().item() < 1e-3
    # Boundary value (index k): a debiased random-subspace estimate of the tail --
    # measured max ~5.2% over 20 seeds; generous 15% margin.
    assert rel_err[k].item() < 0.15


# ----------------------------------------------------------------------- (2) the gate test

# k=8 -> S_vals lays out indices 0..6 (7 values) as the "top" block, then the boundary
# pair at indices 7 (=S[k-1]) and 8 (=S[k]), which is what sv_gap_norm compares.
_GATE_CASES = {
    # sv_gap_norm = 0: an exactly degenerate cluster straddling the k cutoff.
    "degenerate_cluster": [10.0] * 7 + [1.0, 1.0] + [0.5] * 6,
    # sv_gap_norm ~ 5e-4: an insignificant tail (S[k-1], S[k] both << S[0]).
    "insignificant_tail": [10.0] * 7 + [0.02, 0.015] + [0.01] * 6,
    # sv_gap_norm ~ 0.25 / 0.03: safely healthy (well above the 0.01 gate threshold).
    "healthy_separated": [10.0] * 7 + [3.0, 0.5] + [0.3] * 6,
    "healthy_moderate": [10.0] * 7 + [1.5, 1.2] + [0.5] * 6,
}

_GATE_THRESHOLD = 0.01  # matches AdaOjaBlock's default gap_threshold

# d=512, m=256 (rather than the 1024/512 "production" scale) keeps the parametrized/
# repeated-trial tests fast while still giving the complement block enough rows
# (m - k = 248 of a d - k = 504 dim complement) to land the estimated gap safely away
# from the 0.01 threshold in both regimes: measured (30 trials/case, see report) worst
# case degenerate/insignificant-tail estimate 0.0055 and worst case healthy estimate
# 0.0235 -- comfortably on the correct side either way.
_GATE_D, _GATE_NFREQS, _GATE_K, _GATE_M = 512, 48, 8, 256

# Fixed (not hash()-derived -- Python randomizes str hashing per-process by default,
# which would make "deterministic" seeds vary run to run) per-case seed offsets.
_GATE_SEED_BASE = {name: 500 + 10 * i for i, name in enumerate(_GATE_CASES)}


@pytest.mark.parametrize("name", list(_GATE_CASES))
def test_gate_decision_matches_oracle_both_regimes(torch_device, name):
    """Thresholding the estimated sv_gap_norm at 0.01 gives the same trigger decision as
    thresholding the oracle gap, across a degenerate-cluster collapse, an
    insignificant-tail collapse, and two healthy (non-collapsed) spectra."""
    d, n_freqs, k, m = _GATE_D, _GATE_NFREQS, _GATE_K, _GATE_M
    S_vals = _GATE_CASES[name]

    for trial in range(3):
        seed = _GATE_SEED_BASE[name] + trial
        V, U_o, S = _make_frame(d, n_freqs, S_vals, torch_device, seed=seed)
        U = U_o[:, :k]
        A = gen_A_ada(U, m)
        X = A @ V

        oracle_gap = ((S[k - 1] - S[k]) / S[0]).real.item()
        oracle_trigger = oracle_gap < _GATE_THRESHOLD

        out = estimate_boundary_spectrum(X, A, k)
        est_trigger = out["sv_gap_norm"] < _GATE_THRESHOLD
        assert est_trigger == oracle_trigger, (
            f"{name} trial {trial}: oracle_gap={oracle_gap} est_gap={out['sv_gap_norm']}"
        )


# ----------------------------------------------------------------------- (3) noise robustness

@pytest.mark.parametrize("name", list(_GATE_CASES))
def test_gate_decision_robust_to_30db_noise(torch_device, name):
    """With additive measurement noise at ~30 dB SNR, the trigger decision on these
    safely-separated spectra is unchanged."""
    d, n_freqs, k, m = _GATE_D, _GATE_NFREQS, _GATE_K, _GATE_M
    S_vals = _GATE_CASES[name]

    for trial in range(3):
        seed = _GATE_SEED_BASE[name] + 100 + trial
        V, U_o, S = _make_frame(d, n_freqs, S_vals, torch_device, seed=seed)
        U = U_o[:, :k]
        A = gen_A_ada(U, m)
        X = A @ V

        oracle_gap = ((S[k - 1] - S[k]) / S[0]).real.item()
        oracle_trigger = oracle_gap < _GATE_THRESHOLD

        X_noisy = _add_noise(X, snr_db=30.0, seed=seed + 1)
        out_noisy = estimate_boundary_spectrum(X_noisy, A, k)
        est_trigger_noisy = out_noisy["sv_gap_norm"] < _GATE_THRESHOLD
        assert est_trigger_noisy == oracle_trigger, (
            f"{name} trial {trial}: oracle_gap={oracle_gap} "
            f"noisy_est_gap={out_noisy['sv_gap_norm']}"
        )


# ------------------------------------------------------------------ misc / edge behavior

def test_requires_k_less_than_m():
    d, k, m = 32, 8, 8  # m == k: no complement rows
    U = rand_orth_complex(d, k)
    A = gen_A_ada(U, m)
    X = torch.zeros(m, 4, dtype=torch.cfloat, device=A.device)
    with pytest.raises(ValueError):
        estimate_boundary_spectrum(X, A, k)
