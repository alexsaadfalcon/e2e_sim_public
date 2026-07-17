"""Integration tests for the Simulation feed-forward pipeline (e2e.simulation).

These exercise the orchestrator end-to-end with synthetic frames (no Sionna, no .pkl),
across several block configurations -- including the no-AFE path that previously crashed.
"""

import pytest

torch = pytest.importorskip("torch")

from e2e.simulation import Simulation, get_U_true, perturb_basis
from e2e.blocks import (
    RFFEBlock,
    InterconnectBlock,
    AFEBlock,
    AdaOjaBlock,
    FFTBlock,
    SubspaceErrorBlock,
)

D = 16
N_RX = 1024


def _downstream():
    return [FFTBlock(bins=32), SubspaceErrorBlock()]


def test_get_u_true_shape(make_env_block):
    env = make_env_block(n_freqs=32)
    s_pars = env.get_S_pars()
    U = get_U_true(s_pars, D)
    assert U.shape == (N_RX, D)


def test_perturb_basis_stays_orthonormal():
    from e2e.subspace.algorithms import rand_orth_complex
    U = rand_orth_complex(64, 4)
    Up = perturb_basis(U)
    eye = Up.t().conj() @ Up
    assert torch.allclose(eye, torch.eye(4, dtype=eye.dtype, device=eye.device), atol=1e-4)


def test_tracker_warm_started_once_not_reset_every_frame(make_env_block, monkeypatch):
    """The online tracker must be initialized ONCE, then track -- not reset to the
    ground truth on every frame (the old short-circuit that made tracking a no-op)."""
    import e2e.simulation as sim_mod
    calls = {"n": 0}
    real = sim_mod.perturb_basis

    def counting_perturb(U):
        calls["n"] += 1
        return real(U)

    monkeypatch.setattr(sim_mod, "perturb_basis", counting_perturb)
    env = make_env_block(n_frames=4, n_freqs=32)
    sim = Simulation(env, _downstream(), D, subspace_block=AdaOjaBlock(N_RX, D))
    sim.run(n_steps=4)
    # Warm start fires exactly once across all frames, not once per frame.
    assert calls["n"] == 1


def test_pipeline_subspace_only_regression(make_env_block):
    """No AFE and no circuit block: this path raised NameError/TypeError before the fix."""
    env = make_env_block(n_frames=3, n_freqs=32)
    sim = Simulation(
        env, _downstream(), D,
        subspace_block=AdaOjaBlock(N_RX, D),
    )
    out = sim.run(n_steps=2)
    assert set(["fft", "subspace_err"]).issubset(out.keys())
    assert len(out["subspace_err"]) == 2
    assert all(torch.isfinite(v) for v in out["subspace_err"])


def test_pipeline_with_afe(make_env_block):
    env = make_env_block(n_frames=3, n_freqs=32)
    sim = Simulation(
        env, _downstream(), D,
        afe_block=AFEBlock(),
        subspace_block=AdaOjaBlock(N_RX, D),
    )
    out = sim.run(n_steps=2)
    assert len(out["subspace_err"]) == 2


def test_pipeline_with_interconnect(make_env_block):
    env = make_env_block(n_frames=3, n_freqs=32)
    sim = Simulation(
        env, _downstream(), D,
        interconnect_block=InterconnectBlock(case="case3"),
        afe_block=AFEBlock(),
        subspace_block=AdaOjaBlock(N_RX, D),
    )
    out = sim.run(n_steps=2)
    assert len(out["fft"]) == 2


def test_afe_requires_subspace_block(make_env_block):
    env = make_env_block(n_frames=2, n_freqs=32)
    with pytest.raises(ValueError):
        Simulation(env, _downstream(), D, afe_block=AFEBlock(), subspace_block=None)


def test_reset_returns_to_first_frame(make_env_block):
    env = make_env_block(n_frames=4, n_freqs=16)
    for _ in range(3):
        env.step()
    assert env.frame_counter == 3
    env.reset()
    assert env.frame_counter == 0


def test_pipeline_custom_array_shape_explicit(make_env_block):
    """Pipeline works for a non-32x32 array via the explicit array_shape arg."""
    env = make_env_block(n_frames=3, n_freqs=32, n_rx=256, array_shape=(16, 16))
    sim = Simulation(
        env, _downstream(), D,
        afe_block=AFEBlock(),
        subspace_block=AdaOjaBlock(256, D),
        array_shape=(16, 16),
    )
    out = sim.run(n_steps=2)
    assert len(out["subspace_err"]) == 2
    assert sim.n_rx_x == 16 and sim.n_rx_y == 16


def test_pipeline_array_shape_autoderived_from_env(make_env_block):
    """Simulation derives geometry from the environment block when not given one."""
    env = make_env_block(n_frames=2, n_freqs=16, n_rx=256, array_shape=(8, 32))
    sim = Simulation(env, _downstream(), D, subspace_block=AdaOjaBlock(256, D))
    assert (sim.n_rx_x, sim.n_rx_y) == (8, 32)
    out = sim.run(n_steps=1)
    assert len(out["fft"]) == 1


class _ReservedKeyBlock:
    """Dummy downstream block that emits a reserved pipeline key."""

    def __init__(self, key="s_pars"):
        self._key = key

    def apply(self, state_dict):
        return {self._key: state_dict["s_pars"]}


def test_downstream_block_reserved_key_raises(make_env_block):
    """A downstream block clobbering a reserved key must raise ValueError."""
    env = make_env_block(n_frames=2, n_freqs=32)
    sim = Simulation(
        env, [_ReservedKeyBlock("s_pars")], D,
        subspace_block=AdaOjaBlock(N_RX, D),
    )
    with pytest.raises(ValueError, match="reserved key"):
        sim.run(n_steps=1)


def test_multiple_chirps_assertion(make_env_block):
    """A frame with chirp dim (shape[2]) > 1 must trip the multi-chirp guard."""
    env = make_env_block(n_frames=1, n_freqs=32)
    sim = Simulation(
        env, _downstream(), D,
        subspace_block=AdaOjaBlock(N_RX, D),
    )
    sim.reset()

    # Stub get_S_pars to return a frame with two chirps (shape[2] == 2). The reshape of
    # s_pars_orig still needs n_rx_x * n_rx_y * F elements, so widen F accordingly is not
    # required here -- get_U_true / s_pars_orig view operate before the assertion only on
    # the original single-chirp frame, so feed a valid single-chirp frame for those and
    # swap in a multi-chirp tensor for the assertion path.
    real_get_s_pars = env.get_S_pars

    def _two_chirp_s_pars():
        base = real_get_s_pars()  # shape (n_rx, 1, 1, F)
        return torch.cat([base, base], dim=2)  # shape (n_rx, 1, 2, F)

    env.get_S_pars = _two_chirp_s_pars
    with pytest.raises(AssertionError, match="Multiple chirps not supported yet"):
        sim.feed_forward()


@pytest.mark.slow
def test_pipeline_with_rffe_circuit(make_env_block):
    """Full chain including the (heavier) RF front-end circuit model."""
    env = make_env_block(n_frames=2, n_freqs=64)
    sim = Simulation(
        env, _downstream(), D,
        circuit_block=RFFEBlock(n=N_RX),
        afe_block=AFEBlock(),
        subspace_block=AdaOjaBlock(N_RX, D),
    )
    out = sim.run(n_steps=1)
    assert len(out["subspace_err"]) == 1
