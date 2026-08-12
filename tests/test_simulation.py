"""Integration tests for the Simulation feed-forward pipeline (e2e.simulation).

These exercise the orchestrator end-to-end with synthetic frames (no Sionna, no .pkl),
across several block configurations -- including the no-AFE path that previously crashed.
"""

import pytest

torch = pytest.importorskip("torch")

from e2e.simulation import Simulation, get_U_true, perturb_basis, rank_diagnostic
from e2e.blocks import (
    RFFEBlock,
    InterconnectBlock,
    AFEBlock,
    AdaOjaBlock,
    FFTBlock,
    RangeAzBlock,
    SubspaceErrorBlock,
)

K = 16
N_RX = 1024


def _downstream():
    return [FFTBlock(bins=32), SubspaceErrorBlock()]


def test_get_u_true_shape(make_env_block):
    env = make_env_block(n_freqs=32)
    s_pars = env.get_S_pars()
    U = get_U_true(s_pars, K)
    assert U.shape == (N_RX, K)


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
    sim = Simulation(env, _downstream(), K, subspace_block=AdaOjaBlock(N_RX, K))
    sim.run(n_steps=4)
    # Warm start fires exactly once across all frames, not once per frame.
    assert calls["n"] == 1


def test_warm_start_true_frame0_is_perturbed_ground_truth(make_env_block, monkeypatch):
    """warm_start=True (the default) reproduces the historical one-time warm start
    exactly: oja.U is set to perturb_basis(U_true) from frame 0. The tracker's
    per-frame update() is stubbed so we isolate the warm-start assignment itself
    from the subsequent online tracking step that also runs inside frame 0."""
    env = make_env_block(n_frames=2, n_freqs=32)
    subspace_block = AdaOjaBlock(N_RX, K)
    monkeypatch.setattr(subspace_block, "update", lambda *a, **k: None)
    sim = Simulation(env, _downstream(), K, subspace_block=subspace_block)  # warm_start=True default
    assert sim.warm_start is True

    torch.manual_seed(42)
    expected = perturb_basis(get_U_true(env.get_S_pars(), K))

    torch.manual_seed(42)
    sim.run(n_steps=1)
    assert torch.allclose(sim.subspace_block.oja.U, expected, atol=1e-5)


def test_warm_start_false_frame0_is_cold_start(make_env_block, monkeypatch):
    """warm_start=False leaves Oja's own random cold-start basis untouched -- no
    peek at ground truth. update() is stubbed to isolate the (non-)assignment from
    the subsequent tracking step."""
    env = make_env_block(n_frames=2, n_freqs=32)
    subspace_block = AdaOjaBlock(N_RX, K)
    cold_start_U = subspace_block.oja.U.clone()
    monkeypatch.setattr(subspace_block, "update", lambda *a, **k: None)
    sim = Simulation(env, _downstream(), K, subspace_block=subspace_block, warm_start=False)
    assert sim.warm_start is False

    sim.run(n_steps=1)
    # untouched: still exactly the basis Oja's constructor drew (rand_orth_complex)
    assert torch.allclose(sim.subspace_block.oja.U, cold_start_U)

    # sanity: this is nowhere near the ground-truth-derived warm start
    warm = perturb_basis(get_U_true(env.get_S_pars(), K))
    assert not torch.allclose(sim.subspace_block.oja.U, warm, atol=1e-2)


def test_warm_start_false_gives_materially_worse_frame0_subspace_err(make_env_block):
    """Cold-start (no peek at ground truth) tracking is honestly worse on frame 0
    than the warm-started default -- subspace_err should reflect that plainly."""
    env_warm = make_env_block(n_frames=2, n_freqs=32)
    sim_warm = Simulation(env_warm, _downstream(), K, subspace_block=AdaOjaBlock(N_RX, K), warm_start=True)
    out_warm = sim_warm.run(n_steps=1)

    env_cold = make_env_block(n_frames=2, n_freqs=32)
    sim_cold = Simulation(env_cold, _downstream(), K, subspace_block=AdaOjaBlock(N_RX, K), warm_start=False)
    out_cold = sim_cold.run(n_steps=1)

    warm_err = float(out_warm["subspace_err"][0])
    cold_err = float(out_cold["subspace_err"][0])
    assert cold_err > 2 * warm_err
    assert cold_err > 0.5  # cold start is a near-random basis vs. the true subspace


def test_rank_diagnostic_values():
    """rank_diagnostic on a hand-built spectrum with a clean gap at index 3."""
    S = torch.tensor([10.0, 9.0, 8.0, 0.001, 0.0005])
    diag = rank_diagnostic(S, k=3)
    assert diag["effective_rank"] == 3
    assert diag["rank_ok"] is True
    assert diag["sv_gap_at_k"] == pytest.approx(8.0 / 0.001, rel=1e-3)

    diag_over = rank_diagnostic(S, k=4)
    assert diag_over["rank_ok"] is False


def test_low_rank_frame_triggers_rank_warning_and_outputs(make_env_block, torch_device):
    """A rank-2 synthetic frame with k > 2 must report rank_ok=False and warn once."""
    n_freq = 32
    env = make_env_block(n_frames=2, n_freqs=n_freq)
    real_get_s_pars = env.get_S_pars

    def _low_rank_s_pars():
        base = real_get_s_pars()
        n_rx = base.shape[0]
        a = torch.randn(n_rx, 2, dtype=torch.complex64, device=torch_device)
        b = torch.randn(2, n_freq, dtype=torch.complex64, device=torch_device)
        return (a @ b).view(n_rx, 1, 1, n_freq)

    env.get_S_pars = _low_rank_s_pars
    sim = Simulation(env, _downstream(), K, subspace_block=AdaOjaBlock(N_RX, K))
    with pytest.warns(UserWarning, match="exceeds frame effective rank"):
        out = sim.run(n_steps=2)

    assert out["rank_ok"][0] is False
    assert out["effective_rank"][0] <= 2
    # The gap at k sits between two noise-floor singular values of an exactly-rank-2
    # frame: the ratio is >= 1 by construction (descending order) but can be EXACTLY
    # 1.0 when the trailing SVs tie -- CPU LAPACK does this where CUDA's SVD returns
    # a hair above 1.0 (caught on CI). The semantic signal is rank_ok/effective_rank
    # above; this line only sanity-checks the diagnostic is well-formed.
    assert out["sv_gap_at_k"][0] >= 1.0


def test_full_rank_frame_no_rank_warning(make_env_block):
    """A full-rank synthetic (Gaussian noise) frame with k <= effective rank must not
    warn, and rank_ok must be True."""
    env = make_env_block(n_frames=2, n_freqs=32)
    sim = Simulation(env, _downstream(), K, subspace_block=AdaOjaBlock(N_RX, K))
    import warnings as _warnings
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        out = sim.run(n_steps=2)
    assert not any("exceeds frame effective rank" in str(w.message) for w in caught)
    assert out["rank_ok"][0] is True
    assert "effective_rank" in out and "sv_gap_at_k" in out


def test_pipeline_subspace_only_regression(make_env_block):
    """No AFE and no circuit block: this path raised NameError/TypeError before the fix."""
    env = make_env_block(n_frames=3, n_freqs=32)
    sim = Simulation(
        env, _downstream(), K,
        subspace_block=AdaOjaBlock(N_RX, K),
    )
    out = sim.run(n_steps=2)
    assert set(["fft", "subspace_err"]).issubset(out.keys())
    assert len(out["subspace_err"]) == 2
    assert all(torch.isfinite(v) for v in out["subspace_err"])


def test_pipeline_with_afe(make_env_block):
    env = make_env_block(n_frames=3, n_freqs=32)
    sim = Simulation(
        env, _downstream(), K,
        afe_block=AFEBlock(),
        subspace_block=AdaOjaBlock(N_RX, K),
    )
    out = sim.run(n_steps=2)
    assert len(out["subspace_err"]) == 2


def test_pipeline_with_interconnect(make_env_block):
    env = make_env_block(n_frames=3, n_freqs=32)
    sim = Simulation(
        env, _downstream(), K,
        interconnect_block=InterconnectBlock(case="case3"),
        afe_block=AFEBlock(),
        subspace_block=AdaOjaBlock(N_RX, K),
    )
    out = sim.run(n_steps=2)
    assert len(out["fft"]) == 2


def test_afe_requires_subspace_block(make_env_block):
    env = make_env_block(n_frames=2, n_freqs=32)
    with pytest.raises(ValueError):
        Simulation(env, _downstream(), K, afe_block=AFEBlock(), subspace_block=None)


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
        env, _downstream(), K,
        afe_block=AFEBlock(),
        subspace_block=AdaOjaBlock(256, K),
        array_shape=(16, 16),
    )
    out = sim.run(n_steps=2)
    assert len(out["subspace_err"]) == 2
    assert sim.n_rx_x == 16 and sim.n_rx_y == 16


def test_pipeline_array_shape_autoderived_from_env(make_env_block):
    """Simulation derives geometry from the environment block when not given one."""
    env = make_env_block(n_frames=2, n_freqs=16, n_rx=256, array_shape=(8, 32))
    sim = Simulation(env, _downstream(), K, subspace_block=AdaOjaBlock(256, K))
    assert (sim.n_rx_x, sim.n_rx_y) == (8, 32)
    out = sim.run(n_steps=1)
    assert len(out["fft"]) == 1


def test_pipeline_runs_without_subspace_block(make_env_block):
    """subspace_block=None skips the measurement stage: FFT/range products still work
    for users who only want the range/angle maps, with no subspace tracking."""
    env = make_env_block(n_frames=2, n_freqs=16)
    sim = Simulation(env, [FFTBlock(bins=16), RangeAzBlock(bins=16)], K)
    out = sim.run(n_steps=2)
    assert len(out["fft"]) == 2
    assert len(out["range_az"]) == 2


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
        env, [_ReservedKeyBlock("s_pars")], K,
        subspace_block=AdaOjaBlock(N_RX, K),
    )
    with pytest.raises(ValueError, match="reserved key"):
        sim.run(n_steps=1)


def test_multiple_chirps_assertion(make_env_block):
    """A multi-chirp frame must stop at the first stage that declares chirps='single'.

    Since the per-block capability contract landed, the chirp axis is no longer
    rejected pipeline-wide: it flows through the element-wise stages and trips at
    MeasurementStage, whose measurement matrix is defined for one chirp. The error
    names that stage and the blocks it drives.
    """
    env = make_env_block(n_frames=1, n_freqs=32)
    sim = Simulation(
        env, _downstream(), K,
        subspace_block=AdaOjaBlock(N_RX, K),
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
    with pytest.raises(ValueError, match=r"MeasurementStage\[AdaOjaBlock\]: multiple chirps not supported yet"):
        sim.feed_forward()


def test_mimo_assertion(make_env_block):
    """A frame with a TX dim (shape[1]) > 1 must trip the named no-MIMO guard."""
    env = make_env_block(n_frames=1, n_freqs=32)
    sim = Simulation(
        env, _downstream(), K,
        subspace_block=AdaOjaBlock(N_RX, K),
    )
    sim.reset()

    real_get_s_pars = env.get_S_pars

    def _two_tx_s_pars():
        base = real_get_s_pars()  # shape (n_rx, 1, 1, F)
        return torch.cat([base, base], dim=1)  # shape (n_rx, 2, 1, F)

    env.get_S_pars = _two_tx_s_pars
    with pytest.raises(ValueError, match="GridStage: MIMO not supported yet"):
        sim.feed_forward()


class _NoOpStage:
    """Custom serial stage: tags 's_pars' so we can prove it ran and its edit flowed
    through the rest of the pipeline (via the `serial_stages` composability hook)."""

    def __init__(self):
        self.ran = False

    def apply(self, state):
        self.ran = True
        return {"s_pars": state["s_pars"] * 2}


def test_composability_custom_serial_stage_runs_and_flows_through(make_env_block):
    """A custom stage passed via serial_stages= replaces the auto-built list, runs in
    feed_forward, and its s_pars edit is visible to later stages/downstream blocks."""
    from e2e.blocks import GridStage, MeasurementStage

    env = make_env_block(n_frames=2, n_freqs=32)
    custom = _NoOpStage()
    subspace_block = AdaOjaBlock(N_RX, K)
    sim = Simulation(
        env, _downstream(), K,
        subspace_block=subspace_block,
        serial_stages=[custom, GridStage((32, 32)), MeasurementStage(None, subspace_block)],
    )
    out = sim.run(n_steps=1)
    assert custom.ran
    assert len(out["subspace_err"]) == 1
    assert all(torch.isfinite(v) for v in out["subspace_err"])


def test_legacy_args_build_expected_stage_sequence(make_env_block):
    """Legacy positional/kwarg construction auto-builds serial_stages in the right
    order, skipping stages whose backing block is None."""
    from e2e.blocks import CircuitStage, GridStage, InterconnectStage, MeasurementStage

    env = make_env_block(n_frames=1, n_freqs=32)

    # No circuit, no interconnect -> [GridStage, MeasurementStage]
    sim = Simulation(env, _downstream(), K, subspace_block=AdaOjaBlock(N_RX, K))
    assert [type(s) for s in sim.serial_stages] == [GridStage, MeasurementStage]

    # Full stack -> [CircuitStage, GridStage, InterconnectStage, MeasurementStage]
    sim2 = Simulation(
        env, _downstream(), K,
        circuit_block=RFFEBlock(n=N_RX),
        interconnect_block=InterconnectBlock(case="case3"),
        afe_block=AFEBlock(),
        subspace_block=AdaOjaBlock(N_RX, K),
    )
    assert [type(s) for s in sim2.serial_stages] == [
        CircuitStage, GridStage, InterconnectStage, MeasurementStage,
    ]


@pytest.mark.slow
def test_pipeline_with_rffe_circuit(make_env_block):
    """Full chain including the (heavier) RF front-end circuit model."""
    env = make_env_block(n_frames=2, n_freqs=64)
    sim = Simulation(
        env, _downstream(), K,
        circuit_block=RFFEBlock(n=N_RX),
        afe_block=AFEBlock(),
        subspace_block=AdaOjaBlock(N_RX, K),
    )
    out = sim.run(n_steps=1)
    assert len(out["subspace_err"]) == 1


# ------------------------------------------------- multi-chirp flow (capability contract)

def _multichirp_env(make_env_block, n_chirp, n_freqs=32):
    """An environment block whose frames carry `n_chirp` identical chirps."""
    env = make_env_block(n_frames=1, n_freqs=n_freqs)
    real_get_s_pars = env.get_S_pars

    def _stacked():
        base = real_get_s_pars()  # (n_rx, 1, 1, F)
        return torch.cat([base] * n_chirp, dim=2)

    env.get_S_pars = _stacked
    return env


def test_multichirp_frame_flows_through_the_elementwise_and_product_blocks(make_env_block):
    """A multi-chirp frame must reach the FFT product via the element-wise stages, which
    declare CHIRP_NATIVE, and come out stacked on a leading chirp axis."""
    n_chirp, bins = 3, 32
    sim = Simulation(
        _multichirp_env(make_env_block, n_chirp),
        [FFTBlock(bins=bins)], K,
        circuit_block=RFFEBlock(n=N_RX),
        interconnect_block=InterconnectBlock(case='case3'),
    )
    sim.reset()
    sim.feed_forward()
    fft = sim.get_outputs()['fft'][0]
    assert fft.shape == (n_chirp, bins, bins)


def test_single_chirp_product_shape_is_unchanged_by_the_capability_contract(make_env_block):
    """The historical single-chirp path must NOT grow a chirp axis -- broadcast_over_chirps
    is the plain single-chirp call when n_chirp == 1."""
    bins = 32
    sim = Simulation(
        _multichirp_env(make_env_block, 1),
        [FFTBlock(bins=bins), RangeAzBlock(bins=bins)], K,
        circuit_block=RFFEBlock(n=N_RX),
    )
    sim.reset()
    sim.feed_forward()
    outputs = sim.get_outputs()
    assert outputs['fft'][0].shape == (bins, bins)
    assert outputs['range_az'][0].shape == (bins, bins)


def test_multichirp_frame_stops_at_the_first_single_chirp_component(make_env_block):
    """The AFE/subspace path declares chirps='single'; the error must name that stage
    rather than surfacing as a raw matmul shape error deeper in."""
    sim = Simulation(
        _multichirp_env(make_env_block, 2),
        _downstream(), K,
        afe_block=AFEBlock(),
        subspace_block=AdaOjaBlock(N_RX, K),
    )
    sim.reset()
    with pytest.raises(ValueError, match=r"MeasurementStage\[AdaOjaBlock/AFEBlock\]"):
        sim.feed_forward()


# ------------------------------------------------- replay: a chain that starts mid-way

class _AdcSource:
    """Minimal environment block that replays a stored ADC cube, as SourceBlock does:
    it starts the chain already past the dechirp, in the RX-time domain."""

    def __init__(self, cube, labels=None):
        from e2e import frames as _frames
        self.signal_domain = _frames.DOMAIN_RX_TIME
        self._cube = cube
        self._labels = labels

    def get_S_pars(self):
        return self._cube

    def get_state_updates(self):
        return {"labels": self._labels} if self._labels is not None else {}

    def step(self):
        pass

    def reset(self):
        pass


class _AdcProduct:
    """Downstream product block consuming the RX-time cube."""

    def __init__(self):
        from e2e import frames as _frames
        self.frame_capabilities = _frames.FrameCapabilities(
            domain=_frames.DOMAIN_RX_TIME, chirps=_frames.CHIRP_NATIVE
        )

    def apply(self, state):
        return {"cube_power": float(torch.sum(torch.abs(state["adc"]) ** 2))}


def test_chain_can_start_in_the_rx_time_domain_without_ray_tracing():
    """Replay: a stored cube injected part-way down the chain must flow to the products
    without the frequency-domain machinery (the SVD / subspace ground truth) running or
    crashing on a 3-D payload."""
    cube = torch.ones(4, 2, 8, dtype=torch.complex64)
    sim = Simulation(_AdcSource(cube), [_AdcProduct()], K)
    sim.reset()
    sim.feed_forward()
    out = sim.get_outputs()
    assert out["cube_power"][0] == pytest.approx(4 * 2 * 8)
    # The subspace ground truth is deliberately absent rather than invented.
    assert "subspace_err" not in out


def test_replay_carries_environment_state_into_the_chain():
    """Labels attached at the source must reach the blocks, which is what lets a label
    travel with its frame instead of being recomputed downstream."""
    seen = {}

    class _LabelReader:
        def __init__(self):
            from e2e import frames as _frames
            self.frame_capabilities = _frames.FrameCapabilities(
                domain=_frames.DOMAIN_RX_TIME, chirps=_frames.CHIRP_NATIVE
            )

        def apply(self, state):
            seen["labels"] = state.get("labels")
            return {}

    cube = torch.zeros(2, 1, 4, dtype=torch.complex64)
    sim = Simulation(_AdcSource(cube, labels=[{"range_m": 12.0}]), [_LabelReader()], K)
    sim.reset()
    sim.feed_forward()
    assert seen["labels"] == [{"range_m": 12.0}]


def test_reset_rewinds_serial_stages_not_only_downstream_blocks():
    """A sink placed mid-chain must be rewound between runs, or a second run writes its
    frames under the first run's numbering."""
    class _CountingStage:
        def __init__(self):
            from e2e import frames as _frames
            self.frame_capabilities = _frames.FrameCapabilities(
                domain=_frames.DOMAIN_RX_TIME, chirps=_frames.CHIRP_NATIVE
            )
            self.frames_seen = 0

        def apply(self, state):
            self.frames_seen += 1
            return {}

        def reset(self):
            self.frames_seen = 0

    stage = _CountingStage()
    env = _AdcSource(torch.zeros(2, 1, 4, dtype=torch.complex64))
    sim = Simulation(env, [], K, serial_stages=[stage])
    sim.reset()
    sim.feed_forward()
    assert stage.frames_seen == 1
    sim.reset()
    assert stage.frames_seen == 0


# ------------------------------------------ reactive n_refine / gap_response wiring
# See AdaOjaBlock.effective_n_refine (e2e/blocks.py) and the "TRACKER DIVERGENCE
# ROOT-CAUSED" investigation in notes/TODO.md: a near-degenerate SV cluster at the k
# cutoff makes the top-k subspace identity numerically arbitrary and spikes tracking
# error. These pin that Simulation threads the (ground-truth-free) sv_gap_at_k
# diagnostic to MeasurementStage, and that it actually changes tracker behavior.

def _controlled_gap_env(make_env_block, d, n_freqs, k, s_tail, array_shape, seed=0):
    """An env block that hands back ONE fixed frame whose true singular values are
    EXACTLY `s_tail` from index k-1 onward (descending, so `s_tail[0]/s_tail[1]` is
    the frame's real sv_gap_at_k) -- deterministic control of the spectrum diagnostic
    for testing the sv_gap_at_k -> effective_n_refine wiring end to end."""
    env = make_env_block(n_frames=1, n_freqs=n_freqs, seed=seed, n_rx=d, array_shape=array_shape)
    S = torch.linspace(20.0, s_tail[0] * 1.5, k - 1, device=torch_device_of(env))
    S = torch.cat([S, torch.tensor(s_tail, device=S.device)])
    assert torch.all(S[:-1] >= S[1:]), "s_tail must keep S descending"
    Uo = torch.linalg.qr((torch.randn(d, n_freqs, dtype=torch.cfloat, device=S.device)))[0]
    Vo = torch.linalg.qr((torch.randn(n_freqs, n_freqs, dtype=torch.cfloat, device=S.device)))[0]
    frame = (Uo * S.to(torch.cfloat).unsqueeze(0)) @ Vo.conj().T
    frame = frame.view(d, 1, 1, n_freqs)
    env.get_S_pars = lambda: frame
    return env


def torch_device_of(env):
    return env.get_S_pars().device


def test_gap_response_none_matches_baseline_n_refine_call_count(make_env_block, torch_device):
    """gap_response='none' (the default) is bit-for-bit the old behavior: n_refine
    calls to subspace_block.update() every frame, regardless of the spectrum."""
    from e2e.blocks import MeasurementStage

    d, n_freqs, k = 32, 9, 6
    env = _controlled_gap_env(make_env_block, d, n_freqs, k, s_tail=[5.0, 4.762, 3.0, 2.0], array_shape=(4, 8))
    subspace = AdaOjaBlock(d, k, m=16, n_refine=3)
    assert subspace.gap_response == "none"
    calls = {"n": 0}
    real_update = subspace.update
    subspace.update = lambda *a, **kw: (calls.__setitem__("n", calls["n"] + 1), real_update(*a, **kw))[-1]
    sim = Simulation(env, [], k, subspace_block=subspace)
    sim.reset()
    sim.feed_forward()
    assert calls["n"] == 3
    assert sim.outputs["n_refine_used"] == [3]


def test_gap_response_refine_boosts_n_refine_on_collapsed_gap_end_to_end(make_env_block, torch_device):
    """A frame whose true spectrum has a near-degenerate cluster at k (sv_gap_at_k
    ~1.05) makes gap_response='refine' run n_refine_hi passes instead of the base
    n_refine -- driven purely by Simulation's own spectrum diagnostic, with no U_true
    in sight for the tracker."""
    from e2e.blocks import MeasurementStage

    d, n_freqs, k = 32, 9, 6
    collapsed_env = _controlled_gap_env(
        make_env_block, d, n_freqs, k, s_tail=[5.0, 4.762, 3.0, 2.0], array_shape=(4, 8))
    healthy_env = _controlled_gap_env(
        make_env_block, d, n_freqs, k, s_tail=[5.0, 1.0, 0.9, 0.8], array_shape=(4, 8))

    def n_refine_calls(env):
        subspace = AdaOjaBlock(d, k, m=16, n_refine=3, gap_response="refine",
                                gap_threshold=1.15, n_refine_hi=9)
        calls = {"n": 0}
        real_update = subspace.update
        subspace.update = lambda *a, **kw: (calls.__setitem__("n", calls["n"] + 1), real_update(*a, **kw))[-1]
        sim = Simulation(env, [], k, subspace_block=subspace)
        sim.reset()
        sim.feed_forward()
        assert sim.outputs["sv_gap_at_k"][0] == pytest.approx(5.0 / s_tail_gap, rel=1e-2)
        # The per-frame cost log records what actually ran, next to the diagnostic
        # that triggered it.
        assert sim.outputs["n_refine_used"] == [calls["n"]]
        return calls["n"]

    s_tail_gap = 4.762
    assert n_refine_calls(collapsed_env) == 9   # sv_gap_at_k ~1.05 < gap_threshold -> hi
    s_tail_gap = 1.0
    assert n_refine_calls(healthy_env) == 3     # sv_gap_at_k ~5.0 >= gap_threshold -> base


def test_gap_response_coast_freezes_basis_on_collapsed_gap_end_to_end(make_env_block, torch_device):
    """gap_response='coast': the tracker's basis is left untouched (0 update calls)
    while the gap is collapsed."""
    d, n_freqs, k = 32, 9, 6
    env = _controlled_gap_env(make_env_block, d, n_freqs, k, s_tail=[5.0, 4.762, 3.0, 2.0], array_shape=(4, 8))
    subspace = AdaOjaBlock(d, k, m=16, n_refine=3, gap_response="coast", gap_threshold=1.15)
    calls = {"n": 0}
    real_update = subspace.update
    subspace.update = lambda *a, **kw: (calls.__setitem__("n", calls["n"] + 1), real_update(*a, **kw))[-1]
    U_before = subspace.oja.U.clone()
    sim = Simulation(env, [], k, subspace_block=subspace, warm_start=False)
    sim.reset()
    sim.feed_forward()
    assert calls["n"] == 0
    assert torch.allclose(subspace.oja.U, U_before)
    assert sim.outputs["n_refine_used"] == [0]


def test_gap_response_reacts_only_to_spectrum_never_to_ground_truth(make_env_block, torch_device):
    """MeasurementStage.apply must be able to compute the reactive n_refine from a
    state dict that has NO 'U_true' key at all -- the mechanism cannot be peeking at
    ground truth if it doesn't even receive it."""
    from e2e.blocks import MeasurementStage

    d, k, m = 32, 4, 16
    subspace = AdaOjaBlock(d, k, m=m, n_refine=2, gap_response="refine",
                            gap_threshold=1.15, n_refine_hi=6)
    stage = MeasurementStage(None, subspace)
    calls = {"n": 0}
    real_update = subspace.update
    subspace.update = lambda *a, **kw: (calls.__setitem__("n", calls["n"] + 1), real_update(*a, **kw))[-1]

    s_pars = torch.randn(d, 1, 1, 8, dtype=torch.cfloat, device=torch_device)
    state = {"s_pars": s_pars, "sv_gap_at_k": 1.0}  # collapsed; no 'U_true' key present
    assert "U_true" not in state
    stage.apply(state)
    assert calls["n"] == 6
