"""Regression tests for the subspace-tracking capability (AdaOjaBlock 'reestimate').

The tracker must actually FOLLOW a drifting signal subspace -- not sit frozen, which is
what the legacy incremental-Oja step did. These use a controlled, deterministic
low-rank scene that drifts a fixed amount per frame (no munich.pkl, no GPU), and check
the tracker stays near the k-truncated-SVD floor and clearly beats a frozen basis.
"""
import os

import pytest

torch = pytest.importorskip("torch")

from e2e.blocks import AdaOjaBlock
from e2e.subspace.algorithms import rand_orth_complex
from e2e.subspace.subspace_utils import subspace_dist_frob

_MUNICH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       "e2e", "environment", "sionna_sims", "munich.pkl")


def _orth(M):
    return torch.linalg.qr(M)[0]


def _run(block, k, d, F=64, T=10, drift=0.03, seed=0, device=None):
    """Drive `block` over a subspace that drifts `drift`/frame; return (tracker_err,
    frozen_err) trajectories. Mirrors MeasurementStage's no-AFE refine loop."""
    g = torch.Generator(device=device).manual_seed(seed)

    def cn(*shp):
        return (torch.randn(*shp, generator=g, device=device)
                + 1j * torch.randn(*shp, generator=g, device=device))

    Ut = rand_orth_complex(d, k, device=device)
    block.oja.U = Ut.clone()             # warm start on truth
    frozen = Ut.clone()
    track_err, frozen_err = [], []
    for _ in range(T):
        V = Ut @ cn(k, F)                # signal lives in the current subspace
        for _ in range(block.n_refine):  # same refinement loop MeasurementStage runs
            A = block.gen_A_ada()
            block.update(A @ V, A)
        track_err.append(float(subspace_dist_frob(Ut, block.oja.U)))
        frozen_err.append(float(subspace_dist_frob(Ut, frozen)))
        Ut = _orth(Ut + drift * cn(d, k))  # drift the true subspace
    return track_err, frozen_err


def test_reestimate_tracker_follows_drifting_subspace(torch_device):
    d, k, m = 256, 4, 192
    block = AdaOjaBlock(d, k, m=m, n_refine=10, method="reestimate")
    track, frozen = _run(block, k, d, device=torch_device)
    # tracks the drift: stays low, does NOT run away toward the ~sqrt(k) random floor
    assert track[-1] < 0.5
    # clearly beats a frozen basis (which the drift carries away)
    assert sum(track) < 0.5 * sum(frozen)


@pytest.mark.skipif(not os.path.isfile(_MUNICH), reason="needs munich.pkl (real scene)")
def test_reestimate_tracks_fast_real_scene_where_legacy_fails():
    """On the real munich scene (fast, bursty subspace drift), the cheap power-iteration
    re-estimate stays near the SVD floor while the legacy fixed-step Oja runs away -- the
    exact defect the fix addresses (there the legacy tracker was ~indistinguishable from a
    frozen basis)."""
    from e2e.simulation import Simulation
    from e2e.blocks import (SionnaEnvironmentBlock, RFFEBlock, InterconnectBlock,
                            AFEBlock, SubspaceErrorBlock)

    def run(method):
        env = SionnaEnvironmentBlock("munich")
        sim = Simulation(
            env, [SubspaceErrorBlock()], 8,
            RFFEBlock(n=1024, physical_scale=bool(env.physical_scale)),
            InterconnectBlock(case="case3"), AFEBlock(),
            AdaOjaBlock(1024, 8, m=512, n_refine=10, method=method))
        return [float(x) for x in sim.run(n_steps=10)["subspace_err"]]

    re = run("reestimate")
    oja = run("oja")
    assert max(re[3:]) < 0.4            # stays near the k-truncated-SVD floor (~0.08)
    assert oja[-1] > 3 * re[-1]         # legacy runs away instead of tracking


def test_more_measurements_lower_the_floor(torch_device):
    """Observability: with more measurements m the tracker gets closer to the SVD floor
    (mean tracking error should not increase as m grows)."""
    d, k = 256, 4
    errs = []
    for m in (64, 192):
        block = AdaOjaBlock(d, k, m=m, n_refine=10, method="reestimate")
        track, _ = _run(block, k, d, device=torch_device)
        errs.append(sum(track[3:]) / len(track[3:]))
    assert errs[1] <= errs[0] + 1e-3    # more measurements -> no worse
