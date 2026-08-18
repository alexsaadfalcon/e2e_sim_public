"""Unit tests for e2e.comms.constellation_viz (shared constellation-scatter
plotting for the comms/ISAC example scripts)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import matplotlib
matplotlib.use("Agg")            # headless, matches the example scripts' own convention
import matplotlib.pyplot as plt

from e2e.comms.ofdm import qam_constellation
from e2e.comms.constellation_viz import plot_constellation, _constellation_colors


@pytest.mark.parametrize("bits_per_symbol", [2, 4, 6])
def test_constellation_colors_are_all_distinct(bits_per_symbol):
    """Every one of the M constellation points must get its OWN color -- a
    collision would silently make two different symbols indistinguishable."""
    const = qam_constellation(bits_per_symbol).cpu().numpy()
    colors = _constellation_colors(const)
    assert colors.shape == (const.size, 4)
    uniq = {tuple(np.round(c, 6)) for c in colors}
    assert len(uniq) == const.size


def test_plot_constellation_colors_by_ground_truth_not_by_position():
    """A point spatially near cluster B, but ACTUALLY transmitted as cluster A
    (tx_syms=A), must be colored as A -- ground truth, not decision-directed --
    so a genuine symbol error is visible as a wrong-colored point in the wrong
    cluster, per the module's whole reason for existing."""
    const = qam_constellation(4).cpu().numpy()   # 16-QAM
    tx = np.array([const[0], const[1]])
    # rx[0] has drifted almost exactly onto const[1]'s location, but was truly
    # transmitted as const[0].
    rx = np.array([const[1] * 0.999, const[1]])

    fig, ax = plt.subplots()
    sc = plot_constellation(ax, rx, const, tx_syms=tx)
    plt.close(fig)

    colors = _constellation_colors(const)
    face = sc.get_facecolor()
    # compare RGB only -- scatter's own `alpha=` overwrites each facecolor's 4th
    # (alpha) channel independent of the input color array's own alpha.
    assert np.allclose(face[0, :3], colors[0, :3], atol=1e-6)   # colored as its TRUE symbol
    assert np.allclose(face[1, :3], colors[1, :3], atol=1e-6)


def test_plot_constellation_decision_directed_fallback_when_no_tx_syms():
    """Without tx_syms, coloring falls back to nearest-point of rx_syms itself
    (decision-directed) -- documented, not a silent ground-truth claim."""
    const = qam_constellation(4).cpu().numpy()
    rx = np.array([const[3] * 0.98])

    fig, ax = plt.subplots()
    sc = plot_constellation(ax, rx, const, tx_syms=None)
    plt.close(fig)

    colors = _constellation_colors(const)
    assert np.allclose(sc.get_facecolor()[0, :3], colors[3, :3], atol=1e-6)


def test_alpha_is_density_aware_not_hardcoded():
    """A handful of points should render near-opaque; thousands should be
    dimmer -- alpha must actually depend on point count."""
    const = qam_constellation(4).cpu().numpy()
    rng = np.random.default_rng(0)

    few = const[rng.integers(0, 16, 20)]
    many = const[rng.integers(0, 16, 5000)]

    fig, ax = plt.subplots()
    sc_few = plot_constellation(ax, few, const)
    alpha_few = sc_few.get_alpha()
    plt.close(fig)

    fig, ax = plt.subplots()
    sc_many = plot_constellation(ax, many, const)
    alpha_many = sc_many.get_alpha()
    plt.close(fig)

    assert alpha_few > alpha_many
    assert 0.05 <= alpha_many <= 0.9
    assert 0.05 <= alpha_few <= 0.9


def test_mark_ideal_false_skips_the_reference_marker_collection():
    const = qam_constellation(4).cpu().numpy()
    rx = const.copy()

    fig, ax = plt.subplots()
    plot_constellation(ax, rx, const, mark_ideal=False)
    n_with_marker_off = len(ax.collections)
    plt.close(fig)

    fig, ax = plt.subplots()
    plot_constellation(ax, rx, const, mark_ideal=True)
    n_with_marker_on = len(ax.collections)
    plt.close(fig)

    assert n_with_marker_on == n_with_marker_off + 1


def test_plot_constellation_accepts_torch_tensors():
    """The example scripts pass torch tensors (post `.cpu().numpy()` in some
    call sites, raw tensors in others) -- both must work."""
    const = qam_constellation(4)
    rx = const.clone()
    fig, ax = plt.subplots()
    sc = plot_constellation(ax, rx, const)
    plt.close(fig)
    assert sc.get_offsets().shape[0] == const.numel()


def test_works_for_all_supported_qam_orders():
    """2/4/6 bits/symbol (QPSK/16-QAM/64-QAM) must all render without error --
    main_tx_nonideality.py defaults to 64-QAM."""
    for bps in (2, 4, 6):
        const = qam_constellation(bps).cpu().numpy()
        rx = const[np.random.default_rng(0).integers(0, const.size, 50)]
        fig, ax = plt.subplots()
        plot_constellation(ax, rx, const)
        plt.close(fig)
