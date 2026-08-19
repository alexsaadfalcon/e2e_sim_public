"""Unit tests for e2e.comms.constellation_viz (shared constellation-scatter
plotting for the comms/ISAC example scripts)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import matplotlib
matplotlib.use("Agg")            # headless, matches the example scripts' own convention
import matplotlib.pyplot as plt

from e2e.comms.ofdm import qam_constellation
from e2e.comms.constellation_viz import (
    plot_constellation, _constellation_colors, _checkerboard_colors, _grid_row_col,
)


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


@pytest.mark.parametrize("bits_per_symbol", [2, 4, 6])
def test_checkerboard_neighbours_never_share_a_hue(bits_per_symbol):
    """The whole point of checkerboard mode: every 4-/8-connected neighbour on
    the I/Q grid (including diagonals) must get a DIFFERENT hue, so a
    mis-decoded (wrong-cell) symbol is visible as a wrong-coloured dot."""
    const = qam_constellation(bits_per_symbol).cpu().numpy()
    colors = _checkerboard_colors(const)
    row, col, side = _grid_row_col(const)
    pos_to_idx = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(row, col))}
    for (r, c), i in pos_to_idx.items():
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            j = pos_to_idx.get((r + dr, c + dc))
            if j is None:
                continue
            assert not np.allclose(colors[i], colors[j]), \
                f"neighbours {(r, c)} and {(r + dr, c + dc)} share a hue"


def test_checkerboard_uses_at_least_four_hues():
    const = qam_constellation(6).cpu().numpy()   # 64-QAM, 8x8 grid
    colors = _checkerboard_colors(const)
    uniq = {tuple(np.round(c, 6)) for c in colors}
    assert len(uniq) >= 4


def test_plot_constellation_checkerboard_mode_colors_by_ground_truth():
    """Same ground-truth-coloring contract as bitrev mode, just via the
    checkerboard palette -- a drifted point must still be colored as what it
    was actually TRANSMITTED as."""
    const = qam_constellation(4).cpu().numpy()   # 16-QAM
    tx = np.array([const[0], const[1]])
    rx = np.array([const[1] * 0.999, const[1]])

    fig, ax = plt.subplots()
    sc = plot_constellation(ax, rx, const, tx_syms=tx, color_mode="checkerboard")
    plt.close(fig)

    colors = _checkerboard_colors(const)
    face = sc.get_facecolor()
    assert np.allclose(face[0, :3], colors[0, :3], atol=1e-6)
    assert np.allclose(face[1, :3], colors[1, :3], atol=1e-6)


def test_plot_constellation_rejects_unknown_color_mode():
    const = qam_constellation(4).cpu().numpy()
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        plot_constellation(ax, const, const, color_mode="bogus")
    plt.close(fig)


def test_works_for_all_supported_qam_orders():
    """2/4/6 bits/symbol (QPSK/16-QAM/64-QAM) must all render without error --
    main_tx_nonideality.py defaults to 64-QAM."""
    for bps in (2, 4, 6):
        const = qam_constellation(bps).cpu().numpy()
        rx = const[np.random.default_rng(0).integers(0, const.size, 50)]
        fig, ax = plt.subplots()
        plot_constellation(ax, rx, const)
        plt.close(fig)


# ------------------------------------------------------------------------------------
# The colouring rule shared with the webapp's Plotly view.
#
# These exist because the two views HAD drifted: the matplotlib figure coloured every
# symbol by the point it was transmitted as, and the webapp drew flat markers, for months,
# with the data it needed already in `outputs`. Extracting the rule is only half a fix --
# the other half is a test that fails if the two ever diverge again.
# ------------------------------------------------------------------------------------
def _np_const(bits):
    """`qam_constellation` returns a torch tensor; the private colour helpers want numpy."""
    return qam_constellation(bits).cpu().numpy()


def test_symbol_color_indices_uses_ground_truth_over_proximity():
    """THE point of passing tx_syms: a symbol that landed nearer a DIFFERENT ideal point
    must still be labelled by what was actually sent. Otherwise a symbol error colours
    itself correct and becomes invisible -- which is the one thing the plot is for."""
    from e2e.comms.constellation_viz import symbol_color_indices

    const = np.array([-3 - 3j, -3 + 3j, 3 - 3j, 3 + 3j], dtype=np.complex64)
    # Sent const[0], but noise pushed it right next to const[2].
    tx = np.array([const[0]], dtype=np.complex64)
    rx = np.array([2.9 - 3.0j], dtype=np.complex64)

    assert symbol_color_indices(rx, const, tx)[0] == 0          # ground truth
    assert symbol_color_indices(rx, const, None)[0] == 2        # decision-directed


def test_symbol_color_indices_is_exact_on_noiseless_symbols():
    from e2e.comms.constellation_viz import symbol_color_indices

    const = _np_const(4)
    idx_expected = np.arange(len(const))
    got = symbol_color_indices(const, const, const)
    assert np.array_equal(got, idx_expected)


def test_checkerboard_css_colors_matches_the_matplotlib_palette():
    """The Plotly strings must BE the matplotlib RGBA, not a lookalike."""
    from e2e.comms.constellation_viz import _checkerboard_colors, checkerboard_css_colors

    const = _np_const(4)
    rgba = _checkerboard_colors(const)
    css = checkerboard_css_colors(const, alpha=0.85)
    assert len(css) == len(const)
    for (r, g, b, _a), text in zip(rgba, css):
        nums = text[text.index("(") + 1:text.index(")")].split(",")
        assert int(nums[0]) == int(round(r * 255))
        assert int(nums[1]) == int(round(g * 255))
        assert int(nums[2]) == int(round(b * 255))
        assert float(nums[3]) == pytest.approx(0.85)


def test_checkerboard_css_colors_are_well_formed_rgba():
    from e2e.comms.constellation_viz import checkerboard_css_colors

    for text in checkerboard_css_colors(_np_const(6)):
        assert text.startswith("rgba(") and text.endswith(")")
        nums = text[5:-1].split(",")
        assert len(nums) == 4
        assert all(0 <= int(v) <= 255 for v in nums[:3])


def test_adjacent_constellation_points_get_different_colours():
    """The property the checkerboard exists for, restated on the CSS path: a wrong-
    coloured dot one cell over has to be visible, so 4- and 8-connected neighbours must
    never share a hue."""
    from e2e.comms.constellation_viz import _grid_row_col, checkerboard_css_colors

    const = _np_const(6)
    row, col, side = _grid_row_col(const)
    css = checkerboard_css_colors(const)
    pos = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(row, col))}
    for (r, c), i in pos.items():
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            j = pos.get((r + dr, c + dc))
            if j is not None:
                assert css[i] != css[j], f"({r},{c}) and ({r+dr},{c+dc}) share a hue"
