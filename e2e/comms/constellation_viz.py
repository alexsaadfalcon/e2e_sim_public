"""Shared constellation-scatter plotting for the comms/ISAC example scripts.

`main_comms_link.py`, `main_isac.py`, `main_tx_nonideality.py` and
`main_image_link.py` each plotted their own received-QAM scatter as a single
flat color at a hardcoded alpha (0.4 / 0.4 / 0.35). With thousands of points
piled into 16 (or 64, for `main_tx_nonideality`'s default) clusters that
render is one undifferentiated blob -- no way to see cluster tightness, bleed
into a neighbouring cluster, or which points are actually mis-decoded.
`plot_constellation` below is the one shared implementation all four call:

* Colors each point by the IDEAL constellation point it was actually
  TRANSMITTED as (ground truth), via the caller-supplied `tx_syms` -- every
  example script in this repo already has it (it built `tx_freq`/`tx_data_ref`
  itself). A point of the "wrong" colour sitting inside a neighbouring
  cluster is then a genuine symbol error, visible at a glance. If a caller has
  no transmitted-symbol reference at the plotting site, `tx_syms=None` falls
  back to nearest-point coloring of `rx_syms` itself -- this is DECISION-
  DIRECTED, NOT ground truth (a symbol that already crossed a decision
  boundary recolors as whatever it decoded to, hiding its own error), and
  every call site in this repo avoids that branch on purpose.
* Sizes alpha to the point COUNT rather than a hardcoded constant, so a
  32-symbol SNR-sweep snapshot stays visible and a several-thousand-point
  per-pixel-bit run (`main_image_link`) still reads as density, not a solid
  disc.
* Draws the ideal lattice under the cloud (marker style depends on
  `color_mode`, see below), so the reference grid is visible independent of
  hue.
* Assigns point colors one of two ways, via `color_mode`:
  - `"bitrev"` (default, unchanged from the original implementation): a
    bit-reversal permutation of each point's (row, col) position on the
    square QAM grid, NOT sequential colormap index order. Plain sequential
    sampling (`tab20(linspace(0,1,M))`) hands spatially ADJACENT grid points
    near-adjacent colormap entries, which is exactly the case that matters
    most: symbol errors land in the spatially NEAREST cluster. Bit-reversal
    is the standard trick (radix-2 FFT / Sobol-sequence ordering) for
    maximizing the spread between sequential indices; it applies cleanly
    here because every QAM order this repo supports (`qam_constellation` in
    `ofdm.py`: 2/4/6 bits/symbol) is a square, power-of-two grid.
  - `"checkerboard"`: a 2x2 repeating tile of 4 well-separated hues by grid
    parity (`_checkerboard_colors`) -- every 4-/8-connected NEIGHBOURING
    point on the grid is GUARANTEED a different hue (not just "usually
    different" like a large qualitative colormap), which is what
    `main_tx_nonideality.py`'s dense 64-QAM panels need to keep an
    individual mis-decoded symbol visually distinguishable from its
    (correct) neighbours.

Honest legibility note (see also the calling scripts' own comments): in
`"bitrev"` mode, 16 qualitative colors is near the top of what `tab20` can
give without reusing a hue (it has 10 hue families x 2 lightness variants
each); at 16-QAM some of those same-hue light/dark pairs are unavoidably
reused, so on a heavily compressed screen-share a SUBSET of clusters may
still be hard to tell apart by color alone -- the ideal-point markers plus
the "wrong color in the wrong cluster" reasoning are what still work if the
palette does not. `"checkerboard"` mode does not have this problem (only 4
hues, reused by construction, but never on adjacent cells).
"""
from __future__ import annotations

import numpy as np
import matplotlib   # bare `matplotlib` (not `pyplot`) -- no backend/display needed
                     # just to sample colormaps, keeping this importable without a
                     # caller having chosen a backend yet.

try:
    import torch
except ImportError:  # pragma: no cover -- this module works fine without torch
    torch = None


def _to_numpy_complex(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.complex128)
    return np.asarray(x).astype(np.complex128)


def _bit_reverse(x: int, n_bits: int) -> int:
    r = 0
    for _ in range(n_bits):
        r = (r << 1) | (x & 1)
        x >>= 1
    return r


def _grid_row_col(const: np.ndarray):
    """(row, col, side): rank each point of `const` onto a 0..side-1 x 0..side-1
    I/Q grid (works for any square QAM layout without assuming
    `qam_constellation`'s specific index formula). Falls back to a 1 x M row for
    a non-square constellation (shouldn't happen for this repo's QAM orders)."""
    m_total = const.size
    side = int(round(np.sqrt(m_total)))
    if side * side == m_total:
        row = np.searchsorted(np.unique(const.real), const.real)
        col = np.searchsorted(np.unique(const.imag), const.imag)
    else:
        side = m_total
        row = np.zeros(m_total, dtype=int)
        col = np.arange(m_total)
    return row, col, side


def _constellation_colors(const: np.ndarray) -> np.ndarray:
    """`[M, 4]` RGBA, one distinguishable color per point of `const` (`[M]`
    complex), ordered to match `const`'s own indexing. See module docstring
    for the bit-reversal spatial-spread rationale."""
    m_total = const.size
    row, col, side = _grid_row_col(const)

    n_bits = max(1, int(np.ceil(np.log2(side))))
    spread_idx = np.array([
        _bit_reverse(int(r), n_bits) * side + _bit_reverse(int(c), n_bits)
        for r, c in zip(row, col)
    ])
    rank = np.empty(m_total, dtype=int)
    rank[np.argsort(spread_idx)] = np.arange(m_total)

    cmap_name = "tab20" if m_total <= 20 else "hsv"
    cmap = matplotlib.colormaps[cmap_name]
    base = cmap(np.linspace(0.0, 1.0, m_total, endpoint=False))
    return base[rank]


# 4 well-separated, colorblind-friendly hues (Okabe-Ito palette, minus the two
# closest to each other) for `_checkerboard_colors`'s 2x2 repeating tile.
_CHECKERBOARD_PALETTE = np.array([
    matplotlib.colors.to_rgba("#0072B2"),   # blue
    matplotlib.colors.to_rgba("#D55E00"),   # vermillion
    matplotlib.colors.to_rgba("#009E73"),   # bluish green
    matplotlib.colors.to_rgba("#CC79A7"),   # reddish purple
])


def _checkerboard_colors(const: np.ndarray, palette: np.ndarray = None) -> np.ndarray:
    """`[M, 4]` RGBA: colour each ideal point of `const` by `(row + col) % 2` and
    `(row % 2, col % 2)` jointly -- i.e. a 2x2 repeating tile of `palette` (>= 4
    hues) over the I/Q grid, ordered to match `const`'s own indexing.

    A plain 2-colour checkerboard (row+col parity alone) already guarantees no
    two EDGE-adjacent points share a colour, but leaves diagonal neighbours
    sharing one of only two hues. Tiling a 2x2 block of 4 distinct hues instead
    means flipping row OR col (edge-adjacent) OR both (diagonal-adjacent) always
    changes the 2-bit tile index -- so every 4-, 8-connected neighbour on the
    grid gets a different hue, which is what makes a wrong-coloured dot in a
    neighbouring cell (an AM/AM-compression symbol error) visible at a glance.
    """
    if palette is None:
        palette = _CHECKERBOARD_PALETTE
    palette = np.asarray(palette)
    row, col, _side = _grid_row_col(const)
    tile_idx = (row % 2) * 2 + (col % 2)
    return palette[tile_idx % len(palette)]


def symbol_color_indices(rx_syms, const, tx_syms=None) -> np.ndarray:
    """Which ideal constellation point each received symbol belongs to.

    Ground truth when `tx_syms` is given (every caller in this repo has it); otherwise
    decision-directed nearest-neighbour on `rx_syms` itself, which is a genuinely weaker
    labelling -- a symbol that landed closer to the wrong point gets the wrong colour, so
    errors are invisible rather than obvious. See the module docstring.

    Split out of `plot_constellation` so the Plotly view in `webapp/pipeline_runner.py`
    labels symbols by exactly the same rule as the matplotlib figure. Two independent
    implementations of "which point is this" is how the two views drifted apart in the
    first place.
    """
    rx = _to_numpy_complex(rx_syms).reshape(-1)
    const_np = _to_numpy_complex(const).reshape(-1)
    label_src = _to_numpy_complex(tx_syms).reshape(-1) if tx_syms is not None else rx
    return np.argmin(np.abs(label_src[:, None] - const_np[None, :]), axis=1)


def checkerboard_css_colors(const, alpha: float = 0.85):
    """`_checkerboard_colors` as CSS `rgba(...)` strings, one per constellation point.

    For Plotly, which wants colours as strings rather than an `[M, 4]` float array.
    """
    rgba = _checkerboard_colors(_to_numpy_complex(const).reshape(-1))
    return [f"rgba({int(round(r * 255))},{int(round(g * 255))},"
            f"{int(round(b * 255))},{alpha:g})" for r, g, b, _a in rgba]


def plot_constellation(ax, rx_syms, const, tx_syms=None, s=8, title=None,
                        ideal_marker_kw=None, mark_ideal=True, color_mode="bitrev"):
    """Scatter received QAM symbols on `ax`, colored by the ideal constellation
    point each was actually TRANSMITTED as (ground truth if `tx_syms` is
    given), with density-aware alpha and the ideal lattice marked. See module
    docstring for the full rationale.

    Parameters
    ----------
    ax       : matplotlib Axes to draw on.
    rx_syms  : received (or pre-EQ, caller's choice) complex symbols, any
               shape/type (torch or numpy) -- flattened internally.
    const    : the modem's ideal constellation (`modem.const` /
               `qam_constellation(bits_per_symbol)`), `[M]` complex.
    tx_syms  : optional GROUND-TRUTH transmitted symbols, same length/order as
               `rx_syms` (every caller in this repo has this). If `None`, the
               color index is decided by nearest-neighbor of `rx_syms` ITSELF
               -- decision-directed, not ground truth; see module docstring.
    s        : marker size (points^2), passed to `ax.scatter`.
    title    : optional `ax.set_title`.
    mark_ideal : draw the ideal-lattice reference markers (default True). Set
               False for a panel where `rx_syms` already coincides EXACTLY with
               the ideal lattice (e.g. a noiseless TX-side reference) -- there
               every colored point sits exactly under its own marker, and
               drawing both just makes one of the two invisible.
    color_mode : "bitrev" (default, unchanged behavior for existing callers) --
               one `tab20`/`hsv` hue per point via the bit-reversal spatial
               spread (see module docstring). "checkerboard" -- a 2x2 tile of
               4 well-separated hues by grid parity (see `_checkerboard_colors`),
               so every 4-/8-connected NEIGHBOURING cell is guaranteed a
               different hue; also switches the default ideal-lattice marker
               from a heavy "x" to a small unfilled ring, since checkerboard
               mode is meant to be read at higher point density.

    Returns the `PathCollection` for the received-symbol scatter.
    """
    rx = _to_numpy_complex(rx_syms).reshape(-1)
    const_np = _to_numpy_complex(const).reshape(-1)
    idx = symbol_color_indices(rx, const_np, tx_syms)
    if color_mode == "checkerboard":
        colors = _checkerboard_colors(const_np)
        default_mk = dict(marker="o", facecolors="none", edgecolors="black",
                           s=45, linewidths=1.1, zorder=1)
    elif color_mode == "bitrev":
        colors = _constellation_colors(const_np)
        default_mk = dict(marker="x", c="black", s=70, linewidths=1.5, zorder=1)
    else:
        raise ValueError(f"color_mode must be 'bitrev' or 'checkerboard', got {color_mode!r}")

    n = rx.size
    # Density-aware alpha: calibrated so a few dozen points (this repo's
    # SNR-sweep snapshots) render near-opaque, while a several-thousand-point
    # run (per-pixel-bit symbols) still shows cluster density instead of
    # saturating to a solid disc; clipped to stay visible either way.
    # Floor raised 0.05 -> 0.25 and scale 300 -> 500 (owner, 2026-08-19: "dots need
    # to be darker"). At several thousand points the old floor clipped to 0.05, which
    # renders almost invisible once the figure is scaled into a slide box. 0.25 still
    # shows cluster density rather than saturating to a solid disc.
    alpha = float(np.clip(500.0 / max(n, 1), 0.25, 0.9))

    # Ideal markers drawn BENEATH the received cloud (lower zorder) per "the
    # reader sees the reference lattice under the cloud" -- a marker on top
    # would fully occlude a tightly-clustered (or, worse, exactly-coincident)
    # point directly underneath it instead of the other way around. Shrunk
    # (checkerboard mode) or heavy (bitrev, unchanged) per `default_mk` above,
    # so the marker marks the cell rather than covering the cloud beneath it.
    if mark_ideal:
        mk = dict(default_mk)
        if ideal_marker_kw:
            mk.update(ideal_marker_kw)
        ax.scatter(const_np.real, const_np.imag, **mk)

    sc = ax.scatter(rx.real, rx.imag, c=colors[idx], s=s, alpha=alpha,
                     linewidths=0, zorder=2)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    return sc
