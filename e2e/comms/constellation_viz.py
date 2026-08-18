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
* Draws the ideal lattice as black "x" markers under the cloud, so the
  reference grid is visible independent of hue.
* Assigns the M point colors via a bit-reversal permutation of each point's
  (row, col) position on the square QAM grid, NOT sequential colormap index
  order. Plain sequential sampling (`tab20(linspace(0,1,M))`) hands spatially
  ADJACENT grid points near-adjacent colormap entries, which is exactly the
  case that matters most: symbol errors land in the spatially NEAREST
  cluster. Bit-reversal is the standard trick (radix-2 FFT / Sobol-sequence
  ordering) for maximizing the spread between sequential indices; it applies
  cleanly here because every QAM order this repo supports (`qam_constellation`
  in `ofdm.py`: 2/4/6 bits/symbol) is a square, power-of-two grid.

Honest legibility note (see also the calling scripts' own comments): 16
qualitative colors is near the top of what `tab20` can give without reusing
a hue (it has 10 hue families x 2 lightness variants each); at 16-QAM some
of those same-hue light/dark pairs are unavoidably reused, so on a heavily
compressed screen-share a SUBSET of clusters may still be hard to tell apart
by color alone -- the black ideal-point markers plus the "wrong color in the
wrong cluster" reasoning are what still work if the palette does not.
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


def _constellation_colors(const: np.ndarray) -> np.ndarray:
    """`[M, 4]` RGBA, one distinguishable color per point of `const` (`[M]`
    complex), ordered to match `const`'s own indexing. See module docstring
    for the bit-reversal spatial-spread rationale."""
    m_total = const.size
    side = int(round(np.sqrt(m_total)))
    if side * side == m_total:
        # rank real/imag parts onto a 0..side-1 grid (works for any square QAM
        # layout without assuming qam_constellation's specific index formula)
        row = np.searchsorted(np.unique(const.real), const.real)
        col = np.searchsorted(np.unique(const.imag), const.imag)
    else:
        # not a square grid (shouldn't happen for this repo's QAM orders) --
        # degrade to a 1 x M row rather than crash.
        side = m_total
        row = np.zeros(m_total, dtype=int)
        col = np.arange(m_total)

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


def plot_constellation(ax, rx_syms, const, tx_syms=None, s=8, title=None,
                        ideal_marker_kw=None, mark_ideal=True):
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
    mark_ideal : draw the black ideal-lattice "x" markers (default True). Set
               False for a panel where `rx_syms` already coincides EXACTLY with
               the ideal lattice (e.g. a noiseless TX-side reference) -- there
               every colored point sits exactly under its own marker, and
               drawing both just makes one of the two invisible.

    Returns the `PathCollection` for the received-symbol scatter.
    """
    rx = _to_numpy_complex(rx_syms).reshape(-1)
    const_np = _to_numpy_complex(const).reshape(-1)
    label_src = _to_numpy_complex(tx_syms).reshape(-1) if tx_syms is not None else rx

    d = np.abs(label_src[:, None] - const_np[None, :])
    idx = np.argmin(d, axis=1)
    colors = _constellation_colors(const_np)

    n = rx.size
    # Density-aware alpha: calibrated so a few dozen points (this repo's
    # SNR-sweep snapshots) render near-opaque, while a several-thousand-point
    # run (per-pixel-bit symbols) still shows cluster density instead of
    # saturating to a solid disc; clipped to stay visible either way.
    alpha = float(np.clip(300.0 / max(n, 1), 0.05, 0.9))

    # Ideal markers drawn BENEATH the received cloud (lower zorder) per "the
    # reader sees the reference lattice under the cloud" -- a marker on top
    # would fully occlude a tightly-clustered (or, worse, exactly-coincident)
    # point directly underneath it instead of the other way around.
    if mark_ideal:
        mk = dict(marker="x", c="black", s=70, linewidths=1.5, zorder=1)
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
