"""Shared plotting helpers for the `e2e/main/main_*.py` example scripts (and
`e2e.ml.render_scene`).

Two small pieces of range-azimuth (RA) map plotting logic were independently
duplicated across five example scripts plus `e2e.ml.render_scene`:

1. peak-normalized power -> dB, with a DRIFTING epsilon constant at every call site
   (`1e-12`, `1e-30`, `torch.finfo(torch.float32).tiny`, or none at all -- see
   `to_db`'s docstring for the full survey).
2. the `imshow` row/column TRANSPOSE needed because an angle-FFT-then-range map is
   naturally `[n_angle, n_range]` but `imshow` puts rows on y and columns on x, and
   this project's convention is azimuth-on-x/range-on-y. This exact transpose bug has
   been reintroduced independently at least three times in this project's history;
   `e2e.ml.render_scene._draw_radar_view` was the one copy with a regression test
   (`tests/test_ml_render.py::test_draw_radar_view_imshow_orientation_matches_extent`)
   guarding it, and is now a thin wrapper around `imshow_ra` below rather than a
   fourth independent copy.

Both helpers are torch-tolerant (accept a `torch.Tensor` -- CPU or CUDA -- or a numpy
array) and this module itself is importable with no torch installed (only `to_db`'s
torch branch needs it, guarded lazily), matching the rest of this package's
torch-free import boundary (see CLAUDE.md).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover -- exercised by the torch-free import tests
    torch = None


def fig_dir(module_file: str) -> str:
    """`<the calling module's own directory>/figures`, created if missing.

    Every `e2e/main/main_*.py` example script writes its PNGs to a `figures/`
    directory next to itself; `os.path.join(os.path.dirname(__file__), "figures")`
    followed by `os.makedirs(..., exist_ok=True)` was copy-pasted byte-for-byte in
    9 places. Call as `fig_dir(__file__)` from the module that owns the output.
    """
    d = os.path.join(os.path.dirname(os.path.abspath(module_file)), "figures")
    os.makedirs(d, exist_ok=True)
    return d


def to_db(power, floor_db: Optional[float] = -40.0, eps: float = 1e-12):
    """Peak-normalized power -> dB: `10 * log10(power / max(power))`.

    Epsilon convention (surveyed drift across call sites before this consolidation):
    `1e-12` additive in `main_isac.py`'s sensing map / `main_isac_multilink.py`'s
    range waterfall / `main_tx_nonideality.py`'s range-profile plot; `1e-30` in
    `main_comms_head.py`'s radar map / `main_tx_nonideality.py`'s PSD plot (denominator
    only -- `1e-12` in its log argument, so actually two DIFFERENT epsilons in one
    expression); `torch.finfo(torch.float32).tiny` (~1.18e-38) in
    `e2e.ml.render_scene.range_azimuth_map`; and no epsilon at all in
    `main_sionna_blocks.py` (a bare `torch.log10(torch.abs(...))`, i.e. `log10(0) ==
    -inf` on any exact-zero bin). This function standardizes on `eps=1e-12` for BOTH
    the peak-clamp (guards a `power` that is identically zero) and the ratio-clamp
    (guards a bin that underflows to exactly zero after normalization): every one of
    the surveyed call sites separately clips its DISPLAYED range well above -120 dB
    (an imshow `vmin`/`clim` of -30 to -40 dB, or a computed `ylim`), so `1e-12` (a
    -120 dB floor) is already deep enough to never be mistaken for real signal by any
    of them, and unlike `1e-30` or `float32.tiny` it stays comfortably inside
    float32's normal range when added to an `O(1)` ratio (no silent precision loss).

    `floor_db`, if not `None` (default `-40.0`, at or below every surveyed call
    site's own display floor), additionally CLAMPS the returned dB values -- unlike a
    purely display-side `vmin=` on `imshow`, this bounds the array itself, which is
    useful for anything downstream of the plot (e.g. test assertions). Callers that
    need the raw, wide dynamic range preserved for their own inspection (e.g. a PSD
    trace whose `ylim` is computed FROM its minimum) should pass `floor_db=None` to
    skip this clamp and rely on `eps` alone -- the pre-consolidation "numerical safety
    net only" behavior.
    """
    is_torch = torch is not None and isinstance(power, torch.Tensor)
    if is_torch:
        peak = power.max().clamp_min(eps)
        ratio = (power / peak).clamp_min(eps)
        db = 10.0 * torch.log10(ratio)
        if floor_db is not None:
            db = db.clamp_min(floor_db)
        return db

    power = np.asarray(power)
    peak = max(float(power.max()), eps)
    ratio = np.clip(power / peak, eps, None)
    db = 10.0 * np.log10(ratio)
    if floor_db is not None:
        db = np.clip(db, floor_db, None)
    return db


def imshow_ra(ax, ra, sin_az_axis=None, range_axis_m=None, **imshow_kw):
    """Draw a range-azimuth (or any angle x range) power map `ra` on `ax`, azimuth
    (or angle) on x, range on y. Returns the `AxesImage`.

    `ra` is `[n_angle, n_range]` (angle-FFT rows, range columns -- the natural shape
    after an angle FFT over a range-compressed cube; see e.g.
    `e2e.ml.render_scene.range_azimuth_map`'s docstring), but `imshow` indexes its
    array as `[row=y, col=x]`; with azimuth on x and range on y (the convention every
    call site in this project uses), the array must be TRANSPOSED to `[n_range,
    n_angle]` first, or the image content is scrambled relative to its own axes. This
    is the one place that transpose is done now (see module docstring for why that
    matters).

    `sin_az_axis`/`range_axis_m`, if both given, set the imshow `extent` (ascending;
    row 0 of `ra` -> `sin_az_axis[0]`, column 0 -> `range_axis_m[0]`) so the axes
    carry physical units. If either is `None` (a caller with no physical axis, e.g. a
    bin-indexed az/el map), no `extent` is passed and `imshow` falls back to its own
    pixel-index default -- exactly as if this map had no axes at all.

    `origin="lower"` and `aspect="auto"` are the defaults every call site used;
    `**imshow_kw` (`cmap`, `vmin`, `vmax`, ...) is passed straight through to
    `ax.imshow` and can override them.
    """
    if torch is not None and isinstance(ra, torch.Tensor):
        arr = ra.detach().cpu().numpy()
    else:
        arr = np.asarray(ra)

    kw = dict(origin="lower", aspect="auto")
    kw.update(imshow_kw)
    if sin_az_axis is not None and range_axis_m is not None:
        kw.setdefault("extent", [sin_az_axis[0], sin_az_axis[-1], range_axis_m[0], range_axis_m[-1]])
    return ax.imshow(arr.T, **kw)
