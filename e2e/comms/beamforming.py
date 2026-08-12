"""Spatial combining across the full receive aperture, for a communications head.

`ModemBlock` (see `blocks.py`) previously tapped a single spatial channel (element
(0, 0)) -- a SISO link that ignored the other 1023 elements of the array. This
module turns the full aperture into a receive beamformer: build a per-element
channel-frequency-response matrix, form combining weights (EGC, MRC, or a
subspace-tracker-derived broadband direction), and coherently combine
per-element signals into a single stream.

Everything here is a pure function (no state), torch, complex64, on the shared
library `device`. Correctness/array-gain bookkeeping is spelled out in each
function's docstring; noise MUST be injected per element (in `blocks.py`) BEFORE
`combine` is called, or the array-gain accounting below is meaningless.
"""

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batch_interp_common_grid(x_new, x_src, Y):
    """Vectorized linear interpolation of every row of `Y` from a SHARED source
    grid `x_src` onto `x_new`, without looping over rows in Python.

    `x_src` : [F] ascending source grid (shared by every row of `Y`).
    `Y`     : [N, F] real-valued samples (call once for the real part, once for
        the imaginary part, of a complex array).
    `x_new` : [n_new] target grid (already clipped into [x_src[0], x_src[-1]]).

    Returns [N, n_new].
    """
    x_src = np.asarray(x_src, dtype=np.float64)
    x_new = np.asarray(x_new, dtype=np.float64)
    idx = np.searchsorted(x_src, x_new, side="right") - 1
    idx = np.clip(idx, 0, len(x_src) - 2)
    x0 = x_src[idx]
    x1 = x_src[idx + 1]
    frac = (x_new - x0) / (x1 - x0 + 1e-30)
    Y0 = Y[:, idx]
    Y1 = Y[:, idx + 1]
    return Y0 + (Y1 - Y0) * frac[None, :]


def element_channels(s_pars, freqs, target_freqs):
    """Per-element channel frequency response, resampled onto a target grid.

    Flattens `s_pars` to `[N, F]` via `reshape(-1, F)` -- the SAME flat element
    order `MeasurementStage` uses to build `V` (`s_pars.view(-1, s_pars.shape[-1])`)
    and therefore the same order as the tracker's `U` (`U`'s rows index the same
    physical elements as `V`'s rows). Every element's CFR (real/imag parts) is
    then linearly interpolated from the shared `freqs` source grid onto
    `target_freqs` in one vectorized pass over all N rows (no per-element Python
    loop / no per-element `np.interp` call).

    Parameters
    ----------
    s_pars : array-like or torch tensor, any shape whose `reshape(-1, F)` gives
        the aperture's flat element order (e.g. the `[rx_x, rx_y, 1, F]` grid
        `Simulation` hands downstream blocks, or an already-flat `[N, F]`).
    freqs : [F] frequency grid (Hz) `s_pars` is sampled on.
    target_freqs : [n_sc] target frequencies (Hz), e.g. the OFDM subcarrier grid
        (out-of-band targets are clamped to the edge of `freqs`, matching
        `channel.cfr_to_subcarriers`).

    Returns
    -------
    H : [N, n_sc] complex64 torch tensor on the library device.
    """
    if torch.is_tensor(s_pars):
        arr = s_pars.detach().cpu().numpy()
    else:
        arr = np.asarray(s_pars)
    arr = np.asarray(arr, dtype=np.complex64)
    arr = arr.reshape(-1, arr.shape[-1])   # [N, F] -- same flat order as V / U

    freqs = np.asarray(freqs, dtype=np.float64)
    target_freqs = np.asarray(target_freqs, dtype=np.float64)
    tgt_clip = np.clip(target_freqs, freqs.min(), freqs.max())

    Hr = _batch_interp_common_grid(tgt_clip, freqs, arr.real)
    Hi = _batch_interp_common_grid(tgt_clip, freqs, arr.imag)
    H = (Hr + 1j * Hi).astype(np.complex64)
    return torch.from_numpy(H).to(device)


def mrc_weights(H):
    """Per-subcarrier maximum-ratio-combining weights.

    `H` : [N, n_sc] per-element channel (e.g. from `element_channels`). Returns
    `w` : [N, n_sc], unit-norm columns (``||w[:, k]|| == 1``), proportional to
    `H` (matched-filter direction) -- `combine` applies the conjugate at combine
    time, so `w` itself is NOT conjugated here. A subcarrier whose channel is
    (numerically) all-zero across every element gets an all-zero weight column
    (its contribution is defined to be zero rather than NaN/inf).
    """
    H = torch.as_tensor(H, dtype=torch.complex64, device=device)
    norm = torch.linalg.norm(H, dim=0, keepdim=True)          # [1, n_sc]
    safe_norm = torch.clamp(norm, min=1e-12)
    w = H / safe_norm
    return torch.where(norm > 1e-12, w, torch.zeros_like(w))


def egc_weights(H):
    """Per-subcarrier equal-gain-combining weights: the naive phase-only
    beamformer -- co-phase each element (unit-magnitude weight tracking only
    the channel's phase, no amplitude/matched-filter weighting) and sum.

    `H` : [N, n_sc] per-element channel (e.g. from `element_channels`). Returns
    `w` : [N, n_sc], unit-norm columns (``||w[:, k]|| == 1``, matching
    `mrc_weights`'s convention so `combine`'s noise bookkeeping applies
    unchanged): for each element with a nonzero channel tap, `w`'s magnitude is
    ``1 / sqrt(n_valid)`` where `n_valid` is the number of nonzero-channel
    elements in that column (equal gain across all contributing elements,
    unlike MRC's amplitude-proportional weighting), and its phase equals `H`'s
    phase (`combine` conjugates at combine time, so `w` is stored un-conjugated
    here, exactly as `mrc_weights` does). An element whose channel is
    (numerically) zero on a subcarrier has no phase to track and gets a zero
    weight there (same zero-guard behavior as `mrc_weights`); a subcarrier that
    is all-zero across every element gets an all-zero weight column.
    """
    H = torch.as_tensor(H, dtype=torch.complex64, device=device)
    mag = torch.abs(H)
    valid = mag > 1e-12
    phase = torch.where(valid, H / torch.clamp(mag, min=1e-12), torch.zeros_like(H))
    n_valid = valid.sum(dim=0, keepdim=True).to(torch.float32)         # [1, n_sc]
    safe_n = torch.clamp(n_valid, min=1.0)
    w = phase / torch.sqrt(safe_n).to(torch.complex64)
    return torch.where(n_valid > 0, w, torch.zeros_like(w))


def subspace_weights(U):
    """Broadband combining weight vector: the dominant tracked signal direction.

    `U` : [N, d] tracked orthonormal basis (`state['U']` from `AdaOjaBlock`'s Oja
    tracker), in the SAME flat element order as `element_channels` / `V`. Returns
    `u1 = U[:, 0]` (renormalized defensively, though Oja's basis is already
    orthonormal) -- a SINGLE weight vector applied identically at every
    subcarrier (broadband), unlike `mrc_weights` (genuinely per-subcarrier).
    Appropriate when the aperture's dominant tracked direction is a good
    broadband descriptor of where the signal of interest lives spatially.
    """
    U = torch.as_tensor(U, dtype=torch.complex64, device=device)
    u1 = U[:, 0]
    norm = torch.clamp(torch.linalg.norm(u1), min=1e-12)
    return u1 / norm


def combine(rx, w):
    """Coherent spatial combining: ``y[..., k] = w[:, k]^H rx[..., :, k]``.

    `rx` : per-element signal, `[..., N, n_sc]` (the element axis is always
        second-to-last; an optional leading batch axis, e.g. OFDM symbols, is
        supported via broadcasting).
    `w`  : combining weights, either `[N, n_sc]` (per-subcarrier, e.g.
        `mrc_weights`) or `[N]` (a single broadband vector, e.g.
        `subspace_weights`, broadcast across every subcarrier).

    Returns the combined signal with the element axis removed: `[..., n_sc]`
    (or `[n_sc]` for an un-batched `rx`).

    Array-gain bookkeeping: with unit-norm `w` (`||w[:, k]|| == 1`, or `||w||
    == 1` for the broadband case) and i.i.d., unit-variance, INDEPENDENT
    per-element noise injected before this call, the combined noise
    ``w^H n`` has variance ``sum_i |w_i|^2 * Var(n_i) == ||w||^2 * 1 == 1`` --
    unchanged regardless of the number of elements N. So any SNR improvement
    measured after `combine` is honest coherent signal-addition gain, not an
    artifact of the noise having been scaled down by N.
    """
    rx = torch.as_tensor(rx, dtype=torch.complex64, device=device)
    w = torch.as_tensor(w, dtype=torch.complex64, device=device)
    if w.ndim == 1:
        w = w[:, None]                         # [N, 1] broadcasts over subcarriers
    return torch.sum(torch.conj(w) * rx, dim=-2)
