"""THE home of the S-parameter frame shape contract.

Convention (spelled out once, referenced everywhere else): an S-parameter frame is a
torch tensor of shape ``[N_RX, N_TX, n_chirp, N_FREQS]``. The runtime pipeline
(`e2e/simulation.py`, `e2e/blocks.py`) currently only supports **no MIMO** (`N_TX == 1`)
and a **single chirp** (`n_chirp == 1`); frames are commonly reshaped onto the physical
receive-array grid via ``s_pars.view(n_rx_x, n_rx_y, n_chirp, -1)`` (requiring
``n_rx_x * n_rx_y == N_RX``), and downstream blocks frequently slice out the single
chirp with ``s_pars[:, :, 0, :]``.

This module centralizes those shape assumptions (dims/validation helpers) so scattered
`assert`s and bare `[:, :, 0, :]` slices elsewhere can migrate onto a single, clearly
named, well-erroring accessor. It is a runtime-layer module (torch-based, device
agnostic) -- unlike `e2e/scenario.py`, it is NOT dependency-free and may import torch at
module scope.
"""

from collections import namedtuple

import torch


FrameDims = namedtuple("FrameDims", "n_rx n_tx n_chirp n_freqs")


class FrameContractError(ValueError):
    """A frame violated the shape contract or a block's declared capability limit
    (wrong ndim, MIMO, multiple chirps, aperture factorization).

    Subclasses ValueError so existing ``except ValueError`` callers keep working;
    the dedicated type lets UIs distinguish "a documented pipeline constraint was
    hit" from an arbitrary error (see webapp/pipeline_runner.py)."""


def dims(s_pars):
    """Return the `FrameDims` of an S-parameter frame.

    Validates `s_pars` is a 4-D tensor `[N_RX, N_TX, n_chirp, N_FREQS]`; raises
    FrameContractError (a ValueError) with the offending shape otherwise.
    """
    if s_pars.ndim != 4:
        raise FrameContractError(
            f"expected an S-parameter frame with 4 dims [n_rx, n_tx, n_chirp, n_freqs], "
            f"got shape {tuple(s_pars.shape)} ({s_pars.ndim} dims)"
        )
    n_rx, n_tx, n_chirp, n_freqs = s_pars.shape
    return FrameDims(n_rx, n_tx, n_chirp, n_freqs)


def require_no_mimo(s_pars, who: str):
    """Raise FrameContractError unless `s_pars` has a single TX (dim 1 == 1).

    `who` names the caller (e.g. a block class) so the error tells the user exactly
    which component lacks MIMO support and what shape it was handed.
    """
    d = dims(s_pars)
    if d.n_tx != 1:
        raise FrameContractError(
            f"{who}: MIMO not supported yet (expected n_tx == 1), got shape {tuple(s_pars.shape)}"
        )


def require_single_chirp(s_pars, who: str):
    """Raise FrameContractError unless `s_pars` has a single chirp (dim 2 == 1).

    `who` names the caller so the error tells the user exactly which component lacks
    multi-chirp support and what shape it was handed.
    """
    d = dims(s_pars)
    if d.n_chirp != 1:
        raise FrameContractError(
            f"{who}: multiple chirps not supported yet (expected n_chirp == 1), "
            f"got shape {tuple(s_pars.shape)}"
        )


def chirp0(s_pars):
    """Return the canonical `[:, :, 0, :]` slice: `[dim0, dim1, dim3]`.

    Slices out dim 2 (the chirp axis) of ANY 4-D tensor whose dim 2 is the chirp axis --
    both raw frames `[n_rx, n_tx, n_chirp, n_freqs]` and aperture grids
    `[rx_x, rx_y, n_chirp, n_freqs]` (e.g. after `to_aperture_grid`). This is the
    single-chirp view several downstream blocks (FFTBlock, RangeAzBlock, RangeElBlock,
    ...) consume. `s_pars` must be 4-D; ndim is validated via `dims`.
    """
    dims(s_pars)  # validate ndim==4 with a clear error before slicing
    return s_pars[:, :, 0, :]


def to_aperture_grid(s_pars, array_shape):
    """Reshape a frame's RX dimension onto the physical receive-array grid.

    `array_shape` is `(rx_x, rx_y)`; returns a view of shape
    `[rx_x, rx_y, n_chirp, n_freqs]`. Raises FrameContractError (a ValueError) if
    `n_rx != rx_x * rx_y`, naming both the requested grid and the frame's actual `n_rx`.

    Physical ordering: this is a row-major reshape, so `rx_x` must be the axis that
    varies SLOWEST along the frame's flat RX dimension. Sionna's PlanarArray numbers
    antennas column-first (the row index varies fastest), so for Sionna-generated
    frames the correct call is ``array_shape=(num_cols, num_rows)`` — grid dim 0 is
    then columns (horizontal/azimuth) and dim 1 rows (vertical/elevation), the
    convention RangeAzBlock/RangeElBlock assume. `SionnaEnvironmentBlock` performs
    that swap when auto-deriving from v2 metadata (which stores [num_rows, num_cols]).
    """
    d = dims(s_pars)
    rx_x, rx_y = array_shape
    if d.n_rx != rx_x * rx_y:
        raise FrameContractError(
            f"array_shape {(rx_x, rx_y)} (= {rx_x * rx_y} elements) does not factor "
            f"n_rx={d.n_rx}"
        )
    return s_pars.view(rx_x, rx_y, d.n_chirp, d.n_freqs)
