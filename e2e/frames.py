"""THE home of the S-parameter frame shape contract.

Convention (spelled out once, referenced everywhere else): an S-parameter frame is a
torch tensor of shape ``[N_RX, N_TX, n_chirp, N_FREQS]``. Frames are commonly reshaped
onto the physical receive-array grid via ``s_pars.view(n_rx_x, n_rx_y, n_chirp, -1)``
(requiring ``n_rx_x * n_rx_y == N_RX``), and blocks that only handle one chirp slice it
out with ``s_pars[:, :, 0, :]``.

What the runtime pipeline supports is no longer one global rule ("no MIMO, single
chirp") but a **per-block declaration**: every block/stage advertises a
`FrameCapabilities` (whether it accepts MIMO, and how it handles the chirp axis), and
`Simulation` validates each incoming frame against the declaration of the component it
is about to call -- so multi-chirp/MIMO frames flow through the components that
genuinely support them and stop with a named FrameContractError at the ones that don't.

This module centralizes those shape assumptions (dims/validation helpers/capability
declarations) so scattered `assert`s and bare `[:, :, 0, :]` slices elsewhere sit on a
single, clearly named, well-erroring accessor. It is a runtime-layer module (torch-based,
device agnostic) -- unlike `e2e/scenario.py`, it is NOT dependency-free and may import
torch at module scope.
"""

from collections import namedtuple
from dataclasses import dataclass

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


def require_no_mimo(s_pars, who: str, hint: str = None):
    """Raise FrameContractError unless `s_pars` has a single TX (dim 1 == 1).

    `who` names the caller (e.g. a block class) so the error tells the user exactly
    which component lacks MIMO support and what shape it was handed; `hint` appends
    an optional remedy sentence.
    """
    d = dims(s_pars)
    if d.n_tx != 1:
        raise FrameContractError(
            f"{who}: MIMO not supported yet (expected n_tx == 1), got shape "
            f"{tuple(s_pars.shape)}" + (f"; {hint}" if hint else "")
        )


def require_single_chirp(s_pars, who: str, hint: str = None):
    """Raise FrameContractError unless `s_pars` has a single chirp (dim 2 == 1).

    `who` names the caller so the error tells the user exactly which component lacks
    multi-chirp support and what shape it was handed; `hint` appends an optional
    remedy sentence.
    """
    d = dims(s_pars)
    if d.n_chirp != 1:
        raise FrameContractError(
            f"{who}: multiple chirps not supported yet (expected n_chirp == 1), "
            f"got shape {tuple(s_pars.shape)}" + (f"; {hint}" if hint else "")
        )


# ------------------------------------------------------------- per-block capabilities
# How a block/stage handles the chirp axis (dim 2), declared via FrameCapabilities.chirps:
CHIRP_NATIVE = "native"        # consumes the chirp axis itself (element-wise / shape-preserving)
CHIRP_BROADCAST = "broadcast"  # single-chirp core, mapped over chirps (see broadcast_over_chirps)
CHIRP_SINGLE = "single"        # single chirp only; multi-chirp frames are rejected

_CHIRP_MODES = (CHIRP_NATIVE, CHIRP_BROADCAST, CHIRP_SINGLE)

# Frame layouts the pipeline carries. They differ in what dim 1 means, which is why the
# MIMO (dim 1) check only applies to LAYOUT_RAW: after `to_aperture_grid` dim 1 is the
# aperture's second axis (elevation), not TX. The chirp axis is dim 2 in BOTH layouts.
LAYOUT_RAW = "raw"            # [n_rx, n_tx, n_chirp, n_freqs] -- straight from the environment
LAYOUT_APERTURE = "aperture"  # [rx_x, rx_y, n_chirp, n_freqs] -- after to_aperture_grid


@dataclass(frozen=True)
class FrameCapabilities:
    """What frame shapes a pipeline block/stage declares it can consume.

    - `accepts_mimo`: the component handles `n_tx > 1` (only meaningful for raw-layout
      frames; see LAYOUT_RAW).
    - `chirps`: how it handles the chirp axis -- CHIRP_NATIVE (operates over it, e.g.
      element-wise physical blocks), CHIRP_BROADCAST (single-chirp core mapped over the
      axis by `broadcast_over_chirps`, stacking results on a leading chirp axis), or
      CHIRP_SINGLE (rejects `n_chirp > 1`).

    The default is the historical contract -- no MIMO, single chirp -- so a component
    that declares nothing keeps exactly the old behavior.
    """

    accepts_mimo: bool = False
    chirps: str = CHIRP_SINGLE

    def __post_init__(self):
        if self.chirps not in _CHIRP_MODES:
            raise ValueError(
                f"unknown chirp capability {self.chirps!r}; expected one of {_CHIRP_MODES}"
            )

    @property
    def accepts_multichirp(self):
        return self.chirps != CHIRP_SINGLE


#: Conservative fallback for components that declare nothing (legacy custom stages,
#: third-party downstream blocks): the pre-capability contract.
DEFAULT_CAPABILITIES = FrameCapabilities()


def capabilities_of(component):
    """The component's declared `FrameCapabilities`, or DEFAULT_CAPABILITIES."""
    caps = getattr(component, "frame_capabilities", None)
    return caps if isinstance(caps, FrameCapabilities) else DEFAULT_CAPABILITIES


def component_name(component):
    """Name a component in contract errors: its `frame_contract_name` if it sets one
    (stages use it to name the block they wrap), else its class name."""
    return getattr(component, "frame_contract_name", None) or type(component).__name__


def check_capabilities(s_pars, component, layout=LAYOUT_RAW, who=None):
    """Validate `s_pars` against `component`'s declared `FrameCapabilities`.

    Raises FrameContractError naming the component and the offending axis. The MIMO
    (dim 1) check is skipped for non-raw layouts, where dim 1 is not a TX axis.
    """
    caps = capabilities_of(component)
    who = who or component_name(component)
    if layout == LAYOUT_RAW and not caps.accepts_mimo:
        require_no_mimo(s_pars, who, hint="this block declares accepts_mimo=False")
    if not caps.accepts_multichirp:
        require_single_chirp(
            s_pars, who,
            hint="this block declares chirps='single' -- drop the extra chirps (e.g. "
                 "frames.chirp0) or use a block that declares 'native'/'broadcast'",
        )


def broadcast_over_chirps(s_pars, fn):
    """Map a single-chirp function over the frame's chirp axis (dim 2).

    `fn` takes one `[dim0, dim1, n_freqs]` chirp slice. With `n_chirp == 1` this is
    exactly `fn(chirp0(s_pars))` -- the historical single-chirp path, unchanged, with
    the historical output shape. With `n_chirp > 1` the per-chirp results are stacked
    on a NEW LEADING axis, i.e. the output grows a chirp dimension only in the
    multi-chirp case (CHIRP_BROADCAST in FrameCapabilities).
    """
    d = dims(s_pars)
    if d.n_chirp == 1:
        return fn(chirp0(s_pars))
    return torch.stack([fn(s_pars[:, :, c, :]) for c in range(d.n_chirp)], dim=0)


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
