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


# ----------------------------------------------------------------- signal domains
# The chain is symmetric: a transmitted waveform in time, a frequency-domain middle
# where propagation and every analog transfer function live, and a received waveform
# back in time. Blocks declare which domain they consume; the two BRIDGE blocks (the
# modulate and dechirp steps) also declare the domain they emit.
#
# Each domain names the state key carrying its payload, because the tensors genuinely
# differ in rank and meaning -- a 4-D S-parameter frame and a 3-D ADC cube are not the
# same object with a flag on it.
DOMAIN_TX_TIME = "tx_time"   # tx_wave [n_tx, n_chirp, n_t]           -- transmitted envelope
DOMAIN_CFR     = "cfr"       # s_pars  [n_rx, n_tx, n_chirp, n_freqs] -- transfer functions
DOMAIN_RX_TIME = "rx_time"   # adc     [n_rx, n_chirp, n_samples]     -- dechirped beat samples

_DOMAINS = (DOMAIN_TX_TIME, DOMAIN_CFR, DOMAIN_RX_TIME)

#: Declared by blocks that do NOT consume the chain's current payload at all: they work
#: on a side channel of their own. The transmit waveform is the motivating case -- it is
#: a TRIBUTARY that joins the chain rather than a segment of it. A waveform generator
#: and the amplifier that distorts it produce and modify `tx_wave` while the main chain
#: still carries the channel's frequency response; the two merge where the transmitted
#: spectrum multiplies the channel, and only THEN does one signal continue.
#: Blocks declaring this are exempt from the domain check (they must still validate
#: their own inputs, and say so clearly when those are missing).
DOMAIN_ANY = "any"

#: State key carrying each domain's payload. `Simulation` uses this to find the tensor a
#: block is about to consume without every block agreeing on one name for everything.
DOMAIN_PAYLOAD_KEY = {
    DOMAIN_TX_TIME: "tx_wave",
    DOMAIN_CFR: "s_pars",
    DOMAIN_RX_TIME: "adc",
}

#: Human-readable hint naming the block that crosses INTO each domain, used in errors.
_DOMAIN_BRIDGE = {
    DOMAIN_CFR: "ModulateBlock",
    DOMAIN_RX_TIME: "DechirpBlock",
}


@dataclass(frozen=True)
class FrameCapabilities:
    """What frame shapes a pipeline block/stage declares it can consume.

    - `accepts_mimo`: the component handles `n_tx > 1` (only meaningful for raw-layout
      frames; see LAYOUT_RAW).
    - `chirps`: how it handles the chirp axis -- CHIRP_NATIVE (operates over it, e.g.
      element-wise physical blocks), CHIRP_BROADCAST (single-chirp core mapped over the
      axis by `broadcast_over_chirps`, stacking results on a leading chirp axis), or
      CHIRP_SINGLE (rejects `n_chirp > 1`).
    - `domain`: which signal domain the component consumes (see DOMAIN_* above).
    - `emits_domain`: set ONLY by the two bridge blocks, naming the domain they hand
      downstream. `None` means the component leaves the domain as it found it.

    The defaults are the historical contract -- no MIMO, single chirp, frequency domain
    -- so a component that declares nothing keeps exactly the old behavior. The axis
    checks (MIMO, chirps) describe the 4-D S-parameter frame and are applied only in
    DOMAIN_CFR; the time-domain payloads carry their own shapes.
    """

    accepts_mimo: bool = False
    chirps: str = CHIRP_SINGLE
    domain: str = DOMAIN_CFR
    emits_domain: str = None

    def __post_init__(self):
        if self.chirps not in _CHIRP_MODES:
            raise ValueError(
                f"unknown chirp capability {self.chirps!r}; expected one of {_CHIRP_MODES}"
            )
        if self.domain not in _DOMAINS + (DOMAIN_ANY,):
            raise ValueError(
                f"unknown signal domain {self.domain!r}; expected one of "
                f"{_DOMAINS + (DOMAIN_ANY,)}"
            )
        if self.emits_domain is not None and self.emits_domain not in _DOMAINS:
            raise ValueError(
                f"unknown emitted signal domain {self.emits_domain!r}; expected one of "
                f"{_DOMAINS} or None"
            )

    @property
    def accepts_multichirp(self):
        return self.chirps != CHIRP_SINGLE

    @property
    def is_bridge(self):
        """True for the blocks that carry the chain from one domain into another."""
        return self.emits_domain is not None and self.emits_domain != self.domain


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


def require_domain(current_domain, component, who=None):
    """Raise FrameContractError unless `component` consumes `current_domain`.

    This is the check that catches a mis-ordered chain -- an impairment placed before
    the dechirp, say -- and names the bridge block that would fix it, rather than
    letting the mistake surface as a shape error deep inside a transform.
    """
    caps = capabilities_of(component)
    who = who or component_name(component)
    if caps.domain == DOMAIN_ANY:
        return
    if caps.domain != current_domain:
        bridge = _DOMAIN_BRIDGE.get(caps.domain)
        remedy = (f"insert a {bridge} before it" if bridge
                  else f"it must run before the chain leaves the {caps.domain} domain")
        raise FrameContractError(
            f"{who} expects the {caps.domain} domain, but the chain is in the "
            f"{current_domain} domain -- {remedy}."
        )


def check_capabilities(s_pars, component, layout=LAYOUT_RAW, who=None,
                       domain=DOMAIN_CFR):
    """Validate `s_pars` against `component`'s declared `FrameCapabilities`.

    Raises FrameContractError naming the component and the offending axis or domain.
    The domain is checked first: it is the coarser error and produces the more
    actionable message. The MIMO (dim 1) check is skipped for non-raw layouts, where
    dim 1 is not a TX axis; both axis checks apply only in DOMAIN_CFR, since they
    describe the 4-D S-parameter frame specifically.
    """
    caps = capabilities_of(component)
    who = who or component_name(component)
    require_domain(domain, component, who)
    if domain != DOMAIN_CFR or caps.domain == DOMAIN_ANY:
        return
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
