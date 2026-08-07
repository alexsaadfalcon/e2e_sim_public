"""
Single source of truth for the block-diagram pipeline.

Everything the UI knows about the runtime pipeline (which blocks exist, how they
connect, which params are editable, sensible defaults) is *derived* from the
``BLOCKS`` and ``EDGES`` definitions below. The block-diagram layout, the
parameter editor, and the pipeline runner all read from here so there are no
duplicated, hand-maintained strings scattered across the codebase.

This module imports NOTHING heavy (no torch / sionna / e2e). It is pure data so
the UI can be constructed and tested on any machine. The actual block classes
are imported lazily, by id, inside :mod:`webapp.pipeline_runner`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ParamSpec:
    """An editable parameter on a block."""
    key: str
    label: str
    kind: str            # "number" | "int" | "choice"
    default: Any
    choices: Optional[List[Any]] = None
    step: Optional[float] = None
    help: str = ""


@dataclass
class BlockSpec:
    """A node in the pipeline block diagram."""
    id: str
    label: str
    # Whether the user may toggle this block on/off. Environment and the
    # downstream product blocks are structural and always on.
    toggleable: bool = True
    # Whether the block is enabled by default when the UI first loads.
    enabled_default: bool = True
    category: str = "stage"         # "source" | "stage" | "product"
    params: List[ParamSpec] = field(default_factory=list)
    blurb: str = ""


# --------------------------------------------------------------------------------
# The canonical pipeline. Order + optional-ness mirror e2e/simulation.py
# (feed_forward) and the block classes in e2e/blocks.py.
#
#   environment -> [rffe] -> [interconnect] -> [afe (+subspace)] -> products
#
# The four product blocks all read the post-pipeline state_dict in parallel.
# --------------------------------------------------------------------------------

BLOCKS: List[BlockSpec] = [
    BlockSpec(
        id="environment",
        label="Sionna Environment",
        toggleable=False,
        category="source",
        params=[
            ParamSpec("scenario_name", "Scenario", "choice", "munich",
                      choices=["munich", "etoile"],
                      help="Precomputed Sionna RT .pkl frame source."),
        ],
        blurb="Yields S-parameter frames from a precomputed Sionna RT simulation.",
    ),
    BlockSpec(
        id="rffe",
        label="RF Front-End (RFFE)",
        toggleable=True,
        enabled_default=True,  # required by the current backend (records PRX)
        params=[
            ParamSpec("scale_mode", "Scale mode", "choice", "auto",
                      choices=["auto", "legacy", "physical"],
                      help="'auto' follows the frames' own metadata (v2 pkls); "
                           "'physical' forces feeding the frames' absolute volts "
                           "straight into the front-end (for frames generated with "
                           "tx_power_dbm set); 'legacy' forces renormalizing to the "
                           "signal-scaling level below."),
            ParamSpec("signal_scaling", "Signal scaling", "number", 1e-5,
                      step=1e-6, help="Drive level into the analog front-end "
                                      "(legacy scale mode only; ignored in physical)."),
            ParamSpec("freq_span_hz", "Frequency span (Hz)", "number", 3e9,
                      step=1e8, help="Frequency-plan span of the frames; sets the "
                                     "buffer's true sample rate for the noise model."),
        ],
        blurb=("Analog RF front-end circuit distortion (e2e/circuit/rffe_model.py). "
               "Required to run with the current backend."),
    ),
    BlockSpec(
        id="interconnect",
        label="Interconnect",
        toggleable=True,
        enabled_default=False,
        params=[
            ParamSpec("case", "Case", "choice", "default",
                      choices=["default", "case3"],
                      help="'case3' passes the frame through untouched."),
        ],
        blurb="Interconnect filtering applied in the frequency domain.",
    ),
    BlockSpec(
        id="afe",
        label="Adaptive Feature Extraction",
        toggleable=True,
        enabled_default=True,  # required by the current backend's subspace path
        params=[
            ParamSpec("exp", "FP exponent bits", "int", 5, step=1),
            ParamSpec("mantissa", "FP mantissa bits", "int", 6, step=1),
        ],
        blurb=("Quantized matmul / adaptive feature extraction. Pairs with the "
               "subspace block; required to run with the current backend."),
    ),
    BlockSpec(
        id="subspace",
        label="AdaOja Subspace",
        toggleable=True,
        enabled_default=True,
        params=[
            ParamSpec("k", "Subspace dim k", "int", 8, step=1,
                      help="Tracked subspace rank k; also used for U_true."),
        ],
        blurb="Online subspace tracking via Oja's algorithm. Required by AFE.",
    ),
    BlockSpec(
        id="fft",
        label="FFT",
        toggleable=False,
        category="product",
        params=[ParamSpec("bins", "FFT bins", "int", 256, step=1)],
        blurb="Azimuth-elevation power map (coherent aperture FFT, "
              "non-coherent integration over range).",
    ),
    BlockSpec(
        id="range_az",
        label="Range-Azimuth",
        toggleable=False,
        category="product",
        params=[ParamSpec("bins", "FFT bins", "int", 256, step=1)],
        blurb="Range-azimuth power map (non-coherent over elevation).",
    ),
    BlockSpec(
        id="range_el",
        label="Range-Elevation",
        toggleable=False,
        category="product",
        params=[ParamSpec("bins", "FFT bins", "int", 256, step=1)],
        blurb="Range-elevation power map (non-coherent over azimuth).",
    ),
    BlockSpec(
        id="subspace_err",
        label="Subspace Error",
        toggleable=False,
        category="product",
        params=[],
        blurb="Frobenius subspace distance between tracked and true U.",
    ),
    BlockSpec(
        id="comms",
        label="Comms Head (OFDM)",
        toggleable=True,
        enabled_default=False,   # opt-in: existing (radar-only) pipelines unchanged
        category="product",
        params=[
            ParamSpec("combining", "Combining", "choice", "mrc",
                      choices=["element0", "mrc", "subspace"],
                      help="How the array feeds the OFDM demod: 'element0' is the "
                           "historical single-tap SISO shortcut; 'mrc'/'subspace' "
                           "combine across the full aperture."),
            ParamSpec("snr_db", "SNR (dB)", "number", 10.0, step=1.0,
                      help="Per-element AWGN SNR (dB), injected before combining."),
            ParamSpec("fft_size", "FFT size", "int", 64, step=1,
                      help="OFDM subcarrier (FFT) count."),
        ],
        blurb=("Full-aperture spatial combining + OFDM demod; subspace mode uses "
               "the tracker's dominant direction; per-element noise is injected "
               "before combining so any reported array gain is real."),
    ),
]

# Directed dataflow edges (source id -> target id). These define the diagram and
# document the feed-forward order; products fan out from the last serial stage.
EDGES: List[tuple] = [
    ("environment", "rffe"),
    ("rffe", "interconnect"),
    ("interconnect", "afe"),
    ("afe", "subspace"),
    ("subspace", "fft"),
    ("subspace", "range_az"),
    ("subspace", "range_el"),
    ("subspace", "subspace_err"),
    ("subspace", "comms"),
]


# Quick lookups -------------------------------------------------------------------
BLOCKS_BY_ID: Dict[str, BlockSpec] = {b.id: b for b in BLOCKS}

PRODUCT_IDS = [b.id for b in BLOCKS if b.category == "product"]
SERIAL_IDS = [b.id for b in BLOCKS if b.category != "product"]


def default_block_state() -> Dict[str, Dict[str, Any]]:
    """Return the initial {block_id: {"enabled": bool, "params": {...}}} state."""
    state: Dict[str, Dict[str, Any]] = {}
    for b in BLOCKS:
        state[b.id] = {
            "enabled": b.enabled_default,
            "params": {p.key: p.default for p in b.params},
        }
    return state
