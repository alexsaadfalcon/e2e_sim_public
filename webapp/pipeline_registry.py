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

    # ----------------------------------------------------------------------------
    # ADC-cube chain (e2e/chain/*, e2e/environment/blocks.py, e2e/ml/blocks.py). A
    # second, opt-in signal path with two of its own domain bridges: a TX-time
    # waveform crosses into the frequency domain (ModulateBlock), and the
    # frequency-domain channel crosses into RX time (DechirpBlock). All ten blocks
    # default OFF so the existing radar/subspace/comms pipeline above is unaffected
    # until a user opts in. See webapp/pipeline_runner.py for which of these are
    # actually wired into a live run vs. registered for visibility only (the TX-time
    # trio below hits a domain-contract gap in e2e/chain/waveform.py -- see that
    # module's handoff note in the runner).
    # ----------------------------------------------------------------------------
    BlockSpec(
        id="rt_environment",
        label="RT Environment (live ray tracing)",
        toggleable=True,
        enabled_default=False,
        category="source",
        params=[
            ParamSpec("scenario_name", "Scenario", "choice", "munich_radar",
                      choices=["munich_radar", "etoile_radar"],
                      help="Declarative scenario ray-traced fresh every frame "
                           "(needs Sionna/DrJit) -- an alternative to the "
                           "precomputed-.pkl 'Sionna Environment' source above."),
            ParamSpec("base_scene", "Base scene", "choice", "flat",
                      choices=["flat", "free"],
                      help="'flat': a ground plane under the scene objects. "
                           "'free': no ground, objects only."),
            ParamSpec("max_depth", "Max ray bounces", "int", 2, step=1,
                      help="Maximum number of reflections/bounces traced per ray."),
            ParamSpec("include_leakage", "Include TX/RX leakage path", "choice", False,
                      choices=[False, True],
                      help="Also trace the direct antenna-coupling path."),
        ],
        blurb=("Generates each frame by ray tracing a scene live (instead of "
               "reading precomputed frames). Slower, needs Sionna installed, but "
               "reflects the exact scene/target geometry configured below."),
    ),
    BlockSpec(
        id="waveform",
        label="TX Waveform",
        toggleable=True,
        enabled_default=False,
        category="source",
        params=[
            ParamSpec("kind", "Waveform kind", "choice", "fmcw",
                      choices=["fmcw", "narrowband", "wideband"],
                      help="Shape of the synthesized transmitted signal."),
            ParamSpec("bw", "Bandwidth (Hz)", "number", 1e9, step=1e7,
                      help="Swept bandwidth of the transmitted waveform."),
            ParamSpec("sample_rate", "Sample rate (Hz)", "number", 3e9, step=1e8),
            ParamSpec("chirp_duration", "Chirp duration (s)", "number", 1e-6, step=1e-7),
        ],
        blurb=("Synthesizes the actual transmitted waveform (e.g. an FMCW chirp) "
               "instead of assuming an ideal, distortion-free transmitter."),
    ),
    BlockSpec(
        id="tx_pa",
        label="TX Power Amplifier",
        toggleable=True,
        enabled_default=False,
        category="stage",
        params=[
            ParamSpec("gain_db", "Small-signal gain (dB)", "number", 20.0, step=1.0),
            ParamSpec("a_sat", "Saturation amplitude", "number", 1.0, step=0.1,
                      help="Output amplitude the amplifier compresses toward."),
        ],
        blurb=("Distorts the transmit waveform the way a real power amplifier "
               "would: it compresses (and phase-shifts) strong signals instead of "
               "scaling them up linearly forever."),
    ),
    BlockSpec(
        id="modulate",
        label="Modulate (TX -> channel)",
        toggleable=True,
        enabled_default=False,
        category="stage",
        params=[
            ParamSpec("bandwidth_hz", "Ripple axis bandwidth (Hz)", "number", 3e9, step=1e8,
                      help="Frequency span used to phase the PA's ripple response "
                           "(has no effect unless TX Power Amplifier is enabled)."),
        ],
        blurb=("BRIDGE: combines the transmitted waveform's spectrum with the "
               "channel, so downstream stages see what was actually sent, not an "
               "idealized flat transmitter."),
    ),
    BlockSpec(
        id="dechirp",
        label="Dechirp (channel -> ADC)",
        toggleable=True,
        enabled_default=False,
        category="stage",
        params=[
            ParamSpec("preset", "Radar preset", "choice", "ti_iwr1443",
                      choices=["ti_iwr1443", "radial_like"],
                      help="Chirp/frame timing preset shared by this whole ADC-cube "
                           "chain (RT Environment's ray-traced dimensions, this "
                           "block, Impairments, and Radar Cube below)."),
            ParamSpec("mimo", "MIMO scheme", "choice", "tdm",
                      choices=["tdm", "ddma", "single"],
                      help="How multiple transmit antennas share the array; "
                           "overrides the preset's own default."),
        ],
        blurb=("BRIDGE: turns the channel's frequency response into the "
               "dechirped ADC samples a real radar receiver would digitize -- the "
               "entry point of the ADC-cube chain below."),
    ),
    BlockSpec(
        id="impairment",
        label="ADC Impairments",
        toggleable=True,
        enabled_default=False,
        category="stage",
        params=[
            ParamSpec("seed", "Random seed", "int", 0, step=1,
                      help="Seeds the per-frame randomness (deterministic reruns)."),
        ],
        blurb=("Adds realistic receiver imperfections to the digitized signal: "
               "oscillator phase noise, TX/RX antenna leakage, and ground "
               "clutter, all with default physical severities."),
    ),
    BlockSpec(
        id="quantizer",
        label="ADC Quantizer",
        toggleable=True,
        enabled_default=False,
        category="stage",
        params=[
            ParamSpec("bits", "ADC bits", "int", 12, step=1),
            ParamSpec("full_scale", "Full-scale amplitude", "number", 1.0, step=0.1,
                      help="Amplitude (real/imag independently) that hard-clips."),
        ],
        blurb=("Digitizes the signal the way a real analog-to-digital converter "
               "would: a limited number of bits and a hard clip past full scale."),
    ),
    BlockSpec(
        id="radar_cube",
        label="Radar Cube (Range-Doppler)",
        toggleable=True,
        enabled_default=False,
        category="product",
        params=[],
        blurb="Range-Doppler product computed from the digitized ADC samples "
              "(shares the Dechirp block's radar preset).",
    ),
    BlockSpec(
        id="detector",
        label="Neural Detector",
        toggleable=True,
        enabled_default=False,
        category="product",
        params=[
            ParamSpec("input_format", "Model input", "choice", "rd",
                      choices=["rd", "adc"],
                      help="'rd': range-Doppler cube input. 'adc': raw digitized "
                           "samples input."),
            ParamSpec("threshold", "Detection threshold", "number", 0.5, step=0.05),
        ],
        blurb=("Runs a trained neural network (FFTRadNet/SSMRadNet) on the "
               "digitized signal to find targets directly. Needs a trained model "
               "checkpoint, which is not yet selectable from this screen."),
    ),
    BlockSpec(
        id="sink",
        label="Frame Sink (save to disk)",
        toggleable=True,
        enabled_default=False,
        category="product",
        params=[],
        blurb="Saves each frame at this point in the chain to disk, for later "
              "reuse (e.g. to build a training dataset) without rerunning the "
              "chain above it.",
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

    # ADC-cube chain (see the BLOCKS comment above it). Presentational: this is a
    # DAG for the diagram, not a claim that pipeline_runner assembles every one of
    # these edges into one live Simulation call (it does not for the TX-time trio;
    # see pipeline_runner.py). TX-time domain (waveform -> PA) feeds the modulate
    # bridge alongside an existing frequency-domain source (either precomputed
    # 'environment' frames or the live-ray-traced 'rt_environment'); modulate hands
    # back into the same frequency-domain stages (rffe/interconnect) the original
    # pipeline already has; 'interconnect' is also where the RX-time dechirp bridge
    # branches off, continuing through impairments/quantization to the RX-time
    # products (radar cube / neural detector / frame sink).
    ("waveform", "tx_pa"),
    ("tx_pa", "modulate"),
    ("environment", "modulate"),
    ("rt_environment", "modulate"),
    ("rt_environment", "rffe"),
    ("modulate", "rffe"),
    ("interconnect", "dechirp"),
    ("dechirp", "impairment"),
    ("impairment", "quantizer"),
    ("quantizer", "radar_cube"),
    ("quantizer", "detector"),
    ("quantizer", "sink"),
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
