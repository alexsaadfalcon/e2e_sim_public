"""
Single source of truth for the block-diagram pipeline.

Everything the UI knows about which blocks exist, which params are editable and what
their sensible defaults are is *derived* from the ``BLOCKS`` definitions below: the
block-diagram layout, the parameter editor and the pipeline runner all read from here, so
there are no duplicated, hand-maintained strings scattered across the codebase.

HOW THE BLOCKS CONNECT IS NOT DEFINED HERE -- see the note where ``EDGES`` used to be.
The chain's order is the stage list ``e2e.simulation.Simulation`` and
``webapp.pipeline_runner`` build; ``webapp.block_diagram`` draws a collapsed view of that
and nothing restates it.

This module imports NOTHING heavy (no torch / sionna / e2e), with one deliberate
exception: `e2e.interconnect_surrogate` is itself torch-free by contract (see its
own module docstring, pinned by
`tests/test_interconnect_surrogate.py::test_import_does_not_import_torch`), and is
imported here so the Tessera knob ranges below are read from the surrogate's own
recovered training envelope, never hand-typed duplicates that can drift from it.
It is pure data so the UI can be constructed and tested on any machine. The actual
block classes (including `e2e.blocks`, which imports torch) are imported lazily,
by id, inside :mod:`webapp.pipeline_runner`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from e2e.interconnect_surrogate import ARRANGEMENTS, SHIPPED_TSV_DESIGN, VALID_RANGES

from webapp.corpus_catalog import (CORPUS_MANIFESTS, DEFAULT_CORPUS, DEFAULT_SIONNA_SCENARIO,
                                   SIONNA_SCENARIOS)


@dataclass
class ParamSpec:
    """An editable parameter on a block."""
    key: str
    label: str
    kind: str            # "number" | "int" | "choice" | "text" (a free-form string, e.g. a path)
    default: Any
    choices: Optional[List[Any]] = None
    step: Optional[float] = None
    # dcc.Input's `min` -- only declared where a value below it is actually invalid
    # downstream (e.g. subspace k=0 crashes e2e.simulation.rank_diagnostic); None
    # means "no floor", the pre-existing behavior for every other param.
    min: Optional[float] = None
    # dcc.Input's `max` -- same rule as `min`: declared only where a larger value is
    # genuinely invalid downstream (e.g. subspace k >= the tracker's measurement count
    # m makes the sensing matrix unobservable and AdaOjaBlock refuses to construct).
    max: Optional[float] = None
    help: str = ""


@dataclass
class BlockSpec:
    """A node in the pipeline block diagram."""
    id: str
    label: str
    # Whether the user may toggle this block on/off. The precomputed Environment
    # source is structural and always on; every product is switchable (since
    # 2026-09-22 -- a demo preset must be able to hide a panel).
    toggleable: bool = True
    # Whether the block is enabled by default when the UI first loads.
    enabled_default: bool = True
    category: str = "stage"         # "source" | "stage" | "product"
    params: List[ParamSpec] = field(default_factory=list)
    blurb: str = ""


# Measurement count (rows of the adaptive sensing matrix A) the webapp builds its
# AdaOja tracker with. It lives HERE, not next to the AdaOjaBlock construction in
# pipeline_runner, so the UI's bound on k and the value k is actually checked against
# cannot drift apart: AdaOjaBlock refuses k >= m, and before this constant existed the
# ParamSpec had no max at all, so the GUI happily offered k=512 and the resulting
# ValueError escaped run_pipeline's error handling entirely.
SUBSPACE_M = 512

# The Ka-band scale factor `InterconnectBlock(source='tessera')` auto-derives for the
# pipeline's default band (28.5-31.5 GHz munich frames): see e2e/blocks.py
# `_resolve_tessera_scale`/`_TESSERA_SCALE_TARGET_HZ` and F89 (notes/ESTABLISHED_FACTS.md).
# Duplicated as a constant (not imported -- `e2e.blocks` pulls in torch) so the GUI's
# presented knob ranges below can divide the surrogate's own model-space envelope by it;
# `tests/test_webapp.py` cross-checks this against the block's own resolved scale.
_TESSERA_KA_SCALE = 2.0

#: Length parameters the scale model above multiplies/divides (`TESSERA_SCALED_PARAMS`
#: in e2e/blocks.py); `temperature_k` is a material property and is never scaled.
_TESSERA_LENGTH_PARAMS = ("radius_um", "pitch_um", "height_um", "liner_um")


def _tessera_presented_range(name: str) -> tuple:
    """(lo, hi, default) for a Tessera design parameter, PRESENTED (GUI-facing) units:
    the surrogate's own recovered training envelope (`VALID_RANGES`) and shipped demo
    point (`SHIPPED_TSV_DESIGN`), divided by `_TESSERA_KA_SCALE` for the four length
    parameters -- never hand-typed, so a re-measured envelope (e.g. a fresh upstream
    checkout) changes the GUI bounds automatically."""
    lo, hi = VALID_RANGES[name]
    default = SHIPPED_TSV_DESIGN[name]
    if name in _TESSERA_LENGTH_PARAMS:
        lo, hi, default = lo / _TESSERA_KA_SCALE, hi / _TESSERA_KA_SCALE, default / _TESSERA_KA_SCALE
    return lo, hi, default


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
            ParamSpec("scenario_name", "Scenario", "choice", DEFAULT_SIONNA_SCENARIO,
                      choices=SIONNA_SCENARIOS or ["munich"],
                      help="Precomputed Sionna RT .pkl frame source. Only scenarios whose "
                           ".pkl exists under e2e/environment/sionna_sims are listed."),
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
            # step="any": the Thrust 1 preset sets 1e-7, and a browser number input
            # with step=1e-6 reports that as a stepMismatch -> null on blur, even
            # unedited; the runner then fell back to 1e-5 while the field still
            # displayed 1e-7 (rehearsal 2026-09-22). The store also refuses nulls now.
            ParamSpec("signal_scaling", "Signal scaling", "number", 1e-5,
                      step="any", help="Drive level into the analog front-end "
                                       "(legacy scale mode only; ignored in physical)."),
            ParamSpec("freq_span_hz", "Frequency span (Hz)", "number", 3e9,
                      step=1e8, help="Frequency-plan span of the frames; sets the "
                                     "buffer's true sample rate for the noise model."),
            # The two circuit knobs Thrust 1 turns (notes/DEMO_DEFENSE.md). Bounds are
            # the ranges the circuit model was parameterised over (rffe_model.py quotes
            # 0.5-10 mA from the design email); the defaults are the table's own values,
            # so a fresh state is bit-identical to the pre-knob backend. They only move
            # the image near the input-referred noise (signal_scaling ~1e-7); at the
            # 1e-5 default they correctly do nothing, and the help text says so.
            ParamSpec("lna_bias_ma", "LNA bias current (mA)", "number", 8.0,
                      step=0.5, min=0.5, max=10.0,
                      help="Per-element LNA bias. Raises LNA gain and lowers its noise "
                           "contribution; visible only when the signal sits near the "
                           "front-end's own noise (signal scaling ~1e-7). Below ~4 mA "
                           "the modelled LNA is an attenuator."),
            ParamSpec("if_bw_mhz", "IF bandwidth (MHz)", "number", 15.0,
                      step=1.0, min=1.0, max=50.0,
                      help="Receiver IF bandwidth the thermal-noise floor is referenced "
                           "to (noise power scales with it: 1 -> 50 MHz is 17 dB). Does "
                           "not band-limit the signal in this model."),
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
            # NAME COLLISION, deliberately spelled out. "case3" here is a LEGACY alias
            # for an identity pass-through -- it is NOT the Tessera/UIC Case3 transfer
            # function that the README, e2e/data/interconnect/ and the corpus generator
            # all mean by that name. Picking it turns the interconnect OFF, which a user
            # reading the README's Case3 caption would not expect. "passthrough" is the
            # honest label; the alias is kept so old saved states still load.
            ParamSpec("case", "Case", "choice", "default",
                      choices=["default", "passthrough", "case3"],
                      help="'default' = SYNTHETIC 11-tap boxcar PLACEHOLDER (owner ballot "
                           "3A: labelled synthetic wherever it appears; smears a target "
                           "across 11 range bins -- not a real interconnect) -- EXCEPT on "
                           "the live-chain path (Corpus Replay from the stored channel), "
                           "where it is the simulated transfer function that corpus was "
                           "generated with (e2e/data/interconnect/), because that is the "
                           "interconnect those frames actually went through; the run note "
                           "says so. 'passthrough' means pass-through on every path. "
                           "'case3' is a legacy alias for 'passthrough' and does NOT load "
                           "the simulated Case3 hardware response of the same name."),
            ParamSpec("normalize_gain", "Normalize peak gain to 0 dB", "choice", False,
                      choices=[False, True],
                      help="Scale the filter so its peak magnitude is 1. The boxcar "
                           "placeholder otherwise has +20.8 dB of gain, which no passive "
                           "interconnect can have; with this on, only its in-band shape "
                           "(59.7 dB ripple) reaches the chain. Off by default so "
                           "existing runs are unchanged."),
            # Orthogonal to `case`: 'default' keeps the boxcar/CSV behaviour above;
            # 'tessera' evaluates the LIVE public Tessera TSV surrogate instead (see
            # e2e/blocks.py InterconnectBlock's `source=` and F89/F90 in
            # notes/ESTABLISHED_FACTS.md). `case` still gates it: 'passthrough'/'case3'
            # pass the frame through untouched regardless of source.
            ParamSpec("source", "Source", "choice", "default",
                      choices=["default", "tessera"],
                      help="'default' = the boxcar placeholder / transfer_csv above. "
                           "'tessera' = the live public Tessera/UIC TSV surrogate, "
                           "evaluated from the five knobs below over a geometric scale "
                           "model (the run banner's describe() names the factor and the "
                           "frequency it was evaluated at)."),
            # The five continuous design parameters TesseraTSV.s21 takes (TESSERA_DESIGN_
            # PARAMS in e2e/blocks.py), PRESENTED units -- bounds and defaults come from
            # `_tessera_presented_range`, i.e. the surrogate's own recovered training
            # envelope divided by the Ka-band scale factor, never hand-typed here.
            ParamSpec("tessera_radius_um", "Tessera: via radius (um)", "number",
                      _tessera_presented_range("radius_um")[2], step=0.1,
                      min=_tessera_presented_range("radius_um")[0],
                      max=_tessera_presented_range("radius_um")[1],
                      help="TSV via radius. Only applied when source='tessera'; "
                           "presented range is the surrogate's training envelope / "
                           f"{_TESSERA_KA_SCALE:g} (the scale model's Ka-band factor)."),
            ParamSpec("tessera_pitch_um", "Tessera: via pitch (um)", "number",
                      _tessera_presented_range("pitch_um")[2], step=0.5,
                      min=_tessera_presented_range("pitch_um")[0],
                      max=_tessera_presented_range("pitch_um")[1],
                      help="Centre-to-centre via spacing. The public checkpoint's "
                           "NEXT/FEXT-vs-pitch trend is only physical below ~23 GHz "
                           "(F89) -- inverted at our band, so this knob's crosstalk is "
                           "shown as fixed reference numbers on the demo card, not a "
                           "live sweep."),
            # Biggest genuine in-band-shape mover of the five (measured through this
            # same block on the munich Ka band, notes/TESSERA_KNOB_MEASUREMENT_2026-
            # 09-23.md): most of what any knob moves on a peak-normalized range profile
            # is sub-milli-bin bulk DELAY, not distortion -- see that note before adding
            # a "watch the image change shape" claim to a card.
            ParamSpec("tessera_height_um", "Tessera: TSV height (um)", "number",
                      _tessera_presented_range("height_um")[2], step=0.5,
                      min=_tessera_presented_range("height_um")[0],
                      max=_tessera_presented_range("height_um")[1],
                      help="Through-silicon via height/depth. The largest single-knob "
                           "mover of the range-profile skirt of the five (measured "
                           "2026-09-23); the movement is bulk delay (it shifts the "
                           "target), not an in-band shape change."),
            ParamSpec("tessera_liner_um", "Tessera: liner oxide thickness (um)", "number",
                      _tessera_presented_range("liner_um")[2], step=0.05,
                      min=_tessera_presented_range("liner_um")[0],
                      max=_tessera_presented_range("liner_um")[1],
                      help="Oxide liner thickness around each via."),
            ParamSpec("tessera_temperature_k", "Tessera: temperature (K)", "number",
                      _tessera_presented_range("temperature_k")[2], step=5.0,
                      min=_tessera_presented_range("temperature_k")[0],
                      max=_tessera_presented_range("temperature_k")[1],
                      help="Die temperature. Not scaled by the geometric scale model "
                           "(a material property, not a length)."),
            ParamSpec("tessera_arrangement", "Tessera: arrangement", "choice",
                      "ring3x3", choices=sorted(ARRANGEMENTS),
                      help="Signal/ground via grid. 'ring3x3'/'ring5x5' have one signal "
                           "via (no crosstalk terms); 'checker3x3'/'checker5x5' have "
                           "several and expose NEXT/FEXT, at the cost of a bigger S-"
                           "matrix per frequency point."),
        ],
        blurb="Interconnect filtering applied in the frequency domain.",
    ),
    BlockSpec(
        id="afe",
        label="Adaptive Feature Extraction",
        toggleable=True,
        enabled_default=True,  # required by the current backend's subspace path
        params=[
            # min=4: at exp<=3 every combining weight underflows to zero and the radar
            # image is exactly blank (measured on this default state). The backend now
            # raises rather than returning that blank silently, but there is no reason
            # to let the spinner walk into a guaranteed error -- 4 is the last value
            # that still produces a real, if badly degraded, image. The backend guard
            # remains the real backstop, since the underflow point moves with the
            # weight scale and this floor is only a UI convenience.
            ParamSpec("exp", "FP exponent bits", "int", 5, step=1, min=4,
                      help="Exponent bits in the AFE's low-precision float weight "
                           "format. 4 is heavily degraded but still images; the "
                           "default 5 is clean."),
            # min=0: measured -- mantissa=0 still quantizes cleanly (no underflow, ~26%
            # max relative weight error), so 0 is a legitimate, very coarse setting and
            # not a floor to forbid. A NEGATIVE mantissa is what breaks (it drives a
            # negative bit-shift inside the format's packing).
            ParamSpec("mantissa", "FP mantissa bits", "int", 6, step=1, min=0,
                      help="Mantissa bits in the same format. This is the knob that "
                           "moves the subspace-error plot; the range-azimuth image "
                           "barely responds to it."),
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
                      min=1, max=SUBSPACE_M - 1,
                      help=f"Tracked subspace rank k; also used for U_true. Must stay "
                           f"below the tracker's measurement count "
                           f"m={SUBSPACE_M}: at k == m the adaptive sensing matrix is "
                           f"nothing but the anchor rows and the estimate can never "
                           f"update."),
            # Thrust 3 Demo B (notes/DEMO_DEFENSE.md): the shipped tracker is warm-started
            # from the TRUE subspace, so every curve begins ~1e-3 from the answer. A cold
            # start is the honest run and the more impressive one -- it reaches the warm
            # floor by frame 3 at 2:1 compression -- but it was never reachable from the UI.
            ParamSpec("warm_start", "Tracker initialisation", "choice", "warm",
                      choices=["warm", "cold"],
                      help="'warm': start from a perturbed copy of the true subspace "
                           "(the historical default; the error curve then shows tracking "
                           "lag only). 'cold': start from a random basis with no peek at "
                           "ground truth, so the curve shows acquisition from scratch."),
        ],
        blurb="Online subspace tracking via Oja's algorithm. Required by AFE.",
    ),
    BlockSpec(
        id="fft",
        label="FFT",
        toggleable=True,
        category="product",
        params=[ParamSpec("bins", "FFT bins", "int", 256, step=1)],
        blurb="Azimuth-elevation power map (coherent aperture FFT, "
              "non-coherent integration over range).",
    ),
    BlockSpec(
        id="range_az",
        label="Range-Azimuth",
        toggleable=True,
        category="product",
        params=[ParamSpec("bins", "FFT bins", "int", 256, step=1)],
        blurb="Range-azimuth power map (non-coherent over elevation).",
    ),
    BlockSpec(
        id="range_el",
        label="Range-Elevation",
        toggleable=True,
        category="product",
        params=[ParamSpec("bins", "FFT bins", "int", 256, step=1)],
        blurb="Range-elevation power map (non-coherent over azimuth).",
    ),
    BlockSpec(
        id="range_profile",
        label="Range Profile",
        toggleable=True,
        category="product",
        params=[ParamSpec("bins", "FFT bins", "int", 256, step=1)],
        blurb=("Per-channel range profile (FFT along frequency only, no aperture "
               "transform). Compression mixes only the aperture axis, so range "
               "survives compression even where angle does not -- this is the one "
               "product that stays valid on compressed (reduced-dimension) "
               "measurements without a DecompressBlock."),
    ),
    BlockSpec(
        id="subspace_err",
        label="Subspace Error",
        toggleable=True,
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
                      choices=["element0", "egc", "mrc", "subspace"],
                      help="How the array feeds the OFDM demod: 'element0' is the "
                           "historical single-tap SISO shortcut; 'egc' (naive "
                           "phase-only beamforming) / 'mrc' / 'subspace' combine "
                           "across the full aperture."),
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
    # until a user opts in. See webapp/pipeline_runner.py for how each is wired. The
    # TX-time trio (waveform/tx_pa/modulate) IS wired and runs; an earlier version of
    # this comment claimed it hit a "domain-contract gap", which was wrong -- the note it
    # referred to is a CPU-device-placement workaround for RandomWidebandSignal, not a
    # contract blocker.
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
        id="corpus_environment",
        label="Corpus Replay (stored frames)",
        toggleable=True,
        enabled_default=False,
        category="source",
        params=[
            ParamSpec("manifest", "Corpus manifest (path)", "text", DEFAULT_CORPUS,
                      help="Repo-relative or absolute path to a generated corpus's "
                           "manifest.json. Found on this machine: "
                           + (", ".join(CORPUS_MANIFESTS) if CORPUS_MANIFESTS
                              else "none -- corpora are generated locally, not tracked")),
            # Owner directive 2026-09-23 ("store the ray tracing, compute everything
            # else live with the knobs"): a corpus generated with `--store-cfr` keeps
            # the ray-traced channel in a `.cfr.npy` sidecar beside each frame, so the
            # WHOLE analog/digital chain can be re-run from it with the GUI's values
            # instead of being frozen at generation time. "auto" takes that path
            # whenever the corpus has the sidecars and falls back to the stored ADC
            # cube when it does not -- which is every corpus generated before
            # 2026-09-23, so their screens are unchanged.
            ParamSpec("domain", "Replay from", "choice", "auto",
                      choices=["auto", "cfr", "adc"],
                      help="'cfr' = replay the stored RAY-TRACED CHANNEL and run the "
                           "RF front end / dechirp / thermal floor / impairments / IF "
                           "high-pass / quantizer LIVE with the values set below "
                           "(needs a corpus generated with --store-cfr). 'adc' = "
                           "replay the stored ADC cube and skip that chain (the "
                           "pre-2026-09-23 behaviour). 'auto' = cfr when the corpus "
                           "has the sidecars, else adc."),
            ParamSpec("split", "Split", "choice", "test",
                      choices=["test", "val"],
                      help="'test' is the split held out from training. It is the one "
                           "the published detection numbers were scored on for the "
                           "corpus they were scored on (b1_bench_v3); on any other "
                           "corpus it is simply that corpus's held-out split."),
            ParamSpec("start_frame", "First frame index", "int", 0, step=1, min=0,
                      help="Frames are replayed in manifest order from this index; "
                           "each run step advances one frame."),
        ],
        blurb=("Replays the frames of a generated ML corpus -- with their stored "
               "ground-truth labels -- as the source. 'Replay from' decides where the "
               "frame enters: the stored RAY-TRACED CHANNEL (the ADC chain then runs "
               "live with the knobs below, so a front-end setting reaches the "
               "detector), or the stored ADC cube (the chain is skipped -- the frame "
               "is already past it). Either way the frequency-domain products (FFT / "
               "range-azimuth / subspace) do not apply; use Radar Cube and the "
               "Detector. This is how the Thrust 5 demo runs a detector on held-out "
               "ray-traced frames whose ground truth rides along with them."),
    ),
    BlockSpec(
        id="waveform",
        label="TX Waveform",
        toggleable=True,
        enabled_default=False,
        category="source",
        params=[
            ParamSpec("kind", "Waveform kind", "choice", "fmcw",
                      choices=["fmcw", "wideband"],
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
            ParamSpec("preset", "Radar preset", "choice", "radial_like",
                      choices=["ti_iwr1443", "radial_like", "benchmark_v1",
                               "ddma_wide_v1"],
                      help="Chirp/frame timing preset shared by this whole ADC-cube "
                           "chain (RT Environment's ray-traced dimensions, this "
                           "block, Impairments, and Radar Cube below). radial_like "
                           "(12 TX x 16 RX = 192 virtual elements) is the default: the "
                           "detection label grid's 192 azimuth bins only carry "
                           "information at that array size. benchmark_v1 (TDM) and "
                           "ddma_wide_v1 (DDMA, also 192 virtual) are the two presets "
                           "whose targets do NOT alias at scene speeds -- the ones a "
                           "detection benchmark is valid on (F43). IGNORED on the "
                           "live-chain path (Corpus Replay from the stored channel), "
                           "where the geometry is the corpus manifest's: dechirping a "
                           "stored frame at a different chirp count or slope would not "
                           "reproduce it, it would silently produce a different one."),
            ParamSpec("mimo", "MIMO scheme", "choice", "ddma",
                      choices=["tdm", "ddma", "single"],
                      help="How multiple transmit antennas share the array; "
                           "overrides the preset's own default."),
        ],
        blurb=("BRIDGE: turns the channel's frequency response into the "
               "dechirped ADC samples a real radar receiver would digitize -- the "
               "entry point of the ADC-cube chain below."),
    ),
    BlockSpec(
        id="thermal_noise",
        label="Link Budget / Thermal Floor",
        toggleable=True,
        enabled_default=False,
        category="stage",
        params=[
            ParamSpec("seed", "Random seed", "int", 0, step=1,
                      help="Seeds the per-frame noise draw (deterministic reruns). "
                           "IGNORED on the live-chain path (Corpus Replay from the "
                           "stored channel): each replayed frame carries the seed it "
                           "was generated with, and the run uses that, so the live "
                           "chain reproduces the corpus exactly."),
        ],
        blurb=("Puts the cube on an absolute power scale (transmit power) and adds "
               "the physical k*T*B*F thermal noise floor. Enable together with ADC "
               "Impairments: their severities are specified in dB relative to this "
               "floor, and without it they are calibrated against nothing (the "
               "corpus generator always runs this stage)."),
    ),
    BlockSpec(
        id="impairment",
        label="ADC Impairments",
        toggleable=True,
        enabled_default=False,
        category="stage",
        params=[
            ParamSpec("seed", "Random seed", "int", 0, step=1,
                      help="Seeds the per-frame randomness (deterministic reruns). "
                           "IGNORED on the live-chain path (Corpus Replay from the "
                           "stored channel): the replayed frame carries both the seed "
                           "and the domain-randomised severities it was generated "
                           "with, and the run uses those. THERE IS NO SEVERITY/LEVEL "
                           "KNOB on this block -- the severities are per-stage "
                           "dataclasses (phase noise / leakage / clutter), not a "
                           "scalar the UI could offer."),
        ],
        blurb=("Adds realistic receiver imperfections to the digitized signal: "
               "oscillator phase noise, TX/RX antenna leakage, and ground "
               "clutter, all with default physical severities."),
    ),
    BlockSpec(
        id="if_hpf",
        label="IF High-Pass",
        toggleable=True,
        enabled_default=False,
        category="stage",
        params=[
            ParamSpec("corner_range_m", "Corner range (m)", "number", 1.0, step=0.1,
                      help="Range below which returns are suppressed (the corner "
                           "frequency, stated the way a spec sheet states it)."),
            ParamSpec("order", "Filter order", "int", 2, step=1),
        ],
        blurb=("The IF-chain high-pass every FMCW receiver puts between the mixer "
               "and the ADC: suppresses the huge range-0 leakage and bumper tones "
               "before digitization. Enable together with ADC Impairments + "
               "Quantizer -- without it the leakage tone sets the ADC full scale "
               "and the chain reproduces pre-A2 physics (the corpus generator "
               "always runs this stage)."),
    ),
    BlockSpec(
        id="quantizer",
        label="ADC Quantizer",
        toggleable=True,
        enabled_default=False,
        category="stage",
        params=[
            ParamSpec("bits", "ADC bits", "int", 12, step=1, min=1, max=24,
                      help="Converter resolution. On the live-chain path (Corpus "
                           "Replay from the stored channel) this is the knob that "
                           "moves detections: 12 bits is what the corpora were "
                           "generated at."),
            ParamSpec("full_scale", "Full-scale amplitude", "number", 1.0, step=0.1,
                      min=0.0,
                      help="Amplitude (real/imag independently) that hard-clips. "
                           "0 = AUTOMATIC GAIN (full scale set from the frame's own "
                           "peak with 6 dB of headroom) -- what the corpus generator "
                           "uses, and the only workable setting on a physically "
                           "scaled cube, whose returns sit near 1e-7 and would "
                           "quantize to exactly zero against a fixed 1.0."),
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
        label="Detector (CFAR | ML)",
        toggleable=True,
        enabled_default=False,
        category="product",
        params=[
            # ONE detection block with two modes (owner ballot, 2026-09-17): CFAR with its
            # own tunable knobs, or a pretrained network loaded from a checkpoint path.
            # No training happens in the GUI.
            ParamSpec("mode", "Detector", "choice", "cfar",
                      choices=["cfar", "ml"],
                      help="'cfar': classical cell-averaging CFAR on the range-azimuth "
                           "power map (the baseline every published number is compared "
                           "against). 'ml': a trained checkpoint (FFTRadNet, SSMRadNet "
                           "or RADDetNet); its input format -- rd, adc or rad -- is read "
                           "from the checkpoint and derived by the same function the "
                           "training dataset uses."),
            ParamSpec("checkpoint", "ML checkpoint (path)", "text", "",
                      help="ML mode only. Path to a best.pt written by e2e.ml.train, "
                           "e.g. e2e/ml/runs/b5_fftradnet_v3/best.pt. Checkpoints are "
                           "not tracked by git; the demo machine needs the file."),
            # step 0.01: the presets pin each detector at its recall-0.5 operating
            # point (0.66 / 0.22 / 0.44, from e2e/ml/runs/beat_cfar.json), which a
            # 0.05 grid could not hold -- typing 0.44 became null, then 0.5.
            ParamSpec("threshold", "Decode threshold", "number", 0.5, step=0.01,
                      min=0.0, max=1.0,
                      help="Objectness above which a local peak is reported as a "
                           "detection. This is an operating point, not the metric: AP "
                           "in the notes integrates the whole curve."),
            ParamSpec("cfar_guard", "CFAR guard cells", "int", 2, step=1, min=1,
                      help="CFAR mode only. Half-width of the guard ring excluded "
                           "around the cell under test."),
            ParamSpec("cfar_train", "CFAR training cells", "int", 6, step=1, min=1,
                      help="CFAR mode only. Half-width of the annulus the noise level "
                           "is averaged over."),
        ],
        blurb=("Finds targets in the digitized signal and draws an objectness map with "
               "decoded detections -- and the stored ground truth, when the frame came "
               "from the Corpus Replay source. CFAR mode is the shipped baseline "
               "(e2e.ml.baseline); ML mode runs a pretrained network."),
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

# THE DIAGRAM'S TOPOLOGY DOES NOT LIVE HERE ANY MORE.
#
# A hand-maintained `EDGES` list used to sit at this point, and `webapp/block_diagram.py`
# drew it. It is deleted, not moved, because with the one-chain rewrite (2026-09-24) it had
# become a SECOND answer to "how does the pipeline connect": the order the chain actually
# runs in is `e2e.simulation.Simulation._build_spine` plus the stage list
# `webapp/pipeline_runner.py` composes, and the diagram is now derived from the collapsed
# view of that (`block_diagram._CHAIN` / `_PRODUCT_TAP`). Two documents answering one
# question means one of them is already wrong, and this one was: it still carried
# ("interconnect", "dechirp", "alt") and ("quantizer", "radar_cube") after the range
# transform had moved in between them.
#
# `normalize_edge` went with it -- its only reason to exist was the "alt" edge kind, and
# there are no alternative paths left to mark.


# Quick lookups -------------------------------------------------------------------
BLOCKS_BY_ID: Dict[str, BlockSpec] = {b.id: b for b in BLOCKS}

#: Ceiling on frames per run, enforced by BOTH the n_steps spinner and run_pipeline
#: (the spinner alone is advisory: a typed 1000 used to start a run that held the
#: server for an hour with no way to stop it). 50 is well above every demo preset
#: (<= 20, see webapp/demo_presets.py) and below where the per-frame oracle SVD makes
#: a run tedious; raise it here, in one place, if a study needs more.
MAX_N_STEPS = 50

#: Ceiling for the DEMO presets specifically (notes/DEMO_DEFENSE.md DO-NOT-SHOW #9: past
#: ~20 frames the per-frame cost triples and the tracker's rank-collapse spike returns).
#: Lives here so `webapp/demo_presets.py` and the runner's error text quote one number.
MAX_PRESET_N_STEPS = 20

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
