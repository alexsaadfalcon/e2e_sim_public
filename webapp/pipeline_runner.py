"""
Runs the runtime block pipeline from a UI block-state dict, and turns the
outputs into Plotly figures.

ALL heavy imports (torch, e2e.blocks, e2e.simulation) happen lazily inside
:func:`run_pipeline` so this module — and the whole app shell — imports cleanly
on a machine without torch installed. The functions here are the only place the
"Run" action touches the real simulator.

Failure modes are surfaced as :class:`PipelineError` with a friendly message:
  * torch / e2e not importable          -> "torch not installed ..."
  * the scenario .pkl frames are missing -> "generate frames first ..."

The TX-time trio ("waveform" / "tx_pa" / "modulate", from e2e.chain.waveform) IS wired,
as a tributary of the chain rather than a segment of it: the waveform and amplifier
blocks declare `frames.DOMAIN_ANY` because they produce and modify `tx_wave` without
consuming the chain's payload at all, and "modulate" is the merge point where the
transmitted spectrum multiplies the channel response. They therefore run BEFORE the
receive-side stages, whatever domain the chain itself happens to be in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go

from webapp.pipeline_registry import BLOCKS_BY_ID, MAX_N_STEPS, SUBSPACE_M

# Speed of light (m/s), used to convert the frequency-FFT axis to physical range.
_C = 2.99792458e8


class PipelineError(Exception):
    """Raised with a user-facing message when a run cannot complete."""


def _p(state: Dict[str, Dict[str, Any]], block_id: str, key: str) -> Any:
    """Fetch a param value for a block from UI state, falling back to default."""
    spec = BLOCKS_BY_ID[block_id]
    params = state.get(block_id, {}).get("params", {})
    if key in params and params[key] is not None:
        return params[key]
    for ps in spec.params:
        if ps.key == key:
            return ps.default
    raise KeyError(key)


# Params that MUST be strictly positive: a zero/negative value here produces a
# divide-by-zero (e.g. RFFEBlock scales by signal_scaling / mean(|frame|), and a
# 0 scaling yields 0/0 NaNs downstream) or an empty FFT. For these we treat a
# non-positive (or None) value as "unset" and fall back to the registry default,
# rather than letting it poison the run. Other params keep the plain _p policy.
_POSITIVE_PARAMS = {
    ("rffe", "signal_scaling"),
    ("rffe", "freq_span_hz"),
    ("rffe", "lna_bias_ma"),
    ("rffe", "if_bw_mhz"),
    ("fft", "bins"),
    ("range_az", "bins"),
    ("range_el", "bins"),
    ("comms", "fft_size"),
}


def _p_positive(state: Dict[str, Dict[str, Any]], block_id: str, key: str) -> Any:
    """Like :func:`_p` but falls back to the default when the value is <= 0."""
    val = _p(state, block_id, key)
    if val is None or val <= 0:
        spec = BLOCKS_BY_ID[block_id]
        for ps in spec.params:
            if ps.key == key:
                return ps.default
    return val


def _enabled(state: Dict[str, Dict[str, Any]], block_id: str) -> bool:
    return bool(state.get(block_id, {}).get("enabled", False))


def _resolve_physical_scale(mode: str, env_block: Any) -> bool:
    """Map the rffe scale_mode param to RFFEBlock's physical_scale bool.

    'physical'/'legacy' force the value; 'auto' (or anything else) follows the
    environment block's own ``physical_scale`` metadata (v2 pkls only -- legacy
    pkls expose it as absent/None, which falls back to False, i.e. unchanged
    legacy behavior).
    """
    if mode == "physical":
        return True
    if mode == "legacy":
        return False
    return bool(getattr(env_block, "physical_scale", None))


def _resolve_freq_span_hz(state: Dict[str, Dict[str, Any]], env_block: Any) -> float:
    """The RFFE buffer's true sample-rate span (Hz).

    Prefers the environment block's own ``freq_plan`` metadata (v2 pkls) so the
    frontend automatically matches the frames' actual band; falls back to the
    rffe freq_span_hz UI param for legacy pkls (no freq_plan).
    """
    freq_plan = getattr(env_block, "freq_plan", None)
    if freq_plan:
        return float(freq_plan["stop_hz"]) - float(freq_plan["start_hz"])
    return float(_p_positive(state, "rffe", "freq_span_hz"))


def _comms_freqs(state: Dict[str, Dict[str, Any]], env_block: Any) -> np.ndarray:
    """Frequency grid (Hz) for the comms head's `ModemBlock`.

    Prefers the environment block's own ``freq_plan`` metadata (v2 pkls):
    ``linspace(start_hz, stop_hz, num_freqs)`` -- mirrors :func:`_resolve_freq_span_hz`.
    Falls back (legacy pkls, no ``freq_plan``) to a span centered at 30 GHz built
    from the rffe ``freq_span_hz`` UI param, matching the reference scenarios'
    carrier. Either way the point count follows the actual frame's frequency-
    sample count when available (best-effort; a fixed default otherwise).
    """
    try:
        n_freqs = int(env_block.get_S_pars().shape[-1])
    except Exception:
        n_freqs = 64
    freq_plan = getattr(env_block, "freq_plan", None)
    if freq_plan:
        num = int(freq_plan.get("num_freqs") or n_freqs)
        return np.linspace(float(freq_plan["start_hz"]), float(freq_plan["stop_hz"]), num)
    span = _resolve_freq_span_hz(state, env_block)
    carrier = 30e9
    return np.linspace(carrier - span / 2.0, carrier + span / 2.0, n_freqs)


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_repo_path(text: Any) -> Path:
    """A path typed into the UI: absolute as given, otherwise relative to the repo root
    (NOT the process CWD, which for a Dash server is wherever it was launched from)."""
    p = Path(str(text or "").strip())
    return p if p.is_absolute() else _REPO_ROOT / p


def _corpus_source(state: Dict[str, Dict[str, Any]]):
    """Build the Corpus Replay source; returns `(block, cfg, grid)`."""
    manifest_text = str(_p(state, "corpus_environment", "manifest") or "").strip()
    if not manifest_text:
        raise PipelineError(
            "The Corpus Replay source needs a corpus manifest path (Corpus Replay -> "
            "'Corpus manifest'). Corpora are generated locally and are not tracked by "
            "git; none was found under e2e/ml/datasets/ on this machine."
        )
    manifest = _resolve_repo_path(manifest_text)
    if not manifest.is_file():
        raise PipelineError(f"Corpus manifest not found: {manifest}")
    try:
        from e2e.ml.blocks import CorpusSourceBlock
    except ImportError as e:
        raise PipelineError(
            "Could not import the corpus replay source (e2e.ml.blocks). "
            "Underlying error: " + str(e)
        )
    try:
        src = CorpusSourceBlock(
            manifest,
            split=str(_p(state, "corpus_environment", "split")),
            start=int(_p(state, "corpus_environment", "start_frame") or 0),
        )
    except (KeyError, IndexError, FileNotFoundError, ValueError) as e:
        raise PipelineError(f"Could not open the corpus: {e}")
    return src, src.cfg, src.grid


def _build_detector(state: Dict[str, Dict[str, Any]], cfg, grid):
    """The Detector product in either mode. `cfg`/`grid` describe the ADC cube it
    consumes -- from the corpus manifest, or from the dechirp preset for a live chain."""
    mode = str(_p(state, "detector", "mode"))
    threshold = float(_p(state, "detector", "threshold"))
    if mode == "cfar":
        try:
            from e2e.ml.blocks import CFARDetectorBlock
        except ImportError as e:
            raise PipelineError("Could not import the CFAR detector (e2e.ml.blocks). "
                                "Underlying error: " + str(e))
        try:
            return CFARDetectorBlock(
                cfg, grid, threshold=threshold,
                guard=int(_p_positive(state, "detector", "cfar_guard")),
                train=int(_p_positive(state, "detector", "cfar_train")),
            )
        except ValueError as e:
            raise PipelineError(f"Detector: {e}")
    if mode != "ml":
        raise PipelineError(f"Unknown detector mode {mode!r}; choose 'cfar' or 'ml'")
    ckpt_text = str(_p(state, "detector", "checkpoint") or "").strip()
    if not ckpt_text:
        raise PipelineError(
            "Detector in ML mode needs a trained checkpoint path (Detector -> "
            "'ML checkpoint'), e.g. e2e/ml/runs/b5_fftradnet_v3/best.pt. Switch the "
            "mode to 'cfar' to run the classical detector instead."
        )
    ckpt = _resolve_repo_path(ckpt_text)
    if not ckpt.is_file():
        raise PipelineError(f"ML checkpoint not found: {ckpt}")
    try:
        from e2e.ml.blocks import NeuralDetectorBlock
    except ImportError as e:
        raise PipelineError("Could not import the neural detector (e2e.ml.blocks). "
                            "Underlying error: " + str(e))
    try:
        return NeuralDetectorBlock(str(ckpt), mode="infer", cfg=cfg, grid=grid,
                                   threshold=threshold)
    except Exception as e:
        raise PipelineError(f"Could not load the ML checkpoint {ckpt.name}: {e}")


def run_pipeline(state: Dict[str, Dict[str, Any]], n_steps: int = 10,
                 should_stop=None) -> Dict[str, Any]:
    """
    Build blocks from ``state`` and run ``n_steps`` of the simulation.

    Returns the simulation ``outputs`` dict (lists of per-frame torch tensors).
    Raises :class:`PipelineError` for the expected, recoverable failure modes.
    """
    # --- lazy heavy imports -----------------------------------------------------
    try:
        import torch  # noqa: F401
        from e2e.frames import FrameContractError
        from e2e.simulation import Simulation
        from e2e.blocks import (
            SionnaEnvironmentBlock,
            RFFEBlock,
            InterconnectBlock,
            AFEBlock,
            AdaOjaBlock,
            FFTBlock,
            RangeAzBlock,
            RangeElBlock,
            RangeProfileBlock,
            SubspaceErrorBlock,
            CircuitStage,
            InterconnectStage,
        )
    except ImportError as e:  # torch / sionna deps not present
        raise PipelineError(
            "Could not import the simulation backend (torch / e2e). "
            "The Run action needs torch installed. Underlying error: " + str(e)
        )

    N_TX = 1

    scenario_name = _p(state, "environment", "scenario_name")
    # Frame-count ceiling, checked before any block is built so an oversized request
    # costs nothing. The spinner in the UI carries the same bound (block_diagram.py).
    if int(n_steps) > MAX_N_STEPS:
        raise PipelineError(
            f"Frames to run = {int(n_steps)} exceeds the ceiling of {MAX_N_STEPS}. The "
            f"demo presets use at most 20 (past ~20 frames the per-frame cost triples and "
            f"the tracker's rank-collapse spike returns); raise MAX_N_STEPS in "
            f"webapp/pipeline_registry.py for a study."
        )

    k = int(_p(state, "subspace", "k"))
    # AdaOjaBlock raises ValueError for k >= m, and it does so at CONSTRUCTION -- which
    # happens below, outside the try/except that wraps the run. So without this check
    # the exception bypasses every PipelineError handler and reaches the UI as a raw
    # "Unexpected error: m must exceed k ...". The ParamSpec's `max` is only a browser
    # hint; a typed value or a saved state can still arrive out of range, so the floor
    # and the ceiling are both enforced here as well.
    if k >= SUBSPACE_M:
        raise PipelineError(
            f"Subspace dim k must be < {SUBSPACE_M} (the tracker's measurement count "
            f"m); got k={k}. At k >= m the adaptive sensing matrix is nothing but the "
            f"anchor rows, so the estimate can never update."
        )

    # The simulation backend now handles RFFE-off (PRX is initialized to None),
    # AFE-off (the no-AFE subspace branch calls subspace.update(X, A) with two args),
    # and even subspace-off (the measurement stage is skipped) cleanly. The webapp
    # still always constructs a subspace block below because its results view always
    # includes SubspaceErrorBlock, which consumes the tracker's 'U'.

    # --- environment block (loads the .pkl frames; can raise FileNotFoundError) -
    # "RT Environment" (rt_environment) is an opt-in ALTERNATIVE source: instead of
    # reading precomputed .pkl frames it ray-traces a declarative Scenario live, via
    # e2e.environment.blocks.RTEnvironmentBlock (needs Sionna/DrJit -- guarded behind
    # its own lazy import, same per-feature pattern as the comms head below, so a
    # machine without Sionna can still run every pipeline that leaves this off).
    # Corpus replay: the frame enters the chain already digitized, so it is an
    # alternative to BOTH frequency-domain sources and to the whole dechirp chain.
    corpus_mode = _enabled(state, "corpus_environment")
    corpus_cfg = corpus_grid = None
    if corpus_mode and _enabled(state, "rt_environment"):
        raise PipelineError(
            "Corpus Replay and RT Environment are both sources -- enable one, not both."
        )
    if corpus_mode:
        environment_block, corpus_cfg, corpus_grid = _corpus_source(state)
    elif _enabled(state, "rt_environment"):
        try:
            from e2e.environment.blocks import RTEnvironmentBlock
            from e2e.radar_config import PRESETS
            from e2e.scenario import REFERENCE_SCENARIOS
        except ImportError as e:
            raise PipelineError(
                "Could not import the live ray-tracing backend (e2e.environment / "
                "e2e.ml / e2e.scenario -- needs Sionna+DrJit installed). "
                "Underlying error: " + str(e)
            )
        rt_scenario_name = _p(state, "rt_environment", "scenario_name")
        if rt_scenario_name not in REFERENCE_SCENARIOS:
            raise PipelineError(f"Unknown RT scenario '{rt_scenario_name}'")
        # The radar chirp/frame-timing preset is owned by the "dechirp" block's
        # param (see webapp/pipeline_registry.py) because it is shared by the whole
        # ADC-cube chain, not just this source; read regardless of whether "dechirp"
        # itself is enabled.
        rt_preset_name = _p(state, "dechirp", "preset")
        if rt_preset_name not in PRESETS:
            raise PipelineError(f"Unknown radar preset '{rt_preset_name}'")
        try:
            _scenario_for_view = REFERENCE_SCENARIOS[rt_scenario_name]()
            environment_block = RTEnvironmentBlock(
                _scenario_for_view,
                PRESETS[rt_preset_name],
                base_scene=_p(state, "rt_environment", "base_scene"),
                max_depth=int(_p(state, "rt_environment", "max_depth")),
                include_leakage=bool(_p(state, "rt_environment", "include_leakage")),
            )
        except Exception as e:
            raise PipelineError(f"Could not build the RT environment: {e}")
    else:
        try:
            environment_block = SionnaEnvironmentBlock(scenario_name)
        except FileNotFoundError as e:
            raise PipelineError(
                f"No precomputed frames found for scenario '{scenario_name}'. "
                "Generate frames first (Scenario tab -> Generate frames), or pick a "
                f"scenario whose .pkl exists. Missing file: {e}"
            )
        except ValueError as e:
            raise PipelineError(str(e))

    # Derive the receive-array size from the environment block's array_shape so the
    # Oja tracker dimension and Simulation's view() agree with the actual frames.
    # A non-(32,32) array would otherwise silently disagree with a hardcoded 32*32.
    array_shape = getattr(environment_block, "array_shape", (32, 32)) or (32, 32)
    N_RX = int(array_shape[0]) * int(array_shape[1])

    # --- optional serial blocks -------------------------------------------------
    circuit_block = None
    if _enabled(state, "rffe"):
        circuit_block = RFFEBlock(
            n=N_RX * N_TX,
            freq_span_hz=_resolve_freq_span_hz(state, environment_block),
            signal_scaling=float(_p_positive(state, "rffe", "signal_scaling")),
            physical_scale=_resolve_physical_scale(
                _p(state, "rffe", "scale_mode"), environment_block
            ),
            # Thrust 1's circuit knobs. Non-positive values fall back to the registry
            # defaults via _p_positive: both are divisors inside the circuit model.
            lna_bias_ma=float(_p_positive(state, "rffe", "lna_bias_ma")),
            if_bw_mhz=float(_p_positive(state, "rffe", "if_bw_mhz")),
        )

    interconnect_block = None
    if _enabled(state, "interconnect"):
        case = _p(state, "interconnect", "case")
        interconnect_block = InterconnectBlock(
            case=None if case == "default" else case,
            normalize_gain=bool(_p(state, "interconnect", "normalize_gain")),
        )

    afe_block = None
    if _enabled(state, "afe"):
        afe_block = AFEBlock(
            exp=int(_p(state, "afe", "exp")),
            mantissa=int(_p(state, "afe", "mantissa")),
        )

    # The real invariant: a subspace (AdaOja) tracker is ALWAYS built, regardless of
    # the "subspace" toggle's state. AFE draws its combining weights from the
    # tracker whenever it's enabled, and the results view always runs
    # SubspaceErrorBlock, which needs the tracker's 'U' -- so there is no state in
    # which skipping it would be correct. (There used to be a dead "AFE requires
    # subspace" guard here; it could never fire because this line already builds
    # the tracker whenever AFE is on. The UI reflects this honestly: see
    # block_diagram.param_editor, which disables the subspace checkbox while AFE
    # is enabled instead of pretending to toggle a no-op.)
    # gap_response="refine": the reactive-refinement mitigation for the munich
    # rank-deficiency divergence, wired at the ENTRY POINT (the class default stays
    # "none" for bit-compat) -- adversarial-panel finding: the fix existed but no
    # shipped path used it, so default runs past ~frame 22 still diverged.
    subspace_block = AdaOjaBlock(N_RX, k, m=SUBSPACE_M, n_refine=10,
                                 gap_response="refine")

    # --- downstream product blocks (always present unless the ADC-cube chain is --
    # active -- see below) --------------------------------------------------------
    fft_bins = int(_p_positive(state, "fft", "bins"))
    range_az_bins = int(_p_positive(state, "range_az", "bins"))
    range_el_bins = int(_p_positive(state, "range_el", "bins"))
    range_profile_bins = int(_p_positive(state, "range_profile", "bins"))
    # Each classic product is built only when enabled. They were structural (always
    # on) until 2026-09-22; a demo preset needs to hide a panel that contradicts its
    # own story (the FFT az-el view in Thrust 2, notes/DEMO_DEFENSE.md #8), and a
    # product the UI shows as switchable must actually switch.
    classic_products = [
        ("fft", lambda: FFTBlock(bins=fft_bins)),
        ("range_az", lambda: RangeAzBlock(bins=range_az_bins)),
        ("range_el", lambda: RangeElBlock(bins=range_el_bins)),
        ("range_profile", lambda: RangeProfileBlock(bins=range_profile_bins)),
        ("subspace_err", lambda: SubspaceErrorBlock()),
    ]
    downstream_blocks = [build() for bid, build in classic_products if _enabled(state, bid)]
    # `rx_cfg`/`rx_grid` describe the ADC cube the RX-time products consume, when there
    # is one: set by the corpus source, or by the dechirp chain below.
    rx_cfg, rx_grid = corpus_cfg, corpus_grid
    if corpus_mode:
        # A replayed frame is already past every frequency-domain stage and product;
        # Simulation would refuse them at the frame contract. Run none of them.
        downstream_blocks = []

    # --- optional ADC-cube chain (e2e/chain/dechirp.py, e2e/chain/receive.py) ----
    # "dechirp" is this chain's activation toggle: it BRIDGES the frequency-domain
    # frame into a dechirped ADC cube (state['signal_domain'] flips from DOMAIN_CFR
    # to DOMAIN_RX_TIME -- see e2e/frames.py), which is a different domain than the
    # radar/subspace/comms products built above consume. So enabling it REPLACES
    # Simulation's default serial-stage build (the composability hook Simulation
    # itself documents -- see its `serial_stages=` kwarg) and the downstream product
    # list, rather than being appended alongside them; the two chains are mutually
    # exclusive within one run. (The TX-time trio -- waveform/tx_pa/modulate -- is
    # NOT wired in here: see this module's docstring.)
    serial_stages_override = None
    if corpus_mode:
        # No serial stages at all: the corpus frame was generated by this very chain
        # (dechirp -> thermal floor -> impairments -> IF HPF -> quantizer) and stored
        # AFTER it. Re-running any of it here would impair an already-impaired frame.
        serial_stages_override = []
    elif _enabled(state, "dechirp"):
        try:
            from e2e.chain.dechirp import DechirpBlock
            from e2e.radar_config import PRESETS
        except ImportError as e:
            raise PipelineError(
                "Could not import the ADC-cube chain backend (e2e.chain.dechirp / "
                "e2e.radar_config). Underlying error: " + str(e)
            )
        preset_name = _p(state, "dechirp", "preset")
        if preset_name not in PRESETS:
            raise PipelineError(f"Unknown radar preset '{preset_name}'")
        import dataclasses as _dc
        adc_cfg = _dc.replace(PRESETS[preset_name], mimo=_p(state, "dechirp", "mimo"))

        # CircuitStage/InterconnectStage mirror the existing radar path's stages,
        # but run on the RAW [n_rx, n_tx, n_chirp, n_freqs] layout (no GridStage):
        # DechirpBlock's antenna-axis handling needs the raw RX/TX axes, not the
        # aperture-grid reshape GridStage would produce (see e2e/chain/dechirp.py).
        serial_stages_override = []

        # The transmit tributary, if the user enabled it: generate the waveform, distort
        # it in the amplifier, then merge its spectrum into the channel response. These
        # run first because everything after them is the receive side.
        if _enabled(state, "waveform"):
            try:
                from e2e.chain.waveform import ModulateBlock, TxPABlock, WaveformBlock
                from e2e.circuit.tx_pa import TxPA, TxPAConfig
            except ImportError as e:
                raise PipelineError(
                    "Could not import the transmit chain (e2e.chain.waveform / "
                    "e2e.circuit.tx_pa). Underlying error: " + str(e)
                )
            serial_stages_override.append(WaveformBlock(
                kind=_p(state, "waveform", "kind"),
                bw=float(_p(state, "waveform", "bw")),
                sample_rate=float(_p(state, "waveform", "sample_rate")),
                chirp_duration=float(_p(state, "waveform", "chirp_duration")),
            ))
            tx_pa = None
            if _enabled(state, "tx_pa"):
                tx_pa = TxPA(TxPAConfig(
                    small_signal_gain_db=float(_p(state, "tx_pa", "gain_db")),
                    a_sat=float(_p(state, "tx_pa", "a_sat")),
                ))
                serial_stages_override.append(TxPABlock(tx_pa))
            if _enabled(state, "modulate"):
                serial_stages_override.append(ModulateBlock(
                    tx_pa=tx_pa,
                    bandwidth_hz=float(_p(state, "modulate", "bandwidth_hz")),
                ))

        if circuit_block is not None:
            serial_stages_override.append(CircuitStage(circuit_block))
        if interconnect_block is not None:
            serial_stages_override.append(InterconnectStage(interconnect_block))
        serial_stages_override.append(DechirpBlock(adc_cfg))

        # Stage order mirrors e2e.ml.chain_generate.build_chain_simulation exactly:
        # Dechirp -> ThermalNoise -> Impairment -> IFHighPass -> Quantizer (D6 parity;
        # each stage's position is load-bearing -- see the corpus builder's comments).
        if _enabled(state, "thermal_noise"):
            try:
                from e2e.chain.link_budget import ThermalNoiseBlock
            except ImportError as e:
                raise PipelineError(
                    "Could not import the link-budget stage (e2e.chain.link_budget). "
                    "Underlying error: " + str(e)
                )
            serial_stages_override.append(
                ThermalNoiseBlock(adc_cfg, seed=int(_p(state, "thermal_noise", "seed")))
            )

        if _enabled(state, "impairment"):
            # No-impossible-states guard (batch physics review 2026-08-24): the
            # impairment severities are ABSOLUTE dB above the k*T*B*F thermal floor
            # (impairments.DEFAULT_POWER_REFERENCE) -- without the link-budget stage
            # the cube has no absolute scale, so the injected leakage/clutter land at
            # physically meaningless levels and nothing reports a problem. Refuse
            # loudly instead of running the silently-wrong chain.
            if not _enabled(state, "thermal_noise"):
                raise PipelineError(
                    "ADC Impairments needs the Link Budget / Thermal Floor block: its "
                    "severities are calibrated in dB above the thermal floor that "
                    "block establishes. Enable 'Link Budget / Thermal Floor' too (the "
                    "corpus generator always runs both together)."
                )
            try:
                from e2e.chain.receive import ImpairmentBlock
            except ImportError as e:
                raise PipelineError(
                    "Could not import the ADC impairments stage (e2e.chain.receive). "
                    "Underlying error: " + str(e)
                )
            serial_stages_override.append(
                ImpairmentBlock(adc_cfg, seed=int(_p(state, "impairment", "seed")))
            )

        if _enabled(state, "if_hpf"):
            try:
                from e2e.chain.receive import IFHighPassBlock
            except ImportError as e:
                raise PipelineError(
                    "Could not import the IF high-pass stage (e2e.chain.receive). "
                    "Underlying error: " + str(e)
                )
            serial_stages_override.append(IFHighPassBlock(
                adc_cfg,
                corner_range_m=float(_p(state, "if_hpf", "corner_range_m")),
                order=int(_p(state, "if_hpf", "order")),
            ))

        if _enabled(state, "quantizer"):
            try:
                from e2e.chain.receive import QuantizerBlock
            except ImportError as e:
                raise PipelineError(
                    "Could not import the ADC quantizer stage (e2e.chain.receive). "
                    "Underlying error: " + str(e)
                )
            serial_stages_override.append(QuantizerBlock(
                bits=int(_p(state, "quantizer", "bits")),
                full_scale=float(_p(state, "quantizer", "full_scale")),
            ))

        # None of the frequency-domain products above apply once the chain has
        # crossed into RX time; replace them with the RX-time products instead.
        downstream_blocks = []
        # The RX-time products (radar cube, detector) are built in the shared section
        # below, from this chain's cube geometry. The label grid mirrors the corpus
        # generator's convention (e2e.ml.labels.LabelGrid.for_config) so a live chain
        # and a replayed corpus frame draw on the same axes.
        rx_cfg = adc_cfg
        try:
            from e2e.ml.labels import LabelGrid
            rx_grid = LabelGrid.for_config(adc_cfg)
        except ImportError:
            rx_grid = None
        if (_enabled(state, "radar_cube") or _enabled(state, "detector")) \
                and not _enabled(state, "rt_environment"):
            # The radar cube folds the chirp axis, so it needs a frame carrying the
            # preset's full chirp count. Precomputed .pkl frames are always SINGLE-chirp,
            # so this pairing fails deep inside adc_to_rd with a bare shape mismatch
            # ("adc has 1 chirps but cfg.n_chirps=252"), which reads as a bug rather than
            # a configuration error. Say what is actually wrong, as the comms-head guard
            # below does. Not caused by the preset default -- it failed identically at
            # the old preset's 192 chirps.
            raise PipelineError(
                "The Radar Cube and Detector products need a multi-chirp frame, but "
                "the Environment source replays precomputed .pkl frames, which are "
                "single-chirp. Enable the RT Environment source (it ray-traces "
                f"{adc_cfg.n_chirps} chirps to match the '{_p(state, 'dechirp', 'preset')}' "
                "preset), replay a generated corpus with the Corpus Replay source, or "
                "turn them off and use the frequency-domain products instead."
            )
        if _enabled(state, "sink"):
            try:
                from e2e.ml.blocks import SinkBlock
                from e2e.frames import DOMAIN_RX_TIME
            except ImportError as e:
                raise PipelineError(
                    "Could not import the frame-sink product (e2e.ml.blocks). "
                    "Underlying error: " + str(e)
                )
            sink_dir = Path(__file__).resolve().parent / "_sink_output"
            downstream_blocks.append(SinkBlock(sink_dir, tag="webapp", domain=DOMAIN_RX_TIME))

    # --- RX-time products, shared by the live dechirp chain and corpus replay ------
    if rx_cfg is not None:
        if _enabled(state, "radar_cube"):
            try:
                from e2e.chain.receive import RadarCubeBlock
            except ImportError as e:
                raise PipelineError(
                    "Could not import the radar-cube product (e2e.chain.receive). "
                    "Underlying error: " + str(e)
                )
            downstream_blocks.append(RadarCubeBlock(rx_cfg))
        if _enabled(state, "detector"):
            if rx_grid is None:
                raise PipelineError("The Detector needs the label grid (e2e.ml.labels), "
                                    "which could not be imported.")
            downstream_blocks.append(_build_detector(state, rx_cfg, rx_grid))
    elif _enabled(state, "radar_cube") or _enabled(state, "detector"):
        raise PipelineError(
            "The Radar Cube and Detector products consume a digitized ADC cube. Enable "
            "the Dechirp chain (with the RT Environment source) or the Corpus Replay "
            "source to produce one."
        )

    # --- optional comms head (swappable "product": OFDM demod instead of / -------
    # alongside the radar products above). Appended AFTER the radar products so it
    # composes without disturbing their output ordering. Incompatible with the
    # ADC-cube chain above (also a frequency-domain consumer).
    comms_combining = None
    if _enabled(state, "comms"):
        if serial_stages_override is not None:
            raise PipelineError(
                "The Comms Head and the ADC-cube chain (Dechirp/Impairments/"
                "Quantizer/...) consume different signal domains and cannot run "
                "together -- disable one of them."
            )
        try:
            from e2e.comms.blocks import ModemBlock, BERBlock
        except ImportError as e:
            raise PipelineError(
                "Could not import the comms backend (e2e.comms). "
                "Underlying error: " + str(e)
            )
        comms_combining = _p(state, "comms", "combining")
        comms_snr_db = float(_p(state, "comms", "snr_db"))
        comms_fft_size = int(_p_positive(state, "comms", "fft_size"))
        comms_freqs = _comms_freqs(state, environment_block)
        try:
            modem_block = ModemBlock(
                comms_freqs, fft_size=comms_fft_size, snr_db=comms_snr_db,
                combining=comms_combining,
            )
        except ValueError as e:
            raise PipelineError(str(e))
        downstream_blocks.append(modem_block)
        downstream_blocks.append(BERBlock())

    # Enabling "dechirp" alone (no Radar Cube / Neural Detector / Frame Sink, and no
    # Comms Head -- comms is mutually exclusive with dechirp and already rejected
    # above) would otherwise run to completion with an EMPTY downstream_blocks
    # list -- Simulation.run happily produces zero outputs, which the Results tab
    # renders identically to "never ran" ("No results yet"). Fail loudly instead,
    # before sim.run() below. Checked here (after both branches that can populate
    # downstream_blocks for the ADC-cube case) so the more specific comms-conflict
    # message above still wins when both apply.
    if serial_stages_override is not None and not downstream_blocks:
        raise PipelineError(
            "The ADC-cube chain is enabled but no ADC-chain product (Radar "
            "Cube / Neural Detector / Frame Sink) is enabled -- enable one, "
            "or disable Dechirp to run the frequency-domain products."
        )

    sim = Simulation(
        environment_block,
        downstream_blocks,
        k,
        circuit_block,
        interconnect_block,
        afe_block,
        subspace_block,
        array_shape=array_shape,
        serial_stages=serial_stages_override,
        # "cold" leaves Oja's random basis untouched -- the honest acquisition run
        # (Thrust 3 Demo B). The registry default "warm" keeps the historical numbers.
        warm_start=(_p(state, "subspace", "warm_start") != "cold"),
    )

    try:
        outputs = sim.run(n_steps=max(1, int(n_steps)), should_stop=should_stop)
    except FileNotFoundError as e:
        raise PipelineError(
            "A required data file was missing during the run (generate frames "
            f"first). Underlying error: {e}"
        )
    except (AssertionError, FrameContractError) as e:
        # FrameContractError: the e2e/frames.py shape-contract guards (no MIMO,
        # single chirp, aperture factorization) that used to be bare asserts.
        raise PipelineError(f"Pipeline constraint failed: {e}")
    except ValueError as e:
        # e2e.simulation.rank_diagnostic raises ValueError("rank_diagnostic requires
        # k >= 1, got k=...") for a non-positive subspace k; that internal function
        # name means nothing to a UI user, so translate it to the param they can
        # actually fix.
        if "rank_diagnostic" in str(e):
            raise PipelineError("Subspace dim k must be >= 1.")
        # compress.quantize_weights raises when the AFE's float format is too narrow
        # to represent ANY combining weight. Untranslated, that run "succeeds" into a
        # blank heatmap (see that guard's comment); translated, it names the knob.
        if "flushed every" in str(e):
            raise PipelineError(
                f"AFE 'FP exponent bits' is too small: {e} "
                f"(the radar image would be blank)."
            )
        raise PipelineError(f"Pipeline run failed: ValueError: {e}")
    except Exception as e:  # surface anything else cleanly to the UI
        raise PipelineError(f"Pipeline run failed: {type(e).__name__}: {e}")

    # Stash the axis metadata figures_from_outputs needs to label heatmaps physically
    # (bins used per product + the raw frame's frequency-sample count/span) alongside
    # the product outputs. Reserved-key checks in Simulation.feed_forward only guard
    # per-frame state_dict keys, not this top-level dict, so a leading-underscore key
    # here is safe and figures_from_outputs never copies it into its returned figs.
    n_freqs = None
    try:
        n_freqs = int(environment_block.get_S_pars().shape[-1])
    except Exception:
        pass  # axis metadata is best-effort; figures_from_outputs falls back to bins
    # Prefer the environment block's own freq_plan (v2 pkls) so the range axis matches
    # the frames' true band automatically, regardless of whether rffe is enabled.
    freq_plan = getattr(environment_block, "freq_plan", None)
    if freq_plan:
        freq_span_hz = float(freq_plan["stop_hz"]) - float(freq_plan["start_hz"])
        if freq_plan.get("num_freqs") is not None:
            n_freqs = int(freq_plan["num_freqs"])
    else:
        freq_span_hz = (
            float(_p_positive(state, "rffe", "freq_span_hz")) if _enabled(state, "rffe") else 3e9
        )
    outputs["_axis_meta"] = {
        "fft_bins": fft_bins,
        "range_az_bins": range_az_bins,
        "range_el_bins": range_el_bins,
        "range_profile_bins": range_profile_bins,
        "n_freqs": n_freqs,
        "freq_span_hz": freq_span_hz,
        # True when the values above came from the frames' own v2 metadata (env
        # block freq_plan), not from UI params/fallbacks -- lets the UI say so.
        "from_meta": bool(getattr(environment_block, "freq_plan", None)),
    }
    # How many frames actually ran, and whether the run was cut short by Cancel, so
    # the UI labels partial results as partial.
    outputs["_axis_meta"]["n_steps_run"] = int(getattr(sim, "n_steps_run", n_steps))
    outputs["_axis_meta"]["cancelled"] = bool(getattr(sim, "cancelled", False))
    if rx_cfg is not None:
        # Geometry of the ADC cube the RX-time products were built on, so their
        # figures carry physical axes (range in m, radial velocity in m/s, sin(az)).
        outputs["_axis_meta"]["rx"] = {
            "range_resolution_m": float(rx_cfg.range_resolution_m),
            "velocity_resolution_mps": float(rx_cfg.velocity_resolution_mps),
            "max_range_m": float(rx_cfg.max_range_m),
            "grid": (None if rx_grid is None else {
                "n_range": int(rx_grid.n_range), "n_azimuth": int(rx_grid.n_azimuth),
                "max_range_m": float(rx_grid.max_range_m),
            }),
        }
    if comms_combining is not None:
        # Small metadata dict figures_from_outputs reads to label the BER figure
        # (mirrors "_axis_meta" above); leading underscore keeps it out of the
        # reserved per-frame state_dict keys Simulation.feed_forward guards.
        outputs["_comms_meta"] = {"combining": comms_combining}
    return outputs


# --------------------------------------------------------------------------------
# Output -> Plotly figure helpers. Kept torch-tolerant: they only import torch
# when actually given tensors, and accept the outputs dict produced above.
# --------------------------------------------------------------------------------

def _to_numpy_abs_db(tensor):
    import torch
    # Guard the normalization against an all-zero product: dividing by max(|.|)==0
    # would yield NaNs across the whole heatmap. Clamp the denominator to a small
    # epsilon so an all-zero product becomes a finite, floored (-inf-clamped) map.
    peak = torch.max(torch.abs(tensor))
    peak = torch.clamp(peak, min=1e-12)
    t = tensor / peak
    # The fft/range_az/range_el products are POWER maps (|.|^2, non-coherent
    # integration), so dB relative to peak is 10*log10(P/Pmax). (Complex amplitude
    # inputs would take 20*log10; all heatmap products routed here are power.)
    db = 10 * torch.log10(torch.abs(t) + 1e-12)
    return db.T.detach().cpu().numpy()


def _to_numpy_complex(tensor):
    """Flatten a (possibly torch) complex tensor to a 1-D numpy array.

    Torch-tolerant like `_to_numpy_abs_db`: only touches `.detach()`/`.cpu()` when
    the value actually carries them (a torch tensor), so a plain numpy array/list
    (e.g. a hand-built outputs dict in a test) also works.
    """
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu().numpy()
    return np.asarray(tensor).reshape(-1)


def _sin_angle_axis(n_bins: int):
    """fftshifted aperture-FFT bin index -> normalized sine-angle u = sin(theta).

    Half-wavelength element spacing puts the unambiguous field of view at
    u in [-1, 1); bin 0 of the fftshifted axis is the most-negative angle, bin
    n_bins//2 is broadside (u=0).
    """
    return (np.arange(n_bins) - n_bins // 2) / (n_bins / 2)


def _range_axis(n_bins: int, freq_span_hz: float, n_freqs: int):
    """fftshifted range DISPLAY-gate index -> physical range (meters).

    The range blocks compress over the FULL frequency band (all n_freqs samples
    spanning bandwidth freq_span_hz = B, sample spacing df = B / n_freqs, so the
    native round-trip range-per-bin is c / (2*B)), then power-bin the n_freqs
    fftshifted native range bins down to n_bins display gates. This calibration MUST
    mirror e2e.blocks._power_bin exactly: it groups per = ceil(n_freqs / n_bins)
    native bins into each gate, so range-per-gate is per * c / (2*B), NOT the
    exact-ratio c*n_freqs / (2*B*n_bins) (they differ whenever n_bins does not divide
    n_freqs -- the production case n_freqs~5000, n_bins=256 has per=20 vs 19.53).

    Zero range: the block fftshifts the native range axis (zero-delay DC bin -> index
    n_freqs // 2) BEFORE power-binning (which pads at the high-index end), so the
    zero-range gate is (n_freqs // 2) // per, which is n_bins // 2 only when n_bins
    divides n_freqs. Deriving both from `per` keeps the axis aligned in every case.

    Sign: the range blocks take a FORWARD fft over frequency, so a physical delay
    +tau (a target at +R) lands on the NEGATIVE side of the fftshifted axis; the
    axis is negated here so physical targets read at positive range.
    """
    per = -(-n_freqs // n_bins)               # ceil(n_freqs / n_bins); matches _power_bin
    range_per_gate = per * _C / (2.0 * freq_span_hz)
    zero_gate = (n_freqs // 2) // per
    return -(np.arange(n_bins) - zero_gate) * range_per_gate


def _heatmap(data_db, title: str, *, x=None, y=None,
             xlabel: str = "Bin", ylabel: str = "Bin") -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=data_db, x=x, y=y, colorbar=dict(title="power (dB)"), zmin=-40, zmax=0
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        margin=dict(l=40, r=20, t=40, b=40),
        height=360,
    )
    return fig


#: Plan-view styling per object class. Kept here rather than derived from the mesh so a
#: scenario with no assets (the dry-run path) still renders a readable diagram.
_TOPDOWN_STYLE = {
    "vehicle":    dict(color="#2E5A9C", symbol="square",        size=15),
    "pedestrian": dict(color="#B9701A", symbol="circle",        size=10),
    "clutter":    dict(color="#6A1B9A", symbol="diamond",       size=11),
    "scatterer":  dict(color="#57657A", symbol="circle-open",   size=10),
}


def scenario_topdown_figure(scenario) -> "go.Figure":
    """Plan view (x, y) of a `e2e.scenario.Scenario`: nodes, boresight, objects.

    Answers "what is actually in the scene" before any signal-domain plot is read. A
    stripe in sin(azimuth) is only interpretable next to the geometry that produced it.

    Robust to partial scenarios by design: every field it reads is optional in
    `Scenario`, and a scenario with no objects still renders the radar and its boresight
    rather than raising. This runs in the UI, where a figure that throws costs the user
    the whole Results tab.
    """
    fig = go.Figure()

    objs = list(getattr(scenario, "objects", None) or [])
    by_class = {}
    for o in objs:
        by_class.setdefault(getattr(o, "object_class", "scatterer") or "scatterer",
                            []).append(o)

    for cls, items in sorted(by_class.items()):
        style = _TOPDOWN_STYLE.get(cls, _TOPDOWN_STYLE["scatterer"])
        xs = [float(o.position[0]) for o in items]
        ys = [float(o.position[1]) for o in items]
        names = [getattr(o, "name", "") for o in items]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", name=f"{cls} ({len(items)})",
            text=names, hovertemplate="%{text}<br>x=%{x:.1f} m  y=%{y:.1f} m<extra></extra>",
            marker=dict(color=style["color"], symbol=style["symbol"], size=style["size"],
                        line=dict(width=1, color="#2B2B2B")),
        ))

    for node in (getattr(scenario, "nodes", None) or []):
        nx, ny = float(node.position[0]), float(node.position[1])
        fig.add_trace(go.Scatter(
            x=[nx], y=[ny], mode="markers+text", name=getattr(node, "name", "node"),
            text=[getattr(node, "name", "node")], textposition="bottom center",
            marker=dict(color="#2E7D4F", symbol="triangle-up", size=18,
                        line=dict(width=1.5, color="#2B2B2B")),
        ))
        # Boresight, so azimuth on every other plot has a physical reference here.
        look = getattr(node, "look_at", None)
        if look is not None:
            fig.add_trace(go.Scatter(
                x=[nx, float(look[0])], y=[ny, float(look[1])], mode="lines",
                name=f"{getattr(node, 'name', 'node')} boresight",
                line=dict(color="#2E7D4F", width=2, dash="dash"),
                hoverinfo="skip", showlegend=False,
            ))

    fig.update_layout(
        title="Scenario, plan view (x–y). Radar ▲, boresight dashed.",
        xaxis_title="x (m)", yaxis_title="y (m)",
        margin=dict(l=40, r=20, t=40, b=40), height=420,
    )
    # Equal aspect: a plan view with distorted axes misleads about angle, which is the
    # one thing this figure exists to make readable.
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def _add_frame_animation(fig, per_frame, *, key="z", trace_idx=0, trace_type="heatmap"):
    """Attach a frame slider + play control to `fig`, leaving its initial view alone.

    `per_frame` is the already-converted data for each frame, in frame order, matching
    whatever `key` the target trace uses ("z" for a heatmap, "y" for a line). The
    figure's existing trace 0 keeps the LAST frame's data, and the slider starts parked
    on that same index, so the default rendering is byte-for-byte what it was before
    animation existed.

    Returns `fig` unchanged when there are fewer than two frames -- a slider over one
    frame is noise.
    """
    n = len(per_frame)
    if n < 2:
        return fig

    # `type` is REQUIRED: without it Plotly infers Scatter for the frame's trace and
    # rejects "z" as an invalid property.
    fig.frames = [go.Frame(name=str(i), data=[{"type": trace_type, key: d}],
                           traces=[trace_idx])
                  for i, d in enumerate(per_frame)]
    steps = [dict(method="animate", label=str(i + 1),
                  args=[[str(i)], dict(mode="immediate",
                                       frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))])
             for i in range(n)]
    fig.update_layout(
        sliders=[dict(active=n - 1, x=0.08, len=0.9, y=-0.02,
                      currentvalue=dict(prefix="frame ", font=dict(size=12)),
                      pad=dict(t=30, b=4), steps=steps)],
        updatemenus=[dict(type="buttons", showactive=False, direction="left",
                          x=0.0, y=-0.02, xanchor="left", yanchor="top",
                          pad=dict(t=30, r=6),
                          buttons=[dict(label="▶", method="animate",
                                        args=[None, dict(mode="immediate",
                                                         fromcurrent=True,
                                                         frame=dict(duration=350,
                                                                    redraw=True),
                                                         transition=dict(duration=0))]),
                                   dict(label="❚❚", method="animate",
                                        args=[[None], dict(mode="immediate",
                                                           frame=dict(duration=0,
                                                                      redraw=True))])])],
        margin=dict(l=40, r=20, t=40, b=70),
    )
    return fig


def figures_from_outputs(outputs: Dict[str, Any]) -> Dict[str, go.Figure]:
    """Build a dict of named Plotly figures from a simulation outputs dict."""
    figs: Dict[str, go.Figure] = {}

    # Axis metadata (bins per product + the raw frame's freq-sample count/span),
    # stashed by run_pipeline; absent (e.g. a hand-built outputs dict in a test)
    # means every axis falls back to raw bin indices.
    meta = outputs.get("_axis_meta") or {}
    n_freqs = meta.get("n_freqs")
    freq_span_hz = meta.get("freq_span_hz")

    if outputs.get("fft"):
        bins = meta.get("fft_bins") or outputs["fft"][-1].shape[0]
        u = _sin_angle_axis(bins)
        # Coherent 2D aperture FFT, non-coherent (power) integration over range --
        # a target shows up regardless of its range, not just one at range 0.
        figs["fft"] = _add_frame_animation(
            _heatmap(
                _to_numpy_abs_db(outputs["fft"][-1]),
                "Azimuth-Elevation power (non-coherent over range)",
                x=u, y=u, xlabel="azimuth sin(θ)", ylabel="elevation sin(θ)",
            ),
            [_to_numpy_abs_db(f) for f in outputs["fft"]])

    for key, title, aperture_label in [
        ("range_az", "Range-Azimuth power (non-coherent over elevation)", "azimuth sin(θ)"),
        ("range_el", "Range-Elevation power (non-coherent over azimuth)", "elevation sin(θ)"),
    ]:
        if outputs.get(key):
            bins = meta.get(f"{key}_bins") or outputs[key][-1].shape[0]
            x = _sin_angle_axis(bins)
            if n_freqs and freq_span_hz:
                # Full-band range compression + power-binning to `bins` gates means
                # the physical range axis is well-defined for any bins (see
                # _range_axis); only needs the frame's band + freq-sample count.
                y = _range_axis(bins, freq_span_hz, n_freqs)
                ylabel = "range (m)"
            else:
                # Metadata unavailable (e.g. a hand-built outputs dict): fall back
                # to raw display-gate indices.
                y = np.arange(bins)
                ylabel = "range (bins)"
            figs[key] = _add_frame_animation(
                _heatmap(
                    _to_numpy_abs_db(outputs[key][-1]), title,
                    x=x, y=y, xlabel=aperture_label, ylabel=ylabel,
                ),
                [_to_numpy_abs_db(f) for f in outputs[key]])

    if outputs.get("range_profile_agg"):
        prof = outputs["range_profile_agg"][-1]
        if hasattr(prof, "detach"):
            prof = prof.detach().cpu().numpy()
        prof = np.asarray(prof, dtype=float)
        bins_rp = meta.get("range_profile_bins") or prof.shape[0]
        if n_freqs and freq_span_hz:
            x = _range_axis(bins_rp, freq_span_hz, n_freqs)
            xlabel = "range (m)"
        else:
            x = np.arange(bins_rp)
            xlabel = "range (bins)"
        peak = max(float(prof.max()), 1e-12)
        prof_db = 10 * np.log10(prof / peak + 1e-12)
        fig = go.Figure(data=go.Scatter(x=np.asarray(x), y=prof_db, mode="lines"))
        fig.update_layout(
            title="Range profile (non-coherent over channels)",
            xaxis_title=xlabel,
            yaxis_title="power (dB rel. peak)",
            margin=dict(l=40, r=20, t=40, b=40),
            height=360,
        )
        figs["range_profile"] = fig

    rx = meta.get("rx") or {}
    if outputs.get("radar_cube"):
        # [n_channels, range, doppler] complex -> non-coherent power over channels, dB
        # relative to the frame's peak (same display convention as the range products).
        def _rd_db(cube):
            if hasattr(cube, "detach"):
                cube = cube.detach().cpu().numpy()
            p = np.mean(np.abs(np.asarray(cube)) ** 2, axis=0)
            return 10 * np.log10(p / max(float(p.max()), 1e-30) + 1e-12)
        first = _rd_db(outputs["radar_cube"][-1])
        n_r, n_d = first.shape
        if rx.get("range_resolution_m") and rx.get("velocity_resolution_mps"):
            y = np.arange(n_r) * rx["range_resolution_m"]
            # adc_to_rd fftshifts the Doppler axis: zero Doppler sits at bin n_d // 2.
            x = (np.arange(n_d) - n_d // 2) * rx["velocity_resolution_mps"]
            xlabel, ylabel = "radial velocity (m/s)", "range (m)"
        else:
            x, y, xlabel, ylabel = np.arange(n_d), np.arange(n_r), "Doppler (bin)", "range (bin)"
        figs["radar_cube"] = _add_frame_animation(
            _heatmap(first, "Range-Doppler power (non-coherent over channels)",
                     x=x, y=y, xlabel=xlabel, ylabel=ylabel),
            [_rd_db(c) for c in outputs["radar_cube"]])

    for key, title in (("cfar_detection", "CFAR objectness"),
                       ("ml_detection", "Neural detector objectness")):
        if not outputs.get(key):
            continue
        det = outputs[key][-1]
        if hasattr(det, "detach"):
            det = det.detach().cpu().numpy()
        obj = np.asarray(det)[0]                      # [n_range, n_azimuth], in [0, 1]
        n_r, n_a = obj.shape
        g = rx.get("grid") or {}
        max_r = float(g.get("max_range_m") or n_r)
        y = (np.arange(n_r) + 0.5) * max_r / n_r     # range-bin centres, m
        x = -1.0 + (np.arange(n_a) + 0.5) * 2.0 / n_a  # sin(azimuth) bin centres
        fig = go.Figure(data=go.Heatmap(
            z=obj, x=x, y=y, zmin=0.0, zmax=1.0, colorscale="Viridis",
            colorbar=dict(title="objectness"), name="objectness",
        ))
        # Decoded detections (filled) and, for a replayed corpus frame, the stored
        # ground truth (hollow) -- drawn at the surface range the metric matches on.
        dets = (outputs.get(key + "s") or [[]])[-1]
        if dets:
            fig.add_trace(go.Scatter(
                x=[d[1] for d in dets], y=[d[3] for d in dets], mode="markers",
                name=f"detections (n={len(dets)})",
                marker=dict(symbol="x", size=10, color="#ff3b3b", line=dict(width=2)),
                text=[f"score {d[2]:.2f}" for d in dets],
            ))
        gt = (outputs.get("gt_detections") or [[]])[-1]
        if gt:
            fig.add_trace(go.Scatter(
                x=[d[1] for d in gt], y=[d[3] for d in gt], mode="markers",
                name=f"ground truth (n={len(gt)})",
                marker=dict(symbol="circle-open", size=14, color="#ffffff",
                            line=dict(width=2)),
            ))
        fig.update_layout(
            title=title, xaxis_title="azimuth sin(θ)", yaxis_title="range (m)",
            margin=dict(l=40, r=20, t=40, b=40), height=420,
            legend=dict(orientation="h", y=-0.2),
        )
        figs[key] = fig

    if outputs.get("subspace_err"):
        errs = [float(e) for e in outputs["subspace_err"]]
        fig = go.Figure(data=go.Scatter(y=errs, mode="lines+markers"))
        fig.update_layout(
            title="Subspace error (Frobenius) per frame",
            xaxis_title="Frame",
            yaxis_title="Error",
            margin=dict(l=40, r=20, t=40, b=40),
            height=360,
        )
        figs["subspace_err"] = fig

    # Comms head (opt-in "product" -- see webapp/pipeline_registry.py "comms"):
    # BER/EVM-per-frame lines + a constellation snapshot of the last frame.
    if outputs.get("ber"):
        bers = [float(b) for b in outputs["ber"]]
        comms_meta = outputs.get("_comms_meta") or {}
        combining = comms_meta.get("combining", "?")
        title = f"Comms head BER ({combining}"
        gains = [float(g) for g in (outputs.get("comm_array_gain_db") or [])
                 if g is not None and np.isfinite(float(g))]
        if gains:
            title += f", array gain {np.mean(gains):.1f} dB"
        title += ")"
        # BER=0 (no bit errors) is common on good frames but unplottable on a log
        # axis -- Plotly drops the points and the whole figure renders empty. Clamp
        # to a display floor and say so, rather than showing a blank plot.
        ber_floor = 1e-6
        plotted = [max(b, ber_floor) for b in bers]
        fig = go.Figure(data=go.Scatter(y=plotted, mode="lines+markers"))
        fig.update_layout(
            title=title,
            xaxis_title="Frame",
            yaxis_title="BER",
            yaxis_type="log",
            margin=dict(l=40, r=20, t=40, b=40),
            height=360,
        )
        if any(b < ber_floor for b in bers):
            fig.add_annotation(
                text=f"frames with 0 bit errors shown at the {ber_floor:g} floor",
                xref="paper", yref="paper", x=0.5, y=1.02,
                showarrow=False, font=dict(size=11, color="#576574"),
            )
        figs["ber"] = fig

    if outputs.get("evm"):
        evms = [float(e) for e in outputs["evm"]]
        fig = go.Figure(data=go.Scatter(y=evms, mode="lines+markers"))
        fig.update_layout(
            title="Comms head EVM per frame",
            xaxis_title="Frame",
            yaxis_title="EVM",
            margin=dict(l=40, r=20, t=40, b=40),
            height=360,
        )
        figs["evm"] = fig

    if outputs.get("comm_data_eq"):
        data_np = _to_numpy_complex(outputs["comm_data_eq"][-1])
        # Colour each received symbol by the ideal point it was actually TRANSMITTED as,
        # using the same rule as the matplotlib figure (e2e.comms.constellation_viz) so
        # the two views cannot disagree. `comm_tx_data` and `comm_const` are emitted by
        # ModemBlock and forwarded verbatim by Simulation, so no plumbing is needed --
        # but both are guarded, because a pipeline without a ModemBlock has neither and
        # a flat scatter is the honest fallback there.
        marker = dict(size=4)
        const = outputs.get("comm_const")
        tx_data = outputs.get("comm_tx_data")
        if const is not None and tx_data is not None:
            try:
                from e2e.comms.constellation_viz import (checkerboard_css_colors,
                                                          symbol_color_indices)

                const_last = const[-1] if isinstance(const, list) else const
                tx_last = tx_data[-1] if isinstance(tx_data, list) else tx_data
                idx = symbol_color_indices(data_np, const_last, tx_last)
                palette = checkerboard_css_colors(const_last)
                marker = dict(size=5, color=[palette[i] for i in idx])
            except Exception:
                # A colouring failure must not cost the user the plot itself.
                marker = dict(size=4)

        fig = go.Figure(data=go.Scatter(
            x=data_np.real, y=data_np.imag, mode="markers", marker=marker,
        ))
        fig.update_layout(
            title="Comms head constellation (last frame, equalized) — "
                  "coloured by transmitted symbol",
            xaxis_title="I",
            yaxis_title="Q",
            margin=dict(l=40, r=20, t=40, b=40),
            height=360,
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        figs["comm_const"] = fig

    return figs


def placeholder_figure(message: str) -> go.Figure:
    """An empty figure carrying a centered message (used for errors / no-data)."""
    fig = go.Figure()
    fig.add_annotation(
        text=message, showarrow=False, xref="paper", yref="paper",
        x=0.5, y=0.5, font=dict(size=14),
    )
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=20, b=20), height=360,
    )
    return fig
