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

import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go

from webapp.pipeline_registry import (BLOCKS_BY_ID, MAX_N_STEPS, MAX_PRESET_N_STEPS,
                                      SUBSPACE_M)
from webapp import detector_scoreboard

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


def _resolve_interconnect_band_hz(state: Dict[str, Dict[str, Any]], env_block: Any):
    """``(start_hz, stop_hz)`` the frame's frequency axis spans -- what
    ``InterconnectBlock(source='tessera')`` needs as ``band_hz`` to auto-derive its
    scale factor and to place its response on the real axis (see
    `e2e.blocks._resolve_tessera_scale`). Mirrors `_resolve_freq_span_hz`: prefers the
    environment block's own ``freq_plan`` (v2 pkls); legacy pkls (no ``freq_plan``,
    e.g. munich.pkl) fall back to a span centered at 30 GHz, matching `_comms_freqs`'s
    same fallback carrier.
    """
    freq_plan = getattr(env_block, "freq_plan", None)
    if freq_plan:
        return float(freq_plan["start_hz"]), float(freq_plan["stop_hz"])
    span = _resolve_freq_span_hz(state, env_block)
    carrier = 30e9
    return carrier - span / 2.0, carrier + span / 2.0


#: Last-resort fallback for `_corpus_live_interconnect_band_hz`, when `corpus_cfg` is
#: missing or its own resolution raises: the literal every corpus generated at
#: f0=77 GHz used before 2026-09-23 (see `e2e.ml.chain_generate.
#: _LEGACY_77GHZ_INTERCONNECT_BAND_HZ`, which this equals bit-for-bit).
_LEGACY_77GHZ_INTERCONNECT_BAND_HZ = (75e9, 81e9)


def _corpus_live_interconnect_band_hz(corpus_cfg: Any):
    """``(start_hz, stop_hz)`` the live-chain replay path resamples the corpus's own
    interconnect `transfer_csv` onto.

    Reads it straight off the corpus's own `RadarConfig` (its recorded carrier
    `f0_hz`) via `e2e.ml.chain_generate._interconnect_band_hz` -- the SAME function
    the generator used when it built these frames -- rather than a re-typed literal
    that could silently drift from it. A prior version hardcoded `(75e9, 81e9)`
    unconditionally; that function still returns exactly that for every existing
    f0=77 GHz corpus, so this is a no-op for every corpus generated before the
    2026-09-23 Ka-band re-founding, and follows a re-traced corpus's carrier
    automatically once one exists. Falls back to that same literal only if
    `corpus_cfg` is unavailable or the import/call itself fails -- never lets a
    provenance lookup take the run down.
    """
    if corpus_cfg is None:
        return _LEGACY_77GHZ_INTERCONNECT_BAND_HZ
    try:
        from e2e.ml.chain_generate import _interconnect_band_hz

        return _interconnect_band_hz(corpus_cfg)
    except Exception:
        return _LEGACY_77GHZ_INTERCONNECT_BAND_HZ


def prewarm_tessera_interconnect(state: Dict[str, Dict[str, Any]]) -> None:
    """Evaluate the interconnect's Tessera surrogate response once for `state`, through
    its on-disk cache, so a later `run_pipeline` call on the SAME state does not pay
    the cold model-forward cost (~10.9 s, notes/TESSERA_KNOB_MEASUREMENT_2026-09-23.md)
    on its first frame. Meant to be called from a preset-load UI hook, before Run is
    pressed -- best-effort and exception-swallowing throughout: `run_pipeline` is the
    only authority on whether a state can actually run here, and a warm-up must never
    block or fail a preset load.
    """
    if not (_enabled(state, "interconnect")
            and _p(state, "interconnect", "source") == "tessera"):
        return
    if _p(state, "interconnect", "case") in ("passthrough", "case3"):
        return
    try:
        from e2e.blocks import SionnaEnvironmentBlock, TESSERA_DESIGN_PARAMS, tessera_s21_for_axis

        env_block = SionnaEnvironmentBlock(_p(state, "environment", "scenario_name"))
        band_hz = _resolve_interconnect_band_hz(state, env_block)
        n_freqs = int(env_block.get_S_pars().shape[-1])
        freqs = np.linspace(band_hz[0], band_hz[1], n_freqs)
        params = {name: float(_p(state, "interconnect", f"tessera_{name}"))
                 for name in TESSERA_DESIGN_PARAMS}
        tessera_s21_for_axis(freqs, params=params,
                             arrangement=_p(state, "interconnect", "tessera_arrangement"))
    except Exception:
        pass


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
    from e2e.frames import DOMAIN_CFR

    # Which stored payload the frames re-enter the chain as -- see the `domain`
    # ParamSpec. "auto" prefers the ray-traced channel and falls back to the ADC cube
    # for a corpus generated without `--store-cfr` (every corpus older than
    # 2026-09-23), so those screens are unchanged; "cfr" demands it and says what is
    # missing when it is not there.
    domain_choice = str(_p(state, "corpus_environment", "domain") or "auto")
    kwargs = dict(
        split=str(_p(state, "corpus_environment", "split")),
        start=int(_p(state, "corpus_environment", "start_frame") or 0),
    )
    try:
        if domain_choice in ("auto", "cfr"):
            try:
                src = CorpusSourceBlock(manifest, domain=DOMAIN_CFR, **kwargs)
            except FileNotFoundError as e:
                if domain_choice == "cfr":
                    raise PipelineError(
                        "Corpus Replay is set to replay the stored ray-traced channel, "
                        f"but this corpus does not carry one: {e} Regenerate it with "
                        "`python -m e2e.ml.chain_generate ... --store-cfr`, or set "
                        "'Replay from' to 'adc' to replay the stored ADC cube instead."
                    )
                src = CorpusSourceBlock(manifest, **kwargs)
        else:
            src = CorpusSourceBlock(manifest, **kwargs)
    except PipelineError:
        raise
    except (KeyError, IndexError, FileNotFoundError, ValueError) as e:
        raise PipelineError(f"Could not open the corpus: {e}")
    return src, src.cfg, src.grid


# --------------------------------------------------------------------------------
# The live-chain replay path: a corpus frame re-entering the chain as the stored
# RAY-TRACED CHANNEL rather than as the ADC cube it produced (owner directive
# 2026-09-23, "store the ray tracing, compute everything else live with the knobs";
# notes/RT_LIVE_PLAN_2026-09-23.md D3). Both classes are declared here but import
# torch/e2e only inside __init__, so this module still imports without torch.
# --------------------------------------------------------------------------------
class _StoredFrameSettingsStage:
    """Pass-through serial stage: give the noise stages THIS frame's stored identity.

    A corpus frame was generated by exactly the chain the live path now re-runs, but
    three of its stages are stochastic, and their draw is a property OF THE FRAME, not
    of the UI:

      * the RFFE, the thermal floor and the impairments each seed a per-instance
        generator from `seed + frame_index`, and a corpus generates one scene per
        `Simulation`, so every frame carries its OWN base seed (`impairment_params
        ['base_seed']`, `link_budget['seed']`) at frame index 0 -- one UI seed cannot
        reproduce five scenes;
      * the impairment SEVERITIES were domain-randomized per frame at generation and
        are recorded resolved (`impairment_params`), and `ImpairmentBlock` exposes no
        scalar "level" the UI could offer instead -- they are per-stage dataclasses.

    So on this path the stored values win, and the two seed ParamSpecs say so. What
    stays live is every SETTING the operator can actually turn: the front end on/off
    and its circuit knobs, the interconnect on/off (its `case` selects the corpus's own
    response or a pass-through -- see that ParamSpec), the IF high-pass corner and
    order, the ADC's bits and full scale, and the detector. That split is what makes
    the correctness gate meaningful:
    at the frame's own settings the live chain reproduces the stored ADC bit for bit
    (`_StoredADCGateBlock`), so any difference on screen is the knob that was moved.

    Runs FIRST, in the CFR domain, before the front end it re-seeds.
    """

    frame_contract_name = "stored-frame chain settings"

    def __init__(self, noise_blocks, impairment_block, if_hpf_block=None,
                chain_flags=None):
        from e2e import frames as _frames

        self.frame_capabilities = _frames.FrameCapabilities(
            accepts_mimo=True, chirps=_frames.CHIRP_NATIVE, domain=_frames.DOMAIN_CFR,
        )
        self.noise_blocks = [b for b in noise_blocks if b is not None]
        self.impairment_block = impairment_block
        #: The live IF high-pass, if one is in the chain. NOT re-seeded or overridden
        #: (its corner is a demo knob -- Thrust 5's ported-network A/B turns exactly
        #: it), but COMPARED against what each frame records it was filtered with, so
        #: a difference between the live cube and the stored one can be attributed
        #: instead of blamed on the operator. See `if_hpf_mismatch`.
        self.if_hpf_block = if_hpf_block
        #: `(stored_corner_hz, stored_order, live_corner_hz, live_order)` for the first
        #: frame whose recorded IF high-pass is not the one this run is applying, else
        #: None. A corpus generated with `--if-hpf-corner-range` (or with the stage
        #: off) replays through a chain that is NOT the one that wrote it, and before
        #: this the only symptom was a non-zero gate reading that the run note then
        #: attributed to a knob the operator had not touched (reviewed 2026-09-23).
        self.if_hpf_mismatch = None
        #: THIS run's chain-topology settings ({"use_rffe", "use_interconnect",
        #: "use_link_budget", "quant_bits"}), for comparison against whatever a frame
        #: recorded of its own (see `e2e.ml.chain_generate._ChainFlagsStage`).
        self.chain_flags = dict(chain_flags) if chain_flags else {}
        #: A list of `(label, stored, live)` tuples, one per differing setting a frame
        #: actually records -- None until the first frame whose recorded topology is
        #: not this run's. A frame with none of these keys (every corpus generated
        #: before 2026-09-23) leaves this None forever, which is what keeps the gate's
        #: generic wording for that case exactly unchanged.
        self.chain_flags_mismatch = None
        #: Frames whose meta carried no seed at all (an artifact written by something
        #: other than the corpus generator) -- surfaced as a run note rather than
        #: silently running on the UI seed.
        self.frames_without_seed = 0

    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        params = state.get("impairment_params") or {}
        link_budget = state.get("link_budget") or {}
        seed = params.get("base_seed", params.get("seed"))
        if seed is None:
            seed = link_budget.get("seed")
        frame_idx = int(params.get("frame_idx", 0) or 0)
        if seed is None:
            self.frames_without_seed += 1
        else:
            for block in self.noise_blocks:
                block.seed = int(seed)
                block._frame_idx = frame_idx
        self._check_if_hpf(state)
        self._check_chain_flags(state)
        if self.impairment_block is not None:
            severities = {k: params[k] for k in ("phase_noise", "leakage", "clutter")
                          if k in params}
            # `{}` would mean "all stages at their defaults", which is NOT what a frame
            # with no recorded severities says; leave the block's own setting alone.
            if severities:
                self.impairment_block.chain_params = severities
        return {}

    def _check_chain_flags(self, state: Dict[str, Any]) -> None:
        """Record the first frame whose recorded chain topology is not this run's.

        Only settings the frame ACTUALLY recorded are compared -- a frame with none
        of these keys (written before `_ChainFlagsStage` existed) leaves
        `chain_flags_mismatch` at None, same as if nothing were checked at all.
        """
        if self.chain_flags_mismatch is not None:
            return
        diffs = []
        for key, label in (("use_rffe", "RF front end"),
                           ("use_interconnect", "interconnect"),
                           ("use_link_budget", "link budget")):
            stored = state.get(key)
            if stored is None:
                continue
            live = bool(self.chain_flags.get(key))
            if bool(stored) != live:
                diffs.append((label, bool(stored), live))
        stored_bits = state.get("quant_bits")
        if stored_bits is not None:
            live_bits = self.chain_flags.get("quant_bits")
            if live_bits is None or int(stored_bits) != int(live_bits):
                diffs.append(("ADC bit depth", int(stored_bits), live_bits))
        if diffs:
            self.chain_flags_mismatch = diffs

    def _check_if_hpf(self, state: Dict[str, Any]) -> None:
        """Record the first frame whose stored IF high-pass is not this run's.

        `e2e.ml.blocks._EXTRA_META_KEYS` carries `if_hpf_corner_hz`/`if_hpf_order` on
        every frame the generator wrote, so unlike the rest of the chain's topology
        this one IS checkable -- and it is the setting one of the Thrust 5 A/B arms
        turns, so it is also the one most likely to differ on purpose. Compared, never
        overridden: the knob stays live and the difference gets a name.
        """
        if self.if_hpf_mismatch is not None:
            return
        stored_corner = state.get("if_hpf_corner_hz")
        stored_order = state.get("if_hpf_order")
        if stored_corner is None:
            return
        if self.if_hpf_block is None:
            self.if_hpf_mismatch = (float(stored_corner),
                                    None if stored_order is None else int(stored_order),
                                    None, None)
            return
        live_corner = float(getattr(self.if_hpf_block, "corner_hz", float("nan")))
        live_order = int(getattr(self.if_hpf_block, "order", 0))
        same_corner = abs(live_corner - float(stored_corner)) <= 1e-6 * max(
            1.0, abs(float(stored_corner)))
        same_order = stored_order is None or int(stored_order) == live_order
        if not (same_corner and same_order):
            self.if_hpf_mismatch = (float(stored_corner),
                                    None if stored_order is None else int(stored_order),
                                    live_corner, live_order)


def _describe_chain_flag_diff(label: str, stored, live) -> str:
    """One `(label, stored, live)` tuple from `_StoredFrameSettingsStage.
    chain_flags_mismatch`, in the same "this run: ... vs frames: ..." phrasing the
    knobs summary already uses (e.g. "ADC bit depth (this run: ADC 4-bit vs frames:
    12-bit)")."""
    if label == "ADC bit depth":
        live_text = "no quantizer" if live is None else f"ADC {live}-bit"
        return f"{label} (this run: {live_text} vs frames: {stored}-bit)"
    live_text = "on" if live else "off"
    stored_text = "on" if stored else "off"
    return f"{label} (this run: {live_text} vs frames: {stored_text})"


class _StoredADCGateBlock:
    """The correctness gate, ON EVERY LIVE RUN: live cube vs the frame's stored cube.

    A live chain that silently diverged from the corpus it replays would still draw a
    plausible picture, so the divergence is measured rather than asserted once in a
    test: this product re-encodes both cubes in THIS RUN'S OWN quantizer LSBs (its
    live full scale / 2**bits -- not the storage codec's scale, which is a different
    number: a reviewer correctly read "max |diff| 18456 codes" next to a 12-bit
    (4096-code) ADC as a unit mismatch, 2026-09-23) and records the largest LSB
    difference over the run, clipped to the converter's own representable code range
    so the figure can never exceed what a converter of that bit depth can express.

    WHAT ZERO MEANS, EXACTLY (scoped after a review found the unscoped version false,
    2026-09-23): zero says this run's chain reproduced these frames. It is reached when
    the run's settings are the ones the frames were generated with AND the corpus was
    generated with the generator's default chain topology -- because a frame records
    its seeds, its impairment severities and its IF high-pass corner, but NOT whether
    the RF front end, the interconnect and the link budget ran, nor the ADC's bit
    depth. A corpus made with `chain_generate --no-rffe` (say) therefore replays
    through a chain that is not the one that wrote it, and the number here is non-zero
    with no knob having been touched. So a non-zero reading means "the live cube is not
    the stored one" and nothing more precise; the run notes carry what can actually be
    attributed (the IF high-pass IS compared frame by frame -- see
    `_StoredFrameSettingsStage._check_if_hpf`) and say plainly what a frame does not
    record.

    Reads state, writes nothing: `apply` returns `{}` (no reserved key, no product).
    """

    frame_contract_name = "live-vs-stored ADC gate"

    def __init__(self, files, knobs: str = "", quantizer_block=None):
        from e2e import frames as _frames

        self.frame_capabilities = _frames.FrameCapabilities(
            accepts_mimo=True, chirps=_frames.CHIRP_NATIVE, domain=_frames.DOMAIN_RX_TIME,
        )
        self.files = list(files)
        #: A human summary of the knobs this run used, quoted when the cube differs --
        #: a difference is the POINT of an A/B arm, so the note names what was turned
        #: rather than reading as a failure.
        self.knobs = knobs
        #: THIS run's live quantizer (None if the run has none) -- the gate reports the
        #: difference in ITS LSB units (`.lsb`, resolved per frame from `.bits` and the
        #: live full scale), not the storage codec's own scale.
        self.quantizer_block = quantizer_block
        self.bits = int(quantizer_block.bits) if quantizer_block is not None else None
        #: Total representable codes of this run's converter (2**bits) -- the "of N"
        #: half of the "N of TOTAL LSB" phrasing; None with no live quantizer.
        self.lsb_total = (2 ** self.bits) if self.bits is not None else None
        self.frame_counter = 0
        self.n_compared = 0
        self.max_lsb_diff = 0
        self.max_abs_diff = 0.0
        self.problem = None

    def reset(self):
        self.frame_counter = 0

    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        import json

        import numpy as _np

        from e2e.ml import storage

        if not self.files:
            return {}
        # Modulo: the source wraps to frame 0 when a run asks for more frames than the
        # split holds (SourceBlock.step), and the gate must wrap with it.
        idx = self.frame_counter % len(self.files)
        self.frame_counter += 1
        try:
            with _np.load(self.files[idx], allow_pickle=False) as data:
                meta = json.loads(str(data["meta"].item()))
                stored = storage.read_payload(data, meta, meta["payload_key"])
                stored = _np.array(stored)
            live = state["adc"]
            live = live.detach().cpu().numpy() if hasattr(live, "detach") else _np.asarray(live)
            if live.shape != stored.shape:
                self.problem = f"shape {live.shape} vs stored {stored.shape}"
                return {}
            self.max_abs_diff = max(self.max_abs_diff, float(_np.abs(live - stored).max()))
            if self.quantizer_block is not None:
                # `.lsb` is resolved from THIS FRAME's own full scale (the quantizer
                # stage already ran earlier in this same apply cycle -- see
                # QuantizerBlock.apply), so an AGC full scale that moves frame to frame
                # is tracked rather than pinned to frame 0's.
                lsb = float(self.quantizer_block.lsb)
                half_range = 2 ** (self.bits - 1)
                if lsb > 0.0:
                    def _codes(arr):
                        # Clipped to [-half_range, half_range - 1]: the converter's OWN
                        # representable code range (matches QuantizerBlock.apply's own
                        # clamp), so the reported figure can never exceed what a
                        # converter of this bit depth can express -- unlike the old
                        # int16-clipped storage-codec figure, which could (and did).
                        return _np.clip(_np.round(arr / lsb), -half_range,
                                        half_range - 1).astype(_np.int64)
                    diff = max(int(_np.abs(_codes(live.real) - _codes(stored.real)).max()),
                               int(_np.abs(_codes(live.imag) - _codes(stored.imag)).max()))
                    self.max_lsb_diff = max(self.max_lsb_diff, diff)
            self.n_compared += 1
        except Exception as e:      # a gate must never take the run down with it
            self.problem = f"{type(e).__name__}: {e}"
        return {}

    def short(self) -> str:
        """The gate in banner-length form. On the Results tab, not only in the status
        line: the photograph a visitor takes is of the banner, and "is this the corpus's
        own cube or something this run invented?" is exactly the question that picture
        has to answer on its own."""
        if self.problem is not None or not self.n_compared:
            return "live vs stored ADC: NOT COMPARED"
        if self.quantizer_block is None:
            # No live quantizer this run -- LSB units don't apply; fall back to the
            # physical figure alone.
            return (f"live vs stored ADC: max |diff| {self.max_abs_diff:.3e} absolute"
                    + (" (bit-identical)" if self.max_abs_diff == 0.0 else " (differs)"))
        # "(differs)", not "(knob moved)": on a corpus generated with a chain this
        # replay cannot reconstruct, nobody moved anything (see the class docstring).
        # The banner states the fact; the run notes carry the attribution.
        return (f"live vs stored ADC: max |diff| {self.max_lsb_diff} of {self.lsb_total} "
                f"LSB ({self.bits}-bit)"
                + (" (bit-identical)" if self.max_lsb_diff == 0 else " (differs)"))

    def note(self) -> str:
        """The one line this gate contributes to the run notes."""
        if self.problem is not None:
            return ("live-vs-stored ADC gate could not compare this run "
                    f"({self.problem}) -- the live cube is UNVERIFIED against the corpus")
        if not self.n_compared:
            return "live-vs-stored ADC gate compared no frames"
        if self.quantizer_block is None:
            verdict = ("bit-identical, so the live chain IS the chain that wrote them"
                       if self.max_abs_diff == 0.0 else
                       "DIFFERS -- this run's cube is not the stored one"
                       + (f" (this run: {self.knobs})" if self.knobs else ""))
            return (f"live chain vs stored ADC over {self.n_compared} frame(s): "
                    f"max |diff| = {self.max_abs_diff:.3e} absolute -- {verdict}")
        if self.max_lsb_diff == 0:
            verdict = "bit-identical, so the live chain IS the chain that wrote them"
        else:
            # NOT "a knob was moved": the same reading appears when the corpus was
            # generated with a chain this replay cannot know about (see the class
            # docstring). State the fact, and let the attribution notes do the rest.
            verdict = ("DIFFERS -- this run's cube is not the stored one"
                       + (f" (this run: {self.knobs})" if self.knobs else ""))
        return (f"live chain vs stored ADC over {self.n_compared} frame(s): "
                f"max |diff| = {self.max_lsb_diff} of {self.lsb_total} LSB "
                f"({self.bits}-bit) ({self.max_abs_diff:.3e} absolute) -- {verdict}")


def _detector_meta(state: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Mode, operating point and a short label for the detector on screen."""
    mode = str(_p(state, "detector", "mode"))
    threshold = float(_p(state, "detector", "threshold"))
    if mode == "cfar":
        label = (f"CA-CFAR (guard {int(_p_positive(state, 'detector', 'cfar_guard'))}, "
                 f"train {int(_p_positive(state, 'detector', 'cfar_train'))})")
    else:
        ckpt_text = str(_p(state, "detector", "checkpoint") or "").strip()
        label = _resolve_repo_path(ckpt_text).parent.name if ckpt_text else "ML"
    return {"mode": mode, "threshold": threshold, "label": label}


def _build_detector(state: Dict[str, Dict[str, Any]], cfg, grid):
    """The Detector product in either mode, plus a small provenance dict.

    `cfg`/`grid` describe the ADC cube it consumes -- from the corpus manifest, or
    from the dechirp preset for a live chain. The second return value is `{}` for
    CFAR and, for a checkpoint, `{"training_manifest", "input_scale", "input_format"}`
    -- the checkpoint's own numbers, so the run can SAY that the input scaling is the
    training corpus's when the frames on screen come from a different one."""
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
            ), {}
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
        # Loaded HERE rather than inside the block (which accepts an already-loaded
        # checkpoint dict for exactly this reason) so the run can read the checkpoint's
        # own `manifest` -- the corpus whose measured input scaling it was trained
        # against -- without a second torch.load of the weights.
        import torch

        ckpt_dict = torch.load(str(ckpt), map_location="cpu")
        block = NeuralDetectorBlock(ckpt_dict, mode="infer", cfg=cfg, grid=grid,
                                    threshold=threshold)
    except Exception as e:
        raise PipelineError(f"Could not load the ML checkpoint {ckpt.name}: {e}")
    info = {"training_manifest": (str(ckpt_dict.get("manifest"))
                                  if isinstance(ckpt_dict, dict) else None),
            "input_scale": float(getattr(block, "input_scale", 1.0)),
            "input_format": str(getattr(block, "input_format", "rd"))}
    return block, info


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
        from e2e.frames import DOMAIN_CFR, FrameContractError
        from e2e.simulation import Simulation
        from e2e.blocks import (
            SionnaEnvironmentBlock,
            RFFEBlock,
            InterconnectBlock,
            TESSERA_DESIGN_PARAMS,
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
            f"demo presets use at most {MAX_PRESET_N_STEPS} (past ~{MAX_PRESET_N_STEPS} "
            f"frames the per-frame cost triples and the tracker's rank-collapse spike "
            f"returns); raise MAX_N_STEPS in webapp/pipeline_registry.py for a study."
        )
    # Advisory notes about THIS run (blocks ignored, a checkpoint with no provenance
    # stamp, ...) -- surfaced in the UI status line via outputs["_axis_meta"]["notes"].
    run_notes: List[str] = []

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
    # True when the corpus frames re-enter the chain as the stored RAY-TRACED CHANNEL,
    # so the ADC chain below runs LIVE on them instead of being bypassed (owner
    # directive 2026-09-23; see `_StoredFrameSettingsStage`). False for a legacy
    # ADC-only corpus, which keeps the bypass and its "skipped blocks" note.
    corpus_live_cfr = False
    if corpus_mode:
        environment_block, corpus_cfg, corpus_grid = _corpus_source(state)
        corpus_live_cfr = getattr(environment_block, "signal_domain", None) == DOMAIN_CFR
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
        # `scenario_name` is the dropdown's own token (see webapp.corpus_catalog): for
        # plain 'etoile' this token IS the SionnaEnvironmentBlock name, but munich's
        # Ka-band and legacy-3.5-GHz files (F93) are surfaced as two distinct labels
        # under the one name 'munich', so the label resolves to (name, link) rather
        # than being passed straight through.
        from webapp.corpus_catalog import resolve_sionna_scenario

        resolved_name, resolved_link = resolve_sionna_scenario(scenario_name)
        try:
            environment_block = SionnaEnvironmentBlock(resolved_name, link=resolved_link)
        except FileNotFoundError as e:
            raise PipelineError(
                f"No precomputed frames found for scenario '{scenario_name}'. "
                "Generate frames first (Scenario tab -> Generate frames), or pick a "
                f"scenario whose .pkl exists. Missing file: {e}"
            )
        except ValueError as e:
            raise PipelineError(str(e))
        # Surfaced in the Results banner: which carrier these frames actually are, for
        # a v2 pkl (freq_plan present) -- the legacy pkl has no metadata to report.
        if getattr(environment_block, "freq_plan", None):
            carrier_ghz = float(environment_block.freq_plan["carrier_hz"]) / 1e9
            run_notes.append(
                f"Environment '{scenario_name}': frames carry a {carrier_ghz:g} GHz carrier."
            )

    # Derive the receive-array size from the environment block's array_shape so the
    # Oja tracker dimension and Simulation's view() agree with the actual frames.
    # A non-(32,32) array would otherwise silently disagree with a hardcoded 32*32.
    array_shape = getattr(environment_block, "array_shape", (32, 32)) or (32, 32)
    N_RX = int(array_shape[0]) * int(array_shape[1])

    # --- optional serial blocks -------------------------------------------------
    circuit_block = None
    if _enabled(state, "rffe"):
        # On the live-chain path the front end sits on the corpus's own RX axis
        # (`RFFEBlock.apply_circuit` requires n == s_pars.shape[0]; a corpus frame has
        # cfg.n_rx rows, not the imaging array's 1024) and on the ABSOLUTE amplitude
        # scale the corpus was generated with -- `physical_scale=False` would divide
        # that scale away and `ThermalNoiseBlock` would (rightly) refuse the chain.
        # 'legacy' is still honoured, so the refusal stays reachable on purpose.
        rffe_n = int(corpus_cfg.n_rx) if corpus_live_cfr else N_RX * N_TX
        rffe_physical = (_p(state, "rffe", "scale_mode") != "legacy" if corpus_live_cfr
                         else _resolve_physical_scale(
                             _p(state, "rffe", "scale_mode"), environment_block))
        circuit_block = RFFEBlock(
            n=rffe_n,
            freq_span_hz=_resolve_freq_span_hz(state, environment_block),
            signal_scaling=float(_p_positive(state, "rffe", "signal_scaling")),
            physical_scale=rffe_physical,
            # Thrust 1's circuit knobs. Non-positive values fall back to the registry
            # defaults via _p_positive: both are divisors inside the circuit model.
            lna_bias_ma=float(_p_positive(state, "rffe", "lna_bias_ma")),
            if_bw_mhz=float(_p_positive(state, "rffe", "if_bw_mhz")),
        )

    interconnect_block = None
    if _enabled(state, "interconnect"):
        case = _p(state, "interconnect", "case")
        # On the live-chain path the interconnect belongs to the FRAME, like its noise
        # seeds and its dechirp geometry: the stored channel went through the corpus
        # generator's data-driven response, and putting the UI's 11-tap boxcar
        # placeholder there instead would not be "a different setting", it would be a
        # cube no corpus ever produced. So 'default' resolves to that response here
        # (the ParamSpec's help and the run note both say so); 'passthrough'/'case3'
        # still mean pass-through, because removing the interconnect IS a real
        # experiment and the live-vs-stored gate will show exactly what it costs.
        if corpus_live_cfr and case not in ("passthrough", "case3"):
            # Built the way `e2e.ml.chain_generate` builds it (that CSV) -- imported
            # from there rather than re-typed, so the live chain cannot drift from the
            # chain that wrote the frames it replays.
            try:
                from e2e.ml.chain_generate import DEFAULT_INTERCONNECT_CSV
            except ImportError as e:
                raise PipelineError(
                    "Could not import the corpus interconnect response "
                    "(e2e.ml.chain_generate). Underlying error: " + str(e))
            interconnect_block = InterconnectBlock(
                transfer_csv=str(DEFAULT_INTERCONNECT_CSV),
                band_hz=_corpus_live_interconnect_band_hz(corpus_cfg),
                normalize_gain=bool(_p(state, "interconnect", "normalize_gain")),
            )
        elif _p(state, "interconnect", "source") == "tessera" and case not in ("passthrough", "case3"):
            # Live Tessera TSV surrogate (F89/F90/F91, notes/ESTABLISHED_FACTS.md):
            # `case` still gates it above (a passthrough choice means no interconnect at
            # all, regardless of source); band_hz drives both the scale-model factor and
            # the axis the response is placed on, so it must be the frame's REAL band,
            # matching how `_resolve_freq_span_hz` feeds RFFE.
            tessera_params = {name: float(_p(state, "interconnect", f"tessera_{name}"))
                              for name in TESSERA_DESIGN_PARAMS}
            try:
                interconnect_block = InterconnectBlock(
                    case=None if case == "default" else case,
                    source="tessera",
                    tessera_params=tessera_params,
                    tessera_arrangement=_p(state, "interconnect", "tessera_arrangement"),
                    band_hz=_resolve_interconnect_band_hz(state, environment_block),
                    normalize_gain=bool(_p(state, "interconnect", "normalize_gain")),
                )
            except ValueError as e:
                # Construction validates the knobs eagerly (out-of-range geometry, an
                # unknown arrangement) -- caught here, not left to escape past every
                # other PipelineError handler below (the AdaOjaBlock k>=m comment above
                # documents the same failure mode for a different block).
                raise PipelineError(f"Interconnect (Tessera surrogate): {e}")
            # Surfaced in the Results banner (outputs["_axis_meta"]["notes"]) so the
            # scale factor and evaluated frequency are on screen, not only in the code
            # (build item 1: "the run banner shows describe()").
            run_notes.append(interconnect_block.describe())
        else:
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
    # shipped path used it, so default runs past ~frame 22 still diverged. A preset
    # may override both (Thrust 3's cold-start-vs-refine-gate A/B, 2026-09-23): these
    # two have no registry ParamSpec (no operator should be typing a refinement-pass
    # count into a text box mid-demo -- see demo_presets._INTERNAL_PARAMS), so they
    # are read directly off state rather than via `_p`.
    subspace_params = state.get("subspace", {}).get("params", {}) or {}
    subspace_gap_response = subspace_params.get("gap_response")
    if subspace_gap_response is None:
        subspace_gap_response = "refine"
    subspace_n_refine = subspace_params.get("n_refine")
    if subspace_n_refine is None:
        # n_refine's default, absent an explicit value, is DERIVED from gap_response so
        # a preset's single `ab` switch (Thrust 3's cold-start-vs-refine-gate A/B) can
        # move both together: gap_response="none" (fixed-effort arm) defaults to 5 --
        # the largest n_refine of {1, 2, 3, 5} that still took >=3 frames to reach
        # within 1.5x of its own settled level on the real chain at k=2 (re-measured
        # 2026-09-23, munich_ka.pkl, two repeats; n_refine=1 alone reached that bound
        # in 2 frames, same as the shipped gate, so it was rejected). Any other
        # gap_response (or none requested at all) keeps the ORIGINAL shipped default,
        # n_refine=10 -- byte-identical to every preset before this change existed.
        subspace_n_refine = 5 if subspace_gap_response == "none" else 10
    subspace_block = AdaOjaBlock(N_RX, k, m=SUBSPACE_M, n_refine=int(subspace_n_refine),
                                 gap_response=subspace_gap_response)

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
    if corpus_live_cfr:
        # The live path DOES cross the frequency domain (the stored channel enters
        # before the front end), but it crosses OUT of it at the dechirp, so the
        # frequency-domain products still have nothing to consume. Say which ones were
        # dropped -- same honesty as the replay note below -- without claiming the
        # stages were skipped, because here they ran.
        downstream_blocks = []
        ignored = [bid for bid, _ in classic_products if _enabled(state, bid)]
        if ignored:
            run_notes.append("The live chain crosses into RX time at the dechirp, so "
                             "these frequency-domain products produced nothing: "
                             + ", ".join(ignored))
    elif corpus_mode:
        # A replayed frame is already past every frequency-domain stage and product;
        # Simulation would refuse them at the frame contract. Run none of them -- and
        # SAY which enabled blocks were skipped, so a diagram showing them lit does not
        # read as a run that produced nothing for them (reviewed finding, 2026-09-22).
        downstream_blocks = []
        ignored = [bid for bid, _ in classic_products if _enabled(state, bid)]
        ignored += [bid for bid in ("rffe", "interconnect", "afe", "dechirp", "thermal_noise",
                                    "impairment", "if_hpf", "quantizer", "sink")
                    if _enabled(state, bid)]
        if ignored:
            run_notes.append("Corpus Replay skipped the enabled blocks it cannot apply to a "
                             "stored ADC frame: " + ", ".join(ignored))

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
    adc_gate = None
    meta_stage = None
    if corpus_mode and not corpus_live_cfr:
        # No serial stages at all: the corpus frame was generated by this very chain
        # (dechirp -> thermal floor -> impairments -> IF HPF -> quantizer) and stored
        # AFTER it. Re-running any of it here would impair an already-impaired frame.
        # This is the LEGACY replay -- a corpus with no stored channel to run from.
        serial_stages_override = []
    elif _enabled(state, "dechirp") or corpus_live_cfr:
        try:
            from e2e.chain.dechirp import DechirpBlock
            from e2e.radar_config import PRESETS
        except ImportError as e:
            raise PipelineError(
                "Could not import the ADC-cube chain backend (e2e.chain.dechirp / "
                "e2e.radar_config). Underlying error: " + str(e)
            )
        if corpus_live_cfr:
            # The chain that produced these frames is the one being re-run, so its
            # geometry comes from the CORPUS MANIFEST, never from the UI's radar
            # preset: a dechirp at a different chirp count/slope would not reproduce
            # the frame it replays, it would produce a different frame silently.
            if not _enabled(state, "dechirp"):
                raise PipelineError(
                    "Corpus Replay is replaying the stored ray-traced channel, which "
                    "needs the Dechirp bridge to reach the ADC cube the Radar Cube and "
                    "Detector consume. Enable 'Dechirp (channel -> ADC)', or set Corpus "
                    "Replay's 'Replay from' to 'adc' to replay the stored ADC cube.")
            adc_cfg = corpus_cfg
        else:
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
        thermal_block = impairment_block = if_hpf_block = quantizer_block = None

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
            thermal_block = ThermalNoiseBlock(
                adc_cfg, seed=int(_p(state, "thermal_noise", "seed")))
            serial_stages_override.append(thermal_block)

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
            impairment_block = ImpairmentBlock(
                adc_cfg, seed=int(_p(state, "impairment", "seed")))
            serial_stages_override.append(impairment_block)

        if _enabled(state, "if_hpf"):
            try:
                from e2e.chain.receive import IFHighPassBlock
            except ImportError as e:
                raise PipelineError(
                    "Could not import the IF high-pass stage (e2e.chain.receive). "
                    "Underlying error: " + str(e)
                )
            if_hpf_block = IFHighPassBlock(
                adc_cfg,
                corner_range_m=float(_p(state, "if_hpf", "corner_range_m")),
                order=int(_p(state, "if_hpf", "order")),
            )
            serial_stages_override.append(if_hpf_block)

        if _enabled(state, "quantizer"):
            try:
                from e2e.chain.receive import QuantizerBlock
            except ImportError as e:
                raise PipelineError(
                    "Could not import the ADC quantizer stage (e2e.chain.receive). "
                    "Underlying error: " + str(e)
                )
            # full_scale <= 0 means AUTOMATIC GAIN (QuantizerBlock's own `None`, the
            # setting the corpus generator uses): a physically scaled cube's returns
            # sit near 1e-7 and quantize to exactly zero against a fixed 1.0, silently.
            # See that block's "FULL SCALE DEFAULTS TO AUTOMATIC GAIN" docstring.
            quant_full_scale = float(_p(state, "quantizer", "full_scale"))
            quantizer_block = QuantizerBlock(
                bits=int(_p(state, "quantizer", "bits")),
                full_scale=(None if quant_full_scale <= 0.0 else quant_full_scale),
            )
            serial_stages_override.append(quantizer_block)

        if corpus_live_cfr:
            # FIRST in the list: hand each frame's own stored seeds/severities to the
            # stochastic stages before they run (see `_StoredFrameSettingsStage`), so
            # the live chain reproduces the corpus frame for frame rather than only
            # on the first one. The transmit tributary, if someone enabled it, sits
            # after this and is NOT part of the chain that wrote these frames -- the
            # gate below will report the resulting divergence rather than hide it.
            chain_flags = {
                "use_rffe": _enabled(state, "rffe"),
                "use_interconnect": _enabled(state, "interconnect"),
                "use_link_budget": _enabled(state, "thermal_noise"),
                "quant_bits": (int(_p(state, "quantizer", "bits"))
                              if quantizer_block is not None else None),
            }
            meta_stage = _StoredFrameSettingsStage(
                [circuit_block, thermal_block, impairment_block], impairment_block,
                if_hpf_block=if_hpf_block, chain_flags=chain_flags)
            serial_stages_override.insert(0, meta_stage)

        # None of the frequency-domain products above apply once the chain has
        # crossed into RX time; replace them with the RX-time products instead.
        downstream_blocks = []
        # The RX-time products (radar cube, detector) are built in the shared section
        # below, from this chain's cube geometry. The label grid mirrors the corpus
        # generator's convention (e2e.ml.labels.LabelGrid.for_config) so a live chain
        # and a replayed corpus frame draw on the same axes.
        rx_cfg = adc_cfg
        if corpus_live_cfr:
            # The corpus's OWN label grid (from its manifest), not a grid derived from
            # the config: the stored labels riding with each frame are on that grid,
            # and `LabelGrid.for_config`'s defaults (range stride, azimuth count) need
            # not match the ones the corpus was generated with.
            rx_grid = corpus_grid
        else:
            try:
                from e2e.ml.labels import LabelGrid
                rx_grid = LabelGrid.for_config(adc_cfg)
            except ImportError:
                rx_grid = None
        if (_enabled(state, "radar_cube") or _enabled(state, "detector")) \
                and not _enabled(state, "rt_environment") and not corpus_live_cfr:
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
            detector_block, detector_info = _build_detector(state, rx_cfg, rx_grid)
            downstream_blocks.append(detector_block)
            if detector_info.get("training_manifest"):
                # The network's inputs are divided by the measured scale of the corpus
                # it was TRAINED on (e2e.ml.dataset.resolve_input_scale -- the 2026-09-22
                # GUI defect was feeding it unscaled inputs). That is correct and stays;
                # it is also a claim worth stating whenever the frames on screen come
                # from somewhere else, because then the scaling is not this corpus's.
                train_manifest = str(detector_info["training_manifest"]).replace("\\", "/")
                source_manifest = (str(_p(state, "corpus_environment", "manifest") or "")
                                   .replace("\\", "/") if corpus_mode else "")
                if source_manifest and Path(train_manifest).as_posix() != \
                        Path(source_manifest).as_posix():
                    trained_on = (f"{Path(train_manifest).parent.parent.name}/"
                                  f"{Path(train_manifest).parent.name}")
                    if detector_info["input_format"] == "rad":
                        # Not the training corpus's constant: "rad" is dB relative to
                        # each frame's OWN median power, so `resolve_input_scale` pins
                        # it at 1.0 by construction (F83). Say that, rather than
                        # implying a measured scale was carried over.
                        run_notes.append(
                            f"detector input ('rad') is self-normalising per frame, so "
                            f"no training-corpus scale applies; the checkpoint was "
                            f"trained on {trained_on} and these frames are not from it")
                    else:
                        run_notes.append(
                            f"detector input_scale {detector_info['input_scale']:.6g} "
                            f"({detector_info['input_format']}) is resolved from the "
                            f"checkpoint's TRAINING manifest ({trained_on}), not from "
                            "the frames on screen -- the network is normalised for a "
                            "corpus it is not being run on")
            if str(_p(state, "detector", "mode")) == "ml":
                # Provenance of the checkpoint on screen (F84). The guard lives in
                # e2e.ml.beat_cfar; a checkpoint it cannot vouch for is still run --
                # the demo's own checkpoint predates the stamp -- but the run says so.
                try:
                    from e2e.ml.beat_cfar import _stale_reason
                    ckpt = _resolve_repo_path(_p(state, "detector", "checkpoint"))
                    reason = _stale_reason(str(ckpt.parent))
                except Exception:
                    reason = None
                if reason:
                    run_notes.append(f"ML checkpoint {ckpt.parent.name}: {reason} -- "
                                     "run `python -m e2e.ml.recertify` on it to re-verify")
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
            "Cube / Detector / Frame Sink) is enabled -- enable one, "
            "or disable Dechirp to run the frequency-domain products."
        )

    # The correctness gate rides along on every live-chain run (see
    # `_StoredADCGateBlock`). Appended AFTER the "no product enabled" check above so it
    # can never make an otherwise-empty run look productive.
    if corpus_live_cfr:
        knobs = ", ".join([
            f"ADC {int(_p(state, 'quantizer', 'bits'))}-bit" if _enabled(state, "quantizer")
            else "no quantizer",
            f"IF corner {float(_p(state, 'if_hpf', 'corner_range_m')):g} m"
            if _enabled(state, "if_hpf") else "no IF high-pass",
            "front end on" if _enabled(state, "rffe") else "front end OFF",
        ])
        adc_gate = _StoredADCGateBlock(getattr(environment_block, "_files", []), knobs,
                                       quantizer_block=quantizer_block)
        downstream_blocks.append(adc_gate)

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
    # The live-chain gate's verdict, on EVERY live run (not only in the test suite):
    # the max |difference| between the cube this run computed and the cube the corpus
    # stored, so a knob that has drifted off the frame's own value is visible as a
    # number rather than as a picture nobody can check. Prepended, ahead of the
    # advisory notes, because it is the one note that says whether the run is sound.
    if adc_gate is not None:
        run_notes.insert(0, adc_gate.note())
        outputs["_axis_meta"]["gate"] = adc_gate.short()
    if meta_stage is not None:
        run_notes.append(
            "live chain: each frame's stored noise seeds and domain-randomised "
            "impairment severities are used (they belong to the frame); the ADC bits, "
            "full scale, IF high-pass corner and front-end knobs are this run's")
        # ATTRIBUTION, when the cube differs. The gate can only say THAT it differs;
        # these two notes say as much about WHY as the frames allow. Without them a
        # non-zero reading read as "you moved a knob", which is false for a corpus
        # generated with a non-default chain (reviewed 2026-09-23: a corpus written
        # with `--if-hpf-corner-range 3` replayed at the UI default reported 24059
        # codes and blamed the operator).
        mismatch = meta_stage.if_hpf_mismatch
        if mismatch is not None:
            stored_hz, stored_order, live_hz, live_order = mismatch
            this_run = ("not in this run's chain" if live_hz is None
                        else f"{live_hz:.0f} Hz, order {live_order}")
            run_notes.append(
                f"IF high-pass: these frames were filtered at {stored_hz:.0f} Hz"
                + (f", order {stored_order}" if stored_order is not None else "")
                + f"; this run applies {this_run} -- that alone puts the live cube off "
                  "the stored one")
        if adc_gate is not None and adc_gate.max_lsb_diff and mismatch is None:
            if meta_stage.chain_flags_mismatch:
                # A frame generated after e2e.ml.chain_generate._ChainFlagsStage names
                # its own chain topology, so the SPECIFIC setting a mismatch is in is
                # known rather than guessed -- see that stage's docstring.
                run_notes.append(
                    "the live cube differs, and the frames' own recorded chain "
                    "topology names why: "
                    + "; ".join(_describe_chain_flag_diff(*d)
                               for d in meta_stage.chain_flags_mismatch))
            else:
                # A frame written before that provenance existed (every corpus
                # generated before 2026-09-23, including b1_demo_cfr) carries none of
                # these keys, so the cause stays a guess -- unchanged wording.
                run_notes.append(
                    "the live cube differs, and every setting these frames RECORD matches "
                    "this run -- so the difference is in what a frame does NOT record: the "
                    "ADC's bit depth (the usual answer: it is the knob the A/B turns) or "
                    "whether the RF front end, interconnect and link budget ran at "
                    "generation (a corpus made with chain_generate --no-rffe / "
                    "--no-interconnect replays through a chain that never wrote it)")
        if meta_stage.frames_without_seed:
            run_notes.append(
                f"{meta_stage.frames_without_seed} replayed frame(s) recorded no noise "
                "seed, so the UI seeds were used for them -- those frames will NOT "
                "reproduce the stored cube")
    outputs["_axis_meta"]["notes"] = list(run_notes)
    outputs["_axis_meta"]["n_steps_run"] = int(getattr(sim, "n_steps_run", n_steps))
    outputs["_axis_meta"]["cancelled"] = bool(getattr(sim, "cancelled", False))
    # Provenance for the Results banner: which source fed the run, and the
    # detector's operating point (its cross count depends on it).
    if corpus_mode:
        # WHICH PATH RAN, on the banner itself: a live chain and a replay produce the
        # same kind of picture from very different amounts of computation, and a
        # visitor (or the operator, mid-demo) must not have to infer which one it is.
        path = ("live chain from stored channel" if corpus_live_cfr else "ADC replay")
        outputs["_axis_meta"]["source"] = (
            f"Corpus Replay ({path}): {_p(state, 'corpus_environment', 'split')} split "
            f"from frame {_p(state, 'corpus_environment', 'start_frame')}")
    elif _enabled(state, "rt_environment"):
        outputs["_axis_meta"]["source"] = (
            f"RT Environment: {_p(state, 'rt_environment', 'scenario_name')}")
    else:
        outputs["_axis_meta"]["source"] = (
            f"Sionna frames: {_p(state, 'environment', 'scenario_name')}")
    if rx_cfg is not None and _enabled(state, "detector"):
        outputs["_axis_meta"]["detector"] = _detector_meta(state)
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


def _nonnegative_range(axis) -> np.ndarray:
    """Boolean mask selecting the physical (range >= 0) half of an fftshifted range
    axis. The delay profile of a causal channel has no negative-delay content; what
    sits there is sidelobe leakage (measured on munich frame 0, 2026-09-22: 4% of the
    energy). A signed axis put half of every heatmap on a non-physical range and drew
    the question "why negative range?" at every screen (owner decision 1A, 2026-09-22).
    Display only: the products themselves are untouched. A bin-index axis (no
    metadata) is kept whole."""
    axis = np.asarray(axis, dtype=float)
    if axis.size and axis.min() < 0:
        return axis >= 0
    return np.ones(axis.shape, dtype=bool)


def _cropped_nonneg_range_axis(n_bins: int, freq_span_hz: float, n_freqs: int) -> np.ndarray:
    """`_range_axis` restricted to its physical (range >= 0) half, exactly as the
    range-azimuth/range-elevation/range-profile panels below each crop it -- factored
    out so those panels can share ONE computed extent (see `range_az_yaxis_extent` in
    `figures_from_outputs`) instead of each independently computing and cropping the
    same numbers and then relying on Plotly to autorange them identically (it does
    not -- see that variable's comment)."""
    axis = _range_axis(n_bins, freq_span_hz, n_freqs)
    return axis[_nonnegative_range(axis)]


def _range_per_gate_m(n_bins: int, freq_span_hz: float, n_freqs: int) -> float:
    """Physical range (m) spanned by ONE display gate -- the same `per = ceil(n_freqs
    / n_bins)` grouping `_range_axis` uses, factored out so a card/subline can quote
    "X m per gate" without re-deriving `_range_axis`'s own math (wave 7, X6/X7: no
    panel stated this, so a screen's on-screen features had no stated calibration)."""
    per = -(-n_freqs // n_bins)               # ceil(n_freqs / n_bins); matches _power_bin
    return per * _C / (2.0 * freq_span_hz)


def _native_unambiguous_range_m(freq_span_hz: float, n_freqs: int) -> float:
    """One-sided unambiguous range (m) of the frame's OWN frequency sampling --
    independent of the display bin count, unlike `_range_per_gate_m`. This is
    `(n_freqs // 2)` native (per=1) range steps, matching the generator's own stored
    `meta['unambiguous_range_m']` exactly for the munich Ka trace (n_freqs=5000,
    freq_span_hz=3e9 -> 124.9135 m, F94) -- verified against that field rather than
    re-derived from a radar-equation reference, since `_range_axis` already fixes the
    zero-gate/fftshift convention this must agree with."""
    return (n_freqs // 2) * _C / (2.0 * freq_span_hz)


def _native_range_resolution_m(freq_span_hz: float) -> float:
    """Native (un-binned) round-trip range resolution (m): c / (2*B) -- what ONE raw
    frequency sample is worth in range, i.e. `_range_per_gate_m` at `per=1`, before
    display binning groups `per` of them into a gate (wave 8, W13: nothing on screen
    stated the ratio between a display gate and the frame's own native resolution)."""
    return _C / (2.0 * freq_span_hz)


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
    range_per_gate = _range_per_gate_m(n_bins, freq_span_hz, n_freqs)
    zero_gate = (n_freqs // 2) // per
    return -(np.arange(n_bins) - zero_gate) * range_per_gate


#: Podium-distance legibility floor (fresh-context review, 2026-09-22: every figure's
#: browser-default 12-13 px text reads fine on a laptop and fails at the ~2 m a demo
#: audience actually reads from). Applied, as the LAST step, to every figure this
#: module hands back to the UI via `_make_legible`. Raised again (pixel-measured
#: re-check, 2026-09-23): the first pass's 14-15 px tick/colorbar sizes still
#: rendered at 8-13 px of actual ink height against this same 20 px standing
#: threshold, and sat visibly smaller than the Detector scoreboard beside them
#: (webapp/detector_scoreboard.py's table, already at 18-20 px) on a Thrust 5 screen
#: -- the two halves of one screen must not visibly differ in type scale.
_LEGIBLE_FONT_SIZE = 18
_LEGIBLE_TICK_SIZE = 18
_LEGIBLE_COLORBAR_TICK_SIZE = 18
_LEGIBLE_COLORBAR_TITLE_SIZE = 18

# =====================================================================================
# PANEL GEOMETRY AND THE PANEL-META CONTRACT (layout spec, 2026-09-24)
# =====================================================================================
#
# The defect this section exists to remove (hostile round 10, section 2; layout spec
# section 1, measured on the 2026-09-24 14:27 rehearsal PNGs): prose inside the figure
# was given the same visual budget as data. `_heatmap_margin_t()` grew the top margin
# 45 px per wrapped subtitle line while the plot domain stayed pinned, so every honesty
# clause added since 2026-09-22 bought itself a strip of the screen and charged it to
# the picture -- the thrust1 range-azimuth map rendered at 12.4 % of its panel.
#
# THE RULE NOW: a figure carries NO title and NO subtitle. The only text inside a
# figure is axis titles, tick labels, colour-bar ticks, ONE statistic strip above the
# plot, and ONE legend. The panel's title and a ONE-LINE caption are HTML above the
# plot (webapp/app.py `_panel_block`); everything else -- every clause that used to be
# on a subtitle, banner, run-notes line or screen note -- goes in the per-arm
# "Details" disclosure, reachable in one click and never deleted. `panel_text()` below
# is the single accessor that returns "every word this panel carries, wherever it now
# lives", so a test pinning an honesty clause does not have to know which of the three
# places it ended up in.
#
# Figures therefore have ONE height per row kind, not a height derived from their own
# title's line count -- which is what made the same product a different size on every
# screen (acceptance check 20).

#: Row kinds. Every product declares one; `webapp/app.py` sizes the row from it, so
#: the two arms of an A/B pair are always the same height and their plot origins land
#: on the same y (acceptance check 4).
PANEL_ROW_MAP = "map"
PANEL_ROW_TABLE = "table"
PANEL_ROW_PR = "pr"

#: Panel header band (HTML title + one-line caption) and the panel's own padding,
#: px -- see the layout spec's section 2.2 table.
PANEL_HEADER_HEIGHT = 60
PANEL_PADDING = 16

#: Total panel heights, px. No other heights exist: "if a product does not fit one of
#: these, it is the product that changes."
PANEL_HEIGHT = {
    PANEL_ROW_MAP: 540,
    PANEL_ROW_TABLE: 388,
    PANEL_ROW_PR: 552,
}

#: The figure's own height inside each panel (total minus the HTML header and padding).
FIGURE_HEIGHT = {k: v - PANEL_HEADER_HEIGHT - PANEL_PADDING
                 for k, v in PANEL_HEIGHT.items()}

#: Figure margins, px. `t` is the RESERVED STATISTIC STRIP (the on-map callout used to
#: be drawn INSIDE the axes at y-domain 0.94 and covered the 105-125 m range band on
#: every map -- hostile round 10, defect 6); `b` holds the axis title and ticks at the
#: 18 px podium floor; `l` holds the rotated y-axis title; `r` is a small pad, with the
#: colour bar living in the width Plotly reserves beyond it.
_FIG_MARGIN_T = 52
_FIG_MARGIN_B = 60
_FIG_MARGIN_L = 64
_FIG_MARGIN_R = 16

#: Backwards-compatible aliases (several callers and tests still name these).
_HEATMAP_MARGIN_L = _FIG_MARGIN_L
_HEATMAP_MARGIN_R = _FIG_MARGIN_R
_HEATMAP_MARGIN_B = _FIG_MARGIN_B
_HEATMAP_MARGIN_T_BASE = _FIG_MARGIN_T
_HEATMAP_PLOT_DOMAIN_HEIGHT = 540 - PANEL_HEADER_HEIGHT - PANEL_PADDING \
    - _FIG_MARGIN_T - _FIG_MARGIN_B

#: Statistic strip typography (layout spec section 3). The headline statistic is the
#: A/B story on Thrusts 1, 2 and 4 and stays on the picture -- but above the axes, in
#: a strip reserved for it, never over the data.
_STAT_FONT_SIZE = 26
#: The second, smaller line in the same strip: the per-frame readouts that must stay
#: visible without expanding anything (the brightest visible return; which frame the
#: clock is parked on) but must not compete with the headline number.
_STAT_SUB_FONT_SIZE = 17
#: Retained name (tests pin it): the headline statistic's size.
_STAT_CALLOUT_FONT_SIZE = _STAT_FONT_SIZE

#: Arm colours (layout spec section 5). Never `#0fb9b1`/`#8854d0`, which the block
#: diagram owns.
ARM_COLORS = {"a": "#4b6584", "b": "#3867d6"}

#: One plot background for every panel on a screen, maps included (layout spec
#: section 4, "Plot background"): today's white-behind-maps / `#E5ECF6`-behind-charts
#: split made two panels in the same column read as two different products.
PLOT_BGCOLOR = "#E5ECF6"
PAPER_BGCOLOR = "#ffffff"


def _heatmap_margin_t(title: str = "") -> int:
    """Retained name, now a CONSTANT (layout spec section 3): the top margin no longer
    depends on anything a wording change can grow -- that dependency is the mechanism
    behind hostile round 10's defects 1, 3 and 5. Takes and ignores the old `title`
    argument so no caller or test has to care."""
    return _FIG_MARGIN_T


def _base_layout(row: str = PANEL_ROW_MAP) -> Dict[str, Any]:
    """The fixed geometry every figure in this module gets."""
    return dict(
        margin=dict(l=_FIG_MARGIN_L, r=_FIG_MARGIN_R, t=_FIG_MARGIN_T, b=_FIG_MARGIN_B),
        height=FIGURE_HEIGHT[row],
        paper_bgcolor=PAPER_BGCOLOR,
        plot_bgcolor=PLOT_BGCOLOR,
        title=None,
    )


CAPTION_SEP = " · "


def set_panel(fig, *, title: str, caption: List[str], details: List[str],
              row: str = PANEL_ROW_MAP):
    """Attach this panel's WORDS to the figure, for `webapp/app.py` to render as HTML
    above the plot.

    `caption` is a list of clauses joined with CAPTION_SEP into ONE line (the sharing
    pass below rewrites individual clauses, which is why it is a list and not a
    string). `details` is every remaining clause, one per line, for the per-arm
    Details disclosure -- nothing that was ever on screen may be dropped instead of
    moved (acceptance check 15).
    """
    meta = dict(fig.layout.meta or {}) if (fig.layout.meta is not None) else {}
    meta["panel"] = dict(title=title, caption=list(caption),
                         details=[d for d in details if d], row=row)
    fig.update_layout(meta=meta)
    return fig


def _sentence(clause: str) -> str:
    """One subtitle clause as a Details sentence: leading "; " dropped, first letter
    upper-cased, full stop added. `str.capitalize()` is NOT used -- it lower-cases the
    REST of the string, which turned "0 dB = direct path" into "0 db = direct path"
    (caught on the first smoke render, 2026-09-24)."""
    t = (clause or "").strip().lstrip(";").strip()
    if not t:
        return ""
    t = t[0].upper() + t[1:]
    return t if t.endswith(".") else t + "."


def panel_of(fig) -> Dict[str, Any]:
    """This figure's panel meta (see `set_panel`), from either a `go.Figure` or the
    plain dict form the Dash store round-trips. `{}` when absent."""
    if hasattr(fig, "layout"):
        meta = fig.layout.meta
        meta = dict(meta) if meta else {}
    else:
        meta = ((fig.get("layout") or {}).get("meta") or {})
        if not isinstance(meta, dict):
            meta = {}
    panel = meta.get("panel") or {}
    return dict(panel) if isinstance(panel, dict) else {}


def panel_caption(fig) -> str:
    """The panel's one-line caption, clauses joined."""
    return CAPTION_SEP.join(panel_of(fig).get("caption") or [])


def panel_text(fig) -> str:
    """EVERY word this panel carries -- title, caption and the whole Details body --
    as one string.

    The one accessor a test should use to assert an honesty clause is still reachable:
    which of the three places a clause ended up in is a layout decision, and pinning
    the place rather than the clause is what made every subtitle assertion in this
    repo have to be rewritten when the layout changed (2026-09-24)."""
    p = panel_of(fig)
    parts = [p.get("title") or "", CAPTION_SEP.join(p.get("caption") or [])]
    parts.extend(p.get("details") or [])
    return "\n".join(x for x in parts if x)


#: Marks the annotations that make up the reserved statistic strip, so `apply_arm_style`
#: can recolour exactly those and nothing else (an in-plot tag like the detector map's
#: "scoring <= 40 m" must keep its own colour).
_STAT_ANNOTATION_FLAG = "stat_strip"


def _stat_annotations(stat: str, sub: str = "", *, arm: str = "a") -> List[Dict[str, Any]]:
    """The reserved statistic strip above the plot: the headline number left-aligned in
    the arm's colour, and (optionally) the smaller per-frame readouts right-aligned on
    the same line.

    Anchored to the AXES' own domain (`x domain`/`y domain`) at y > 1, so the strip sits
    entirely in the figure's top margin and can never cover a return (hostile round 10,
    defect 6 / acceptance check 8). No background pill: there is nothing underneath it
    to hide any more."""
    out = [dict(text=stat, xref="x domain", yref="y domain", x=0.0, y=1.06,
                showarrow=False, xanchor="left", yanchor="bottom", align="left",
                name=_STAT_ANNOTATION_FLAG,
                font=dict(size=_STAT_FONT_SIZE,
                          color=ARM_COLORS.get(arm, ARM_COLORS["a"])))]
    if sub:
        out.append(dict(text=sub, xref="x domain", yref="y domain", x=1.0, y=1.06,
                        showarrow=False, xanchor="right", yanchor="bottom", align="right",
                        name=_STAT_ANNOTATION_FLAG + "_sub",
                        font=dict(size=_STAT_SUB_FONT_SIZE, color="#576574")))
    return out


def _corner_annotation(text: str, *, y: float = 1.06) -> Dict[str, Any]:
    """Retained name: the headline statistic annotation. It is no longer a "corner" --
    see `_stat_annotations` for why the old INSIDE-the-axes placement was retired."""
    return _stat_annotations(text)[0]


def apply_arm_style(figs: Dict[str, Any], arm: str) -> None:
    """Recolour every figure's statistic strip in `arm`'s colour, in place, on the
    stored figure DICTS (and on each animation frame's layout override, which replaces
    the whole annotations list when the clock steps).

    `figures_from_outputs` builds one run's figures and has no idea which arm it will
    be shown as; the A/B assignment is `webapp/app.py`'s (this run = A, the stored
    previous run = B), so the colour is applied there."""
    color = ARM_COLORS.get(arm, ARM_COLORS["a"])

    def _recolor(layout):
        for ann in (layout.get("annotations") or []):
            if isinstance(ann, dict) and ann.get("name") == _STAT_ANNOTATION_FLAG:
                font = ann.get("font")
                if not isinstance(font, dict):
                    font = {}
                    ann["font"] = font
                font["color"] = color

    for fig in figs.values():
        if not isinstance(fig, dict):
            continue
        _recolor(fig.get("layout") or {})
        for frame in (fig.get("frames") or []):
            _recolor(frame.get("layout") or {})


def _make_legible(fig: go.Figure) -> go.Figure:
    """Bump every text element on `fig` to the podium-distance floor (>= 17 px inside a
    figure, layout spec section 3), IN PLACE, and return it. Safe on any figure --
    heatmap or scatter, with or without a colorbar -- since each update targets an
    element that may simply not be present."""
    fig.update_layout(font=dict(size=_LEGIBLE_FONT_SIZE, color="#2d3a4a"))
    fig.update_xaxes(tickfont=dict(size=_LEGIBLE_TICK_SIZE),
                     title_font=dict(size=_LEGIBLE_FONT_SIZE))
    fig.update_yaxes(tickfont=dict(size=_LEGIBLE_TICK_SIZE),
                     title_font=dict(size=_LEGIBLE_FONT_SIZE))
    for trace in fig.data:
        cbar = getattr(trace, "colorbar", None)
        if cbar is not None:
            cbar.tickfont = dict(size=_LEGIBLE_COLORBAR_TICK_SIZE)
            if cbar.title is not None and cbar.title.text:
                cbar.title.font = dict(size=_LEGIBLE_COLORBAR_TITLE_SIZE)
    for slider in (fig.layout.sliders or ()):
        slider.currentvalue.font = dict(size=_LEGIBLE_FONT_SIZE)
    return fig


def _peak_minus_median_db(db: np.ndarray) -> float:
    """Peak minus median of an already peak-normalized power-dB map -- the T1/T2 card
    headline statistic ("+12 dB", "-14 dB"), computed on the UNCLIPPED map (before any
    display clip). Matches notes/tools/demo_thrust1_rescue.py::q exactly: that script's
    `db.max() - np.median(db)` on `10*log10(ra/ra.max())` is the same quantity as
    `db.max() - np.median(db)` here, since `_to_numpy_abs_db` already peak-normalizes."""
    return float(np.max(db) - np.median(db))


def _radar_cube_clip_db(db: np.ndarray) -> float:
    """Display clip (dB rel. peak) for the range-Doppler panel ONLY -- every other
    heatmap keeps the shared -40 dB clip.

    Physics review, 2026-09-23, measured on the benchmark corpus's first 5 test
    frames (the Thrust 5 presets' own source): the ambient floor sits at median -47.8
    to -39.2 dB and p95 -46.6 to -38.2 dB depending on the frame, sometimes within
    ~1-2 dB of the fixed -40 dB clip -- close enough that floor fluctuation crosses
    the clip boundary and lights up whole Doppler rows as false "smear" (the true
    per-target mainlobe is 6-8 of 64 bins, ~2 m/s). Fixed candidates (-30, -35 dB)
    were tried and left too little margin on the highest-floor frame measured (-35 dB
    left only ~0.2 dB against that frame's p95); an adaptive clip is used instead, so
    the frame's own floor is guaranteed >= 3 dB below the clip: never looser (more
    negative) than the shared -40 dB, but tightened toward the frame's own median +
    3 dB when that median sits above -43 dB.
    """
    return max(-40.0, float(np.median(db)) + 3.0)


#: The colour bar carries NO title any more (layout spec section 4): at 20 px,
#: "dB rel. peak (clipped at -67.1)" was ~310 px wide and Plotly bought that width by
#: shrinking the plot (hostile round 10, defect 5). Units, clip and sharing are ONE
#: sentence in the HTML caption instead. The prefix is kept as the CAPTION clause that
#: names the scale, and `db_colorbar` in `layout.meta` is now what tells the sharing
#: pass which colour bars are dB-scaled.
_DB_COLORBAR_PREFIX = "dB rel. peak"

#: Colour-bar geometry: thin, tall, tick-labels only.
_COLORBAR = dict(thickness=14, len=0.90)

Z_SHARE_REACH_FLOOR = "reach_floor"
Z_SHARE_KEEP_CLIP = "keep_clip"

#: How far BELOW the deepest per-frame median floor of either arm the shared
#: `Z_SHARE_REACH_FLOOR` limit is placed. Mirrors `_radar_cube_clip_db`'s own 3 dB
#: margin, in the opposite direction: that one guarantees the floor stays OUT of the
#: displayed range, this one guarantees it stays IN it on both arms.
_SHARED_FLOOR_MARGIN_DB = 3.0

#: The caption clause that carries the display clip, and the ones the sharing pass
#: adds. Recognised by prefix so `_apply_shared_z` REPLACES rather than appends (two
#: clips printed on one panel read as a bug -- wave 11, 2026-09-24).
CLIP_CLAUSE_PREFIX = "clipped at "
SHARED_SCALE_CLAUSE = "same colour scale on both arms"
INDEPENDENT_SCALE_CLAUSE = "independent colour scales"
#: Marker the Details line that names the exact shared limits starts with. The clause
#: itself is word-for-word the one that used to sit on the panel subtitle, so a reader
#: (and a test) finds the same words -- in Details.
_SHARED_LIMITS_MARKER = "colour limits shared with arm "


def _heatmap(data_db, *, x=None, y=None,
             xlabel: str = "Bin", ylabel: str = "Bin", zmin: float = -40.0,
             zmax: float = 0.0, z_share: str = None, colorscale=None,
             db_colorbar: bool = True) -> go.Figure:
    """One heat-map panel at the FIXED map-row geometry. No title, no subtitle, no
    colour-bar title -- see this section's header comment."""
    kwargs = {}
    if colorscale is not None:
        kwargs["colorscale"] = colorscale
    fig = go.Figure(
        data=go.Heatmap(z=data_db, x=x, y=y, zmin=zmin, zmax=zmax,
                        colorbar=dict(_COLORBAR), **kwargs)
    )
    fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel, **_base_layout())
    meta = {"db_colorbar": bool(db_colorbar)}
    if z_share is not None:
        # Carried on the figure itself, not looked up by product key later -- see the
        # `Z_SHARE_*` constants. `layout.meta` is Plotly's own free-form slot and
        # survives `to_dict()`/the Dash store round trip, which is where
        # `share_heatmap_z_limits` reads it back.
        meta["z_share"] = z_share
    fig.update_layout(meta=meta)
    return fig


#: Range (m) around the direct-path/leakage gate (range 0, the delay-normalised
#: munich frames' own 0 dB reference -- see `earliest_arrival_note` below) excluded
#: before looking for the "brightest visible return" (item 7, coordinator addendum,
#: 2026-09-23, measured on thrust1_circuit_knobs frame 5): the leakage smears into a
#: couple of native bins around the true zero, not only the exact zero gate, so a
#: narrower exclusion would still pick a leakage sidelobe rather than a genuine target.
_DIRECT_PATH_EXCLUSION_M = 2.0

#: Minimum y-axis upper bound for the subspace-error plot (Change 3, 2026-09-22
#: review), so a near-floor curve reads as flat rather than filling the plot height.
#: Picked from the presets' own measured range: Thrust 2's B arm (AFE mantissa 6->1)
#: reaches ~0.63 and Thrust 3's cold start begins ~0.57 (webapp/demo_presets.py
#: blurbs, measured 2026-09-22).
_SUBSPACE_ERR_MIN_YMAX = 0.65
#: The warm-started settled tracking floor the Thrust 2/3 cards quote.
_SUBSPACE_ERR_SETTLED_LEVEL = 0.06
#: Minimum y-axis upper bound for the "refinement passes/frame" right-hand axis
#: (wave 8, W3): the two Thrust 3 arms' right axes used to each autoscale to their own
#: max (A: 0-5, B: 0-10), so a real 2x difference in compute spent per frame rendered
#: at the SAME pixel height on both screens -- pinning both to one common range is
#: what makes that difference visible as a difference in bar/marker height rather than
#: only in the printed numbers. 10 is AdaOjaBlock's own default n_refine ceiling
#: (`subspace_n_refine`'s "none"-arm fallback default, see `run_pipeline`).
_REFINE_AXIS_MIN_YMAX = 10.0


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
        xaxis_title="x (m)", yaxis_title="y (m)",
        margin=dict(l=_FIG_MARGIN_L, r=_FIG_MARGIN_R, t=_FIG_MARGIN_T, b=_FIG_MARGIN_B),
        height=FIGURE_HEIGHT[PANEL_ROW_MAP],
        paper_bgcolor=PAPER_BGCOLOR, plot_bgcolor=PLOT_BGCOLOR,
    )
    # Equal aspect: a plan view with distorted axes misleads about angle, which is the
    # one thing this figure exists to make readable.
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return _make_legible(fig)


#: Assumed paper-x extent of the play/pause button row (two icon buttons, from
#: x=0.0, at the podium-distance 16 px font -- see updatemenus in
#: `_add_frame_animation`). The slider's own x is required to clear this (Change,
#: rehearsal 2026-09-23); a small margin is added on top for the currentvalue
#: label's own width even though it is left-anchored (defense in depth against a
#: future font-size bump).
#:
#: RAISED 0.16 -> 0.34 (coordinator report, 2026-09-24, read on
#: thrust5_detector_cfar_results.png's Range-Doppler panel: the currentvalue label
#: "frame 5" rendered as "rame 5", its "f" hidden under the pause button). The
#: buttons occupy a FIXED PIXEL width (~139 px, measured via a standalone
#: Playwright script -- icon glyph size does not scale with the figure), while the
#: slider's own `x` is necessarily a FRACTION of the card's width (Plotly's slider
#: layout has no pixel-anchored x) -- so one fixed fraction can only clear a fixed
#: pixel width down to some minimum card width, and 0.16 (`_SLIDER_X`=0.24) cleared
#: it only on the WIDE single-card Thrust 1/single-figure layouts this was
#: originally calibrated against (rehearsal 2026-09-23): on Thrust 5's ~700 px
#: multi-card cards, 0.24 lands the label's left edge at ~131-159 px, inside or
#: barely past the ~139 px button width. Re-measured against the SAME real card
#: widths (600-700 px, Thrust 5's own two-column layout) and the wide single-card
#: case (Thrust 1, ~1560 px): 0.34 (`_SLIDER_X`=0.42) clears the buttons by
#: 11-52 px across that whole range.
#: RETIRED (layout spec section 4, "Slider / play controls"; hostile round 10,
#: defect 8): every animated figure used to carry its own `updatemenus` play/pause
#: pair and its own frame slider, up to EIGHT copies of the same transport on one
#: screen, each costing 130 px of bottom margin -- and the presenter never touched
#: any of them, because `webapp/assets/results_clock.js` has driven every animated
#: figure from ONE clock since wave 11. The figure now carries its `frames` and
#: nothing else; the single transport lives in the run-identity row
#: (`webapp/app.py::_transport_bar`, wired to the same clock).
#:
#: These constants are kept, unused by the figure layer, only so the reason they went
#: away is recorded where the next person looks for them.
_SLIDER_BUTTONS_X_EXTENT = 0.34
_SLIDER_MARGIN_B = 0


def _add_frame_animation(fig, per_frame, *, key="z", trace_idx=0, trace_type="heatmap",
                         frame_layouts=None):
    """Attach the per-frame data to `fig` as Plotly `frames`, leaving its initial view
    and its fixed geometry alone.

    `per_frame` is the already-converted data for each frame, in frame order, matching
    whatever `key` the target trace uses ("z" for a heatmap, "y" for a line). The
    figure's existing trace 0 keeps the LAST frame's data, so the default rendering is
    byte-for-byte what it was before animation existed.

    `frame_layouts`, if given, is a per-frame list of layout dicts (same length and
    order as `per_frame`) merged into each `go.Frame` -- used by the map panels to keep
    their statistic strip in step with the clock.

    NO slider and NO play/pause buttons are added: there is one transport per screen,
    not one per panel (see `_SLIDER_BUTTONS_X_EXTENT` above). The clock steps frames by
    NAME ("0".."n-1"), which is unchanged.

    Returns `fig` unchanged when there are fewer than two frames -- an animation over
    one frame is noise.
    """
    n = len(per_frame)
    if n < 2:
        return fig

    # `type` is REQUIRED: without it Plotly infers Scatter for the frame's trace and
    # rejects "z" as an invalid property.
    fig.frames = [
        go.Frame(name=str(i), data=[{"type": trace_type, key: d}], traces=[trace_idx],
                 **({"layout": frame_layouts[i]} if frame_layouts is not None else {}))
        for i, d in enumerate(per_frame)
    ]
    return fig


# =====================================================================================
# A/B shared colour limits (applied at render time, on the stored figure DICTS)
# =====================================================================================

def decode_plotly_array(v) -> list:
    """A trace's x/y/z, after `go.Figure.to_dict()`, is EITHER a plain (possibly
    nested) list OR plotly's compact typed-array encoding
    (`{"dtype": ..., "bdata": <base64>}`, used for large numpy arrays since plotly
    5.20+ -- every range_az/range_el/radar_cube axis and z here is a numpy array).
    Decode either into a list; the typed-array form comes back FLAT (shape is not
    restored), which is all any caller here needs. Found live in the 2026-09-22
    rehearsal: `webapp.app._share_y_ranges` crashed reading 'dtype' as a coordinate
    because it assumed the plain-list form. Used only for bookkeeping -- the stored
    trace dict itself (what actually renders) is untouched.

    ONE authority (`webapp.app._decode_plotly_array` delegates here): both the axis
    sharing in the app and the colour-limit sharing below need it, and two copies of a
    decoder for plotly's own wire format is exactly the kind of duplicate that drifts.
    """
    if v is None:
        return []
    if isinstance(v, dict) and "bdata" in v:
        import base64
        raw = base64.b64decode(v["bdata"])
        return np.frombuffer(raw, dtype=np.dtype(v.get("dtype", "f8"))).tolist()
    return list(v)


def _heatmap_floors_db(fig: Dict[str, Any]) -> List[float]:
    """Median of this panel's own dB data for the INITIAL view and for EVERY animation
    frame -- i.e. the panel's own noise floor, per frame, over the whole run.

    Median (not min): these maps are peak-normalized, so the median IS the ambient
    floor -- the same quantity `_peak_minus_median_db` subtracts from the peak for the
    on-screen "peak - median" statistic, so a colour limit derived from it lines up
    with the number the card quotes.
    """
    floors: List[float] = []

    def _add(trace):
        if trace is None or trace.get("type") not in (None, "heatmap"):
            return
        z = decode_plotly_array(trace.get("z"))
        if len(z):
            floors.append(float(np.median(np.asarray(z, dtype=float))))

    data = fig.get("data") or []
    if data and data[0].get("type") == "heatmap":
        _add(data[0])
    for frame in (fig.get("frames") or []):
        for trace in (frame.get("data") or []):
            _add(trace)
    return floors


def _panel_dict(fig: Dict[str, Any]) -> Dict[str, Any]:
    """The mutable `layout.meta.panel` dict of a stored figure dict, created if
    missing. Mutating what this returns mutates the figure."""
    layout = fig.setdefault("layout", {})
    meta = layout.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        layout["meta"] = meta
    panel = meta.get("panel")
    if not isinstance(panel, dict):
        panel = {"title": "", "caption": [], "details": [], "row": PANEL_ROW_MAP}
        meta["panel"] = panel
    panel.setdefault("caption", [])
    panel.setdefault("details", [])
    return panel


def _apply_shared_z(fig: Dict[str, Any], zmin: float, zmax: float,
                    other_arm: str, *, say_shared: bool) -> None:
    """Pin one figure dict's heat-map traces to `zmin`/`zmax` and SAY so, in the two
    places the layout now keeps words: the panel's one-line CAPTION (which clip is in
    force, and -- on arm A only -- that the scale is shared) and its DETAILS (the exact
    zmin/zmax pair, and the value sharing superseded).

    The clause is a fact about the picture that a photograph of the screen has no other
    way to carry: two maps that share a colour scale look like two maps that happen to
    be the same colour otherwise.

    `say_shared` is the layout spec's "stated once per row, not once per panel" rule:
    the sharing sentence goes at the end of ARM A's caption only, and the exact limits
    go in BOTH arms' Details. The absence of the clause must never be what carries the
    meaning, so the caller states one of `SHARED_SCALE_CLAUSE` /
    `INDEPENDENT_SCALE_CLAUSE` explicitly.
    """
    own_zmin = None
    for trace in (fig.get("data") or []):
        if trace.get("type") != "heatmap":
            continue
        if own_zmin is None and trace.get("zmin") is not None:
            own_zmin = float(trace["zmin"])
        trace["zmin"], trace["zmax"] = zmin, zmax

    panel = _panel_dict(fig)
    caption = [c for c in panel["caption"]
               if not (isinstance(c, str)
                       and (c.startswith(CLIP_CLAUSE_PREFIX)
                            or c == SHARED_SCALE_CLAUSE
                            or c == INDEPENDENT_SCALE_CLAUSE))]
    caption.append(f"{CLIP_CLAUSE_PREFIX}{zmin:.1f} dB")
    if say_shared:
        caption.append(SHARED_SCALE_CLAUSE)
    panel["caption"] = caption

    # "(was X)" whenever sharing MOVED this arm's limit. The superseded value is
    # deliberate provenance (hostile round 10, section 5.3: keep the "(was X)") and is
    # named exactly once, here.
    moved = (f" (was {own_zmin:.1f})"
             if own_zmin is not None and abs(own_zmin - zmin) > 0.05 else "")
    line = (f"{_SHARED_LIMITS_MARKER}{other_arm}: zmin {zmin:.1f} dB{moved}, "
            f"zmax {zmax:.0f} dB")
    details = [d for d in panel["details"]
               if not (isinstance(d, str) and d.startswith(_SHARED_LIMITS_MARKER))]
    details.append(line)
    panel["details"] = details


def share_heatmap_z_limits(figs: Dict[str, Any], prev_figs: Dict[str, Any],
                           labels=("B", "A")) -> None:
    """Give each heat-map product that BOTH arms rendered ONE zmin/zmax, computed over
    BOTH arms and ALL their animation frames, in place.

    Why this is not just "take the tighter of the two clips" (which
    `webapp.app._share_y_ranges` already did): on a `Z_SHARE_REACH_FLOOR` panel the
    NOISE FLOOR is the thing the A/B knob moves, and both arms' floors sat below the
    -40 dB display clip, so both backgrounds rendered as the one clip colour. Owner,
    live test 2026-09-24 (Thrust 1, LNA 8 mA vs 0.5 mA): "images look fine, but 55 vs.
    65 dB peak to median is not discernible by human eye". Pushing the shared limit 3
    dB BELOW the deepest floor either arm reaches puts both floors inside the colour
    ramp, where an 11-12 dB difference is a visible difference in background colour.

    `labels` is (the other arm's name as seen FROM `figs`, the other arm's name as seen
    from `prev_figs`) -- i.e. the default says `figs` is arm A (so its panels are
    "shared with arm B") and `prev_figs` is arm B.
    """
    for key in set(figs) & set(prev_figs):
        pair = (figs[key], prev_figs[key])
        policies = [((fig.get("layout") or {}).get("meta") or {}).get("z_share")
                    if isinstance((fig.get("layout") or {}).get("meta"), dict) else None
                    for fig in pair]
        if policies[0] is None or policies[0] != policies[1]:
            # An untagged panel (the detector objectness maps: heat maps, but their z
            # is a 0-1 score, not dB) keeps whatever limits it was built with.
            continue
        if policies[0] == Z_SHARE_REACH_FLOOR:
            floors = [f for fig in pair for f in _heatmap_floors_db(fig)]
            if not floors:
                continue
            zmin = min(floors) - _SHARED_FLOOR_MARGIN_DB
        else:
            zmins = [float(tr["zmin"]) for fig in pair
                     for tr in (fig.get("data") or [])
                     if tr.get("type") == "heatmap" and tr.get("zmin") is not None]
            if not zmins:
                continue
            # The TIGHTER (higher) of the two adaptive clips: each arm's own clip is a
            # promise that its floor stays >= 3 dB below it (`_radar_cube_clip_db`),
            # and the higher clip keeps that promise for both arms.
            zmin = max(zmins)
        zmax = 0.0
        # Arm A (the first of the pair) carries the "shared" sentence; arm B does not
        # repeat it. Saying it twice invites "why does it need saying twice?" --
        # hostile round 10, section 5.8, on the PR panel's own once-only clause.
        for i, (fig, other) in enumerate(zip(pair, labels)):
            _apply_shared_z(fig, zmin, zmax, other, say_shared=(i == 0))


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
        fft_frames = [_to_numpy_abs_db(f) for f in outputs["fft"]]
        fft_stats = [f"{_peak_minus_median_db(d):.1f} dB peak−median"
                     for d in fft_frames]
        fft_subs = [f"frame {i + 1} of {len(fft_frames)}"
                    for i in range(len(fft_frames))]
        fig = _heatmap(fft_frames[-1], x=u, y=u,
                       xlabel="azimuth sin(θ)", ylabel="elevation sin(θ)",
                       z_share=Z_SHARE_KEEP_CLIP)
        fig.update_layout(annotations=_stat_annotations(fft_stats[-1], fft_subs[-1]))
        set_panel(fig, title="Azimuth-elevation power",
                  caption=[_DB_COLORBAR_PREFIX, f"{CLIP_CLAUSE_PREFIX}-40.0 dB"],
                  details=["Non-coherent (power) integration over range, so a target "
                           "shows up regardless of its range, not just one at range 0."],
                  row=PANEL_ROW_MAP)
        figs["fft"] = _make_legible(_add_frame_animation(
            fig, fft_frames,
            frame_layouts=[dict(annotations=_stat_annotations(s, sub))
                           for s, sub in zip(fft_stats, fft_subs)]))

    # Shared range extent for the range-azimuth heatmap and the range-profile line
    # plot below (Change 4, 2026-09-23 hostile-expert re-read): both compress the
    # SAME physical range axis and, at the UI's default matching bin counts, cover
    # the identical physical span -- but a Heatmap trace's autorange pads its axis
    # differently than a Scatter trace's, so a screen showing both side by side
    # displayed 0-22 m on the heatmap and 0-25 m on the profile despite identical
    # underlying data (measured, thrust4_interconnect_range_profile rehearsal PNG).
    # Pin the heatmap to the profile's own computed extent (arbitrarily the profile,
    # since the finding named it) rather than let each panel autorange independently.
    # `None` (no forcing -- range_az keeps its historical autorange) whenever the
    # profile panel is not part of this run or axis metadata is unavailable: there is
    # then no extent to share.
    range_az_yaxis_extent = None
    if outputs.get("range_profile_agg") and n_freqs and freq_span_hz:
        _bins_rp_for_extent = meta.get("range_profile_bins")
        if _bins_rp_for_extent:
            _cropped_for_extent = _cropped_nonneg_range_axis(
                _bins_rp_for_extent, freq_span_hz, n_freqs)
            if _cropped_for_extent.size:
                range_az_yaxis_extent = float(_cropped_for_extent.max())

    for key, title, qualifier, aperture_label in [
        ("range_az", "Range-azimuth power", "non-coherent over elevation", "azimuth sin(θ)"),
        ("range_el", "Range-elevation power", "non-coherent over azimuth", "elevation sin(θ)"),
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
                # 0 is not "no range" here: the .pkl frames these two panels ever run
                # on (the plain frequency-domain path -- see CLAUDE.md's classic-
                # products/corpus-mode split) were generated with Sionna's
                # normalize_delays=True (sionna_simple_channel.py), which subtracts
                # the shortest path's delay, so range 0 is the earliest arrival, not
                # literally zero range -- the bright full-azimuth band there read as
                # an unlabelled target (hostile-expert re-read, 2026-09-23). Stated in
                # the title SUBLINE below rather than the y-axis title itself: the
                # rotated axis-title text ran into the heatmap's own title at podium
                # font size (fresh-context re-check, 2026-09-23). The corpus-replay
                # panels below (radar_cube/cfar_detection/ml_detection) use a
                # DIFFERENT, absolute range axis from the ADC dechirp geometry and
                # must NOT carry this note -- webapp/demo_presets.py's thrust5
                # scripts already say the absolute-range story for those on screen.
                # Wave 8, W2: a hostile-expert read found no screen states that the 0 dB
                # reference AT range 0 is this leakage band, not a target -- stated
                # directly; "(0 = earliest arrival)" is kept verbatim so it still reads
                # as the SAME claim tests/test_webapp_figures_wave3.py (an unowned file)
                # already pins in this subline. "at range 0" dropped (wave 11, F96):
                # the longer, computed F96 range-window clause below pushed this
                # subline to 7 wrapped lines, past the established 6-line budget
                # (`_apply_shared_z`'s own comment) -- shortened here rather than
                # accepting a 7th line; the 3 substrings the wave-8/wave-3 tests pin
                # ("0 dB = direct path", "(0 = earliest arrival)", "not a target") are
                # untouched, only the connective "at range 0" between them is gone.
                earliest_arrival_note = ("; 0 dB = direct path "
                                         "(0 = earliest arrival), not a target")
                # Calibration nobody stated on screen (wave 7, X6/X7; sharpened wave 8,
                # W13): no panel said what a display gate is worth relative to the
                # frame's own NATIVE frequency-sampling resolution, or how far the axis
                # can go before it wraps. Computed from this frame's own freq_plan +
                # display bin count, never hand-typed.
                _native_m = _native_range_resolution_m(freq_span_hz)
                _gate_m = _range_per_gate_m(bins, freq_span_hz, n_freqs)
                _ratio = _gate_m / _native_m if _native_m > 0 else float("nan")
                # F96 (notes/ESTABLISHED_FACTS.md): the old "unambig 125 m" wording read
                # as a physical range limit, but it is HALF of the frame's own N-point
                # FFT period -- the axis is fftshifted to +-125 m and `_nonnegative_range`
                # (this loop) then crops the negative-delay half away, so a real return
                # between 125 and 250 m lands on the cropped side and is silently
                # discarded, not out of range. Both numbers are computed from this
                # frame's own freq_plan, never typed: the full period is N*c/(2B) (the
                # native per-bin resolution above times the frame's own n_freqs), and the
                # displayed half is exactly that divided by 2.
                _full_window_m = n_freqs * _native_m
                _half_window_m = _full_window_m / 2.0
                gate_note = (
                    f"; {_gate_m:.2f} m/gate ({_native_m * 100:.0f} cm native, "
                    f"{_ratio:.0f}:1); display 0-{_half_window_m:.0f} m of "
                    f"{_full_window_m:.0f} m unambig (neg.-delay half cropped)")
            else:
                # Metadata unavailable (e.g. a hand-built outputs dict): fall back
                # to raw display-gate indices.
                y = np.arange(bins)
                ylabel = "range (bins)"
                earliest_arrival_note = ""
                gate_note = ""
            # Peak-median dB, per frame, on the UNCLIPPED map BEFORE the nonneg-range
            # crop below -- matches notes/tools/demo_thrust1_rescue.py::q, the T1/T2/T4
            # cards' own dynamic-range definition (Change 2). Computed for range_el too
            # (4th hostile-expert read, 2026-09-23): a screen note about the elevation
            # cut had no number beside it while range_az's identical note did, reading
            # as if only range_az's dynamic range had been checked.
            dyn_range_db = [_peak_minus_median_db(d) for d in
                            (_to_numpy_abs_db(f) for f in outputs[key])]
            frames_db = [_to_numpy_abs_db(f) for f in outputs[key]]
            keep = _nonnegative_range(y)
            if frames_db[-1].shape[0] == keep.size:
                frames_db = [f[keep] for f in frames_db]
                y = y[keep]
            # Item 7 (coordinator addendum, wave 9, 2026-09-23, measured on
            # thrust1_circuit_knobs frame 5): the 0 dB reference cell (range 0, the
            # direct-path/leakage gate the caption above already names) renders as
            # roughly ONE native display gate against this panel's own fixed
            # plot-domain height -- sub-pixel at the geometry these presets ship
            # with, so no viewer can actually see the "0 dB" a caption references.
            # Computed per frame from the CROPPED (physical, nonneg-range) map, so
            # "visible" matches what is actually on screen -- never typed. Scoped
            # exactly like `earliest_arrival_note` above (a physical range axis
            # only): this never fires on the corpus-replay radar_cube/detector
            # panels, which use a different, absolute range axis.
            if n_freqs and freq_span_hz and y.size:
                _beyond_direct_path = y >= _DIRECT_PATH_EXCLUSION_M
                # RETRACTED (coordinator re-check, same wave, 2026-09-24): this used
                # to also quote a "~N px" figure from `_HEATMAP_PLOT_DOMAIN_HEIGHT /
                # y.size` -- that divides the module's own DECLARED plot-domain
                # constant by the bin count, not the browser's actual rendered pixel
                # height (measured on the PNG: ~206 px for 126 gates = 1.6 px here,
                # not the 2.2 the old formula printed) -- a wrong, un-reproducible
                # number this module has no way to verify from inside Python. Says
                # "sub-pixel" (true at any plausible render size for a 1 m gate
                # against a 100+ m axis) instead of a specific, wrong pixel count.
                #
                # RETRACTED (wave 9, second hostile-expert read, 2026-09-24, item
                # 4): "not visible" was itself a claim this module cannot verify --
                # the direct-path gate's BRIGHTNESS varies frame to frame (it is
                # real data, not a fixed leakage floor), and on
                # cancel_results.png frame 2 it renders as a visible bright stripe
                # across the full azimuth axis at range 0. What IS true regardless
                # of the frame's own brightness is the gate's geometric size: one
                # display gate tall against a 100+ m axis. Says that, and where a
                # stripe would sit if it IS bright enough to show, rather than
                # asserting invisibility this module cannot check.
                def _direct_path_note(frame_db_2d: np.ndarray) -> str:
                    if not _beyond_direct_path.any():
                        return ""
                    sub = frame_db_2d[_beyond_direct_path]
                    i_flat = int(np.argmax(sub))
                    r_idx = np.unravel_index(i_flat, sub.shape)[0]
                    bright_db = float(sub.flat[i_flat])
                    bright_range_m = float(y[_beyond_direct_path][r_idx])
                    return (f"; 0 dB cell at range 0 is one {_gate_m:.2g} m gate "
                            f"(may show as a thin stripe at the bottom edge); "
                            f"brightest visible return: {bright_db:.1f} dB at "
                            f"{bright_range_m:.0f} m")
                direct_path_notes = [_direct_path_note(f) for f in frames_db]
            else:
                direct_path_notes = [""] * len(frames_db)
            # Adaptive display clip (wave 7, X4/X5): on the pre-diffuse-scattering
            # munich Ka trace the shared -40 dB clip left these screens near-black --
            # 99%+ of pixels sat below it (F94) -- because nothing followed the
            # frame's own floor. Same treatment as `_radar_cube_clip_db`, on the
            # CROPPED (physical-range) last frame so the clip reflects what is
            # actually on screen; on the re-traced file (real multipath restored,
            # 2026-09-23) the floor still sits below -43 dB so this reads "shared
            # floor" (unchanged from -40) -- measured, not assumed either way.
            clip_db = _radar_cube_clip_db(frames_db[-1])
            if clip_db > -40.0:
                clip_provenance = f"clip {clip_db:.1f} dB (median floor + 3 dB)"
            else:
                clip_provenance = f"clip {clip_db:.1f} dB (shared floor)"
            fig = _heatmap(frames_db[-1], x=x, y=y, xlabel=aperture_label,
                           ylabel=ylabel, zmin=clip_db, z_share=Z_SHARE_REACH_FLOOR)
            if key == "range_az" and range_az_yaxis_extent is not None:
                # See `range_az_yaxis_extent`'s definition above the loop.
                fig.update_yaxes(range=[0.0, range_az_yaxis_extent])

            # ---- The panel's words -------------------------------------------------
            # TITLE: one line, HTML above the plot (layout spec section 2.2).
            # CAPTION: one line, <= 110 characters, the units/clip sentence that
            # replaced the colour-bar title, the "colour limits shared" subline and the
            # screen note's clip clause -- three statements of one fact become one
            # (layout spec section 4). `_apply_shared_z` rewrites the clip clause and
            # appends the sharing sentence at render time.
            # DETAILS: every other clause the old six-line subtitle carried, verbatim,
            # so nothing on screen today is deleted -- only moved (acceptance check 15).
            # The two PER-FRAME statistics (peak - median; the brightest visible
            # return) stay visible without expanding anything: they are the reserved
            # statistic strip above the plot, restepped by the clock exactly as the
            # old title was.
            n_frames_key = len(frames_db)
            stat_texts = [f"{d:.1f} dB peak−median" for d in dyn_range_db]
            sub_texts = []
            for i, dp in enumerate(direct_path_notes):
                bright = ""
                if "brightest visible return: " in dp:
                    bright = dp.split("brightest visible return: ", 1)[1].strip()
                    bright = f"brightest visible return {bright} · "
                sub_texts.append(f"{bright}frame {i + 1} of {n_frames_key}")
            fig.update_layout(annotations=_stat_annotations(stat_texts[-1],
                                                            sub_texts[-1]))
            details = [f"Integration: ({qualifier}).",
                       f"peak - median, dB: {dyn_range_db[-1]:.1f} (last frame).",
                       clip_provenance + "; superseded by the shared limits below "
                       "when both arms render this product."]
            if earliest_arrival_note:
                details.append(_sentence(earliest_arrival_note))
            if gate_note:
                details.append(_sentence(gate_note))
            if direct_path_notes[-1]:
                details.append(_sentence(direct_path_notes[-1]))
            set_panel(fig, title=title,
                      caption=[_DB_COLORBAR_PREFIX,
                               f"{CLIP_CLAUSE_PREFIX}{clip_db:.1f} dB"],
                      details=details, row=PANEL_ROW_MAP)

            # The per-frame layout override REPLACES the whole annotations list when
            # the clock steps, so every frame carries its own copy of the strip.
            frame_layouts = [dict(annotations=_stat_annotations(s, sub))
                             for s, sub in zip(stat_texts, sub_texts)]
            figs[key] = _make_legible(_add_frame_animation(fig, frames_db,
                                                           frame_layouts=frame_layouts))

    if outputs.get("range_profile_agg"):
        prof = outputs["range_profile_agg"][-1]
        if hasattr(prof, "detach"):
            prof = prof.detach().cpu().numpy()
        prof = np.asarray(prof, dtype=float)
        bins_rp = meta.get("range_profile_bins") or prof.shape[0]
        if n_freqs and freq_span_hz:
            x = _range_axis(bins_rp, freq_span_hz, n_freqs)
            # Same "0 = earliest arrival" caveat as the range-azimuth/range-elevation
            # panels above (see that loop's comment) -- this panel only ever runs on
            # the same delay-normalised munich frames, never a corpus-replay frame.
            xlabel = "range (m; 0 = earliest arrival)"
            # Wave 8, W2: state the SAME 0 dB = direct-path caveat the range-azimuth/
            # range-elevation sublines now carry (see that loop) -- this panel's own
            # 0 dB point (range 0) is exactly that leakage band.
            direct_path_note = "; 0 dB = direct path at range 0, not a target"
        else:
            x = np.arange(bins_rp)
            xlabel = "range (bins)"
            direct_path_note = ""
        peak = max(float(prof.max()), 1e-12)
        prof_db = 10 * np.log10(prof / peak + 1e-12)
        x = np.asarray(x)
        keep = _nonnegative_range(x)
        if prof_db.shape[0] == keep.size:
            x, prof_db = x[keep], prof_db[keep]
        fig = go.Figure(data=go.Scatter(x=x, y=prof_db, mode="lines"))
        if x.size:
            # Explicit, rather than Scatter's own autorange padding -- this panel IS
            # the reference `range_az_yaxis_extent` (above) pins the heatmap to; a
            # padded autorange here would defeat that match (Change 4, 2026-09-23).
            fig.update_xaxes(range=[0.0, float(x.max())])
        # Fixed display floor, like the heatmaps' -40 dB: an exactly-zero bin
        # (the notched DC bin) otherwise drops to -120 dB and autoscale hangs the
        # whole profile off that one cliff (seen on the Thrust 4 preset, 2026-09-22).
        fig.update_yaxes(range=[-60.0, 2.0])
        # Median floor level, dB rel. peak, of the DISPLAYED (cropped) profile -- the
        # T4 card's "~14 dB floor" statistic (Change 2), computed the same way as the
        # range_az dynamic-range statistic above (median), just without the peak term
        # since a range profile's own peak is already 0 dB by construction. In the
        # title (a "<br><sup>" subline), not a floating annotation: a corner
        # annotation collided with the title text at this font size (rehearsal,
        # 2026-09-23).
        floor_db = float(np.median(prof_db)) if prof_db.size else float("nan")
        # The median-floor statistic is the one the Thrust 4 card quotes, so it goes in
        # the reserved strip above the plot, not into a subtitle (layout spec section 4).
        # This panel is built from the LAST frame and does not animate -- the strip says
        # so, which is the disclosure hostile round 10 (defect 3.2) found missing.
        fig.update_layout(annotations=_stat_annotations(
            f"{floor_db:.1f} dB median floor",
            f"last frame of {len(outputs['range_profile_agg'])} (static)"))
        fig.update_layout(xaxis_title=xlabel,
                          yaxis_title="power (dB rel. peak)",
                          **_base_layout())
        set_panel(fig, title="Range profile",
                  caption=[_DB_COLORBAR_PREFIX, "non-coherent over channels"],
                  details=[
                      "Non-coherent (power) integration over channels.",
                      f"median floor, dB rel. peak: {floor_db:.1f}.",
                      _sentence(direct_path_note),
                      "Built from the LAST frame and static: the range-azimuth map "
                      "above it loops on the clock, this panel does not.",
                  ], row=PANEL_ROW_MAP)
        figs["range_profile"] = _make_legible(fig)

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
        # Adaptive clip (see _radar_cube_clip_db): the shared -40 dB clip sits too
        # close to this corpus's own ambient floor and lights up whole Doppler rows.
        # One clip for the whole animation, from the LAST frame (same convention as
        # every other stat/annotation this module attaches to the animated view).
        rd_clip = _radar_cube_clip_db(first)
        # The computed clip is a float (e.g. -36.233...); the raw value both read as
        # false precision and said nothing about where it came from. Round to one
        # decimal, and say what it is -- but ONLY when it actually came from "3 dB
        # above median" (rd_clip > -40): a quiet frame keeps the plain shared -40 dB
        # clip, which is not median-derived (rehearsal, 2026-09-23).
        # The provenance of the clip goes in the panel's title subline, NOT the colorbar
        # title: a long colorbar title squeezed the heat map to a sliver at two-card
        # width (rehearsal 2026-09-23, all three Thrust 5 screens).
        # ONE line (item 5S, wave 9 second hostile-expert read, 2026-09-24): see
        # `_heatmap`'s own colorbar_title comment.
        rd_clip_caption = f"{CLIP_CLAUSE_PREFIX}{rd_clip:.1f} dB"
        # Shortened (Change 3, 2026-09-23 hostile-expert re-read): the previous
        # subline ("(non-coherent over channels); clip -36.2 dB = this frame's median
        # floor + 3 dB") ran past the two-card (~600 px) panel edge and was clipped
        # mid-sentence ("...median floor + "). Dropping the "(non-coherent over
        # channels)" qualifier keeps this under 60 characters even at the worst case
        # (a negative two-digit clip); see test_webapp_figures_wave3.py for the
        # length pin. The qualifier itself is not lost -- radar_cube's own block
        # comment above and the docstring still state it.
        if rd_clip > -40.0:
            rd_clip_provenance = f"clip {rd_clip:.1f} dB (median floor + 3 dB)"
        else:
            rd_clip_provenance = f"clip {rd_clip:.1f} dB (shared floor)"
        rd_frames = [_rd_db(c) for c in outputs["radar_cube"]]
        rd_stats = [f"{_peak_minus_median_db(d):.1f} dB peak−median"
                    for d in rd_frames]
        rd_subs = [f"frame {i + 1} of {len(rd_frames)}" for i in range(len(rd_frames))]
        fig = _heatmap(first, x=x, y=y, xlabel=xlabel, ylabel=ylabel, zmin=rd_clip,
                       z_share=Z_SHARE_KEEP_CLIP)
        fig.update_layout(annotations=_stat_annotations(rd_stats[-1], rd_subs[-1]))
        set_panel(fig, title="Range-Doppler power",
                  caption=[_DB_COLORBAR_PREFIX, rd_clip_caption],
                  details=[
                      "Non-coherent (power) integration over channels.",
                      rd_clip_provenance + ": this panel's display clip is a "
                      "deliberate decision about what to hide, so sharing it across "
                      "arms only unifies the two clips instead of pushing the limit "
                      "down to the floor.",
                      "A sparse automotive scene at a ~25 dB clip is mostly flat dark "
                      "blue on purpose; the scale is not stretched to make it look "
                      "busy.",
                  ], row=PANEL_ROW_MAP)
        figs["radar_cube"] = _make_legible(_add_frame_animation(
            fig, rd_frames,
            frame_layouts=[dict(annotations=_stat_annotations(s, sub))
                           for s, sub in zip(rd_stats, rd_subs)]))

    det_meta = meta.get("detector") or {}
    for key, title in (("cfar_detection", "CFAR objectness"),
                       ("ml_detection", "Neural detector objectness")):
        if not outputs.get(key):
            continue
        n_frames = len(outputs[key])
        det_label = str(det_meta.get("label", "")) if det_meta else ""
        det_threshold = float(det_meta.get("threshold", 0.0)) if det_meta else None
        # Name the detector and its operating point ON the panel: the three Thrust 5
        # presets are compared across screens, and their cross counts are set by the
        # threshold as much as by the detector. This panel is the last frame while the
        # cube beside it animates; the statistic strip says which frame it is.
        panel_title = f"{title} — {det_label}" if det_label else title
        det = outputs[key][-1]
        if hasattr(det, "detach"):
            det = det.detach().cpu().numpy()
        obj = np.asarray(det)[0]                      # [n_range, n_azimuth], in [0, 1]
        n_r, n_a = obj.shape
        g = rx.get("grid") or {}
        max_r = float(g.get("max_range_m") or n_r)
        y = (np.arange(n_r) + 0.5) * max_r / n_r     # range-bin centres, m
        # The range CROP scoring is actually done at (e2e.ml.compare_detectors'
        # `--max-range-m`, e.g. 40 m) is far smaller than the label grid's own
        # physical extent (max_r above, ~102 m) -- labels themselves are sparse past
        # it (F83, notes/ESTABLISHED_FACTS.md), so ~60% of this panel's range axis was
        # empty (hostile-expert re-read, 2026-09-23, finding 5). Read from
        # beat_cfar.json rather than a literal so this cannot silently drift from what
        # the offline scoring actually used; `None` (file missing) leaves the axis at
        # its old, uncropped behaviour rather than inventing a crop.
        scoring_max_r = detector_scoreboard.scoring_max_range_m()
        x = -1.0 + (np.arange(n_a) + 0.5) * 2.0 / n_a  # sin(azimuth) bin centres
        fig = go.Figure(data=go.Heatmap(
            z=obj, x=x, y=y, zmin=0.0, zmax=1.0, colorscale="Viridis",
            colorbar=dict(_COLORBAR), name="objectness",
        ))
        # Decoded detections (filled) and, for a replayed corpus frame, the stored
        # ground truth (hollow) -- drawn at the surface range the metric matches on.
        dets = (outputs.get(key + "s") or [[]])[-1]
        n_dets = len(dets) if dets else 0
        if dets:
            fig.add_trace(go.Scatter(
                x=[d[1] for d in dets], y=[d[3] for d in dets], mode="markers",
                name=f"✕ detections (n={len(dets)})",
                marker=dict(symbol="x", size=14, color="#ff3b3b", line=dict(width=2)),
                text=[f"score {d[2]:.2f}" for d in dets],
            ))
        gt = (outputs.get("gt_detections") or [[]])[-1]
        n_gt = len(gt) if gt else 0
        hit_rule = ""
        if gt:
            # Ground truth drawn as its own match-tolerance BOX, in DATA coordinates,
            # rather than a fixed-pixel circle: a fixed 18 px circle drew LARGER than
            # what the scoreboard actually scores with (+-2 m range / +-0.06 sin-az is
            # ~5x12 px on this axis), so a cross sitting on the circle could still be a
            # scored miss (adversarial finding, 2026-09-23 -- CFAR frame 5 showed 8
            # crosses on 7 circles yet scored TP 3 / FP 5). The box IS the tolerance,
            # so a cross inside it is genuinely a hit. Numbers come from
            # e2e.ml.metrics.MatchCriterion at call time -- never typed here, so this
            # cannot silently drift from what score_frames (below) actually enforces.
            from e2e.ml.metrics import MatchCriterion
            crit = MatchCriterion()
            r_tol, az_tol = crit.max_range_err_m, crit.max_sin_az_err
            for d in gt:
                cx, cy = d[1], d[3]
                fig.add_shape(
                    type="rect", xref="x", yref="y",
                    x0=cx - az_tol, x1=cx + az_tol, y0=cy - r_tol, y1=cy + r_tol,
                    line=dict(color="#ffffff", width=2), fillcolor="rgba(0,0,0,0)",
                )
            # The hit RULE moves to the caption (layout spec section 4, "Detector
            # map"): as a legend entry it was a 100-character sentence inside a dark
            # block that took 72 px of the panel.
            hit_rule = (f"hit = cross inside the box (±{r_tol:g} m, "
                        f"±{az_tol:g} sin az)")
            fig.add_trace(go.Scatter(
                x=[d[1] for d in gt], y=[d[3] for d in gt], mode="markers",
                name=f"● ground truth (n={len(gt)})",
                marker=dict(symbol="circle", size=6, color="#ffffff",
                            line=dict(width=1, color="#2d3436")),
            ))
        fig.update_layout(
            xaxis_title="azimuth sin(θ)", yaxis_title="range (m)",
            # ONE inline legend line on the panel background, 17 px, 32 px tall (was a
            # 72 px dark block): the marker glyphs are in the trace names above, so the
            # entry reads as "x detections (n=7)" whatever the swatch does.
            legend=dict(orientation="h", yanchor="top", y=-0.22, x=0.0,
                        xanchor="left", bgcolor="rgba(0,0,0,0)",
                        font=dict(size=17, color="#2d3a4a")),
            **_base_layout(),
        )
        if scoring_max_r is not None:
            # 50 m is a fixed display margin above the scoring crop (not itself a
            # claim about anything); the crop value drawn/labelled below IS one, so
            # only it comes from `scoring_max_r`.
            fig.update_yaxes(range=[0.0, 50.0])
            fig.add_hline(
                y=scoring_max_r, line_dash="dash", line_color="#ffffff",
                # A short TAG at the right end of the line, not a centred white
                # sentence across the middle of the data (layout spec section 4).
                annotation_text=f" scoring ≤ {scoring_max_r:g} m ",
                annotation_position="top right",
                annotation_xshift=-10, annotation_yshift=6,
                annotation_bgcolor="rgba(45,58,74,0.7)",
                annotation_font=dict(size=16, color="#ffffff"),
            )
        thr_txt = "n/a" if det_threshold is None else f"{det_threshold:.2f}"
        fig.update_layout(annotations=list(fig.layout.annotations or ())
                          + _stat_annotations(
                              f"{n_dets} detections, {n_gt} labelled",
                              f"frame {n_frames} of {n_frames} (last)"))
        set_panel(fig, title=panel_title,
                  caption=[f"objectness ≥ {thr_txt}"]
                          + ([hit_rule] if hit_rule else []),
                  details=[
                      f"detections at objectness >= {thr_txt} -- frame {n_frames} of "
                      f"{n_frames} (last).",
                      "This panel is pinned to the LAST frame while the range-Doppler "
                      "cube above it loops on the clock, so the two can read as "
                      "different frames.",
                      (f"Ground truth boxes ARE the match tolerance: {hit_rule}."
                       if hit_rule else ""),
                      (f"labels & scoring stop at {scoring_max_r:g} m."
                       if scoring_max_r is not None else ""),
                  ], row=PANEL_ROW_MAP)
        figs[key] = _make_legible(fig)

        # Scoreboard: TP/FP/FN this frame + cumulative hits/false alarms/FA-per-frame/
        # hit-rate over the run, and the match rule in words -- the numbers a hostile-
        # expert read (2026-09-22) said the objectness panel alone does not show. Reuses
        # e2e.ml.metrics' own matcher (webapp/detector_scoreboard.py); appears whenever
        # this detector product is on, next to its objectness panel.
        det_scores = detector_scoreboard.score_frames(
            outputs.get(key + "s") or [], outputs.get("gt_detections"),
            threshold=det_meta.get("threshold"))
        # Which beat_cfar.json arm (if any) this on-screen detector corresponds to --
        # feeds the scoreboard's offline AP/FA/stripe/CI block (Change 1c, 2026-09-23
        # hostile-expert re-read). Best-effort: a missing/malformed beat_cfar.json
        # must not break the live run, only skip that block (arm_name_for_detector
        # itself already returns None for a detector outside the comparison).
        try:
            beat_cfar_arm_name = detector_scoreboard.arm_name_for_detector(det_meta)
        except (FileNotFoundError, ValueError):
            beat_cfar_arm_name = None
        figs[key + "_scoreboard"] = _make_legible(detector_scoreboard.scoreboard_figure(
            det_scores, arm_name=det_meta.get("label", title),
            threshold=det_meta.get("threshold"),
            match_rule_text=detector_scoreboard.match_rule_text(),
            beat_cfar_arm_name=beat_cfar_arm_name))

    if outputs.get("subspace_err"):
        errs = [float(e) for e in outputs["subspace_err"]]
        # Frames are numbered 1..n, matching the heatmap animation slider (which
        # labels its steps 1-based); an implicit 0-based x autoticked at 0.5 on
        # short runs ("Frame 0.5" after a Cancel, rehearsal 2026-09-22).
        # Named (not the "trace 0" default) because adding the n_refine_used trace
        # below turns the legend on -- an unnamed primary trace read as "trace 0" next
        # to "refinement passes/frame" (found in the Thrust 3 rehearsal, 2026-09-23).
        fig = go.Figure(data=go.Scatter(x=list(range(1, len(errs) + 1)), y=errs,
                                        mode="lines+markers", name="subspace error"))
        # Anchor at zero AND give the axis a MINIMUM upper bound (Change 3, 2026-09-22
        # review): the as-loaded Thrust 2 curve rises 0.04 -> 0.06 on an axis that used
        # to autoscale/tozero to 0.06, which reads as "the tracker is diverging". The
        # floor comes from the presets' own measured range -- T2's B arm (mantissa
        # 6->1) reaches ~0.63, T3's cold start begins ~0.57 -- so a near-floor curve now
        # reads as flat near zero instead of filling the plot height. Headroom above
        # the curve's own max is BOTH relative (5%) AND a fixed absolute cushion
        # (wave 8, W15: a cold-start arm's frame-1 point sat visually on the axis
        # ceiling -- close enough to `top` that the marker's own radius touched the
        # border even though the data value was strictly below it).
        top = max(_SUBSPACE_ERR_MIN_YMAX,
                 (max(errs) * 1.05 + 0.03) if errs else 0.0)
        fig.update_yaxes(range=[0.0, top])
        fig.update_xaxes(dtick=1)
        # The settled warm-start level the cards quote, so "is 0.06 good?" has an
        # on-screen answer instead of living only in the operator's script. Labelled
        # "reference" (4th hostile-expert read, 2026-09-23): on a run whose OWN curve
        # sits well above this line (e.g. a cold-start/rank-collapse run reaching
        # ~0.62), an unqualified "settled level (0.06)" reads as if it were THIS run's
        # level rather than a separate warm-start reference case.
        # Collision avoidance (wave 8, W15): "top left" puts the text right beside the
        # FIRST few frames at y ~= the settled level -- a cold-start/refine-gate curve
        # that itself passes near 0.06 early (e.g. T3-B's frame 2) puts a marker right
        # under the annotation. Moved to the right end of the line instead whenever an
        # early frame's error sits within this band; unaffected runs (the common case)
        # keep the original "top left" placement.
        _early = errs[:min(4, len(errs))]
        _annotation_collides = any(
            abs(e - _SUBSPACE_ERR_SETTLED_LEVEL) <= 0.03 for e in _early)
        # Padding off the right plot edge (item 3, wave 9 hostile-expert read,
        # 2026-09-23): "top right" sat the label flush against x=1 (paper), which then
        # ran into the right-hand axis's own tick label (the primary y-axis when there
        # is no refinement-passes trace, or that trace's y2 axis when there is one) --
        # a SECOND collision this branch introduced while fixing the first (the early-
        # frame marker). `annotation_xshift` nudges it left, off that edge, without
        # touching which side ("top right" vs "top left") is chosen -- keeping that
        # choice intact matters: it is what a cold-start/refine-gate run (an early
        # frame near the settled level) versus a clear run actually differ on.
        fig.add_hline(y=_SUBSPACE_ERR_SETTLED_LEVEL, line_dash="dash", line_color="#576574",
                     annotation_text=(f"warm-start settled level "
                                      f"({_SUBSPACE_ERR_SETTLED_LEVEL:g}, reference)"),
                     annotation_position=("top right" if _annotation_collides
                                          else "top left"),
                     annotation_xshift=(-15 if _annotation_collides else 0),
                     annotation_font=dict(size=_LEGIBLE_TICK_SIZE, color="#576574"))
        fig.update_layout(
            xaxis_title="frame",
            # Unnormalised: said in the caption; the rotated axis title at 20 px
            # clipped when it carried the word (rehearsal 2026-09-23).
            yaxis_title="subspace error (Frobenius)",
            **_base_layout(),
        )
        # The statistic the presenter reads off this panel: where the curve ended, and
        # against what reference. Reserved strip above the plot, same as every map.
        fig.update_layout(annotations=_stat_annotations(
            f"{errs[-1]:.2f} at frame {len(errs)}",
            f"warm-start reference {_SUBSPACE_ERR_SETTLED_LEVEL:g}"))
        set_panel(fig, title="Subspace error per frame",
                  caption=["Frobenius, unnormalised distance",
                           "grows ~sqrt(k), not a fraction"],
                  details=[
                      "Unnormalised distance; grows ~sqrt(k), not a fraction.",
                      f"The dashed line is the warm-start settled level "
                      f"({_SUBSPACE_ERR_SETTLED_LEVEL:g}, reference) -- a separate "
                      "warm-start case, not this run's own level.",
                  ], row=PANEL_ROW_MAP)
        fig.update_yaxes(automargin=True)
        # Compute spent per frame (Thrust 3's cold-start-vs-refine-gate A/B, 2026-09-23):
        # AdaOjaBlock's own effective_n_refine() decision (e2e/blocks.py), reported back
        # per frame as outputs["n_refine_used"] -- the "adaptive effort" statistic
        # belongs on the screen, not only in a preset's prose. A second, right-axis
        # trace so an A/B pair reads side by side: two arms spending identical compute
        # (the gate never engaging) draw two overlapping flat lines, which is itself
        # the finding, not a missing feature.
        n_refine_used = outputs.get("n_refine_used")
        if n_refine_used and len(n_refine_used) == len(errs):
            fig.add_trace(go.Scatter(
                x=list(range(1, len(errs) + 1)), y=[int(n) for n in n_refine_used],
                mode="lines+markers", name="refinement passes/frame",
                line=dict(dash="dot", color="#c0392b"), marker=dict(symbol="square"),
                yaxis="y2"))
            # Pinned range, not `rangemode="tozero"` (wave 8, W3): that only anchors
            # zero, it does not stop each arm autoscaling to its OWN max -- an A/B pair
            # at fixed-effort 5 vs the refine gate's 10 rendered as 0-5 and 0-10, so the
            # 2x difference in compute spent sat at the SAME pixel height on both
            # screens and was invisible. This function only ever sees ONE run's own
            # data, so it cannot look at the OTHER arm's max directly -- instead the
            # floor (`_REFINE_AXIS_MIN_YMAX`, AdaOjaBlock's own refine-gate ceiling) is
            # what both arms of the shipped Thrust 3 A/B actually share: the "none" arm
            # never exceeds its fixed effort (5) and the "refine" arm never exceeds its
            # own ceiling (10), so both floor to the SAME 0-10 range and a genuine 2x
            # reads as a height difference. Headroom (a fixed +1, not a %, so it cannot
            # round back down to the floor) only grows the axis past 10 for a run whose
            # OWN data exceeds that shared ceiling (e.g. a more aggressive gate at a
            # different aperture, wave 7 X1) -- otherwise-clipped, not otherwise-shared.
            _max_refine_used = max(int(n) for n in n_refine_used)
            refine_top = (_max_refine_used + 1 if _max_refine_used > _REFINE_AXIS_MIN_YMAX
                         else _REFINE_AXIS_MIN_YMAX)
            # 20% headroom above the shared top (still common to both arms, since
            # `refine_top` itself is): without it, a trace that reaches the top value
            # exactly sits on the plot's top edge and runs through the subtitle text
            # (rehearsal PNGs, 2026-09-23).
            fig.update_layout(
                yaxis2=dict(title="refinement passes/frame", overlaying="y",
                           side="right", range=[0, 1.2 * refine_top], showgrid=False),
                # Legend below the plot, not the default top-right: at top-right it sat
                # on top of the new right-hand axis's own tick labels, clipping "10"
                # into "1C" (found in the Thrust 3 rehearsal, 2026-09-23).
                legend=dict(orientation="h", yanchor="top", y=-0.30,
                           xanchor="left", x=0.0, bgcolor="rgba(0,0,0,0)",
                           font=dict(size=17, color="#2d3a4a")),
                showlegend=True,
            )
            # The compute-per-frame trace is the Thrust 3 A/B. Say where it ended,
            # beside the error statistic, so the "2x compute" claim has a number on
            # screen that does not require reading the right-hand axis.
            panel = panel_of(fig)
            fig.update_layout(annotations=_stat_annotations(
                f"{errs[-1]:.2f} at frame {len(errs)}",
                f"{int(n_refine_used[-1])} refinement passes/frame"))
            set_panel(fig, title=panel["title"], caption=panel["caption"],
                      details=list(panel["details"]) + [
                          "The dotted red trace (right axis) is AdaOjaBlock's own "
                          "effective_n_refine() decision per frame -- the compute "
                          "actually spent, not a preset's prose. Both arms' right "
                          "axes are pinned to one range, so a real 2x reads as a "
                          "height difference.",
                      ], row=panel["row"])
        figs["subspace_err"] = _make_legible(fig)

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
            xaxis_title="Frame",
            yaxis_title="BER",
            yaxis_type="log",
            **_base_layout(),
        )
        fig.update_layout(annotations=_stat_annotations(
            f"BER {bers[-1]:.2e} (last frame)"))
        set_panel(fig, title="Comms head BER", caption=[title.strip("()") or combining],
                  details=[f"Combining: {combining}.",
                           f"Frames with 0 bit errors are shown at the {ber_floor:g} "
                           "display floor -- a log axis cannot plot an exact zero."],
                  row=PANEL_ROW_MAP)
        if any(b < ber_floor for b in bers):
            fig.add_annotation(
                text=f"frames with 0 bit errors shown at the {ber_floor:g} floor",
                xref="paper", yref="paper", x=0.5, y=1.02,
                showarrow=False, font=dict(size=11, color="#576574"),
            )
        figs["ber"] = _make_legible(fig)

    if outputs.get("evm"):
        evms = [float(e) for e in outputs["evm"]]
        fig = go.Figure(data=go.Scatter(y=evms, mode="lines+markers"))
        fig.update_layout(xaxis_title="Frame", yaxis_title="EVM", **_base_layout())
        fig.update_layout(annotations=_stat_annotations(
            f"EVM {evms[-1]:.3f} (last frame)"))
        set_panel(fig, title="Comms head EVM per frame",
                  caption=["error vector magnitude, per frame"],
                  details=["Error vector magnitude of the equalized data symbols, "
                           "one point per frame."],
                  row=PANEL_ROW_MAP)
        figs["evm"] = _make_legible(fig)

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
        fig.update_layout(xaxis_title="I", yaxis_title="Q", **_base_layout())
        set_panel(fig, title="Comms head constellation",
                  caption=["last frame, equalized",
                           "coloured by transmitted symbol"],
                  details=["Last frame, equalized; each received symbol is coloured "
                           "by the ideal point it was actually TRANSMITTED as."],
                  row=PANEL_ROW_MAP)
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        figs["comm_const"] = _make_legible(fig)

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
        margin=dict(l=20, r=20, t=20, b=20), height=FIGURE_HEIGHT[PANEL_ROW_MAP],
        paper_bgcolor=PAPER_BGCOLOR, plot_bgcolor=PLOT_BGCOLOR,
    )
    set_panel(fig, title="No data", caption=[message], details=[message],
              row=PANEL_ROW_MAP)
    return fig
