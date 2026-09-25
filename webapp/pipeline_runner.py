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
from typing import Any, Dict, List, Optional

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


class _OFDMTxCfg:
    """The single-TX radar config the OFDM/JSAC mixing block reads.

    `SymbolDivisionBlock` needs exactly two things off a cfg -- the MIMO scheme and the
    transmit-element count -- because its tail IS `e2e.chain.dechirp`'s (`beat_from_cfr`
    + `mimo_combine`, imported rather than reimplemented, which is what makes the FMCW
    parity oracle bit-exact). The stored munich frames are single-TX, and a chirp/frame
    TIMING preset means nothing to a waveform whose symbol duration comes from the
    subcarrier spacing instead, so a full `RadarConfig` here would be four numbers that
    are never read and could drift.
    """

    mimo = "single"
    n_tx = 1


#: Products whose frames carry one image PER SYMBOL on a JSAC run.
_SYMBOL_IMAGE_PRODUCTS = ("fft", "range_az", "range_el")


def _display_symbol_for(spec) -> int:
    """WHICH symbol's image the screen shows, and it is not a free choice.

    `sensing_source="preamble"`: symbol 0. It is the all-pilot symbol -- the one whose
    transmitted grid is identically 1 -- so its image is the FMCW bit-parity point
    (oracle O2) rather than an arbitrary pick, and every other symbol's sensing
    reference is masked to zero, i.e. their cubes are empty by construction.

    `sensing_source="pilots_only"`: symbol 1, THE FIRST DATA SYMBOL, and this is the
    whole demonstration. Measured 2026-09-24, and it is what a first pass gets wrong:
    under this source `OFDMFrame.reference_grid` keeps symbol 0 as the FULL all-pilot
    preamble and puts the comb on symbols 1..M-1 only. So symbol 0's image is the full
    499.55 m window WHATEVER the pilot spacing is -- showing it makes the resource-split
    A/B render two IDENTICAL pictures (both arms measured 72.86-72.88 dB peak-median,
    to the digit). The split lives on the data symbols, so that is the symbol to show.
    """
    frame = getattr(spec, "frame", None)
    source = getattr(frame, "sensing_source", "preamble")
    n_symbols = int(getattr(frame, "n_symbols", 1) or 1)
    return 1 if (source == "pilots_only" and n_symbols > 1) else 0


def _waveform_run_notes(spec) -> List[str]:
    """What the OFDM/JSAC chain COMPUTED, in the run notes, from `ChainSpec.notes`.

    Every number here is derived from the source's own frequency plan by
    `e2e.comms.ofdm_isac` -- subcarrier spacing from the plan's endpoint-inclusive step,
    sample rate from `N * delta_f`, the sensing window from `c / (P * delta_f)`, the data
    rate from the comb the frame actually carries. None of it is typed into a preset,
    which is the point: the resource-split knob moves the window and the rate in opposite
    directions and the screen reads both off the frame it just ran.
    """
    n = dict(spec.notes or {})
    notes: List[str] = []
    bits = {2: "QPSK", 4: "16-QAM", 6: "64-QAM"}.get(int(n.get("bits_per_symbol", 0)),
                                                     "%s b/symbol"
                                                     % n.get("bits_per_symbol"))
    head = ("%s waveform: %s subcarriers at %.1f kHz spacing, %s symbols, %s, "
            "pilot spacing %s (%s)."
            % (spec.kind.upper(), n.get("n_fast"), float(n.get("delta_f_hz", 0)) / 1e3,
               n.get("n_slow"), bits, n.get("pilot_spacing"),
               n.get("sensing_source")))
    notes.append(head)
    rate = n.get("data_rate_bps")
    dur = n.get("frame_duration_s")
    if rate is not None and dur is not None:
        notes.append(
            "Burst data rate %.3f Gb/s over a %.3f us frame -- uncoded, and it is a "
            "BURST rate: the average depends on a duty cycle this run does not define."
            % (float(rate) / 1e9, float(dur) * 1e6))
    if spec.sensing and n.get("sensing_window_m") is not None:
        shown = _display_symbol_for(spec)
        which = ("symbol %d, the all-pilot preamble -- the symbol whose transmitted "
                 "grid is identically 1, which is also the FMCW bit-parity point"
                 % shown) if shown == 0 else (
            "symbol %d, the first DATA symbol -- the one carrying the sensing comb, "
            "which is where the resource split actually is (symbol 0 stays an all-pilot "
            "preamble at every pilot spacing, so its image would not move with the knob)"
            % shown)
        notes.append(
            "Sensing window %.2f m of excess path (c / (pilot spacing x subcarrier "
            "spacing)); the image is %s. Its range-Doppler is deliberately absent: an "
            "FFT over OFDM symbols of one time-invariant stored channel is a delta at "
            "bin 0 dressed up as a velocity."
            % (float(n["sensing_window_m"]), which))
    rise = n.get("qam_noise_rise_db")
    if rise is not None and float(rise[0]) > 0.01:
        notes.append(
            "Symbol division amplifies noise by 1/|X|^2 where the transmitted symbol is "
            "not constant-modulus: %.2f dB mean / %.2f dB worst-subcarrier rise in the "
            "cube's floor at this constellation (closed form, from the constellation "
            "this run transmits)." % (float(rise[0]), float(rise[1])))
    return notes


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


#: The Ka-band carrier the munich frames (and every Sionna-path screen) run at, in GHz
#: -- used ONLY to decide whether a replayed corpus's own recorded carrier is that
#: band or a legacy one worth marking as such in the run-identity line (see the
#: `_axis_meta["band"]` assignment in `run_pipeline`). Matches
#: `_resolve_interconnect_band_hz`'s own 30 GHz fallback carrier above.
KA_BAND_CARRIER_GHZ = 30.0

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


def _stored_frame_composition(files) -> str:
    """WHICH chain composition wrote these corpus frames -- read off the frames.

    The live-chain replay has to build the stage order the corpus was generated with,
    or the gate reads a non-zero difference that no knob on screen explains. Since
    2026-09-24 every frame's meta records it (`e2e.ml.chain_generate._ChainFlagsStage`:
    `composition` = "full" | "legacy_impulse"). A frame written before that key existed
    carries none, and every such corpus on disk was written by the v1.0 order, so the
    absence IS the answer: `"legacy_impulse"`.

    Deliberately reads the FIRST readable frame only: the composition is a constant of a
    `Simulation` run, and the per-frame comparison of the flags that can differ is
    `_StoredFrameSettingsStage`'s job. Never raises -- a provenance lookup must not take
    a run down; an unreadable corpus falls through to the legacy answer and the gate
    then reports whatever difference results.
    """
    import json

    for path in list(files)[:1] or []:
        try:
            with np.load(path, allow_pickle=False) as data:
                meta = json.loads(str(data["meta"].item()))
        except Exception:
            return "legacy_impulse"
        recorded = str(meta.get("composition") or "").strip()
        if recorded in ("full", "legacy_impulse"):
            return recorded
        return "legacy_impulse"
    return "legacy_impulse"


#: The stage classes that MIX -- the one point on the spine where a frequency response
#: becomes a sampled record. Named here so `_composition_record` reads the crossing off
#: the stage list instead of assuming which one a run used.
_MIXING_STAGES = ("DechirpBlock", "OFDMReceiveBlock")

#: Where the analog front end sat, as a word the cards and the help text may both use.
#: "beat"    -- `FrontEndBlock` on the sampled beat record, AFTER the mixing block
#:              (the FULL contract, section 1.2 row 5).
#: "symbol"  -- `CircuitStage` ahead of an OFDM receiver: `ifft` of a received OFDM grid
#:              IS the received time-domain symbol, so the cascade sees a real sampled
#:              signal here too (F98; `OFDMReceiveBlock`'s docstring).
#: "impulse" -- `CircuitStage` ahead of a dechirp, i.e. on `ifft(CFR)`, the channel
#:              impulse response: the v1.0 order, kept ONLY to reproduce corpora that
#:              were generated with it bit-for-bit (F96, F97c).
#: "none"    -- no front end on this run.
FRONT_END_PLACEMENTS = ("beat", "symbol", "impulse", "none")


def _composition_record(stages) -> Dict[str, Any]:
    """WHICH composition this run actually built, read off the stage list that ran.

    Measured, never assumed: the placement is a property of the ORDER of the stages the
    run composed, and three different code paths in this module compose one (the
    frequency-domain spine `Simulation` builds, the receive chain this module builds for
    a corpus replay, and the OFDM/JSAC spine). Reading the built list is the only answer
    that cannot drift from any of them.

    Why this exists (measured 2026-09-25, all eight presets at `3017497`): a hostile
    round read `chain_composition` at the corpus branch below and concluded that Thrusts
    1-4 run the front end on `ifft(CFR)`. They do not -- that variable governs only the
    branch that composes its own `serial_stages`, and the munich presets leave
    `serial_stages=None`, so `Simulation._build_spine` builds its default, which is
    `composition="full"`: `DechirpBlock -> FrontEndBlock -> RangeTransformBlock`. A
    claim about the placement now comes from this record, on the run, rather than from
    reading one branch of the builder.

    Returns `composition` ("full" | "legacy_impulse"), `front_end_placement` (one of
    `FRONT_END_PLACEMENTS`) and `noise_injected_by` (the stages that actually draw a
    thermal sample, in chain order -- empty when nothing does).
    """
    names = [type(s).__name__ for s in stages]
    mix = next((i for i, n in enumerate(names) if n in _MIXING_STAGES), None)
    mixer = names[mix] if mix is not None else None
    fe_beat = next((i for i, n in enumerate(names) if n == "FrontEndBlock"), None)
    fe_cfr = next((i for i, n in enumerate(names) if n == "CircuitStage"), None)

    if fe_beat is not None:
        placement = "beat"
    elif fe_cfr is None:
        placement = "none"
    elif mixer == "OFDMReceiveBlock":
        placement = "symbol"
    else:
        placement = "impulse"

    injectors: List[str] = []
    for i, stage in enumerate(stages):
        name = names[i]
        if name == "FrontEndBlock" and getattr(stage, "inject_noise", True):
            injectors.append("front end (Friis cascade on the beat record)")
        elif name == "CircuitStage":
            rffe = getattr(stage, "rffe_block", None)
            if getattr(rffe, "inject_noise", True):
                injectors.append("front end (Friis cascade, %s domain)"
                                 % ("symbol" if mixer == "OFDMReceiveBlock"
                                    else "impulse"))
        elif name == "ThermalNoiseBlock":
            mode = str(getattr(stage, "mode", "legacy"))
            if mode == "legacy":
                injectors.append("link budget / thermal floor (k.T.B.F, legacy mode)")
            elif not injectors:
                injectors.append("link budget / thermal floor (k.T.B.F, the one "
                                 "injection -- no front end ran)")
    return {
        # "full" is every placement the contract endorses; only the v1.0 order that
        # exists for stored-corpus bit parity is the legacy one.
        "composition": "legacy_impulse" if placement == "impulse" else "full",
        "front_end_placement": placement,
        "noise_injected_by": injectors,
        "mixing_block": mixer,
        "stages": names,
    }


def _composition_note(record: Dict[str, Any]) -> str:
    """The run note that says WHERE the front end ran on this run, in the words the
    RFFE help text uses. One authority: the help text points at this note rather than
    asserting a placement that is only true on some screens."""
    placement = record.get("front_end_placement")
    where = {
        "beat": "the SAMPLED BEAT RECORD, after the mixing block (the full "
                "composition: dechirp -> front end -> range transform)",
        "symbol": "the RECEIVED OFDM SYMBOL, ahead of the receiver -- ifft of the "
                  "received grid is the time-domain symbol a real amplifier sees",
        "impulse": "the CHANNEL IMPULSE RESPONSE ifft(CFR), before the mixing block "
                   "-- the v1.0 order, which is what these stored frames were "
                   "generated with, so the live-vs-stored gate can read zero codes "
                   "(F96, F97c: below one LSB on the signal path, floors identical)",
        "none": "nowhere: no front end is enabled on this run",
    }.get(placement, str(placement))
    floors = record.get("noise_injected_by") or []
    return (
        "Front end applied to " + where + ". Thermal floor injected by: "
        + (", ".join(floors) if floors else "nothing on this run")
        + "."
    )


def _placement_caption(record: Dict[str, Any]) -> str:
    """`_composition_note` at ARM-CAPTION length (hostile round 14, K2): the measured
    placement reached only Details, so no visible surface said where the front end ran.
    Same record, same authority; "" when nothing was measured."""
    return {
        "beat": "front end on the beat record",
        "symbol": "front end on the OFDM symbol",
        "impulse": "front end on ifft(CFR), the frames' own order",
        "none": "no front end",
    }.get(str((record or {}).get("front_end_placement") or ""), "")


def _frequency_chain_radar_cfg(state: Dict[str, Dict[str, Any]], env_block: Any,
                               array_shape, run_notes: List[str]):
    """The `RadarConfig` the FULL composition's spine is built from, on a run whose
    source is a channel frequency response (Thrusts 1-4).

    `Simulation`'s default spine puts the front end on the SAMPLED BEAT RECORD, which
    needs a beat sample rate to reference its noise bandwidth to (`min(if_bw, fs)`), and
    it refuses to invent one -- see `FrontEndBlock.fs_hz`. The plan is DERIVED from the
    frame's own frequency grid by `fmcw_plan_from_freq_plan`, which enforces
    `S/fs == (stop - start)/(num_freqs - 1)` on the endpoint-inclusive grid the frames
    were traced on (F97d: getting that `N/(N-1)` factor wrong is a silent 0.02 % range
    scale error, ~0.45 m at the far end of the munich window).

    A v2 pkl carries the plan. A LEGACY pkl (munich.pkl) and the synthetic test
    fixtures do not, and for those the grid is reconstructed from the same UI span the
    rest of this module already falls back to (`_resolve_freq_span_hz`, centred on
    `KA_BAND_CARRIER_GHZ`) -- with a run note, because the sweep time that follows is a
    STATED convention and not a measurement. It changes no image: any `(S, fs)` with
    `S/fs = df` yields the same beat record; it changes only what sample rate the noise
    and IF filters are referenced to.

    Returns None when even that is impossible, in which case a chain that configures a
    front end raises `Simulation`'s own error naming all three ways out.
    """
    try:
        from e2e.chain.waveform import fmcw_plan_from_freq_plan
    except ImportError:
        return None
    n_rx = int(array_shape[0]) * int(array_shape[1])
    plan = getattr(env_block, "freq_plan", None)
    if plan:
        try:
            return fmcw_plan_from_freq_plan(plan, n_rx=n_rx, name="fmcw_from_frame_plan")
        except ValueError:
            return None
    try:
        n_freqs = int(env_block.get_S_pars().shape[-1])
    except Exception:
        return None
    if n_freqs < 2:
        return None
    span = float(_resolve_freq_span_hz(state, env_block))
    carrier = KA_BAND_CARRIER_GHZ * 1e9
    try:
        cfg = fmcw_plan_from_freq_plan(
            {"start_hz": carrier - span / 2.0, "stop_hz": carrier + span / 2.0,
             "num_freqs": n_freqs},
            n_rx=n_rx, name="fmcw_from_ui_span")
    except ValueError:
        return None
    run_notes.append(
        f"these frames carry no frequency plan, so the beat sample rate the front end "
        f"references its noise band to ({cfg.fs_hz / 1e6:.3f} MS/s) is derived from the "
        f"UI's {span / 1e9:.3f} GHz span centred at {KA_BAND_CARRIER_GHZ:g} GHz on a "
        f"nominal 200 us sweep -- a stated convention, not a measurement; it changes no "
        f"image, only the band the noise and IF filters are referenced to")
    return cfg


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

    def caption(self, frames_bits: Optional[int] = None) -> str:
        """The gate at ARM-CAPTION length, WITH its verdict (hostile round 14, J4):
        "max |diff| = 0 of 4096 LSB vs stored: bit-identical" / "max |diff| = 1 of 8 LSB vs stored: differs
        (this run's ADC 3-bit)". The verdict word is the one `note()` and `short()`
        use, from the same comparison, so the caption cannot disagree with Details.
        `frames_bits` (the depth the frames were WRITTEN at, when they record it) lets
        a difference be attributed to the bit depth only when the bit depth is what
        differs; otherwise "differs" stands alone and the run notes attribute it."""
        if self.problem is not None or not self.n_compared:
            return "live vs stored ADC: not compared"
        if self.quantizer_block is None:
            return ("live vs stored ADC: "
                    + ("bit-identical" if self.max_abs_diff == 0.0 else "differs"))
        # "max |diff| = " (shard 3f, 2026-09-25): the bare "2815 of 4096 LSB vs
        # stored" did not say what the 2815 IS -- a count, a code, a mean?
        head = f"max |diff| = {self.max_lsb_diff} of {self.lsb_total} LSB vs stored: "
        if self.max_lsb_diff == 0:
            return head + "bit-identical"
        if frames_bits is not None and self.bits is not None and int(frames_bits) != self.bits:
            return head + f"differs (this run's ADC {self.bits}-bit)"
        return head + "differs"

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
    meta = {"mode": mode, "threshold": threshold, "label": label,
            "display_label": label}
    if mode != "cfar":
        # ONE NAME PER CHECKPOINT ON SCREEN (hostile round 12, item 16). `label` is the
        # checkpoint's DIRECTORY name (`b15_fftradnet_rd_ka`) and it stays `label`,
        # because it is the key `arm_name_for_detector` matches the scored arms on. But
        # the panel title printed it while the PR legend and the scoreboard heading
        # printed the SCORED ARM's name (`fftradnet_rd_b15`) -- two names for one
        # checkpoint, on one screen, neither of them wrong. The screen now says the
        # scored arm's name everywhere; the directory name is in the detector's own
        # Details. CFAR keeps its label ("CA-CFAR (guard 2, train 6)"), which carries
        # the operating point the arm name does not.
        try:
            from webapp import detector_scoreboard as _ds

            arm = _ds.arm_name_for_detector(meta)
            if arm:
                meta["display_label"] = _ds._display_arm_name(arm)
        except (FileNotFoundError, ValueError, ImportError):
            pass
    return meta


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

    # --- the RECEIVE segment of the one chain (e2e/chain/dechirp.py, receive.py) --
    # "dechirp" is the MIXING block's activation toggle: it carries the frame across
    # from the channel frequency response into the sampled beat record
    # (state['signal_domain'] flips from DOMAIN_CFR to DOMAIN_RX_TIME -- see
    # e2e/frames.py) and everything after it is the receive side. When it is on, this
    # module composes the whole spine itself (Simulation's documented `serial_stages=`
    # hook) because it has stages Simulation's own default spine does not carry --
    # impairments, the IF high-pass, the quantiser, and each frame's stored-settings
    # stage. It is ONE list either way: the frequency-domain products and the
    # beat/cube products are taps at different points on it, not two pipelines. (The
    # 2026-09-24 one-chain contract, section 1.2; the "mutually exclusive chains" rule
    # that used to live here is gone, and so is the comms-vs-dechirp refusal below.)
    #
    # PRODUCTS ARE ORDERED TAPS, and they have to be: `Simulation` drops the previous
    # domain's payload at every crossing, so a detector that reads `adc` cannot run
    # after the range transform and `RadarCubeBlock`, which reads `cube`, cannot run
    # before it. Thrust 5 enables both. See `Simulation._register_product_taps`.
    spine_stages = None
    adc_gate = None
    meta_stage = None
    #: Stages on `spine_stages` whose returns are products (passed to Simulation as
    #: `product_taps=`), in the order they tap the chain.
    product_taps: List[Any] = []
    #: Set by the OFDM/JSAC branch below: the `ChainSpec` this run's waveform class
    #: contributed. None on an FMCW run.
    wave_spec = None
    #: THE WAVEFORM CLASS. `fmcw` is the chain every preset before this one ran: on a
    #: unit-modulus chirp the dechirp identity IS the modulation, so a stored CFR needs
    #: no transmit tributary at all. `ofdm` and `jsac` are the other two rows of
    #: `e2e.comms.ofdm_isac.waveform_chain_spec` -- READ from there rather than
    #: re-decided here, so the dropdown, the diagram's one branch point and the products
    #: a screen shows cannot disagree about what a class is.
    wave_kind = (str(_p(state, "waveform", "kind")) if _enabled(state, "waveform")
                 else "fmcw")
    if wave_kind in ("ofdm", "jsac"):
        if corpus_mode:
            raise PipelineError(
                "The %r waveform class transmits a grid onto the stored CHANNEL, and "
                "Corpus Replay serves frames that are already past that point. Use the "
                "Sionna Environment source, or set the waveform back to 'fmcw'."
                % wave_kind)
        if _enabled(state, "dechirp"):
            raise PipelineError(
                "The %r waveform class brings its own mixing block, so the Dechirp "
                "bridge must be off. Turn off 'Dechirp (channel -> ADC)', or set the "
                "waveform back to 'fmcw'." % wave_kind)
        try:
            from e2e.chain.receive import RangeTransformBlock
            from e2e.comms.blocks import BERBlock
            from e2e.comms.ofdm_isac import waveform_chain_spec
        except ImportError as e:
            raise PipelineError(
                "Could not import the OFDM/JSAC waveform backend (e2e.comms.ofdm_isac "
                "/ e2e.chain.receive). Underlying error: " + str(e))
        freq_plan = getattr(environment_block, "freq_plan", None)
        if not freq_plan:
            raise PipelineError(
                "The %r waveform places its subcarriers ON the stored channel's own "
                "frequency grid, so the source must carry a freq_plan (a v2 .pkl). "
                "This scenario's frames do not." % wave_kind)
        try:
            wave_spec = waveform_chain_spec(
                wave_kind, _OFDMTxCfg(), freq_plan=freq_plan,
                n_symbols=int(_p_positive(state, "waveform", "n_symbols")),
                pilot_spacing=int(_p_positive(state, "waveform", "pilot_spacing")),
                bits_per_symbol=int(_p_positive(state, "waveform", "bits_per_symbol")),
                sensing_source=str(_p(state, "waveform", "sensing_source")),
                combining=str(_p(state, "waveform", "combining")),
            )
        except (ValueError, TypeError) as e:
            raise PipelineError("%s waveform: %s" % (wave_kind, e))
        # THE SPINE, in the contract's order. The only thing the waveform class changes
        # is the mixing block and which half of the products exists -- everything else
        # is the same list every other preset runs.
        spine_stages = []
        spine_stages.extend(wave_spec.tributary_stages())      # Y = H . X
        if interconnect_block is not None:
            spine_stages.append(InterconnectStage(interconnect_block))
        if circuit_block is not None:
            # THE FRONT END IS IN THE FREQUENCY DOMAIN ON THIS PATH, and that is the
            # physics rather than a workaround (see `OFDMReceiveBlock`'s docstring):
            # `ifft(s_pars)` of a RECEIVED OFDM grid IS the received time-domain symbol,
            # the signal a real amplifier sees -- which is what makes `RFFEBlock`'s
            # round trip correct here and wrong on an FMCW CFR (F96). Placed here it
            # runs BEFORE the comms head, so its floor reaches the constellation AND the
            # image: one knob, both products, one frame.
            spine_stages.append(CircuitStage(circuit_block))
        ber_block = BERBlock()
        spine_stages.extend([wave_spec.receive_block, ber_block])
        product_taps.extend([wave_spec.receive_block, ber_block])
        if wave_spec.sensing:
            spine_stages.append(wave_spec.mixing_block)
            # The imaging identity point, the same one `Simulation._build_spine` uses
            # for the frequency-domain presets (window="none", dc_removal=False): the
            # munich trace is generated with normalize_delays=True, so the
            # line-of-sight path sits AT bin 0 and a fast-time mean subtraction would
            # zero it.
            spine_stages.append(RangeTransformBlock(None, window="none",
                                                    dc_removal=False))
        else:
            # `ofdm` is `jsac` with the mixer omitted: no cube, so no sensing product
            # can exist. Refusing by name beats rendering an empty panel.
            unsupported = [bid for bid in ("fft", "range_az", "range_el",
                                           "range_profile", "subspace_err")
                           if _enabled(state, bid)]
            if unsupported:
                raise PipelineError(
                    "The 'ofdm' waveform class is comms only -- a comms receiver "
                    "equalises the grid and never forms a cube -- so it cannot produce "
                    + ", ".join(unsupported) + ". Switch the waveform to 'jsac' (same "
                    "frame, one extra block, and the image appears), or turn those "
                    "products off.")
        # v1.1 SCOPE, stated rather than discovered live (the JSAC build spec's section
        # 3.9, option (c)): the AFE compressor, the subspace tracker and the range
        # profile read one snapshot per chirp and reject an M-symbol frame by name.
        # They are FMCW-arm products on this release, and the preset's card says so.
        symbol_rejecting = [bid for bid in ("range_profile", "subspace_err")
                            if _enabled(state, bid)]
        if wave_spec.sensing and symbol_rejecting:
            raise PipelineError(
                "The adaptive front end, the subspace tracker and the range profile "
                "read one snapshot per chirp and reject a multi-SYMBOL frame, so they "
                "are FMCW-arm products in v1.1: turn off "
                + ", ".join(symbol_rejecting) + " on a jsac run.")
        downstream_blocks = [b for b in downstream_blocks
                             if type(b).__name__ in ("FFTBlock", "RangeAzBlock",
                                                     "RangeElBlock")]
        run_notes.extend(_waveform_run_notes(wave_spec))
        if _enabled(state, "comms"):
            run_notes.append(
                "The Comms Head block is ignored on this run: the %r class brings its "
                "own receiver (OFDMReceiveBlock), which reads the chain's OWN received "
                "grid and injects no noise of its own -- the legacy head synthesises a "
                "channel and a floor of its own, which would count the noise twice."
                % wave_kind)
    elif corpus_mode and not corpus_live_cfr:
        # The stored ADC replay ENTERS the one chain after the quantiser: the corpus
        # frame was generated by this very chain (dechirp -> thermal floor ->
        # impairments -> IF HPF -> quantizer) and stored AFTER it, so re-running any of
        # that here would impair an already-impaired frame. The spine therefore starts
        # at the beat-record taps and the range transform -- the stages the stored
        # frame has NOT been through -- and the run notes below name every enabled
        # block that was skipped. This is the LEGACY replay: a corpus with no stored
        # channel to run from.
        spine_stages = []
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
        spine_stages = []
        thermal_block = impairment_block = if_hpf_block = quantizer_block = None
        #: The beat-placement front end, when this chain uses one. It, not the
        #: `RFFEBlock` it was built from, is the stage that actually draws the front
        #: end's noise, so it is the one the stored-settings stage must re-seed per
        #: frame -- re-seeding the `RFFEBlock` instead left the live chain one LSB off
        #: the stored cube on every frame but the first (measured 2026-09-24).
        front_end_block = None

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
            spine_stages.append(WaveformBlock(
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
                spine_stages.append(TxPABlock(tx_pa))
            if _enabled(state, "modulate"):
                spine_stages.append(ModulateBlock(
                    tx_pa=tx_pa,
                    bandwidth_hz=float(_p(state, "modulate", "bandwidth_hz")),
                ))

        # WHERE THE FRONT END SITS *ON THIS BRANCH ONLY* -- the branch that composes its
        # own `serial_stages` for a corpus source. SCOPE, because it has been misread
        # (hostile round 13, refuted by measurement 2026-09-25): the frequency-domain
        # presets (Thrusts 1-4) never reach this line. They leave `serial_stages=None`
        # and `Simulation._build_spine` builds `composition="full"` for them, so their
        # front end is a `FrontEndBlock` on the beat record. What this variable decides
        # is the order for a corpus whose frames were WRITTEN with one of the two orders.
        #
        # On a replay the choice is not this module's to make: it is a recorded fact
        # about the frames (`_stored_frame_composition`). Under the FULL composition the front end
        # acts on the SAMPLED BEAT RECORD after the mixing block, `sqrt(P_tx)` moves to
        # the source and thermal noise is injected exactly once; under
        # "legacy_impulse" it acts on `ifft(CFR)` before it, which is how every corpus
        # generated before 2026-09-24 was written. Getting this wrong is not a subtle
        # difference: it is what the live-vs-stored gate reads as a non-zero code
        # difference no knob on screen explains.
        chain_composition = ("legacy_impulse" if not corpus_live_cfr
                             else _stored_frame_composition(
                                 getattr(environment_block, "_files", [])))
        legacy_placement = chain_composition != "full"
        if not legacy_placement:
            try:
                from e2e.chain.frontend import FrontEndBlock
                from e2e.chain.link_budget import TxPowerStage
            except ImportError as e:
                raise PipelineError(
                    "Could not import the beat-placement front end (e2e.chain.frontend "
                    "/ e2e.chain.link_budget). Underlying error: " + str(e)
                )
            if _enabled(state, "thermal_noise"):
                # sqrt(P_tx) at the SOURCE, so receiver noise cannot scale with
                # transmit power (contract section 1.4, the coupling F81 is about).
                spine_stages.append(TxPowerStage(adc_cfg))
        if legacy_placement and circuit_block is not None:
            spine_stages.append(CircuitStage(circuit_block))
        if interconnect_block is not None:
            spine_stages.append(InterconnectStage(interconnect_block))
        spine_stages.append(DechirpBlock(adc_cfg))
        if not legacy_placement and circuit_block is not None:
            # Every knob carried across rather than restated -- including a
            # hand-edited per-element config table (`FrontEndBlock.from_rffe`).
            front_end_block = FrontEndBlock.from_rffe(circuit_block, adc_cfg)
            spine_stages.append(front_end_block)
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
            # mode="once" under the FULL composition: it adds NOTHING when the front
            # end already injected (it becomes the provenance record) and IS the one
            # injection when no front end ran. "legacy" is the mode every stored corpus
            # was generated with, and the gate reads zero codes only against it.
            thermal_block = ThermalNoiseBlock(
                adc_cfg, seed=int(_p(state, "thermal_noise", "seed")),
                mode=("legacy" if legacy_placement else "once"))
            spine_stages.append(thermal_block)

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
            spine_stages.append(impairment_block)

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
            spine_stages.append(if_hpf_block)

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
            spine_stages.append(quantizer_block)
            # SAID ON THE SCREEN, from what the run is about to do rather than from a
            # preset's prose (hostile round 12, item 15b). Automatic gain is why a
            # 3-bit converter still produces a picture at all, and until now it was
            # disclosed only in the parameter editor's help text on the other tab.
            _bits = int(_p(state, "quantizer", "bits"))
            if quant_full_scale <= 0.0:
                run_notes.append(
                    f"ADC full scale: PER-FRAME AUTOMATIC GAIN -- set from each "
                    f"frame's own peak with 6 dB of headroom (the 'Full-scale "
                    f"amplitude 0' setting the corpus generator uses), so the "
                    f"{_bits}-bit converter's codes span the frame in front of it. "
                    f"A fixed full scale would clip or starve it instead.")
            else:
                run_notes.append(
                    f"ADC full scale: FIXED at {quant_full_scale:g} (no automatic "
                    f"gain), {_bits} bits.")

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
                [front_end_block or circuit_block, thermal_block, impairment_block],
                impairment_block,
                if_hpf_block=if_hpf_block, chain_flags=chain_flags)
            spine_stages.insert(0, meta_stage)

        # The frequency-domain products cannot consume a frame that has crossed into
        # the beat record, so on this chain they produce nothing (said in the run notes
        # above); the beat/cube products tap the chain instead, in the shared section
        # below.
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
            sink_block = SinkBlock(sink_dir, tag="webapp", domain=DOMAIN_RX_TIME)
            # A tap on the beat record, at the point the corpus's own generator
            # persists (`e2e.ml.chain_generate`: the sink is a serial stage between the
            # quantiser and the range transform, because what a corpus sample STORES is
            # the digitised beat record and the range transform crosses out of that
            # domain).
            spine_stages.append(sink_block)
            product_taps.append(sink_block)

    # --- the beat-record and cube products: ORDERED TAPS on the one chain ----------
    # Shared by the live dechirp chain and the stored-ADC replay. Order is the
    # contract's (section 1.2): the scored detectors read `adc` and therefore tap
    # BEFORE the range transform; `RadarCubeBlock` reads `cube` and taps AFTER it.
    if rx_cfg is not None:
        if _enabled(state, "detector"):
            if rx_grid is None:
                raise PipelineError("The Detector needs the label grid (e2e.ml.labels), "
                                    "which could not be imported.")
            detector_block, detector_info = _build_detector(state, rx_cfg, rx_grid)
            spine_stages.append(detector_block)
            product_taps.append(detector_block)
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

        # The live-vs-stored correctness gate (see `_StoredADCGateBlock`) reads the
        # DIGITISED BEAT RECORD, so it taps here, beside the detectors, and not after
        # the range transform where `adc` no longer exists. It is appended after the
        # real products so it can never make an otherwise-empty run look productive
        # (the "no product enabled" check below excludes it by identity).
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
            spine_stages.append(adc_gate)
            product_taps.append(adc_gate)

        # THE ONE RANGE TRANSFORM on this chain (contract section 1.2 row 11). Always
        # present, like the mixing block: it is what makes the chain one chain, and
        # `range_transform_for` is the single constructor that pins the SCORED protocol
        # (`transforms.RD_RANGE_PROTOCOL`: hann, DC removal, uncropped) the cube product
        # checks by name -- which is what keeps Thrust 5's live-vs-stored gate reading
        # max |diff| = 0 codes on the beat record while the cube stays on the protocol
        # the corpora were scored with.
        try:
            from e2e.chain.transforms import range_transform_for
        except ImportError as e:
            raise PipelineError(
                "Could not import the range transform (e2e.chain.transforms). "
                "Underlying error: " + str(e)
            )
        spine_stages.append(range_transform_for(rx_cfg))
        if _enabled(state, "radar_cube"):
            try:
                from e2e.chain.receive import RadarCubeBlock
            except ImportError as e:
                raise PipelineError(
                    "Could not import the radar-cube product (e2e.chain.receive). "
                    "Underlying error: " + str(e)
                )
            cube_block = RadarCubeBlock(rx_cfg)
            spine_stages.append(cube_block)
            product_taps.append(cube_block)
    elif _enabled(state, "radar_cube") or _enabled(state, "detector"):
        raise PipelineError(
            "The Radar Cube and Detector products consume a digitized ADC cube. Enable "
            "the Dechirp chain (with the RT Environment source) or the Corpus Replay "
            "source to produce one."
        )

    # --- the comms head: a TAP on the one chain, not a rival pipeline -------------
    # It reads the received channel frequency response, so it taps the chain BEFORE the
    # mixing block (the `IC -> comms head` edge on the contract's one diagram) and emits
    # only `comm_*` keys -- it changes nothing the sensing products read. That is what
    # makes it a tap rather than a branch, and it is why the old refusal here ("the
    # Comms Head and the ADC-cube chain ... cannot run together") is deleted: with the
    # products as ordered taps there is no second pipeline for it to be exclusive with.
    # On a chain whose mixing block is a dechirp the head is still its own noise source
    # (a beat-placement front end acts on a tensor that does not exist at the tap --
    # F98), which `ModemBlock` states for itself.
    comms_combining = None
    comms_head = None
    if _enabled(state, "comms"):
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
        comms_head_blocks = [modem_block, BERBlock()]
        if spine_stages is None:
            # The frequency-domain chain: `Simulation` builds the spine, so the head is
            # handed to it as `comms_head=` and inserted at the mixing block's input.
            comms_head = comms_head_blocks
        else:
            # This module composed the spine itself. The head goes in at the same
            # place -- ahead of the mixing block, while a channel response still
            # exists -- which is the front of the receive segment the meta stage
            # excepted.
            insert_at = 1 if meta_stage is not None else 0
            for offset, block in enumerate(comms_head_blocks):
                spine_stages.insert(insert_at + offset, block)
            product_taps.extend(comms_head_blocks)

    # Enabling the mixing block alone (no Radar Cube / Neural Detector / Frame Sink and
    # no Comms Head) would otherwise run to completion with NO PRODUCT at all --
    # Simulation.run happily produces zero outputs, which the Results tab renders
    # identically to "never ran" ("No results yet"). Fail loudly instead, before
    # sim.run() below. The live-vs-stored gate is excluded by identity: it is a
    # correctness check riding along on a run, never the reason a run happened.
    if spine_stages is not None:
        real_products = [t for t in product_taps if t is not adc_gate]
        if not real_products and not downstream_blocks:
            raise PipelineError(
                "The receive chain is enabled but no product that consumes it (Radar "
                "Cube / Detector / Frame Sink / Comms Head) is enabled -- enable one, "
                "or disable Dechirp to run the frequency-domain products."
            )

    # The FULL composition's front end lives on the beat record, so a frequency-domain
    # run has to be able to NAME a beat sample rate (see `_frequency_chain_radar_cfg`).
    # On the receive chain this module composed itself, the cfg is the one its own
    # dechirp/quantiser were built from.
    frequency_chain_cfg = (_frequency_chain_radar_cfg(state, environment_block,
                                                      array_shape, run_notes)
                           if spine_stages is None else rx_cfg)

    sim = Simulation(
        environment_block,
        downstream_blocks,
        k,
        circuit_block,
        interconnect_block,
        afe_block,
        subspace_block,
        array_shape=array_shape,
        serial_stages=spine_stages,
        # The products that tap the chain mid-way (the scored detectors on the beat
        # record, the radar cube after the range transform, the comms head at the
        # mixing block's input). Only meaningful with `serial_stages=`; when
        # `Simulation` builds its own spine it registers the head itself.
        product_taps=(product_taps if spine_stages is not None else None),
        comms_head=comms_head,
        # The beat plan the FULL composition's front end references its noise band to.
        # DERIVED from the frame's own frequency grid, never typed -- see
        # `_frequency_chain_radar_cfg`.
        radar_cfg=frequency_chain_cfg,
        # "cold" leaves Oja's random basis untouched -- the honest acquisition run
        # (Thrust 3 Demo B). The registry default "warm" keeps the historical numbers.
        warm_start=(_p(state, "subspace", "warm_start") != "cold"),
        # STATED, not inherited. `Simulation`'s default already is "full", and on the
        # frequency-domain path (Thrusts 1-4, `serial_stages=None`) that default is what
        # builds the spine -- dechirp -> FrontEndBlock on the beat record -> range
        # transform. Passing it explicitly is the difference between a placement this
        # module chose and one it happened to inherit: a hostile round read the corpus
        # branch's `chain_composition` above and concluded T1-T4 ran the v1.0 impulse
        # order, which measurement refuted (2026-09-25, all eight presets). Ignored when
        # `serial_stages=` is passed, which is why the record below is read off the
        # BUILT list rather than off this argument.
        composition="full",
    )

    # WHAT THIS RUN ACTUALLY COMPOSED. Measured off the stage list that was built, so
    # the note below and the placement test cannot drift from whichever of the three
    # spine-composing branches above ran (see `_composition_record`).
    composition_record = _composition_record(sim.serial_stages)

    # THE DRIVE IS A DISPLAY CHOICE, and the screen says so in its own words rather than
    # a card quoting a level (seat's read of the 2026-09-24 renders, item 1d).
    #
    # EMITTED HERE, on every chain that HAS a front end (hostile round 13, round-12 item
    # 15a): it used to be emitted inside the branch that composes its own `serial_stages`,
    # i.e. only on a corpus replay -- so on Thrust 1, the screen whose entire subject is
    # that drive, it never appeared at all, and the disclosure existed only in the runbook
    # and in Details. The arm caption is the headline of the run's first note, which is
    # why the first clause is short and ends at a " -- ": `webapp.app._note_headline` cuts
    # there, and a clause cut mid-sentence would print a truncation mark (acceptance
    # check 12 forbids one in visible text).
    #
    # Measured 2026-09-24 on the Thrust 1 preset, munich Ka: the source reports
    # physical_scale=False, and forcing scale_mode='physical' on it does not make the
    # level physical -- it feeds a unitless channel response to the LNA, and the A/B
    # INVERTS (8 mA minus 0.5 mA = -3.0 to -3.6 dB over the preset's 5 frames, against
    # +11.69 to +11.71 dB at 3e-5 in legacy scale mode). That is why the preset stays in
    # legacy scale mode and why no card may read the drive as an input level.
    if circuit_block is not None and not rffe_physical:
        _drive = float(_p_positive(state, "rffe", "signal_scaling"))
        _src_abs = getattr(environment_block, "physical_scale", None)
        run_notes.append(
            f"Front-end drive {_drive:.3g} is a DISPLAY choice, not a measured input "
            "level -- legacy scale mode renormalises every frame to that mean |signal| "
            "before the circuit cascade, because these frames carry no absolute volts "
            f"(this source reports physical_scale={_src_abs!r}). Forcing 'physical' "
            "would not make the number physical: it would feed a unitless channel "
            "response to the LNA. What the drive decides is where the picture sits "
            "between the front end's own noise floor and its clamp, and every A/B on "
            "this screen is measured at the drive it ran with.")

    run_notes.append(_composition_note(composition_record))

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

    # ONE SYMBOL ON SCREEN. A JSAC frame carries M symbols, and the image products are
    # CHIRP_BROADCAST, so they emit one image per symbol ([M, bins, gates]). The panels
    # draw a 2-D map, so the run shows symbol 0 -- the all-pilot preamble, the parity
    # point -- and the run note above says so. Sliced HERE rather than inside the product
    # so that the products keep emitting what they computed, and so an FMCW run (one
    # slow index, already 2-D) is untouched.
    if wave_spec is not None and wave_spec.sensing:
        shown = _display_symbol_for(wave_spec)
        for key in _SYMBOL_IMAGE_PRODUCTS:
            frames_list = outputs.get(key)
            if not frames_list:
                continue
            outputs[key] = [(f[shown] if getattr(f, "ndim", 2) == 3 else f)
                            for f in frames_list]

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
    # The spine's own range calibration (see `_spine_range_meta`): read off the
    # `RangeTransformBlock` that actually ran, never re-derived here.
    _range_transform = next(
        (st for st in sim.serial_stages
         if type(st).__name__ == "RangeTransformBlock"), None)
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
        # The measured composition of THIS run (front-end placement, who injected the
        # floor, the stage list) -- what the placement test reads and what the run note
        # above states in words.
        "composition": composition_record,
        "placement_caption": _placement_caption(composition_record),
        **_spine_range_meta(environment_block, _range_transform),
    }
    # THE SENSING WAVEFORM'S OWN UNAMBIGUOUS WINDOW, when the run had one.
    #
    # `_spine_range_meta` describes the TRANSFORM (a full-grid N-point FFT over the
    # frame's 5000 frequency samples), which is the same for every waveform class. It is
    # NOT the window the SENSING product is unambiguous over: a `pilots_only` comb of
    # spacing P samples the channel every P-th subcarrier, so its effective spacing is
    # `P * df` and its window is `c / (P * df)` -- `OFDMFrame.sensing_window_m`, the one
    # authority, read here rather than recomputed. Beyond it the same scene is drawn
    # again, P times, inside the transform's own 0-249.8 m half-window: measured on the
    # first JSAC renders (2026-09-24/25), arm B (P = 8, window 62.44 m) drew FOUR
    # copies of the scene and its brightest-return statistic reported an alias
    # (0.0 dB @ 125 m, which is the 62.4 m window's second copy). Stored so
    # `figures_from_outputs` can crop each arm's map to the window its own knob buys.
    if wave_spec is not None and wave_spec.sensing:
        _window = (wave_spec.notes or {}).get("sensing_window_m")
        if _window is not None and float(_window) > 0:
            outputs["_axis_meta"]["sensing_window_m"] = float(_window)
    if wave_spec is not None:
        # THE OTHER HALF OF THE RESOURCE SPLIT (hostile round 13, N9). The knob buys a
        # longer unambiguous window by spending data rate, and only the window reached
        # the screen: the rate lived in a run note (`_waveform_run_notes`) and nowhere a
        # photograph of the two maps could show it, so the A/B looked like a free lunch.
        # Read off the frame the run actually transmitted, never typed.
        _rate = (wave_spec.notes or {}).get("data_rate_bps")
        if _rate is not None and float(_rate) > 0:
            outputs["_axis_meta"]["data_rate_bps"] = float(_rate)
        # ...and WHICH RECEIVER produced these products, so the panels can be named for
        # it instead of for the legacy comms head the diagram draws disabled (N10).
        outputs["_axis_meta"]["waveform_kind"] = str(wave_spec.kind)
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
    # THE RUN'S ADC BIT DEPTH, and the depth the frames were written at when the frames
    # record it (hostile round 14, J1/J4). `figures_from_outputs` needs both to say WHY
    # the Range-Doppler floor above the scoring crop is higher on one arm than the other
    # (re-digitising the same stored frames at fewer bits raises the quantisation floor,
    # which is a different sentence from "this arm's peak was attenuated"), and the
    # gate's caption needs them to name the bit depth as the difference only when it is.
    _frames_bits = None
    if _enabled(state, "quantizer"):
        if meta_stage is not None:
            for _label, _stored, _live in (meta_stage.chain_flags_mismatch or ()):
                if _label == "ADC bit depth":
                    _frames_bits = int(_stored)
        outputs["_axis_meta"]["adc"] = {
            "bits": int(_p(state, "quantizer", "bits")),
            # None when the frames do not record it (every corpus before 2026-09-23) or
            # when it MATCHES this run -- `chain_flags_mismatch` only records diffs, so
            # "no entry" means "same as this run" on a corpus that carries the key.
            "frames_bits": _frames_bits,
        }
    if adc_gate is not None:
        # The same verdict, at ARM-CAPTION length (hostile round 14, J4). The caption
        # carried `_note_headline(note)` -- everything before the note's first " -- " --
        # which is the LSB count and nothing else, so the Thrust 5 screens printed the
        # count with the one word that says whether it is fine ("bit-identical") or the
        # point of the arm ("differs") on the far side of the separator, in Details.
        # Built by the gate from its OWN comparison, so the two cannot drift apart.
        outputs["_axis_meta"]["gate_caption"] = adc_gate.caption(_frames_bits)
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
        # The corpus's OWN carrier, in the run-identity line (hostile round 11, H4):
        # these frames are a 77 GHz trace while every Sionna screen announces
        # "munich (Ka-band, 30 GHz)" in the same 18 px line, and that discrepancy was
        # only disclosed mid-sentence in the 15 px foot note. Read off the corpus's
        # recorded `RadarConfig.f0_hz` -- never typed here, so a re-traced corpus
        # relabels itself. "(legacy)" marks a carrier that is NOT the Ka-band
        # re-founding every other screen runs at (F93, 2026-09-23); a corpus at that
        # carrier prints the band alone.
        _f0 = getattr(corpus_cfg, "f0_hz", None)
        if _f0:
            _ghz = float(_f0) / 1e9
            # ONE ENVIRONMENT-NAME FORMAT (hostile round 12 residue, closed 2026-09-25):
            # this printed "Corpus Replay (30 GHz corpus)" -- the one screen family that
            # never said "Ka", while the four munich screens announced "munich (Ka-band,
            # 30 GHz)" in the same slot of the same line. A reader comparing the two had
            # to know that 30 GHz IS Ka to see they were the same band. Names the band
            # and the carrier, in that order, like every other environment on the page.
            outputs["_axis_meta"]["band"] = (
                f"Ka corpus, {_ghz:g} GHz" if abs(_ghz - KA_BAND_CARRIER_GHZ) < 1.0
                else f"{_ghz:g} GHz corpus (legacy)")
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
    if comms_combining is None and wave_spec is not None and wave_spec.comms:
        # The OFDM/JSAC head is not the legacy `ModemBlock`, so `comms_combining` (read
        # off the "comms" block's params) is None on this path and the BER/EVM caption
        # rendered a literal "? combining" -- read on the first JSAC render, 2026-09-24.
        # The combining a JSAC run used is the waveform class's own.
        comms_combining = getattr(wave_spec.receive_block, "combining", None)
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

def _to_float_array(t) -> np.ndarray:
    """A torch tensor or array-like as a plain float ndarray, on the CPU."""
    t = t.detach().cpu().numpy() if hasattr(t, "detach") else t
    return np.asarray(t, dtype=float)


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


def _matched_detection_indices(dets, gt) -> set:
    """Indices of `dets` that MATCH a ground-truth target, by the scoreboard's matcher.

    Calls `e2e.ml.metrics.match_detections` -- the same function
    `webapp.detector_scoreboard.score_frames` pools per frame and the same one
    `compare_detectors` scores with -- so the glyph on the picture and the TP count in
    the table beneath it can never disagree. Returns an empty set when there is no
    ground truth for this frame (nothing is claimed matched) or when the import fails
    (the panel then draws every detection as unmatched, which is the pre-2026-09-24
    behaviour and visibly conservative rather than quietly wrong).
    """
    if not dets or not gt:
        return set()
    try:
        from e2e.ml.metrics import MatchCriterion, match_detections
    except ImportError:
        return set()
    try:
        matches, _unmatched_det, _unmatched_gt = match_detections(
            list(dets), list(gt), MatchCriterion())
    except Exception:
        return set()
    return {int(d_i) for d_i, _t_i in matches}


def _spine_range_meta(env_block, range_transform):
    """The ONE range calibration, read off the spine that actually ran.

    THIS REPLACES the six helpers deleted below, which were this module's own
    re-derivation of an axis the products no longer produce: they assumed each product
    range-compressed `n_freqs` samples itself, fftshifted, negated and displayed the
    non-negative half. Under the one-chain contract the spine's `RangeTransformBlock` does
    the transform ONCE in front of every product and crops the negative-delay half itself,
    so the old fftshift/negate/crop was applied to an already-cropped cube and the numbers
    on the card ("1.00 m/gate; display 0-125 m of a 250 m window") described a pipeline
    that no longer runs.

    Two things move with it, and both are the owner's calls:
      * the convention is `bistatic_path` (2026-09-24, ballot 2B: "if bistatic, math
        should be correct") -- excess path length c*tau, so every metre DOUBLES against
        the v1.0 c*tau/2 numbers;
      * the grid spacing is the frame's own endpoint-inclusive `(stop-start)/(N-1)`
        (F97d), not `B/N` -- 0.02 %, ~0.45 m at the far end of the munich window.

    Computed by the SAME functions the spine uses (`delta_f_from_freq_plan` /
    `range_axis_m`), so there is one authority for the number and this module reads it.
    Returns {} for a frame with no plan; the caller then falls back to
    `_range_meta_from_grid` or to bin indices, rather than quoting metres off an
    invented grid.
    """
    try:
        from e2e.chain.receive import delta_f_from_freq_plan, range_axis_m
    except ImportError:
        return {}
    plan = getattr(env_block, "freq_plan", None)
    if not plan:
        return {}
    try:
        delta_f = float(delta_f_from_freq_plan(plan))
        n_fft = int(plan["num_freqs"])
    except Exception:
        return {}
    convention = getattr(range_transform, "convention", None) or "bistatic_path"
    cropped = bool(getattr(range_transform, "crop_negative_delay", True))
    n_keep = n_fft // 2 + 1 if cropped else n_fft
    axis = range_axis_m(n_keep, delta_f, n_fft, convention)
    if axis is None or len(axis) < 2:
        return {}
    per_bin = float(axis[1] - axis[0])
    return {
        "range_m_per_bin": per_bin,
        "range_convention": convention,
        "range_cropped": cropped,
        "range_n_fft": n_fft,
        "range_n_bins": int(n_keep),
        "range_delta_f_hz": delta_f,
        # The FULL unambiguous window in this convention (N bins of the uncropped
        # transform) and the part of it the cropped cube actually shows.
        "range_window_m": float(per_bin * n_fft),
        "range_displayed_m": float(per_bin * (n_keep - 1)),
    }


def _range_meta_from_grid(n_freqs, freq_span_hz, convention="bistatic_path",
                          cropped=True):
    """`_spine_range_meta`'s numbers from a bare (band, sample count) pair.

    For a caller with no `freq_plan` to read -- a legacy pkl, or a hand-built outputs dict
    in a test. The grid is taken as ENDPOINT-INCLUSIVE, `span/(N-1)`, because that is what
    the generator writes (F97d): two different spacings for one frame is exactly the
    0.02 % error that finding is about.
    """
    try:
        from e2e.chain.receive import range_axis_m
    except ImportError:
        return {}
    try:
        n_fft = int(n_freqs)
        span = float(freq_span_hz)
    except (TypeError, ValueError):
        return {}
    if n_fft < 2 or not span > 0:
        return {}
    delta_f = span / (n_fft - 1)
    n_keep = n_fft // 2 + 1 if cropped else n_fft
    axis = range_axis_m(n_keep, delta_f, n_fft, convention)
    if axis is None or len(axis) < 2:
        return {}
    per_bin = float(axis[1] - axis[0])
    return {
        "range_m_per_bin": per_bin,
        "range_convention": convention,
        "range_cropped": bool(cropped),
        "range_n_fft": n_fft,
        "range_n_bins": int(n_keep),
        "range_delta_f_hz": delta_f,
        "range_window_m": float(per_bin * n_fft),
        "range_displayed_m": float(per_bin * (n_keep - 1)),
    }


def _conform_range_axis(axis, n_rows):
    """Make a display axis exactly `n_rows` long, extending at its own step if short.

    `_power_bin` returns `ceil(n_range / per)` gates, which is `n_bins` only when `per`
    divides evenly -- and a hand-built outputs dict in a test need not match either. The
    axis is uniform by construction, so extending it is arithmetic, not interpolation of
    data. Loud only in the sense that it never silently pairs a 7-gate axis with an
    8-row map: it conforms, which is what every caller then relies on.
    """
    axis = np.asarray(axis, dtype=float)
    n_rows = int(n_rows)
    if axis.size == n_rows:
        return axis
    if axis.size > n_rows:
        return axis[:n_rows]
    step = float(axis[1] - axis[0]) if axis.size > 1 else 1.0
    start = float(axis[0]) if axis.size else 0.0
    return start + step * np.arange(n_rows)


def _display_range_axis(n_bins, rmeta):
    """Display-gate index -> metres, for a product that power-binned the SPINE's cube.

    `e2e.blocks._power_bin` groups `per = ceil(n_range / n_bins)` native bins into each
    gate and pads at the high-index end, and the spine's cube starts at bin 0 = zero
    excess delay with no fftshift -- so the axis is simply `gate * per * m_per_bin`,
    ascending, with no negative half to crop. Returns (axis, metres_per_gate).
    """
    native = int(rmeta["range_n_bins"])
    per = -(-native // max(1, int(n_bins)))
    m_per_gate = per * float(rmeta["range_m_per_bin"])
    n = min(int(n_bins), -(-native // per))
    return np.arange(n) * m_per_gate, m_per_gate


# SUPERSEDED AND DELETED (2026-09-24, one-chain integration): `_nonnegative_range`,
# `_cropped_nonneg_range_axis`, `_range_per_gate_m`, `_native_unambiguous_range_m`,
# `_native_range_resolution_m` and `_range_axis`. Together they were this module's own
# range calibration, written when each product ran its OWN range FFT over `n_freqs`
# samples, fftshifted, negated and displayed the non-negative half. The spine's
# `RangeTransformBlock` now does the transform once in front of every product and crops
# the negative-delay half itself, so all six described a pipeline that no longer runs --
# and `_native_unambiguous_range_m`'s `B/n_freqs` carried the F97d off-by-one besides.
# `_spine_range_meta` / `_range_meta_from_grid` / `_display_range_axis` replace them, and
# the metres come from `e2e.chain.receive.range_axis_m`, which is the one authority.

#: How the y/x axis of a range panel is labelled, per convention. The owner's 2026-09-24
#: ballot answer 2B ("if bistatic, math should be correct") makes `bistatic_path` the
#: default, and the label has to SAY so: the munich link is bistatic (TX at [8.5,21,27],
#: RX array at [45,90,1.5], ~82 m apart) with `normalize_delays=True`, so what the axis
#: measures is EXCESS PATH LENGTH c*tau over the line of sight -- not a monostatic range,
#: and not a distance from the array. An axis labelled just "range (m)" invites both
#: readings, and every metre on it is twice the v1.0 number.
# ONE RANGE VOCABULARY (hostile round 13, N7). Three labels were in use across one
# rehearsal: "excess path (m)" on the munich maps, a bare "range (m)" on the Thrust 5
# corpus panels (which never said which convention, and it is the OTHER one), and
# "excess path (m; 0 = earliest arrival)" on the Thrust 4 profile -- a third definition
# of zero on a screen whose map above it used the first. Both entries now NAME their
# convention, and every panel takes its label from here; where zero means something
# particular (the munich traces are generated with `normalize_delays=True`), that is a
# caption/Details clause, never a second axis label.
_RANGE_AXIS_LABEL = {
    # "excess path" already names the convention: it is c*tau, and it is not a range.
    # Left exactly as it was, because this title also carries the computed
    # "displayed half of 499.6 m" clause and the rotated title's budget is 46 characters
    # against the 363 px plot height (`test_range_az_yaxis_title_names_the_convention_and
    # _the_half_window`) -- adding ", c*tau" here put the real munich title at 49.
    "bistatic_path": "excess path (m)",
    # THIS is the one that said nothing: a bare "range (m)" on the Thrust 5 corpus panels,
    # which are the OTHER convention. It carries no half-window clause, so it has the
    # room to say which.
    "monostatic": "range (m), c*tau/2",
}

#: The one-clause version of the same fact, for a panel caption.
_RANGE_CONVENTION_PHRASE = {
    "bistatic_path": "bistatic excess path (c*tau)",
    "monostatic": "equivalent monostatic range (c*tau/2)",
}

#: Podium-distance legibility floor (fresh-context review, 2026-09-22: every figure's
#: browser-default 12-13 px text reads fine on a laptop and fails at the ~2 m a demo
#: audience actually reads from). Applied, as the LAST step, to every figure this
#: module hands back to the UI via `_make_legible`.
#:
#: RETRACTED, 2026-09-24 (flagged by a test-rewrite review of this same comment): this
#: block used to cite a "20 px standing threshold" and a scoreboard "already at
#: 18-20 px". Neither is current. The layout spec of 2026-09-24 sets the floor at
#: **17 px inside a figure** and 15 px anywhere on a results screen (section 3), the
#: constants below are 18 (one px of margin over that floor), and the scoreboard's own
#: table is now 17 px. The reason the number came DOWN rather than up is the same
#: reason the subtitles left: at 20 px the axis/colour-bar text was itself competing
#: with the picture for a fixed panel. `tests/test_webapp_layout_acceptance.py`
#: (`MIN_FIGURE_FONT_PX`) is the authority; this comment is a pointer, not a copy.
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
#: The panel's own CSS padding, px per side. 8, not 12: acceptance check 5 wants the
#: plot at >= 50 % of the panel's AREA, and at 12 px this measured 47.3 % (rendered,
#: 2026-09-24) -- every px of chrome here is bought straight out of the picture.
PANEL_CSS_PADDING = 8

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
#: 46, down from 68 (hostile round 11, D8): the strip is ONE line now -- headline
#: number bold, per-frame readouts after a separator, one size (`_stat_annotations`)
#: -- instead of a 17 px prose line stacked above a 26 px number. The 22 px that
#: bought goes straight back into the picture.
_FIG_MARGIN_T = 46
_FIG_MARGIN_B = 46
#: 72, not 44 (hostile round 11, D7): `margin.l` is a FLOOR, not the value -- Plotly's
#: auto-expansion then grows it to whatever THIS panel's y-axis ticks happen to need,
#: so the plot origin landed at x = 100 (objectness), 107 (subspace), 112 (maps),
#: 115 (range profile) down ONE column and the left edge visibly staggered. 72 is
#: above the widest of those (71, measured on the rendered page 2026-09-24), so
#: auto-expansion has nothing left to add and every panel of every kind starts at the
#: same x. Raise it, never lower it, if a future axis needs more room.
_FIG_MARGIN_L = 72
_FIG_MARGIN_R = 4

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
#: 19, down from 26 (hostile round 11, D8): the strip is one line now, so the whole
#: statistic -- headline plus per-frame readouts, ~62 characters at its longest --
#: has to cross a 594 px map at ONE size. 18 px keeps that inside the figure width
#: (measured on the rendered page: 8.7 px per character at 17 px, so ~68 characters
#: fit the 658 px between the y-axis and the figure edge) and stays above the 17 px
#: in-figure floor; the
#: headline is separated by WEIGHT (bold) instead of by size.
_STAT_FONT_SIZE = 18
#: Retained name: the per-frame readouts are in the same annotation, at the same
#: size, as the headline now -- see `_stat_annotations`.
_STAT_SUB_FONT_SIZE = _STAT_FONT_SIZE
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
    """The reserved statistic strip above the plot: ONE line, ONE size, headline
    number bold, then the per-frame readouts after a separator --

        **66.0 dB peak−median** · brightest −13.8 dB @ 37 m · frame 4 of 5

    Anchored to the AXES' own domain (`x domain`/`y domain`) at y > 1, so the strip sits
    entirely in the figure's top margin and can never cover a return (hostile round 10,
    defect 6 / acceptance check 8). No background pill: there is nothing underneath it
    to hide any more.

    ONE annotation, not two (hostile round 11, D8): the strip used to stack a 17 px
    prose line ABOVE a 26 px number, re-creating in miniature the inverted hierarchy
    the whole redesign was about -- and it cost the strip 22 px of plot height on
    every map. One line at one weight-graded size does the same job in less space.
    """
    text = f"<b>{stat}</b>" + (f"{CAPTION_SEP}{sub}" if sub else "")
    return [dict(text=text, xref="x domain", yref="y domain", x=0.0, y=1.02,
                 showarrow=False, xanchor="left", yanchor="bottom", align="left",
                 name=_STAT_ANNOTATION_FLAG,
                 font=dict(size=_STAT_FONT_SIZE,
                           color=ARM_COLORS.get(arm, ARM_COLORS["a"])))]


def _frame_tag(i: int, n: int) -> str:
    """Frame `i` (0-based index) of `n`, in the TRANSPORT's own words and numbering:
    "frame 5 of 5". One authority for the phrase, so a statistic strip, a Details line
    and the transport beside them can be compared by eye (hostile round 14, K1/K4:
    Details said "(last frame)", the strip "frame 4 of 5", and the T6 EVM axis counted
    from 0 while the transport counts from 1)."""
    return f"frame {int(i) + 1} of {int(n)}"


def _n_frames_phrase(frames: List[int]) -> str:
    """1-based frame numbers as a short phrase: [3, 4, 5] -> "frames 3-5",
    [1] -> "frame 1", [1, 3] -> "frames 1, 3"."""
    frames = sorted(int(f) for f in frames)
    if len(frames) == 1:
        return f"frame {frames[0]}"
    if frames == list(range(frames[0], frames[-1] + 1)):
        return f"frames {frames[0]}-{frames[-1]}"
    return "frames " + ", ".join(str(f) for f in frames)


#: `fig.layout.meta` key: the panel's headline statistic at the LAST frame, with its
#: frame tag -- what an arm's one-line caption quotes (hostile round 14, J3), read off
#: the figure so the caption and the strip cannot disagree.
_HEADLINE_META = "headline_last_frame"


def _keep_non_stat_annotations(fig) -> List[Dict[str, Any]]:
    """Every annotation on `fig` that is NOT the statistic strip, as plain dicts.

    `fig.update_layout(annotations=...)` REPLACES the list, and `add_hline(...,
    annotation_text=...)` puts its label in that same list -- so setting the statistic
    after drawing a reference line silently deleted the line's label. That is exactly
    how the subspace panels came to carry an unlabelled grey dashed line at 0.06,
    from a DIFFERENT run on the STATIC scene, sitting where arm B lands, with its
    "reference, not this run" caveat reachable only in Details (hostile round 11, H1).
    Callers that set a statistic on a figure with reference lines compose the two
    lists with this."""
    return [a.to_plotly_json() if hasattr(a, "to_plotly_json") else dict(a)
            for a in (fig.layout.annotations or ())
            if not str(getattr(a, "name", "") or "").startswith(_STAT_ANNOTATION_FLAG)]


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
#: `xpad=2`: Plotly's default 10 px pad on each side of the bar is pure reserved
#: width, and the bar is only 14 px wide.
_COLORBAR = dict(thickness=14, len=0.90, xpad=0)

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
#: ...and the same line on a ONE-arm screen, where there is no other arm to name.
_SINGLE_ARM_FLOOR_MARKER = "colour limits, this arm only: "

#: `layout.meta` key holding a y-extent this panel set ON PURPOSE, in the axis's own
#: units, which the cross-arm axis pass must NOT widen. Today exactly one panel sets it:
#: a sensing map cropped to its waveform's unambiguous window (`sensing_window_m`), where
#: the two arms' windows differ BECAUSE of the A/B knob. Sharing that axis would undo the
#: crop on the shorter arm -- the same failure `_union_fixed_range` was written for, one
#: level up.
_Y_EXTENT_LOCK = "y_extent_lock_m"

#: `layout.meta` key holding this panel's own brightest-visible-return range (m), the
#: number its statistic strip prints. Recorded so the cross-arm pass can say where a
#: return one arm shows lands on the other arm's ALIASED axis (round 13, N4: arm A's
#: brightest sat at 72 m and arm B's at 9 m, one 62.4 m window apart, with nothing on
#: either screen saying the second was the first folded).
_BRIGHTEST_M = "brightest_m"

#: The caption clause naming that window, and the Details clause naming the pair when the
#: two arms differ. Prefix-recognised for the same reason the clip clause is.
_WINDOW_CLAUSE_PREFIX = "window "
_WINDOW_DIFFERS_PREFIX = "y-axis NOT shared: "


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
#: The settled tracking floor the Thrust 2/3 cards quote, drawn as the dashed
#: "settled level (reference)" line on every subspace-error panel.
#: RE-MEASURED 2026-09-24 (seat's read of the renders, item 1g), munich Ka, k=2, through
#: this runner: Thrust 2's A arm (AFE mantissa 6) settles at 0.0771-0.0829 over frames
#: 2-6 and Thrust 3's B arm (the shipped adaptive gate) at 0.0785-0.0806 from frame 4 --
#: two independent arms on the same floor. The old 0.06 predates the Ka retrace and the
#: one-chain receiver; it drew the line a third BELOW the curve the cards said settled on
#: it. Re-measure this whenever the frames or the receiver change; it is a measurement,
#: not a constant of nature.
#:
#: TWO SCOPES ON IT, both load-bearing and both easy to lose (2026-09-25):
#:   * it was measured on the STATIC scene (`munich_ka.pkl`). Thrust 3 now runs the
#:     swept-line-of-sight file, where the same two arms hold about 0.55 and 0.26 and
#:     neither comes near this line -- the line is a reference from another scene there,
#:     and every string below says "static scene" so a reader cannot take it for this
#:     run's own floor.
#:   * it is NOT a warm-start number. Both arms it was measured from (Thrust 2's A and
#:     Thrust 3's B) run `warm_start="cold"`; the screen called it a "warm-start settled
#:     level" until 2026-09-25, which was a claim about a start condition the
#:     measurement never had.
_SUBSPACE_ERR_SETTLED_LEVEL = 0.08
#: The static-scene settled level PER FIXED PASS COUNT (shard 3f, 2026-09-25). The
#: 0.08 above is the 10-pass tracker's; the 5-pass tracker settles at about 0.16 on
#: the same static scene (Thrust 3's card and runbook: "A about 0.55, B about 0.26 ...
#: against 0.16 and 0.08 on the static scene"; the 0.16-0.17 reading from frame 5 is
#: wave 12's, 2026-09-24, k=2 cold). Both Thrust 3 arms used to draw 0.08, so arm A's
#: dashed line was the OTHER arm's static level. A run whose passes/frame is flat at
#: a count listed here draws that count's level; anything else keeps 0.08.
_SUBSPACE_ERR_STATIC_LEVEL_BY_PASSES = {5: 0.16, 10: _SUBSPACE_ERR_SETTLED_LEVEL}


def _static_reference_level(n_refine_used) -> float:
    """This run's own static-scene reference: the level for its passes/frame when
    that count is flat over the run and measured, else `_SUBSPACE_ERR_SETTLED_LEVEL`."""
    if n_refine_used:
        counts = {int(n) for n in n_refine_used}
        if len(counts) == 1:
            return _SUBSPACE_ERR_STATIC_LEVEL_BY_PASSES.get(
                counts.pop(), _SUBSPACE_ERR_SETTLED_LEVEL)
    return _SUBSPACE_ERR_SETTLED_LEVEL
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
    set_panel(fig, title="Scenario, plan view",
              caption=["x-y plane", "radar ▲, boresight dashed"],
              details=["Plan view (x-y) of the scenario the Scenario tab holds. "
                       "Radar drawn as a triangle, boresight as a dashed line."],
              row=PANEL_ROW_MAP)
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
    panel = _panel_dict(fig)
    # The arm's OWN clip, before any sharing -- remembered on the panel the first time,
    # because this pass runs again on every Dash render (a tab switch re-fires
    # `_render_results`) and by then the trace already holds the SHARED value. Without
    # this, the second pass silently dropped the "(was X)" provenance the layout spec
    # and hostile round 10 (section 5.3) both say to keep; caught by
    # `test_sharing_is_idempotent`, 2026-09-24.
    own_zmin = panel.get("own_zmin")
    for trace in (fig.get("data") or []):
        if trace.get("type") != "heatmap":
            continue
        if own_zmin is None and trace.get("zmin") is not None:
            own_zmin = float(trace["zmin"])
        trace["zmin"], trace["zmax"] = zmin, zmax
    if own_zmin is not None:
        panel["own_zmin"] = own_zmin

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


def reach_floor_single_arm(figs: Dict[str, Any]) -> None:
    """The SAME colour rule on a one-arm screen as on a two-arm one, in place.

    `share_heatmap_z_limits` puts a `Z_SHARE_REACH_FLOOR` panel's limit 3 dB below the
    deepest per-frame median floor EITHER ARM reaches, because that is what makes an
    11-12 dB difference in background a visible difference in colour (owner, live test
    2026-09-24). With one arm there is no pair, so that pass never ran and the panel
    kept `_radar_cube_clip_db`'s own limit -- which is `max(-40, median + 3)`, i.e. a
    hard -40 dB whenever the floor is below it. Measured on the 2026-09-24 renders: the
    cancel screen clipped at -40.0 dB and rendered a UNIFORMLY DARK map, against -61.3 dB
    and visible structure for the identical product on the two-arm Thrust 1 screen
    (hostile round 12, item 8). Same rule, one arm.

    Only panels that declare `Z_SHARE_REACH_FLOOR`; a `Z_SHARE_KEEP_CLIP` panel keeps
    its own clip by design, and an untagged one (the detector's 0-1 objectness map) is
    not in dB at all.
    """
    for fig in figs.values():
        meta = (fig.get("layout") or {}).get("meta")
        if not isinstance(meta, dict) or meta.get("z_share") != Z_SHARE_REACH_FLOOR:
            continue
        floors = _heatmap_floors_db(fig)
        if not floors:
            continue
        zmin = min(floors) - _SHARED_FLOOR_MARGIN_DB
        own = [float(tr["zmin"]) for tr in (fig.get("data") or [])
               if tr.get("type") == "heatmap" and tr.get("zmin") is not None]
        if own and zmin >= min(own) - 0.05:
            # Already at or below the floor-reaching limit (a frame whose own median
            # sits above -43 dB): leave the tighter, adaptive clip alone.
            continue
        panel = _panel_dict(fig)
        panel.setdefault("own_zmin", own[0] if own else None)
        for trace in (fig.get("data") or []):
            if trace.get("type") == "heatmap":
                trace["zmin"], trace["zmax"] = zmin, 0.0
        caption = [c for c in panel["caption"]
                   if not (isinstance(c, str) and c.startswith(CLIP_CLAUSE_PREFIX))]
        caption.append(f"{CLIP_CLAUSE_PREFIX}{zmin:.1f} dB")
        panel["caption"] = caption
        was = (f" (was {panel['own_zmin']:.1f})"
               if panel.get("own_zmin") is not None else "")
        panel["details"] = [d for d in panel["details"]
                            if not (isinstance(d, str)
                                    and d.startswith(_SINGLE_ARM_FLOOR_MARKER))] + [
            f"{_SINGLE_ARM_FLOOR_MARKER}zmin {zmin:.1f} dB{was}, zmax 0 dB -- "
            f"{_SHARED_FLOOR_MARGIN_DB:.0f} dB below the deepest per-frame median "
            f"floor this run reaches, the same rule a two-arm screen applies across "
            f"both arms."]


def _fig_meta_get(fig, key: str):
    """A raw `layout.meta` value off a `go.Figure` or its stored dict form, or None."""
    if hasattr(fig, "layout"):
        meta = dict(fig.layout.meta) if fig.layout.meta else {}
    else:
        meta = ((fig.get("layout") or {}).get("meta") or {})
    return meta.get(key) if isinstance(meta, dict) else None


def _meta_number(fig, key: str):
    """A numeric `layout.meta` value off a `go.Figure` or the stored dict form, or None.
    Never raises: a missing/garbled provenance value must not take a render down."""
    if hasattr(fig, "layout"):
        meta = dict(fig.layout.meta) if fig.layout.meta else {}
    else:
        meta = ((fig.get("layout") or {}).get("meta") or {})
        if not isinstance(meta, dict):
            meta = {}
    try:
        val = meta.get(key)
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def y_extent_lock_of(fig) -> float:
    """The deliberate y-extent this figure dict/Figure set, or None. See
    `_Y_EXTENT_LOCK`."""
    if hasattr(fig, "layout"):
        meta = dict(fig.layout.meta) if fig.layout.meta else {}
    else:
        meta = ((fig.get("layout") or {}).get("meta") or {})
        if not isinstance(meta, dict):
            meta = {}
    val = meta.get(_Y_EXTENT_LOCK)
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


#: Name of the shape/annotation that marks the other arm's shorter window on a wider
#: arm's map, so re-running the pass replaces it instead of stacking copies.
_OTHER_WINDOW_MARK = "other_arm_window"


def _shade_other_arm_window(fig, other_m: float, other_label: str) -> None:
    """Mark the other arm's (shorter) unambiguous window on THIS arm's map, in place.

    A translucent band from 0 to `other_m` plus one labelled edge line. Added to the
    figure's own layout, which every animation frame inherits (the frames here override
    `annotations` only -- see `_add_frame_animation`'s callers), and REPLACED rather than
    appended on a second pass so an A/B re-render cannot stack two bands.
    """
    layout = fig.get("layout") if isinstance(fig, dict) else None
    if layout is None:
        return
    shapes = [sh for sh in (layout.get("shapes") or [])
              if not (isinstance(sh, dict) and sh.get("name") == _OTHER_WINDOW_MARK)]
    shapes.append(dict(
        type="rect", name=_OTHER_WINDOW_MARK, xref="x domain", yref="y",
        x0=0.0, x1=1.0, y0=0.0, y1=float(other_m), layer="above",
        line=dict(color="#ffffff", width=1, dash="dot"),
        fillcolor="rgba(255,255,255,0.10)",
    ))
    layout["shapes"] = shapes
    mark = dict(
        name=_OTHER_WINDOW_MARK, text=f" arm {other_label}'s window ends {other_m:.0f} m ",
        xref="x domain", yref="y", x=1.0, y=float(other_m), xanchor="right",
        yanchor="bottom", showarrow=False, bgcolor="rgba(45,58,74,0.7)",
        font=dict(size=17, color="#ffffff"),
    )

    def _put(target: Dict[str, Any]) -> None:
        anns = [a for a in (target.get("annotations") or [])
                if not (isinstance(a, dict) and a.get("name") == _OTHER_WINDOW_MARK)]
        anns.append(dict(mark))
        target["annotations"] = anns

    _put(layout)
    # EVERY FRAME TOO. A frame's layout override REPLACES the whole annotations list
    # (that is how the statistic strip re-steps), so a label added only to the base
    # layout vanishes the moment the clock advances one frame -- the same mechanism that
    # once deleted the scoring-crop line's label (see `_keep_non_stat_annotations`).
    for frame in (fig.get("frames") or []):
        if isinstance(frame, dict) and isinstance(frame.get("layout"), dict):
            _put(frame["layout"])


def note_differing_y_extents(figs: Dict[str, Any], prev_figs: Dict[str, Any],
                             labels=("B", "A")) -> None:
    """SAY IT when an A/B pair's two panels do not share a y-axis, in place.

    Two maps side by side are read as one comparison, and the strongest thing a
    photograph of the screen cannot recover is that their axes are different. When both
    arms locked their own y-extent (a sensing map cropped to its own unambiguous window
    -- see `_Y_EXTENT_LOCK`) and the two differ, arm A's caption says so with both
    numbers and both arms' Details repeat the pair. When they are EQUAL, nothing is
    said: the axes then agree and a clause about them would be noise.

    `labels` matches `share_heatmap_z_limits`: (the other arm's name seen from `figs`,
    the other arm's name seen from `prev_figs`).
    """
    for key in set(figs) & set(prev_figs):
        pair = (figs[key], prev_figs[key])
        extents = [y_extent_lock_of(fig) for fig in pair]
        if any(e is None for e in extents):
            continue
        if abs(extents[0] - extents[1]) <= 0.01 * max(extents):
            continue
        # `pair[0]` is arm A by the caller's convention, and `labels[0]` names the arm
        # `prev_figs` holds, so the pair reads (this arm, the other arm) on each side.
        for i, (fig, other) in enumerate(zip(pair, labels)):
            panel = _panel_dict(fig)
            other_fig = pair[1 - i]
            this_m, other_m = extents[i], extents[1 - i]
            line = (f"{_WINDOW_DIFFERS_PREFIX}this arm draws {this_m:.1f} m, arm "
                    f"{other} draws {other_m:.1f} m -- each panel shows its own "
                    f"unambiguous window, so the two maps are not to the same "
                    f"vertical scale.")
            details = [d for d in panel["details"]
                       if not (isinstance(d, str)
                               and d.startswith(_WINDOW_DIFFERS_PREFIX))]
            details.append(line)
            # THE FOLD, NAMED (hostile round 13, N4). On the JSAC screen arm A's
            # brightest return sat at 72 m and arm B's at 9 m, and nothing on either
            # panel said the second was the first wrapped by one 62.4 m window -- the
            # room read a 63 m disagreement between two arms of one A/B. Only stated on
            # the NARROWER arm (the one whose axis actually aliases), and only when the
            # arithmetic checks out to within one display gate, because an unverified
            # "that is an alias" is exactly the kind of claim this repo retracts.
            if this_m < other_m:
                other_bright = _meta_number(other_fig, _BRIGHTEST_M)
                this_bright = _meta_number(fig, _BRIGHTEST_M)
                if other_bright is not None and this_bright is not None:
                    n_wraps = round((other_bright - this_bright) / this_m)
                    folded = other_bright - n_wraps * this_m
                    if n_wraps >= 1 and abs(folded - this_bright) <= 1.5:
                        details.append(
                            f"THE BRIGHTEST RETURN HERE IS AN ALIAS: arm {other}'s "
                            f"brightest sits at {other_bright:.0f} m, and this arm's "
                            f"axis has a period of {this_m:.1f} m, so it folds to "
                            f"{folded:.0f} m -- which is the {this_bright:.0f} m this "
                            f"panel prints. Same return, one window of aliasing apart, "
                            f"not a second scene.")
                        # ON THE VISIBLE CAPTION TOO (hostile round 14, M2): the runbook
                        # told the presenter to point at a fold the screen only carried
                        # in Details. Short -- this caption is one line of ~94
                        # characters and already carries units, rate and clip -- and
                        # made of the same three computed numbers as the sentence above.
                        # TAGGED WITH ITS FRAME (shard 3f, 2026-09-25): the fold is
                        # measured on the last frame, and it sat beside arm A's strip
                        # reading "@ 72 m, frame 4 of 5" -- two frames, one sentence.
                        _ftag = _fig_meta_get(fig, _BRIGHTEST_M + "_frame")
                        # With the tag, the window clause is dropped: tag + window
                        # measured 102 characters on the 2026-09-25 render, past the
                        # ~94 that fit; this arm's window is in its Details.
                        _fold = (f"{other_bright:.0f} m folds to {this_bright:.0f} m "
                                 + (f"({_ftag})" if _ftag
                                    else f"({this_m:.1f} m window)"))
                        panel["caption"] = [
                            c for c in (panel.get("caption") or [])
                            if not (isinstance(c, str) and " folds to " in c)] + [_fold]
            panel["details"] = details
            # NOTHING IS ADDED TO THE CAPTION HERE any more (2026-09-25). Arm A's
            # caption used to be rewritten to carry BOTH windows, which with the
            # burst-rate clause (N9) pushed it past the ~94 characters that fit one line
            # and the browser clipped it mid-word. The cross-arm fact is now DRAWN --
            # `_shade_other_arm_window` shades the other arm's window on this map and
            # labels its edge in the picture's own units -- and each arm's own window is
            # on its axis title. The sentence stays in both arms' Details.
            if this_m > other_m:
                # SHADE THE OTHER ARM'S WINDOW ON THIS MAP (round 13, N9). The two maps
                # render at 4x different vertical scales, so arm B's 62.4 m of scene
                # looked DENSER than arm A's 249.8 m purely by zoom. A band across this
                # map marking where the other arm's axis ends is the one mark that makes
                # the two panels comparable by eye without changing either axis.
                _shade_other_arm_window(fig, other_m, other)


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
                  details=["Integration: (non-coherent over range).",
                           "Non-coherent (power) integration over range, so a target "
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
    # The spine's calibration when the run recorded one; otherwise derived from whatever
    # band/point count the caller does have (a legacy pkl, a hand-built outputs dict).
    _rmeta = meta if meta.get("range_m_per_bin") else (
        _range_meta_from_grid(n_freqs, freq_span_hz) if (n_freqs and freq_span_hz) else None)
    _rmeta = _rmeta or None
    #: The range profile's frames as plain arrays, and how many of its display gates
    #: actually CARRY DATA. Computed here, before the extent below, because both panels
    #: in that column have to be pinned to the same number.
    #:
    #: THE LAST-BIN PLUNGE (hostile round 13, N13), measured on
    #: `thrust4_interconnect_range_profile` at HEAD before writing a word about it: the
    #: profile's last six gates read -120 dB (clamped to the panel's -60 dB floor) beside
    #: a -34 dB median, which read on screen as a ~30 dB cliff at the window edge -- a
    #: physical-looking feature. It is not one. `e2e.blocks._power_bin` groups the cube's
    #: 2500 native range bins into the requested 256 display gates by CEIL division
    #: (per = 10) and zero-pads the remainder, so 2560 - 2500 = 60 padded bins fill the
    #: last 6 gates with exact zeros. Those gates are not scene, and not window edge:
    #: they are arithmetic. Dropped, and the drop is stated in Details.
    _prof_all = [_to_float_array(t) for t in (outputs.get("range_profile_agg") or [])]
    _prof_gates_padded = 0
    if _prof_all:
        _rows = int(_prof_all[0].shape[0])
        #: FULLY-FILLED gates only. Two artefacts, one cause: the all-zero gates at the
        #: end (exact zeros -> the dB floor) and, just before them, ONE gate holding a
        #: single native bin of ten, i.e. a tenth of a gate's power -- measured -43.8 dB
        #: against a -34.0 dB median on the Thrust 4 preset, a 10 dB dip that is the fill
        #: fraction and nothing else. Computed from the cube's own native bin count when
        #: the run recorded one; the exact-zero scan is the fallback for a hand-built
        #: outputs dict that has no `range_n_bins`.
        #: The crop fires ONLY on evidence that this profile carries padded gates -- a
        #: trailing run of exact zeros. Without that test it also cropped a hand-built
        #: `range_profile_agg` (a test fixture, or any caller assembling outputs itself),
        #: where all gates are real and the arithmetic below describes a binning that
        #: never happened: two existing tests caught exactly that, cropping 8 real gates
        #: to 6 and 16 to 11. Padding is the only thing being removed here.
        _last_data = max((int(np.max(np.nonzero(a)[0])) + 1 if np.any(a) else 0)
                         for a in _prof_all)
        _keep = _rows
        if 0 < _last_data < _rows:
            _keep = _last_data
            _native = (_rmeta or {}).get("range_n_bins")
            if _native:
                # ...and the gate just before the zeros is part-filled whenever the
                # native bins do not divide evenly: `_power_bin` groups by CEIL division,
                # so only `native // per` gates hold all `per` bins.
                _per = -(-int(_native) // _rows)
                _keep = min(_keep, max(1, int(_native) // _per))
        if 0 < _keep < _rows:
            _prof_gates_padded = _rows - _keep
            _prof_all = [a[:_keep] for a in _prof_all]
    if _prof_all and _rmeta:
        _bins_rp_for_extent = meta.get("range_profile_bins")
        if _bins_rp_for_extent:
            _axis_for_extent, _ = _display_range_axis(_bins_rp_for_extent, _rmeta)
            # Conformed to the profile's OWN row count, exactly as the profile panel
            # below does it -- otherwise the heatmap is pinned to an extent the profile
            # never draws, which is the mismatch this shared extent exists to remove.
            # The row count is the CROPPED one (see `_prof_gates_padded`): pinning the
            # map to gates the profile no longer draws would re-create that mismatch.
            _prof_rows = int(_prof_all[-1].shape[0])
            _axis_for_extent = _conform_range_axis(_axis_for_extent, _prof_rows)
            if _axis_for_extent.size:
                range_az_yaxis_extent = float(_axis_for_extent.max())

    for key, title, qualifier, aperture_label in [
        ("range_az", "Range-azimuth power", "non-coherent over elevation", "azimuth sin(θ)"),
        ("range_el", "Range-elevation power", "non-coherent over azimuth", "elevation sin(θ)"),
    ]:
        if outputs.get(key):
            bins = meta.get(f"{key}_bins") or outputs[key][-1].shape[0]
            x = _sin_angle_axis(bins)
            if _rmeta:
                # Full-band range compression + power-binning to `bins` gates means
                # the physical range axis is well-defined for any bins (see
                # _range_axis); only needs the frame's band + freq-sample count.
                y, _gate_m = _display_range_axis(bins, _rmeta)
                ylabel = _RANGE_AXIS_LABEL[_rmeta["range_convention"]]
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
                _native_m = float(_rmeta["range_m_per_bin"])
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
                _full_window_m = float(_rmeta["range_window_m"])
                _displayed_m = float(_rmeta["range_displayed_m"])
                # PRECISION IS THE POINT here (hostile round 12, item 13): at `.1f` cm
                # and `.0f` m this clause printed "10.0 cm native ... 0-250 m shown of a
                # 500 m window" -- the exact three nominal-B figures F97d RETRACTED, so
                # a reader who knew the finding read the panel as still carrying the
                # bug. The frame's own endpoint-inclusive grid gives 9.99 cm / 249.8 m /
                # 499.6 m, and two decimals on the centimetre / one on the metre is what
                # it takes to show the difference.
                gate_note = (
                    f"; {_gate_m:.2f} m/gate ({_native_m * 100:.2f} cm native, "
                    f"{_ratio:.0f}:1); 0-{_displayed_m:.1f} m shown of a "
                    f"{_full_window_m:.1f} m window, "
                    f"{_RANGE_CONVENTION_PHRASE[_rmeta['range_convention']]}")
            else:
                # Metadata unavailable (e.g. a hand-built outputs dict): fall back
                # to raw display-gate indices.
                y = np.arange(bins)
                ylabel = "range (bins)"
                earliest_arrival_note = ""
                gate_note = ""
            frames_db = [_to_numpy_abs_db(f) for f in outputs[key]]
            # NO NEGATIVE-DELAY CROP HERE ANY MORE. The spine's range transform already
            # keeps only the non-negative-delay half (`crop_negative_delay`), so the axis
            # this module builds is ascending from 0 and every gate on it is physical. The
            # old fftshift-then-crop was the second half of a calibration the products
            # stopped doing (see `_spine_range_meta`); applying it to an already-cropped
            # cube threw away the far half of the window a second time.
            if y.size and frames_db:
                y = _conform_range_axis(y, frames_db[-1].shape[0])
            # THE SENSING WAVEFORM'S WINDOW, when this run has one that is SHORTER than
            # the transform's own displayed half (see `_axis_meta["sensing_window_m"]`).
            # Everything past it is the same scene drawn again -- measured on the first
            # JSAC screens: arm B (pilot spacing 8, window 62.44 m) drew four copies
            # inside the 249.8 m axis and its brightest-return statistic named an alias
            # at 125 m. Cropping the DATA rather than only the axis range is deliberate:
            # the clip, the peak-median and the brightest return below are then all
            # computed on exactly the pixels that reach the screen. `ceil` keeps the last
            # gate that still STARTS inside the window; at P = 2 the window is the whole
            # transform half-window and the one gate this drops is the wrap of range 0
            # onto its own period (that gate is what printed "brightest -0.0 dB @ 250 m").
            window_m = None
            if _rmeta and y.size and meta.get("sensing_window_m"):
                window_m = float(meta["sensing_window_m"])
                n_keep_gates = max(1, int(np.ceil(window_m / _gate_m - 1e-9)))
                if n_keep_gates < y.size:
                    y = y[:n_keep_gates]
                    frames_db = [f[:n_keep_gates] for f in frames_db]
                else:
                    # The window is at or past the axis this panel could draw anyway;
                    # nothing is cropped and no caption may claim otherwise.
                    window_m = float(min(window_m, float(y.max()) + _gate_m))
            # K3 (hostile round 14): PER ARM, not preset-wide. `gate_note` above is built
            # from the TRANSFORM's geometry, which is identical on both arms -- so the
            # narrower JSAC arm, whose map is cropped to 62.4 m two lines below, printed
            # "0-249.8 m shown of a 499.6 m window" in its own Details. The transform's
            # half-window is still the right denominator (it is what the FFT computed and
            # what a return past the window folds inside), so it stays; what changes is
            # that the "shown" number is now the number this arm's axis actually ends on.
            # Only when this arm's window is actually NARROWER than the transform's
            # half: on the P = 2 arm the two are the same axis and the old sentence is
            # already the true one.
            if (window_m is not None and gate_note
                    and window_m < _displayed_m - 0.5 * _gate_m):
                gate_note = gate_note.replace(
                    f"0-{_displayed_m:.1f} m shown of a {_full_window_m:.1f} m window",
                    f"0-{window_m:.1f} m shown (this arm's own sensing window), of the "
                    f"transform's 0-{_displayed_m:.1f} m half of a "
                    f"{_full_window_m:.1f} m period")
            # Peak-median dB, per frame, on the UNCLIPPED map -- matches
            # notes/tools/demo_thrust1_rescue.py::q, the T1/T2/T4 cards' own
            # dynamic-range definition (Change 2). Computed for range_el too (4th
            # hostile-expert read, 2026-09-23): a screen note about the elevation cut had
            # no number beside it while range_az's identical note did, reading as if only
            # range_az's dynamic range had been checked. AFTER the window crop above, so
            # the number describes the map on screen and not the aliased copies beside it.
            dyn_range_db = [_peak_minus_median_db(d) for d in frames_db]
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
            if _rmeta and y.size:
                _beyond_direct_path = y >= _DIRECT_PATH_EXCLUSION_M
                if window_m is not None:
                    # ON AN ALIASED AXIS, RANGE 0 IS ALSO RANGE W. The window is the
                    # period of the sensing comb, so the direct-path cell this exclusion
                    # exists to skip appears again at the TOP of the map -- measured on
                    # the first cropped render (2026-09-25): arm B's statistic read
                    # "brightest -0.0 dB @ 62 m" on a 62.44 m window, i.e. the same 0 dB
                    # leakage gate the bottom of the map is excluded for. The same
                    # exclusion, wrapped.
                    _beyond_direct_path &= y <= (window_m - _DIRECT_PATH_EXCLUSION_M)
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
                    wrap = ("" if window_m is None else
                            f" and, because range 0 and range {window_m:.1f} m are the "
                            f"same cell on this arm's aliased axis, at the TOP edge too "
                            f"-- both are excluded from the statistic below")
                    return (f"; 0 dB cell at range 0 is one {_gate_m:.2g} m gate "
                            f"(may show as a thin stripe at the bottom edge){wrap}; "
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
            if window_m is None and _rmeta and y.size and ylabel.startswith("excess path"):
                # THE AXIS SAYS IT IS A HALF-WINDOW (hostile round 12, item 13). The
                # transform's period is 499.6 m and this axis draws the non-negative
                # half of it; before this the only place that said so was a closed
                # Details disclosure, so a reader of the SCREEN saw an axis ending at
                # 249.8 m with nothing to say what was past it. Computed from the
                # frame's own plan, never typed.
                fig.update_yaxes(
                    title_text=f"{ylabel}, displayed half of "
                               f"{float(_rmeta['range_window_m']):.1f} m")
            if window_m is not None:
                # THE AXIS SAYS WHAT IT IS. A y-axis reading "excess path (m)" that stops
                # at 62.4 on one arm and 249.8 on the other is the one fact a photograph
                # of a single panel cannot recover, and the panel caption is already at
                # its one-line budget (see below). Rotated, 18 px, ~35 characters against
                # the ~42 the 363 px plot height allows.
                fig.update_yaxes(title_text=f"{ylabel}, {window_m:.1f} m window")
                # This arm draws its OWN window and nothing past it. The axis is pinned
                # to the window rather than to the last gate's centre, so the number in
                # the caption is the number the axis ends on, and `_Y_EXTENT_LOCK` tells
                # `webapp.app._share_y_ranges` not to union it back up to the other arm's
                # (two arms with different pilot spacings have different windows BY
                # CONSTRUCTION -- that is the product the knob buys, and unioning them
                # would put the shorter arm's 62.4 m map back on a 249.8 m axis).
                fig.update_yaxes(range=[0.0, window_m])
                _meta = dict(fig.layout.meta or {}) if fig.layout.meta else {}
                _meta[_Y_EXTENT_LOCK] = float(window_m)
                fig.update_layout(meta=_meta)
            elif key == "range_az" and range_az_yaxis_extent is not None:
                # See `range_az_yaxis_extent`'s definition above the loop.
                fig.update_yaxes(range=[0.0, range_az_yaxis_extent])
            # A SEPARATE STATEMENT, deliberately: written as an `if` inside the chain
            # above it swallowed the `elif` that pins the range-azimuth extent to the
            # profile's (caught by `test_range_az_and_range_profile_share_the_same_extent
            # _when_both_present`, which is exactly why that test exists).
            #
            # THE BRIGHTEST RETURN, as a number rather than as a sentence inside a
            # Details string -- so the cross-arm pass can say what happened to it on an
            # aliased axis (round 13, N4) without parsing prose back out of a caption.
            if direct_path_notes[-1] and "brightest visible return: " in direct_path_notes[-1]:
                try:
                    _tail = direct_path_notes[-1].split(
                        "brightest visible return: ", 1)[1]
                    _bm = float(_tail.split(" at ", 1)[1].split(" m", 1)[0])
                    _meta2 = dict(fig.layout.meta or {}) if fig.layout.meta else {}
                    _meta2[_BRIGHTEST_M] = _bm
                    # WHICH frame that brightest return is from (shard 3f): the last,
                    # in the transport's words, so the fold clause can say so beside
                    # a strip that may be parked on another frame.
                    _meta2[_BRIGHTEST_M + "_frame"] = _frame_tag(
                        len(direct_path_notes) - 1, len(direct_path_notes))
                    fig.update_layout(meta=_meta2)
                except (IndexError, ValueError):
                    pass

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
                    # SHORT (measured on the first render, 2026-09-24): at 17 px the
                    # full phrase plus the frame counter was ~370 px and collided with
                    # the 26 px headline statistic at the other end of the same 52 px
                    # strip on a 604 px plot. The word "visible" and the full phrasing
                    # are in Details; what has to be READABLE here is the number.
                    bright = dp.split("brightest visible return: ", 1)[1].strip()
                    bright = f"brightest {bright.replace(' at ', ' @ ')} · "
                sub_texts.append(f"{bright}{_frame_tag(i, n_frames_key)}")
            fig.update_layout(annotations=_stat_annotations(stat_texts[-1],
                                                            sub_texts[-1]))
            # EVERY STATISTIC IN DETAILS NAMES ITS FRAME, in the transport's own words
            # (hostile round 14, K1). Both the strip and these lines are computed at
            # index -1, so they have always been the same frame -- but the strip printed
            # "frame 5 of 5" while Details said "(last frame)" or nothing at all, and the
            # strip RE-STEPS with the clock. Read on the 2026-09-25 renders with the
            # animation mid-loop, that is a Details line saying "brightest visible
            # return: -14.5 dB at 71 m" beside a strip saying "brightest -14.3 dB @ 74 m"
            # with nothing on either to say they are two different frames. The tag is the
            # same string the strip and the transport use, so the two can be compared
            # instead of guessed at (pinned by
            # `test_details_statistics_name_the_same_frame_the_strip_does`).
            _last_tag = _frame_tag(n_frames_key - 1, n_frames_key)
            _meta_h = dict(fig.layout.meta or {}) if fig.layout.meta else {}
            _meta_h[_HEADLINE_META] = f"{stat_texts[-1]}, {_last_tag}"
            fig.update_layout(meta=_meta_h)
            details = [f"Integration: ({qualifier}).",
                       f"peak - median, dB: {dyn_range_db[-1]:.1f} ({_last_tag}, the "
                       f"last).",
                       clip_provenance + "; superseded by the shared limits below "
                       "when both arms render this product."]
            if earliest_arrival_note:
                details.append(_sentence(earliest_arrival_note))
            if gate_note:
                details.append(_sentence(gate_note))
            if direct_path_notes[-1]:
                details.append(_sentence(direct_path_notes[-1])
                               + f" ({_last_tag}; the strip above re-steps with the "
                                 f"clock and names the frame it shows.)")
            caption = [_DB_COLORBAR_PREFIX, f"{CLIP_CLAUSE_PREFIX}{clip_db:.1f} dB"]
            if window_m is not None:
                # SHORT, and the number is on the y-axis title as well: this caption is
                # one line in a 746 px panel and the longest one that has ever rendered
                # unclipped there is 94 characters (measured across the rehearsal's own
                # geometry dumps, 2026-09-25). The units clause, the clip clause and the
                # colour-sharing clause already cost 67 of them.
                # THE WINDOW IS ON THE AXIS TITLE, NOT IN THE CAPTION (2026-09-25).
                # Measured on this round's own render: with the burst-rate clause added
                # (N9) arm A's caption ran to 115 characters against the ~94 that fit one
                # line in a 746 px panel, and the browser clipped it mid-word at "same
                # colour scale o..." -- a truncation mark in visible text, which
                # acceptance check 12 forbids. The window is the one clause with a second
                # home already on the panel: `fig.update_yaxes(title_text=...)` above
                # prints "excess path (m), 249.8 m window" on the axis itself, and the
                # OTHER arm's window is drawn ON the map as a shaded band with a labelled
                # edge (`_shade_other_arm_window`). Both arms' Details still spell the
                # pair out in a sentence.
                # N9: both halves of the split on one line. "window 62.4 m" alone said
                # what the knob BUYS and never what it costs; 4 more characters say both.
                _rate_bps = meta.get("data_rate_bps")
                if _rate_bps:
                    caption.append(f"burst {float(_rate_bps) / 1e9:.2f} Gb/s")
                    details.append(
                        f"Burst data rate at this pilot spacing: "
                        f"{float(_rate_bps) / 1e9:.3f} Gb/s uncoded. The knob trades it "
                        f"against the {window_m:.1f} m window above -- a wider comb "
                        f"samples the channel more often (longer window) and carries "
                        f"fewer data subcarriers (lower rate).")
                details.append(
                    f"Unambiguous window of THIS arm's sensing waveform: "
                    f"{window_m:.2f} m of excess path (c / (pilot spacing x subcarrier "
                    f"spacing)). The map is cropped to it: past that gate the same "
                    f"scene repeats, so nothing there would be a new return.")
            set_panel(fig, title=title, caption=caption,
                      details=details, row=PANEL_ROW_MAP)

            # The per-frame layout override REPLACES the whole annotations list when
            # the clock steps, so every frame carries its own copy of the strip.
            frame_layouts = [dict(annotations=_stat_annotations(s, sub))
                             for s, sub in zip(stat_texts, sub_texts)]
            figs[key] = _make_legible(_add_frame_animation(fig, frames_db,
                                                           frame_layouts=frame_layouts))

    if _prof_all:
        prof_all = _prof_all          # already converted and padding-cropped, above
        prof = prof_all[-1]
        bins_rp = meta.get("range_profile_bins") or prof.shape[0]
        if _rmeta:
            x, _ = _display_range_axis(bins_rp, _rmeta)
            # Same "0 = earliest arrival" caveat as the range-azimuth/range-elevation
            # panels above (see that loop's comment) -- this panel only ever runs on
            # the same delay-normalised munich frames, never a corpus-replay frame.
            # N7: the axis title is the SAME string the maps above it use, verbatim.
            # "0 = earliest arrival" is a fact about the frames, not a second axis
            # convention, so it rides in this panel's caption (below) where the maps
            # state it too -- the profile used to print it as a third axis label.
            xlabel = _RANGE_AXIS_LABEL[_rmeta["range_convention"]]
            # Wave 8, W2: state the SAME 0 dB = direct-path caveat the range-azimuth/
            # range-elevation sublines now carry (see that loop) -- this panel's own
            # 0 dB point (range 0) is exactly that leakage band.
            direct_path_note = "; 0 dB = direct path at range 0, not a target"
        else:
            x = np.arange(bins_rp)
            xlabel = "range (bins)"
            direct_path_note = ""
        # EACH FRAME NORMALISED TO ITS OWN PEAK, exactly as the single static frame
        # always was -- this panel's y axis is "dB rel. peak" and the peak is the
        # frame's own.
        def _prof_db(a):
            return 10 * np.log10(a / max(float(a.max()), 1e-12) + 1e-12)

        prof_db_all = [_prof_db(a) for a in prof_all]
        prof_db = prof_db_all[-1]
        x = np.asarray(x)
        # Already the non-negative half -- see the range_az loop's note.
        if x.size:
            x = _conform_range_axis(x, prof_db.shape[0])
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
        # IT ANIMATES NOW (hostile round 12, item 14). It was built from the LAST frame
        # and said so, while the range-azimuth map beside it looped on the clock -- so
        # Thrust 4's two panels printed "frame 2 of 3" and "last frame of 3 (static)"
        # side by side, on one screen, about one run. Every frame of the profile was
        # already computed and thrown away; stepping it on the same clock costs nothing
        # and makes the column one frame again. The median-floor statistic (the number
        # the Thrust 4 card quotes) steps with it.
        prof_floors = [float(np.median(d)) if d.size else float("nan")
                       for d in prof_db_all]
        n_prof = len(prof_db_all)
        prof_stats = [(f"{f:.1f} dB median floor", f"frame {i + 1} of {n_prof}")
                      for i, f in enumerate(prof_floors)]
        fig.update_layout(annotations=_stat_annotations(*prof_stats[-1]))
        fig.update_layout(xaxis_title=xlabel,
                          yaxis_title="power (dB rel. peak)",
                          **_base_layout())
        set_panel(fig, title="Range profile",
                  caption=["power, dB rel. peak", "non-coherent over channels",
                           # N7: the zero, said in the caption -- the same fact the
                           # range-azimuth map above it states, in the same words,
                           # instead of a second axis label that disagreed with it.
                           "0 = earliest arrival, not a target"],
                  details=[
                      "Non-coherent (power) integration over channels.",
                      "median floor, dB rel. peak, per frame: "
                      + ", ".join(f"{f:.1f}" for f in prof_floors)
                      + f" (last frame {floor_db:.1f}).",
                      _sentence(direct_path_note),
                      "Steps with the same clock as the range-azimuth map above it, "
                      "so the two panels in this column always show the same frame; "
                      "each frame is normalised to its own peak.",
                      # N13: say what was dropped, so the shorter axis is not itself a
                      # silent edit. Only when something WAS dropped.
                      (f"{_prof_gates_padded} trailing display gate(s) are not drawn: "
                       f"the cube's native range bins do not divide evenly into the "
                       f"{bins_rp} gates requested, so `_power_bin`'s last groups are "
                       f"part-filled or zero-padded and carry a fraction of a gate's "
                       f"power. On the dB scale that reads as a cliff at the window "
                       f"edge; it is the fill fraction, not the scene."
                       if _prof_gates_padded else ""),
                  ], row=PANEL_ROW_MAP)
        figs["range_profile"] = _make_legible(_add_frame_animation(
            fig, prof_db_all, key="y", trace_type="scatter",
            frame_layouts=[dict(annotations=_stat_annotations(st, sub))
                           for st, sub in prof_stats]))

    rx = meta.get("rx") or {}
    # ONE RANGE CROP FOR THE WHOLE COLUMN (hostile round 14, J1/J5). The scoring crop the
    # objectness panel draws its dashed line at, hoisted above the Range-Doppler branch
    # so both panels in that column can be pinned to the same axis. Read from
    # beat_cfar.json, never typed; `None` (file missing) leaves both panels uncropped
    # rather than inventing a crop.
    scoring_max_r = detector_scoreboard.scoring_max_range_m()
    #: The display margin above the scoring crop, shared by the Range-Doppler and
    #: objectness panels. Not itself a claim about anything -- the crop value drawn and
    #: labelled on both IS one, and only it comes from `scoring_max_r`.
    det_column_max_r = 50.0
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
            # The corpus chain's range axis IS the monostatic one (`RadarConfig.
            # range_resolution_m` is c/(2B)) -- labelled from the one vocabulary above,
            # so this panel and the munich panels cannot disagree about what a metre on
            # a range axis means (N7).
            xlabel, ylabel = ("radial velocity (m/s)",
                              _RANGE_AXIS_LABEL["monostatic"])
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
        # THE HEADLINE IS THE FRAME ON SCREEN, and the run median is the readout beside
        # it. The two swapped places on 2026-09-25 (hostile round 12, item 11), and the
        # history matters because both orders have been defended:
        #
        #   * round 11 (H2) made the headline the RUN MEDIAN, because arm A's per-frame
        #     peak−median went 52.8 dB (frame 4) -> 32.6 dB (frame 5) and a reviewer who
        #     dragged the transport got a different headline than the presenter said.
        #   * round 12 read the result on the screen: bold "36.6 dB peak−median (run
        #     median) · this frame 38.7" -- a 26 px number in the arm's colour, directly
        #     above a map of ONE frame, with the qualifier in 17 px beside it. It is read
        #     as the frame's value, because that is what a statistic over a picture is.
        #
        # The fix keeps both facts and reverses which one is bold: every animated panel
        # on these screens now headlines the frame the clock is parked on (the same rule
        # the peak−median maps, the EVM panel and the scoreboard rows follow), and the
        # run median -- the stable number the presenter quotes -- is on the same strip,
        # one size down, with the full per-frame spread in Details. Both computed here,
        # never typed.
        rd_per_frame = [_peak_minus_median_db(d) for d in rd_frames]
        rd_median = float(np.median(rd_per_frame))
        rd_stats = [f"{v:.1f} dB peak−median" for v in rd_per_frame]
        rd_subs = [f"run median {rd_median:.1f} · frame {i + 1} of {len(rd_frames)}"
                   for i in range(len(rd_frames))]
        fig = _heatmap(first, x=x, y=y, xlabel=xlabel, ylabel=ylabel, zmin=rd_clip,
                       z_share=Z_SHARE_KEEP_CLIP)
        fig.update_layout(annotations=_stat_annotations(rd_stats[-1], rd_subs[-1]))
        # ---- The crop this panel shares with the objectness map below it -------------
        # HOSTILE ROUND 14, J1 (SEVERE) and J5. This panel drew 0-102 m directly above
        # an objectness panel drawn 0-50 m, in ONE column, with no axis tick in common:
        # so the two pictures of the same frame could not be read against each other,
        # and the 40 m line that says where scoring stops existed on only one of them.
        # Worse, the top 60 m of THIS panel is where arm B's re-digitised floor shows
        # (measured below), which is outside the scoring crop entirely -- so the hero
        # panel of the detector thrust argued the opposite of what the screen claims.
        #
        # The AXIS is cropped, not the data: every statistic on this panel (the clip,
        # the per-frame peak-median, the run median) is defined over the whole cube and
        # is quoted on the T5 cards, so cropping the data would silently move a card
        # number. Details says so, and says what the axis now ends at.
        rd_window_m = float(y[-1]) if len(y) else 0.0
        # THE BIN SIZE ON THE AXIS, like the munich maps carry their calibration on
        # theirs (hostile round 14, low item: T5 never stated its range bin). Computed
        # from the run's own RadarConfig; the full sentence is in Details.
        if rx.get("range_resolution_m"):
            fig.update_yaxes(title_text=(
                f"{ylabel}, {float(rx['range_resolution_m']):.2f} m bins"))
        rd_visible_above_crop = 0
        #: 1-based frames on which that band has anything above the clip -- named in
        #: the caption when it is not every frame, so a parked clock on a clean frame
        #: does not contradict the clause.
        rd_frames_above_crop: List[int] = []
        if scoring_max_r is not None and rd_window_m > det_column_max_r:
            fig.update_yaxes(range=[0.0, det_column_max_r])
            fig.add_hline(
                y=scoring_max_r, line_dash="dash", line_color="#ffffff",
                annotation_text=f" scoring ≤ {scoring_max_r:g} m ",
                annotation_position="top right",
                annotation_xshift=-10, annotation_yshift=6,
                annotation_bgcolor="rgba(45,58,74,0.7)",
                annotation_font=dict(size=17, color="#ffffff"),
            )
            # MEASURED, on the pixels that actually reach the screen: how many cells of
            # the drawn window sit ABOVE the scoring line and ABOVE this panel's own
            # display clip, over every frame of this run. Zero on the 12-bit arm; that
            # is what makes the caption clause below an observation rather than a guess.
            _band = (y > float(scoring_max_r)) & (y <= det_column_max_r)
            if _band.any():
                _per_frame = [int((d[_band] > rd_clip).sum()) for d in rd_frames]
                rd_visible_above_crop = max(_per_frame)
                rd_frames_above_crop = [i + 1 for i, c in enumerate(_per_frame) if c]
        # WHY anything is visible up there, when it is -- and only from what this run
        # recorded. MEASURED 2026-09-25 on thrust5_detector_cfar and _raddetnet (same
        # 5 corpus frames, both arms, j1_measure2 in the shard-3e scratchpad): at the 12
        # bits the frames were written at, NO cell above 40 m clears the panel's clip on
        # any frame (0 of 5 frames); re-digitised at 3 bits, 55-1064 cells per frame do
        # (0-345 of them inside the 40-50 m band this panel now draws), spread over up
        # to 41 of 64 Doppler bins and 196 of 311 range rows. So they are the 3-bit
        # converter's own products -- they vanish at 12 bits on the same frames -- and
        # spread like a risen floor rather than a few discrete spurs. When the bit depth
        # is NOT what moved (thrust5_detector_ml turns the IF high-pass corner at 12
        # bits and also clears the clip up there, 169 cells on frame 1: the corner
        # lowers the near-range peak every dB is referenced to), the clause says only
        # what is measurable and does not borrow the other screen's mechanism.
        _adc = meta.get("adc") or {}
        _bits_now, _bits_frames = _adc.get("bits"), _adc.get("frames_bits")
        rd_above_is_quant = bool(
            _bits_now is not None and _bits_frames is not None
            and int(_bits_now) < int(_bits_frames))
        # THE NEAR-RANGE SMEAR INSIDE THE CROP (shard 3f, 2026-09-25). On the 3-bit
        # arm a smear at 0-5 m, -8 to -2 m/s shows inside the scoring crop. MEASURED
        # 2026-09-25 on thrust5_detector_cfar, same 5 frames, same chain, only the bit
        # depth moved (m7_3f.py in the shard-3e scratchpad): cells above the panel's
        # own clip in that box per frame were 0,0,8,0,2 at 12 bits, 0,0,7,0,2 at 6 bits
        # and 4,0,109,56,49 at 3 bits (of 208) -- the 3-bit converter's products, like
        # the band above 40 m. Counted here per run, and only NAMED when the bit depth
        # is what moved (`rd_above_is_quant`).
        rd_near_smear_cells = 0
        if rd_above_is_quant and rx.get("velocity_resolution_mps"):
            _near = np.ix_(y <= 5.0, (x >= -8.0) & (x <= -2.0))
            if _near[0].size and _near[1].size:
                rd_near_smear_cells = max(int((d[_near] > rd_clip).sum())
                                          for d in rd_frames)
        rd_above_clause = ""
        if rd_visible_above_crop:
            _where = f"above {scoring_max_r:g} m"
            if len(rd_frames_above_crop) < len(rd_frames):
                _where += (" (" + _n_frames_phrase(rd_frames_above_crop) + ")")
            if rd_above_is_quant:
                rd_above_clause = (f"{_where}: {int(_bits_now)}-bit quantisation "
                                   f"floor, unscored")
                # The 0-5 m smear (the same converter's products INSIDE the crop) is
                # NOT added here: with the frames qualifier the combined clause runs
                # past the ~86-character line. It is in Details and on the runbook's
                # Say list (shard 3f, 2026-09-25).
            else:
                rd_above_clause = f"{_where}: floor over the clip, unscored"
        # "sparse scene: mostly dark on purpose" ON THE DEFAULT SCREEN (hostile round
        # 11, D9): these two near-empty blue panels are a quarter of the first screen
        # on three of seven presets, and the sentence explaining that the emptiness is
        # deliberate (and the scale honest) was only in Details.
        # The caption budget is ONE 16 px line in a 746 px column (~86 characters) and
        # the sharing pass appends "same colour scale on both arms" to arm A's copy,
        # so the unit clause moves to Details to make room: the colour bar is the
        # only thing this clause was labelling, and it is beside the panel.
        # ONE LINE, ~86 characters in a 746 px column, so the new clause displaces the
        # "sparse scene" one rather than joining it: on the arm where it fires the panel
        # is NOT empty up there, which is the whole point, and "sparse, dark on purpose"
        # beside it would be two half-true sentences. The displaced clause is unchanged
        # in Details (acceptance check 15) and still on the caption of the arm that IS
        # dark, which is where it is true.
        rd_caption = [rd_clip_caption,
                      rd_above_clause or "sparse scene, dark on purpose"]
        set_panel(fig, title="Range-Doppler power",
                  caption=rd_caption,
                  details=[_d for _d in [
                      "Non-coherent (power) integration over channels; the colour "
                      f"scale is {_DB_COLORBAR_PREFIX} of this frame.",
                      # THE GEOMETRY THIS PANEL DRAWS, in the same words the munich
                      # panels use (hostile round 14, low item: "T5 screens state their
                      # range bin size and window like T1-T4"). Computed from the run's
                      # own cube, never typed.
                      (f"{float(rx['range_resolution_m']):.2f} m/gate "
                       f"({len(y)} range bins, 0-{rd_window_m:.1f} m computed), "
                       f"monostatic range c*tau/2"
                       if rx.get("range_resolution_m") and len(y) else ""),
                      # WHAT THE CROP DOES AND DOES NOT DO (J1/J5).
                      (f"The range axis is cropped to 0-{det_column_max_r:g} m so this "
                       f"panel and the objectness map below it share one axis, with the "
                       f"{scoring_max_r:g} m scoring line on both. The AXIS is cropped, "
                       f"not the data: the clip and every peak-median below are computed "
                       f"over the whole {rd_window_m:.1f} m cube."
                       if scoring_max_r is not None and rd_window_m > det_column_max_r
                       else ""),
                      (f"Between {scoring_max_r:g} and {det_column_max_r:g} m this arm "
                       f"draws up to {rd_visible_above_crop} cell(s) above its own "
                       f"{rd_clip:.1f} dB clip, on "
                       f"{_n_frames_phrase(rd_frames_above_crop or [0])}. That band is "
                       f"outside the "
                       f"scoring crop, so no detector metric on this screen sees it."
                       if rd_visible_above_crop else ""),
                      ("Why it is there (measured 2026-09-25 on the corpus these "
                       "presets replay): at the 12 bits the frames were written at, no "
                       "cell above 40 m clears the clip on any of the 5 frames; "
                       "re-digitised at 3 bits, on the SAME frames, 55-1064 cells per "
                       "frame do, over up to 41 of 64 Doppler bins -- the 3-bit "
                       "converter's own quantisation products, spread like a risen "
                       "floor rather than a few discrete spurs."
                       if (rd_visible_above_crop and rd_above_is_quant) else ""),
                      (f"The 0-5 m smear at -8 to -2 m/s is INSIDE the scoring crop: "
                       f"up to {rd_near_smear_cells} cell(s) above the clip on this "
                       f"run. Measured 2026-09-25 on the same 5 frames with only the "
                       f"bit depth moved: 0-8 cells at 12 or 6 bits, 0-109 at 3 bits "
                       f"-- the 3-bit converter's products, not a target."
                       if (rd_near_smear_cells and rd_above_is_quant) else ""),
                      rd_clip_provenance + ": this panel's display clip is a "
                      "deliberate decision about what to hide, so sharing it across "
                      "arms only unifies the two clips instead of pushing the limit "
                      "down to the floor.",
                      "A sparse automotive scene at a ~25 dB clip is mostly flat dark "
                      "blue on purpose; the scale is not stretched to make it look "
                      "busy.",
                      # The spread the headline median hides, stated where the
                      # headline is defined (H2).
                      f"peak−median per frame over this run: "
                      + ", ".join(f"{v:.1f}" for v in rd_per_frame)
                      + f" dB (median {rd_median:.1f}). The strip's bold number is the "
                      "FRAME the clock is parked on; the run median is the readout "
                      "beside it, and it is the number that does not move when the "
                      "transport does.",
                  ] if _d], row=PANEL_ROW_MAP)
        # EACH ARM'S ABSOLUTE PEAK, per frame, in dB of the cube's own power units
        # (item 1, shard 3f, 2026-09-25). Every value on this panel is relative to the
        # frame's own peak, so an A/B knob that lowers the PEAK (thrust5_detector_ml's
        # 25 m IF corner attenuates the near-range return every dB is referenced to)
        # makes the rest of the map read BRIGHTER on that arm. The cross-arm pass in
        # app.py (`note_rd_peak_drop`) reads this to put the drop on arm B's caption.
        def _abs_peak_db(cube):
            if hasattr(cube, "detach"):
                cube = cube.detach().cpu().numpy()
            p = np.mean(np.abs(np.asarray(cube)) ** 2, axis=0)
            return float(10 * np.log10(max(float(p.max()), 1e-30)))
        fig.update_layout(meta={**dict(fig.layout.meta or {}),
                                "rd_peak_abs_db": [_abs_peak_db(c)
                                                   for c in outputs["radar_cube"]]})
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
        det_label = str(det_meta.get("display_label")
                        or det_meta.get("label", "")) if det_meta else ""
        det_dir = str(det_meta.get("label", "")) if det_meta else ""
        det_threshold = float(det_meta.get("threshold", 0.0)) if det_meta else None
        # Name the detector and its operating point ON the panel: the three Thrust 5
        # presets are compared across screens, and their cross counts are set by the
        # threshold as much as by the detector. This panel FOLLOWS THE CLOCK, like
        # every other animated panel on the screen (hostile round 11, H3): it used to
        # be pinned to the last frame while the range-Doppler cube above it looped, so
        # the transport said "frame 4 of 5", this panel said "frame 5 of 5 (last)",
        # and the crosses the room counted were from a different frame than the cube
        # they were counted against. The per-frame objectness, detections and ground
        # truth are all already in `outputs`; nothing new is computed.
        panel_title = f"{title} — {det_label}" if det_label else title

        def _obj_map(d):
            if hasattr(d, "detach"):
                d = d.detach().cpu().numpy()
            return np.asarray(d)[0]                   # [n_range, n_azimuth], in [0, 1]

        obj_frames = [_obj_map(d) for d in outputs[key]]
        obj = obj_frames[-1]
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
        # Hoisted above the Range-Doppler branch (hostile round 14, J1/J5) so BOTH
        # panels in this column pin to the same axis; the read itself is unchanged.
        x = -1.0 + (np.arange(n_a) + 0.5) * 2.0 / n_a  # sin(azimuth) bin centres
        fig = go.Figure(data=go.Heatmap(
            z=obj, x=x, y=y, zmin=0.0, zmax=1.0, colorscale="Viridis",
            colorbar=dict(_COLORBAR), name="objectness",
        ))
        # Decoded detections (filled) and, for a replayed corpus frame, the stored
        # ground truth (hollow) -- drawn at the surface range the metric matches on.
        # Per frame, in the same frame order as the objectness maps above, so the
        # animation steps all three together (H3).
        det_frames = list(outputs.get(key + "s") or [])
        gt_frames = list(outputs.get("gt_detections") or [])

        def _at(frames, i):
            return (frames[i] if i < len(frames) else []) or []

        dets = _at(det_frames, len(obj_frames) - 1)
        n_dets = len(dets)
        gt_now = _at(gt_frames, len(obj_frames) - 1)
        # HITS AND MISSES ARE DIFFERENT GLYPHS (hostile round 11, H8). The room counts
        # crosses; the table counts matches; on `thrust5_detector_ml` arm B the table
        # said "this frame: TP 0" while three crosses sat on the top edge of a
        # ground-truth box, and "inside the box" is not decidable by eye at a marker
        # whose arms reach past the tolerance. So the drawing says which is which.
        #
        # THE SAME MATCHER THE SCOREBOARD SCORES WITH -- `e2e.ml.metrics.
        # match_detections`, via `_matched_detection_indices` -- not a second
        # implementation of the rule. A picture that disagreed with the table beneath it
        # would be worse than the ambiguity it replaced.
        matched_idx = _matched_detection_indices(dets, gt_now)
        hit_pts = [d for i, d in enumerate(dets) if i in matched_idx]
        miss_pts = [d for i, d in enumerate(dets) if i not in matched_idx]
        # ALWAYS added, even when this frame has no detections: the animation
        # addresses traces by INDEX, so a trace that appears only on some frames
        # would shift the ground-truth trace under it.
        fig.add_trace(go.Scatter(
            x=[d[1] for d in miss_pts], y=[d[3] for d in miss_pts], mode="markers",
            name=f"✕ unmatched (n={len(miss_pts)})",
            # 10 px, down from 14: at the shipped panel geometry the tolerance box is
            # ~36 x 27 px, so a 14 px cross with 2 px arms reached to its edge. The
            # marker now sits inside its own tolerance, which is what makes "the centre
            # is what counts" checkable rather than asserted.
            marker=dict(symbol="x", size=10, color="#ff3b3b", line=dict(width=2)),
            text=[f"score {d[2]:.2f}" for d in miss_pts],
        ))
        fig.add_trace(go.Scatter(
            x=[d[1] for d in hit_pts], y=[d[3] for d in hit_pts], mode="markers",
            name=f"◆ matched (n={len(hit_pts)})",
            marker=dict(symbol="diamond", size=11, color="#20bf6b",
                        line=dict(width=1, color="#0b5c34")),
            text=[f"score {d[2]:.2f}" for d in hit_pts],
        ))
        gt = gt_now
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

            def _gt_rects(gt_list):
                """This frame's tolerance boxes, as layout shapes (a frame's layout
                REPLACES the shapes list, so each frame carries its own)."""
                return [dict(type="rect", xref="x", yref="y",
                             x0=d[1] - az_tol, x1=d[1] + az_tol,
                             y0=d[3] - r_tol, y1=d[3] + r_tol,
                             line=dict(color="#ffffff", width=2),
                             fillcolor="rgba(0,0,0,0)")
                        for d in gt_list]
            # The hit RULE moves to the caption (layout spec section 4, "Detector
            # map"): as a legend entry it was a 100-character sentence inside a dark
            # block that took 72 px of the panel.
            hit_rule = (f"hit = cross inside the box (±{r_tol:g} m, "
                        f"±{az_tol:g} sin az); matched ones are drawn as green "
                        f"diamonds, by the scoreboard's own matcher")
            fig.add_trace(go.Scatter(
                x=[d[1] for d in gt], y=[d[3] for d in gt], mode="markers",
                name=f"● ground truth (n={len(gt)})",
                marker=dict(symbol="circle", size=6, color="#ffffff",
                            line=dict(width=1, color="#2d3436")),
            ))
        fig.update_layout(
            xaxis_title="azimuth sin(θ)",
            yaxis_title=_RANGE_AXIS_LABEL["monostatic"],
            # NO legend strip. The spec asked for a 32 px inline line (down from a
            # 72 px dark block), but MEASURED on the rendered page a horizontal legend
            # below the plot costs ~50 px of PLOT height, not 32 px of panel: Plotly
            # takes the legend out of the axis domain via margin auto-expansion, which
            # dropped this panel to 44 % of its own area against the spec's own >= 50 %
            # (acceptance check 5, measured 2026-09-24). The legend said exactly what
            # the statistic strip above the plot now says -- "7 detections, 5 labelled"
            # -- so the counts are kept and the duplicate is what goes. Which glyph is
            # which is a caption clause.
            showlegend=False,
            legend=dict(orientation="h", yanchor="top", y=-0.22, x=0.0,
                        xanchor="left", bgcolor="rgba(0,0,0,0)",
                        font=dict(size=17, color="#2d3a4a")),
            **_base_layout(),
        )
        if scoring_max_r is not None:
            # `det_column_max_r` is the fixed display margin above the scoring crop
            # (not itself a claim about anything), shared with the Range-Doppler panel
            # above -- one axis for the column (J5); the crop value drawn/labelled below
            # IS a claim, so only it comes from `scoring_max_r`.
            fig.update_yaxes(range=[0.0, det_column_max_r])
            fig.add_hline(
                y=scoring_max_r, line_dash="dash", line_color="#ffffff",
                # A short TAG at the right end of the line, not a centred white
                # sentence across the middle of the data (layout spec section 4).
                annotation_text=f" scoring ≤ {scoring_max_r:g} m ",
                annotation_position="top right",
                annotation_xshift=-10, annotation_yshift=6,
                annotation_bgcolor="rgba(45,58,74,0.7)",
                # 17 px, the in-figure floor (layout spec section 3); at 16 it was
                # the one element on the Thrust 5 screens under it (measured,
                # 2026-09-24).
                annotation_font=dict(size=17, color="#ffffff"),
            )
        thr_txt = "n/a" if det_threshold is None else f"{det_threshold:.2f}"
        # Everything on this figure that is NOT per-frame: the scoring-crop line and
        # its tag. A frame's layout REPLACES both lists, so each frame below is built
        # as "these, plus this frame's own boxes / statistic".
        _static_shapes = [sh.to_plotly_json() if hasattr(sh, "to_plotly_json")
                          else dict(sh) for sh in (fig.layout.shapes or ())]
        _static_anns = _keep_non_stat_annotations(fig)

        def _det_stat(i: int):
            d, g = _at(det_frames, i), _at(gt_frames, i)
            # "1 detections" is a grammar error in 20 px bold on the LEAD screen of the
            # detector thrust (hostile round 12, item 16). It is reachable on every arm
            # -- a single detection on one frame -- so it is pluralised, not hoped away.
            def _n(k: int, word: str) -> str:
                return f"{k} {word}" if k == 1 else f"{k} {word}s"

            # HOW MANY OF THEM MATCHED, on THIS frame (hostile round 13, N3 item 1).
            # The strip used to print only a total ("45 detections, 5 labelled") while
            # the table below printed TP/FP for a DIFFERENT frame, and the room read the
            # two as one contradiction. The per-frame triple has moved into the
            # scoreboard's Details, which makes this strip the one authority for the
            # frame on screen -- so it has to carry the split the glyphs already draw,
            # counted with the scoreboard's own matcher (`_matched_detection_indices`).
            n_hit = len(_matched_detection_indices(d, g))
            # THE SCOPE WORD, IN THE BOLD (hostile round 14, J2). The strip and the
            # scoreboard beside it were both already correct -- this strip is one frame,
            # the table's visible rows are the run ("cumulative hits", "unmatched /
            # frame, these N frames", "recall ... this run") -- and read together on the
            # ML screen they still land as "11 detections, 0 matched" beside "recall
            # 0.43", which is a contradiction until you know which is which. The sub-line
            # already named the frame; the bold number, which is what the eye reads
            # first, did not say it was about one. Nothing is recomputed: the same counts
            # get the word that scopes them.
            return _stat_annotations(f"this frame: {_n(len(d), 'detection')}, "
                                     f"{n_hit} matched, {len(g)} labelled",
                                     f"frame {i + 1} of {n_frames}")

        if gt:
            fig.update_layout(shapes=_static_shapes + _gt_rects(gt))
        fig.update_layout(annotations=_static_anns + _det_stat(n_frames - 1))
        # THE HIT RULE IS IN DETAILS, and the caption says where (hostile round 12,
        # item 10; spec check 12). It is a 100-character sentence and the caption is
        # ONE line in a 746 px panel -- measured on the 2026-09-24 and 2026-09-25
        # renders, all six objectness panels printed it truncated at "...(±2 m, ±0.06
        # sin a...", i.e. the panel's only statement of what counts as a hit ended
        # mid-symbol. A pointer that fits beats a rule that does not.
        # EVERY MARK ON THE PANEL IS NAMED (hostile round 13, N5). The plot draws FOUR:
        # a red cross, a green diamond, a small white dot and a white rectangle -- and
        # the caption named two of them, calling the rectangle a circle ("○
        # ground-truth boxes"). A viewer counting green diamonds against a caption that
        # does not mention them is reading an unlabelled picture. Kept to one line by
        # folding the hit-rule pointer into the same clause it belongs to.
        set_panel(fig, title=panel_title,
                  caption=[f"objectness ≥ {thr_txt}",
                           "✕ unmatched · ◆ matched · • label in its white tolerance "
                           "box" + (" · hit rule in Details" if hit_rule else "")],
                  details=[
                      f"detections at objectness >= {thr_txt}; the panel shows the "
                      f"frame the transport is parked on, of {n_frames}.",
                      "This panel steps with the same clock as the range-Doppler "
                      "cube above it, so both always show the SAME frame; the "
                      "scoreboard's \"last frame\" rows do not follow the clock and "
                      "say so.",
                      (f"Ground truth boxes ARE the match tolerance: {hit_rule}."
                       if hit_rule else ""),
                      (f"labels & scoring stop at {scoring_max_r:g} m."
                       if scoring_max_r is not None else ""),
                      # The other name this one checkpoint has, said once, here
                      # (hostile round 12, item 16).
                      (f"Checkpoint directory: {det_dir}; scored on this screen as "
                       f"\"{det_label}\", the name the scoreboard and the PR legend "
                       f"use." if det_dir and det_dir != det_label else ""),
                  ], row=PANEL_ROW_MAP)
        # One frame per stored frame: the objectness map (trace 0), this frame's
        # UNMATCHED detections (trace 1), its MATCHED ones (trace 2), the ground-truth
        # markers (trace 3) and its boxes + statistic (layout). Both detection traces are
        # always added, so the indices are the same on every frame (H3).
        #
        # THE SPLIT HAS TO STEP WITH THE CLOCK. Caught on the 22:2x render: with only
        # traces [0, 1, 2] updated, the ground-truth positions were written into the
        # MATCHED trace, so every screen drew a green diamond on each label and the panel
        # read "7 detections" beside eight diamonds while the table said TP 0. A trace
        # added to this figure without a row here is not a cosmetic omission; it silently
        # feeds one trace's data to another.
        if len(obj_frames) > 1:
            def _frame(i, z):
                d_i = _at(det_frames, i)
                matched_i = _matched_detection_indices(d_i, _at(gt_frames, i))
                miss_i = [d for j, d in enumerate(d_i) if j not in matched_i]
                hit_i = [d for j, d in enumerate(d_i) if j in matched_i]
                data = [{"type": "heatmap", "z": z},
                        {"type": "scatter",
                         "x": [d[1] for d in miss_i], "y": [d[3] for d in miss_i]},
                        {"type": "scatter",
                         "x": [d[1] for d in hit_i], "y": [d[3] for d in hit_i]}]
                traces = [0, 1, 2]
                if gt:
                    # Trace 3 is the ground-truth marker trace, which only exists on a
                    # replayed corpus frame -- it has to step too, or the white dots
                    # stay on the last frame's labels while their boxes move.
                    data.append({"type": "scatter",
                                 "x": [d[1] for d in _at(gt_frames, i)],
                                 "y": [d[3] for d in _at(gt_frames, i)]})
                    traces.append(3)
                return go.Frame(
                    name=str(i), data=data, traces=traces,
                    layout=dict(
                        shapes=_static_shapes + (_gt_rects(_at(gt_frames, i))
                                                 if gt else []),
                        annotations=_static_anns + _det_stat(i)))

            fig.frames = [_frame(i, z) for i, z in enumerate(obj_frames)]
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
        ref_level = _static_reference_level(outputs.get("n_refine_used"))
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
        # The settled level the cards quote, so "is 0.26 good?" has an
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
        # THE LINE CARRIES NO LABEL OF ITS OWN (hostile round 12, item 16). It used
        # to, and two waves were spent moving that label around the plot: "top left"
        # put it beside the first frames, "top right" ran it into the right-hand axis,
        # and a collision test over the first four frames chose between them. On the
        # Thrust 3 arm-B curve -- which SETTLES on this very level -- both ends are on
        # the data, and round 12 read it sitting across the curve at frames 3-4. There
        # is no free position on a line the curve lies along.
        #
        # It does not need one. The same fact is already stated twice on the default
        # screen, in places that cannot collide with data: the caption ("dashed =
        # warm-start settled level (reference run)") and the statistic strip's own
        # sub-line, which prints the level. The plot keeps the dashed line; the words
        # stay off the picture, which is the layout spec's rule (check 8).
        fig.add_hline(y=ref_level, line_dash="dash",
                      line_color="#576574")
        fig.update_layout(
            xaxis_title="frame",
            # Unnormalised: said in the caption; the rotated axis title at 20 px
            # clipped when it carried the word (rehearsal 2026-09-23).
            yaxis_title="subspace error (Frobenius)",
            **_base_layout(),
        )
        # The statistic the presenter reads off this panel: where the curve ended, and
        # against what reference. Reserved strip above the plot, same as every map.
        # COMPOSED with the reference line's own label, never replacing it (H1) --
        # see `_keep_non_stat_annotations`.
        fig.update_layout(annotations=_keep_non_stat_annotations(fig)
                          + _stat_annotations(
                              f"{errs[-1]:.2f} at {_frame_tag(len(errs) - 1, len(errs))}",
                              f"dashed = static-scene reference "
                              f"{ref_level:g}"))
        # SHORT (measured on the rendered page, 2026-09-24): the caption renders on ONE
        # line with no wrap in a 746 px column at 16 px, which is ~86 characters -- the
        # spec's 110-character budget is the hard cap, not the fitting width, and a
        # three-clause caption here was CSS-clipped (acceptance check 12). The clause
        # dropped from the caption ("grows ~sqrt(k), not a fraction") is unchanged in
        # Details below.
        # The caption names EVERY line style on the plot. It used to enumerate two of
        # the three ("solid = error, dotted = passes/frame"), leaving the grey dashed
        # horizontal -- a level from a different, warm-start run, drawn exactly where
        # arm B settles -- as the one unexplained mark on the screen (H1).
        set_panel(fig, title="Subspace error per frame",
                  caption=["Frobenius, unnormalised",
                           "dashed = settled level, static scene (reference)"],
                  details=[
                      # Verbatim from the retired subtitle, lower case and all: the
                      # honesty pin in tests/test_webapp_layout_acceptance.py matches
                      # the clause as it was written, not a re-punctuated version.
                      "unnormalised distance; grows ~sqrt(k), not a fraction.",
                      f"The dashed line is this tracker's settled level on the "
                      f"STATIC scene ({ref_level:g}, reference) at this run's "
                      "passes/frame (0.16 at 5, 0.08 at 10), measured on other runs "
                      "-- NOT this run's own level, and on a swept scene no arm "
                      "reaches it.",
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
                # No legend strip: measured on the rendered page, a horizontal legend
                # below the plot costs ~65 px of PLOT height (Plotly takes it out of
                # the axis domain), which put this panel at 42 % of its own area
                # against the spec's >= 50 % (acceptance check 5, 2026-09-24). Both
                # traces are already named by their own AXIS TITLES -- "subspace error
                # (Frobenius)" on the left, "refinement passes/frame" on the right --
                # and the caption says which line is which, so the legend was spending
                # a sixth of the picture to repeat the axes.
                showlegend=False,
            )
            # The compute-per-frame trace is the Thrust 3 A/B. Say where it ended,
            # beside the error statistic, so the "2x compute" claim has a number on
            # screen that does not require reading the right-hand axis.
            panel = panel_of(fig)
            fig.update_layout(annotations=_keep_non_stat_annotations(fig)
                              + _stat_annotations(
                                  f"{errs[-1]:.2f} at {_frame_tag(len(errs) - 1, len(errs))}",
                                  f"{int(n_refine_used[-1])} refinement passes/frame"))
            # ALL THREE line styles, and only them: with the passes/frame trace on,
            # "Frobenius, unnormalised" moves out of the caption (the y-axis title
            # and Details both still carry it) so the three styles fit one line at
            # 16 px in a 746 px column (H1).
            set_panel(fig, title=panel["title"],
                      caption=["solid = error (left axis)",
                               "dotted = passes/frame",
                               "dashed = static-scene reference"],
                      details=list(panel["details"]) + [
                          "The dotted red trace (right axis) is AdaOjaBlock's own "
                          "effective_n_refine() decision per frame -- the compute "
                          "actually spent, not a preset's prose. Both arms' right "
                          "axes are pinned to one range, so a real 2x reads as a "
                          "height difference.",
                      ], row=panel["row"])
        # THE TRANSPORT (hostile round 11, D10): every other screen has exactly one
        # clock; Thrust 3's had none, because nothing on it carried animation frames.
        # The line now GROWS with the clock -- frame i draws the first i points and
        # prints that frame's own error (and passes/frame) in the statistic strip --
        # so the same single transport drives this screen too, and the acquisition
        # story is something the audience watches happen rather than reads off a
        # finished curve. The reference line's own label is re-attached to every
        # frame: a frame layout REPLACES the annotations list (H1).
        _keep = _keep_non_stat_annotations(fig)
        _n_refine = None
        if n_refine_used and len(n_refine_used) == len(errs):
            _n_refine = [int(n) for n in n_refine_used]
        if len(errs) > 1:
            frames = []
            for i in range(len(errs)):
                data = [{"type": "scatter", "y": errs[:i + 1]}]
                traces = [0]
                sub = f"dashed = static-scene reference {ref_level:g}"
                if _n_refine is not None:
                    data.append({"type": "scatter", "y": _n_refine[:i + 1]})
                    traces.append(1)
                    sub = f"{_n_refine[i]} refinement passes/frame"
                frames.append(go.Frame(
                    name=str(i), data=data, traces=traces,
                    layout=dict(annotations=_keep + _stat_annotations(
                        f"{errs[i]:.2f} at {_frame_tag(i, len(errs))}", sub))))
            fig.frames = frames
        _meta_h = dict(fig.layout.meta or {}) if fig.layout.meta else {}
        _meta_h[_HEADLINE_META] = (f"tracker error {errs[-1]:.2f}, "
                                   f"{_frame_tag(len(errs) - 1, len(errs))}")
        # The passes actually spent per frame, for the arm caption's "did the gate
        # escalate" clause (app.py `_arm_facts`) -- read off the run, never typed.
        if _n_refine is not None:
            _meta_h["n_refine_used"] = list(_n_refine)
        fig.update_layout(meta=_meta_h)
        figs["subspace_err"] = _make_legible(fig)

    # Comms head (opt-in "product" -- see webapp/pipeline_registry.py "comms"):
    # BER/EVM-per-frame lines + a constellation snapshot of the last frame.
    bers = [float(b) for b in (outputs.get("ber") or [])]
    comms_meta = outputs.get("_comms_meta") or {}
    # WHOSE RECEIVER THIS IS (hostile round 13, N10). On a JSAC/OFDM run these products
    # come from the waveform class's own `OFDMReceiveBlock`, and the block the diagram
    # labels "Comms head" is drawn DISABLED -- so three panels titled "Comms head ..."
    # sat on a screen whose diagram said the comms head was off. The title now names the
    # receiver that ran, read off the run's own waveform kind.
    _wave_kind = str(meta.get("waveform_kind") or "")
    _head = ("JSAC receiver" if _wave_kind == "jsac"
             else "OFDM receiver" if _wave_kind == "ofdm" else "Comms head")
    combining = comms_meta.get("combining", "?")
    gains = [float(g) for g in (outputs.get("comm_array_gain_db") or [])
             if g is not None and np.isfinite(float(g))]
    #: BER as a SENTENCE for the EVM panel, set when every frame scored exactly zero.
    #: A line plot of a constant zero is not a plot: clamped to a display floor on a log
    #: axis (what this module did before) Plotly rendered an empty grid whose ticks read
    #: 1.0000024 ... 1, with one invisible point -- 460 px of page spent on nothing,
    #: read on the first JSAC render (2026-09-24). The number moves to the panel beside
    #: it, where a photograph can read it, and the row disappears.
    ber_zero_sentence = None
    if bers and max(bers) <= 0.0:
        n_bits = None
        tx_bits = outputs.get("comm_tx_bits")
        if tx_bits:
            try:
                n_bits = int(np.asarray(_to_numpy_complex(tx_bits[-1])).size)
            except Exception:
                n_bits = None
        # SHORT, because this rides in a one-line caption that also has to carry the
        # panel's own units clause and its "does not step with the clock" clause: the
        # long form ("... on every frame (5 frames, 15000 bits each)") left the caption
        # at ~150 characters against the layout spec's 110. Same two facts.
        ber_zero_sentence = (
            "BER 0.0 on every frame (%d%s)"
            % (len(bers), (" x %d bits" % n_bits) if n_bits else " frames"))
    if bers and ber_zero_sentence is None:
        # Combining and array gain are CAPTION clauses now, not a parenthesised title
        # (layout spec section 2.2: the title is one short line, the caption carries
        # the qualifiers). Same two facts, same computation.
        caption_clauses = [f"{combining} combining"]
        if gains:
            caption_clauses.append(f"array gain {np.mean(gains):.1f} dB")
        # A run with SOME errors still plots on a log axis, and its zero frames still
        # need a floor to be drawn at.
        ber_floor = 1e-6
        plotted = [max(b, ber_floor) for b in bers]
        # 1-based x, the transport's numbering (hostile round 14, K4).
        fig = go.Figure(data=go.Scatter(x=list(range(1, len(plotted) + 1)), y=plotted,
                                        mode="lines+markers"))
        fig.update_layout(
            xaxis_title="Frame",
            yaxis_title="BER",
            yaxis_type="log",
            **_base_layout(),
        )
        fig.update_layout(annotations=_stat_annotations(
            f"BER {bers[-1]:.2e} ({_frame_tag(len(bers) - 1, len(bers))})"))
        set_panel(fig, title=f"{_head} BER", caption=caption_clauses,
                  details=[f"Combining: {combining}."
                           + (f" Array gain {np.mean(gains):.1f} dB." if gains else ""),
                           f"Frames with 0 bit errors are shown at the {ber_floor:g} "
                           "display floor -- a log axis cannot plot an exact zero."],
                  row=PANEL_ROW_MAP)
        if any(b < ber_floor for b in bers):
            # Was an 11 px in-figure annotation at y=1.02, i.e. under the statistic
            # strip AND below the 17 px in-figure floor (layout spec section 3). It is
            # a caption clause now: same statement, legible, and it cannot collide.
            panel = panel_of(fig)
            set_panel(fig, title=panel["title"],
                      caption=list(panel["caption"])
                              + [f"0-error frames drawn at the {ber_floor:g} floor"],
                      details=panel["details"], row=panel["row"])
        figs["ber"] = _make_legible(fig)

    if outputs.get("evm"):
        evms = [float(e) for e in outputs["evm"]]
        # 1-BASED FRAMES, the transport's own numbering (hostile round 14, K4): this
        # axis counted 0-4 while the transport beside it said "frame 5 of 5" and the
        # subspace panels on T2/T3 already counted from 1. Integer ticks only.
        fig = go.Figure(data=go.Scatter(x=list(range(1, len(evms) + 1)), y=evms,
                                        mode="lines+markers"))
        fig.update_layout(xaxis_title="frame", yaxis_title="EVM", **_base_layout())
        fig.update_xaxes(tickmode="linear", tick0=1, dtick=1)
        # THE HEADLINE NAMES ITS FRAME, in the same words the scoreboard rows use
        # ("last frame N/N"), and the caption says the panel is static. This panel
        # draws every frame at once and never steps, while the transport beside it
        # reads "frame 4 of 5" -- a bold "(last frame)" over a static curve beside a
        # clock on another frame is exactly the table-and-picture desync hostile
        # round 12 items 1 and 11 are about. `.2e`, not `.3f`: at this operating point
        # every frame's EVM is ~1e-3 and three decimals printed all five of them as
        # "0.001".
        fig.update_layout(annotations=_stat_annotations(
            f"EVM {evms[-1]:.2e} ({_frame_tag(len(evms) - 1, len(evms))}, the last)"))
        # Both clauses are short because this caption is one line in a 746 px panel:
        # measured on the rehearsal's geometry dump, 97 characters render and 133 clip.
        # "per frame" is already in the panel title.
        evm_caption = ["error vector magnitude", "static: all frames drawn"]
        evm_details = ["Error vector magnitude of the equalized data symbols, "
                       "one point per frame, against the symbols actually "
                       "TRANSMITTED (not a decision-directed reference).",
                       "EVM over the run: %.2e (best frame) to %.2e (worst), "
                       "%.2e at the last frame." % (min(evms), max(evms), evms[-1])]
        if ber_zero_sentence:
            evm_caption.append(ber_zero_sentence)
            evm_details.append(
                ber_zero_sentence + " -- so there is no BER curve to plot: a constant "
                "zero on a log axis is an empty panel, and the number says more here.")
        if combining != "?":
            evm_details.append(
                "Combining: %s." % combining
                + (" Array gain %.1f dB." % np.mean(gains) if gains else ""))
        set_panel(fig, title=f"{_head} EVM per frame", caption=evm_caption,
                  details=evm_details, row=PANEL_ROW_MAP)
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
        # EXPLICIT SYMMETRIC RANGES. With `scaleanchor` and autorange, Plotly rendered
        # the QPSK cloud with its y-axis running 0 to 0.8 -- the LOWER HALF OF THE
        # CONSTELLATION OFF THE PANEL, two of four points visible, on data that is
        # symmetric by construction (read on the first JSAC render, 2026-09-24; the data
        # spans +-0.711 in both I and Q). A constellation missing half its points is not
        # a cosmetic defect: it is the panel's whole content.
        _lim = float(np.max(np.abs(np.concatenate([data_np.real, data_np.imag]))))
        _lim = _lim * 1.25 if np.isfinite(_lim) and _lim > 0 else 1.0
        fig.update_layout(xaxis_title="I", yaxis_title="Q", **_base_layout())
        # `constrain="range"`, NOT "domain" (hostile round 13, N8). With "domain" Plotly
        # honours the 1:1 scale by SHRINKING the plot to a square: measured on the
        # 2026-09-25 render, 362 x 363 px inside a 746 x 540 panel -- 0.33 of the panel
        # against the layout spec's >= 0.5 (acceptance check 5), with ~250 px of white
        # either side. With "range" it honours the same 1:1 scale by WIDENING the I range
        # instead, so the plot fills the panel and the constellation is still
        # undistorted: a circle is still a circle, there is just more empty I either side
        # of the cloud, which is what the axis ticks say.
        fig.update_xaxes(range=[-_lim, _lim], constrain="range")
        fig.update_yaxes(range=[-_lim, _lim], constrain="range")
        # NAMES ITS FRAME, and says it is a snapshot: this panel is a scatter of ONE
        # frame and never steps, while the transport above it reads "frame 4 of 5"
        # (hostile round 12, items 1 and 11 -- a caption that says only "last frame"
        # beside a clock on another frame reads as a contradiction).
        _n_eq = len(outputs["comm_data_eq"])
        _tag = _frame_tag(_n_eq - 1, _n_eq)
        # 1:1 after the ranges are set, so the constraint above has something to act on.
        fig.update_yaxes(scaleanchor="x", scaleratio=1, constrain="range")
        fig = _make_legible(fig)
        # WHY THE PANEL LOOKS LIKE FOUR DOTS (hostile round 14, L1): thousands of
        # symbols sit on four points because the EVM is ~1e-3, so every received symbol
        # lands inside its own marker -- a correct picture that read as an empty one.
        # The clause is MEASURED, not asserted: the largest distance from any received
        # symbol to its nearest ideal point, converted to pixels on this panel's own
        # y-axis (the 1:1 axis whose range is [-_lim, _lim]), against the marker radius.
        _n_sym = int(data_np.size)
        _evm_last = (float(outputs["evm"][-1]) if outputs.get("evm") else None)
        _inside = False
        _max_err_px = None
        try:
            _c = np.asarray(_to_numpy_complex(const[-1] if isinstance(const, list)
                                              else const)).ravel()
            _err = np.min(np.abs(data_np.ravel()[:, None] - _c[None, :]), axis=1)
            _m = fig.layout.margin
            _plot_h = float(fig.layout.height) - float(_m.t or 0) - float(_m.b or 0)
            _max_err_px = float(_err.max()) * _plot_h / (2.0 * _lim)
            _inside = _max_err_px <= float(marker.get("size", 4)) / 2.0
        except Exception:
            _inside = False
        _stat = f"{_tag}: {_n_sym:,} symbols".replace(",", " ")
        if _evm_last is not None:
            _mant, _exp = f"{_evm_last:.1e}".split("e")
            _stat += f", EVM {_mant}e{int(_exp)}"
        caption = [_stat]
        if _inside:
            caption.append("every point inside its marker")
        caption.append("static snapshot")
        details = [f"Last frame ({_tag}), equalized: {_n_sym} received symbols; each "
                   "is coloured by the ideal point it was actually TRANSMITTED as. A "
                   "snapshot of one frame: this panel does not step with the transport."]
        if _max_err_px is not None:
            details.append(
                f"Largest distance from a received symbol to its nearest ideal point: "
                f"{_max_err_px:.2f} px on this panel, against a marker radius of "
                f"{float(marker.get('size', 4)) / 2.0:.1f} px -- "
                + ("so every symbol is drawn inside its own marker and the thousands "
                   "of points read as four dots." if _inside else
                   "so the spread around each point is visible."))
        set_panel(fig, title=f"{_head} constellation", caption=caption,
                  details=details, row=PANEL_ROW_MAP)
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
        margin=dict(l=20, r=20, t=20, b=20), height=FIGURE_HEIGHT[PANEL_ROW_MAP],
        paper_bgcolor=PAPER_BGCOLOR, plot_bgcolor=PLOT_BGCOLOR,
    )
    set_panel(fig, title="No data", caption=[message], details=[message],
              row=PANEL_ROW_MAP)
    return fig
