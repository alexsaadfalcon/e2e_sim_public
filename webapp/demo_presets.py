"""Demo presets: one-click block states for the CogniSense Annual Review (Sept 28-30, 2026).

A preset is a block state plus a frame count, with the words that go with it: what the
operator turns live, what to say, and what NOT to say. The last two are not decoration.
Every preset here was built against notes/DEMO_DEFENSE.md, a hostile review of the five
thrusts that found three of them demonstrate the right effect through the wrong
mechanism; its DO-NOT-SHOW list is reproduced per preset as `do_not_say`, and the
preset's own configuration is chosen so the listed traps are not one click away.

Torch-free and import-cheap: this module only touches the registry, so the app shell can
list presets without the simulator installed. `apply_preset` validates every override
against the registry (unknown block or param, a choice not in the list, a number outside
its declared bounds) and raises, so a preset can never silently fall back to a default
the operator did not intend -- the failure mode the interconnect "case3" alias taught.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from webapp.pipeline_registry import BLOCKS_BY_ID, default_block_state

#: notes/DEMO_DEFENSE.md DO-NOT-SHOW #9: runs longer than ~20 frames. Defined in the
#: registry beside MAX_N_STEPS so the runner's error text and this ceiling agree.
from webapp.pipeline_registry import MAX_PRESET_N_STEPS  # noqa: E402

#: The checkpoint for the ML detector: rd format, test AP 0.127 under the beat_cfar
#: protocol. It PREDATES the pipeline fingerprint (F84) -- it carries no stamp and is
#: trusted because re-scoring it under current code reproduces 0.127 (2026-09-22,
#: `e2e/ml/runs/beat_cfar.json`, arm `fftradnet_rd_b5`). Not tracked by git -- the demo
#: machine needs the file. The rad-format checkpoints are NOT used here: see F84.
ML_CHECKPOINT = "e2e/ml/runs/b5_fftradnet_v3/best.pt"

#: The repo-native architecture that scored 0.476 vs CFAR 0.301 (F85; seed 42,
#: deterministic, fingerprint-clean, recertified). Its recall-0.5 operating point is
#: objectness 0.44 (`beat_cfar.json`). Presented ONLY after F85's verification addendum.
RADDETNET_CHECKPOINT = "e2e/ml/runs/b7_raddetnet/best.pt"
RADDETNET_THRESHOLD = 0.44

#: Classical CA-CFAR's recall-0.5 operating point on the same split (`beat_cfar.json`,
#: `operating_point.score_threshold` 0.661). All three Thrust 5 presets sit at their
#: recall-0.5 points on the 172-frame beat_cfar.json split -- not three arbitrary
#: thresholds -- so the FA/frame numbers quoted from that file (6.2 / 26 / 3.0 per
#: frame) are the comparison. RETRACTED (hostile-expert read, 2026-09-23, item 4):
#: this comment used to say the on-screen CROSS COUNTS (5 live frames) ARE that
#: comparison; they are not -- recall varies frame to frame on 5 frames (arm-A hit
#: rates measured 2026-09-23: CFAR 0.53, RADDetNet 0.33, ML 0.50 -- nowhere near
#: matched), so only the 172-frame rows are comparable across arms.
CFAR_THRESHOLD = 0.66

#: Decode threshold for that checkpoint: its recall-0.5 operating point, objectness
#: 0.22 (`beat_cfar.json`). At the registry default 0.5 the checkpoint draws NO
#: detections on the test frames -- the "demo landmine" in DEMO_DEFENSE.md, measured
#: again on the real frames. Pinned here so the figure is never blank.
ML_THRESHOLD = 0.22

#: The demo corpus that stores the RAY-TRACED CHANNEL beside each frame (50 scenes,
#: seed 4242, splits 40/5/5, every frame with a `.cfr.npy` sidecar -- generated
#: 2026-09-23 by `chain_generate --store-cfr`). Replaying it in the `cfr` domain runs
#: the whole analog/digital chain LIVE from that channel, which is what makes a
#: front-end knob reach the detector at all (owner directive, notes/STATE.md §0.1).
#: The predecessor screens replayed a stored ADC cube and could not do that; the two
#: pre-generated bridge corpora that stood in for a live ADC knob are retired with
#: them (the directories may stay on disk).
DEMO_CFR_CORPUS = "e2e/ml/datasets/b1_demo_cfr/benchmark_v1_D2/manifest.json"


#: Shared Results-tab screen note for the three Thrust 5 detector presets. The opening
#: sentence is the owner's wording (2026-09-23) for the live-chain screens: it states
#: what is stored, what is computed, which corpus the OFFLINE numbers belong to, and
#: that a live count is a demonstration rather than a re-measurement. Reworded tighter
#: (same 2026-09-23, owner course-correction) to make room for the mandatory corpus-band
#: disclosure clause -- these corpora were traced with the `benchmark_v1` RadarConfig
#: preset at 77 GHz, a DIFFERENT band from the munich frames (Ka, 28.5-31.5 GHz) every
#: other thrust's screen shows; a Ka-band regeneration of this corpus is scheduled, so
#: the clause is a disclosure of today's state, not a permanent fact. "{VMAX_CLAUSE}" is
#: filled in (or dropped, if the manifest cannot be read) at render time -- see
#: `_read_corpus_v_max` in webapp/app.py -- so the number is never typed here. It no
#: longer fits one 16 px line on the 1600 px results page and wraps to two; that was
#: read on the rendered PNG and accepted, the content being mandatory. (The banner, the
#: OTHER place the owner asked for this "if there is room": there isn't, safely --
#: `outputs["_axis_meta"]["source"]` (pipeline_runner.py) is pinned byte-for-byte by
#: `tests/test_webapp_live_chain.py`'s `.startswith("Corpus Replay (live chain from
#: stored channel)")` / `"Corpus Replay (ADC replay)"` checks, owned by another coder's
#: shard; inserting the band clause there breaks those on sight. So this screen note is
#: the only place it appears; recorded here rather than silently dropped.)
#: wave 9 (2026-09-24, T5 item 3): "clip follows arm's floor, not the knob" is
#: inserted below the character budget the other two coding waves already found (the
#: ML preset's note has the least headroom, ~42 chars, once the VMAX clause and its
#: own "Loses to CFAR" suffix are in) -- kept terse rather than dropped, so all three
#: screens carry the fact that the display clip is recomputed per arm from its own
#: printed median floor, so a brighter background on one arm is the clip moving, not
#: the knob's mechanism (see also the `say` line on each T5 card for the full version).
#: wave 9 (cross-shard fix, 2026-09-24): moved BEFORE "40 m{VMAX_CLAUSE}" rather than
#: after it -- `tests/test_webapp_ab.py::
#: test_resolve_screen_note_drops_vmax_clause_when_manifest_is_unreadable` pins that
#: the resolved note still ends on "40 m." (the scoring crop) when the VMAX clause
#: drops, with no dangling separator; appending the clip clause after it broke that.
_T5_SCREEN_NOTE = (
    "frames: stored ray-traced channel (b1_demo_cfr); ADC chain LIVE; offline numbers "
    "(beat_cfar.json, b1_bench_v3, 12-bit) are a demo when live, not a re-measurement; "
    "ML leaves training distribution on any knob change; corpus traced at 77 GHz "
    "(legacy preset; Ka-band regeneration scheduled); clip follows arm's floor, not "
    "the knob; 40 m{VMAX_CLAUSE}."
)


#: Thrust 4's A/B runs the LIVE Tessera surrogate at its canonical (arm A) vs. TSV
#: height dropped to the low end of its presented envelope (arm B) -- NOT hand-typed:
#: read off the same ParamSpec the GUI slider uses (webapp/pipeline_registry.py
#: `_tessera_presented_range`), so a re-measured envelope moves both together. Height
#: is OFFLINE the biggest single-knob mover of the range-profile skirt of the five
#: continuous knobs, measured through this same InterconnectBlock(source='tessera') on
#: the real munich Ka band (2026-09-23: baseline skirt -53.90 dB -> height-low
#: -57.43 dB, a 3.53 dB native flat-frame metric move; that is NOT the statistic the
#: rendered range-profile panel itself prints -- see the card's own `blurb`, corrected
#: wave 7 X3, 2026-09-23, after the panel's own "median floor" statistic was quoted as
#: if it were this number). notes/TESSERA_KNOB_MEASUREMENT_2026-09-23.md found the
#: same direction/order-of-magnitude at scale=1.
_TESSERA_HEIGHT_SPEC = next(p for p in BLOCKS_BY_ID["interconnect"].params
                           if p.key == "tessera_height_um")
_TESSERA_ARM_B_HEIGHT_UM = _TESSERA_HEIGHT_SPEC.min
_TESSERA_CANONICAL_HEIGHT_UM = _TESSERA_HEIGHT_SPEC.default
#: Display-rounded copy for card text -- the override itself (`ab=` below) uses the
#: exact `.min`, so validation against the ParamSpec's own bound cannot drift.
_TESSERA_ARM_B_HEIGHT_DISPLAY = round(_TESSERA_ARM_B_HEIGHT_UM, 2)
#: Model-geometry (pre-scale) copy of the canonical height, PRESENTED value * the
#: hardcoded x2 scale factor already stated in prose throughout this preset (wave 7
#: X3, 2026-09-23): the card's Arm A header used to say "h 100 um" -- the MODEL
#: geometry -- while every other reference on the same card (`live_knobs`,
#: `screen_note`) used the PRESENTED 50 um, two numbers for one arm. Every label now
#: uses the presented value; this constant exists so the "= 100 um model geometry"
#: clause is stated from a computed number, once, in the blurb.
_TESSERA_CANONICAL_HEIGHT_MODEL_UM = _TESSERA_CANONICAL_HEIGHT_UM * 2

#: Wave 7 (X4-X8, hostile-expert read, 2026-09-23): the array disclosures the review
#: asked every card that mentions the array to carry, read once here from the munich
#: Ka trace's own stored generation meta (`e2e/environment/sionna_sims/munich_ka.pkl`,
#: F93/F94: `rx_spacing_m` 4.997e-3, `aperture_m` 0.1549, `boresight_offset_deg` 35.0,
#: `scattering_coefficient` 0.4 -- the last flagged in that same meta as an ASSUMPTION,
#: not a measured material property) rather than re-typed per preset. Not read live
#: from the pkl at import: this module is deliberately torch-free and import-cheap
#: (module docstring), and the pkl is ~1.2 GB.
_ARRAY_DISCLOSURE = (
    "array: 32x32 at 5 mm spacing (15.5 cm aperture), Ka-band 28.5-31.5 GHz, "
    "boresight 35 deg off the transmitter, diffuse scattering assumed "
    "(coefficient 0.4)."
)


@dataclass(frozen=True)
class DemoPreset:
    id: str
    label: str
    thrust: int
    n_steps: int
    #: {block_id: {"enabled": bool, "params": {key: value}}} -- only what differs from
    #: `default_block_state()`.
    overrides: Dict[str, Dict[str, Any]]
    #: One paragraph: what this preset shows and how to run it.
    blurb: str
    #: The knob(s) the operator turns live, as (block_id, param_key, "from -> to").
    live_knobs: List[Tuple[str, str, str]] = field(default_factory=list)
    #: What to say, each a quotable sentence with its measured basis.
    say: List[str] = field(default_factory=list)
    #: What NOT to say or show -- from the adversarial review, each item measured.
    do_not_say: List[str] = field(default_factory=list)
    #: One-click A/B comparison (2026-09-22 hostile-expert read: three headline claims
    #: showed no evidence of themselves on a single screen). (block_id, param_key,
    #: value_for_run_B); run A is this preset AS LOADED (its own `overrides`), run B is
    #: the same state with ONLY this one param replaced. `apply_preset(preset, arm="b")`
    #: builds run B's state. None (the default) keeps a preset single-run, unchanged.
    ab: Optional[Tuple[str, str, Any]] = None
    #: Human labels for the two arms, quoted in the Results banner, e.g. "8 mA" / "0.5 mA".
    #: Required whenever `ab` is set (apply_preset does not check this; the webapp does).
    ab_label_a: str = ""
    ab_label_b: str = ""
    #: A visitor photographs the Results-tab card, not the presenter (hostile-expert
    #: third read, 2026-09-23): the operator's card (`say`/`do_not_say`) lives on the
    #: Block Diagram tab and never reaches whoever looks at the screenshot later. One
    #: legible line rendered on the Results tab itself, directly under the run
    #: banner(s), shared once by both A/B panels when this preset's run is on screen
    #: (webapp/app.py `_resolve_screen_note`/`_render_results`). May contain the
    #: literal token "{VMAX_CLAUSE}", filled in (or dropped) at render time from the
    #: corpus manifest -- see `_read_corpus_v_max` in webapp/app.py.
    screen_note: str = ""


def _only_products(*keep: str) -> Dict[str, Dict[str, Any]]:
    """Overrides that disable every product block except `keep`."""
    out: Dict[str, Dict[str, Any]] = {}
    for b in BLOCKS_BY_ID.values():
        if b.category == "product" and b.toggleable:
            out[b.id] = {"enabled": b.id in keep}
    return out


def _merge(*parts: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for part in parts:
        for bid, ov in part.items():
            slot = out.setdefault(bid, {})
            if "enabled" in ov:
                slot["enabled"] = ov["enabled"]
            if ov.get("params"):
                slot.setdefault("params", {}).update(ov["params"])
    return out


#: Block state shared by all three Thrust 5 presets: the demo corpus replayed as its
#: stored channel, through the chain that GENERATED it -- the corpus front end
#: (`rffe`, absolute scale) and its own data-driven interconnect (which is what the
#: 'default' case means on this path -- see that ParamSpec), dechirp,
#: thermal floor, impairments, IF high-pass, ADC. The quantizer's full scale is 0 =
#: automatic gain, the corpus generator's setting; a fixed 1.0 would quantize a
#: physically scaled cube (returns near 1e-7) to exactly zero. With every knob here at
#: the value each frame was generated with, the runner's gate reports max |diff| = 0
#: ADC codes against the stored cube -- measured 2026-09-23, printed on every run.
_T5_LIVE_CHAIN = _merge(
    {"corpus_environment": {"enabled": True, "params": {
        "manifest": DEMO_CFR_CORPUS, "split": "test", "start_frame": 0,
        "domain": "cfr"}}},
    {"rffe": {"enabled": True, "params": {"scale_mode": "auto"}},
     "interconnect": {"enabled": True},
     "dechirp": {"enabled": True, "params": {"preset": "benchmark_v1", "mimo": "tdm"}},
     "thermal_noise": {"enabled": True},
     "impairment": {"enabled": True},
     "if_hpf": {"enabled": True, "params": {"corner_range_m": 1.0, "order": 2}},
     "quantizer": {"enabled": True, "params": {"bits": 12, "full_scale": 0.0}},
     "afe": {"enabled": False}, "subspace": {"enabled": False}},
    _only_products("radar_cube", "detector"),
)


PRESETS: List[DemoPreset] = [
    DemoPreset(
        id="thrust1_circuit_knobs",
        label="Thrust 1 - RF circuit knobs vs image quality",
        thrust=1,
        n_steps=5,
        overrides=_merge(
            {"rffe": {"enabled": True, "params": {
                "scale_mode": "legacy", "signal_scaling": 1e-7,
                "lna_bias_ma": 8.0, "if_bw_mhz": 15.0}}},
            {"interconnect": {"enabled": False}},
            _only_products("range_az"),
        ),
        blurb=("Press Run once: both arms run and appear alike (A, top, 8 mA / B, "
               "bottom, 0.5 mA), each panel printing its own peak-median statistic "
               "(about 66 vs 54 dB) -- the ~12 dB difference is in the statistic, "
               "not the picture. Signal is deliberately set just below the model's "
               "input-referred noise (1e-7 vs 1.36e-7 V) -- a real 1024-element "
               "radar's per-element SNR, recovered by coherent gain. Manual path: "
               "LNA bias is the A/B above; second knob: IF bandwidth 15 -> 50 MHz "
               "(see live_knobs)."),
        live_knobs=[("rffe", "lna_bias_ma", "8 -> 0.5 mA (about -12 dB)"),
                    ("rffe", "if_bw_mhz", "manual second knob: 15 -> 50 MHz and run again "
                                          "(about -5 dB; 1 -> 50 MHz is -13.5 dB here, "
                                          "re-measured 2026-09-23 -- the block panel's "
                                          "17 dB is the front end's own noise-power "
                                          "scaling, a different quantity)")],
        # A/B (Change 1, 2026-09-22 hostile-expert read): as-loaded IS the 8 mA arm;
        # run B drops to 0.5 mA, the direction the card's headline (+12 dB) quotes.
        ab=("rffe", "lna_bias_ma", 0.5),
        ab_label_a="8 mA", ab_label_b="0.5 mA",
        # Wave 7 (X4/X6/X7, 2026-09-23): clip now follows the frame's own median floor
        # (it reads "shared floor" here -- the Ka trace's median sits far below -43 dB,
        # so the adaptive clip has nothing to tighten); gate/unambiguous range are on
        # the panel itself (pipeline_runner.py); the array disclosure is mandatory on
        # any card that mentions the array.
        screen_note=("dB rel. peak; clip follows the frame's median floor + 3 dB, falls "
                     "back to the shared -40 dB below -43 dB (as here, both arms); range "
                     "(m; 0 = earliest arrival; 1.00 m/gate; unambiguous 125 m); both "
                     "panels show the same streaks at the same visible brightness -- "
                     "the floor difference is in the printed peak-median number, not "
                     "the picture; all 1024 elements share one front-end config; "
                     + _ARRAY_DISCLOSURE),
        say=[
            # wave 9 (2026-09-24): "about twelve dB" not "12 dB" -- the screen shows
            # 65.9 vs 54.3 = 11.6 dB, not a clean 12.
            "LNA bias 0.5->8 mA is worth about twelve dB (+-0.6-0.9 dB).",
            "At default signal level (1e-5) these knobs do nothing (0.5 dB, under "
            "the 40 dB floor).",
            "Below ~4 mA the LNA is a LOSS stage (-8.5 dB at 0.5 mA); most of the "
            "twelve dB leaves the attenuator regime (4->8 mA: +1.6 dB).",
            "There is no trade-off today: nothing clips; the IF filter only sets noise "
            "variance (1 MHz = 1 ms sweep vs 20 us at 50 MHz).",
            # wave 9 (2026-09-24): "brightest band" was wrong -- the hottest visible
            # pixel on this map is the 37 m return, not range 0; reworded to Thrust 4's
            # framing (the display-normalisation fact, not a brightness claim).
            "Range 0-2 m is not a target: it is the direct path the display "
            "normalises to (0 dB); real multipath sits near 37 m and 68 m (F93/F94).",
            "Noise figure IS quotable: Friis 11.97 dB vs measured 11.80 dB (0.17 dB "
            "agreement, 2026-09-21). Absolute dBm is NOT: input level is free; "
            "quote noise figure/relative dB only.",
            "Channel mismatch: all 1024 elements share one config (get_RX_config "
            "broadcasts one value); gain/phase mismatch is structurally zero; a "
            "per-element spread is a small change, on the list.",
            "What end-to-end buys over Friis: 0.17 dB agreement validates the noise "
            "mechanism; Friis can't give this knob's effect on the AFE, tracker or "
            "detector.",
            "Spacing: lambda/2 at 30 GHz, 0.525 lambda at 31.5 GHz -- grating lobes "
            "beyond |sin theta| ~0.90; native range resolution 5 cm, binned 20:1 to "
            "1 m gates.",
            # wave 9 (2026-09-24): peak-to-median caveat (also on Thrust 2's card) --
            # applies here too, since this is exactly the statistic the LNA-bias knob
            # moves.
            "Peak-to-median dynamic range measures how empty the map is (median set "
            "by empty gates), not target SNR; it moves with the noise floor, which "
            "this knob changes.",
            # wave 9 (2026-09-24, orchestrator course-correction): the 0 dB reference
            # cell (range-0 gate) is ~1.6 px tall on screen, anti-aliased away -- the
            # colour bar's 0 dB is never actually visible on the map. Numbers not
            # typed here on purpose: the other coder prints them on the panel
            # subtitle.
            "The 0 dB reference is a single range-0 gate too small to see; the "
            "panel prints the brightest visible return (read it off the screen); "
            "every dB on the map is relative to the direct path.",
        ],
        do_not_say=[
            "Any DC power readout: PRX is U-shaped, MINIMUM at best quality (8.45 V "
            "rail, 100-200 mV chain).",
            "The compression regime (scaling 1e-1..1e-3): worse live than silence.",
            "That gm scales linearly with bias -- true only at 8 mA; say "
            "'constant-overdrive power scaling'.",
            "Anything about IIP3: constant across 0.5-10 mA, unlike a real LNA.",
            "Any gain knob: peak normalization removes it.",
            "Any absolute dBm sensitivity: the input scale is arbitrary.",
            "That the brightest thing on the picture is the 0 dB reference -- it is "
            "not visible.",
        ],
    ),
    DemoPreset(
        id="thrust2_feature_reduction_error",
        label="Thrust 2 - feature-reduction (AFE) error vs end result",
        thrust=2,
        n_steps=6,
        overrides=_merge(
            {"afe": {"enabled": True, "params": {"exp": 5, "mantissa": 6}}},
            {"interconnect": {"enabled": False}},
            # Wave 7 tracker re-pick (2026-09-23, F94 measured on the diffuse-scattering
            # Ka retrace, real multipath restored): k=8 (the old default) is DEGENERATE
            # at Ka -- effective rank is 3-4 and arm A spikes hard mid-run (frame 3:
            # 0.00->0.34, >2x its own settled level). k=4 is ALSO unstable on this file
            # (arm A: 0.00, 0.07, 0.07, 0.70, 0.51, 0.08 -- an 8-10x spike at frame 3,
            # reproduced across repeated runs). k=2 is the largest k with no spike (arm
            # A stays 0.00-0.07 across all 6 frames) and the arms still clearly
            # separate (A ~0.06 vs B ~0.32, both settled). See the preset's `say` list
            # for the re-measured numbers.
            {"subspace": {"params": {"k": 2}}},
            _only_products("range_az", "range_el", "subspace_err"),
        ),
        # INTEGRITY (hostile-expert third read, 2026-09-23): the FFT range-elevation
        # panel used to be deliberately off because it contradicts the "image barely
        # moves" framing -- hiding a contradicting panel is worse than showing it. It
        # is back on; the card now tells the no-exact-pair version below.
        # wave 9 (2026-09-24): the blurb used to say each panel PRINTS a subspace-error
        # statistic -- no panel does; the tracker panel plots a curve against a dashed
        # "warm-start settled level (0.06, reference)" line, read by eye.
        # wave 9 update #3 (2026-09-24, orchestrator course-correction): RETRACTED the
        # specific dB pair. A same-day re-render (03:0x) printed range-azimuth
        # 76.7 -> 76.5 and range-elevation 76.8 -> 76.6 (0.2 dB each) -- neither the
        # 76.9->76.5/0.4 dB pair this card quoted minutes earlier, nor the 76.7->76.4/
        # 0.3 dB azimuth pair. The pipeline is nondeterministic at ~5e-3, so these
        # peak-median statistics drift ~0.1-0.2 dB run to run: NO exact pair belongs
        # on the card, ever -- only the printed numbers on THIS run's own screen, and
        # the fact that the move sits at the run-to-run noise floor.
        blurb=("Press Run once: both arms appear alike (A, top, mantissa 6 bit / B, "
               "bottom, 1 bit); the tracker panel plots a subspace-error curve "
               "against a dashed 0.06 reference line -- A settles on it, B sits "
               "about 5x above. Both images move a few tenths of a dB (read the "
               "printed numbers) -- at the run-to-run floor, not the story; the "
               "tracker curve is. Manual: AFE mantissa 6 -> 1 bit, run again."),
        live_knobs=[("afe", "mantissa", "6 -> 1 bit (subspace_err 0.06 -> 0.32 at k=2)")],
        # A/B (Change 1): as-loaded IS mantissa=6 (the settled 0.06 arm); run B drops
        # to 1 bit, the 0.32 arm the card's headline quotes.
        ab=("afe", "mantissa", 1),
        ab_label_a="6 bit", ab_label_b="1 bit",
        # Rewritten (hostile-expert fourth read, 2026-09-23): the old note claimed the
        # elevation cut moves ~2.7 dB, a number no panel on THIS screen shows (that
        # figure was a cross-arm mean |dB| difference on the off-screen FFT az-el
        # product). The note now claims only what the two displayed images and the
        # tracker curve do -- RETRACTED (wave 9 update #3, 2026-09-24): every specific
        # dB pair quoted here across three earlier waves (2.7, then 0.52 mean
        # 76.85->76.57, then 0.3/0.4 at 76.x->76.x) is superseded by the blurb's own
        # comment above: the statistic drifts run to run, so the note (and the card)
        # names the drift, never a pair.
        # Numbers re-measured wave 7 at k=2 (see the `overrides` comment above); array
        # disclosure appended (wave 7, X4-X8: mandatory on any card naming the array).
        # wave 9 (2026-09-24, item 2.6): the tracker panel also draws a red dotted
        # "refinement passes/frame" trace on a right-hand axis, flat at 10 for this
        # preset -- the card never named it; it is the AFE mantissa knob on screen, not
        # a tracker setting.
        screen_note=("range-azimuth and range-elevation images barely move on the "
                     "adaptive display clip (statistics printed on each; range m/gate "
                     "and unambiguous range on the panel); the tracker error moves "
                     "about 5x; subspace error is unnormalised, ceiling sqrt(k) = 1.41 "
                     "for k = 2; the tracker panel's red dotted trace (right axis) is "
                     "refinement passes/frame, fixed at 10 here -- the knob on this "
                     "screen is AFE precision, not the tracker. " + _ARRAY_DISCLOSURE),
        say=[
            # wave 9 (2026-09-24, item 1.5): frame 1 reads 0.00 on screen, frame 2
            # ~0.065 -- "by frame 1" was wrong by one frame.
            "As loaded the curve starts near 0, settles at about 0.06 by frame 2; "
            "the knob compares SETTLED levels, 0.06 vs 0.32.",
            "Headline in ANGLES: 0.32 -> 0.06 is unnormalized, bounded by sqrt(k); "
            "converted, principal angle goes 13.1 -> 2.6 deg.",
            "Both images move by only a few tenths of a dB -- read the two printed "
            "peak-median numbers on the screen; that is at the ~0.1 dB run-to-run "
            "floor, so the image is not the story; the tracker curve is (about 5x "
            "above the 0.06 reference on arm B).",
            "No detection metric is wired to this view; say so before asked what "
            "it means for P_d or false alarms.",
            "Range 0-2 m is not a target: it is the direct path the display "
            "normalises to (0 dB); real multipath sits near 37 m and 68 m.",
            "Tracker k was re-picked: k=8 (old default) and k=4 spike mid-run on "
            "the Ka retrace (rank 3-4, F94); k=2 is the largest stable k.",
            "Spacing: lambda/2 at 30 GHz, 0.525 lambda at 31.5 GHz -- grating lobes "
            "beyond |sin theta| ~0.90; range resolution 5 cm, binned 20:1 to 1 m "
            "gates.",
            # wave 9 (2026-09-24, item 3.3): prepared answer for "what is the 0.06
            # floor made of?" -- an interpretation, not a re-measurement.
            "Prepared answer -- 'what is the 0.06 floor made of?': at k=2 (rank "
            "~3-4, F94) part of A's residual is rank mismatch; the 5x gap to B is "
            "the knob (interpretation).",
            # wave 9 (2026-09-24, item 3.12): the array-spread caveat lives on Thrust
            # 1's card; naming it here since the AFE/tracker story is what a
            # per-element spread would actually change.
            "All 1024 elements share one front-end config (Thrust 1); a spread would "
            "show up in the AFE weights/tracker curve, not the picture.",
            # wave 9 (2026-09-24, orchestrator course-correction): same line as
            # Thrust 1/4 -- see that card's comment for the measurement it stands
            # in for.
            "The 0 dB reference is a single range-0 gate too small to see; the "
            "panel prints the brightest visible return (read it off the screen); "
            "every dB on the map is relative to the direct path.",
        ],
        do_not_say=[
            "That the mantissa sweep models analog hardware error: AFEBlock's "
            "WEIGHT_FLOAT is right for a compute datapath, wrong for analog "
            "control (compress.py).",
            "That the picture does not respond: both images move a few tenths of "
            "a dB, at the run-to-run floor -- not evidence either way; the "
            "tracker curve is.",
            "That a higher compression ratio would look better: 512 of 1024 lets the "
            "tracker see drift; observability drops at 16x.",
            "Peak-to-median dynamic range as evidence compression is good: it improves as "
            "compression worsens.",
            "That k=8 still applies: degenerate here (F94); Thrust 2/3 both run "
            "at k=2 now, so they compare.",
        ],
    ),
    DemoPreset(
        id="thrust3_cold_start_acquisition",
        label="Thrust 3 - adaptive feature extraction: cold-start acquisition",
        thrust=3,
        n_steps=8,
        # Owner decision (option A, 2026-09-23), round 2: an identical-arms screen is a
        # null demo (round-6 review). Arm A is now the FIXED-effort arm at the largest
        # n_refine of {1, 2, 3, 5} that still took >=3 frames to reach within 1.5x of
        # its own settled level (real chain, k=2, cold start, 8 frames, two repeats
        # each, scratch 2026-09-23): n_refine=1 already got there in 2 frames (same as
        # the shipped gate below), so it was rejected; 2, 3 and 5 all took 3 frames --
        # 5 is the largest of those, keeping arm A closest to the shipped budget while
        # still visibly slower to acquire. n_refine is not set explicitly here: the
        # runner derives 5 from gap_response="none" (webapp/pipeline_runner.py), which
        # is what lets the single `ab` switch below move both n_refine AND
        # gap_response together. Arm B is the shipped default (gap_response="refine",
        # which derives n_refine=10 the same way) -- on this file the gate never
        # escalates past that baseline (see below), so B is exactly a fixed 10-pass
        # tracker. k=2 is the largest spike-free rank measured on this file (matches
        # Thrust 2's own pick, F94); k=4 spikes ~0.98 on both arms and is not shipped.
        # At k=2 sv_gap_norm sits at 0.086-0.096 every frame, 9x the gate's 0.01
        # threshold, so gap_response="refine" NEVER escalates to n_refine_hi=60 here --
        # this CONTRADICTS notes/ESTABLISHED_FACTS.md F94's tracker addendum ("the
        # shipped gate... spends 60 power iterations per frame"), which was measured at
        # k=8 on an EARLIER munich_ka.pkl, before this file's geometry fix (8d1e251)
        # changed the spectrum; it does not hold at k=2 on the current file (flagged in
        # notes/STATE.md for re-verification/retraction).
        overrides=_merge(
            {"afe": {"enabled": True}},
            {"subspace": {"params": {"k": 2, "warm_start": "cold",
                                     "gap_response": "none"}}},
            {"interconnect": {"enabled": False}},
            _only_products("subspace_err"),
        ),
        # wave 9 (2026-09-24, items 3.4/3.5): dropped "within 1.5x by frame 2" for B --
        # true by 0.001 against a ~5e-3 nondeterminism floor, not a real margin. Added
        # the stronger claim that IS on screen: A never reaches B's floor in 8 frames.
        # Reworded "up to 60 when unhealthy": the right axis here runs 0-12 and this
        # file's gate never escalates past 10 -- 60 is what the gate would do on a
        # file with a small spectral gap, not this one.
        blurb=("Cold start on BOTH arms, k=2 (k=4 spikes ~0.98, see say). Arm A: "
               "FIXED 5 passes/frame, settling near 0.16, never reaching B's ~0.06 "
               "floor in 8 frames -- about 2.5x higher throughout. Arm B: the "
               "shipped adaptive gate, 10 passes/frame baseline (right axis 0-12; a "
               "small-gap file would climb to 60, not this one). Over 8 frames: A "
               "about 0.6 -> 0.31 -> 0.19; B about 0.30 -> 0.09 by frame 2, settled "
               "from frame 3 -- a frame ahead, at a lower floor, for 2x the passes "
               "(flat 5 vs flat 10). It never escalates here: k=2's gap stays well "
               "clear of 0.01."),
        # gap_response has no registry ParamSpec (no UI slider -- see
        # demo_presets._INTERNAL_PARAMS); it is the `ab` knob below, not listed here as
        # a manually-turned live_knob. warm_start does have a slider and stays
        # reachable, though it is no longer part of this preset's A/B.
        live_knobs=[("subspace", "warm_start",
                     "manual: cold -> warm (perturbed truth; not part of this A/B)")],
        ab=("subspace", "gap_response", "refine"),
        ab_label_a="fixed effort (5 passes/frame)",
        ab_label_b="adaptive gate (shipped default, 10 passes/frame baseline)",
        screen_note=("Frames are 1 m of platform travel each (no time base); the "
                     "right-axis trace is AdaOjaBlock's own refinement-passes-per-frame "
                     "(n_refine_used) -- flat at 5 (A) vs flat at 10 (B), because this "
                     "run's spectral gap (about 0.09) never drops below the gate's 0.01 "
                     "threshold, so B never escalates past its baseline. "
                     + _ARRAY_DISCLOSURE),
        say=[
            "Cold start, k=2 (largest spike-free rank, F94), measured over 8 frames. "
            "Quote 'about' -- nondeterministic at ~5e-3, never the third decimal.",
            "The A/B statistic is frames-to-acquire vs passes-per-frame, both on "
            "screen: B pays 2x the compute for a lower floor and one frame sooner.",
            "This is 2:1 compression (m=512 of 1024). At 16:1 or 64:1 neither arm "
            "converges in this many frames -- why m is not a live knob here.",
            "There is deliberately no image here: the picture doesn't change during "
            "acquisition -- the error curve shows what the tracker hasn't yet "
            "learned. Without the AFE it looks identical -- do not toggle it.",
            "Prepared answer -- 'does your gap diagnostic work at Ka?': at k=2 the gap "
            "sits far above 0.01, so the gate never escalates -- the 2x on screen IS "
            "baseline, not reaction. At k=4 (not shipped) the gap collapses and the "
            "gate spends 6x more, but the cluster still spikes -- 'mitigated'.",
            "The run is not faster than a full SVD: scoring runs the full SVD every frame "
            "for ground truth. The 45x microbenchmark is real, the run time is not.",
            "Spacing: lambda/2 at 30 GHz, 0.525 lambda at 31.5 GHz -- grating lobes "
            "beyond |sin theta| ~0.90; native range resolution 5 cm, binned 20:1 to "
            "1 m gates.",
            # wave 9 (2026-09-24, item 3.12): see the same note on Thrust 2's card.
            "All 1024 elements share one front-end config (Thrust 1); a spread would "
            "show up in the tracker's acquisition curve here, not Thrust 1's "
            "picture.",
        ],
        do_not_say=[
            "Anything with an interferer: three confounders, and the sign of the response "
            "flips with a knob that is not on screen.",
            "'k = 2 is the optimum' in raw subspace_err: the metric ceilings at "
            "sqrt(2) = 1.41.",
            "The AFE on/off toggle as 'the effect of adaptive feature extraction': 2:1, "
            "costs 3 dB, the picture looks identical.",
            "subspace_err beside an m=16 run: the warm tracker's error GROWS with frames "
            "there (0.125 -> 0.807 over six).",
            "That the gate 'always fires' or 'escalates' at Ka: measured false here "
            "(flat 5 on A / 10 on B, 8/8 frames). F94's escalation claim was "
            "pre-fix (8d1e251), at k=8 -- retracted at k=2.",
        ],
    ),
    DemoPreset(
        id="thrust4_interconnect_range_profile",
        label="Thrust 4 - a worse interconnect, on the range profile",
        thrust=4,
        n_steps=3,
        overrides=_merge(
            {"interconnect": {"enabled": True, "params": {"source": "tessera"}}},
            _only_products("range_profile", "range_az"),
        ),
        # X3 fix (wave 7, 2026-09-23): the blurb used to quote "skirt -53.90 ->
        # -57.43 dB" as if it were on the rendered panel -- it is an OFFLINE
        # native-flat-frame measurement (TESSERA_KNOB_MEASUREMENT note); the range
        # profile panel itself prints its own "median floor, dB rel. peak" statistic,
        # and the offline skirt move sits below that displayed floor. Same fix,
        # geometry convention: Arm A used to be "h 100 um" (model geometry) while
        # every other reference on this card used 50 um presented -- one convention
        # now (presented), with the model-geometry equivalence stated once below.
        # wave 9 (2026-09-24, items 1.2/1.3/2.2): led with the honest story (the
        # interconnect is not the limiting element at this display floor); stated the
        # printed median floor plainly instead of "quote ... if asked".
        # wave 9 (cross-shard fix #2, 2026-09-24, orchestrator): the run-notes line
        # HAS landed on the Results tab (read on the rendered PNG, under each arm's
        # banner) -- shortened the "Block Diagram tab ... after this wave" hedge
        # everywhere on this card. Also: this run printed 76.6/76.6 (floors
        # -50.2/-50.3), not the 76.6/76.5 pair an earlier wave quoted -- that pair
        # drifts run to run, so the card now names the ~0.1 dB run-to-run floor
        # instead of a specific pair of numbers.
        blurb=("THE HONEST STORY: the interconnect is NOT the limiting element at "
               "this display's floor -- median floor is -50.3 dB on both arms; "
               "any difference of about 0.1 dB between the arms' printed "
               "range-azimuth peak-median statistic is the run-to-run floor, not "
               "the knob. This is the LIVE public Tessera/UIC TSV surrogate "
               "(InterconnectBlock(source='tessera'), scale x2 / half frequency "
               "at Ka band; see the run-notes line under the banner). "
               f"Arm A: canonical geometry, {_TESSERA_CANONICAL_HEIGHT_UM:g} um "
               f"presented (= {_TESSERA_CANONICAL_HEIGHT_MODEL_UM:g} um model "
               "geometry at scale x2). Arm B drops TSV height to its presented "
               "low end -- OFFLINE the biggest single-knob mover of the skirt "
               "(3.53 dB native flat-frame move) -- bulk DELAY, an offline "
               "metric the range profile cannot show; a group-delay/|S21| "
               "overlay would. That move sits below the printed median floor."),
        live_knobs=[("interconnect", "tessera_height_um",
                     f"{_TESSERA_CANONICAL_HEIGHT_UM:g} -> {_TESSERA_ARM_B_HEIGHT_DISPLAY:g} um "
                     "(the A/B above)"),
                    ("interconnect", "source",
                     "manual third option, not part of the A/B: source='default' + "
                     "case='default' selects the old SYNTHETIC 11-tap boxcar placeholder")],
        ab=("interconnect", "tessera_height_um", _TESSERA_ARM_B_HEIGHT_UM),
        ab_label_a=f"canonical Tessera geometry ({_TESSERA_CANONICAL_HEIGHT_UM:g} um presented)",
        ab_label_b=f"TSV height -> {_TESSERA_ARM_B_HEIGHT_DISPLAY:g} um presented (largest skirt mover)",
        screen_note=("LIVE Tessera surrogate, scale model x2 (see the run-notes "
                     "line under the banner); in-band "
                     "|S21| moves <0.03 dB across every knob -- invisible on a "
                     "peak-normalized display. The 3.53 dB skirt figure on the card "
                     "is an offline flat-frame metric (bulk delay, not distortion), "
                     "and sits ~50 dB below this display's real noise floor: read "
                     "both panels and say so if they look identical. "
                     "Crosstalk (NEXT/FEXT) is modelled for a multi-via arrangement, not "
                     "this default single-via one: worst-pair band mean, checker3x3, "
                     "28.5-31.5 GHz, pitch 40 um (in training box) NEXT -31.5 / "
                     "FEXT -39.2 dB; pitch 60 um (our shipped geometry) NEXT -30.3 / "
                     "FEXT -34.5 dB -- the pitch trend itself inverts above ~23 GHz on "
                     "this public checkpoint (F89), so these are fixed reference values "
                     "for checker3x3, not numbers from this run."),
        say=[
            "This is the LIVE public Tessera/UIC surrogate (checkpoint, not a CSV) -- "
            "the run-notes line under the banner names the scale factor and "
            "frequency.",
            "Credit UIC by name (Mohamed Gharib, Leonid Popryho, Inna Partin-Vaisband; "
            "doi 10.1109/TCAD.2026.3718807) -- block, wrapper and six S21 CSVs are "
            "theirs.",
            "In-band |S21| is invisible on this display for every knob (<0.03 dB "
            "span); A/B moves TSV height because it measurably moves the skirt.",
            "Crosstalk is now modelled -- NEXT/FEXT between vias -- with F89's "
            "numbers on the screen note; per-ELEMENT broadcast is still "
            "unmodelled.",
            "77 GHz shipped CSVs are not reconciled with the 30 GHz frames -- "
            "caption real-CSV results shape-only.",
            "Range 0-2 m is not a target: range 0 = earliest arrival "
            "(normalize_delays=True). Peaks near 37, 46, 68, 79, 113 m are "
            "multipath; the 120-125 m rise is the skirt wrapping at 125 m.",
            "Skin depth goes as f^-1/2, not f^-1: conductor loss is under-estimated "
            "by sqrt(2) (~0.2 dB of 0.5 dB in-band loss), substrate coupling by up "
            "to 2x; trends/shape exact (F91).",
            "Spacing: lambda/2 at 30 GHz, 0.525 lambda at 31.5 GHz -- grating lobes "
            "beyond |sin theta| ~0.90; range resolution 5 cm, binned 20:1 to 1 m "
            "gates.",
            # wave 9 (2026-09-24, orchestrator course-correction): same line as
            # Thrust 1/2 -- see Thrust 1's comment for the measurement it stands
            # in for.
            "The 0 dB reference is a single range-0 gate too small to see; the "
            "panel prints the brightest visible return (read it off the screen); "
            "every dB on the map is relative to the direct path.",
        ],
        do_not_say=[
            "That crosstalk is structurally absent -- RETRACTED: the surrogate models "
            "NEXT/FEXT for multi-via layouts (F89); one broadcast S21 for 1024 "
            "elements remains true.",
            "'Drag pitch, watch crosstalk change' at our band: the checkpoint's pitch "
            "trend is physical only below ~23 GHz (F89) and runs backwards.",
            "That the A/B skirt movement is a worse shape: the measurement note "
            "attributes most of it to a delay artifact.",
            "'Case3' from the dropdown as UIC Case3: a legacy passthrough alias, "
            "not reachable here.",
        ],
    ),
    DemoPreset(
        id="thrust5_detector_cfar",
        label="Thrust 5 - live chain from the stored channel: classical CFAR",
        thrust=5,
        n_steps=5,
        overrides=_merge(
            _T5_LIVE_CHAIN,
            {"detector": {"params": {"mode": "cfar", "threshold": CFAR_THRESHOLD,
                                     "cfar_guard": 2, "cfar_train": 6}}},
        ),
        # MEASURED 2026-09-23 (CUDA, two identical runs per arm): A (12-bit) 61
        # detections over the 5 test frames, 16 hits / 45 unmatched / 9.0 per frame;
        # B (3-bit) 49 detections, 13 hits / 36 unmatched / 7.2 per frame. The gate
        # printed max |diff| = 0 ADC codes on the A arm (bit-identical to the corpus)
        # and a nonzero live-vs-stored diff on the B arm -- the quantizer, which is
        # the point. 3-bit, not 4 (item 2, hostile-expert read, 2026-09-23): swept
        # {2, 3, 4, 6} bits on both this preset and RADDetNet -- 4-bit was the value
        # at which RADDetNet's hit count went UP relative to 12-bit (10 -> 11) on
        # these 5 frames, reading backwards on screen; 3-bit is the largest depth in
        # the sweep at which BOTH detectors lose hits relative to 12-bit.
        blurb=("The held-out TEST frames, replayed as the STORED RAY-TRACED CHANNEL: "
               "the RF front end, dechirp, thermal floor, impairments, IF high-pass "
               "and ADC all run LIVE from that channel with the values on screen, then "
               "CA-CFAR. Press Run once: A is the 12-bit ADC the corpus was generated "
               "at, B the same frames re-digitised at 3 bits. Measured over the 5 "
               "frames: 16 hits / 45 unmatched (9.0 per frame) at 12 bits, 13 / 36 "
               "(7.2) at 3 bits. Thresholds are each detector's recall-0.5 point on "
               "the 172-frame split; on 5 frames recall varies, so compare the "
               "172-frame FA/frame rows, not the crosses."),
        live_knobs=[("quantizer", "bits", "12 -> 3 (the ADC is re-run, not re-loaded)"),
                    ("detector", "threshold", "0.66 -> 0.8 (fewer detections)")],
        ab=("quantizer", "bits", 3),
        ab_label_a="12-bit ADC (as built)",
        ab_label_b="3-bit ADC (same frames)",
        screen_note=_T5_SCREEN_NOTE,
        say=[
            "SAY FIRST: the frames change here. Thrusts 1-4 ran ray-traced munich "
            "frames (125 m, range-azimuth); this is the benchmark corpus (100 m, "
            "range-Doppler). STORED is the ray-traced channel -- everything after "
            "runs live, so the ADC knob reaches the detector.",
            "The gate that makes this honest: at generation settings, the live cube "
            "is BIT-IDENTICAL to the stored one -- max |diff| = 0 ADC codes. Move a "
            "knob and that number leaves zero; that difference is the whole "
            "demonstration.",
            "Classical CFAR scores AP 0.301 on this split offline; chance floor "
            "0.081 (b1_bench_v3, 12-bit default impairments) -- the counts on "
            "screen are 5 live frames of a different corpus, a demonstration not a "
            "re-measurement.",
            # wave 9 (2026-09-24, item 1.7): the old line was wrong on both counts --
            # the detector map spans 0-50 m (a 10 m unscored strip above the 40 m
            # dashed line), and the 0-100 m panel is Range-Doppler power, unlabelled.
            "The CFAR map spans 0-50 m; the 10 m strip above the dashed line is "
            "unscored (labels/scoring stop at 40 m). The 0-100 m panel is "
            "Range-Doppler power, unlabelled.",
            "Ground truth omits ~3 real strongly-scattering objects per frame "
            "inside 40 m, so a detector firing on every real object has a "
            "precision ceiling of 0.64 -- some 'false alarms' are real objects.",
            "Unambiguous velocity is +-v_max from the manifest (~9.7 m/s); corpus "
            "targets are slower by construction, so a 20 m/s car would alias.",
            "Unmatched detections can DROP at deeper quantisation: quantisation "
            "noise raises the CA-CFAR estimate, so fewer weak peaks clear the "
            "threshold -- a loss of sensitivity, not a quality gain (5 frames "
            "cannot resolve the knob).",
            # wave 9 (2026-09-24, T5 item 5): prepared answer, shared across the three
            # T5 cards.
            "This detector sits at its 172-frame recall-0.5 threshold, yet gives "
            "0.53 recall on these 5 frames; the matched-recall FA comparison is "
            "made on that 172-frame split -- 5 frames cannot reproduce a recall.",
        ],
        do_not_say=[
            "That 16 vs 13 hits measures 3-bit quantisation's cost: 5 frames at one "
            "threshold is a demonstration that the knob reaches the detector, not a "
            "measurement of it.",
            # wave 9 (2026-09-24, T5 item 2): 0.138 (fftradnet_rad, on every PR legend
            # here) is not one of the retracted numbers -- it is the 2026-09-22
            # rescoring under the current beat_cfar.json protocol; only 0.229 (the old
            # rad-format figure) is retracted.
            "Any learned-detector number before 2026-09-22 except rd-format "
            "0.127/0.123: the rad-format 0.138 (PR legend) is the 2026-09-22 "
            "rescoring under beat_cfar.json's protocol, not the retracted 0.229 "
            "(F84).",
            "That fewer CFAR training cells means more false alarms: measured, the "
            "count went 46 -> 39 (train 6 -> 2). Do not turn that knob on stage.",
        ],
    ),
    DemoPreset(
        id="thrust5_detector_ml",
        label=("Thrust 5 - live chain, ported network (the arm that LOSES, shown on "
               "purpose)"),
        thrust=5,
        n_steps=5,
        overrides=_merge(
            _T5_LIVE_CHAIN,
            {"detector": {"params": {"mode": "ml", "checkpoint": ML_CHECKPOINT,
                                     "threshold": ML_THRESHOLD}}},
        ),
        # MEASURED 2026-09-23 (CUDA, repeated): A (corner 1.0 m) 148 detections over
        # the 5 frames, 15 hits / 133 unmatched / 26.6 per frame; B (corner 25 m) 65
        # detections, 4 hits / 61 unmatched / 12.2 per frame. 25 m was chosen by
        # sweeping: 4 m and 8 m move the count by 1-2 crosses (invisible on stage),
        # 40 m empties the screen (3 arms of 5 frames show nothing at all).
        # ATTENUATION, MEASURED not assumed (item 1, hostile-expert read, 2026-09-23):
        # the caption used to say the 25 m corner "discards everything closer", which
        # the plot contradicts (targets at 20-25 m stay bright). The filter is an
        # order-2 Butterworth high-pass (e2e/chain/receive.py IFHighPassBlock); its
        # |H(R)| = 1/sqrt(1+(corner/R)^(2*order)) gives 4.27 dB of attenuation at the
        # targets' ~22 m range for a 25 m corner (order 2) -- computed directly from
        # `IFHighPassBlock.response`, not eyeballed. That is well above the 3 dB floor
        # the task set for keeping the corner as-is, so 25 m is unchanged; the wording
        # now states the measured dB instead of "discards".
        blurb=("The same live chain, decoded by the ported FFTRadNet checkpoint (rd "
               "input; offline test AP 0.127 vs CFAR's 0.301). Its objectness map is "
               "a range-profile x fixed-azimuth-prior STRIPE, not peaks: the network "
               "never learns azimuth (F83). A/B moves the IF high-pass corner from "
               "1 m (real receiver) to a deliberately broken 25 m, attenuating (not "
               "discarding) returns inside 25 m -- about 4.3 dB at the targets' "
               "~22 m range: unmatched/frame falls 26.60 -> 12.20, hits 15 -> 4 "
               "(scoreboard). Threshold is pinned at recall-0.5 (0.22); at default "
               "0.5 this checkpoint draws nothing."),
        live_knobs=[("if_hpf", "corner_range_m",
                     "1 m (as built) -> 25 m (attenuates returns inside 25 m; about "
                     "4.3 dB at the targets' 22 m range)"),
                    ("detector", "threshold", "0.22 -> 0.5 (the figure goes blank)")],
        ab=("if_hpf", "corner_range_m", 25.0),
        ab_label_a="IF high-pass corner 1 m (as built)",
        ab_label_b="IF high-pass corner 25 m (attenuates ~4.3 dB at 22 m)",
        # ".": the shared note ends on the render-time v_max clause, so the sentence
        # separator has to be added back here or the two run together on screen
        # ("v_max +-9.69 m/s Loses to CFAR", read off the rehearsal PNG).
        screen_note=_T5_SCREEN_NOTE.rstrip(".")
        + ". Loses to CFAR 0.127 vs 0.301, shown on purpose.",
        say=[
            "The learned detector LOSES to CFAR: 0.127 vs 0.301, chance floor 0.081. "
            "Say it first; the diagnosis is the result.",
            "B is not a plausible receiver -- a 25 m high-pass corner, 25x the real "
            "one -- the point is a front-end setting reaches the detector at all: "
            "unmatched/frame 26.60 -> 12.20, hits 15 -> 4 (scoreboard).",
            "At A's operating point, offline expects ~29 crosses/frame = 26.4 FA + "
            "3.0 hits (beat_cfar.json); these 5 live frames give 26.60 "
            "unmatched/frame -- same regime, not the same number.",
            "Both ported networks emit a near-separable f(range) * g(azimuth) map: "
            "rank-1 energy fraction 0.89 / 0.76 against 0.31 for ground truth. Under "
            "azimuth-only matching they score no better than a constant map.",
            "F83's mechanism: neither head converts channel phase into an angle bin. "
            "An architecture with range x azimuth as its spatial plane scores 0.476 "
            "and passes the controls this one fails -- load the RADDetNet preset, "
            "read its caveats first.",
            "This checkpoint trained on a different corpus from these frames; its "
            "rd input scaling comes from that corpus (screen note). Moving a knob "
            "takes it further out of its training distribution.",
            "On Arm B some frames score TP = 0 (every cross a miss): that is the "
            "mechanism on display, not an accident -- the 25 m corner attenuates "
            "the same near-range returns this checkpoint was trained to fire on.",
            # wave 9 (2026-09-24, T5 item 4): the honest line for why the 4.3 dB
            # number cannot be read off the picture.
            "You cannot see the 4.3 dB: both maps are peak-normalised and the peak "
            "sits inside 25 m; what you see is the floor coming up relative to a "
            "peak attenuated along with it.",
            # wave 9 (2026-09-24, T5 item 5): prepared answer, shared across the
            # three T5 cards.
            "This detector sits at its 172-frame recall-0.5 threshold, yet gives "
            "0.50 recall on these 5 frames; the matched-recall FA comparison is "
            "made on that 172-frame split -- 5 frames cannot reproduce a recall.",
        ],
        do_not_say=[
            "'The rad input doubles AP' or any 0.229 / 0.484 figure: retracted, F84.",
            "'We fixed azimuth': the stripe statistic refutes it on the next slide.",
            "That the 25 m corner measures receiver-design sensitivity: it is an "
            "illustration on 5 frames, not a sweep.",
            "That this is a benchmark of SSMRadNet/FFTRadNet: the fault is on our "
            "side, and the collaborator README says so.",
        ],
    ),
    DemoPreset(
        id="thrust5_detector_raddetnet",
        label="Thrust 5 (LEAD) - RADDetNet vs CFAR: AP 0.476 vs 0.301, in-distribution",
        thrust=5,
        n_steps=5,
        overrides=_merge(
            _T5_LIVE_CHAIN,
            {"detector": {"params": {"mode": "ml", "checkpoint": RADDETNET_CHECKPOINT,
                                     "threshold": RADDETNET_THRESHOLD}}},
        ),
        # MEASURED 2026-09-23 (CUDA, repeated): A (12-bit) 23 detections over the 5
        # frames, 10 hits / 13 unmatched / 2.6 per frame; B (3-bit) 16 detections, 5
        # hits / 11 unmatched / 2.2 per frame. 3-bit, not 4 (item 2, hostile-expert
        # read, 2026-09-23): at 4-bit the hit count moved the WRONG way (10 -> 11,
        # reading backwards on screen); swept {2, 3, 4, 6} bits on both this preset
        # and CFAR -- 3-bit is the largest depth at which BOTH detectors lose hits
        # relative to 12-bit, so the direction on screen now agrees with the claim.
        # X2 fix (wave 7, 2026-09-23): the screen shows fewer crosses AND fewer hits
        # for RADDetNet than CFAR on 5 unmatched-recall frames, which reads as a
        # loss; the real, defensible claim is fewer false alarms AT MATCHED recall
        # (2.99 vs 6.24 FA/frame, 172 frames). Reworded to lead with that, not with
        # the architecture description -- the scoreboard panel is reordered the same
        # way (`webapp/detector_scoreboard.py`).
        blurb=("THE REAL CLAIM, first: at matched recall (0.5) on the 172-frame split, "
               "RADDetNet racks up fewer false alarms than CFAR -- 2.99 vs 6.24 "
               "FA/frame (6.24 on the CFAR screen; AP 0.476 vs 0.301, controls pass). "
               "On the 5 live frames below, fewer crosses can mean fewer hits, since "
               "these frames aren't recall-matched -- read the scoreboard's FA rows, "
               "not the crosses. The same live chain runs RADDetNet (Doppler as "
               "channels, range x azimuth as the spatial plane) on CFAR's cube. A/B "
               "re-digitises the stored channel at 3 bits: hits go 10 -> 5, five "
               "frames -- read it as 'the knob reaches the detector', not a ranking. "
               "Out of distribution the result is seed-dependent (F86); say so "
               "unprompted."),
        live_knobs=[("quantizer", "bits", "12 -> 3 (the ADC is re-run, not re-loaded)"),
                    ("detector", "threshold", "0.44 -> 0.2 (more, weaker detections)")],
        ab=("quantizer", "bits", 3),
        ab_label_a="12-bit ADC (as built)",
        ab_label_b="3-bit ADC (same frames)",
        screen_note=_T5_SCREEN_NOTE,
        say=[
            "The defensible sentence: a learned head on the classical front end beats a "
            "CFAR threshold on the same cube, in-distribution -- say that, not 'beats "
            "CFAR' (F85 addendum).",
            "Every offline number comes from e2e/ml/runs/beat_cfar.json (seed 42, "
            "b1_bench_v3, 12-bit default impairments); re-scored bit-identically. "
            "Paired scene bootstrap: +0.175 AP vs shipped CFAR (0.301), 95% CI "
            "[+0.145, +0.208]; +0.148 vs the best of nine classical baselines (0.328).",
            # wave 9 (2026-09-24, T5 item 5): folded the recall-0.5 caveat into this
            # bullet -- the card's say list is already at its 6-bullet cap.
            "The counts on screen are 5 live frames of a different corpus -- a "
            "demonstration, never a re-measurement of AP; recall here (0.33) cannot "
            "reproduce the 172-frame recall-0.5 threshold these arms are set at.",
            "The controls are F83's, which the shipped nets FAILED (deranged-label "
            "retention 12%, CFAR 10%, shipped nets 48-51%); nine classical baselines "
            "were scored too, best 0.328 -- above shipped CFAR (0.301), but the best "
            "classical still loses to RADDetNet (0.476).",
            "Four learned arms were screened on this test split: three ported "
            "architectures and this one designed to the F83 diagnosis; all four are in "
            "beat_cfar.json, none dropped.",
            "THE CAVEAT: on an unseen earlier-generator corpus (b1_bench_v2), CFAR "
            "scores 0.179/13.2 FA; this checkpoint (seed 42) 0.208/15.1; seed 43 "
            "0.153/20.7 (F86). Out of distribution it does NOT reliably beat CFAR.",
        ],
        do_not_say=[
            "'Beats CFAR', unqualified: the verified claim is in-distribution and on "
            "CFAR's own front end (F85 addendum).",
            "That 5 hits at 3 bits vs 10 at 12 bits measures the cost of 3-bit "
            "quantisation: 5 frames at one threshold shows the knob reaches the "
            "detector, not measures it.",
            # wave 9 (2026-09-24, T5 item 1): the screen itself prints OOD and
            # 3rd-corpus rows now, so "anything" contradicted what is on screen --
            # reworded to don't-volunteer-but-read-what's-printed.
            "Do not volunteer generalisation/robustness; if asked, read the OOD and "
            "3rd-corpus rows as printed (a third corpus never trained on; the lead "
            "holds) and stop -- 0.487 on v2 is in-distribution, not OOD (F86).",
            "That this is what the professor asked for: it is a detector designed to "
            "the F83 diagnosis, not a port of the collaborators' architectures.",
            "That the model converged: val AP peaks at epoch 14 of 40 and decays to "
            "0.35-0.41 while train loss keeps falling (F85 addendum); early stopping "
            "on val is load-bearing.",
        ],
    ),
]

PRESETS_BY_ID: Dict[str, DemoPreset] = {p.id: p for p in PRESETS}


class PresetError(ValueError):
    """A preset that does not fit the registry -- raised, never papered over."""


def _check_n_refine(preset_id: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PresetError(f"{preset_id}: subspace.n_refine={value!r} must be a positive int")


def _check_gap_response(preset_id: str, value: Any) -> None:
    choices = ("none", "refine", "coast")
    if value not in choices:
        raise PresetError(f"{preset_id}: subspace.gap_response={value!r} not in {choices}")


#: Tracker knobs AdaOjaBlock (e2e/blocks.py) accepts that carry no registry ParamSpec
#: -- no operator should be typing a refinement-pass count or gap-response mode into a
#: text box mid-demo (Thrust 3's cold-start-vs-refine-gate A/B, 2026-09-23). Validated
#: here directly, against AdaOjaBlock's own accepted values, so a typo still fails
#: `apply_preset` loudly, the same way an unknown UI param does; keyed by
#: (block_id, param_key) and checked ahead of the registry-backed loop below.
_INTERNAL_PARAMS: Dict[Tuple[str, str], Any] = {
    ("subspace", "n_refine"): _check_n_refine,
    ("subspace", "gap_response"): _check_gap_response,
}


def ab_key_is_known(bid: str, key: str) -> bool:
    """True if ``(bid, key)`` is a registered UI ParamSpec or one of `_INTERNAL_PARAMS`
    -- the single predicate `apply_preset` and its tests both use, so an `ab` tuple can
    reference either kind of knob without the "stale/typo'd key" guard losing teeth."""
    spec = BLOCKS_BY_ID.get(bid)
    if spec is not None and any(ps.key == key for ps in spec.params):
        return True
    return (bid, key) in _INTERNAL_PARAMS


def apply_preset(preset: DemoPreset, *, arm: str = "a") -> Dict[str, Dict[str, Any]]:
    """`default_block_state()` with the preset's overrides applied and VALIDATED.

    ``arm="b"`` (only meaningful when ``preset.ab`` is set) additionally applies the
    preset's single A/B override on top of its own overrides -- see `DemoPreset.ab`.
    """
    if arm not in ("a", "b"):
        raise PresetError(f"{preset.id}: unknown ab arm {arm!r}")
    overrides = preset.overrides
    if arm == "b":
        if preset.ab is None:
            raise PresetError(f"{preset.id}: has no `ab` override to apply arm 'b'")
        bid_ab, key_ab, value_b = preset.ab
        overrides = _merge(overrides, {bid_ab: {"params": {key_ab: value_b}}})
    state = default_block_state()
    for bid, ov in overrides.items():
        spec = BLOCKS_BY_ID.get(bid)
        if spec is None:
            raise PresetError(f"{preset.id}: unknown block {bid!r}")
        if "enabled" in ov:
            if not spec.toggleable and bool(ov["enabled"]) != spec.enabled_default:
                raise PresetError(f"{preset.id}: {bid} is not toggleable")
            state[bid]["enabled"] = bool(ov["enabled"])
        specs = {p.key: p for p in spec.params}
        for key, value in (ov.get("params") or {}).items():
            if (bid, key) in _INTERNAL_PARAMS:
                _INTERNAL_PARAMS[(bid, key)](preset.id, value)
                state[bid]["params"][key] = value
                continue
            ps = specs.get(key)
            if ps is None:
                raise PresetError(f"{preset.id}: {bid} has no param {key!r}")
            if ps.kind == "choice":
                if value not in (ps.choices or []):
                    raise PresetError(f"{preset.id}: {bid}.{key}={value!r} not in {ps.choices}")
            elif ps.kind == "text":
                if not isinstance(value, str):
                    raise PresetError(f"{preset.id}: {bid}.{key} must be a string")
            else:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise PresetError(f"{preset.id}: {bid}.{key}={value!r} is not numeric")
                if ps.kind == "int" and int(value) != value:
                    raise PresetError(f"{preset.id}: {bid}.{key}={value!r} is not an int")
                if ps.min is not None and value < ps.min:
                    raise PresetError(f"{preset.id}: {bid}.{key}={value!r} below min {ps.min}")
                if ps.max is not None and value > ps.max:
                    raise PresetError(f"{preset.id}: {bid}.{key}={value!r} above max {ps.max}")
            state[bid]["params"][key] = value
    if preset.n_steps < 1 or preset.n_steps > MAX_PRESET_N_STEPS:
        raise PresetError(f"{preset.id}: n_steps={preset.n_steps} outside 1..{MAX_PRESET_N_STEPS}")
    return state


def validate_all() -> None:
    """Apply every preset; any PresetError propagates. Cheap enough to run at import in
    tests, deliberately NOT run at app import so a bad preset cannot take the shell down."""
    for p in PRESETS:
        apply_preset(p)
        if p.ab is not None:
            apply_preset(p, arm="b")
            assert p.ab_label_a and p.ab_label_b, (
                f"{p.id}: ab is set but ab_label_a/ab_label_b are missing")
