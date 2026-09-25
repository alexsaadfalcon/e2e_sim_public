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

#: wave 10 (2026-09-24, item 1.3): the Ka scale factor Tessera geometry is presented
#: through (model um / this = presented um) -- imported, never re-typed as a literal
#: "2", so the crosstalk clause's model-to-presented pitch conversion stays correct if
#: the scale is ever re-measured. See pipeline_registry.py's own comment on the
#: constant for where it comes from (e2e.blocks._resolve_tessera_scale, F89).
from webapp.pipeline_registry import _TESSERA_KA_SCALE

#: notes/DEMO_DEFENSE.md DO-NOT-SHOW #9: runs longer than ~20 frames. Defined in the
#: registry beside MAX_N_STEPS so the runner's error text and this ceiling agree.
from webapp.pipeline_registry import MAX_PRESET_N_STEPS  # noqa: E402

#: The LOSING arm, at Ka (owner 2026-09-24, ballot 4A): the ported FFTRadNet recipe
#: retrained on the Ka corpus (b5's recipe, 30 epochs, --deterministic, recertify PASS).
#: Ka D2 test AP 0.105 vs shipped CFAR 0.218, FA/frame 26.28 -- and only +0.012
#: [+0.001, +0.026] above the null arm's 0.093, i.e. barely above the chance floor
#: (F96 addendum, `e2e/ml/runs/beat_cfar_ka.json`, arm `fftradnet_rd_b15`). That is a
#: STRONGER F83 statement than 77 GHz's 0.127 and the screen should make it, not soften
#: it. Not tracked by git -- the demo machine needs the file.
ML_CHECKPOINT = "e2e/ml/runs/b15_fftradnet_rd_ka/best.pt"

#: The repo-native architecture, retrained at Ka (F95): D2 test AP 0.468 vs shipped
#: CFAR 0.218 (+0.250 [+0.217, +0.282]) and 0.511 vs 0.281 on the never-trained-on D4
#: corpus; all four controls pass on both; seed 42, deterministic, recertify PASS. Its
#: recall-0.5 operating point is objectness 0.4753 (`beat_cfar_ka.json`). F95's
#: INDEPENDENT pass is the condition on presenting it: any Ka screen prints BOTH CFAR
#: baselines, because against a val-tuned CFAR that detects before collapsing Doppler and
#: is unclamped the honest lead is 0.468 vs 0.326, not 0.468 vs 0.218.
RADDETNET_CHECKPOINT = "e2e/ml/runs/b14_raddetnet_ka/best.pt"
RADDETNET_THRESHOLD = 0.475

#: Classical CA-CFAR's recall-0.5 operating point on the same split (`beat_cfar_ka.json`,
#: `operating_point.score_threshold` 0.6155). All three Thrust 5 presets sit at their
#: recall-0.5 points on the 172-frame beat_cfar.json split -- not three arbitrary
#: thresholds -- so the FA/frame numbers quoted from that file (6.2 / 26 / 3.0 per
#: frame) are the comparison. RETRACTED (hostile-expert read, 2026-09-23, item 4):
#: this comment used to say the on-screen CROSS COUNTS (5 live frames) ARE that
#: comparison; they are not -- recall varies frame to frame on 5 frames (arm-A hit
#: rates measured 2026-09-23: CFAR 0.53, RADDetNet 0.33, ML 0.50 -- nowhere near
#: matched), so only the 172-frame rows are comparable across arms.
CFAR_THRESHOLD = 0.615

#: Decode threshold for that checkpoint: its recall-0.5 operating point, objectness
#: 0.2203 (`beat_cfar_ka.json`). At the registry default 0.5 the checkpoint draws NO
#: detections on the test frames -- the "demo landmine" in DEMO_DEFENSE.md, measured
#: again on the real frames. Pinned here so the figure is never blank.
ML_THRESHOLD = 0.220

#: The demo corpus that stores the RAY-TRACED CHANNEL beside each frame (50 scenes,
#: seed 4242, splits 40/5/5, every frame with a `.cfr.npy` sidecar -- generated
#: 2026-09-24 at Ka by `chain_generate --store-cfr`). Replaying it in the `cfr` domain runs
#: the whole analog/digital chain LIVE from that channel, which is what makes a
#: front-end knob reach the detector at all (owner directive, notes/STATE.md §0.1).
#: The predecessor screens replayed a stored ADC cube and could not do that; the two
#: pre-generated bridge corpora that stood in for a live ADC knob are retired with
#: them (the directories may stay on disk).
DEMO_CFR_CORPUS = ("e2e/ml/datasets/b1_demo_cfr_ka/benchmark_v1_ka_D2/"
                   "benchmark_v1_ka_D2/manifest.json")


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
#: wave 10 (2026-09-24, items 2.12/2.13/4.6, hostile round 9): moved the 77 GHz
#: disclosure to be the FIRST clause -- round 9 found it buried sixth of eight,
#: "the single most attackable fact on the screen"; fixed the "are a demo when
#: live" grammar; and named the bare "40 m" number as the scoring crop. The note
#: still ends on the crop clause (kept last so the VMAX_CLAUSE substitution --
#: webapp/app.py `_resolve_screen_note` -- reads cleanly whether or not it fills
#: in), under the 400-char budget on all three T5 cards (checked, ML has the
#: least headroom because of its own appended sentence).
#: wave 12 (2026-09-24, item 1.3): "clip follows arm's floor, not the knob" is now
#: false -- the shared-scale pass takes the TIGHTER of the two arms' own clips on
#: Range-Doppler (printed as "zmin -25.9 (was -29.6)", the opposite direction from
#: the munich maps' "reach the deeper floor" rule). Replaced with the rule that is
#: actually in force -- trimmed elsewhere in the same edit (dropped the "12-bit"
#: aside and two connector words) to stay under the 400-char budget on all three
#: T5 cards; ML has the least headroom because of its own appended sentence.
# KA (owner 2026-09-24, ballot 4A). The 77 GHz disclosure that used to lead this note is
# GONE because the fact it disclosed is gone: these screens replay the Ka corpus, the same
# band as every other thrust. In its place, the two things a Ka screen must carry -- both
# CFAR baselines (F95's independent pass) and the chance floor beside any AP.
_T5_SCREEN_NOTE = (
    "Ka corpus, 28.5-31.5 GHz (b1_demo_cfr_ka); stored ray-traced channel, "
    "ADC chain LIVE; offline numbers (beat_cfar_ka.json) for reference, not "
    "re-measured live; BOTH CFAR baselines, shipped 0.218 and val-tuned 0.326, "
    "chance floor 0.093; a knob moves ML off training distribution; "
    "scoring crop 40 m{VMAX_CLAUSE}."
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

#: wave 10 (2026-09-24, item 1.3): F89's two crosstalk reference PITCHES are MODEL
#: (pre-scale) values -- 60 um is `SHIPPED_TSV_DESIGN["pitch_um"]` itself (matches the
#: shipped `tessera_pitch_um` ParamSpec's own presented default * this scale, i.e. the
#: "our shipped geometry" pitch), 40 um is the training-box reference F89 measured
#: crosstalk at, not a registry default -- so only the two literal MODEL micron
#: figures are hand-typed; the model->presented CONVERSION uses the imported scale
#: constant, never a hardcoded "2". The screen note used to quote both MODEL, four
#: lines below a run-notes line that quotes PRESENTED (30 um) for the same shipped
#: geometry -- two unit conventions on one screen (hostile-expert round 9, item 1.3).
_TESSERA_CROSSTALK_TRAINING_PITCH_MODEL_UM = 40.0
_TESSERA_CROSSTALK_SHIPPED_PITCH_MODEL_UM = 60.0
_TESSERA_CROSSTALK_TRAINING_PITCH_UM = (
    _TESSERA_CROSSTALK_TRAINING_PITCH_MODEL_UM / _TESSERA_KA_SCALE)
_TESSERA_CROSSTALK_SHIPPED_PITCH_UM = (
    _TESSERA_CROSSTALK_SHIPPED_PITCH_MODEL_UM / _TESSERA_KA_SCALE)

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
        # wave 10 (2026-09-24, item 4.5, hostile round 9): "vs image quality"
        # oversold the screen -- the A/B moves peak-median 11.6 dB while both
        # panels are pixel-identical (5.1, do-not-change) and the card's own say
        # list concedes the statistic "measures how empty the map is ... not
        # target SNR". Renamed to name the statistic the screen actually shows.
        label="Thrust 1 - RF circuit knobs vs the image's noise floor",
        thrust=1,
        n_steps=5,
        overrides=_merge(
            # DRIVE LEVEL RE-PICKED ON THE UNIFIED RECEIVER (2026-09-24, shard 3).
            # The front end now acts on the SAMPLED BEAT RECORD and normalises by
            # `mean|beat|`, where the v1.0 placement normalised by `mean|ifft(CFR)|`.
            # On a trace whose line-of-sight tap carries ~93 % of the PDP energy those
            # two references differ by ~2.5 decades, so the shipped 1e-7 put the signal
            # far below the front end's own floor and BOTH arms rendered as noise
            # (measured: peak-median 6.7 dB / 1.0 dB, a 5.7 dB A/B -- the screen's whole
            # claim gone). Measured sweep on munich_ka frame 0, arms 8 mA / 0.5 mA:
            #   1e-7  6.65 / 0.98  (delta 5.67)      1e-4  67.06 / 56.88 (10.18)
            #   1e-6 29.07 / 17.03 (12.04)           3e-4  70.57 / 65.38 ( 5.19)
            #   1e-5 49.04 / 37.13 (11.91)           1e-3  71.23 / 70.27 ( 0.96)
            #   3e-5 58.33 / 46.64 (11.69)           1e-2  71.39 / 71.31 ( 0.08)
            # 3e-5 is the pick: the A/B is still "about twelve dB" (11.69), the map has
            # 58 dB of dynamic range to show it in, and the drive sits two decades below
            # where F97b measured the baseband clamp starting to break the commutation
            # identity this placement rests on (1e-4 -> 1.3e-5 relative error; 1e-2 ->
            # 0.14). Over the preset's 5 frames the pair is 58.31-58.34 / 46.60-46.64 dB,
            # i.e. stable to +-0.02 dB, so it is not a run-to-run drifting statistic.
            {"rffe": {"enabled": True, "params": {
                "scale_mode": "legacy", "signal_scaling": 3e-5,
                "lna_bias_ma": 8.0, "if_bw_mhz": 15.0}}},
            {"interconnect": {"enabled": False}},
            _only_products("range_az"),
        ),
        # wave 10 (2026-09-24, item 1.9, hostile round 9): "~12 dB" in the blurb was
        # the one instance the earlier "about twelve dB" fix (item 1.10, wave 9)
        # missed -- the say list was corrected, the blurb was not. Screen: 65.9 -
        # 54.3 = 11.6 dB.
        # wave 11 (2026-09-24): the owner's live test put both A/B maps on one
        # shared colour scale (zmin set by the deeper arm's floor) -- the old
        # "both arms appear alike ... the difference is in the statistic, not
        # the picture" line is now false: the shared scale puts the floor
        # difference IN the picture (arm B's background reads visibly brighter).
        # wave 12 (2026-09-24): "A, top / B, bottom" -> "A, left / B, right" --
        # the A/B arms render side by side (left/right columns), not stacked;
        # and dropped the drifting (66 vs 54 dB) pair -- the printed peak-median
        # numbers move run to run, so no exact pair belongs on the card (see
        # thrust2's own no-exact-pair rule, same standard applied here).
        blurb=("Press Run once: the two arms (A, left, 8 mA / B, right, 0.5 mA) "
               "share one colour scale down to the deeper floor, so arm B's "
               "background reads visibly brighter -- ~12 dB by the printed "
               "numbers; streaks match. Second knob, manual: IF bandwidth "
               "15 -> 50 MHz."),
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
        # wave 11 (2026-09-24): same fix as the blurb above -- the shared colour
        # scale (owner's live test) makes the floor difference visible, so "same
        # streaks at the same visible brightness" is retracted; and F96 (munich's
        # "unambiguous 125 m" is only the positive-delay half of a 250 m window).
        # wave 12 (2026-09-24, item 1.2): "falls back to the shared -40 dB below
        # -43 dB (as here, both arms)" described the PRE-wave-11 single-arm clip
        # rule and is now false -- the panel subtitle prints the shared-scale
        # result instead ("zmin -67.1 dB (was -40.0)"). Replaced with the actual
        # shared-scale rule; the two meanings of "shared" (a default clip vs the
        # A/B pass) no longer collide in one sentence.
        # wave 13 (2026-09-24, round 10 §1.10): the cancel screen only ever renders
        # ONE arm (B never started), so an unconditional "the 0.5 mA arm's background
        # is visibly brighter" was false on that screen -- the two "share one colour
        # scale" clauses (one generic, one naming arm B by value) are merged into a
        # single clause gated on "when two arms run", true whether one or both render.
        screen_note=("dB rel. peak; clip follows the frame's median floor + 3 dB; "
                     "when two arms run, they share one colour scale down to the "
                     "deeper arm's floor (the printed zmin), and the 0.5 mA arm's "
                     "background reads visibly brighter than the 8 mA arm's, about "
                     "twelve dB by the printed numbers, streaks matching; all 1024 "
                     "elements share one front-end config; " + _ARRAY_DISCLOSURE),
        # THE RANGE CLAUSE IS GONE FROM HERE, deliberately (shard 3, 2026-09-24). It
        # used to read "1.00 m/gate; display 0-125 m of a 250 m unambiguous window" --
        # typed, and by today wrong twice: the convention is the owner's bistatic excess
        # path (ballot 2B), which doubles the window to 499.6 m with 249.8 m shown, and
        # the grid is endpoint-inclusive (F97d). The panel's own caption COMPUTES all of
        # it from the frame's freq_plan (`pipeline_runner._spine_range_meta`), so this
        # card now has no second, typed copy to drift from it. One authority per
        # question -- and a metre on a card is exactly the kind of number that drifts.
        say=[
            # wave 11 (2026-09-24): runbook.py's "While it runs, say" line is
            # preset.say[0] -- it must carry the shared-colour-scale story, not
            # the old "look alike" framing. Folds in the old first bullet's
            # error bars (+-0.6-0.9 dB) rather than keeping both (card word
            # budget).
            "With the shared colour scale, arm B's background reads about "
            "twelve dB brighter than arm A's (+-0.6-0.9 dB); streaks match.",
            # RETRACTED as written (shard 3, 2026-09-24): "default signal level (1e-5)"
            # named a drive that is no longer the default and no longer means the same
            # thing -- on the beat-record placement 1e-5 is a WORKING point (49.0 /
            # 37.1 dB, an 11.9 dB A/B), not a level where the knobs do nothing. What is
            # still true, and is the point the bullet was making, is that the knobs go
            # quiet once the drive is far enough above the front end's own floor:
            # measured on this preset, the A/B collapses to 0.96 dB at 1e-3 and 0.08 dB
            # at 1e-2.
            "Drive it harder and these knobs stop mattering: the A/B is 11.7 dB at "
            "3e-5, 0.96 dB at 1e-3, 0.08 dB at 1e-2 (2026-09-24).",
            "Below ~4 mA the LNA is a LOSS stage (-8.5 dB); most of the 12 dB "
            "leaves the attenuator (4->8 mA: +1.6 dB).",
            "No trade-off today: nothing clips; the IF filter only sets noise "
            "variance (1 MHz = 1 ms sweep, 20 us at 50 MHz).",
            # wave 9 (2026-09-24): "brightest band" was wrong -- the hottest visible
            # pixel on this map is the 37 m return, not range 0; reworded to Thrust 4's
            # framing (the display-normalisation fact, not a brightness claim).
            # wave 10 (2026-09-24, item 1.7, hostile round 9): the panel's own
            # subtitle prints "36 m" (1 m display gates round the true 37.1 m delay,
            # F94) -- the card told the presenter to read a number off the screen
            # that was not the one printed there.
            # wave 12 (2026-09-24, item 1.4): "~36 m on the panel" read as "the
            # panel prints 36" when it prints 37 -- reworded to name what the
            # panel actually shows without asserting a specific digit.
            # BISTATIC METRES (owner ballot 2B, 2026-09-24): the munich link is
            # bistatic and the axis is EXCESS PATH LENGTH c*tau, so F94's 37.1 m and
            # 68 m families read 74.2 m and 136 m of excess path. Same returns, same
            # delays, one convention -- and the panel's axis now says "excess path (m)"
            # so the card and the screen cannot disagree about which quantity it is.
            "Excess path 0-4 m is not a target: it is the direct path the display "
            "normalises to (0 dB). Multipath: the ~74 m return the panel names "
            "(74.2 m; F94's 37.1 m at c*tau/2) and 136 m.",
            # wave 10 (2026-09-24): merged the old "noise figure" and "what
            # end-to-end buys over Friis" bullets -- both turned on the same 0.17 dB
            # agreement figure.
            # wave 12 (2026-09-24): trimmed for the 450-word card ceiling.
            "Noise figure IS quotable: Friis 11.97 vs measured 11.80 dB "
            "validates the mechanism and this knob's downstream effect. "
            "Absolute dBm is NOT: the input level is free.",
            "Channel mismatch: all 1024 elements share one config; mismatch is "
            "structurally zero; a per-element spread is a small change.",
            # Re-derived on the unified receiver: the spine's cube bin is 9.99 cm of
            # EXCESS PATH (c*tau on the endpoint-inclusive grid, F97d -- not the 10 cm a
            # nominal 3 GHz would give), and the display bins 2501 of them 10:1 into
            # 1.00 m gates. The old line said "5 cm, binned 20:1" -- the same physical
            # resolution stated in the other convention over the uncropped 5000-bin
            # transform.
            "Spacing: lambda/2 at 30 GHz (0.525 at 31.5) -- grating lobes "
            "beyond |sin theta| ~0.90; 9.99 cm excess-path bins, 10:1 to 1.00 m "
            "gates.",
            # wave 9 (2026-09-24): peak-to-median caveat (also on Thrust 2's card) --
            # applies here too, since this is exactly the statistic the LNA-bias knob
            # moves.
            "Peak-to-median measures how empty the map is (median = empty gates), "
            "not target SNR; it moves with the noise floor this knob changes.",
            # wave 9 (2026-09-24, orchestrator course-correction): the 0 dB reference
            # cell (range-0 gate) is ~1.6 px tall on screen, anti-aliased away -- the
            # colour bar's 0 dB is never actually visible on the map. Numbers not
            # typed here on purpose: the other coder prints them on the panel
            # subtitle.
            # wave 12 (2026-09-24): dropped "(read it off the screen)" -- now
            # redundant with the dedicated pause-before-reading bullet below.
            "The 0 dB reference is a single range-0 gate too small to see; "
            "every dB is relative to the direct path.",
            # wave 10 (2026-09-24, item 4.1, hostile round 9): the noise floor
            # moves 11 dB between this slide and the next; name the cause before
            # the room asks.
            "Thrust 1 runs at signal_scaling 3e-5 (legacy): peak-median ~58 dB, "
            "~11 dB below Thrust 2's ~77 dB -- a different operating point. That "
            "drive is a DISPLAY choice, not an input level (Details).",
            # SEAT'S READ OF THE 2026-09-24 RENDERS, item 1d. The obvious question from
            # the floor is why a hand-picked drive is used when the block offers a
            # `physical` mode. Answered by measurement, taken 2026-09-24 on this preset:
            # forcing scale_mode='physical' on the munich Ka source (which itself reports
            # physical_scale=False -- the frames are not volts) puts both arms at ~69-74
            # dB and INVERTS the A/B to -3.0/-3.6 dB, i.e. the 0.5 mA arm reads better.
            # And the drive is inside F97b's exact regime with room to spare: the
            # front end's phase-equivariance error on munich Ka frame 0 is 5.4e-07 at
            # 3e-5 (float32 rounding), 2.1e-06 at 1e-4, and only reaches 2.1e-04 at 1e-3.
            # wave 12 (2026-09-24, item 1.5): the panel subtitle and its
            # peak-median callout rebuild every 700 ms while the Results clock
            # loops -- a presenter reading a number off a moving screen without
            # warning reads as reciting a fixed figure that then contradicts
            # itself on the next loop.
            "The printed statistics update per frame while the panels loop; "
            "pause before reading one.",
            # wave 12 (2026-09-24, items 4.1/4.2): the shared-scale rule runs in
            # opposite directions here vs Thrust 5's Range-Doppler; trimmed for
            # the 450-word card ceiling.
            "Shared-scale direction flips by screen (munich: deeper floor; "
            "Thrust 5 Range-Doppler: tighter clip); a fainter streak is the "
            "floor rising, contrast not level.",
        ],
        do_not_say=[
            "Any DC power readout: PRX is U-shaped, MINIMUM at best quality.",
            "The compression regime (scaling 1e-1..1e-3): worse live than silence.",
            "That gm scales linearly with bias -- true only at 8 mA; say "
            "'constant-overdrive scaling'.",
            "Anything about IIP3: constant across 0.5-10 mA.",
            "Any gain knob: peak normalization removes it.",
            "Any absolute dBm sensitivity: the input scale is arbitrary.",
            # wave 10 (2026-09-24, item 2.6, hostile round 9): RETRACTED "it is not
            # visible" -- false on some frames (frame 2 shows a yellow 1-px stripe
            # at range 0). Reworded to a claim that holds on every frame: the
            # stripe, when it shows, is the 0 dB reference, not a target. The other
            # coder is matching the panel subtitle to this wording.
            "That the thin stripe at the bottom edge, when it shows, is a target "
            "-- it is the range-0 direct path, the 0 dB reference.",
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
        # wave 11 (2026-09-24): the Results screen now shares one colour scale
        # between A/B (owner's live test) -- "images move a few tenths of a dB"
        # still holds (T2's images genuinely don't move much), but the framing
        # now names the shared scale so the presenter reads it as a deliberate
        # display fact, not silence.
        # wave 12 (2026-09-24, item 1.1): "A, top / B, bottom" -> "A, left / B,
        # right" -- the A/B arms render side by side, not stacked.
        blurb=("Press Run once (A, left, mantissa 6 bit / B, right, 1 bit); the "
               "tracker panel plots a subspace-error curve against a dashed "
               "0.06 reference line -- A settles on it, B sits about 5x above. "
               "With the shared colour scale, compare the backgrounds; any "
               "difference at the ~0.1 dB run-to-run floor is not the knob. "
               "Manual: AFE mantissa 6 -> 1 bit."),
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
        # wave 12 (2026-09-24, item 1.6): the note still named the pre-wave-11
        # per-arm adaptive clip; the panels now share one colour scale instead.
        screen_note=("range-azimuth and range-elevation images barely move under the "
                     "shared colour scale (statistics printed on each; the range "
                     "calibration is on the panel); the tracker error moves "
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
            "No detection metric is wired here; say so before asked what it means "
            "for P_d or false alarms.",
            # wave 10 (2026-09-24, item 1.7, hostile round 9): see Thrust 1's card
            # for the same fix -- the panel prints "36 m" (37.1 m true delay, F94).
            # wave 12 (2026-09-24, item 1.4): same reword as Thrust 1's card.
            "Range 0-2 m is not a target: it is the direct path the display "
            # BISTATIC METRES (owner ballot 2B, 2026-09-24): the axis is EXCESS
            # PATH LENGTH c*tau over the line of sight, so F94's 37.1 m and 68 m
            # families read 74.2 m and 136 m. Same returns, same delays, one
            # convention -- and the panel's own axis now says "excess path (m)".
            "normalises to (0 dB); multipath: the ~74 m return the panel names "
            "(74.2 m; F94's 37.1 m at c*tau/2) and 136 m.",
            "Tracker k re-picked: k=8 (old default) and k=4 spike mid-run on the "
            "Ka retrace (rank 3-4, F94); k=2 is the largest stable k.",
            "Spacing: lambda/2 at 30 GHz, 0.525 lambda at 31.5 GHz -- grating lobes "
            "beyond |sin theta| ~0.90; 9.99 cm excess-path bins, 10:1 to 1.00 m "
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
            # wave 12 (2026-09-24): dropped "(read it off the screen)" -- now
            # redundant with the dedicated pause-before-reading bullet below.
            "The 0 dB reference is a single range-0 gate too small to see; "
            "every dB on the map is relative to the direct path.",
            # wave 12 (2026-09-24, item 1.5): same warning as Thrust 1/4 -- the
            # printed peak-median numbers rebuild per frame while the clock loops.
            "The printed statistics update per frame while the panels loop; "
            "pause before reading one.",
        ],
        do_not_say=[
            "That the mantissa sweep models analog hardware error: AFEBlock's "
            "WEIGHT_FLOAT is right for a compute datapath, wrong for analog "
            "control (compress.py).",
            # wave 11 (2026-09-24): "the picture does not respond" is no longer
            # the honest framing now the two panels share one colour scale --
            # replaced with what to actually look at.
            "That the picture does not respond: with the shared colour scale, "
            "compare the backgrounds; any difference at the ~0.1 dB "
            "run-to-run floor is not the knob -- the tracker curve is the "
            "story.",
            "That a higher compression ratio looks better: 512 of 1024 lets the "
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
        # wave 12 (2026-09-24, item 1.9): "settling near 0.16" and the trajectory's
        # closing "about 0.2" named two different values for arm A's settled
        # level on the same card; the screen settles at 0.16-0.17 from frame 5 --
        # one number now, stated once.
        blurb=("Cold start on BOTH arms, k=2 (k=4 spikes ~0.98, see say). Arm A: "
               "FIXED 5 passes/frame, never reaching B's ~0.06 "
               "floor in 8 frames -- about 2.5x higher. Arm B: the shipped "
               "adaptive gate, 10 passes/frame baseline (right axis 0-12; a "
               "small-gap file would climb to 60, not this one). Over 8 frames: A "
               "about 0.6 -> 0.31 -> settles about 0.16-0.17 from frame 5; B about "
               "0.30 -> 0.09 by frame 2, settled from frame 3. It never escalates "
               "here: k=2's gap stays well clear of 0.01."),
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
            # wave 10 (2026-09-24, item 1.4, hostile round 9): RETRACTED
            # "frames-to-acquire ... one frame sooner" -- arm A never reaches B's
            # floor in 8 frames (the blurb's own honest claim), so
            # frames-to-acquire has no value for A and the old bullet contradicted
            # the blurb on the same card. The A/B statistic is the settled FLOOR at
            # two fixed pass counts, not a race.
            "The A/B statistic is the settled floor at 5 vs 10 passes/frame -- A "
            "never reaches B's, about 2.5x higher throughout.",
            "This is 2:1 compression (m=512 of 1024). At 16:1/64:1 neither arm "
            "converges in this many frames -- why m is not a live knob here.",
            "There is deliberately no image here: the picture doesn't change during "
            "acquisition -- the error curve shows what the tracker hasn't learned "
            "yet. Without the AFE it looks identical -- do not toggle it.",
            "Prepared answer -- 'does your gap diagnostic work at Ka?': at k=2 the "
            "gap sits far above 0.01, so it never escalates -- the 2x on screen IS "
            "baseline. At k=4 (not shipped) the gap collapses, the gate spends 6x "
            "more, but the cluster still spikes -- 'mitigated'.",
            "The run is not faster than a full SVD: scoring runs the full SVD every "
            "frame. The 45x microbenchmark is real, the run time is not.",
            "Spacing: lambda/2 at 30 GHz, 0.525 lambda at 31.5 GHz -- grating lobes "
            "beyond |sin theta| ~0.90; 9.99 cm excess-path bins (F97d), 10:1 to "
            "1.00 m gates.",
            # wave 9 (2026-09-24, item 3.12): see the same note on Thrust 2's card.
            "All 1024 elements share one front-end config (Thrust 1); a spread would "
            "show up in the tracker's acquisition curve here, not Thrust 1's "
            "picture.",
            # wave 10 (2026-09-24, item 4.3, hostile round 9): prepared answer for
            # "why does one cold run hit the warm floor and the other never does?"
            "Prepared answer -- both arms are cold starts; the dashed line is a "
            "WARM-started settled level. Arm B (10 passes) reaches it from cold; "
            "arm A (5 passes) does not.",
        ],
        do_not_say=[
            "Anything with an interferer: three confounders, and the sign of the "
            "response flips with a knob not on screen.",
            "'k = 2 is the optimum' in raw subspace_err: the metric ceilings at "
            "sqrt(2) = 1.41.",
            "The AFE on/off toggle as 'the effect of adaptive feature extraction': 2:1, "
            "costs 3 dB, the picture looks identical.",
            "subspace_err beside an m=16 run: the warm tracker's error GROWS with frames "
            "there (0.125 -> 0.807 over six).",
            "That the gate 'always fires' or 'escalates' at Ka: false here (flat 5 "
            "on A / 10 on B, 8/8 frames). F94's escalation claim was pre-fix "
            "(8d1e251), at k=8 -- retracted at k=2.",
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
        # wave 10 (2026-09-24, item 1.2, hostile round 9): RETRACTED "on both arms" --
        # the 03:0x re-render's own median floor printed -50.3 (A) vs -50.4 (B), i.e.
        # the median floor is the statistic that DID move by ~0.1 dB, while
        # range-azimuth peak-median printed identically (76.6/76.6) that same run --
        # the opposite of what the sentence named. Generalised to "either panel" so
        # the card is never pinned to whichever one happened to hold still this time.
        # wave 11 (2026-09-24): named the shared colour scale (owner's live
        # test) as what to compare -- same physics as before (T4's images still
        # move at the run-to-run floor), just stated against the new screen.
        # wave 12 (2026-09-24): trimmed for the 450-word card ceiling.
        blurb=("THE HONEST STORY: the interconnect is NOT the limiting element "
               "here; compare the backgrounds under the shared colour scale -- "
               "any ~0.1 dB difference between the arms' printed statistics "
               "(either panel) is the run-to-run floor, not the knob. This is "
               "the LIVE public Tessera/UIC TSV surrogate "
               "(InterconnectBlock(source='tessera'), scale x2 / half frequency "
               "at Ka band). "
               f"Arm A: {_TESSERA_CANONICAL_HEIGHT_UM:g} um presented "
               f"(= {_TESSERA_CANONICAL_HEIGHT_MODEL_UM:g} um model geometry). "
               "Arm B drops TSV height to its presented low end -- OFFLINE the "
               "biggest single-knob mover of the skirt (3.53 dB native "
               "flat-frame move) -- bulk DELAY, sits below the printed median "
               "floor."),
        live_knobs=[("interconnect", "tessera_height_um",
                     f"{_TESSERA_CANONICAL_HEIGHT_UM:g} -> {_TESSERA_ARM_B_HEIGHT_DISPLAY:g} um "
                     "(the A/B above)"),
                    ("interconnect", "source",
                     "manual third option, not part of the A/B: source='default' + "
                     "case='default' selects the old SYNTHETIC 11-tap boxcar placeholder")],
        ab=("interconnect", "tessera_height_um", _TESSERA_ARM_B_HEIGHT_UM),
        ab_label_a=f"canonical Tessera geometry ({_TESSERA_CANONICAL_HEIGHT_UM:g} um presented)",
        ab_label_b=f"TSV height -> {_TESSERA_ARM_B_HEIGHT_DISPLAY:g} um presented (largest skirt mover)",
        # wave 10 (2026-09-24, item 1.8, hostile round 9): RETRACTED "sits ~50 dB
        # below this display's real noise floor" -- the range-profile axis runs 0 to
        # -60 dB and prints its own median floor at -50.3; the quoted offline skirt
        # levels (-53.90 -> -57.43 dB) are INSIDE that range, not 50 dB below it. The
        # sentence right after it ("an offline flat-frame metric... not the statistic
        # the panel prints") already carries the honest claim; the retracted one is
        # deleted, not reworded.
        # wave 10 (item 1.3/1.11, hostile round 9): the crosstalk clause used to quote
        # MODEL pitches (40/60 um) while the run-notes line four lines above quotes
        # PRESENTED geometry (pitch 30 um) for the same shipped design -- one screen,
        # two unit conventions. Converted to presented (model value in parens),
        # computed from the imported Ka scale, never a hand-typed "/2". Also matched
        # the run-notes line's own vocabulary: "ring3x3 (one signal via)", not
        # "single-via" (both true, e2e/interconnect_surrogate/tessera.py; two
        # vocabularies on one screen was the finding, not a factual error).
        # wave 13 (2026-09-24, layout redesign): "under the banner" was the
        # pre-redesign banner; the run-notes line is now only reachable inside each
        # arm's Details disclosure (the caption slot under the banner is spent on the
        # TSV height value, which overflows the arm chip on this preset -- see
        # `_ab_arm_chip_overflow`/`_arm_caption` in webapp/app.py, confirmed on the
        # rendered Results PNG: the scale/frequency clause does not appear until
        # Details is opened).
        screen_note=("LIVE Tessera surrogate, scale model x2 (see the run-notes "
                     "line inside each arm's Details disclosure); in-band "
                     "|S21| moves <0.03 dB across every knob -- invisible on a "
                     "peak-normalized display. The 3.53 dB skirt figure on the card "
                     "is an offline flat-frame metric (bulk delay, not distortion), "
                     "not the statistic the panel itself prints. With the shared "
                     "colour scale, compare the backgrounds; any difference at "
                     "the ~0.1 dB run-to-run floor is not the knob. "
                     "Crosstalk (NEXT/FEXT) is modelled for a multi-via arrangement, "
                     "not this default ring3x3 (one signal via): worst-pair band "
                     "mean, checker3x3, 28.5-31.5 GHz, pitch "
                     f"{_TESSERA_CROSSTALK_TRAINING_PITCH_UM:g} um presented "
                     f"({_TESSERA_CROSSTALK_TRAINING_PITCH_MODEL_UM:g} um model, in "
                     "training box) NEXT -31.5 / FEXT -39.2 dB; pitch "
                     f"{_TESSERA_CROSSTALK_SHIPPED_PITCH_UM:g} um presented "
                     f"({_TESSERA_CROSSTALK_SHIPPED_PITCH_MODEL_UM:g} um model, our "
                     "shipped geometry) NEXT -30.3 / FEXT -34.5 dB -- the pitch "
                     "trend itself inverts above ~23 GHz on this public checkpoint "
                     "(F89), so these are fixed reference values for checker3x3, "
                     "not numbers from this run."),
        say=[
            # wave 12 (2026-09-24, item 1.7): folded in the one on-screen pointer
            # for "did arm B actually run?".
            # wave 13 (2026-09-24, layout redesign): RETRACTED "the run-notes line
            # under each banner names the scale factor and frequency, and is the one
            # place arm B's height shows on screen" -- read on the rendered PNG, the
            # scale/frequency clause is Details-only on this preset (the caption slot
            # under the banner is spent on the TSV height VALUE instead, since the
            # arm chip itself overflows); that value IS the one place the height
            # shows without a click, so the two facts are now attributed correctly.
            "This is the LIVE public Tessera/UIC surrogate (checkpoint, not a "
            "CSV); scale factor and frequency are in each arm's Details. "
            "Arm B's height shows without a click, in the caption under "
            f"each banner: {_TESSERA_CANONICAL_HEIGHT_UM:g} um (A) vs "
            f"{_TESSERA_ARM_B_HEIGHT_DISPLAY:g} um (B).",
            "Credit UIC by name (Mohamed Gharib, Leonid Popryho, Inna Partin-Vaisband; "
            "doi 10.1109/TCAD.2026.3718807) -- block, wrapper and six S21 CSVs are "
            "theirs.",
            "In-band |S21| is invisible on this display (<0.03 dB span); A/B "
            "moves TSV height because it measurably moves the skirt.",
            "Crosstalk is now modelled -- NEXT/FEXT between vias -- with F89's "
            "numbers on the screen note; per-ELEMENT broadcast is still "
            "unmodelled.",
            "77 GHz shipped CSVs are not reconciled with the 30 GHz frames -- "
            "caption real-CSV results shape-only.",
            # wave 11 (2026-09-24, F96): the display crops the negative-delay
            # half of the 250 m FFT window at 125 m -- the rise there is the
            # range-0 skirt's negative-delay side folding into the crop edge,
            # not a "wrap".
            "Range 0-2 m is not a target: range 0 = earliest arrival "
            "(normalize_delays=True). Peaks near 37-113 m are multipath; the "
            "rise at the window's top is the range-0 skirt's negative-delay side "
            "at the crop edge (F96).",
            "Skin depth goes as f^-1/2, not f^-1: conductor loss under-estimated "
            "by sqrt(2) (~0.2 dB of 0.5 dB loss), substrate coupling up to 2x; "
            "trends/shape exact (F91).",
            "Spacing: lambda/2 at 30 GHz, 0.525 lambda at 31.5 GHz -- grating lobes "
            "beyond |sin theta| ~0.90; 9.99 cm excess-path bins, 10:1 to 1.00 m "
            "gates.",
            # wave 9 (2026-09-24, orchestrator course-correction): same line as
            # Thrust 1/2 -- see Thrust 1's comment for the measurement it stands
            # in for.
            # wave 12 (2026-09-24): trimmed, redundant with the pause bullet below.
            "The 0 dB reference is a single range-0 gate too small to see; "
            "every dB on the map is relative to the direct path.",
            # wave 10 (2026-09-24, item 4.2, hostile round 9): the prepared answer
            # for "then why is this a thrust?" -- a negative result stated as such,
            # not hidden behind "THE HONEST STORY" alone.
            # wave 12 (2026-09-24): trimmed -- the blurb already states "the
            # interconnect is NOT the limiting element", so this bullet no
            # longer repeats it.
            "What Thrust 4 DID establish: six knobs run live end to end, and "
            "in-band |S21| moves <0.03 dB across all -- a negative result, "
            "stated as one.",
            # wave 12 (2026-09-24, item 1.5): same warning as Thrust 1/2 -- the
            # printed panel statistics rebuild per frame while the clock loops.
            "The printed statistics update per frame while the panels loop; "
            "pause before reading one.",
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
               "and ADC all run LIVE from that channel, then CA-CFAR. Press Run "
               "once: A is the 12-bit ADC the corpus was generated at, B the same "
               "frames re-digitised at 3 bits. Measured over 5 frames: 16 hits / "
               "45 unmatched (9.0/frame) at 12 bits, 13 / 36 (7.2) at 3 bits. "
               "Thresholds are each detector's recall-0.5 point on the 172-frame "
               "split; on 5 frames recall varies, so compare the 172-frame "
               "FA/frame rows, not the crosses."),
        live_knobs=[("quantizer", "bits", "12 -> 3 (the ADC is re-run, not re-loaded)"),
                    ("detector", "threshold", "0.66 -> 0.8 (fewer detections)")],
        ab=("quantizer", "bits", 3),
        ab_label_a="12-bit ADC (as built)",
        ab_label_b="3-bit ADC (same frames)",
        screen_note=_T5_SCREEN_NOTE,
        say=[
            "SAY FIRST: the frames change -- Thrusts 1-4 ran munich (249.8 m, "
            "range-azimuth); this is the benchmark corpus (100 m, "
            "range-Doppler). STORED is the ray-traced channel; everything after "
            "runs live, so the ADC knob reaches the detector.",
            "The gate that makes this honest: at generation settings the live "
            "cube is BIT-IDENTICAL to the stored one (max |diff| = 0 ADC codes); "
            "moving a knob breaks that -- the whole demonstration.",
            "Classical CFAR scores AP 0.218, a val-tuned CFAR 0.326, chance "
            "floor 0.093; these 5 frames demonstrate, not measure.",
            # wave 9 (2026-09-24, item 1.7): the old line was wrong on both counts --
            # the detector map spans 0-50 m (a 10 m unscored strip above the 40 m
            # dashed line), and the 0-100 m panel is Range-Doppler power, unlabelled.
            # wave 11 (2026-09-24): the Results screen now auto-plays every
            # animated panel on one shared clock -- the Range-Doppler cube
            # loops while the detector/scoreboard/PR panels hold the last
            # frame; pause the cube before pointing at one frame's crosses.
            "The CFAR map spans 0-50 m; the top 10 m is unscored. Detector, "
            "scoreboard and PR panels hold the LAST "
            "frame; the Range-Doppler cube loops beside them -- pause it "
            "before discussing one frame's detections.",
            "Ground truth omits ~3 real objects per frame inside 40 m, so a "
            "detector catching every real object caps precision at 0.64 -- "
            "some 'false alarms' are real.",
            "Unambiguous velocity is +-v_max from the manifest (~9.7 m/s); corpus "
            "targets are slower by construction, so a 20 m/s car would alias.",
            "Unmatched detections can DROP at deeper quantisation: quantisation "
            "noise raises the CA-CFAR estimate, so fewer weak peaks clear "
            "threshold -- a loss of sensitivity, not a quality gain.",
            # wave 9 (2026-09-24, T5 item 5): prepared answer, shared across the three
            # T5 cards.
            "This detector sits at its 172-frame recall-0.5 point, yet gives "
            "0.53 recall here; matched-recall FA comparisons use that "
            "172-frame split -- 5 frames cannot reproduce a recall.",
            # wave 12 (2026-09-24, item 4.3): how the cube becomes a map, read
            # from classical_detection_map/cfar_objectness (e2e/ml/baseline.py):
            # the Doppler axis is summed away first (non-coherent integration,
            # the shipped default), leaving one 2-D range-azimuth power map;
            # CA-CFAR's guard/train cells then form ONE square annulus over
            # range AND azimuth together, not two separate 1-D passes.
            "How the cube becomes the map: Doppler sums away first (angle FFT) "
            "into a range-azimuth map; CA-CFAR's guard 2 / train 6 cells form "
            "one square annulus over range and azimuth, not two 1-D passes.",
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
            "0.105/0.093: every number on this screen is the Ka scoring "
            "(beat_cfar_ka.json, 2026-09-24); the 77 GHz figures (0.127, 0.138, "
            "0.229) belong to a different corpus and band.",
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
               "input; offline test AP 0.105 vs CFAR's 0.218). Its objectness map is "
               "a range-profile x fixed-azimuth-prior STRIPE, not peaks: the network "
               "never learns azimuth (F83). A/B moves the IF high-pass corner from "
               "1 m (real receiver) to a deliberately broken 25 m, attenuating (not "
               "discarding) returns inside 25 m -- ~4.3 dB at ~22 m: unmatched/frame "
               "falls sharply, and hits with it. Threshold is pinned at "
               "recall-0.5 (0.22); at default 0.5 this checkpoint draws nothing."),
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
        + " Loses to CFAR 0.105 vs 0.218, shown on purpose.",
        say=[
            "The learned detector LOSES to CFAR: 0.105 vs 0.218, chance floor "
            "0.081. Say it first.",
            "B is not a plausible receiver -- a 25 m high-pass corner -- the "
            "point is a front-end setting reaches the detector at all: "
            "unmatched/frame and hits both fall.",
            "At A's operating point, offline expects ~29 crosses/frame = 26.4 FA "
            "+ 3.0 hits (beat_cfar_ka.json); these 5 live frames give their own "
            "unmatched/frame -- same regime, not the same number.",
            "Both ported networks emit a near-separable f(range)*g(azimuth) map: "
            "rank-1 energy 0.89/0.76 vs 0.31 for ground truth. Under azimuth-only "
            "matching they score no better than a constant map.",
            "F83's mechanism: neither head converts channel phase into an angle "
            "bin. An architecture with range x azimuth as its spatial plane "
            "scores 0.468 and passes controls this one fails -- load RADDetNet, "
            "read its caveats first.",
            "This checkpoint trained on a different corpus; its rd input scaling "
            "comes from that corpus (screen note). Moving a knob takes it "
            "further out of its training distribution.",
            "On Arm B some frames score TP = 0: the mechanism on display, not an "
            "accident -- the 25 m corner attenuates the same near-range returns "
            "this checkpoint was trained to fire on.",
            # wave 9 (2026-09-24, T5 item 4): the honest line for why the 4.3 dB
            # number cannot be read off the picture.
            "You cannot see the 4.3 dB: both maps are peak-normalised and the "
            "peak sits inside 25 m; you see the floor coming up relative to a "
            "peak attenuated along with it.",
            # wave 9 (2026-09-24, T5 item 5): prepared answer, shared across the
            # three T5 cards.
            # wave 11 (2026-09-24): replaced the "no slider here" clause -- the
            # Range-Doppler panel now auto-plays and loops on the shared clock
            # rather than sitting still; the detector panel still holds the
            # last frame regardless.
            "This detector sits at its 172-frame recall-0.5 threshold, yet "
            "gives 0.50 recall on these 5 frames (the LAST) -- 5 frames "
            "cannot reproduce a recall. Detector, scoreboard and PR panels "
            "hold that frame while Range-Doppler loops; pause it to discuss "
            "one frame.",
            # wave 10 (2026-09-24, item 3.7, hostile round 9): one name for the
            # detector, not three.
            "The PR legend's 'fftradnet_rd_b15' is this checkpoint "
            "(b15_fftradnet_rd_ka).",
            # RETIRED (shard 3, 2026-09-24, F95 addendum): this bullet existed for a
            # scoreboard row that no longer appears on THIS screen. The seed spread
            # 0.040 is a RADDetNet measurement (seeds 42 vs 43, 77 GHz, F86) and the
            # caption is now gated to that arm -- printing it here attributed one
            # architecture's training variance to a different network. In its place, the
            # fact that IS this screen's:
            "At Ka this arm scores 0.105 against a 0.093 chance floor -- +0.012 "
            "[+0.001, +0.026] above chance. It does not merely lose to CFAR; it "
            "barely beats random.",
        ],
        do_not_say=[
            "'The rad input doubles AP' or any 0.229 / 0.484 figure: retracted, F84.",
            "'We fixed azimuth': the stripe statistic refutes it on the next slide.",
            "That the 25 m corner measures receiver-design sensitivity: an "
            "illustration on 5 frames, not a sweep.",
            "That this benchmarks SSMRadNet/FFTRadNet: the fault is on our side "
            "(collaborator README).",
        ],
    ),
    DemoPreset(
        id="thrust5_detector_raddetnet",
        label="Thrust 5 (LEAD) - RADDetNet vs CFAR at Ka: AP 0.468 vs 0.218 "
              "(val-tuned 0.326), in-distribution",
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
        # wave 10 (2026-09-24, item 1.6, hostile round 9): "6.24 on the CFAR screen"
        # sent the presenter to another preset for a number printed on THIS one, same
        # row ("FA/frame at recall 0.5, 172 frames | 2.99 (CFAR 6.24)").
        blurb=("THE REAL CLAIM, first: at matched recall (0.5) on the 172-frame split, "
               "RADDetNet racks up fewer false alarms than CFAR -- 3.31 vs 10.35 "
               "FA/frame (both printed on this screen; AP 0.468 vs 0.218, "
               "controls pass). On the 5 live frames below, fewer crosses can mean "
               "fewer hits since these aren't recall-matched -- read the "
               "scoreboard's FA rows, not the crosses. The same live chain runs "
               "RADDetNet (Doppler as channels, range x azimuth as the spatial "
               "plane) on CFAR's cube. A/B re-digitises the stored channel at 3 "
               "bits: hits go 10 -> 5 -- the knob reaching the detector, not a "
               "ranking. Out of distribution the result is seed-dependent (F86); "
               "say so unprompted."),
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
            "Every offline number comes from e2e/ml/runs/beat_cfar_ka.json (seed 42, "
            "b1_bench_v3_ka, 12-bit default impairments); re-scored bit-identically. "
            "Paired scene bootstrap: +0.250 AP vs shipped CFAR (0.218), 95% CI "
            "[+0.145, +0.208]; +0.148 vs the best of nine classical baselines (0.328).",
            # wave 9 (2026-09-24, T5 item 5): folded the recall-0.5 caveat into this
            # bullet -- the card's say list is already at its 6-bullet cap.
            # wave 10 (2026-09-24, item 2.2, hostile round 9): folded in that the
            # detector panel has no slider (it's pinned to the last frame) -- the
            # say list is at its 6-bullet cap.
            # wave 11 (2026-09-24): replaced "no slider here" -- the
            # Range-Doppler panel now auto-plays/loops on the shared clock;
            # kept under the 45-word RADDetNet bullet cap.
            "The counts on screen are 5 live frames, LAST shown -- a "
            "demonstration, not a re-measurement of AP; recall here (0.33) "
            "cannot reproduce recall-0.5. Detector, scoreboard and PR panels "
            "hold that frame while Range-Doppler loops; pause it to discuss "
            "one frame.",
            "The controls are F83's, which the shipped nets FAILED (deranged-label "
            "retention 12%, CFAR 10%, shipped nets 48-51%); nine classical baselines "
            "were scored too, best 0.326 -- above shipped CFAR (0.218), but still "
            "behind RADDetNet (0.468, a +0.142 lead).",
            "Four learned arms were screened: three ported architectures and "
            "this one designed to the F83 diagnosis; all four are in "
            "beat_cfar_ka.json, none dropped.",
            # wave 10 (2026-09-24, item 4.4, hostile round 9): folded in the
            # seed-spread-vs-CI caveat -- the say list is at its 6-bullet cap, and
            # this is the OOD bullet the caveat actually belongs on.
            # wave 12 (2026-09-24, item 1.8): "3 seeds + the joint arm" read as a
            # bigger sample than the scoreboard shows -- it prints two seed rows
            # (0.208/0.153) plus the joint checkpoint, not three seeds.
            "THE CAVEAT: one Ka seed exists. The seed spread 0.040 (F86, at "
            "77 GHz) exceeds this arm's CI half-width 0.032, so the lead rests "
            "on one training run, not the CI alone.",
        ],
        do_not_say=[
            "'Beats CFAR', unqualified: the verified claim is in-distribution, on "
            "CFAR's front end (F85 addendum).",
            "That 5 hits at 3 bits vs 10 at 12 bits measures quantisation cost: "
            "5 frames at one threshold shows the knob reaches the detector, not "
            "measures it.",
            # wave 9 (2026-09-24, T5 item 1): the screen itself prints OOD and
            # 3rd-corpus rows now, so "anything" contradicted what is on screen --
            # reworded to don't-volunteer-but-read-what's-printed.
            # wave 10 (2026-09-24, item 1.5, hostile round 9): RETRACTED the "0.487 on
            # v2" clause -- 0.487 is the JOINT checkpoint's number (trained partly on
            # v2), not this seed-42 checkpoint's; the card's own `say` list already
            # names a DIFFERENT number for THIS checkpoint on v2 (0.208), so the two
            # "v2" statements read as contradictory. Dropped, not disambiguated.
            "Do not volunteer generalisation/robustness; if asked, read the OOD and "
            "3rd-corpus rows as printed (a third corpus never trained on; the lead "
            "holds) and stop there.",
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
