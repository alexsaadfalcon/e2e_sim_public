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

from webapp.corpus_catalog import DEFAULT_CORPUS
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
#: recall-0.5 points so the cross counts on screen are the false-alarm comparison the
#: cards quote (6.2 / 26 / 3.0 per frame), not three arbitrary thresholds.
CFAR_THRESHOLD = 0.66

#: Decode threshold for that checkpoint: its recall-0.5 operating point, objectness
#: 0.22 (`beat_cfar.json`). At the registry default 0.5 the checkpoint draws NO
#: detections on the test frames -- the "demo landmine" in DEMO_DEFENSE.md, measured
#: again on the real frames. Pinned here so the figure is never blank.
ML_THRESHOLD = 0.22

#: The bridge corpora (verified 2026-09-23): 50 scenes each, seed 4242, generated into
#: ONE path with a rename between runs so the scene salt is shared; splits 40/5/5. The
#: 5 TEST frames carry IDENTICAL target lists frame by frame (checked via each frame's
#: `meta["targets"]`) and differ ONLY in ADC quantizer resolution -- per-frame
#: `meta["quant_snr_db"]` ~57-58 dB (12-bit) vs ~9-10 dB (4-bit). This is the one T5
#: screen where a front-end setting (ADC bits) reaches a detection count, and it does so
#: baked into the corpus at generation time, not on a live knob.
BRIDGE_CORPUS_12BIT = "e2e/ml/datasets/b1_bridge_12bit/benchmark_v1_D2/manifest.json"
BRIDGE_CORPUS_4BIT = "e2e/ml/datasets/b1_bridge_4bit/benchmark_v1_D2/manifest.json"

#: Shared Results-tab screen note for the three Thrust 5 detector presets (hostile-
#: expert third read, 2026-09-23; fourth read, 2026-09-23: prepended the "frames:
#: ..." clause below so the note itself says the RF chain is bypassed, not just the
#: `say` list). "{VMAX_CLAUSE}" is filled in (or dropped, if the manifest cannot be
#: read) at render time -- see `_read_corpus_v_max` in webapp/app.py -- so the number
#: is never typed here. Kept terse (measured against the rendered PNG, 2026-09-23) so
#: the whole line -- including the per-preset addition on `thrust5_detector_ml` --
#: still fits one line at 16 px on the 1600 px results page.
_T5_SCREEN_NOTE = (
    "frames: stored ADC corpus (benchmark_v1_D2), replayed -- the RF chain of "
    "Thrusts 1-4 is bypassed; 40 m; in-distribution; seed 42 of two (0.476 / 0.436){VMAX_CLAUSE}."
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
        blurb=("Press Run once: both arms run and appear as before (A, top, 8 mA) / after "
               "(B, bottom, 0.5 mA), each panel printing its own peak-median statistic "
               "(about 42 vs 30 dB). The range-azimuth image loses dynamic range as the "
               "front-end's own noise rises. The signal is deliberately set just below "
               "the model's input-referred noise (1e-7 vs 1.36e-7 V) -- a real "
               "1024-element radar's per-element SNR, recovered by coherent gain. Manual "
               "path: LNA bias is the A/B above; second knob: IF bandwidth 15 -> 50 MHz "
               "(about -5 dB; 1 -> 50 MHz is -16 dB)."),
        live_knobs=[("rffe", "lna_bias_ma", "8 -> 0.5 mA (about -12 dB)"),
                    ("rffe", "if_bw_mhz", "manual second knob: 15 -> 50 MHz and run again "
                                          "(about -5 dB; 1 -> 50 MHz is -16 dB, measured "
                                          "2026-09-21)")],
        # A/B (Change 1, 2026-09-22 hostile-expert read): as-loaded IS the 8 mA arm;
        # run B drops to 0.5 mA, the direction the card's headline (+12 dB) quotes.
        ab=("rffe", "lna_bias_ma", 0.5),
        ab_label_a="8 mA", ab_label_b="0.5 mA",
        screen_note=("dB rel. peak on every panel; range 0 = earliest arrival (delays "
                     "normalised at generation); all 1024 elements share one front-end "
                     "config."),
        say=[
            "LNA bias 0.5->8 mA is worth about +12 dB (+-0.6-0.9 dB).",
            "At default signal level (1e-5) these knobs do nothing (0.5 dB, under the "
            "40 dB floor); show it if asked.",
            "Below ~4 mA the LNA is a LOSS stage (-8.5 dB at 0.5 mA); most of the 12 dB "
            "is leaving the attenuator regime. Sub-claim: 4->8 mA = +1.6 dB (measured "
            "2026-09-22).",
            "There is no trade-off today: nothing clips; the IF filter only sets noise "
            "variance. Missing half: 1 MHz IF is a 1 ms sweep vs 20 us at 50 MHz; a "
            "20 m/s car moves two wavelengths in that time.",
            "The brightest band at range 0-2 m is not a target: Sionna's "
            "normalize_delays=True subtracts the shortest path's delay (see the screen "
            "note); the 20-22 m stripe is real drifting multipath (2-21 m across "
            "frames), not fixed.",
            "Noise figure IS quotable: Friis gives 11.97 dB, the measured end-to-end "
            "floor is 11.80 dB -- a 0.17 dB agreement through 1024 elements, the FFT "
            "chain, the AFE and the tracker (2026-09-21). Absolute sensitivity in dBm "
            "is NOT: the input level is a free parameter, so quote noise figure and "
            "relative dB only.",
            "Channel mismatch: all 1024 elements share one config (get_RX_config "
            "broadcasts one value), so element-to-element gain/phase mismatch is "
            "structurally zero here; rx_config is already per-element, so adding a "
            "spread is a small change and is on the list.",
            "What the end-to-end run buys over Friis: the 0.17 dB agreement validates "
            "the noise mechanism; what Friis cannot give is the coupling downstream -- "
            "the same knob's effect on the AFE, the tracker (Thrust 2/3) and the "
            "detector.",
        ],
        do_not_say=[
            "Any DC power readout: PRX is U-shaped, MINIMUM at best quality, implying an "
            "8.45 V rail in a 100-200 mV chain.",
            "The compression regime (scaling 1e-1..1e-3): worse live than silence.",
            "That gm scales linearly with bias -- true only at 8 mA (a weak-inversion "
            "law); say 'constant-overdrive power scaling'.",
            "Anything about IIP3: constant to five decimals across 0.5-10 mA here, unlike "
            "a real LNA whose IIP3 improves with bias.",
            "Any gain knob: peak normalization removes it.",
            "Any absolute dBm sensitivity: the input scale is arbitrary.",
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
            _only_products("range_az", "range_el", "subspace_err"),
        ),
        # INTEGRITY (hostile-expert third read, 2026-09-23): the FFT range-elevation
        # panel used to be deliberately off because it contradicts the "image barely
        # moves" framing -- hiding a contradicting panel is worse than showing it. It
        # is back on; the card now tells the three-number version below.
        blurb=("Press Run once: both arms run and appear as before (A, top, mantissa 6 "
               "bit) / after (B, bottom, mantissa 1 bit), each panel printing its "
               "subspace-error statistic (about 0.06 for A vs about 0.63 for B, measured "
               "on screen 2026-09-23). Three numbers carry the story: the range-azimuth "
               "image barely moves (peak-median about 0.6 dB, 61.2 -> 60.6), the "
               "elevation cut moves about 2.7 dB (mean image move in unclipped dB, "
               "offline measurement 2026-09-22, not the on-screen statistic), and the "
               "tracker error moves 10x (0.06 -> 0.63). The tracker is far more "
               "sensitive to weight precision than either picture is; the elevation cut "
               "is on screen precisely because it is the one that moves. The manual path "
               "still works: lower the AFE weight mantissa 6 -> 1 bit and run again."),
        live_knobs=[("afe", "mantissa", "6 -> 1 bit (subspace_err 0.06 -> 0.63)")],
        # A/B (Change 1): as-loaded IS mantissa=6 (the settled 0.06 arm); run B drops
        # to 1 bit, the 0.63 arm the card's headline quotes.
        ab=("afe", "mantissa", 1),
        ab_label_a="6 bit", ab_label_b="1 bit",
        # Rewritten (hostile-expert fourth read, 2026-09-23): the old note claimed the
        # elevation cut moves ~2.7 dB, a number no panel on THIS screen shows (that
        # figure is a different, offline metric -- see the card's `say`/`blurb`). The
        # note now claims only what the two displayed images and the tracker curve do.
        screen_note=("range-azimuth and range-elevation images barely move on the "
                     "displayed -40 dB range (statistics printed on each); the tracker "
                     "error moves 10x; subspace error is unnormalised, ceiling sqrt(k) "
                     "= 2.83 for k = 8."),
        say=[
            "As loaded the curve starts at 0.04 and settles at 0.06 within two frames: "
            "the tracker is warm-started from a perturbed copy of the true subspace and "
            "relaxes to its steady tracking error. The knob compares the SETTLED level, "
            "0.06 against 0.63.",
            "Say the headline in ANGLES: subspace error 0.63 -> 0.06 is an unnormalized "
            "distance bounded by sqrt(k); converted, the average principal angle goes "
            "12.8 deg -> 1.3 deg.",
            "The three numbers together: range-azimuth barely moves (about 0.6 dB, "
            "61.2 -> 60.6), the elevation cut moves about 2.7 dB (an offline metric, "
            "not on screen), and the tracker error moves 10x. The AFE does "
            "something real; the range-azimuth picture just is not where it shows.",
            "No detection metric is wired to this view. Say so before being asked what it "
            "means for P_d or false alarms.",
            "The brightest band at range 0-2 m is not a target: Sionna's "
            "normalize_delays=True subtracts the shortest path's delay, so range 0 is "
            "the earliest arrival (47.6 dB above the frame-0 median). The 20-22 m stripe "
            "is real drifting multipath (2-21 m across frames), not fixed.",
            "Read with Thrust 3: at 1 bit the tracker's steady state (0.62) is worse than "
            "a cold start's FIRST frame (about 0.6) -- at that precision it never acquires "
            "(measured on screen 2026-09-23).",
        ],
        do_not_say=[
            "That the mantissa sweep models analog hardware error: AFEBlock uses "
            "WEIGHT_FLOAT, which compress.py's own docstring calls 'right for a compute "
            "datapath and wrong for an analog control'.",
            "That the picture does not respond: the elevation cut does, by about 2.7 dB "
            "(an offline metric, not on screen) for the same 6 -> 1 bit sweep.",
            "That a higher compression ratio would look better: 512 of 1024 was chosen so "
            "the tracker can observe drift; observability drops from 0.50 to 0.055 at 16x, "
            "unmeasured.",
            "Peak-to-median dynamic range as evidence compression is good: it improves as "
            "compression worsens.",
        ],
    ),
    DemoPreset(
        id="thrust3_cold_start_acquisition",
        label="Thrust 3 - adaptive feature extraction: cold-start acquisition",
        thrust=3,
        n_steps=6,
        overrides=_merge(
            {"afe": {"enabled": True}},
            {"subspace": {"params": {"k": 8, "warm_start": "cold"}}},
            {"interconnect": {"enabled": False}},
            _only_products("subspace_err"),
        ),
        blurb=("The tracker starts from a RANDOM basis with no peek at ground truth and "
               "acquires the scene's 8-dimensional subspace from 512 adaptive measurements "
               "of 1024 elements. Watch subspace_err: about 0.6 -> 0.15 -> 0.07 -> 0.06, at "
               "the warm-started floor by frame 3. Press Run once: both arms run and appear as "
               "before (A, top, cold start) / after (B, bottom, warm start), so the cold "
               "curve's acquisition sits above the historical warm curve, which starts at "
               "0.04 -- already below the dashed 0.06 line both curves converge to, because "
               "it begins from a perturbed copy of the true subspace."),
        live_knobs=[("subspace", "warm_start", "cold <-> warm")],
        # A/B (2026-09-23 hostile-expert read #1): the screen never named its own knob
        # ("tracker initialisation = cold" with no warm curve to compare against). As
        # loaded IS the cold arm; run B flips to warm, the historical curve the blurb and
        # do_not_say list both already assumed a reader could see.
        ab=("subspace", "warm_start", "warm"),
        ab_label_a="cold start (random basis)", ab_label_b="warm start (perturbed truth)",
        screen_note=("Frames are 1 m of platform travel each (no time base); warm start = "
                     "perturbed copy of the true subspace, so its curve is tracking lag "
                     "only."),
        say=[
            "Three frames are three metres of platform travel: the frames carry no time "
            "base (1 m per frame at generation), so quote convergence in frames, never "
            "seconds.",
            "There is deliberately no image on this screen: the range-azimuth product does "
            "not change visibly during acquisition on the displayed 40 dB range (it is the "
            "same panel Thrusts 1 and 2 show), and the point of the error curve is that the "
            "picture cannot show you what the tracker has not yet learned. If asked what "
            "the image would look like without the AFE, the answer is 'the same at this "
            "compression' -- do not toggle it.",
            "Three frames to converge, at ten refinement passes per frame (n_refine=10 -- "
            "say it before someone reads it).",
            "This is 2:1 compression (m=512 of 1024). At 16:1 or 64:1 a cold start does not "
            "converge in this many frames, which is why m is not a live knob here.",
            "The tracker still spikes when the scene's rank genuinely collapses (frames 23-26 "
            "on the shipped path): 'mitigated' is not 'fixed'. Have the singular-value "
            "spectrum as a backup slide.",
            "The run is not faster than a full SVD, because scoring runs the full SVD every "
            "frame to build ground truth; the 45x microbenchmark is real, the run time is not.",
            "The run is nondeterministic at the ~5e-3 level and the cold-start first frame "
            "varies run to run; quote 'about', never the third decimal.",
        ],
        do_not_say=[
            "Anything with an interferer: three confounders, and the sign of the response "
            "flips with a knob that is not on screen.",
            "'k = 8 is the optimum' in raw subspace_err: the metric ceilings at sqrt(k).",
            "The AFE on/off toggle as 'the effect of adaptive feature extraction': 2:1, "
            "costs 3 dB, the picture looks identical.",
            "Do not put subspace_err beside an m=16 run: the warm tracker's error GROWS "
            "with frames there (0.125 -> 0.807 over six).",
        ],
    ),
    DemoPreset(
        id="thrust4_interconnect_range_profile",
        label="Thrust 4 - a worse interconnect, on the range profile",
        thrust=4,
        n_steps=3,
        overrides=_merge(
            {"interconnect": {"enabled": True, "params": {
                "case": "passthrough", "normalize_gain": True}}},
            _only_products("range_profile", "range_az"),
        ),
        blurb=("A SYNTHETIC bad interconnect: the 11-tap boxcar placeholder, normalized to a "
               "0 dB peak so it has no gain a passive part could not have, leaving ~60 dB "
               "of in-band ripple. Press Run once: both arms run and appear as before "
               "(A, top, passthrough) / after (B, bottom, SYNTHETIC boxcar), each panel "
               "printing its own median-floor statistic (about -49 dB for A vs about "
               "-34 dB for B, measured on screen 2026-09-23). "
               "On the heatmap the filter streaks each bright return along the range "
               "axis: the 11-tap boxcar is applied along frequency unwindowed, the worst "
               "possible filter shape (first sidelobe -13 dB, 6 dB per octave), so one "
               "clean point target's sidelobes reach -40 dB over 14.4 m of the 25 m axis "
               "(72 of 126 gates) -- measured through the real InterconnectBlock and "
               "RangeProfileBlock. That, not the main-lobe smear, is what the audience "
               "sees; the floor rise is the same energy. Lead with the floor; the two "
               "vertical streaks are the filter's sidelobes on its two strongest returns. "
               "Manual path: set Case -> default and run again."),
        live_knobs=[("interconnect", "case", "passthrough -> default (synthetic boxcar)")],
        # A/B (hostile-expert fourth read, 2026-09-23: polarity was inverted relative to
        # T1/T2, where "after"/B is always the degraded arm -- here "after" used to be
        # the HEALTHY passthrough arm). As-loaded IS passthrough; run B swaps to the
        # synthetic boxcar, the ~14 dB floor-rise direction the card's headline quotes,
        # so "after" is the degraded arm on every A/B screen, same as T1/T2.
        ab=("interconnect", "case", "default"),
        # Labels carry "SYNTHETIC" explicitly (owner ballot 3A: labelled synthetic
        # wherever it appears) -- before 2026-09-23 the banner read "Case default
        # (boxcar)" with no hint the filter is a placeholder, not a measured part.
        ab_label_a="passthrough (no interconnect)",
        ab_label_b="SYNTHETIC 11-tap boxcar placeholder -- added",
        screen_note=("SYNTHETIC filter: the real Tessera/UIC responses differ by <= 0.021 "
                     "dB in band and are invisible on this display -- for the real parts "
                     "the honest result is a null."),
        say=[
            "This filter is synthetic and labelled as such wherever it appears (owner "
            "ballot 3A). It stands in for a bad interconnect; it is not a model of any "
            "hardware.",
            "Credit UIC by name (Gharib & Partin-Vaisband): the interconnect thrust, the "
            "Interconnect block and the six Tessera S21 responses shipped in "
            "e2e/data/interconnect are theirs; the finding that flat in-band loss divides "
            "out of a peak-normalized image is ours. The boxcar is our placeholder, not "
            "derived from their designs.",
            "The real Tessera/UIC designs are INVISIBLE on a peak-normalized display: the "
            "actual Case3 response in-band gives correlation 0.999999, max 0.021 dB "
            "difference. Flat insertion loss divides out. That is why the demo shows a "
            "shaped filter, and why the range profile, not the image, is the view.",
            "Crosstalk -- the dominant real array-interconnect impairment -- is structurally "
            "absent: one S21 is broadcast to all 1024 elements. Say it up front.",
            "The 77 GHz parts are not reconciled with the 30 GHz frames; today's "
            "reconciliation is 'relabel the axis'. Caption real-data results as shape-only.",
            "The brightest band at range 0-2 m is not a target: Sionna's "
            "normalize_delays=True subtracts the shortest path's delay, so range 0 is "
            "the earliest arrival (47.6 dB above the frame-0 median). The 20-22 m stripe "
            "is real drifting multipath (2-21 m across frames), not fixed.",
        ],
        do_not_say=[
            "'Case3' from the dropdown as the UIC Case3: it is a legacy alias for "
            "passthrough. The real CSV is not reachable from this screen yet.",
            "That the boxcar is physically legitimate: unnormalized it has +20.8 dB gain.",
            "That the real designs 'do nothing' -- they are invisible on THIS display, which "
            "is a statement about the display.",
        ],
    ),
    DemoPreset(
        id="thrust5_detector_cfar",
        label="Thrust 5 - detector on benchmark frames: classical CFAR",
        thrust=5,
        n_steps=5,
        overrides=_merge(
            {"corpus_environment": {"enabled": True, "params": {
                "manifest": DEFAULT_CORPUS, "split": "test", "start_frame": 0}}},
            {"rffe": {"enabled": False}, "interconnect": {"enabled": False},
             "afe": {"enabled": False}, "subspace": {"enabled": False}},
            _only_products("radar_cube", "detector"),
            {"detector": {"params": {"mode": "cfar", "threshold": CFAR_THRESHOLD,
                                     "cfar_guard": 2, "cfar_train": 6}}},
        ),
        blurb=("Replays the held-out TEST frames every published number was scored on, "
               "labels included, and runs the classical CA-CFAR baseline on them: the "
               "range-Doppler cube, then the objectness map with detections (red x) over "
               "ground truth's match-tolerance box (white; a cross inside is a hit, scored "
               "beside it). The threshold is CFAR's recall-0.5 point (0.66), matching the "
               "two network presets, so the cross counts across the three screens ARE the "
               "false-alarm comparison. Run this first, then load RADDetNet on the same "
               "frames: the previous run stays on the Results tab underneath."),
        live_knobs=[("detector", "threshold", "0.66 -> 0.8 (fewer detections; the knob "
                                              "that moves the way it sounds)")],
        screen_note=_T5_SCREEN_NOTE,
        say=[
            "Unambiguous velocity is +-v_max from the manifest (read it: ~9.7 m/s); the "
            "corpus targets are slower by construction, so a 20 m/s car would alias here "
            "-- say so if asked.",
            "SAY FIRST: the frames change here. Thrusts 1-4 ran ray-traced munich frames "
            "(25 m, range-azimuth); this is the benchmark corpus: stored ADC frames "
            "(100 m, range-Doppler cube), already impaired at generation. On replay the "
            "ADC-cube blocks are SKIPPED (the run note says so) and the Thrust 1-4 blocks "
            "are off.",
            "Classical CFAR scores AP 0.301 on this split; the data-blind chance floor is "
            "0.081. Both numbers reproduced today from the public repo.",
            "At this operating point CFAR averages 6.2 false alarms per frame over the 172 "
            "test frames (a single frame can show more or fewer). The objectness map is a "
            "clipped CFAR ratio, near zero away from detections -- it looks dark because "
            "CFAR is a threshold test, not a probability field.",
            "The top 60 m of the map is empty because the labels stop at 40 m, which is "
            "also the scoring crop; say it before someone asks what is up there.",
            "Range-azimuth heatmaps elsewhere in the demo come from the munich frames' "
            "delay-normalised channel (range 0 = earliest arrival); this panel's range is "
            "absolute because the ADC cube is dechirped -- different pipeline.",
            "Ground truth omits about 3 real strongly-scattering objects per frame inside "
            "40 m, so any detector that fires on every real object has a precision ceiling "
            "of 0.64. Some of the 'false alarms' are real objects.",
            "Streaked targets in Doppler: ANSWERED. True mainlobe is 6-8 of 64 bins "
            "(~2 m/s at 0.303 m/s/bin); ambient floor sits at median -41.6 dB / p95 "
            "-40.6 dB, within 1 dB of the -40 dB clip -- floor fluctuation lights up "
            "whole rows, a display-threshold coincidence.",
        ],
        do_not_say=[
            "Any learned-detector number from before 2026-09-22 except the rd-format 0.127 "
            "and 0.123: the rad-format results were retracted (ESTABLISHED_FACTS F84).",
            "That fewer CFAR training cells means more false alarms: measured on these 5 "
            "frames at threshold 0.66 the count went 46 -> 39 (train 6 -> 2). Do not turn "
            "that knob on stage.",
        ],
    ),
    DemoPreset(
        id="thrust5_bridge_adc_bits_vs_detections",
        label=("Thrust 5 (bridge) - ADC resolution vs detections: same 5 scenes, "
               "12-bit vs 4-bit"),
        thrust=5,
        n_steps=5,
        overrides=_merge(
            {"corpus_environment": {"enabled": True, "params": {
                "manifest": BRIDGE_CORPUS_12BIT, "split": "test", "start_frame": 0}}},
            {"rffe": {"enabled": False}, "interconnect": {"enabled": False},
             "afe": {"enabled": False}, "subspace": {"enabled": False}},
            _only_products("radar_cube", "detector"),
            {"detector": {"params": {"mode": "cfar", "threshold": CFAR_THRESHOLD,
                                     "cfar_guard": 2, "cfar_train": 6}}},
        ),
        # MEASURED (2026-09-23, CUDA, two independent runs each arm -- both runs agreed
        # bit-for-bit, so the gap below is the quantizer, not run-to-run noise): over the
        # 5 test frames, cumulative hits 19 (A, 12-bit) vs 17 (B, 4-bit); unmatched
        # detections 25 both arms (redistributed frame to frame, not net reduced); hit
        # rate 0.613 vs 0.548.
        blurb=("Same CA-CFAR run as the classical-CFAR preset (threshold 0.66, guard 2, "
               "train 6), but the CORPUS itself was regenerated at two ADC resolutions "
               "from one scene salt (seed 4242): the same 5 held-out test scenes, same "
               "target ranges/velocities/classes frame by frame -- only the ADC "
               "quantizer changed (per-frame quant_snr_db ~57-58 dB at 12-bit vs ~9-10 "
               "dB at 4-bit). Press Run once: both arms run and appear as before (A, "
               "top, 12-bit) / after (B, bottom, 4-bit). Measured over the 5 test "
               "frames: cumulative hits 19 (A) vs 17 (B), unmatched detections 25 both "
               "arms (redistributed across frames, not reduced), hit rate 0.613 vs "
               "0.548. This is the one Thrust 5 screen where a front-end setting (ADC "
               "bit depth) visibly moves a detection count -- and it moves it baked "
               "into the corpus at generation time, not on a live knob: the manifest "
               "path IS the knob, changed by re-running the generator, never live on "
               "the ADC itself."),
        live_knobs=[("corpus_environment", "manifest",
                     "12-bit corpus -> 4-bit corpus (same 5 scenes, run again)")],
        ab=("corpus_environment", "manifest", BRIDGE_CORPUS_4BIT),
        ab_label_a="12-bit ADC (as generated)",
        ab_label_b="4-bit ADC, same scenes re-generated -- added quantisation",
        screen_note=("frames: stored ADC corpus (b1_bridge_12bit vs b1_bridge_4bit, "
                     "benchmark_v1_D2), replayed -- same 5 scenes (seed 4242), "
                     "quantizer 12 vs 4 bits is the only difference; the RF chain of "
                     "Thrusts 1-4 is bypassed; 40 m; in-distribution."),
        say=[
            "Read the cumulative-hits row: 19 (A, 12-bit) vs 17 (B, 4-bit) over the 5 "
            "test frames -- 2 fewer real targets crossed CFAR's threshold at 4-bit. "
            "Unmatched detections stayed at 25 both arms, just redistributed "
            "frame to frame, so the story is fewer hits, not fewer false alarms.",
            "4-bit quantisation raises the ADC noise floor by construction "
            "(quant_snr_db ~57-58 dB at 12-bit vs ~9-10 dB at 4-bit, stored per frame "
            "in the corpus meta). CFAR's own adaptive threshold tracks that floor, so a "
            "real return that cleared the 12-bit threshold can fall back under the "
            "4-bit one.",
            "The Thrust 1-4 blocks (RFFE, interconnect, AFE, subspace) are off here, "
            "same as the other two Thrust 5 presets -- this corpus enters the chain "
            "already digitized.",
            "Both arms are deterministic and reproduced bit-for-bit run to run "
            "(checked twice each, 2026-09-23): the 2-hit gap is the quantizer, not "
            "run-to-run noise.",
            "The two manifests share one scene salt (seed 4242) with a rename between "
            "generator runs, so geometry and target lists are identical frame by "
            "frame -- verified by diffing each test frame's stored meta.",
        ],
        do_not_say=[
            "That this isolates a learned detector: like thrust5_detector_cfar, this "
            "preset only runs classical CFAR on 5 frames -- no network is loaded here.",
            "'Unmatched detections' as false alarms: ground truth omits real objects "
            "(F83), so the precision ceiling is 0.64 and some unmatched crosses are "
            "real, unlabelled targets.",
        ],
    ),
    DemoPreset(
        id="thrust5_detector_ml",
        label=("Thrust 5 - detector on benchmark frames: ported network (the arm that "
               "LOSES, shown on purpose)"),
        thrust=5,
        n_steps=5,
        overrides=_merge(
            {"corpus_environment": {"enabled": True, "params": {
                "manifest": DEFAULT_CORPUS, "split": "test", "start_frame": 0}}},
            {"rffe": {"enabled": False}, "interconnect": {"enabled": False},
             "afe": {"enabled": False}, "subspace": {"enabled": False}},
            _only_products("radar_cube", "detector"),
            {"detector": {"params": {"mode": "ml", "checkpoint": ML_CHECKPOINT,
                                     "threshold": ML_THRESHOLD}}},
        ),
        blurb=("The same frames, through the ported FFTRadNet checkpoint (rd input; test AP "
               "0.127 against CFAR's 0.301 under the same protocol). Its objectness map is "
               "a range-profile x fixed-azimuth-prior STRIPE, not peaks: the network never "
               "learns azimuth (F83). The decode threshold is pinned at its recall-0.5 "
               "operating point (0.22), matching the CFAR and RADDetNet presets; at the "
               "default 0.5 this checkpoint draws nothing."),
        live_knobs=[("detector", "threshold", "0.22 -> 0.5 (the figure goes blank -- that is the point)")],
        # This is the arm the demo shows on purpose to LOSE (hostile-expert fourth
        # read, 2026-09-23): the shared T5 note plus a one-clause reminder, kept short
        # so the combined line still fits 16 px on the 1600 px results page.
        screen_note=_T5_SCREEN_NOTE.replace("; seed 42 of two (0.476 / 0.436)", "").rstrip(".")
        + "; loses to CFAR 0.127 vs 0.301, shown on purpose.",
        say=[
            "The learned detector LOSES to CFAR: 0.127 vs 0.301, chance floor 0.081. Say it "
            "first; the diagnosis is the result.",
            "At this operating point (0.22) expect ~29 crosses per frame ON AVERAGE over the "
            "172 test frames = 26.4 false alarms + 3.0 hits (beat_cfar.json); any single "
            "frame differs (the rehearsal's frame 5 showed 25). CFAR's false-alarm rate is "
            "6.2, RADDetNet's 3.0. ALL inside 40 m: the network never fires beyond the "
            "labelled range.",
            "Both ported networks emit a near-separable f(range) * g(azimuth) map: rank-1 "
            "energy fraction 0.89 / 0.76 against 0.31 for ground truth. Under azimuth-only "
            "matching they score no better than a constant frame-independent map.",
            "F83's mechanism, verbatim: neither head converts channel phase into an angle "
            "bin. Tested two ways (F85): the same beamformed input into this decoder is "
            "worth +0.011 (0.138); an architecture with range x azimuth as its spatial "
            "plane, on that input, scores 0.476 and passes the same controls this one "
            "fails. Load the RADDetNet preset for that -- and read its caveats first.",
            "How two results died this week and what stops it recurring: every checkpoint "
            "trained since 2026-09-21 records a fingerprint of the code that built its "
            "inputs (F84). The checkpoint on screen predates the field; it is trusted "
            "because re-scoring it today reproduces its number, not because of a stamp.",
        ],
        do_not_say=[
            "'The rad input doubles AP' or any 0.229 / 0.484 figure: retracted, F84.",
            "'We fixed azimuth': the stripe statistic refutes it on the next slide.",
            "That this is a benchmark of SSMRadNet or FFTRadNet: the fault is on our side "
            "of the integration, and the collaborator README says so.",
        ],
    ),
    DemoPreset(
        id="thrust5_detector_raddetnet",
        label="Thrust 5 (LEAD) - RADDetNet vs CFAR: AP 0.476 vs 0.301, in-distribution",
        thrust=5,
        n_steps=5,
        overrides=_merge(
            {"corpus_environment": {"enabled": True, "params": {
                "manifest": DEFAULT_CORPUS, "split": "test", "start_frame": 0}}},
            {"rffe": {"enabled": False}, "interconnect": {"enabled": False},
             "afe": {"enabled": False}, "subspace": {"enabled": False}},
            _only_products("radar_cube", "detector"),
            {"detector": {"params": {"mode": "ml", "checkpoint": RADDETNET_CHECKPOINT,
                                     "threshold": RADDETNET_THRESHOLD}}},
        ),
        blurb=("The same frames through RADDetNet (Doppler as channels, range x azimuth as "
               "the spatial plane) on CFAR's own beamformed cube. Test AP 0.476 vs CFAR's "
               "0.301, 3.0 FA/frame at recall 0.5 vs CFAR's 6.2, controls pass (F85). "
               "Threshold pinned at recall-0.5 (0.44). Independently verified (F85/F86 "
               "addenda): bit-identical, no leakage, baseline fair. On an unseen "
               "earlier-generator corpus the result is SEED-DEPENDENT (F86): seed 42 leads "
               "CFAR by +0.03 with worse false alarms, seed 43 trails by -0.03. "
               "In-distribution both seeds beat CFAR. Trained on both corpora, one "
               "checkpoint beats CFAR on both held-out splits (F86 addendum) -- but "
               "neither corpus is unseen; two seeds agree (0.584 / 0.577). Owner decision: "
               "LEADS Thrust 5, caveat volunteered."),
        live_knobs=[("detector", "threshold", "0.44 -> 0.2 (more, weaker detections)")],
        screen_note=_T5_SCREEN_NOTE,
        say=[
            "The defensible sentence: a learned head on the classical front end beats a "
            "CFAR threshold on the same cube, in-distribution -- say that, not 'beats "
            "CFAR' (F85 addendum).",
            "Every number comes from e2e/ml/runs/beat_cfar.json (seed 42, deterministic); "
            "independently re-scored bit-identically, controls re-implemented to 1e-6. "
            "Paired scene-level bootstrap: +0.175 AP, 95% CI [+0.145, +0.208] "
            "(raddetnet_ci.json, F85 addendum).",
            "The controls are F83's, which the shipped nets FAILED (deranged-label "
            "retention 12%, CFAR 10%, shipped nets 48-51%; azimuth-only AP 0.657 vs 0.472 "
            "prior); the baseline is honest too -- nine classical configs scored, best "
            "0.328, shipped CFAR beats every alternative (F85 addendum).",
            # Word-count trade (hostile-expert fourth read, 2026-09-23): this bullet was
            # ADDED and the two above it merged into one to keep the say list at its
            # tested ceiling of 6 while staying inside the 450-word card ceiling.
            "Four learned arms were screened on this test split: three ported "
            "architectures and this one designed to the F83 diagnosis; all four are in "
            "beat_cfar.json, none dropped.",
            "THE CAVEAT: on an unseen earlier-generator corpus (b1_bench_v2), CFAR scores "
            "0.179/13.2 FA; this checkpoint (seed 42) 0.208/15.1; seed 43 0.153/20.7 "
            "(F86). Out of distribution it does NOT reliably beat CFAR -- one seed +0.03, "
            "the other -0.03.",
            "Trained on both corpora, the network beats CFAR on both test splits "
            "(0.584/1.4 FA v3, 0.487/2.9 FA v2; controls pass -- F86 addendum). Say this "
            "as a data-diversity result, not generalisation: it is also one training "
            "seed.",
        ],
        do_not_say=[
            "'Beats CFAR', unqualified: the verified claim is in-distribution and on "
            "CFAR's own front end (F85 addendum); the first radar person in the room will "
            "ask about both.",
            "Anything about generalisation or robustness: the two single-corpus seeds "
            "straddle CFAR out of distribution (+0.03/-0.03); the joint 0.487 on v2 is "
            "NOT out-of-distribution -- it trained on v2 (F86).",
            "The joint numbers as generalisation: two seeds agree (F86) but both corpora were "
            "trained on; the only unseen corpus is the third (F87: 0.539 vs 0.274, one backdrop).",

            "That this is what the professor asked for: it is a detector designed to the "
            "F83 diagnosis, not a port of the collaborators' architectures.",
            "That the model converged: val AP peaks at epoch 14 of 40 and decays to "
            "0.35-0.41 while train loss keeps falling (F85 addendum); early stopping on "
            "val is load-bearing.",
        ],
    ),
]

PRESETS_BY_ID: Dict[str, DemoPreset] = {p.id: p for p in PRESETS}


class PresetError(ValueError):
    """A preset that does not fit the registry -- raised, never papered over."""


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
