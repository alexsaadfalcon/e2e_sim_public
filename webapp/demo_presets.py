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
from typing import Any, Dict, List, Tuple

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

#: Decode threshold for that checkpoint. Its recall-0.5 operating point is objectness
#: 0.22 (compare_detectors, 2026-09-22); at the registry default of 0.5 it draws ZERO
#: detections -- the "demo landmine" in DEMO_DEFENSE.md, measured again today on the
#: real frames. Pinned here so the figure is never blank.
ML_THRESHOLD = 0.2


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
        blurb=("Run once as loaded, then turn ONE knob and run again: LNA bias 8 -> 0.5 mA, "
               "or IF bandwidth 15 -> 50 MHz. The range-azimuth image loses dynamic range as "
               "the front-end's own noise rises. The signal is deliberately set just below "
               "the model's input-referred noise (1e-7 vs 1.36e-7 V), which is where a real "
               "1024-element radar operates: per-element SNR below 0 dB, recovered by "
               "coherent gain."),
        live_knobs=[("rffe", "lna_bias_ma", "8 -> 0.5 mA (about -12 dB)"),
                    ("rffe", "if_bw_mhz", "15 -> 50 MHz (about -5 dB; 1 -> 50 is -16 dB)")],
        say=[
            "LNA bias 0.5 -> 8 mA is worth about +12 dB of image dynamic range at this "
            "operating point; IF bandwidth 1 -> 50 MHz costs about -16 dB. Two significant "
            "figures; the numbers move 0.6-0.9 dB between operating points.",
            "The noise mechanism is right: the analytic Friis cascade predicts 11.97 dB and "
            "the end-to-end chain measures 11.80 dB, through 1024 elements, an FFT, the AFE "
            "and the subspace tracker. The IF number lands on 10*log10(50) = 17.0.",
            "At the default signal level (1e-5) these knobs correctly do nothing (0.5 dB, "
            "under the 40 dB display floor) -- show that as the control if asked.",
            "Below about 4 mA the modelled LNA is a LOSS stage (-8.5 dB at 0.5 mA), so most "
            "of the 12 dB is the LNA leaving the attenuator regime. The defensible sub-claim "
            "is 4 -> 8 mA = +1.6 dB (measured 2026-09-22). Say it before someone does.",
            "There is no trade-off in the model today: nothing clips and the IF filter only "
            "sets the noise variance. Volunteer the missing half: 1000 frequency points at "
            "1 MHz IF is a 1 ms sweep versus 20 us at 50 MHz, and a 20 m/s car moves two "
            "wavelengths in that time.",
        ],
        do_not_say=[
            "Any DC power readout: PRX is U-shaped with its MINIMUM at the best-quality "
            "point, and implies an 8.45 V rail in a 100-200 mV chain.",
            "The compression regime (signal scaling 1e-1..1e-3): explained, but worse on "
            "stage than silence.",
            "That gm scales linearly with bias -- it does at 8 mA only because the model "
            "uses a weak-inversion law; say 'constant-overdrive power scaling'.",
            "Anything about IIP3: it is constant to five decimals across 0.5-10 mA here. "
            "A real LNA's IIP3 improves with bias.",
            "Any gain knob at any level: peak normalization removes it.",
            "Any absolute dBm sensitivity or noise-figure number for the chain.",
            "That the 1024 receivers are modelled individually: every column of the "
            "config table holds one value, broadcast to all elements. Channel mismatch is "
            "structurally absent (cheap to add; not added).",
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
            _only_products("range_az", "subspace_err"),
        ),
        blurb=("Run as loaded, then lower the AFE weight mantissa 6 -> 1 bit and run again. "
               "The subspace error rises sharply (0.06 -> 0.63 measured on this preset) "
               "while the range-azimuth image barely moves ON THE DISPLAYED 40 dB RANGE -- "
               "the changes sit 40-80 dB below the peak, under the colour floor. The "
               "tracker is far more sensitive to weight precision than the picture is. The "
               "FFT az-el panel is deliberately off (it contradicts this framing)."),
        live_knobs=[("afe", "mantissa", "6 -> 1 bit (subspace_err 0.06 -> 0.63)")],
        say=[
            "Say the headline in ANGLES: subspace error 0.63 -> 0.06 (measured on this preset "
            "2026-09-22: 0.626 -> 0.062) is an unnormalized distance bounded by sqrt(k); "
            "converted, the average principal angle goes 12.8 deg -> 1.3 deg.",
            "The AFE is doing something real but modest: on vs fully removed moves the "
            "displayed range-azimuth image 0.14 dB -- bigger than the mantissa knob's own "
            "0.04 dB on the same displayed range.",
            "No detection metric is wired to this view. Say so before being asked what it "
            "means for P_d or false alarms.",
        ],
        do_not_say=[
            "That the mantissa sweep models analog hardware error: AFEBlock uses "
            "WEIGHT_FLOAT, which compress.py's own docstring calls 'right for a compute "
            "datapath and wrong for an analog control'.",
            "The FFT az-elevation panel: it moves 2.70 dB mean while range-azimuth moves "
            "0.05 dB, and undoes the 'image barely moves' story.",
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
            _only_products("subspace_err", "range_az"),
        ),
        blurb=("The tracker starts from a RANDOM basis with no peek at ground truth and "
               "acquires the scene's 8-dimensional subspace from 512 adaptive measurements "
               "of 1024 elements. Watch subspace_err: 0.57 -> 0.15 -> 0.07 -> 0.06, at the "
               "warm-started floor by frame 3. Flip 'Tracker initialisation' back to warm "
               "to show the historical curve, which starts at 0.04 -- already below the "
               "0.06 tracking floor it then settles to, because it begins from a perturbed "
               "copy of the true subspace."),
        live_knobs=[("subspace", "warm_start", "cold <-> warm")],
        say=[
            "Three frames to converge, at ten refinement passes per frame (n_refine=10 -- "
            "say it before someone reads it).",
            "This is 2:1 compression (m=512 of 1024). At 16:1 or 64:1 a cold start does not "
            "converge in this many frames, which is why m is not a live knob here.",
            "The tracker still spikes when the scene's rank genuinely collapses (frames 23-26 "
            "on the shipped path): 'mitigated' is not 'fixed'. Have the singular-value "
            "spectrum as a backup slide.",
            "The run is not faster than a full SVD, because scoring runs the full SVD every "
            "frame to build ground truth; the 45x microbenchmark is real, the run time is not.",
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
                "case": "default", "normalize_gain": True}}},
            _only_products("range_profile", "range_az"),
        ),
        blurb=("A SYNTHETIC bad interconnect: the 11-tap boxcar placeholder, normalized to a "
               "0 dB peak so it has no gain a passive part could not have, leaving ~60 dB "
               "of in-band ripple. Run as loaded, then set Case -> passthrough and run "
               "again: the range profile's floor drops ~14 dB and the smearing disappears "
               "(11 NATIVE range cells; about 3 gates on the 256-bin display, which pools 4 "
               "cells per gate). Lead with the range profile, not the heatmap."),
        live_knobs=[("interconnect", "case", "default (synthetic boxcar) -> passthrough")],
        say=[
            "This filter is synthetic and labelled as such wherever it appears (owner "
            "ballot 3A). It stands in for a bad interconnect; it is not a model of any "
            "hardware.",
            "The real Tessera/UIC designs are INVISIBLE on a peak-normalized display: the "
            "actual Case3 response in-band gives correlation 0.999999, max 0.021 dB "
            "difference. Flat insertion loss divides out. That is why the demo shows a "
            "shaped filter, and why the range profile, not the image, is the view.",
            "Crosstalk -- the dominant real array-interconnect impairment -- is structurally "
            "absent: one S21 is broadcast to all 1024 elements. Say it up front.",
            "The 77 GHz parts are not reconciled with the 30 GHz frames; today's "
            "reconciliation is 'relabel the axis'. Caption real-data results as shape-only.",
        ],
        do_not_say=[
            "'Case3' from the dropdown as the UIC Case3: it is a legacy alias for "
            "passthrough. The real CSV is not reachable from this screen yet.",
            "That the boxcar is physically legitimate: unnormalized it has +20.8 dB of gain.",
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
             "afe": {"enabled": False}},
            _only_products("radar_cube", "detector"),
            {"detector": {"params": {"mode": "cfar", "threshold": 0.5,
                                     "cfar_guard": 2, "cfar_train": 6}}},
        ),
        blurb=("Replays the held-out TEST frames every published number was scored on, "
               "labels included, and runs the classical CA-CFAR baseline on them: the "
               "range-Doppler cube, then the objectness map with detections (x) over ground "
               "truth (o). Run this first, then load the ML preset on the same frames."),
        live_knobs=[("detector", "cfar_train", "6 -> 2 cells (noisier estimate, more false alarms)"),
                    ("detector", "threshold", "0.5 -> 0.8 (fewer detections)")],
        say=[
            "Classical CFAR scores AP 0.301 on this split; the data-blind chance floor is "
            "0.081. Both numbers reproduced today from the public repo.",
            "Ground truth omits about 3 real strongly-scattering objects per frame inside "
            "40 m, so any detector that fires on every real object has a precision ceiling "
            "of 0.64. Some of the 'false alarms' are real objects.",
        ],
        do_not_say=[
            "Any learned-detector number from before 2026-09-22 except the rd-format 0.127 "
            "and 0.123: the rad-format results were retracted (ESTABLISHED_FACTS F84).",
        ],
    ),
    DemoPreset(
        id="thrust5_detector_ml",
        label="Thrust 5 - detector on benchmark frames: trained network",
        thrust=5,
        n_steps=5,
        overrides=_merge(
            {"corpus_environment": {"enabled": True, "params": {
                "manifest": DEFAULT_CORPUS, "split": "test", "start_frame": 0}}},
            {"rffe": {"enabled": False}, "interconnect": {"enabled": False},
             "afe": {"enabled": False}},
            _only_products("radar_cube", "detector"),
            {"detector": {"params": {"mode": "ml", "checkpoint": ML_CHECKPOINT,
                                     "threshold": ML_THRESHOLD}}},
        ),
        blurb=("The same frames, through the ported FFTRadNet checkpoint (rd input; test AP "
               "0.127 against CFAR's 0.301 under the same protocol). Its objectness map is "
               "a range-profile x fixed-azimuth-prior STRIPE, not peaks: the network never "
               "learns azimuth (F83). The decode threshold is pinned at 0.2 because at the "
               "default 0.5 this checkpoint draws nothing."),
        live_knobs=[("detector", "threshold", "0.2 -> 0.5 (the figure goes blank -- that is the point)")],
        say=[
            "The learned detector LOSES to CFAR: 0.127 vs 0.301, chance floor 0.081. Say it "
            "first; the diagnosis is the result.",
            "At threshold 0.2 expect roughly 2-70 crosses per frame (mean ~38 over 30 test "
            "frames, 2026-09-22), ALL inside 40 m: the network never fires beyond the "
            "labelled range. At the scored operating point (0.22) it is ~27 per frame = 26 "
            "false alarms + ~3 hits, which is the published number; CFAR's is 6.2.",
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
        label="Thrust 5 - detector on benchmark frames: RADDetNet (0.476 vs CFAR 0.301, verified in-distribution; owner decision pending)",
        thrust=5,
        n_steps=5,
        overrides=_merge(
            {"corpus_environment": {"enabled": True, "params": {
                "manifest": DEFAULT_CORPUS, "split": "test", "start_frame": 0}}},
            {"rffe": {"enabled": False}, "interconnect": {"enabled": False},
             "afe": {"enabled": False}},
            _only_products("radar_cube", "detector"),
            {"detector": {"params": {"mode": "ml", "checkpoint": RADDETNET_CHECKPOINT,
                                     "threshold": RADDETNET_THRESHOLD}}},
        ),
        blurb=("The same frames through RADDetNet -- Doppler as channels, range x azimuth "
               "as the spatial plane, on the beamformed ('rad') input with the classical "
               "front end. Test AP 0.476 against CFAR's 0.301 under the same protocol, 3.0 "
               "false alarms per frame at recall 0.5 against CFAR's 6.2, and the F83 "
               "controls say it reads the frame (deranged-label retention 12%, azimuth-only "
               "0.657 vs 0.472 for a train-density prior). Threshold pinned at its recall-0.5 "
               "operating point (0.44). Independently verified 2026-09-22 (F85 addendum): "
               "reproduces bit-identically, splits scene-disjoint, baseline fair -- and on an "
               "UNSEEN corpus from an earlier generator the lead falls to +0.03 with a WORSE "
               "matched-recall false-alarm rate than CFAR. Present only after the owner has "
               "re-decided Thrust 5 on that basis."),
        live_knobs=[("detector", "threshold", "0.44 -> 0.2 (more, weaker detections)")],
        say=[
            "The defensible sentence, verbatim from the verifier: a learned head on the "
            "classical front end beats a CFAR threshold on the same cube, in-distribution. "
            "The network is fed CFAR's own beamformed cube, notch and TDM compensation "
            "included; the ladder on that identical cube is global threshold 0.18-0.22, "
            "CFAR 0.30, this 0.48. Say that, not 'beats CFAR'.",
            "Every number comes from one file, e2e/ml/runs/beat_cfar.json (seed 42, "
            "deterministic kernels); the checkpoint records the fingerprint of the code that "
            "built its inputs and reproduces its own validation number under current code. "
            "An independent verifier re-scored it bit-identically and re-implemented the "
            "controls to 1e-6. Paired scene-level bootstrap: +0.176 AP, 95% CI [+0.145, +0.206].",
            "The controls are the ones F83 defined and the shipped nets FAILED: AP retention "
            "under deranged labels 12% (CFAR 10%; the FFTRadNets 48-51%; stable across random "
            "derangements), and azimuth-only AP 0.657 against 0.472 for the strongest "
            "frame-independent prior (the mean of all training label maps) -- which the "
            "shipped nets could not beat (0.421).",
            "The baseline is honest: nine classical configurations were scored, the best "
            "reaches 0.328 (Doppler-resolved CFAR, unclamped score), and the shipped "
            "guard/train beats every alternative tried. The 40 m crop, the score floor and "
            "the unlabelled clutter move nothing.",
            "Where the gain is: pedestrians. Hit rate at the floor 0.945 vs CFAR's 0.798; on "
            "vehicles 0.983 vs 0.929. Physically sensible, not suspicious.",
            "What changed is the architecture, not the input: the same beamformed input into "
            "the RADIal-style decoder (b8) scores 0.138 and keeps 47% under deranged labels. "
            "Range x azimuth had to be the spatial plane.",
            "THE CAVEAT, volunteered: on b1_bench_v2 test -- 173 unseen scenes, same radar and "
            "grid, an earlier generator with a different impairment model -- CFAR scores "
            "0.179 at 13.2 FA/frame and this network 0.208 at 15.1 FA/frame. The lead is "
            "+0.03, and at matched recall the network is worse. One seed, one corpus, one "
            "tier; nothing here shows learning buys robustness.",
            "The stripe statistic is 0.62 (0.60 over all frames) against 0.31 for ground "
            "truth: the map is still partly separable. Quote it beside the AP.",
        ],
        do_not_say=[
            "'Beats CFAR', unqualified. The verified claim is in-distribution and on CFAR's "
            "own front end; the first radar person in the room will ask about both.",
            "Anything about generalisation or robustness: the one out-of-distribution "
            "measurement went the other way at matched recall.",
            "That this is what the professor asked for in the ML thrust: it is a detector "
            "we designed to the diagnosis, not a port of the collaborators' architectures.",
            "That the model converged: val AP peaks at epoch 14 of 40 and decays to 0.35-0.41 "
            "while train loss keeps falling. Early stopping on val is load-bearing.",
        ],
    ),
]

PRESETS_BY_ID: Dict[str, DemoPreset] = {p.id: p for p in PRESETS}


class PresetError(ValueError):
    """A preset that does not fit the registry -- raised, never papered over."""


def apply_preset(preset: DemoPreset) -> Dict[str, Dict[str, Any]]:
    """`default_block_state()` with the preset's overrides applied and VALIDATED."""
    state = default_block_state()
    for bid, ov in preset.overrides.items():
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
