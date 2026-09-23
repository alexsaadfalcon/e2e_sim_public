# Wave 1 — the first fan-out (fundamentals and the demo path)

Twelve nodes, chosen because everything downstream inherits them or because a Thrust preset
runs through them. Each brief is neutral by design: it names what to establish, not what to
find. Every brief carries the same standing text (below) plus its own paragraph.

**Standing text for every brief.** Repo `C:\Users\asf3\workspace\e2e_sim_public`. Read
`docs/validation/README.md`, `CARD_TEMPLATE.md`, your node's row(s) in `TREE.md`, and
`notes/RIGOR_STANDARD.md` first. `notes/ESTABLISHED_FACTS.md` and `notes/PHYSICAL_ASSUMPTIONS.md`
contain claims about your node: treat them as claims to re-verify with a command, never as
answers to copy. Record `git rev-parse --short HEAD` and `git status --short` at the start;
the tree is under concurrent edit, so anchor every citation `path:line (symbol)`. You write
only under `docs/validation/` (your card, and ≤200-line scripts under `evidence/<NODE-ID>/`);
scratch goes in your owned dir and is disposable. Do not edit code, tests, notes or the tree.
Nothing runs on the GPU unless your brief says `RUN_SIONNA=1`. **Deliverable:** one filled
card per node ID in the brief, every section answered or `UNKNOWN` with the reason.
**Stop conditions:** the card is complete; or a `SUSPECT` verdict (stop, hand back the finding
immediately); or 40 tool calls without a new result (hand back a `PARTIAL` card that says what
is missing); or a needed artifact/GPU is unavailable (`UNKNOWN`, never a guess).

Scratch root: `C:\Users\asf3\workspace\_scratch\validation\` (sibling of the repo, gitignored
by location). One subdirectory per brief, named by its first node ID.

| # | nodes | tier | scratch dir | GPU |
|---|---|---|---|---|
| 1 | L1-C2 (+ L2-C2.1, L2-C2.2, L2-C2.3) | Opus | `L1-C2/` | no |
| 2 | L1-C1 (+ L2-C1.1) and L1-B5 (+ L2-B5.2) | Sonnet | `L1-C1/` | no |
| 3 | L1-B1, L1-B2 (+ L2-B1.1, L2-B2.1, L2-B2.2) | Opus | `L1-B1/` | `RUN_SIONNA=1`, one scene |
| 4 | L1-C3 (+ L2-C3.1–C3.4) | Opus | `L1-C3/` | no |
| 5 | L1-C4 and L1-B12 (+ L2-C4.1–C4.3, L2-B12.1) | Opus | `L1-C4/` | no |
| 6 | L1-B6 and L1-B11 (+ L2-B6.1, L2-B11.1) | Sonnet | `L1-B6/` | no |
| 7 | L1-B14 (+ L2-B14.1, L2-B14.2) | Sonnet | `L1-B14/` | no |
| 8 | L1-B8 (+ L2-B8.1, L2-B8.2, L2-B8.4) | Opus | `L1-B8/` | no (surrogate is CPU) |
| 9 | L1-B17 (+ L2-B17.1, L2-B17.2) | Sonnet | `L1-B17/` | no |
| 10 | L1-B15 (+ L2-B15.1–B15.5) | Sonnet | `L1-B15/` | no |
| 11 | L1-B7 (+ L2-B7.1, L2-B7.2) | Opus | `L1-B7/` | no |
| 12 | L1-C5 and L1-B10 (+ L2-C5.1, L2-C5.2, L2-B10.1–B10.4) | Sonnet | `L1-C5/` | no |

## Briefs

**1 · Frequency plan (L1-C2).** Establish, artifact by artifact, what frequency axis each
stored frame set and each consumer actually carries or assumes: the declared `FrequencyPlan`
(`e2e/scenario.py`), every `.pkl` under `e2e/environment/sionna_sims/` (read its shape and any
`meta` on CPU), the generator(s) that claim to have produced them, the interconnect `band_hz`
mapping, the `radar_config` presets, and the corpus `.npz` meta under `e2e/ml/datasets/`.
For each, state the carrier, span, point count, whether the axis is absolute or relative to a
scene carrier, and where a consumer's assumption differs from what the artifact carries. Your
oracle is a frame whose frequency axis you know by construction (the dry-run scenario runner
produces one). Card sections 3–7 apply to the plan-to-axis mapping, not to Sionna.

**2 · Frame contract and serialisation (L1-C1, L1-B5).** Establish that the frame contract
(`e2e/frames.py`) and the round trip through every serialisation path (v2 pkl via
`scenario_runner`, legacy pkl via `SionnaIterator`, corpus `.npz` + `.cfr.npy` via
`e2e/ml/storage.py` and `SourceBlock`) preserve shape, dtype, device, ordering and metadata
bit-for-bit, and refuse what they should. Include the metadata a consumer needs to interpret
a frame (array shape, band, power convention) and say for each path whether it travels.

**3 · RT environment, materials, frequency (L1-B1, L1-B2).** With `RUN_SIONNA=1` on one small
scene, establish what `build_rt_scene` and the munich generator set as `scene.frequency`, in
what order arrays and materials are built relative to it, which radio materials each scene
uses and whether each is inside its ITU validity range at the scene's frequency, what the
out-of-band policy does numerically, and what ground roughness and element pattern are in
force. Cross-check one CFR against Sionna's own `paths.cfr()` and one against a re-traced
reference; state agreement as numbers. Section 11 must list every scene you did not run.

**4 · Array geometry and antenna ordering (L1-C3).** Establish, with a target at a known
angle, that every path that turns an element index into an angle agrees on handedness, on
which axis is azimuth, on the flat-axis ordering (row-first or column-first) and on the
spacing it assumes: `rd_synth`, `rt_signal_chain`, `frames.to_aperture_grid`, the
`RangeAz`/`RangeEl`/`FFT` blocks, `baseline.range_azimuth_power`, and the webapp's angle axis.
Where a path assumes a spacing rather than computing it, say so and say what happens when
the frame's true spacing differs.

**5 · Link budget and absolute scale (L1-C4, L1-B12).** Establish the chain of units from
CFR to volts to the thermal floor: the amplitude scale in `scenario_runner`, the legacy vs
physical normalisation in `RFFEBlock`, the noise bandwidth and reference temperature used in
`link_budget.py` and in `rffe_model.py`, and the injected noise power vs the budget. Do an
independent hand calculation at the Ka-band operating point and report agreement in dB.
State separately which constants are measured, which are datasheet-derived, and at which band.

**6 · Waveform, chirp and dechirp (L1-B6, L1-B11).** Establish that the FMCW ramp, the
beat-frequency grid and the CFR-to-beat mapping are mutually consistent: that a point target
at a known range and radial velocity placed through `rd_synth` and through the CFR path lands
in the same range/Doppler/angle bin, that the mapping is exactly invertible, and that every
MIMO scheme's combine matches its demux. Report what the transmit tributary changes when
enabled and what approximations the ramp makes.

**7 · IF high-pass and ADC (L1-B14).** Establish the filter's response against a closed-form
Butterworth of the same order and corner, the effect of the band-edge taper on the far-range
gates, the quantiser's step, bias, clip and SNR against the textbook formula across bit
depths (including the 12 and 3 used on stage), and the behaviour of the AGC full-scale rule on
a physically small cube. Say what "corner as a range" means numerically on each preset.

**8 · Interconnect (L1-B8).** Establish, for each of the three sources (boxcar, CSV, Tessera
surrogate), what frequency each sample of `H` is evaluated at and how that maps onto the
frame's axis; for the CSV path, what in the file is data and what is reconstruction; for the
surrogate, whether the scale-model evaluation reproduces upstream's output and whether the
quantity shown on stage (height A/B) is invariant under the scale rule. Include a passivity
check and the cache's effect on determinism.

**9 · Labels and detection metrics (L1-B17).** Establish that the metric returns AP = 1 when
ground truth is fed as detections, that the match criterion is bounded, symmetric in the
sense it claims, and consistent with the label footprint convention; hand-compute AP and
false-alarms-at-recall on a five-detection example and compare. Section 11 must state which
corpora and which grid geometries were exercised.

**10 · Radar cube and the FFT products (L1-B15).** Establish, with a synthetic CFR of known
delay and angle, where each product (`RadarCube`, `FFTBlock`, `RangeAz`, `RangeEl`,
`RangeProfile`) places the target, what window and padding each applies on which axis, and
how the webapp converts bins to metres and to sin(angle). Report every axis convention
(sign, zero position, shift) you find and whether the screen's labelled value equals the
known one.

**11 · RF front end (L1-B7).** Establish the small-signal gain, the compression point, the
noise floor and their dependence on the two stage knobs (LNA bias, IF bandwidth) by
measurement at the Thrust 1 operating point and at one other, and compare with a Friis
cascade computed by hand. State what the model makes each knob change and what it leaves
unchanged, and whether the IF filter is part of the path the webapp runs.

**12 · Determinism and the subspace tracker (L1-C5, L1-B10).** Establish which random draws
in a full pipeline run are seeded and from what, whether two runs with every seed fixed are
bit-identical (frames, cube, tracker basis), and what the tracker's error metric measures
against. For the tracker, recover a known static subspace and a known drifting one, report
the error floor against the k-truncated SVD, and say what the gap gate reads and whether a
receiver could compute it.

## After the wave

Each returned card is read by the seat, then by one fresh-context reviewer at the tier above
(automatic). Findings go into the flow in `README.md`. The generator (spec in `README.md`)
is written as a wave-1 coder shard so that wave 2's tree is generated, not hand-edited.
