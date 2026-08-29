# Physics justification

This page exists because a simulator can be internally consistent — every block runs,
every test passes — while still not modeling anything real. The question this document
asks of every stage in the receive chain is: **what real-world effect is this block
standing in for, why was this particular model chosen over the alternatives, what does
it approximate away, and what evidence do we have that it behaves the way the physics
says it should?** That last question matters for you, the reader: if a result you get
out of this simulator looks surprising, the first thing to check is whether it's real
physics or an artifact of one of the approximations listed below.

Unfamiliar term (frame, block, `n_freqs`/`num_freqs`, answerability tier, ...)? See
[`docs/GLOSSARY.md`](GLOSSARY.md). For how to set up and run the pipeline yourself, see
[`docs/FIRST_SCENARIO.md`](FIRST_SCENARIO.md); for test conventions and how to add a new
block (and why every new block should get an entry here), see
[`CONTRIBUTING.md`](../CONTRIBUTING.md).

## How to read an entry

Every module below answers the same five questions, in order:

- **Effect** — the real-world physical phenomenon being represented.
- **Model** — the concrete mathematical/computational model chosen to represent it.
- **Approximations** — what the model leaves out or simplifies, stated explicitly
  rather than left implicit.
- **Evidence** — what in the repository (a test, a reproducible measurement from a
  script you can run) supports the claim that the model behaves as intended.
- **Verdict** — one of three grades:
  - **JUSTIFIED** — the physics case is sound and written down; approximations are
    stated rather than hidden.
  - **UNDER-JUSTIFIED** — the model is defensible in principle, but the implementation
    is a cruder form than the model it claims to be, or the case for it is incomplete.
  - **UNJUSTIFIED** — the implementation contradicts the physics it claims to model.
    (Nothing currently shipping in the default pipeline carries this grade; where a
    mechanism was found in this state during development it was fixed before merge —
    see "Known open items" at the end for what's still outstanding.)

A module can carry different verdicts for different aspects of itself; where that
happens, both are stated.

Every entry below has had its final review pass. Two of them — the interconnect
(data-driven transfer functions from a collaborator) and the subspace tracker — were
held back longer than the rest because their wording touches a collaborator's data or
had been recently rewritten; both were cleared on 2026-08-29. Every entry reflects the
state of the code at the time of writing; re-run the cited test or script to check it
against whatever you have checked
out.

---

## 1. TX waveform — `e2e/chain/waveform.py`

**Effect:** the transmitted complex envelope — for FMCW radar, a linear frequency
sweep (chirp); for OFDM comms, a multicarrier symbol.

**Model:** `WaveformBlock` synthesizes the envelope from one of two waveform classes
(`FMCWSignal`, `RandomWidebandSignal`, standing in for a generic wideband/OFDM-like
source), then `ModulateBlock` puts the transmitted spectrum on the exact same
frequency grid as the channel's stored response (`s_pars`) by construction — both are
`n_freqs`-point DFTs over the same span — so there is no resampling/interpolation
seam between the TX and channel models (see the module docstring's "Modulate
convention" for the grid-alignment argument in full).

**Approximations:** the chirp sweep is assumed perfectly linear — real VCOs have
ppm-level nonlinearity that smears range resolution, and this is not modeled. Both
waveform classes hardcode a disabled carrier (`fc` is accepted as metadata but has no
numerical effect on today's output).

**Evidence:** downstream range-processing tests assume and confirm an exactly linear
sweep (the dechirp bijection tests, entry 7).

**Verdict: JUSTIFIED**, with chirp nonlinearity listed as a known, currently-unmodeled
approximation rather than an implicit gap.

## 2. TX power amplifier — `e2e/circuit/tx_pa.py`, wired via `TxPABlock` in `e2e/chain/waveform.py`

**Effect:** solid-state power-amplifier (SSPA) compression near saturation and the
drive-dependent phase shift ("AM/PM") that comes with it.

**Model:** the memoryless Rapp AM/AM curve (the standard behavioral SSPA model in the
literature) plus a smooth AM/PM term that saturates with drive, applied per-sample to
the transmitted envelope.

**Approximations:** memoryless — no memory effects (adjacent-symbol dependence) and no
thermal drift.

**Evidence:** `python -m e2e.main.main_tx_nonideality` sweeps input backoff and reports
EVM for an ideal (linear) TX vs. the non-ideal PA. A representative run: EVM starts
around 22% at 0 dB backoff for the non-ideal TX and falls toward roughly 3.6–4%,
converging on the *fixed* noise-only EVM floor of the ideal TX (which does not move
with backoff) — the convergence is what isolates the PA's nonlinearity as the only
variable, rather than a change in the noise draw. The same script also reports the
sensing-side cost (peak sidelobe level degradation with drive).

**Verdict: JUSTIFIED.**

## 3. Ray-traced propagation channel (Sionna RT) — `e2e/environment/`

**Effect:** propagation through a real or synthetic scene — geometry, occlusion,
multipath, ground interaction.

**Model:** deterministic ray tracing (via [Sionna RT](https://nvlabs.github.io/sionna/))
over triangle meshes, both real-world scans (the shipped `munich` scene) and
generated scenes (`e2e/environment/rt_scenes.py`, entry 18).

**Approximations, and one structural gap worth understanding clearly:** Sionna RT is
built to answer *comms* questions (does a signal reach a receiver, along what paths),
not *radar* questions (does a signal reflect straight back to where it came from).
Its specular-path search essentially never finds a monostatic backscatter path off a
curved surface: the specular point on a curved reflector displaces from the true
return point by an amount proportional to range and inversely proportional to the
surface's local curvature, and for typical automotive-scale curvature and radar range
that displacement exceeds any single mesh facet — so the specular search misses the
return regardless of how finely the mesh is tessellated. This is a genuine limitation
of using a comms ray tracer for a radar problem, not a bug; entry 4 below exists
specifically to fill the resulting gap.

**Verdict: JUSTIFIED for propagation** (geometry, occlusion, multipath, diffuse
return) — **the monostatic-backscatter gap is real and is handled by entry 4, not
here.** See also the README's "Physical modeling scope & limitations" section: diffuse
scattering from rough surfaces/foliage is currently disabled in the ray-tracing call
(`max_depth=5`, specular/LOS/refraction paths only), so clutter from those surfaces is
absent from ray-traced frames — the separate `apply_clutter` impairment (entry 10)
stands in for road/ground clutter statistically instead.

## 4. Coherent point-scatterer hybrid — `e2e/environment/rt_signal_chain.py` (`coherent_target_cfr`)

**Effect:** the coherent monostatic radar return that plain ray tracing cannot produce
(entry 3's gap).

**Model:** a hybrid, the standard automotive "scattering-center" approach in its
minimal form. Ray tracing still supplies geometry, occlusion, multipath, and the
diffuse (speckle) half of each target's return; a deterministic point (or small set of
points) per object supplies the coherent half. The split is energy-conserving — for a
target with reflection coefficient `S`, the coherent return carries `1 - S²` of the
object's radar cross-section and the diffuse return carries `S²`, so the two halves sum
to the object's total RCS rather than double-counting or losing energy. Where traced
paths reach the object, the phase center is taken from those paths directly (which also
gives correct occlusion silence — an occluded target simply has no traced path to take
a phase center from); otherwise a specular point on the object's bounding geometry is
used as a fallback. Sionna's path-amplitude convention (`paths.a`) is used directly
rather than re-derived, and per-element antenna gain is applied.

**Approximations:** since the model's multi-center revision, each object contributes
several phase centers rather than one: the traced phase center plus the visible
vertical-edge midpoints of the object's own yawed body-frame footprint (computed from
the same `extent_m`/`yaw_rad` geometry the detection labels use), with the far side of
the object culled by adjacent-face visibility so a typical object contributes three to
four centers with equal power split. This is a deterministic geometric placement, not a
measured scattering-center model — there is no aspect-dependent scintillation, and
placement enters aspect-dependence only through which edges the visibility cull keeps.
A single-center mode (one phase center per object) is also available and reproduces the
model's earlier, simpler behavior; it is pinned by a regression test so the two modes
stay distinguishable.

**Evidence:** an internal comparison across the model family (diffuse-only vs.
single-center vs. multi-center) on a shared ray-tracer solve found the multi-center
model monotonically better on every coherence-sensitive metric tested (phase RMS error,
peak-to-background ratio, angular extent of the return) than the single-center model,
which is why multi-center shipped as the default. That comparison was run on one scene
configuration and is a reasoned generalization, not a broad sweep; the falsifying
measurement would be aspect-resolved extent/RCS data from real automotive targets,
which this repository does not have.

**Verdict: JUSTIFIED at the model family's current operating point** for the coherent
half of the return; the diffuse half (entry 3) is the larger open question for how
target returns actually look in the final map, and is out of this entry's scope.

## 5. RF front end (RFFE) — `e2e/circuit/rffe_model.py`, applied via `RFFEBlock`

**Effect:** analog receive-chain distortion — LNA/mixer envelope nonlinearity — and the
receiver's thermal noise floor.

**Model:** a behavioral envelope nonlinearity operating at physical drive levels
(volts in / volts out, not a normalized abstraction), plus thermal noise generated as
`4kTR` at `T = 290 K`, band-referenced to the IF bandwidth and injected at the
baseband-output seam of the chain (deliberately *after* any IF filtering block, so a
downstream filter does not act on it — documented in-code where it's injected).

**Approximations:** behavioral, not transistor-level; a single nonlinearity stands in
for the whole analog chain rather than modeling each stage separately.

**Evidence:** `tests/test_link_budget.py::test_thermal_floor_matches_independent_hand_calculation`
and `::test_target_snr_matches_independent_hand_calculation` check the derived noise
floor and target SNR against an independently worked RF link-budget calculation.

**Verdict: JUSTIFIED.**

## 6. Interconnect — `e2e/blocks.py` (`InterconnectBlock`) + `e2e/data/interconnect/`

**Effect:** the physical interconnect's frequency response between the RF front end
and downstream processing — every real receive chain has *some* transfer function
here, and a flat/ideal assumption hides its cost.

**Model:** `InterconnectBlock` defaults to a fixed placeholder (an 11-tap boxcar
frequency response, independent of the scenario's frequency plan). Passing
`transfer_csv=` switches it to a data-driven mode: it loads a simulated magnitude
response `|S21|(f)` and resamples it onto the scenario's frequency band. Seven derived
datasets ship under `e2e/data/interconnect/` (six 77 GHz automotive interconnect
designs plus a Ka-band TSV interconnect), each derived from an HFSS S-parameter
simulation.

**Provenance and attribution.** The interconnect transfer functions were simulated by
**Mohamed Gharib and Prof. Inna Partin-Vaisband (University of Illinois Chicago)**. The
`.csv` files in this repository are simulated interconnect responses, not laboratory
measurements. The simulation code that produced them is **not** distributed with this
repository and is available on request to those authors.

**Approximations:** the underlying HFSS export is magnitude-only, so the model
reconstructs phase via a minimum-phase assumption (a Hilbert transform of the log
magnitude). That is the physically correct phase for a passive, causal, minimum-phase
structure, but it is an assumption rather than a derived result, and it applies to all
seven datasets.

**Evidence:** the in-band ripple of each dataset predicts its sidelobe floor in the way
transform theory says it should (more ripple, higher/worse sidelobes) — see the README's
interconnect section for the ripple-to-sidelobe numbers and
`python -m e2e.main.main_interconnect` to reproduce the comparison figure yourself.
Regression tests pin the resampled response for each dataset.

**Verdict: JUSTIFIED** for the data-driven mode; the default placeholder remains a
stated, opt-out-of stand-in (see the README's "Interconnect: placeholder vs. simulated"
section).

## 7. Dechirp — `e2e/chain/dechirp.py`

**Effect:** the FMCW receive mix (stretch processing) — in hardware, an analog
per-element multiply of the received signal against the transmitted reference ramp,
producing a low-frequency "beat" signal whose frequency encodes range.

**Model:** an exact change of coordinates from the frequency-domain channel response
to the beat-frequency domain (conjugate plus an antenna-axis flip), bit-exact
invertible and pinned by test; TDM/DDMA MIMO combination is verified against a
reference implementation.

**Verdict: JUSTIFIED** — with one standing caveat that is a code-composability issue,
not a physics one: the block/frame contract currently cannot compose the dechirp stage
with the AFE compression stage (entry 12) in either order within one pipeline run, so
the physically correct sensing chain (dechirp, *then* adaptively compress the
resulting beat signal) is not fully expressible yet. See "Known open items."

## 8. Phase noise — `e2e/chain/impairments.py::apply_phase_noise`

**Effect:** oscillator phase noise, seen through FMCW's range-dependent correlation
property: because the same oscillator generates both the transmitted ramp and the
receive reference, a target's phase-noise contribution partially cancels for near
returns (short round-trip delay, so the oscillator's phase has barely moved) and does
not cancel for far returns — producing "skirts" around strong returns that grow with
range.

**Model:** the residual phase difference `φ(t) − φ(t − τ)` applied in fast time
(genuinely smearing range as the physical effect does), plus a chirp-to-chirp
treatment for the analogous Doppler-axis skirts.

**Approximations:** the round-trip delay `τ` cannot vary continuously per range gate
within a single time-domain multiply, so range gates are grouped into
`n_range_bands` bands that share one `τ`, each applying the same phase-noise draw
(a fidelity/cost knob — more bands means finer range resolution of the effect, at
proportionally higher synthesis cost). The bands are **log-spaced**, with the very
first range gate (`τ = 0`) isolated in its own band — this matters because it is
exactly at `τ = 0` that phase noise should cancel completely (the direct TX/RX leakage
tap has no round-trip delay at all), and a coarse/uniform banding scheme would average
that gate in with a band of otherwise-nonzero delay, destroying the exact cancellation
the physics requires there. The banding used today is asymmetric (fine near zero delay,
coarser further out) for exactly this reason.

**Evidence:** `tests/test_impairments.py::test_phase_noise_zero_delay_gate_is_exactly_cancelled`
pins the zero-delay cancellation exactly; the correlation structure across range and
Doppler is checked by `test_phase_noise_range_axis_correlation` and
`test_phase_noise_doppler_axis_range_correlation` in the same file.

**Verdict: JUSTIFIED.**

## 9. TX–RX leakage, bumper return, and the IF high-pass filter — `e2e/chain/impairments.py`, `e2e/chain/receive.py::IFHighPassBlock`

**Effect:** finite TX/RX isolation (energy leaking directly from the transmitter into
the receive path) and the near-field bumper/radome return — the two strongest,
closest-in tones every automotive FMCW receiver has to contend with — and the analog
high-pass filter real receivers use ahead of the ADC specifically to suppress them
before digitization.

**Model:** the leakage and bumper tones are injected as coherent beat-frequency tones
at their physical beat frequencies, calibrated from an isolation budget (derived
in-code, cross-checked against the link budget). Each transmit/receive antenna pair
gets its own tap carrying that pair's own MIMO code (TDM chirp gating or DDMA code, via
`_mimo_tx_factor`) rather than one shared tap replicated across the array — this
matters because a shared, uncoded tap would, after MIMO demultiplexing, reappear as a
spurious spatially-periodic pattern that has nothing to do with the physical scene (see
entry 10 for the same failure mode in the clutter model, where it was originally
found). `IFHighPassBlock` then applies a causal analog Butterworth high-pass response by
linear (zero-padded-FFT) convolution — matching how a real receiver's anti-leakage
filter behaves — with an edge-replicated settling prefix/suffix so the DC leakage tone
is suppressed essentially completely, and a raised-cosine anti-alias taper at the
Nyquist edge so the one-sided filter kernel's spectrum stays continuous there. The
corner is specified as a physical range (default 1.0 m) and converted to a frequency
via the scenario's chirp slope, so the same setting means the same physical thing
regardless of which radar preset it's applied to.

**Approximations:** the filter's causal settling behavior leaves a residual near the
edge of the suppressed region (roughly the top few percent of the configured range
axis is still somewhat attenuated by the filter's own roll-off) — stated, not hidden.

**Evidence:** `tests/test_chain_receive.py::test_if_hpf_suppresses_off_bin_tone_skirt_at_far_range`
measures the filter's ability to suppress a leakage tone's spectral-leakage skirt at
far range (tens of dB of suppression where an earlier, purely per-bin-magnitude
implementation of the same idea measured essentially none, because a per-bin |H|
weighting is a circular, not linear, operation and cannot remove an off-bin tone's
skirt).

**Verdict: the leakage/bumper tones themselves are JUSTIFIED; the high-pass filter is
now a physically real (linear, causal) filter and is JUSTIFIED.** Whether the filter
alone accounts for a receiver's full false-alarm floor is a separate, still-open
measurement question (see "Known open items").

## 10. Ground clutter — `e2e/chain/impairments.py::apply_clutter`

**Effect:** diffuse road/ground return — many weak, near-zero-Doppler, heavy-tailed
scatterers spread across range.

**Model, per frame:** K-distributed gains (a gamma-distributed "texture" times complex
Gaussian "speckle" — the standard statistical model for heavy-tailed sea/ground
clutter), with scatterer density set per range bin and a small Doppler spread. The
clutter-to-noise ratio is a stated, explicit assumption (flagged as such in-code) rather
than derived from a physical reflectivity model.

**Model, across frames:** the clutter field is drawn once per scenario from a
frame-independent base seed and evolves frame-to-frame by a deterministic
per-scatterer phase advance, rather than being redrawn independently every frame —
independent per-frame redraws would make the ground clutter flicker in a way real,
mostly-static ground clutter does not.

**Model, across the array:** each clutter scatterer draws a real azimuth (uniform in
`sin(azimuth)`, a stated approximation) and contributes its own virtual-array steering
vector plus the correct per-chirp MIMO code (the same `_mimo_tx_factor` convention used
in entry 9), rather than an azimuth-agnostic per-receiver draw. This distinction
matters physically: ground clutter is a spatially extended, azimuth-varying return, not
a per-receiver electronic artifact, and an azimuth-agnostic draw that is later run
through the MIMO deinterleave/demultiplex machinery can produce a spurious, perfectly
periodic spatial pattern that traces the array's channel-multiplexing structure instead
of anything physical.

**Evidence:** `tests/test_impairments.py::test_clutter_persists_across_frames_and_still_evolves`
checks the frame-to-frame temporal behavior (static scene implies correlated, not
independent, frames); `test_clutter_azimuth_recovered_through_tdm_deinterleave` and
`test_clutter_azimuth_recovered_through_ddma_demux` pin that a clutter scatterer's
azimuth survives correctly through both MIMO schemes, and are structured to catch a
sign error in that recovery.

**Verdict: JUSTIFIED**, across all three axes (per-frame statistics, temporal
persistence, and array/MIMO structure).

## 11. ADC quantizer — `e2e/chain/receive.py::QuantizerBlock`

**Effect:** digitization — a fixed full-scale clip plus uniform quantization, which
lays down a noise floor that buries weak targets under strong ones exactly the way
real hardware does.

**Model:** mid-tread uniform quantization, with clipping applied per rail (the real
and imaginary parts of each IQ sample independently) rather than to the complex
magnitude.

**Evidence:** `tests/test_chain_receive.py::test_quantizer_is_uniform_and_matches_the_textbook_adc_snr`
checks the measured quantization noise floor against the textbook `6.02·bits + 1.76`
dB formula for a full-scale sinusoid; `test_weak_signal_sits_at_the_quantization_floor_not_below_it`
checks the floor actually limits weak-signal visibility as it should. The quantizer's
float-precision format is deliberately distinct from the AFE's own float weight format
(entry 12) — they model different hardware (an ADC vs. a compute datapath) and the
distinction is documented rather than left to look like duplication.

**Verdict: JUSTIFIED.**

## 12. AFE / compute-in-memory combiner — `e2e/blocks.py::AFEBlock`, `e2e/chain/compress.py`, `e2e/afe/afe_utils.py`

**Effect:** a compute-in-memory analog combining architecture in which a combining
matrix stored in SRAM performs an analog multiply-accumulate (MAC) and an ADC *in the
same macro* reads the accumulator out — so aperture compression and analog-to-digital
conversion happen as one operation, and converter count scales with the number of
compressed measurements rather than the full array size. (This architecture is
published — see the paper cited in `e2e/afe/afe_utils.py` — and this module is a
software model of it, not a from-scratch design.)

**Model as implemented:** `AFEBlock` quantizes the combining weights using a
low-precision *floating-point* model (constant relative error — appropriate for a
digital compute datapath), then performs an *exact* (infinite-precision) matmul on
those quantized weights. `e2e/chain/compress.py` actually implements a third,
different non-ideality — `COMPUTE_NOISY`, a multiplicative-output-noise model of a
noisy compute-in-memory array (`e2e/afe/afe_utils.py::approx_matmul`) — but `AFEBlock`
does not use that mode; it always takes the quantized-weight/exact-matmul path.

**Approximations, and why the verdict is what it is:** per the published architecture,
the physical chip's weights are a 4-bit *fixed*-point format, and the ADC reading the
accumulator is where the paper reports the dominant source of error — but the
simulator's weight-quantization model is a floating-point format (right operand, wrong
number format) and the accumulator/MAC output itself is never quantized at all (the
step the paper says matters most is currently modeled as infinite-precision). The
building blocks to fix this already exist in `compress.py`
(`COMPUTE_NOISY`/`approx_matmul`) — they are simply not the mode `AFEBlock` is wired to
use.

**Verdict: UNDER-JUSTIFIED** — the cited architecture is real, but the simulator
currently models the wrong halves of it (a float weight format instead of the chip's
4-bit fixed-point, and no output-stage quantization/noise at all). See "Known open
items."

## 13. AdaOja subspace tracker — `e2e/subspace/`

**Effect:** the adaptation loop of the same compute-in-memory hardware (entry 12) —
digital feedback that steers which subspace the analog combining weights track,
online, frame by frame. Waveform-agnostic: the same tracker feeds both the radar path
and, optionally, an OFDM combining head (`ModemBlock(combining="subspace")`).

**Model:** an online subspace tracker (`AdaOjaBlock`) whose default method
(`method="reestimate"`) performs a warm-started power-iteration step on the
back-projected measurements, tracking the array's top-`k` signal subspace at
`O(d·n·k)` cost per step — no SVD, no pseudo-inverse. A legacy incremental-gradient
variant (`method="oja"`) is kept for reference/tests but does not track fast drift
well, because its fixed-size gradient step carries no information about how much the
subspace has actually moved.

**A known failure mode and its fix:** when the tracked rank sits inside a
near-degenerate cluster of singular values (the top-`k` and (`k`+1)-th singular
values are close, i.e. a small `sv_gap_norm` — see the glossary), the "true" subspace
the tracker is chasing is itself ill-defined and rotates unusually fast frame to
frame, and any tracker's error rises during that window regardless of how much
compute it spends. `AdaOjaBlock(gap_response="refine")` (opt-in; default is
`"none"`, which preserves the tracker's original behavior) spends extra refinement
passes reactively, triggered when `sv_gap_norm` drops below a configurable threshold,
rather than spending extra effort on every frame regardless of need.

**Approximations:** the reactive gate is a heuristic keyed on one scalar diagnostic
(`sv_gap_norm`); it does not guarantee recovery within any fixed number of frames, and
some residual tracking-error spikes inside a degenerate window persist at *any* fixed
compute effort, because the underlying target subspace is itself ill-defined there —
that is a property of the scene's singular-value spectrum, not a tracker deficiency.

**Evidence:** `tests/test_subspace_tracking.py::test_reactive_gate_holds_error_down_through_a_degeneracy_episode`
drives a synthetic degeneracy episode (a scene whose subspace briefly becomes
near-degenerate, then recovers) and checks the trajectory, averaged over three seeds
because a single draw is not a stable measurement. It pins two things: the gated
tracker's mean error *through* the episode against a fixed-low-effort arm's — measured
~0.002 gated against ~1.46 ungated, a ~700× margin — and that the extra refinement
passes are spent only inside the episode window, not throughout the run. The test is
constructed to fail against a gate that never boosts effort, so it is a genuine
regression check, not a tautology.

What the test deliberately does **not** assert is any post-episode advantage. An
earlier version did, and it was wrong: once the spectral gap reopens the gate drops
back to `n_refine`, and at one pass per frame neither arm holds this synthetic drift —
both settle at the same error (~1.28). The gate buys accuracy where the diagnostic says
to spend it, not a lasting head start. The test now asserts the two arms track *alike*
after the episode, which is what would catch a gate stuck open.

A separate measurement on real ray-traced `munich` frames (frames 22–59) compares three
arms rather than two, so the trade is visible rather than implied: the reactive gate
matches the constant-maximum-effort arm's accuracy through the collapse (mean error
0.164 against 0.170) at 64% of its compute, while the fixed-low-effort baseline's error
is 8.7× higher. That low-effort arm is a demonstration point, not a production default
effort level.

**Verdict: JUSTIFIED** for the tracker's core algorithm and for the reactive-gating
mitigation, given the pinned synthetic regression test; the real-scenario numbers
above are corroborating, not load-bearing.

## 14. Radar products & windowing — `e2e/blocks.py` (`FFTBlock`, `RangeAzBlock`, `RangeElBlock`), `e2e/ml/baseline.py`

**Effect:** range/Doppler/angle compression into the final radar products, and
detection on top of them.

**Model:** FFT-based range/Doppler/angle processing at native resolution — no
zero-padding, because a zero-padded FFT manufactures sidelobe structure of its own that
does not correspond to anything physical and would misrepresent what the array
actually resolves. A Hann taper is applied on the range and Doppler axes, and (to
control azimuth sidelobes, after they were found to drive a large false-alarm rate)
also on the angle axis (`e2e/blocks.py::_aperture_window`). Detection on top of these
products uses cell-averaging CFAR (`e2e/ml/baseline.py`): each cell's power is
compared against the mean of a surrounding training annulus, with guard cells excluded
so a target's own energy does not pollute its own noise estimate; the guard region is
sized to the measured extent a target actually occupies in the grid.

**Evidence:** `e2e/ml/baseline.py`'s module docstring documents its own evolution
(a pre-CFAR prototype scored far worse and is kept there as a historical, explicitly
non-comparable reference point) and states the CFAR threshold sweep's correspondence
to the detection metric's own score sweep, so the reported operating curve is honestly
a CFAR threshold sweep, not an arbitrary score transform.

**Verdict: JUSTIFIED** — of everything in this document, this stage has the most
direct, reproducible evidence behind it.

## 15. Comms head (OFDM) — `e2e/comms/`

**Effect:** an OFDM communications link running over the same physical receive chain
as the radar path — channel estimation, equalization, BER/EVM.

**Model:** per-subcarrier flat-fading channel estimation and zero-forcing/MMSE
equalization; independent AWGN is injected per array element *before* spatial
combining (`e2e/comms/beamforming.py`), so a reported array gain from combining
(`"mrc"` or `"subspace"` combining modes) is genuine coherent-combining gain, not an
artifact of averaging correlated noise after the fact.

**Approximations, and one currently-unmodeled asymmetry:** the channel is applied as a
per-subcarrier frequency-domain multiply, not a time-domain convolution, so
inter-symbol-interference / cyclic-prefix-overrun effects cannot occur in the shipped
examples (the cyclic-prefix length is exercised only by the modem's own isolated
time-domain round trip, not by any end-to-end example — see the README's "Physical
modeling scope & limitations" section). More importantly: the comms branch currently
sees **no phase noise and no ADC quantization**, because the existing phase-noise and
quantizer implementations are formulated in the dechirped-beat domain that only the
radar path passes through — so today, the comms head is silently the highest-fidelity
signal path in the repository, not because comms hardware is actually better, but
because two of the radar path's impairments have nowhere to attach on that branch yet.

**Verdict: the core estimation/equalization/combining model is JUSTIFIED; the
impairment asymmetry with the radar path is an open item** (see "Known open items").

## 16. Absolute link budget — `e2e/chain/link_budget.py`

**Effect:** the absolute power reference the rest of the chain needs to mean anything
in physical units — the thermal noise floor (`kTBF`), transmit power, and the target
SNR the radar equation predicts for a given range and radar cross-section.

**Model:** standard radar-equation/link-budget relations, with the noise bandwidth
taken equal to the ADC sample rate (`B = fs`), which is the correct convention for
complex (I/Q) sampling — stated explicitly rather than assumed silently.

**Evidence:** `tests/test_link_budget.py::test_thermal_floor_matches_independent_hand_calculation`
and `::test_target_snr_matches_independent_hand_calculation` cross-check the module's
output against an independently worked RF hand calculation; further tests in the same
file check the expected one-for-one scaling with RCS, TX power, and noise figure, and
the expected fourth-power falloff with range.

**Verdict: JUSTIFIED.**

## 17. Motion & scheduling — `e2e/environment/motion.py`

**Effect:** per-frame kinematics for every node and object in a scenario.

**Model:** three motion primitives — constant velocity, waypoint following, and
yaw-about-a-pivot — implemented in plain NumPy and resolved to per-frame trajectories
that the scenario generator consumes; independent of Sionna, so it is unit-testable on
its own.

**Evidence:** unit tests exercise all three primitives directly. (A historical bug that
silently inflated velocities by roughly 10× when more than one frame was generated per
scene has been found and fixed; it is inert at default settings.)

**Verdict: JUSTIFIED.**

## 18. Scene generation & difficulty tiers — `e2e/environment/rt_scenes.py`

**Effect:** a difficulty ladder of generated scenes with real-dimension
vehicle/pedestrian meshes, street-furniture-scale clutter objects, and per-frame
motion, used to build labeled training/benchmark corpora.

**Model:** tiers D1 and above draw target speeds uniformly from a configured speed
range for each generated scenario.

**A cross-layer constraint, and the guard now in place for it:** a scene's drawn
target speeds and a radar configuration's *unambiguous* velocity (the highest speed the
array's Doppler processing can resolve without aliasing) are independent numbers set in
different layers of the codebase. If a scene's speed range exceeds a radar config's
unambiguous velocity, moving targets alias in Doppler, and every detection metric
computed against such a pairing is measuring something other than what it claims to.
Generation now checks this automatically: `e2e.radar_config.answerability_problems`
computes whether a given radar configuration can, in principle, resolve the range/angle
distinctions a benchmark's tolerance and speed range require, and
`e2e.ml.dataset`'s generation entry point refuses to build a corpus that fails this
check unless `--allow-unanswerable` is passed explicitly (see the glossary's
"answerability tier" entry).

**Verdict: JUSTIFIED** — the scene draws themselves are a reasonable difficulty ladder,
and the speed/Doppler cross-layer constraint that used to be silently checkable-only-
by-hand is now an enforced precondition at generation time.

## 19. Analytic ADC synthesizer — `e2e/chain/rd_synth.py`

**Effect:** the classical closed-form stop-and-hop FMCW point-target model — treats the
target as stationary during each chirp, giving an exact analytic beat-frequency signal.

**Role:** not a corpus source (an earlier evaluation judged an all-analytic training
corpus not representative enough of ray-traced physics, and generation was moved to
ray tracing — entry 3 — for that reason). Today this module serves as **oracle
infrastructure**: an independently-derived implementation used to validate ray-traced
output against (the channel-response-to-ADC bridge tests, and the point-scatterer
energy checks in entry 4's tests), and as a fast, millisecond-scale synthetic signal
source for unit tests that don't need ray tracing at all.

**Verdict: JUSTIFIED as an oracle and as a fast synthetic-signal source for tests; it
would be UNJUSTIFIED if used as a training-corpus generator, and it is not used as one.**

## 20. ISAC helpers — `e2e/comms/isac.py`

**Effect:** running both a sensing product and a communications link off one
declarative scene and one shared waveform — range-profile and range-angle estimators
built on the same channel-frequency-response representation the rest of the simulator
produces, rather than a separate sensing-specific data path.

**Model:** shared-waveform estimators consuming the same `s_pars` representation the
radar and comms blocks both operate on.

**Verdict: JUSTIFIED** — it composes existing, already-justified building blocks rather
than introducing a new physical model of its own.

## 21. Detection labels & metrics — `e2e/ml/labels.py`, `e2e/ml/metrics.py`

**Effect:** what a trained (or classical) detector is scored against, and how that
score is computed.

**Model:** a target's detection footprint (the "objectness" region a detector should
light up) is placed on the nearest visible **surface** point of the target, since
returns physically come from a target's surface, not its center of mass; the target's
*center* is kept separately for regression (a downstream tracker integrates
center-of-mass kinematics, which the surface point does not represent well over time).
The analytic oracle (entry 19) places its own point return on that same surface
definition, so the oracle and the labels agree by construction rather than by
coincidence. Metrics use the standard all-points interpolated precision-recall average
precision (the VOC2010+/COCO convention), recall reported at one stated operating
point, and a false-alarms-at-matched-recall metric — each with its known failure modes
documented alongside the implementation.

**Verdict: JUSTIFIED** — alongside entry 14, this is one of the most directly
evidenced parts of the repository; the label/geometry rationale is written down where
the code lives, not just asserted here.

---

## Known open items

These are honest, currently-open gaps rather than resolved verdicts — listed here so a
reader building on top of this simulator knows what not to over-trust yet, without the
internal tracking detail of exactly how or when each will be addressed:

- **AFE output quantization** (entry 12): the compute-in-memory accumulator/ADC stage
  the cited architecture describes is not currently quantized in the simulator, and the
  weight-precision model is attached to the wrong operand (a floating-point model where
  the physical chip uses fixed-point). The building blocks for a fix
  (`e2e/chain/compress.py`'s `COMPUTE_NOISY` mode) already exist but are not wired into
  `AFEBlock`.
- **Dechirp ∘ AFE composability** (entry 7): the block/frame contract cannot currently
  express "dechirp, then adaptively compress the beat signal" in one pipeline run, even
  though that ordering is the physically correct one; today's pipelines compress before
  dechirping instead.
- **Comms-branch impairment asymmetry** (entry 15): the comms head sees no phase noise
  and no ADC quantization, because those impairments are formulated in a domain
  (post-dechirp beat frequency) the comms branch never enters. A physically honest
  comparison between the radar and comms legs of a shared scenario needs a comms-domain
  phase-noise/converter model of its own.
- **Chirp nonlinearity** (entry 1): currently unmodeled; listed as a stated
  approximation rather than an implementation task with a scheduled date.
- **Whether the IF high-pass filter (entry 9) is sufficient to explain a receiver's
  observed false-alarm floor** is still an open, unresolved measurement question — the
  filter itself is now physically real, but its downstream effect on detection-metric
  false-alarm rates has not been isolated and confirmed.

If you pick up one of these, the standing expectation for this repository (see
[`CONTRIBUTING.md`](../CONTRIBUTING.md)) is that a new or changed signal-chain block
gets an entry in this document as part of the same change, not as follow-up
documentation debt.
