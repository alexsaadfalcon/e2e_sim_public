# Shard 1 report — the one serial spine (MVC) + the FULL contract items

**Clone:** `<scratchpad>/spine`, branched from `c33bb64`. Nothing pushed; the main tree
is untouched. Data dirs (`e2e/ml/runs`, `e2e/ml/datasets`, `e2e/environment/sionna_sims`,
`e2e/environment/sionna_frames`) are NTFS junctions to the main tree; nothing was deleted
through them. GPU: `CUDA_VISIBLE_DEVICES=1` throughout.

**Status:** interim, end of Thursday 2026-09-24, after the orchestrator's follow-up
(make the FULL composition the auto-default). Everything below is committed and green:
**414 passed, 4 skipped** across the 21 owned test files. Open items are listed in §7.
§4 is written to be filed as F97 as-is.

---

## 1. What the spine looks like

`Simulation` builds ONE stage list, the same way whatever domain the source starts in.
**The default is the FULL composition** (owner 2026-09-24, ballot answer 1B):

```
source ─▶ [TxPowerStage] ─▶ [InterconnectStage] ─▶ DechirpBlock ─▶ [FrontEndBlock]
       ─▶ [ThermalNoiseBlock(mode="once")] ─▶ [Impairments/IFHighPass/Quantizer]
       ─▶ RangeTransformBlock ─▶ [MeasurementStage(AFE+AdaOja)]
                                        │
                                        ├─▶ RangeAz / RangeEl / AzEl
                                        ├─▶ RangeProfile
                                        └─▶ SubspaceError
   (RadarCube / detectors tap `adc`, before the range transform)
```

Square brackets are **physical options**, present only when configured. The dechirp and
the range transform are never optional: they are what make the chain one chain.

`composition="legacy_impulse"` reaches the v1.0 order — `CircuitStage` BEFORE the
dechirp, no link-budget stages — and exists for the stored corpora's bit-parity gate
only (F97c). A legacy `circuit_block=RFFEBlock(...)` is otherwise **translated** onto
the beat placement by `FrontEndBlock.from_rffe`, which copies the per-element config
TABLE rather than re-deriving it from `lna_bias_ma`/`if_bw_mhz` (RFFEBlock applies those
into `rx_config` at construction and does not keep them, so re-deriving would silently
drop a hand-edited table).

Two build rules that are decisions, not hedges:

- **`link_budget="auto"`** (default) adds `TxPowerStage` + `ThermalNoiseBlock` iff a
  `RadarConfig` is available AND the source advertises an absolute amplitude scale.
  `munich_ka` is `physical_scale=False` (measured), so the imaging screens get neither
  — which is F63 and contract §1.2 row 7, not an omission: there is no absolute
  reference for a `k·T·B·F` floor to sit beneath, and `ThermalNoiseBlock` would refuse
  the frame by name if it were built.
- **A chain with a front end and no way to know its beat sample rate is REFUSED**,
  naming all three ways out. The beat-placement noise band is `min(if_bw, fs)`;
  inventing `fs` is a silent multi-dB error in the floor.
  `RFFEBlock.freq_span_hz` is deliberately NOT a fallback — it is the CFR's frequency
  SPAN (3 GHz), a different quantity, ~21 dB out. The cfg is otherwise DERIVED from the
  source's own `freq_plan` (`fmcw_plan_from_freq_plan`), so nothing has to type one.

Structural changes, all in `e2e/`:

- **`RangeTransformBlock`** (new, `e2e/chain/receive.py`) is THE range FFT: `adc` →
  `cube` (new `frames.DOMAIN_CUBE`). It owns the one fast-time window, the one range
  axis (both conventions), and the `crop_negative_delay` flag.
- **The products' own range FFTs are deleted.** `RangeAzBlock`, `RangeElBlock`,
  `FFTBlock`, `RangeProfileBlock` consume the cube and own only their angle transform.
- **The compressor stays SERIAL.** `MeasurementStage` consumes and rewrites `cube`; the
  tracker's snapshots are the cube's range bins instead of the frame's frequency bins.
- **`GridStage` is off the spine.** The element axis must stay flat through the
  compressor, so the aperture view happens inside each product
  (`frames.cube_to_aperture_grid`) and the geometry travels in `state['aperture_shape']`.
- **One feed-forward loop.** `_feed_forward_from` and the "non-CFR source gets an EMPTY
  stage list" branch are gone. Replay is a **start index** into the same list; the
  stages it skips are named in `sim.skipped_stages` / `outputs['skipped_stages']`.
- **Loud failures**: a product without the spine names `RangeTransformBlock`; a
  wrong-waveform cube is refused by `frames.require_cube_axes`; two sources on one chain
  raise; a MIMO frame the spine's dechirp cfg cannot combine raises **naming the cfg**
  (the old refusal came from `GridStage`, which no longer exists on the spine — and this
  matters, because `DechirpBlock` *accepts* MIMO and, told `mimo="single"`, would keep
  only TX 0 and say nothing).

---

## 2. Diff stat

```
 e2e/blocks.py                            | 393 ++++++++++-------
 e2e/chain/frontend.py                    | 293 +++++++++++++
 e2e/chain/link_budget.py                 |  94 ++++-
 e2e/chain/receive.py                     | 236 ++++++++++-
 e2e/chain/waveform.py                    | 321 +++++++++++++-
 e2e/circuit/rffe_model.py                | 107 ++++-
 e2e/environment/sionna_simple_channel.py |  64 ++-
 e2e/frames.py                            |  64 ++-
 e2e/main/main_sionna_blocks.py           |  30 +-
 e2e/radar_config.py                      |  36 +-
 e2e/simulation.py                        | 551 +++++++++++++++++-------
 notes_shard1_report.md                   | 248 +++++++++++
 tests/test_blocks.py                     | 107 +++--
 tests/test_blocks_range_profile.py       |  82 ++--
 tests/test_chain_compress.py             |  17 +-
 tests/test_full_chain_frontend.py        | 571 +++++++++++++++++++++++++
 tests/test_one_chain_spine.py            | 695 +++++++++++++++++++++++++++++++
 tests/test_rffe_physics.py               |  12 +
 tests/test_simulation.py                 | 178 ++++++--
 19 files changed, 3677 insertions(+), 422 deletions(-)
```

`e2e/chain/transforms.py` was **not** touched (F84 training freeze, and the brief
forbids it). `webapp/` and `e2e/ml/` were not touched.

---

## 3. The oracle numbers

| Oracle | Result | Tolerance asked |
|---|---|---|
| Identity: spine vs v1.0 RangeAz/RangeEl/AzEl/RangeProfile, synthetic frame | **≤ 4.0e-7** rel | 1e-5 |
| Identity: same, **real `munich_ka` frame 0**, 1024×5000, bins=256 | **1.7e-7** (range-az), **1.3e-7** (range-el), **6.7e-10** (profile) | 1e-5 |
| Known-delay, both grids (5000 @ B/4999 and 512 @ B/512), both conventions | exact bin, axis to 1e-9 rel | — |
| Compressor still in series: mantissa 6 vs 1, mean \|ΔdB\| | **0.104 dB** (v1.0's own: 0.111) | > 0 |
| AFE off ⇒ cube bit-identical | `torch.equal` | bit |
| Tracker subspace after one warm frame | **0.044140** (v1.0: **0.044097**), Δ = 4.3e-5 | 5e-3 floor |
| Front-end freeze: `CircuitStage` == `fft(rffe(ifft(CFR)))` | `torch.equal` | 0.0 |
| **Noise-once vs analytic Friis cascade** | **−0.021 dB** | 0.2 dB |
| **Placement parity (5 Ka frames, 12 bits), signal path** | **8.6e-05** rel-RMSE vs one LSB **1.76e-04** — BELOW by ~2× | one LSB |
| **Placement parity, noise floor LEVEL** | the two placements agree to **8.1e-7 dB** | — |
| F81-a: +24 dB of P_tx moves the noise-only floor | **0.00 dB** | < 0.5 dB |
| F81-b: +20 dB of NF moves the floor (no front end) | **+20.0 dB** | > 10 dB |
| Legacy `circuit_model_batch` call bit-identical | `torch.equal` | bit |

Other measured numbers now written into the code:

- Munich Ka grid spacing is **600 120.02 Hz** (`B/4999`, endpoint-inclusive), not
  600 000. Full window **499.55 m** bistatic / **249.78 m** monostatic — *not* the
  500/250 m the contract prose quotes. 0.45 m of off-by-one; the axis is derived from
  `freq_plan`, never from a literal.
- OFDM symbol PAPR **5.6 dB** at fft_size 64 (so the TX PA and the front-end clamp are
  genuinely live on that path where they are inert on FMCW).
- RFFE cascade: **24.0 dB** voltage gain, **1.95 dB** noise figure.

---

## 4. Ledger-ready findings — for filing as F97

Four sub-findings, written to be pasted into `notes/ESTABLISHED_FACTS.md` as-is. Each
states what was measured, when, and the conditions it holds under, per the
provenance rule. All four are pinned by tests in this clone; the test name is given so
a later reader can re-verify rather than re-adopt.

---

### F97a — The dechirp's conjugate CANCELS its element-index flip on the angle axis. The contract's "both angle axes mirrored" is wrong.

**Measured** 2026-09-24, derived analytically and confirmed numerically on a synthetic
1024×64 frame and on `munich_ka.pkl` frame 0 (1024×5000).

`DechirpBlock` applies `conj` *and* reverses the RX antenna index
(`e2e/chain/dechirp.py:beat_from_cfr`). `ONE_CHAIN_CONTRACT_2026-09-24.md` §5.3 states
the consequence as "conj ⇒ range bin `k ↔ N−k`; antenna flip ⇒ both angle axes
mirrored". **The second clause is the flip taken without its accompanying conjugate,
and is wrong.**

For an aperture element `v_new[x,y,f] = conj(v_old[X,Y,f])` with `X = n_az−1−x`,
`Y = n_el−1−y`, the joint angle-and-range transform satisfies

```
R_new[m, r] = e^(−j 2π (n_az−1) m / B) · conj( R_old[m, (−r) mod N] )
```

at the mirrored elevation row. Therefore:

- the **angle POWER map is unchanged** — the conjugate exactly cancels the mirror the
  flip alone would produce;
- only the **range axis is negated**: `P_new[k] = P_old[(−k) mod N]`;
- the mirrored elevation row is summed away by the products' non-coherent integration,
  so it is unobservable in RangeAz/RangeEl.

**Scope.** Holds for any product that integrates the collapsed aperture axis
non-coherently (RangeAz, RangeEl, FFTBlock) and for the per-channel range profile up to
a reversal of the CHANNEL index, which that one product does still expose. It is a
property of `ANTENNA_INDEX_REVERSED = True` combined with the conjugate; if either is
changed independently, the cancellation stops.

**Evidence.** Every azimuth/elevation bin assertion in `tests/test_blocks.py` survived
the migration onto the cube **unchanged**, which is the test of this claim.
`tests/test_one_chain_spine.py::v10_range_reorder` carries the derivation; the identity
oracle agrees with the v1.0 products to ≤4.0e-7 relative (synthetic) and 1.7e-7 /
1.3e-7 / 6.7e-10 (munich range-az / range-el / range-profile) with **only** the range
reorder applied.

---

### F97b — The RFFE cascade is phase-equivariant only BELOW baseband clipping: the baseband stage clamps I and Q separately (a square region, not a circular one).

**Measured** 2026-09-24, `benchmark_v1_ka` config, 4 elements × 256 samples, float32 on
CUDA, thermal noise off.

Moving the front end onto beat samples (the FULL contract, §1.1 fact 3) requires the
cascade to be a function of the ENVELOPE alone — i.e. equivariant under a global phase
rotation, `g(e^{jθ}x) = e^{jθ}g(x)`. The LNA and mixer stages are: both are written in
baseband-equivalent envelope form (`rffe_model.py`, "Bandpass cubic nonlinearity,
baseband-equivalent (envelope) form"). **The baseband stage is not**: it applies
`torch.clamp(VBB_I, −Vbias_BB, Vbias_BB)` and the same to Q, which is a SQUARE region
in the complex plane rather than a circular one. Once that clamp engages, the output
depends on the signal's phase.

Relative max error in `g(e^{jθ}x) − e^{jθ}g(x)`, vs input drive amplitude:

| drive | relative error |
|---|---|
| 1e-6 | 1.6e-7 — float32 rounding; structurally exact |
| 1e-4 | 1.3e-5 |
| 1e-2 | **1.4e-1** — baseband clamp engaging |
| 3e-2 | 4.7e-1 |
| ≥1e-1 | 4.7e-1 (saturated) |

**Consequence, and its scope.** The front-end-on-beat-samples placement is exact only
below baseband clipping. The demo operating point (`signal_scaling = 1e-7`; nothing
clips — STATE §5) is far inside the exact regime, so the shipped T1–T4 screens are
covered. **A preset that drives the baseband stage into its clamp is not covered by the
commutation argument** and would need the RF-rate record the contract rules out on cost
(~4.9 GB/frame for munich).

This is a limitation of the model's square baseband clamp, not of the placement. Fixing
it means a circular (envelope) baseband clamp, which would move v1.0 numbers and is
therefore a separate decision, not a quiet edit.

**Evidence.**
`tests/test_full_chain_frontend.py::test_the_cascade_is_phase_equivariant_below_baseband_clipping`
(asserts both the exactness below and the breakdown above). Documented at the head of
`e2e/chain/frontend.py`.

---

### F97c — Moving the front end onto beat samples changes NO measurable quantity, but still fails a bit-parity gate. The legacy placement flag is needed for bit parity only.

**Measured** 2026-09-24 on 5 `b1_demo_cfr_ka` frames (`benchmark_v1_ka` cfg, 16 RX,
512 samples), at 12 bits, comparing the range-compressed cube from the two placements.

**Signal path** (thermal noise off on both arms — see below for why):

```
mean rel-RMSE            8.6e-05
one LSB / |cube| rms     1.76e-04
→ BELOW one LSB, by a factor of ~2
```

**Noise floor LEVEL**: the two placements agree to **8.1e-7 dB** on a zeroed frame
(16 elements × 512 samples). This is exactly the contract's own prediction (§1.1
fact 2): the legacy path injects `NBB·BW_IF / nt` per time sample and the caller's
UNNORMALISED forward FFT multiplies the variance by `nt`, landing on `NBB·BW_IF` per
frequency bin; the beat path injects `NBB·min(if_bw, fs)` per sample with no FFT at
all. **The same number whenever `if_bw ≤ fs`**, which every shipped preset satisfies
(15 MHz IF, 25 MHz fs).

**Why noise must be switched off for the measurement.** With the floor in, the first
run of this measurement returned **1.7e-1** rel-RMSE — 2000× the correct answer. That
is not a placement difference: it is two independent draws of the *same-variance* noise,
which differ by ~√2 × the floor. Reporting it as a placement difference would have been
a wrong diagnosis of exactly the kind this ledger exists to prevent. `inject_noise=False`
was added to both `RFFEBlock` and `FrontEndBlock` so the oracle can isolate the signal
path.

**Consequence.** The legacy `placement="impulse"` /
`Simulation(composition="legacy_impulse")` flag is **NOT load-bearing for numerical
fidelity** — signal below one LSB, floors identical. It **is** load-bearing for BIT
parity: `tests/test_ml_store_cfr.py` and `tests/test_webapp_live_chain.py` read
max |diff| = **0 codes**, and a differently-ordered RNG consumption changes the noise
realization, which fails a zero tolerance at any floor. The flag therefore stays until
the corpora are regenerated, and its justification is "these files on disk were made
that way", not "the physics differ".

**Evidence.**
`tests/test_full_chain_frontend.py::test_placement_parity_on_the_ka_corpus_is_below_one_lsb`
(gated on the corpus being present),
`::test_the_two_placements_agree_on_the_noise_floor_LEVEL`,
`::test_inject_noise_false_is_a_true_bypass_on_both_blocks`.

---

### F97d — The munich Ka unambiguous window is 499.55 m (bistatic) / 249.78 m (monostatic), NOT 500 / 250 m. The stored grid is endpoint-inclusive.

**Measured** 2026-09-24 against the shipped `munich_ka.pkl` header
(`freq_plan = {start 28.5 GHz, stop 31.5 GHz, num_freqs 5000}`).

`sionna_simple_channel.build_frequencies` is `np.linspace(start, stop, num_freqs)` —
**endpoint-inclusive** — so the grid spacing is `(stop − start)/(num_freqs − 1)` =
**600 120.02 Hz**, not `B/num_freqs` = 600 000.0 Hz. A 0.02 % error, which is worth
**~0.45 m at the far end** of the window and ~0.5 m at 125 m.

Derived quantities, all computed from `N·Δf` and never from a nominal `B`:

| quantity | bistatic (`c·τ`, the default since the owner's 2026-09-24 ballot) | equivalent monostatic (`c·τ/2`) |
|---|---|---|
| range per bin | **9.99 cm** | **4.995 cm** |
| full unambiguous window | **499.55 m** | **249.78 m** |
| displayed half (`crop_negative_delay=True`) | 249.8 m | 124.9 m |

The contract's own prose quotes 10 cm / 500 m and 5 cm / 250 m; those use the nominal
`B = 3 GHz` and are the off-by-one this finding names.

**Corpus grids are different and must not be conflated.**
`rt_signal_chain.beat_frequencies` samples `f0 + S·n/fs` for `n < n_samples` — NOT
endpoint-inclusive — so a corpus grid's spacing is `B/n_samples`. Two formulas, two
grid constructions; `e2e/chain/receive.py` has both
(`delta_f_from_freq_plan`, `delta_f_from_cfg`) and the known-delay oracle exercises
both.

**Also retracted here (F96's sibling).**
`sionna_simple_channel.unambiguous_range_m` was documented as "the USABLE
(non-negative-range) unambiguous range window". It is the **display half** of a
`num_freqs·c/(2B)` period: with IQ sampling, beat frequencies `0..fs` map to delays
`0..1/Δf` and **the whole FFT period is physical delay**. The "negative range" the v1.0
screens showed was an artifact of fftshifting and then negating an axis that had no
negative half. The docstring now carries the retraction; the stored
`meta["unambiguous_range_m"]` keeps its name AND value (every card quotes it) with
`unambiguous_range_full_m` added beside it.

**Evidence.**
`tests/test_one_chain_spine.py::test_munich_ka_fmcw_preset_samples_the_stored_grid`,
`::test_delta_f_resolution_order_and_the_two_grid_formulas`,
`::test_known_delay_oracle` (parametrised over both grids),
`tests/test_full_chain_frontend.py::test_the_munich_plan_in_both_conventions`.

---

## 5. Every test file touched, and why

| File | Why |
|---|---|
| `tests/test_one_chain_spine.py` | **NEW** — the MVC oracle suite (§5.3). The v1.0 product math is transcribed *verbatim* from HEAD `c33bb64` rather than shipped as a binary, so it is hands-off from a clean clone; the transcription was checked against tensors dumped from that HEAD before any edit (≤4e-7 on every product). |
| `tests/test_full_chain_frontend.py` | **NEW** — the FULL contract oracles: phase-equivariance and its limit, the Friis floor, noise-once, the bistatic default, the waveform registry, the Ka placement-parity decision. |
| `tests/test_blocks.py` | Products are fed the spine's `cube`. Angle-bin assertions **unchanged** (finding (a)); range assertions move to gate 0 because the cube's bin 0 is zero delay. `test_range_az_nondivisible_nfreqs_zero_gate_and_energy` pinned an indexing rule the consolidation *retires* (the zero gate used to depend on whether `bins` divided `n_freqs`); it now pins the property it was guarding, with the reason on the spot. |
| `tests/test_blocks_range_profile.py` | `_delay_frame` → `_delay_cube`, built through the two spine stages; expected bin is the delay bin itself. |
| `tests/test_simulation.py` | Stage-order assertion updated to the contract's order (GridStage gone); the MIMO refusal moved to `Simulation._check_source_frame`; warm-start oracle goes through `to_beat_basis`; the replay test now asserts the start index and the skipped-stage names; `_ReservedKeyBlock` declares `DOMAIN_ANY` so the domain check cannot make it pass for the wrong reason. |
| `tests/test_chain_compress.py` | The reduced payload is a `cube [M, n_chirp, n_range]`, so its Recorder declares `DOMAIN_CUBE`. |
| `tests/test_rffe_physics.py` | Its full-pipeline test configures a front end, so under the default composition it must name a beat sample rate; one explicit `RadarConfig` added with the reason. |

Nothing was deleted. The one assertion retired by design
(`test_range_az_nondivisible_nfreqs_zero_gate_and_energy`) carries its one-line reason
and the contract section.

All 21 owned test files run green: **414 passed, 4 skipped** in 75 s
(`CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg python -m pytest <the 21 files>`, run at
`35c9cc2`). The 4 skips are the marker-gated ones (`sionna`/`slow`/`gui`); the two
`munich_ka`-gated oracles RAN and passed, since the file is present on this machine.

---

## 6. Wall time

Thursday afternoon/evening, ~5 h of agent time. Longest single measurement: the Ka
placement-parity run (~40 s for 5 frames on GPU 1). No full-suite run yet (see §7).

---

## 7. What I could not finish / hand-offs

1. ~~The FULL composition is not the auto-default.~~ **DONE** (`35c9cc2`) — it is the
   default; see §1. **This moves every T1–T4 number**: shard 3 must re-render and the
   hostile rounds must read the new pixels, not the wave-11 ones.
2. **The two `tests/test_ml_link_budget.py` xfails are still xfails.** Both properties
   now hold and are tested at block level (§3), but those two tests run through
   `e2e/ml/chain_generate.py`, which composes the legacy chain and which I am forbidden
   to touch. **Shard 2**: compose `TxPowerStage → Dechirp → FrontEndBlock →
   ThermalNoiseBlock(mode="once")` there and the xfails flip.
3. **`adc_to_rd` is not yet refactored to call `RangeTransformBlock`.**
   `e2e/chain/transforms.py` is under the F84 freeze and outside my brief. Until it
   lands, `tests/test_one_chain_spine.py::test_adc_to_rd_range_half_parity` is what
   keeps the two copies honest.
4. **`RadarCubeBlock` still reads `adc`** (not `cube`), deliberately: it calls the frozen
   `adc_to_rd`, and T5's live-vs-stored gate reads max |diff| = 0. A ~10-line change once
   (3) lands.
5. **`notes/JSAC_WAVEFORM_2026-09-24.md` does not exist yet.** `JSACSignal` is registered
   and raises `NotImplementedError` with the contract and both candidate hybrids written
   into its docstring.
6. **`ofdm`'s RECEIVE bridge is not wired.** The class builds the transmitted symbol and
   keeps the grid the receiver divides by; `Y/X → cube` lives in `e2e/comms/` (shard 2).
7. **No full-suite run yet** — the brief says once, at the end. `webapp/` and `e2e/ml/`
   tests will be red until shards 2 and 3 land, so it is the orchestrator's integration
   gate, not mine.

### Things shard 3 must know now (interface + consequences)

- **`Simulation(composition=...)`** defaults to `"full"`. If a chain configures a front
  end it MUST be able to name a beat sample rate — pass `radar_cfg=`, or rely on the
  source's `freq_plan` (both `SionnaEnvironmentBlock` and the corpus sources carry one),
  or pass `front_end=FrontEndBlock(..., fs_hz=...)`. Otherwise the build raises with all
  three ways out named. `composition="legacy_impulse"` is for the T5 stored-vs-live gate
  and nothing else.
- New `Simulation` kwargs: `composition`, `front_end`, `link_budget`, `radar_cfg`,
  `range_transform`. New attributes: `sim.composition`, `sim.link_budget_active`,
  `sim.radar_cfg` (resolved), `sim.skipped_stages`.

- State keys: `cube` (`DOMAIN_CUBE`), `cube_axes`, `range_axis` (metres, float64, or
  `None` when the frame carries no `freq_plan`), `range_axis_m_per_bin`,
  `range_convention`, `range_n_fft`, `range_delta_f_hz`, `range_cropped`,
  `aperture_shape`, `waveform` / `waveform_kind`, `noise_injected_by`,
  `front_end_placement`, `skipped_stages`.
- `RangeTransformBlock(cfg, window=, dc_removal=, convention=, crop_negative_delay=,
  delta_f_hz=)`. Defaults: `window="hann"`, `dc_removal=True` (the ML protocol); the
  imaging spine passes `window="none", dc_removal=False` (the LoS path sits at bin 0 on
  `normalize_delays=True` traces, and the mean subtraction would zero it).
- **`crop_negative_delay` defaults to True**, keeping bins `0 … N//2` — exactly the half
  `pipeline_runner._nonnegative_range` kept. **Consequence:** the products now bin
  2501 native bins down to 256 gates instead of 5000, so range-per-gate **halves** —
  today's "1 m/gate" card line becomes ~0.50 m/gate (monostatic) or ~1.00 m/gate
  (bistatic, the new default). Compute it from `state['range_axis_m_per_bin']`, never
  from a literal.
- **The metre convention default is now `bistatic_path`.** Every card metre doubles
  relative to the v1.0 numbers: 37.1 m → 74.2 m, 68 m → 136 m, "5 cm" → "9.99 cm",
  window 249.78 m → 499.55 m.
- `webapp/pipeline_runner.py`'s `_range_axis`, `_nonnegative_range`,
  `_cropped_nonneg_range_axis`, `_range_per_gate_m`, `_native_unambiguous_range_m` are
  all superseded by the spine's one axis, and `_native_unambiguous_range_m`'s `B/n_freqs`
  is the off-by-one (§3).

### `git log --oneline` of the clone (this shard)

```
35c9cc2 feat(spine): the FULL composition is the auto-default (owner 1B, no hedging)
c8b254e docs(shard1): exact test tally (412 passed, 4 skipped) instead of an estimate
223db23 docs(shard1): interim report -- spine, FULL items, oracle numbers, hand-offs
00c0202 test(full): pin the placement-parity decision numbers
f1c15eb feat(full): front end on beat samples, noise once, bistatic axis, waveform classes
72be765 fix(spine): read the MIMO guard's cfg from the spine's own dechirp
a3b1bfd test(spine): move the block/simulation tests onto the one chain
6a10a7d fix(spine): strip the UTF-8 BOM PowerShell left on e2e/simulation.py
44b1dc8 feat(spine): one serial chain -- RangeTransformBlock in front of every product
c33bb64 (base) feat(webapp): wave 11 from the owner's live test
```
