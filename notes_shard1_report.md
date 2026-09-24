# Shard 1 report — the one serial spine (MVC) + the FULL contract items

**Clone:** `<scratchpad>/spine`, branched from `c33bb64`. Nothing pushed; the main tree
is untouched. Data dirs (`e2e/ml/runs`, `e2e/ml/datasets`, `e2e/environment/sionna_sims`,
`e2e/environment/sionna_frames`) are NTFS junctions to the main tree; nothing was deleted
through them. GPU: `CUDA_VISIBLE_DEVICES=1` throughout.

**Status:** interim, end of Thursday 2026-09-24. Everything below is committed and
green. Open items are listed in §7.

---

## 1. What the spine looks like

`Simulation` builds ONE stage list, in the contract's §1.2 order, the same way whatever
domain the source starts in:

```
source ─▶ [CircuitStage(RFFE)] ─▶ [InterconnectStage] ─▶ DechirpBlock ─▶ RangeTransformBlock ─▶ [MeasurementStage(AFE+AdaOja)]
                                                                                │                       │
                                                                                │                       ├─▶ RangeAz / RangeEl / AzEl
                                                                                │                       ├─▶ RangeProfile
                                                                                │                       └─▶ SubspaceError
                                                                                └─▶ (RadarCube / detectors read `adc`)
```

Square brackets = present only when configured. The dechirp and the range transform are
never optional: they are what make the chain one chain.

The FULL composition (built explicitly, not yet the auto-default — see §7):

```
source ─▶ TxPowerStage ─▶ DechirpBlock ─▶ FrontEndBlock ─▶ ThermalNoiseBlock(mode="once")
       ─▶ [Impairments] ─▶ [IFHighPass] ─▶ [Quantizer] ─▶ RangeTransformBlock ─▶ MeasurementStage ─▶ products
```

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
 e2e/chain/frontend.py                    | 266 +++++++++++++   (new)
 e2e/chain/link_budget.py                 |  94 ++++-
 e2e/chain/receive.py                     | 236 ++++++++++-
 e2e/chain/waveform.py                    | 321 ++++++++++++++-
 e2e/circuit/rffe_model.py                | 107 ++++-
 e2e/environment/sionna_simple_channel.py |  64 ++-
 e2e/frames.py                            |  64 ++-
 e2e/main/main_sionna_blocks.py           |  12 +-
 e2e/radar_config.py                      |  36 +-
 e2e/simulation.py                        | 428 +++++++++++-------
 tests/test_blocks.py                     | 107 +++--
 tests/test_blocks_range_profile.py       |  82 ++--
 tests/test_chain_compress.py             |  17 +-
 tests/test_full_chain_frontend.py        | 571 +++++++++++++++++++   (new)
 tests/test_one_chain_spine.py            | 647 +++++++++++++++++++++   (new)
 tests/test_simulation.py                 | 148 +++++--
 17 files changed, 3171 insertions(+), 422 deletions(-)
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

## 4. Two findings that contradict prior written claims

**(a) The contract's index map is wrong, and the test says so.**
§5.3 says "antenna flip ⇒ both angle axes mirrored". That is the flip taken *without*
its accompanying conjugate. Derived and measured: `R_new[m,r] = phase ·
conj(R_old[m, (−r) mod N])`, so the **angle power map is unchanged** (the conjugate
exactly cancels the mirror) and only the **range axis is negated**. Every angle-bin
assertion in the migrated tests is therefore unchanged from v1.0, which is itself the
evidence. See `tests/test_one_chain_spine.py::v10_range_reorder`.

**(b) The front-end commutation licence has a drive limit nobody had stated.**
The move onto beat samples rests on the cascade being a function of the envelope alone.
The LNA and mixer are (written in baseband-equivalent envelope form). **The baseband
stage is not**: it clamps I and Q *separately* — a square region in the complex plane,
not a circular one — so once that clamp engages the result depends on the signal's
phase. Measured relative error in `g(e^{iθ}x) == e^{iθ}g(x)`:

| drive | error |
|---|---|
| 1e-6 | 1.6e-7 (float32 rounding; structurally exact) |
| 1e-4 | 1.3e-5 |
| 1e-2 | **1.4e-1** ← clamp engaging |
| ≥3e-2 | 4.7e-1 (saturated) |

The demo operating point (`signal_scaling=1e-7`) is far inside the exact regime, so the
shipped screens are covered. A preset that clips the baseband stage is **not**. Fixing
it properly means a circular baseband clamp, which would move v1.0 numbers and is a
separate decision. Documented in `e2e/chain/frontend.py` and pinned by a test.

**(c) The 125 m claim, retracted in place.** `sionna_simple_channel.unambiguous_range_m`
was documented as "the usable unambiguous range window"; it is the **display half** of a
`num_freqs·c/(2B)` period (F96). The docstring now carries the retraction and the
reason; the stored `meta["unambiguous_range_m"]` keeps its name *and value* (every card
quotes it) with `unambiguous_range_full_m` added beside it.

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

Nothing was deleted. The one assertion retired by design
(`test_range_az_nondivisible_nfreqs_zero_gate_and_energy`) carries its one-line reason
and the contract section.

All 21 owned test files run green: **412 passed, 4 skipped** in 104 s
(`CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg python -m pytest <the 21 files>`, run at
`223db23`). The 4 skips are the marker-gated ones (`sionna`/`slow`/`gui`); the two
`munich_ka`-gated oracles RAN and passed, since the file is present on this machine.

---

## 6. Wall time

Thursday afternoon/evening, ~5 h of agent time. Longest single measurement: the Ka
placement-parity run (~40 s for 5 frames on GPU 1). No full-suite run yet (see §7).

---

## 7. What I could not finish / hand-offs

1. **The FULL composition is not the auto-default.** `Simulation`'s auto-built spine is
   still the MVC one (`CircuitStage` before the dechirp, `ThermalNoiseBlock` absent).
   The FULL order is built explicitly and tested
   (`test_the_full_spine_runs_front_end_after_the_dechirp`). Flipping the default moves
   every T1–T4 number and should be one decision, taken with shard 3 present.
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
00c0202 test(full): pin the placement-parity decision numbers
f1c15eb feat(full): front end on beat samples, noise once, bistatic axis, waveform classes
72be765 fix(spine): read the MIMO guard's cfg from the spine's own dechirp
a3b1bfd test(spine): move the block/simulation tests onto the one chain
6a10a7d fix(spine): strip the UTF-8 BOM PowerShell left on e2e/simulation.py
44b1dc8 feat(spine): one serial chain -- RangeTransformBlock in front of every product
c33bb64 (base) feat(webapp): wave 11 from the owner's live test
```
