# Shard 2 report — the ML/corpus chain, the comms fold-in, and the three waveform classes

**Clone:** `<scratchpad>/shard2`, branched from shard 1's `da3b76a`. Nothing pushed; the
main tree and the spine clone are untouched. Data dirs (`e2e/ml/runs`, `e2e/ml/datasets`,
`e2e/environment/sionna_sims`, `e2e/environment/sionna_frames`) are NTFS junctions to the
main tree; nothing was deleted through them. GPU: `CUDA_VISIBLE_DEVICES=1` throughout.

**Rung reached on the fallback ladder: the TOP one.** `jsac` is implemented, green and
oracled — image *and* BER from one frame, with the resource-split knob — so neither the
`ofdm`-only rung nor the no-knob rung was needed. Details in §4.

---

## 1. The composition in `chain_generate`

`build_chain_simulation` builds THE CONTRACT'S ORDER by default
(`composition="full"`, `e2e/ml/chain_generate.py`):

```
RTEnvironmentBlock / SourceBlock (ray-traced CFR + labels)
  -> _ChainFlagsStage              # provenance, incl. composition + front_end_placement
  -> [CFRCaptureStage]             # store_cfr
  -> [WaveformBlock/TxPABlock/ModulateBlock]   # use_transmit_chain, off by default
  -> [TxPowerStage]                # sqrt(P_tx) AT THE SOURCE      (link budget on)
  -> [InterconnectStage]
  -> DechirpBlock
  -> [FrontEndBlock]               # the RF front end ON THE BEAT RECORD
  -> [ThermalNoiseBlock(mode="once")]          # the ONE thermal injection
  -> ImpairmentBlock
  -> [IFHighPassBlock]
  -> QuantizerBlock
  -> SinkBlock                     # SERIAL now, see below
  -> RangeTransformBlock           # the ONE range FFT, on the scored protocol
     downstream: [RadarCubeBlock]  # the Doppler half only
```

Three placements that are decisions, not conveniences:

- **The sink is a serial stage**, between the quantiser and the range transform. What a
  corpus sample stores is the digitised beat record; the range transform crosses into
  the cube domain, and `Simulation` drops the previous domain's payload at a crossing
  (it must — a stale `adc` outliving the crossing is how a block computes silently on
  pre-transform data). So the sink runs where the record it persists still exists.
- **`FrontEndBlock.from_rffe`** carries every `RFFEBlock` knob across, including a
  hand-edited per-element config table, so no caller restates its knobs.
- **`composition="legacy_impulse"`** reaches the v1.0 order and exists for the stored
  corpora's bit-parity gates only. Every frame's meta now records `composition` and
  `front_end_placement`, so a live-vs-stored diff can name its cause.

### The two `test_ml_link_budget.py` xfails are gone — both pass

Measured on the ML fixture (`_CFG` + `_FakeRTEnvironment`), 2026-09-24:

| property | FULL | v1.0 order, same fixture, same run |
|---|---|---|
| noise-only floor vs +24 dB of `P_tx` | **0.00 dB** | +22.34 dB (**F81**) |
| noise figure 5 → 25 dB, **no front end** | **+20.00 dB** | +1.12 dB |
| noise figure 5 → 25 dB, **front end present** | **+0.04 dB** | — |

The third row is the one worth reading twice. It is **not** a residual defect: contract
§1.4 says that when a front end is present the floor is *its own Friis cascade's* and
`cfg.noise_figure_db` is not in that path. So `test_noise_figure_moves_the_corpus_noise_floor`
now says `use_rffe=False` with that scope written into it, and a companion test
(`test_cfg_noise_figure_is_inert_behind_a_front_end`) pins the other half so the dial's
real scope is read off the suite rather than rediscovered from a flat sweep. A corpus
generator that wants to sweep difficulty behind a front end must move the front end's own
knobs.

**A consequence to design around, not discover.** `sqrt(P_tx)` at the source means
transmit power now reaches the receiver's nonlinearity. On this fixture the cube tracks
`P_tx` to 0.05 dB below −24 dBm and loses **16.8 dB of gain over the last 24 dB of
drive** (−60 → −53.09, −48 → −41.05, −36 → −29.03, −24 → −17.04, −12 → −5.28,
0 → +4.01, +12 → +6.96, +24 → +7.24 dB). Under the v1.0 order this could not happen —
`P_tx` was applied *downstream* of the front end. The scaling test moved to the measured
linear span and the compression is pinned separately
(`test_transmit_power_drives_the_front_end_into_compression`). **Shard 3 / preset work
should check that no demo preset sits in that compressed region by accident.**

The retracted behaviour is kept reachable and kept MEASURED
(`test_the_legacy_composition_still_carries_F81`), so "someone fixed the legacy arm" and
"the corpora stopped reproducing" cannot be confused.

---

## 2. Parity numbers

`adc_to_rd` is now `RangeTransformBlock` + the new `rd_from_cube`, operation for
operation; `RadarCubeBlock` consumes the spine's `cube` and applies only the Doppler
half. `transforms.RD_RANGE_PROTOCOL` names the scored protocol (hann / DC removal /
uncropped) in ONE place and `RadarCubeBlock` refuses a cube built off it by name.

### The Ka corpus, 5 scenes of `b1_demo_cfr_ka`

Each scene replayed from its own `.cfr.npy` sidecar with its OWN recorded seeds,
impairment severities and IF-HPF corner fed back in.

| arm | result |
|---|---|
| **legacy, full chain, vs the stored `adc`** | `torch.equal`, max \|diff\| = **0.0**, all 5 scenes |
| **legacy, full chain, vs the stored cube** | `torch.equal`, max \|diff\| = **0.0**, all 5 scenes |
| **FULL vs legacy, SIGNAL PATH** (every injection off) | rel-RMSE **3.63e-05 .. 4.82e-05**, mean **4.08e-05**; max \|diff\| 3.8e-07 .. 3.4e-06 |
| one LSB, referred into cube units | **2.52e-05 .. 4.05e-05** RMS (0.26 %–0.76 % of the cube's own RMS) |

**At or below one LSB in RMS, one to two orders below it in peak.** The legacy flag is
load-bearing for BIT parity only — "these files on disk were made that way", not "the
physics differ". That is F97c, re-measured on the FULL default.

**The control is the measurement.** With noise on, the two compositions draw INDEPENDENT
realisations (different tensor shapes at different points), and the same question returns
**rel-RMSE ≈ 0.97** — four orders larger and entirely a property of the RNG. An earlier
measurement of this exact question returned 1.7e-1 for that reason. The test disables
every injection on both arms and says so; the legacy arm's noise draw is *patched to
identity* rather than disabled, because its `"legacy"` mode bundles `sqrt(P_tx)` into the
same call and disabling it would put the two arms on different absolute footings.

Pinned in `tests/test_ml_composition_parity.py` (2 scenes, `slow`-marked, gated on the
corpus), which passed first run — an independent re-derivation of the same numbers.

### Two things recorded rather than smoothed over

1. **An unexplained magnitude.** With noise ON, the ADC-level gap between the FULL replay
   and the stored corpus (~3.6e-4) is larger than two independent draws of a 1.26e-6-σ
   floor explain. Plausible contributor: under FULL the floor is the front end's own
   cascade rather than `cfg.noise_figure_db = 15 dB`, so the two placements' floors may
   differ in LEVEL and not only in realisation, and the leakage/clutter severities are
   referenced to whatever floor is local to them. The mechanism is pinned; the magnitude
   on this corpus is **not measured** and no claim is made about it.
2. **`adc_to_rd` does NOT TDM-de-interleave while `RadarCubeBlock` does.** On this
   `mimo="tdm"` config their outputs are `[16,512,256]` and `[64,512,64]` — not
   comparable. The asymmetry predates this work; the parity test uses `RadarCubeBlock`'s
   own computation as the reference and names the substitution rather than making it
   silently.

---

## 3. The comms package, folded onto the chain

`Simulation(comms_head=[...])` inserts the comms blocks as **serial stages before the
mixing block**, in the frequency domain — the `IC → comms head` edge on the contract's one
diagram. Serial and not downstream for a structural reason: downstream blocks run after
the whole spine, by which point the chain is in the cube domain and `s_pars` has been
dropped. Whatever a tap returns IS recorded in `outputs`, so `out["ber"]` still works.

The head emits only `comm_*` keys, so it changes nothing the sensing products read: a tap,
not a branch — which is why the old "the comms head is mutually exclusive with the ADC
chain" rule has nothing left to protect.

**One noise source.** `ModemBlock` was the third floor, beside the front end's cascade and
the link budget's kTBF. `add_noise=None` (the default) is AUTO: no AWGN of its own
whenever `state['noise_injected_by']` is set. Pinned STRUCTURALLY — two applies of one
frame are bit-identical — because a power comparison passes either way.

**The equaliser's SNR is measured.** When the chain is the noise source, nothing in the
pipeline knows the post-combining SNR, so it comes from the pilot residual
(`channel.estimate_snr_db`, new) and is reported as `comm_snr_db` beside
`comm_noise_source`. `None` back (one symbol, nothing to pool) falls through to
zero-forcing, which needs no SNR and makes the same hard decision.

`Simulation` also seeds `state['U']` at the HEAD of the chain from the tracker's current
basis, so a tap can beamform with the tracker's estimate **as of the previous frame** —
the causally correct thing for a receiver to have, since this frame's tracker output does
not exist until this frame has been received. It is overwritten after the serial loop, so
no downstream product's view changes.

---

## 4. The three waveform classes — the interface shard 3 builds against

A class is the triple **(source waveform, mixing mode, product set)**:

| `kind` | source | `mixing` | products |
|---|---|---|---|
| `fmcw` | chirp | `"dechirp"` | sensing only |
| `ofdm` | OFDM grid | `None` | comms only |
| `jsac` | OFDM grid | `"symbol_division"` | **both, from one frame**, with a resource split |

`ofdm` having **no mixing block** is the substantive part: a comms receiver equalises the
grid and never forms a cube. That is what makes the difference between `ofdm` and `jsac` a
*screenshot* — same waveform, same front end, one extra block, and a radar image appears —
rather than a claim in prose.

### The one call shard 3 needs

```python
from e2e.comms.ofdm_isac import waveform_chain_spec
spec = waveform_chain_spec(kind, cfg, freq_plan=<the source's plan>,
                           n_symbols=4, pilot_spacing=8, bits_per_symbol=2,
                           sensing_source="preamble", combining="mrc")
```

`ChainSpec` fields: `.kind`, `.mixing_block` (the stage that replaces `DechirpBlock`, or
`None`), `.sensing` / `.comms` (whether to build the sensing half / the comms head),
`.channel_block` (`OFDMChannelBlock`, the `Y = H·X` apply, goes before the interconnect),
`.receive_block` (`OFDMReceiveBlock`, goes in `comms_head=`), `.frame` (the `OFDMFrame`),
`.notes` (every card-bound number, all COMPUTED — `sample_rate_hz`, `data_rate_bps`,
`frame_duration_s`, `sensing_window_m`, `qam_noise_rise_db`, `delta_f_hz`, `cp_len`,
`pilot_spacing`, `sensing_source`).

**State keys** the JSAC path adds or changes:

| key | value |
|---|---|
| `tx_grid` | the transmitted grid `X [M, N]` — what the mixer divides by and the demapper demaps against |
| `cube_axes` | `{"slow": "symbol", "fast": "subcarrier"}` before the range transform, `{"slow": "symbol", "fast": "range_bin"}` after it |
| `sensing_reference` | which `sensing_source` produced the cube |
| `comm_snr_db` | the SNR the equaliser actually used — MEASURED on the chain path |
| `comm_noise_source` | `"frontend"` / `"modem"` / `"none"` |
| `comm_array_gain_db` | realised (estimated-weights) array gain |

`RangeTransformBlock` now **carries the slow-axis label through** from the mixing block
instead of stamping `"chirp"` on every cube.

**Product ids** are unchanged (`range_az`, `range_el`, `fft`, `range_profile`,
`subspace_err`, `radar_cube`, `ber`, `evm`, `comm_*`). Two contract changes:

- The angle products and the range profile accept **any** slow axis
  (`require_cube_axes(..., slow=None)`): they broadcast over it and read only the fast
  one, so a JSAC cube's SYMBOL axis is as meaningful to them as a chirp axis.
  `RangeAzBlock` on an `M`-symbol frame emits `[M, bins, gates]`.
- **`RadarCubeBlock` refuses a symbol-slow cube by name.** Its product IS an FFT along
  that axis, and a Doppler transform over OFDM symbols of one time-invariant stored
  channel is a delta at bin 0 dressed up as a velocity. **The JSAC screen shows
  range-azimuth, not range-Doppler** — and the FMCW arm on the same single-chirp munich
  frames has exactly the same degenerate Doppler axis, so this is a property of the
  corpus, not a defect of JSAC. Say it that way from the floor.

### The knobs, in the order they matter

1. **`pilot_spacing P` with `sensing_source="pilots_only"`** — the resource split, built
   FIRST per the review. One knob, both products, opposite directions: the sensing
   window is `c/(P·Δf)` and shrinks by `P` (the scene's multipath visibly aliases) while
   `P−1` of every `P` subcarriers carry data and the rate rises. `P = 1` is all-pilot:
   zero bits, and it is the O2 parity point.
2. **`bits_per_symbol`** — QPSK ↔ 16-QAM ↔ 64-QAM. Measured floor rise from the repo's
   own constellation: **QPSK 0.00 / 0.00 dB, 16-QAM +2.762 / +6.990 dB, 64-QAM
   +4.290 / +13.222 dB** (mean / worst subcarrier). Xiong et al.'s deterministic–random
   tradeoff in one A/B.
3. **`combining`** — `element0` / `egc` / `mrc`. `subspace` is deliberately NOT offered
   on the OFDM path: it would read the tracker's basis, and the tracker sits downstream
   of the mixing block.

### The CHIRP_SINGLE lift

The review enumerated five blocks that reject an `M`-symbol frame and offered three
resolutions, two of which fork the spine. The restriction was never a property of the
algorithm — the sensing matrix `A` acts on the element axis alone — it was a property of
`cube.view(-1, n_range)`, which folds the slow axis INTO the element axis and tracks the
subspace of something that is not an aperture. That is a genuine silent-wrong-answer, so
the pin was right to exist; fixing the reshape retires it.

`MeasurementStage` now stacks one snapshot column per `(slow index, range bin)` pair via
`cube.reshape(n_el, n_slow * n_range)`. On a single-slow cube that is **the same tensor,
bit for bit** (asserted), so no FMCW number moves. `RangeProfileBlock` integrates the slow
axis non-coherently (identity for one slow index) with a `slow_index=` knob a JSAC preset
uses to show symbol 0. `AFEBlock` / `AdaOjaBlock` declare `_SNAPSHOT_STACK`.

Three tests pinned the restriction; all three are rewritten in place to pin the property
it was guarding, with the reason on the spot. None deleted.

### The PAPR question — answered by measurement, and it runs BACKWARDS

The review's item 1. `RFFEBlock` normalises by the MEAN magnitude of `ifft(s_pars)`, so
whether the clamp engages is governed by the peak-to-mean of *that* tensor, not by the
transmitted waveform's PAPR. On the munich Ka trace the LoS tap carries ~93 % of the PDP
energy, so `ifft(H)` is a spike: **the shipped FMCW preset is ~27 dB peakier at the LNA
than a JSAC frame will be** (FMCW as shipped 36.67 dB; FMCW with the waveform+modulate
blocks enabled 3.95 dB; JSAC 9.61 dB; the TX-side OFDM symbol 9.55 dB at Nyquist).

`measure_lna_input_papr` is provided so a card quotes a measured number for the
configuration it actually runs, and a test pins the DIRECTION so the requirement cannot
lapse. **No card may claim "the OFDM waveform is what makes the front end clip" without
naming which FMCW configuration it is A/B'd against and quoting both numbers at that
preset's own operating point.** As shipped, the claim is false.

The withdrawn number: the design note's "99.9th pct 12.41 dB" was a percentile of pooled
instantaneous sample power, not a per-symbol PAPR percentile, and sits above the observed
per-symbol maximum. `measure_papr_db` quotes the mean at a STATED oversampling factor
(9.55 dB at Nyquist, ~10.0 dB at 4×).

---

## 5. The oracles, and what each is worth

`tests/test_waveform_classes.py`, 29 tests, all green.

| # | Oracle | Result |
|---|---|---|
| **O2** | FMCW parity, all-pilot symbol | **`torch.equal`** on the `adc` AND the cube, every symbol. Not a tolerance. |
| O2′ | constant-modulus QPSK grid | rel < 1e-5 (float noise; the division is exact for unit modulus) |
| **O1** | known delay, both arms | exact cube bin at 0 / 7 / 23, written against `N·Δf` and never `B` |
| **O5** | handedness, through the shipped angle product | the two arms' images peak in the SAME `(azimuth bin, range gate)` |
| **O4** | division noise | 16-QAM +2.762 / +6.990 dB, 64-QAM +4.290 / +13.222 dB; measured through the chain against what THIS draw predicts, to ±0.05 dB |
| **O3** | BER floor | **exactly 0.0** on a clean flat channel and on a random multipath CFR |
| — | resource split | window and rate move in opposite directions; `P=1` gives 0 b/s |
| — | measured SNR | +20 dB of injected noise moves the reported SNR by 20 ± 3 dB; one symbol returns `None`, never a guess |

**What O2 is worth, stated because it will be over-read otherwise.** It certifies the
PLUMBING — conjugate present, antenna flip present, `mimo_combine` indexing right, FFT
direction right, symbol axis mapped to the chirp axis right. It CANNOT catch a wrong `Δf`,
a wrong frequency ordering, a CP bug or anything in the data-bearing path, because both
arms share all of those. It is the right gate for "is this one chain or two" and the wrong
gate for "is the image correct". **O1 and O5 carry the physics.** O2 is bit-exact only
because `SymbolDivisionBlock` IMPORTS `beat_from_cfr` and `mimo_combine` rather than
reimplementing them; if it ever needs a tolerance, the tails have drifted and that is the
bug.

---

## 6. JSAC on the REAL munich Ka frames — the demo claim, measured

`munich_ka.pkl` frame 0, `[1024, 1, 1, 5000]`, one frame, one waveform, one front end.
Everything below is derived from the source's own `freq_plan`; nothing is typed.

| quantity | value |
|---|---|
| Δf | **600 120.024 Hz** (endpoint-inclusive — not 600 000) |
| JSAC sample rate | **3.0006 GS/s** (FMCW's plan: 25.005 MS/s over a 199.96 µs sweep) |
| symbol / frame duration | 1.6663 µs / 6.6653 µs at `cp_len=0`, `M=4` |
| burst data rate | **3.938 Gb/s** QPSK, raw uncoded burst (`~262 kb/s` average at 10 Hz — state the duty cycle or say it is undefined) |
| cube | `[1024, 4, 2501]`, axes `{"slow": "symbol", "fast": "range_bin"}` |
| range per bin / displayed window | **0.0999 m / 249.78 m** (bistatic path, F97d's numbers, derived) |
| **O2 on the real frame** | **`torch.equal = True`, max \|diff\| = 0.0** |
| image | `range_az [4, 256, 256]`; symbol 0 peaks at the LoS (gate 0), 38.8 dB dynamic range |
| comms, SAME frame | **BER 0.0**, EVM 3.0e-3, MRC array gain **30.10 dB** = `10·log10(1024)` exactly |
| PDP (99 % / 99.9 % / 99.99 %) | 517.2 ns / 756.5 ns / 971.5 ns — reproduces the design note digit for digit |

**The resource-split knob, on the real plan** (`sensing_source="pilots_only"`):

| `P` | sensing window | QPSK burst rate |
|---|---|---|
| 1 (all-pilot) | 499.55 m | **0 Gb/s** — the O2 parity point |
| 4 | 124.89 m | 3.376 Gb/s |
| 8 | 62.44 m | 3.938 Gb/s |
| 16 | 31.22 m | 4.219 Gb/s |

One knob, both products, opposite directions, on one frame.

**The array-gain figure has a scope.** 30.10 dB is the IDEAL-weights number, and the
weights came from a NOISELESS preamble, so there is no estimation loss to see. The
review's R-new-3 (MRC weights are a single noisy snapshot of a 1024-dimensional vector
per subcarrier, and nobody has bounded the loss) is answered **only for the noiseless
case**. `measure_array_gain_db` is exported so the lossy case can be measured against
ideal weights in one line; it has not been.

---

## 7. Hand-offs

### 7.1 SHARD 3 / the webapp — 15 tests are RED and it is my change (SUPERSEDED, see 9.2)

`webapp/pipeline_runner.py`'s `serial_stages_override` ends at `QuantizerBlock` and puts
`RadarCubeBlock` downstream. `RadarCubeBlock` now consumes `cube`, so that chain raises:

```
PipelineError: Pipeline constraint failed: RadarCubeBlock expects the cube domain,
but the chain is in the rx_time domain -- insert a RangeTransformBlock before it.
```

Red: `tests/test_webapp_live_chain.py` (5) and `tests/test_webapp_run_notes.py` (1).
**The fix is to append `e2e.chain.transforms.range_transform_for(adc_cfg)` to that
override list** — that helper pins the SCORED protocol (`RD_RANGE_PROTOCOL`: hann, DC
removal, uncropped) that `RadarCubeBlock` now checks, so the T5 live-vs-stored gate keeps
reading max |diff| = 0. I did not make the edit: shard 3 owns the file and the contract
has it deleting `serial_stages_override` wholesale, so the fix lands inside work that
must happen anyway. No other test outside `webapp/` is red.

Two more things shard 3 needs:

- **`composition="legacy_impulse"`** on any chain whose job is to reproduce a stored
  corpus bit-for-bit. `build_chain_simulation` takes it; `Simulation` takes it.
- **`waveform_chain_spec`** (§4) is the one place the three classes' rows live. Read the
  dropdown, the diagram's branch point and the preset's product list off it rather than
  writing a second `if kind == ...` ladder.

### 7.2 Card text that is BLOCKED on a measurement

No card may claim that the OFDM waveform is what drives the front end into its clamp. As
shipped it is false twice over (§4, §6): the preamble symbol IS the FMCW preset's tensor,
and the data symbols are 17.3 dB *less* peaky than the bare CFR. If a drive-knob A/B is
wanted, it must name which FMCW configuration it is against and quote
`measure_lna_input_papr` for both at that preset's operating point.

### 7.3 Open, and deliberately not closed

1. The **ADC-level gap with noise on** between the FULL replay and the stored corpus
   (~3.6e-4) is bigger than independent draws of a 1.26e-6-σ floor explain. The
   mechanism is pinned; the magnitude is unmeasured (§2).
2. **MRC's estimation loss** with a noisy preamble is unbounded (§6).
3. **Peak GPU allocation** for a 1024 × 5000 × M JSAC frame in a web callback is not
   measured. The symbol division was rewritten to avoid materialising expanded copies
   (164 MB per symbol-set), but the review's R2 asked for a rehearsal-time measurement
   and that has not been done.
4. **The cyclic prefix has no numerical effect in this model** — the channel is applied
   as a per-subcarrier multiply, which is a cyclic convolution by construction, so no CP
   length can create or prevent ISI. `cp_guard_ok` checks the premise the multiply is
   justified by; it cannot detect ISI, because the model cannot produce any. Said once,
   out loud, in the module docstring.

---

## 8. The examples (`e2e/main/`), and a defect found while re-pointing them

- **`main_comms_head.py`** — the comms head moves into `comms_head=`; the front end's
  beat-sample-rate refusal is answered with a `radar_cfg` DERIVED from the example's own
  frequency plan. `add_noise=True` is passed EXPLICITLY: this example compares combining
  modes at a stated `snr_db`, so it is the one that keeps its own noise source. Measured
  over 5 frames at `snr_db=5`: element0 BER 6.7e-2 / egc 0.0 (29.99 dB) / mrc 0.0
  (30.10 dB) / subspace 1.3e-2 (23.23 dB). `subspace` now reads the tracker's basis as
  of the **previous** frame (the head is tapped before the mixing block) — causally
  correct, and stated in the code.
- **`main_comms_link.py` / `main_channel_estimation.py`** — the channel is sourced
  through the chain's own `InterconnectBlock`; the sweep's AWGN stays explicit because
  it is the swept variable. BER(ZF)@20 dB **4.340e-03** and @0 dB **2.929e-01** — unmoved
  from before the re-point, which is the honest null result for a 15 MHz comms band
  inside a 3 GHz sweep.
- **`main_isac.py`** — re-pointed onto `waveform_chain_spec("jsac", ...)`: one CFR, one
  frame, both products. **SCOPE CHANGE, and the orchestrator should confirm it:** the old
  version ran the multi-node scenario's TWO independent channels (spatial-division ISAC)
  through a hand-rolled range-angle map. The `jsac` spec is waveform-division ISAC and
  has one channel, so the second node's link is named in the printout but no longer
  driven. **If the two-channel story still needs a home it now needs its own script.**

### 8.1 KNOWN DEFECT: at Ka the interconnect is a CONSTANT, not a filter

Found while deciding which interconnect the examples should use.
`DEFAULT_INTERCONNECT_CSV` covers **70–90 GHz**; `_interconnect_band_hz` hands a Ka
config **27–33 GHz**, entirely below it; `InterconnectBlock` clamps out-of-range grid
points to the CSV's endpoints. Measured 2026-09-24 on a `benchmark_v1_ka` frame:

| config | band | \|S21\| ripple | phase span |
|---|---|---|---|
| `benchmark_v1` (77 GHz) | 75–81 GHz | **0.0336 dB** | 0.089° |
| `benchmark_v1_ka` (30 GHz) | 27–33 GHz | **0.000000 dB** | **0.000000°** |

Every grid point of the Ka response is bit-identical at −0.4861 dB / +1.2946°. The
"0.034 dB in-band ripple" `chain_generate` quotes was measured at 77 GHz and holds only
there.

**Consequence.** Every Ka corpus (`b1_bench_v3_ka`, `b1_bench_v4_ka`, `b1_demo_cfr_ka`)
was generated with `use_interconnect=True` and an interconnect that contributed a
scalar. A constant complex gain changes no measurable quantity downstream, so **the
corpora are not wrong** — but any claim that they carry a *modelled* interconnect at Ka
is, and **a Ka card must not say the interconnect shapes the band on this path.** If T4's
"in-band |S21| moves <0.03 dB" null result is quoted for a Ka screen, check which
interconnect source that screen runs: the CSV path is inert there, while
`InterconnectBlock(source="tessera")` has a geometric scale model (F91) built precisely
for a non-77 GHz carrier and is presumably fine.

**NOT FIXED, deliberately.** Changing the band, the CSV or the clamp changes what
`use_interconnect=True` computes, which regenerates every Ka corpus and breaks the
bit-parity gates. `test_the_ka_interconnect_is_a_clamped_constant` pins the current
behaviour AND the 77 GHz ripple so the defect cannot be fixed into a parity break by
accident, and `_interconnect_band_hz`'s docstring carries the measurement, its date and
its conditions.

The two comms examples therefore use the CSV's **band-agnostic** mode, which maps its own
span across the frame's samples — the real response shape on a nominal frequency mapping
— rather than passing the true band and silently getting a constant. Which of the two you
have is stated at the call site.

---

## 9. What the adversarial review found, and what changed because of it

A fresh-context reviewer was given the diff, the author's claims as claims to test, and
an instruction to reproduce rather than reason. It confirmed the exactness claims
(`torch.equal` for `adc_to_rd == RangeTransform + rd_from_cube` across odd/even/single
shapes on CPU and CUDA; the `reshape`/`view` equivalence; the PAPR and power-sweep
docstring numbers; the Ka corpus parity tests re-run against the real corpus). It found
**one blocker**, and it was right.

### 9.1 The blocker: the noise coupling did not exist on the path I claimed it on

`ModemBlock`'s "no longer injects its own AWGN when the chain already did" was FALSE in
the integrated pipeline. The comms head is tapped before the mixing block; under the
FULL composition the front end runs AFTER the dechirp, so `noise_injected_by` has not
been written when the tap executes. The reviewer built a real `Simulation` with a fully
configured front end and link budget and observed `comm_noise_source == "modem"` and
`comm_snr_db == 10.0` (the constructor value) — the pre-refactor behaviour, silently,
with the front-end knobs moving the image and not the BER.

**My tests could not have caught it**, and that is the more useful half: they set
`state['noise_injected_by']` by hand and tested the boolean in isolation, so they could
not see which SIDE of the tap a stage lands on.

Two different things were wrong and they needed different answers:

1. **A missing seam.** `CircuitStage` — the FREQUENCY-domain front end, which is where
   the front end belongs for OFDM/JSAC, because `ifft(s_pars)` of a received OFDM grid
   IS the received time-domain symbol — never stamped `noise_injected_by`. It does now.
   On that chain the coupling is real: one front end, one floor, both products, one knob.
2. **An overclaim.** On an FMCW chain no fix is possible and none should be attempted:
   the beat-placement front end acts on `adc`, a tensor that does not exist at the tap,
   in a domain the head does not read. **No stage ordering can make its noise reach a
   channel-frequency-response tap.** So the head is correctly its own noise source there,
   says so, and **the front-end knobs do not move the comms BER on an FMCW arm.** That
   scope is now in `ModemBlock`'s and `OFDMReceiveBlock`'s docstrings.

Two real-pipeline tests now pin BOTH halves. **Card consequence: "one knob moves both
products" is true on the JSAC screen and false on any FMCW screen with a comms panel.**

### 9.2 The webapp hand-off is bigger than one line — I tried the fix and reverted it

§7.1 said the 15 red webapp tests were a one-line repair (append `range_transform_for`
to `serial_stages_override`). **I made that change and it does not work**, so it is
reverted and the analysis is the hand-off instead:

```
PipelineError: CFARDetectorBlock expects the rx_time domain, but the chain is in
the cube domain -- insert a DechirpBlock before it.
```

The collision is architectural. `RadarCubeBlock` now consumes `cube`; the detectors must
keep consuming `adc` (contract §1.2 — their scored front ends own their own transforms,
which is what keeps F85/F95 bit-for-bit). Both are DOWNSTREAM blocks, downstream blocks
run after the whole serial list, and `Simulation` drops the previous domain's payload at
a crossing — deliberately, because a stale `adc` outliving the crossing is how a block
computes silently on pre-transform data. So no placement of the range transform in the
serial list satisfies both, and T5 enables both.

**The shape of the fix that does work**, and shard 3 already has the mechanism: make the
products ORDERED TAPS rather than a flat downstream list, using the `_product_stages`
machinery added for the comms head. Each product taps the domain it belongs to, in order,
on one chain:

```
... -> Quantizer -> [CFAR / learned detectors]   (taps, read `adc`)
                 -> RangeTransformBlock
                 -> [RadarCubeBlock]              (tap, reads `cube`)
```

That is arguably more faithful to the contract than "everything is downstream", and it is
inside the rewrite shard 3 owes anyway (the contract deletes `serial_stages_override`).

**Baseline, measured so the attribution is exact.** At shard 1's HEAD `da3b76a`, running
the same five webapp test files: **5 already red** (3 in `test_demo_presets.py`, 2 in
`test_webapp.py`), all `"the 'full' composition puts the front end on the beat record,
which needs the beat SAMPLE RATE"` — shard 1's hand-off, answered by giving the runner a
`radar_cfg`. **15 more are mine**, all one cause (`test_webapp_detector.py` 1,
`test_webapp_live_chain.py` 12, `test_webapp_run_notes.py` 2). Nothing outside `webapp/`
is red.

---

## 10. Test results, diff stat, git log

**Full suite, once, at the end** (`CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg python -m
pytest -p no:randomly`, at `2d3c419`):

```
20 failed, 2383 passed, 184 skipped in 581.46s (0:09:41)
```

All 20 failures are `webapp/`-driven and analysed in §9.2: **5 pre-existing at shard 1's
`da3b76a`** (measured by running the same five files in the untouched spine clone) and
**15 mine, from one cause**. Nothing outside `webapp/` is red. The `slow`-marked Ka
parity tests skip by default and pass under `RUN_SLOW=1` (verified independently by the
reviewer against the real corpus).

### Diff stat

```
 e2e/blocks.py                       | 125 ++++--
 e2e/chain/frontend.py               |   7 +
 e2e/chain/receive.py                |  87 +++-
 e2e/chain/transforms.py             |  84 +++-
 e2e/chain/waveform.py               | 265 ++++++-----
 e2e/comms/blocks.py                 | 122 ++++-
 e2e/comms/channel.py                |  45 ++
 e2e/comms/ofdm_isac.py              | 867 ++++++++++++++++++++++++++++++++++++
 e2e/frames.py                       |  24 +-
 e2e/main/main_channel_estimation.py |  50 ++-
 e2e/main/main_comms_head.py         |  32 +-
 e2e/main/main_comms_link.py         |  72 ++-
 e2e/main/main_isac.py               | 221 +++++----
 e2e/ml/chain_generate.py            | 177 +++++++-
 e2e/simulation.py                   |  49 +-
 tests/test_chain_end_to_end.py      |   8 +
 tests/test_chain_receive.py         |  65 ++-
 tests/test_comms_beamforming.py     |   3 +-
 tests/test_comms_blocks.py          | 143 +++++-
 tests/test_full_chain_frontend.py   |  56 ++-
 tests/test_ml_chain_generate.py     | 166 ++++++-
 tests/test_ml_composition_parity.py | 219 +++++++++
 tests/test_ml_link_budget.py        | 173 ++++++-
 tests/test_one_chain_spine.py       |  38 +-
 tests/test_simulation.py            |  53 ++-
 tests/test_waveform_classes.py      | 616 +++++++++++++++++++++++++
 26 files changed, 3363 insertions(+), 404 deletions(-)
```

### `git log --oneline`

```
2d3c419 fix(comms): the noise coupling is real only where the front end is on the TAP's side -- scoped, stamped and pinned
3e8d7b1 feat(main): the comms/ISAC examples run on the chain -- and a KNOWN DEFECT found doing it
ed48a0a fix(comms): the JSAC path on the REAL munich Ka frames -- and the PAPR story's sharpest edge
dea423c test(ml): the Ka corpus parity question, measured on the real corpus with its control
c7c78b8 feat(comms): the comms head becomes a TAP on the one chain, with one noise source
d78668d feat(comms): three waveform CLASSES on one spine -- fmcw / ofdm / jsac, with the oracles
d5aaa53 feat(blocks): lift CHIRP_SINGLE off the compressor and the range profile; the slow axis is a snapshot axis
7e561e2 feat(ml): chain_generate composes the FULL contract chain; the two F81 xfails become passes
7cc5308 feat(chain): one range transform -- adc_to_rd calls RangeTransformBlock, RadarCubeBlock reads the cube
```
