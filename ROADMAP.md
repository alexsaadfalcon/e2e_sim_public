# Roadmap

This document sketches the post-1.0 direction for the Array Processing End-to-End
Simulator. It is forward-looking and will evolve; nothing here is a commitment.

## Where v1.0 stands

v1.0 delivers a working block-based receive-chain pipeline with three user-facing
capabilities layered on the radar core:

- a **web block-diagram UI** to compose, configure, and run pipelines;
- **scenario scheduling** — a declarative, JSON-serializable `Scenario` spec plus an
  offline Sionna RT generator (with a GPU-free dry-run);
- **communications and joint radar/comms (ISAC)** examples.

The runtime pipeline is fast and CPU-only (consumes precomputed `.pkl` frames); the
heavy ray-traced generation is isolated behind the `[sionna]` extra. The suite is
hands-off and green, packaging is in place, and the real Sionna RT path is validated
on GPU.

## Design north star

Every stage is a swappable **block** over a single data contract (an S-parameter
frame, `[N_RX, N_TX, chirp, N_FREQS]`). The roadmap's through-line is making that
contract **self-describing and general** so blocks compose freely and the geometry
/ waveform assumptions stop being hardcoded.

## Near-term — release polish + the keystone

- **★ A1 — Self-describing frames (KEYSTONE). ✅ DONE 2026-07:** frames carry freq plan / per-link geometry / scale convention; iterator, env block, and web UI consume it; validated on the real GPU path. Original rationale below for context. Today a generated `.pkl` is a bare
  array; the `FrequencyPlan` and per-link array geometry/role the generator already
  holds are dropped at the file boundary, so several runtime consumers re-hardcode
  the 32×32 aperture and the frequency grid. Embed a `meta` block (freq plan, array
  shape, role) in the frame (or a sidecar) and have consumers read it. **Unblocks
  almost everything below.**
- **Subspace-tracking mode.** Make the online (Oja) tracker's warm-start behavior
  explicit/configurable so the reported subspace error reflects a clear, documented
  setting.
- **RF front-end operating point.** Review and document the default RFFE
  configuration (bandwidth/gain/scaling) so the front-end distortion study runs at a
  representative operating point.
- **Subspace ground-truth diagnostic.** A rank / singular-value-gap check on
  `get_U_true`'s input frame so `subspace_err` reports (or flags) when the requested
  `d` exceeds the frame's effective rank, instead of silently tracking noise.
- **Docs/UX:** finish the cookbook; tidy entry points and base-scene options.

## Mid-term — widen the contract, pay down abstraction debt

- **A3 — Widen the runtime shape contract. ✅ DONE 2026-07** (e2e/frames.py accessor + named per-block capability errors). Route geometry through one accessor and
  relax the no-MIMO / single-chirp assertions into scoped per-block capabilities.
  Unblocks **MIMO** and **multi-chirp / Doppler**. (Depends on A1.)
- **A2 — Unify the Block protocol. ✅ DONE 2026-07** (serial stages share the apply-state protocol; feed_forward is two loops; serial_stages override hook; verified bit-exact). Original: give every stage one `apply(state) -> state`
  interface and a typed pipeline state, then collapse the hardwired serial chain into
  a loop. Enables pipeline reordering in the UI and removes the webapp's mirror
  wiring. (Do A1 first.)
- **A4 — Multi-link ISAC consumer. ✅ DONE 2026-07** (main_isac_multilink: metadata-native band/kind, per-frame BER). Original: ship an example that generates a multi-link ISAC
  scenario and consumes the per-link `.pkl` dict directly; unify link naming.
- **Beamforming comms path. ✅ DONE 2026-07:** the pipeline comms head
  (`ModemBlock(combining=...)` + `e2e/comms/beamforming.py`) now does real full-aperture
  spatial combining (MRC / subspace-tracker-derived weights) with independent
  per-element noise injected before combining, so the reported array gain is honest;
  see `python -m e2e.main.main_comms_head` and the web UI's "Comms Head (OFDM)" block.
  Originally scoped as: give the comms leg a spatial-combining option instead of the
  single-tap SISO shortcut. Still open: **link-budget-coupled SNR** below — the
  per-element/per-stream noise target is still set post-hoc rather than derived from
  `tx_power_dbm` / receiver noise figure.
- **Unify the channel-frequency-response loaders** into one helper (folds into A1).
- **Single config/device module** to replace the device globals re-derived per file.
- **Per-node antenna patterns in the UI.**
- **Dual-polarization support through the frame contract.** Sionna's PlanarArray
  reports 2x antenna ports for VH/cross-pol; dual-pol scenarios are currently rejected
  at generation (see CHANGELOG) until the shape contract can carry a pol axis.
- **Link-budget-coupled comms SNR.** Derive the comms noise power from `tx_power_dbm` /
  receiver noise figure instead of enforcing SNR post-hoc, so the comms leg shares the
  radar leg's physical channel gain (see the README's physical-modeling-scope note).
- **Optional diffuse reflection + configurable solver settings.** Ray tracing currently
  runs specular/LOS/refraction only (`max_depth=5`); expose diffuse scattering and
  solver depth/settings for scenes where rough-surface/foliage clutter matters.
- **Aperture tapering / window options for the radar maps.** `FFTBlock` /
  `RangeAzBlock` / `RangeElBlock` currently apply no amplitude taper before the
  aperture FFT; expose windowing (Hamming/Taylor/etc.) for sidelobe control.
- **Interconnect model parameterized by the frequency plan.** `InterconnectBlock` is a
  fixed 11-tap boxcar independent of the scenario's `FrequencyPlan`; make its response
  derive from (or at least track) the configured bandwidth/center frequency.
- **Range FFT should decimate, not truncate.** The range transform passes `n=bins` to
  `torch.fft.fft`, which crops an `n_freqs`≈5000 frequency axis to its first `bins`
  samples — discarding ~95% of the swept band (~13 dB integration gain and the
  documented range resolution). Bin/decimate or chunk the full axis instead.

## Long-term — reach & rigor

- **Live / interactive generation mode** (streaming `step()`-driven generation; hooks
  already exist).
- **Wire the `signal_generator` waveforms** (Narrowband / Wideband / FMCW) into the
  pipeline as a transmit-source stage.
- **GPU-accelerated / batched runtime** (vectorize the RFFE; batch frames).
- **More scenes and full array-size parameterization** end to end.
- **Model validation** against measured or reference data.

## Suggested 1.1 milestone

**"Frames that describe themselves."** A tight, shippable scope:

1. **A1 — self-describing frames** (the keystone; everything compounds off it).
2. **Subspace-tracking mode** + **RFFE operating-point** review (low effort, high
   credibility payoff).
3. **Unify the CFR loaders** (folds into A1).
4. **A3 — widen the shape contract** (A1 makes it nearly free; sets up Doppler/MIMO).
5. Polish: cookbook, base-scene options.

A2 (block-protocol unification) and A4 (multi-link ISAC consumer) are deferred to
1.2 to keep 1.1 focused.
