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

- **★ A1 — Self-describing frames (KEYSTONE).** Today a generated `.pkl` is a bare
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
- **Docs/UX:** finish the cookbook; tidy entry points and base-scene options.

## Mid-term — widen the contract, pay down abstraction debt

- **A3 — Widen the runtime shape contract.** Route geometry through one accessor and
  relax the no-MIMO / single-chirp assertions into scoped per-block capabilities.
  Unblocks **MIMO** and **multi-chirp / Doppler**. (Depends on A1.)
- **A2 — Unify the Block protocol.** Give every stage one `apply(state) -> state`
  interface and a typed pipeline state, then collapse the hardwired serial chain into
  a loop. Enables pipeline reordering in the UI and removes the webapp's mirror
  wiring. (Do A1 first.)
- **A4 — Multi-link ISAC consumer.** Ship an example that generates a multi-link ISAC
  scenario and consumes the per-link `.pkl` dict directly; unify link naming.
- **Unify the channel-frequency-response loaders** into one helper (folds into A1).
- **Single config/device module** to replace the device globals re-derived per file.
- **Per-node antenna patterns in the UI.**

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
