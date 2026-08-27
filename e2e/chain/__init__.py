"""Signal-chain blocks: the stages between a transmitted waveform and a radar cube.

This package holds the block families that have no older home in the tree -- waveform
generation and the transmit amplifier, the two domain bridges (modulate and dechirp),
and the receive-side processing (impairments, quantization, radar cubes). Blocks that
belong beside an existing dependency live there instead: `RTEnvironmentBlock` sits in
`e2e/environment/` with the rest of the Sionna code, and the neural detector and
dataset blocks sit in `e2e/ml/` with the models.

**This file imports nothing on purpose.** Each module is imported directly
(`from e2e.chain.dechirp import DechirpBlock`) so that a heavy dependency in one family
cannot be dragged in by a caller that wanted another. That isolation is structural
rather than a rule reviewers have to keep enforcing: a module-scope `import sionna.rt`
here can only ever cost the module that writes it.

The dependency rule runs one way AT MODULE SCOPE, and it inverts the historical
direction: core (this package, `e2e.blocks`, `e2e.simulation`, `e2e.environment`) never
imports `e2e.ml` at module scope; `e2e.ml` may import core freely. That is the point of
the C1 move (`notes/C1_MOVE_PLAN.md`) -- the signal-model modules `e2e.ml` used to own
(geometry/scatterers, and rd_synth/transforms/impairments/link_budget/radar_config) are
core, not ML-specific, and belong on this side of the line.

As of batch 3, `geometry`/`scatterers` (batch 1), `rd_synth`/`transforms`/
`impairments`/`link_budget`/`radar_config` (batch 2), and the RT sextet --
`rt_scene_build`/`rt_signal_chain`/`rt_doppler_study`/`rt_gen`/`rt_scenes`/`assets`
(batch 3) -- have all moved to `e2e.environment`, `e2e.chain`, and `e2e.radar_config`
respectively; this package's own `receive.py` now imports them from their new homes,
closing the hole the rule used to have during the migration.

The one lazy import worth naming, now that it no longer crosses the core/ml line, is
`e2e.environment.rt_signal_chain` (what `e2e.environment.rt_gen` re-exports its
beat-mapping and MIMO-combining names from) importing `e2e.chain.dechirp` *inside
functions*: that code now lives here and rt_signal_chain delegates to it rather than
keeping a second copy of conventions that were validated against re-traced ground
truth. Function-local keeps it off the import graph, so no cycle exists at load time --
it is core importing core, unremarkable under the rule above, but the history is worth
keeping since the two used to live on opposite sides of the line it draws.

The hard part of the rule stands: `e2e/ml/dataset.py` stays a pure producer of data.
The dataset blocks depend on it, never the reverse. That is the one genuine cycle this
layout could grow, and it is the one to keep watching.

Every block declares a `frames.FrameCapabilities` naming the signal domain it consumes
(and, for the two bridges, the domain it emits); `Simulation` validates each frame
against that declaration before calling the block. See `e2e/frames.py`.
"""
