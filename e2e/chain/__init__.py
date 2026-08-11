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

The dependency rule runs one way AT MODULE SCOPE: modules here may import from
`e2e.blocks`, `e2e.ml`, and `e2e.frames` freely; nothing in those packages may import
`e2e.chain` at module scope.

There is one deliberate exception, and it is worth stating rather than pretending the
rule is absolute. `e2e/ml/rt_gen.py` imports `e2e.chain.dechirp` *inside functions*,
because the beat-mapping and MIMO-combining code now lives here and rt_gen delegates to
it -- having two copies of conventions that were validated against re-traced ground
truth would be far worse than one lazy import. Function-local keeps it off the import
graph, so no cycle exists at load time.

The hard part of the rule stands: `e2e/ml/dataset.py` stays a pure producer of data.
The dataset blocks depend on it, never the reverse. That is the one genuine cycle this
layout could grow, and it is the one to keep watching.

Every block declares a `frames.FrameCapabilities` naming the signal domain it consumes
(and, for the two bridges, the domain it emits); `Simulation` validates each frame
against that declaration before calling the block. See `e2e/frames.py`.
"""
