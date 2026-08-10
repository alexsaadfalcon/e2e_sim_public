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

The dependency rule runs one way: modules here may import from `e2e.blocks`, `e2e.ml`,
and `e2e.frames`; nothing in those packages may import from `e2e.chain`. In particular
`e2e/ml/dataset.py` stays a pure producer of data -- the dataset blocks depend on it,
never the reverse -- which is the one import cycle this layout could otherwise grow.

Every block declares a `frames.FrameCapabilities` naming the signal domain it consumes
(and, for the two bridges, the domain it emits); `Simulation` validates each frame
against that declaration before calling the block. See `e2e/frames.py`.
"""
