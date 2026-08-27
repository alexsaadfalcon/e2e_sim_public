"""Deprecated shim: this module moved to `e2e.chain.rd_synth` (C1 move,
`notes/C1_MOVE_PLAN.md`) -- it is core signal synthesis (receive.py's DechirpBlock
mirrors it), not ML-specific.

Re-exports the old module's public names so existing imports of `e2e.ml.rd_synth`
keep working. Kept through v1.1; removal left to successors. Import
`e2e.chain.rd_synth` directly in new code.
"""

import warnings

warnings.warn(
    "e2e.ml.rd_synth has moved to e2e.chain.rd_synth; import from there instead. "
    "This shim is kept through v1.1 and will be removed after.",
    DeprecationWarning,
    stacklevel=2,
)

from e2e.chain.rd_synth import (  # noqa: F401,E402
    C_LIGHT,
    RadarPose,
    array_axis,
    device,
    synthesize_adc,
)
