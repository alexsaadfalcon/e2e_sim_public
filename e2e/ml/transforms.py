"""Deprecated shim: this module moved to `e2e.chain.transforms` (C1 move,
`notes/C1_MOVE_PLAN.md`) -- it is core signal processing (raw ADC -> range-Doppler),
not ML-specific.

Re-exports the old module's public names so existing imports of `e2e.ml.transforms`
keep working. Kept through v1.1; removal left to successors. Import
`e2e.chain.transforms` directly in new code.
"""

import warnings

warnings.warn(
    "e2e.ml.transforms has moved to e2e.chain.transforms; import from there instead. "
    "This shim is kept through v1.1 and will be removed after.",
    DeprecationWarning,
    stacklevel=2,
)

from e2e.chain.transforms import (  # noqa: F401,E402
    adc_to_rd,
    ddma_demux,
    input_stats,
    normalize,
    rd_power_db,
    rd_to_input,
    tdm_deinterleave,
)
