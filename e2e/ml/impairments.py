"""Deprecated shim: this module moved to `e2e.chain.impairments` (C1 move,
`notes/C1_MOVE_PLAN.md`) -- it is a core receive-chain stage (`ImpairmentBlock`
wraps it in `e2e/chain/receive.py`), not ML-specific.

Re-exports the old module's public names so existing imports of `e2e.ml.impairments`
keep working. Kept through v1.1; removal left to successors. Import
`e2e.chain.impairments` directly in new code.
"""

import warnings

warnings.warn(
    "e2e.ml.impairments has moved to e2e.chain.impairments; import from there instead. "
    "This shim is kept through v1.1 and will be removed after.",
    DeprecationWarning,
    stacklevel=2,
)

from e2e.chain.impairments import (  # noqa: F401,E402
    C_MPS,
    DEFAULT_POWER_REFERENCE,
    REFERENCE_NOISE,
    REFERENCE_PEAK,
    REFERENCE_THERMAL,
    ClutterParams,
    LeakageParams,
    PhaseNoiseParams,
    apply_all,
    apply_clutter,
    apply_leakage,
    apply_phase_noise,
    stage_seed,
)
