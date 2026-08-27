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
    # private names below are consumed by out-of-git reproducibility scripts under
    # notes/tools (a15_clutter_anchor_probe, a17_comet_tail_probe) — the exact class
    # of consumer the shim policy exists to protect (C1 close review finding).
    _k_distributed_gain,
    _mimo_tx_factor,
    _thermal_reference,
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
