"""Deprecated shim: this module moved to `e2e.radar_config` (C1 move,
`notes/C1_MOVE_PLAN.md`) -- it is core FMCW radar configuration, not ML-specific, and
sits beside `e2e/scenario.py` as its own dependency-free module.

Re-exports the old module's public names so existing imports of `e2e.ml.radar_config`
keep working. Kept through v1.1; removal left to successors. Import `e2e.radar_config`
directly in new code.
"""

import warnings

warnings.warn(
    "e2e.ml.radar_config has moved to e2e.radar_config; import from there instead. "
    "This shim is kept through v1.1 and will be removed after.",
    DeprecationWarning,
    stacklevel=2,
)

from e2e.radar_config import (  # noqa: F401,E402
    BENCHMARK_V1,
    C_MPS,
    DDMA_WIDE_V1,
    PRESETS,
    RADIAL_LIKE,
    TI_IWR1443,
    RadarConfig,
    answerability_problems,
)
