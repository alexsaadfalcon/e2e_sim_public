"""Deprecated shim: this module moved to `e2e.environment.rt_scenes` (C1 move,
`notes/C1_MOVE_PLAN.md`) -- it is core difficulty-tiered RT scenario construction
(pure Python/numpy, no Sionna needed), not ML-specific; the ML corpus generators are
consumers, not the owner, of these scenarios.

Re-exports the old module's public names so existing imports of `e2e.ml.rt_scenes`
keep working. Kept through v1.1; removal left to successors. Import
`e2e.environment.rt_scenes` directly in new code.
"""

import warnings

warnings.warn(
    "e2e.ml.rt_scenes has moved to e2e.environment.rt_scenes; import from there "
    "instead. This shim is kept through v1.1 and will be removed after.",
    DeprecationWarning,
    stacklevel=2,
)

from e2e.environment.rt_scenes import (  # noqa: F401,E402
    RT_DIFFICULTY_TIERS,
    VEHICLE_CLASS_POOLS,
    VEHICLE_CLASS_WEIGHTS,
    VEHICLE_FOOTPRINT_M,
    _CLUTTER_LOS_MARGIN_SIN_AZ,
    _CLUTTER_LOS_RANGE_MARGIN_M,
    _asset_vehicle_class,
    _draw_vehicle_asset,
    _footprint,
    _footprint_radius,
    _stable_seed,
    build_rt_tier_scenario,
    tier_summary,
    vehicle_asset_class,
)
