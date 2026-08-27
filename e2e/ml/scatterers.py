"""Deprecated shim: this module moved to `e2e.environment.scatterers` (C1 move,
`notes/C1_MOVE_PLAN.md`) -- it is the core Scenario -> point-scatterer bridge, not
ML-specific, and the ML package now only consumes it.

Re-exports the old module's public names so existing imports of `e2e.ml.scatterers`
keep working. Kept through v1.1; removal left to successors. Import
`e2e.environment.scatterers` directly in new code.
"""

import warnings

warnings.warn(
    "e2e.ml.scatterers has moved to e2e.environment.scatterers; import from there "
    "instead. This shim is kept through v1.1 and will be removed after.",
    DeprecationWarning,
    stacklevel=2,
)

from e2e.environment.scatterers import (  # noqa: F401,E402
    DEFAULT_DT_S,
    DEFAULT_RCS_DBSM,
    SYNTHETIC_BASE_SCENE,
    RadarPose,
    Scatterer,
    frame_scatterers,
    pedestrian,
    radar_pose,
    vehicle,
)
