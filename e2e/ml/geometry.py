"""Deprecated shim: this module moved to `e2e.environment.geometry` (C1 move,
`notes/C1_MOVE_PLAN.md`) -- it is core geometry (object extents, yaw, the monostatic
surface point), not ML-specific, and the ML package now only consumes it.

Re-exports the old module's public names so existing imports of `e2e.ml.geometry` keep
working. Kept through v1.1; removal left to successors. Import `e2e.environment.geometry`
directly in new code.
"""

import warnings

warnings.warn(
    "e2e.ml.geometry has moved to e2e.environment.geometry; import from there instead. "
    "This shim is kept through v1.1 and will be removed after.",
    DeprecationWarning,
    stacklevel=2,
)

from e2e.environment.geometry import (  # noqa: F401,E402
    BOX_EXTENT_M,
    LOCAL_ASSET_EXTENT_M,
    PEDESTRIAN_EXTENT_M,
    SIONNA_CAR_EXTENT_M,
    Vec3,
    nearest_surface_point,
    object_extent_m,
    object_yaw_rad,
    scene_seed_for,
    surface_range_offset_m,
)
