"""Deprecated shim: this module moved to `e2e.environment.assets` (C1 move,
`notes/C1_MOVE_PLAN.md`) -- it is core Sionna RT asset handling (download/extract/
decimate/normalize the vehicle meshes the ray tracer places), not ML-specific; the ML
package only consumes it via `e2e.environment.rt_scene_build`/`rt_scenes`.

Re-exports the old module's public names so existing imports of `e2e.ml.assets` keep
working. Kept through v1.1; removal left to successors. Import `e2e.environment.assets`
directly in new code.
"""

import warnings

warnings.warn(
    "e2e.ml.assets has moved to e2e.environment.assets; import from there instead. "
    "This shim is kept through v1.1 and will be removed after.",
    DeprecationWarning,
    stacklevel=2,
)

from e2e.environment.assets import (  # noqa: F401,E402
    ARCHIVE_SPECS,
    DECIMATE_MAX_TRIS,
    DOWNLOADED_ASSET_SPECS,
    DOWNLOADED_BUS_ASSET_NAMES,
    DOWNLOADED_CAR_ASSET_NAMES,
    DOWNLOADED_TROLLEY_ASSET_NAMES,
    DOWNLOADED_TRUCK_ASSET_NAMES,
    DownloadedAssetSpec,
    _read_obj_np,
    _read_stl_np,
    _vertex_cluster_pass,
    decimate_to_budget,
    ensure_extracted,
    normalize_mesh,
    process_all,
    process_asset,
    processed_dir,
)
