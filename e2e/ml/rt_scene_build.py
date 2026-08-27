"""Deprecated shim: this module moved to `e2e.environment.rt_scene_build` (C1 move,
`notes/C1_MOVE_PLAN.md`) -- it is core Sionna RT mesh/asset/scene construction, not
ML-specific; `e2e.environment.rt_gen` (also moved) is its main consumer.

Re-exports the old module's public names so existing imports of
`e2e.ml.rt_scene_build` keep working. Kept through v1.1; removal left to successors.
Import `e2e.environment.rt_scene_build` directly in new code.
"""

import warnings

warnings.warn(
    "e2e.ml.rt_scene_build has moved to e2e.environment.rt_scene_build; import from "
    "there instead. This shim is kept through v1.1 and will be removed after.",
    DeprecationWarning,
    stacklevel=2,
)

from e2e.environment.rt_scene_build import (  # noqa: F401,E402
    ASSET_LICENSES,
    CAR_ASSET_NAMES,
    DEFAULT_ANTENNA_PATTERN,
    DEFAULT_GROUND_SCATTERING_COEFFICIENT,
    DEFAULT_SCATTERING_COEFFICIENT,
    DEFAULT_SCATTERING_PATTERN,
    LOCAL_ASSET_SPECS,
    LOCAL_PEDESTRIAN_ASSET_NAMES,
    LOCAL_VEHICLE_ASSET_NAMES,
    LocalAssetSpec,
    PEDESTRIAN_ASSET_NAME,
    RTScene,
    SIONNA_CAR_REPRESENTATIVE,
    SKIN_CONDUCTIVITY_SPM,
    SKIN_RELATIVE_PERMITTIVITY,
    _FLAT_SCENE_XML,
    _GROUND_MATERIAL,
    _OBJECT_COLOR_CLUTTER_BOX,
    _OBJECT_COLOR_DEFAULT,
    _OBJECT_COLOR_PEDESTRIAN,
    _OBJECT_COLOR_SPHERE,
    _OBJECT_COLOR_VEHICLE,
    _YAW_MOVING_EPS_MPS,
    _box_mesh_path,
    _car_mesh_path,
    _default_object_render_color,
    _flat_scene_xml,
    _load_base_scene,
    _load_local_asset,
    _local_asset_dir,
    _local_asset_source_path,
    _object_mesh,
    _pedestrian_mesh_path,
    _read_obj,
    _read_stl,
    _synthetic_scene_path,
    _write_ply,
    build_rt_scene,
    object_local_height_m,
    object_yaw_rad,
)
