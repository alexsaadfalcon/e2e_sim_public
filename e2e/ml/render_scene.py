"""Deprecated shim: this module moved to `e2e.render_scene` (C1 move,
`notes/C1_MOVE_PLAN.md`) -- it is a leaf media/visualization tool beside `e2e.viz`,
not ML-specific.

Re-exports the old module public names so existing imports of `e2e.ml.render_scene`
keep working. Kept through v1.1; removal left to successors. Import
`e2e.render_scene` directly in new code.
"""

import warnings

warnings.warn(
    "e2e.ml.render_scene has moved to e2e.render_scene; import from there instead. "
    "This shim is kept through v1.1 and will be removed after.",
    DeprecationWarning,
    stacklevel=2,
)

from e2e.render_scene import (  # noqa: F401,E402
    RENDER_CORPUS_TAG,
    TOP_DOWN_CAMERA_DIR,
    _COLORS,
    _COLOR_CLUTTER,
    _COLOR_PEDESTRIAN,
    _COLOR_RADAR,
    _COLOR_VEHICLE,
    _GT_MARKERS,
    _MARKERS,
    _NEAR_VERTICAL_DOT,
    _OBJECT_FRAMING_RADIUS_M,
    _PEDESTRIAN_FLAG_HEIGHT_M,
    _PEDESTRIAN_FLAG_RADIUS_M,
    _RADAR_BORESIGHT_LEN_M,
    _RADAR_MARKED_CLASSES,
    _RADAR_MARKER_COLOR,
    _RADAR_MARKER_RADIUS_M,
    _RENDER_CAMERA_MARGIN,
    _RENDER_FOV_DEG,
    _TOPDOWN_FRAMING_MARGIN_M,
    _TOP_DOWN_YAW_RAD,
    _VELOCITY_ARROW_S,
    _add_pedestrian_flags,
    _build_camera,
    _build_rt_scene_for_render,
    _caption_render,
    _draw_birdseye,
    _draw_radar_view,
    _fit_camera_position,
    _overlay_legend,
    _render_rt_topdown_frames,
    _resolve_frames,
    argparse,
    build_arg_parser,
    dataclasses,
    inspect,
    main,
    math,
    matplotlib,
    np,
    os,
    plt,
    range_azimuth_map,
    range_azimuth_power,
    render_rt_tier_png,
    render_rt_topdown_gif,
    render_scene_gif,
    render_scene_gif_2x2,
    sys,
    tempfile,
    torch,
)
