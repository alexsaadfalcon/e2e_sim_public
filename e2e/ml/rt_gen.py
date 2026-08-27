"""Deprecated shim: this module moved to `e2e.environment.rt_gen` (C1 move,
`notes/C1_MOVE_PLAN.md`) -- it is core Sionna RT raw-ADC generation, not ML-specific;
the ML dataset/training code is a consumer, not the owner, of ray-traced ADC frames.

Re-exports the old module's public names (its own re-export of `rt_scene_build`/
`rt_signal_chain`/`rt_doppler_study`) so existing imports of `e2e.ml.rt_gen` /
monkeypatch targets keep working. Kept through v1.1; removal left to successors.
Import `e2e.environment.rt_gen` directly in new code.
"""

import warnings

warnings.warn(
    "e2e.ml.rt_gen has moved to e2e.environment.rt_gen; import from there instead. "
    "This shim is kept through v1.1 and will be removed after.",
    DeprecationWarning,
    stacklevel=2,
)

from e2e.environment.rt_gen import (  # noqa: F401,E402
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
    _ANTENNA_INDEX_REVERSED,
    _FLAT_SCENE_XML,
    _GROUND_MATERIAL,
    _add_awgn,
    _beat_from_paths,
    _box_mesh_path,
    _car_mesh_path,
    _coherent_gain,
    _demo_scenario,
    _load_base_scene,
    _load_local_asset,
    _local_asset_dir,
    _local_asset_source_path,
    _object_mesh,
    _peak_reference_amplitude,
    _pedestrian_mesh_path,
    _rd_peak_bin,
    _read_obj,
    _read_stl,
    _resolve_device,
    _snr_reference_chirps,
    _solve,
    _write_ply,
    beat_frequencies,
    build_arg_parser,
    build_rt_scene,
    cfr_from_paths,
    cfr_sum_over_paths,
    cfr_sum_over_paths_budgeted,
    coherent_target_cfr,
    doppler_error_study,
    doppler_validity,
    format_error_study,
    main,
    mimo_combine,
    object_local_height_m,
    rt_cfr_frame,
    rt_retrace_reference,
    rt_synthesize_adc,
    warn_if_doppler_invalid,
)

if __name__ == "__main__":
    raise SystemExit(main())
