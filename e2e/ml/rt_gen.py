"""
Ray-traced (Sionna RT) raw-ADC generation for the FMCW MIMO radar ML package.

This is the high-fidelity sibling of `e2e.ml.rd_synth`: instead of evaluating the
closed-form point-target model, it ray-traces the scene with Sionna RT and turns the
resulting channel frequency response into the **same** dechirped ADC cube --
`complex64 [n_rx, n_chirps, n_samples]` -- so `e2e.ml.transforms`, `e2e.ml.labels`
and `e2e.ml.dataset` cannot tell which generator produced a frame.

This module used to hold all of that in one ~1700-line file; it has since been split
into three focused submodules, kept as ONE import surface here (explicit re-exports
below, no star imports) so every existing `from e2e.ml.rt_gen import ...` / monkeypatch
target keeps working unchanged:

* `e2e.ml.rt_scene_build` -- mesh/asset/scene construction: `RTScene`, `build_rt_scene`,
  the local (unshipped) asset library, `object_local_height_m`, the mesh-path/PLY
  helpers. See that module's docstring for the "Materials" and "Ground-rest placement"
  notes.
* `e2e.ml.rt_signal_chain` -- ray-paths -> ADC signal physics: the CFR->beat mapping
  (`rt_cfr_frame`/`rt_synthesize_adc`/`rt_retrace_reference`), Doppler-validity
  helpers, SNR calibration. See that module's docstring for the load-bearing CFR->beat
  derivation this package used to carry here.
* `e2e.ml.rt_doppler_study` -- the native-vs-re-trace experiment harness
  (`doppler_error_study`/`format_error_study`) and the `python -m e2e.ml.rt_gen` CLI
  (`main`/`build_arg_parser`), unchanged.

Sionna is imported lazily (inside functions) by all three, so `import e2e.ml.rt_gen`
still works on a machine without Sionna/DrJit -- only the generation calls need it.
"""

from __future__ import annotations

# --------------------------------------------------------------------------------
# Back-compat re-exports -- explicit, no star imports. Every name below is imported
# by tests/ or e2e/ elsewhere in this repo (see the wave2/rtgen-split task notes);
# keep this list in sync with those call sites rather than pruning "unused" entries.
# --------------------------------------------------------------------------------
from e2e.ml.rt_scene_build import (  # noqa: F401
    ASSET_LICENSES,
    CAR_ASSET_NAMES,
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
    _box_mesh_path,
    _car_mesh_path,
    _load_base_scene,
    _load_local_asset,
    _local_asset_dir,
    _local_asset_source_path,
    _object_mesh,
    _pedestrian_mesh_path,
    _read_obj,
    _read_stl,
    _write_ply,
    build_rt_scene,
    object_local_height_m,
)
from e2e.ml.rt_signal_chain import (  # noqa: F401
    _ANTENNA_INDEX_REVERSED,
    _add_awgn,
    _beat_from_paths,
    _coherent_gain,
    _peak_reference_amplitude,
    _resolve_device,
    _snr_reference_chirps,
    _solve,
    beat_frequencies,
    cfr_from_paths,
    cfr_sum_over_paths,
    doppler_validity,
    mimo_combine,
    rt_cfr_frame,
    rt_retrace_reference,
    rt_synthesize_adc,
    warn_if_doppler_invalid,
)
from e2e.ml.rt_doppler_study import (  # noqa: F401
    _demo_scenario,
    _rd_peak_bin,
    build_arg_parser,
    doppler_error_study,
    format_error_study,
    main,
)

if __name__ == "__main__":
    raise SystemExit(main())
