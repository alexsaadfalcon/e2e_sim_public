"""Environment blocks that source pipeline frames from Sionna RT ray tracing.

`RTEnvironmentBlock` is the `e2e.ml` radar-generation family's counterpart of
`e2e.blocks.SionnaEnvironmentBlock`: instead of iterating precomputed `.pkl`
S-parameter frames, it ray-traces `e2e.ml.rt_gen`'s FMCW radar scene per frame and
exposes the same `get_S_pars()`/`step()`/`reset()`/`array_shape` interface
`e2e.simulation.Simulation` expects -- so a declarative `Scenario` + `RadarConfig` can
drive the runtime pipeline directly. It emits the RAW channel frequency response
(DOMAIN_CFR, pre-dechirp, see `e2e/frames.py`); `e2e.chain.dechirp.DechirpBlock(cfg)`
is the bridge from there to a dechirped ADC cube.

Sionna is imported LAZILY (inside `get_S_pars`, via `e2e.ml.rt_gen`'s own lazy
imports) so `import e2e.environment.blocks` never requires DrJit/Sionna -- matching
`e2e.environment.scenario_runner`'s disciplined pattern.

`base_scene="munich"`/`"etoile"` (or any other Sionna built-in city scene / custom
scene path) works here at whatever frequency `cfg` describes -- including automotive
mmWave, where Sionna's bundled ITU materials would otherwise hard-raise -- via
`e2e.environment.city_scenes` (see that module's docstring for the mechanism and the
`material_policy` trade-off). `"flat"`, the default, is unaffected: it never enters
that path.
"""

from __future__ import annotations

from typing import Optional, Sequence


class RTEnvironmentBlock:
    """Ray-traced environment block: one `Scenario` + `RadarConfig` -> per-frame raw
    `s_pars` (DOMAIN_CFR) + ground-truth detection labels.

    Interface (mirrors `e2e.blocks.SionnaEnvironmentBlock`):
      * `get_S_pars()` -- ray-traces the CURRENT frame (`self.frame_counter`) and
        returns `complex64 [n_rx, n_tx, n_chirp, n_samples]` on `self.device` -- the
        raw CFR, NOT yet dechirped; feed it to `DechirpBlock(cfg)` for `adc`.
      * `step()` / `reset()` -- advance / rewind `self.frame_counter`, wrapping at
        `scenario.num_frames` (same semantics as `SionnaEnvironmentBlock`).
      * `array_shape` -- `(cfg.n_rx, 1)`: `e2e.ml.rt_gen.build_rt_scene` gives the
        radar a 1 x `cfg.n_rx` receive ULA (not a 2-D planar array), so the aperture
        grid it factors into is one-dimensional.

    Ground truth labels: `get_S_pars()` also computes this frame's detection labels
    from the SAME scatterer/pose solve used for the ray trace (no re-derivation
    downstream), caching them on `self.last_labels` (`e2e.ml.labels.
    encode_detection_labels`'s float32 `[3, n_range, n_azimuth]` map) and
    `self.last_targets` (the raw `(range_m, sin_azimuth, object_class)` list from
    `targets_in_grid`). Read them right after `get_S_pars()`, before the next `step()`
    (or the next `get_S_pars()` call at a new frame) overwrites them.

    Those labels reach the chain through `get_state_updates()`, which `Simulation` calls
    when it seeds each frame's state (see `Simulation._environment_state_updates`). That
    is why the labels are computed HERE rather than at the far end of the chain: they
    describe the scene that produced THIS frame, and a downstream re-derivation would be
    reading a scene that has since stepped. A caller driving this block directly can
    still read `last_labels`/`last_targets` after `get_S_pars()`.

    City-scene material provenance: when `base_scene` is not `"flat"`/`"free"`,
    `get_S_pars()` also runs `e2e.environment.city_scenes.prepare_scene_for_frequency`
    on the loaded scene (see that module) and caches its report on
    `self.last_material_report` (`{material_name: MaterialSubstitution}`, empty if
    nothing needed substituting) -- read alongside `last_labels`/`last_targets` for a
    corpus's provenance of what its city was actually made of.

    A fresh scene is ray-traced (via `e2e.ml.rt_gen.build_rt_scene` + `rt_cfr_frame`) on
    every `get_S_pars()` call -- the same per-frame rebuild `rt_synthesize_adc` does by
    default (not an incremental/cached scene) -- so moving-object geometry is always
    exactly resolved for `self.frame_counter`.
    """

    def __init__(self, scenario, cfg, *, base_scene: str = "flat", device=None,
                max_depth: int = 2, include_leakage: bool = False,
                diffuse_reflection: bool = True, specular_reflection: bool = True,
                refraction: bool = False, solver_seed: int = 41, freq_chunk: int = 128,
                scattering_coefficient: Optional[float] = None,
                scattering_pattern: Optional[str] = None,
                material_policy: str = "extrapolated",
                stand_in_material: str = "concrete",
                label_grid=None,
                label_classes: Optional[Sequence[str]] = ("vehicle", "pedestrian")):
        self.scenario = scenario
        self.cfg = cfg
        self.base_scene = base_scene
        self.device = device
        # Only consulted for city scenes (base_scene not "flat"/"free") -- see
        # `e2e.environment.city_scenes.EXTRAPOLATED`/`STAND_IN`.
        self.material_policy = material_policy
        self.stand_in_material = stand_in_material
        self.max_depth = max_depth
        self.include_leakage = include_leakage
        self.diffuse_reflection = diffuse_reflection
        self.specular_reflection = specular_reflection
        self.refraction = refraction
        self.solver_seed = solver_seed
        self.freq_chunk = freq_chunk
        # None -> rt_gen's own defaults (DEFAULT_SCATTERING_COEFFICIENT/_PATTERN),
        # resolved lazily in get_S_pars so this constructor stays Sionna-import-free.
        self.scattering_coefficient = scattering_coefficient
        self.scattering_pattern = scattering_pattern
        # None -> LabelGrid.for_config(cfg)'s defaults; see e2e.ml.labels.LabelGrid.
        self.label_grid = label_grid
        self.label_classes = tuple(label_classes) if label_classes is not None else None

        self.frame_counter = 0
        # 1 x cfg.n_rx receive ULA (see build_rt_scene) -- not a 2-D planar array, but
        # Simulation/frames.to_aperture_grid only need a factorization of n_rx.
        self.array_shape = (int(cfg.n_rx), 1)
        # Populated by get_S_pars(); see the class docstring's "Ground truth labels".
        self.last_labels = None
        self.last_targets = None
        # Populated by get_S_pars() for city scenes; see "City-scene material
        # provenance" above. Stays None for "flat"/"free".
        self.last_material_report = None

    def get_state_updates(self):
        """Per-frame state `Simulation` seeds the chain with, alongside the frame.

        Returns the ground truth for the frame `get_S_pars()` just produced, so labels
        travel WITH their frame down the chain. Empty before the first `get_S_pars()`.
        """
        if self.last_labels is None:
            return {}
        return {"labels": self.last_labels, "targets": self.last_targets}

    def reset(self):
        self.frame_counter = 0
        self.last_labels = None
        self.last_targets = None
        self.last_material_report = None

    def step(self):
        self.frame_counter += 1
        if self.frame_counter >= int(self.scenario.num_frames):
            self.frame_counter = 0

    def get_S_pars(self):
        # Lazy: this is the only method that needs Sionna (via e2e.ml.rt_gen), and only
        # at call time -- see the module docstring.
        from e2e.ml.labels import LabelGrid, encode_detection_labels, targets_in_grid
        from e2e.ml.rt_gen import (
            DEFAULT_SCATTERING_COEFFICIENT,
            DEFAULT_SCATTERING_PATTERN,
            _resolve_device,
            build_rt_scene,
            rt_cfr_frame,
        )
        from e2e.ml.scatterers import frame_scatterers, radar_pose

        dev = _resolve_device(self.device)
        sc_coeff = (DEFAULT_SCATTERING_COEFFICIENT if self.scattering_coefficient is None
                   else self.scattering_coefficient)
        sc_pattern = (DEFAULT_SCATTERING_PATTERN if self.scattering_pattern is None
                     else self.scattering_pattern)

        build_kwargs = dict(base_scene=self.base_scene, frame_idx=self.frame_counter,
                           scattering_coefficient=sc_coeff, scattering_pattern=sc_pattern)
        if self.base_scene in ("flat", "free"):
            # Synthetic scenes only ever use in-band materials (see
            # `e2e.ml.rt_gen._GROUND_MATERIAL`) -- unmodified, so "flat" (the
            # default) stays exactly as it was before city scenes existed.
            rt_scene = build_rt_scene(self.scenario, self.cfg, **build_kwargs)
        else:
            # A Sionna built-in city scene (or a custom scene path): repair its
            # materials for this radar's frequency one Python frame before
            # `build_rt_scene` sets `scene.frequency` -- see
            # `city_scenes.patched_builtin_loader`'s docstring for why this is a
            # monkeypatch rather than a direct call.
            from e2e.environment.city_scenes import patched_builtin_loader

            f_center_hz = float(self.cfg.f0_hz) + float(self.cfg.bandwidth_hz) / 2.0
            self.last_material_report = {}
            with patched_builtin_loader(f_center_hz, policy=self.material_policy,
                                        stand_in_itu_type=self.stand_in_material,
                                        report_sink=self.last_material_report):
                rt_scene = build_rt_scene(self.scenario, self.cfg, **build_kwargs)
        s_pars = rt_cfr_frame(
            self.cfg, self.scenario, frame_idx=self.frame_counter,
            base_scene=self.base_scene, device=dev, rt_scene=rt_scene,
            max_depth=self.max_depth, include_leakage=self.include_leakage,
            diffuse_reflection=self.diffuse_reflection,
            specular_reflection=self.specular_reflection, refraction=self.refraction,
            solver_seed=self.solver_seed, freq_chunk=self.freq_chunk,
        )

        dt = 1.0 / float(self.cfg.frame_rate_hz)
        pose = radar_pose(self.scenario, self.frame_counter)
        scats = frame_scatterers(self.scenario, self.frame_counter, dt=dt)
        grid = self.label_grid or LabelGrid.for_config(self.cfg)
        self.last_labels = encode_detection_labels(grid, scats, pose,
                                                    classes=self.label_classes)
        self.last_targets = targets_in_grid(grid, scats, pose, classes=self.label_classes)

        return s_pars
