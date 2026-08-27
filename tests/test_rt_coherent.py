"""Tests for the 2026-08-17 hybrid-RT target-physics fix.

The defect: Sionna's image-method specular search cannot find a monostatic backscatter
path off a tessellated convex body at range (see the "HYBRID RT" banner in
`e2e.environment.rt_signal_chain` for the geometry and the measurements), so every ray-traced
target's entire return was Monte-Carlo diffuse speckle -- roughly the right ENERGY but
spread over range and decorrelated across the aperture, earning almost none of the
chain's coherent processing gain. `coherent_target_cfr` supplies the coherent complement.

Split by what each test needs:
  * ungated -- the closed-form coherent term's own physics (energy split, aperture phase
    ramp, Doppler) against a stub scene, and the PLUMBING from `chain_generate` /
    `RTEnvironmentBlock` down to `rt_cfr_frame`. No Sionna.
  * @pytest.mark.sionna -- the end-to-end claims that need a real solve: coherence
    restored on a real target, the diffuse lobe's S^2 energy law, and
    `coherent_targets=False` reproducing the old path bit-for-bit.
"""

import dataclasses
import math
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from e2e.environment import rt_signal_chain as rsc
from e2e.radar_config import RadarConfig
from e2e.environment.rt_scene_build import (DEFAULT_ANTENNA_PATTERN,
                                            DEFAULT_GROUND_SCATTERING_COEFFICIENT,
                                            DEFAULT_SCATTERING_COEFFICIENT,
                                            _flat_scene_xml, _synthetic_scene_path)
from e2e.scenario import (ArrayConfig, Motion, Node, NodeRole, ObjectKind, Scenario,
                          SceneObject)

_C = 299792458.0

# Small, fast config; a real 12-element virtual ULA so the aperture-phase check has
# something to measure.
_CFG = RadarConfig(
    name="test_coh_cfg", f0_hz=77e9, bandwidth_hz=1e9, n_tx=3, n_rx=4,
    n_chirps=6, n_samples=32, fs_hz=5e6, chirp_period_s=20e-6, mimo="tdm",
)


def _scenario(position=(12.0, 3.0, 0.5), velocity=(0.0, 0.0, 0.0), rcs_dbsm=10.0):
    return Scenario(
        name="coh", base_scene="flat", num_frames=1,
        nodes=[Node(name="radar", role=NodeRole.RADAR, position=(0.0, 0.0, 1.5),
                    look_at=(1.0, 0.0, 1.5), array=ArrayConfig(num_rows=1, num_cols=1))],
        objects=[SceneObject(name="t", kind=ObjectKind.SPHERE, position=position,
                             scaling=0.5, material="metal", object_class="vehicle",
                             rcs_dbsm=rcs_dbsm, motion=Motion(),
                             velocity_mps=velocity)],
    )


class _FakeDrJitArray:
    """Minimal stand-in for a DrJit array: only `.numpy()`/`.shape`, which is all
    `_rt_phase_centres` touches."""

    def __init__(self, arr):
        self._a = np.asarray(arr)

    def numpy(self):
        return self._a

    @property
    def shape(self):
        return self._a.shape


def _stub_rt_scene(cfg, centre_world, *, object_id=7, half=(0.5, 0.5, 0.5)):
    """An `RTScene`-shaped stub whose one object reports a world-space AABB, plus a
    matching `paths` stub whose single first-interaction vertex is `centre_world`.

    Lets the closed-form half of the fix be tested with no ray tracer at all: the two
    inputs `coherent_target_cfr` actually reads from Sionna are (a) each object's bbox and
    (b) the traced nearest-visible-surface vertex.
    """
    c = np.asarray(centre_world, dtype=float)
    h = np.asarray(half, dtype=float)
    bbox = types.SimpleNamespace(min=list(c - h), max=list(c + h))
    so = types.SimpleNamespace(object_id=object_id,
                               mi_mesh=types.SimpleNamespace(bbox=lambda: bbox))
    scene = types.SimpleNamespace(scene=None, objects={"t": so}, cfg=cfg)
    # objects: [depth, rx, rxa, tx, txa, P]; vertices: same + (3,); tau: [rx,rxa,tx,txa,P]
    tau = np.full((1, int(cfg.n_rx), 1, int(cfg.n_tx), 1),
                  float(np.linalg.norm(c) * 2 / _C))
    paths = types.SimpleNamespace(
        objects=_FakeDrJitArray(np.full((2, 1, int(cfg.n_rx), 1, int(cfg.n_tx), 1),
                                        object_id)),
        vertices=_FakeDrJitArray(
            np.broadcast_to(c, (2, 1, int(cfg.n_rx), 1, int(cfg.n_tx), 1, 3)).copy()),
        tau=_FakeDrJitArray(tau),
    )
    return scene, paths


# --------------------------------------------------------------------------------
# The coherent term's own physics -- no Sionna
# --------------------------------------------------------------------------------
def test_coherent_term_energy_is_the_diffuse_complement():
    """The coherent term carries `1 - S^2` of the object's RCS, exactly.

    That is the energy-conserving split against Sionna's diffuse lobe, which carries
    `S^2` (MEASURED: the traced diffuse energy moved 10.5 dB between S=0.3 and S=1.0
    against 10.46 dB predicted by S^2 -- see `test_diffuse_lobe_follows_s_squared`).
    Power is quadratic in the CFR, so the ratio between two `S` values must be
    `(1 - S_a^2) / (1 - S_b^2)`.
    """
    scn = _scenario()
    scene, paths = _stub_rt_scene(_CFG, (12.0, 3.0, 0.5))
    powers = {}
    for s in (0.0, 0.3, 0.6, 0.9):
        h = rsc.coherent_target_cfr(_CFG, scene, scn, frame_idx=0,
                                    scattering_coefficient=s, paths=paths)
        powers[s] = float(np.sum(np.abs(h) ** 2))

    assert powers[0.0] > 0.0
    for s in (0.3, 0.6, 0.9):
        expected = powers[0.0] * (1.0 - s ** 2)
        assert powers[s] == pytest.approx(expected, rel=1e-5), s
    # And the headline number in dB: S=0.3 keeps essentially all of it, S=0.9 almost none.
    assert 10 * math.log10(powers[0.3] / powers[0.0]) == pytest.approx(-0.41, abs=0.02)


def test_coherent_term_is_a_clean_aperture_phase_ramp():
    """The point of the fix: a coherent plane wave across the virtual ULA.

    The traced diffuse-only return has a 1.73 rad RMS phase residual against the ideal
    `pi * v * sin(az)` ramp (MEASURED; the theoretical maximum for uniform random phase
    is pi/sqrt(3) = 1.81 rad). The closed-form coherent term must be far better than
    that -- this pins it at essentially zero, which is what earns the array gain.
    """
    scn = _scenario(position=(12.0, 3.0, 1.5))
    p = np.array([12.0, 3.0, 1.5])
    scene, paths = _stub_rt_scene(_CFG, p)
    # n_centers=1: this is the PER-CENTRE cleanliness oracle. The v1.1 default (5) is a
    # superposition of ramps from slightly different azimuths -- deliberately not a
    # single clean ramp; its properties are pinned by the multi-centre tests below.
    h = rsc.coherent_target_cfr(_CFG, scene, scn, frame_idx=0, paths=paths, n_centers=1)

    from e2e.chain.dechirp import beat_from_cfr

    beat = beat_from_cfr(torch.from_numpy(h)).numpy()      # [n_rx, n_tx, chirp, sample]
    # Virtual element v = t*n_rx + r, matching build_rt_scene's spacing convention.
    v = np.transpose(beat, (1, 0, 2, 3)).reshape(-1, h.shape[2], h.shape[3])
    vec = v[:, 0, 0]

    radar = np.array([0.0, 0.0, 1.5])
    d = p - radar
    r = float(np.linalg.norm(d))
    ax = np.cross([0.0, 0.0, 1.0], np.array([1.0, 0.0, 0.0]))
    ax = ax / np.linalg.norm(ax)
    s_az = float(np.dot(d, ax) / r)

    ideal = np.exp(1j * np.pi * np.arange(vec.size) * s_az)
    z = vec * np.conj(ideal)
    z = z * np.exp(-1j * np.angle(z.sum()))
    resid = float(np.sqrt(np.mean(np.angle(z) ** 2)))
    assert resid < 0.05, f"aperture phase residual {resid:.3f} rad is not a clean ramp"
    # Uniform illumination too -- speckle shows up as amplitude spread across elements.
    spread_db = 20 * math.log10(np.abs(vec).max() / np.abs(vec).min())
    assert spread_db < 1.0, spread_db


def test_coherent_term_range_and_rcs_follow_the_radar_equation():
    """Amplitude must be Sionna's own per-path convention,
    `|a|^2 = sigma lambda^2 / ((4pi)^3 R_t^2 R_r^2)` -- that is what licenses ADDING it
    to `cfr_from_paths`' output rather than scaling it by an arbitrary constant."""
    scn_near = _scenario(position=(10.0, 0.0, 1.5))
    scn_far = _scenario(position=(20.0, 0.0, 1.5))
    sc_n, pa_n = _stub_rt_scene(_CFG, (10.0, 0.0, 1.5))
    sc_f, pa_f = _stub_rt_scene(_CFG, (20.0, 0.0, 1.5))
    # n_centers=1: the amplitude-convention oracle wants ONE path at a known range;
    # the multi-centre composition is tested separately below.
    p_near = float(np.sum(np.abs(rsc.coherent_target_cfr(
        _CFG, sc_n, scn_near, frame_idx=0, paths=pa_n, n_centers=1)) ** 2))
    p_far = float(np.sum(np.abs(rsc.coherent_target_cfr(
        _CFG, sc_f, scn_far, frame_idx=0, paths=pa_f, n_centers=1)) ** 2))
    # R^-4 two-way: doubling the range costs 12 dB. Ranges are radar-to-surface, so use
    # the actual geometry rather than the nominal 10/20 m.
    r_n = np.linalg.norm(np.array([10.0, 0.0, 1.5]) - np.array([0.0, 0.0, 1.5]))
    r_f = np.linalg.norm(np.array([20.0, 0.0, 1.5]) - np.array([0.0, 0.0, 1.5]))
    assert 10 * math.log10(p_near / p_far) == pytest.approx(
        40 * math.log10(r_f / r_n), abs=0.3)

    # RCS enters linearly in power.
    scn_hot = _scenario(position=(10.0, 0.0, 1.5), rcs_dbsm=20.0)
    sc_h, pa_h = _stub_rt_scene(_CFG, (10.0, 0.0, 1.5))
    p_hot = float(np.sum(np.abs(rsc.coherent_target_cfr(
        _CFG, sc_h, scn_hot, frame_idx=0, paths=pa_h, n_centers=1)) ** 2))
    assert 10 * math.log10(p_hot / p_near) == pytest.approx(10.0, abs=0.05)


@pytest.mark.parametrize("speed", [-8.0, 8.0])
def test_coherent_term_doppler_matches_the_analytic_point_target(speed):
    """Cross-check against the ORACLE rather than against my reading of the sign
    convention: `rd_synth.synthesize_adc` is the closed-form point-target model the whole
    package is calibrated on, so the coherent term's chirp-to-chirp phase advance must
    equal its own for the same radial velocity. `speed` is the +x velocity component, so
    negative = approaching (the radar looks along +x from the origin).

    Position-independent by construction: the per-chirp advance depends only on `v_r`,
    which is what lets this compare a surface-point phase centre against rd_synth's
    object-centre one.
    """
    from e2e.chain.dechirp import beat_from_cfr
    from e2e.environment.scatterers import frame_scatterers, radar_pose
    from e2e.chain.rd_synth import synthesize_adc

    scn = _scenario(position=(12.0, 0.0, 1.5), velocity=(speed, 0.0, 0.0))
    scene, paths = _stub_rt_scene(_CFG, (12.0, 0.0, 1.5))
    # n_centers=1: rd_synth's oracle is a single point with one LOS; each of the five
    # default centres has its own LOS and hence its own slightly different f_d.
    h = rsc.coherent_target_cfr(_CFG, scene, scn, frame_idx=0, paths=paths, n_centers=1)
    beat = beat_from_cfr(torch.from_numpy(h)).numpy()
    got = np.angle(beat[0, 0, 1:, 0] * np.conj(beat[0, 0, :-1, 0]))

    # Single-TX view so consecutive chirps are one chirp period apart (TDM interleaves).
    single = dataclasses.replace(_CFG, n_tx=1, mimo="single")
    scats = frame_scatterers(scn, 0, dt=1.0 / float(_CFG.frame_rate_hz))
    adc = synthesize_adc(single, scats, radar_pose(scn, 0), snr_db=None, seed=0,
                         random_phase=False).cpu().numpy()
    want = np.angle(adc[0, 1:, 0] * np.conj(adc[0, :-1, 0]))

    assert np.allclose(got, want[:got.size], atol=2e-3), (got[:3], want[:3])
    # And it is actually non-trivial: a moving target must rotate chirp to chirp.
    assert abs(float(np.mean(got))) > 1e-3


def test_occluded_object_gets_no_coherent_return():
    """No traced path to an object means the ray tracer says it is not visible; the
    coherent term must respect that rather than shining through the occluder."""
    scn = _scenario()
    scene, paths = _stub_rt_scene(_CFG, (12.0, 3.0, 0.5), object_id=7)
    # Every traced interaction belongs to a DIFFERENT object -> our object is unseen.
    paths.objects = _FakeDrJitArray(
        np.full((2, 1, int(_CFG.n_rx), 1, int(_CFG.n_tx), 1), 999))
    h = rsc.coherent_target_cfr(_CFG, scene, scn, frame_idx=0, paths=paths)
    assert np.all(h == 0)


# --------------------------------------------------------------------------------
# Scene-level defaults -- no Sionna
# --------------------------------------------------------------------------------
def test_directive_antenna_pattern_is_the_default():
    """`build_rt_scene`'s element pattern must not be isotropic any more: an isotropic
    element put a nadir ground bounce 54 dB above the target into every flat-scene
    frame (see `DEFAULT_ANTENNA_PATTERN`)."""
    import inspect

    from e2e.environment.rt_scene_build import build_rt_scene

    assert DEFAULT_ANTENNA_PATTERN != "iso"
    sig = inspect.signature(build_rt_scene)
    assert sig.parameters["pattern"].default == DEFAULT_ANTENNA_PATTERN


def test_flat_scene_xml_declares_its_ground_roughness_and_is_parameterised():
    """The ground's scattering coefficient is now explicit in the scene XML and
    settable; leaving it implicit took Sionna's 0.0 (an optical mirror)."""
    default_xml = _flat_scene_xml()
    assert f'name="scattering_coefficient" value="{float(DEFAULT_GROUND_SCATTERING_COEFFICIENT)}"' \
        in default_xml
    rough = _flat_scene_xml(0.25)
    assert 'name="scattering_coefficient" value="0.25"' in rough
    # Distinct roughness values must not collide in the written-scene cache.
    a = _synthetic_scene_path("flat", 0.0)
    b = _synthetic_scene_path("flat", 0.25)
    assert a != b
    assert _synthetic_scene_path("flat", 0.25) == b          # cached, not rewritten
    with open(b) as f:
        assert 'value="0.25"' in f.read()
    # "free" has no ground at all -- the parameter must not fabricate one.
    assert _synthetic_scene_path("free", 0.25) == _synthetic_scene_path("free")


# --------------------------------------------------------------------------------
# Plumbing: does the generation path actually carry the fix? -- no Sionna
# --------------------------------------------------------------------------------
def test_rt_environment_block_defaults_to_the_fixed_physics():
    from e2e.environment.blocks import RTEnvironmentBlock

    blk = RTEnvironmentBlock(_scenario(), _CFG)
    assert blk.coherent_targets is True
    assert blk.antenna_pattern is None          # -> DEFAULT_ANTENNA_PATTERN at solve time
    # And the old behaviour stays reachable for regressions/reproduction.
    old = RTEnvironmentBlock(_scenario(), _CFG, coherent_targets=False,
                             antenna_pattern="iso")
    assert old.coherent_targets is False and old.antenna_pattern == "iso"


def test_rt_environment_block_forwards_the_fix_to_rt_cfr_frame(monkeypatch):
    """`get_S_pars()` is the only seam corpus generation goes through, so the kwargs have
    to arrive THERE. Sionna-free: `get_S_pars` resolves `build_rt_scene`/`rt_cfr_frame`
    from `e2e.environment.rt_gen` at call time, so both are monkeypatchable."""
    import e2e.environment.rt_gen as rt_gen
    from e2e.environment.blocks import RTEnvironmentBlock

    seen = {}

    def fake_build_rt_scene(scenario, cfg, **kwargs):
        seen["build"] = kwargs
        return types.SimpleNamespace(objects={}, scene=None, cfg=cfg)

    def fake_rt_cfr_frame(cfg, scenario, **kwargs):
        seen["cfr"] = kwargs
        return torch.zeros((int(cfg.n_rx), int(cfg.n_tx), int(cfg.n_chirps),
                            int(cfg.n_samples)), dtype=torch.complex64)

    monkeypatch.setattr(rt_gen, "build_rt_scene", fake_build_rt_scene)
    monkeypatch.setattr(rt_gen, "rt_cfr_frame", fake_rt_cfr_frame)

    blk = RTEnvironmentBlock(_scenario(), _CFG, device="cpu")
    blk.get_S_pars()
    assert seen["cfr"]["coherent_targets"] is True
    assert seen["build"]["pattern"] == DEFAULT_ANTENNA_PATTERN
    # The coherent/diffuse split is only energy-conserving if BOTH sides see the same S.
    assert seen["cfr"]["scattering_coefficient"] == seen["build"]["scattering_coefficient"]
    assert seen["cfr"]["scattering_coefficient"] == DEFAULT_SCATTERING_COEFFICIENT

    seen.clear()
    blk_old = RTEnvironmentBlock(_scenario(), _CFG, device="cpu",
                                 coherent_targets=False, antenna_pattern="iso",
                                 ground_scattering_coefficient=0.25,
                                 samples_per_src=10 ** 5)
    blk_old.get_S_pars()
    assert seen["cfr"]["coherent_targets"] is False
    assert seen["build"]["pattern"] == "iso"
    assert seen["build"]["ground_scattering_coefficient"] == 0.25
    assert seen["cfr"]["samples_per_src"] == 10 ** 5


def test_chain_generate_builds_a_fixed_environment_block(tmp_path):
    """`build_chain_simulation` is what `generate_chain_corpus` calls per scene; the
    default `RTEnvironmentBlock` it constructs must carry the fix."""
    from e2e.ml import chain_generate
    from e2e.environment.blocks import RTEnvironmentBlock
    from e2e.ml.labels import LabelGrid

    grid = LabelGrid.for_config(_CFG)
    sim = chain_generate.build_chain_simulation(
        _scenario(), _CFG, out_dir=tmp_path, label_grid=grid, device="cpu",
        use_rffe=False, use_interconnect=False)
    env = sim.environment_block if hasattr(sim, "environment_block") else None
    if env is None:                     # attribute name differs across Simulation revs
        env = next(v for v in vars(sim).values()
                   if isinstance(v, RTEnvironmentBlock))
    assert isinstance(env, RTEnvironmentBlock)
    assert env.coherent_targets is True
    assert env.antenna_pattern is None

    sim_old = chain_generate.build_chain_simulation(
        _scenario(), _CFG, out_dir=tmp_path, label_grid=grid, device="cpu",
        use_rffe=False, use_interconnect=False,
        coherent_targets=False, antenna_pattern="iso")
    env_old = next(v for v in vars(sim_old).values()
                   if isinstance(v, RTEnvironmentBlock))
    assert env_old.coherent_targets is False and env_old.antenna_pattern == "iso"


def test_chain_generate_cli_exposes_the_reproduction_flags():
    from e2e.ml import chain_generate

    p = chain_generate.build_arg_parser()
    args = p.parse_args(["--config", "ti_iwr1443", "--tier", "D1", "--n", "1"])
    assert args.no_coherent_targets is False
    assert args.antenna_pattern is None
    assert args.ground_scattering is None
    assert args.samples_per_src is None
    # The TX tributary is ON by default in `generate_chain_corpus` and MEASURED to erase
    # the target-physics fix (D1 median target-vs-p99 +7.5 dB with it off, -1.6 dB with
    # it on), so a corpus run must be able to turn it off from the command line.
    assert args.no_transmit_chain is False
    args = p.parse_args(["--config", "ti_iwr1443", "--tier", "D1", "--n", "1",
                         "--no-coherent-targets", "--antenna-pattern", "iso",
                         "--ground-scattering", "0.3", "--samples-per-src", "100000",
                         "--no-transmit-chain"])
    assert args.no_coherent_targets is True
    assert args.antenna_pattern == "iso"
    assert args.ground_scattering == pytest.approx(0.3)
    assert args.samples_per_src == 100000
    assert args.no_transmit_chain is True


# --------------------------------------------------------------------------------
# End-to-end, real ray tracing
# --------------------------------------------------------------------------------
def _d0_scene():
    from e2e.environment.rt_scenes import build_rt_tier_scenario

    return build_rt_tier_scenario("D0", corpus_tag="unit-test", frame_idx=0, seed=0, num_frames=1,
                                  use_local_assets=False)


@pytest.mark.sionna
def test_specular_only_finds_no_path_on_a_curved_target():
    """The root cause, pinned so nobody "simplifies" the coherent term away: Sionna's
    image method finds NOTHING off a tessellated sphere, at any tessellation or range."""
    from e2e.radar_config import PRESETS
    from e2e.environment.rt_scene_build import build_rt_scene

    cfg = PRESETS["ti_iwr1443"]
    scn = _d0_scene()
    rts = build_rt_scene(scn, cfg, base_scene="free")
    paths = rsc._solve(rts, max_depth=2, include_leakage=False,
                       diffuse_reflection=False, specular_reflection=True,
                       refraction=False, seed=41)
    tau = np.asarray(paths.tau.numpy())[0, 0, 0, 0]
    assert int(np.sum(np.isfinite(tau) & (tau > 0))) == 0


@pytest.mark.sionna
def test_diffuse_lobe_follows_s_squared():
    """The traced diffuse energy is `S^2` of the reflected power -- the law the coherent
    term's `1 - S^2` complements. MEASURED 10.5 dB between S=0.3 and S=1.0 against
    10.46 dB predicted."""
    from e2e.radar_config import PRESETS
    from e2e.environment.rt_scene_build import build_rt_scene

    cfg = PRESETS["ti_iwr1443"]
    scn = _d0_scene()
    energies = {}
    for s in (0.3, 1.0):
        rts = build_rt_scene(scn, cfg, base_scene="free", scattering_coefficient=s)
        paths = rsc._solve(rts, max_depth=2, include_leakage=False,
                           diffuse_reflection=True, specular_reflection=False,
                           refraction=False, seed=41)
        a_re, a_im = paths.a
        a = np.asarray(a_re.numpy()) + 1j * np.asarray(a_im.numpy())
        tau = np.asarray(paths.tau.numpy())
        good = np.isfinite(tau[0, 0, 0, 0]) & (tau[0, 0, 0, 0] > 0)
        energies[s] = float(np.sum(np.abs(a[0, 0, 0, 0][good]) ** 2))
    measured_db = 10 * math.log10(energies[1.0] / energies[0.3])
    predicted_db = 10 * math.log10(1.0 / 0.3 ** 2)
    assert measured_db == pytest.approx(predicted_db, abs=0.5), (measured_db, predicted_db)


@pytest.mark.sionna
def test_coherent_targets_false_reproduces_the_old_path_exactly():
    """The reproduction escape hatch must be exact, not approximate -- it is how a
    pre-2026-08-17 corpus gets regenerated for comparison."""
    from e2e.radar_config import PRESETS
    from e2e.environment.rt_scene_build import build_rt_scene

    cfg = PRESETS["ti_iwr1443"]
    scn = _d0_scene()
    rts = build_rt_scene(scn, cfg, base_scene="flat", pattern="iso")
    paths = rsc._solve(rts, max_depth=2, include_leakage=False, diffuse_reflection=True,
                       specular_reflection=True, refraction=False, seed=41)
    reference = np.asarray(rsc.cfr_from_paths(paths, cfg, n_chirps=int(cfg.n_chirps)),
                           dtype=np.complex64)

    def rel(a, b):
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))

    # `rt_cfr_frame` re-solves even when handed a built scene, and Sionna's diffuse
    # sampling is not bit-reproducible across repeated solves of the same scene object
    # (a pre-existing property of the solver, not of this fix). So establish the
    # solve-to-solve floor empirically instead of assuming bit-equality...
    off_a = rsc.rt_cfr_frame(cfg, scn, frame_idx=0, rt_scene=rts, device="cpu",
                             coherent_targets=False).cpu().numpy()
    off_b = rsc.rt_cfr_frame(cfg, scn, frame_idx=0, rt_scene=rts, device="cpu",
                             coherent_targets=False).cpu().numpy()
    floor = max(rel(off_a, off_b), rel(off_a, reference), 1e-6)
    assert floor < 1e-2, f"solver reproducibility floor {floor:.2e} is suspiciously large"

    # ...and require the OFF arm to sit on that floor relative to the pre-fix reference,
    # i.e. it adds nothing.
    assert rel(off_b, reference) <= floor

    # The ON arm must be an unmistakable change, well clear of the solver floor. Note
    # the raw-CFR relative change is only a few percent even though the fix is worth
    # tens of dB after processing: this arm keeps the old `pattern="iso"` scene, whose
    # nadir ground bounce dominates the cube's ENERGY. The fix's size shows up only
    # after coherent integration -- which is exactly the defect, and is what
    # `test_fix_restores_aperture_coherence_on_a_real_traced_target` measures.
    on = rsc.rt_cfr_frame(cfg, scn, frame_idx=0, rt_scene=rts, device="cpu",
                          coherent_targets=True).cpu().numpy()
    assert rel(on, reference) > 20.0 * floor, (rel(on, reference), floor)


@pytest.mark.sionna
def test_fix_restores_aperture_coherence_on_a_real_traced_target():
    """The headline before/after: on the D0 single-sphere scene the aperture phase
    residual at the target's own range-Doppler cell collapses from ~1.0 rad (speckle) to
    ~0.1 rad (a plane wave), and the target rises well above its own map background."""
    from e2e.radar_config import PRESETS
    from e2e.ml.render_scene import _resolve_frames
    from e2e.environment.rt_scene_build import build_rt_scene
    from e2e.chain.transforms import adc_to_rd, tdm_deinterleave

    cfg = PRESETS["ti_iwr1443"]
    scn = _d0_scene()
    scats_pf, pose_pf = _resolve_frames(scn, cfg, 1, 1.0 / float(cfg.frame_rate_hz))
    scats, pose = scats_pf[0], pose_pf[0]
    radar_p = np.asarray(pose.position, float)
    tgt_p = np.asarray(scats[0].position, float)
    r = float(np.linalg.norm(tgt_p - radar_p))
    ax = np.cross([0.0, 0.0, 1.0], np.asarray(pose.boresight, float))
    ax = ax / np.linalg.norm(ax)
    s_az = float(np.dot(tgt_p - radar_p, ax) / r)

    def measure(coherent, pattern):
        rts = build_rt_scene(scn, cfg, base_scene="flat", pattern=pattern)
        adc = rsc.rt_synthesize_adc(cfg, scn, frame_idx=0, snr_db=None, seed=0,
                                    rt_scene=rts, device="cpu",
                                    coherent_targets=coherent)
        sub = dataclasses.replace(cfg, n_tx=1, mimo="single",
                                  n_chirps=cfg.n_chirps_per_tx)
        rd = adc_to_rd(sub, tdm_deinterleave(cfg, adc))       # [virt, range, doppler]
        spec = torch.fft.fftshift(torch.fft.fft(rd, n=256, dim=0), dim=0)
        ra = (spec.abs() ** 2).max(dim=2).values.cpu().numpy()
        n_ang, n_rng = ra.shape
        ab = int(round(s_az * n_ang / 2.0 + n_ang // 2))
        r0 = max(0, int((r - 1.0) / cfg.range_resolution_m))
        r1 = min(n_rng, int((r + 0.5) / cfg.range_resolution_m) + 1)
        win = ra[max(0, ab - 4):ab + 5, r0:r1]
        rbin = r0 + int(np.unravel_index(int(np.argmax(win)), win.shape)[1])
        db = 10 * np.log10(max(float(win.max()), 1e-30)
                           / max(float(np.percentile(ra, 99)), 1e-30))
        cell = rd[:, rbin, :]
        k = int(torch.argmax(cell.abs().sum(dim=0)).item())
        vec = cell[:, k].cpu().numpy()
        z = vec * np.conj(np.exp(1j * np.pi * np.arange(vec.size) * s_az))
        z = z * np.exp(-1j * np.angle(z.sum()))
        return db, float(np.sqrt(np.mean(np.angle(z) ** 2)))

    db_old, ph_old = measure(False, "iso")
    db_new, ph_new = measure(True, DEFAULT_ANTENNA_PATTERN)

    assert ph_old > 0.5, f"speckle arm should be incoherent, got {ph_old:.3f} rad"
    assert ph_new < 0.3, f"fixed arm should be coherent, got {ph_new:.3f} rad"
    assert db_new > db_old + 10.0, (db_old, db_new)
    assert db_new > 15.0, db_new


# --------------------------------------------------------------------------------
# The multi-centre model (v1.1 default n_centers=5) -- no Sionna
# --------------------------------------------------------------------------------
def test_n_centers_one_matches_the_pre_v11_implementation():
    """Refactor regression: the separable-einsum rewrite at `n_centers=1` must
    reproduce the pre-v1.1 broadcast implementation (frozen, inert, local to this
    test -- the A1-test pattern) to float tolerance. The two are algebraically
    identical; only the association order differs."""
    scn = _scenario(position=(12.0, 3.0, 1.5), velocity=(3.0, 1.0, 0.0))
    scene, paths = _stub_rt_scene(_CFG, (12.0, 3.0, 1.5))

    def _legacy(cfg, rt_scene, scenario, *, frame_idx, paths):
        from e2e.environment.scatterers import frame_scatterers, radar_pose
        from e2e.environment.rt_scene_build import DEFAULT_SCATTERING_COEFFICIENT
        coh_frac = max(0.0, 1.0 - float(DEFAULT_SCATTERING_COEFFICIENT) ** 2)
        n_chirps = int(cfg.n_chirps)
        freqs = rsc.beat_frequencies(cfg)
        f_c = float(cfg.f0_hz) + float(cfg.bandwidth_hz) / 2.0
        lam = _C / f_c
        pose = radar_pose(scenario, frame_idx)
        scats = frame_scatterers(scenario, frame_idx, dt=1.0 / float(cfg.frame_rate_hz))
        radar_pos = np.asarray(pose.position, dtype=np.float64)
        tx_pos = rsc._array_element_positions(radar_pos, pose.boresight, int(cfg.n_tx),
                                              0.5 * int(cfg.n_rx), lam)
        rx_pos = rsc._array_element_positions(radar_pos, pose.boresight, int(cfg.n_rx), 0.5, lam)
        t = np.arange(n_chirps, dtype=np.float64) * float(cfg.chirp_period_s)
        out = np.zeros((int(cfg.n_rx), int(cfg.n_tx), n_chirps, freqs.size), dtype=np.complex128)
        centres = rsc._rt_phase_centres(paths, rt_scene)
        for obj, sc in zip(scenario.objects, scats):
            rt_p, visible = centres.get(obj.name, (None, True))
            if not visible or rt_p is None:
                continue
            p = np.asarray(rt_p, dtype=np.float64)
            sigma = coh_frac * 10.0 ** (float(sc.rcs_dbsm) / 10.0)
            d_t = np.linalg.norm(tx_pos - p[None, :], axis=1)
            d_r = np.linalg.norm(rx_pos - p[None, :], axis=1)
            tau = (d_r[:, None] + d_t[None, :]) / _C
            los = p - radar_pos
            los = los / max(float(np.linalg.norm(los)), 1e-12)
            g = rsc._element_field_amplitude(getattr(rt_scene, "antenna_pattern", "iso"),
                                             pose.boresight, los)
            amp = (g * g * math.sqrt(sigma) * lam
                   / ((4.0 * math.pi) ** 1.5 * d_r[:, None] * d_t[None, :]))
            v_r = float(np.dot(np.asarray(sc.velocity, dtype=np.float64), los))
            f_d = -2.0 * v_r / lam
            tau_b = tau[:, :, None, None]
            t_b = t[None, None, :, None]
            f_b = freqs[None, None, None, :]
            tau_bb = tau_b - (f_d / f_c) * t_b
            phase = (np.exp(-2j * np.pi * f_c * tau_b)
                     * np.exp(2j * np.pi * f_d * t_b)
                     * np.exp(-2j * np.pi * f_b * tau_bb))
            out += amp[:, :, None, None] * phase
        return np.ascontiguousarray(out.astype(np.complex64))

    new = rsc.coherent_target_cfr(_CFG, scene, scn, frame_idx=0, paths=paths, n_centers=1)
    old = _legacy(_CFG, scene, scn, frame_idx=0, paths=paths)
    assert np.allclose(new, old, rtol=1e-5, atol=1e-10 * float(np.abs(old).max()))


def test_multi_center_conserves_total_energy_within_a_db():
    """Five centres at sigma/5 each must carry about the same TOTAL energy as one at
    sigma: the centres sit metres apart, so their cross terms average out over the
    frequency axis and the powers add. Not exact -- each centre has its own range --
    hence a dB-scale tolerance, not an equality."""
    scn = _scenario(position=(12.0, 3.0, 1.5))
    # The premise needs the centres METRES apart (else cross terms stay partially
    # coherent over the frequency axis); give the body a van-sized footprint.
    scn.objects[0].extent_m = (4.0, 4.0, 1.5)
    scene, paths = _stub_rt_scene(_CFG, (12.0, 3.0, 1.5), half=(1.0, 1.0, 0.5))
    p1 = float(np.sum(np.abs(rsc.coherent_target_cfr(
        _CFG, scene, scn, frame_idx=0, paths=paths, n_centers=1)) ** 2))
    p5 = float(np.sum(np.abs(rsc.coherent_target_cfr(
        _CFG, scene, scn, frame_idx=0, paths=paths, n_centers=5)) ** 2))
    assert abs(10 * math.log10(p5 / p1)) < 1.5


def test_multi_center_widens_azimuth_extent():
    """The spike's headline (audit entry 4): one centre collapses target extent to ~1
    cell where real returns measure ~6 (F44). A truck-sized bbox's 5 centres span
    enough cross-range that the -6 dB azimuth extent must widen vs the point model."""
    from e2e.chain.dechirp import beat_from_cfr

    def _extent(n_centers):
        scn = _scenario(position=(12.0, 0.0, 1.5))
        # Truck-sized body via the explicit extent override `e2e.environment.geometry.
        # object_extent_m` honours (the corners come from the scatterer layer's
        # body-frame extent since A13, not from the stub mesh's world AABB).
        scn.objects[0].extent_m = (5.0, 5.0, 1.6)
        scene, paths = _stub_rt_scene(_CFG, (12.0, 0.0, 1.5), half=(2.5, 2.5, 0.8))
        h = rsc.coherent_target_cfr(_CFG, scene, scn, frame_idx=0, paths=paths,
                                    n_centers=n_centers)
        beat = beat_from_cfr(torch.from_numpy(h)).numpy()
        v = np.transpose(beat, (1, 0, 2, 3)).reshape(-1, h.shape[2], h.shape[3])[:, 0, 0]
        spec = np.abs(np.fft.fftshift(np.fft.fft(v, n=256))) ** 2
        return int(np.sum(spec >= spec.max() * 10 ** (-6 / 10)))

    e1, e5 = _extent(1), _extent(5)
    assert e5 > e1, (e1, e5)


def test_multi_center_falls_back_to_one_point_without_extent(monkeypatch):
    """An object with no resolvable extent is a POINT target (the same convention
    `e2e.ml.labels` uses): there is nowhere deterministic to put extra centres, so
    n_centers=5 must equal n_centers=1 exactly."""
    import e2e.environment.scatterers as scatterers_mod
    monkeypatch.setattr(scatterers_mod, "object_extent_m", lambda obj: None)
    scn = _scenario(position=(12.0, 3.0, 1.5))
    scene, paths = _stub_rt_scene(_CFG, (12.0, 3.0, 1.5))
    h5 = rsc.coherent_target_cfr(_CFG, scene, scn, frame_idx=0, paths=paths, n_centers=5)
    h1 = rsc.coherent_target_cfr(_CFG, scene, scn, frame_idx=0, paths=paths, n_centers=1)
    assert np.array_equal(h5, h1)
    assert np.abs(h5).max() > 0.0


def test_visible_corner_centers_sit_on_the_yawed_body():
    """A13 oracle: corners must map back EXACTLY to (+-L/2, +-W/2) in the object's
    own frame under any yaw -- the pre-2026-08-24 world-AABB corners sat up to ~3 m
    off-body at oblique yaw because an axis-aligned box inflates around a rotated
    body."""
    centre = np.array([20.0, 5.0, 1.0])
    extent = (5.0, 2.0, 1.5)
    yaw = math.radians(37.0)
    radar = np.array([0.0, 0.0, 1.5])
    pts = rsc._visible_corner_centers(centre, extent, yaw, radar)
    assert pts.shape[0] >= 2
    c, s = math.cos(yaw), math.sin(yaw)
    for p in pts:
        d = p - centre
        bx = c * d[0] + s * d[1]          # rotate back by -yaw
        by = -s * d[0] + c * d[1]
        assert abs(abs(bx) - 2.5) < 1e-9 and abs(abs(by) - 1.0) < 1e-9, p
        assert p[2] == centre[2]


def test_visible_corner_centers_dedupe_degenerate_extent():
    """A zero-width extent override collapses a corner pair to one point; the pair
    must be deduplicated so it does not carry double its sigma share (review
    finding 2026-08-24)."""
    pts = rsc._visible_corner_centers(np.array([20.0, 5.0, 1.0]), (5.0, 0.0, 1.5),
                                      math.radians(37.0), np.array([0.0, 0.0, 1.5]))
    assert pts.shape[0] == len({tuple(p) for p in pts})


def test_visible_corner_centers_far_side_cull():
    """A13 oracle: exact broadside keeps exactly the 2 near-face corners (the far
    face's corners are shadowed by the body); a generic oblique aspect keeps 3."""
    centre = np.array([20.0, 0.0, 1.0])
    extent = (5.0, 2.0, 1.5)
    radar = np.array([0.0, 0.0, 1.5])

    # yaw=90 deg: body +x (length) points along world +y -> the radar stares at the
    # broadside face; visible corners are the two with body-frame y = +W/2 toward it.
    pts = rsc._visible_corner_centers(centre, extent, math.pi / 2, radar)
    assert pts.shape[0] == 2
    assert all(p[0] < centre[0] for p in pts)     # both on the radar side

    # Oblique 37 deg: two faces visible -> shared corner + 2 silhouette corners.
    pts = rsc._visible_corner_centers(centre, extent, math.radians(37.0), radar)
    assert pts.shape[0] == 3
    # And every kept corner is strictly nearer the radar than the culled one.
    all4 = {(sx, sy) for sx in (1.0, -1.0) for sy in (1.0, -1.0)}
    c, s = math.cos(math.radians(37.0)), math.sin(math.radians(37.0))
    kept = {(round(np.sign(c * (p - centre)[0] + s * (p - centre)[1]) or 1.0),
             round(np.sign(-s * (p - centre)[0] + c * (p - centre)[1]) or 1.0))
            for p in pts}
    (culled,) = all4 - kept
    culled_w = centre[:2] + np.array([c * culled[0] * 2.5 - s * culled[1] * 1.0,
                                      s * culled[0] * 2.5 + c * culled[1] * 1.0])
    assert all(np.linalg.norm(p[:2] - radar[:2]) < np.linalg.norm(culled_w - radar[:2])
               for p in pts)


def test_n_centers_zero_raises():
    scn = _scenario()
    scene, paths = _stub_rt_scene(_CFG, (12.0, 3.0, 0.5))
    with pytest.raises(ValueError, match="n_centers"):
        rsc.coherent_target_cfr(_CFG, scene, scn, frame_idx=0, paths=paths, n_centers=0)


def test_element_gain_enters_amplitude_squared(monkeypatch):
    """The element gain applies once on TX and once on RX (F32), so CFR amplitude
    carries g^2 and power g^4. Review finding 2026-08-23: every stub in this file
    resolves to the isotropic pattern (g=1), so a dropped/doubled square was invisible
    to CI -- this pins it with a monkeypatched non-unity gain."""
    scn = _scenario(position=(12.0, 3.0, 1.5))
    scene, paths = _stub_rt_scene(_CFG, (12.0, 3.0, 1.5))
    p_iso = float(np.sum(np.abs(rsc.coherent_target_cfr(
        _CFG, scene, scn, frame_idx=0, paths=paths, n_centers=1)) ** 2))
    monkeypatch.setattr(rsc, "_element_field_amplitude", lambda *a, **k: 2.0)
    p_gain = float(np.sum(np.abs(rsc.coherent_target_cfr(
        _CFG, scene, scn, frame_idx=0, paths=paths, n_centers=1)) ** 2))
    # amplitude x g^2 = x4 -> power x16 = +12.04 dB, exactly.
    assert 10 * math.log10(p_gain / p_iso) == pytest.approx(12.04, abs=0.01)


def test_n_centers_other_than_1_or_5_raises():
    scn = _scenario()
    scene, paths = _stub_rt_scene(_CFG, (12.0, 3.0, 0.5))
    for bad in (0, 2, 3, 4, 6, 10):
        with pytest.raises(ValueError, match="n_centers"):
            rsc.coherent_target_cfr(_CFG, scene, scn, frame_idx=0, paths=paths,
                                    n_centers=bad)
