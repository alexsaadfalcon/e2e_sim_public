"""`e2e.environment.city_scenes` -- Sionna built-in city scenes (munich, etoile) made
usable at frequencies their bundled ITU materials don't cover (see that module's
docstring for the mechanism/policy trade-off), plus the `RTEnvironmentBlock` wiring
(`e2e.environment.blocks`) that lets `base_scene="munich"`/`"etoile"` work end to end.

Real Sionna RT ray tracing, gated behind `@pytest.mark.sionna` (RUN_SIONNA=1) -- see
`tests/test_ml_rt_gen.py` for why the Sionna import lives in a session fixture rather
than at module scope (keeps a plain `pytest` run from touching DrJit/CUDA at all).

Munich's ~1150+ individually-meshed objects (`merge_shapes=False`, hardcoded in
`e2e.ml.rt_gen._load_base_scene` -- not owned by this file) make a full-fidelity
`PathSolver` call there measurably expensive (tens of seconds on this box; see
`test_measure_city_vs_flat_frame_cost`), so this module keeps the number of actual
`_solve()` calls to a minimum and uses `city_scenes` functions directly (load + swap,
no solve) wherever a test doesn't need one.
"""
import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from e2e.ml.radar_config import TI_IWR1443
from e2e.scenario import Node, NodeRole, Scenario

_REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------------
# Torch-free import boundary (ungated -- runs in the default `pytest` invocation)
# --------------------------------------------------------------------------------
def test_module_imports_without_sionna():
    """`import e2e.environment.city_scenes` must not require Sionna/DrJit.

    Checked in a FRESH SUBPROCESS, never by popping/reloading sionna in-process (see
    CLAUDE.md / tests/test_webapp.py::_import_without_torch for why).
    """
    code = (
        "import importlib, sys; "
        "importlib.import_module('e2e.environment.city_scenes'); "
        "sys.exit(0 if 'sionna' not in sys.modules else 3)"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(_REPO_ROOT),
                          capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"


# --------------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def sionna_rt():
    """Import Sionna RT once per session (skips the module if it is unavailable)."""
    return pytest.importorskip("sionna.rt")


@pytest.fixture(scope="module")
def cfg():
    """A TI_IWR1443 preset shrunk to a small virtual array -- `_solve()`'s cost on
    these scenes is dominated by scene geometry (see module docstring), not `n_rx`, so
    shrinking it mainly keeps the CFR-sampling tail of each test cheap."""
    return dataclasses.replace(TI_IWR1443, name="ti_test", n_tx=1, n_rx=4, n_chirps=8,
                               n_samples=32, mimo="single")


def _munich_node():
    # Same square used by e2e.scenario.munich_radar_scenario, aimed inward.
    return Node(name="radar", role=NodeRole.RADAR, position=(45.0, 90.0, 1.5),
               look_at=(45.0, 60.0, 1.5))


def _etoile_node():
    # Same avenue used by e2e.scenario.etoile_radar_scenario, aimed at the roundabout.
    return Node(name="radar", role=NodeRole.RADAR, position=(60.0, 0.0, 1.5),
               look_at=(0.0, 0.0, 1.5))


# --------------------------------------------------------------------------------
# Detection: programmatic, not a hardcoded material list
# --------------------------------------------------------------------------------
@pytest.mark.sionna
def test_out_of_band_materials_finds_munichs_marble_and_brick(sionna_rt):
    """Munich uses ITU `marble` (valid 1-60 GHz) and `brick` (valid 1-40 GHz) -- both
    must be flagged at 78 GHz (the ti_iwr1443 chirp centre), and nothing else (`wood`,
    `metal`, `concrete` are tabulated to 100 GHz)."""
    import sionna.rt as rt

    from e2e.environment.city_scenes import out_of_band_materials

    scene = rt.load_scene(rt.scene.munich, merge_shapes=False)
    stale = out_of_band_materials(scene, 78e9)
    stale_types = {mat.itu_type for mat in stale.values()}
    assert stale_types == {"marble", "brick"}


@pytest.mark.sionna
def test_out_of_band_materials_finds_etoiles_marble_only(sionna_rt):
    """Etoile does NOT use `brick` -- detection must reflect the scene it is actually
    given, not a list carried over from munich. This is the generality check for
    'detect programmatically, don't hardcode marble/brick'."""
    import sionna.rt as rt

    from e2e.environment.city_scenes import out_of_band_materials

    scene = rt.load_scene(rt.scene.etoile, merge_shapes=False)
    stale = out_of_band_materials(scene, 78e9)
    stale_types = {mat.itu_type for mat in stale.values()}
    assert stale_types == {"marble"}


# --------------------------------------------------------------------------------
# The trap: registered-but-unused materials still block `scene.frequency`
# --------------------------------------------------------------------------------
@pytest.mark.sionna
def test_removal_of_stale_material_is_genuinely_required(sionna_rt):
    """Reassigning every OBJECT off a stale material is not enough on its own --
    `Scene.frequency`'s setter iterates every REGISTERED material (`is_used` or not),
    so the stale material must also be `scene.remove()`d. Proven independent of this
    module's own code: does the reassignment by hand, WITHOUT calling
    `prepare_scene_for_frequency`, and checks that a naive fix which skips the removal
    step still fails -- so a future refactor can't silently drop it.

    Uses etoile (one out-of-band material, `marble`) so there is exactly one stale
    material in play and no ambiguity about which one is still blocking.
    """
    import sionna.rt as rt

    from e2e.environment.city_scenes import out_of_band_materials

    scene = rt.load_scene(rt.scene.etoile, merge_shapes=False)
    freq = 78e9
    stale = out_of_band_materials(scene, freq)
    assert stale, "etoile has no out-of-band material at 78 GHz -- test assumption broken"
    name, mat = next(iter(stale.items()))

    stand_in = rt.ITURadioMaterial(f"{name}-standin", "concrete", thickness=0.1)
    for obj in scene.objects.values():
        if obj.radio_material.name == name:
            obj.radio_material = stand_in
    assert not scene.get(name).is_used, "reassignment did not free the stale material"

    # Unused, but still REGISTERED -- must still block.
    with pytest.raises(ValueError):
        scene.frequency = freq

    # Only removing it lets the assignment through.
    scene.remove(name)
    scene.frequency = freq
    assert float(scene.frequency[0]) == pytest.approx(freq, rel=1e-6)


# --------------------------------------------------------------------------------
# Both substitution policies
# --------------------------------------------------------------------------------
@pytest.mark.sionna
def test_extrapolated_policy_is_frequency_independent_for_marble(sionna_rt):
    """Marble's ITU curve fit has exponent b=0 (eps_r = a * f_GHz^0), so `EXTRAPOLATED`
    must report the SAME relative permittivity at two different out-of-band
    frequencies -- the physical claim the module docstring makes."""
    from e2e.environment.city_scenes import EXTRAPOLATED, load_city_scene

    _, report_77 = load_city_scene("munich", 77e9, policy=EXTRAPOLATED)
    _, report_90 = load_city_scene("munich", 90e9, policy=EXTRAPOLATED)

    assert report_77["marble"].relative_permittivity == pytest.approx(7.074, rel=1e-3)
    assert report_77["marble"].relative_permittivity == pytest.approx(
        report_90["marble"].relative_permittivity, rel=1e-6)
    # Conductivity DOES vary with frequency (d != 0) -- not a frozen constant.
    assert report_77["marble"].conductivity != pytest.approx(
        report_90["marble"].conductivity, rel=1e-3)


@pytest.mark.sionna
def test_stand_in_policy_swaps_to_an_in_band_material(sionna_rt):
    """`STAND_IN` must leave nothing out-of-band behind (checked via
    `out_of_band_materials`, not by re-deriving what got swapped) and must record which
    ITU type it stood in for each substitution."""
    from e2e.environment.city_scenes import (
        DEFAULT_STAND_IN_ITU_TYPE,
        STAND_IN,
        load_city_scene,
        out_of_band_materials,
    )

    scene, report = load_city_scene("munich", 78e9, policy=STAND_IN)
    assert set(report) == {"marble", "brick"}
    for sub in report.values():
        assert sub.stand_in_itu_type == DEFAULT_STAND_IN_ITU_TYPE
    assert out_of_band_materials(scene, 78e9) == {}


@pytest.mark.sionna
def test_both_policies_produce_different_material_constants(sionna_rt):
    """The two policies are not cosmetically different -- they hand the solver
    different `(eps_r, sigma)` for the same original material."""
    from e2e.environment.city_scenes import EXTRAPOLATED, STAND_IN, load_city_scene

    _, extrap = load_city_scene("munich", 78e9, policy=EXTRAPOLATED)
    _, stand_in = load_city_scene("munich", 78e9, policy=STAND_IN)

    assert extrap["marble"].relative_permittivity != pytest.approx(
        stand_in["marble"].relative_permittivity, rel=1e-3)
    assert extrap["marble"].conductivity != pytest.approx(
        stand_in["marble"].conductivity, rel=1e-3)
    # STAND_IN's constant is concrete's, regardless of which material it replaced.
    assert stand_in["marble"].relative_permittivity == pytest.approx(
        stand_in["brick"].relative_permittivity, rel=1e-6)


@pytest.mark.sionna
def test_both_policies_produce_a_working_scene(sionna_rt):
    """Both policies must actually let `scene.frequency` be assigned (no raise) and
    leave the scene fully in-band afterwards."""
    from e2e.environment.city_scenes import (
        EXTRAPOLATED,
        STAND_IN,
        load_city_scene,
        out_of_band_materials,
    )

    for policy in (EXTRAPOLATED, STAND_IN):
        scene, report = load_city_scene("etoile", 78e9, policy=policy)
        assert report, f"{policy}: expected at least one substitution for etoile"
        assert float(scene.frequency[0]) == pytest.approx(78e9, rel=1e-6)
        assert out_of_band_materials(scene, 78e9) == {}


# --------------------------------------------------------------------------------
# Wired into RTEnvironmentBlock / build_rt_scene: real solves
# --------------------------------------------------------------------------------
@pytest.mark.sionna
def test_munich_loads_and_solves_at_the_radars_frequency(sionna_rt, cfg):
    """End to end through the SAME path `RTEnvironmentBlock.get_S_pars()` uses
    (`patched_builtin_loader` wrapping `e2e.ml.rt_gen.build_rt_scene`): munich must
    solve at 77-ish GHz and find real paths, not silently return zero."""
    from e2e.environment.city_scenes import EXTRAPOLATED, patched_builtin_loader
    from e2e.ml.rt_gen import _solve, build_rt_scene

    sc = Scenario(name="munich_test", base_scene="munich", num_frames=1,
                 nodes=[_munich_node()], objects=[])
    f_center = float(cfg.f0_hz) + float(cfg.bandwidth_hz) / 2.0

    with patched_builtin_loader(f_center, policy=EXTRAPOLATED):
        rts = build_rt_scene(sc, cfg, base_scene="munich")
    assert rts.scene.frequency[0] == pytest.approx(f_center, rel=1e-6)

    paths = _solve(rts, max_depth=2, include_leakage=False, diffuse_reflection=True,
                  specular_reflection=True, refraction=False, seed=41)
    assert int(paths.tau.shape[-1]) > 0, "munich at 77 GHz found no paths"


@pytest.mark.sionna
def test_etoile_loads_and_solves_at_the_radars_frequency(sionna_rt, cfg):
    from e2e.environment.city_scenes import EXTRAPOLATED, patched_builtin_loader
    from e2e.ml.rt_gen import _solve, build_rt_scene

    sc = Scenario(name="etoile_test", base_scene="etoile", num_frames=1,
                 nodes=[_etoile_node()], objects=[])
    f_center = float(cfg.f0_hz) + float(cfg.bandwidth_hz) / 2.0

    with patched_builtin_loader(f_center, policy=EXTRAPOLATED):
        rts = build_rt_scene(sc, cfg, base_scene="etoile")
    assert rts.scene.frequency[0] == pytest.approx(f_center, rel=1e-6)

    paths = _solve(rts, max_depth=2, include_leakage=False, diffuse_reflection=True,
                  specular_reflection=True, refraction=False, seed=41)
    assert int(paths.tau.shape[-1]) > 0, "etoile at 77 GHz found no paths"


@pytest.mark.sionna
def test_rt_environment_block_produces_a_real_frame_for_munich(sionna_rt, cfg,
                                                                torch_device):
    """`RTEnvironmentBlock(base_scene="munich")` -- the actual class this feature was
    built for -- must return a finite, non-degenerate frame and populate
    `last_material_report` with what it substituted."""
    from e2e.environment.blocks import RTEnvironmentBlock

    sc = Scenario(name="munich_block_test", base_scene="munich", num_frames=1,
                 nodes=[_munich_node()], objects=[])
    blk = RTEnvironmentBlock(sc, cfg, base_scene="munich", device=torch_device,
                             max_depth=2)
    s_pars = blk.get_S_pars()

    assert s_pars.dtype == torch.complex64
    assert s_pars.device.type == torch_device.type
    assert torch.isfinite(s_pars.real).all() and torch.isfinite(s_pars.imag).all()
    assert blk.last_material_report, "munich should have needed material substitutions"
    assert {sub.original_itu_type for sub in blk.last_material_report.values()} \
        == {"marble", "brick"}


@pytest.mark.sionna
def test_rt_environment_block_flat_default_is_unaffected(sionna_rt, cfg, torch_device):
    """`base_scene="flat"` (the default) must never touch `city_scenes` at all --
    `last_material_report` stays `None`, not an empty dict."""
    from e2e.environment.blocks import RTEnvironmentBlock

    sc = Scenario(name="flat_block_test", base_scene="flat", num_frames=1,
                 nodes=[Node(name="radar", role=NodeRole.RADAR, position=(0.0, 0.0, 1.5),
                            look_at=(10.0, 0.0, 1.5))],
                 objects=[])
    blk = RTEnvironmentBlock(sc, cfg, device=torch_device, max_depth=2)   # base_scene default
    assert blk.base_scene == "flat"
    s_pars = blk.get_S_pars()
    assert torch.isfinite(s_pars.real).all() and torch.isfinite(s_pars.imag).all()
    assert blk.last_material_report is None


# --------------------------------------------------------------------------------
# Cost measurement: is a city tier affordable for the ML corpus?
# --------------------------------------------------------------------------------
@pytest.mark.sionna
def test_measure_city_vs_flat_frame_cost(sionna_rt, torch_device):
    """MEASURE wall time for one full `RTEnvironmentBlock` frame (scene build + swap +
    solve + CFR sampling), munich vs flat, AT THE ti_iwr1443 PRESET -- this is the
    number the ML corpus's city-tier affordability call rests on. Printed (run with
    `pytest -s` to see the numbers); the only hard assertions are that both frames
    actually come out finite and that city is (unsurprisingly) the slower one -- the
    wall-clock figures themselves are hardware-dependent and belong in the run's
    report, not a time-budget assertion that would be flaky across machines.
    """
    import time

    from e2e.environment.blocks import RTEnvironmentBlock

    sc_munich = Scenario(name="munich_cost", base_scene="munich", num_frames=1,
                         nodes=[_munich_node()], objects=[])
    sc_flat = Scenario(name="flat_cost", base_scene="flat", num_frames=1,
                       nodes=[Node(name="radar", role=NodeRole.RADAR,
                                  position=(0.0, 0.0, 1.5), look_at=(10.0, 0.0, 1.5))],
                       objects=[])

    blk_munich = RTEnvironmentBlock(sc_munich, TI_IWR1443, base_scene="munich",
                                    device=torch_device, max_depth=2)
    t0 = time.time()
    s_munich = blk_munich.get_S_pars()
    t_munich = time.time() - t0

    blk_flat = RTEnvironmentBlock(sc_flat, TI_IWR1443, base_scene="flat",
                                  device=torch_device, max_depth=2)
    t0 = time.time()
    s_flat = blk_flat.get_S_pars()
    t_flat = time.time() - t0

    print(f"\n[city_scenes cost] ti_iwr1443 preset, max_depth=2, one frame each: "
         f"munich={t_munich:.2f}s  flat={t_flat:.2f}s  "
         f"(munich/flat ratio={t_munich / max(t_flat, 1e-9):.0f}x)")

    assert torch.isfinite(s_munich.real).all() and torch.isfinite(s_munich.imag).all()
    assert torch.isfinite(s_flat.real).all() and torch.isfinite(s_flat.imag).all()
    assert t_munich > t_flat, "a city scene should never be cheaper than the flat one"
