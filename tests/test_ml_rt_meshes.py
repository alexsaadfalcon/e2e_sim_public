"""Real object meshes for ray-traced ML data (campaign R2).

Two halves:

* Ungated (default `pytest`, no Sionna): `e2e.ml.rt_scenes`' tier determinism and the
  procedural pedestrian placeholder's geometry -- neither needs a ray tracer, only
  `e2e.scenario` / `e2e.ml.rt_gen`'s asset-name constants (torch is imported by
  `e2e.ml.rt_gen` at module level, so this whole file needs torch installed, same as
  the rest of `e2e.ml`'s test suite -- see CLAUDE.md).
* Gated (`@pytest.mark.sionna`, RUN_SIONNA=1): every bundled car mesh loads and is
  automotive-scale, the pedestrian placeholder loads through Sionna itself, a D1 (car
  mesh) scene solves to a nonzero return monostatically (regression for the specular/
  diffuse-invisibility bug), and `scenario_runner`'s three bug fixes (diffuse
  scattering, BOX mesh resolution, real car meshes in `sionna_env.add_cars`).

Sionna is imported inside a session fixture (`sionna_rt`), never at module level --
see `tests/test_ml_rt_gen.py`'s module docstring for why (DrJit/CUDA init at collection
would break the ungated suite).
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from e2e.ml.rt_gen import ASSET_LICENSES, CAR_ASSET_NAMES, PEDESTRIAN_ASSET_NAME
from e2e.ml.rt_scenes import RT_DIFFICULTY_TIERS, build_rt_tier_scenario, tier_summary
from e2e.scenario import ObjectKind


@pytest.fixture(scope="session")
def sionna_rt():
    """Import Sionna RT once per session (skips the module if it is unavailable)."""
    return pytest.importorskip("sionna.rt")


# --------------------------------------------------------------------------------
# Asset inventory (ungated -- pure constants/dicts, no Sionna needed)
# --------------------------------------------------------------------------------
def test_seventeen_bundled_car_assets_recorded():
    assert len(CAR_ASSET_NAMES) == 17
    assert "low_poly_car" in CAR_ASSET_NAMES
    assert len(set(CAR_ASSET_NAMES)) == 17  # no accidental duplicate names


def test_asset_licenses_cover_cars_and_pedestrian():
    assert ASSET_LICENSES["sionna_cars"]["license"] == "Apache-2.0"
    assert ASSET_LICENSES["sionna_cars"]["count"] == len(CAR_ASSET_NAMES)
    assert PEDESTRIAN_ASSET_NAME in ASSET_LICENSES
    # the pedestrian placeholder is explicitly NOT a third-party asset.
    assert "PLACEHOLDER" in ASSET_LICENSES[PEDESTRIAN_ASSET_NAME]["status"]


# --------------------------------------------------------------------------------
# Procedural pedestrian placeholder geometry (ungated -- pure Python mesh generation)
# --------------------------------------------------------------------------------
def _read_ascii_ply_vertices(path):
    with open(path) as f:
        lines = f.readlines()
    n_v = int(next(l for l in lines if l.startswith("element vertex")).split()[-1])
    hdr_end = lines.index("end_header\n")
    verts = [tuple(float(x) for x in l.split()) for l in lines[hdr_end + 1:hdr_end + 1 + n_v]]
    return verts


def test_pedestrian_placeholder_is_generated_not_committed(tmp_path):
    from e2e.ml.rt_gen import _pedestrian_mesh_path

    path = _pedestrian_mesh_path()
    # lives under a temp dir (regenerated per process), not inside the repo tree.
    assert "e2e-rt-pedestrian" in path.replace("\\", "/")
    verts = _read_ascii_ply_vertices(path)
    assert len(verts) > 0


def test_pedestrian_placeholder_height_in_range():
    from e2e.ml.rt_gen import _pedestrian_mesh_path

    verts = _read_ascii_ply_vertices(_pedestrian_mesh_path())
    zs = [v[2] for v in verts]
    height = max(zs) - min(zs)
    assert 1.6 <= height <= 1.9, f"pedestrian placeholder height {height:.2f} m out of range"
    assert min(zs) == pytest.approx(0.0, abs=1e-6)  # feet on the ground


def test_pedestrian_placeholder_path_is_cached():
    from e2e.ml.rt_gen import _pedestrian_mesh_path

    assert _pedestrian_mesh_path() == _pedestrian_mesh_path()


# --------------------------------------------------------------------------------
# RT difficulty tiers (ungated -- e2e.ml.rt_scenes is pure Python/numpy)
# --------------------------------------------------------------------------------
@pytest.mark.parametrize("tier", sorted(RT_DIFFICULTY_TIERS))
def test_every_tier_builds_a_valid_scenario(tier):
    sc = build_rt_tier_scenario(tier, frame_idx=0, seed=1, num_frames=2)
    assert sc.validate() == []
    assert len(sc.nodes) == 1
    summary = tier_summary(sc)
    assert summary["tier"] == tier


def test_tier_determinism_same_triple_is_byte_identical():
    a = build_rt_tier_scenario("D2", frame_idx=3, seed=7, num_frames=4)
    b = build_rt_tier_scenario("D2", frame_idx=3, seed=7, num_frames=4)
    assert a.to_json() == b.to_json()


def test_tier_determinism_num_frames_is_not_part_of_the_key():
    """Same (tier, frame_idx, seed) with a different num_frames keeps the same object
    mix/positions -- only the returned Scenario's own frame count changes."""
    a = build_rt_tier_scenario("D1", frame_idx=0, seed=5, num_frames=2)
    b = build_rt_tier_scenario("D1", frame_idx=0, seed=5, num_frames=9)
    assert [o.name for o in a.objects] == [o.name for o in b.objects]
    assert [o.position for o in a.objects] == [o.position for o in b.objects]
    assert a.num_frames == 2 and b.num_frames == 9


@pytest.mark.parametrize("vary", ["frame_idx", "seed", "tier"])
def test_tier_determinism_changing_any_key_component_changes_the_draw(vary):
    base = dict(tier="D2", frame_idx=0, seed=1)
    changed = dict(base)
    if vary == "tier":
        changed["tier"] = "D3"
    else:
        changed[vary] = base[vary] + 1

    a = build_rt_tier_scenario(**base, num_frames=2)
    b = build_rt_tier_scenario(**changed, num_frames=2)
    assert a.to_json() != b.to_json()


def test_d0_is_spheres_only_d1_is_cars_only():
    d0 = build_rt_tier_scenario("D0", frame_idx=0, seed=0)
    assert all(o.kind == ObjectKind.SPHERE for o in d0.objects)

    d1 = build_rt_tier_scenario("D1", frame_idx=0, seed=0)
    assert len(d1.objects) > 0
    assert all(o.kind == ObjectKind.MESH and o.asset in CAR_ASSET_NAMES for o in d1.objects)


def test_d2_and_d3_mix_cars_and_pedestrians():
    for tier in ("D2", "D3"):
        sc = build_rt_tier_scenario(tier, frame_idx=1, seed=2)
        classes = {o.object_class for o in sc.objects}
        assert "vehicle" in classes
        # not every draw is guaranteed a pedestrian (count ranges include 0), so check
        # the asset is wired correctly whenever one IS drawn.
        peds = [o for o in sc.objects if o.object_class == "pedestrian"]
        for p in peds:
            assert p.kind == ObjectKind.MESH
            assert p.asset == PEDESTRIAN_ASSET_NAME
            assert p.material == "skin"


def test_car_objects_reference_a_real_bundled_asset_name():
    sc = build_rt_tier_scenario("D3", frame_idx=4, seed=9)
    cars = [o for o in sc.objects if o.object_class == "vehicle" and o.kind == ObjectKind.MESH]
    assert cars, "expected at least one car in this draw"
    for c in cars:
        assert c.asset in CAR_ASSET_NAMES


# --------------------------------------------------------------------------------
# Gated: mesh loading + scale checks (real Sionna RT)
# --------------------------------------------------------------------------------
@pytest.mark.sionna
@pytest.mark.parametrize("name", CAR_ASSET_NAMES)
def test_every_bundled_car_mesh_loads_and_is_automotive_scale(sionna_rt, name):
    from e2e.ml.rt_gen import _car_mesh_path

    path = _car_mesh_path(sionna_rt, name)
    mesh = sionna_rt.load_mesh(path)
    bbox = mesh.bbox()
    extents = np.asarray((bbox.max - bbox.min).numpy()).reshape(-1)
    length = max(float(extents[0]), float(extents[1]))
    height = float(extents[2])
    assert 3.5 <= length <= 5.5, f"{name}: length {length:.2f} m out of range"
    assert 1.2 <= height <= 2.0, f"{name}: height {height:.2f} m out of range"


@pytest.mark.sionna
def test_pedestrian_placeholder_loads_through_sionna(sionna_rt):
    from e2e.ml.rt_gen import _pedestrian_mesh_path

    mesh = sionna_rt.load_mesh(_pedestrian_mesh_path())
    bbox = mesh.bbox()
    extents = np.asarray((bbox.max - bbox.min).numpy()).reshape(-1)
    assert 1.6 <= float(extents[2]) <= 1.9


@pytest.mark.sionna
def test_object_mesh_dispatch_resolves_cars_and_pedestrian(sionna_rt):
    """`rt_gen._object_mesh` resolves sphere/box/car-name/pedestrian-sentinel/raw-path."""
    import dataclasses

    from e2e.ml.rt_gen import _object_mesh
    from e2e.scenario import SceneObject

    sphere = _object_mesh(sionna_rt, SceneObject(name="s", kind=ObjectKind.SPHERE))
    assert sphere == sionna_rt.scene.sphere

    box = _object_mesh(sionna_rt, SceneObject(name="b", kind=ObjectKind.BOX))
    assert box.endswith(".ply") and "box" in box.lower()

    car = _object_mesh(sionna_rt, SceneObject(name="c", kind=ObjectKind.MESH, asset="low_poly_car"))
    assert car.endswith("low_poly_car.ply")

    ped = _object_mesh(sionna_rt, SceneObject(name="p", kind=ObjectKind.MESH, asset=PEDESTRIAN_ASSET_NAME))
    assert ped.endswith(".ply")

    raw = _object_mesh(sionna_rt, SceneObject(name="r", kind=ObjectKind.MESH, asset="/some/path.ply"))
    assert raw == "/some/path.ply"


# --------------------------------------------------------------------------------
# Gated: D1 (real car meshes) solves to a nonzero monostatic return
# --------------------------------------------------------------------------------
@pytest.mark.sionna
def test_d1_tier_scene_solves_to_nonzero_return_monostatically(sionna_rt):
    """Regression for the specular/diffuse-invisibility bug: a real object (here, a
    car mesh, not a bare sphere) must actually scatter energy back to a monostatic
    radar, not just geometrically exist in the scene."""
    import dataclasses

    from e2e.ml.radar_config import TI_IWR1443
    from e2e.ml.rt_gen import _beat_from_paths, _solve, build_rt_scene

    cfg = dataclasses.replace(TI_IWR1443, name="rt_mesh_test", n_chirps=8, n_samples=64)
    scenario = build_rt_tier_scenario("D1", frame_idx=0, seed=0, num_frames=1)

    rt_scene = build_rt_scene(scenario, cfg, base_scene="free", frame_idx=0)
    paths = _solve(rt_scene, max_depth=2, include_leakage=False, diffuse_reflection=True,
                   specular_reflection=True, refraction=False, seed=41)
    beat = _beat_from_paths(paths, cfg, n_chirps=cfg.n_chirps)
    assert np.isfinite(beat).all()
    assert np.abs(beat).sum() > 0, "no monostatic return from the D1 (car mesh) scene"


# --------------------------------------------------------------------------------
# Gated: scenario_runner bug fixes
# --------------------------------------------------------------------------------
@pytest.mark.sionna
def test_scenario_runner_box_mesh_path_resolves_a_real_ply(sionna_rt):
    """Bug (b): `rt.scene.box` is a SCENE path (box.xml), not a mesh -- the fix must
    resolve to the mesh one level down."""
    from e2e.environment.scenario_runner import ScenarioRunner

    path = ScenarioRunner._box_mesh_path(sionna_rt)
    assert path.endswith(".ply")
    mesh = sionna_rt.load_mesh(path)  # raises if not a loadable mesh
    bbox = mesh.bbox()
    assert float(bbox.max.x) > float(bbox.min.x)


@pytest.mark.sionna
def test_scenario_runner_box_object_builds_in_a_real_scene(sionna_rt):
    """End-to-end: a BOX SceneObject added via `_add_scene_object` to a real scene."""
    from e2e.environment.scenario_runner import ScenarioRunner
    from e2e.scenario import (ArrayConfig, FrequencyPlan, Node, NodeRole, Scenario,
                              SceneObject)

    sc = Scenario(
        name="box_object_test", base_scene="floor_wall",
        frequency=FrequencyPlan(carrier_hz=30e9, start_hz=29.5e9, stop_hz=30.5e9, num_freqs=4),
        num_frames=1,
        nodes=[Node(name="radar", role=NodeRole.RADAR, position=(-1.5, 0.0, 1.0),
                    look_at=(1.5, 0.0, 1.0), array=ArrayConfig(num_rows=2, num_cols=2))],
        objects=[SceneObject(name="crate", kind=ObjectKind.BOX, position=(-0.5, 0.0, 1.0),
                             scaling=0.3, material="metal")],
    )
    runner = ScenarioRunner(sc, dry_run=False)
    runner._setup_sionna()
    ls = runner._sionna["link_scenes"][runner.primary_link.name]
    so = ls["scene_objs"]["crate"]
    assert so.position.x[0] == pytest.approx(-0.5, abs=1e-3)


@pytest.mark.sionna
def test_scenario_runner_real_sphere_target_is_visible_monostatically(sionna_rt, tmp_path):
    """Regression for bug (a): before the fix (scattering_coefficient unset,
    diffuse_reflection=False at solve time), a curved object had no specular return
    straight back at a monostatic radar and was effectively invisible."""
    from e2e.environment.scenario_runner import ScenarioRunner
    from e2e.scenario import ArrayConfig, FrequencyPlan, Node, NodeRole, Scenario, SceneObject

    sc = Scenario(
        name="sphere_visibility_test", base_scene="floor_wall",
        frequency=FrequencyPlan(carrier_hz=30e9, start_hz=29.5e9, stop_hz=30.5e9, num_freqs=8),
        num_frames=1,
        nodes=[Node(name="radar", role=NodeRole.RADAR, position=(-1.5, 0.0, 1.0),
                    look_at=(1.5, 0.0, 1.0), array=ArrayConfig(num_rows=2, num_cols=2))],
        objects=[SceneObject(name="ball", kind=ObjectKind.SPHERE, position=(-0.7, 0.0, 1.0),
                             scaling=0.2, material="metal")],
    )
    runner = ScenarioRunner(sc, dry_run=False)
    payload = runner.run(out_path=str(tmp_path / "sphere.pkl"), verbose=False)
    arr = payload["links"][runner.primary_link.name]
    assert np.isfinite(arr).all()
    assert np.abs(arr).sum() > 0, "monostatic sphere target produced no return at all"


@pytest.mark.sionna
def test_sionna_env_add_cars_default_places_real_car_meshes(sionna_rt):
    """Bug (c): `add_cars()` must place real car meshes by default (docstring said
    "cars", code placed spheres); `shape='sphere'` stays available as an explicit
    opt-out."""
    from e2e.environment.sionna_env import SionnaEnvironment

    env = SionnaEnvironment(scene_name="etoile", num_cars=2, car_radius=20.0,
                            car_center=(0.0, 0.0, 0.0))
    env.add_cars()  # default shape="car"
    assert len(env.cars) == 2
    for car in env.cars:
        bbox = car._mi_mesh.bbox()
        extents = np.asarray((bbox.max - bbox.min).numpy()).reshape(-1)
        length = max(float(extents[0]), float(extents[1]))
        # real car mesh at scaling=1.0 -> automotive length; the old sphere placeholder
        # at scaling=5.0 would have measured ~9.8 m (2*0.987*5), well outside this range.
        assert 3.5 <= length <= 5.5, f"car extent {length:.2f} m looks like a sphere, not a car"


@pytest.mark.sionna
def test_sionna_env_add_cars_sphere_opt_out_still_works(sionna_rt):
    from e2e.environment.sionna_env import SionnaEnvironment

    env = SionnaEnvironment(scene_name="etoile", num_cars=2, car_radius=20.0,
                            car_center=(0.0, 0.0, 0.0))
    env.add_cars(shape="sphere")
    assert len(env.cars) == 2
    bbox = env.cars[0]._mi_mesh.bbox()
    extents = np.asarray((bbox.max - bbox.min).numpy()).reshape(-1)
    diameter = float(extents[0])
    assert 8.0 <= diameter <= 11.0  # ~2 * 0.987 m radius * 5.0 scaling


@pytest.mark.sionna
def test_sionna_env_add_cars_rejects_unknown_shape(sionna_rt):
    from e2e.environment.sionna_env import SionnaEnvironment

    env = SionnaEnvironment(scene_name="etoile", num_cars=1, car_radius=20.0,
                            car_center=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        env.add_cars(shape="not_a_real_shape")
