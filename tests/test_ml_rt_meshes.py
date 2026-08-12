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

from e2e.ml.assets import DOWNLOADED_ASSET_SPECS
from e2e.ml.rt_gen import (ASSET_LICENSES, CAR_ASSET_NAMES, LOCAL_ASSET_SPECS,
                          LOCAL_PEDESTRIAN_ASSET_NAMES, LOCAL_VEHICLE_ASSET_NAMES,
                          PEDESTRIAN_ASSET_NAME, SIONNA_CAR_REPRESENTATIVE,
                          object_local_height_m)
from e2e.ml.rt_scenes import (RT_DIFFICULTY_TIERS, VEHICLE_CLASS_POOLS,
                             build_rt_tier_scenario, tier_summary, vehicle_asset_class)
from e2e.scenario import ObjectKind

# Every asset name a "vehicle" object can legitimately carry in the DEFAULT (use_local_
# assets=False) pool -- one Sionna representative + every downloaded car/truck/bus/
# trolley mesh (present or not on this machine; a missing raw source is a LOAD-time
# degrade, not a name-pool restriction -- see rt_gen._object_mesh).
_DEFAULT_VEHICLE_POOL = frozenset(n for pool in VEHICLE_CLASS_POOLS.values() for n in pool)


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
    assert all(o.kind == ObjectKind.MESH and o.asset in _DEFAULT_VEHICLE_POOL for o in d1.objects)


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


def test_car_objects_reference_a_known_vehicle_asset_name():
    sc = build_rt_tier_scenario("D3", frame_idx=4, seed=9)
    cars = [o for o in sc.objects if o.object_class == "vehicle" and o.kind == ObjectKind.MESH]
    assert cars, "expected at least one vehicle in this draw"
    for c in cars:
        assert c.asset in _DEFAULT_VEHICLE_POOL


# --------------------------------------------------------------------------------
# Pool composition (ungated -- regression for "17 names, ONE geometry": a car-class
# draw must be able to land on more than one distinct mesh, and a tier's vehicles must
# be a MIX of classes, not always "car").
# --------------------------------------------------------------------------------
def test_duplicate_sionna_car_name_problem_cannot_come_back():
    """Across many draws, the car CLASS specifically (not just "any vehicle") must
    produce more than one distinct asset name -- the exact failure mode this campaign
    fixes (17 Sionna names, all `low_poly_car.ply`, drawn uniformly)."""
    seen = set()
    for tier in ("D2", "D3"):
        for seed in range(30):
            sc = build_rt_tier_scenario(tier, frame_idx=0, seed=seed)
            for o in sc.objects:
                if o.object_class == "vehicle" and o.kind == ObjectKind.MESH \
                        and vehicle_asset_class(o.asset) == "car":
                    seen.add(o.asset)
    assert len(seen) > 1, f"only ever drew {seen!r} for the car class"


def test_no_uniform_draw_over_all_seventeen_sionna_names():
    """None of the 16 duplicate-geometry Sionna scene-slot names (car-0..7/car_1..8)
    should appear in a default-pool draw -- only the one kept representative."""
    dup_names = set(CAR_ASSET_NAMES) - {SIONNA_CAR_REPRESENTATIVE}
    for tier in ("D1", "D2", "D3"):
        for seed in range(20):
            sc = build_rt_tier_scenario(tier, frame_idx=0, seed=seed)
            for o in sc.objects:
                if o.object_class == "vehicle" and o.kind == ObjectKind.MESH:
                    assert o.asset not in dup_names


def test_a_tier_draws_more_than_just_cars_over_many_seeds():
    """D3 (widest n_cars range) should, over enough draws, produce at least one
    non-"car" vehicle class (truck/bus/trolley) -- the "realistic mix" fix."""
    classes_seen = set()
    for seed in range(60):
        sc = build_rt_tier_scenario("D3", frame_idx=0, seed=seed)
        for o in sc.objects:
            if o.object_class == "vehicle" and o.kind == ObjectKind.MESH:
                classes_seen.add(vehicle_asset_class(o.asset))
    assert classes_seen - {"car"}, f"never drew a non-car vehicle class: {classes_seen!r}"


# --------------------------------------------------------------------------------
# Clutter-box occlusion (ungated -- pure geometry, item 5 fix)
# --------------------------------------------------------------------------------
def test_clutter_boxes_do_not_sit_in_a_targets_line_of_sight():
    import math

    from e2e.ml.rt_scenes import _CLUTTER_LOS_MARGIN_SIN_AZ, _CLUTTER_LOS_RANGE_MARGIN_M

    def _polar(dx, dy):
        r = math.hypot(dx, dy)
        return r, (dy / r if r > 1e-6 else 0.0)

    for tier in ("D2", "D3"):
        for seed in range(40):
            sc = build_rt_tier_scenario(tier, frame_idx=0, seed=seed)
            rx0, ry0, _ = sc.nodes[0].position
            targets = [(o.position[0] - rx0, o.position[1] - ry0) for o in sc.objects
                      if o.object_class in ("vehicle", "pedestrian")]
            boxes = [(o.position[0] - rx0, o.position[1] - ry0) for o in sc.objects
                    if o.kind == ObjectKind.BOX]
            target_polar = [_polar(*t) for t in targets]
            for b in boxes:
                br, bsin = _polar(*b)
                for tr, tsin in target_polar:
                    occludes = (abs(bsin - tsin) < _CLUTTER_LOS_MARGIN_SIN_AZ
                               and br < tr + _CLUTTER_LOS_RANGE_MARGIN_M)
                    assert not occludes, \
                        f"{tier}/seed={seed}: a clutter box occludes a target"


# --------------------------------------------------------------------------------
# Ground-rest placement (ungated -- object_local_height_m/build_rt_tier_scenario's
# z-placement math is pure Python; only ASSETS THAT NEED SIONNA TO LOAD are gated).
# --------------------------------------------------------------------------------
def test_object_local_height_m_known_kinds():
    assert object_local_height_m(ObjectKind.SPHERE) > 0
    assert object_local_height_m(ObjectKind.BOX) > 0
    assert object_local_height_m(ObjectKind.MESH, "low_poly_car") > 0
    assert object_local_height_m(ObjectKind.MESH, PEDESTRIAN_ASSET_NAME) == pytest.approx(1.74)
    with pytest.raises(ValueError):
        object_local_height_m(ObjectKind.MESH, "not-a-real-asset")


@pytest.mark.parametrize("tier", sorted(RT_DIFFICULTY_TIERS))
def test_every_object_position_z_is_ground_rest_centre(tier):
    """Regression for the sinking-into-the-ground bug: `build_rt_tier_scenario` must
    place each object's CENTER at `0.5 * object_local_height_m(...) * scaling`, not at
    world z=0 -- z=0 is what Sionna's `SceneObject.position` setter would then use as
    the bbox CENTER, burying the bottom half of the object below ground (see
    `e2e.ml.rt_gen`'s module docstring)."""
    sc = build_rt_tier_scenario(tier, frame_idx=0, seed=1, num_frames=1)
    for o in sc.objects:
        expected_z = 0.5 * object_local_height_m(o.kind, o.asset) * float(o.scaling)
        assert o.position[2] == pytest.approx(expected_z, abs=1e-9), \
            f"{tier}/{o.name}: z={o.position[2]} != ground-rest centre {expected_z}"
        assert o.position[2] > 0.0, f"{tier}/{o.name}: z <= 0 can't be a ground-rest centre"


# --------------------------------------------------------------------------------
# Local (unshipped) asset library -- degrades gracefully when the files aren't present
# (this file must pass whether or not this happens to be the workstation that has
# them; see e2e.ml.rt_gen's module docstring).
# --------------------------------------------------------------------------------
def test_local_asset_specs_are_recorded_with_unknown_license():
    assert len(LOCAL_ASSET_SPECS) > 0
    for name, spec in LOCAL_ASSET_SPECS.items():
        assert name in ASSET_LICENSES
        assert "UNKNOWN" in ASSET_LICENSES[name]["license"]
        assert spec.category in ("vehicle", "pedestrian")
    assert set(LOCAL_VEHICLE_ASSET_NAMES) | set(LOCAL_PEDESTRIAN_ASSET_NAMES) \
        == set(LOCAL_ASSET_SPECS)


def test_downloaded_asset_licenses_recorded_as_unverified():
    """The five original freestl.com meshes have unchecked provenance -- generic
    UNKNOWN/INTERNAL-ONLY text. The Kenney fleet (spec.license set) is the deliberate
    exception, checked separately below."""
    for name, spec in DOWNLOADED_ASSET_SPECS.items():
        if spec.license is not None:
            continue
        assert name in ASSET_LICENSES
        lic = ASSET_LICENSES[name]["license"]
        assert "UNKNOWN" in lic and "INTERNAL-ONLY" in lic
        assert ASSET_LICENSES[name]["category"] == spec.vehicle_class


def test_kenney_fleet_licenses_recorded_as_verified_cc0():
    """The Kenney fleet's `spec.license` must flow through into ASSET_LICENSES verbatim
    (not the generic UNKNOWN text every other downloaded asset gets)."""
    kenney_names = [n for n in DOWNLOADED_ASSET_SPECS if n.startswith("kn_")]
    # 15 vehicles only -- bus/trolley box-body stand-ins were drafted then deferred by
    # the project owner (that search concluded separately -- see
    # test_owner_approved_bus_and_tram_licenses_recorded_verbatim below).
    assert len(kenney_names) == 15
    for name in kenney_names:
        spec = DOWNLOADED_ASSET_SPECS[name]
        assert spec.license is not None
        assert ASSET_LICENSES[name]["license"] == spec.license
        assert "CC0" in ASSET_LICENSES[name]["license"]
        assert ASSET_LICENSES[name]["category"] == spec.vehicle_class


def test_owner_approved_bus_and_tram_licenses_recorded_verbatim():
    """The OWNER-APPROVED bus (CC0) and tram (CC-BY 3.0, attribution required) also
    carry a verified `spec.license` -- must flow through into ASSET_LICENSES verbatim,
    same as the Kenney fleet, even though the tram's is not CC0."""
    for name in ("dl_bus_ajanhallinta", "dl_tram_google"):
        spec = DOWNLOADED_ASSET_SPECS[name]
        assert spec.license is not None
        assert ASSET_LICENSES[name]["license"] == spec.license
        assert ASSET_LICENSES[name]["category"] == spec.vehicle_class
    assert "CC-BY 3.0" in ASSET_LICENSES["dl_tram_google"]["license"]


def test_use_local_assets_false_stays_within_the_default_pool():
    """Default behaviour (use_local_assets=False) must draw ONLY from the default
    vehicle pool (SIONNA_CAR_REPRESENTATIVE + downloaded car/truck/bus/trolley meshes)
    / PEDESTRIAN_ASSET_NAME -- never a `LOCAL_ASSET_SPECS` (workstation-only,
    non-downloaded) name -- regardless of whether this machine happens to have those
    local files."""
    for tier in sorted(RT_DIFFICULTY_TIERS):
        sc = build_rt_tier_scenario(tier, frame_idx=2, seed=5, num_frames=1)
        for o in sc.objects:
            if o.object_class == "vehicle" and o.kind == ObjectKind.MESH:
                assert o.asset in _DEFAULT_VEHICLE_POOL
                assert o.asset not in LOCAL_VEHICLE_ASSET_NAMES
            if o.object_class == "pedestrian":
                assert o.asset == PEDESTRIAN_ASSET_NAME


def test_use_local_assets_true_stays_within_the_expanded_pool():
    """use_local_assets=True may ADDITIONALLY draw `LOCAL_ASSET_SPECS` names (if this
    machine has the files) but must never draw anything OUTSIDE the expanded pool --
    works whether or not the local files are actually present (graceful degrade)."""
    car_pool = _DEFAULT_VEHICLE_POOL | set(LOCAL_VEHICLE_ASSET_NAMES)
    ped_pool = {PEDESTRIAN_ASSET_NAME} | set(LOCAL_PEDESTRIAN_ASSET_NAMES)
    for tier in ("D1", "D2", "D3"):
        for seed in range(5):
            sc = build_rt_tier_scenario(tier, frame_idx=0, seed=seed, num_frames=1,
                                        use_local_assets=True)
            for o in sc.objects:
                if o.object_class == "vehicle" and o.kind == ObjectKind.MESH:
                    assert o.asset in car_pool
                if o.object_class == "pedestrian":
                    assert o.asset in ped_pool


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


@pytest.mark.sionna
@pytest.mark.parametrize("name", sorted(DOWNLOADED_ASSET_SPECS))
def test_object_mesh_dispatch_resolves_downloaded_assets(sionna_rt, name):
    """`rt_gen._object_mesh` for a downloaded (car/truck/bus/trolley) asset name --
    either the real processed PLY (if the source is present on this machine) or a
    graceful degrade to `SIONNA_CAR_REPRESENTATIVE`, either way a loadable mesh."""
    from e2e.ml.rt_gen import _object_mesh
    from e2e.scenario import SceneObject

    path = _object_mesh(sionna_rt, SceneObject(name="v", kind=ObjectKind.MESH, asset=name))
    mesh = sionna_rt.load_mesh(path)  # raises if not a loadable mesh
    assert float(mesh.bbox().max.x) > float(mesh.bbox().min.x)


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
# Gated: ground-rest placement regression (objects must not sink into the ground)
# --------------------------------------------------------------------------------
@pytest.mark.sionna
@pytest.mark.parametrize("tier", sorted(RT_DIFFICULTY_TIERS))
@pytest.mark.parametrize("use_local_assets", [False, True])
def test_every_placed_object_rests_on_or_above_ground(sionna_rt, tier, use_local_assets):
    """Regression: `SceneObject.position` re-centers the mesh's own AABB on the given
    point, so placing every object at world z=0 (the old behaviour) buried the bottom
    half of it below the ground plane. Every REAL object in a built RT scene must have
    a bbox min-z at or above the z=0 ground plane, for every tier and with/without the
    local (unshipped) asset pool wired in (use_local_assets degrades gracefully to
    Sionna meshes on a machine without the local files -- see rt_gen's module
    docstring -- so this must hold either way)."""
    from e2e.ml.radar_config import TI_IWR1443

    scenario = build_rt_tier_scenario(tier, frame_idx=0, seed=0, num_frames=1,
                                      use_local_assets=use_local_assets)
    if not scenario.objects:
        pytest.skip(f"{tier} draw at this seed placed zero objects")

    from e2e.ml.rt_gen import build_rt_scene

    rt_scene = build_rt_scene(scenario, TI_IWR1443, base_scene="flat", frame_idx=0)
    for obj in scenario.objects:
        so = rt_scene.objects[obj.name]
        bbox = so._mi_mesh.bbox()
        min_z = float(np.asarray(bbox.min.numpy()).reshape(-1)[2])
        assert min_z >= -1e-3, f"{tier}/{obj.name} (asset={obj.asset!r}) sinks into the ground: min_z={min_z:.4f}"


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


def test_placed_objects_do_not_interpenetrate():
    """No two objects may occupy the same patch of ground.

    Footprints are BOXES, not circles. Circles were the first attempt and the user spotted
    the result in a render: a 3.2 m radius inscribed in a 6x6 m clutter box leaves the
    corners exposed, so two boxes cleared the distance test and still intersected. Measured
    at the time: 0.57% of pairs overlapped. This asserts zero, using the module's own
    footprints so the test tracks placement rather than re-guessing it.
    """
    from e2e.ml.rt_scenes import build_rt_tier_scenario, _footprint

    def box(obj):
        if obj.name.startswith("pedestrian"):
            ext = _footprint("pedestrian")
        elif obj.name.startswith("sphere"):
            ext = _footprint("sphere")
        elif obj.name.startswith("clutter"):
            ext = _footprint("scatterer")
        else:
            ext = _footprint("vehicle", obj.asset)
        x, y = obj.position[0], obj.position[1]
        return (x - ext[0] / 2, x + ext[0] / 2, y - ext[1] / 2, y + ext[1] / 2)

    overlaps = pairs = 0
    for tier in ("D0", "D1", "D2", "D3"):
        for frame in range(25):
            objects = build_rt_tier_scenario(tier, frame_idx=frame, seed=17).objects
            for i in range(len(objects)):
                for j in range(i + 1, len(objects)):
                    a, b = box(objects[i]), box(objects[j])
                    pairs += 1
                    if min(a[1], b[1]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[2], b[2]):
                        overlaps += 1

    assert overlaps == 0, f"{overlaps}/{pairs} object footprints intersect"


def test_no_vehicle_extends_back_through_the_radar():
    """A 16 m semi-trailer centred at the 6 m minimum range reaches BEHIND the antenna.
    A review render caught exactly that. Minimum range now grows with the object's own
    half-length, so every vehicle's near edge stays in front of the radar."""
    from e2e.ml.rt_scenes import build_rt_tier_scenario, _footprint_radius

    closest = float("inf")
    for tier in ("D1", "D2", "D3"):
        for frame in range(30):
            for obj in build_rt_tier_scenario(tier, frame_idx=frame, seed=3,
                                              use_local_assets=True).objects:
                if obj.object_class != "vehicle" or obj.name.startswith("sphere"):
                    continue
                r = (obj.position[0] ** 2 + obj.position[1] ** 2) ** 0.5
                closest = min(closest, r - _footprint_radius("vehicle", obj.asset))

    assert closest > 1.0, f"a vehicle's near edge comes within {closest:.2f} m of the radar"


def test_local_assets_are_classified_by_their_real_class():
    """A local asset must not be misread as a car.

    The class lookup consulted only the DOWNLOADED pools, and local assets join a pool
    only when use_local_assets is set — so every local asset fell through to "car". That
    gave the 16 m tractor-trailer a 4.4 m car footprint, letting it overlap its
    neighbours and defeating the separation check that depends on this function. It also
    skewed the vehicle mix: trucks drew at 5.5% against a 12% target.
    """
    from e2e.ml.rt_scenes import _asset_vehicle_class, _footprint

    assert _asset_vehicle_class("local_tractor_trailer") == "truck"
    assert _footprint("vehicle", "local_tractor_trailer")[0] > 10.0
    assert _asset_vehicle_class("local_mustang") == "car"


def test_vehicle_class_mix_matches_its_weights():
    """The drawn mix must follow VEHICLE_CLASS_WEIGHTS. A misclassification upstream
    shows up here as a deficit, which is how the tractor-trailer bug surfaced."""
    import collections
    import numpy as np
    from e2e.ml.rt_scenes import (VEHICLE_CLASS_WEIGHTS, _asset_vehicle_class,
                                  _draw_vehicle_asset)

    rng = np.random.default_rng(0)
    counts = collections.Counter(
        _asset_vehicle_class(_draw_vehicle_asset(rng, True)) for _ in range(4000)
    )
    total = sum(counts.values())
    for cls, want in VEHICLE_CLASS_WEIGHTS.items():
        got = counts[cls] / total
        assert abs(got - want) < 0.03, f"{cls}: drew {got:.1%}, weights say {want:.0%}"


def test_kenney_fleet_is_registered_in_the_default_pool():
    """The Kenney fleet must be reachable through the normal (use_local_assets=False)
    draw path -- it's registered via DOWNLOADED_ASSET_SPECS, same as every other
    downloaded mesh, not through the separate (workstation-only) LOCAL_ASSET_SPECS."""
    from e2e.ml.assets import DOWNLOADED_ASSET_SPECS

    kenney_names = {n for n, s in DOWNLOADED_ASSET_SPECS.items() if n.startswith("kn_")}
    assert len(kenney_names) == 15
    assert kenney_names <= _DEFAULT_VEHICLE_POOL

    car_pool = set(VEHICLE_CLASS_POOLS["car"])
    truck_pool = set(VEHICLE_CLASS_POOLS["truck"])
    assert "kn_sedan" in car_pool
    assert "kn_delivery" in truck_pool


def test_bus_and_trolley_pools_gain_the_owner_approved_assets():
    """The Kenney fleet itself is car/truck only -- bus/trolley box-body stand-ins were
    drafted then explicitly DEFERRED by the project owner, pending a parallel search
    for real freely-licensed bus/trolley meshes. That search has SINCE concluded (this
    replaces the older pin of "unchanged" -- i.e. still just dl_school_bus/dl_trolley
    -- which was only ever true for the pre-approval state): each pool now holds its
    original (UNKNOWN-license) mesh PLUS one new OWNER-APPROVED asset."""
    assert set(VEHICLE_CLASS_POOLS["bus"]) == {"dl_school_bus", "dl_bus_ajanhallinta"}
    assert set(VEHICLE_CLASS_POOLS["trolley"]) == {"dl_trolley", "dl_tram_google"}


def test_draw_vehicle_asset_survives_an_empty_class_pool():
    """REGRESSION: every non-car class has exactly ONE mesh, so removing a single
    UNKNOWN-licence asset -- the project's own stated plan -- empties that pool and used
    to crash inside numpy with `ValueError: high <= 0` and nothing naming the cause.
    Generation must degrade the vehicle MIX, loudly, not break."""
    import warnings

    import numpy as np

    from e2e.ml import rt_scenes

    original = dict(rt_scenes.VEHICLE_CLASS_POOLS)
    try:
        rt_scenes.VEHICLE_CLASS_POOLS["truck"] = ()
        rng = np.random.default_rng(0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            names = [rt_scenes._draw_vehicle_asset(rng, False) for _ in range(400)]
        assert all(isinstance(n, str) and n for n in names)
        assert any("empty asset pool" in str(w.message) for w in caught), \
            "falling back silently would hide a changed vehicle mix"
    finally:
        rt_scenes.VEHICLE_CLASS_POOLS.clear()
        rt_scenes.VEHICLE_CLASS_POOLS.update(original)
