"""`e2e.ml.assets`: extraction/decimation/normalization of the downloaded vehicle
meshes (campaign R3).

Two halves, same split as `tests/test_ml_rt_meshes.py`:

* Ungated (default `pytest`, no Sionna): the asset registry itself (pure dataclasses),
  the mesh-processing primitives (OBJ group filtering, STL reading, vertex-clustering
  decimation, axis-permutation/scale normalization) exercised on small SYNTHETIC meshes
  built in-test (no dependency on this workstation's downloaded archives -- CI has
  neither the archives nor `7z`, and must stay green), and graceful degradation when the
  asset cache/downloads directory is absent.
* Gated (`@pytest.mark.sionna`, RUN_SIONNA=1): the REAL downloaded meshes (this
  workstation's cache) load into a Sionna scene, land in their class's physical size
  range, and are under the decimation triangle budget; a D1-style scene built from one
  loads and solves to a nonzero monostatic return.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")  # e2e.ml.rt_gen (imported transitively) needs it

from e2e.ml.assets import (ARCHIVE_SPECS, DECIMATE_MAX_TRIS, DOWNLOADED_ASSET_SPECS,
                          DOWNLOADED_BUS_ASSET_NAMES, DOWNLOADED_CAR_ASSET_NAMES,
                          DOWNLOADED_TROLLEY_ASSET_NAMES, DOWNLOADED_TRUCK_ASSET_NAMES,
                          _read_obj_np, _read_stl_np, _vertex_cluster_pass,
                          decimate_to_budget, normalize_mesh, process_all, process_asset)


@pytest.fixture(scope="session")
def sionna_rt():
    return pytest.importorskip("sionna.rt")


# --------------------------------------------------------------------------------
# Asset registry (ungated -- pure dataclasses/dicts)
# --------------------------------------------------------------------------------
def test_every_class_is_populated():
    for names in (DOWNLOADED_CAR_ASSET_NAMES, DOWNLOADED_TRUCK_ASSET_NAMES,
                 DOWNLOADED_BUS_ASSET_NAMES, DOWNLOADED_TROLLEY_ASSET_NAMES):
        assert len(names) >= 1


def test_car_class_has_more_than_one_distinct_mesh():
    """The whole point of this campaign: a car-class draw must be able to land on more
    than one distinct geometry (delorean vs. audi_r8, on top of the one Sionna
    representative kept in e2e.ml.rt_gen)."""
    assert len(DOWNLOADED_CAR_ASSET_NAMES) >= 2


def test_bus_is_its_own_class_not_lumped_with_truck():
    for name in DOWNLOADED_BUS_ASSET_NAMES:
        assert DOWNLOADED_ASSET_SPECS[name].vehicle_class == "bus"
    for name in DOWNLOADED_TRUCK_ASSET_NAMES:
        assert DOWNLOADED_ASSET_SPECS[name].vehicle_class == "truck"
    assert set(DOWNLOADED_BUS_ASSET_NAMES).isdisjoint(DOWNLOADED_TRUCK_ASSET_NAMES)


def test_school_bus_replaces_the_max_bus_which_is_not_registered():
    """The Mercedes O403 (.max, needs 3ds Max) was dropped outright, superseded by the
    Type B school bus -- neither the raw archive key nor any asset spec should
    reference it."""
    assert "dl_school_bus" in DOWNLOADED_ASSET_SPECS
    assert DOWNLOADED_ASSET_SPECS["dl_school_bus"].vehicle_class == "bus"
    for key, spec in ARCHIVE_SPECS.items():
        assert "o403" not in spec.archive_filename.lower()
        assert "o403" not in key.lower()
    for name, spec in DOWNLOADED_ASSET_SPECS.items():
        assert not spec.filename.lower().endswith(".max")


def test_real_length_ranges_match_the_class_brief():
    for name, spec in DOWNLOADED_ASSET_SPECS.items():
        lo, hi = spec.real_length_range_m
        assert lo < hi
        if spec.vehicle_class == "car":
            assert 4.0 <= lo and hi <= 5.2
        elif spec.vehicle_class == "truck":
            assert 7.0 <= lo and hi <= 17.0
        elif spec.vehicle_class == "bus":
            assert 6.0 <= lo and hi <= 10.0
        elif spec.vehicle_class == "trolley":
            assert 9.0 <= lo and hi <= 15.0


def test_axis_permutation_is_a_valid_permutation():
    for name, spec in DOWNLOADED_ASSET_SPECS.items():
        assert sorted(spec.axis_permutation) == [0, 1, 2], name


# --------------------------------------------------------------------------------
# Mesh I/O primitives (ungated -- synthetic tiny meshes, no downloaded archives needed)
# --------------------------------------------------------------------------------
def _write_ascii_obj(path, groups):
    """`groups`: list of (name, verts, faces) with faces as 0-based LOCAL indices into
    that group's own `verts` -- writes a multi-object OBJ with global (cumulative)
    indices, matching how real exporters lay these files out."""
    with open(path, "w") as f:
        base = 0
        for name, verts, faces in groups:
            f.write(f"o {name}\n")
            for v in verts:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for a, b, c in faces:
                f.write(f"f {a + base + 1} {b + base + 1} {c + base + 1}\n")
            base += len(verts)


def test_read_obj_np_excludes_named_groups(tmp_path):
    path = tmp_path / "two_group.obj"
    keep_verts = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    keep_faces = [(0, 1, 2)]
    drop_verts = [(10, 10, 10), (11, 10, 10), (10, 11, 10)]
    drop_faces = [(0, 1, 2)]
    _write_ascii_obj(path, [("Body", keep_verts, keep_faces),
                            ("Ground_Plane", drop_verts, drop_faces)])

    verts, faces = _read_obj_np(str(path), exclude_name_substrings=("Ground_Plane",))
    assert verts.shape == (3, 3)
    assert faces.shape == (1, 3)
    np.testing.assert_allclose(verts, np.asarray(keep_verts, dtype=np.float64))
    assert faces.max() < verts.shape[0]  # reindexed, no dangling reference


def test_read_obj_np_keeps_everything_with_no_exclusions(tmp_path):
    path = tmp_path / "one_group.obj"
    verts = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
    faces = [(0, 1, 2), (1, 3, 2)]
    _write_ascii_obj(path, [("Body", verts, faces)])

    v, f = _read_obj_np(str(path))
    assert v.shape == (4, 3)
    assert f.shape == (2, 3)


def test_read_stl_np_binary_roundtrip(tmp_path):
    import struct

    path = tmp_path / "tiny.stl"
    tris = [
        ((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        ((0, 0, 1), (1, 0, 1), (0, 1, 1)),
    ]
    with open(path, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", len(tris)))
        for tri in tris:
            f.write(struct.pack("<3f", 0.0, 0.0, 1.0))  # normal (unused)
            for v in tri:
                f.write(struct.pack("<3f", *v))
            f.write(struct.pack("<H", 0))

    verts, faces = _read_stl_np(str(path))
    assert verts.shape == (6, 3)
    assert faces.shape == (2, 3)
    np.testing.assert_allclose(verts[3:6], np.asarray(tris[1], dtype=np.float64))


# --------------------------------------------------------------------------------
# Decimation (ungated -- synthetic dense mesh, no downloaded archive needed)
# --------------------------------------------------------------------------------
def _synthetic_box_mesh(nx=40, ny=40, length=(4.0, 2.0, 1.5)):
    """A dense (over-tessellated) axis-aligned box surface: `nx*ny` grid on the top
    face alone gives thousands of triangles from a trivially simple, exactly-known
    bbox -- enough to exercise decimation without a real downloaded mesh."""
    L, W, H = length
    xs = np.linspace(0, L, nx)
    ys = np.linspace(0, W, ny)
    xv, yv = np.meshgrid(xs, ys, indexing="ij")
    top = np.stack([xv.ravel(), yv.ravel(), np.full(xv.size, H)], axis=1)
    bottom = np.stack([xv.ravel(), yv.ravel(), np.zeros(xv.size)], axis=1)
    verts = np.concatenate([top, bottom], axis=0)

    faces = []
    def idx(i, j, offset):
        return offset + i * ny + j
    for surf_offset in (0, nx * ny):
        for i in range(nx - 1):
            for j in range(ny - 1):
                a, b = idx(i, j, surf_offset), idx(i + 1, j, surf_offset)
                c, d = idx(i + 1, j + 1, surf_offset), idx(i, j + 1, surf_offset)
                faces.append((a, b, c))
                faces.append((a, c, d))
    return verts, np.asarray(faces, dtype=np.int64)


def test_decimate_to_budget_respects_the_cap():
    verts, faces = _synthetic_box_mesh(nx=100, ny=100)
    assert faces.shape[0] > DECIMATE_MAX_TRIS  # the synthetic mesh really is over budget

    dec_verts, dec_faces = decimate_to_budget(verts, faces, target_tris=1000, max_tris=2000)
    assert dec_faces.shape[0] <= 2000
    assert dec_faces.shape[0] > 0


def test_decimate_to_budget_preserves_bbox_within_one_cell():
    verts, faces = _synthetic_box_mesh(nx=50, ny=50, length=(4.0, 2.0, 1.5))
    orig_extent = verts.max(axis=0) - verts.min(axis=0)

    dec_verts, dec_faces = decimate_to_budget(verts, faces, target_tris=800, max_tris=1500)
    dec_extent = dec_verts.max(axis=0) - dec_verts.min(axis=0)
    # Vertex clustering can only shrink the bbox (corner clusters average inward, never
    # extrapolate outward) -- and only by up to ~one grid cell; for a mesh this dense
    # relative to its own bbox that is a small fraction of each dimension.
    assert np.all(dec_extent <= orig_extent + 1e-9)
    assert np.all(dec_extent >= orig_extent * 0.85)


def test_decimate_to_budget_is_a_noop_under_budget():
    verts, faces = _synthetic_box_mesh(nx=5, ny=5)
    assert faces.shape[0] <= DECIMATE_MAX_TRIS
    dec_verts, dec_faces = decimate_to_budget(verts, faces)
    assert dec_faces.shape[0] == faces.shape[0]
    np.testing.assert_array_equal(dec_verts, verts)


def test_vertex_cluster_pass_collapses_a_degenerate_triangle():
    # Two points closer together than the cell size collapse to the same cluster;
    # a triangle referencing only those two (plus itself) degenerates and is dropped.
    verts = np.array([[0.0, 0.0, 0.0], [0.001, 0.001, 0.0], [5.0, 5.0, 0.0]])
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    new_verts, new_faces = _vertex_cluster_pass(verts, faces, cell_size=1.0,
                                                bbox_min=verts.min(axis=0))
    # 0 and 1 collapse into the same cluster as each other but not with 2 -- 2 distinct
    # clusters, and the one triangle referencing all three degenerates (two indices equal).
    assert new_verts.shape[0] == 2
    assert new_faces.shape[0] == 0


# --------------------------------------------------------------------------------
# Normalization (ungated -- synthetic verts + a hand-built spec)
# --------------------------------------------------------------------------------
def test_normalize_mesh_permutes_scales_and_rebases():
    from e2e.ml.assets import DownloadedAssetSpec

    # Raw axes: "length" lives on raw z, "width" on raw x, "height" on raw y (mirrors
    # the delorean/audi_r8/truck_daf finding) -- permutation (2, 0, 1).
    raw = np.array([
        [1.0, 2.0, 0.0],
        [1.0, 2.0, 10.0],   # z=10 -> the raw "length" extent
        [3.0, 2.0, 5.0],    # x=3  -> the raw "width" extent (relative to min x=1)
        [1.0, 6.0, 5.0],    # y=6  -> the raw "height" extent (relative to min y=2)
    ])
    spec = DownloadedAssetSpec(
        name="synthetic", vehicle_class="car", archive_key="none", filename="none",
        axis_permutation=(2, 0, 1), scale_m=0.5, real_length_range_m=(0.0, 100.0),
    )
    out, stats = normalize_mesh(raw, spec)

    assert stats["length_m"] == pytest.approx(10.0 * 0.5)
    assert stats["width_m"] == pytest.approx(2.0 * 0.5)
    assert stats["height_m"] == pytest.approx(4.0 * 0.5)
    assert out[:, 2].min() == pytest.approx(0.0, abs=1e-9)  # ground-rest rebase


# --------------------------------------------------------------------------------
# Graceful degradation (ungated -- must hold with no cache/downloads present, i.e. CI)
# --------------------------------------------------------------------------------
def test_process_asset_returns_none_without_cache_or_downloads(tmp_path, monkeypatch):
    monkeypatch.setenv("E2E_ML_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("E2E_ML_ASSET_DOWNLOADS_DIR", str(tmp_path / "downloads"))
    import e2e.ml.assets as assets_mod
    assets_mod._process_cache.clear()

    for name in DOWNLOADED_ASSET_SPECS:
        assert process_asset(name) is None

    results = process_all()
    assert all(v is None for v in results.values())
    assets_mod._process_cache.clear()


def test_ensure_extracted_false_when_7z_or_archive_missing(tmp_path, monkeypatch):
    from e2e.ml.assets import ensure_extracted

    monkeypatch.setenv("E2E_ML_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("E2E_ML_ASSET_DOWNLOADS_DIR", str(tmp_path / "nonexistent_downloads"))
    assert ensure_extracted("delorean") is False


# --------------------------------------------------------------------------------
# Gated: the REAL downloaded meshes (this workstation's cache), via Sionna
# --------------------------------------------------------------------------------
@pytest.mark.sionna
@pytest.mark.parametrize("name", sorted(DOWNLOADED_ASSET_SPECS))
def test_downloaded_asset_loads_and_is_class_scale_and_under_budget(sionna_rt, name):
    result = process_asset(name)
    if result is None:
        pytest.skip(f"{name}: raw source not found on this machine (graceful degrade)")

    assert result.tris_after < DECIMATE_MAX_TRIS, \
        f"{name}: {result.tris_after} triangles >= the {DECIMATE_MAX_TRIS} budget"

    mesh = sionna_rt.load_mesh(result.ply_path)
    bbox = mesh.bbox()
    extents = np.asarray((bbox.max - bbox.min).numpy()).reshape(-1)
    spec = DOWNLOADED_ASSET_SPECS[name]
    lo, hi = spec.real_length_range_m
    length = float(extents[0])  # normalize_mesh puts the length axis at x
    assert lo - 0.5 <= length <= hi + 0.5, \
        f"{name}: mesh length {length:.2f} m outside class range [{lo}, {hi}] m"
    assert float(extents[2]) > 0.1, f"{name}: near-zero height -- looks flattened/on its side"


@pytest.mark.sionna
@pytest.mark.parametrize("name", sorted(DOWNLOADED_ASSET_SPECS))
def test_downloaded_asset_scene_solves_to_nonzero_monostatic_return(sionna_rt, name):
    """One real object per class, ray-traced monostatically -- regression for both the
    group-exclusion re-indexing (truck_daf/audi_r8 drop a backdrop/prop group before
    decimation; a broken remap would leave dangling/garbage face indices) and the
    axis-normalization (a mesh lying on its side or squashed flat would still "solve"
    geometrically but is the exact bug item 3 exists to catch -- this doesn't assert
    orientation directly, `test_downloaded_asset_loads_and_is_class_scale_and_under_
    budget` does, but a badly malformed mesh is more likely to break path solving too)."""
    result = process_asset(name)
    if result is None:
        pytest.skip(f"{name}: raw source not found on this machine (graceful degrade)")

    import dataclasses

    from e2e.ml.radar_config import TI_IWR1443
    from e2e.ml.rt_gen import _beat_from_paths, _solve, build_rt_scene
    from e2e.scenario import Motion, Node, NodeRole, ObjectKind, Scenario, SceneObject

    cfg = dataclasses.replace(TI_IWR1443, name="dl_asset_test", n_chirps=8, n_samples=64)
    scenario = Scenario(
        name=f"dl_asset_test_{name}", base_scene="free", num_frames=1,
        nodes=[Node(name="radar", role=NodeRole.RADAR, position=(0.0, 0.0, 1.5),
                    look_at=(1.0, 0.0, 1.5))],
        objects=[SceneObject(name="target", kind=ObjectKind.MESH, asset=name,
                             position=(12.0, 0.0, 0.5 * result.height_m), scaling=1.0,
                             material="metal", object_class="vehicle", motion=Motion())],
    )
    rt_scene = build_rt_scene(scenario, cfg, base_scene="free", frame_idx=0)
    paths = _solve(rt_scene, max_depth=2, include_leakage=False, diffuse_reflection=True,
                   specular_reflection=True, refraction=False, seed=7)
    beat = _beat_from_paths(paths, cfg, n_chirps=cfg.n_chirps)
    assert np.isfinite(beat).all()
    assert np.abs(beat).sum() > 0, f"no monostatic return from the {name} scene"
