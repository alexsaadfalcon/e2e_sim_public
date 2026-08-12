"""Tests for `e2e.ml.export_ssm`: a tiny, Sionna-free synthetic corpus mimicking the
RT-generated on-disk schema (int16-coded ADC + `meta["targets"]`, see
`e2e.ml.chain_generate`/`e2e.environment.blocks.RTEnvironmentBlock`), export N=3
samples, and check the deliverables: copied npz, labels JSON schema, int16 round-trip,
GT-mismatch detection, and a standalone (no-`e2e`-import) `loader_ssm.py`.

Mirrors `tests/test_afe_sweep.py`'s "build one tiny corpus with a temp-registered
RadarConfig" pattern, but the FIXTURE writes samples directly via
`e2e.ml.storage.write_sample_npz` (matching `SinkBlock`'s schema) instead of
`e2e.ml.dataset.generate_dataset`, since `export_ssm.reconstruct_scene` needs the RT
tier machinery (`e2e.ml.rt_scenes.build_rt_tier_scenario`) and `meta["targets"]`, which
the analytic fallback path never writes.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("e2e.ml.labels", reason="sibling shard e2e.ml.labels not present")
pytest.importorskip("e2e.ml.rt_scenes", reason="sibling shard e2e.ml.rt_scenes not present")

from e2e.ml import export_ssm, storage
from e2e.ml.dataset import write_manifest
from e2e.ml.labels import LabelGrid, encode_detection_labels, targets_in_grid
from e2e.ml.radar_config import RadarConfig
from e2e.ml.rt_scenes import build_rt_tier_scenario
from e2e.ml.scatterers import frame_scatterers, radar_pose

TIER = "D0"          # single deterministic sphere-as-"vehicle" scatterer, no Sionna needed
SEED = 9000
N = 3

_TINY_CFG = RadarConfig(
    # n_samples=256 (not 16) so max_range_m (~51 m) comfortably covers rt_scenes'
    # D0/D1 placement envelope (targets drawn in [6, 34] m) -- a smaller grid left
    # every target out of range and every sample's target list empty.
    name="test_export_ssm_tiny", f0_hz=77e9, bandwidth_hz=749.5e6, n_tx=2, n_rx=4,
    n_chirps=8, n_samples=256, fs_hz=10e6, chirp_period_s=7.6e-5, mimo="ddma",
    frame_rate_hz=10.0,
)


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)}")


@pytest.fixture(scope="module")
def tiny_rt_manifest(tmp_path_factory):
    """A tiny, Sionna-free corpus that mimics `generate_chain_corpus`'s on-disk schema:
    `sample_scene?????_frame_00000.npz` files with int16-coded ADC + `meta["targets"]`,
    scenes drawn from `build_rt_tier_scenario` (real geometry, no ray tracing)."""
    dataset_dir = tmp_path_factory.mktemp("tiny_rt_corpus")
    cfg = _TINY_CFG
    grid = LabelGrid.for_config(cfg, range_stride=4, n_azimuth=8)
    label_classes = ("vehicle", "pedestrian")

    sequences = []
    rng = np.random.default_rng(1)
    for i in range(N):
        scenario = build_rt_tier_scenario(TIER, frame_idx=i, seed=SEED, num_frames=1,
                                          use_local_assets=True)
        scats = frame_scatterers(scenario, 0, dt=1.0)
        pose = radar_pose(scenario, 0)
        labels = encode_detection_labels(grid, scats, pose, classes=label_classes)
        targets = targets_in_grid(grid, scats, pose, classes=label_classes)

        # Fake ADC: random values snapped onto an exact int16-code grid so
        # storage.encode_payload's round-trip check accepts CODEC_INT16 -- matching the
        # real chain's post-QuantizerBlock corpus, not the analytic fallback's raw floats.
        full_scale = 2.0
        scale = full_scale / 2 ** 15
        codes_re = rng.integers(-2000, 2000, size=(cfg.n_rx, cfg.n_chirps, cfg.n_samples))
        codes_im = rng.integers(-2000, 2000, size=(cfg.n_rx, cfg.n_chirps, cfg.n_samples))
        adc = (codes_re * scale + 1j * codes_im * scale).astype(np.complex64)

        fname = f"sample_scene{i:05d}_frame_00000.npz"
        meta = {
            "tag": "sample", "frame_idx": 0, "domain": "rx_time", "payload_key": "adc",
            "shape": list(adc.shape), "dtype": "complex64",
            "impairment_params": {"seed": SEED + i},
            "targets": targets,
        }
        storage.write_sample_npz(
            dataset_dir / fname, {"adc": adc, "labels": labels.cpu().numpy()}, meta,
            payload_key="adc", full_scale=full_scale, json_default=_json_default,
        )
        sequences.append([fname])

    manifest_path = write_manifest(dataset_dir, cfg, TIER, sequences, grid=grid, seed=SEED,
                                   snr_db=None, frames_per_scene=1, splits=(1.0, 0.0, 0.0),
                                   label_classes=label_classes)
    return manifest_path


# --------------------------------------------------------------------------------
# select_train_files / filename parsing
# --------------------------------------------------------------------------------
def test_select_train_files_returns_exact_manifest_slice(tiny_rt_manifest):
    with open(tiny_rt_manifest) as f:
        manifest = json.load(f)
    selected = export_ssm.select_train_files(manifest, N)
    assert selected == manifest["files"]["train"][:N]
    assert len(selected) == N


def test_select_train_files_rejects_n_too_large(tiny_rt_manifest):
    with open(tiny_rt_manifest) as f:
        manifest = json.load(f)
    with pytest.raises(ValueError):
        export_ssm.select_train_files(manifest, N + 100)


def test_parse_scene_frame():
    assert export_ssm._parse_scene_frame("sample_scene00042_frame_00007.npz") == (42, 7)
    with pytest.raises(ValueError):
        export_ssm._parse_scene_frame("not_a_match.npz")


# --------------------------------------------------------------------------------
# GT verification
# --------------------------------------------------------------------------------
def test_verify_gt_match_accepts_identical_targets():
    export_ssm.verify_gt_match([[1.0, 0.5, "vehicle"]], [(1.0, 0.5, "vehicle")], context="t")


def test_verify_gt_match_rejects_value_mismatch():
    with pytest.raises(AssertionError):
        export_ssm.verify_gt_match([[1.0, 0.5, "vehicle"]], [(1.5, 0.5, "vehicle")], context="t")


def test_verify_gt_match_rejects_count_mismatch():
    with pytest.raises(AssertionError):
        export_ssm.verify_gt_match([[1.0, 0.5, "vehicle"]], [], context="t")


def test_reconstruct_scene_gt_matches_fixture(tiny_rt_manifest):
    """The one-sample (here, every-sample) empirical check the brief calls for: the
    fixture's OWN stored `meta["targets"]` must match a fresh reconstruction."""
    dataset_dir = tiny_rt_manifest.parent
    with open(tiny_rt_manifest) as f:
        manifest = json.load(f)
    grid = LabelGrid(**manifest["grid"])
    for fname in manifest["files"]["train"]:
        scene_idx, frame_idx = export_ssm._parse_scene_frame(fname)
        _scenario, scats, pose = export_ssm.reconstruct_scene(
            TIER, SEED, scene_idx, frame_idx, 1, use_local_assets=True)
        rebuilt = targets_in_grid(grid, scats, pose, classes=("vehicle", "pedestrian"))
        with np.load(dataset_dir / fname, allow_pickle=True) as data:
            meta = json.loads(str(data["meta"].item()))
        export_ssm.verify_gt_match(meta["targets"], rebuilt, context=fname)


# --------------------------------------------------------------------------------
# Full export
# --------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def exported(tiny_rt_manifest, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("ssm_export")
    result = export_ssm.export(tiny_rt_manifest, N, out_dir, render=True, dpi=60)
    return out_dir, result


def test_export_writes_expected_files(exported):
    out_dir, result = exported
    assert result["n_samples"] == N
    assert result["gt_verified"] is True

    for i in range(N):
        assert (out_dir / "npz" / f"sample_{i:05d}.npz").is_file()
        assert (out_dir / "labels" / f"sample_{i:05d}.json").is_file()
        png = out_dir / "renders" / f"sample_{i:05d}.png"
        assert png.is_file()
        assert png.stat().st_size > 0

    assert (out_dir / "README.md").is_file()
    assert (out_dir / "manifest.json").is_file()
    assert (out_dir / "loader_ssm.py").is_file()


def test_export_npz_is_byte_identical_copy(tiny_rt_manifest, exported):
    out_dir, result = exported
    dataset_dir = tiny_rt_manifest.parent
    for i, src_name in enumerate(result["selected_files"]):
        src_bytes = (dataset_dir / src_name).read_bytes()
        dst_bytes = (out_dir / "npz" / f"sample_{i:05d}.npz").read_bytes()
        assert src_bytes == dst_bytes


def test_labels_json_schema(exported):
    out_dir, _result = exported
    with open(out_dir / "labels" / "sample_00000.json") as f:
        payload = json.load(f)
    for key in ("sample_index", "source_file", "scene_index", "frame_idx", "tier",
               "config", "impairment_params", "targets"):
        assert key in payload
    assert len(payload["targets"]) >= 1   # D0 tier: exactly one sphere-as-vehicle target
    target = payload["targets"][0]
    for key in ("class", "range_m", "sin_azimuth", "azimuth_deg", "x_m", "y_m",
               "extent_m", "rcs_dbsm", "velocity_mps", "grid_row", "grid_col"):
        assert key in target
    assert target["class"] == "vehicle"
    assert target["extent_m"] == {"length": 1.0, "width": 1.0}   # sphere footprint


def test_labels_json_geometry_is_self_consistent(exported):
    """x_m/y_m must reconstruct range_m/sin_azimuth (radar at origin, boresight +x)."""
    out_dir, _result = exported
    with open(out_dir / "labels" / "sample_00000.json") as f:
        payload = json.load(f)
    t = payload["targets"][0]
    r = (t["x_m"] ** 2 + t["y_m"] ** 2) ** 0.5
    assert r == pytest.approx(t["range_m"], abs=1e-6)
    assert t["y_m"] / r == pytest.approx(t["sin_azimuth"], abs=1e-6)


def test_top_level_manifest_records_selected_files(tiny_rt_manifest, exported):
    out_dir, result = exported
    with open(out_dir / "manifest.json") as f:
        manifest = json.load(f)
    assert manifest["n_samples"] == N
    assert [s["source_file"] for s in manifest["samples"]] == result["selected_files"]
    assert manifest["tier"] == TIER
    assert manifest["seed"] == SEED


def test_readme_mentions_int16_and_coordinate_frame(exported):
    out_dir, _result = exported
    text = (out_dir / "README.md").read_text()
    assert "int16" in text
    assert "boresight" in text.lower()
    assert "codec_meta" in text


# --------------------------------------------------------------------------------
# int16 round trip
# --------------------------------------------------------------------------------
def test_int16_roundtrip_matches_original(tiny_rt_manifest, exported):
    out_dir, result = exported
    dataset_dir = tiny_rt_manifest.parent
    src_name = result["selected_files"][0]
    with np.load(dataset_dir / src_name, allow_pickle=True) as data:
        meta = json.loads(str(data["meta"].item()))
        original = storage.read_payload(data, meta, "adc")

    with np.load(out_dir / "npz" / "sample_00000.npz", allow_pickle=True) as data:
        meta2 = json.loads(str(data["meta"].item()))
        decoded = storage.read_payload(data, meta2, "adc")

    assert meta["codec"] == "int16"          # the fixture is built to guarantee this
    np.testing.assert_array_equal(decoded, original)


# --------------------------------------------------------------------------------
# Standalone loader_ssm.py: must import with NO `e2e` package on sys.modules
# --------------------------------------------------------------------------------
def test_loader_ssm_standalone_no_e2e_import(exported):
    out_dir, _result = exported
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(out_dir)!r})
        import loader_ssm
        assert not any(m == "e2e" or m.startswith("e2e.") for m in sys.modules), (
            "loader_ssm.py must not import e2e")

        ds = loader_ssm.SSMExportDataset({str(out_dir)!r})
        assert len(ds) == {N}
        adc_cube, targets = ds[0]
        import torch
        assert adc_cube.dtype == torch.complex64
        assert adc_cube.dim() == 3
        assert isinstance(targets, list) and len(targets) >= 1
        assert targets[0]["class"] == "vehicle"
        print("OK")
    """)
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "OK" in proc.stdout


def test_loader_ssm_matches_export_npz_decode(exported):
    """Loader's decode must reproduce the same array `storage.read_payload` would."""
    out_dir, _result = exported
    sys_path_added = str(out_dir)
    import importlib.util

    spec = importlib.util.spec_from_file_location("loader_ssm_test", out_dir / "loader_ssm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ds = mod.SSMExportDataset(out_dir)
    adc_cube, _targets = ds[0]

    with np.load(out_dir / "npz" / "sample_00000.npz", allow_pickle=True) as data:
        meta = json.loads(str(data["meta"].item()))
        expected = storage.read_payload(data, meta, "adc")

    np.testing.assert_allclose(adc_cube.numpy(), expected, atol=1e-6)


# --------------------------------------------------------------------------------
# Extent lookup helper
# --------------------------------------------------------------------------------
def test_target_extent_defaults_for_unknown_class():
    class _FakeObj:
        kind = "mesh"
        object_class = "scatterer"
        asset = None

    from e2e.scenario import ObjectKind

    class _FakeMesh:
        kind = ObjectKind.MESH
        object_class = "scatterer"
        asset = None

    assert export_ssm._target_extent_m(_FakeMesh()) is None
