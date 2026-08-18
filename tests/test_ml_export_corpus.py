"""
Tests for `e2e.ml.export_corpus`: packaging a manifest-based radar corpus into a
self-contained bundle (frames + manifest + DATA_FORMAT.md + ATTRIBUTION.md +
MANIFEST.sha256 + README.md).

Builds its own tiny synthetic corpus directly with `e2e.ml.storage.write_sample_npz`
(the same writer `e2e.ml.dataset`/`e2e.ml.chain_generate` use) rather than importing
`e2e.ml.dataset` -- this module doesn't need scenes/labels/scatterers to exist, only
the on-disk npz+manifest contract `export_corpus.py` packages.
"""
import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from e2e.ml import export_corpus, storage


# --------------------------------------------------------------------------------
# Tiny synthetic corpus fixture
# --------------------------------------------------------------------------------
def _quantized_adc(shape=(2, 3, 4), bits=10, fs=1.5, seed=0):
    """A uniformly-quantized complex64 cube -- verifies as CODEC_INT16 (see
    e2e.ml.storage's docstring / test_ml_storage.py's identical helper)."""
    rng = np.random.default_rng(seed)
    top = 2 ** (bits - 1) - 1
    lsb = fs / (2 ** (bits - 1))
    re_code = rng.integers(-top - 1, top + 1, size=shape)
    im_code = rng.integers(-top - 1, top + 1, size=shape)
    array = (re_code.astype(np.float32) + 1j * im_code.astype(np.float32)) * np.float32(lsb)
    return array.astype(np.complex64), fs


def _continuous_adc(shape=(2, 3, 4), seed=1):
    """Continuous (non-uniformly-quantized) complex64 cube -- falls back to
    CODEC_RAW (never verifies as an exact int16 grid)."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)


_CFG = {
    "name": "test_tiny", "f0_hz": 77e9, "bandwidth_hz": 2e9, "n_tx": 1, "n_rx": 2,
    "n_chirps": 3, "n_samples": 4, "fs_hz": 25e6, "chirp_period_s": 25e-6,
    "mimo": "single", "frame_rate_hz": 10.0,
}
_GRID = {"n_range": 2, "n_azimuth": 3, "max_range_m": 12.0}


def _frame_meta(frame_idx, scene_index, targets):
    return {
        "frame_idx": frame_idx, "snr_db": 30.0, "seed": scene_index, "config": _CFG["name"],
        "mimo": _CFG["mimo"], "pose_position": [0.0, 0.0, 0.0], "pose_boresight": [1.0, 0.0, 0.0],
        "target_extras": [{"rcs_dbsm": 10.0, "velocity_mps": [1.0, 0.0, 0.0]} for _ in targets],
        "scene_index": scene_index, "targets": targets,
        "scene": {"n_vehicles": 1, "n_pedestrians": 0, "n_clutter": 0, "clutter": [],
                  "placement_attempts": 1},
    }


@pytest.fixture
def tiny_corpus(tmp_path):
    """Writes a 2-frame manifest_version-2 corpus (one CODEC_INT16 frame, one
    CODEC_RAW frame -- exercising both branches of the documented codec) under
    `tmp_path / "src" / "test_tiny_D0"`. Returns the manifest path."""
    src_dir = tmp_path / "src" / "test_tiny_D0"
    src_dir.mkdir(parents=True)

    labels0 = np.zeros((3, _GRID["n_range"], _GRID["n_azimuth"]), dtype=np.float32)
    adc0, fs0 = _quantized_adc(seed=0)
    storage.write_sample_npz(
        src_dir / "frame_00000.npz", {"adc": adc0, "labels": labels0},
        _frame_meta(0, 0, [(5.0, 0.1, "vehicle", 4.5)]),
        payload_key="adc", full_scale=fs0,
    )

    labels1 = np.zeros((3, _GRID["n_range"], _GRID["n_azimuth"]), dtype=np.float32)
    adc1 = _continuous_adc(seed=1)
    storage.write_sample_npz(
        src_dir / "frame_00001.npz", {"adc": adc1, "labels": labels1},
        _frame_meta(0, 1, [(6.0, -0.2, "pedestrian")]),  # 3-tuple: back-compat schema
        payload_key="adc",  # no full_scale hint -> continuous data falls back to CODEC_RAW
    )

    manifest = {
        "manifest_version": 2, "config": _CFG, "tier": "D0", "grid": _GRID,
        "snr_db": 30.0, "seed": 0, "frames_per_scene": 1,
        "label_classes": ["vehicle", "pedestrian"],
        "files": {"train": ["frame_00000.npz"], "val": ["frame_00001.npz"], "test": []},
        "sequences": [["frame_00000.npz"], ["frame_00001.npz"]],
    }
    manifest_path = src_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


# --------------------------------------------------------------------------------
# Bundle contents
# --------------------------------------------------------------------------------
def test_export_writes_every_expected_file(tiny_corpus, tmp_path):
    out_dir = tmp_path / "bundle"
    result = export_corpus.export_corpus(tiny_corpus, out_dir)
    assert result == out_dir

    expected = [
        "README.md", "DATA_FORMAT.md", "ATTRIBUTION.md", "MANIFEST.sha256",
        "data/manifest.json", "data/frame_00000.npz", "data/frame_00001.npz",
    ]
    for rel in expected:
        p = out_dir / rel
        assert p.is_file(), f"missing {rel}"
        assert p.stat().st_size > 0, f"empty {rel}"

    # manifest.json in the bundle is byte-identical to the source (copy, not re-encode).
    assert (out_dir / "data" / "manifest.json").read_bytes() == tiny_corpus.read_bytes()

    readme = (out_dir / "README.md").read_text()
    assert "python -m e2e.ml.dataset" in readme
    assert "python -m e2e.ml.train" in readme
    assert "--eval-only" in readme

    attribution = (out_dir / "ATTRIBUTION.md").read_text()
    assert "SSMRadNet" in attribution and "RADIal" in attribution
    assert "Kenney" in attribution and "CC0" in attribution
    assert "MIT License" in attribution


def test_export_copies_frame_bytes_unmodified(tiny_corpus, tmp_path):
    """'copy or hardlink; do not re-encode' -- bundled npz bytes match the source
    exactly, not just their decoded contents."""
    out_dir = tmp_path / "bundle"
    export_corpus.export_corpus(tiny_corpus, out_dir)
    src_dir = tiny_corpus.parent
    for fname in ("frame_00000.npz", "frame_00001.npz"):
        assert (out_dir / "data" / fname).read_bytes() == (src_dir / fname).read_bytes()


def test_export_link_mode_still_produces_readable_files(tiny_corpus, tmp_path):
    out_dir = tmp_path / "bundle_linked"
    export_corpus.export_corpus(tiny_corpus, out_dir, link=True)
    with np.load(out_dir / "data" / "frame_00000.npz") as data:
        assert "adc_code_re" in data  # frame_00000 is the CODEC_INT16 one


def test_export_raises_on_missing_frame_file(tiny_corpus, tmp_path):
    (tiny_corpus.parent / "frame_00001.npz").unlink()
    with pytest.raises(FileNotFoundError):
        export_corpus.export_corpus(tiny_corpus, tmp_path / "bundle")


# --------------------------------------------------------------------------------
# Checksum manifest
# --------------------------------------------------------------------------------
def test_checksum_manifest_verifies_every_file(tiny_corpus, tmp_path):
    out_dir = tmp_path / "bundle"
    export_corpus.export_corpus(tiny_corpus, out_dir)

    lines = (out_dir / "MANIFEST.sha256").read_text().strip().splitlines()
    assert len(lines) >= 6  # README/DATA_FORMAT/ATTRIBUTION + manifest.json + 2 frames
    checked = set()
    for line in lines:
        digest, rel = line.split("  ", 1)
        actual = hashlib.sha256((out_dir / rel).read_bytes()).hexdigest()
        assert actual == digest, f"checksum mismatch for {rel}"
        checked.add(rel)

    # MANIFEST.sha256 itself is not (and cannot meaningfully be) listed in itself.
    assert "MANIFEST.sha256" not in checked
    assert "data/manifest.json" in checked
    assert "data/frame_00000.npz" in checked
    assert "data/frame_00001.npz" in checked


def test_checksum_manifest_has_lf_line_endings(tiny_corpus, tmp_path):
    """`sha256sum -c` (a POSIX tool) misparses a CRLF-terminated filename -- must stay
    LF-only even when the bundle is built on Windows (see _write_checksum_manifest)."""
    out_dir = tmp_path / "bundle"
    export_corpus.export_corpus(tiny_corpus, out_dir)
    raw = (out_dir / "MANIFEST.sha256").read_bytes()
    assert b"\r" not in raw


def test_checksum_manifest_detects_tampering(tiny_corpus, tmp_path):
    out_dir = tmp_path / "bundle"
    export_corpus.export_corpus(tiny_corpus, out_dir)
    (out_dir / "data" / "frame_00000.npz").write_bytes(b"tampered")

    lines = (out_dir / "MANIFEST.sha256").read_text().strip().splitlines()
    mismatches = 0
    for line in lines:
        digest, rel = line.split("  ", 1)
        actual = hashlib.sha256((out_dir / rel).read_bytes()).hexdigest()
        if actual != digest:
            mismatches += 1
    assert mismatches == 1


# --------------------------------------------------------------------------------
# --dry-run
# --------------------------------------------------------------------------------
def test_dry_run_writes_nothing(tiny_corpus, tmp_path):
    out_dir = tmp_path / "bundle_dry"
    rc = export_corpus.main(["--manifest", str(tiny_corpus), "--out", str(out_dir), "--dry-run"])
    assert rc == 0
    assert not out_dir.exists()


def test_dry_run_cli_subprocess_prints_plan_and_writes_nothing(tiny_corpus, tmp_path):
    """Exercises the real `python -m e2e.ml.export_corpus --dry-run` entry point."""
    out_dir = tmp_path / "bundle_dry_subproc"
    proc = subprocess.run(
        [sys.executable, "-m", "e2e.ml.export_corpus",
         "--manifest", str(tiny_corpus), "--out", str(out_dir), "--dry-run"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "NOT written -- dry-run" in proc.stdout
    assert not out_dir.exists()


def test_main_without_dry_run_writes_the_bundle(tiny_corpus, tmp_path):
    out_dir = tmp_path / "bundle_cli"
    rc = export_corpus.main(["--manifest", str(tiny_corpus), "--out", str(out_dir)])
    assert rc == 0
    assert (out_dir / "README.md").is_file()
    assert (out_dir / "data" / "manifest.json").is_file()


# --------------------------------------------------------------------------------
# DATA_FORMAT.md decode round trip -- the doc is executable, not just descriptive.
#
# This re-implements ONLY what DATA_FORMAT.md's "ADC codec" section documents
# (numpy only, no e2e.ml.storage import for the actual dequantization step) and
# checks it reproduces exactly what `e2e.ml.storage.read_payload` (the real reader)
# returns -- for BOTH codecs the fixture exercises. If the codec ever changes, this
# test (not just storage's own tests) will catch a DATA_FORMAT.md that drifted.
# --------------------------------------------------------------------------------
def _decode_adc_per_data_format_md(npz_path):
    with np.load(npz_path) as data:
        meta = json.loads(str(data["meta"].item()))
        codec = meta.get("codec", "raw")
        if codec == "raw":
            return np.asarray(data["adc"])
        assert codec == "int16"
        scale = meta["codec_meta"]["scale"]
        dtype = np.dtype(meta["codec_meta"]["dtype"])
        # Exact expression from DATA_FORMAT.md's "ADC codec" section.
        re = data["adc_code_re"].astype(np.float32) * np.float32(scale)
        im = data["adc_code_im"].astype(np.float32) * np.float32(scale)
        return (re + 1j * im).astype(dtype)


def test_data_format_md_dequantization_matches_storage_reader_int16(tiny_corpus):
    src_dir = tiny_corpus.parent
    path = src_dir / "frame_00000.npz"  # the CODEC_INT16 fixture frame
    with np.load(path) as data:
        meta = json.loads(str(data["meta"].item()))
        assert meta["codec"] == "int16"  # sanity: fixture actually exercises this branch
        expected = storage.read_payload(data, meta, "adc")

    decoded = _decode_adc_per_data_format_md(path)
    assert decoded.dtype == expected.dtype
    assert np.array_equal(decoded, expected)  # exact, not approximate (lossless codec)


def test_data_format_md_dequantization_matches_storage_reader_raw(tiny_corpus):
    src_dir = tiny_corpus.parent
    path = src_dir / "frame_00001.npz"  # the CODEC_RAW fixture frame
    with np.load(path) as data:
        meta = json.loads(str(data["meta"].item()))
        assert meta["codec"] == "raw"
        expected = storage.read_payload(data, meta, "adc")

    decoded = _decode_adc_per_data_format_md(path)
    assert np.array_equal(decoded, expected)


def test_data_format_md_documents_which_key_signals_which_codec(tiny_corpus):
    """DATA_FORMAT.md's npz-keys table: `adc` (raw) vs `adc_code_re`/`adc_code_im`
    (int16) are mutually exclusive, and codec is derivable from key presence alone."""
    src_dir = tiny_corpus.parent
    with np.load(src_dir / "frame_00000.npz") as data:
        assert "adc_code_re" in data and "adc_code_im" in data and "adc" not in data
    with np.load(src_dir / "frame_00001.npz") as data:
        assert "adc" in data and "adc_code_re" not in data


# --------------------------------------------------------------------------------
# Target tuple schema (3-tuple vs 4-tuple), as documented in DATA_FORMAT.md.
# --------------------------------------------------------------------------------
def test_target_tuple_schema_length_distinguishes_3_vs_4_tuple(tiny_corpus):
    src_dir = tiny_corpus.parent
    with np.load(src_dir / "frame_00000.npz") as data:
        meta = json.loads(str(data["meta"].item()))
    targets = meta["targets"]
    assert len(targets[0]) == 4  # (centre_range_m, sin_azimuth, class, surface_range_m)
    centre_range_m, sin_azimuth, object_class, surface_range_m = targets[0]
    assert object_class == "vehicle"

    with np.load(src_dir / "frame_00001.npz") as data:
        meta = json.loads(str(data["meta"].item()))
    targets = meta["targets"]
    assert len(targets[0]) == 3  # older back-compat schema, no surface range
    centre_range_m, sin_azimuth, object_class = targets[0]
    assert object_class == "pedestrian"


# --------------------------------------------------------------------------------
# DATA_FORMAT.md exists at the location export_corpus.py copies from, and is
# non-trivial (guards against an accidental empty-file regression).
# --------------------------------------------------------------------------------
def test_data_format_source_file_present_and_substantial():
    assert export_corpus.DATA_FORMAT_SRC.is_file()
    text = export_corpus.DATA_FORMAT_SRC.read_text()
    assert len(text) > 2000
    for needle in ("codec_meta", "manifest_version", "sin_azimuth", "surface_range_m"):
        assert needle in text
