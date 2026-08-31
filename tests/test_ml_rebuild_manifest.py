"""Tests for `e2e.ml.rebuild_manifest` -- recovering a corpus whose run died mid-flight.

The failure this tool exists for is expensive (hours of GPU time stranded behind a missing
manifest) and the ways it can go WRONG are silent, so every refusal below is asserted as
carefully as the success path.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from e2e.ml.rebuild_manifest import group_frames, rebuild
from e2e.radar_config import PRESETS


def _touch_corpus(root, n_scenes: int, frames_per_scene: int = 1, tag: str = "sample"):
    """A directory of correctly-named empty frames. Contents are irrelevant here -- this
    module only ever reads FILENAMES, never frame payloads."""
    d = root / "b1_fake_v9" / "benchmark_v1_D2"
    d.mkdir(parents=True)
    for s in range(n_scenes):
        for f in range(frames_per_scene):
            (d / f"{tag}_scene{s:05d}_frame_{f:05d}.npz").write_bytes(b"")
    return d


def test_groups_frames_by_scene_in_frame_order(tmp_path):
    d = _touch_corpus(tmp_path, n_scenes=3, frames_per_scene=2)
    seqs = group_frames(d)
    assert len(seqs) == 3 and all(len(s) == 2 for s in seqs)
    assert seqs[0] == ["sample_scene00000_frame_00000.npz",
                       "sample_scene00000_frame_00001.npz"]


def test_rebuilt_manifest_records_the_two_component_salt(tmp_path):
    """THE bug this tool is most likely to introduce, so it is pinned first.

    `chain_generate` salts scene sampling with `parent/leaf`, but `write_manifest`'s own
    default is the leaf alone. A manifest carrying the leaf-only tag describes a corpus
    whose scenes were never drawn with that salt -- `export_ssm` would reconstruct
    different scenes and its ground-truth check would fail, or worse, quietly pass on a
    corpus where the geometry happens not to diverge.
    """
    d = _touch_corpus(tmp_path, n_scenes=4)
    out = rebuild(d, "benchmark_v1", "D2", seed=123, write=True)
    manifest = json.loads((d / "manifest.json").read_text())
    assert manifest["corpus_tag"] == "b1_fake_v9/benchmark_v1_D2"
    assert out["corpus_tag"] == manifest["corpus_tag"]


def test_rebuilt_manifest_matches_the_generators_own_schema(tmp_path):
    """It must be indistinguishable from a normally-written manifest, not a lookalike."""
    d = _touch_corpus(tmp_path, n_scenes=10)
    rebuild(d, "benchmark_v1", "D2", seed=7, write=True)
    m = json.loads((d / "manifest.json").read_text())
    assert m["manifest_version"] == 2
    assert m["config"]["name"] == "benchmark_v1"
    assert m["tier"] == "D2" and m["seed"] == 7
    assert sum(len(v) for v in m["files"].values()) == 10
    assert len(m["sequences"]) == 10
    # The absolute-scale constant must be present, or a consumer silently falls back to
    # 1.0 and trains on an unnormalised cube (F63).
    from e2e.ml.dataset import _input_scale
    assert m["input_scale"] == pytest.approx(_input_scale(PRESETS["benchmark_v1"]))


def test_refuses_a_non_contiguous_scene_range(tmp_path):
    """A hole means a scene index no longer identifies the scene the salt drew."""
    d = _touch_corpus(tmp_path, n_scenes=5)
    (d / "sample_scene00002_frame_00000.npz").unlink()
    with pytest.raises(ValueError, match="not contiguous"):
        group_frames(d)


def test_refuses_a_torn_final_scene(tmp_path):
    """An interrupted run can leave some frames of its last scene. Accepting that would
    let the SCENE-level split straddle a partial motion sequence."""
    d = _touch_corpus(tmp_path, n_scenes=4, frames_per_scene=3)
    (d / "sample_scene00003_frame_00002.npz").unlink()
    with pytest.raises(ValueError, match="partial frame set"):
        group_frames(d)


def test_refuses_to_overwrite_an_existing_manifest(tmp_path):
    d = _touch_corpus(tmp_path, n_scenes=2)
    rebuild(d, "benchmark_v1", "D2", seed=1, write=True)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        rebuild(d, "benchmark_v1", "D2", seed=1, write=True)


def test_expect_commit_mismatch_is_loud(tmp_path):
    """Provenance must fail hard, not plausibly. `generator_git_commit` is recorded as
    HEAD *now*, which is only the truth if HEAD has not moved since the run."""
    d = _touch_corpus(tmp_path, n_scenes=2)
    with pytest.raises(ValueError, match="does not match the expected"):
        rebuild(d, "benchmark_v1", "D2", seed=1, expect_commit="0000000", write=True)


def test_dry_run_writes_nothing(tmp_path):
    d = _touch_corpus(tmp_path, n_scenes=3)
    out = rebuild(d, "benchmark_v1", "D2", seed=1)
    assert out["written"] is False
    assert not (d / "manifest.json").exists()


def test_rejects_foreign_filenames(tmp_path):
    d = tmp_path / "x" / "benchmark_v1_D2"
    d.mkdir(parents=True)
    (d / "frame_00000.npz").write_bytes(b"")   # the analytic generator's naming
    with pytest.raises(ValueError, match="does not match"):
        group_frames(d)
