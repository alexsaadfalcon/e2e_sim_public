"""Tests for the corpus-identity seed salt (`e2e.ml.rt_scenes._stable_seed` /
`e2e.ml.dataset._stable_scene_seed`), the manifest provenance fields
(`write_manifest`'s `corpus_tag`/`generator_git_commit`), and the pre-ship dedup gate
(`e2e.ml.check_corpus_overlap`).

Root cause under test: `_stable_seed` used to hash only `(tier, frame_idx, seed)`, so
two corpora sharing a `(tier, seed)` drew byte-identical scenes at every `frame_idx` --
measured live on disk as `rt_ablation_txoff`'s entire val+test split sitting inside two
other corpora's train splits. `corpus_tag` closes that FOR NEW GENERATION; this file's
gate tests cover the complementary case of corpora already on disk.
"""
import dataclasses
import json
import subprocess

import numpy as np
import pytest

from e2e.ml import check_corpus_overlap as gate
from e2e.ml.rt_scenes import _stable_seed, build_rt_tier_scenario

# --------------------------------------------------------------------------------
# Seed salting (rt_scenes)
# --------------------------------------------------------------------------------
def test_stable_seed_requires_corpus_tag():
    with pytest.raises(TypeError):
        _stable_seed("D1", 0, 0)  # type: ignore[call-arg]


def test_stable_seed_changes_with_corpus_tag():
    a = _stable_seed("D1", 0, 0, "corpus-a")
    b = _stable_seed("D1", 0, 0, "corpus-b")
    assert a != b


def test_build_rt_tier_scenario_requires_corpus_tag():
    with pytest.raises(TypeError):
        build_rt_tier_scenario("D1", frame_idx=0, seed=0)  # type: ignore[call-arg]


def test_build_rt_tier_scenario_same_corpus_tag_is_reproducible():
    a = build_rt_tier_scenario("D1", corpus_tag="same", frame_idx=0, seed=0, num_frames=1)
    b = build_rt_tier_scenario("D1", corpus_tag="same", frame_idx=0, seed=0, num_frames=1)
    assert a.to_json() == b.to_json()


def test_build_rt_tier_scenario_different_corpus_tag_draws_a_different_scene():
    """The regression this whole shard exists for: two 'corpora' sharing (tier,
    frame_idx, seed) must NOT draw the same scene once corpus_tag differs."""
    a = build_rt_tier_scenario("D1", corpus_tag="corpus-one", frame_idx=0, seed=0, num_frames=1)
    b = build_rt_tier_scenario("D1", corpus_tag="corpus-two", frame_idx=0, seed=0, num_frames=1)
    assert a.to_json() != b.to_json()


# --------------------------------------------------------------------------------
# Seed salting (dataset.py's analytic-path sibling)
# --------------------------------------------------------------------------------
def test_dataset_stable_scene_seed_requires_corpus_tag():
    ml_dataset = pytest.importorskip("e2e.ml.dataset")
    with pytest.raises(TypeError):
        ml_dataset._stable_scene_seed(0, 0)  # type: ignore[call-arg]


def test_dataset_stable_scene_seed_changes_with_corpus_tag():
    ml_dataset = pytest.importorskip("e2e.ml.dataset")
    a = ml_dataset._stable_scene_seed("corpus-a", 0, 0)
    b = ml_dataset._stable_scene_seed("corpus-b", 0, 0)
    assert a != b


# --------------------------------------------------------------------------------
# Manifest provenance (write_manifest)
# --------------------------------------------------------------------------------
def test_write_manifest_records_corpus_tag_and_git_commit(tmp_path):
    ml_dataset = pytest.importorskip("e2e.ml.dataset")
    from e2e.ml.radar_config import TI_IWR1443

    cfg = dataclasses.replace(TI_IWR1443, name="test_tiny_provenance", n_chirps=4, n_samples=8)
    dataset_dir = tmp_path / "some_corpus_ti_iwr1443_D1"
    dataset_dir.mkdir()

    manifest_path = ml_dataset.write_manifest(dataset_dir, cfg, "D1", [], grid=None)
    manifest = json.loads(manifest_path.read_text())

    # Default corpus_tag is the dataset directory's own name (both real producers
    # already name that directory f"{cfg_name}_{tier}" and pass it as the seed salt).
    assert manifest["corpus_tag"] == "some_corpus_ti_iwr1443_D1"
    assert isinstance(manifest["generator_git_commit"], str) and manifest["generator_git_commit"]


def test_write_manifest_explicit_corpus_tag_overrides_directory_name(tmp_path):
    ml_dataset = pytest.importorskip("e2e.ml.dataset")
    from e2e.ml.radar_config import TI_IWR1443

    cfg = dataclasses.replace(TI_IWR1443, name="test_tiny_provenance2", n_chirps=4, n_samples=8)
    dataset_dir = tmp_path / "irrelevant_dir_name"
    dataset_dir.mkdir()

    manifest_path = ml_dataset.write_manifest(
        dataset_dir, cfg, "D1", [], grid=None, corpus_tag="explicit-tag",
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["corpus_tag"] == "explicit-tag"


def test_generator_git_commit_never_raises(monkeypatch):
    """Provenance is best-effort: git being unavailable must degrade to "unknown",
    never crash a generation run."""
    ml_dataset = pytest.importorskip("e2e.ml.dataset")

    def _boom(*a, **k):
        raise FileNotFoundError("no git on this box")

    monkeypatch.setattr(subprocess, "run", _boom)
    assert ml_dataset._generator_git_commit() == "unknown"


# --------------------------------------------------------------------------------
# check_corpus_overlap: helpers
# --------------------------------------------------------------------------------
def _write_fake_corpus(root, name, files_by_split):
    """Write a minimal corpus at `root/name`: one tiny `.npz` per frame (only a
    "meta" entry -- the gate never reads "adc"/"labels") plus a `manifest.json` with
    just enough structure (`files`) for `check_corpus_overlap` to scan.

    `files_by_split` is `{"train": [targets_list, ...], "val": [...], "test": [...]}`
    where each `targets_list` is a `meta["targets"]`-shaped list (each entry either a
    3-tuple `(range_m, sin_az, object_class)` or a 4-tuple with a trailing
    `surface_range_m`, matching the two on-disk vintages).
    """
    dataset_dir = root / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    files = {"train": [], "val": [], "test": []}
    for split, scenes in files_by_split.items():
        for i, targets in enumerate(scenes):
            fname = f"{split}_{i:03d}.npz"
            meta = {"targets": [list(t) for t in targets]}
            np.savez(dataset_dir / fname, meta=np.array(json.dumps(meta)))
            files[split].append(fname)
    with open(dataset_dir / "manifest.json", "w") as f:
        json.dump({"manifest_version": 2, "files": files}, f)
    return dataset_dir


_SCENE_A = [(15.0, 0.1, "vehicle")]
_SCENE_B = [(20.0, -0.2, "vehicle"), (10.0, 0.05, "pedestrian")]
_SCENE_C = [(28.0, 0.3, "vehicle")]
_SCENE_A_NEAR_DUP = [(15.02, 0.1001, "vehicle")]      # within rounding tolerance of A
_SCENE_A_4TUPLE = [(15.0, 0.1, "vehicle", 14.5)]      # newer vintage, same first 3 fields


# --------------------------------------------------------------------------------
# check_corpus_overlap: scene-hashing
# --------------------------------------------------------------------------------
def test_frame_scene_key_ignores_empty_targets():
    assert gate.frame_scene_key({"targets": []}) is None
    assert gate.frame_scene_key({}) is None


def test_frame_scene_key_common_fields_only_ignores_tuple_length():
    """A 3-tuple and a 4-tuple that agree on (range_m, sin_az, object_class) must hash
    the same -- the trailing surface_range_m field is a newer-vintage-only extra."""
    key3 = gate.frame_scene_key({"targets": _SCENE_A})
    key4 = gate.frame_scene_key({"targets": _SCENE_A_4TUPLE})
    assert key3 is not None
    assert key3 == key4


def test_exact_key_separates_jitter_but_near_key_does_not():
    """The two tiers must DISAGREE on sub-cm drift -- that disagreement is the design.

    Sub-cm / sub-1e-3 drift is what the same seed produces under different
    `rt_scenes.py` code versions. The EXACT tier must treat those as different scenes
    (it is reserved for bit-identical seed collisions, which is why it can hard-FAIL
    with no false positives); the NEAR tier must treat them as the same scene, because
    cross-version drift is exactly what it exists to warn about.

    If both tiers agreed, one would be redundant and the gate would be back to failing
    on coincidence: two independent single-target D0 scenes 2 cm apart land in the same
    0.1 m bin, and over 1200 scenes that is a near-certainty, not bad luck.
    """
    exact_a = gate.frame_scene_key({"targets": _SCENE_A})
    exact_dup = gate.frame_scene_key({"targets": _SCENE_A_NEAR_DUP})
    near_a = gate.frame_scene_key_near({"targets": _SCENE_A})
    near_dup = gate.frame_scene_key_near({"targets": _SCENE_A_NEAR_DUP})
    assert exact_a != exact_dup, "EXACT tier must not absorb jitter"
    assert near_a == near_dup, "NEAR tier must absorb jitter"


def test_exact_key_matches_identical_geometry():
    """The exact tier's whole claim: identical geometry hashes identically."""
    assert gate.frame_scene_key({"targets": _SCENE_A}) == gate.frame_scene_key(
        {"targets": list(_SCENE_A)})


def test_frame_scene_key_distinguishes_different_scenes():
    assert gate.frame_scene_key({"targets": _SCENE_A}) != gate.frame_scene_key({"targets": _SCENE_B})


def test_frame_scene_key_malformed_target_is_skipped_not_raised():
    # A too-short entry is dropped, not fatal; the frame still hashes on what's left.
    key_ok = gate.frame_scene_key({"targets": _SCENE_A})
    key_with_junk = gate.frame_scene_key({"targets": _SCENE_A + [(1.0,)]})
    assert key_ok == key_with_junk


# --------------------------------------------------------------------------------
# check_corpus_overlap: the gate itself
# --------------------------------------------------------------------------------
def test_check_root_clean_corpora_pass(tmp_path):
    _write_fake_corpus(tmp_path, "corpus_a", {
        "train": [_SCENE_A, _SCENE_B], "val": [_SCENE_C], "test": [],
    })
    _write_fake_corpus(tmp_path, "corpus_b", {
        "train": [[(1.0, 0.0, "vehicle")]], "val": [], "test": [],
    })
    ok, report = gate.check_root(tmp_path, allowlist_path=None)
    assert ok
    assert "RESULT: PASS" in report


def test_check_root_self_overlap_is_a_failure(tmp_path):
    """A scene shared between one corpus's own train and val is a structural bug
    (write_manifest splits at the SCENE level) and must fail regardless of any
    allowlist -- the allowlist only ever excuses CROSS-corpus reuse."""
    _write_fake_corpus(tmp_path, "corpus_a", {
        "train": [_SCENE_A], "val": [_SCENE_A], "test": [],
    })
    ok, report = gate.check_root(tmp_path, allowlist_path=None)
    assert not ok
    assert "self-overlap" in report
    assert "FAIL corpus_a: train" in report


def test_check_root_cross_corpus_leak_is_a_failure_without_allowlist(tmp_path):
    """The rt_ablation_txoff-shaped case: corpus B's val/test sits inside corpus A's
    train, two DIFFERENTLY-NAMED corpora, no declared relationship."""
    _write_fake_corpus(tmp_path, "corpus_a", {
        "train": [_SCENE_A, _SCENE_B], "val": [], "test": [],
    })
    _write_fake_corpus(tmp_path, "corpus_b", {
        "train": [_SCENE_C], "val": [_SCENE_A], "test": [],
    })
    ok, report = gate.check_root(tmp_path, allowlist_path=None)
    assert not ok
    assert "FAIL corpus_b val/test OVERLAP corpus_a train: 1 scene(s)" in report


def test_check_root_allowlisted_pair_is_reported_but_not_a_failure(tmp_path):
    _write_fake_corpus(tmp_path, "corpus_a", {
        "train": [_SCENE_A, _SCENE_B], "val": [], "test": [],
    })
    _write_fake_corpus(tmp_path, "corpus_b", {
        "train": [_SCENE_C], "val": [_SCENE_A], "test": [],
    })
    allowlist_path = tmp_path / "allowlist.json"
    allowlist_path.write_text(json.dumps([["corpus_a", "corpus_b"]]))

    ok, report = gate.check_root(tmp_path, allowlist_path=allowlist_path)
    assert ok
    assert "ALLOWLISTED" in report
    assert "RESULT: PASS" in report


def test_check_root_allowlist_is_pairwise_not_transitive(tmp_path):
    """Declaring (a, b) intentional must NOT also excuse an undeclared (a, c) leak."""
    _write_fake_corpus(tmp_path, "corpus_a", {
        "train": [_SCENE_A], "val": [], "test": [],
    })
    _write_fake_corpus(tmp_path, "corpus_b", {
        "train": [], "val": [_SCENE_A], "test": [],
    })
    _write_fake_corpus(tmp_path, "corpus_c", {
        "train": [], "val": [_SCENE_A], "test": [],
    })
    allowlist_path = tmp_path / "allowlist.json"
    allowlist_path.write_text(json.dumps([["corpus_a", "corpus_b"]]))

    ok, report = gate.check_root(tmp_path, allowlist_path=allowlist_path)
    assert not ok  # (a, c) is still undeclared
    assert "ALLOWLISTED" in report
    assert "FAIL" in report


def test_check_root_allowlist_dict_form_with_comment(tmp_path):
    """The shipped corpus_overlap_allowlist.json wraps pairs in {"pairs": [...]}
    alongside a "_comment" -- load_allowlist must accept that form too."""
    _write_fake_corpus(tmp_path, "corpus_a", {"train": [_SCENE_A], "val": [], "test": []})
    _write_fake_corpus(tmp_path, "corpus_b", {"train": [], "val": [_SCENE_A], "test": []})
    allowlist_path = tmp_path / "allowlist.json"
    allowlist_path.write_text(json.dumps({"_comment": "note", "pairs": [["corpus_a", "corpus_b"]]}))

    ok, _report = gate.check_root(tmp_path, allowlist_path=allowlist_path)
    assert ok


def test_check_root_missing_allowlist_file_means_nothing_declared(tmp_path):
    _write_fake_corpus(tmp_path, "corpus_a", {"train": [_SCENE_A], "val": [], "test": []})
    _write_fake_corpus(tmp_path, "corpus_b", {"train": [], "val": [_SCENE_A], "test": []})
    ok, _report = gate.check_root(tmp_path, allowlist_path=tmp_path / "does_not_exist.json")
    assert not ok


def test_check_root_nested_corpus_names_use_posix_separators(tmp_path):
    """Corpora nested under a group directory (e.g. sweep/gentle/<cfg>_<tier>) are
    named with '/' regardless of platform, matching the allowlist's key convention."""
    _write_fake_corpus(tmp_path, "sweep/gentle", {"train": [_SCENE_A], "val": [], "test": []})
    names = [name for name, _dir in gate.find_corpora(tmp_path)]
    assert names == ["sweep/gentle"]  # POSIX separator even if tmp_path is a Windows path


def test_find_corpora_empty_root_returns_nothing(tmp_path):
    assert gate.find_corpora(tmp_path) == []


def test_main_cli_exit_codes(tmp_path):
    _write_fake_corpus(tmp_path, "clean_a", {"train": [_SCENE_A], "val": [_SCENE_B], "test": []})
    assert gate.main(["--root", str(tmp_path), "--allowlist", str(tmp_path / "nope.json")]) == 0

    _write_fake_corpus(tmp_path, "dup_b", {"train": [_SCENE_A], "val": [_SCENE_A], "test": []})
    assert gate.main(["--root", str(tmp_path), "--allowlist", str(tmp_path / "nope.json")]) == 1


# --------------------------------------------------------------------------------
# The real corpora on disk (skips cleanly if the gitignored datasets/ dir is absent,
# e.g. a fresh checkout with no generated corpora yet).
# --------------------------------------------------------------------------------
def test_gate_runs_clean_over_real_datasets_dir_structure():
    """Not an assertion that today's on-disk corpora are leak-free (they are NOT --
    that is the whole reason this module exists) -- just that the gate runs to
    completion over whatever is actually on disk and returns a bool + a report,
    without raising."""
    if not gate.DEFAULT_ROOT.exists():
        pytest.skip("e2e/ml/datasets/ is gitignored and absent on this checkout")
    ok, report = gate.check_root(gate.DEFAULT_ROOT, allowlist_path=gate.DEFAULT_ALLOWLIST)
    assert isinstance(ok, bool)
    assert "RESULT:" in report
