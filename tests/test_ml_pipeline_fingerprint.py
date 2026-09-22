"""A checkpoint must be able to say what code it was trained by.

THE BUG THIS PINS (2026-09-21). `e2e/ml/dataset.py` was edited at 17:19 while a run trained
16:36-19:36. The run recorded `val_AP 0.484`; its checkpoint, reloaded against the edited
dataset, scored `0.023` on the same split -- the network was being fed an input
distribution it had never seen, which reads as a catastrophic model failure rather than as
the bookkeeping error it was. A second run lost its headline number the same way. Worse,
`beat_cfar` skipped retraining whenever the epoch count already matched, so the stale
checkpoints would have been re-scored silently and the wrong numbers published.

The tests below cover the two things that failure needed: a fingerprint that actually
changes when the pipeline changes, and a staleness check that refuses the skip.

Deliberately NOT tested here: that an mtime bump alone is caught. It is not caught for a
mid-run edit -- the final checkpoint write stamps later than the edit -- which is precisely
why the recorded fingerprint, not the mtime, is the authority.
"""

import json

import pytest

torch = pytest.importorskip("torch")

from e2e.ml.beat_cfar import _completed_epochs, _stale_reason
from e2e.ml.train import INPUT_PIPELINE_SOURCES, pipeline_fingerprint


def _fake_tree(root, dataset_body="X = 1\n"):
    """A minimal stand-in for the real source tree, laid out at the same relative paths."""
    (root / "e2e" / "ml" / "models").mkdir(parents=True)
    (root / "e2e" / "chain").mkdir(parents=True)
    (root / "e2e" / "ml" / "dataset.py").write_text(dataset_body)
    (root / "e2e" / "ml" / "baseline.py").write_text("def range_azimuth_power(): return 1\n")
    (root / "e2e" / "chain" / "transforms.py").write_text("# transforms\n")
    (root / "e2e" / "ml" / "labels.py").write_text("# labels\n")
    (root / "e2e" / "ml" / "train.py").write_text("# train\n")
    (root / "e2e" / "ml" / "metrics.py").write_text("# metrics\n")
    (root / "e2e" / "ml" / "models" / "m.py").write_text("# model\n")
    return root


def test_fingerprint_covers_the_classical_front_end(tmp_path):
    """`baseline.range_azimuth_power` is the function that actually caused the incident.

    The first version of the source list omitted it and caught the 2026-09-21 failure only
    because the call site in `dataset.py` moved as well. A change confined to the front end
    must move the digest on its own (reviewed finding, same day).
    """
    root = _fake_tree(tmp_path)
    before = pipeline_fingerprint(root)
    (root / "e2e" / "ml" / "baseline.py").write_text(
        "def range_azimuth_power(): return 2  # notch default retuned\n")
    assert pipeline_fingerprint(root) != before


def test_docstring_and_comment_edits_do_not_invalidate_a_run(tmp_path):
    """Prose is not behaviour.

    A guard that forces a multi-hour retrain because a docstring was fixed gets switched
    off, and then it protects nothing. Hashing the AST with docstrings stripped keeps the
    guard sensitive to what can move a number and blind to what cannot.
    """
    root = _fake_tree(tmp_path, dataset_body='"""Old prose."""\nX = 1  # a comment\n')
    before = pipeline_fingerprint(root)
    (root / "e2e" / "ml" / "dataset.py").write_text(
        '"""Completely rewritten prose, several paragraphs of it."""\nX = 1  # reworded\n')
    assert pipeline_fingerprint(root) == before, "a prose-only edit must not invalidate"

    # ...but the very next character of real code must.
    (root / "e2e" / "ml" / "dataset.py").write_text('"""Old prose."""\nX = 2\n')
    assert pipeline_fingerprint(root) != before


def test_unparseable_source_falls_back_to_raw_bytes(tmp_path):
    """Be conservative exactly when the clever path is unavailable."""
    root = _fake_tree(tmp_path)
    (root / "e2e" / "ml" / "dataset.py").write_text("def broken( :\n")
    before = pipeline_fingerprint(root)
    assert before is not None
    (root / "e2e" / "ml" / "dataset.py").write_text("def broken( :  # changed\n")
    assert pipeline_fingerprint(root) != before


def test_fingerprint_is_stable_for_unchanged_sources(tmp_path):
    root = _fake_tree(tmp_path)
    assert pipeline_fingerprint(root) == pipeline_fingerprint(root)


def test_fingerprint_changes_when_the_input_pipeline_changes(tmp_path):
    """Content, not mtime: this is the signal the checkpoint records."""
    root = _fake_tree(tmp_path)
    before = pipeline_fingerprint(root)
    (root / "e2e" / "ml" / "dataset.py").write_text("X = 2  # the front-end parity fix\n")
    assert pipeline_fingerprint(root) != before


def test_fingerprint_covers_every_declared_source(tmp_path):
    """Each declared source must actually contribute, or the guard has a blind spot."""
    edits = {
        "e2e/ml/dataset.py": "X = 99\n",
        "e2e/ml/baseline.py": "def range_azimuth_power(): return 99\n",
        "e2e/chain/transforms.py": "Y = 1\n",
        "e2e/ml/labels.py": "Z = 1\n",
        "e2e/ml/train.py": "W = 1\n",
        "e2e/ml/metrics.py": "V = 1\n",
        "e2e/ml/models": "U = 1\n",
    }
    assert set(edits) == set(INPUT_PIPELINE_SOURCES), "source list changed; update this test"
    for src, body in edits.items():
        root = _fake_tree(tmp_path / src.replace("/", "_"))
        before = pipeline_fingerprint(root)
        target = (root / src / "m.py") if src.endswith("models") else (root / src)
        target.write_text(body)
        assert pipeline_fingerprint(root) != before, f"{src} does not affect the fingerprint"


def test_fingerprint_is_none_when_sources_are_absent(tmp_path):
    assert pipeline_fingerprint(tmp_path) is None


def _write_checkpoint(out_dir, *, fingerprint, epochs=3):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": {}, "model_name": "fake", "pipeline_fingerprint": fingerprint},
               out_dir / "best.pt")
    (out_dir / "history.json").write_text(json.dumps({"epoch": list(range(1, epochs + 1))}))
    return out_dir


def test_matching_fingerprint_is_not_stale(tmp_path, monkeypatch):
    out = _write_checkpoint(tmp_path / "run", fingerprint=pipeline_fingerprint())
    assert _stale_reason(str(out)) is None


def test_mismatched_fingerprint_is_stale(tmp_path):
    """The case that cost two results: recorded code != current code."""
    out = _write_checkpoint(tmp_path / "run", fingerprint="0" * 64)
    reason = _stale_reason(str(out))
    assert reason is not None and "fingerprint" in reason


def test_missing_checkpoint_is_not_stale(tmp_path):
    """Nothing to invalidate; `_train` handles the absent-run case by its epoch count."""
    (tmp_path / "run").mkdir()
    assert _stale_reason(str(tmp_path / "run")) is None


def test_completed_epochs_reads_history(tmp_path):
    out = _write_checkpoint(tmp_path / "run", fingerprint=pipeline_fingerprint(), epochs=7)
    assert _completed_epochs(str(out)) == 7
