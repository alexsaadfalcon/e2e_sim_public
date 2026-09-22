"""Fixes from the 2026-09-22 fresh-context review of the demo work.

1. The GUI's ML detector died unless the Dash server's CWD was the repo root: the
   checkpoint's recorded manifest path is repo-relative and was opened bare.
2. Corpus replay silently ignored enabled blocks it cannot apply; the diagram showed them
   lit and the results showed nothing for them. The run now says which it skipped.
3. Two frame ceilings existed (20 in the presets, 50 in the registry, "20" typed into the
   runner's error text); they now come from one constant each.
"""

import json
import os

import pytest

torch = pytest.importorskip("torch")

from e2e.ml.blocks import _REPO_ROOT, _resolve_manifest_path
from webapp.pipeline_registry import MAX_N_STEPS, MAX_PRESET_N_STEPS


# ------------------------------------------------------------------------------------
# 1. manifest resolution is CWD-independent
# ------------------------------------------------------------------------------------
def test_absolute_manifest_path_is_used_as_given(tmp_path):
    p = tmp_path / "manifest.json"
    assert _resolve_manifest_path(str(p)) == p


def test_relative_manifest_path_resolves_against_repo_root_from_another_cwd(tmp_path, monkeypatch):
    rel = "e2e/ml/datasets/_resolver_probe/manifest.json"
    target = _REPO_ROOT / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.write_text("{}")
        monkeypatch.chdir(tmp_path)            # anywhere but the repo root
        assert not (tmp_path / rel).exists()
        assert _resolve_manifest_path(rel) == target
        assert _resolve_manifest_path(rel.replace("/", os.sep)) == target
    finally:
        target.unlink(missing_ok=True)
        target.parent.rmdir()


def test_relative_manifest_path_prefers_the_cwd_when_it_exists_there(tmp_path, monkeypatch):
    (tmp_path / "m.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    assert _resolve_manifest_path("m.json").resolve() == (tmp_path / "m.json").resolve()


# ------------------------------------------------------------------------------------
# 2. corpus replay reports what it skipped
# ------------------------------------------------------------------------------------
def test_corpus_mode_reports_skipped_blocks_in_run_notes(tmp_path):
    from tests.test_webapp_detector import _corpus_state, _tiny_corpus
    from webapp.pipeline_runner import run_pipeline
    st = _corpus_state(_tiny_corpus(tmp_path))
    st["fft"]["enabled"] = True          # lit on the diagram, inapplicable to a stored ADC frame
    st["rffe"]["enabled"] = True
    out = run_pipeline(st, n_steps=1)
    notes = out["_axis_meta"]["notes"]
    assert len(notes) == 1 and "fft" in notes[0] and "rffe" in notes[0]
    assert "fft" not in out


def test_corpus_mode_with_nothing_to_skip_has_no_notes(tmp_path):
    from tests.test_webapp_detector import _corpus_state, _tiny_corpus
    from webapp.demo_presets import PRESETS_BY_ID, apply_preset
    from webapp.pipeline_runner import run_pipeline
    # The demo preset disables everything a replayed frame cannot use -- so no note.
    st = apply_preset(PRESETS_BY_ID["thrust5_detector_cfar"])
    st["corpus_environment"]["params"]["manifest"] = str(_tiny_corpus(tmp_path))
    st["detector"]["params"].update({"cfar_guard": 1, "cfar_train": 2})
    out = run_pipeline(st, n_steps=1)
    assert out["_axis_meta"]["notes"] == []


def test_run_status_carries_the_notes():
    """The app appends run notes to the status line (the only place an operator looks)."""
    import inspect
    import webapp.app as appmod
    src = inspect.getsource(appmod._run_pipeline)
    assert 'axis_meta.get("notes")' in src


# ------------------------------------------------------------------------------------
# 3. one ceiling each
# ------------------------------------------------------------------------------------
def test_preset_ceiling_is_the_registry_constant():
    from webapp import demo_presets
    assert demo_presets.MAX_PRESET_N_STEPS is MAX_PRESET_N_STEPS
    assert MAX_PRESET_N_STEPS <= MAX_N_STEPS


def test_runner_ceiling_message_quotes_the_preset_constant():
    from webapp.pipeline_registry import default_block_state
    from webapp.pipeline_runner import PipelineError, run_pipeline
    with pytest.raises(PipelineError, match=f"at most {MAX_PRESET_N_STEPS}"):
        run_pipeline(default_block_state(), n_steps=MAX_N_STEPS + 1)


# ------------------------------------------------------------------------------------
# 4. the staleness guard's mtime fallback must not depend on the CWD either
# ------------------------------------------------------------------------------------
def test_stale_reason_fallback_sees_the_sources_from_another_cwd(tmp_path, monkeypatch):
    """An unfingerprinted checkpoint older than the pipeline sources is stale from ANY
    directory. Before the fix, from outside the repo root the sources 'did not exist'
    and the fallback vouched for everything."""
    from e2e.ml.beat_cfar import _stale_reason
    run = tmp_path / "old_run"
    run.mkdir()
    torch.save({"model_state": {}, "model_name": "fake"}, run / "best.pt")   # no fingerprint
    os.utime(run / "best.pt", (0, 0))                                         # 1970: older than everything
    (run / "history.json").write_text(json.dumps({"epoch": [1]}))
    monkeypatch.chdir(tmp_path)
    reason = _stale_reason(str(run))
    assert reason is not None and "no recorded fingerprint" in reason
