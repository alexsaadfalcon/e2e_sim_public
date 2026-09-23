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
    """ADC replay with every inapplicable block already off says nothing -- unchanged.

    The vehicle changed on 2026-09-23, not the invariant: this used to load the
    thrust5_detector_cfar preset, which since the live-chain change replays the stored
    RAY-TRACED CHANNEL (`corpus_environment.domain == "cfr"`) and turns the whole ADC
    chain ON. On a corpus with no `.cfr.npy` sidecar that preset now (correctly) fails
    loudly, and on one with sidecars it always emits notes -- the live-vs-stored gate
    is a run note. The state is therefore built directly, which is what the test was
    always about: an ADC-replay run with nothing to skip.
    """
    from tests.test_webapp_detector import _corpus_state, _tiny_corpus
    from webapp.pipeline_runner import run_pipeline
    st = _corpus_state(_tiny_corpus(tmp_path))
    for bid in ("fft", "range_az", "range_el", "range_profile", "subspace_err",
                "rffe", "interconnect", "afe", "dechirp", "thermal_noise",
                "impairment", "if_hpf", "quantizer", "sink"):
        st[bid]["enabled"] = False
    st["detector"]["params"].update({"cfar_guard": 1, "cfar_train": 2})
    out = run_pipeline(st, n_steps=1)
    assert out["_axis_meta"]["notes"] == []
    assert out["_axis_meta"]["source"].startswith("Corpus Replay (ADC replay)")


def test_the_live_chain_preset_refuses_a_corpus_with_no_stored_channel(tmp_path):
    """The other half of the change above, pinned where the reader of that test is:
    thrust5_detector_cfar asks for the stored ray-traced channel, so a corpus that
    does not carry one is an error naming --store-cfr, never a silent ADC replay of
    frames the preset's card describes as computed live."""
    from tests.test_webapp_detector import _tiny_corpus
    from webapp.demo_presets import PRESETS_BY_ID, apply_preset
    from webapp.pipeline_runner import PipelineError, run_pipeline
    st = apply_preset(PRESETS_BY_ID["thrust5_detector_cfar"])
    st["corpus_environment"]["params"]["manifest"] = str(_tiny_corpus(tmp_path))
    with pytest.raises(PipelineError, match="store-cfr"):
        run_pipeline(st, n_steps=1)


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
