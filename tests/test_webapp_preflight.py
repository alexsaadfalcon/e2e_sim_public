"""Tests for webapp/preflight.py -- the pre-demo check the RUNBOOK points at.

Every check function accepts its heavy dependencies (repo root, preset list, a
`run_pipeline` callable, a port prober) as optional parameters, so these tests exercise
missing-file / busy-port / missing-checkpoint / slow-run branches without a GPU, real
frames, or a bound socket. Shared fixtures come from tests/conftest.py; not redefined
here.
"""
from __future__ import annotations

import pytest

from webapp.demo_presets import DemoPreset
from webapp.pipeline_registry import BLOCKS_BY_ID, default_block_state
from webapp.preflight import (
    CheckResult,
    _preset_asset_paths,
    _step_grid_violations,
    check_assets,
    check_environment,
    check_port,
    check_presets,
    check_timing_budget,
    check_warmup,
)


def _preset(**overrides) -> DemoPreset:
    return DemoPreset(id="x", label="x preset", thrust=1, n_steps=3, overrides=overrides,
                      blurb="b", say=["s"], do_not_say=["d"])


# =====================================================================================
# 1. Assets
# =====================================================================================

def test_check_assets_all_present(tmp_path):
    from webapp.corpus_catalog import discover_sionna_scenarios

    manifest = tmp_path / "corpus" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    checkpoint = tmp_path / "runs" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("")
    sims_dir = tmp_path / "e2e" / "environment" / "sionna_sims"
    sims_dir.mkdir(parents=True)
    (sims_dir / "munich_ka.pkl").write_bytes(b"x")  # garbage bytes: _munich_ka_label's
    (sims_dir / "munich.pkl").write_bytes(b"x")     # pickle.load fails closed to a
    # Thrust 3's swept-line-of-sight scene (2026-09-25): checked by name, because a
    # missing file makes that preset fall back to the static scene and draw a screen its
    # card no longer describes.
    (sims_dir / "munich_ka_losweep.pkl").write_bytes(b"x")
                                                     # metadata-free label, never raises
    tessera_csv = tmp_path / "e2e" / "data" / "interconnect" / "tessera_tsv_s21_public.csv"
    tessera_csv.parent.mkdir(parents=True)
    tessera_csv.write_text("freq_hz,s21_re,s21_im\n")
    checkpoint_dir = (tmp_path / "e2e" / "interconnect_surrogate" / "_models"
                     / "tessera_checkout" / "models")
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "best_model.pth").write_bytes(b"x")
    (checkpoint_dir / "input_scaler.pt").write_bytes(b"x")

    presets = [_preset(
        corpus_environment={"enabled": True, "params": {"manifest": str(manifest)}},
        detector={"params": {"mode": "ml", "checkpoint": str(checkpoint)}},
    )]

    # registry_choices comes from the SAME scan here, mirroring how the real registry is
    # built from discover_sionna_scenarios -- this test checks check_assets' PASS branch
    # when they agree, not the exact label text (that's _discover_sionna_scenario_specs'
    # concern, tested in test_corpus_catalog.py-style tests elsewhere).
    scanned = discover_sionna_scenarios(sims_dir)
    results = check_assets(repo_root=tmp_path, presets=presets, sims_dir=sims_dir,
                           registry_choices=scanned)
    by_name = {r.name: r for r in results}
    assert by_name["assets.munich_ka_pkl"].status == "PASS"
    assert by_name["assets.munich_ka_losweep_pkl"].status == "PASS"
    assert by_name["assets.munich_legacy_pkl"].status == "PASS"
    assert by_name["assets.tessera_public_csv"].status == "PASS"
    assert by_name["assets.tessera_checkpoint"].status == "PASS"
    assert by_name["assets.sionna_registry_match"].status == "PASS"
    assert by_name[f"assets.{manifest}"].status == "PASS"
    assert by_name[f"assets.{checkpoint}"].status == "PASS"
    assert by_name["assets.summary"].status == "PASS"


def test_check_assets_reports_missing_checkpoint_and_manifest(tmp_path):
    """A preset referencing a corpus/checkpoint that is not on disk is a named FAIL,
    not a silent pass -- this is the failure mode a demo machine actually hits."""
    presets = [_preset(
        corpus_environment={"enabled": True, "params": {"manifest": "e2e/ml/datasets/nope/manifest.json"}},
        detector={"params": {"mode": "ml", "checkpoint": "e2e/ml/runs/nope/best.pt"}},
    )]
    results = check_assets(repo_root=tmp_path, presets=presets, sims_dir=tmp_path / "sims",
                           registry_choices=["munich"])
    by_name = {r.name: r for r in results}
    assert by_name["assets.munich_ka_pkl"].status == "FAIL"
    assert by_name["assets.munich_legacy_pkl"].status == "FAIL"
    assert by_name["assets.tessera_public_csv"].status == "FAIL"
    assert by_name["assets.tessera_checkpoint"].status == "FAIL"
    assert by_name["assets.e2e/ml/datasets/nope/manifest.json"].status == "FAIL"
    assert "x" in by_name["assets.e2e/ml/datasets/nope/manifest.json"].message  # names preset id
    assert by_name["assets.e2e/ml/runs/nope/best.pt"].status == "FAIL"
    assert by_name["assets.summary"].status == "FAIL"
    assert "manifest.json" in by_name["assets.summary"].message


def test_check_assets_reports_missing_tessera_checkpoint_file_individually(tmp_path):
    """Only one of the two checkpoint files present is still a FAIL (an incomplete
    fetch is as unusable as no fetch) -- the real failure mode
    `python -m e2e.interconnect_surrogate.fetch` interrupted mid-clone would produce."""
    checkpoint_dir = (tmp_path / "e2e" / "interconnect_surrogate" / "_models"
                     / "tessera_checkout" / "models")
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "best_model.pth").write_bytes(b"x")  # input_scaler.pt missing
    results = check_assets(repo_root=tmp_path, presets=[], sims_dir=tmp_path / "sims",
                           registry_choices=["munich"])
    by_name = {r.name: r for r in results}
    assert by_name["assets.tessera_checkpoint"].status == "FAIL"


def test_check_assets_cfar_preset_has_no_checkpoint_to_check():
    """mode='cfar' presets carry no checkpoint path; the scan must not invent one (the
    fixed infra checks like assets.tessera_checkpoint are unrelated to this scan, so this
    exercises the preset-path helper directly rather than substring-matching result names)."""
    presets = [_preset(detector={"params": {"mode": "cfar", "checkpoint": ""}})]
    assert _preset_asset_paths(presets) == {}


def test_check_assets_flags_sionna_registry_mismatch(tmp_path):
    (tmp_path / "e2e" / "environment" / "sionna_sims").mkdir(parents=True)
    results = check_assets(repo_root=tmp_path, presets=[],
                           sims_dir=tmp_path / "e2e" / "environment" / "sionna_sims",
                           registry_choices=["munich", "etoile"])
    by_name = {r.name: r for r in results}
    # nothing on disk -> scan is [] but the registry claims two scenarios -> mismatch
    assert by_name["assets.sionna_registry_match"].status == "FAIL"


# =====================================================================================
# 2. Environment
# =====================================================================================

class _FakeCuda:
    def __init__(self, available, count=0, name="Fake GPU", mem_info=(1_000_000_000, 2_000_000_000)):
        self._available = available
        self._count = count
        self._name = name
        self._mem_info = mem_info

    def is_available(self):
        return self._available

    def device_count(self):
        return self._count

    def get_device_name(self, idx):
        return self._name

    def mem_get_info(self, idx):
        return self._mem_info


class _FakeTorch:
    __version__ = "0.0.0-fake"

    def __init__(self, cuda):
        self.cuda = cuda


def test_check_environment_reports_cuda_available():
    fake = _FakeTorch(_FakeCuda(available=True, count=1))
    results = check_environment(torch_module=fake)
    cuda = next(r for r in results if r.name == "env.cuda")
    assert cuda.status == "PASS"
    assert "1 device" in cuda.message and "Fake GPU" in cuda.message


def test_check_environment_fails_without_cuda():
    """Monkeypatch-without-GPU path: a torch stub reporting zero devices is a named FAIL,
    not a silent skip -- the demo needs a real device."""
    fake = _FakeTorch(_FakeCuda(available=False, count=0))
    results = check_environment(torch_module=fake)
    cuda = next(r for r in results if r.name == "env.cuda")
    assert cuda.status == "FAIL"
    assert "device count 0" in cuda.message


def test_check_environment_survives_a_cuda_query_exception():
    class _BoomCuda(_FakeCuda):
        def is_available(self):
            raise RuntimeError("no driver")
    fake = _FakeTorch(_BoomCuda(available=False))
    results = check_environment(torch_module=fake)
    cuda = next(r for r in results if r.name == "env.cuda")
    assert cuda.status == "FAIL"
    assert "no driver" in cuda.message
    # exactly one env.cuda result -- the exception path must not ALSO append the
    # generic "device count 0" FAIL underneath it.
    assert sum(1 for r in results if r.name == "env.cuda") == 1


def test_check_environment_reports_dash_plotly_and_app_import():
    pytest.importorskip("dash")
    pytest.importorskip("plotly")
    fake = _FakeTorch(_FakeCuda(available=False, count=0))
    results = check_environment(torch_module=fake)
    by_name = {r.name: r for r in results}
    assert by_name["env.dash_plotly"].status == "PASS"
    assert by_name["env.app_import"].status == "PASS"


# =====================================================================================
# 3. Port
# =====================================================================================

def test_check_port_free():
    r = check_port(host="127.0.0.1", port=8074, prober=lambda h, p: False)
    assert r.status == "PASS"
    assert "8074" in r.message


def test_check_port_busy():
    r = check_port(host="127.0.0.1", port=8074, prober=lambda h, p: True)
    assert r.status == "FAIL"
    assert "already in use" in r.message


def test_check_port_defaults_to_the_app_module_host_and_port():
    pytest.importorskip("dash")
    from webapp.app import HOST, PORT
    seen = {}

    def prober(h, p):
        seen["host"], seen["port"] = h, p
        return False

    r = check_port(prober=prober)
    assert seen == {"host": HOST, "port": PORT}
    assert r.status == "PASS"


# =====================================================================================
# 5. Presets / step-grid
# =====================================================================================

def test_step_grid_violations_flags_off_grid_value():
    p = _preset(rffe={"params": {"lna_bias_ma": 8.3}})  # step=0.5, min=0.5 -> off grid
    problems = _step_grid_violations(p, BLOCKS_BY_ID)
    assert problems and "lna_bias_ma" in problems[0]


def test_step_grid_violations_accepts_on_grid_value():
    p = _preset(rffe={"params": {"lna_bias_ma": 8.0}})
    assert _step_grid_violations(p, BLOCKS_BY_ID) == []


def test_step_grid_violations_ignores_step_any_and_choice_params():
    p = _preset(rffe={"params": {"signal_scaling": 1e-7}},  # step="any"
               interconnect={"params": {"case": "passthrough"}})  # kind="choice"
    assert _step_grid_violations(p, BLOCKS_BY_ID) == []


def test_check_presets_all_pass_for_the_real_presets():
    results = check_presets()
    summary = next(r for r in results if r.name == "presets.summary")
    assert summary.status == "PASS", summary.message
    assert "->" in summary.message  # stage order printed


def test_check_presets_reports_a_bad_override_as_a_named_fail():
    bad = _preset(rffe={"params": {"lna_bias_ma": 99.0}})  # above max=10.0
    results = check_presets(presets=[bad])
    by_name = {r.name: r for r in results}
    assert by_name["preset.x"].status == "FAIL"
    assert by_name["presets.summary"].status == "FAIL"
    assert "0/1" in by_name["presets.summary"].message


def test_check_presets_reports_off_step_override_as_a_named_fail():
    bad = _preset(rffe={"params": {"lna_bias_ma": 8.3}})
    results = check_presets(presets=[bad])
    assert next(r for r in results if r.name == "preset.x").status == "FAIL"


# =====================================================================================
# 4. Warm-up / 6. Timing budget (fake run_pipeline -- no torch/GPU needed)
# =====================================================================================

def _presets_of(n_steps_list):
    return [DemoPreset(id=f"p{i}", label=f"p{i}", thrust=1, n_steps=n, overrides={},
                       blurb="b", say=["s"], do_not_say=["d"])
           for i, n in enumerate(n_steps_list)]


def test_check_warmup_picks_the_cheapest_preset_and_reports_wall_time():
    presets = _presets_of([5, 1, 3])
    seen = {}

    def fake_run_pipeline(state, n_steps):
        seen["n_steps"] = n_steps
        return {}

    r = check_warmup(presets=presets, run_pipeline_fn=fake_run_pipeline)
    assert r.status == "PASS"
    assert seen["n_steps"] == 1  # warm-up always runs exactly 1 frame
    assert "p1" in r.message  # the n_steps=1 preset is the cheapest


def test_check_warmup_reports_a_run_failure():
    presets = _presets_of([1])

    def boom(state, n_steps):
        raise RuntimeError("no frames on this machine")

    r = check_warmup(presets=presets, run_pipeline_fn=boom)
    assert r.status == "FAIL"
    assert "no frames on this machine" in r.message


def test_check_timing_budget_reports_wall_seconds_per_preset():
    presets = _presets_of([1, 1])

    def fast_run_pipeline(state, n_steps):
        return {}

    results = check_timing_budget(presets=presets, run_pipeline_fn=fast_run_pipeline)
    assert len(results) == 2
    assert all(r.status == "PASS" for r in results)
    assert all("1 run" in r.message for r in results)


def test_check_timing_budget_warns_above_15s(monkeypatch):
    """Fake time.time() (via the monkeypatch fixture, auto-restored) rather than
    actually sleeping 15s in the test suite."""
    import webapp.preflight as preflight_mod

    presets = _presets_of([1])
    calls = {"n": 0}

    def fake_time():
        calls["n"] += 1
        return 0.0 if calls["n"] == 1 else 20.0

    monkeypatch.setattr(preflight_mod.time, "time", fake_time)
    results = check_timing_budget(presets=presets, run_pipeline_fn=lambda state, n_steps: {})
    assert results[0].status == "WARN"
    assert "over the 15s budget" in results[0].message


def test_check_timing_budget_runs_both_arms_of_an_ab_preset():
    p = DemoPreset(id="ab", label="ab", thrust=1, n_steps=2,
                  overrides={"rffe": {"params": {"lna_bias_ma": 8.0}}},
                  blurb="b", say=["s"], do_not_say=["d"],
                  ab=("rffe", "lna_bias_ma", 0.5), ab_label_a="8 mA", ab_label_b="0.5 mA")
    calls = []

    def fake_run_pipeline(state, n_steps):
        calls.append(state["rffe"]["params"]["lna_bias_ma"])
        return {}

    results = check_timing_budget(presets=[p], run_pipeline_fn=fake_run_pipeline)
    assert calls == [8.0, 0.5]  # arm A then arm B, one click on stage runs both
    assert "2 arms" in results[0].message


def test_check_timing_budget_reports_a_run_failure():
    presets = _presets_of([1])

    def boom(state, n_steps):
        raise RuntimeError("ML checkpoint not found")

    results = check_timing_budget(presets=presets, run_pipeline_fn=boom)
    assert results[0].status == "FAIL"
    assert "ML checkpoint not found" in results[0].message


# =====================================================================================
# CLI exit code
# =====================================================================================

def test_main_exit_code_follows_results(monkeypatch):
    import webapp.preflight as preflight_mod

    monkeypatch.setattr(preflight_mod, "run_all", lambda quick: [
        CheckResult("a", "PASS", "ok"), CheckResult("b", "WARN", "slow")])
    assert preflight_mod.main(["--quick"]) == 0

    monkeypatch.setattr(preflight_mod, "run_all", lambda quick: [
        CheckResult("a", "PASS", "ok"), CheckResult("b", "FAIL", "broken")])
    assert preflight_mod.main(["--quick"]) == 1


def test_main_quick_flag_skips_warmup_and_timing(monkeypatch):
    import webapp.preflight as preflight_mod

    seen = []
    monkeypatch.setattr(preflight_mod, "check_assets", lambda: [])
    monkeypatch.setattr(preflight_mod, "check_environment", lambda: [])
    monkeypatch.setattr(preflight_mod, "check_port", lambda: CheckResult("port", "PASS", "free"))
    monkeypatch.setattr(preflight_mod, "check_presets", lambda: [])
    monkeypatch.setattr(preflight_mod, "check_warmup", lambda: seen.append("warmup"))
    monkeypatch.setattr(preflight_mod, "check_timing_budget", lambda: seen.append("timing"))

    results = preflight_mod.run_all(quick=True)
    assert seen == []  # neither warm-up nor timing ran
    statuses = {r.name: r.status for r in results}
    assert statuses["warmup"] == "SKIP" and statuses["timing"] == "SKIP"
