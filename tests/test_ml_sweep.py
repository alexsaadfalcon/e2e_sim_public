"""Tests for `e2e.ml.sweep`: grid expansion, resume/skip, `pick_best`, and the CLI.

Kept FAST and torch-free-at-the-training-level: every test monkeypatches
`e2e.ml.sweep.train` (the module-level name `sweep.py` imports from `e2e.ml.train`,
same pattern as `test_ml_train.py`'s `detection_loss` spy) with a canned stub -- no real
model is ever built or trained here.
"""
from __future__ import annotations

import json
import math

import pytest

from e2e.ml import sweep


# --------------------------------------------------------------------------------
# DEFAULT_GRID / grid expansion
# --------------------------------------------------------------------------------
def test_default_grid_matches_design_doc():
    # Verbatim numbers from the panel's sweep design doc S2.
    assert sweep.DEFAULT_GRID == {
        "reg_weight": [1, 10, 30, 100],
        "lr": [1e-4, 3e-4],
        "gamma_default": 2.0,
        "stage2_gamma": [0.0, 2.0],
    }


def test_stage1_trials_order_and_count():
    trials = sweep._stage1_trials(sweep.DEFAULT_GRID)
    assert len(trials) == 8  # 4 reg_weight x 2 lr
    expected = [
        {"reg_weight": 1.0, "lr": 1e-4, "gamma": 2.0},
        {"reg_weight": 1.0, "lr": 3e-4, "gamma": 2.0},
        {"reg_weight": 10.0, "lr": 1e-4, "gamma": 2.0},
        {"reg_weight": 10.0, "lr": 3e-4, "gamma": 2.0},
        {"reg_weight": 30.0, "lr": 1e-4, "gamma": 2.0},
        {"reg_weight": 30.0, "lr": 3e-4, "gamma": 2.0},
        {"reg_weight": 100.0, "lr": 1e-4, "gamma": 2.0},
        {"reg_weight": 100.0, "lr": 3e-4, "gamma": 2.0},
    ]
    assert trials == expected


def test_stage2_trials_crossed_with_winner():
    winner_params = {"reg_weight": 10.0, "lr": 3e-4}
    trials = sweep._stage2_trials(sweep.DEFAULT_GRID, winner_params)
    assert trials == [
        {"reg_weight": 10.0, "lr": 3e-4, "gamma": 0.0},
        {"reg_weight": 10.0, "lr": 3e-4, "gamma": 2.0},
    ]


def test_trial_slug_deterministic():
    params = {"reg_weight": 10.0, "lr": 0.0003, "gamma": 2.0}
    assert sweep.trial_slug(params) == sweep.trial_slug(dict(params))
    assert sweep.trial_slug({"reg_weight": 1.0, "lr": 1e-4, "gamma": 2.0}) != \
        sweep.trial_slug({"reg_weight": 2.0, "lr": 1e-4, "gamma": 2.0})


def test_validate_grid_missing_key_raises():
    with pytest.raises(ValueError):
        sweep._validate_grid({"reg_weight": [1], "lr": [1e-4]})  # no stage2_gamma


# --------------------------------------------------------------------------------
# Objective / guard helpers
# --------------------------------------------------------------------------------
def test_mean_last_n_uses_available_epochs_when_fewer_than_window():
    assert sweep._mean_last_n([0.1, 0.3]) == pytest.approx(0.2)
    assert sweep._mean_last_n([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) == pytest.approx(0.6)  # last 5


def test_is_ar_declining_true_only_when_strictly_monotonic_over_window():
    assert sweep._is_ar_declining([0.5, 0.4, 0.3, 0.2, 0.1]) is True
    assert sweep._is_ar_declining([0.1, 0.2, 0.3, 0.4, 0.5]) is False
    assert sweep._is_ar_declining([0.5, 0.5, 0.5]) is False  # flat, not "still falling"
    assert sweep._is_ar_declining([0.9]) is False  # can't tell a trend from one point


# --------------------------------------------------------------------------------
# pick_best
# --------------------------------------------------------------------------------
def _trial(objective, ar_declining=False, final_ar=0.3):
    return {"objective_mean_ap_last5": objective, "ar_declining": ar_declining,
            "final_val_AR": final_ar}


def test_pick_best_picks_highest_objective_excluding_declining():
    trials = [
        _trial(0.9, ar_declining=True),   # best objective but reproduces the AR-collapse bug
        _trial(0.5, ar_declining=False),  # winner among non-declining
        _trial(0.2, ar_declining=False),
    ]
    assert sweep.pick_best(trials) is trials[1]


def test_pick_best_falls_back_when_every_trial_is_declining():
    trials = [_trial(0.9, ar_declining=True), _trial(0.5, ar_declining=True)]
    assert sweep.pick_best(trials) is trials[0]  # best objective, with the caveat documented


def test_pick_best_tie_breaks_on_final_val_ar():
    trials = [_trial(0.5, final_ar=0.1), _trial(0.5, final_ar=0.4)]
    assert sweep.pick_best(trials) is trials[1]


def test_pick_best_accepts_wrapped_payload():
    trials = [_trial(0.1), _trial(0.7)]
    payload = {"trials": trials}
    assert sweep.pick_best(payload) is trials[1]


def test_pick_best_empty_raises():
    with pytest.raises(ValueError):
        sweep.pick_best([])


# --------------------------------------------------------------------------------
# run_sweep: grid iteration, resume/skip, results schema (stubbed train())
# --------------------------------------------------------------------------------
def _canned_history(epochs: int, ap: float, ar: float = 0.4) -> dict:
    return {
        "epoch": list(range(1, epochs + 1)),
        "train_loss": [1.0] * epochs,
        "train_cls_loss": [0.6] * epochs,
        "train_reg_loss": [0.2] * epochs,
        "val_AP": [ap] * epochs,
        "val_AR": [ar] * epochs,
        "val_range_rmse_m": [0.0] * epochs,
    }


def _score(reg_weight: float, lr: float, gamma: float) -> float:
    """Deterministic, tie-free score: monotone in the stage-1 grid's own listing
    order, EXCEPT stage 2's gamma=0.0 on the eventual winner scores higher still
    (so the test can assert the sweep actually "found" that improvement)."""
    base = {
        (1.0, 1e-4): 0.10, (1.0, 3e-4): 0.11,
        (10.0, 1e-4): 0.12, (10.0, 3e-4): 0.13,
        (30.0, 1e-4): 0.14, (30.0, 3e-4): 0.15,
        (100.0, 1e-4): 0.16, (100.0, 3e-4): 0.17,
    }[(reg_weight, lr)]
    if (reg_weight, lr) == (100.0, 3e-4) and gamma == 0.0:
        return 0.99
    return base


def _make_stub_train(epochs: int):
    calls = []

    def _stub(manifest_path, model_name, *, epochs=epochs, batch_size=8, seed=0, out_dir=None,
               device=None, reg_weight=100.0, lr=1e-4, gamma=2.0, **kwargs):
        calls.append({"reg_weight": reg_weight, "lr": lr, "gamma": gamma})
        return _canned_history(epochs, ap=_score(reg_weight, lr, gamma))

    return _stub, calls


def test_run_sweep_full_grid_call_count_resume_and_results_schema(tmp_path, monkeypatch):
    stub, calls = _make_stub_train(epochs=5)
    monkeypatch.setattr(sweep, "train", stub)

    manifest_path = tmp_path / "manifest.json"  # never opened by run_sweep itself (train stubbed)
    out_dir = tmp_path / "sweep_out"
    results_path = sweep.run_sweep(manifest_path, "fftradnet", epochs=5, batch_size=2, seed=0,
                                    out_dir=out_dir)

    assert results_path == out_dir / "sweep_results.json"
    payload = json.loads(results_path.read_text())
    trials = payload["trials"]
    assert len(trials) == 10  # 8 stage1 + 2 stage2

    # Stage-2 gamma=2.0 exactly repeats the stage-1 winner's (reg_weight, lr, gamma) ->
    # same trial_slug -> resumed, not re-trained. Everything else is a fresh call.
    assert len(calls) == 9

    expected_keys = {"stage", "params", "out_dir", "best_val_AP", "final_val_AP", "final_val_AR",
                      "final_train_cls_loss", "final_train_reg_loss", "objective_mean_ap_last5",
                      "ar_declining", "wall_s", "resumed"}
    for t in trials:
        assert set(t.keys()) == expected_keys

    # Sorted descending by the documented objective.
    objectives = [t["objective_mean_ap_last5"] for t in trials]
    assert objectives == sorted(objectives, reverse=True)

    top = trials[0]
    assert top["params"] == {"reg_weight": 100.0, "lr": 3e-4, "gamma": 0.0}
    assert top["stage"] == "stage2"
    assert top["resumed"] is False
    assert top["objective_mean_ap_last5"] == pytest.approx(0.99)

    resumed_trials = [t for t in trials if t["resumed"]]
    assert len(resumed_trials) == 1
    assert resumed_trials[0]["params"] == {"reg_weight": 100.0, "lr": 3e-4, "gamma": 2.0}
    assert resumed_trials[0]["wall_s"] == 0.0

    # pick_best on the actual written payload agrees with the printed leaderboard's top row.
    assert sweep.pick_best(payload)["params"] == top["params"]


def test_run_sweep_preseeded_trial_is_skipped_not_retrained(tmp_path, monkeypatch):
    grid = {"reg_weight": [1.0, 2.0], "lr": [1e-4], "gamma_default": 2.0, "stage2_gamma": [2.0]}
    out_dir = tmp_path / "sweep_out"

    preseeded_params = {"reg_weight": 1.0, "lr": 1e-4, "gamma": 2.0}
    preseeded_dir = out_dir / sweep.trial_slug(preseeded_params)
    preseeded_dir.mkdir(parents=True)
    preseeded_history = _canned_history(3, ap=0.55, ar=0.33)
    (preseeded_dir / "history.json").write_text(json.dumps(preseeded_history))

    calls = []

    def _stub(manifest_path, model_name, *, epochs, batch_size, seed, out_dir, device=None,
              reg_weight, lr, gamma, **kwargs):
        assert (reg_weight, lr) != (1.0, 1e-4), "preseeded trial must not be retrained"
        calls.append({"reg_weight": reg_weight, "lr": lr, "gamma": gamma})
        # (2.0, 1e-4) scores higher so it becomes the stage-1 winner deterministically.
        return _canned_history(epochs, ap=0.7)

    monkeypatch.setattr(sweep, "train", _stub)
    results_path = sweep.run_sweep(tmp_path / "manifest.json", "fftradnet", grid, epochs=3,
                                    batch_size=2, seed=0, out_dir=out_dir)
    payload = json.loads(results_path.read_text())
    trials = payload["trials"]
    assert len(trials) == 3  # 2 stage1 + 1 stage2 (dup of the (2.0, 1e-4) winner -> also skipped)

    # Only the un-preseeded stage-1 combo actually hit the stub train().
    assert calls == [{"reg_weight": 2.0, "lr": 1e-4, "gamma": 2.0}]

    resumed = [t for t in trials if t["params"]["reg_weight"] == 1.0]
    assert len(resumed) == 1
    assert resumed[0]["resumed"] is True
    assert resumed[0]["final_val_AP"] == pytest.approx(0.55)
    assert resumed[0]["final_val_AR"] == pytest.approx(0.33)

    dup = [t for t in trials if t["stage"] == "stage2"][0]
    assert dup["resumed"] is True  # stage2's gamma=2.0 duplicates the (2.0, 1e-4) stage-1 winner
    assert dup["params"] == {"reg_weight": 2.0, "lr": 1e-4, "gamma": 2.0}


def test_run_sweep_unknown_model_raises(tmp_path):
    with pytest.raises(ValueError):
        sweep.run_sweep(tmp_path / "manifest.json", "not_a_real_model", out_dir=tmp_path)


# --------------------------------------------------------------------------------
# CLI: --dry-run, --grid-json, argument wiring
# --------------------------------------------------------------------------------
def test_dry_run_exits_zero_without_training(tmp_path, monkeypatch, capsys):
    def _explode(*args, **kwargs):
        raise AssertionError("train() must not be called in --dry-run")

    monkeypatch.setattr(sweep, "train", _explode)
    rc = sweep.main(["--manifest", str(tmp_path / "manifest.json"), "--model", "fftradnet",
                     "--dry-run", "--epochs", "5"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Stage 1" in out
    assert "Stage 2" in out
    assert "Total trials: 10" in out


def test_dry_run_grid_json_override(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sweep, "train", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    grid_json = json.dumps({"reg_weight": [5, 6], "lr": [1e-3], "stage2_gamma": [1.0]})
    rc = sweep.main(["--manifest", str(tmp_path / "manifest.json"), "--model", "ssmradnet",
                     "--dry-run", "--grid-json", grid_json])
    assert rc == 0
    out = capsys.readouterr().out
    assert "reg_weight=5" in out and "reg_weight=6" in out
    assert "Total trials: 3" in out  # 2 stage1 (2x1) + 1 stage2


def test_grid_json_missing_required_key_raises(tmp_path):
    bad_grid_json = json.dumps({"reg_weight": [1], "lr": [1e-4]})  # no stage2_gamma
    with pytest.raises(ValueError):
        sweep.main(["--manifest", str(tmp_path / "manifest.json"), "--model", "fftradnet",
                    "--dry-run", "--grid-json", bad_grid_json])


def test_cli_wires_run_sweep_arguments(tmp_path, monkeypatch):
    captured = {}

    def _fake_run_sweep(manifest_path, model_name, grid=None, **kwargs):
        captured["manifest_path"] = manifest_path
        captured["model_name"] = model_name
        captured["grid"] = grid
        captured.update(kwargs)
        return tmp_path / "sweep_results.json"

    monkeypatch.setattr(sweep, "run_sweep", _fake_run_sweep)
    rc = sweep.main(["--manifest", "dummy_manifest.json", "--model", "ssmradnet",
                     "--epochs", "7", "--batch-size", "4", "--seed", "3", "--out", str(tmp_path)])

    assert rc == 0
    assert captured["manifest_path"] == "dummy_manifest.json"
    assert captured["model_name"] == "ssmradnet"
    assert captured["grid"] == sweep.DEFAULT_GRID
    assert captured["epochs"] == 7
    assert captured["batch_size"] == 4
    assert captured["seed"] == 3
    assert captured["out_dir"] == str(tmp_path)


def test_cli_model_required_and_validated():
    with pytest.raises(SystemExit):
        sweep.build_arg_parser().parse_args(["--manifest", "m.json"])  # --model missing
    with pytest.raises(SystemExit):
        sweep.build_arg_parser().parse_args(["--manifest", "m.json", "--model", "nope"])
