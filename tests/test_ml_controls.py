"""`e2e.ml.controls`: the reviewer controls run as one committed module.

The benchmark calibration (F83's numbers for b5_fftradnet_v3) needs the real corpus and is
recorded in the ledger, not here. These tests pin the MECHANICS on a tiny trained
checkpoint: every control is produced, a derangement really scores against other frames'
labels, and a constant map is really frame-independent.
"""

import pytest

torch = pytest.importorskip("torch")

from e2e.ml import controls as ctl
from tests.test_ml_recertify import tiny_manifest_path, trained_run  # noqa: F401  (fixtures)


def test_controls_produce_every_number(trained_run, tiny_manifest_path):
    r = ctl.controls_for(str(trained_run / "best.pt"), manifest=tiny_manifest_path,
                         split="val", max_range_m=None)
    for key in ("AP", "deranged_AP", "deranged_retention", "az_only_AP",
                "az_only_constant_AP", "range_only_AP", "range_only_constant_AP",
                "stripe_rank1"):
        assert key in r and isinstance(r[key], float), key
    assert 0.0 <= r["AP"] <= 1.0 and 0.0 <= r["deranged_AP"] <= 1.0
    assert 0.0 <= r["stripe_rank1"] <= 1.0


def test_derangement_is_a_cyclic_shift_of_targets(trained_run, tiny_manifest_path, monkeypatch):
    """Frame i must be scored against frame (i+1)'s labels -- never its own."""
    seen = []
    real = ctl._ap

    def spy(preds, targets, grid, **kw):
        seen.append(targets)
        return real(preds, targets, grid, **kw)

    monkeypatch.setattr(ctl, "_ap", spy)
    ctl.controls_for(str(trained_run / "best.pt"), manifest=tiny_manifest_path,
                     split="val", max_range_m=None)
    own, deranged = seen[0], seen[1]
    assert deranged == own[1:] + own[:1]
    assert len(own) >= 2


def test_constant_map_is_identical_for_every_frame(trained_run, tiny_manifest_path, monkeypatch):
    seen = []
    real = ctl._ap

    def spy(preds, targets, grid, **kw):
        seen.append(preds)
        return real(preds, targets, grid, **kw)

    monkeypatch.setattr(ctl, "_ap", spy)
    ctl.controls_for(str(trained_run / "best.pt"), manifest=tiny_manifest_path,
                     split="val", max_range_m=None)
    constant_calls = [p for p in seen if all(torch.equal(p[0], q) for q in p[1:])]
    assert len(constant_calls) == 2, "az-only and range-only constant-map scores"
    assert torch.allclose(constant_calls[0][0], torch.stack(seen[0]).mean(dim=0))


def test_stripe_is_one_for_a_rank_one_map():
    m = torch.outer(torch.rand(8), torch.rand(12))
    assert ctl._stripe([torch.stack([m, m * 0, m * 0])]) == pytest.approx(1.0, abs=1e-6)
