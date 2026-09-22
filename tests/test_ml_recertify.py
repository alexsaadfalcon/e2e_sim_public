"""`e2e.ml.recertify`: measure whether a checkpoint still reproduces its own metric under
the current code, and only then re-stamp its pipeline fingerprint.

The fingerprint guard (F84) is blind to WHAT changed. This is the tool that turns an
inert refactor's false "stale" into evidence, and a real mismatch into a refusal.
"""

import pytest

torch = pytest.importorskip("torch")

from e2e.ml import recertify as recert_mod
from e2e.ml import train as train_mod
from e2e.ml.beat_cfar import _stale_reason
from tests.test_ml_train import tiny_manifest_path  # noqa: F401  (pytest fixture)


@pytest.fixture(scope="module")
def trained_run(tiny_manifest_path, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("recert") / "run"
    train_mod.train(tiny_manifest_path, "fftradnet", epochs=1, batch_size=2,
                    out_dir=out_dir, seed=0)
    return out_dir


def _rewrite(run_dir, **fields):
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu")
    ckpt.update(fields)
    torch.save(ckpt, run_dir / "best.pt")


def test_fresh_checkpoint_passes_and_is_not_rewritten(trained_run):
    r = recert_mod.recertify(trained_run)
    assert r["passed"] and not r["stamped"]
    assert abs(r["delta"]) <= r["tol"]
    assert _stale_reason(str(trained_run)) is None


def test_inert_change_is_cleared_by_measurement(trained_run):
    """A fingerprint that no longer matches -- but a metric that reproduces -- gets
    re-stamped, with the measurement recorded beside it."""
    _rewrite(trained_run, pipeline_fingerprint="0" * 64)
    assert _stale_reason(str(trained_run)) is not None
    r = recert_mod.recertify(trained_run)
    assert r["passed"] and r["stamped"]
    ckpt = torch.load(trained_run / "best.pt", map_location="cpu")
    assert ckpt["pipeline_fingerprint"] == train_mod.pipeline_fingerprint()
    assert ckpt["recertified"][-1]["reproduced_val_AP"] == pytest.approx(r["reproduced"])
    assert _stale_reason(str(trained_run)) is None


def test_dry_run_measures_but_never_writes(trained_run):
    _rewrite(trained_run, pipeline_fingerprint="1" * 64)
    r = recert_mod.recertify(trained_run, dry_run=True)
    assert r["passed"] and not r["stamped"]
    assert torch.load(trained_run / "best.pt", map_location="cpu")["pipeline_fingerprint"] == "1" * 64
    recert_mod.recertify(trained_run)   # restore a current stamp for the next test


def test_a_checkpoint_that_does_not_reproduce_its_metric_is_refused(trained_run):
    """The F84 case in miniature: the recorded number is not what reloading gives."""
    real = torch.load(trained_run / "best.pt", map_location="cpu")["best_val_AP"]
    _rewrite(trained_run, best_val_AP=float(real) + 0.5, pipeline_fingerprint="2" * 64)
    r = recert_mod.recertify(trained_run)
    assert not r["passed"] and not r["stamped"]
    ckpt = torch.load(trained_run / "best.pt", map_location="cpu")
    assert ckpt["pipeline_fingerprint"] == "2" * 64, "a failed certification writes nothing"
    assert recert_mod.main([str(trained_run)]) == 1
    _rewrite(trained_run, best_val_AP=float(real))


def test_missing_checkpoint_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        recert_mod.recertify(tmp_path)
