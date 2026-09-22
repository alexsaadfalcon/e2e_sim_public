"""The two training knobs added for the generalisation campaign (owner, 2026-09-22, after
F85's out-of-distribution caveat): decoupled weight decay and joint training across
corpora. Each is pinned as (1) does what it says and (2) leaves the default path
bit-identical -- every existing checkpoint's provenance depends on the latter.
"""

import pytest

torch = pytest.importorskip("torch")

from e2e.ml import train as train_mod
from tests.test_ml_train import tiny_manifest_path  # noqa: F401  (pytest fixture)


def test_default_path_still_uses_plain_adam(tiny_manifest_path, tmp_path, monkeypatch):
    used = []
    real_adam, real_adamw = torch.optim.Adam, torch.optim.AdamW
    monkeypatch.setattr(torch.optim, "Adam", lambda *a, **k: used.append("Adam") or real_adam(*a, **k))
    monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **k: used.append("AdamW") or real_adamw(*a, **k))
    train_mod.train(tiny_manifest_path, "fftradnet", epochs=1, batch_size=2,
                    out_dir=tmp_path / "run", seed=0)
    assert used == ["Adam"]
    ckpt = torch.load(tmp_path / "run" / "best.pt", map_location="cpu")
    assert ckpt["train_config"]["weight_decay"] == 0.0
    assert ckpt["train_config"]["extra_manifests"] == []


def test_weight_decay_switches_to_adamw_and_is_recorded(tiny_manifest_path, tmp_path, monkeypatch):
    used = []
    real_adamw = torch.optim.AdamW

    def spy(params, lr, weight_decay):
        used.append(weight_decay)
        return real_adamw(params, lr=lr, weight_decay=weight_decay)

    monkeypatch.setattr(torch.optim, "AdamW", spy)
    train_mod.train(tiny_manifest_path, "fftradnet", epochs=1, batch_size=2,
                    out_dir=tmp_path / "run", seed=0, weight_decay=1e-4)
    assert used == [1e-4]
    ckpt = torch.load(tmp_path / "run" / "best.pt", map_location="cpu")
    assert ckpt["train_config"]["weight_decay"] == 1e-4


def test_extra_manifests_concatenate_train_splits_and_keep_val_on_the_primary(
        tiny_manifest_path, tmp_path, monkeypatch):
    """Joint training doubles the train set (same manifest passed twice here); validation
    still comes from the primary manifest alone."""
    from e2e.ml.dataset import RadarFrameDataset
    seen = {}
    real_loader = train_mod.DataLoader

    def spy(ds, *a, **kw):
        # The FIRST DataLoader built is the training one (shuffle=True); record its size.
        if kw.get("shuffle"):
            seen["train_len"] = len(ds)
        return real_loader(ds, *a, **kw)

    monkeypatch.setattr(train_mod, "DataLoader", spy)
    n_train = len(RadarFrameDataset(tiny_manifest_path, split="train"))
    train_mod.train(tiny_manifest_path, "fftradnet", epochs=1, batch_size=2,
                    out_dir=tmp_path / "run", seed=0, extra_manifests=[tiny_manifest_path])
    assert seen["train_len"] == 2 * n_train
    ckpt = torch.load(tmp_path / "run" / "best.pt", map_location="cpu")
    assert ckpt["train_config"]["extra_manifests"] == [str(tiny_manifest_path)]
    assert len(ckpt["history"]["val_AP"]) == 1
