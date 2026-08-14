"""Tests for `e2e.ml.compressed_domain`: CPU-only, synthetic tiny cubes -- no real
corpus/GPU (mirrors `tests/test_afe_sweep.py`'s pattern/fixtures).
"""
from __future__ import annotations

import dataclasses
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("e2e.ml.labels", reason="sibling shard e2e.ml.labels not present")
pytest.importorskip("e2e.ml.scenes", reason="sibling shard e2e.ml.scenes not present")

from e2e.ml import afe_sweep
from e2e.ml import compressed_domain as cd
from e2e.ml import dataset as ml_dataset
from e2e.ml import train as train_mod
from e2e.ml.radar_config import PRESETS, TI_IWR1443
from e2e.ml.scenes import DIFFICULTY_TIERS

TIER = sorted(DIFFICULTY_TIERS)[0]
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def tiny_cd_fixture(tmp_path_factory):
    """Shrunk `ti_iwr1443` clone, 8 frames, no checkpoint trained yet -- just the
    manifest/corpus, shared by every test below (each test trains its own tiny
    checkpoint(s) on top of it, cheap at this size)."""
    cfg = dataclasses.replace(TI_IWR1443, name="test_cd_tiny", n_chirps=12, n_samples=64)
    PRESETS[cfg.name] = cfg
    try:
        dataset_dir = tmp_path_factory.mktemp("cd_dataset")
        manifest_path = ml_dataset.generate_dataset(
            cfg.name, TIER, 8, out_dir=dataset_dir, seed=0, device=CPU,
            splits=(0.5, 0.25, 0.25),
        )
        yield {"manifest_path": manifest_path, "cfg": cfg}
    finally:
        PRESETS.pop(cfg.name, None)


# --------------------------------------------------------------------------------
# train_native_m
# --------------------------------------------------------------------------------
def test_train_native_m_writes_checkpoint_sized_to_m(tiny_cd_fixture, tmp_path):
    manifest_path = tiny_cd_fixture["manifest_path"]
    n_rx = tiny_cd_fixture["cfg"].n_rx
    m = max(n_rx // 2, 1)
    assert m != n_rx

    out_dir = tmp_path / "native_m_run"
    r = cd.train_native_m(manifest_path, m, epochs=1, batch_size=2, seed=0, device=CPU,
                          out_dir=out_dir)

    assert r["m"] == m
    for key in ("AP", "AR", "range_rmse_m"):
        assert key in r["model"]
        assert r["model"][key] == r["model"][key]  # not NaN
    assert (out_dir / "best.pt").is_file()
    assert (out_dir / "history.json").is_file()

    ckpt = torch.load(out_dir / "best.pt", map_location=CPU)
    assert ckpt["m"] == m
    assert ckpt["no_reconstruct"] is True
    assert ckpt["model_name"] == "fftradnet"

    # the checkpoint really is sized to M, not n_rx: rebuilding a full-aperture model
    # (n_rx unmodified) and loading this state dict must fail (shape mismatch).
    with open(manifest_path) as f:
        manifest = json.load(f)
    full_manifest = afe_sweep._manifest_at_m(manifest, n_rx, ckpt["input_format"])
    full_model = train_mod.build_model("fftradnet", full_manifest, device=CPU)
    with pytest.raises(RuntimeError):
        full_model.load_state_dict(ckpt["model_state"])

    # ... but an M-sized model loads it cleanly.
    m_manifest = afe_sweep._manifest_at_m(manifest, m, ckpt["input_format"])
    m_model = train_mod.build_model("fftradnet", m_manifest, device=CPU)
    m_model.load_state_dict(ckpt["model_state"])  # no raise


def test_train_native_m_history_matches_epoch_count(tiny_cd_fixture, tmp_path):
    manifest_path = tiny_cd_fixture["manifest_path"]
    n_rx = tiny_cd_fixture["cfg"].n_rx
    m = max(n_rx // 2, 1)

    r = cd.train_native_m(manifest_path, m, epochs=2, batch_size=2, seed=0, device=CPU,
                          out_dir=tmp_path / "run2")
    assert r["history"]["epoch"] == [1, 2]
    assert len(r["history"]["val_AP"]) == 2


# --------------------------------------------------------------------------------
# run_compressed_domain_grid
# --------------------------------------------------------------------------------
def test_run_compressed_domain_grid_schema(tiny_cd_fixture, tmp_path):
    manifest_path = tiny_cd_fixture["manifest_path"]
    n_rx = tiny_cd_fixture["cfg"].n_rx
    m_list = [n_rx, max(n_rx // 2, 1)]

    payload = cd.run_compressed_domain_grid(manifest_path, m_list=m_list, epochs=1,
                                            batch_size=2, seed=0, device=CPU,
                                            out_root=tmp_path / "grid")
    assert payload["m_list"] == m_list
    assert len(payload["results"]) == len(m_list)
    for m, r in zip(m_list, payload["results"]):
        assert r["m"] == m
        for scorer in ("native_model", "classical_reconstructed"):
            for key in ("AP", "AR", "range_rmse_m"):
                assert r[scorer][key] == r[scorer][key]  # finite, not NaN

    # JSON-serializable, as the CLI writes it
    dumped = json.dumps(payload)
    reloaded = json.loads(dumped)
    assert reloaded["m_list"] == m_list


# --------------------------------------------------------------------------------
# zero_shot_identity_probe (arm 2)
# --------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def full_aperture_ckpt(tiny_cd_fixture, tmp_path_factory):
    manifest_path = tiny_cd_fixture["manifest_path"]
    ckpt_dir = tmp_path_factory.mktemp("cd_full_ckpt")
    train_mod.train(manifest_path, "fftradnet", epochs=1, batch_size=2, out_dir=ckpt_dir,
                    seed=0, device=CPU)
    return ckpt_dir / "best.pt"


def test_zero_shot_identity_probe_matches_undegraded_eval(tiny_cd_fixture, full_aperture_ckpt):
    manifest_path = tiny_cd_fixture["manifest_path"]
    n_rx = tiny_cd_fixture["cfg"].n_rx

    r = cd.zero_shot_identity_probe(manifest_path, full_aperture_ckpt, seed=0, device=CPU)
    assert r["m"] == n_rx
    assert r["matches_undistorted"] is True
    assert r["model"]["AP"] == pytest.approx(r["undistorted_AP"], abs=1e-5)


# --------------------------------------------------------------------------------
# reporting helpers
# --------------------------------------------------------------------------------
def test_reconstruct_then_detect_ap_reads_existing_sweep_json(tmp_path):
    fake_sweep = {
        "results": [
            {"m": 16, "model": {"AP": 0.04}},
            {"m": 12, "model": {"AP": 0.001}},
            {"m": 8, "model": {"AP": 0.0}},
        ]
    }
    path = tmp_path / "fftradnet_results.json"
    path.write_text(json.dumps(fake_sweep))

    by_m = cd._reconstruct_then_detect_ap(path, [16, 12, 4])
    assert by_m == {16: 0.04, 12: 0.001}  # 4 absent from the sweep file, silently dropped
    assert 8 not in by_m  # not requested, even though present in the file


def test_plot_ap_vs_m_writes_a_file(tiny_cd_fixture, tmp_path):
    manifest_path = tiny_cd_fixture["manifest_path"]
    n_rx = tiny_cd_fixture["cfg"].n_rx
    m_list = [n_rx, max(n_rx // 2, 1)]

    payload = cd.run_compressed_domain_grid(manifest_path, m_list=m_list, epochs=1,
                                            batch_size=2, seed=0, device=CPU,
                                            out_root=tmp_path / "grid")
    fig_path = tmp_path / "nested" / "ap_vs_m.png"
    out = cd.plot_ap_vs_m(payload, fig_path,
                          reconstruct_then_detect={n_rx: 0.05})
    assert out == fig_path
    assert fig_path.is_file()
    assert fig_path.stat().st_size > 0


def test_markdown_summary_runs_without_error(tiny_cd_fixture, tmp_path):
    manifest_path = tiny_cd_fixture["manifest_path"]
    n_rx = tiny_cd_fixture["cfg"].n_rx
    m_list = [n_rx]

    payload = cd.run_compressed_domain_grid(manifest_path, m_list=m_list, epochs=1,
                                            batch_size=2, seed=0, device=CPU,
                                            out_root=tmp_path / "grid")
    text = cd.markdown_summary(payload, zero_shot=None, reconstruct_then_detect=None)
    assert "Compressed-domain-v1" in text
    assert "impossibility" in text.lower() or "cannot consume" in text.lower()
    assert f"| {n_rx} |" in text


def test_markdown_summary_with_zero_shot(tiny_cd_fixture, full_aperture_ckpt, tmp_path):
    manifest_path = tiny_cd_fixture["manifest_path"]
    n_rx = tiny_cd_fixture["cfg"].n_rx

    zero_shot = cd.zero_shot_identity_probe(manifest_path, full_aperture_ckpt, seed=0, device=CPU)
    payload = cd.run_compressed_domain_grid(manifest_path, m_list=[n_rx], epochs=1,
                                            batch_size=2, seed=0, device=CPU,
                                            out_root=tmp_path / "grid")
    text = cd.markdown_summary(payload, zero_shot=zero_shot, reconstruct_then_detect=None)
    assert "permutation-free" in text.lower()


# --------------------------------------------------------------------------------
# CLI wiring
# --------------------------------------------------------------------------------
def test_build_arg_parser_defaults():
    args = cd.build_arg_parser().parse_args(["--manifest", "m.json"])
    assert args.m_list == list(cd.DEFAULT_M_LIST)
    assert args.model == "fftradnet"
    assert args.epochs == 15
    assert args.batch_size == 4
    assert args.lr == pytest.approx(3e-4)
    assert args.out == "report/rt_ml/compressed_domain_v1"


def test_main_end_to_end_tiny(tiny_cd_fixture, full_aperture_ckpt, tmp_path):
    manifest_path = tiny_cd_fixture["manifest_path"]
    n_rx = tiny_cd_fixture["cfg"].n_rx
    m = max(n_rx // 2, 1)
    out_dir = tmp_path / "cd_out"

    rc = cd.main([
        "--manifest", str(manifest_path), "--full-aperture-ckpt", str(full_aperture_ckpt),
        "--m-list", str(n_rx), str(m), "--epochs", "1", "--batch-size", "2",
        "--device", "cpu", "--out", str(out_dir),
    ])
    assert rc == 0
    assert (out_dir / "results.json").is_file()
    assert (out_dir / "ap_vs_m.png").is_file()
    assert (out_dir / "summary.md").is_file()

    payload = json.loads((out_dir / "results.json").read_text())
    assert payload["m_list"] == [n_rx, m]
    assert payload["zero_shot_probe"]["matches_undistorted"] is True
