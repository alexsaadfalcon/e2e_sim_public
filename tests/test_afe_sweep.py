"""Tests for `e2e.ml.afe_sweep`: CPU-only, synthetic tiny cubes -- no real corpus/GPU.

`e2e.ml.dataset`/`labels`/`scenes`/`train` are sibling shards -- if any hasn't landed
yet, this whole module skips cleanly via the `importorskip` calls below (matching
`tests/test_ml_train.py`'s pattern).
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("e2e.ml.labels", reason="sibling shard e2e.ml.labels not present")
pytest.importorskip("e2e.ml.scenes", reason="sibling shard e2e.ml.scenes not present")

from e2e.ml import afe_sweep
from e2e.ml import dataset as ml_dataset
from e2e.ml import train as train_mod
from e2e.ml.radar_config import PRESETS, TI_IWR1443
from e2e.ml.scenes import DIFFICULTY_TIERS

TIER = sorted(DIFFICULTY_TIERS)[0]
CPU = torch.device("cpu")


# --------------------------------------------------------------------------------
# sensing_matrix / degrade_adc_cube -- pure-tensor, no dataset needed
# --------------------------------------------------------------------------------
def test_sensing_matrix_identity_control_at_m_equal_n():
    a = afe_sweep.sensing_matrix(6, 6, seed=0)
    assert torch.equal(a, torch.eye(6, dtype=torch.complex64))


def test_sensing_matrix_unit_norm_rows_when_m_below_n():
    a = afe_sweep.sensing_matrix(10, 4, seed=0)
    assert a.shape == (4, 10)
    norms = a.norm(dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_sensing_matrix_rejects_out_of_range_m():
    with pytest.raises(ValueError):
        afe_sweep.sensing_matrix(4, 5, seed=0)
    with pytest.raises(ValueError):
        afe_sweep.sensing_matrix(4, 0, seed=0)


def test_degrade_adc_cube_deterministic_per_seed():
    torch.manual_seed(0)
    adc = torch.randn(8, 3, 5, dtype=torch.complex64)
    a = afe_sweep.degrade_adc_cube(adc, 4, seed=7)
    b = afe_sweep.degrade_adc_cube(adc, 4, seed=7)
    c = afe_sweep.degrade_adc_cube(adc, 4, seed=8)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_degrade_adc_cube_m_equal_n_rx_is_bit_exact_pass_through():
    """The control row: M == n_rx uses the identity (see `sensing_matrix`), so
    degradation must reproduce the input exactly -- no RNG, no quantization loss."""
    torch.manual_seed(1)
    n_rx = 6
    adc = torch.randn(n_rx, 4, 7, dtype=torch.complex64)
    out = afe_sweep.degrade_adc_cube(adc, n_rx, seed=42, weight_bits=4)
    assert torch.equal(out, adc)


def test_degrade_adc_cube_shape_and_dtype_preserved():
    adc = torch.randn(8, 3, 5, dtype=torch.complex64)
    for m in (8, 5, 1):
        out = afe_sweep.degrade_adc_cube(adc, m, seed=0)
        assert out.shape == adc.shape
        assert out.dtype == adc.dtype


def test_degrade_adc_cube_energy_decreases_as_m_decreases():
    """Minimum-norm-LS reconstruction recovers only the row-space component of the
    aperture (see `e2e.chain.compress`'s module docstring): fewer measurements ->
    smaller row space -> less energy survives reconstruction, for a generic
    (isotropic random) cube. Checked as a monotonic trend across a fixed seed/cube,
    not asserted for every possible cube (a cube aligned with a particular sensing
    matrix's row space could, in principle, buck the trend)."""
    torch.manual_seed(3)
    n_rx = 10
    adc = torch.randn(n_rx, 4, 6, dtype=torch.complex64)
    m_list = [10, 8, 6, 4, 2, 1]
    energies = []
    for m in m_list:
        out = afe_sweep.degrade_adc_cube(adc, m, seed=0)
        energies.append(float((out.abs() ** 2).sum()))
    assert energies == sorted(energies, reverse=True)
    assert energies[0] == pytest.approx(float((adc.abs() ** 2).sum()), rel=1e-4)


def test_degrade_adc_cube_no_quantize_differs_from_quantized():
    torch.manual_seed(4)
    adc = torch.randn(8, 2, 4, dtype=torch.complex64)
    q = afe_sweep.degrade_adc_cube(adc, 3, seed=0, weight_bits=2, quantize=True)
    nq = afe_sweep.degrade_adc_cube(adc, 3, seed=0, quantize=False)
    assert not torch.equal(q, nq)


def test_degrade_adc_cube_rejects_bad_shape():
    with pytest.raises(ValueError):
        afe_sweep.degrade_adc_cube(torch.randn(4, 5, dtype=torch.complex64), 2)


# --------------------------------------------------------------------------------
# End-to-end fixture: tiny CPU corpus + one trained fftradnet checkpoint
# --------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_afe_fixture(tmp_path_factory):
    """Shrunk `ti_iwr1443` clone (12 chirps, 64 samples), 8 frames, trained 1 epoch on
    CPU -- mirrors `test_ml_train.py`'s `tiny_manifest_path` fixture, forced onto CPU
    (per this shard's "no real corpus/GPU" instruction) and module-scoped so the sweep
    tests below share one trained checkpoint instead of retraining per test.
    """
    cfg = dataclasses.replace(TI_IWR1443, name="test_afe_sweep_tiny_tdm", n_chirps=12, n_samples=64)
    PRESETS[cfg.name] = cfg
    try:
        dataset_dir = tmp_path_factory.mktemp("afe_sweep_dataset")
        manifest_path = ml_dataset.generate_dataset(
            cfg.name, TIER, 8, out_dir=dataset_dir, seed=0, device=CPU,
            splits=(0.5, 0.25, 0.25),
        )
        ckpt_dir = tmp_path_factory.mktemp("afe_sweep_ckpt")
        train_mod.train(manifest_path, "fftradnet", epochs=1, batch_size=2,
                        out_dir=ckpt_dir, seed=0, device=CPU)
        yield {"manifest_path": manifest_path, "ckpt_path": ckpt_dir / "best.pt", "cfg": cfg}
    finally:
        PRESETS.pop(cfg.name, None)


# --------------------------------------------------------------------------------
# _DegradedRadarFrameDataset: the eval-loop seam
# --------------------------------------------------------------------------------
def test_degraded_dataset_matches_direct_degrade_adc_cube(tiny_afe_fixture):
    manifest_path = tiny_afe_fixture["manifest_path"]
    n_rx = tiny_afe_fixture["cfg"].n_rx
    m = max(n_rx // 2, 1)

    plain = ml_dataset.RadarFrameDataset(manifest_path, split="val", input_format="adc")
    array0, is_adc0, _labels, _meta = plain._load_raw(0)
    assert is_adc0
    expected = afe_sweep.degrade_adc_cube(torch.from_numpy(array0).to(torch.complex64), m, seed=0)

    degraded = afe_sweep._DegradedRadarFrameDataset(manifest_path, "val", "adc", m=m, seed=0)
    got = degraded.raw_adc(0)
    assert torch.equal(got, expected)


def test_degraded_dataset_targets_are_not_degraded(tiny_afe_fixture):
    """Ground truth (`.targets()`) reads only the npz's "meta" entry -- degradation must
    never touch it, or the sweep would be scoring against a moving target."""
    manifest_path = tiny_afe_fixture["manifest_path"]
    n_rx = tiny_afe_fixture["cfg"].n_rx
    plain = ml_dataset.RadarFrameDataset(manifest_path, split="val", input_format="adc")
    degraded = afe_sweep._DegradedRadarFrameDataset(manifest_path, "val", "adc",
                                                     m=max(n_rx // 2, 1), seed=0)
    assert degraded.targets(0) == plain.targets(0)


# --------------------------------------------------------------------------------
# evaluate_at_m / run_afe_sweep: full plumbing on the tiny checkpoint
# --------------------------------------------------------------------------------
def test_evaluate_at_m_control_row_matches_undegraded_evaluate(tiny_afe_fixture):
    """M == n_rx (identity control) MUST reproduce the checkpoint's undegraded val_AP."""
    manifest_path = tiny_afe_fixture["manifest_path"]
    ckpt_path = tiny_afe_fixture["ckpt_path"]
    n_rx = tiny_afe_fixture["cfg"].n_rx

    undistorted = train_mod.evaluate(manifest_path, ckpt_path, split="val", device=CPU)
    r = afe_sweep.evaluate_at_m(manifest_path, ckpt_path, "val", n_rx, seed=0, device=CPU,
                                batch_size=2)

    assert r["m"] == n_rx
    assert r["model"]["AP"] == pytest.approx(undistorted["AP"], abs=1e-5)
    assert r["model"]["AR"] == pytest.approx(undistorted["AR"], abs=1e-5)
    assert r["model"]["range_rmse_m"] == pytest.approx(undistorted["range_rmse_m"], abs=1e-5)


def test_evaluate_at_m_returns_finite_classical_and_model_metrics(tiny_afe_fixture):
    manifest_path = tiny_afe_fixture["manifest_path"]
    ckpt_path = tiny_afe_fixture["ckpt_path"]
    n_rx = tiny_afe_fixture["cfg"].n_rx
    m = max(n_rx // 2, 1)

    r = afe_sweep.evaluate_at_m(manifest_path, ckpt_path, "val", m, seed=0, device=CPU,
                                batch_size=2)
    assert set(r.keys()) == {"m", "model", "classical"}
    for scorer in ("model", "classical"):
        for key in ("AP", "AR", "range_rmse_m"):
            assert math.isfinite(r[scorer][key])


def test_run_afe_sweep_end_to_end_schema_and_order(tiny_afe_fixture):
    manifest_path = tiny_afe_fixture["manifest_path"]
    ckpt_path = tiny_afe_fixture["ckpt_path"]
    n_rx = tiny_afe_fixture["cfg"].n_rx
    m_list = [n_rx, max(n_rx // 2, 1)]

    payload = afe_sweep.run_afe_sweep(manifest_path, ckpt_path, split="val", m_list=m_list,
                                      seed=0, device=CPU, batch_size=2)

    assert payload["model_name"] == "fftradnet"
    assert payload["n_rx"] == n_rx
    assert payload["m_list"] == m_list
    assert [r["m"] for r in payload["results"]] == m_list
    # the control row (M == n_rx) is first and exactly reproduces an undegraded eval
    assert payload["results"][0]["m"] == n_rx

    # JSON-serializable (as the CLI writes it), no NaN/inf sneaking through
    dumped = json.dumps(payload)
    reloaded = json.loads(dumped)
    assert reloaded["results"][0]["model"]["AP"] == payload["results"][0]["model"]["AP"]


def test_run_afe_sweep_rejects_m_above_n_rx(tiny_afe_fixture):
    manifest_path = tiny_afe_fixture["manifest_path"]
    ckpt_path = tiny_afe_fixture["ckpt_path"]
    n_rx = tiny_afe_fixture["cfg"].n_rx
    with pytest.raises(ValueError):
        afe_sweep.run_afe_sweep(manifest_path, ckpt_path, m_list=[n_rx + 1], device=CPU)


# --------------------------------------------------------------------------------
# Reporting: plot + markdown table
# --------------------------------------------------------------------------------
def test_plot_ap_vs_m_writes_a_file(tiny_afe_fixture, tmp_path):
    manifest_path = tiny_afe_fixture["manifest_path"]
    ckpt_path = tiny_afe_fixture["ckpt_path"]
    n_rx = tiny_afe_fixture["cfg"].n_rx

    payload = afe_sweep.run_afe_sweep(manifest_path, ckpt_path, split="val",
                                      m_list=[n_rx, max(n_rx // 2, 1)], seed=0, device=CPU,
                                      batch_size=2)
    fig_path = tmp_path / "nested" / "afe_sweep_ap.png"
    out = afe_sweep.plot_ap_vs_m(payload, fig_path)
    assert out == fig_path
    assert fig_path.is_file()
    assert fig_path.stat().st_size > 0


def test_print_markdown_table_runs_without_error(tiny_afe_fixture, capsys):
    manifest_path = tiny_afe_fixture["manifest_path"]
    ckpt_path = tiny_afe_fixture["ckpt_path"]
    n_rx = tiny_afe_fixture["cfg"].n_rx

    payload = afe_sweep.run_afe_sweep(manifest_path, ckpt_path, split="val",
                                      m_list=[n_rx], seed=0, device=CPU, batch_size=2)
    afe_sweep.print_markdown_table(payload)
    out = capsys.readouterr().out
    assert "AFE compression sweep" in out
    assert f"| {n_rx} |" in out


# --------------------------------------------------------------------------------
# CLI wiring
# --------------------------------------------------------------------------------
def test_build_arg_parser_defaults():
    args = afe_sweep.build_arg_parser().parse_args([
        "--manifest", "m.json", "--ckpt", "c.pt",
    ])
    assert args.m_list == list(afe_sweep.DEFAULT_M_LIST)
    assert args.split == "val"
    assert args.weight_bits == 8
    assert args.no_quantize is False
    assert args.out == "report/rt_ml/overnight_0811/afe_sweep/results.json"


def test_main_end_to_end_writes_json_and_figure(tiny_afe_fixture, tmp_path):
    manifest_path = tiny_afe_fixture["manifest_path"]
    ckpt_path = tiny_afe_fixture["ckpt_path"]
    n_rx = tiny_afe_fixture["cfg"].n_rx

    out_path = tmp_path / "results.json"
    rc = afe_sweep.main([
        "--manifest", str(manifest_path), "--ckpt", str(ckpt_path),
        "--model", "fftradnet", "--split", "val", "--device", "cpu",
        "--m-list", str(n_rx), str(max(n_rx // 2, 1)),
        "--batch-size", "2", "--out", str(out_path),
    ])
    assert rc == 0
    assert out_path.is_file()
    payload = json.loads(out_path.read_text())
    assert payload["m_list"] == [n_rx, max(n_rx // 2, 1)]
    assert (tmp_path / "afe_sweep_ap.png").is_file()


def test_main_warns_on_model_name_mismatch(tiny_afe_fixture, tmp_path, capsys):
    manifest_path = tiny_afe_fixture["manifest_path"]
    ckpt_path = tiny_afe_fixture["ckpt_path"]
    n_rx = tiny_afe_fixture["cfg"].n_rx

    rc = afe_sweep.main([
        "--manifest", str(manifest_path), "--ckpt", str(ckpt_path),
        "--model", "ssmradnet", "--device", "cpu",  # checkpoint is actually fftradnet
        "--m-list", str(n_rx), "--batch-size", "2", "--out", str(tmp_path / "r.json"),
    ])
    assert rc == 0
    assert "warning" in capsys.readouterr().out.lower()
