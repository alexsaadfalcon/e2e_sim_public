"""
Tests for `e2e.ml.compare_detectors` -- the matched-recall detector comparison.

The module's whole claim is that its number cannot be gamed by a threshold, so the tests
that matter are the ones where something tries: a saturating detector, an arm that never
reaches the target recall, and arms scored over different numbers of frames (which would
make false-alarms-PER-FRAME quietly incomparable).

The `false_alarms_at_recall` arithmetic itself is pinned in `test_ml_metrics.py`; here we
pin the plumbing around it -- argument parsing, the classical arm against a real tiny
corpus, the shared-frame requirement, and the CLI's refusal to call an empty comparison a
result.
"""

import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from e2e.ml import compare_detectors
from e2e.ml.labels import LabelGrid
from e2e.ml.radar_config import PRESETS


# ------------------------------------------------------------------------------------
# NAME=PATH parsing
# ------------------------------------------------------------------------------------
def test_parse_checkpoint_arg_splits_on_the_first_equals_only():
    # Display names carry spaces and parentheses; only the first '=' separates.
    name, path = compare_detectors.parse_checkpoint_arg(
        "FFTRadNet (old corpus)=report/rt_ml/kenney_d2/fftradnet_e30/best.pt")
    assert name == "FFTRadNet (old corpus)"
    assert path == "report/rt_ml/kenney_d2/fftradnet_e30/best.pt"


def test_parse_checkpoint_arg_accepts_a_bare_path():
    name, path = compare_detectors.parse_checkpoint_arg("runs/fftradnet_e30/best.pt")
    assert path == "runs/fftradnet_e30/best.pt"
    assert name == "fftradnet_e30"


@pytest.mark.parametrize("bad", ["=path", "name=", "="])
def test_parse_checkpoint_arg_rejects_a_half_empty_spec(bad):
    with pytest.raises(ValueError):
        compare_detectors.parse_checkpoint_arg(bad)


# ------------------------------------------------------------------------------------
# _score_arm -- the decode floor is a floor, not the operating point
# ------------------------------------------------------------------------------------
def _one_hot_map(grid: LabelGrid, cells, score: float):
    """A detection map with `score` at each `(range_bin, az_bin)` in `cells`.

    Channel layout matches `e2e.ml.labels.decode_detections`' contract: channel 0 is
    objectness, the rest are regressions left at zero (the cell centre).
    """
    m = torch.zeros(3, grid.n_range, grid.n_azimuth)
    for ri, ai in cells:
        m[0, ri, ai] = score
    return m


def test_score_arm_reports_an_operating_point_above_the_decode_floor():
    """A detector whose confidences all sit at 0.2 still gets an operating point at 0.2
    when decoded at a floor of 0.01 -- the floor must not be mistaken for the threshold."""
    grid = LabelGrid(n_range=32, n_azimuth=32, max_range_m=32.0)
    # One true target per frame at a cell the detector fires on, plus one it invents.
    pred_maps = [_one_hot_map(grid, [(10, 16), (20, 8)], 0.2) for _ in range(4)]
    target_lists = [[(10.5 * grid.range_bin_m, 0.0, "vehicle")] for _ in range(4)]

    res = compare_detectors._score_arm(pred_maps, target_lists, grid, n_frames=4,
                                       decode_threshold=0.01, target_recall=0.5)
    op = res["operating_point"]
    assert op["reached"] is True
    assert op["score_threshold"] == pytest.approx(0.2)
    assert op["score_threshold"] > res["decode_threshold"]


def test_score_arm_reports_not_reached_when_the_floor_clips_the_curve():
    """A decode floor ABOVE every confidence leaves no curve, so no operating point.

    This is the trap the module exists to avoid: silently reporting the false alarms at
    whatever recall a clipped curve managed would answer a different question.
    """
    grid = LabelGrid(n_range=32, n_azimuth=32, max_range_m=32.0)
    pred_maps = [_one_hot_map(grid, [(10, 16)], 0.05) for _ in range(4)]
    target_lists = [[(10.5 * grid.range_bin_m, 0.0, "vehicle")] for _ in range(4)]

    res = compare_detectors._score_arm(pred_maps, target_lists, grid, n_frames=4,
                                       decode_threshold=0.5, target_recall=0.5)
    assert res["operating_point"]["reached"] is False
    assert math.isnan(res["operating_point"]["fp_per_frame"])


# ------------------------------------------------------------------------------------
# compare() -- arm assembly, and the shared-frame requirement
# ------------------------------------------------------------------------------------
def _tiny_corpus(tmp_path, cfg):
    """A 4-frame corpus carrying raw ADC, so the classical arm has something to beamform.

    Deliberately hand-built rather than generated: integer codes round-trip byte-exact
    through `storage`'s int16 codec, and the content is irrelevant here -- only that
    every frame has `adc_code_re`/`adc_code_im` and a known target list.
    """
    from e2e.ml import dataset as ml_dataset
    from e2e.ml import storage
    from e2e.environment.scatterers import RadarPose, Scatterer
    from e2e.ml.labels import encode_detection_labels

    grid = LabelGrid.for_config(cfg)
    pose = RadarPose()
    d = tmp_path / "tiny_compare_corpus"
    d.mkdir()
    rng = np.random.default_rng(0)

    sequences = []
    for i in range(4):
        r = (25 + 10 * i + 0.5) * grid.range_bin_m
        sc = [Scatterer(position=(r, 0.0, 0.0), velocity=(0.0, 0.0, 0.0),
                        rcs_dbsm=20.0, object_class="vehicle")]
        labels = encode_detection_labels(grid, sc, pose, classes=("vehicle",))
        codes = rng.integers(-1000, 1000, size=(cfg.n_rx, cfg.n_chirps, cfg.n_samples))
        fname = f"frame_{i:05d}.npz"
        storage.write_sample_npz(
            d / fname, {"adc": codes.astype(np.complex64),
                        "labels": labels.cpu().numpy()},
            {"targets": [(r, 0.0, "vehicle")]}, payload_key="adc",
            full_scale=float(2 ** 15))
        sequences.append([fname])

    return ml_dataset.write_manifest(d, cfg, "test_tier", sequences, grid=grid,
                                     splits=(0.0, 1.0, 0.0))


def test_compare_refuses_an_empty_comparison():
    with pytest.raises(ValueError, match="nothing to compare"):
        compare_detectors.compare("unused.json", classical=False, checkpoints=())


def test_compare_scores_the_classical_arm_and_self_describes(tmp_path):
    cfg = PRESETS["ti_iwr1443"]
    manifest = _tiny_corpus(tmp_path, cfg)

    res = compare_detectors.compare(manifest, split="val", classical=True,
                                    target_recall=0.5, decode_threshold=0.01,
                                    device="cpu")
    # This corpus has NO train split, so the (default-on) null arm cannot fit its
    # box -- it must be a RECORDED skip, never a silent one or a crash.
    assert [a["name"] for a in res["arms"]] == ["classical CFAR"]
    assert "no targets" in res["null_skipped"]
    # The result carries the question it answered, so a stored JSON needs no context.
    assert res["target_recall"] == 0.5
    assert res["decode_threshold"] == 0.01
    assert res["split"] == "val"
    assert res["arms"][0]["operating_point"]["n_frames"] == 4


def test_null_arm_is_data_blind_boxed_and_deterministic(tmp_path):
    """B4 (from the B2 adversarial review): the chance floor every AP table needs.
    The null arm must (a) fit its box on the TRAIN split only, (b) score without
    touching the RF, (c) confine detections to the box, (d) reproduce exactly."""
    cfg = PRESETS["ti_iwr1443"]
    manifest = _tiny_corpus_with_train(tmp_path, cfg)

    a = compare_detectors.score_null(manifest, "val", decode_threshold=0.01,
                                     target_recall=0.5)
    b = compare_detectors.score_null(manifest, "val", decode_threshold=0.01,
                                     target_recall=0.5)
    assert a["AP"] == b["AP"] and a["n_detections"] == b["n_detections"]
    assert a["null_box"]["fit_split"] == "train"
    r_lo, r_hi = a["null_box"]["range_bins"]
    assert 0 <= r_lo < r_hi  # a real, nonempty box
    # And the default-on integration: a compare() on this corpus carries the arm.
    res = compare_detectors.compare(manifest, split="val", classical=True,
                                    target_recall=0.5, decode_threshold=0.01,
                                    device="cpu")
    assert [x["name"] for x in res["arms"]] == ["classical CFAR",
                                                "null (random-in-GT-box)"]
    assert "null_skipped" not in res


def _tiny_corpus_with_train(tmp_path, cfg):
    """`_tiny_corpus`, but with 2 of the 4 frames in the train split so the null
    arm has targets to fit its box on."""
    from e2e.ml import dataset as ml_dataset
    from e2e.ml import storage
    from e2e.environment.scatterers import RadarPose, Scatterer
    from e2e.ml.labels import encode_detection_labels

    grid = LabelGrid.for_config(cfg)
    pose = RadarPose()
    d = tmp_path / "tiny_compare_corpus_train"
    d.mkdir()
    rng = np.random.default_rng(0)
    sequences = []
    for i in range(4):
        r = (25 + 10 * i + 0.5) * grid.range_bin_m
        sc = [Scatterer(position=(r, 0.0, 0.0), velocity=(0.0, 0.0, 0.0),
                        rcs_dbsm=20.0, object_class="vehicle")]
        labels = encode_detection_labels(grid, sc, pose, classes=("vehicle",))
        codes = rng.integers(-1000, 1000, size=(cfg.n_rx, cfg.n_chirps, cfg.n_samples))
        fname = f"frame_{i:05d}.npz"
        storage.write_sample_npz(
            d / fname, {"adc": codes.astype(np.complex64),
                        "labels": labels.cpu().numpy()},
            {"targets": [(r, 0.0, "vehicle")]}, payload_key="adc",
            full_scale=float(2 ** 15))
        sequences.append([fname])
    return ml_dataset.write_manifest(d, cfg, "test_tier", sequences, grid=grid,
                                     splits=(0.5, 0.5, 0.0))


def test_compare_gives_every_arm_the_same_frames(tmp_path, monkeypatch):
    """`--limit` must reach EVERY arm. False alarms per frame divided by two different
    frame counts is not a comparison, and the denominator is invisible in the headline
    number, so a limit that applied to only one arm would be silently wrong."""
    cfg = PRESETS["ti_iwr1443"]
    manifest = _tiny_corpus(tmp_path, cfg)

    seen = {}

    def _fake_checkpoint(mpath, cpath, split, *, limit=None, **kw):
        seen["limit"] = limit
        return {"AP": 0.0, "AR_at_decode_floor": 0.0, "n_targets": 0,
                "n_detections": 0, "decode_threshold": 0.01,
                "operating_point": {"reached": False, "target_recall": 0.5,
                                    "recall_achieved": float("nan"), "recall_max": 0.0,
                                    "score_threshold": float("nan"), "fp": None,
                                    "tp": None, "fp_per_frame": float("nan"),
                                    "n_frames": limit or 0}}

    monkeypatch.setattr(compare_detectors, "score_checkpoint", _fake_checkpoint)

    res = compare_detectors.compare(manifest, split="val", classical=True,
                                    checkpoints=[("m", "ckpt.pt")], limit=2,
                                    device="cpu")
    assert seen["limit"] == 2                                  # reached the model arm
    assert res["arms"][0]["operating_point"]["n_frames"] == 2  # and the classical arm


# ------------------------------------------------------------------------------------
# Presentation and exit codes
# ------------------------------------------------------------------------------------
def _result(reached: bool):
    op = ({"reached": True, "target_recall": 0.5, "recall_achieved": 0.5,
           "recall_max": 0.9, "score_threshold": 0.2, "fp": 100, "tp": 5,
           "fp_per_frame": 25.0, "n_frames": 4}
          if reached else
          {"reached": False, "target_recall": 0.5, "recall_achieved": float("nan"),
           "recall_max": 0.1, "score_threshold": float("nan"), "fp": None, "tp": None,
           "fp_per_frame": float("nan"), "n_frames": 4})
    return {"manifest": "m.json", "split": "test", "target_recall": 0.5,
            "decode_threshold": 0.01,
            "arms": [{"name": "arm", "AP": 0.07, "operating_point": op}]}


def test_format_table_never_prints_a_number_for_an_unreached_arm():
    out = compare_detectors.format_table(_result(reached=False))
    assert "did not reach" in out
    assert "nan" not in out.lower().replace("nan'", "")  # no NaN leaking into the table


def test_format_table_prints_the_false_alarm_rate_when_reached():
    out = compare_detectors.format_table(_result(reached=True))
    assert "25.0" in out
    assert "did not reach" not in out


def test_main_exits_nonzero_when_no_arm_reached(monkeypatch, capsys, tmp_path):
    """An overnight pipeline must not mistake an empty comparison for a result."""
    monkeypatch.setattr(compare_detectors, "compare",
                        lambda *a, **k: _result(reached=False))
    out = tmp_path / "r.json"
    rc = compare_detectors.main(["--manifest", "m.json", "--classical",
                                 "--out", str(out)])
    assert rc == 2
    assert "NO ARM REACHED" in capsys.readouterr().err
    # It still writes the JSON: the evidence of the empty run is worth keeping.
    assert json.loads(out.read_text())["arms"][0]["operating_point"]["reached"] is False


def test_main_exits_zero_when_an_arm_reached(monkeypatch):
    monkeypatch.setattr(compare_detectors, "compare",
                        lambda *a, **k: _result(reached=True))
    assert compare_detectors.main(["--manifest", "m.json", "--classical"]) == 0
