"""Thrust 5 in the GUI: replay a generated corpus frame and run a detector on it.

Owner ballot 2026-09-17: ONE Detection block, mode CFAR (tunable knobs) | ML
(pretrained checkpoint path). Before this, the webapp's detector block could only raise.

The end-to-end tests build a real (tiny) corpus with the same `SinkBlock` the generator
uses and a manifest in the generator's format, so `CorpusSourceBlock` is exercised on
exactly the artifact shape it will meet, not a mock of it.
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from e2e import frames
from e2e.ml.blocks import CFARDetectorBlock, CorpusSourceBlock, SinkBlock
from e2e.ml.labels import LabelGrid
from e2e.radar_config import RadarConfig

# Small enough for CFAR to run in well under a second, large enough for a 1-guard /
# 2-train annulus to fit inside the label grid.
_CFG = RadarConfig(name="tiny_corpus", f0_hz=77e9, bandwidth_hz=500e6, n_tx=1, n_rx=4,
                   n_chirps=8, n_samples=32, fs_hz=5e6, chirp_period_s=10e-6, mimo="single")
_GRID = LabelGrid(n_range=8, n_azimuth=12, max_range_m=_CFG.max_range_m)


def _tiny_corpus(tmp_path, n_frames=3):
    """A corpus directory + manifest in the generator's layout; returns the manifest path."""
    sink = SinkBlock(tmp_path, tag="sample")
    names = []
    for i in range(n_frames):
        g = torch.Generator(device="cpu").manual_seed(i)
        adc = (torch.randn(_CFG.n_rx, _CFG.n_chirps, _CFG.n_samples, generator=g)
               + 1j * torch.randn(_CFG.n_rx, _CFG.n_chirps, _CFG.n_samples, generator=g))
        labels = torch.zeros(3, _GRID.n_range, _GRID.n_azimuth)
        labels[0, 2 + i, 5] = 1.0          # one ground-truth target per frame
        sink.apply({"signal_domain": frames.DOMAIN_RX_TIME,
                    "adc": adc.to(torch.complex64), "labels": labels})
        names.append(f"sample_frame_{i:05d}.npz")
    manifest = {
        "config": _CFG.to_dict(),
        "grid": {"n_range": _GRID.n_range, "n_azimuth": _GRID.n_azimuth,
                 "max_range_m": _GRID.max_range_m},
        "files": {"train": [], "val": [], "test": names},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


# ------------------------------------------------------------------------------------
# CorpusSourceBlock
# ------------------------------------------------------------------------------------
def test_corpus_source_replays_manifest_order_with_labels(tmp_path):
    src = CorpusSourceBlock(_tiny_corpus(tmp_path), split="test")
    assert src.signal_domain == frames.DOMAIN_RX_TIME
    assert src.cfg.n_samples == _CFG.n_samples and src.grid.n_azimuth == _GRID.n_azimuth
    f0 = src.get_S_pars()
    assert tuple(f0.shape) == (_CFG.n_rx, _CFG.n_chirps, _CFG.n_samples)
    assert "labels" in src.get_state_updates()
    src.step()
    assert not torch.equal(src.get_S_pars(), f0)


def test_corpus_source_start_index_and_bounds(tmp_path):
    path = _tiny_corpus(tmp_path)
    src = CorpusSourceBlock(path, split="test", start=2)
    assert len(src) == 1
    with pytest.raises(IndexError):
        CorpusSourceBlock(path, split="test", start=3)
    with pytest.raises(KeyError):
        CorpusSourceBlock(path, split="nope")


# ------------------------------------------------------------------------------------
# CFARDetectorBlock
# ------------------------------------------------------------------------------------
def test_cfar_block_emits_map_detections_and_ground_truth(torch_device):
    g = torch.Generator(device="cpu").manual_seed(0)
    adc = (torch.randn(_CFG.n_rx, _CFG.n_chirps, _CFG.n_samples, generator=g)
           + 1j * torch.randn(_CFG.n_rx, _CFG.n_chirps, _CFG.n_samples, generator=g))
    labels = torch.zeros(3, _GRID.n_range, _GRID.n_azimuth)
    labels[0, 3, 4] = 1.0
    state = {"signal_domain": frames.DOMAIN_RX_TIME,
             "adc": adc.to(torch.complex64).to(torch_device), "labels": labels}
    out = CFARDetectorBlock(_CFG, _GRID, guard=1, train=2, threshold=0.5).apply(state)
    det = out["cfar_detection"]
    assert tuple(det.shape) == (3, _GRID.n_range, _GRID.n_azimuth)
    assert torch.all(det[0] >= 0) and torch.all(det[0] <= 1)
    assert torch.all(det[1:] == 0), "a CFAR has no sub-cell regression"
    assert isinstance(out["cfar_detections"], list)
    assert len(out["gt_detections"]) == 1
    assert "adc" not in out


def test_cfar_block_omits_ground_truth_when_the_frame_has_none(torch_device):
    adc = torch.zeros(_CFG.n_rx, _CFG.n_chirps, _CFG.n_samples, dtype=torch.complex64)
    out = CFARDetectorBlock(_CFG, _GRID, guard=1, train=2).apply(
        {"signal_domain": frames.DOMAIN_RX_TIME, "adc": adc.to(torch_device)})
    assert "gt_detections" not in out


def test_cfar_block_rejects_degenerate_annulus():
    with pytest.raises(ValueError):
        CFARDetectorBlock(_CFG, _GRID, guard=0, train=2)


# ------------------------------------------------------------------------------------
# Registry / diagram
# ------------------------------------------------------------------------------------
def test_detector_block_has_both_modes_and_a_checkpoint_path():
    from webapp.pipeline_registry import BLOCKS_BY_ID
    spec = BLOCKS_BY_ID["detector"]
    keys = {p.key: p for p in spec.params}
    assert keys["mode"].choices == ["cfar", "ml"]
    assert keys["checkpoint"].kind == "text"
    assert {"cfar_guard", "cfar_train", "threshold"} <= set(keys)


def test_corpus_source_is_registered_and_drawn():
    from webapp import block_diagram
    from webapp.pipeline_registry import BLOCKS_BY_ID, default_block_state
    assert BLOCKS_BY_ID["corpus_environment"].category == "source"
    assert BLOCKS_BY_ID["corpus_environment"].enabled_default is False
    elements = block_diagram.build_elements(default_block_state())
    ids = {e["data"]["id"] for e in elements if "id" in e.get("data", {})}
    assert "corpus_environment" in ids


def test_param_editor_renders_text_kind_as_text_input():
    from webapp import block_diagram
    from webapp.pipeline_registry import default_block_state
    children = block_diagram.param_editor("detector", default_block_state())
    inputs = [c for c in children if getattr(c, "type", None) == "text"]
    assert inputs, "the checkpoint path must render as a text input"


# ------------------------------------------------------------------------------------
# run_pipeline end to end, corpus replay -> radar cube + CFAR detector -> figures
# ------------------------------------------------------------------------------------
def _corpus_state(manifest_path, **detector_params):
    from webapp.pipeline_registry import default_block_state
    st = default_block_state()
    st["corpus_environment"]["enabled"] = True
    st["corpus_environment"]["params"]["manifest"] = str(manifest_path)
    st["radar_cube"]["enabled"] = True
    st["detector"]["enabled"] = True
    st["detector"]["params"].update({"cfar_guard": 1, "cfar_train": 2, **detector_params})
    return st


def test_run_pipeline_replays_corpus_into_cube_and_cfar(tmp_path):
    from webapp.pipeline_runner import figures_from_outputs, run_pipeline
    outputs = run_pipeline(_corpus_state(_tiny_corpus(tmp_path)), n_steps=2)
    assert len(outputs["radar_cube"]) == 2
    assert len(outputs["cfar_detection"]) == 2
    assert len(outputs["gt_detections"]) == 2 and len(outputs["gt_detections"][0]) == 1
    assert "fft" not in outputs and "range_az" not in outputs, \
        "frequency-domain products must not run on a replayed ADC frame"
    rxm = outputs["_axis_meta"]["rx"]
    assert rxm["grid"]["n_azimuth"] == _GRID.n_azimuth
    assert rxm["max_range_m"] == pytest.approx(_CFG.max_range_m)

    figs = figures_from_outputs(outputs)
    assert "radar_cube" in figs and "cfar_detection" in figs
    det_fig = figs["cfar_detection"]
    names = [t.name for t in det_fig.data]
    assert any(n.startswith("ground truth") for n in names)
    # Physical axes: the objectness heatmap spans the label grid's range and sin(az).
    hm = det_fig.data[0]
    assert float(np.max(hm.y)) < _GRID.max_range_m and float(np.min(hm.x)) > -1.0


def test_run_pipeline_corpus_missing_manifest_is_a_friendly_error(tmp_path):
    from webapp.pipeline_runner import PipelineError, run_pipeline
    with pytest.raises(PipelineError, match="manifest not found"):
        run_pipeline(_corpus_state(tmp_path / "nope.json"), n_steps=1)


def test_run_pipeline_ml_mode_without_checkpoint_is_a_friendly_error(tmp_path):
    from webapp.pipeline_runner import PipelineError, run_pipeline
    st = _corpus_state(_tiny_corpus(tmp_path), mode="ml", checkpoint="")
    with pytest.raises(PipelineError, match="checkpoint"):
        run_pipeline(st, n_steps=1)


def test_run_pipeline_ml_mode_with_missing_checkpoint_names_the_path(tmp_path):
    from webapp.pipeline_runner import PipelineError, run_pipeline
    st = _corpus_state(_tiny_corpus(tmp_path), mode="ml", checkpoint=str(tmp_path / "x.pt"))
    with pytest.raises(PipelineError, match="x.pt"):
        run_pipeline(st, n_steps=1)


def test_run_pipeline_refuses_two_sources(tmp_path):
    from webapp.pipeline_runner import PipelineError, run_pipeline
    st = _corpus_state(_tiny_corpus(tmp_path))
    st["rt_environment"]["enabled"] = True
    with pytest.raises(PipelineError, match="one, not both"):
        run_pipeline(st, n_steps=1)


def test_detector_without_a_cube_source_is_a_friendly_error():
    """Detector on the precomputed .pkl source: no ADC cube exists to detect on."""
    from webapp.pipeline_registry import default_block_state
    from webapp.pipeline_runner import PipelineError, run_pipeline
    st = default_block_state()
    st["detector"]["enabled"] = True
    with pytest.raises(PipelineError, match="ADC cube"):
        run_pipeline(st, n_steps=1)
