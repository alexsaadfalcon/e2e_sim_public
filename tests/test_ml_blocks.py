"""
Tests for `e2e.ml.blocks` (SinkBlock / SourceBlock / NeuralDetectorBlock).

Synthetic only -- no Sionna, no trained checkpoints (anything needing a real
checkpoint is guarded/skipped, per the shard brief).
"""

import json

import pytest

torch = pytest.importorskip("torch")

from e2e import frames
from e2e.ml.blocks import NeuralDetectorBlock, SinkBlock, SourceBlock
from e2e.ml.models.ssmradnet import SSMRadNet


# --------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------
def _rx_time_state(torch_device, n_rx=4, n_chirp=6, n_samples=32, seed=0, **extra):
    g = torch.Generator(device="cpu").manual_seed(seed)
    adc = (torch.randn(n_rx, n_chirp, n_samples, generator=g)
          + 1j * torch.randn(n_rx, n_chirp, n_samples, generator=g))
    state = {"signal_domain": frames.DOMAIN_RX_TIME, "adc": adc.to(torch.complex64).to(torch_device)}
    state.update(extra)
    return state


# --------------------------------------------------------------------------------
# SinkBlock / SourceBlock
# --------------------------------------------------------------------------------
def test_sink_source_roundtrip_identical_tensor(tmp_path, torch_device):
    state = _rx_time_state(torch_device)
    sink = SinkBlock(tmp_path, tag="rt")
    out = sink.apply(state)
    assert out == {}   # never rewrites state

    src = SourceBlock(tmp_path, tag="rt")
    loaded = src.get_S_pars()
    assert torch.equal(loaded, state["adc"])
    assert loaded.device.type == torch_device.type
    assert src.signal_domain == frames.DOMAIN_RX_TIME


def test_two_sinks_different_tags_do_not_collide(tmp_path, torch_device):
    state_clean = _rx_time_state(torch_device, seed=1)
    state_sample = _rx_time_state(torch_device, seed=2)

    SinkBlock(tmp_path, tag="clean").apply(state_clean)
    SinkBlock(tmp_path, tag="sample").apply(state_sample)

    files = sorted(p.name for p in tmp_path.glob("*.npz"))
    assert files == ["clean_frame_00000.npz", "sample_frame_00000.npz"]

    clean = SourceBlock(tmp_path, tag="clean").get_S_pars()
    sample = SourceBlock(tmp_path, tag="sample").get_S_pars()
    assert torch.equal(clean, state_clean["adc"])
    assert torch.equal(sample, state_sample["adc"])
    assert not torch.equal(clean, sample)   # distinct content, not accidentally sharing a file

    # each artifact self-describes: tag/frame_idx/domain/shape all recoverable from meta
    import numpy as np

    for tag, expected_state in (("clean", state_clean), ("sample", state_sample)):
        with np.load(tmp_path / f"{tag}_frame_00000.npz") as data:
            meta = json.loads(str(data["meta"].item()))
        assert meta["tag"] == tag
        assert meta["frame_idx"] == 0
        assert meta["domain"] == frames.DOMAIN_RX_TIME
        assert meta["payload_key"] == "adc"
        assert meta["shape"] == list(expected_state["adc"].shape)


def test_sink_records_impairment_params_and_labels(tmp_path, torch_device):
    labels = torch.zeros(3, 4, 5, device=torch_device)
    state = _rx_time_state(torch_device, labels=labels,
                           impairment_params={"phase_noise": {"sigma": 0.1}})
    SinkBlock(tmp_path, tag="sample").apply(state)

    import numpy as np

    with np.load(tmp_path / "sample_frame_00000.npz") as data:
        meta = json.loads(str(data["meta"].item()))
        assert "labels" in data
    assert meta["impairment_params"] == {"phase_noise": {"sigma": 0.1}}

    src = SourceBlock(tmp_path, tag="sample")
    extra = src.get_state_updates()
    assert torch.equal(extra["labels"], labels)
    assert extra["impairment_params"] == {"phase_noise": {"sigma": 0.1}}


def test_sink_missing_payload_key_raises(tmp_path):
    sink = SinkBlock(tmp_path, tag="sample")
    with pytest.raises(KeyError):
        sink.apply({"signal_domain": frames.DOMAIN_RX_TIME})   # no 'adc' in state


def test_source_no_artifacts_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        SourceBlock(tmp_path, tag="nope")


def test_source_satisfies_environment_block_interface(tmp_path, torch_device):
    n_frames = 3
    sink = SinkBlock(tmp_path, tag="rt")   # one instance -> its own counter advances the filenames
    for i in range(n_frames):
        sink.apply(_rx_time_state(torch_device, seed=i))

    src = SourceBlock(tmp_path, tag="rt")
    assert hasattr(src, "get_S_pars") and callable(src.get_S_pars)
    assert hasattr(src, "step") and callable(src.step)
    assert hasattr(src, "reset") and callable(src.reset)
    assert src.signal_domain == frames.DOMAIN_RX_TIME

    assert src.frame_counter == 0
    frame0 = src.get_S_pars()
    src.step()
    assert src.frame_counter == 1
    frame1 = src.get_S_pars()
    assert not torch.equal(frame0, frame1)   # distinct frames (different seeds)

    src.step()
    src.step()   # wraps: 2 -> 0 with only 3 files (matches SionnaEnvironmentBlock.step)
    assert src.frame_counter == 0

    src.reset()
    assert src.frame_counter == 0


def test_two_sinks_at_two_stages_reflect_different_state(tmp_path, torch_device):
    """A chain that inserts SinkBlock twice (an intermediate stage + the final
    product stage) with the SAME tag semantics as the design notes' example --
    each call advances its OWN counter, and re-derives from whatever `state`
    holds at that point, so an upstream mutation between the two calls is
    correctly reflected in each artifact."""
    sink_clean = SinkBlock(tmp_path, tag="clean")
    sink_sample = SinkBlock(tmp_path, tag="sample")

    state = _rx_time_state(torch_device, seed=7)
    sink_clean.apply(state)               # "clean" artifact: pre-impairment
    state = dict(state)
    state["adc"] = state["adc"] * 2.0     # stand-in for an impairment/quantizer stage
    sink_sample.apply(state)              # "sample" artifact: post-impairment

    clean = SourceBlock(tmp_path, tag="clean").get_S_pars()
    sample = SourceBlock(tmp_path, tag="sample").get_S_pars()
    assert torch.allclose(sample, clean * 2.0)


# --------------------------------------------------------------------------------
# NeuralDetectorBlock
# --------------------------------------------------------------------------------
def _tiny_ssmradnet(torch_device, n_rx=4, n_chirp=6, n_samples=32,
                    n_range_out=8, n_azimuth_out=12):
    torch.manual_seed(0)
    model = SSMRadNet(2 * n_rx, n_samples, n_chirp, n_range_out, n_azimuth_out,
                      d_model=8, d_state=4, n_layers_fast=1, n_layers_slow=1,
                      head_channels=4, backend="torch", input_mode="adc")
    return model.to(torch_device)


def test_neural_detector_infer_emits_detections_and_leaves_adc_unchanged(torch_device):
    n_rx, n_chirp, n_samples = 4, 6, 32
    n_range_out, n_azimuth_out = 8, 12
    model = _tiny_ssmradnet(torch_device, n_rx, n_chirp, n_samples, n_range_out, n_azimuth_out)

    block = NeuralDetectorBlock(model, mode="infer", input_format="adc")
    state = _rx_time_state(torch_device, n_rx=n_rx, n_chirp=n_chirp, n_samples=n_samples)
    adc_before = state["adc"].clone()

    out = block.apply(state)

    assert "adc" not in out
    assert torch.equal(state["adc"], adc_before)   # left untouched

    assert "ml_detection" in out
    det = out["ml_detection"]
    assert det.shape == (3, n_range_out, n_azimuth_out)
    assert det.dtype == torch.float32
    # channel 0 is a sigmoid objectness probability
    assert torch.all(det[0] >= 0.0) and torch.all(det[0] <= 1.0)


def test_neural_detector_capabilities_match_docstring():
    caps = NeuralDetectorBlock.frame_capabilities
    assert caps.domain == frames.DOMAIN_RX_TIME
    assert caps.accepts_mimo is True
    assert caps.chirps == frames.CHIRP_NATIVE


def test_neural_detector_train_mode_requires_manifest_and_model_name(tmp_path):
    with pytest.raises(ValueError):
        NeuralDetectorBlock(mode="train")   # no manifest_path
    with pytest.raises(ValueError):
        NeuralDetectorBlock(mode="train", manifest_path=tmp_path)   # no model_name


def test_neural_detector_train_mode_apply_is_a_noop(tmp_path, torch_device):
    block = NeuralDetectorBlock(mode="train", manifest_path=tmp_path, model_name="ssmradnet")
    state = _rx_time_state(torch_device)
    adc_before = state["adc"].clone()
    out = block.apply(state)
    assert out == {}
    assert torch.equal(state["adc"], adc_before)
    assert block.model is None   # not built until fit() (which needs a real corpus -- skipped)


def test_neural_detector_infer_requires_model_or_ckpt():
    with pytest.raises(ValueError):
        NeuralDetectorBlock(mode="infer")


# --------------------------------------------------------------------------------
# Capability declarations (SinkBlock / SourceBlock)
# --------------------------------------------------------------------------------
def test_sink_block_capabilities_match_docstring(tmp_path):
    sink = SinkBlock(tmp_path, tag="sample", domain=frames.DOMAIN_RX_TIME)
    caps = frames.capabilities_of(sink)
    assert caps.domain == frames.DOMAIN_RX_TIME
    assert caps.accepts_mimo is True
    assert caps.chirps == frames.CHIRP_NATIVE

    sink_cfr = SinkBlock(tmp_path, tag="cfr", domain=frames.DOMAIN_CFR)
    assert frames.capabilities_of(sink_cfr).domain == frames.DOMAIN_CFR
