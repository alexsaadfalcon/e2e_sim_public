"""The network input scale is ONE number resolved by ONE function, for training and the GUI.

The 2026-09-22 defect: `NeuralDetectorBlock` derived its input without the manifest's
`input_scale` while `RadarFrameDataset` divided by it, so the GUI fed a checkpoint inputs
on a different scale from training (objectness peak 0.12 vs 0.49 on the same frame; the
demo's ML preset drew nothing). These tests pin the resolver's rules and that the block
actually divides by what it resolves.
"""

import pytest

torch = pytest.importorskip("torch")

from e2e import frames
from e2e.ml.blocks import NeuralDetectorBlock
from e2e.ml.dataset import resolve_input_scale


# ------------------------------------------------------------------------------------
# resolve_input_scale: the rules the dataset always had, now callable
# ------------------------------------------------------------------------------------
def test_per_format_scale_is_used_for_its_format():
    m = {"input_scale_by_format": {"rd": 4.0, "adc": 0.5}, "input_scale": 4.0}
    assert resolve_input_scale(m, "rd") == 4.0
    assert resolve_input_scale(m, "adc") == 0.5


def test_a_format_the_manifest_does_not_list_is_refused_not_borrowed():
    """F80: another format's constant rescales the input by orders of magnitude."""
    with pytest.raises(ValueError, match="not for input_format='adc'"):
        resolve_input_scale({"input_scale_by_format": {"rd": 4.0}}, "adc")


def test_legacy_single_scale_is_rd_only():
    assert resolve_input_scale({"input_scale": 3.0}, "rd") == 3.0
    with pytest.raises(ValueError, match="measured for input_format='rd'"):
        resolve_input_scale({"input_scale": 3.0}, "adc")


def test_rad_is_self_normalising_and_pinned_at_one():
    assert resolve_input_scale({"input_scale_by_format": {"rd": 4.0}}, "rad") == 1.0
    assert resolve_input_scale({"input_scale": 3.0}, "rad") == 1.0


def test_a_corpus_predating_the_constant_gets_one():
    assert resolve_input_scale({}, "rd") == 1.0 and resolve_input_scale({}, "adc") == 1.0


def test_non_positive_scale_is_refused():
    with pytest.raises(ValueError, match="non-positive"):
        resolve_input_scale({"input_scale_by_format": {"rd": 0.0}}, "rd")


def test_dataset_and_resolver_agree(tmp_path):
    """The dataset must be USING the resolver, not a copy of its rules."""
    import json
    from e2e.ml.dataset import RadarFrameDataset
    manifest = {"files": {"test": []}, "input_scale_by_format": {"rd": 7.5, "adc": 2.5},
                "grid": {"n_range": 4, "n_azimuth": 4, "max_range_m": 10.0}}
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))
    try:
        ds = RadarFrameDataset(p, split="test", input_format="adc")
    except TypeError:
        pytest.skip("installed RadarFrameDataset predates input_format")
    assert ds.input_scale == resolve_input_scale(manifest, "adc") == 2.5


# ------------------------------------------------------------------------------------
# The block divides by what it resolves
# ------------------------------------------------------------------------------------
class _Identity(torch.nn.Module):
    """A 'model' whose detection output is a fixed map; only the input path is under test."""
    def forward(self, x):
        b = x.shape[0]
        return {"detection": torch.zeros(b, 3, 4, 4, device=x.device)}


def test_block_divides_adc_input_by_its_scale(torch_device):
    blk = NeuralDetectorBlock(_Identity(), mode="infer", input_format="adc", device=torch_device)
    assert blk.input_scale == 1.0, "an nn.Module has no manifest: scale stays 1.0"
    adc = (torch.randn(2, 3, 8) + 1j * torch.randn(2, 3, 8)).to(torch.complex64)
    unscaled = blk._derive_input(adc)
    blk.input_scale = 4.0
    assert torch.allclose(blk._derive_input(adc), unscaled / 4.0)


def test_block_and_dataset_derive_the_same_tensor_for_every_format(torch_device):
    """One function, both callers: the GUI must feed a checkpoint exactly what training
    fed it, for rd, adc AND rad (the format the architecture that beat CFAR uses)."""
    from e2e.ml.dataset import derive_network_input
    from e2e.radar_config import RadarConfig
    cfg = RadarConfig(name="t", f0_hz=77e9, bandwidth_hz=500e6, n_tx=1, n_rx=4, n_chirps=8,
                      n_samples=32, fs_hz=5e6, chirp_period_s=10e-6, mimo="single")
    g = torch.Generator(device="cpu").manual_seed(0)
    adc = (torch.randn(4, 8, 32, generator=g) + 1j * torch.randn(4, 8, 32, generator=g)).to(torch.complex64)
    for fmt in ("rd", "adc", "rad"):
        blk = NeuralDetectorBlock(_Identity(), mode="infer", input_format=fmt, cfg=cfg,
                                  device=torch_device)
        blk.input_scale = 2.0
        expected = derive_network_input(cfg, adc, fmt) / 2.0
        assert torch.allclose(blk._derive_input(adc).cpu(), expected.cpu(), atol=1e-6), fmt


def test_block_needs_cfg_for_the_beamformed_formats(torch_device):
    blk = NeuralDetectorBlock(_Identity(), mode="infer", input_format="rad", device=torch_device)
    with pytest.raises(ValueError, match="needs cfg"):
        blk._derive_input(torch.zeros(2, 3, 8, dtype=torch.complex64))
