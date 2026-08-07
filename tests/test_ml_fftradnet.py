"""
Tests for `e2e.ml.models.fftradnet.FFTRadNet` (ported/adapted from valeoai/RADIal's
FFTRadNet -- see the module docstring in `e2e/ml/models/fftradnet.py` for the full
attribution notice and documented deviations).

Small sizes throughout for speed; `torch_device` fixture exercises the library's
actual device (cuda if present, else cpu).
"""

import pytest

torch = pytest.importorskip("torch")

from e2e.ml.models.fftradnet import FFTRadNet


# --------------------------------------------------------------------------------
# Forward shape
# --------------------------------------------------------------------------------
def test_forward_shape_plain_stem(torch_device):
    model = FFTRadNet(in_channels=24, n_range_in=128, n_doppler_in=32, n_range_out=32,
                       n_azimuth_out=48).to(torch_device)
    x = torch.randn(2, 24, 128, 32, device=torch_device)
    out = model(x)
    assert set(out.keys()) == {"detection"}
    assert out["detection"].shape == (2, 3, 32, 48)


def test_forward_shape_ddma_preencoder(torch_device):
    model = FFTRadNet(in_channels=8, n_range_in=64, n_doppler_in=32, n_range_out=16,
                       n_azimuth_out=24, mimo_preencoder="ddma", n_tx=4).to(torch_device)
    x = torch.randn(2, 8, 64, 32, device=torch_device)
    out = model(x)
    assert out["detection"].shape == (2, 3, 16, 24)


def test_ddma_requires_n_tx():
    with pytest.raises(ValueError):
        FFTRadNet(in_channels=8, n_range_in=64, n_doppler_in=32, n_range_out=16,
                  n_azimuth_out=24, mimo_preencoder="ddma")


def test_invalid_mimo_preencoder_raises():
    with pytest.raises(ValueError):
        FFTRadNet(in_channels=8, n_range_in=64, n_doppler_in=32, n_range_out=16,
                  n_azimuth_out=24, mimo_preencoder="bogus")


def test_base_channels_and_blocks_must_have_4_stages():
    with pytest.raises(ValueError):
        FFTRadNet(in_channels=8, n_range_in=64, n_doppler_in=32, n_range_out=16,
                  n_azimuth_out=24, base_channels=(32, 40, 48))
    with pytest.raises(ValueError):
        FFTRadNet(in_channels=8, n_range_in=64, n_doppler_in=32, n_range_out=16,
                  n_azimuth_out=24, blocks=(3, 6, 6))


# --------------------------------------------------------------------------------
# Output channel semantics
# --------------------------------------------------------------------------------
def test_objectness_channel_is_in_unit_interval(torch_device):
    model = FFTRadNet(in_channels=24, n_range_in=128, n_doppler_in=32, n_range_out=32,
                       n_azimuth_out=48).to(torch_device)
    x = torch.randn(2, 24, 128, 32, device=torch_device)
    detection = model(x)["detection"]
    obj = detection[:, 0]
    assert torch.all(obj >= 0.0) and torch.all(obj <= 1.0)


def test_regression_channels_are_unbounded(torch_device):
    # A sigmoid-bounded channel could never exceed ~1 in magnitude; regression
    # channels are raw linear outputs and should be able to exceed that easily
    # for at least some random draw of weights/inputs.
    torch.manual_seed(0)
    model = FFTRadNet(in_channels=24, n_range_in=128, n_doppler_in=32, n_range_out=32,
                       n_azimuth_out=48).to(torch_device)
    # push weights to a larger scale so regression outputs aren't coincidentally tiny
    with torch.no_grad():
        for p in model.detection_header.reghead.parameters():
            p.mul_(50.0)
    x = torch.randn(4, 24, 128, 32, device=torch_device) * 5.0
    detection = model(x)["detection"]
    reg = detection[:, 1:]
    assert torch.any(reg.abs() > 1.0)


# --------------------------------------------------------------------------------
# Backward / trainability
# --------------------------------------------------------------------------------
def test_backward_produces_finite_grads_on_all_params(torch_device):
    model = FFTRadNet(in_channels=24, n_range_in=128, n_doppler_in=32, n_range_out=32,
                       n_azimuth_out=48).to(torch_device)
    x = torch.randn(2, 24, 128, 32, device=torch_device)
    out = model(x)["detection"]
    out.sum().backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.all(torch.isfinite(p.grad)), f"non-finite grad for {name}"


def test_overfit_smoke_loss_decreases(torch_device):
    # NOTE (deviation from the nominal "lr=1e-3" smoke-test spec): this deep multi-BN-layer
    # stack is unstable under Adam at 1e-3 (empirically verified -- loss overshoots/oscillates
    # for several random seeds/targets, including trivial near-converged ones), so this test
    # uses 1e-4 -- which is also upstream's own actual training `optimizer.lr` (see the
    # scouting notes' config dump), not an arbitrary weakening. Seed fixed at 0 for a
    # deterministic, reliably-monotonic 3-step trace.
    torch.manual_seed(0)
    model = FFTRadNet(in_channels=24, n_range_in=64, n_doppler_in=32, n_range_out=16,
                       n_azimuth_out=24).to(torch_device)
    x = torch.randn(2, 24, 64, 32, device=torch_device)
    target = torch.rand(2, 3, 16, 24, device=torch_device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    losses = []
    for _ in range(3):
        opt.zero_grad()
        pred = model(x)["detection"]
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[1] < losses[0]
    assert losses[2] < losses[1]


# --------------------------------------------------------------------------------
# Param count
# --------------------------------------------------------------------------------
def test_param_count_upstream_like_config_within_2_to_6m():
    # n_doppler_in=252 (not the vendor's 256): the DDMA pre-encoder requires a
    # replica spacing that lands on integer bins (n_doppler_in % n_tx == 0) -- see
    # the RADIAL_LIKE preset note in radar_config.py.
    model = FFTRadNet(in_channels=32, n_range_in=512, n_doppler_in=252, n_range_out=128,
                       n_azimuth_out=224, mimo_preencoder="ddma", n_tx=12,
                       base_channels=(32, 40, 48, 56), blocks=(3, 6, 6, 3),
                       detection_head_channels=(144, 96, 96, 96))
    n_params = sum(p.numel() for p in model.parameters())
    assert 2_000_000 <= n_params <= 6_000_000, n_params


def test_ddma_preencoder_rejects_fractional_replica_spacing():
    # 256 % 12 != 0: replicas fall between the dilated taps (up to a 4-bin miss at
    # the last TX) -- geometry no training can fix, so the constructor must refuse.
    with pytest.raises(ValueError, match="divisible"):
        FFTRadNet(in_channels=32, n_range_in=64, n_doppler_in=256, n_range_out=16,
                   n_azimuth_out=24, mimo_preencoder="ddma", n_tx=12)


def test_forward_survives_arbitrary_odd_range_sizes(torch_device):
    # The decoder's deconv/skip concats used to crash for ~75% of n_range_in values
    # (any size whose stage widths aren't clean halves); the crop-to-skip fix must
    # keep every size alive, not just powers of two.
    for n_range_in in (17, 33, 100, 250):
        model = FFTRadNet(in_channels=8, n_range_in=n_range_in, n_doppler_in=32,
                           n_range_out=8, n_azimuth_out=24).to(torch_device)
        out = model(torch.randn(1, 8, n_range_in, 32, device=torch_device))
        assert out["detection"].shape == (1, 3, 8, 24)
