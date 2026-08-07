"""
Tests for `e2e.ml.losses` (RADIal/FFTRadNet-style detection loss, ported to
`e2e.ml.labels`'s `[3, n_range, n_azimuth]` target format).
"""

import math

import pytest

torch = pytest.importorskip("torch")

from e2e.ml.losses import detection_loss, focal_loss, masked_regression_loss


# --------------------------------------------------------------------------------
# focal_loss
# --------------------------------------------------------------------------------
def test_focal_loss_perfect_prediction_is_near_zero(torch_device):
    target = torch.zeros((2, 4, 4), device=torch_device)
    target[:, 1, 1] = 1.0
    pred = target.clone()
    # exactly 0/1 predictions clamp against the log-eps; nudge to "as perfect as
    # a probability can realistically get" instead of literal 0/1.
    pred = torch.where(pred == 1.0, torch.full_like(pred, 1.0 - 1e-7),
                        torch.full_like(pred, 1e-7))

    loss = focal_loss(pred, target, gamma=2.0, reduction="sum")
    assert loss.item() < 1e-3


def test_focal_loss_gamma_downweights_easy_negatives(torch_device):
    # mostly-empty map: one positive cell, everything else a confident-but-imperfect
    # negative prediction (pt close to but not exactly 1). gamma=2 should down-weight
    # the (many) easy-negative contributions relative to gamma=0 (plain BCE-like sum).
    target = torch.zeros((1, 20, 20), device=torch_device)
    target[0, 10, 10] = 1.0
    pred = torch.full((1, 20, 20), 0.05, device=torch_device)  # easy negatives: pred~0
    pred[0, 10, 10] = 0.05  # hard positive: pred says "no" where target says "yes"

    loss_gamma0 = focal_loss(pred, target, gamma=0.0, reduction="sum")
    loss_gamma2 = focal_loss(pred, target, gamma=2.0, reduction="sum")
    assert loss_gamma2.item() < loss_gamma0.item()


def test_focal_loss_reduction_mean_vs_sum(torch_device):
    target = torch.zeros((1, 4, 4), device=torch_device)
    pred = torch.full((1, 4, 4), 0.5, device=torch_device)
    total = focal_loss(pred, target, reduction="sum")
    mean = focal_loss(pred, target, reduction="mean")
    assert total.item() == pytest.approx(mean.item() * pred.numel())


def test_focal_loss_invalid_reduction_raises(torch_device):
    target = torch.zeros((1, 2, 2), device=torch_device)
    pred = torch.full((1, 2, 2), 0.5, device=torch_device)
    with pytest.raises(ValueError):
        focal_loss(pred, target, reduction="bogus")


# --------------------------------------------------------------------------------
# masked_regression_loss
# --------------------------------------------------------------------------------
def test_masked_regression_loss_perfect_prediction_is_zero(torch_device):
    mask = torch.zeros((2, 5, 5), device=torch_device)
    mask[:, 2, 2] = 1.0
    target_reg = torch.randn((2, 2, 5, 5), device=torch_device) * mask.unsqueeze(1)
    pred_reg = target_reg.clone()

    loss = masked_regression_loss(pred_reg, target_reg, mask, kind="smooth_l1")
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_masked_regression_loss_ignores_cells_outside_mask(torch_device):
    mask = torch.zeros((1, 5, 5), device=torch_device)
    mask[0, 2, 2] = 1.0
    target_reg = torch.zeros((1, 2, 5, 5), device=torch_device)
    target_reg[0, :, 2, 2] = 0.3
    pred_reg = target_reg.clone()

    base_loss = masked_regression_loss(pred_reg, target_reg, mask, kind="smooth_l1")

    # perturb both pred and target well outside the masked cell
    perturbed_pred = pred_reg.clone()
    perturbed_pred[0, :, 0, 0] = 999.0
    perturbed_target = target_reg.clone()
    perturbed_target[0, :, 4, 4] = -999.0

    perturbed_loss = masked_regression_loss(perturbed_pred, perturbed_target, mask, kind="smooth_l1")
    assert perturbed_loss.item() == pytest.approx(base_loss.item(), abs=1e-6)


def test_masked_regression_loss_zero_positives_is_exact_zero(torch_device):
    mask = torch.zeros((2, 5, 5), device=torch_device)
    pred_reg = torch.randn((2, 2, 5, 5), device=torch_device)
    target_reg = torch.randn((2, 2, 5, 5), device=torch_device)

    loss = masked_regression_loss(pred_reg, target_reg, mask, kind="smooth_l1")
    assert loss.item() == 0.0
    assert not torch.isnan(loss)


def test_masked_regression_loss_smooth_l1_vs_l1_agree_for_small_residuals(torch_device):
    # Smooth-L1's quadratic branch (0.5*x^2) and L1's linear branch (|x|) only both
    # collapse toward the SAME near-zero absolute value when x itself is tiny (x=0.01
    # already gives a 200x relative gap, 0.00125 vs 0.02 absolute -- not "agreeing");
    # x=1e-4 makes both losses individually ~1e-4-scale-or-smaller, so they agree to
    # within a tolerance far below either residual's own magnitude.
    mask = torch.zeros((1, 3, 3), device=torch_device)
    mask[0, 1, 1] = 1.0
    target_reg = torch.zeros((1, 2, 3, 3), device=torch_device)
    pred_reg = target_reg.clone()
    pred_reg[0, :, 1, 1] = 1e-4  # small residual: well within smooth-L1's quadratic region

    smooth = masked_regression_loss(pred_reg, target_reg, mask, kind="smooth_l1")
    l1 = masked_regression_loss(pred_reg, target_reg, mask, kind="l1")
    assert smooth.item() == pytest.approx(l1.item(), abs=1e-3)


def test_masked_regression_loss_smooth_l1_vs_l1_differ_for_large_residuals(torch_device):
    mask = torch.zeros((1, 3, 3), device=torch_device)
    mask[0, 1, 1] = 1.0
    target_reg = torch.zeros((1, 2, 3, 3), device=torch_device)
    pred_reg = target_reg.clone()
    pred_reg[0, :, 1, 1] = 10.0  # large residual: smooth-L1 is linear (Huber) here, L1 differs in scale

    smooth = masked_regression_loss(pred_reg, target_reg, mask, kind="smooth_l1")
    l1 = masked_regression_loss(pred_reg, target_reg, mask, kind="l1")
    assert smooth.item() != pytest.approx(l1.item(), abs=1e-2)


def test_masked_regression_loss_invalid_kind_raises(torch_device):
    mask = torch.ones((1, 2, 2), device=torch_device)
    pred_reg = torch.zeros((1, 2, 2, 2), device=torch_device)
    target_reg = torch.zeros((1, 2, 2, 2), device=torch_device)
    with pytest.raises(ValueError):
        masked_regression_loss(pred_reg, target_reg, mask, kind="bogus")


# --------------------------------------------------------------------------------
# detection_loss
# --------------------------------------------------------------------------------
def test_detection_loss_perfect_prediction_is_near_zero(torch_device):
    target = torch.zeros((1, 3, 6, 6), device=torch_device)
    target[0, 0, 3, 3] = 1.0
    target[0, 1, 3, 3] = 0.2
    target[0, 2, 3, 3] = -0.1

    pred = target.clone()
    pred[0, 0] = torch.where(pred[0, 0] == 1.0, torch.full_like(pred[0, 0], 1.0 - 1e-7),
                              torch.full_like(pred[0, 0], 1e-7))

    total, info = detection_loss(pred, target)
    assert total.item() < 1e-2
    assert info["cls"] < 1e-2
    assert info["reg"] == pytest.approx(0.0, abs=1e-6)


def test_detection_loss_returns_detached_float_info(torch_device):
    target = torch.zeros((1, 3, 4, 4), device=torch_device)
    target[0, 0, 1, 1] = 1.0
    pred = torch.full((1, 3, 4, 4), 0.1, device=torch_device)

    total, info = detection_loss(pred, target)
    assert isinstance(info["cls"], float)
    assert isinstance(info["reg"], float)
    assert math.isfinite(total.item())


def test_detection_loss_gradients_flow_to_both_channels(torch_device):
    target = torch.zeros((1, 3, 6, 6), device=torch_device)
    target[0, 0, 2, 2] = 1.0
    target[0, 1, 2, 2] = 0.1
    target[0, 2, 2, 2] = -0.2

    pred = torch.full((1, 3, 6, 6), 0.3, device=torch_device, requires_grad=True)
    total, _ = detection_loss(pred, target)
    total.backward()

    assert pred.grad is not None
    assert torch.any(pred.grad[0, 0] != 0.0)   # classification channel got gradient
    assert torch.any(pred.grad[0, 1:] != 0.0)  # regression channels got gradient


# --------------------------------------------------------------------------------
# integration with e2e.ml.labels.encode_detection_labels
# --------------------------------------------------------------------------------
def test_detection_loss_with_encoded_labels_target(torch_device):
    from e2e.ml.labels import LabelGrid, encode_detection_labels
    from e2e.ml.scatterers import RadarPose, Scatterer

    grid = LabelGrid(n_range=20, n_azimuth=20, max_range_m=20.0)
    pose = RadarPose()

    def _target(r, sin_az):
        y = r * sin_az
        x = math.sqrt(max(r * r - y * y, 0.0))
        return Scatterer(position=(x, y, 0.0), velocity=(0.0, 0.0, 0.0), rcs_dbsm=0.0,
                          object_class="vehicle")

    label_a = encode_detection_labels(grid, [_target(5.0, -0.3)], pose)
    label_b = encode_detection_labels(grid, [_target(15.0, 0.4)], pose)
    target = torch.stack([label_a, label_b], dim=0).to(torch_device)

    pred = torch.rand(target.shape, device=torch_device, requires_grad=True)

    total, info = detection_loss(pred, target)
    assert math.isfinite(total.item())
    assert math.isfinite(info["cls"])
    assert math.isfinite(info["reg"])

    total.backward()
    assert pred.grad is not None
    assert torch.all(torch.isfinite(pred.grad))
