"""
Detection loss for the FMCW radar detection head.

Adapted from valeoai/RADIal (no LICENSE file upstream; license inquiry sent
2026-08-07, used with attribution and removed on request -- see
e2e/ml/models/fftradnet.py's header). Deviations documented inline. The upstream reference is
`FFTRadNet/loss/loss.py` (`FocalLoss` + `pixor_loss`); this module ports its
math to our `e2e.ml.labels` target format:

    pred, target : float `[B, 3, n_range, n_azimuth]`
      channel 0        objectness -- `pred[:, 0]` is already a probability
                       (the model applies sigmoid internally, unlike upstream's
                       `pixor_loss` which is also fed post-sigmoid values here).
      channels 1-2     range/azimuth regression residuals, meaningful only
                       where `target[:, 0] == 1` (the 3x3 footprint cells, see
                       `e2e.ml.labels`).

Upstream computes the same three pieces (`FocalLoss`, `SmoothL1Loss`/`L1Loss`
regression, batch-summed) but flattens/reshapes a bit differently and carries
a config-driven classification/regression switch plus a segmentation loss we
have no equivalent for (no segmentation head in this port).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

# Numerical-stability epsilon inside log(pt + eps) -- matches upstream's
# `FocalLoss.forward` (`torch.log(pt + 1e-6)`) so gamma=0 focal loss reduces to
# the same (clamped) binary cross-entropy magnitude as upstream's BCE fallback.
_LOG_EPS = 1e-6


# --------------------------------------------------------------------------------
# Classification: binary focal loss
# --------------------------------------------------------------------------------
def focal_loss(pred_prob: Tensor, target: Tensor, *, gamma: float = 2.0,
               reduction: str = "sum") -> Tensor:
    """Binary focal loss on probabilities (not logits).

    `pred_prob` must already be in `[0, 1]` (the model applies sigmoid, unlike
    the upstream reference which is agnostic and can be fed either); `target`
    is a `{0, 1}` map of the same shape. Matches upstream's `FocalLoss`:
    `pt = where(target==1, pred, 1-pred)`, `loss = -(1-pt)^gamma * log(pt+eps)`,
    with `eps=1e-6` guarding `log(0)` when a prediction is exactly wrong.

    `gamma=0` recovers (unweighted) binary cross-entropy -- easy negatives
    dominate the sum in that case, which is exactly the effect `gamma=2`
    exists to suppress.
    """
    if reduction not in ("sum", "mean"):
        raise ValueError(f"reduction must be 'sum' or 'mean', got {reduction!r}")
    pt = torch.where(target == 1.0, pred_prob, 1.0 - pred_prob)
    loss = -(1.0 - pt).pow(gamma) * torch.log(pt + _LOG_EPS)
    return loss.sum() if reduction == "sum" else loss.mean()


# --------------------------------------------------------------------------------
# Regression: masked Smooth-L1 / L1
# --------------------------------------------------------------------------------
def masked_regression_loss(pred_reg: Tensor, target_reg: Tensor, mask: Tensor, *,
                            kind: str = "smooth_l1") -> Tensor:
    """Masked regression loss over `pred_reg`/`target_reg`, summed then normalized.

    `pred_reg`/`target_reg` : `[B, C, n_range, n_azimuth]` (C=2 for range/azimuth
    residuals here, but this is not assumed). `mask` : `[B, n_range, n_azimuth]`
    (or already `[B, 1, n_range, n_azimuth]`), typically `target[:, 0]`.

    Both `pred_reg` and `target_reg` are multiplied by `mask` before the loss is
    computed. Upstream's `pixor_loss` only masks the *prediction*
    (`reg_loss_fct(P*M, T)`), implicitly relying on the label encoder having
    already zeroed `T` outside the footprint -- true for our labels too, but
    masking both sides here makes the "ignore cells outside the mask" contract
    hold unconditionally (a deliberate, safer deviation from upstream).

    The summed loss is divided by the number of positive **cells**
    (`mask.sum()`, counted once per spatial cell, not per regression channel --
    matches upstream's `NbPts = M.sum()`, computed before `M` is broadcast
    across channels). Sums (both numerator and denominator) are taken over the
    whole batch, matching upstream's un-per-sample-normalized `pixor_loss`.
    If there are zero positive cells, returns an exact zero (guarding the
    division instead of letting `0/0` produce NaN).
    """
    if kind == "smooth_l1":
        loss_fn = F.smooth_l1_loss
    elif kind == "l1":
        loss_fn = F.l1_loss
    else:
        raise ValueError(f"kind must be 'smooth_l1' or 'l1', got {kind!r}")

    if mask.dim() == pred_reg.dim() - 1:
        mask = mask.unsqueeze(1)
    mask = mask.to(dtype=pred_reg.dtype)
    n_pos = mask.sum()

    if n_pos <= 0:
        return pred_reg.new_zeros(())

    numerator = loss_fn(pred_reg * mask, target_reg * mask, reduction="sum")
    return numerator / n_pos


# --------------------------------------------------------------------------------
# Total detection loss
# --------------------------------------------------------------------------------
def detection_loss(pred: Tensor, target: Tensor, *, gamma: float = 2.0,
                    reg_weight: float = 100.0, kind: str = "smooth_l1"
                    ) -> Tuple[Tensor, Dict[str, float]]:
    """Total detection loss = focal(classification) + `reg_weight` * masked regression.

    `pred`/`target` : `[B, 3, n_range, n_azimuth]` (see module docstring for the
    channel layout). Both loss terms are summed over the whole batch (not
    averaged), matching upstream's `pixor_loss` + the `weight=[1, 100, 100]`
    scaling in its training config (segmentation weight has no equivalent
    here, there is no segmentation head).

    Returns `(total, {"cls": float, "reg": float})` -- the dict values are
    detached scalars for logging, not part of the autograd graph.
    """
    cls_pred, cls_target = pred[:, 0], target[:, 0]
    reg_pred, reg_target = pred[:, 1:], target[:, 1:]

    cls_loss = focal_loss(cls_pred, cls_target, gamma=gamma, reduction="sum")
    reg_loss = masked_regression_loss(reg_pred, reg_target, mask=cls_target, kind=kind)
    total = cls_loss + reg_weight * reg_loss

    return total, {"cls": float(cls_loss.detach().item()), "reg": float(reg_loss.detach().item())}
