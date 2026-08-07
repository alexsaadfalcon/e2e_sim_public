"""
Neural detection models for `e2e.ml`'s radar detection labels
(`e2e.ml.labels.LabelGrid` / `encode_detection_labels`).

Unlike the rest of `e2e.ml` (torch-free at import time), this sub-package expects
torch to be installed -- these are model definitions, not the dependency-free
scene/label plumbing. Imports below are eager but cheap (`torch.nn` module
definitions only, no dataset/weight loading).

Sub-modules
-----------
* `fftradnet` -- `FFTRadNet`, a ResNet-Bottleneck FPN + range-angle decoder +
  PIXOR-style detection head, adapted from valeoai/RADIal (FFTRadNet, Rebut et
  al., CVPR 2022) to this repo's `[B, C_in, R_in, D_in]` input / `[B, 3,
  n_range_out, n_azimuth_out]` detection-target contract. See `fftradnet.py`'s
  module docstring for the full attribution notice and documented deviations.
* `ssm`       -- `SelectiveSSM`/`MambaBlock`, a pure-PyTorch Mamba-1-style
  selective state-space layer (log-space parallel scan) with an optional
  `mamba_ssm` CUDA fast path when that package is installed.
* `ssmradnet` -- `SSMRadNet`, a two-scale (range-axis + Doppler-axis) SSM
  detector adapted from AnuvabSen1/SSMRadNet to the same input/target contract
  as `FFTRadNet`. See `ssmradnet.py` for attribution and deviations.
"""

from e2e.ml.models.fftradnet import FFTRadNet
from e2e.ml.models.ssm import MambaBlock, SelectiveSSM
from e2e.ml.models.ssmradnet import SSMRadNet

__all__ = ["FFTRadNet", "MambaBlock", "SelectiveSSM", "SSMRadNet"]
