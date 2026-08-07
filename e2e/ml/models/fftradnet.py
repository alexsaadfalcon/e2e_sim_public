"""
FFTRadNet detection model.

Adapted from valeoai/RADIal (https://github.com/valeoai/RADIal), FFTRadNet by
Rebut et al., CVPR 2022. The upstream repository ships no LICENSE file; a license
inquiry was sent to the authors (2026-08-07) and this adaptation proceeds under
research-community norms of attribution, consistent with the upstream project's
evident open-research intent. It will be removed or relicensed promptly on the
authors' request. Deviations from upstream are documented inline (and summarized
here).

Upstream source ported: `FFTRadNet/model/FFTRadNet.py` (`MIMO_PreEncoder`,
`FPN_BackBone`, `Bottleneck`, `BasicBlock`, `RangeAngle_Decoder`,
`Detection_Header`, `FFTRadNet`).

Data-contract differences from upstream (see `e2e.ml.labels`/`e2e.ml.dataset`)
--------------------------------------------------------------------------
Upstream is hardwired to one radar (`NbTxAntenna=12`, `NbRxAntenna=16`, fixed
`(N, 32, 512, 256)` input, fixed `(N, 3, 128, 224)` detection output, plus a
segmentation head we have no labels for). This port takes the input/output
geometry and MIMO scheme as constructor arguments instead:

* `in_channels`/`n_range_in`/`n_doppler_in` replace the hardcoded `32`/`512`/
  `256`; `n_range_out`/`n_azimuth_out` replace the hardcoded `128`/`224`.
* `mimo_preencoder` chooses the pre-encoder: `"ddma"` reproduces upstream's
  dilated-conv MIMO demux (see `_MIMOPreEncoder`, generalized below since
  upstream's kernel/dilation only work for its specific 12x16 radar); `None`
  (our TDM/virtual-array inputs, where a classical virtual-array is already
  formed upstream of this model) uses a plain 3x3 conv stem to the same
  channel count -- there is nothing left to "de-multiplex" in that case.
* The segmentation head is OMITTED entirely: our labels
  (`e2e.ml.labels.LabelGrid`) are detection-only, so there is nothing to
  supervise a `Segmentation` output with. `RangeAngle_Decoder`/`FPN_BackBone`/
  `Detection_Header` are otherwise ported faithfully (see class docstrings for
  the handful of places genuinely new parameterization was required -- upstream
  bakes several "shapes as a side effect of unrelated hyperparameters" tricks
  that only work for its one fixed config; those are called out below, per the
  "896-width bookkeeping is a known decoy" scouting note).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _conv3x3(in_planes: int, out_planes: int, stride=1, bias: bool = False) -> nn.Conv2d:
    """3x3 conv with padding=1 (upstream's `conv3x3`)."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)


def _downsampled_len(n: int) -> int:
    """Spatial length after one stride-2, padding-1 conv (kernel 3 or kernel 1, both agree).

    Upstream's `Bottleneck` downsamples via a stride-2 kernel-3/padding-1 main-path conv
    AND a stride-2 kernel-1/padding-0 shortcut conv; both give `floor((n-1)/2) + 1`, so the
    two paths always agree on the output length regardless of `n`'s parity. We use this to
    derive exact backbone-stage widths from `n_range_in`/`n_doppler_in` instead of assuming
    they are powers of two (upstream implicitly assumes this since it hardcodes 512/256).
    """
    return (n - 1) // 2 + 1


# --------------------------------------------------------------------------------
# MIMO pre-encoder
# --------------------------------------------------------------------------------
class _DdmaPreEncoder(nn.Module):
    """DDMA MIMO pre-encoder: gathers per-TX Doppler replicas into virtual channels.

    Upstream's `MIMO_PreEncoder` hardcodes `kernel_size=(1, NbTxAntenna=12)`,
    `dilation=(1, NbRxAntenna=16)` and a fixed circular pad of `NbVirtualAntenna/2=96`
    -- all three numbers only work together for its one fixed 12-TX/16-RX radar and a
    Doppler axis of width 256. We generalize instead of copying the magic numbers:

    * kernel `(1, n_tx)` reads one sample from each of the `n_tx` DDMA replicas.
    * dilation `(1, n_doppler_in // n_tx)` matches the actual physical replica
      spacing: DDMA places each TX's replica every `n_doppler_in / n_tx` Doppler bins,
      so a `n_tx`-tap conv with that dilation looks at exactly the `n_tx` replica
      positions for a given target. This is only geometrically valid when
      `n_doppler_in % n_tx == 0` -- with a fractional spacing the replicas both
      smear across bins (spectral leakage in the synthesizer) and drift off the
      conv's fixed taps (up to (n_tx-1) * frac bins at the last tap), which no
      amount of training can repair. The constructor therefore REJECTS
      non-divisible combinations; pick a chirp count divisible by n_tx (see the
      RADIAL_LIKE preset's n_chirps=252 note in radar_config.py).
    * The circular pad is derived from the conv's own receptive field
      (`(n_tx - 1) * dilation`), not a fixed constant, so the crop-back-to-original-width
      step stays valid for any `(n_tx, n_doppler_in)` combination instead of only
      upstream's specific one.
    """

    def __init__(self, in_channels: int, out_channels: int, n_tx: int, n_doppler_in: int,
                 use_bn: bool = True):
        super().__init__()
        if n_doppler_in % n_tx != 0:
            raise ValueError(
                f"DDMA pre-encoder needs n_doppler_in divisible by n_tx to place its "
                f"dilated taps on the replica bins (got {n_doppler_in} % {n_tx} = "
                f"{n_doppler_in % n_tx}); use a chirp count divisible by n_tx"
            )
        dilation = max(1, n_doppler_in // n_tx)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, n_tx),
                               dilation=(1, dilation), bias=not use_bn)
        self.use_bn = use_bn
        self.bn = nn.BatchNorm2d(out_channels)
        self.pad = (n_tx - 1) * dilation

    def forward(self, x: Tensor) -> Tensor:
        width = x.shape[-1]
        if self.pad > 0:
            x = torch.cat([x[..., -self.pad:], x, x[..., :self.pad]], dim=3)
        x = self.conv(x)
        start = (x.shape[-1] - width) // 2
        x = x[..., start:start + width]
        if self.use_bn:
            x = self.bn(x)
        return x


# --------------------------------------------------------------------------------
# ResNet-Bottleneck FPN backbone
# --------------------------------------------------------------------------------
class _Bottleneck(nn.Module):
    """1x1 -> 3x3(stride) -> 1x1 residual bottleneck, expansion=4 (upstream's `Bottleneck`)."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None, expansion: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(expansion * planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return F.relu(residual + out)


class _BasicBlock(nn.Module):
    """Two 3x3 convs (upstream's `BasicBlock`).

    NOTE (faithful port of an upstream oddity): despite the name, this has NO residual
    add -- `forward` never adds the block's input back in, so it is really just a plain
    2-conv stack (an optional trailing `downsample` transform is applied to the *output*,
    not used for a skip connection). Ported as-is since `RangeAngle_Decoder` relies on
    exactly this behaviour; `downsample` is accepted for fidelity but never passed by
    any caller in this file (matching upstream).
    """

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = _conv3x3(in_planes, planes, stride, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        if self.downsample is not None:
            out = self.downsample(out)
        return out


class _FpnBackbone(nn.Module):
    """MIMO pre-encoder + conv stem + 4-stage `_Bottleneck` FPN (upstream's `FPN_BackBone`).

    `num_blocks`/`channels` must each have exactly 4 entries -- the FPN depth is fixed at
    4 stages (matching upstream and `_RangeAngleDecoder`'s wiring to stages x2/x3/x4).
    Each stage's first block strides 2, so stage `i` output has spatial size
    `_downsampled_len` applied `i+1` times to the pre-encoder's (unchanged) input size.

    Deviation: the pre-encoder's hidden channel width ("mimo_layer" upstream, a `192`
    hardcoded independently of `channels`) is not exposed as a constructor argument here
    (see `FFTRadNet`'s docstring); we derive it as `channels[0] * expansion` so it scales
    with the backbone instead of being an unexplained extra magic number.
    """

    def __init__(self, in_channels: int, num_blocks: Sequence[int], channels: Sequence[int],
                 mimo_preencoder: Optional[str], n_tx: Optional[int], n_doppler_in: int,
                 expansion: int = 4):
        super().__init__()
        mimo_layer = channels[0] * expansion

        if mimo_preencoder == "ddma":
            self.pre_enc = _DdmaPreEncoder(in_channels, mimo_layer, n_tx=n_tx,
                                            n_doppler_in=n_doppler_in, use_bn=True)
        else:
            # Plain 3x3 conv stem: our TDM/virtual-array inputs already have a formed
            # virtual array, so there is no per-TX Doppler demultiplexing left to learn --
            # this stem only needs to reproject to the backbone's channel width.
            self.pre_enc = nn.Sequential(
                _conv3x3(in_channels, mimo_layer, bias=False),
                nn.BatchNorm2d(mimo_layer),
            )

        self.in_planes = mimo_layer
        self.conv = _conv3x3(self.in_planes, self.in_planes)
        self.bn = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)

        self.block1 = self._make_layer(channels[0], num_blocks[0], expansion)
        self.block2 = self._make_layer(channels[1], num_blocks[1], expansion)
        self.block3 = self._make_layer(channels[2], num_blocks[2], expansion)
        self.block4 = self._make_layer(channels[3], num_blocks[3], expansion)

    def _make_layer(self, planes: int, num_blocks: int, expansion: int) -> nn.Sequential:
        downsample = nn.Sequential(
            nn.Conv2d(self.in_planes, planes * expansion, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(planes * expansion),
        )
        layers = [_Bottleneck(self.in_planes, planes, stride=2, downsample=downsample,
                               expansion=expansion)]
        self.in_planes = planes * expansion
        for _ in range(1, num_blocks):
            layers.append(_Bottleneck(self.in_planes, planes, stride=1, expansion=expansion))
            self.in_planes = planes * expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.pre_enc(x)
        x = self.relu(self.bn(self.conv(x)))
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        return {"x1": x1, "x2": x2, "x3": x3, "x4": x4}


# --------------------------------------------------------------------------------
# Range-Angle decoder
# --------------------------------------------------------------------------------
class _RangeAngleDecoder(nn.Module):
    """The range-angle "transpose trick" decoder (upstream's `RangeAngle_Decoder`).

    Upstream's key idea: `transpose(1, 3)` swaps a feature map's channel axis with its
    (downsampled) Doppler-width axis, so the backbone's *channel* dimension becomes a
    *spatial* azimuth axis, while the old Doppler-width axis becomes the new channel
    count. Two `ConvTranspose2d(kernel=3, stride=(2,1))` stages then upsample only the
    *range* axis (stride 1 on the transposed/azimuth axis), concatenating with a
    shallower feature (via a 1x1 `L2`/`L3` channel projection) at each step.

    Deviation (this is the one place we could not just parameterize an upstream
    constant): upstream applies `L3`/`L2` projections to x3/x2 to make their channel
    count match x4's *native* channel count (`channels[3] * expansion`) before
    transposing -- x4 itself is used unprojected, so upstream's final azimuth width is
    always exactly `channels[3] * expansion` (224 for its default config; the scouting
    notes call this the "known decoy", since `channels[3]` was picked backwards from the
    desired 224, not the other way around). That ties azimuth resolution rigidly to
    `base_channels`, which conflicts with `n_azimuth_out` being an independent
    constructor argument here. We add a matching 1x1 projection `L4` (x4's native
    channel count -> `n_azimuth_out`) alongside `L3`/`L2` (also -> `n_azimuth_out`
    instead of `channels[3] * expansion`), so all three decoder inputs project onto the
    same, independently configurable azimuth width. Everything else (the two
    deconv/concat/`_BasicBlock` stages) is an unmodified, faithful port.

    `w2`/`w3`/`w4` are the *Doppler-axis* backbone widths at stages 2/3/4 (from
    `_downsampled_len`), which become the pre-transpose channel counts consumed by
    `deconv4`/`conv_block4`/`deconv3`.
    """

    def __init__(self, s2: int, s3: int, s4: int, w2: int, w3: int, w4: int,
                 n_azimuth_out: int, decoder_ch4: int, decoder_ch3: int):
        super().__init__()
        self.L2 = nn.Conv2d(s2, n_azimuth_out, kernel_size=1)
        self.L3 = nn.Conv2d(s3, n_azimuth_out, kernel_size=1)
        self.L4 = nn.Conv2d(s4, n_azimuth_out, kernel_size=1)  # deviation: see class docstring

        self.deconv4 = nn.ConvTranspose2d(w4, w4, kernel_size=3, stride=(2, 1), padding=1,
                                           output_padding=(1, 0))
        self.conv_block4 = _BasicBlock(w4 + w3, decoder_ch4)

        self.deconv3 = nn.ConvTranspose2d(decoder_ch4, decoder_ch4, kernel_size=3, stride=(2, 1),
                                           padding=1, output_padding=(1, 0))
        self.conv_block3 = _BasicBlock(decoder_ch4 + w2, decoder_ch3)

    @staticmethod
    def _crop_range_to(x: Tensor, ref: Tensor) -> Tensor:
        """Crop `x`'s range axis (dim 2 post-transpose) to `ref`'s length.

        The backbone halves the range axis with floor((n-1)/2)+1, so a stride-2
        deconv lands on either the skip's exact length (even n) or one sample
        long (odd n). Cropping the surplus keeps the skip concatenations valid
        for ARBITRARY n_range_in -- without this, any input whose stage widths
        aren't exact powers-of-two halves crashes at the concat below.
        """
        return x[:, :, :ref.shape[2], :]

    def forward(self, features: Dict[str, Tensor]) -> Tensor:
        t4 = self.L4(features["x4"]).transpose(1, 3)
        t3 = self.L3(features["x3"]).transpose(1, 3)
        t2 = self.L2(features["x2"]).transpose(1, 3)

        s4 = torch.cat([self._crop_range_to(self.deconv4(t4), t3), t3], dim=1)
        s4 = self.conv_block4(s4)

        s43 = torch.cat([self._crop_range_to(self.deconv3(s4), t2), t2], dim=1)
        return self.conv_block3(s43)


# --------------------------------------------------------------------------------
# Detection head
# --------------------------------------------------------------------------------
class _DetectionHeader(nn.Module):
    """PIXOR-style detection head (upstream's `Detection_Header`).

    Deviation: upstream has three hardcoded stride branches keyed off
    `input_angle_size in {224, 448, 896}` (its way of coping with a few fixed azimuth
    widths without ever resizing). Since `_RangeAngleDecoder` above already projects
    azimuth to exactly `n_azimuth_out` for any value (see its docstring) and
    `FFTRadNet.forward` explicitly resizes to `(n_range_out, n_azimuth_out)` before this
    head runs, a single stride-1 branch suffices regardless of `n_azimuth_out` -- the
    upstream branching is dead weight once the input width is always already correct.
    Layer widths come from `detection_head_channels` instead of upstream's hardcoded
    `144, 96, 96, 96`. No ReLU between conv/BN layers, matching upstream exactly (an
    unusual but faithfully-ported choice -- upstream's head is a plain conv+BN stack
    until the final `sigmoid`/linear heads).
    """

    def __init__(self, in_channels: int, channels: Sequence[int], reg_layer: int = 2):
        super().__init__()
        c0, c1, c2, c3 = channels
        self.conv1 = _conv3x3(in_channels, c0)
        self.bn1 = nn.BatchNorm2d(c0)
        self.conv2 = _conv3x3(c0, c1)
        self.bn2 = nn.BatchNorm2d(c1)
        self.conv3 = _conv3x3(c1, c2)
        self.bn3 = nn.BatchNorm2d(c2)
        self.conv4 = _conv3x3(c2, c3)
        self.bn4 = nn.BatchNorm2d(c3)

        self.clshead = _conv3x3(c3, 1, bias=True)
        self.reghead = _conv3x3(c3, reg_layer, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
        x = self.bn4(self.conv4(x))

        cls = torch.sigmoid(self.clshead(x))
        reg = self.reghead(x)
        return torch.cat([cls, reg], dim=1)


# --------------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------------
class FFTRadNet(nn.Module):
    """ResNet-Bottleneck FPN + range-angle decoder + PIXOR detection head.

    Input: float32 `[B, in_channels, n_range_in, n_doppler_in]`.
    Output: `{"detection": [B, 3, n_range_out, n_azimuth_out]}` -- channel 0 is a
    sigmoid objectness probability in `[0, 1]`; channels 1-2 are raw (unnormalized)
    regression residuals, matching `e2e.ml.labels`'s target convention and upstream's
    output convention.

    Segmentation head OMITTED (see module docstring): our labels are detection-only.

    Parameters
    ----------
    mimo_preencoder : None | "ddma"
        `"ddma"` reproduces upstream's dilated-conv MIMO pre-encoder (requires `n_tx`);
        `None` (default) uses a plain 3x3 conv stem, appropriate for inputs where a
        virtual array has already been formed upstream (e.g. our TDM preset).
    base_channels, blocks : each length-4
        Per-stage bottleneck planes / block counts (upstream's `channels`/
        `backbone_block`). Stage output channels are `base_channels[i] * 4`
        (bottleneck expansion).
    detection_head_channels : length-4
        Per-layer widths of the detection head's conv stack (upstream's hardcoded
        `144, 96, 96, 96`).

    A final `F.interpolate` to exactly `(n_range_out, n_azimuth_out)` is applied to the
    decoder output before the detection head. The azimuth axis is already exactly
    `n_azimuth_out` by construction (see `_RangeAngleDecoder`); the range axis is
    exact only when `n_range_in` divides cleanly through 4 stride-2 stages then 2x
    2x upsamples (i.e. `n_range_out == n_range_in // 4` and `n_range_in` a multiple of
    16) -- the interpolate is a deviation from upstream (which relies on exact integer
    arithmetic and would simply crash on a shape mismatch) that makes the model robust
    to arbitrary `n_range_in`/`n_range_out` instead.
    """

    def __init__(self, in_channels: int, n_range_in: int, n_doppler_in: int, n_range_out: int,
                 n_azimuth_out: int, *, mimo_preencoder: Optional[str] = None,
                 n_tx: Optional[int] = None, base_channels: Tuple[int, int, int, int] = (32, 40, 48, 56),
                 blocks: Tuple[int, int, int, int] = (3, 6, 6, 3),
                 detection_head_channels: Tuple[int, int, int, int] = (144, 96, 96, 96)):
        super().__init__()
        if len(base_channels) != 4 or len(blocks) != 4:
            raise ValueError("base_channels and blocks must each have exactly 4 stages "
                              "(FPN depth is fixed at 4, matching upstream)")
        if mimo_preencoder not in (None, "ddma"):
            raise ValueError(f"mimo_preencoder must be None or 'ddma', got {mimo_preencoder!r}")
        if mimo_preencoder == "ddma" and n_tx is None:
            raise ValueError("n_tx is required when mimo_preencoder='ddma'")

        self.n_range_out = n_range_out
        self.n_azimuth_out = n_azimuth_out

        expansion = 4
        self.backbone = _FpnBackbone(in_channels, blocks, base_channels, mimo_preencoder, n_tx,
                                      n_doppler_in, expansion=expansion)

        # Doppler-axis backbone widths at stages 2/3/4 -- see _downsampled_len docstring.
        d1 = _downsampled_len(n_doppler_in)
        d2 = _downsampled_len(d1)
        d3 = _downsampled_len(d2)
        d4 = _downsampled_len(d3)

        s2 = base_channels[1] * expansion
        s3 = base_channels[2] * expansion
        s4 = base_channels[3] * expansion
        # Decoder's own internal channel widths (upstream hardcodes 128/256 independently
        # of `channels`); derived from base_channels[0] so they still equal 128/256 at
        # upstream's default sizes but scale with a smaller/larger backbone.
        decoder_ch4 = base_channels[0] * expansion
        decoder_ch3 = base_channels[0] * expansion * 2

        self.decoder = _RangeAngleDecoder(s2, s3, s4, w2=d2, w3=d3, w4=d4,
                                           n_azimuth_out=n_azimuth_out,
                                           decoder_ch4=decoder_ch4, decoder_ch3=decoder_ch3)
        self.detection_header = _DetectionHeader(decoder_ch3, detection_head_channels, reg_layer=2)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        features = self.backbone(x)
        ra = self.decoder(features)
        ra = F.interpolate(ra, size=(self.n_range_out, self.n_azimuth_out), mode="bilinear",
                            align_corners=False)
        return {"detection": self.detection_header(ra)}
