"""RADDetNet — a detector whose spatial plane IS the plane it predicts.

WHY THIS EXISTS (measured, 2026-09-21; see notes/ESTABLISHED_FACTS.md F83 and
notes/ML_DETECTION_BRIEF.md)
---------------------------------------------------------------------------------
The two ported detectors (`fftradnet`, `ssmradnet`) both emit a near-separable
`f(range) * g(azimuth)` objectness map — a full-field-of-view stripe at every true range
rather than peaks. Rank-1 energy fraction 0.89 / 0.76 against 0.31 for ground truth, and
under azimuth-only matching they score no better than a constant frame-independent map.
They never learn azimuth.

The first cause was the INPUT: azimuth reached them only as phase across a virtual-channel
axis. Fixing that (`input_format="rad"`, which hands the network the classical
beamformer's own range-azimuth-Doppler cube) raised FFTRadNet's test AP 0.127 -> 0.229 and
halved its false alarms, 26.3 -> 12.4 per frame. But the stripe barely moved: rank-1
0.894 -> 0.853. The gain came from range behaviour and normalization, not azimuth.

The second cause is ARCHITECTURAL, and it is why this module exists.
`fftradnet._RangeAngleDecoder` inherits upstream RADIal's "transpose trick": it builds the
OUTPUT azimuth axis out of the BACKBONE'S CHANNEL AXIS via 1x1 projections. Upstream that
was sound — channels there carried MIMO coding. Here the `rad` input puts azimuth IN the
channel axis, so the backbone's first convolution mixes it away, and the decoder then
re-invents an azimuth axis from learned channel weights. Direction is destroyed in layer
one and hallucinated at the end.

THE DESIGN, stated as the one decision it turns on
---------------------------------------------------------------------------------
Put (range, azimuth) in the SPATIAL axes and Doppler in the channel axis, so every
convolution is local in exactly the plane the labels live in, and the output geometry is
reached by resampling a real azimuth axis rather than by synthesizing one.

    input   [B, A, R, D]   (dataset's rad layout: azimuth, range, doppler)
      permute             -> [B, D, R, A]    doppler = channels
      encoder (stride on RANGE only)         azimuth is never strided away
      resample            -> (n_range_out, n_azimuth_out)
      head                -> [B, 3, n_range_out, n_azimuth_out]

Doppler as the channel axis is the right call and not merely convenient: a target's
signature is a *pattern across* Doppler at one (range, azimuth) cell, which is precisely
what a 1x1-in-space, all-channels convolution reads. Azimuth is never given a stride
because the corpus's whole difficulty is angular: the label grid is 192 azimuth bins from
a 64-bin native axis, so azimuth is upsampled, never reduced.

This is deliberately a small, legible model rather than a third ported architecture. The
point being tested is the REPRESENTATION-TO-GEOMETRY match, and a large model would
confound that with capacity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _ConvBlock(nn.Module):
    """Two 3x3 convs + BN + SiLU, optionally striding the RANGE axis only."""

    def __init__(self, cin: int, cout: int, range_stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=(range_stride, 1), padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
        )
        # Projection so the residual add survives a channel or range-stride change.
        self.skip = (nn.Identity() if (cin == cout and range_stride == 1)
                     else nn.Conv2d(cin, cout, 1, stride=(range_stride, 1), bias=False))

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x) + self.skip(x)


class RADDetNet(nn.Module):
    """Range-azimuth-Doppler detector. Input `[B, A, R, D]`, output the standard
    `{"detection": [B, 3, n_range_out, n_azimuth_out]}` contract (channel 0 = sigmoid
    objectness, channels 1-2 = raw regression residuals), identical to `FFTRadNet` /
    `SSMRadNet` so every downstream consumer — `detection_loss`, `decode_detections`,
    `evaluate_dataset`, `compare_detectors` — works unchanged.

    `in_azimuth` / `n_doppler` describe the INPUT cube; `n_range_out` / `n_azimuth_out`
    the label grid. They need not match: the azimuth axis is resampled once, explicitly,
    at a single named place (`F.interpolate` below) rather than being manufactured inside
    a decoder.
    """

    def __init__(self, in_azimuth: int, n_range_in: int, n_doppler: int,
                 n_range_out: int, n_azimuth_out: int, width: int = 64):
        super().__init__()
        self.in_azimuth = int(in_azimuth)
        self.n_range_in = int(n_range_in)
        self.n_doppler = int(n_doppler)
        self.n_range_out = int(n_range_out)
        self.n_azimuth_out = int(n_azimuth_out)

        w = int(width)
        # Doppler is the channel axis on entry. Stride only RANGE (512 -> 64 here), so the
        # angular axis is carried at full native resolution all the way to the resample.
        self.stem = _ConvBlock(self.n_doppler, w)
        self.down1 = _ConvBlock(w, w * 2, range_stride=2)
        self.down2 = _ConvBlock(w * 2, w * 2, range_stride=2)
        self.down3 = _ConvBlock(w * 2, w * 4, range_stride=2)
        self.mix = _ConvBlock(w * 4, w * 4)
        self.head = nn.Sequential(
            nn.Conv2d(w * 4, w * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(w * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(w * 2, 3, 3, padding=1),
        )

    def forward(self, x: Tensor) -> dict:
        if x.dim() != 4:
            raise ValueError(f"expected [B, A, R, D], got {tuple(x.shape)}")
        b, a, r, d = x.shape
        if (a, r, d) != (self.in_azimuth, self.n_range_in, self.n_doppler):
            raise ValueError(
                f"expected [B, {self.in_azimuth}, {self.n_range_in}, {self.n_doppler}], "
                f"got {tuple(x.shape)}"
            )
        # [B, A, R, D] -> [B, D, R, A]: Doppler becomes channels, (range, azimuth) the
        # spatial plane. This single permute is the whole idea of the module.
        z = x.permute(0, 3, 2, 1).contiguous()
        z = self.mix(self.down3(self.down2(self.down1(self.stem(z)))))
        # One explicit resample onto the label grid. Bilinear, not a learned deconv, so
        # nothing can invent angular structure that the input did not carry.
        z = F.interpolate(z, size=(self.n_range_out, self.n_azimuth_out),
                          mode="bilinear", align_corners=False)
        out = self.head(z)
        # Objectness through a sigmoid; regression channels raw. Matches the shipped
        # contract exactly (see e2e.ml.labels.decode_detections).
        return {"detection": torch.cat([torch.sigmoid(out[:, :1]), out[:, 1:]], dim=1)}
