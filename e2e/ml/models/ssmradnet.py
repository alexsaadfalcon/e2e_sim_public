"""
SSMRadNet: a two-scale selective-state-space radar detector.

Adapted from AnuvabSen1/SSMRadNet (https://github.com/AnuvabSen1/SSMRadNet). The upstream
repository ships no LICENSE file; the SSMRadNet authors requested this integration
(training/evaluation on this simulator's synthetic data) and gave written approval for
this adaptation's redistribution (2026-08-07). Deviations documented inline.

What upstream does
------------------
`FFTRadNet/model/ssmradnet_channelmixing.py` (the variant that is fully wired end to end)
runs two Mamba scans at two time scales on the *raw ADC* cube `(B, 512, 256, 32)` =
`(batch, fast-time samples, chirps, Rx x I/Q)`:

1. a **fast-time / intra-chirp** SSM scanning the 512 ADC samples of each chirp
   (batched over `B * n_chirps`), average-pooled to one embedding per chirp;
2. a **slow-time / inter-chirp** SSM scanning the resulting 256-chirp sequence;
3. a linear projection of the chirp sequence onto a 2-D grid, then a conv decoder with a
   PIXOR-style `cls`(sigmoid) + `reg` head.

The paper's selling point is that this replaces the range/Doppler FFTs with learned scans.

Documented deviations
---------------------
* **Input is post-RD-FFT `[B, C, R, D]`, not raw ADC.** This repo's dataset layer
  (`e2e.ml.transforms.rd_to_input` -> `e2e.ml.dataset`) serves range-Doppler tensors with
  real/imag stacked on the channel axis, and the sibling `fftradnet.py` consumes the same
  tensor. Serving raw ADC purely for this model would fork the dataset contract. The
  two-scale idea survives the change intact: the "fast axis" scan still runs along the
  *range* axis (one sequence per Doppler column) and the "slow axis" scan still runs along
  the *Doppler* axis -- the axes carry the same physical meaning, they are just already in
  the frequency domain, so the scans refine rather than replace the FFT. This is an honest
  weakening of the paper's "no FFT" claim and is stated as such.
* **Doppler is grouped at the stem** (`n_doppler_tokens`, adaptive average pool).
  Upstream's fast-axis scan is `B * n_chirps` independent sequences, which only fits in
  memory because `mamba_ssm`'s fused kernel never materializes the `[batch, L, d_inner,
  d_state]` state; our portable scan (`e2e.ml.models.ssm`) does. Grouping neighbouring
  Doppler bins -- which are heavily correlated after the Doppler FFT, and are marginalized
  out entirely by the range-azimuth output map -- keeps the scan tractable in pure torch.
* **Channel mixing is a pointwise MLP, not 32 independent Mamba blocks.** Upstream's
  `PerChannelFastSSM` instantiates one `Mamba` per input channel and loops over them in
  Python; that is a throughput landmine (see the scout notes) and does not scale to this
  repo's virtual-array channel counts. We embed channels once at the stem and mix them
  with an MLP between the two SSM stages, which is the same information path at a fraction
  of the cost.
* **Detection head only.** Upstream also carries a free-space *segmentation* head; this
  repo has no free-space labels (`e2e.ml.labels` encodes detections only), so that head
  would be untrainable dead weight.
* Residual SSM stages use pre-LayerNorm (upstream adds the raw block output); this only
  affects trainability, not the architecture.

Output contract (shared with `e2e.ml.models.fftradnet`)
-------------------------------------------------------
`forward` returns `{"detection": [B, 3, n_range_out, n_azimuth_out]}` matching
`e2e.ml.labels.encode_detection_labels`: channel 0 is objectness passed through a sigmoid
(so it is a probability in `[0, 1]`), channels 1-2 are raw range/azimuth regression
residuals in bin units.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from e2e.ml.models.ssm import MambaBlock


__all__ = ["SSMRadNet"]


class _SSMStage(nn.Module):
    """`n_layers` pre-LayerNorm residual `MambaBlock`s over a `[B, L, d_model]` sequence."""

    def __init__(self, d_model, d_state, n_layers, backend, d_conv=4, expand=2):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand, backend=backend)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        for norm, block in zip(self.norms, self.blocks):
            x = x + block(norm(x))
        return x


class _ChannelMix(nn.Module):
    """Pointwise (per range-Doppler cell) gated MLP -- the 'channelmixing' stage."""

    def __init__(self, d_model, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, expand * d_model)
        self.fc2 = nn.Linear(expand * d_model, d_model)

    def forward(self, x):
        return x + self.fc2(F.silu(self.fc1(self.norm(x))))


class SSMRadNet(nn.Module):
    """Two-scale selective-SSM radar detector on range-Doppler input.

    Parameters
    ----------
    in_channels, n_range_in, n_doppler_in
        Shape of the input tensor `[B, in_channels, n_range_in, n_doppler_in]` (the
        `[2*n_rx, range_bin, doppler_bin]` layout produced by `e2e.ml.transforms`).
    n_range_out, n_azimuth_out
        Detection-map geometry (see `e2e.ml.labels.LabelGrid`).
    d_model, d_state, n_layers_fast, n_layers_slow
        SSM width / state size / depth of the range and Doppler scan stages.
    backend
        Forwarded to `MambaBlock` ("auto" | "torch" | "mamba_ssm").
    n_doppler_tokens
        Doppler bins are average-pooled to at most this many tokens at the stem (no-op if
        `n_doppler_in` is already smaller). See "Documented deviations".
    head_channels
        Width of the conv decoder that produces the detection map.
    """

    def __init__(
        self,
        in_channels: int,
        n_range_in: int,
        n_doppler_in: int,
        n_range_out: int,
        n_azimuth_out: int,
        *,
        d_model: int = 64,
        d_state: int = 16,
        n_layers_fast: int = 2,
        n_layers_slow: int = 2,
        backend: str = "auto",
        n_doppler_tokens: int = 16,
        head_channels: int = 32,
    ):
        super().__init__()
        for name, value in (
            ("in_channels", in_channels), ("n_range_in", n_range_in),
            ("n_doppler_in", n_doppler_in), ("n_range_out", n_range_out),
            ("n_azimuth_out", n_azimuth_out), ("d_model", d_model),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        self.in_channels = int(in_channels)
        self.n_range_in = int(n_range_in)
        self.n_doppler_in = int(n_doppler_in)
        self.n_range_out = int(n_range_out)
        self.n_azimuth_out = int(n_azimuth_out)
        self.d_model = int(d_model)
        self.n_doppler_tokens = min(int(n_doppler_tokens), self.n_doppler_in)

        # ---- stem: embed channels, group the Doppler axis ------------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.d_model),
            nn.SiLU(),
        )
        self.doppler_pool = nn.AdaptiveAvgPool2d((self.n_range_in, self.n_doppler_tokens))

        # ---- two SSM scales + channel mixing -------------------------------------------
        self.fast_ssm = _SSMStage(self.d_model, d_state, n_layers_fast, backend)   # scans range
        self.chan_mix = _ChannelMix(self.d_model)
        self.range_pool = nn.AdaptiveAvgPool2d((self.n_range_out, self.n_doppler_tokens))
        self.slow_ssm = _SSMStage(self.d_model, d_state, n_layers_slow, backend)   # scans Doppler
        self.post_norm = nn.LayerNorm(self.d_model)

        # ---- detection head -------------------------------------------------------------
        # One 1x1 conv fans the per-range embedding out over the azimuth axis (the axis the
        # network has to synthesize: azimuth lives in the input's *channel* dimension, not
        # in a spatial one), then a small conv stack refines the 2-D map.
        self.head_channels = int(head_channels)
        self.az_proj = nn.Conv1d(self.d_model, self.head_channels * self.n_azimuth_out, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.head_channels),
            nn.SiLU(),
            nn.Conv2d(self.head_channels, self.head_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.head_channels),
            nn.SiLU(),
        )
        self.cls_head = nn.Conv2d(self.head_channels, 1, kernel_size=3, padding=1)
        self.reg_head = nn.Conv2d(self.head_channels, 2, kernel_size=3, padding=1)

    # --------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> dict:
        if x.dim() != 4:
            raise ValueError(f"expected [B, C, R, D], got {tuple(x.shape)}")
        b, c, r, d = x.shape
        if (c, r, d) != (self.in_channels, self.n_range_in, self.n_doppler_in):
            raise ValueError(
                f"expected [B, {self.in_channels}, {self.n_range_in}, {self.n_doppler_in}], "
                f"got {tuple(x.shape)}"
            )

        z = self.doppler_pool(self.stem(x))                    # [B, d_model, R, Dt]
        n_dop = self.n_doppler_tokens

        # ---- fast axis: one sequence per (batch, Doppler token), scanning range ---------
        seq = z.permute(0, 3, 2, 1).reshape(b * n_dop, r, self.d_model)
        seq = self.fast_ssm(seq)
        seq = self.chan_mix(seq)
        z = seq.view(b, n_dop, r, self.d_model).permute(0, 3, 2, 1)   # [B, d_model, R, Dt]

        # ---- downsample range to the output grid, then scan the Doppler axis ------------
        z = self.range_pool(z)                                 # [B, d_model, R_out, Dt]
        seq = z.permute(0, 2, 3, 1).reshape(b * self.n_range_out, n_dop, self.d_model)
        seq = self.slow_ssm(seq)
        seq = self.post_norm(seq).mean(dim=1)                  # collapse Doppler
        feat = seq.view(b, self.n_range_out, self.d_model).transpose(1, 2)   # [B, d_model, R_out]

        # ---- detection head -------------------------------------------------------------
        az = self.az_proj(feat).view(b, self.head_channels, self.n_azimuth_out, self.n_range_out)
        az = az.permute(0, 1, 3, 2).contiguous()               # [B, head_ch, R_out, A_out]
        h = self.decoder(az)
        detection = torch.cat([torch.sigmoid(self.cls_head(h)), self.reg_head(h)], dim=1)
        return {"detection": detection}
