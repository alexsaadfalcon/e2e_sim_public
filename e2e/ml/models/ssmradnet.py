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

Raw-ADC input mode (`input_mode="adc"`)
----------------------------------------
`e2e.ml.dataset`'s `input_format="adc"` (see `e2e.ml.dataset.RadarFrameDataset._derive_input`) serves the
raw, un-deinterleaved physical-channel ADC cube `[2*n_rx, n_samples, n_chirps]` instead of
the post-RD-FFT tensor described above. This restores upstream's literal "no FFT" premise
for the fast axis (it now scans real ADC samples, not range bins) -- but the RD stem above
is *wrong* for it: `doppler_pool` average-pools the chirp axis *before* either SSM has seen
it, which for RD is harmless (Doppler bins are already FFT-correlated) but for raw ADC
would average together chirps before `slow_ssm` ever gets to learn their sequence -- exactly
backwards from "let the SSM learn the raw structure". `input_mode="adc"` therefore selects
a different stem (`_ADCStem`, pointwise channel embed, no spatial mixing) and disables the
pre-scan chirp pooling entirely (`doppler_pool` becomes the identity, achieved by setting
`n_doppler_tokens == n_doppler_in`) so `slow_ssm` scans the *full*, unpooled chirp sequence;
pooling only happens after a scan has run (the existing `range_pool`-after-`fast_ssm`
pattern, unchanged, and `slow_ssm`'s own post-scan `.mean(dim=1)` collapse), matching
upstream's intent that pooling never precedes the SSM that is supposed to model that axis.

Honest fork from upstream, not resolved by this code: upstream's raw ADC is **DDMA** (all
TX transmit every chirp, so consecutive chirps are simultaneous-TX snapshots); this
simulator's TDM MIMO config fires **one TX per chirp, round-robin** (chirp `c` was
illuminated by TX `c % n_tx` only -- see `e2e.ml.transforms.tdm_deinterleave`'s docstring).
`input_mode="adc"` does **not** deinterleave (deinterleaving is itself a hand-engineered
MIMO-demux step that would defeat the "raw signal" premise this input mode exists to
serve), so for TDM configs the slow axis `slow_ssm` scans is TX-interleaved: it must learn
the `n_tx`-chirp periodicity itself from raw, abruptly-alternating-aperture data, a strictly
harder (and physically different) inductive-bias problem than DDMA's simultaneous-every-
chirp raw ADC. This is flagged, not solved: whether the extra difficulty is worth the
"honest" raw-signal input is an open research question for this port.

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

    def __init__(self, d_model, d_state, n_layers, backend, d_conv=4, expand=2, chunk_size=None):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand, backend=backend,
                      chunk_size=chunk_size)
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


class _ADCStem(nn.Module):
    """Pointwise (1x1) channel embed for raw-ADC input -- no spatial mixing at all.

    Unlike the RD stem's `3x3` `Conv2d` (which deliberately smooths neighbouring
    range/Doppler bins, already-correlated FFT output), raw ADC samples/chirps carry no
    such correlation to exploit -- a `3x3` conv would just be an uninvited, hand-rolled
    local-smoothing prior with no upstream equivalent (upstream's fast SSM sees each
    chirp's raw samples untouched). `1x1` embeds each `(sample, chirp)` cell's channel
    vector independently, leaving all fast/slow-axis structure for the two SSM scans
    to find.
    """

    def __init__(self, in_channels: int, d_model: int):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.embed(x)


class SSMRadNet(nn.Module):
    """Two-scale selective-SSM radar detector on range-Doppler *or* raw-ADC input.

    Parameters
    ----------
    in_channels, n_range_in, n_doppler_in
        Shape of the input tensor `[B, in_channels, n_range_in, n_doppler_in]`. For
        `input_mode="rd"` (default) this is the `[2*n_rx, range_bin, doppler_bin]` layout
        produced by `e2e.ml.transforms.rd_to_input`; for `input_mode="adc"` it is the raw
        `[2*n_rx, n_samples, n_chirps]` layout produced by `e2e.ml.dataset.RadarFrameDataset._derive_input`
        -- same tensor rank/argument order, different physical axis meaning (samples
        instead of range bins, chirps instead of Doppler bins). See "Raw-ADC input mode"
        in the module docstring.
    n_range_out, n_azimuth_out
        Detection-map geometry (see `e2e.ml.labels.LabelGrid`).
    d_model, d_state, n_layers_fast, n_layers_slow
        SSM width / state size / depth of the range and Doppler scan stages.
    backend
        Forwarded to `MambaBlock` ("auto" | "torch" | "mamba_ssm").
    input_mode
        `"rd"` (default, current/original behavior) or `"adc"` (see module docstring).
    n_doppler_tokens
        `input_mode="rd"` only: Doppler bins are average-pooled to at most this many
        tokens at the stem (no-op if `n_doppler_in` is already smaller; see "Documented
        deviations"). Ignored for `input_mode="adc"`, where the pre-scan pool is disabled
        outright (see "Raw-ADC input mode").
    head_channels
        Width of the conv decoder that produces the detection map.
    ssm_chunk_size
        Forwarded to every `MambaBlock`/`SelectiveSSM` in `fast_ssm`/`slow_ssm` as
        `chunk_size` (see `e2e.ml.models.ssm.selective_scan`'s "CHUNKED SCAN" docs, and
        `e2e.ml.train`'s `--ssm-chunk`): `None` (default) is the original, unchunked
        scan; a smaller positive int trades scan compute for peak activation memory,
        with no change to the model's output (the chunked scan is numerically
        equivalent, see `tests/test_ml_ssmradnet.py`).
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
        input_mode: str = "rd",
        n_doppler_tokens: int = 16,
        head_channels: int = 32,
        ssm_chunk_size=None,
    ):
        super().__init__()
        for name, value in (
            ("in_channels", in_channels), ("n_range_in", n_range_in),
            ("n_doppler_in", n_doppler_in), ("n_range_out", n_range_out),
            ("n_azimuth_out", n_azimuth_out), ("d_model", d_model),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if input_mode not in ("rd", "adc"):
            raise ValueError(f"input_mode must be 'rd' or 'adc', got {input_mode!r}")

        self.in_channels = int(in_channels)
        self.n_range_in = int(n_range_in)
        self.n_doppler_in = int(n_doppler_in)
        self.n_range_out = int(n_range_out)
        self.n_azimuth_out = int(n_azimuth_out)
        self.d_model = int(d_model)
        self.input_mode = input_mode

        # ---- stem: embed channels; RD groups the Doppler axis, ADC pools nothing -------
        if input_mode == "rd":
            self.stem = nn.Sequential(
                nn.Conv2d(self.in_channels, self.d_model, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.d_model),
                nn.SiLU(),
            )
            self.n_doppler_tokens = min(int(n_doppler_tokens), self.n_doppler_in)
            self.doppler_pool = nn.AdaptiveAvgPool2d((self.n_range_in, self.n_doppler_tokens))
        else:  # "adc"
            self.stem = _ADCStem(self.in_channels, self.d_model)
            # No pre-scan chirp pooling: n_doppler_tokens == n_doppler_in makes
            # `doppler_pool` a size-preserving (identity) average pool, so `slow_ssm`
            # below scans the full, unpooled chirp sequence -- see "Raw-ADC input mode".
            self.n_doppler_tokens = self.n_doppler_in
            self.doppler_pool = nn.Identity()

        # ---- two SSM scales + channel mixing -------------------------------------------
        self.fast_ssm = _SSMStage(self.d_model, d_state, n_layers_fast, backend,
                                  chunk_size=ssm_chunk_size)                        # scans range
        self.chan_mix = _ChannelMix(self.d_model)
        self.range_pool = nn.AdaptiveAvgPool2d((self.n_range_out, self.n_doppler_tokens))
        self.slow_ssm = _SSMStage(self.d_model, d_state, n_layers_slow, backend,
                                  chunk_size=ssm_chunk_size)                        # scans Doppler
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
