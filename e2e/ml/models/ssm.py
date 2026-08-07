"""
Mamba-1 style selective state-space layer in pure PyTorch (no CUDA extensions).

Adapted from AnuvabSen1/SSMRadNet (https://github.com/AnuvabSen1/SSMRadNet). The upstream
repository ships no LICENSE file; the SSMRadNet authors gave written approval for this
adaptation's redistribution (2026-08-07) and requested this integration
(training/evaluation on this simulator's synthetic data). Deviations documented inline.

Why this file exists
--------------------
Every model file upstream does `from mamba_ssm import Mamba` at *module import time*.
`mamba_ssm` ships hand-written CUDA extensions (`selective_scan_cuda`, `causal_conv1d`)
with no Windows wheels, so the whole upstream `model/` package is unimportable on this
box. `SelectiveSSM` below reimplements the same Mamba-1 recurrence in plain torch ops so
the architecture runs (and trains) anywhere torch runs; `MambaBlock` transparently prefers
the real fused kernel when it is importable.

The recurrence (identical to Mamba-1)
-------------------------------------
For an input sequence ``x: [B, L, d_model]``::

    x, z    = in_proj(x).chunk(2)                # [B, L, d_inner] each  (d_inner = expand*d_model)
    x       = silu(depthwise_causal_conv1d(x))   # kernel width d_conv, left-padded
    dt,B_,C = x_proj(x).split([dt_rank, N, N])   # data-dependent (selective) parameters
    dt      = softplus(dt_proj(dt))              # [B, L, d_inner], strictly positive
    A       = -exp(A_log)                        # [d_inner, N], diagonal and negative
    h_t     = exp(dt_t * A) h_{t-1} + (dt_t * B_t) x_t
    y_t     = C_t . h_t + D * x_t
    y       = out_proj(y * silu(z))

Scan algorithm (the only real deviation from upstream, which calls a fused CUDA kernel)
---------------------------------------------------------------------------------------
A python `for t in range(L)` loop is unusable at L=512 (kernel-launch bound, and it stores
L separate autograd graphs). Instead `selective_scan` runs a **Hillis-Steele inclusive
parallel scan in log space** over the first-order linear recurrence, which is associative:

    (a1, b1) then (a2, b2)  ==>  (a1*a2,  a2*b1 + b2)      [h -> a*h + b]

Element ``t`` starts as ``(log a_t, b_t) = (dt_t*A, dt_t*B_t*x_t)`` and is combined with the
element ``d`` positions to its left for ``d = 1, 2, 4, ... >= L`` (identity ``(0, 0)`` when
the left partner falls off the front). After ``ceil(log2 L)`` fully-vectorized steps the
``b`` slot holds ``h_t``. Cost is ``O(L log L)`` elementwise work in ``ceil(log2 L)`` kernel
launches instead of ``O(L)`` launches.

The decay factor is kept in **log space** and only ever exponentiated as ``exp(log a)`` with
``log a <= 0`` (because ``A < 0`` and ``dt > 0``), so every exponential is in ``(0, 1]``.
This is what makes it stable; the tempting alternative -- ``h_t = exp(cum_t) * cumsum(
exp(-cum_s) u_s)`` -- needs ``exp(-cum_s)``, which overflows float32 after a few hundred
steps of real decay.

MEMORY CEILING: each of the ``ceil(log2 L)`` scan steps materializes full
``[B, L, d_inner, d_state]`` tensors, and all of them stay alive in the autograd graph
until backward completes -- peak memory is roughly ``log2(L)`` times one state tensor.
At upstream-RADIal-like dims (L=512, 128 parallel sequences) this OOMs an 8 GB GPU at
batch size 8; use small batches (2 worked on 8 GB), fewer ``n_doppler_tokens``, or a
shorter scan axis. A chunked/checkpointed scan that trades recompute for memory is a
known follow-up, not yet implemented.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["selective_scan", "SelectiveSSM", "MambaBlock", "mamba_ssm_available"]


# --------------------------------------------------------------------------------
# Parallel scan
# --------------------------------------------------------------------------------
def _log_assoc_scan(log_a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Inclusive scan of ``h_t = exp(log_a_t) * h_{t-1} + b_t`` along dim 1 (``h_{-1} = 0``).

    `log_a`, `b` are broadcast-compatible ``[..., L, ...]`` tensors with the sequence on
    dim 1. Hillis-Steele: at stride `d`, every position absorbs the composed operator of
    the position `d` to its left (identity-padded at the front). See the module docstring.
    """
    seq_len = b.shape[1]
    # zero-pad on the *front* of the sequence axis == composing with the identity (1, 0).
    pad = (0,) * (2 * (b.dim() - 2))          # F.pad pads from the last dim backwards
    stride = 1
    while stride < seq_len:
        a_prev = F.pad(log_a[:, :-stride], pad + (stride, 0))
        b_prev = F.pad(b[:, :-stride], pad + (stride, 0))
        b = torch.exp(log_a) * b_prev + b     # uses this step's a_t and the shifted b
        log_a = log_a + a_prev
        stride *= 2
    return b


def selective_scan(u, dt, A, B, C, D=None):
    """Mamba-1 selective scan: ``h_t = exp(dt_t A) h_{t-1} + dt_t B_t u_t``, ``y_t = C_t.h_t + D u_t``.

    Shapes (``P`` = d_inner, ``N`` = d_state)::

        u  : [batch, L, P]      pre-scan input (post conv + SiLU)
        dt : [batch, L, P]      strictly positive step sizes
        A  : [P, N]             diagonal state matrix, must be negative
        B  : [batch, L, N]      input-projection (selective)
        C  : [batch, L, N]      output-projection (selective)
        D  : [P] or None        skip/`feedthrough` gain
        -> y : [batch, L, P]

    Runs in float32 regardless of the caller's dtype (the scan is the numerically delicate
    part of the layer) and casts back on the way out.
    """
    in_dtype = u.dtype
    u32, dt32 = u.float(), dt.float()
    log_decay = dt32.unsqueeze(-1) * A.float()                       # [b, L, P, N] <= 0
    b_term = (dt32 * u32).unsqueeze(-1) * B.float().unsqueeze(2)     # [b, L, P, N]
    h = _log_assoc_scan(log_decay, b_term)                           # [b, L, P, N]
    y = torch.einsum("blpn,bln->blp", h, C.float())
    if D is not None:
        y = y + u32 * D.float()
    return y.to(in_dtype)


# --------------------------------------------------------------------------------
# Layer
# --------------------------------------------------------------------------------
class SelectiveSSM(nn.Module):
    """Pure-PyTorch selective state-space (Mamba-1 style) layer.

    forward(x: [B, L, D]) -> [B, L, D]

    Parameter names/shapes and the initialization schedule follow the reference Mamba
    implementation so that a `mamba_ssm.Mamba` state_dict could be mapped onto this layer
    (and so the dynamics are in the regime the architecture was designed for): `A` is
    initialized to `S4D-Real` (`-1..-d_state` per row) and `dt_proj.bias` is set so that
    `softplus(bias)` is log-uniform in `[dt_min, dt_max]`.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        *,
        dt_rank=None,
        bias: bool = False,
        conv_bias: bool = True,
        dt_min: float = 1e-3,
        dt_max: float = 0.1,
    ):
        super().__init__()
        if d_model <= 0 or d_state <= 0 or d_conv <= 0 or expand <= 0:
            raise ValueError("d_model, d_state, d_conv and expand must all be positive")
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.d_inner = self.expand * self.d_model
        self.dt_rank = int(math.ceil(self.d_model / 16)) if dt_rank is None else int(dt_rank)

        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=bias)
        # depthwise + left-padded => strictly causal (padding is trimmed in forward)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=self.d_conv,
            groups=self.d_inner, padding=self.d_conv - 1, bias=conv_bias,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        # A is stored as log(-A): guarantees A stays negative (=> a stable, decaying scan).
        a = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(a.expand(self.d_inner, self.d_state).contiguous().log())
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self._init_dt(dt_min, dt_max)

    def _init_dt(self, dt_min: float, dt_max: float, floor: float = 1e-4) -> None:
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp_min(floor)
        with torch.no_grad():
            # inverse of softplus, so softplus(bias) == dt at init
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"SelectiveSSM expects [B, L, d_model={self.d_model}], got {tuple(x.shape)}"
            )
        seq_len = x.shape[1]

        u, z = self.in_proj(x).chunk(2, dim=-1)                       # [B, L, P] each
        u = self.conv1d(u.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        u = F.silu(u)

        dt, B, C = self.x_proj(u).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)

        y = selective_scan(u, dt, A, B, C, self.D)
        return self.out_proj(y * F.silu(z))


def mamba_ssm_available() -> bool:
    """True if the fused-CUDA `mamba_ssm` package can be imported."""
    try:
        import mamba_ssm  # noqa: F401
    except Exception:
        return False
    return True


class MambaBlock(nn.Module):
    """SSM block: selective SSM -> Linear -> SiLU (upstream's `MambaSSMBlock`).

    Uses `mamba_ssm.Mamba` when importable (CUDA fast path), else `SelectiveSSM`.

    backend
        ``"auto"``       -- `mamba_ssm` if it imports *and* CUDA is available, else torch.
        ``"torch"``      -- always the pure-PyTorch `SelectiveSSM` (portable, testable).
        ``"mamba_ssm"``  -- force the fused kernel; raises ImportError if unavailable.

    Note: the two backends hold different parameter tensors (`mamba_ssm.Mamba` fuses
    `dt_proj`/`x_proj` differently), so a checkpoint is not portable across backends.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        *,
        backend: str = "auto",
    ):
        super().__init__()
        if backend not in ("auto", "torch", "mamba_ssm"):
            raise ValueError(f"backend must be 'auto', 'torch' or 'mamba_ssm', got {backend!r}")

        resolved = backend
        if backend == "auto":
            resolved = "mamba_ssm" if (mamba_ssm_available() and torch.cuda.is_available()) else "torch"

        if resolved == "mamba_ssm":
            try:
                from mamba_ssm import Mamba
            except Exception as exc:  # pragma: no cover - depends on the local env
                raise ImportError(
                    "backend='mamba_ssm' requires the `mamba_ssm` package (fused CUDA "
                    "selective-scan kernels), which is not installed/importable here "
                    f"({exc}). It has no Windows wheels; use backend='torch' (pure "
                    "PyTorch, works everywhere) or backend='auto' to fall back "
                    "automatically."
                ) from exc
            self.ssm = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.ssm = SelectiveSSM(d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        self.backend = resolved
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.out_proj(self.ssm(x)))
