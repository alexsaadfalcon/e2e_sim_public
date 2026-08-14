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
shorter scan axis.

CHUNKED SCAN (opt-in escalation, ``chunk_size``/``checkpoint_chunks``): `selective_scan`
accepts an optional `chunk_size`. When set (and smaller than `L`), the sequence is split
into chunks of that length; each chunk runs the *same* Hillis-Steele scan above, but
wrapped in `torch.utils.checkpoint` (its ``ceil(log2 chunk_size)`` intermediates are
discarded after the forward pass and recomputed on demand during backward, so at most one
chunk's worth of intermediates is ever live), and only the chunk's final state (`h` at its
last position) is carried to the next chunk as an initial condition. Because the
recurrence is associative (see above), splitting the scan into chunks-with-carried-state
changes only the order of evaluation, not the result: `chunk_size=None` (default) is
today's single-chunk behavior, bit-for-bit unchanged. Peak memory for the scan drops from
``O(L log L)`` to ``O(chunk_size log chunk_size + L)`` (one chunk's Hillis-Steele
intermediates, plus the ``O(L)`` unavoidable inputs/outputs/carries) at the cost of
~2x the scan's forward FLOPs (each chunk's forward is computed twice: once to produce
its output, once more during its checkpointed backward). See
`e2e.ml.train`'s `--ssm-chunk` for how this is threaded from the CLI down to
`SSMRadNet`/`MambaBlock`/`SelectiveSSM`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint


__all__ = ["selective_scan", "SelectiveSSM", "MambaBlock", "mamba_ssm_available"]


# --------------------------------------------------------------------------------
# Parallel scan
# --------------------------------------------------------------------------------
def _log_assoc_scan(log_a: torch.Tensor, b: torch.Tensor):
    """Inclusive scan of ``h_t = exp(log_a_t) * h_{t-1} + b_t`` along dim 1 (``h_{-1} = 0``).

    `log_a`, `b` are broadcast-compatible ``[..., L, ...]`` tensors with the sequence on
    dim 1. Hillis-Steele: at stride `d`, every position absorbs the composed operator of
    the position `d` to its left (identity-padded at the front). See the module docstring.

    Returns ``(h, log_a_cum)``: `h` is the scan output (unchanged from the previous,
    single-return-value version -- callers that only used `h` are unaffected bit-for-bit).
    `log_a_cum` is `log_a` after the same loop, which the loop invariant makes the
    *cumulative* sum of the original per-step `log_a` from position 0 through `t` --
    i.e. ``exp(log_a_cum[t])`` is the total decay applied to an initial state `h_{-1}`
    carried into position 0 by the time it reaches position `t`. `selective_scan`'s
    unchunked path ignores it; the chunked path uses it to fold a carried-in state from
    the previous chunk into this chunk's (locally ``h_{-1} = 0``) local scan.
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
    return b, log_a


def _scan_chunk_with_carry(u_chunk: torch.Tensor, dt_chunk: torch.Tensor, A: torch.Tensor,
                            B_chunk: torch.Tensor, carry: torch.Tensor) -> torch.Tensor:
    """One chunk's scan, folding in the previous chunk's final state `carry` (``h_{-1}``).

    Takes `u`/`dt`/`B` (``[..., P]``/``[..., N]``, no `d_state` axis on `u`/`dt`) rather
    than the already-broadcast ``log_a``/``b`` (``[..., P, N]``) `_log_assoc_scan` needs --
    those full `[B, L, d_inner, d_state]` tensors are exactly the memory ceiling this
    whole module exists to cut, so they must never be materialized for more than one
    chunk at a time (materializing them for the *whole* sequence before chunking, then
    only slicing, would silently defeat the point: slicing is a view, not a copy, so the
    full-length tensor would still be the thing actually resident in memory). By
    linearity of the recurrence, `h_t` (global) = ``exp(log_a_cum[t]) * carry +
    local_h[t]``, where `local_h`/`log_a_cum` are `_log_assoc_scan`'s outputs treating
    this chunk in isolation (``h_{-1} = 0``). This whole function is the unit that gets
    checkpointed (see `_chunked_scan`): only its single output `h_chunk` (one chunk's
    worth of `[B, L, d_inner, d_state]` state, not `ceil(log2 chunk_size)` copies of it,
    and not the full-sequence-length version of it either) needs to survive from
    forward to backward.
    """
    log_a_chunk = dt_chunk.unsqueeze(-1) * A                                   # [b, Lc, P, N]
    b_chunk = (dt_chunk * u_chunk).unsqueeze(-1) * B_chunk.unsqueeze(2)        # [b, Lc, P, N]
    local_h, log_a_cum = _log_assoc_scan(log_a_chunk, b_chunk)
    return torch.exp(log_a_cum) * carry.unsqueeze(1) + local_h


def _chunked_scan(u32: torch.Tensor, dt32: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
                   chunk_size: int) -> torch.Tensor:
    """Chunked, checkpointed selective scan from the *pre-broadcast* `u`/`dt`/`A`/`B`.

    Splits the sequence (dim 1) into chunks of `chunk_size` (the last chunk may be
    shorter), scans each chunk under `torch.utils.checkpoint` -- discarding its
    ``ceil(log2 chunk_size)`` Hillis-Steele intermediates (and the chunk-local
    ``[b, chunk_size, P, N]`` ``log_a``/``b`` tensors `_scan_chunk_with_carry` builds
    them from) after the forward pass, recomputing them only if/when backward reaches
    this chunk -- and threads each chunk's final state to the next chunk's `carry`.
    See the module docstring.
    """
    seq_len = u32.shape[1]
    p, n = A.shape
    carry = u32.new_zeros(u32.shape[0], p, n)   # h_{-1} for the first chunk: the identity
    need_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in (u32, dt32, A, B)
    )
    chunks = []
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        u_c, dt_c, B_c = u32[:, start:end], dt32[:, start:end], B[:, start:end]
        if need_grad:
            h_chunk = torch_checkpoint.checkpoint(
                _scan_chunk_with_carry, u_c, dt_c, A, B_c, carry, use_reentrant=False,
            )
        else:
            # No backward will ever run (eval / no_grad): checkpointing would only add
            # the recompute overhead for nothing, so call the plain function instead.
            h_chunk = _scan_chunk_with_carry(u_c, dt_c, A, B_c, carry)
        chunks.append(h_chunk)
        carry = h_chunk[:, -1]
    return torch.cat(chunks, dim=1)


def selective_scan(u, dt, A, B, C, D=None, chunk_size=None):
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

    `chunk_size` (default `None`) is the opt-in memory escalation described in the module
    docstring's "CHUNKED SCAN" section: `None` (or a `chunk_size >= L`) is *exactly* the
    original single-shot Hillis-Steele scan (bit-for-bit -- the chunked code path is not
    even entered), any smaller positive int runs the mathematically equivalent chunked +
    checkpointed scan instead. Raises `ValueError` for `chunk_size < 1`.
    """
    if chunk_size is not None and chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1 or None, got {chunk_size}")
    in_dtype = u.dtype
    u32, dt32, A32 = u.float(), dt.float(), A.float()
    seq_len = u32.shape[1]
    if chunk_size is None or chunk_size >= seq_len:
        # Unchunked path -- bit-for-bit today's original code (log_decay/b_term built
        # once, for the whole sequence). Deliberately NOT reused by the chunked path
        # below: building these full [b, L, P, N] tensors even once, then slicing them
        # per chunk, would keep the whole-sequence allocation resident throughout (a
        # slice is a view, not a copy) and defeat chunking's entire memory point.
        log_decay = dt32.unsqueeze(-1) * A32                             # [b, L, P, N] <= 0
        b_term = (dt32 * u32).unsqueeze(-1) * B.float().unsqueeze(2)     # [b, L, P, N]
        h, _ = _log_assoc_scan(log_decay, b_term)                        # [b, L, P, N]
    else:
        h = _chunked_scan(u32, dt32, A32, B.float(), chunk_size)         # [b, L, P, N]
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

    `chunk_size` (default `None`) is forwarded to `selective_scan` unchanged on every
    `forward` call -- see `selective_scan`'s docstring and the module docstring's
    "CHUNKED SCAN" section for the memory/compute tradeoff it buys.
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
        chunk_size=None,
    ):
        super().__init__()
        if d_model <= 0 or d_state <= 0 or d_conv <= 0 or expand <= 0:
            raise ValueError("d_model, d_state, d_conv and expand must all be positive")
        if chunk_size is not None and chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1 or None, got {chunk_size}")
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.chunk_size = chunk_size
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

        y = selective_scan(u, dt, A, B, C, self.D, chunk_size=self.chunk_size)
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

    `chunk_size` (default `None`) is forwarded to `SelectiveSSM` when `backend` resolves
    to `"torch"` (see `selective_scan`'s "CHUNKED SCAN" docs); `mamba_ssm.Mamba` has no
    such knob (its fused kernel never materializes the `[B, L, d_inner, d_state]` state
    this trades off in the first place), so `chunk_size` is silently unused when the
    fused backend is selected -- not an error, since `backend="auto"` may resolve either
    way depending on the machine, and a caller threading one `chunk_size` through both
    should not have to branch on which backend it landed on.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        *,
        backend: str = "auto",
        chunk_size=None,
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
            self.ssm = SelectiveSSM(d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                                    chunk_size=chunk_size)

        self.backend = resolved
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.out_proj(self.ssm(x)))
