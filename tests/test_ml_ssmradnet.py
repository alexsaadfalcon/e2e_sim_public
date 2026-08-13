"""
Tests for `e2e.ml.models.ssm` (pure-PyTorch selective scan) and `e2e.ml.models.ssmradnet`.

Everything pins ``backend="torch"``: `mamba_ssm` is not installable on this platform (no
Windows wheels for its CUDA extensions), and the portable path is the one that has to be
correct everywhere. Sizes are deliberately tiny except for the one wall-clock guard.
"""

import time

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F

from e2e.ml.models.ssm import MambaBlock, SelectiveSSM, selective_scan
from e2e.ml.models.ssmradnet import SSMRadNet


# --------------------------------------------------------------------------------
# Reference implementations (naive sequential loops -- the thing the parallel scan
# has to reproduce). Deliberately written here, not imported, so they cannot drift
# with the implementation under test.
# --------------------------------------------------------------------------------
def _sequential_scan(u, dt, A, B, C, D):
    """h_t = exp(dt_t A) h_{t-1} + dt_t B_t u_t ; y_t = C_t . h_t + D u_t, one t at a time."""
    b, seq_len, p = u.shape
    n = A.shape[-1]
    h = torch.zeros(b, p, n, dtype=u.dtype, device=u.device)
    ys = []
    for t in range(seq_len):
        decay = torch.exp(dt[:, t].unsqueeze(-1) * A)              # [b, p, n]
        h = decay * h + (dt[:, t] * u[:, t]).unsqueeze(-1) * B[:, t].unsqueeze(1)
        ys.append((h * C[:, t].unsqueeze(1)).sum(-1) + u[:, t] * D)
    return torch.stack(ys, dim=1)


def _sequential_selective_ssm(layer, x):
    """Re-run `layer`'s forward with a python for-loop over the sequence axis."""
    seq_len = x.shape[1]
    u, z = layer.in_proj(x).chunk(2, dim=-1)
    u = layer.conv1d(u.transpose(1, 2))[..., :seq_len].transpose(1, 2)
    u = F.silu(u)
    dt, B, C = layer.x_proj(u).split([layer.dt_rank, layer.d_state, layer.d_state], dim=-1)
    dt = F.softplus(layer.dt_proj(dt))
    A = -torch.exp(layer.A_log)
    y = _sequential_scan(u, dt, A, B, C, layer.D)
    return layer.out_proj(y * F.silu(z))


# --------------------------------------------------------------------------------
# Selective scan correctness
# --------------------------------------------------------------------------------
def test_parallel_scan_matches_sequential_reference(torch_device):
    torch.manual_seed(0)
    b, seq_len, p, n = 2, 24, 8, 5
    u = torch.randn(b, seq_len, p, device=torch_device)
    dt = torch.rand(b, seq_len, p, device=torch_device) * 0.2 + 0.01
    A = -(torch.rand(p, n, device=torch_device) * 3.0 + 0.1)
    B = torch.randn(b, seq_len, n, device=torch_device)
    C = torch.randn(b, seq_len, n, device=torch_device)
    D = torch.randn(p, device=torch_device)

    fast = selective_scan(u, dt, A, B, C, D)
    ref = _sequential_scan(u, dt, A, B, C, D)
    torch.testing.assert_close(fast, ref, rtol=1e-4, atol=1e-5)


def test_selective_ssm_layer_matches_sequential_reference(torch_device):
    torch.manual_seed(1)
    layer = SelectiveSSM(d_model=8, d_state=6, d_conv=4).to(torch_device).eval()
    x = torch.randn(2, 24, 8, device=torch_device)
    with torch.no_grad():
        torch.testing.assert_close(layer(x), _sequential_selective_ssm(layer, x),
                                   rtol=1e-4, atol=1e-5)


def test_selective_scan_stable_over_long_sequences(torch_device):
    """Strong decay over L=512 must not overflow (the log-space scan's whole point)."""
    torch.manual_seed(2)
    b, seq_len, p, n = 1, 512, 4, 4
    u = torch.randn(b, seq_len, p, device=torch_device)
    dt = torch.full((b, seq_len, p), 0.5, device=torch_device)
    A = -torch.full((p, n), 16.0, device=torch_device)   # exp(-8) per step
    B = torch.randn(b, seq_len, n, device=torch_device)
    C = torch.randn(b, seq_len, n, device=torch_device)
    y = selective_scan(u, dt, A, B, C, None)
    assert torch.isfinite(y).all()


# --------------------------------------------------------------------------------
# Chunked / checkpointed scan (memory-ceiling fix): `chunk_size` must not change the
# scan's *result*, only how it is evaluated -- see ssm.py's "CHUNKED SCAN" docstring.
# --------------------------------------------------------------------------------
def _random_scan_inputs(device, b, seq_len, p, n, seed):
    torch.manual_seed(seed)
    u = torch.randn(b, seq_len, p, device=device, requires_grad=True)
    dt = (torch.rand(b, seq_len, p, device=device) * 0.2 + 0.01).requires_grad_()
    A = (-(torch.rand(p, n, device=device) * 3.0 + 0.1)).requires_grad_()
    B = torch.randn(b, seq_len, n, device=device, requires_grad=True)
    C = torch.randn(b, seq_len, n, device=device, requires_grad=True)
    D = torch.randn(p, device=device, requires_grad=True)
    return u, dt, A, B, C, D


@pytest.mark.parametrize("seq_len,chunk_size", [
    (24, 1),     # maximally fine-grained chunking
    (24, 5),     # L not evenly divisible by chunk_size -> a short final chunk
    (24, 8),     # evenly divisible
    (24, 24),    # chunk_size == L (single chunk, still routes through the chunked path... )
    (24, 100),   # ... this one does NOT (chunk_size >= L short-circuits to the unchunked path)
    (37, 6),     # a second, differently-shaped L not evenly divisible by chunk_size
])
def test_selective_scan_chunked_matches_unchunked_forward_and_grad(torch_device, seq_len, chunk_size):
    """Chunking is an evaluation-order change on an associative recurrence, not a
    different computation: forward output AND every input's gradient must match the
    unchunked (`chunk_size=None`) scan within float32 tolerance, for L both divisible
    and not divisible by chunk_size."""
    b, p, n = 2, 6, 5
    inputs_ref = _random_scan_inputs(torch_device, b, seq_len, p, n, seed=0)
    y_ref = selective_scan(*inputs_ref)
    y_ref.sum().backward()
    grads_ref = [t.grad.clone() for t in inputs_ref]

    inputs = _random_scan_inputs(torch_device, b, seq_len, p, n, seed=0)   # same values, fresh graph
    y = selective_scan(*inputs, chunk_size=chunk_size)
    torch.testing.assert_close(y, y_ref, rtol=1e-4, atol=1e-5)

    y.sum().backward()
    for t, t_ref, name in zip(inputs, grads_ref, ("u", "dt", "A", "B", "C", "D")):
        torch.testing.assert_close(t.grad, t_ref, rtol=1e-3, atol=1e-4,
                                   msg=lambda m, name=name: f"grad mismatch for {name}: {m}")


def test_selective_scan_chunk_size_none_is_bitwise_unchanged_regression(torch_device):
    """Pin the unchunked (`chunk_size=None`, today's default) scan's output against a
    hardcoded value, so a future edit to the default path (not just the new chunked
    one) cannot silently drift without a test catching it."""
    # Inputs are generated on CPU with a fixed seed (reproducible regardless of which
    # device the suite runs on -- CPU/CUDA RNG streams differ for the same seed) and
    # then moved to `torch_device`; the scan itself still runs on the library's device.
    torch.manual_seed(42)
    b, seq_len, p, n = 1, 6, 2, 2
    u = torch.randn(b, seq_len, p).to(torch_device)
    dt = (torch.rand(b, seq_len, p) * 0.2 + 0.01).to(torch_device)
    A = (-(torch.rand(p, n) * 3.0 + 0.1)).to(torch_device)
    B = torch.randn(b, seq_len, n).to(torch_device)
    C = torch.randn(b, seq_len, n).to(torch_device)
    D = torch.randn(p).to(torch_device)

    y = selective_scan(u, dt, A, B, C, D)
    expected = torch.tensor([[
        [0.6545, -0.0227], [0.4255, -0.0713], [-2.1623, 0.0399],
        [4.0586, 0.1623], [0.9564, -0.0689], [0.9418, -0.1328],
    ]], device=torch_device)
    torch.testing.assert_close(y, expected, rtol=1e-3, atol=1e-4)


def test_selective_scan_rejects_non_positive_chunk_size(torch_device):
    u, dt, A, B, C, D = _random_scan_inputs(torch_device, 1, 4, 2, 2, seed=0)
    for bad in (0, -1):
        with pytest.raises(ValueError, match="chunk_size"):
            selective_scan(u, dt, A, B, C, D, chunk_size=bad)


def test_selective_ssm_chunk_size_matches_unchunked(torch_device):
    """Same equivalence check at the `SelectiveSSM` layer level (shared weights)."""
    torch.manual_seed(7)
    layer = SelectiveSSM(d_model=8, d_state=6, d_conv=4).to(torch_device)
    layer_chunked = SelectiveSSM(d_model=8, d_state=6, d_conv=4, chunk_size=5).to(torch_device)
    layer_chunked.load_state_dict(layer.state_dict())

    x = torch.randn(2, 22, 8, device=torch_device, requires_grad=True)
    x_chunked = x.detach().clone().requires_grad_()

    y = layer(x)
    y.sum().backward()
    y_chunked = layer_chunked(x_chunked)
    y_chunked.sum().backward()

    torch.testing.assert_close(y_chunked, y, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(x_chunked.grad, x.grad, rtol=1e-3, atol=1e-4)


def test_selective_ssm_rejects_non_positive_chunk_size():
    with pytest.raises(ValueError, match="chunk_size"):
        SelectiveSSM(d_model=8, chunk_size=0)


def test_mamba_block_threads_chunk_size_to_selective_ssm(torch_device):
    block = MambaBlock(8, backend="torch", chunk_size=3).to(torch_device)
    assert block.ssm.chunk_size == 3
    assert block(torch.randn(2, 10, 8, device=torch_device)).shape == (2, 10, 8)


def test_selective_ssm_is_causal(torch_device):
    """Perturbing position t leaves every output at position < t untouched."""
    torch.manual_seed(3)
    layer = SelectiveSSM(d_model=8, d_state=6).to(torch_device).eval()
    x = torch.randn(2, 16, 8, device=torch_device)
    t = 9
    x_perturbed = x.clone()
    x_perturbed[:, t] += 3.0

    with torch.no_grad():
        y, y_perturbed = layer(x), layer(x_perturbed)
    torch.testing.assert_close(y[:, :t], y_perturbed[:, :t], rtol=0, atol=1e-6)
    # sanity: the perturbation is not simply being ignored
    assert (y[:, t:] - y_perturbed[:, t:]).abs().max() > 1e-3


def test_selective_ssm_shape_and_device(torch_device):
    layer = SelectiveSSM(d_model=8).to(torch_device)
    y = layer(torch.randn(3, 7, 8, device=torch_device))
    assert y.shape == (3, 7, 8)
    assert y.device.type == torch.device(torch_device).type
    with pytest.raises(ValueError):
        layer(torch.randn(3, 7, 9, device=torch_device))


# --------------------------------------------------------------------------------
# Backend selection
# --------------------------------------------------------------------------------
def test_backend_torch_uses_pure_pytorch(torch_device):
    block = MambaBlock(8, backend="torch").to(torch_device)
    assert block.backend == "torch"
    assert isinstance(block.ssm, SelectiveSSM)
    assert block(torch.randn(2, 5, 8, device=torch_device)).shape == (2, 5, 8)


def test_backend_auto_falls_back_without_mamba_ssm():
    from e2e.ml.models.ssm import mamba_ssm_available
    block = MambaBlock(8, backend="auto")
    expected = "mamba_ssm" if (mamba_ssm_available() and torch.cuda.is_available()) else "torch"
    assert block.backend == expected


def test_backend_mamba_ssm_raises_clear_import_error():
    from e2e.ml.models.ssm import mamba_ssm_available
    if mamba_ssm_available():
        pytest.skip("mamba_ssm is installed here; the failure path cannot be exercised")
    with pytest.raises(ImportError, match="mamba_ssm"):
        MambaBlock(8, backend="mamba_ssm")


def test_backend_rejects_unknown_name():
    with pytest.raises(ValueError, match="backend"):
        MambaBlock(8, backend="s4")


# --------------------------------------------------------------------------------
# SSMRadNet
# --------------------------------------------------------------------------------
def _small_net(device, **kwargs):
    kwargs.setdefault("d_model", 16)
    kwargs.setdefault("d_state", 8)
    kwargs.setdefault("n_layers_fast", 1)
    kwargs.setdefault("n_layers_slow", 1)
    kwargs.setdefault("head_channels", 8)
    return SSMRadNet(24, 128, 32, 32, 48, backend="torch", **kwargs).to(device)


def test_ssmradnet_forward_shape_and_detection_contract(torch_device):
    torch.manual_seed(4)
    net = SSMRadNet(24, 128, 32, 32, 48, backend="torch").to(torch_device)
    x = torch.randn(2, 24, 128, 32, device=torch_device)

    out = net(x)
    assert set(out) == {"detection"}
    det = out["detection"]
    assert det.shape == (2, 3, 32, 48)
    assert det.device.type == torch.device(torch_device).type
    # channel 0 is objectness through a sigmoid; channels 1-2 are unbounded residuals
    assert det[:, 0].min() >= 0.0 and det[:, 0].max() <= 1.0
    assert torch.isfinite(det).all()

    det.sum().backward()
    grads = [p.grad for p in net.parameters()]
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)
    assert max(g.abs().max().item() for g in grads) > 0.0


def test_ssmradnet_rejects_wrong_input_shape(torch_device):
    net = _small_net(torch_device)
    with pytest.raises(ValueError, match=r"\[B, C, R, D\]"):
        net(torch.randn(2, 24, 128, device=torch_device))
    with pytest.raises(ValueError, match="expected"):
        net(torch.randn(2, 24, 64, 32, device=torch_device))


def test_ssmradnet_param_count_is_small(torch_device):
    net = SSMRadNet(24, 128, 32, 32, 48, backend="torch")
    n_params = sum(p.numel() for p in net.parameters())
    assert 1e5 < n_params < 3e6, n_params


def test_ssmradnet_overfits_a_fixed_batch(torch_device):
    """3 optimizer steps on one fixed batch must strictly reduce the loss."""
    torch.manual_seed(5)
    net = _small_net(torch_device).train()
    x = torch.randn(4, 24, 128, 32, device=torch_device)
    target = torch.rand(4, 3, 32, 48, device=torch_device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)

    losses = []
    for _ in range(3):
        opt.zero_grad()
        loss = F.mse_loss(net(x)["detection"], target)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    with torch.no_grad():
        losses.append(F.mse_loss(net(x)["detection"], target).item())

    assert all(b < a for a, b in zip(losses, losses[1:])), losses


# --------------------------------------------------------------------------------
# SSMRadNet, input_mode="adc" (raw-ADC stem)
# --------------------------------------------------------------------------------
def _small_adc_net(device, **kwargs):
    """A tiny `input_mode="adc"` net: [B, 8, n_samples=32, n_chirps=6] -> detection."""
    kwargs.setdefault("d_model", 16)
    kwargs.setdefault("d_state", 8)
    kwargs.setdefault("n_layers_fast", 1)
    kwargs.setdefault("n_layers_slow", 1)
    kwargs.setdefault("head_channels", 8)
    return SSMRadNet(8, 32, 6, 8, 12, backend="torch", input_mode="adc", **kwargs).to(device)


def test_adcstem_forward_shape_and_detection_contract(torch_device):
    torch.manual_seed(10)
    net = _small_adc_net(torch_device)
    x = torch.randn(2, 8, 32, 6, device=torch_device)

    out = net(x)
    assert set(out) == {"detection"}
    det = out["detection"]
    assert det.shape == (2, 3, 8, 12)
    assert det.device.type == torch.device(torch_device).type
    assert det[:, 0].min() >= 0.0 and det[:, 0].max() <= 1.0
    assert torch.isfinite(det).all()


def test_adcstem_disables_pre_scan_chirp_pooling(torch_device):
    """`input_mode="adc"` must not pool the chirp axis before any SSM sees it."""
    net = _small_adc_net(torch_device)
    assert isinstance(net.doppler_pool, torch.nn.Identity)
    assert net.n_doppler_tokens == net.n_doppler_in == 6
    assert isinstance(net.stem, type(net.stem))  # sanity: constructed without error
    from e2e.ml.models.ssmradnet import _ADCStem
    assert isinstance(net.stem, _ADCStem)


def test_adcstem_grads_finite(torch_device):
    torch.manual_seed(11)
    net = _small_adc_net(torch_device).train()
    x = torch.randn(2, 8, 32, 6, device=torch_device)

    det = net(x)["detection"]
    det.sum().backward()
    grads = [p.grad for p in net.parameters()]
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)
    assert max(g.abs().max().item() for g in grads) > 0.0


def test_adcstem_overfits_a_fixed_batch(torch_device):
    """3 optimizer steps on one fixed batch must strictly reduce the loss (adc mode)."""
    torch.manual_seed(12)
    net = _small_adc_net(torch_device).train()
    x = torch.randn(4, 8, 32, 6, device=torch_device)
    target = torch.rand(4, 3, 8, 12, device=torch_device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)

    losses = []
    for _ in range(3):
        opt.zero_grad()
        loss = F.mse_loss(net(x)["detection"], target)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    with torch.no_grad():
        losses.append(F.mse_loss(net(x)["detection"], target).item())

    assert all(b < a for a, b in zip(losses, losses[1:])), losses


def test_adcstem_fast_axis_scan_is_causal(torch_device):
    """The sample-axis (fast) scan must stay causal through the pointwise ADC stem.

    Perturbing raw sample index `t` (across all channels/chirps) must leave every
    fast-axis-scan output at position < t untouched -- checked directly on
    `net.stem` + `net.fast_ssm` (before `range_pool`/the conv decoder mix positions
    together spatially, which would otherwise mask a causality break in the scan
    itself).
    """
    torch.manual_seed(13)
    net = _small_adc_net(torch_device).eval()
    x = torch.randn(2, 8, 32, 6, device=torch_device)
    t = 20
    x_perturbed = x.clone()
    x_perturbed[:, :, t, :] += 5.0

    def _fast_axis_seq(inp):
        z = net.doppler_pool(net.stem(inp))                       # identity pool for adc
        b, d_model, r, n_dop = z.shape
        seq = z.permute(0, 3, 2, 1).reshape(b * n_dop, r, d_model)
        return net.fast_ssm(seq)

    with torch.no_grad():
        y = _fast_axis_seq(x)
        y_perturbed = _fast_axis_seq(x_perturbed)

    torch.testing.assert_close(y[:, :t], y_perturbed[:, :t], rtol=0, atol=1e-5)
    assert (y[:, t:] - y_perturbed[:, t:]).abs().max() > 1e-3


def test_input_mode_rd_default_unchanged(torch_device):
    """`input_mode="rd"` (implicit default) matches an explicit `input_mode="rd"` build.

    Regression guard: adding the `input_mode`/adc path must not perturb the existing
    RD default in any way (same stem type, same pooling, identical output given the
    same seed/weights).
    """
    torch.manual_seed(14)
    net_default = _small_net(torch_device)
    torch.manual_seed(14)
    net_explicit = _small_net(torch_device, input_mode="rd")

    assert net_default.input_mode == "rd"
    assert type(net_default.stem) is type(net_explicit.stem)
    assert isinstance(net_default.doppler_pool, torch.nn.AdaptiveAvgPool2d)

    x = torch.randn(2, 24, 128, 32, device=torch_device)
    with torch.no_grad():
        out_default = net_default(x)["detection"]
        out_explicit = net_explicit(x)["detection"]
    torch.testing.assert_close(out_default, out_explicit)


def test_ssmradnet_rejects_unknown_input_mode(torch_device):
    with pytest.raises(ValueError, match="input_mode"):
        SSMRadNet(8, 32, 6, 8, 12, backend="torch", input_mode="raw")


def test_ssmradnet_forward_wall_clock(torch_device):
    """A python for-loop over L=512 would blow this budget by orders of magnitude.

    One layer per scale keeps the CPU-only CI comfortable; the guard is on the scan
    implementation, not on absolute throughput.
    """
    torch.manual_seed(6)
    net = SSMRadNet(24, 512, 64, 32, 48, backend="torch",
                    n_layers_fast=1, n_layers_slow=1).to(torch_device).eval()
    x = torch.randn(2, 24, 512, 64, device=torch_device)

    with torch.no_grad():
        start = time.perf_counter()
        out = net(x)
        if torch.device(torch_device).type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    assert out["detection"].shape == (2, 3, 32, 48)
    assert elapsed < 10.0, f"forward took {elapsed:.2f}s"
