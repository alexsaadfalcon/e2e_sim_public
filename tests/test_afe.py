"""Tests for the adaptive-feature-extraction quantizer (e2e.afe.afe_utils)."""

import pytest

torch = pytest.importorskip("torch")

from e2e.afe.afe_utils import quantizer_fp, create_custom_fp, interpret_custom_fp


def test_quantizer_preserves_shape():
    x = torch.randn(8, 5)
    xq = quantizer_fp(x, exp=5, mantissa=6)
    assert xq.shape == x.shape


def test_quantizer_is_approximately_lossless_with_many_bits():
    torch.manual_seed(0)
    x = torch.randn(64)
    # Wide mantissa -> small quantization error.
    xq = quantizer_fp(x, exp=8, mantissa=16)
    assert torch.max(torch.abs(x - xq)).item() < 1e-2


def test_more_mantissa_bits_reduce_error():
    torch.manual_seed(0)
    x = torch.randn(256)
    err_lo = torch.mean((x - quantizer_fp(x, exp=5, mantissa=2)) ** 2).item()
    err_hi = torch.mean((x - quantizer_fp(x, exp=5, mantissa=8)) ** 2).item()
    assert err_hi < err_lo


def test_roundtrip_components_are_finite():
    x = torch.randn(32)
    fp = create_custom_fp(x, 5, 6)
    back = interpret_custom_fp(fp.int(), 5, 6)
    assert torch.all(torch.isfinite(back))


def test_exact_zero_maps_to_zero():
    # Previously 0.0 round-tripped to a small finite number (~3e-5/7.8e-3).
    x = torch.zeros(4)
    xq = quantizer_fp(x, exp=4, mantissa=3)
    assert torch.all(xq == 0.0)


def test_normal_value_roundtrip_unchanged():
    # A normal-range value must still quantize with the same small error as
    # before the edge-case fix (this is the regression guard for the normal path).
    x = torch.tensor([1.5, 0.3, -0.3, 7.0])
    xq = quantizer_fp(x, exp=4, mantissa=3)
    assert torch.all(torch.isfinite(xq))
    assert torch.max(torch.abs(x - xq)).item() < 0.05
    # 1.5 and 7.0 are exactly representable in this format.
    assert xq[0].item() == pytest.approx(1.5)
    assert xq[3].item() == pytest.approx(7.0)


def test_non_finite_inputs_propagate():
    # Documented choice: inf/-inf propagate to inf/-inf, NaN propagates to NaN
    # (they no longer silently become small finite numbers).
    x = torch.tensor([float("inf"), float("-inf"), float("nan")])
    xq = quantizer_fp(x, exp=4, mantissa=3)
    assert torch.isposinf(xq[0])
    assert torch.isneginf(xq[1])
    assert torch.isnan(xq[2])


def test_large_magnitude_saturates_not_wraps():
    # Previously 1e30 wrapped to a wrong-sign / small finite value; it must now
    # saturate to +/-inf and keep its sign.
    x = torch.tensor([1e30, -1e30])
    xq = quantizer_fp(x, exp=4, mantissa=3)
    assert torch.isposinf(xq[0])
    assert torch.isneginf(xq[1])
