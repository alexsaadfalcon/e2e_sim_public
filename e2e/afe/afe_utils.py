###
# AFE pytorch port
# take MATLAB code which does floating point truncation
# convert it to pytorch for flexible CPU/GPU computation
###

import torch
import numpy as np

# Sentinel exponent codes used by the custom-FP encoding.
#
# The packed integer produced by ``create_custom_fp_from_components`` uses a
# biased exponent field of width ``exp_width``.  Mirroring IEEE-754 we reserve
# two exponent codes so that special values survive the round-trip:
#   * biased exponent ``0``                 -> zero (mantissa ignored)
#   * biased exponent ``(1<<exp_width)-1``  -> non-finite (inf if mantissa==0,
#                                              NaN otherwise)
# Normal (finite, non-zero) values use biased exponents in
# ``[1, (1<<exp_width)-2]``; the corresponding real exponent therefore lies in
# ``[1 - bias, (1<<exp_width)-2 - bias]`` where ``bias = 2**(exp_width-1) - 1``.
# Inputs whose magnitude exceeds the largest representable normal saturate to
# +/-inf; inputs smaller than the smallest representable normal flush to zero.

def create_custom_fp(x, exp_width, mantissa_width):
    x_size = x.shape
    x = x.flatten().float()
    x_uint = x.view(torch.int32)

    x_sign, x_exp, x_mantissa = extract_custom_fp_components(x_uint, 8, 23)

    return create_custom_fp_from_components(x_sign, x_exp, x_mantissa, exp_width, mantissa_width).reshape(x_size)

def extract_custom_fp_components(x_uint_raw, exp_width, mantissa_width):
    x_uint = x_uint_raw & ((1 << (exp_width + mantissa_width + 1)) - 1)
    x_sign = (x_uint >> (exp_width + mantissa_width)).float()
    x_exp = (x_uint >> mantissa_width) & ((1 << exp_width) - 1)
    x_exp = x_exp.float() - (2**(exp_width-1) - 1)
    x_mantissa = (x_uint & ((1 << mantissa_width) - 1)).float() / (2**mantissa_width)
    return x_sign, x_exp, x_mantissa

def create_custom_fp_from_components(x_sign, x_exp, x_mantissa, exp_width, mantissa_width):
    # ``x_exp`` is the real (unbiased) IEEE-754 float32 exponent and
    # ``x_mantissa`` the fractional mantissa in [0, 1).  IEEE-754 encodes the
    # following specials in those components (bias 127):
    #   * zero / denormal -> stored exp 0  -> x_exp == -127
    #   * inf / NaN        -> stored exp 255 -> x_exp == 128 (inf: mant==0)
    bias = 2**(exp_width - 1) - 1
    max_exp_code = (1 << exp_width) - 1            # reserved: non-finite
    max_normal_exp_code = max_exp_code - 1         # largest finite biased exp

    src_zero = x_exp <= -127            # zero or (flushed) denormal from float32
    src_inf = (x_exp >= 128) & (x_mantissa == 0)
    src_nan = (x_exp >= 128) & (x_mantissa != 0)

    y_mantissa = torch.floor(x_mantissa * 2**mantissa_width)
    y_exp = x_exp + bias

    # Underflow (biased exponent <= 0, includes the float32 zero/denormal case)
    # flushes to the zero code.
    zero_mask = (y_exp < 1) | src_zero
    y_mantissa = torch.where(zero_mask, torch.zeros_like(y_mantissa), y_mantissa)
    y_exp = torch.where(zero_mask, torch.zeros_like(y_exp), y_exp)

    # Exponent overflow saturates to +/-inf instead of integer-wrapping into a
    # wrong-sign small value.
    overflow_mask = (y_exp > max_normal_exp_code) & ~zero_mask & ~src_inf & ~src_nan
    inf_mask = src_inf | overflow_mask
    y_exp = torch.where(inf_mask, torch.full_like(y_exp, max_exp_code), y_exp)
    y_mantissa = torch.where(inf_mask, torch.zeros_like(y_mantissa), y_mantissa)

    # NaN: reserved exponent with a non-zero mantissa.
    y_exp = torch.where(src_nan, torch.full_like(y_exp, max_exp_code), y_exp)
    y_mantissa = torch.where(
        src_nan, torch.full_like(y_mantissa, max(1.0, 2**mantissa_width - 1)), y_mantissa
    )

    return (x_sign * 2**(exp_width+mantissa_width) + y_exp * 2**mantissa_width + y_mantissa).int()

def interpret_custom_fp(x_uint_raw, exp_width, mantissa_width):
    x_uint = x_uint_raw & ((1 << (exp_width + mantissa_width + 1)) - 1)
    x_sign = (x_uint >> (exp_width + mantissa_width)).float()
    exp_code = (x_uint >> mantissa_width) & ((1 << exp_width) - 1)
    mant_code = (x_uint & ((1 << mantissa_width) - 1)).float()

    x_exp = exp_code.float() - (2**(exp_width-1) - 1)
    x_mantissa = mant_code / (2**mantissa_width)

    sign = 1 - 2 * x_sign
    value = sign * (2**x_exp) * (1 + x_mantissa)

    # Decode reserved exponent codes.
    zero_mask = exp_code == 0
    special_mask = exp_code == (1 << exp_width) - 1
    inf_mask = special_mask & (mant_code == 0)
    nan_mask = special_mask & (mant_code != 0)

    value = torch.where(zero_mask, torch.zeros_like(value), value)
    value = torch.where(inf_mask, sign * torch.full_like(value, float("inf")), value)
    value = torch.where(nan_mask, torch.full_like(value, float("nan")), value)
    return value

# floating point quantization method
def quantizer_fp(A, exp, mantissa):
    """Round-trip ``A`` through a custom low-precision floating-point format.

    The format has 1 sign bit, ``exp`` exponent bits and ``mantissa`` mantissa
    bits.  Following IEEE-754, two exponent codes are reserved so that special
    values survive the round-trip:

      * exact ``0`` (and any float32 zero/denormal/underflow) maps to ``0``;
      * ``+/-inf`` round-trips to ``+/-inf``;
      * ``NaN`` round-trips to ``NaN``;
      * magnitudes above the largest representable normal saturate to
        ``+/-inf`` (they no longer integer-wrap to a wrong-sign small value).

    Representable normal magnitudes span exponents in
    ``[2 - 2**(exp-1), 2**(exp-1) - 1]``.  Normal-range inputs are quantized
    exactly as before (this fix only changes the edge behavior).
    """
    A_fp = create_custom_fp(A, exp, mantissa)
    return interpret_custom_fp(A_fp, exp, mantissa)

matmul_noise_settings = {
    'mean': 0,
    'std': 0.10,
}
def approx_matmul(A, B, noise_settings=None):
    if noise_settings is None:
        noise_settings = matmul_noise_settings
    C = A @ B
    # compute exact matmul
    # then modify by going from 1.0-std to 1.0+std times each element
    std = noise_settings['std']
    # scaling_matrix = (1 - std) + 2 * std * torch.rand_like(C)
    scaling_matrix = 1 + std * torch.randn_like(C)
    C_pert = C * scaling_matrix
    return C_pert

def approx_matmul_afe(A, B):
    raise NotImplementedError

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.randn(20, 10, dtype=torch.float32, device=device)
    exp = 4
    mantissa = 3
    Aq = quantizer_fp(A, exp, mantissa)
    print('quantization error:', torch.max(torch.abs(A - Aq)).item(), torch.mean((A - Aq) ** 2).item())

    afe_matmul_mean = np.load('matmul/HW_errors_based_on_output_val_mean_0p60v_power.npy')
    afe_matmul_std = np.load('matmul/HW_errors_based_on_output_val_stdev_0p60v_power.npy')
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(afe_matmul_mean, label='mean')
    plt.plot(afe_matmul_std, label='std')
    plt.legend()
    plt.show()

    B = torch.randn(10, 20, dtype=torch.float32, device=device)
    C = approx_matmul(A, B)
    print('approximation error:', torch.max(torch.abs(A @ B - C)).item(), torch.mean((A @ B - C) ** 2).item())
