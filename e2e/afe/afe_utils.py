###
# AFE pytorch port
# take MATLAB code which does floating point truncation
# convert it to pytorch for flexible CPU/GPU computation
###

import torch
import numpy as np

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
    y_mantissa = torch.floor(x_mantissa * 2**mantissa_width)
    y_exp = x_exp + 2**(exp_width-1) - 1
    
    zero_mask = y_exp < 0
    y_mantissa[zero_mask] = 0
    y_exp[zero_mask] = 0
    
    return (x_sign * 2**(exp_width+mantissa_width) + y_exp * 2**mantissa_width + y_mantissa).int()

def interpret_custom_fp(x_uint_raw, exp_width, mantissa_width):
    x_sign, x_exp, x_mantissa = extract_custom_fp_components(x_uint_raw, exp_width, mantissa_width)
    return (1 - 2 * x_sign) * (2**x_exp) * (1 + x_mantissa)

# floating point quantization method
def quantizer_fp(A, exp, mantissa):
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
