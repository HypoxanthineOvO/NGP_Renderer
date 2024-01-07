import torch
from torch.autograd import Variable
import math
import warnings
########## Floating-Point Quantize ##########
def Floating_Point_Quantize(x: torch.Tensor, exponent_length: int = 5, mantissa_length: int = 10):

    x_data = x.view(torch.int32)
    
    # Float32
    x_sign = x_data >> 31
    x_exponent = (x_data >> 23) & 0xFF
    x_mantissa = x_data & 0x7FFFFF
    # Generate Float: Sign, Exponent, Mantissa: 1, exponent_bits, mantissa_bits
    x_gf_sign = x_sign
    x_gf_exponent = (x_exponent - 127 + ((1 <<(exponent_length - 1)) - 1))
    x_gf_mantissa = (x_mantissa >> (23 - mantissa_length)) & ((1 << mantissa_length) - 1)
    # Overflow Check
    overflow_mask = torch.where(x_gf_exponent > ((1 << exponent_length) - 1))
    x_gf_exponent[overflow_mask] = ((1 << exponent_length) - 1)
    x_gf_mantissa[overflow_mask] = ((1 << mantissa_length) - 1)
    # Underflow mask
    underflow_mask = torch.where(x_gf_exponent < 0)
    x_gf_exponent[underflow_mask] = 0
    x_gf_mantissa[underflow_mask] = 0
    # Combine
    x_gf_data = (x_gf_sign << 31) | ((x_gf_exponent + (128 - (1 << (exponent_length - 1)))) << 23) | (x_gf_mantissa << (23 - mantissa_length))

    return x_gf_data.view(torch.float32)

########## Fixed-Point Quantize ##########
def Fixed_Point_Quantize(x: torch.Tensor, integral_length: int = 4, mantissa_length: int = 4):
    left_bound, right_bound = -(2 ** (integral_length - 1)), 2 ** (integral_length - 1)
    q_x = torch.clip(x, left_bound, right_bound)
    q_x = torch.floor(q_x * (2 ** mantissa_length) + 0.5) / (2 ** mantissa_length)
    out = (q_x - x).detach() + x
    return out

########## Linear Quantize ##########
warnings.filterwarnings("ignore")

def Get_int_Part(input, overflow_rate):
    """
    input: tensor that need to compute
    overflow_rate: overflow_rate after quantize
    """
    abs_value = input.abs().view(-1)
    sorted_value = abs_value.sort(dim=0, descending=True)[0]
    split_idx = int(overflow_rate * len(sorted_value))
    v = sorted_value[split_idx]
    if isinstance(v, Variable):
        v = float(v.data.cpu())
    sf = math.ceil(math.log2(v+1e-12))
    return sf

def Get_ScaleFactor_From_int_Part(bits, sf_int):
    return bits - 1. - sf_int

def Compute_Scale_Factor(input, bits, ov=0.0):
    sfind = Get_int_Part(input, overflow_rate=ov)
    sfd = Get_ScaleFactor_From_int_Part(bits=bits, sf_int=sfind)
    return sfd

def Quantize_with_ScaleFactor(input, sf, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input) - 1
    delta = math.pow(2.0, -sf)
    bound = math.pow(2.0, bits-1)
    min_val = - bound
    max_val = bound - 1
    rounded = torch.floor(input / delta + 0.5)

    clipped_value = torch.clamp(rounded, min_val, max_val) * delta
    
    original_value = input
    output = (clipped_value - original_value).detach() + original_value
    
    return output

def Linear_Quantize(input: torch.Tensor, bits, ov=0.0):
    sf = Compute_Scale_Factor(input, bits, ov)
    quant_t = Quantize_with_ScaleFactor(input, sf, bits)
    torch.cuda.empty_cache()
    return quant_t


if __name__ == "__main__":
    x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    print(x)
    print("Floating-Point Quantize")
    qx = Floating_Point_Quantize(x)
    print(qx)