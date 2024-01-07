import numpy as np
import torch

### Ray Marching Utils
def get_init_t_value(aabb, ray_o, ray_d):
    """
    Given a ray with ray_o, ray_d and an axis-aligned bounding box
    return the initial t value.
    AABB is a 2x3 array, each row is the min and max of the AABB
    """
    ray_o, ray_d = torch.reshape(ray_o, (3,)), torch.reshape(ray_d, (3,))
    # Noticed that for each axis, tmin and tmax is not fixed order
    ts = (aabb - ray_o )/ ray_d
    
    tmins, tmaxs = torch.min(ts, dim = 0), torch.max(ts, dim = 0)
    t_enter = torch.max(tmins.values)
    t_exit = torch.min(tmaxs.values)
    
    if (t_enter < t_exit) and (t_exit >= 0):
        return t_enter + 1e-4
    return "No Intersection"
def get_next_voxel(position, direction):
    index = (position + 0.5) * 64
    next_index = torch.floor(index + 0.5 + 0.5 * torch.sign(direction))
    
    if (next_index <= 0).any() or (next_index >= 128).any():
        return "Termination"
    delta_distance = next_index - index
    
    dts = delta_distance / 64. / direction
    dt = torch.min(dts + 1e-5)
    if dt <= 0:
        return 0.0
    return dt

def get_index(pos):
    """
    Utils function to get index of a position
    Assume that:
    resolution of grid is (128, 128, 128) and is located at (-0.5, -0.5, -0.5) to (1.5, 1.5, 1.5)
    """
    return np.floor((pos + 0.5) * 64)


### Rendering Utils
def cumprod_exclusive_ngp(tensor: torch.Tensor) -> torch.Tensor:
    ### Support for my implementation
    in_shape = (tensor.shape[0],)
    out_shape = (tensor.shape[0], 1)
    tensor = tensor.reshape(in_shape)
    # Only works for the last dimension (dim=-1)
    dim = -1
    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, dim)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, dim)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.0

    return cumprod.reshape(out_shape)

### ISCA 2024 Method
def gen_normal(bins = 9):
    #print(bins)
    #print(bins // 2)
    #print(-bins // 2 + 1, bins // 2 + 1)
    indexs = np.arange(-bins//2+1, bins//2+1, 1)
    #print(indexs)
    
    ans = np.zeros(bins)
    for i, index in enumerate(indexs):
        ans[i] = (1/np.sqrt(2*np.pi)) * np.exp(-(index * index) / 2.5)
    # ! 目前的归一化是归一化到“中间”为1
    #print(ans, bins // 2)
    return ans / ans[bins//2]

def generate_curve(oc_res: torch.tensor, normals: torch.tensor):
    l = oc_res.shape[0]
    
    res = torch.zeros(l, device = oc_res.device)
    idxs = torch.where(oc_res)[0]
    #print(idxs)
    offsets = idxs - l // 2
    left_idxs = torch.clip(offsets, 0, l).type(torch.int32)
    right_idxs = torch.clip(offsets + l, 0, l).type(torch.int32)
    #print(offsets)
    #print(left_idxs)
    #print(right_idxs)
    #print(normals)
    for i in range(idxs.shape[0]):
        res[left_idxs[i]:right_idxs[i]] += normals[left_idxs[i]-offsets[i]:right_idxs[i]-offsets[i]]
    return res #/ torch.sum(res) * torch.sum(oc_res)

### Quantize utils
### * Integer Quantize * ###

def get_minmax_scale_factor(input_tensor):
    """Compute Min-Max Scale

    Args:
        input_tensor (torch.tensor): The tensor to be quantize

    Returns:
        scale_factor: The scale factor of thie method
        
    Notices:
        This Method use the max valud of abs(input) as the interval
        of input tensor.
    """
    input_abs_bound = torch.max(torch.abs(input_tensor))
    return torch.ceil(torch.log2(input_abs_bound + 1e-12))

def Integer_Quant(input_tensor, scale_factor, zero_point, bounds = None):
    """Basic Integer Quantize

    Args:
        input_tensor (torch.tensor): The tensor to be quantize
        scale_factor (float): The scale factor for quantize
        zero_point (float): The zero point for quantize
        bounds (tuple(int, int)): The bound of quantize
    Returns:
        Quantized_tensor: The tensor after quantize and dequantize
    """
    integer_tensor = torch.round(input_tensor/scale_factor + zero_point)
    # If given the bound, do clamp
    if bounds is not None:
        integer_tensor = torch.clamp(integer_tensor, bounds[0], bounds[1])
    return (integer_tensor - zero_point) * scale_factor

def MinMax_Quant(input_tensor, bits):
    """Linear Min Max Quantization

    Args:
        input_tensor (torch.tensor): The tensor to be quantize
        bits (int): The bits of quantization
    
    Returns:
        Quantized_tensor: The tensor after quantize and dequantize
    """
    assert bits >= 1
    bound = pow(2.0, bits - 1)
    min_bound, max_bound = -bound, bound - 1
    # Compute Scale Factor
    ## sf_interval - 1 - bits is the log space operation
    ## It means (2^(sf-1)) / 2^(bits)
    log_space_scale_factor = get_minmax_scale_factor(input_tensor) - 1 - bits
    scale_factor = torch.pow(2, log_space_scale_factor)
    
    return Integer_Quant(input_tensor, scale_factor, 0, (min_bound, max_bound))

### * Floating Point Quantize * ###
def FloatingPoint_Quantize(input_tensor, bits = 8):
    """Convert input torch.float32 to float8
    This Implementation is E4M3

    Args:
        input_tensor (torch.tensor): input float32 tensor
        bits (int, optional): The length of quantization. Defaults to 8.
    """


if __name__ == "__main__":
    gen_normal(7)
    gen_normal(8)
    #oc = torch.tensor([1, 0, 0, 1, 1, 1, 0])
    #c = generate_curve(oc, gen_normal(7))
    #print(c)