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
    indexs = np.arange(-bins//2+1, bins//2+1, 1)
    ans = np.zeros(bins)
    for i, index in enumerate(indexs):
        ans[i] = (1/np.sqrt(2*np.pi)) * np.exp(-(index * index) / 2.5)
    # ! 目前的归一化是归一化到“中间”为1
    return ans / ans[bins//2]

def generate_curve(ts: torch.tensor, oc_res: torch.tensor, normals: torch.tensor):
    assert(ts.shape[0] == oc_res.shape[0])
    l = ts.shape[0]
    #normals = gen_normal(bins = l)
    
    res = torch.zeros(l, device = ts.device)
    idxs = torch.where(oc_res)[0]
    #print(idxs)
    offsets = idxs - l // 2
    left_idxs = torch.clip(offsets, 0, l).type(torch.int32)
    right_idxs = torch.clip(offsets + l, 0, l).type(torch.int32)
    for i in range(idxs.shape[0]):
        
        res[left_idxs[i]:right_idxs[i]] += normals[left_idxs[i]-offsets[i]:right_idxs[i]-offsets[i]]
    return res #/ torch.sum(res) * torch.sum(oc_res)

if __name__ == "__main__":
    ll = 25
    print(gen_normal(ll)[(ll-1)//2])