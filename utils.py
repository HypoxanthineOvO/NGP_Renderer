import numpy as np
import torch

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

if __name__ == "__main__":
    aabb = torch.tensor([[-0.5, -0.5, -0.5], [1.5, 1.5, 1.5]])
    ray_o = torch.tensor([3.2373, 3.4593, 0.5000])
    ray_d = torch.tensor([-0.8733, -0.5749, -0.1085])
    
    # Simulate
    t = get_init_t_value(aabb, ray_o, ray_d)
    if isinstance(t, str):
        # continue
        exit()
    while(t <= 6.):
        pos = ray_o + t * ray_d
        print(t, pos)
        # if grid.intersect(pos):
        dt = get_next_voxel(pos, ray_d)
        print(dt)
        if isinstance(dt, str):
            print("END!")
            break
        t += dt