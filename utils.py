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
        return t_enter
    return "No Intersection"

def get_axis_steplength(aabb, ray_d):
    ray_d = torch.reshape(ray_d, (3,))
    return (aabb[1] - aabb[0]) / (128 * torch.abs(ray_d))

def distance_to_next_voxel(position, dir):
    """
    Given a position and a direction, return the distance to the next voxel
    """
    print("DISTANCE TO NEXT VOXEL")
    position, dir = torch.reshape(position, (3,)), torch.reshape(dir, (3,))
    print(position, dir)
    p = 128 * (position - 0.5)
    print(p)
    sign_dir = torch.sign(dir)
    print(sign_dir)
    ts = (torch.floor(p + 0.5 + 0.5 * sign_dir) - p) / dir
    print(ts)
    t = torch.min(ts)
    
    return max(t.item() / 128, 0.0)
    

if __name__ == "__main__":
    aabb = torch.tensor([[-0.5, -0.5, -0.5], [1.5, 1.5, 1.5]])
    ray_o = torch.tensor([3.2373, 3.4593, 0.5000])
    ray_d = torch.tensor([-0.8733, -0.4749, -0.1085])
    it = get_init_t_value(aabb, ray_o, ray_d)
    
    
    target_point = torch.tensor([0.44849876, 0.7497844, 0.585095])
    for t in np.arange(0, 6, 0.01):
        print(t, ray_o + t * ray_d-0.05)