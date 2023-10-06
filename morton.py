import torch
import numpy as np

def Part_1_By_2(x: torch.tensor):
    x &= 0x000003ff;                 # x = ---- ---- ---- ---- ---- --98 7654 3210
    x = (x ^ (x << 16)) & 0xff0000ff # x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x <<  8)) & 0x0300f00f # x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x <<  4)) & 0x030c30c3 # x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x <<  2)) & 0x09249249 # x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    return x

def morton_naive(x: torch.tensor, y: torch.tensor, z: torch.tensor):
    return Part_1_By_2(x) + (Part_1_By_2(y) << 1) + (Part_1_By_2(z) << 2)

def morton(input):
    return morton_naive(input[..., 0], input[..., 1], input[..., 2])

def inv_Part_1_By_2(x: torch.tensor):
    x = ((x >> 2) | x) & 0x030C30C3
    x = ((x >> 4) | x) & 0x0300F00F
    x = ((x >> 8) | x) & 0x030000FF
    x = ((x >>16) | x) & 0x000003FF
    return x

def inv_morton_naive(input: torch.tensor):
    x = input &        0x09249249
    y = (input >> 1) & 0x09249249
    z = (input >> 2) & 0x09249249
    
    return inv_Part_1_By_2(x), inv_Part_1_By_2(y), inv_Part_1_By_2(z)

def inv_morton(input:torch.tensor):
    x,y,z = inv_morton_naive(input)
    return torch.stack([x,y,z], dim = -1)
