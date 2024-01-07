import numpy as np
import torch

class DensityGrid:
    def __init__(self, grid, aabb = [[0,0,0], [1,1,1]]):
        '''
        Initialize the Density Grid
        '''
        self.grid = grid.clone().detach().cuda()
        self.aabb = torch.tensor(aabb, device = "cuda")
    
    def intersect(self, points):
        idxs = torch.sum(
            torch.floor(
                (points - self.aabb[0]) / (self.aabb[1] - self.aabb[0]) * 128) 
                * 
                torch.tensor([128 * 128, 128, 1], device = points.device
            ),dim = -1, dtype = torch.int32)
        
        # Noticed that: a point out of aabb may map to a index in [0, 128**3)
        # So we must check by this
        masks_raw = ((points >= self.aabb[0]) & (points <= self.aabb[1]))
        masks = torch.all(masks_raw, dim = -1).type(torch.int32)
        valid_idxs = idxs * masks
        return self.grid[valid_idxs]