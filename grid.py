import numpy as np
import torch

class DensityGrid:
    def __init__(self, grid, aabb = [[0,0,0], [1,1,1]]):
        '''
        Initialize the Density Grid
        '''
        self.grid = grid
        if(isinstance(aabb, list)):
            aabb = np.array(aabb)
        assert(aabb.shape == (2,3))
        self.aabb = aabb
        self.scales = (self.aabb[1] - self.aabb[0]) * self.grid.shape
        #print(self.scales)
    
    def intersect(self, point):
        if(isinstance(point, list)):
            point = np.array(point)
        if(isinstance(point, torch.Tensor)):
            point = point.detach().cpu().numpy()
        assert(point.shape == (3,))
        # If the point is out of the grid, return 0
        if(np.any(point <= self.aabb[0]) or np.any(point >= self.aabb[1])):
            return 0
        point_idx = np.floor((point - self.aabb[0]) / (self.aabb[1] - self.aabb[0]) * self.grid.shape)
        return self.grid[tuple(point_idx.astype(np.int8))]