import numpy as np
import torch


### Camera

class Camera:
    def __init__(self, resolution, camera_angle, camera_matrix):
        # Resolution: For Generate Image
        self.resolution = resolution
        self.w = self.resolution[0]
        self.h = self.resolution[1]
        self.image = np.zeros((resolution[0], resolution[1], 3)) # RGB Image
        # Parameters
        self.position = np.array([0.0, 0.0, 0.0])
        self.camera_to_world = np.zeros((3, 3))
        self.focal_length = 1.0

        # Camera Coordinate Directions
        self.directions = None
        # Rays Origin and Direction
        self.rays_o = np.zeros((resolution[0], resolution[1], 3))
        self.rays_d = np.zeros((resolution[0], resolution[1], 3))

        assert camera_matrix.shape == (3, 4) or camera_matrix.shape == (4, 4)
        if(camera_matrix.shape == (4, 4)):
            camera_matrix = camera_matrix[:3]

        self.position = camera_matrix[:3, -1]
        self.camera_to_world = camera_matrix[:3, :3]
        self.w = self.resolution[0]
        self.h = self.resolution[1]
        self.focal_length = .5 * self.w / np.tan(.5 * camera_angle)
        # Generate Directions
        i, j = np.meshgrid(
            np.linspace(0, self.w-1, self.w), 
            np.linspace(0, self.h-1, self.h), 
            indexing='xy'
        )
        self.directions = np.stack([(i-0.5 * self.w)/ self.focal_length, -(j-0.5 * self.h)/self.focal_length, -np.ones_like(i)], -1)
        # Transform to World Coordinate
        self.rays_o = np.broadcast_to(self.position, self.directions.shape)
        self.rays_d = np.sum(self.directions[..., np.newaxis, :] * self.camera_to_world, -1)
        
        # Try to transpose
        self.rays_o = self.rays_o[..., [1,2,0]]
        self.rays_d = self.rays_d[..., [1,2,0]]
        # Normalize rays_d
        #self.rays_d = self.rays_d / np.linalg.norm(self.rays_d, axis=-1, keepdims=True)

