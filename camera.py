import numpy as np
import torch


### Camera
SCALE = 0.33

def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0.5, 0.5, 0.5]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose

def get_ray(x, y, hw, transform_matrix, focal, principal = [0.5, 0.5]):
    x = (x + 0.5) / hw[0]
    y = (y + 0.5) / hw[1]
    ray_o = transform_matrix[:3, 3]
    ray_d = np.array([
        (x - principal[0]) * hw[0] / focal,
        (y - principal[1]) * hw[1] / focal,
        1.0,
    ])
    ray_d = np.matmul(transform_matrix[:3, :3], ray_d)
    ray_d = ray_d / np.linalg.norm(ray_d)
    return ray_o, ray_d

class Camera:
    def __init__(self, resolution, camera_angle, camera_matrix):
        # Resolution: For Generate Image
        self.resolution = resolution
        self.w = self.resolution[0]
        self.h = self.resolution[1]
        self.image = np.zeros((resolution[0] * resolution[1], 3)) # RGB Image
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
        ngp_mat = nerf_matrix_to_ngp(camera_matrix)

        rays_o, rays_d = [], []
        for i in range(self.h):
            for j in range(self.w):
                ro, rd = get_ray(j, i, [self.h, self.w], ngp_mat, self.focal_length)
                rays_o.append(ro)
                rays_d.append(rd)
        
        self.rays_o = np.array(rays_o).reshape((-1, 3))
        self.rays_d = np.array(rays_d).reshape((-1, 3))

