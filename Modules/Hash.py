import torch
from torch.nn import Module
import math

def Hashing(vertex: torch.Tensor, size: int, non_hashing_resolution: float = 0.0):
    x,y,z = vertex[..., 0], vertex[..., 1], vertex[..., 2]
    
    if(non_hashing_resolution == 0.0):
        index = (((x * 1) ^ (y * 2654435761) ^ (z * 805459861)) % size + size) % size
    else:
        int_scale = (int)(non_hashing_resolution)
        index = (x + y * int_scale + z * int_scale * int_scale) % size
    index = index.type(torch.int32)
    return index

class HashEncoding(Module):
    def __init__(self, n_input_dims: int, encoding_config: dict):
        super().__init__()
        
        self.n_levels: int = encoding_config.get("n_levels", 16)
        self.n_feature_per_level: int = encoding_config.get("n_features_per_level", 2)
        self.log2_hashmap_size: int = encoding_config.get("log2_hashmap_size", 19)
        self.base_resolution: int = encoding_config.get("base_resolution", 16)
        self.per_level_scale: float = encoding_config.get("per_level_scale", 1.38191288)
        
        # Generate Parameters
        hash_grids = []
        scales = []
        sizes = []
        
        for i in range(self.n_levels):
            scale_raw = math.pow(
                2.0, i * math.log2(self.per_level_scale)
            ) * self.base_resolution - 1.0
            resolution: int = math.ceil(scale_raw) + 1
            num_of_features_raw: int = math.ceil(math.pow(resolution, 3) / 8) * 8
            THREDHOLD: int = int(math.pow(2, self.log2_hashmap_size))
            num_of_features: int = min(num_of_features_raw, THREDHOLD)
            
            hash_grid = torch.nn.Embedding(
                num_embeddings = num_of_features,
                embedding_dim = self.n_feature_per_level
            )
            scales.append(scale_raw)
            sizes.append(num_of_features)
            
            hash_grids.append(hash_grid)
        self.grid = torch.nn.ModuleList(hash_grids)
        self.scales = scales
        self.sizes = sizes
    def load_states(self, states: torch.Tensor):
        state_dict = {}
        for i, size in enumerate(self.sizes):
            offset = size * self.n_feature_per_level
            state_dict[f"grid.{i}.weight"] = states[:offset].reshape([size, self.n_feature_per_level])
            states = states[offset:]
        self.load_state_dict(state_dict)
        
    def forward(self, inputs: torch.Tensor):
        shape = list(inputs.shape)
        assert(len(shape) == 2)
        assert(shape[-1] == 3)
        
        outputs = torch.zeros(
            [shape[0], self.n_levels * self.n_feature_per_level], 
            dtype = torch.float32, device = inputs.device)
        
        for level in range(self.n_levels):
            scale = self.scales[level]
            resolution = math.ceil(scale) + 1.0
            if(self.sizes[level] >= (1 << self.log2_hashmap_size)):
                resolution = 0.0
            
            inputs_scale = inputs * scale + 0.5
            inputs_grid = torch.floor(inputs_scale)
            inputs_delta = inputs_scale - inputs_grid
            
            inputs_grid = inputs_grid.type(torch.int32)
            x_grid, y_grid, z_grid = inputs_grid[..., 0], inputs_grid[..., 1], inputs_grid[..., 2]
            dx, dy, dz = inputs_delta[..., 0], inputs_delta[..., 1], inputs_delta[..., 2]
            
            v_000 = torch.stack([x_grid, y_grid, z_grid], dim = -1)
            v_001 = torch.stack([x_grid, y_grid, z_grid + 1], dim = -1)
            v_010 = torch.stack([x_grid, y_grid + 1, z_grid], dim = -1)
            v_011 = torch.stack([x_grid, y_grid + 1, z_grid + 1], dim = -1)
            v_100 = torch.stack([x_grid + 1, y_grid, z_grid], dim = -1)
            v_101 = torch.stack([x_grid + 1, y_grid, z_grid + 1], dim = -1)
            v_110 = torch.stack([x_grid + 1, y_grid + 1, z_grid], dim = -1)
            v_111 = torch.stack([x_grid + 1, y_grid + 1, z_grid + 1], dim = -1)
            
            w_000 = torch.reshape((1 - dx) * (1 - dy) * (1 - dz), (-1, 1))
            w_001 = torch.reshape((1 - dx) * (1 - dy) * dz, (-1, 1))
            w_010 = torch.reshape((1 - dx) * dy * (1 - dz), (-1, 1))
            w_011 = torch.reshape((1 - dx) * dy * dz, (-1, 1))
            w_100 = torch.reshape(dx * (1 - dy) * (1 - dz), (-1, 1))
            w_101 = torch.reshape(dx * (1 - dy) * dz, (-1, 1))
            w_110 = torch.reshape(dx * dy * (1 - dz), (-1, 1))
            w_111 = torch.reshape(dx * dy * dz, (-1, 1))
            
            f_000 = self.grid[level](Hashing(v_000, self.sizes[level], resolution))
            f_001 = self.grid[level](Hashing(v_001, self.sizes[level], resolution))
            f_010 = self.grid[level](Hashing(v_010, self.sizes[level], resolution))
            f_011 = self.grid[level](Hashing(v_011, self.sizes[level], resolution))
            f_100 = self.grid[level](Hashing(v_100, self.sizes[level], resolution))
            f_101 = self.grid[level](Hashing(v_101, self.sizes[level], resolution))
            f_110 = self.grid[level](Hashing(v_110, self.sizes[level], resolution))
            f_111 = self.grid[level](Hashing(v_111, self.sizes[level], resolution))
            
            features = f_000 * w_000 + f_001 * w_001 + f_010 * w_010 + f_011 * w_011 \
            + f_100 * w_100 + f_101 * w_101 + f_110 * w_110 + f_111 * w_111
            
            outputs[..., 
                level * self.n_feature_per_level: (level + 1) * self.n_feature_per_level
                ] = features
        return outputs