import numpy as np
import torch
import json, time, os, msgpack
from grid import DensityGrid
from morton import *
import nerfacc
    
def load_msgpack(path: str):
    print(f"Loding Msgpack from {path}")
    # Return Value
    res = {}
    # Get Morton3D Object
    
    # Set File Path
    assert (os.path.isfile(path))
    dir_name, full_file_name = os.path.split(path)
    file_name, ext_name = os.path.splitext(full_file_name)
    # Load the msgpack
    with open(path, 'rb') as f:
        unpacker = msgpack.Unpacker(f, raw = False, max_buffer_size = 0)
        config = next(unpacker)

    # Set Model Parameters
    # Total: 12206480 Parameters
    params_binary = np.frombuffer(config["snapshot"]["params_binary"], dtype = np.float16, offset = 0)
    # Transform to torch tensor
    params_binary = torch.tensor(params_binary, dtype = torch.float32)
    # Generate Parameters Dictionary
    params = {}
    # Params for Hash Encoding Network
    ## Network Params Size: 32 * 64 + 64 * 16 = 3072
    hashenc_params_network = params_binary[:(32 * 64 + 64 * 16)]
    params_binary = params_binary[(32 * 64 + 64 * 16):]
    # Params for RGB Network
    ## Network Params Size: 32 * 64 + 64 * 64 + 64 * 16 = 7168
    rgb_params_network = params_binary[:(32 * 64 + 64 * 64 + 64 * 16)]
    params_binary = params_binary[(32 * 64 + 64 * 64 + 64 * 16):]
    # Params for Hash Encoding Grid
    ## Grid size: 12196240
    hashenc_params_grid = params_binary

    # Generate Final Parameters
    params["HashEncoding"] = torch.concat([hashenc_params_network, hashenc_params_grid,  ])
    params["RGB"] = rgb_params_network
    res["params"] = params
    # Occupancy Grid Part
    grid_raw = torch.tensor(np.clip(
        np.frombuffer(config["snapshot"]["density_grid_binary"],dtype=np.float16).astype(np.float32),
        0, 1) > 0.01, dtype = torch.int8)
    grid = torch.zeros([128 * 128 * 128], dtype = torch.int8)
    x, y, z = inv_morton_naive(torch.arange(0, 128**3, 1))
    grid[x * 128 * 128 + y * 128 + z] = grid_raw
    
    # For AABB: we only consider k = 1
    ## The Domain is [-0.5, -0.5, -0.5] to [1.5, 1.5, 1.5]
    oc_grid = DensityGrid(grid,[[-0.5, -0.5, -0.5], [1.5, 1.5, 1.5]])
    
    res["OccupancyGrid"] = oc_grid
    
    print("Msgpack Loaded!")
    return res

def load_msgpack_new(path: str):
    print(f"Loding Msgpack from {path}")
    # Return Value
    res = {}
    # Get Morton3D Object
    
    # Set File Path
    assert (os.path.isfile(path))
    dir_name, full_file_name = os.path.split(path)
    file_name, ext_name = os.path.splitext(full_file_name)
    # Load the msgpack
    with open(path, 'rb') as f:
        unpacker = msgpack.Unpacker(f, raw = False, max_buffer_size = 0)
        config = next(unpacker)
    Hash_Hyper_param = config["encoding"]
    print(Hash_Hyper_param['base_resolution'])
    print(Hash_Hyper_param['log2_hashmap_size'])
    print(Hash_Hyper_param['n_features_per_level'])
    print(Hash_Hyper_param['n_levels'])

    # Set Model Parameters
    # Total: 12206480 Parameters
    params_binary = np.frombuffer(config["snapshot"]["params_binary"], dtype = np.float16, offset = 0)
    # Transform to torch tensor
    params_binary = torch.tensor(params_binary, dtype = torch.float32)
    # Generate Parameters Dictionary
    params = {}
    # Params for Hash Encoding Network
    ## Network Params Size: 32 * 64 + 64 * 16 = 3072
    hashenc_params_network = params_binary[:(32 * 64 + 64 * 16)]
    params_binary = params_binary[(32 * 64 + 64 * 16):]
    # Params for RGB Network
    ## Network Params Size: 32 * 64 + 64 * 64 + 64 * 16 = 7168
    rgb_params_network = params_binary[:(32 * 64 + 64 * 64 + 64 * 16)]
    params_binary = params_binary[(32 * 64 + 64 * 64 + 64 * 16):]
    # Params for Hash Encoding Grid
    ## Grid size: 12196240
    hashenc_params_grid = params_binary

    # Generate Final Parameters
    params["HashEncoding"] = torch.concat([hashenc_params_network, hashenc_params_grid,  ])
    params["RGB"] = rgb_params_network
    res["params"] = params
    # Occupancy Grid Part
    
    snapshot = config["snapshot"]
    density_grid_size = snapshot["density_grid_size"]
    aabb_min = snapshot["aabb"]["min"]
    aabb_max = snapshot["aabb"]["max"]
    #exit()
    grid_raw = torch.tensor(
        np.frombuffer(snapshot["density_grid_binary"],dtype=np.float16).astype(np.float32),
        dtype = torch.float32, device = "cuda"
        )

    num_of_layers = grid_raw.shape[0] // (density_grid_size ** 3)
    #grid_raw = torch.ones_like(grid_raw, dtype = torch.float32, device = 'cuda')
    total_grid = []
    total_grid_binary = []
    aabbs = []
    for i in range(num_of_layers):
        aabb_left = 0.5 - 0.5 * (2 ** i)
        aabb_right = 0.5 + 0.5 * (2 ** i)
        aabbs.append([aabb_left, aabb_left, aabb_left, aabb_right, aabb_right, aabb_right])
        grid = torch.zeros([density_grid_size ** 3], dtype = torch.float32, device = 'cuda')

        x, y, z = inv_morton_naive(torch.arange(0, 128**3, 1))
        grid[x * 128 * 128 + y * 128 + z] = grid_raw[i * (density_grid_size ** 3): (i + 1) * (density_grid_size ** 3)]
        grid_3d = torch.reshape(grid > 0.01, [1, density_grid_size, density_grid_size, density_grid_size]).type(torch.bool)
        total_grid.append(grid)
        total_grid_binary.append(grid_3d)
    #exit()
    total_grid = torch.cat(total_grid, dim = 0)
    total_grid_binary = torch.cat(total_grid_binary, dim = 0)
    estimator = nerfacc.OccGridEstimator(
        aabb_min + aabb_max,
        resolution = 128, levels = num_of_layers
    )
    params_grid = {
        "resolution": torch.tensor([density_grid_size, density_grid_size, density_grid_size], dtype = torch.int32),
        #"aabbs": torch.tensor([[-0.5, -0.5, -0.5, 1.5, 1.5, 1.5]]),
        "aabbs": torch.tensor(aabbs),
        "occs":total_grid,
        "binaries": total_grid_binary#grid_3d
    }
    estimator.load_state_dict(params_grid)
    
    res["OccupancyGrid"] = estimator
    
    print("Msgpack Loaded!")
    return res