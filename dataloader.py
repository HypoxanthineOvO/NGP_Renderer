import numpy as np
import torch
import json, time, os, msgpack, Morton3D
from grid import DensityGrid
from tqdm import tqdm
    
def load_msgpack(path: str):
    print("Loding Msgpack...")
    # Return Value
    res = {}
    # Get Morton3D Object
    mt = Morton3D.zorder()
    
    # Set File Path
    assert (os.path.isfile(path))
    dir_name, full_file_name = os.path.split(path)
    file_name, ext_name = os.path.splitext(full_file_name)
    # Load the msgpack
    with open(path, 'rb') as f:
        unpacker = msgpack.Unpacker(f, raw = False)
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
    grid_raw = np.clip(
        np.frombuffer(config["snapshot"]["density_grid_binary"],dtype=np.float16).astype(np.float32),
        0, 1)
    grid = np.zeros([128 * 128 * 128], dtype = np.int8)
    for idx in range(grid_raw.shape[0]):
        x,y,z = mt.deMorton(idx, False)
        grid[x * 128 * 128 + y * 128 + z] = (grid_raw[idx] > 0.01)
    
    # grid = np.zeros([128, 128, 128], dtype = np.int8)
    # for idx in range(grid_raw.shape[0]):
    #     x,y,z = mt.deMorton(idx, 0)
    #     grid[x,y,z] = (grid_raw[idx] > 0.01)
    # For AABB: we only consider k = 1
    ## The Domain is [-0.5, -0.5, -0.5] to [1.5, 1.5, 1.5]
    oc_grid = DensityGrid(grid,[[-0.5, -0.5, -0.5], [1.5, 1.5, 1.5]])
    
    res["OccupancyGrid"] = oc_grid
    
    print("Msgpack Loaded!")
    return res