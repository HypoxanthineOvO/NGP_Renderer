import numpy as np
import torch
import tinycudann as tcnn
import matplotlib.pyplot as plt
import json
import os
import argparse
import nerfacc


parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=str, default="lego", help="scene name")
parser.add_argument("--steps", type=int, default = 1024, help="steps of each ray")
parser.add_argument("--w", "--width", type = int, default = 800, help = "width of the image")
parser.add_argument("--h", "--height", type = int, default = 800, help = "height of the image")
parser.add_argument("--name", help = "Name Of the Output Image")
parser.add_argument("--test_id", type = int, default = 0, help = "ID of out image in test datasets")
parser.add_argument("--config", default = "base", help = "Config Json Name")

from camera import Camera
from dataloader import load_msgpack_new

if __name__ == "__main__":
    # Deal with Arguments
    ## arguments
    args = parser.parse_args()
    ### Scene Name
    scene = args.scene
    DATA_PATH = f"./snapshots/TotalData/{scene}.msgpack"
    #DATA_PATH = f"./{scene}.msgpack"
    ### Resolution
    img_w, img_h = args.w, args.h    
    resolution = (img_w, img_h)    
    ### Steps
    NERF_STEPS = args.steps
    SQRT3 = 1.7320508075688772
    STEP_LENGTH = SQRT3 / NERF_STEPS
    

    CONFIG_PATH = f"./configs/{args.config}.json"

    ## Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## Test Data
    ID = args.test_id
    ## Constants
    NEAR_DISTANCE = 0.6
    FAR_DISTANCE = 2.0
    
    # Visualize Informations
    print("==========Hypoxanthine's Instant NGP==========")
    print(f"Scene: NeRF-Synthetic {scene}")
    print(f"Image: {img_w} x {img_h}")
    NAME = args.name if args.name is not None else f"Test_{scene}_{ID}"
    print(f"The output image is {NAME}.png")

    # Camera Parameters
    with open(f"./data/nerf_synthetic/{scene}/transforms_test.json", "r") as f:
        meta = json.load(f)
    m_Camera_Angle_X = float(meta["camera_angle_x"])
    m_C2W = np.array(meta["frames"][ID]["transform_matrix"]).reshape(4, 4)
    
    # Load Configs and Generate Components
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    hashenc = tcnn.NetworkWithInputEncoding(
        n_input_dims = 3,
        n_output_dims = 16,
        encoding_config = config["HashEnc"],
        network_config = config["HashNet"]
    ).to(DEVICE)
    shenc = tcnn.Encoding(
        n_input_dims = 3,
        encoding_config = config["SHEnc"],
        dtype = torch.float32
    ).to(DEVICE)
    rgb_net = tcnn.Network(
        n_input_dims = 32,
        n_output_dims = 3,
        network_config = config["RGBNet"]
    ).to(DEVICE)
    
    
    camera = Camera(resolution, m_Camera_Angle_X, m_C2W)
    # exit()
    print("==========HyperParameters==========")
    print(f"Steps: {args.steps}")
    log2_hashgrid_size = config["HashEnc"]["log2_hashmap_size"]
    print(f"Hash Grid Size: 2 ^ {log2_hashgrid_size}")
    print("AABB: (-0.5, -0.5, -0.5) ~ (1.5, 1.5, 1.5)")

    # Load Parameters
    snapshots = load_msgpack_new(DATA_PATH)
    hashenc.load_state_dict({"params":snapshots["params"]["HashEncoding"]})
    
    rgb_net.load_state_dict({"params":snapshots["params"]["RGB"]})
    estimator: nerfacc.OccGridEstimator =  snapshots["OccupancyGrid"].to(DEVICE)
    
    print("==========Begin Running==========")
    
    rays_o_total = torch.tensor(camera.rays_o, dtype = torch.float32, device = DEVICE)
    rays_d_total = torch.tensor(camera.rays_d, dtype = torch.float32, device = DEVICE)
    total_color = np.zeros([resolution[0] * resolution[1], 3], dtype = np.float32)
    
    BATCH_SIZE = min(400 * 400, resolution[0] * resolution[1])
    
    for index in range(0, resolution[0] * resolution[1], BATCH_SIZE):
        BATCH = min(BATCH_SIZE, resolution[0] * resolution[1] - index)
        rays_o = rays_o_total[index: index + BATCH]
        rays_d = rays_d_total[index: index + BATCH]

        def alpha_fn(t_starts, t_ends, ray_indices):
            origins = rays_o[ray_indices]
            directions = rays_d[ray_indices]
            ts = torch.reshape((t_starts + t_ends) / 2.0, (-1, 1))
            positions = origins + directions * ts
            hash_features = hashenc(positions)
            alphas_raw = hash_features[..., 0]
            alphas = (1. - torch.exp(-torch.exp(alphas_raw.type(torch.float32)) * STEP_LENGTH))
            return alphas
        
        def rgb_alpha_fn(t_starts, t_ends, ray_indices):
            origins = rays_o[ray_indices]
            directions = rays_d[ray_indices]
            ts = torch.reshape((t_starts + t_ends) / 2.0, (-1, 1))
            positions = origins + directions * ts
            
            hash_features = hashenc(positions)
            sh_features = shenc((directions + 1) / 2)
            
            
            features = torch.concat([hash_features, sh_features], dim = -1)
            alphas_raw = hash_features[..., 0]
            alphas = (1. - torch.exp(-torch.exp(alphas_raw.type(torch.float32)) * STEP_LENGTH))
            rgbs_raw = rgb_net(features)
            rgbs = torch.sigmoid(rgbs_raw)
            return rgbs, alphas    

        
        ray_indices, t_starts, t_ends = estimator.sampling(
            rays_o, rays_d, near_plane = NEAR_DISTANCE, far_plane = FAR_DISTANCE, 
            render_step_size = STEP_LENGTH
        )
        if(ray_indices.shape[0] <= 0):
            continue

        color, opacity, depth, extras = nerfacc.rendering(
            t_starts, t_ends, ray_indices, 
            n_rays = BATCH, rgb_alpha_fn = rgb_alpha_fn
        )
        total_color[index: index + BATCH] = (color).cpu().detach().numpy()
    

    # * Only show image and don't show the axis
    dpi = 100
    fig = plt.figure(figsize = (img_w / dpi, img_h / dpi), dpi = dpi)
    axes = fig.add_axes([0, 0, 1, 1])
    axes.set_axis_off()
    axes.imshow(total_color.reshape(camera.w, camera.h, 3))
    
    output_dir = os.path.join("outputs")
    os.makedirs(output_dir, exist_ok = True)
    plt.savefig(os.path.join(output_dir, NAME))
    print(f"Done! Image was saved to ./{output_dir}/{NAME}.png")
