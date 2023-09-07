import numpy as np
import torch
import tinycudann as tcnn
import matplotlib.pyplot as plt
import json, time, os, msgpack, Morton3D, cv2
from tqdm import trange, tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=str, default="lego", help="scene name")
parser.add_argument("--data", type = str, default = "TotalData", help = "Name of data dir")
parser.add_argument("--steps", type=int, default = 1024, help="steps of each ray")
parser.add_argument("--w", "--width", type = int, default = 800, help = "width of the image")
parser.add_argument("--h", "--height", type = int, default = 800, help = "height of the image")
parser.add_argument("--name", help = "Name Of the Output Image")
parser.add_argument("--test_id", type = int, default = 0, help = "ID of out image in test datasets")
parser.add_argument("--fast", type = int, default = 0, help = "Use Fast Compute Method")

from camera import Camera
from dataloader import load_msgpack
from renderer import render_ray

torch.set_printoptions(precision=4,sci_mode=False)

if __name__ == "__main__":
    # Deal with Arguments
    ## arguments
    args = parser.parse_args()
    ### Scene Name
    scene = args.scene
    data_dir = args.data
    DATA_PATH = f"./snapshots/{data_dir}/{scene}.msgpack"
    ### Resolution
    img_w, img_h = args.w, args.h    
    resolution = (img_w, img_h)    
    ### Steps
    NERF_STEPS = args.steps
    SQRT3 = 1.7320508075688772
    STEP_LENGTH = SQRT3 / NERF_STEPS
    

    CONFIG_PATH = "./configs/base.json"

    ## Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## Test Data
    ID = args.test_id
    ## Constants
    NEAR_DISTANCE = 0.6#torch.tensor([0.6], dtype = torch.float32, device = DEVICE)
    FAR_DISTANCE = 2.0#torch.tensor([2.], dtype = torch.float32, device = DEVICE)
    
    
    
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
    snapshots = load_msgpack(DATA_PATH)
    hashenc.load_state_dict({"params":snapshots["params"]["HashEncoding"]})
    
    rgb_net.load_state_dict({"params":snapshots["params"]["RGB"]})
    grid = snapshots["OccupancyGrid"]

    
    print("==========Begin Running==========")
    pixels = camera.resolution[0] * camera.resolution[1]
    valid_points_counter = np.zeros(camera.resolution[0] * camera.resolution[1])
    
    stg01_time = 0
    stgother_time = 0
    if args.fast:
        BATCH_SIZE = args.fast
        for pixel_index in trange(0, pixels, BATCH_SIZE):
            BATCH = min(BATCH_SIZE, pixels - pixel_index)
            ray_o = torch.from_numpy(camera.rays_o[pixel_index: pixel_index + BATCH]).to(DEVICE)
            ray_d = torch.from_numpy(camera.rays_d[pixel_index: pixel_index + BATCH]).to(DEVICE)
            
            """
            Naive Ray Marching
            """ 
            t = NEAR_DISTANCE
            color = torch.zeros([BATCH, 3], dtype = torch.float32, device = DEVICE)
            opacity = torch.zeros([BATCH, 1], dtype = torch.float32, device = DEVICE)
            while (t <= FAR_DISTANCE):
                
                position = ray_o + t * ray_d
                
                masks = grid.intersect(position * 2 - 0.5).reshape((-1, 1))
                
                hash_feature = hashenc(position)
                
                sh_feature = shenc((ray_d + 1)/2)
                feature = torch.concat([hash_feature, sh_feature], dim = -1)
                alpha_raw = hash_feature[:, 0:1]
                rgb_raw = rgb_net(feature)
                T = 1 - opacity
                alpha = 1 - torch.exp(-torch.exp(alpha_raw) * STEP_LENGTH)
                
                weight = T * alpha * masks
                rgb = torch.sigmoid(rgb_raw) * weight
                
                opacity += weight
                color += rgb
                
                t += STEP_LENGTH
            
            camera.image[pixel_index: pixel_index + BATCH_SIZE] = (color).cpu().detach().numpy()
    else:
        for pixel_index in trange(0, pixels):
            ray_o = torch.from_numpy(camera.rays_o[pixel_index: pixel_index + 1]).to(DEVICE)
            ray_d = torch.from_numpy(camera.rays_d[pixel_index: pixel_index + 1]).to(DEVICE)

            ts = torch.reshape(torch.linspace(NEAR_DISTANCE, FAR_DISTANCE, NERF_STEPS, device = DEVICE), (-1, 1))
            pts = ray_o + ts * ray_d
            occupancy = grid.intersect(pts * 2 - 0.5)
            if(torch.sum(occupancy) == 0):
                continue
            color = torch.zeros([1, 3], dtype = torch.float32, device = DEVICE)
            opacity = torch.zeros([1, 1], dtype = torch.float32, device = DEVICE)
            pts_truth = pts[torch.where(occupancy)]

            hash_features = hashenc(pts_truth)
            sh_features = torch.tile(shenc((ray_d+1) / 2), (hash_features.shape[0], 1))
            features = torch.concat([hash_features, sh_features], dim = -1)

            alphas_raw = hash_features[..., 0:1]
            rgbs_raw = rgb_net(features)
            camera.image[pixel_index] = render_ray(alphas_raw, rgbs_raw, STEP_LENGTH)
            valid_points_counter[pixel_index] = torch.sum(occupancy).detach().cpu().numpy()
        
    # * Only show image and don't show the axis
    dpi = 100
    fig = plt.figure(figsize = (img_w / dpi, img_h / dpi), dpi = dpi)
    axes = fig.add_axes([0, 0, 1, 1])
    axes.set_axis_off()
    axes.imshow(camera.image.reshape(camera.w, camera.h, 3).astype(np.float32))
    output_dir = os.path.join("outputs")
    os.makedirs(output_dir, exist_ok = True)
    
    
    plt.savefig(os.path.join(output_dir, NAME))
    print(f"Done! Image was saved to ./{output_dir}/{NAME}.png")
