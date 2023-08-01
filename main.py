import numpy as np
import torch
import tinycudann as tcnn
import matplotlib.pyplot as plt
import json, time, os, msgpack, Morton3D, cv2
from tqdm import trange, tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=str, default="lego", help="scene name")
parser.add_argument("--steps", type=int, default = 1024, help="steps of each ray")
parser.add_argument("--w", "--width", type = int, default = 800, help = "width of the image")
parser.add_argument("--h", "--height", type = int, default = 800, help = "height of the image")
parser.add_argument("--name", help = "Name Of the Output Image")


from camera import Camera
from dataloader import load_msgpack
from renderer import render_ray, render_ray_original
from utils import get_init_t_value, get_next_voxel



if __name__ == "__main__":
    # Deal with Arguments
    ## arguments
    args = parser.parse_args()
    ### Scene Name
    scene = args.scene
    DATA_PATH = f"./snapshots/NsightComputeData/{scene}.msgpack"    
    ### Resolution
    img_w, img_h = args.w, args.h    
    resolution = (img_w, img_h)    
    ### Steps
    NERF_STEPS = args.steps
    SQRT3 = 1.7320508075688772
    STEP_LENGTH = SQRT3 / NERF_STEPS
    
    ## Constants
    NEAR_DISTANCE = 0.6
    FAR_DISTANCE = 2.

    CONFIG_PATH = "./configs/base.json"

    ## Compute Batch Size
    BATCH_SIZE = 5000

    ## Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Visualize Informations
    print("==========Hypoxanthine's Instant NGP==========")
    print(f"Scene: NeRF-Synthetic {scene}")
    print(f"Image: {img_w} x {img_h}")
    NAME = args.name if args.name is not None else f"Test_{scene}"
    print(f"The output image is {NAME}.png")

    # Camera Parameters
    with open(f"./data/nerf_synthetic/{scene}/transforms_test.json", "r") as f:
        meta = json.load(f)
    m_Camera_Angle_X = float(meta["camera_angle_x"])
    m_C2W = np.array(meta["frames"][0]["transform_matrix"]).reshape(4, 4)
    
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
        encoding_config = config["SHEnc"]
    ).to(DEVICE)
    rgb_net = tcnn.Network(
        n_input_dims = 32,
        n_output_dims = 3,
        network_config = config["RGBNet"]
    ).to(DEVICE)
    camera = Camera(resolution, m_Camera_Angle_X, m_C2W)
    
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
    print(f"Batch Size: {BATCH_SIZE}")
    pixels = camera.resolution[0] * camera.resolution[1]
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
            #if(grid.intersect(position[0] * 2 + 0.5)):
            masks = grid.intersect(position * 2 + 0.5).reshape((-1, 1))
            # Case of we need run
            pos_hash = position + 0.5
            hash_feature = hashenc(pos_hash)
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

        camera.image[pixel_index: pixel_index + BATCH_SIZE] = color.cpu().detach().numpy()
        #tqdm.write(f"Device Memory Usage: {torch.cuda.memory_allocated() // (1024 * 1024)}MB")
    # Only show image and don't show the axis
    dpi = 100
    fig = plt.figure(figsize = (img_w / dpi, img_h / dpi), dpi = dpi)
    axes = fig.add_axes([0, 0, 1, 1])
    axes.set_axis_off()
    axes.imshow(camera.image.reshape(camera.w, camera.h, 3))
    output_dir = os.path.join("outputs")
    os.makedirs(output_dir, exist_ok = True)
    
    
    plt.savefig(os.path.join(output_dir, NAME))
    print(f"Done! Image was saved to ./{output_dir}/{NAME}.png")
