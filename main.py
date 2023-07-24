import numpy as np
import torch
import tinycudann as tcnn
import matplotlib.pyplot as plt
import json, time, os, msgpack, Morton3D, cv2
from tqdm import trange, tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=str, default="lego", help="scene name")
parser.add_argument("--steps", type=int, default = 128, help="steps of each ray")
parser.add_argument("--w", "--width", type = int, default = 100, help = "width of the image")
parser.add_argument("--h", "--height", type = int, default = 100, help = "height of the image")


### Constants
NEAR_DISTANCE = 0.05
FAR_DISTANCE = 6.
BATCH_SIZE = 1

CONFIG_PATH = "./configs/base.json"

from camera import Camera
from dataloader import load_msgpack
from renderer import render_rays
from utils import get_init_t_value, get_next_voxel

scales = {
    "chair": 0.33,
    "drums": 0.33,
    "ficus": 0.33,
    "hotdog": 0.33,
    "lego": 0.33,
    "materials": 0.33,
    "mic": 0.33,
    "ship": 0.33
}

if __name__ == "__main__":
    # Deal with Arguments
    args = parser.parse_args()
    scene = args.scene
    img_w, img_h = args.w, args.h
    #DATA_PATH = f"./snapshots/TestData/{scene}_16.msgpack"
    DATA_PATH = f"./snapshots/NsightComputeData/{scene}.msgpack"
    NERF_STEPS = args.steps
    SQRT3 = 1.7320508075688772
    STEP_LENGTH = SQRT3 / NERF_STEPS
    resolution = (img_w, img_h)
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
    # Load Parameters
    snapshots = load_msgpack(DATA_PATH)
    hashenc.load_state_dict({"params":snapshots["params"]["HashEncoding"]})
    rgb_net.load_state_dict({"params":snapshots["params"]["RGB"]})
    grid = snapshots["OccupancyGrid"]
    aabb = torch.tensor([[-0.5, -0.5, -0.5], [1.5, 1.5, 1.5]], device = DEVICE)
    for i in trange(camera.resolution[0]):
        for j in range(0, camera.resolution[1], BATCH_SIZE):
            ray_o = torch.from_numpy(camera.rays_o[i, j: j + BATCH_SIZE]).to(DEVICE)
            ray_d = torch.from_numpy(camera.rays_d[i, j: j + BATCH_SIZE]).to(DEVICE)

            t = 0.05
            
            # Skip the empty ray
            if isinstance(t, str):
                continue
            color = torch.zeros(3, dtype = torch.float32, device = DEVICE)
            opacity = torch.zeros(1, dtype = torch.float32, device = DEVICE)
            while (t <= 6.):
                position = ray_o + t * ray_d
                if(grid.intersect(position[0] * scales[scene] * 2 + 0.5)):
                    # Case of we need run
                    pos_hash = position * scales[scene] + 0.5
                    hash_feature = hashenc(pos_hash)
                    sh_feature = shenc((ray_d + 1)/2)
                    feature = torch.concat([hash_feature, sh_feature], dim = -1)

                    alpha_raw = hash_feature[:, 0]
                    rgb_raw = rgb_net(feature)
                    T = 1 - opacity
                    alpha = 1 - torch.exp(-torch.exp(alpha_raw) * STEP_LENGTH)
                    weight = T * alpha
                    rgb = torch.sigmoid(rgb_raw) * weight
                    opacity += weight
                    color += rgb[0]
                    
                t += STEP_LENGTH
            camera.image[i, j: j + BATCH_SIZE] = color.cpu().detach().numpy()

    # Only show image and don't show the axis
    dpi = 100
    fig = plt.figure(figsize = (img_w / dpi, img_h / dpi), dpi = dpi)
    axes = fig.add_axes([0, 0, 1, 1])
    axes.set_axis_off()
    axes.imshow(camera.image)
    output_dir = os.path.join("outputs")
    os.makedirs(output_dir, exist_ok = True)
    plt.savefig(os.path.join(output_dir, f"Test_{scene}.png"))
    
