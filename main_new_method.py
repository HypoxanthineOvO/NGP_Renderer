import numpy as np
import torch
import tinycudann as tcnn
import matplotlib.pyplot as plt
import json, time, os, msgpack, Morton3D, cv2
from tqdm import trange, tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=str, default="lego", help="scene name")
parser.add_argument("--steps", type=int, default = 256, help="steps of each ray")
parser.add_argument("--w", "--width", type = int, default = 800, help = "width of the image")
parser.add_argument("--h", "--height", type = int, default = 800, help = "height of the image")
parser.add_argument("--name", help = "Name Of the Output Image")
parser.add_argument("--thredhold", help = "Thredhold of new method", type = float, default = 0.2)


from camera import Camera
from dataloader import load_msgpack
from renderer import render_ray
from utils import generate_curve, gen_normal

if __name__ == "__main__":
    # Deal with Arguments
    ## arguments
    args = parser.parse_args()
    ### Scene Name
    scene = args.scene
    DATA_PATH = f"./snapshots/ISCAData/{scene}.msgpack"
    ### Resolution
    img_w, img_h = args.w, args.h    
    resolution = (img_w, img_h)    
    ### Steps
    NERF_STEPS = args.steps
    SQRT3 = 1.7320508075688772
    STEP_LENGTH = SQRT3 / NERF_STEPS
    ## THREDHOLD
    THREDHOLD = args.thredhold
    ## Constants
    NEAR_DISTANCE = 0.6
    FAR_DISTANCE = 2.0

    CONFIG_PATH = "./configs/base.json"

    ## Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Visualize Informations
    print("==========Hypoxanthine's Instant NGP==========")
    print(f"Scene: NeRF-Synthetic {scene}")
    print(f"Image: {img_w} x {img_h}")
    NAME = args.name if args.name is not None else f"NEWMethod_{scene}"
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
    pixels = camera.resolution[0] * camera.resolution[1]
    ts = torch.reshape(torch.linspace(NEAR_DISTANCE, FAR_DISTANCE, NERF_STEPS, device = DEVICE), (-1, 1))
    
    NORMAL = torch.tensor(gen_normal(NERF_STEPS), device = DEVICE)
    for pixel_index in trange(0, pixels):
        ray_o = torch.from_numpy(camera.rays_o[pixel_index: pixel_index + 1]).to(DEVICE)
        ray_d = torch.from_numpy(camera.rays_d[pixel_index: pixel_index + 1]).to(DEVICE)

        """
        Naive Ray Marching
        """
        
        pts = ray_o + ts * ray_d
        occupancy = grid.intersect(pts * 2 + 0.5)
        if(torch.sum(occupancy) == 0):
            continue
        ### New Method
        
        density_curve = generate_curve(occupancy, NORMAL)
        oc = torch.where(density_curve > THREDHOLD)[0]
        ts_final = torch.cat(
            [torch.arange(
                (ts[oc[i]] - 0.5 * STEP_LENGTH).item(), (ts[oc[i]] + 0.5 * STEP_LENGTH).item(), SQRT3/1024, device = DEVICE
                )
            for i in range(oc.shape[0])], dim = -1).reshape((-1, 1))
        
        pts_final = ray_o + ts_final * ray_d
        color = torch.zeros([1, 3], dtype = torch.float32, device = DEVICE)
        opacity = torch.zeros([1, 1], dtype = torch.float32, device = DEVICE)
        #pts_truth = pts[torch.where(occupancy)]

        hash_features = hashenc(pts_final + 0.5)
        sh_features = torch.tile(shenc((ray_d+1) / 2), (hash_features.shape[0], 1))
        features = torch.concat([hash_features, sh_features], dim = -1)

        alphas_raw = hash_features[..., 0:1]
        rgbs_raw = rgb_net(features)
        camera.image[pixel_index] = render_ray(alphas_raw, rgbs_raw, SQRT3/1024)#STEP_LENGTH)

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
