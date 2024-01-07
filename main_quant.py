import numpy as np
import torch
### Quantize Results
import Quantize.QHash as QHash
import Quantize.QSH as QSH
import Quantize.QNetWorks as QNetwork
from Quantize.QuantUtils import Fixed_Point_Quantize
### Normal Results
import Modules.Hash as Hash
import Modules.SphericalHarmonics as SH
import Modules.Networks as Network
import matplotlib.pyplot as plt
import json
import os
import argparse
import nerfacc
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
import cv2 as cv

parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=str, default="lego", help="scene name")
parser.add_argument("--steps", type=int, default = 1024, help="steps of each ray")
parser.add_argument("--data", type = str, default = "TotalData", help = "Name of data dir")
parser.add_argument("--w", "--width", type = int, default = 800, help = "width of the image")
parser.add_argument("--h", "--height", type = int, default = 800, help = "height of the image")
parser.add_argument("--name", help = "Name Of the Output Image")
parser.add_argument("--test_id", type = int, default = 0, help = "ID of out image in test datasets")
parser.add_argument("--config", default = "base", help = "Config Json Name")
parser.add_argument("--quant_type", "--qt", default = "Fixed-Point")
parser.add_argument("--white_bkgd", action = "store_true", help = "Use white background instead of black background")

from camera import Camera
from dataloader import load_msgpack_new

def PSNR(img1, path2):
    img2_raw = cv.imread(path2, cv.IMREAD_UNCHANGED) / 255.
    img2_raw = img2_raw[..., :3] * img2_raw[..., 3:]
    img2 = np.array(img2_raw, dtype=np.float32)[..., [2, 1, 0]]
    print(img1.shape, img2.shape)
    return compute_psnr(img1, img2)

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
    CONFIG_PATH = f"./configs/{args.config}.json"
    if(args.quant_type == "Fixed-Point"):
        Quant_CONFIG_PATH = f"./configs/quantize_fixed.json"
    elif (args.quant_type == "Floating-Point" or args.quant_type == "FloatingPoint"):
        Quant_CONFIG_PATH = f"./configs/quantize_float.json"

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
    NAME = args.name if args.name is not None else f"Q_{scene}_{ID}"
    print(f"The output image is {NAME}.png")

    # Camera Parameters
    with open(f"./data/nerf_synthetic/{scene}/transforms_test.json", "r") as f:
        meta = json.load(f)
    m_Camera_Angle_X = float(meta["camera_angle_x"])
    m_C2W = np.array(meta["frames"][ID]["transform_matrix"]).reshape(4, 4)
    
    # Load Configs and Generate Components
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
        
    if(args.quant_type == "Fixed-Point"):
        Quant_Part_1 = "Integral"
        Quant_Part_2 = "Fraction"
        QType = "Fixed"
    elif (args.quant_type == "Floating-Point" or args.quant_type == "FloatingPoint"):
        Quant_Part_1 = "Exponent"
        Quant_Part_2 = "Mantissa"
        QType = "Float"



    # Quantize Parameters
    with open(Quant_CONFIG_PATH, "r") as f:
        Quant_config = json.load(f)
    ## Ray Marching
    if Quant_config["RayMarching"] == "Float":
        ray_marching_q = None
        print("Ray Marching is Float")
    else:
        ray_marching_q = Quant_config["RayMarching"]
        print("Ray Marching is Quantized:")
        print(f"\tPoint  \t{Quant_Part_1}:{ray_marching_q['Point'][0]}, {Quant_Part_2}:{ray_marching_q['Point'][1]}")
        print(f"\tDirection \t{Quant_Part_1}:{ray_marching_q['Direction'][0]}, {Quant_Part_2}:{ray_marching_q['Direction'][1]}")
    ## Hash Encoding
    if Quant_config["Hash"] == "Float":
        hashgrid = Hash.HashEncoding(
            n_input_dims = 3,
            encoding_config = config["HashEnc"]
        ).to(DEVICE)
        print("Hash Encoding is Float")
    else:
        hashgrid = QHash.QHashEncoding(
            n_input_dims = 3,
            encoding_config = config["HashEnc"],
            FeatureBits = Quant_config["Hash"]["Feature"],
            ResultBits = Quant_config["Hash"]["Result"],
            qtype = QType
        ).to(DEVICE)
        print("Hash Encoding is Quantized:")
        print(f"\tFeature  \t{Quant_Part_1}:{Quant_config['Hash']['Feature'][0]}, {Quant_Part_2}:{Quant_config['Hash']['Feature'][1]}")
        print(f"\tResult   \t{Quant_Part_1}:{Quant_config['Hash']['Result'][0]}, {Quant_Part_2}:{Quant_config['Hash']['Result'][1]}")
    ## Sphereical Harmonics
    if Quant_config["SH"] == "Float":
        shenc = SH.SHEncoding(
            n_input_dims = 3,
            encoding_config = config["SHEnc"]
        ).to(DEVICE)
        print("Spherical Harmonics is Float")
    else:
        shenc = QSH.QSHEncoding(
            n_input_dims = 3,
            encoding_config = config["SHEnc"],
            FeatureBits = Quant_config["SH"]["Feature"],
            ResultBits = Quant_config["SH"]["Result"]
        ).to(DEVICE)
        print("Spherical Harmonics is Quantized:")
        print(f"\tFeature  \t{Quant_Part_1}:{Quant_config['SH']['Feature'][0]}, {Quant_Part_2}:{Quant_config['SH']['Feature'][1]}")
        print(f"\tResult \t{Quant_Part_1}:{Quant_config['SH']['Result'][0]}, {Quant_Part_2}:{Quant_config['SH']['Result'][1]}")
    ## Two MLP
    if Quant_config["MLP"] == "Float":
        sig_net = Network.MLP(
            n_input_dims = 32,
            n_output_dims = 16,
            network_config = config["HashNet"]
        ).to(DEVICE)
        rgb_net = Network.MLP(
            n_input_dims = 32,
            n_output_dims = 3,
            network_config = config["RGBNet"]
        ).to(DEVICE)
        print("MLP is Float")
    else:
        sig_net = QNetwork.QMLP(
            n_input_dims = 32,
            n_output_dims = 16,
            network_config = config["HashNet"],
            WeightBits=Quant_config["MLP"]["Weight"],
            FeatureBits=Quant_config["MLP"]["Feature"],
            qtype = QType
        ).to(DEVICE)

        rgb_net = QNetwork.QMLP(
            n_input_dims = 32,
            n_output_dims = 3,
            network_config = config["RGBNet"],
            WeightBits=Quant_config["MLP"]["Weight"],
            FeatureBits=Quant_config["MLP"]["Feature"],
            qtype = QType
        ).to(DEVICE)
        print("MLP is Quantized:")
        print(f"\tWeight  \t{Quant_Part_1}:{Quant_config['MLP']['Weight'][0]}, {Quant_Part_2}:{Quant_config['MLP']['Weight'][1]}")
        print(f"\tFeature  \t{Quant_Part_1}:{Quant_config['MLP']['Feature'][0]}, {Quant_Part_2}:{Quant_config['MLP']['Feature'][1]}")
    
    camera = Camera(resolution, m_Camera_Angle_X, m_C2W)
    # exit()
    print("==========HyperParameters==========")
    print(f"Steps: {args.steps}")
    log2_hashgrid_size = config["HashEnc"]["log2_hashmap_size"]
    print(f"Hash Grid Size: 2 ^ {log2_hashgrid_size}")
    print("AABB: (-0.5, -0.5, -0.5) ~ (1.5, 1.5, 1.5)")

    # Load Parameters
    snapshots = load_msgpack_new(DATA_PATH)
    #hashenc.load_state_dict({"params":snapshots["params"]["HashEncoding"]})
    hashgrid.load_states(snapshots["params"]["HashEncoding"][3072:])
    sig_net.load_states(snapshots["params"]["HashEncoding"][:3072])
    
    rgb_net.load_states(snapshots["params"]["RGB"])
    estimator: nerfacc.OccGridEstimator =  snapshots["OccupancyGrid"].to(DEVICE)
    
    print("==========Begin Running==========")
    
    rays_o_total = torch.tensor(camera.rays_o, dtype = torch.float32, device = DEVICE)
    rays_d_total = torch.tensor(camera.rays_d, dtype = torch.float32, device = DEVICE)
    total_color = np.zeros([resolution[0] * resolution[1], 3], dtype = np.float32)
    total_opacity = np.zeros([resolution[0] * resolution[1], 1], dtype = np.float32)
    
    BATCH_SIZE = min(100 * 100, resolution[0] * resolution[1])
    
    for index in range(0, resolution[0] * resolution[1], BATCH_SIZE):
        BATCH = min(BATCH_SIZE, resolution[0] * resolution[1] - index)
        rays_o = rays_o_total[index: index + BATCH]
        rays_d = rays_d_total[index: index + BATCH]
        if ray_marching_q is not None:
            rays_o = Fixed_Point_Quantize(rays_o, ray_marching_q["Point"][0], ray_marching_q["Point"][1])
            rays_d = Fixed_Point_Quantize(rays_d, ray_marching_q["Direction"][0], ray_marching_q["Direction"][1])

        def alpha_fn(t_starts, t_ends, ray_indices):
            origins = rays_o[ray_indices]
            directions = rays_d[ray_indices]
            ts = torch.reshape((t_starts + t_ends) / 2.0, (-1, 1))
            Points = origins + directions * ts
            if ray_marching_q is not None:
                Points = Fixed_Point_Quantize(Points, ray_marching_q["Point"][0], ray_marching_q["Point"][1])
            hash_features_raw = hashgrid(Points)
            hash_features = sig_net(hash_features_raw)
            alphas_raw = hash_features[..., 0]
            alphas = (1. - torch.exp(-torch.exp(alphas_raw.type(torch.float32)) * STEP_LENGTH))
            return alphas
        
        def rgb_alpha_fn(t_starts, t_ends, ray_indices):
            origins = rays_o[ray_indices]
            directions = rays_d[ray_indices]
            ts = torch.reshape((t_starts + t_ends) / 2.0, (-1, 1))
            Points = origins + directions * ts
            if ray_marching_q is not None:
                Points = Fixed_Point_Quantize(Points, ray_marching_q["Point"][0], ray_marching_q["Point"][1])
            hash_features_raw = hashgrid(Points)
            hash_features = sig_net(hash_features_raw)
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
        total_opacity[index: index + BATCH] = (opacity).cpu().detach().numpy()

    # * Only show image and don't show the axis
    dpi = 100
    fig = plt.figure(figsize = (img_w / dpi, img_h / dpi), dpi = dpi)
    axes = fig.add_axes([0, 0, 1, 1])
    axes.set_axis_off()
    if args.white_bkgd:
        img = total_color + (1. - total_opacity) * np.ones_like(total_color)
    else:
        img = total_color
    img = img.reshape(camera.w, camera.h, 3)
    axes.imshow(img)
    

    
    output_dir = os.path.join("outputs")
    os.makedirs(output_dir, exist_ok = True)
    # plt.savefig(os.path.join(output_dir, NAME + f"_{Quant_config['Hash']['Feature'][0]}_{Quant_config['Hash']['Feature'][1]}"))
    # print(f"Image was saved to ./{output_dir}/{NAME}_{Quant_config['Hash']['Feature'][0]}_{Quant_config['Hash']['Feature'][1]}.png")
    plt.savefig(os.path.join(output_dir, NAME))
    print(f"Image was saved to ./{output_dir}/{NAME}.png")
    print("==========Done!==========")
    
    print("==========Evaluate==========")
    ref_img_path = f"./data/nerf_synthetic/{scene}/test/r_{ID}.png"
    psnr = PSNR(img, ref_img_path)
    print(f"PSNR = {psnr}")
