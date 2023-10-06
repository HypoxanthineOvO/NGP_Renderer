import shutil
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
import matplotlib.pyplot as plt
import cv2 as cv
import json

scenes = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
scenes = ["lego"]

def PSNR_ip(img1, path2):
    img2_raw = cv.imread(path2, cv.IMREAD_UNCHANGED) / 255.
    img2_raw = img2_raw[..., :3] * img2_raw[..., 3:]
    img2 = np.array(img2_raw, dtype=np.float32)
    return compute_psnr(img1, img2)

def Show_Diff(path1, path2, name = None):
    img1 = np.array(cv.imread(path1) / 255., dtype=np.float32)
    img2_raw = cv.imread(path2, cv.IMREAD_UNCHANGED) / 255.
    img2_raw = img2_raw[..., :3] * img2_raw[..., 3:]
    img2 = np.array(img2_raw, dtype=np.float32)
    
    diff = np.abs(img1 - img2)
    out_name = "Diff"
    if name is not None:
        out_name = name
    plt.imsave(f"{out_name}.png",diff)
    
def resize(path):
    image = np.array(cv.imread(path) / 255., dtype=np.float32)
    return cv.resize(image, (800, 800),interpolation = cv.INTER_AREA)

if __name__ == "__main__":
    res = {"Mean": 0.0}
    outdir = "Test_Results"
    os.makedirs(outdir, exist_ok = True)
    mean_psnrs = []
    for i, name in enumerate(scenes):
        psnrs = []
        os.makedirs(os.path.join(outdir, name), exist_ok = True)
        for id in range(0, 200, 8):
            out_file_name = "Test_" + name + f"_{id}.png"
            if not (os.path.exists(os.path.join(outdir, name, out_file_name))):
                os.system(f"python main_new.py --scene {name} --w 1600 --h 1600 --test_id {id} --data BigData --config big")
                psnr = round(PSNR_ip(resize(os.path.join("outputs", out_file_name)), f"./data/nerf_synthetic/{name}/test/r_{id}.png"), 4)
                shutil.move(os.path.join("outputs", out_file_name), os.path.join(outdir, name, out_file_name))
            else:
                psnr = round(PSNR_ip(resize(os.path.join(outdir, name, out_file_name)), f"./data/nerf_synthetic/{name}/test/r_{id}.png"), 4)
            print(f'PSNR Of {name}: {psnr}')
            psnrs.append(psnr)
            
            res[name] = [round(np.mean(psnrs), 4), psnrs]
            with open(os.path.join(outdir, "results.json"), "w") as f:
                json.dump(res, f, indent = 4)
        mean_psnrs += psnrs
        res["Mean"] = round(np.mean(mean_psnrs), 4)
    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(res, f, indent = 4)