import shutil
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
import matplotlib.pyplot as plt
import cv2 as cv

scenes = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]


def PSNR(name, path, id):
    path1 = f"{path}/Test_{name}_{id}.png"
    path2 = f"./data/nerf_synthetic/{name}/test/r_{id}.png"
    print(path1)
    img1 = np.array(cv.imread(path1) / 255., dtype=np.float32)
    img2 = np.array(cv.imread(path2) / 255., dtype=np.float32)
    
    return compute_psnr(img1, img2)

if __name__ == "__main__":
    out_basedir = os.path.join(".", "test_out")
    STEP = 8
    
    for scene in scenes:
        psnrs = []
        out_dir = os.path.join(out_basedir, scene)
        os.makedirs(out_dir, exist_ok = True)
        for id in range(0, 200, STEP):
            command = f"python main.py --scene {scene} --test_id {id} --fast 5000"
            os.system(command)
            command_out_dir = os.path.join(".", "outputs")
            psnr = PSNR(scene, command_out_dir, id)
            psnrs.append(psnr)
            os.remove(os.path.join(out_dir, f"Test_{scene}_{id}.png"))
            shutil.move(os.path.join(command_out_dir, f"Test_{scene}_{id}.png"),os.path.join(out_dir))
        mean_psnr = round(np.mean(psnrs), 4)
        print(f"{scene.capitalize()}'s PSNR: {mean_psnr}")
        with open(os.path.join(out_dir, "PSNR.txt"), "w") as f:
            f.write(f"{scene.capitalize()}, mean psnr: {mean_psnr}\n")
            for i, psnr in enumerate(psnrs):
                f.write(f"ID: {i * STEP}, psnr = {round(psnr, 4)}\n")