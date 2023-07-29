import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def compute_mse(img1, img2):
    return np.mean((img1 - img2)**2)

def mse_to_psnr(mse):
    return 20 * np.log10(1 / np.sqrt(mse))

def resize_image(img, res):
    if res == 800:
        return img
    return cv.resize(img, (res, res))

scenes = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
res = 800

def PSNR(name:str, res = 400):
    path1 = f"./outputs/Test_{name}.png"
    path2 = f"./data/nerf_synthetic/{name}/test/r_0.png"
    img1 = np.array(cv.imread(path1) / 255., dtype=np.float32)
    img2 = np.array(cv.imread(path2) / 255., dtype=np.float32)
    delta = np.abs(img1 - img2)
    img2 = resize_image(img2, res)
    return mse_to_psnr(compute_mse(img1, img2))

if __name__ == "__main__":
    for scene in scenes:
        print(PSNR(scene, res))
