import torch
import numpy as np
from utils import cumprod_exclusive_ngp

def render_ray(alpha_raws, color_raws, step_length = 1.7320508075688772 / 1024):
	"""
	Do Volume Rendering for a single ray's data
	"""    
	alphas = (1. - torch.exp(-torch.exp(alpha_raws) * step_length))
	
	# Cumprod_exclusive need a 1D array input!
	weights = alphas * cumprod_exclusive_ngp((1. - alphas)) 
	colors = weights * torch.sigmoid(color_raws)
	return torch.sum(colors, dim = -2).cpu().detach().numpy()
