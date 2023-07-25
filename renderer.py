import torch
import numpy as np


### Renderer
def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    # Only works for the last dimension (dim=-1)
    dim = -1
    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, dim)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, dim)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.0

    return cumprod

def render_ray(alpha_raws, color_raws, step_length = 1.7320508075688772 / 1024):
    """
    Do Volume Rendering for a single ray's data
    """
    
    alphas = 1 - torch.exp(-torch.exp(alpha_raws) * step_length)
    weights = alphas * cumprod_exclusive(1.0 - alphas + 1e-10)
    rgbs = torch.sigmoid(color_raws) * weights
    return torch.sum(rgbs, dim = -2)

def render_ray_original(alpha_raws, color_raws, step_length = 1.7320508075688772 / 1024):
	"""
	Do Volume Rendering for a single ray's data
	"""
	DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	opacity = torch.zeros(1, dtype = torch.float32, device = DEVICE)
	color = torch.zeros(3, dtype = torch.float32, device = DEVICE)
	for i in range(alpha_raws.shape[0]):
		T = 1 - opacity
		#print(alpha_raws.shape, color_raws.shape)
		alpha = 1 - torch.exp(-torch.exp(alpha_raws[i]) * step_length)
		#print(alpha.shape)
		weight = T * alpha
		rgb = torch.sigmoid(color_raws[i]) * weight
		#print(rgb.shape)
		opacity += weight
		color += rgb
	return color