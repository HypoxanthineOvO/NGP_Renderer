import torch
import numpy as np


### Renderer
def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    r"""Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

    Args:
    tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
      is to be computed.

    Returns:
    cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
      tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
    """
    # TESTED
    # Only works for the last dimension (dim=-1)
    dim = -1
    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, dim)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, dim)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.0

    return cumprod

def render_rays(z_values, sigmas, rgbs, batch_size = 1):
    one_e_10 = torch.tensor([1e10])
    if torch.cuda.is_available():
        one_e_10 = one_e_10.to("cuda")
    z_values = z_values.expand([batch_size, -1]).to("cuda")
    
    dists = torch.cat([z_values[...,1:] - z_values[...,:-1],one_e_10.expand(z_values[...,:1].shape)],dim = -1)
    #print("Sigmas: ",sigmas.shape)
    #print("Dists: ",dists.shape)
    alpha = 1.0 - torch.exp(-sigmas * dists) * 1.73205080757 / 1024
    #print(alpha.shape)
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)
    
    #print(weights.shape, rgbs.shape)
    rgb_map = (weights[...,np.newaxis] * rgbs).sum(dim = -2)
    depth_map = (weights * z_values).sum(dim = -1)
    acc_map = weights.sum(-1)
    
    return rgb_map,depth_map,acc_map
