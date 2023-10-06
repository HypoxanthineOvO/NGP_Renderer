import torch
from torch.nn import Module

class SHEncoding(Module):
    def __init__(self, n_input_dims, encoding_config: dict):
        super().__init__()
        self.n_input_dims = n_input_dims
        self.degree = encoding_config["nested"][0]["degree"]
    
    def forward(self, inputs: torch.Tensor):
        shape = list(inputs.shape)
        assert(len(shape) == 2)
        assert(shape[-1] == 3)
        
        x = inputs[..., 0] * 2 - 1
        y = inputs[..., 1] * 2 - 1
        z = inputs[..., 2] * 2 - 1
        xy = x * y
        xz = x * z
        yz = y * z
        x2 = x * x
        y2 = y * y
        z2 = z * z
        
        outputs = torch.zeros([shape[0], 16], dtype = torch.float32, device = inputs.device)
        
        if(self.degree >= 1):
            outputs[..., 0] = (0.28209479177387814)
        if(self.degree >= 2):
            outputs[..., 1] = (-0.48860251190291987 * y)
            outputs[..., 2] = (0.48860251190291987 * z)
            outputs[..., 3] = (
                -0.48860251190291987 * x)
        if(self.degree >= 3):
            outputs[..., 4] = (1.0925484305920792 * xy)
            outputs[..., 5] = (-1.0925484305920792 * yz)
            outputs[..., 6] = (0.94617469575755997 * z2 - 0.31539156525251999)
            outputs[..., 7] = (-1.0925484305920792 * xz)
            outputs[..., 8] = (0.54627421529603959 * x2 - 0.54627421529603959 * y2)
        if(self.degree >= 4):
            outputs[..., 9] = (0.59004358992664352 * y * (-3.0 * x2 + y2))
            outputs[..., 10] = (2.8906114426405538 * xy * z)
            outputs[..., 11] = (0.45704579946446572 * y * (1.0 - 5.0 * z2))
            outputs[..., 12] = (0.3731763325901154 * z * (5.0 * z2 - 3.0))
            outputs[..., 13] = (0.45704579946446572 * x * (1.0 - 5.0 * z2))
            outputs[..., 14] = (1.4453057213202769 * z * (x2 - y2))
            outputs[..., 15] = (0.59004358992664352 * x * (-x2 + 3.0 * y2))

        return outputs
        