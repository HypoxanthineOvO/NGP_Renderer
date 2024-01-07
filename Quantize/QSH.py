import torch
from torch.nn import Module
from .QuantUtils import Fixed_Point_Quantize

class QSHEncoding(Module):
    ### TODO
    def __init__(self, n_input_dims, encoding_config: dict, FeatureBits = 8, ResultBits = 8):
        super().__init__()
        self.n_input_dims = n_input_dims
        self.degree = encoding_config["nested"][0]["degree"]
        self.FeatureBits = FeatureBits
        self.ResultBits = ResultBits
    
    def forward(self, inputs: torch.Tensor):
        shape = list(inputs.shape)
        assert(len(shape) == 2)
        assert(shape[-1] == 3)
        
        # x = inputs[..., 0] * 2 - 1
        # y = inputs[..., 1] * 2 - 1
        # z = inputs[..., 2] * 2 - 1
        # xy = x * y
        # xz = x * z
        # yz = y * z
        # x2 = x * x
        # y2 = y * y
        # z2 = z * z
        x = Fixed_Point_Quantize(inputs[..., 0] * 2 - 1, self.FeatureBits[0], self.FeatureBits[1])
        y = Fixed_Point_Quantize(inputs[..., 1] * 2 - 1, self.FeatureBits[0], self.FeatureBits[1])
        z = Fixed_Point_Quantize(inputs[..., 2] * 2 - 1, self.FeatureBits[0], self.FeatureBits[1])
        xy = Fixed_Point_Quantize(x * y, self.FeatureBits[0], self.FeatureBits[1])
        xz = Fixed_Point_Quantize(x * z, self.FeatureBits[0], self.FeatureBits[1])
        yz = Fixed_Point_Quantize(y * z, self.FeatureBits[0], self.FeatureBits[1])
        x2 = Fixed_Point_Quantize(x * x, self.FeatureBits[0], self.FeatureBits[1])
        y2 = Fixed_Point_Quantize(y * y, self.FeatureBits[0], self.FeatureBits[1])
        z2 = Fixed_Point_Quantize(z * z, self.FeatureBits[0], self.FeatureBits[1])
        
        
        outputs = torch.zeros([shape[0], 16], dtype = torch.float32, device = inputs.device)
        
        if(self.degree >= 1):
            outputs[..., 0] = Fixed_Point_Quantize(0.28209479177387814, self.ResultBits[0], self.ResultBits[1])
        # if(self.degree >= 2):
        #     outputs[..., 1] = (-0.48860251190291987 * y)
        #     outputs[..., 2] = (0.48860251190291987 * z)
        #     outputs[..., 3] = (
        #         -0.48860251190291987 * x)
        # if(self.degree >= 3):
        #     outputs[..., 4] = (1.0925484305920792 * xy)
        #     outputs[..., 5] = (-1.0925484305920792 * yz)
        #     outputs[..., 6] = (0.94617469575755997 * z2 - 0.31539156525251999)
        #     outputs[..., 7] = (-1.0925484305920792 * xz)
        #     outputs[..., 8] = (0.54627421529603959 * x2 - 0.54627421529603959 * y2)
        # if(self.degree >= 4):
        #     outputs[..., 9] = (0.59004358992664352 * y * (-3.0 * x2 + y2))
        #     outputs[..., 10] = (2.8906114426405538 * xy * z)
        #     outputs[..., 11] = (0.45704579946446572 * y * (1.0 - 5.0 * z2))
        #     outputs[..., 12] = (0.3731763325901154 * z * (5.0 * z2 - 3.0))
        #     outputs[..., 13] = (0.45704579946446572 * x * (1.0 - 5.0 * z2))
        #     outputs[..., 14] = (1.4453057213202769 * z * (x2 - y2))
        #     outputs[..., 15] = (0.59004358992664352 * x * (-x2 + 3.0 * y2))
        if(self.degree >= 2):
            outputs[..., 1] = Fixed_Point_Quantize(Fixed_Point_Quantize(
                -0.48860251190291987, self.FeatureBits[0], self.ResultBits[1])
                * Fixed_Point_Quantize(y, self.FeatureBits[0], self.FeatureBits[1]),
                self.ResultBits[0], self.ResultBits[1])
            outputs[..., 2] = Fixed_Point_Quantize(Fixed_Point_Quantize(
                0.48860251190291987, self.FeatureBits[0], self.ResultBits[1])
                * Fixed_Point_Quantize(z, self.FeatureBits[0], self.FeatureBits[1]),
                self.ResultBits[0], self.ResultBits[1])
            outputs[..., 3] = Fixed_Point_Quantize(Fixed_Point_Quantize(
                -0.48860251190291987, self.FeatureBits[0], self.ResultBits[1])
                * Fixed_Point_Quantize(x, self.FeatureBits[0], self.FeatureBits[1]),
                self.ResultBits[0], self.ResultBits[1])
        if(self.degree >= 3):
            outputs[..., 4] = Fixed_Point_Quantize(Fixed_Point_Quantize(
                1.0925484305920792, self.FeatureBits[0], self.ResultBits[1])
                * Fixed_Point_Quantize(xy, self.FeatureBits[0], self.FeatureBits[1]),
                self.ResultBits[0], self.ResultBits[1])
            outputs[..., 5] = Fixed_Point_Quantize(Fixed_Point_Quantize(
                -1.0925484305920792, self.FeatureBits[0], self.ResultBits[1])
                * Fixed_Point_Quantize(yz, self.FeatureBits[0], self.FeatureBits[1]),
                self.ResultBits[0], self.ResultBits[1])
            outputs[..., 6] = Fixed_Point_Quantize(Fixed_Point_Quantize(
                0.94617469575755997, self.FeatureBits[0], self.ResultBits[1])
                * Fixed_Point_Quantize(z2, self.FeatureBits[0], self.FeatureBits[1])
                - Fixed_Point_Quantize(0.31539156525251999, self.FeatureBits[0], self.ResultBits[1]),
                self.ResultBits[0], self.ResultBits[1])
            outputs[..., 7] = Fixed_Point_Quantize(Fixed_Point_Quantize(
                -1.0925484305920792, self.FeatureBits[0], self.ResultBits[1])
                * Fixed_Point_Quantize(xz, self.FeatureBits[0], self.FeatureBits[1]),
                self.ResultBits[0], self.ResultBits[1])
            outputs[..., 8] = Fixed_Point_Quantize(Fixed_Point_Quantize(
                0.54627421529603959, self.FeatureBits[0], self.ResultBits[1])
                * Fixed_Point_Quantize(x2, self.FeatureBits[0], self.FeatureBits[1])
                - Fixed_Point_Quantize(0.54627421529603959, self.FeatureBits[0], self.ResultBits[1])
                * Fixed_Point_Quantize(y2, self.FeatureBits[0], self.FeatureBits[1]),
                self.ResultBits[0], self.ResultBits[1])
        if(self.degree >= 4):
            outputs[..., 9] = Fixed_Point_Quantize(Fixed_Point_Quantize(
                0.59004358992664352, self.FeatureBits[0], self.ResultBits[1])
                * Fixed_Point_Quantize(y, self.FeatureBits[0], self.FeatureBits[1])
                * Fixed_Point_Quantize(-3.0 * x2 + y2, self.FeatureBits[0], self.FeatureBits[1]),
                self.ResultBits[0], self.ResultBits[1])
            outputs[..., 10] = Fixed_Point_Quantize(Fixed_Point_Quantize(
                2.8906114426405538, self.FeatureBits[0], self.ResultBits[1])
                * Fixed_Point_Quantize(xy, self.FeatureBits[0], self.FeatureBits[1])
                * Fixed_Point_Quantize(z, self.FeatureBits[0], self.FeatureBits[1]),
                self.ResultBits[0], self.ResultBits[1])
            outputs[..., 11] = Fixed_Point_Quantize(Fixed_Point_Quantize(
                0.45704579946446572, self.FeatureBits[0], self.ResultBits[1])
                * Fixed_Point_Quantize(y, self.FeatureBits[0], self.FeatureBits[1])
                * Fixed_Point_Quantize(1.0 - 5.0 * z2, self.FeatureBits[0], self.FeatureBits[1]),
                self.ResultBits[0], self.ResultBits[1])
            outputs[..., 12] = Fixed_Point_Quantize(Fixed_Point_Quantize(
                0.3731763325901154, self.FeatureBits[0], self.ResultBits[1])
                * Fixed_Point_Quantize(z, self.FeatureBits[0], self.FeatureBits[1])
                * Fixed_Point_Quantize(5.0 * z2 - 3.0, self.FeatureBits[0], self.FeatureBits[1]),
                self.ResultBits[0], self.ResultBits[1])
            outputs[..., 13] = Fixed_Point_Quantize(Fixed_Point_Quantize(
                0.45704579946446572, self.FeatureBits[0], self.ResultBits[1])
                * Fixed_Point_Quantize(x, self.FeatureBits[0], self.FeatureBits[1])
                * Fixed_Point_Quantize(1.0 - 5.0 * z2, self.FeatureBits[0], self.FeatureBits[1]),
                self.ResultBits[0], self.ResultBits[1])
            outputs[..., 14] = Fixed_Point_Quantize(Fixed_Point_Quantize(
                1.4453057213202769, self.FeatureBits[0], self.ResultBits[1])
                * Fixed_Point_Quantize(z, self.FeatureBits[0], self.FeatureBits[1])
                * Fixed_Point_Quantize(x2 - y2, self.FeatureBits[0], self.FeatureBits[1]),
                self.ResultBits[0], self.ResultBits[1])
            outputs[..., 15] = Fixed_Point_Quantize(Fixed_Point_Quantize(
                0.59004358992664352, self.FeatureBits[0], self.ResultBits[1])
                * Fixed_Point_Quantize(x, self.FeatureBits[0], self.FeatureBits[1])
                * Fixed_Point_Quantize(-x2 + 3.0 * y2, self.FeatureBits[0], self.FeatureBits[1]),
                self.ResultBits[0], self.ResultBits[1])
        return outputs
        