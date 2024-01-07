import torch
from torch.nn import Module
from .QuantUtils import Fixed_Point_Quantize, Floating_Point_Quantize


class QMLP(Module):
    def __init__(self, n_input_dims: int, n_output_dims: int, network_config: dict, WeightBits = [3, 5], FeatureBits = [3, 13], qtype = "Fixed"):
        super().__init__()
        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims
        self.n_neurons = network_config.get("n_neurons", 64)
        self.n_hidden_layers = network_config.get("n_hidden_layers", 1)
        
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(self.n_input_dims, self.n_neurons, bias = False)] + 
            [torch.nn.Linear(self.n_neurons, self.n_neurons, bias = False) for i in range(self.n_hidden_layers - 1)] + 
            [torch.nn.Linear(self.n_neurons, self.n_output_dims, bias = False)]
        )
        
        self.WeightBits = WeightBits
        self.FeatureBits = FeatureBits
        if qtype == "Fixed":
            self.Quant_Function = Fixed_Point_Quantize
        else:
            self.Quant_Function = Floating_Point_Quantize
        
    def load_states(self, states: torch.Tensor):
        state_dict = {}
        for i, layer in enumerate(self.layers):
            if(i == 0):
                weight = states[:self.n_input_dims * self.n_neurons].reshape(
                    [self.n_neurons, self.n_input_dims]
                )
                state_dict[f"layers.{i}.weight"] = self.Quant_Function(weight, self.WeightBits[0], self.WeightBits[1])
                states = states[self.n_input_dims * self.n_neurons:]
            elif (i == self.n_hidden_layers):
                weight = states[:self.n_neurons * self.n_output_dims].reshape(
                    [self.n_output_dims, self.n_neurons]
                )
                state_dict[f"layers.{i}.weight"] = self.Quant_Function(weight, self.WeightBits[0], self.WeightBits[1])
                states = states[self.n_neurons * self.n_output_dims:]
            else:
                weight = states[:self.n_neurons * self.n_neurons].reshape(
                    [self.n_neurons, self.n_neurons]
                )
                state_dict[f"layers.{i}.weight"] = self.Quant_Function(weight, self.WeightBits[0], self.WeightBits[1])
                states = states[self.n_neurons * self.n_neurons:]
        self.load_state_dict(state_dict)
        
    def forward(self, inputs: torch.Tensor):
        x: torch.Tensor = inputs
        for i, layer in enumerate(self.layers):
            x = self.Quant_Function(x, self.FeatureBits[0], self.FeatureBits[1])
            
            x = layer(x)
            if(i < len(self.layers) - 1):
                x = torch.nn.functional.relu(x)
        x = self.Quant_Function(x, self.FeatureBits[0], self.FeatureBits[1])
        return x