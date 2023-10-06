import torch
from torch.nn import Module

class MLP(Module):
    def __init__(self, n_input_dims: int, n_output_dims: int, network_config: dict):
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
    def load_states(self, states: torch.Tensor):
        state_dict = {}
        for i, layer in enumerate(self.layers):
            if(i == 0):
                state_dict[f"layers.{i}.weight"] = states[:self.n_input_dims * self.n_neurons].reshape(
                    [self.n_neurons, self.n_input_dims]
                )
                states = states[self.n_input_dims * self.n_neurons:]
            elif (i == self.n_hidden_layers):
                state_dict[f"layers.{i}.weight"] = states[:self.n_neurons * self.n_output_dims].reshape(
                    [self.n_output_dims, self.n_neurons]
                )
                states = states[self.n_neurons * self.n_output_dims:]
            else:
                state_dict[f"layers.{i}.weight"] = states[:self.n_neurons * self.n_neurons].reshape(
                    [self.n_neurons, self.n_neurons]
                )
                states = states[self.n_neurons * self.n_neurons:]
        self.load_state_dict(state_dict)
        
    def forward(self, inputs: torch.Tensor):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if(i < len(self.layers) - 1):
                x = torch.nn.functional.relu(x)
        return x