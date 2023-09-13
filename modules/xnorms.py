import torch
import torch.nn as nn
import torch.nn.functional as F
class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: int=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epslion = eps
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epslion)
        return self.weight * hidden_states.to(input_dtype)

# class DeepNorm(nn.Moduel):
#     def __init__(self, ):
#         super().__init__()
#     def forward(self):
#         x=0
