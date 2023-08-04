import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

class ActGLU(nn.Module):
    def __init__(
        self,
        hidden_size,
        ffn_hidden_size,
        act='relu',
    ):
        super(ActGLU).__init__()
        if act == 'relu':
            self.act_fn = nn.ReLU()
        elif act == 'gelu':
            self.act_fn = nn.GELU()
        elif act == 'sigmoid':
            self.act_fn = nn.Sigmoid()
        elif act == 'swish':
            self.act_fn = nn.SiLU()
        elif act == 'none':
            self.act_fn=None
        self.w_x = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.w_u = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.w_out = nn.Linear(ffn_hidden_size, hidden_size, bias=False)
    def forward(self, x):
        if self.act_fn is None:
            return self.w_out(self.w_x(x)*self.w_u(x))
        else:
            return self.w_out(self.act_fn(self.w_x(x))*self.w_u(x))

class ShiftGatedUnit(nn.Module):
    def __init__(self, config):
        super(ShiftGatedUnit, self).__init__()
        self.w_q = nn.Linear()
        self.w_k = nn.Linear()
        self.w_v = nn.Linear()
    def _create_shift_index(self):
        x=0


    def forward(self):
        x=0



class ShiftGAUConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ShiftGAU(PreTrainedModel):
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
    def forward(self, inputs):
        x=0

