import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Optional
from transformers import PretrainedConfig, PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

# def rope(x, dim):
#     shape = x.shape
#     if isinstance(dim, int):
#         dim = [dim]
#     x1, x2 = torch.chunk(x, 2, dim=-1)

#     return torch.cat([x1*cos-x2*sin])


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
    def forward(self, x, a):
        if self.act_fn is None:
            return self.w_out(self.w_x(x)*self.w_u(a))
        else:
            return self.w_out(self.act_fn(self.w_x(x))*self.w_u(a))

class ShiftGatedUnit(nn.Module):
    def __init__(self, config):
        super(ShiftGatedUnit, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = self.hidden_size//self.num_heads
        self.ffn_hidden_size = config.intermediate_size
        self.w_q = nn.Linear(self.hidden_size,self.hidden_size,bias=False)
        self.w_k = nn.Linear(self.hidden_size,self.hidden_size,bias=False)
        self.actglu = ActGLU(hidden_size=self.hidden_size, ffn_hidden_size=self.ffn_hidden_size,act='swish')
    def forward(self, x, ):
        bsz, seq_len, _ = x.shape
        q = self.w_q(x).view(bsz, seq_len, self.num_heads, self.head_size)
        k = self.w_k(x).view(bsz, seq_len, self.num_heads, self.head_size)
        k = torch.roll()
    



class ShiftGatedNetConfig(PretrainedConfig):
    model_type = 'ShiftGatedNet'
    def __init__(
        self,
        vocab_size=0,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id,**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout


class ShiftGatedNet(PreTrainedModel):
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
    def forward(self, inputs):
        x=0

