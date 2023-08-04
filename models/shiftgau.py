import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

# def rope(x, dim):
#     shape = x.shape
#     if isinstance(dim, int):
#         dim = [dim]
#     x1, x2 = torch.chunk(x, 2, dim=-1)

#     return torch.cat([x1*cos-x2*sin])
class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None) -> None:
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions)



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
        q = self.w_q(x).view()
        k = self.w_k(x).view()
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

