from typing import Optional
import torch
import numpy as np
from torch import Tensor
import torch.nn as nn

class SinusoidalPositionEmbedding(nn.Embedding):
    def __init__(
        self, 
        num_positions: int, 
        embedding_dim: int, 
        padding_idx: int | None = None
    ) -> None:
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False
        out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out
    
    def forward(self, seq_len: int, past_key_values_length) -> Tensor:
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions)

class RotaryPositionalEmbedding(nn.Embedding):
    def __init__(
        self,
        num_positions: int,
        embedding_dim: int,
        padding_idx: int | None = None,
    ) -> None:
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, seq_len: int, past_key_values_length: int = 0) -> Tensor:
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions)
    
def apply_rotary_position_embeddings(sinusoidal_pos, q, k, v=None):
    sin, cos = sinusoidal_pos.chunk(2, dim=-1)
    sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
    cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
    rotate_half_q = torch.stack([-q[..., 1::2], q[..., 0::2]], dim=-1).reshape_as(q)
    q = q * cos_pos + rotate_half_q * sin_pos
    rotate_half_k = torch.stack([-k[..., 1::2], k[..., 0::2]], dim=-1).reshape_as(k)
    k = k * cos_pos + rotate_half_k * sin_pos

    if v is not None:
        rotate_half_v = torch.stack([-v[..., 1::2], v[..., 0::2]], dim=-1).reshape_as(k)
        v = v * cos_pos + rotate_half_v * sin_pos
        return q, k, v
    return q, k

# class ALiBiEmbedding(nn.Embedding):
#     def __init__(self):
#         super().__init__()
#     def forward(self, input: Tensor) -> Tensor:
#         return super().forward(input)


if __name__ == '__main__':
    rope = RotaryPositionalEmbedding(1024, 512)
    print(rope)
    print(rope(torch.Size([100,1024]),0)[0].shape)
    p = rope(torch.Size([100,1024]),0)
    print(p.chunk(2,dim=-1)[1])
