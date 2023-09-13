import torch
import torch.nn as nn
from .pos_embeddings import SinusoidalPositionEmbedding, RotaryPositionalEmbedding, apply_rotary_position_embeddings
class GAU(nn.Module):
    def __init__(self, ):
        super().__init__()
    def forward(self):
        x=0


#We implement transformer's multi-head self attention by using pytorch.
# class MultiHeadAttention():

    