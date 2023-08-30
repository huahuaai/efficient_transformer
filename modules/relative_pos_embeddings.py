import torch
import numpy as np
import torch.nn as nn
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embed_dim, batch_size):
        super().__init__()
        self.sinpos = np.array(
            [[pos / np.power(10000, 2 * (j//2) / embed_dim) for j in range(embed_dim)] for pos in range(max_seq_len)]
        )
        self.pos = torch.FloatTensor(torch.zeros((batch_size, max_seq_len, )))

    def forward(self,x):
        x = 0



class AliBiEmbedding(nn.Module):
    def __init__(self,) -> None:
        super().__init__(*args, **kwargs)
    def forward(self):
        x=0