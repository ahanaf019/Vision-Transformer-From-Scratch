import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from modules.attention import Attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.atten_dim = d_model // num_heads

        self.heads = nn.ModuleList([Attention(self.d_model, self.atten_dim, self.atten_dim, self.atten_dim) for _ in range(num_heads)])
        self.W_o = nn.Linear(d_model, d_model)
    

    def forward(self, x: torch.Tensor):
        head_output = torch.cat([head(x, x, x) for head in self.heads], dim=-1)
        return self.W_o(head_output)


if __name__ == '__main__':
    model = MultiHeadAttention(128, 4).to('cuda')
    summary(model, input_size=[5, 256, 128])