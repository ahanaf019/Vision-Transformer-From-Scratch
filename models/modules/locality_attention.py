# Reference: https://arxiv.org/abs/2112.13492

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class LocalityAttention(nn.Module):
    def __init__(self, attention_dim, key_dim, query_dim, value_dim):
        super().__init__()
        self.attention_dim = attention_dim
        self.key_dim = key_dim
        self.query_dim = query_dim
        self.value_dim = value_dim

        self.query = nn.Linear(query_dim, attention_dim)
        self.key = nn.Linear(key_dim, attention_dim)
        self.value = nn.Linear(value_dim, attention_dim)

        self.temperature = nn.Parameter(
            torch.tensor(1.0)
        )
        nn.init.trunc_normal_(self.temperature, std=1)

    

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        Q = self.query(q)
        K = self.key(k)
        V = self.value(v)

        x = torch.matmul(Q, torch.transpose(K, dim0=1, dim1=2))
        
        # Scale with temperature
        x = x / self.temperature

        # Diagonal Masking
        mask = torch.eye(x.shape[-1], device=x.device, dtype=torch.bool)
        mask = mask.expand(x.shape[0], -1, -1)
        x.masked_fill_(mask, value=-torch.inf)
        return torch.matmul(F.softmax(x, dim=-1), V)



if __name__ == '__main__':
    model = LocalityAttention(512, 512, 512, 512).to('cuda')
    data = torch.rand(
        size=[15, 5, 512]
    ).to('cuda')
    summary(model, input_data=[data, data, data], col_names=["input_size", "output_size", "num_params"])
    # summary(model, input_size=[1, 147, 512], col_names=["input_size", "output_size", "num_params"])