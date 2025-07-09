import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class Attention(nn.Module):
    def __init__(self, attention_dim, q_dim, k_dim, v_dim, bias=False):
        super().__init__()
        self.attention_dim = attention_dim
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.query = nn.Linear(attention_dim, q_dim)
        self.key = nn.Linear(attention_dim, k_dim)
        self.value = nn.Linear(attention_dim, v_dim)

        # if bias:
        #     self.bias = torch.

    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        Q = self.query(q)
        K = self.key(k)
        V = self.value(v)
        
        x = torch.matmul(Q, torch.transpose(K, dim0=1, dim1=2))
        x = x / (self.k_dim ** 0.5)
        x = F.softmax(x, dim=-1)
        # print(x.shape)
        return torch.matmul(x, V)


if __name__ == '__main__':
    model = Attention(128, 128, 128, 128).to('cuda')
    data = torch.rand(5, 384, 128).to('cuda')
    data = [data, data, data]
    summary(model, input_data=data)