import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchinfo import summary

from models.modules.multihead_attention import MultiHeadAttention

class TransformerEncoder(nn.Module):
    def __init__(
            self, 
            d_model, 
            num_heads, 
            mlp_dim,
            dropout_rate,
            stochastic_path_rate,
            attention_type='attention'
            ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate

        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads=num_heads, attention_type=attention_type)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, d_model),
            # nn.GELU()
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        self.sd1 = torchvision.ops.StochasticDepth(p=stochastic_path_rate, mode='row')
        self.sd2 = torchvision.ops.StochasticDepth(p=stochastic_path_rate, mode='row')
    

    def forward(self, x: torch.Tensor):
        x_norm = self.ln1(x)
        atten_out = self.dropout1(self.mha(x_norm))
        x = x + self.sd1(atten_out)
        mlp_out = self.dropout2(self.mlp(self.ln2(x)))
        x = x + self.sd2(mlp_out)
        return x


if __name__ == '__main__':
    model = TransformerEncoder(128, 4, 512, 0.1).to('cuda')
    summary(model, input_size=[5, 256, 128])