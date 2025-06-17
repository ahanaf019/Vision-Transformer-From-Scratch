import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, sequence_len):
        super().__init__()
        self.d_model = d_model
        self.sequence_len = sequence_len

        self.embed = nn.Parameter(
            torch.zeros(size=[sequence_len, d_model])
        )
        nn.init.trunc_normal_(self.embed, std=0.02)


    def forward(self, x:torch.Tensor):
        return x + self.embed.expand(x.size(0), -1, -1)

if __name__ == '__main__':
    model = PositionalEmbedding(128, 256).to('cuda')
    summary(model, input_size=[5, 256, 128])