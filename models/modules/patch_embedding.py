import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, d_model, patch_size):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.patch_size = patch_size
        self.embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x: torch.Tensor):
        x = self.embed(x)
        x = x.flatten(2, 3)
        return x.transpose(1, 2)


model = PatchEmbedding(3, 128, 16).to('cuda')
summary(model, input_size=[5, 3, 256, 256])