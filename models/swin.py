import torch
import torch.nn as nn
import torch.nn.functional as F


class SwinT(nn.Module):
    def __init__(self, d_model, image_size, patch_size, num_classes):
        super().__init__()
    

    def forward(self, x):
        return x