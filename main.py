import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from einops.layers.torch import Rearrange
from einops import rearrange

PATCH_SIZE = 16
IMAGE_SIZE = 112


def read_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    return image / 255.0

image_path = 'image.jpg'
image = read_image(image_path)

patches = rearrange(image, '(h p1) (w p2) c -> (h w) p1 p2 c', p1=PATCH_SIZE, p2=PATCH_SIZE)
print(patches.shape)

grid_size = int(IMAGE_SIZE / PATCH_SIZE)
fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))

for i, ax in enumerate(axes.flat):
    ax.imshow(patches[i])  # Show patch
    ax.axis("off")
plt.show()


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, in_channels, patch_size, embed_dim=128):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.proj = nn.Sequential(
            Rearrange('b (w p1) (h p2) c -> b (w h) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(in_features=patch_size*patch_size*in_channels, out_features=embed_dim)
        )
        

    def forward(self, x):
        return self.proj(x)


# class PositionalEmbedding(nn.Module):
#     def __init__(self,):
#         super().__init__()

#     def forward(self, x):

embed_dim=128
num_patches = patches.shape[0]
print(torch.randn(1, num_patches + 1, embed_dim))