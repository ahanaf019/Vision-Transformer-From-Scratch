# Reference: https://arxiv.org/abs/2112.13492

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from einops.layers.torch import Rearrange


class ShiftedPatchTokenization(nn.Module):
    def __init__(self, in_channels, d_model, patch_size, shift_ratio):
        super().__init__()
        self.patch_size = patch_size
        self.shift_ratio = shift_ratio
        self.d_model = d_model
        self.shift_pixels = int(patch_size * shift_ratio)


        self.patch_embed = nn.Sequential(
            Rearrange(
            'B C (p1 h) (p2 w) -> B (h w) (C p1 p2)', p1=patch_size, p2=patch_size
            ),
            nn.LayerNorm(normalized_shape=[patch_size * patch_size * 5 * in_channels]),
            
            nn.Linear(
                in_features=patch_size * patch_size * 5 * in_channels,
                out_features=d_model
            )
        )

    
    def forward(self, x):
        dx, dy = self.shift_pixels, self.shift_pixels
        right_up = self.__shift_image(x, +dx, -dy)
        left_up = self.__shift_image(x, -dx, -dy)
        right_down = self.__shift_image(x, +dx, +dy)
        left_down = self.__shift_image(x, -dx, +dy)

        x = torch.cat([x, left_up, right_up, left_down, right_down], dim=1)
        x = self.patch_embed(x)
        return x



    def __shift_image(self, image, shift_x, shift_y):
        B, C, H, W = image.size()
        shift_x = shift_x * 2 / W
        shift_y = shift_x * 2 / H
        theta = torch.tensor(
            [[[1, 0, shift_x],
              [0, 1, shift_y]]] * B,
            dtype=image.dtype, device=image.device
        )
        grid = F.affine_grid(theta, image.size(), align_corners=False)
        shifted = F.grid_sample(image, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        return shifted




if __name__ == '__main__':
    model = ShiftedPatchTokenization(patch_size=16, in_channels=3, d_model=128, shift_ratio=0.5).to('cuda')
    summary(model, input_size=[1, 3, 224, 224], col_names=["input_size", "output_size", "num_params"])
