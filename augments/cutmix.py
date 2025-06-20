# Reference: https://arxiv.org/abs/1905.04899

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class CutMix():
    def __init__(self, alpha, cutmix_rate=0.5):
        self.alpha = alpha
        self.cutmix_rate = cutmix_rate
        self.beta = torch.distributions.Beta(concentration0=alpha, concentration1=alpha)
    

    def generate(self, images: torch.Tensor, one_hot_targets: torch.Tensor):
        B, C, H, W = images.shape
        mask = (torch.rand(size=[B]) <= self.cutmix_rate).type(torch.int32).to(images.device)

        _lambda = self.beta.sample(sample_shape=[B]).to(images.device)
        _lambda *= mask
        cx = torch.randint(low=0, high=W, size=[B]).to(images.device)
        cy = torch.randint(low=0, high=H, size=[B]).to(images.device)

        rw = (W * (1 - _lambda) ** 0.5).type(torch.int32)
        rh = (H * (1 - _lambda) ** 0.5).type(torch.int32)
        rw *= mask
        rh *= mask

        x1 = torch.clamp(cx - rw // 2, min=0, max=W)
        x2 = torch.clamp(cx + rw // 2, min=0, max=W)
        y1 = torch.clamp(cy - rh // 2, min=0, max=H)
        y2 = torch.clamp(cy + rh // 2, min=0, max=H)


        idx = torch.randperm(n=B)
        images_cutmix = images.clone()
        for i in range(B):
            images_cutmix[i, :, y1[i]: y2[i], x1[i]: x2[i]] = images[idx[i], :, y1[i]: y2[i], x1[i]: x2[i]]
        
        _lambda = 1 - (x2 - x1) * (y2 - y1) / (W * H)
        one_hot_targets = _lambda.view(B, 1) * one_hot_targets + (1 - _lambda.view(B, 1)) * one_hot_targets[idx]
        return images_cutmix, one_hot_targets