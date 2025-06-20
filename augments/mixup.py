# Reference https://arxiv.org/abs/1710.09412

import torch


class MixUp():
    def __init__(self, alpha, mixup_rate=0.5):
        self.alpha = alpha
        self.mixup_rate = mixup_rate
        self.beta = torch.distributions.Beta(concentration0=alpha, concentration1=alpha)

    def generate(self, images: torch.Tensor, one_hot_targets: torch.Tensor):
        B, C, H, W = images.shape
        _lambda = self.beta.sample(sample_shape=[B]).to(images.device)
        mask = (torch.rand(size=[B]) <= self.mixup_rate).type(torch.float)
        _lambda *= mask
        
        idx = torch.randperm(n=B)
        images = _lambda.view(B, 1, 1, 1) * images + (1 - _lambda.view(B, 1, 1, 1)) * images[idx]
        one_hot_targets = _lambda.view(B, 1) * one_hot_targets + (1 - _lambda.view(B, 1)) * one_hot_targets[idx]
        return images, one_hot_targets


if __name__ == '__main__':
    rate = 0.5
    x = torch.rand(size=[100])
    print((x <= rate).type(torch.float).sum())
    beta = torch.distributions.Beta(concentration0=.4, concentration1=.4)
    _lambda = beta.sample(sample_shape=[32])
    mask = (torch.rand(size=[32]) <= rate).type(torch.float)
    _lambda *= mask
    print((_lambda == 0).sum())