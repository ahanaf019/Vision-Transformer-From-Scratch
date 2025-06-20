import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        B, C = outputs.shape
        outputs = F.log_softmax(outputs, dim=-1)

        if targets.dtype == torch.long:
            targets = F.one_hot(targets, num_classes=C).float()
        
        # Label Smoothing: 
        # target = target * (1 - ε) + ε / (C-1) * (1 - target)
        targets = targets * (1 - self.label_smoothing) + self.label_smoothing / C * (1 - targets)

        loss = -(targets * outputs).sum(dim=1)
        return loss.mean()


if __name__ == '__main__':
    x = torch.Tensor([[0, 1, 0, 0]])
    ls = 0
    x = x * (1 - ls) + ls / x.shape[1] * (1 - x)
    print(x)