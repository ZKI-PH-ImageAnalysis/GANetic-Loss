import torch
from torch import nn

class GANetic(nn.Module):

    def __init__(self, eps, task, reduction):
        super(GANetic, self).__init__()
        self.eps = eps
        self.task = task
        self.reduction = reduction

    def forward(self, x, target):
        if self.task == "binary":
            x = torch.sigmoid(x)
        else:
            x = torch.softmax(x, dim = -1)
        loss = x**3.0 + torch.sqrt(torch.abs(3.985*target/torch.sum(x + self.eps)) + self.eps)
        if self.reduction == "mean":
            loss = loss.mean()
        return loss
