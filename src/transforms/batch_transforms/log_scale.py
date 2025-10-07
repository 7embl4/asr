import torch
from torch import nn, Tensor

class LogScale(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data: Tensor):
        return torch.log(data + 1e-8)