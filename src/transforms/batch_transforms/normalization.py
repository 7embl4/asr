import torch
import torch.nn as nn

class Normalize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        mean = torch.mean(data, dim=1).unsqueeze(1)
        std = torch.std(data, dim=1).unsqueeze(1)
        return (data - mean) / (std + 1e-8)
