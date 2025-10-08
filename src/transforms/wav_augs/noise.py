import torch
from torch import Tensor, nn, distributions


class Noise(nn.Module):
    def __init__(self, p=0.5, *args, **kwargs):
        super().__init__()
        self.p = p
        self._noise = distributions.Normal(*args, **kwargs)
        self._add_noise_proba = distributions.Uniform(0.0, 1.0)

    def forward(self, data: Tensor):
        add_noise = self._add_noise_proba.sample(torch.tensor([1])).item() <= self.p
        if (add_noise):
            data = data + self._noise.sample(data.shape)
        return data
