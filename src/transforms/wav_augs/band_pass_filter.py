import torch_audiomentations
from torch import Tensor, nn


class BandPassFilter(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.BandPassFilter(*args, **kwargs)

    def forward(self, data: Tensor):
        data = data.unsqueeze(1)
        return self._aug(data).squeeze(1)
