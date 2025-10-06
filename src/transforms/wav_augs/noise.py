from torch import Tensor, nn, distributions


class Noise(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._noise = distributions.Normal(*args, **kwargs)

    def forward(self, data: Tensor):
        data = data + self._noise.sample(data.shape)
        return data
