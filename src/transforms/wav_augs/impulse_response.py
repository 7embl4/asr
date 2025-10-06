import torch_audiomentations
import torchaudio
from torch import Tensor, nn, conv1d


class ImpulseResponse(nn.Module):
    def __init__(self, ir_wav, *args, **kwargs):
        super().__init__()
        self._ir = torchaudio.load(ir_wav)

    def forward(self, data: Tensor):
        left_pad = right_pad = self._ir.shape[-1] - 1    
        flipped_rir = self._ir.squeeze().flip(0)

        data = nn.functional.pad(data, [left_pad, right_pad]).view(1, 1, -1)
        convolved_audio = conv1d(data, flipped_rir.view(1, 1, -1)).squeeze()
        
        if convolved_audio.abs().max() > 1:
            convolved_audio /= convolved_audio.abs().max()

        return convolved_audio