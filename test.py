import torch
import torchaudio
import torchaudio.transforms as T

import matplotlib.pyplot as plt

from src.model import CTCModel


wav, sample_rate = torchaudio.load('test_audio.mp3')
print(wav.shape)
print(sample_rate)

mel_spec_transform = T.MelSpectrogram(
    sample_rate,
    n_fft=2048
)
mel_spec = mel_spec_transform(wav)
print(mel_spec.shape)
print(mel_spec)

#plt.imshow(torch.log(mel_spec))