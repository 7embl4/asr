import torch
import torchaudio
from torch_audiomentations import Gain
import torchaudio.transforms as T
from torchvision.transforms.v2 import Compose

from omegaconf import OmegaConf
import matplotlib.pyplot as plt

from src.model import CTCModel
from src.text_encoder import CTCTextEncoder
from src.datasets import LibrispeechDataset
from src.datasets.collate import collate_fn
#from src.transforms.wav_augs import Gain

import warnings
warnings.filterwarnings('ignore')


dataset = LibrispeechDataset(
    part='train-clean-100', 
    text_encoder=CTCTextEncoder(),
    instance_transforms={
        'get_spectrogram': T.MelSpectrogram(sample_rate=16000),
        #'audio': Compose([Gain()])
    })

items = [dataset[i] for i in range(10)]
batch = collate_fn(items)


# print('dataset item')
# print('audio: ', items[0]['audio'].shape)
# print('spectrogram: ', items[0]['spectrogram'].shape)
# print('text: ', items[0]['text'])
# print('text_encoded: ', items[0]['text_encoded'].shape)
# print('audio_path: ', items[0]['audio_path'])
# print('---------------------------')

print('batch sizes')
print('spectrogram: ', batch['spectrogram'].shape)
print('text_encoded: ', batch['text_encoded'].shape)
print('input_lens: ', batch['input_lens'].shape)
print('target_lens: ', batch['target_lens'].shape)
print('---------------------------')

cfg = OmegaConf.load('src/configs/model/ctc_model.yaml')
model = CTCModel(cfg)
out = model(batch)
print(out.shape)

# print('batch element')
# print('spectrogram: ', batch['spectrogram'][0])
# print('text_encoded: ', batch['text_encoded'][0])
# print('input_lens: ', batch['input_lens'])
# print('target_lens: ', batch['target_lens'])
# print('---------------------------')


