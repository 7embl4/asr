import torch
import torchaudio
from torchinfo import summary
import torch.nn as nn
import torchaudio.transforms as T
from torchvision.transforms.v2 import Compose
from math import ceil

from omegaconf import OmegaConf
from hydra.utils import instantiate
import matplotlib.pyplot as plt

from src.model import CTCModel
from src.text_encoder import CTCTextEncoder
from src.datasets import LibrispeechDataset
from src.datasets.collate import collate_fn
from src.loss import CTCLossWrapper
from src.transforms.wav_augs import (
    Gain, 
    Pitch, 
    Noise, 
    ImpulseResponse,
    BandPassFilter
)

from src.metrics.utils import calc_wer
from src.metrics.utils import calc_cer

import warnings
warnings.filterwarnings('ignore')
torch.manual_seed(0)

# m = nn.Conv1d(16, 32, kernel_size=5, padding=2)
# i = torch.rand(10, 16, 80)
# out = m(i)
# print(out.shape)

# n = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
# a = torch.rand(10, 3, 128, 128)
# out = n(a)
# print(out.shape)

dataset = LibrispeechDataset(
    part='train-clean-100', 
    text_encoder=CTCTextEncoder(),
    instance_transforms={
        'get_spectrogram': T.MelSpectrogram(sample_rate=16000),
        'audio': Compose([
            Gain(), 
            Pitch(sample_rate=16000),
            Noise(loc=0.0, scale=0.03),
            BandPassFilter(sample_rate=16000)
        ])
    })

items = [dataset[i] for i in range(10)]
batch = collate_fn(items)

print(ceil(4.3))

# model = CTCModel(
#     in_channels=128,  # n_mel
#     out_channels=128,
#     subsampling_out=256,
#     n_blocks=4,
#     decoder_dim=640,
#     out_feat=28  # vocab len
# )
# out = model(**batch)
# print(summary(model))
# print(out["log_probs"].shape)

# encoder = CTCTextEncoder()
# beam_search_res = encoder.beam_search(out["log_probs"])
# print(len(beam_search_res))
# print(len(beam_search_res[0]))
# print(beam_search_res[0][0].tokens)

# t = torch.tensor([1, 2, 3])
# print(t.unsqueeze(0).shape)
# print(t.unsqueeze(1).shape)



# text_encoder = CTCTextEncoder()
# print(text_encoder.beam_search(out['log_probs'][0]))

# text_encoder = CTCTextEncoder()
# text = "^^^^^^hhhh^^^^e^ll^^^^llooooo ^^w^orrl^^^^ddd"
# encoded_text = text_encoder.encode(text).squeeze().tolist()
# print('encoded text: ', encoded_text)
# print('decoded text: ', text_encoder.ctc_decode(encoded_text))




# print('batch sizes')
# print('spectrogram: ', batch['spectrogram'].shape)
# print('text_encoded: ', batch['text_encoded'].shape)
# print('log_probs_length: ', batch['log_probs_length'].shape)
# print('text_encoded_length: ', batch['text_encoded_length'].shape)
# print('---------------------------')







# config = OmegaConf.load('src/configs/metrics/example.yaml')

# text_encoder = CTCTextEncoder()
# metrics = {"train": [], "inference": []}
# for metric_type in ["train", "inference"]:
#     for metric_config in config.get(metric_type, []):
#         # use text_encoder in metrics
#         metrics[metric_type].append(
#             instantiate(metric_config, text_encoder=text_encoder)
#         )
# print(metrics)

# print('dataset item')
# print('audio: ', items[0]['audio'].shape)
# print('spectrogram: ', items[0]['spectrogram'].shape)
# print('text: ', items[0]['text'])
# print('text_encoded: ', items[0]['text_encoded'].shape)
# print('audio_path: ', items[0]['audio_path'])
# print('---------------------------')

# cfg = OmegaConf.load('src/configs/model/ctc_model.yaml')
# model = CTCModel(cfg)
# out = model(batch)
# batch.update(out)
# # print(out.shape)
# # model output is [batch_size, seq_len, vocab_len]

# ctc_wrapper = CTCLossWrapper()
# ctc_criterion = nn.CTCLoss()
# wrapper_loss = ctc_wrapper(**batch)
# real_loss = ctc_criterion(
#     log_probs=torch.transpose(batch['log_probs'], 0, 1),
#     targets=batch['text_encoded'],
#     input_lengths=batch['log_probs_length'],
#     target_lengths=batch['text_encoded_length']
# )
# print(wrapper_loss)
# print(real_loss)

# print('batch element')
# print('spectrogram: ', batch['spectrogram'][0])
# print('text_encoded: ', batch['text_encoded'][0])
# print('input_lens: ', batch['input_lens'])
# print('target_lens: ', batch['target_lens'])
# print('---------------------------')


