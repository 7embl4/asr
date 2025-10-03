import torch
import torchaudio
import torch.nn as nn
from torch_audiomentations import Gain
import torchaudio.transforms as T
from torchvision.transforms.v2 import Compose

from omegaconf import OmegaConf
from hydra.utils import instantiate
import matplotlib.pyplot as plt

from src.model import CTCModel
from src.text_encoder import CTCTextEncoder
from src.datasets import LibrispeechDataset
from src.datasets.collate import collate_fn
from src.loss import CTCLossWrapper
#from src.transforms.wav_augs import Gain

import warnings
warnings.filterwarnings('ignore')
torch.manual_seed(0)

dataset = LibrispeechDataset(
    part='train-clean-100', 
    text_encoder=CTCTextEncoder(),
    instance_transforms={
        'get_spectrogram': T.MelSpectrogram(sample_rate=16000),
        #'audio': Compose([Gain()])
    })

items = [dataset[i] for i in range(10)]
batch = collate_fn(items)

config = OmegaConf.load('src/configs/metrics/example.yaml')

text_encoder = CTCTextEncoder()
metrics = {"train": [], "inference": []}
for metric_type in ["train", "inference"]:
    for metric_config in config.get(metric_type, []):
        # use text_encoder in metrics
        metrics[metric_type].append(
            instantiate(metric_config, text_encoder=text_encoder)
        )
print(metrics)

# print('dataset item')
# print('audio: ', items[0]['audio'].shape)
# print('spectrogram: ', items[0]['spectrogram'].shape)
# print('text: ', items[0]['text'])
# print('text_encoded: ', items[0]['text_encoded'].shape)
# print('audio_path: ', items[0]['audio_path'])
# print('---------------------------')

# print('batch sizes')
# print('spectrogram: ', batch['spectrogram'].shape)
# print('text_encoded: ', batch['text_encoded'].shape)
# print('log_probs_length: ', batch['log_probs_length'].shape)
# print('text_encoded_length: ', batch['text_encoded_length'].shape)
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


