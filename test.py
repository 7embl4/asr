import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from torchvision.transforms.v2 import Compose

from src.model import CTCModel
from src.text_encoder import CTCTextEncoder
from src.datasets import LibrispeechDataset, CustomDirAudioDataset
from src.datasets.collate import collate_fn
from string import ascii_lowercase

from torchaudio.models.decoder._ctc_decoder import ctc_decoder, download_pretrained_files
from src.metrics.utils import calc_cer, calc_wer
from pathlib import Path
from torchinfo import summary

import warnings
warnings.filterwarnings('ignore')
#torch.manual_seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = LibrispeechDataset(
    'train-clean-100',
    text_encoder=CTCTextEncoder(),
    instance_transforms={
        'get_spectrogram': T.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            f_min=50,
            f_max=8000,
            n_mels=80,
            center=False
        )
    }
)

dataloader = DataLoader(
    dataset=dataset,
    collate_fn=collate_fn,
    drop_last=True,
    shuffle=True,
    batch_size=8
)

print(len(dataloader))


model = CTCModel(
    in_channels=80,
    out_channels=80,
    subsampling_rate=2,
    subsampling_out=144,
    n_blocks=4,
    decoder_dim=320,
    out_feat=28
)
print(summary(model))