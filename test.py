import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.model import CTCModel
from src.text_encoder import CTCTextEncoder
from src.datasets import LibrispeechDataset, CustomDirAudioDataset
from src.datasets.collate import collate_fn
from string import ascii_lowercase

from torchaudio.models.decoder._ctc_decoder import ctc_decoder, download_pretrained_files
from src.metrics.utils import calc_cer, calc_wer
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')
#torch.manual_seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

path = Path('data/datasets/custom_dataset/audio')
print(path)
print(path.suffix)
print(next(iter(path.iterdir())))
print(path.stem)

dataset = CustomDirAudioDataset('data/datasets/custom_dataset/audio', 'data/datasets/custom_dataset/transcriptions')
print(dataset[0])