import torch.nn as nn
from hydra.utils import instantiate


class CTCModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = instantiate(config.encoder)
        self.fc = instantiate(config.classifier)
        # maybe softmax here
    
    def forward(self, batch):
        spec_batch = batch['spectrogram']
        print(spec_batch.shape)
        x = self.encoder(spec_batch.permute(0, 2, 1))
        x = self.fc(x)
        return x
