import torch.nn as nn
from hydra.utils import instantiate


class CTCModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = instantiate(config.encoder)
        self.fc = instantiate(config.classifier)
        self.log_softmax = nn.LogSoftmax()
    
    def forward(self, batch):
        spec_batch = batch['spectrogram']
        out = self.encoder(spec_batch.permute(0, 2, 1))
        out = self.fc(out)
        out = self.log_softmax(out)
        return {'log_probs': out}
