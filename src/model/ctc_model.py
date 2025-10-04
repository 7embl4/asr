import torch
import torch.nn as nn
from hydra.utils import instantiate


class Subsampling(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels,
            kernel_size,
            fc_out,
            dropout_rate
        ):
        super().__init__()
        
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size
        )
        self.linear = nn.Linear(in_channels, fc_out)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # x_in: [batch_size, seq_len, n_mel]
        # x_out: [batch_size, seq_len // kernel_size, fc_out]
        x = torch.transpose(x, 1, 2)
        x = self.conv(x)
        x = torch.transpose(x, 1, 2)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class FeedForwardModule(nn.Module):
    def __init__(
            self,
            in_feat,
            fc_multiplier,
            dropout_rate
        ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.LayerNorm(in_feat),
            nn.Linear(in_feat, in_feat * fc_multiplier),
            nn.SiLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_feat * fc_multiplier, in_feat),
            nn.Dropout(p=dropout_rate)
        )
        
    def forward(self, x):
        _x = self.layers(x)
        _x = 0.5 * _x
        x = _x + x
        return x


class MultiHeadAttentionModule(nn.Module):
    def __init__(
            self,
            in_feat,
            n_attention_heads,
            fc_dim,
            n_layers
        ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.LayerNorm(in_feat),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=in_feat,
                    nhead=n_attention_heads,
                    dim_feedforward=fc_dim,
                    batch_first=True
                ),
                num_layers=n_layers
            )
        )

    def forward(self, x):
        _x = self.layers(x)
        x = _x + x
        return x
    

class ConvolutionModule(nn.Module):
    def __init__(
            self,
            in_feat,
            kernel_size,
            dropout_rate
        ):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(in_feat)
        self.layers = nn.Sequential(
            nn.Conv1d(  # pointwise conv
                in_channels=in_feat,
                out_channels=in_feat * 2,
                kernel_size=1
            ),  
            nn.GLU(dim=1),
            nn.Conv1d(  # 1d deepwise conv
                in_channels=in_feat,
                out_channels=in_feat,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                groups=in_feat
            ),
            nn.BatchNorm1d(in_feat),
            nn.SiLU(),
            nn.Conv1d(  # pointwise conv
                in_channels=in_feat,
                out_channels=in_feat,
                kernel_size=1
            ),  
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, x):
        _x = self.layer_norm(x)
        _x = torch.transpose(_x, 1, 2)
        _x = self.layers(_x)
        _x = torch.transpose(_x, 1, 2)
        x = _x + x
        return x


class ConformerBlock(nn.Module):
    def __init__(
            self,
            fc_module_1,
            attention_module,
            convolution_module,
            fc_module_2,
            subsampling_out
        ):
        super().__init__()

        self.fc_module_1 = fc_module_1
        self.attention_module = attention_module
        self.convolution_module = convolution_module
        self.fc_module_2 = fc_module_2
        self.layer_norm = nn.LayerNorm(subsampling_out)

    def forward(self, x):
        x = self.fc_module_1(x)
        x = self.attention_module(x)
        x = self.convolution_module(x)
        x = self.fc_module_2(x)
        x = self.layer_norm(x)
        return x


class CTCModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.subsampling = instantiate(config.subsampling)
        modules = []
        for _ in range(config.get('n_blocks', 4)):
            modules.append(ConformerBlock(
                instantiate(config.fc_module),
                instantiate(config.attention_module),
                instantiate(config.convolution_module),
                instantiate(config.fc_module),
                config.get('subsampling_out', 1024)
            ))
        self.conf_blocks = nn.Sequential(*modules)
        self.decoder = instantiate(config.decoder)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, spectrogram, **batch):
        x = torch.transpose(spectrogram, 1, 2)
        res = {}
        x = self.subsampling(x)
        x = self.conf_blocks(x)
        x = self.decoder(x)
        res['logits'] = x
        x = self.log_softmax(x)
        res['log_probs'] = x
        return res
