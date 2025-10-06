import torch
import torch.nn as nn
from hydra.utils import instantiate


class Subsampling(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels,
            kernel_size=4,
            fc_out=256,
            dropout_rate=0.1
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
            fc_multiplier=4,
            dropout_rate=0.1
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
            n_attention_heads=4,
            fc_dim=256,
            n_layers=16
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
            kernel_size=31,
            dropout_rate=0.1
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
    def __init__(self, in_feat):
        super().__init__()

        self.layers = nn.Sequential(
            FeedForwardModule(in_feat),
            MultiHeadAttentionModule(in_feat),
            ConvolutionModule(in_feat),
            FeedForwardModule(in_feat)    
        )
        self.layer_norm = nn.LayerNorm(in_feat)

    def forward(self, x):
        x = self.layers(x)
        x = self.layer_norm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_feat, decoder_dim, out_feat):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_feat, decoder_dim),
            nn.ReLU(),
            nn.Linear(decoder_dim, out_feat)
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x


class CTCModel(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            subsampling_out,
            n_blocks,
            decoder_dim,
            out_feat
        ):
        super().__init__()

        self.subsampling = Subsampling(
            in_channels, 
            out_channels, 
            fc_out=subsampling_out
        )
        modules = []
        for _ in range(n_blocks):
            modules.append(ConformerBlock(subsampling_out))
        self.conf_blocks = nn.Sequential(*modules)

        self.decoder = Decoder(subsampling_out, decoder_dim, out_feat)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, spectrogram, **batch):
        x = torch.transpose(spectrogram, 1, 2)
        x = self.subsampling(x)
        x = self.conf_blocks(x)
        x = self.decoder(x)
        x = self.log_softmax(x)
        return {'log_probs': x}
