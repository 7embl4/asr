import torch.nn as nn


class CTCEncoder(nn.Module):
    def __init__(self, encoder_type, config):
        super().__init__()

        if (encoder_type == 'rnn'):
            self.encoder = nn.RNN(
                config['in_features'],
                config['rnn']['hidden_size'],
                config['rnn']['num_layers']
            )
        elif (encoder_type == 'transformer'):    
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    config['in_features'],
                    config['transformer']['nhead'],
                    config['transformer']['dim_fc']
                ),
                config['transformer']['num_layers']
            )
        else:
            raise f'Unknown encoder type {encoder_type}. Use either rnn or transformer'

    def forward(self, x):
        return self.encoder(x)

class CTCClassifier(nn.Module):
    def __init__(self, encoder_type, config):
        super().__init__()

        if (encoder_type == 'rnn'):
            self.fc = nn.Linear(config['rnn']['hidden_size'], config['out_features'])
        elif (encoder_type == 'transformer'):
            self.fc = nn.Linear(config['in_features'], config['out_features'])
        else:
            raise f'Unknown encoder type {encoder_type}. Use either rnn or transformer'
    
    def forward(self, x):
        return self.fc(x)

class CTCModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        encoder_type = config.get('encoder_type', None)
        self.encoder = CTCEncoder(encoder_type, config)
        self.fc = CTCClassifier(encoder_type, config)
        # maybe softmax here    
    
    def forward(self, x):
        return self.fc(self.encoder(x))
