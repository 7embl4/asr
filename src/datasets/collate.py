import torch
from torch.nn.utils.rnn import pad_sequence

from src.text_encoder import CTCTextEncoder


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    text_encoded_pad = CTCTextEncoder().char2ind['']
    result_batch = {
        # 'audio': pad_sequence([item['audio'].squeeze() for item in dataset_items], batch_first=True),
        'spectrogram': pad_sequence([item['spectrogram'].squeeze().T for item in dataset_items], batch_first=True).permute(0, 2, 1),
        # 'text': None,
        'text_encoded': pad_sequence([item['text_encoded'].squeeze() for item in dataset_items], batch_first=True, padding_value=text_encoded_pad),
        'log_probs_length': torch.tensor([item['spectrogram'].shape[-1] for item in dataset_items]),  # lengths of spectrograms without padding
        'text_encoded_length': torch.tensor([item['text_encoded'].shape[-1] for item in dataset_items]),  # lengths of encoded texts without padding
    }
    return result_batch
    
