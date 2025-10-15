from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer

from src.text_encoder import BeamSearch

from torchaudio.models.decoder._ctc_decoder import ctc_decoder


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)


class BeamSearchCERMetric(BaseMetric):
    def __init__(
            self, 
            text_encoder, 
            nbest, 
            beam_size,
            lexicon_path=None,
            vocab_path=None, 
            lm_path=None, 
            type="torch",
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        if type == 'torch':
            self.decoder = ctc_decoder(
                lexicon=lexicon_path,
                tokens=vocab_path if vocab_path else text_encoder.vocab,
                lm=lm_path,
                nbest=nbest,
                beam_size=beam_size,
                blank_token=text_encoder.EMPTY_TOK,
                sil_token=text_encoder.EMPTY_TOK
            )
        elif type == 'handcrafted':
            self.decoder = BeamSearch(
                vocab=text_encoder.vocab,
                beam_size=beam_size,
                empty_token=text_encoder.EMPTY_TOK
            )
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        cers = []
        predictions = log_probs.detach().cpu()
        lengths = log_probs_length.detach().numpy()
        for pred, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text, cer = self._get_best_prediction(pred.unsqueeze(0), target_text)
            cers.append(cer)
        return sum(cers) / len(cers)
    
    def _get_best_prediction(self, predictions, target_text):
        hypos = self.decoder(predictions)
        best_hypo = ""
        best_cer = 1000
        for hypo in hypos[0]:
            if len(hypo.words) != 0:
                hypo_text = "".join(hypo.words)
            else:
                hypo_text = self.text_encoder.ctc_decode(hypo.tokens.toist())
            hypo_cer = calc_cer(target_text, hypo_text)
            if hypo_cer <= best_cer:
                best_hypo = hypo_text
                best_cer = hypo_cer
        
        return best_hypo, best_cer