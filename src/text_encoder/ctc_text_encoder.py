import re
from string import ascii_lowercase
from collections import defaultdict

import torch
import numpy as np
from torchaudio.models.decoder._ctc_decoder import ctc_decoder


class Hypo:
    def __init__(self, words, score):
        self.words = words
        self.score = score

class BeamSearch:
    def __init__(self, vocab, beam_size, empty_token):
        self.vocab = vocab
        self.beam_size = beam_size
        self.empty_token = empty_token

    def __call__(self, log_probs):
        res = []
        for log_prob in log_probs:
            dp = {
                ("", self.empty_token): 0.0
            }
            for curr_step_log_prob in log_prob:
                dp = self.__expand_and_merge_beams(dp, curr_step_log_prob)
                dp = self.__truncate_beams(dp)
            res.append([Hypo(words=list(hypo[0][0]), score=hypo[1]) for hypo in dp.items()])
        return res

    def __expand_and_merge_beams(self, dp, curr_step_log_prob): 
        new_dp = defaultdict(float)
        for (pref, last_char), pref_log_proba in dp.items():
            for idx, char in enumerate(self.vocab):
                curr_log_proba = pref_log_proba + curr_step_log_prob[idx]
                curr_pref = pref
                if (char != self.empty_token and last_char != char):
                    curr_pref += char
                curr_char = char


                new_dp[(curr_pref, curr_char)] = np.logaddexp(new_dp[(curr_pref, curr_char)], curr_log_proba.item())
        return new_dp
    
    def __truncate_beams(self, dp):
        return dict(sorted(list(dp.items()), key=lambda x: x[1], reverse=True)[:self.beam_size])    


class CTCTextEncoder:
    EMPTY_TOK = "^"

    def __init__(self, alphabet=None, nbest=1, beam_size=1, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        self.beam_search_decoder = ctc_decoder(
            lexicon=None,
            tokens=self.vocab,
            lm=None,
            nbest=nbest,
            beam_size=beam_size,
            blank_token=self.EMPTY_TOK,
            sil_token=self.EMPTY_TOK
        )

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        prev_symbol = None
        res = ""
        for ind in inds:
            if (prev_symbol != ind):
                res += self.ind2char[ind]
                prev_symbol = ind
        return res.replace(self.EMPTY_TOK, "").strip()

    def beam_search(self, log_probs):
        res = self.beam_search_decoder(log_probs.detach().cpu())
        return res

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
