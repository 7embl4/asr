import re
from string import ascii_lowercase
from collections import defaultdict

import torch
import torchaudio 

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = "^"

    def __init__(self, alphabet=None, **kwargs):
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

    def beam_search(self, log_probs, beam_size=4):
        dp = {
            ("", self.EMPTY_TOK): 1.0
        }
        for curr_step_prob in log_probs:
            dp = self.__expand_and_merge_beams(dp, curr_step_prob)
            dp = self.__truncate_beams(dp, beam_size)
        return dp

    def __expand_and_merge_beams(self, dp, curr_step_prob): 
        new_dp = defaultdict(float)
        for (pref, last_char), pref_proba in dp.items():
            for idx, char in enumerate(self.vocab):
                curr_pref_proba = pref_proba * curr_step_prob[idx]
                curr_pref = pref
                if (char != self.EMPTY_TOK and last_char != char):
                    curr_pref += char
                curr_char = char
                new_dp[(curr_pref, curr_char)] += curr_pref_proba.item()
        return new_dp
    
    def __truncate_beams(self, dp, beam_size):
        return dict(sorted(list(dp.items()), key=lambda x: x[1])[:beam_size])

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^^a-z ]", "", text)
        return text
