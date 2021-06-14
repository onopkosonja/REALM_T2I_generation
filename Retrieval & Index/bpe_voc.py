from bpemb import BPEmb
import numpy as np
import torch
from sentencepiece import SentencePieceProcessor
import re


def sentencepiece_load(file):
    """Load a SentencePiece model"""
    spm = SentencePieceProcessor()
    spm.Load(str(file))
    return spm


def preprocess(text):
        return re.sub(r"\d", "0", text.lower())


class BpeVocabulary:
    def __init__(self, emb_size=300, model_file=None, emb_file=None, saved_embs=None):
        if saved_embs:
            self.spm = sentencepiece_load(model_file)
            self.BOS_str = "<s>"
            self.EOS_str = "</s>"
            self.BOS = self.spm.PieceToId(self.BOS_str)
            self.EOS = self.spm.PieceToId(self.EOS_str)
            self.do_preproc = True
            self.preprocess = preprocess
            self.embs = torch.Tensor(np.load(saved_embs))

        # elif emb_file:
        #     self.bpemb = BPEmb(lang="en", dim=emb_size, add_pad_emb=True, model_file=model_file, emb_file=emb_file)
        #     self.embs = torch.Tensor(self.bpemb.vectors)
        #
        # else:
        #     self.bpemb = BPEmb(lang="en", dim=emb_size, add_pad_emb=True)
        #     self.embs = torch.Tensor(self.bpemb.vectors)

    def _encode(self, texts, fn):
        if isinstance(texts, str):
            if self.do_preproc:
                texts = self.preprocess(texts)
            return fn(texts)
        if self.do_preproc:
            texts = map(self.preprocess, texts)
        return list(map(fn, texts))

    def encode_ids_with_bos_eos(self, texts):
        return self._encode(texts, lambda t: [self.BOS] + self.spm.EncodeAsIds(t) + [self.EOS])

    def __call__(self, cap):
        return self.encode_ids_with_bos_eos(cap)

    def __len__(self):
        return len(self.embs)

