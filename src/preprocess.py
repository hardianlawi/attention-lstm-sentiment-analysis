import os
import pickle
from typing import List

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer


class Preprocessor(object):
    def __init__(self, maxlen, vocab_size, oov_token):
        self.__oov_token = oov_token
        self._tokenizer = Tokenizer(oov_token=oov_token)
        self._vocab_size = vocab_size
        self._maxlen = maxlen

    def fit_on_texts(self, texts):
        self._tokenizer.fit_on_texts(texts)
        self._word2id = dict(
            zip(self._tokenizer.index_word.values(), self._tokenizer.index_word.keys())
        )
        self.__oov_id = self._word2id[self.__oov_token]

    def transform(self, sentences: List[str]):
        sequences = self._tokenizer.texts_to_sequences(sentences)
        padded_sequences = sequence.pad_sequences(sequences, maxlen=self._maxlen)
        padded_sequences = self._filter_oov(padded_sequences)
        return padded_sequences

    def _filter_oov(self, X):
        X = X.copy()
        X[X >= self._vocab_size] = self.__oov_id
        return X

    def save(self, path):
        _make_dir_if_not_exists(path)
        with open(path, "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as handle:
            return pickle.load(handle)


def _make_dir_if_not_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
