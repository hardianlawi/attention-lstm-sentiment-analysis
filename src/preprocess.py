import pickle
from typing import List

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer


class Preprocessor(object):
    def __init__(self, maxlen, oov_token):
        self._tokenizer = Tokenizer(oov_token=oov_token)
        self._maxlen = maxlen

    def fit_on_texts(self, texts):
        self._tokenizer.fit_on_texts(texts)

    def transform(self, sentences: List[str]):
        sequences = self._tokenizer.texts_to_sequences(sentences)
        padded_sequences = sequence.pad_sequences(sequences, maxlen=self._maxlen)
        return padded_sequences

    def save(self, path):
        with open(path, "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as handle:
            return pickle.load(handle)
