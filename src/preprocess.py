import json
from typing import List

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json


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

    def to_json(self):
        self._tokenizer.to_json()

    def load_tokenizer(self, json_path):
        with open(json_path, "r") as f:
            self._tokenizer = tokenizer_from_json(json.loads(f))
