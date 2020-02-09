import os
import pickle
from typing import List

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer


class Preprocessor(object):
    def __init__(self, maxlen, vocab_size, oov_token):
        self.__oov_token = oov_token
        self._tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
        self._vocab_size = vocab_size
        self._maxlen = maxlen

    @property
    def word2id(self):
        return self._tokenizer.word_index

    @property
    def id2word(self):
        return self._tokenizer.index_word

    def fit_on_texts(self, texts):
        self._tokenizer.fit_on_texts(texts)
        self.__oov_id = self.word2id[self.__oov_token]

    def transform(
        self,
        sentences: List[str],
        return_len: bool = False,
        return_oov_pctg: bool = False,
    ):
        sequences = self._tokenizer.texts_to_sequences(sentences)
        padded_sequences = sequence.pad_sequences(sequences, maxlen=self._maxlen)
        outputs = (padded_sequences,)

        if return_len:
            outputs += (list(map(len, sequences)),)

        if return_oov_pctg:
            outputs += (
                list(
                    map(
                        lambda x: len(["" for t in x if t == self.__oov_id]) / len(x),
                        sequences,
                    )
                ),
            )

        if len(outputs) == 1:
            return outputs[0]

        return outputs

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
