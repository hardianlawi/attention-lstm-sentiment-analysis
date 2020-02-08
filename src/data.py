from tensorflow.keras.datasets import imdb

PAD_ID = 0
START_ID = 1
OOV_ID = 2
INDEX_OFFSET = 2
WORD2ID = imdb.get_word_index()
ID2WORD = {i + INDEX_OFFSET: word for word, i in WORD2ID.items()}
ID2WORD[PAD_ID] = "<PAD>"
ID2WORD[START_ID] = "<START>"
ID2WORD[OOV_ID] = "<OOV>"


def get_data(vocab_size):
    (X_train, y_train), (X_test, y_test) = imdb.load_data(
        num_words=vocab_size, start_char=START_ID, oov_char=OOV_ID
    )
    str_X_train = list(
        map(lambda x: " ".join([ID2WORD[c] for c in x if c != PAD_ID]), X_train)
    )
    str_X_test = list(
        map(lambda x: " ".join([ID2WORD[c] for c in x if c != PAD_ID]), X_test)
    )
    return (str_X_train, y_train), (str_X_test, y_test)
