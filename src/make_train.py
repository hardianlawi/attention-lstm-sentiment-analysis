import click
from tensorflow.keras.datasets import imdb

from src.models import get_model
from src.preprocess import Preprocessor

WORD2ID = imdb.get_word_index()
ID2WORD = dict(zip(WORD2ID.values(), WORD2ID.keys()))


def _preprocess_raw_data(X):
    """Preprocess raw data to avoid training serving skew"""
    return list(map(lambda x: [ID2WORD[c] for c in x if c != 0], X))


@click.command()
@click.argument("log_dir", type=str)
@click.argument("model_name", type=str)
@click.option("--vocab_size", type=int, default=5000)
@click.option("--emb_size", type=int, default=32)
@click.option("--batch_size", type=int, default=64)
@click.option("--epochs", type=int, default=3)
@click.option("--maxlen", type=int, default=500)
def main(log_dir, model_name, vocab_size, emb_size, batch_size, epochs, maxlen):
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

    X_train = _preprocess_raw_data(X_train)
    X_test = _preprocess_raw_data(X_test)

    preprocessor = Preprocessor(maxlen=maxlen)
    preprocessor.fit_on_texts(X_train + X_test)

    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Define model
    model = get_model(model_name, maxlen, vocab_size, emb_size)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(
        X_train[batch_size:],
        y_train[batch_size:],
        validation_data=(X_train[:batch_size], y_train[:batch_size]),
        batch_size=batch_size,
        epochs=epochs,
    )

    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Test accuracy:", scores[1])


if __name__ == "__main__":
    main()
