from os.path import join

import click

from src.data import ID2WORD, OOV_ID, get_data
from src.models import get_model
from src.preprocess import Preprocessor


def _preprocess_raw_data(X):
    """Preprocess raw data to avoid training serving skew"""
    return list(map(lambda x: [ID2WORD[c] for c in x if c != 0], X))


@click.command()
@click.argument("log_dir", type=str)
@click.argument("model_type", type=str)
@click.option("--vocab_size", type=int, default=5000)
@click.option("--emb_size", type=int, default=32)
@click.option("--batch_size", type=int, default=64)
@click.option("--epochs", type=int, default=3)
@click.option("--maxlen", type=int, default=500)
@click.option("--min_acc", type=float, default=0.85)
def main(
    log_dir, model_type, vocab_size, emb_size, batch_size, epochs, maxlen, min_acc
):
    (X_train, y_train), (X_test, y_test) = get_data(vocab_size)

    preprocessor = Preprocessor(maxlen=maxlen, oov_token=ID2WORD[OOV_ID])
    preprocessor.fit_on_texts(X_train + X_test)

    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Define model
    model = get_model(model_type, maxlen, vocab_size, emb_size)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(
        X_train[batch_size:],
        y_train[batch_size:],
        validation_data=(X_train[:batch_size], y_train[:batch_size]),
        batch_size=batch_size,
        epochs=epochs,
    )

    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test accuracy:", score[1])
    assert score[1] > min_acc, f"score doesnt meet the minimum threshold {min_acc}"

    preprocessor.save(join(log_dir, "preprocessor.pkl"))
    model.save(join(log_dir, "saved_model"))


if __name__ == "__main__":
    main()
