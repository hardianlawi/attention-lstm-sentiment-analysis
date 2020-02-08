import json
import logging
import os
from os.path import join

import click
import numpy as np

from src.data import get_data
from src.models import get_model
from src.preprocess import Preprocessor


def _create_dir_if_not_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def _generate_test_requests(model, preprocessor, test_samples, num_sentences_per_req=3):
    logging.info("Generating test requests.")
    processed_test_samples = preprocessor.transform(test_samples)
    probabilities = model(processed_test_samples).numpy().squeeze().tolist()

    test_requests = []
    for x in range(0, len(test_samples), num_sentences_per_req):
        sentences = list(test_samples[x : x + num_sentences_per_req])
        probs = list(probabilities[x : x + num_sentences_per_req])
        test_requests.append({"sentences": sentences, "probabilities": probs})

    logging.info("Sample test request:", test_requests[0])

    return test_requests


@click.command()
@click.argument("log_dir", type=str)
@click.argument("model_type", type=str)
@click.option("--vocab_size", type=int, default=5000)
@click.option("--emb_size", type=int, default=32)
@click.option("--batch_size", type=int, default=64)
@click.option("--epochs", type=int, default=3)
@click.option("--maxlen", type=int, default=500)
@click.option("--min_acc", type=float, default=0.85)
@click.option("--num_samples", type=int, default=180)
@click.option("--oov_token", type=str, default="<OOV>")
def main(
    log_dir,
    model_type,
    vocab_size,
    emb_size,
    batch_size,
    epochs,
    maxlen,
    min_acc,
    num_samples,
    oov_token,
):
    _create_dir_if_not_exist(log_dir)

    (str_X_train, y_train), (str_X_val, y_val), (str_X_test, y_test) = get_data()
    test_samples = np.random.choice(str_X_test, num_samples)

    preprocessor = Preprocessor(
        maxlen=maxlen, vocab_size=vocab_size, oov_token=oov_token
    )
    preprocessor.fit_on_texts(str_X_train + str_X_val + str_X_test)

    X_train = preprocessor.transform(str_X_train)
    X_val = preprocessor.transform(str_X_val)
    X_test = preprocessor.transform(str_X_test)

    # Define model
    model = get_model(model_type, maxlen, vocab_size, emb_size)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
    )

    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test accuracy:", score[1])
    assert score[1] > min_acc, f"score doesnt meet the minimum threshold {min_acc}"

    test_requests = _generate_test_requests(model, preprocessor, test_samples)
    _save_json(join(log_dir, "test_requests.json"), test_requests)
    preprocessor.save(join(log_dir, "preprocessor.pkl"))
    model.save(join(log_dir, "saved_model"))


if __name__ == "__main__":
    main()
