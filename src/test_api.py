import json
import logging
from os.path import join

import click
import numpy as np
import requests

from src.logging import setup_logging

setup_logging()


def _read_json(path):
    with open(path, "r") as f:
        return json.load(f)


@click.command()
@click.argument("log_dir", type=str)
@click.option("--port", type=int, default=8080)
def main(log_dir, port):

    test_requests = _read_json(join(log_dir, "test_requests.json"))
    count = 0

    for request in test_requests:
        response = requests.post(
            f"http://localhost:{port}/predict", data=json.dumps(request)
        )
        expected_probabilities = np.array(request["probabilities"])

        flag = True
        if response.status_code == 200:
            predictions = response.json()["predictions"]
            for pred, exp_prob in zip(predictions, expected_probabilities):
                if not np.isclose(pred["probability"], exp_prob):
                    flag = False
                    logging.info(
                        "Prediction is different: ", pred["probability"], exp_prob
                    )

        if flag:
            count += 1

    logging.info(f"Success requests: {(count / len(test_requests) * 100):.2f}%")

    assert np.isclose(
        count / len(test_requests), 1.0
    ), "There are some failed test requests."


if __name__ == "__main__":
    main()
