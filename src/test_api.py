import json
from os.path import join

import click
import numpy as np
import requests


def _read_json(path):
    with open(path, "r") as f:
        return json.load(f)


@click.command()
@click.argument("log_dir", type=str)
def main(log_dir):

    test_requests = _read_json(join(log_dir, "test_requests.json"))
    count = 0

    for request in test_requests:
        response = requests.post(
            "http://localhost:8080/predict", data=json.dumps(request)
        )
        expected_probabilities = np.array(request["probabilities"])

        if response.status_code == 200:
            probabilities = np.array(response.json()["probabilities"])
            if np.all(np.isclose(expected_probabilities, probabilities)):
                count += 1

    print("Success requests:", count / len(test_requests))

    assert np.isclose(
        count / len(test_requests), 1.0
    ), "There are some failed test requests."


if __name__ == "__main__":
    main()
