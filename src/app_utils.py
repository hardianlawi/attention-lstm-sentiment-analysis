import os
from typing import List

import numpy as np

from src.models import load_model
from src.preprocess import Preprocessor

PREPROCESSOR = None
MODEL = None


def load(log_dir: str):
    global MODEL, PREPROCESSOR
    model_path = os.path.join(log_dir, "model.h5")
    preprocessor_path = os.path.join(log_dir, "preprocessor.pkl")
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        raise Exception(
            "Please run `train.py` before building the app. Or make sure the needed files exist"
        )
    MODEL = load_model(model_path)
    PREPROCESSOR = Preprocessor.load(preprocessor_path)
    assert MODEL is not None, "Model has not been properly loaded"
    assert PREPROCESSOR is not None, "Preprocessor has not been properly loaded"


def preprocess(sentences: List[str]):
    if PREPROCESSOR is None:
        raise ValueError("Preprocessor is not properly loaded")
    return PREPROCESSOR.transform(sentences, return_len=True, return_oov_pctg=True)


def generate_predictions(preprocessed_sentences: np.ndarray):
    if MODEL is None:
        raise ValueError("Models are not properly loaded")
    probabilities = MODEL.predict(preprocessed_sentences).squeeze(axis=1)
    sentiments = (probabilities > 0.5).astype(int).squeeze().tolist()
    return probabilities, sentiments


def validate_request(request_json):
    if "sentences" not in request_json:
        raise ValueError("Could not find `sentences` in the request")
    sentences = request_json["sentences"]
    if not isinstance(sentences, list):
        raise ValueError("`sentences` has to be a list")
