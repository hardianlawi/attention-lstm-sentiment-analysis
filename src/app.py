import argparse
from copy import copy

from sanic import Sanic, response
from src.app_utils import generate_predictions, load, preprocess, validate_request

app = Sanic()

MODEL_TYPE = None


@app.route("/predict", methods=["POST"])
def predict(request):
    data = {"model_type": MODEL_TYPE, "predictions": []}

    request_json = request.json
    validate_request(request_json)

    sentences = copy(request_json["sentences"])
    for i, sentence in enumerate(sentences):
        try:
            sentence = str(sentence)
        except ValueError:
            sentences[i] = ""

    preprocessed_sentences = preprocess(sentences)
    probabilities, sentiments = generate_predictions(preprocessed_sentences)

    for sentence, prob, sent in zip(
        request_json["sentences"], probabilities, sentiments
    ):
        data["predictions"].append(
            {"sentence": sentence, "probability": prob, "sentiment": sent,}
        )

    return response.json(data)


@app.route("/", methods=["GET"])
async def test(request):
    return response.json({"hello": "world"})


@app.route("/ping", methods=["GET"])
async def ping(request):
    return response.json({"ping": "ok"})


@app.route("/healthz", methods=["GET"])
async def healthz(request):
    return response.json({"health": "ok"})


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=str)
    parser.add_argument("model_type", type=str)

    args = parser.parse_args()
    MODEL_TYPE = args.model_type

    load(args.log_dir)

    app.run(host="0.0.0.0", port=8080)
