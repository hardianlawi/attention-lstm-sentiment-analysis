import argparse
from copy import copy

from sanic import Sanic, response
from src.app_utils import generate_predictions, load, preprocess, validate_request
from src.logging import setup_logging

setup_logging()

app = Sanic(__name__)

MODEL_TYPE = None


@app.route("/predict", methods=["POST"])
def predict(request):
    resp = {"model_type": MODEL_TYPE, "predictions": []}

    request_json = request.json
    validate_request(request_json)

    sentences = request_json["sentences"]
    preprocessed_seqs, seqs_len, seqs_oov_pctgs = preprocess(sentences)
    probabilities, sentiments = generate_predictions(preprocessed_seqs)

    for sentence, prob, sent, ps, seq_len, seq_oov_pctg in zip(
        sentences,
        probabilities,
        sentiments,
        preprocessed_seqs,
        seqs_len,
        seqs_oov_pctgs,
    ):
        data = {"sentence": sentence}
        if seq_len <= 2:
            data.update(
                {
                    "message": "model not generating prediction due to sequence too short "
                }
            )
        elif seq_oov_pctg > 0.5:
            data.update(
                {
                    "message": "model not generating prediction due to high pctg of OOV tokens"
                }
            )
        else:
            data.update({"probability": float(prob), "sentiment": int(sent)})

        resp["predictions"].append(data)

    return response.json(resp)


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
