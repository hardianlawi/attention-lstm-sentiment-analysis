import argparse
from os.path import join

from sanic import Sanic, response

from src.models import load_model
from src.preprocess import Preprocessor

app = Sanic()
PREPROCESSOR = None
MODEL = None
MODEL_TYPE = None


def load(log_dir: str, model_type: str):
    global MODEL, PREPROCESSOR, MODEL_TYPE
    MODEL = load_model(join(log_dir, "saved_model"))
    PREPROCESSOR = Preprocessor.load(join(log_dir, "preprocessor.pkl"))
    MODEL_TYPE = model_type
    assert MODEL is not None, "Model has not been properly loaded"
    assert PREPROCESSOR is not None, "Preprocessor has not been properly loaded"


def validate_request(request_json):
    if "sentences" not in request_json:
        raise ValueError("Could not find `sentences` in the request")
    sentences = request_json["sentences"]
    if not isinstance(sentences, list):
        raise ValueError("`sentences` has to be a list")
    for sentence in sentences:
        try:
            sentence = str(sentence)
        except ValueError:
            raise ValueError(
                "one of the value in `sentences` could not be converted to string."
            )


@app.route("/predict", methods=["POST"])
def predict(request):
    data = {
        "success": False,
        "model_type": MODEL_TYPE,
    }

    request_json = request.json
    validate_request(request_json)
    sentences = request_json["sentences"]

    preprocessed_sentences = PREPROCESSOR.transform(sentences)
    probabilities = MODEL(preprocessed_sentences)
    sentiments = (probabilities > 0.5).astype(int).squeeze().tolist()

    data["success"] = True
    data["probabilities"] = probabilities
    data["sentiments"] = sentiments

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
    load(args.log_dir, args.model_type)

    app.run(host="0.0.0.0", port=8080)
