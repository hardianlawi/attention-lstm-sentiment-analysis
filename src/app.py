from sanic import Sanic, response

from src.models import load_model
from src.preprocess import load_preprocessor

app = Sanic()
preprocessor = None
model = None


def load():
    return


@app.route("/predict", methods=["POST"])
def predict(request):
    data = {"success": False}
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
    load()
    app.run(host="0.0.0.0", port=8080)
