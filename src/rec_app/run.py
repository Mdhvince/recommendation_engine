import configparser
from pathlib import Path

from flask import Flask, request, jsonify

from src.recommender import Recommender

app = Flask(__name__)

config = configparser.ConfigParser(inline_comment_prefixes="#")
config.read(Path(__file__).parent.parent.parent / "config.ini")
recommender = Recommender(config)


@app.route("/predict", methods=["POST"])  # endpoint
def predict():
    if request.method == "POST":
        user = request.args.get("user")
        item = request.args.get("item")
        prediction = recommender.predict(user, item)
        msg = f"User {user} or item {item} not found in dataset" if prediction == -1 else ""
        return jsonify({"prediction": prediction, "message": msg})


if __name__ == "__main__":
    app.run(debug=True)

