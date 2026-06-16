import json
import os
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

RESULT_PATH = "result.json"


@app.route("/")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["GET"])
def predict():
    if not os.path.exists(RESULT_PATH):
        return jsonify({
            "error": "result.json not found. Run: python stock.py predict"
        }), 500

    try:
        with open(RESULT_PATH, "r") as f:
            result = json.load(f)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)