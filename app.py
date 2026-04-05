from flask import Flask, jsonify
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "API is running"

@app.route("/predict")
def predict():
    with open("result.json") as f:
        return jsonify(json.load(f))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)