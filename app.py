from flask import Flask, jsonify
from flask_cors import CORS
from stock import run_pipeline

app = Flask(__name__)
CORS(app)   

@app.route("/")
def home():
    return "API is running"

@app.route("/predict")
def predict():
    result = run_pipeline()
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    