from flask import Flask, jsonify
from flask_cors import CORS
import json
import pandas as pd 

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "API is running"

@app.route("/predict")
def predict():
    import pandas as pd

@app.route("/predict")
def predict():
    # Load prediction result
    with open("result.json") as f:
        result = json.load(f)

    # Load historical price data
    df = pd.read_csv("data/processed/final_data_1.csv")

    # Take last 7 days
    df = df.tail(7)

    # Prepare chart data
    dates = df["Date"].tolist()
    prices = df["Close Price"].tolist()

    result["dates"] = dates
    result["prices"] = prices

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)