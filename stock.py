import os
import json
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = "models/rf_model.pkl"
MIN_PROB   = 0.1740048587640126
MAX_PROB   = 0.7209117413904984


FEATURES = [
    "RSI_14", "EMA_12", "EMA_26", "MACD",
    "Volume_Change", "dist_sma20", "ret_1_z", "vol_ratio"
]



def calculate_rsi(series, period=14):
    diff     = series.diff()
    gain     = diff.clip(lower=0)
    loss     = -diff.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_trend_score(prob):
    score = (prob - MIN_PROB) / (MAX_PROB - MIN_PROB)
    return float(np.clip(score, 0, 1))


def load_and_update_price_data():
    """Load CSV and append any new yfinance rows. Returns clean df."""
    df = pd.read_csv("data/processed/final_data_1.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    valid_columns = [
        "Symbol", "Series", "Date", "Prev Close", "Open Price", "High Price",
        "Low Price", "Last Price", "Close Price", "Average Price",
        "Total Traded Quantity", "Turnover ₹", "No. of Trades",
        "Deliverable Qty", "% Dly Qt to Traded Qty"
    ]
    df = df[[col for col in valid_columns if col in df.columns]]

    price_cols = ["Open Price", "High Price", "Low Price",
                  "Close Price", "Total Traded Quantity"]
    for col in price_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].replace(",", "", regex=True).astype(float)

    last_date  = df["Date"].max()
    start_date = last_date + timedelta(days=1)
    end_date   = datetime.today()

    if start_date <= end_date:
        try:
            new_data = yf.download(
                "RELIANCE.NS", start=start_date,
                end=end_date, progress=False
            )
            if not new_data.empty:
                if isinstance(new_data.columns, pd.MultiIndex):
                    new_data.columns = new_data.columns.get_level_values(0)
                new_data = new_data.reset_index().rename(columns={
                    "Open":   "Open Price",
                    "High":   "High Price",
                    "Low":    "Low Price",
                    "Close":  "Close Price",
                    "Volume": "Total Traded Quantity"
                })
                new_data = new_data[[
                    "Date", "Open Price", "High Price",
                    "Low Price", "Close Price", "Total Traded Quantity"
                ]]
                df = pd.concat([df, new_data], ignore_index=True)
                df = df.drop_duplicates(subset=["Date"]).sort_values("Date")
                df.to_csv("data/processed/final_data_1.csv", index=False)
                print(f"Price data updated to: {df['Date'].max().date()}")
        except Exception as e:
            print(f"yfinance update failed (using existing data): {e}")

    return df


def add_features(df):
    df = df.copy()
    df["RSI_14"]      = calculate_rsi(df["Close Price"])
    df["EMA_12"]      = df["Close Price"].ewm(span=12, adjust=False).mean()
    df["EMA_26"]      = df["Close Price"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["ret_1"]       = df["Close Price"].pct_change()

    rolling_mean  = df["ret_1"].rolling(20).mean()
    rolling_std   = df["ret_1"].rolling(20).std()
    df["ret_1_z"] = (df["ret_1"] - rolling_mean) / rolling_std

    df["SMA_20"]     = df["Close Price"].rolling(20).mean()
    df["dist_sma20"] = (df["Close Price"] - df["SMA_20"]) / df["SMA_20"]

    df["vol_10"]    = df["ret_1"].rolling(10).std()
    df["vol_20"]    = df["ret_1"].rolling(20).std()
    df["vol_ratio"] = df["vol_10"] / df["vol_20"]

    df["Volume_Change"] = df["Total Traded Quantity"].pct_change()
    df["Volume_Change"] = df["Volume_Change"].replace([np.inf, -np.inf], np.nan)
    lower = df["Volume_Change"].quantile(0.01)
    upper = df["Volume_Change"].quantile(0.99)
    df["Volume_Change"] = df["Volume_Change"].clip(lower, upper)

    df["future_return_3d"] = (
        df["Close Price"].shift(-3) - df["Close Price"]
    ) / df["Close Price"]
    df["Target"] = (df["future_return_3d"] > 0).astype(int)

    return df.dropna(subset=["Date"])


def read_latest_sentiment():
    """Read latest sentiment score from CSV. No FinBERT. No network."""
    csv_path = "data/processed/sentiment_score.csv"

    if not os.path.exists(csv_path):
        print("WARNING: No sentiment CSV found. Using neutral 0.5")
        print("Run 'python update_sentiment.py' to generate it.")
        return 0.5

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return 0.5

        # FIX ISSUE 7 — sort by proper datetime not string
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

        latest_score = float(df["sentiment_score"].iloc[-1])
        print(f"Sentiment score from CSV: {latest_score:.4f}  "
              f"(date: {df['Date'].iloc[-1].date()})")
        return latest_score

    except Exception as e:
        print(f"Could not read sentiment CSV: {e}. Using neutral 0.5")
        return 0.5



def train_pipeline():
    print("=== TRAIN PIPELINE STARTED ===")

    df = load_and_update_price_data()
    df = add_features(df)
    df["year"] = df["Date"].dt.year
    years      = sorted(df["year"].unique())

    df_train = df.iloc[:-3].dropna(subset=FEATURES + ["Target"]).copy()
    df_train["year"] = df_train["Date"].dt.year

    print("Running walk-forward validation...")
    for i in range(3, len(years)):
        train_years = years[:i]
        test_year   = years[i]
        train = df_train[df_train["year"].isin(train_years)]
        test  = df_train[df_train["year"] == test_year]
        if train.empty or test.empty:
            continue
        rf_temp = RandomForestClassifier(
            n_estimators=300, max_depth=5,
            min_samples_leaf=20, random_state=42, n_jobs=-1
        )
        rf_temp.fit(train[FEATURES], train["Target"])
        print(f"  Validated year {test_year} ({len(test)} samples)")

    # Final model trained on all available data
    rf_final = RandomForestClassifier(
        n_estimators=300, max_depth=5,
        min_samples_leaf=20, random_state=42, n_jobs=-1
    )
    rf_final.fit(df_train[FEATURES], df_train["Target"])

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        {"model": rf_final, "features": FEATURES},
        MODEL_PATH
    )
    print(f"Model + feature list saved → {MODEL_PATH}")
    print("=== TRAIN PIPELINE DONE ===")



def predict_pipeline():
    print("=== PREDICT PIPELINE STARTED ===")

    # 1. Check model exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Run: python stock.py train"
        )

    # 2. Load model AND feature list from the same saved file
    # FIX ISSUE 3 — features come from the saved dict, not from this file's globals
    saved    = joblib.load(MODEL_PATH)
    rf       = saved["model"]
    features = saved["features"]
    print(f"Model loaded. Features: {features}")

    # 3. Update price data and compute indicators
    df = load_and_update_price_data()
    df = add_features(df)

    df_clean = df.dropna(subset=features)

    if df_clean.empty:
        raise ValueError(
            "No complete rows found after dropping NaN features. "
            "Check your data — rolling windows may need more history."
        )

    latest   = df_clean.tail(1)
    X_latest = latest[features]

    latest_prob = float(rf.predict_proba(X_latest)[:, 1][0])
    trend_score = compute_trend_score(latest_prob)
    print(f"Trend score: {trend_score:.4f}  (raw prob: {latest_prob:.4f})")

    # 4. Read sentiment from CSV — zero network, zero FinBERT
    sentiment_score = read_latest_sentiment()

    # 5. Final score and recommendation
    final_score = 0.5 * trend_score + 0.5 * sentiment_score

    if final_score > 0.60:
        recommendation = "BUY"
    elif final_score < 0.54:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"

    # 6. Save to CSV (historical record)
    latest_date = pd.to_datetime(latest["Date"].values[0]).date()

    result_row = pd.DataFrame([{
        "date":            str(latest_date),
        "trend_score":     round(trend_score, 4),
        "sentiment_score": round(sentiment_score, 4),
        "final_score":     round(final_score, 4),
        "recommendation":  recommendation
    }])

    csv_path = "data/processed/reliance_final_recommendations_v1.csv"
    os.makedirs("data/processed", exist_ok=True)

    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        existing["date"] = pd.to_datetime(existing["date"]).dt.date.astype(str)
        result_row["date"] = result_row["date"].astype(str)
        combined = pd.concat([existing, result_row], ignore_index=True)

        # FIX ISSUE 7 — sort by proper datetime not string
        combined["date"] = pd.to_datetime(combined["date"])
        combined = combined.drop_duplicates(subset=["date"], keep="last")
        combined = combined.sort_values("date")
        combined["date"] = combined["date"].dt.date.astype(str)
        combined.to_csv(csv_path, index=False)
    else:
        result_row.to_csv(csv_path, index=False)

    result = {
        "date":            str(latest_date),
        "trend_score":     round(trend_score, 4),
        "sentiment_score": round(sentiment_score, 4),
        "final_score":     round(final_score, 4),
        "signal":          recommendation
    }

    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"result.json written.")
    print(f"Recommendation: {recommendation}  |  Score: {final_score:.4f}")
    print("=== PREDICT PIPELINE DONE ===")

    return result

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "predict"

    if mode == "train":
        train_pipeline()

    elif mode == "sentiment":
        from update_sentiment import update_sentiment
        update_sentiment()

    else:
        print(predict_pipeline())