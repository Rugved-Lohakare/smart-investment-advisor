import os
import re
import pandas as pd
import requests
from dotenv import load_dotenv
from transformers import pipeline as hf_pipeline

load_dotenv()


def update_sentiment():
    print("=== SENTIMENT UPDATE STARTED ===")

    API_KEY = os.getenv("NEWS_API_KEY")
    if not API_KEY:
        raise ValueError("NEWS_API_KEY not set in .env file")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "Reliance Industries OR Reliance stock OR Reliance shares OR RIL OR Jio Ambani OR Reliance OR Jio",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data     = response.json()
        articles = data.get("articles", [])
    except Exception as e:
        print(f"News API failed: {e}")
        articles = []

    if not articles:
        print("No articles returned. Keeping existing sentiment_score.csv unchanged.")
        return

    df_news = pd.DataFrame(articles)

    for col in ["publishedAt", "title", "description"]:
        if col not in df_news.columns:
            df_news[col] = ""

    df_news = df_news[["publishedAt", "title", "description"]]
    df_news["Date"] = pd.to_datetime(df_news["publishedAt"], errors="coerce").dt.date
    df_news["Date"] = pd.to_datetime(df_news["Date"])
    df_news["text"] = df_news["title"].fillna("") + " " + df_news["description"].fillna("")

    keywords = [
    "reliance",
    "ril",
    "jio",
    "ambani",
    "reliance industries",
    "reliance stock"
]
    pattern  = "|".join(keywords)
    df_news  = df_news[df_news["text"].str.contains(pattern, case=False, na=False)]

    if df_news.empty:
        print("No relevant articles after keyword filter. Skipping update.")
        return

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"https?\s*:\s*//\S+", "", text)
        text = re.sub(r"[^a-z\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    df_news["clean_text"] = df_news["text"].apply(clean_text)

    # FIX ISSUE 1 — do NOT pass truncation/max_length to pipeline()
    # Pass truncation=True inside the inference call instead
    print("Loading FinBERT model...")
    sentiment_model = hf_pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert"
    )

    def finbert_score(text):
        try:
            # FIX ISSUE 1 — truncation goes here, on the inference call
            result = sentiment_model(
                str(text)[:512],
                truncation=True
            )[0]
            label = result["label"].lower()
            prob  = result["score"]
            if label == "positive":  return prob
            elif label == "negative": return -prob
            return 0.0
        except Exception as e:
            print(f"FinBERT inference error: {e}")
            return 0.0

    df_news["sentiment_score"] = df_news["clean_text"].apply(finbert_score)

    daily = (
        df_news.groupby("Date", as_index=False)["sentiment_score"]
        .mean()
        .rename(columns={"sentiment_score": "daily_sentiment_score"})
    )
    daily["sentiment_score"] = (daily["daily_sentiment_score"] + 1) / 2

    # FIX ISSUE 7 — sort by proper datetime, not string
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily = daily.sort_values("Date")
    daily["Date"] = daily["Date"].dt.date.astype(str)

    os.makedirs("data/processed", exist_ok=True)
    csv_path = "data/processed/sentiment_score.csv"

    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        existing["Date"] = pd.to_datetime(existing["Date"]).dt.date.astype(str)
        combined = pd.concat([existing, daily], ignore_index=True)
        combined["Date"] = pd.to_datetime(combined["Date"])
        combined = combined.drop_duplicates(subset=["Date"], keep="last")
        combined = combined.sort_values("Date")
        combined["Date"] = combined["Date"].dt.date.astype(str)
        combined.to_csv(csv_path, index=False)
    else:
        daily.to_csv(csv_path, index=False)

    print(f"Sentiment updated. Latest: {daily['Date'].max()}")
    print("=== SENTIMENT UPDATE DONE ===")


if __name__ == "__main__":
    update_sentiment()