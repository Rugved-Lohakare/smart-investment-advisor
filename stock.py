def run_pipeline():
    # CLEANING PHASE

    import pandas as pd
    import numpy as np
    import yfinance as yf
    from datetime import datetime, timedelta
    import requests
    import re
    from transformers import pipeline
    from sklearn.ensemble import RandomForestClassifier


    pd.set_option('display.max_rows', 25)
    pd.set_option('display.max_columns', 23)

    df_me = pd.read_csv('data/processed/final_data_1.csv')
    df_me = df_me.copy()

    df_me["Date"] = pd.to_datetime(df_me["Date"])

    print(df_me)

    valid_columns = [
        "Symbol","Series","Date","Prev Close","Open Price","High Price",
        "Low Price","Last Price","Close Price","Average Price",
        "Total Traded Quantity","Turnover ₹","No. of Trades",
        "Deliverable Qty","% Dly Qt to Traded Qty"
    ]

    df_me = df_me[[col for col in valid_columns if col in df_me.columns]]

    df_me.to_csv("data/processed/final_data_1.csv", index=False)

    cols = ["Date","Open Price","High Price","Low Price","Close Price","Total Traded Quantity"]
    df_me[cols] = df_me[cols].replace(',', '', regex=True)

    df_me[["Open Price","High Price","Low Price","Close Price","Total Traded Quantity"]] = \
    df_me[["Open Price","High Price","Low Price","Close Price","Total Traded Quantity"]].astype(float)

    print(df_me)

    print(df_me.info())

    # YFINANCE UPDATE

    last_date = df_me["Date"].max()
    start_date = last_date + timedelta(days=1)
    end_date = datetime.today()

    if start_date <= end_date:
        new_data = yf.download("RELIANCE.NS", start=start_date, end=end_date, progress=False)
    else:
        new_data = pd.DataFrame()

    if new_data.empty:
        print("up to date.")
    else:
        if isinstance(new_data.columns, pd.MultiIndex):
            new_data.columns = new_data.columns.get_level_values(0)

        new_data = new_data.reset_index()

        new_data = new_data.rename(columns={
            "Open": "Open Price",
            "High": "High Price",
            "Low": "Low Price",
            "Close": "Close Price",
            "Volume": "Total Traded Quantity"
        })

        new_data = new_data[[
            "Date","Open Price","High Price","Low Price","Close Price","Total Traded Quantity"
        ]]

        df_me = pd.concat([df_me, new_data], ignore_index=True)
        df_me = df_me.drop_duplicates(subset=["Date"])
        df_me = df_me.sort_values("Date")

        print("New data appended successfully.")

    df_me.to_csv("data/processed/final_data_1.csv", index=False)

    print("Dataset updated till:", df_me["Date"].max())
    df_me.info()

    print(df_me)

    # FEATURE ENGINEERING

    def calculate_rsi(series, period=14):
        difference = series.diff()
        gain = difference.clip(lower=0)
        loss = -difference.clip(upper=0)

        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100/(1+rs))
        return rsi

    df_me['RSI_14'] = calculate_rsi(df_me['Close Price'])

    df_me['EMA_12'] = df_me['Close Price'].ewm(span=12, adjust=False).mean()
    df_me['EMA_26'] = df_me['Close Price'].ewm(span=26, adjust=False).mean()

    df_me['MACD'] = df_me['EMA_12'] - df_me['EMA_26']
    df_me['MACD_Signal'] = df_me['MACD'].ewm(span=9, adjust=False).mean()
    df_me['MACD_Hist'] = df_me['MACD'] - df_me['MACD_Signal']

    df_me["Volume_Change"] = df_me["Total Traded Quantity"].pct_change()

    df_me['future_return_3d'] = (
        df_me['Close Price'].shift(-3) - df_me['Close Price']
    ) / df_me['Close Price']

    df_me['Target'] = (df_me['future_return_3d'] > 0).astype(int)

    df_me["ret_1"] = df_me["Close Price"].pct_change()

    rolling_mean_20 = df_me["ret_1"].rolling(20).mean()
    rolling_std_20 = df_me["ret_1"].rolling(20).std()

    df_me["ret_1_z"] = (df_me["ret_1"] - rolling_mean_20) / rolling_std_20

    df_me["SMA_20"] = df_me["Close Price"].rolling(20).mean()
    df_me["dist_sma20"] = (
        df_me["Close Price"] - df_me["SMA_20"]
    ) / df_me["SMA_20"]

    df_me["vol_10"] = df_me["ret_1"].rolling(10).std()
    df_me["vol_20"] = df_me["ret_1"].rolling(20).std()

    df_me["vol_ratio"] = df_me["vol_10"] / df_me["vol_20"]

    df_me["Volume_Change"] = df_me["Volume_Change"].replace([np.inf, -np.inf], np.nan)

    lower = df_me["Volume_Change"].quantile(0.01)
    upper = df_me["Volume_Change"].quantile(0.99)
    df_me["Volume_Change"] = df_me["Volume_Change"].clip(lower, upper)

    df_me = df_me.dropna(subset=["Date"])

    print(df_me)

    # TRAIN / TEST SPLIT

    df_train = df_me.iloc[:-3]
    df_future = df_me.tail(3)

    df_train = df_train.dropna(subset=[
        "RSI_14","EMA_12","EMA_26","MACD",
        "Volume_Change","dist_sma20","ret_1_z","vol_ratio","Target"
    ])

    print(df_train)

    # Sentiment analysis

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": "Reliance Industries OR Reliance stock OR Reliance shares OR RIL OR Jio Ambani",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": "f16b1fd1909946edaf7a8983eb98abed"
    }

    response = requests.get(url, params=params)
    print("Status Code:", response.status_code)

    # see if request was successfull , sometimes it might give error
    data = response.json()

    articles = data['articles']

    df_news = pd.DataFrame(articles)

    df_news = df_news[['publishedAt','title','description','source','url']]

    df_news["Date"] = pd.to_datetime(df_news["publishedAt"]).dt.date

    df_news['Date'] = pd.to_datetime(df_news['Date'])

    print(df_news)

    df_news.to_csv("data/raw/reliance_news.csv")

    df_news["text"] = (df_news["title"].fillna("") + " " +df_news["description"].fillna("") )


    keywords = ["reliance stock","reliance industries"]
    pattern = "|".join(keywords)

    df_news = df_news[df_news["text"].str.contains(pattern, case=False, na=False)]

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"https?\s*:\s*//\S+", "", text)   
        text = re.sub(r"[^a-z\s]", "", text)           
        text = re.sub(r"\s+", " ", text).strip()        
        return text

    df_news['clean_text'] = df_news['text'].apply(clean_text)

    df_news = df_news[["Date", "clean_text"]]

    sentiment_model = pipeline("sentiment-analysis",model="ProsusAI/finbert")

    def finbert_score(clean_text):
        result = sentiment_model(str(clean_text)[:512])[0]
        label = result["label"].lower()
        prob = result["score"]

        if label == "positive":
            return prob
        elif label == "negative":
            return -prob
        else:
            return 0.0
        
    df_news["sentiment_score"] = df_news["clean_text"].apply(finbert_score)

    df_news.to_csv("data/raw/reliance_news_cleaned.csv")

    df_news['Date'] = pd.to_datetime(df_news['Date'])

    daily_sentiment = (df_news.groupby('Date',as_index=False)['sentiment_score'].mean().rename(columns={'sentiment_score' : 'daily_sentiment_score'}))
    daily_sentiment.set_index('Date',inplace=True)
    daily_sentiment.sort_index(ascending=False,inplace=True)

    daily_sentiment["sentiment_score"] = (daily_sentiment["daily_sentiment_score"] + 1) / 2

    print(daily_sentiment)

    daily_sentiment.to_csv('data/processed/sentiment_score.csv')

    # ML Model Training Process
    df_me["Date"] = pd.to_datetime(df_me["Date"])
    df_me["year"] = df_me["Date"].dt.year

    df_train["Date"] = pd.to_datetime(df_train["Date"])
    df_train["year"] = df_train["Date"].dt.year

    df_future["Date"] = pd.to_datetime(df_future["Date"])
    df_future["year"] = df_future["Date"].dt.year

    y_col = "Target"


    features = [
        "RSI_14",
        "EMA_12",
        "EMA_26",
        "MACD",
        "Volume_Change",
        "dist_sma20",
        "ret_1_z",
        "vol_ratio"
    ]

    years = sorted(df_me["year"].unique())

    all_probs_rf = []
    all_future_returns_rf = []
    all_dates_rf = []

    df_train = df_train.dropna(subset=features + ["Target"])

    for i in range(3, len(years)):

        train_years = years[:i]
        test_year = years[i]

        train = df_train[df_train["year"].isin(train_years)]
        test = df_train[df_train["year"] == test_year]

        
        if train.empty or test.empty:
            print(f"Skipping year {test_year} due to insufficient data")
            continue

        X_train = train[features]
        X_test = test[features]

        y_train = train["Target"]
        y_test = test["Target"]

        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1
        )

        rf.fit(X_train, y_train)

        rf_probs = rf.predict_proba(X_test)[:, 1]

        # Store OUT-OF-SAMPLE results
        all_probs_rf.extend(rf_probs)
        all_future_returns_rf.extend(test["future_return_3d"])
        all_dates_rf.extend(test["Date"])

    rf.fit(df_train[features], df_train["Target"])

    MIN_PROB = 0.1740048587640126
    MAX_PROB = 0.7209117413904984

    def compute_trend_score(prob):
        score = (prob - MIN_PROB) / (MAX_PROB - MIN_PROB)
        score = np.clip(score, 0, 1)
        return score

    latest = df_me.tail(1)

    X_latest = latest[features]

    latest_prob = rf.predict_proba(X_latest)[:,1][0]

    latest_trend_score = compute_trend_score(latest_prob)

    print(latest)
    print("\nLatest Date:", latest["Date"].values[0])
    print("Latest Trend Score:", latest_trend_score)

    probs = np.array(all_probs_rf)
    future_returns = np.array(all_future_returns_rf)
    dates = pd.to_datetime(all_dates_rf)

    df_eval = pd.DataFrame({
        "Date": dates,
        "trend_score": probs,
        "future_ret_3": future_returns
    })

    latest_eval_row = pd.DataFrame({
        "Date": [latest["Date"].values[0]],
        "trend_score": [latest_prob],
        "future_ret_3": [np.nan]
    })

    df_eval = pd.concat([df_eval, latest_eval_row], ignore_index=True)

    print(df_eval)

    probs = np.array(all_probs_rf)
    trend_scores = compute_trend_score(probs)

    final_df = pd.DataFrame({
        "Date": df_eval["Date"].values,
        "trend_score": df_eval["trend_score"].values
    })

    latest_row = pd.DataFrame({
        "Date": [latest["Date"].values[0]],
        "trend_score": [latest_trend_score]
    })

    final_df = pd.concat([final_df, latest_row], ignore_index=True)

    final_df = final_df.drop_duplicates(subset=["Date"], keep="last")

    final_df.to_csv("data/processed/reliance_trend_score.csv", index=False)

    sentiment_df = daily_sentiment.reset_index()

    # Ensure datetime consistency
    df_me["Date"] = pd.to_datetime(df_me["Date"])
    sentiment_df["Date"] = pd.to_datetime(sentiment_df["Date"])
    final_df["Date"] = pd.to_datetime(final_df["Date"])

    # Rename Date → date (lowercase) for consistency
    df_me.rename(columns={"Date":"date"},inplace=True)
    sentiment_df.rename(columns={"Date": "date"}, inplace=True)
    final_df.rename(columns={"Date": "date"}, inplace=True)

    # Ensure date column is datetime after renaming
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
    final_df["date"] = pd.to_datetime(final_df["date"])

    master = final_df.merge(sentiment_df[["date", "sentiment_score"]],on="date",how="left")

    master = master.sort_values("date").reset_index(drop=True)

    master = master.merge(df_me[["date", "Close Price"]],on="date",how="left")

    master["sentiment_score"] = master["sentiment_score"].fillna(0.5)

    last_date = master["date"].max()

    start_date = last_date - pd.Timedelta(days=30)

    master["sentiment_score"] = master["sentiment_score"].fillna(0.5)

    master["next_day_return"] = (
        master["Close Price"].shift(-1) / master["Close Price"] - 1
    )

    master = master[[
        "date", "trend_score", "sentiment_score",
        "Close Price", "next_day_return"
    ]]

    last_idx = master.index[-1]

    master = master[
        master["next_day_return"].notna() | (master.index == last_idx)
    ]

    master_last_30 = master[master["date"] >= start_date].copy()

    wt = 0.5
    ws = 0.5
    threshold = 0.55

    df_eval = master_last_30.copy()

    df_eval["final_score"] = (
        wt * df_eval["trend_score"] +
        ws * df_eval["sentiment_score"]
    )

    df_eval["signal"] = (
        df_eval["final_score"] > threshold
    ).astype(int)

    df_eval["actual"] = (
        df_eval["next_day_return"] > 0
    ).astype(int)

    df_final = master_last_30.copy()

    # 1️ Compute Final Hybrid Score
    wt = 0.5
    ws = 0.5

    df_final["final_score"] = (
        wt * df_final["trend_score"] +
        ws * df_final["sentiment_score"]
    )

    # 2️ Create Recommendation Signal
    df_final["recommendation"] = "HOLD"

    df_final.loc[df_final["final_score"] > 0.60, "recommendation"] = "BUY"
    df_final.loc[df_final["final_score"] < 0.54, "recommendation"] = "SELL"

    print("Signal Distribution:")
    print(df_final["recommendation"].value_counts())

    # 3️ Evaluate Classification
    df_final["actual_direction"] = np.where(
        df_final["next_day_return"] > 0,
        "UP",
        "DOWN"
    )

    conf_matrix = pd.crosstab(
        df_final["recommendation"],
        df_final["actual_direction"]
    )

    print("\nConfusion Matrix:")
    print(conf_matrix)

    # 4️ Long-Short Strategy Simulation
    df_final["strategy_return"] = 0.0

    # Long for BUY
    df_final.loc[
        df_final["recommendation"] == "BUY",
        "strategy_return"
    ] = df_final["next_day_return"]

    # Short for SELL
    df_final.loc[
        df_final["recommendation"] == "SELL",
        "strategy_return"
    ] = -df_final["next_day_return"]

    # HOLD remains 0

    # 5️ Performance Metrics
    df_final["cumulative_return"] = (
        1 + df_final["strategy_return"]
    ).cumprod()

    df_final[[
        "date",
        "trend_score",
        "sentiment_score",
        "final_score",
        "recommendation"
    ]].to_csv("data/processed/reliance_final_recommendations_v1.csv", index=False)

    print(df_final)

    latest_row = df_final.iloc[-1]

    return {
        "trend_score": float(latest_row["trend_score"]),
        "sentiment_score": float(latest_row["sentiment_score"]),
        "final_score": float(latest_row["final_score"]),
        "signal": str(latest_row["recommendation"])
    }

if __name__ == "__main__":
    print(run_pipeline())