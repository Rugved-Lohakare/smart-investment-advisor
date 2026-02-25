#  Smart Investment Advisor

An AI-powered stock investment advisory system that combines historical stock data, news sentiment analysis, and machine learning to generate intelligent investment recommendations.

---

##  Project Overview

Smart Investment Advisor is a full end-to-end machine learning pipeline that:

- Collects historical stock data (2011–2026)
- Fetches real-time financial news using NewsAPI
- Performs NLP-based sentiment analysis using FinBERT
- Generates daily sentiment scores
- Creates trend-based features
- Trains a machine learning model for stock movement prediction
- Produces final investment recommendations

This project demonstrates data engineering, NLP, feature engineering, and ML model development in a financial domain.

---

##  System Pipeline

1. **Data Collection**
   - Historical stock data (CSV files)
   - Financial news using NewsAPI

2. **Data Cleaning & Preprocessing**
   - News filtering & relevance scoring
   - Date alignment with stock data
   - Missing value handling

3. **Sentiment Analysis**
   - FinBERT-based sentiment scoring
   - Daily aggregated sentiment score

4. **Feature Engineering**
   - Trend score calculation
   - Sentiment normalization
   - ML-ready dataset creation

5. **Machine Learning Model**
   - Stock movement classification
   - Probability-based prediction
   - Performance evaluation

6. **Recommendation Engine**
   - Buy / Hold / Sell logic
   - Final recommendation CSV output

---

##  Repository Structure

