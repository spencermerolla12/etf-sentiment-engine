# Macro Markets Sentiment Analytics Engine

**[Live Dashboard]**(https://etf-sentiment-spencer.streamlit.app/)

An automated, cloud-native NLP data pipeline and macroeconomic dashboard tracking the divergence between equity prices and aggregated media sentiment forecasts. 

## Project Objective & Economic Framework
* **The Goal:** To analyze whether high-velocity, aggregated financial news headlines serve as a leading, lagging, or coincident indicator for major index price movements (SPY, QQQ, DIA).
* **The Hypothesis:** Gathering a large volume of headline sentiment across multiple RSS feeds provides a real-time proxy for market mood, testing short-term behavioral finance volatility without the heavy compute overhead of full-text parsing.

## Architecture & Data Pipeline
This project operates on a 100% automated, zero-maintenance cloud architecture:
* **Data Sourcing:** Asset pricing and volume are pulled via the `yfinance` API. Market news is aggregated daily from free public RSS syndication (Google News, Yahoo Finance).
* **Sentiment Engine:** Parses 90+ daily headlines using the `FinBERT` Natural Language Processing transformer model.
* **Database:** Relational time-series data is stored in an **Aiven PostgreSQL** database, utilizing composite primary keys `(date, ticker)` to maintain data integrity.
* **Automation:** A **GitHub Actions** cron-job wakes up an Ubuntu server every weekday at 4:30 PM EST to execute the scraping script, score the sentiment, and upsert the database.
* **Frontend:** A multi-page interactive dashboard deployed via **Streamlit Community Cloud**.

## Local Installation
If you wish to run this pipeline locally on your own machine:

1. Clone the repository:
   `git clone https://github.com/spencermerolla12/etf-sentiment-engine.git`
2. Install dependencies:
   `pip install -r requirements.txt`
3. Set up your `.env` file with your database credentials:
   `DATABASE_URL=postgres://user:password@host:port/defaultdb`
4. Run the application:
   `python -m streamlit run 1_Dashboard.py`
