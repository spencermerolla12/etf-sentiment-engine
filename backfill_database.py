"""Backfill the daily_sentiment table with recent index ETF trading days.

This script is meant to seed the dashboard with enough historical data to make
the trend chart useful immediately. It uses yfinance for ETF price/volume data,
Google News headlines for date-specific index text, FinBERT for sentiment, and
upserts the result into the Aiven PostgreSQL database.
"""

from __future__ import annotations

import os
import time
from datetime import date, timedelta
from typing import Any
from urllib.parse import quote

import feedparser
import psycopg2
import yfinance as yf
from dotenv import load_dotenv

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
load_dotenv()

from transformers import pipeline
from transformers.utils import logging as transformers_logging


DATABASE_URL = os.getenv("DATABASE_URL")
TICKERS = [
    {"symbol": "SPY", "query": "S&P 500 OR SPY ETF"},
    {"symbol": "QQQ", "query": "NASDAQ OR QQQ ETF"},
    {"symbol": "DIA", "query": "Dow Jones OR DIA ETF"},
]
BACKFILL_SYMBOLS = {"QQQ", "DIA"}
TRADING_DAYS_TO_BACKFILL = 60
LOOKBACK_CALENDAR_DAYS = 120
MAX_HEADLINES_PER_DAY = 10
REQUEST_PAUSE_SECONDS = 0.75
FINBERT_MODEL = "ProsusAI/finbert"
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"

CREATE_DAILY_SENTIMENT_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS daily_sentiment (
    date DATE NOT NULL,
    ticker VARCHAR(10) NOT NULL DEFAULT 'SPY',
    spy_close NUMERIC,
    spy_volume BIGINT,
    sentiment_score NUMERIC,
    PRIMARY KEY (date, ticker)
);
"""

MIGRATE_DAILY_SENTIMENT_PRIMARY_KEY_SQL = """
UPDATE daily_sentiment
SET ticker = 'SPY'
WHERE ticker IS NULL;

ALTER TABLE daily_sentiment
ALTER COLUMN ticker SET NOT NULL;

DO $$
DECLARE
    current_primary_key_name TEXT;
    current_primary_key_columns TEXT[];
BEGIN
    SELECT
        constraint_name,
        columns
    INTO
        current_primary_key_name,
        current_primary_key_columns
    FROM (
        SELECT
            primary_constraint.conname AS constraint_name,
            array_agg(attribute.attname ORDER BY key_column.ordinality) AS columns
        FROM pg_constraint AS primary_constraint
        CROSS JOIN LATERAL unnest(primary_constraint.conkey)
            WITH ORDINALITY AS key_column(attnum, ordinality)
        JOIN pg_attribute AS attribute
            ON attribute.attrelid = primary_constraint.conrelid
            AND attribute.attnum = key_column.attnum
        WHERE
            primary_constraint.conrelid = 'daily_sentiment'::regclass
            AND primary_constraint.contype = 'p'
        GROUP BY primary_constraint.conname
    ) AS primary_key;

    IF current_primary_key_columns = ARRAY['date']::TEXT[] THEN
        EXECUTE format(
            'ALTER TABLE daily_sentiment DROP CONSTRAINT %I',
            current_primary_key_name
        );
        ALTER TABLE daily_sentiment ADD PRIMARY KEY (date, ticker);
    ELSIF current_primary_key_columns IS NULL THEN
        ALTER TABLE daily_sentiment ADD PRIMARY KEY (date, ticker);
    END IF;
END $$;
"""

UPSERT_DAILY_SENTIMENT_SQL = """
INSERT INTO daily_sentiment (
    date,
    ticker,
    spy_close,
    spy_volume,
    sentiment_score
)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (date, ticker) DO UPDATE
SET
    spy_close = EXCLUDED.spy_close,
    spy_volume = EXCLUDED.spy_volume,
    sentiment_score = EXCLUDED.sentiment_score;
"""


def fetch_ticker_history(symbol: str):
    end_date = date.today() + timedelta(days=1)
    start_date = end_date - timedelta(days=LOOKBACK_CALENDAR_DAYS)

    history = yf.Ticker(symbol).history(
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        interval="1d",
        auto_adjust=False,
    )

    if history.empty:
        raise RuntimeError(f"No historical price data returned for {symbol}.")

    history = history.dropna(subset=["Close", "Volume"])
    history = history[history["Volume"] > 0]

    if len(history) < TRADING_DAYS_TO_BACKFILL:
        raise RuntimeError(
            f"Only found {len(history)} valid trading days for {symbol}; "
            f"expected {TRADING_DAYS_TO_BACKFILL}."
        )

    return history.tail(TRADING_DAYS_TO_BACKFILL)


def fetch_google_news_titles(target_date: date, query: str) -> list[str]:
    next_date = target_date + timedelta(days=1)
    dated_query = (
        f"({query}) "
        f"after:{target_date.isoformat()} "
        f"before:{next_date.isoformat()}"
    )
    url = (
        f"{GOOGLE_NEWS_RSS_URL}?q={quote(dated_query)}"
        "&hl=en-US&gl=US&ceid=US:en"
    )
    feed = feedparser.parse(
        url,
        request_headers={"User-Agent": "Mozilla/5.0"},
    )

    if feed.bozo:
        raise RuntimeError(f"Unable to parse Google News RSS: {feed.bozo_exception}")

    titles = []
    seen = set()
    for entry in feed.entries:
        title = (entry.get("title") or "").strip()
        if not title or title in seen:
            continue

        seen.add(title)
        titles.append(title)

        if len(titles) >= MAX_HEADLINES_PER_DAY:
            break

    return titles


def load_finbert_pipeline() -> Any:
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()
    return pipeline(
        "sentiment-analysis",
        model=FINBERT_MODEL,
        tokenizer=FINBERT_MODEL,
    )


def score_headlines(titles: list[str], sentiment_pipeline: Any) -> float:
    if not titles:
        return 0.0

    results = sentiment_pipeline(
        titles,
        truncation=True,
        max_length=512,
        batch_size=8,
    )

    daily_scores = []
    for result in results:
        label = result["label"].lower()
        confidence = float(result["score"])

        if label == "positive":
            daily_scores.append(confidence)
        elif label == "negative":
            daily_scores.append(-confidence)
        else:
            daily_scores.append(0.0)

    return sum(daily_scores) / len(daily_scores)


def upsert_daily_sentiment(
    connection,
    symbol: str,
    target_date: date,
    spy_close: float,
    spy_volume: int,
    sentiment_score: float,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            UPSERT_DAILY_SENTIMENT_SQL,
            (
                target_date,
                symbol,
                spy_close,
                spy_volume,
                sentiment_score,
            ),
        )


def main() -> None:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is missing from the .env file.")

    print("Loading FinBERT model...")
    sentiment_pipeline = load_finbert_pipeline()

    with psycopg2.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute(CREATE_DAILY_SENTIMENT_TABLE_SQL)
            cursor.execute(MIGRATE_DAILY_SENTIMENT_PRIMARY_KEY_SQL)
        connection.commit()

        for target in TICKERS:
            symbol = target["symbol"]
            if symbol not in BACKFILL_SYMBOLS:
                print(f"Skipping {symbol}; historical data is already backfilled.")
                continue

            print(
                f"\nFetching last {TRADING_DAYS_TO_BACKFILL} "
                f"{symbol} trading days..."
            )
            history = fetch_ticker_history(symbol)
            print(f"Beginning {symbol} backfill for {len(history)} trading days.\n")

            for index, row in history.iterrows():
                target_date = index.date()
                ticker_close = float(row["Close"])
                ticker_volume = int(row["Volume"])

                print(f"Processing {symbol} on {target_date}...")

                try:
                    titles = fetch_google_news_titles(target_date, target["query"])
                except Exception as exc:
                    titles = []
                    print(
                        "  News lookup failed, using neutral sentiment. "
                        f"Reason: {exc}"
                    )

                if titles:
                    sentiment_score = score_headlines(titles, sentiment_pipeline)
                    print(f"  Headlines scored: {len(titles)}")
                else:
                    sentiment_score = 0.0
                    print("  No headlines found; using neutral sentiment.")

                upsert_daily_sentiment(
                    connection,
                    symbol,
                    target_date,
                    ticker_close,
                    ticker_volume,
                    sentiment_score,
                )
                connection.commit()

                print(
                    f"  Close: ${ticker_close:,.2f} | "
                    f"Volume: {ticker_volume:,} | "
                    f"Sentiment: {sentiment_score:.3f}"
                )

                time.sleep(REQUEST_PAUSE_SECONDS)

    print("\nSUCCESS: QQQ and DIA 60-trading-day backfill complete.")


if __name__ == "__main__":
    main()
