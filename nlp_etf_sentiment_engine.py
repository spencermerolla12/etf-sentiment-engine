"""Fetch financial headlines, FinBERT sentiment, and index ETF market data.

This is an early data-ingestion script for the NLP ETF Sentiment Engine project.
It pulls ticker-specific headlines from multiple free RSS sources and the
latest available daily close/volume, then scores financial news sentiment with
FinBERT for each target ETF.
"""

from __future__ import annotations

import html
import os
import re
from collections import defaultdict
from datetime import date, datetime
from typing import Any
from urllib.parse import quote_plus

import feedparser
import psycopg2
import yfinance as yf
from dotenv import load_dotenv

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
load_dotenv()

try:
    from transformers import pipeline
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()
except ImportError:  # pragma: no cover - handled at runtime for clearer setup guidance.
    pipeline = None


YAHOO_FINANCE_RSS_URL = (
    "https://feeds.finance.yahoo.com/rss/2.0/headline"
    "?s={symbol}&region=US&lang=en-US"
)
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"
TICKERS = [
    {"symbol": "SPY", "query": "S&P 500 OR SPY ETF"},
    {"symbol": "QQQ", "query": "NASDAQ OR QQQ ETF"},
    {"symbol": "DIA", "query": "Dow Jones OR DIA ETF"},
]
INDEX_NEWS_QUERY_TEMPLATES = (
    "{query} when:1d",
    "({query}) stock market when:1d",
    "({query}) earnings economy when:1d",
    "({query}) Federal Reserve inflation when:1d",
)
ARTICLES_PER_SOURCE = 20
FINBERT_BATCH_SIZE = 16
FINBERT_MODEL = "ProsusAI/finbert"
MISSING_SUMMARY = "Summary not provided by this RSS item."
DATABASE_URI = os.getenv("DATABASE_URL")
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


def clean_text(value: str) -> str:
    """Convert RSS HTML snippets into readable console text."""
    without_tags = re.sub(r"<[^>]+>", " ", value or "")
    normalized = " ".join(html.unescape(without_tags).split())
    return normalized


def extract_summary(entry: feedparser.util.FeedParserDict) -> str:
    for field in ("summary", "description"):
        if entry.get(field):
            return clean_text(entry[field])

    if entry.get("content"):
        return clean_text(entry.content[0].get("value", ""))

    return MISSING_SUMMARY


def load_finbert_pipeline() -> Any:
    if pipeline is None:
        raise RuntimeError(
            "The transformers library is not installed. Run: "
            "python -m pip install -r requirements.txt"
        )

    return pipeline(
        "sentiment-analysis",
        model=FINBERT_MODEL,
        tokenizer=FINBERT_MODEL,
    )


def choose_sentiment_text(article: dict[str, str]) -> tuple[str, str]:
    return article.get("title", "").strip(), "headline"


def parse_published_date(published: str) -> str:
    if not published or published == "No publish date available":
        return "unknown"

    try:
        normalized = published.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).date().isoformat()
    except ValueError:
        return "unknown"


def summarize_finbert_scores(scores: list[dict[str, Any]]) -> dict[str, float | str]:
    probabilities = {
        item["label"].lower(): float(item["score"])
        for item in scores
    }

    positive = probabilities.get("positive", 0.0)
    neutral = probabilities.get("neutral", 0.0)
    negative = probabilities.get("negative", 0.0)

    return {
        "label": max(probabilities, key=probabilities.get),
        "score": positive - negative,
        "positive_probability": positive,
        "neutral_probability": neutral,
        "negative_probability": negative,
    }


def score_texts_with_finbert(
    texts: list[str],
    sentiment_pipeline: Any,
) -> list[dict[str, float | str]]:
    if not texts:
        return []

    raw_results = sentiment_pipeline(
        texts,
        truncation=True,
        max_length=512,
        batch_size=FINBERT_BATCH_SIZE,
        top_k=None,
    )

    if len(raw_results) != len(texts):
        raise RuntimeError(
            "FinBERT returned a different number of results than input headlines."
        )

    return [summarize_finbert_scores(scores) for scores in raw_results]


def score_text_with_finbert(text: str, sentiment_pipeline: Any) -> dict[str, float | str]:
    return score_texts_with_finbert([text], sentiment_pipeline)[0]


def calculate_daily_aggregated_sentiment(
    articles: list[dict[str, str]],
    sentiment_pipeline: Any | None = None,
) -> dict[str, Any]:
    sentiment_pipeline = sentiment_pipeline or load_finbert_pipeline()
    scored_articles = []
    scores_by_date: dict[str, list[float]] = defaultdict(list)
    articles_to_score = []
    texts_to_score = []

    for article in articles:
        text, text_source = choose_sentiment_text(article)
        if not text:
            continue

        articles_to_score.append((article, text_source))
        texts_to_score.append(text)

    sentiments = score_texts_with_finbert(texts_to_score, sentiment_pipeline)

    for (article, text_source), sentiment in zip(
        articles_to_score,
        sentiments,
        strict=True,
    ):
        published_date = parse_published_date(article.get("published", ""))
        score = float(sentiment["score"])

        scores_by_date[published_date].append(score)
        scored_articles.append(
            {
                "title": article["title"],
                "source": article.get("source", "Unknown source"),
                "published_date": published_date,
                "text_source": text_source,
                "sentiment_label": sentiment["label"],
                "sentiment_score": score,
                "positive_probability": sentiment["positive_probability"],
                "neutral_probability": sentiment["neutral_probability"],
                "negative_probability": sentiment["negative_probability"],
            }
        )

    daily_scores = {
        published_date: sum(scores) / len(scores)
        for published_date, scores in scores_by_date.items()
    }
    overall_score = (
        sum(item["sentiment_score"] for item in scored_articles) / len(scored_articles)
        if scored_articles
        else 0.0
    )

    return {
        "model": FINBERT_MODEL,
        "daily_scores": daily_scores,
        "overall_score": overall_score,
        "scored_article_count": len(scored_articles),
        "scored_articles": scored_articles,
    }


def build_google_news_rss_url(query: str) -> str:
    return (
        f"{GOOGLE_NEWS_RSS_URL}?q={quote_plus(query)}"
        "&hl=en-US&gl=US&ceid=US:en"
    )


def normalize_headline_for_deduplication(title: str) -> str:
    normalized = clean_text(title).casefold()
    normalized = re.sub(r"\s+-\s+[^-]+$", "", normalized)
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


def fetch_rss_source(
    source_name: str,
    url: str,
    limit: int = ARTICLES_PER_SOURCE,
) -> list[dict[str, str]]:
    feed = feedparser.parse(
        url,
        request_headers={
            "User-Agent": "Mozilla/5.0 (compatible; ETFSentimentEngine/1.0)",
            "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
        },
    )

    if feed.bozo and not feed.entries:
        raise RuntimeError(f"Unable to parse {source_name}: {feed.bozo_exception}")

    articles = []
    for entry in feed.entries[:limit]:
        title = clean_text(entry.get("title", ""))
        if not title:
            continue

        articles.append(
            {
                "title": title,
                "summary": extract_summary(entry),
                "link": entry.get("link", "No link available"),
                "published": entry.get(
                    "published",
                    entry.get("updated", "No publish date available"),
                ),
                "source": source_name,
            }
        )

    return articles


def fetch_financial_news(target: dict[str, str]) -> list[dict[str, str]]:
    symbol = target["symbol"]
    query = target["query"]
    sources = [
        (
            f"Yahoo Finance: {symbol}",
            YAHOO_FINANCE_RSS_URL.format(symbol=symbol),
        )
    ]
    sources.extend(
        (
            f"Google News: {search_query}",
            build_google_news_rss_url(search_query),
        )
        for search_query in (
            template.format(query=query)
            for template in INDEX_NEWS_QUERY_TEMPLATES
        )
    )

    collected_articles = []
    failed_sources = []
    for source_name, url in sources:
        try:
            collected_articles.extend(fetch_rss_source(source_name, url))
        except Exception as exc:
            failed_sources.append(f"{source_name}: {exc}")

    unique_articles = []
    seen_headlines = set()
    for article in collected_articles:
        headline_key = normalize_headline_for_deduplication(article["title"])
        if not headline_key or headline_key in seen_headlines:
            continue

        seen_headlines.add(headline_key)
        unique_articles.append(article)

    if not unique_articles:
        failures = "; ".join(failed_sources) or "No RSS entries were returned."
        raise RuntimeError(f"No financial headlines were collected. {failures}")

    if failed_sources:
        print("RSS source warnings:")
        for failure in failed_sources:
            print(f"  - {failure}")

    return unique_articles


def fetch_ticker_daily_data(symbol: str) -> dict[str, str | float | int]:
    ticker = yf.Ticker(symbol)
    history = ticker.history(period="7d", interval="1d", auto_adjust=False)

    if history.empty:
        raise RuntimeError(f"No price history returned for {symbol}.")

    valid_history = history.dropna(subset=["Close", "Volume"])
    if valid_history.empty:
        raise RuntimeError(f"No valid close/volume rows returned for {symbol}.")

    latest_date = valid_history.index[-1].date()
    latest_row = valid_history.iloc[-1]

    label = "today" if latest_date == date.today() else "latest available trading day"

    return {
        "ticker": symbol,
        "date": latest_date.isoformat(),
        "date_label": label,
        "close": float(latest_row["Close"]),
        "volume": int(latest_row["Volume"]),
    }


def upload_daily_sentiment(
    ticker_data: dict[str, str | float | int],
    sentiment_data: dict[str, Any],
) -> tuple[bool, str]:
    if not DATABASE_URI:
        return (
            False,
            "Database upload failed: DATABASE_URL is missing from the .env file.",
        )

    try:
        with psycopg2.connect(DATABASE_URI) as connection:
            with connection.cursor() as cursor:
                cursor.execute(CREATE_DAILY_SENTIMENT_TABLE_SQL)
                cursor.execute(MIGRATE_DAILY_SENTIMENT_PRIMARY_KEY_SQL)
                cursor.execute(
                    UPSERT_DAILY_SENTIMENT_SQL,
                    (
                        ticker_data["date"],
                        ticker_data["ticker"],
                        ticker_data["close"],
                        ticker_data["volume"],
                        sentiment_data["overall_score"],
                    ),
                )

        return (
            True,
            f"Database upload succeeded for "
            f"{ticker_data['ticker']} on {ticker_data['date']}.",
        )
    except Exception as exc:
        return False, f"Database upload failed: {exc}"


def print_news(symbol: str, articles: list[dict[str, str]]) -> None:
    print("=" * 80)
    print(f"{symbol} MULTI-SOURCE FINANCIAL NEWS - {len(articles)} UNIQUE HEADLINES")
    print("=" * 80)

    for index, article in enumerate(articles, start=1):
        print(f"\n{index}. {article['title']}")
        print(f"   Source: {article['source']}")
        print(f"   Published: {article['published']}")
        print(f"   Summary: {article['summary']}")
        print(f"   Link: {article['link']}")


def print_market_data(data: dict[str, str | float | int]) -> None:
    print("\n" + "=" * 80)
    print(f"{data['ticker']} ETF MARKET DATA")
    print("=" * 80)
    print(f"Date: {data['date']} ({data['date_label']})")
    print(f"Closing Price: ${data['close']:,.2f}")
    print(f"Total Trading Volume: {data['volume']:,} shares")


def print_sentiment_report(symbol: str, sentiment_data: dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print(f"{symbol} FINBERT SENTIMENT REPORT - {sentiment_data['model']}")
    print("=" * 80)
    print(f"Articles scored: {sentiment_data['scored_article_count']}")
    print(f"Overall aggregated score: {sentiment_data['overall_score']:.3f}")
    print("Score range: -1.000 highly negative | 0.000 neutral | 1.000 highly positive")

    print("\nDaily aggregated sentiment:")
    for published_date, score in sorted(sentiment_data["daily_scores"].items()):
        print(f"  {published_date}: {score:.3f}")

    print("\nArticle-level sentiment:")
    for index, article in enumerate(sentiment_data["scored_articles"], start=1):
        print(
            f"  {index}. {article['sentiment_score']:.3f} "
            f"({article['sentiment_label']}, source: {article['text_source']}) - "
            f"{article['title']}"
        )


def print_database_status(succeeded: bool, message: str) -> None:
    print("\n" + "=" * 80)
    print("DATABASE UPLOAD STATUS")
    print("=" * 80)
    status = "SUCCESS" if succeeded else "FAILED"
    print(f"{status}: {message}")


def main() -> None:
    sentiment_pipeline = load_finbert_pipeline()

    for target in TICKERS:
        symbol = target["symbol"]
        print("\n" + "#" * 80)
        print(f"PROCESSING {symbol}")
        print("#" * 80)

        articles = fetch_financial_news(target)
        ticker_data = fetch_ticker_daily_data(symbol)
        sentiment_data = calculate_daily_aggregated_sentiment(
            articles,
            sentiment_pipeline,
        )
        database_succeeded, database_message = upload_daily_sentiment(
            ticker_data,
            sentiment_data,
        )

        print_news(symbol, articles)
        print_market_data(ticker_data)
        print_sentiment_report(symbol, sentiment_data)
        print_database_status(database_succeeded, database_message)


if __name__ == "__main__":
    main()
