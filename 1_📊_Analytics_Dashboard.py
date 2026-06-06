"""Streamlit dashboard for the ETF Sentiment Engine."""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import psycopg2
import streamlit as st
from dotenv import load_dotenv
from plotly.subplots import make_subplots


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
TICKER_OPTIONS = ["SPY", "QQQ", "DIA"]
TICKER_CONTEXT = {
    "SPY": (
        "Tracks the S&P 500 index, representing the top 500 large-cap "
        "US corporations."
    ),
    "QQQ": (
        "Tracks the NASDAQ-100 index, heavily weighted toward high-growth "
        "technology and innovation sectors."
    ),
    "DIA": (
        "Tracks the Dow Jones Industrial Average, representing 30 blue-chip, "
        "value-oriented industrial leaders."
    ),
}

st.set_page_config(
    page_title="ETF Sentiment Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)
st.markdown(
    """
    <style>
        /* Remove empty space at top of main page and sidebar */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 0rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=300)
def load_daily_sentiment() -> pd.DataFrame:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is missing from the .env file.")

    query = """
        SELECT *
        FROM daily_sentiment
        ORDER BY date;
    """

    with psycopg2.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]

    dataframe = pd.DataFrame(rows, columns=columns)

    if dataframe.empty:
        return dataframe

    dataframe["date"] = pd.to_datetime(dataframe["date"])
    dataframe["spy_close"] = pd.to_numeric(dataframe["spy_close"])
    dataframe["spy_volume"] = pd.to_numeric(dataframe["spy_volume"])
    dataframe["sentiment_score"] = pd.to_numeric(dataframe["sentiment_score"])
    dataframe = dataframe.sort_values(["ticker", "date"])
    dataframe["sentiment_3d_average"] = (
        dataframe.groupby("ticker")["sentiment_score"]
        .transform(lambda scores: scores.rolling(window=3).mean())
    )
    return dataframe


def filter_by_ticker(dataframe: pd.DataFrame, ticker: str) -> pd.DataFrame:
    return dataframe[dataframe["ticker"] == ticker].copy()


def filter_by_date_range(
    dataframe: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    return dataframe[
        (dataframe["date"].dt.date >= start_date)
        & (dataframe["date"].dt.date <= end_date)
    ]


def build_sentiment_chart(dataframe: pd.DataFrame, ticker: str) -> go.Figure:
    figure = make_subplots(specs=[[{"secondary_y": True}]])

    figure.add_trace(
        go.Scatter(
            x=dataframe["date"],
            y=dataframe["spy_close"],
            name=f"{ticker} Close",
            mode="lines",
            line={"color": "#FFA500", "width": 4},
            hovertemplate=(
                f"Date: %{{x|%Y-%m-%d}}<br>{ticker} Close: "
                "$%{y:,.2f}<extra></extra>"
            ),
        ),
        secondary_y=False,
    )

    figure.add_trace(
        go.Scatter(
            x=dataframe["date"],
            y=dataframe["sentiment_score"],
            name="Daily Sentiment",
            mode="lines",
            line={"color": "rgba(37, 99, 235, 0.45)", "width": 2, "dash": "dash"},
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "Daily Sentiment: %{y:.3f}<extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    figure.add_trace(
        go.Scatter(
            x=dataframe["date"],
            y=dataframe["sentiment_3d_average"],
            name="3-Day Sentiment Average",
            mode="lines",
            line={"color": "#2563eb", "width": 4},
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "3-Day Average: %{y:.3f}<extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    figure.update_layout(
        title=f"{ticker} Price vs. FinBERT Sentiment",
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.15,
            "xanchor": "center",
            "x": 0.5,
        },
        dragmode="pan",
        margin={"t": 80, "r": 70, "b": 50, "l": 70},
        template="plotly_white",
    )
    figure.update_xaxes(title_text="", tickformat="%m/%Y", dtick="M1")
    figure.update_yaxes(
        title_text=f"{ticker} Close Price",
        tickprefix="$",
        tickformat=",.2f",
        secondary_y=False,
    )
    figure.update_yaxes(
        title_text="Sentiment Score",
        range=[-1, 1],
        tickformat=".2f",
        secondary_y=True,
    )

    return figure


def main() -> None:
    try:
        dataframe = load_daily_sentiment()
    except Exception as exc:
        st.error(f"Unable to load dashboard data: {exc}")
        return

    if dataframe.empty:
        st.warning("No rows found in the daily_sentiment table yet.")
        return

    st.sidebar.header("Dashboard Configuration")
    selected_ticker = st.sidebar.radio(
        "Select ETF",
        ["SPY", "QQQ", "DIA"],
        captions=[
            "S&P 500 Large-Cap",
            "Nasdaq 100 Tech",
            "Dow Jones Industrial",
        ],
    )
    st.sidebar.divider()

    ticker_dataframe = filter_by_ticker(dataframe, selected_ticker)

    st.title("Macro Markets Sentiment Analytics")
    st.caption(TICKER_CONTEXT[selected_ticker])

    if ticker_dataframe.empty:
        st.warning(f"No rows found for {selected_ticker} yet.")
        return

    min_date = ticker_dataframe["date"].min().date()
    max_date = ticker_dataframe["date"].max().date()
    latest_row = ticker_dataframe.sort_values("date").iloc[-1]
    trading_days = ticker_dataframe["date"].dt.date.nunique()

    st.sidebar.caption(f"Available data: {min_date:%b %d, %Y} to {max_date:%b %d, %Y}")

    start_column, end_column = st.sidebar.columns(2)
    with start_column:
        selected_start = st.date_input(
            "Start date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            format="YYYY-MM-DD",
            disabled=min_date == max_date,
        )
    with end_column:
        selected_end = st.date_input(
            "End date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            format="YYYY-MM-DD",
            disabled=min_date == max_date,
        )

    if selected_start > selected_end:
        st.warning("The start date must be on or before the end date.")
        return

    close_metric, sentiment_metric, horizon_metric = st.columns(3)
    close_metric.metric(
        f"Latest {selected_ticker} Close",
        f"${latest_row['spy_close']:,.2f}",
    )
    sentiment_metric.metric(
        "Latest Sentiment Score",
        f"{latest_row['sentiment_score']:.3f}",
    )
    horizon_metric.metric("Data Horizon", f"{trading_days} trading days")

    filtered_dataframe = filter_by_date_range(
        ticker_dataframe,
        selected_start,
        selected_end,
    )

    if filtered_dataframe.empty:
        st.warning("No data is available for the selected date range.")
        return

    chart = build_sentiment_chart(filtered_dataframe, selected_ticker)
    st.plotly_chart(
        chart,
        use_container_width=True,
        config={
            "displaylogo": False,
            "modeBarButtonsToRemove": [
                "toImage",
                "select2d",
                "lasso2d",
                "hoverCompareCartesian",
                "hoverClosestCartesian",
                "toggleSpikelines",
            ],
        },
    )

    with st.expander("View underlying data"):
        st.dataframe(
            filtered_dataframe.sort_values("date", ascending=False),
            width="stretch",
            hide_index=True,
        )


if __name__ == "__main__":
    main()
