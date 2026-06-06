"""Methodology and economic theory page for the sentiment analytics dashboard."""

import streamlit as st


st.set_page_config(
    page_title="Methodology & Economic Theory",
    page_icon=":material/architecture:",
    layout="wide",
)

st.sidebar.header("Project Context")
st.sidebar.caption(
    "This application is an independent portfolio project demonstrating cloud-native "
    "data engineering, NLP sentiment scoring, and macroeconomic analysis."
)

st.sidebar.header("Financial Disclaimer")
st.sidebar.caption(
    "All data, sentiment scores, and analyses provided by this dashboard are strictly "
    "for educational and demonstration purposes. This does not constitute investment "
    "advice, financial guidance, or a recommendation to buy or sell any security."
)

st.title("Methodology & Economic Theory")

st.markdown(
    """
    ## Project Objective & Hypothesis

    **Goal:** To analyze whether aggregated financial news headlines serve as a
    leading, lagging, or coincident indicator for major index price movements.

    **Hypothesis:** Gathering a large volume of headline sentiment across
    multiple feeds can give us a clean, real-time look at market mood. This lets
    us capture the general trend quickly and cheaply without needing massive
    computing power to read full articles.

    ---

    ## Macroeconomic & Financial Framework

    **The Efficient Market Hypothesis (EMH):** This project acts as an empirical
    test of semi-strong form EMH by evaluating whether public news sentiment
    serves as a leading indicator of asset price movements, or if public
    information is already instantly priced into the indices.

    **Behavioral Finance & Market Sentiment:** While classical economics assumes
    rational actors, asset prices frequently experience short-term volatility
    driven by systemic market sentiment and herd behavior. Aggregating
    index-level sentiment attempts to quantify this psychological momentum.

    **Strategic Signaling & Information Asymmetry:** Financial markets
    frequently experience information asymmetry where institutional actors or
    corporate entities utilize media channels to release strategic narratives.
    Often associated with 'noise trading,' this dynamic suggests that
    orchestrated press releases can intentionally suppress or inflate asset
    valuations to create favorable entry or exit liquidity. By tracking
    aggregated headline sentiment against actual price action, this dashboard
    provides a framework to observe whether media narratives act as genuine
    leading indicators or strategic misdirection.

    ---

    ## Data Sourcing & Engineering Trade-offs

    **Market Data:** Asset pricing and volume are pulled directly from the Yahoo
    Finance API (yfinance).

    **Sentiment Sourcing:** News data is aggregated from free, public RSS
    syndication (Google News & Yahoo) rather than a premium financial news API.

    **The Engineering Trade-off:** To maintain a 100% automated, zero-cost
    pipeline that runs daily on cloud servers without human intervention, this
    engine parses headline titles using FinBERT rather than extracting full
    article text or using advanced LLMs (like GPT-4). While full-text parsing or
    newer language models would yield higher nuance, it introduces paywalls,
    anti-bot blocking, and heavy compute costs. This architecture intentionally
    sacrifices deep textual accuracy in favor of high velocity, broad sample
    size, and zero-maintenance automation.
    """
)
