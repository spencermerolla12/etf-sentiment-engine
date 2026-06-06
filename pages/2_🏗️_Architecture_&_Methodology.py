import streamlit as st

st.title("Architecture & Methodology")

st.header("Project Objective & Hypothesis")
st.write("**Goal:** To analyze whether aggregated financial news headlines serve as a leading, lagging, or coincident indicator for major index price movements.")
st.write("**Hypothesis:** Gathering a large volume of headline sentiment across multiple feeds can give us a clean, real-time look at market mood. This lets us capture the general trend quickly and cheaply without needing massive computing power to read full articles.")

st.header("System Architecture")
st.write("This pipeline is 100% automated using a modern cloud-native stack:")
st.markdown("""
* **Data Ingestion:** GitHub Actions cron-jobs automatically scrape daily RSS feeds and ping the `yfinance` API for market closing prices.
* **NLP Processing:** The Hugging Face `FinBERT` transformer model processes and scores headline sentiment specifically trained on financial lexicons.
* **Database:** Time-series data is structured and stored in an Aiven PostgreSQL relational database utilizing composite primary keys.
* **Frontend:** Streamlit Community Cloud handles the interactive data visualization and user routing.
""")

st.sidebar.caption("This application is an independent portfolio project demonstrating cloud-native data engineering, NLP sentiment scoring, and macroeconomic analysis.")
