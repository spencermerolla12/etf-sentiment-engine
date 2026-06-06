import streamlit as st

st.title("Macroeconomic Framework")

st.header("Theoretical Foundations")

st.subheader("The Efficient Market Hypothesis (EMH)")
st.write("This project acts as an empirical test of semi-strong form EMH by evaluating whether public news sentiment serves as a leading indicator of asset price movements, or if public information is already instantly priced into the indices.")

st.subheader("Behavioral Finance & Market Sentiment")
st.write("While classical economics assumes rational actors, asset prices frequently experience short-term volatility driven by systemic market sentiment and herd behavior. Aggregating index-level sentiment attempts to quantify this psychological momentum.")

st.subheader("Strategic Signaling & Information Asymmetry")
st.write("Financial markets frequently experience information asymmetry where institutional actors or corporate entities utilize media channels to release strategic narratives. Often associated with 'noise trading,' this dynamic suggests that orchestrated press releases can intentionally suppress or inflate asset valuations to create favorable entry or exit liquidity. By tracking aggregated headline sentiment against actual price action, this dashboard provides a framework to observe whether media narratives act as genuine leading indicators or strategic misdirection.")

st.sidebar.caption("All data, sentiment scores, and analyses provided by this dashboard are strictly for educational and demonstration purposes. This does not constitute investment advice, financial guidance, or a recommendation to buy or sell any security.")
