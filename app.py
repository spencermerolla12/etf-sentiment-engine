"""Interactive carbon-intensity and valuation regression dashboard."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import streamlit as st


DATA_PATH = (
    Path(__file__).resolve().parent
    / "data for project 2 (Emissions)"
    / "valuation_dataset.csv"
)
MAX_EV_TO_EBITDA = 50


@st.cache_data
def load_and_clean_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required_columns = {
        "parent_company",
        "listed_company_name",
        "ticker",
        "sector",
        "total_scope1_emissions_mtco2e",
        "Carbon_Intensity",
        "enterprise_value_usd",
        "ebitda_usd",
        "ev_to_ebitda",
        "log_market_cap",
        "ebitda_margin",
    }
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"Valuation dataset is missing columns: {sorted(missing_columns)}"
        )

    numeric_columns = [
        "Carbon_Intensity",
        "total_scope1_emissions_mtco2e",
        "enterprise_value_usd",
        "ebitda_usd",
        "ev_to_ebitda",
        "log_market_cap",
        "ebitda_margin",
    ]
    data[numeric_columns] = data[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    data = data.dropna(subset=numeric_columns + ["sector"]).copy()
    data = data[
        (data["enterprise_value_usd"] > 0)
        & (data["ebitda_usd"] > 0)
        & (data["ev_to_ebitda"] <= MAX_EV_TO_EBITDA)
    ].copy()

    data["Sector"] = data["sector"].astype(str).str.strip()
    data = data[data["Sector"] != ""].copy()
    return data.sort_values("Carbon_Intensity").reset_index(drop=True)


def run_ols(data: pd.DataFrame):
    return smf.ols(
        formula=(
            "ev_to_ebitda ~ Carbon_Intensity + log_market_cap "
            "+ ebitda_margin + C(Sector)"
        ),
        data=data,
    ).fit()


def format_p_value(value: float) -> str:
    return f"{value:.4f}" if value >= 0.0001 else f"{value:.2e}"


def run_carbon_tax_stress_test(
    data: pd.DataFrame,
    tax_rate: int,
) -> pd.DataFrame:
    simulation = data.copy()
    simulation["Carbon_Tax_Liability"] = (
        simulation["total_scope1_emissions_mtco2e"] * tax_rate
    )
    simulation["Adjusted_EBITDA"] = (
        simulation["ebitda_usd"] - simulation["Carbon_Tax_Liability"]
    )
    profitable = simulation["Adjusted_EBITDA"] > 0
    simulation["Adjusted_EV_to_EBITDA"] = np.where(
        profitable,
        simulation["enterprise_value_usd"] / simulation["Adjusted_EBITDA"],
        np.nan,
    )
    simulation["Stress_Status"] = np.where(
        profitable,
        "Profitable",
        "Negative Earnings",
    )
    return simulation


def add_fixed_effect_lines(
    figure: go.Figure,
    data: pd.DataFrame,
    model,
) -> None:
    color_by_sector = {
        trace.name: trace.marker.color
        for trace in figure.data
        if trace.mode == "markers"
    }

    for sector, sector_data in data.groupby("Sector", observed=True):
        median_log_market_cap = sector_data["log_market_cap"].median()
        median_ebitda_margin = sector_data["ebitda_margin"].median()
        x_values = np.linspace(
            sector_data["Carbon_Intensity"].min(),
            sector_data["Carbon_Intensity"].max(),
            50,
        )
        predictions = model.predict(
            pd.DataFrame(
                {
                    "Carbon_Intensity": x_values,
                    "log_market_cap": median_log_market_cap,
                    "ebitda_margin": median_ebitda_margin,
                    "Sector": sector,
                }
            )
        )
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=predictions,
                mode="lines",
                name=f"{sector} fixed-effect fit",
                line={
                    "color": color_by_sector.get(sector),
                    "width": 2,
                    "dash": "dash",
                },
                hovertemplate=(
                    f"Sector: {sector}<br>"
                    "Carbon Intensity: %{x:.6f}<br>"
                    "Predicted EV/EBITDA: %{y:.2f}<extra></extra>"
                ),
                showlegend=False,
            )
        )


def main() -> None:
    st.set_page_config(
        page_title="Climate Transition Risk & Valuation Modeler",
        layout="wide",
    )
    st.title("Climate Transition Risk & Valuation Modeler")
    st.caption(
        "Testing whether carbon-intensive companies trade at different "
        "enterprise-value multiples after controlling for sector, company "
        "size, and profitability."
    )

    try:
        data = load_and_clean_data(DATA_PATH)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    if (
        len(data) < 3
        or data["Carbon_Intensity"].nunique() < 2
        or data["Sector"].nunique() < 2
    ):
        st.error(
            "The cleaned dataset does not contain enough variation for "
            "sector fixed-effects OLS."
        )
        st.stop()

    tax_rate = st.sidebar.slider(
        "Hypothetical Carbon Tax ($/Metric Ton)",
        min_value=0,
        max_value=150,
        value=0,
        step=5,
    )

    model = run_ols(data)
    carbon_coefficient = model.params["Carbon_Intensity"]
    carbon_p_value = model.pvalues["Carbon_Intensity"]

    st.header("Current Market Reality")
    r_squared_col, p_value_col, coefficient_col = st.columns(3)
    r_squared_col.metric("R-squared", f"{model.rsquared:.4f}")
    p_value_col.metric("P-value", format_p_value(carbon_p_value))
    coefficient_col.metric(
        "Carbon Coefficient (Beta 1)",
        f"{carbon_coefficient:,.2f}",
    )

    figure = px.scatter(
        data,
        x="Carbon_Intensity",
        y="ev_to_ebitda",
        color="Sector",
        hover_name="listed_company_name",
        hover_data={
            "ticker": True,
            "parent_company": True,
            "Sector": True,
            "Carbon_Intensity": ":.6f",
            "ev_to_ebitda": ":.2f",
        },
        labels={
            "Carbon_Intensity": "Carbon Intensity (Scope 1 mtCO2e / Revenue USD)",
            "ev_to_ebitda": "EV / EBITDA",
            "ticker": "Ticker",
            "parent_company": "EPA Parent Company",
        },
        title="Carbon Intensity vs. EV/EBITDA with Sector Fixed Effects",
        template="plotly_white",
    )
    figure.update_traces(
        marker={"size": 10, "opacity": 0.8},
        selector={"mode": "markers"},
    )
    add_fixed_effect_lines(figure, data, model)
    figure.update_layout(
        hovermode="closest",
        legend_title_text="Sector",
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
    )

    st.plotly_chart(figure, width="stretch")
    st.caption(
        f"Multivariate OLS sample: {len(data)} companies across "
        f"{data['Sector'].nunique()} sector classifications after excluding "
        f"non-positive EV or EBITDA and multiples above {MAX_EV_TO_EBITDA}x."
    )

    st.header("Transition Risk Stress Test")
    simulation = run_carbon_tax_stress_test(data, tax_rate)
    unprofitable_count = int((simulation["Adjusted_EBITDA"] <= 0).sum())

    unprofitable_col, tax_rate_col = st.columns(2)
    unprofitable_col.metric(
        "Companies with Negative Earnings",
        f"{unprofitable_count} of {len(simulation)}",
    )
    tax_rate_col.metric(
        "Carbon Tax Scenario",
        f"${tax_rate}/Metric Ton",
    )

    top_emitters = simulation.nlargest(
        10,
        "total_scope1_emissions_mtco2e",
    ).copy()
    chart_data = top_emitters[
        ["listed_company_name", "ticker", "ebitda_usd", "Adjusted_EBITDA"]
    ].melt(
        id_vars=["listed_company_name", "ticker"],
        value_vars=["ebitda_usd", "Adjusted_EBITDA"],
        var_name="Scenario",
        value_name="EBITDA_USD",
    )
    chart_data["Scenario"] = chart_data["Scenario"].map(
        {
            "ebitda_usd": "Baseline EBITDA",
            "Adjusted_EBITDA": "Adjusted EBITDA",
        }
    )
    chart_data["Company"] = (
        chart_data["listed_company_name"] + " (" + chart_data["ticker"] + ")"
    )
    chart_data["EBITDA_Billions"] = chart_data["EBITDA_USD"] / 1_000_000_000

    stress_figure = px.bar(
        chart_data,
        x="Company",
        y="EBITDA_Billions",
        color="Scenario",
        barmode="group",
        color_discrete_map={
            "Baseline EBITDA": "#2F6BFF",
            "Adjusted EBITDA": "#E45756",
        },
        labels={
            "Company": "Company",
            "EBITDA_Billions": "EBITDA (USD Billions)",
        },
        title=(
            "Top 10 Emitters: Baseline vs. Carbon-Tax-Adjusted EBITDA "
            f"at ${tax_rate}/Metric Ton"
        ),
        template="plotly_white",
    )
    stress_figure.add_hline(y=0, line_color="black", line_width=1)
    stress_figure.update_layout(
        legend_title_text="Scenario",
        xaxis_tickangle=-35,
        margin={"l": 20, "r": 20, "t": 70, "b": 100},
    )
    st.plotly_chart(stress_figure, width="stretch")

    if unprofitable_count:
        failed_companies = simulation.loc[
            simulation["Stress_Status"] == "Negative Earnings",
            ["listed_company_name", "ticker"],
        ]
        company_labels = ", ".join(
            f"{row.listed_company_name} ({row.ticker})"
            for row in failed_companies.itertuples(index=False)
        )
        st.warning(f"Negative Earnings under this scenario: {company_labels}")


if __name__ == "__main__":
    main()
