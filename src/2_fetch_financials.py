"""Map high-emitting EPA parent companies to public equities and fetch fundamentals."""

from __future__ import annotations

import re
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data for project 2 (Emissions)"
INPUT_PATH = DATA_DIR / "cleaned_emissions.csv"
OUTPUT_PATH = DATA_DIR / "valuation_dataset.csv"
CACHE_DIR = PROJECT_ROOT / ".yfinance_cache"

TOP_COMPANIES = 100
US_LISTED_EXCHANGES = {"ASE", "NCM", "NGM", "NMS", "NYQ"}

# EPA parent names frequently differ from the listed issuer's legal or trading name.
# Values are US exchange-listed tickers; None marks known private/public-sector entities.
TICKER_OVERRIDES: dict[str, str | None] = {
    "Vistra Corp": "VST",
    "THE SOUTHERN CO": "SO",
    "DUKE ENERGY CORP": "DUK",
    "BERKSHIRE HATHAWAY INC": "BRK-B",
    "AMERICAN ELECTRIC POWER CO INC": "AEP",
    "CPN MANAGEMENT LP": "CPN",
    "NEXTERA ENERGY INC": "NEE",
    "US GOVERNMENT": None,
    "ENTERGY CORP": "ETR",
    "XCEL ENERGY INC": "XEL",
    "DOMINION ENERGY INC": "D",
    "EXXON MOBIL CORP": "XOM",
    "MARATHON PETROLEUM CORP": "MPC",
    "PPL CORP": "PPL",
    "PHILLIPS 66": "PSX",
    "EVERGY INC": "EVRG",
    "DTE ENERGY CO": "DTE",
    "NRG ENERGY INC": "NRG",
    "VALERO ENERGY CORP": "VLO",
    "BASIN ELECTRIC POWER COOPERATIVE": None,
    "CF INDUSTRIES HOLDINGS INC": "CF",
    "AMEREN CORP": "AEE",
    "LIGHTSTONE GENERATION LLC": None,
    "WEC Energy Group Inc": "WEC",
    "CMS ENERGY CORP": "CMS",
    "SALT RIVER PROJECT AGRICULTURAL IMPROVEMENT & POWER DISTRICT": None,
    "ASSOCIATED ELECTRIC COOPERATIVE INC": None,
    "CHEVRON CORP": "CVX",
    "DOW INC": "DOW",
    "FIRSTENERGY CORP": "FE",
    "SOUTH CAROLINA PUBLIC SERVICE AUTHORITY": None,
    "AIR PRODUCTS & CHEMICALS INC": "APD",
    "KOCH INDUSTRIES INC": None,
    "ALLIANT ENERGY CORP": "LNT",
    "AES CORP": "AES",
    "PRAIRIE STATE ENERGY CAMPUS MANAGEMENT CO": None,
    "LS Power Development, LLC": None,
    "PBF ENERGY INC": "PBF",
    "CPS ENERGY": None,
    "PINNACLE WEST CAPITAL CORP": "PNW",
    "TALEN ENERGY CORP": "TLN",
    "ARCLIGHT ENERGY PARTNERS FUND VII LP": None,
    "OGLETHORPE POWER CORP": None,
    "BUCKEYE POWER INC": None,
    "CONSTELLATION ENERGY CORP": "CEG",
    "LINDE INC": "LIN",
    "CLECO CORPORATE HOLDINGS LLC": None,
    "LOWER COLORADO RIVER AUTHORITY": None,
    "LYONDELLBASELL INDUSTRIES INC": "LYB",
    "NEBRASKA PUBLIC POWER DISTRICT": None,
    "TRI-STATE GENERATION & TRANSMISSION ASSOC INC": None,
    "EAST KENTUCKY POWER COOPERATIVE INC": None,
    "BP AMERICA INC": "BP",
    "PDV AMERICA INC": None,
    "OGE ENERGY CORP": "OGE",
    "OMAHA PUBLIC POWER DISTRICT": None,
    "REMC ASSETS LP": None,
    "TECO ENERGY INC": None,
    "PUGET HOLDINGS LLC": None,
    "UNS ENERGY CORP": "FTS",
    "SEMINOLE ELECTRIC COOPERATIVE INC": None,
    "SHELL PETROLEUM INC": "SHEL",
    "ARKANSAS ELECTRIC COOPERATIVE CORP": None,
    "ARAMCO SERVICES CO": None,
    "INVENERGY LLC": None,
    "OCCIDENTAL PETROLEUM CORP": "OXY",
    "Eastman Chemical Co": "EMN",
    "FORMOSA PLASTICS CORP USA": None,
    "LS POWER EQUITY PARTNERS LP": None,
    "PORTLAND GENERAL ELECTRIC CO": "POR",
    "JACKSONVILLE ELECTRIC AUTHORITY": None,
    "MINNKOTA POWER COOPERATIVE INC": None,
    "IDACORP INC": "IDA",
    "ARCLIGHT CAPITAL HOLDINGS LLC": None,
    "EDGEWATER GENERATION HOLDINGS LLC": None,
    "ALLETE INC": "ALE",
    "Nutrien US Topco LLC": "NTR",
    "HALLADOR ENERGY CO": "HNRG",
    "ALCOA CORP": "AA",
    "Omnis Pleasants, LLC": None,
    "TRANSALTA USA INC": "TAC",
    "NISOURCE INC": "NI",
    "Intermountain Power Agency": None,
    "HF SINCLAIR CORP": "DINO",
    "ARGO INFRASTRUCTURE PARTNERS LP": None,
    "COOPERATIVE ENERGY A MISSISSIPPI ELECTRIC COOPERATIVE": None,
    "BLACK HILLS CORP": "BKH",
    "ORLANDO UTILITIES COMMISSION (INC)": None,
    "AUSTIN ENERGY CORP": None,
    "DESERET GENERATION & TRANSMISSION COOPERATIVE": None,
    "AMERICAN AIR LIQUIDE HOLDINGS INC": None,
    "CENTERPOINT ENERGY INC": "CNP",
    "LONGVIEW INTERMEDIATE HOLDINGS C LLC": None,
    "ASCEND PERFORMANCE MATERIALS HOLDINGS INC": None,
    "CVR ENERGY INC": "CVI",
    "BIG RIVERS ELECTRIC CORP": None,
    "DEER PARK REFINING LP": None,
    "POWERSOUTH ENERGY COOPERATIVE": None,
    "WESTLAKE CHEMICAL CORP": "WLK",
    "GENERATION HOLDINGS LP": None,
}


def normalize_company_name(value: str) -> str:
    value = value.upper().replace("&", " AND ")
    value = re.sub(r"[^A-Z0-9 ]", " ", value)
    suffixes = {
        "CO",
        "COMPANY",
        "CORP",
        "CORPORATION",
        "HOLDING",
        "HOLDINGS",
        "INC",
        "LLC",
        "LP",
        "LTD",
        "PLC",
    }
    return " ".join(word for word in value.split() if word not in suffixes)


def name_match_score(company_name: str, candidate_name: str) -> float:
    company = normalize_company_name(company_name)
    candidate = normalize_company_name(candidate_name)
    if not company or not candidate:
        return 0.0

    company_tokens = set(company.split())
    candidate_tokens = set(candidate.split())
    token_score = len(company_tokens & candidate_tokens) / len(company_tokens)
    sequence_score = SequenceMatcher(None, company, candidate).ratio()
    return max(token_score, sequence_score)


def search_public_ticker(company_name: str) -> str | None:
    """Return a strongly matched US-listed equity ticker, otherwise None."""
    try:
        quotes = yf.Search(company_name, max_results=8).quotes
    except Exception as exc:
        print(f"  Search failed for {company_name}: {exc}")
        return None

    candidates: list[tuple[float, str]] = []
    for quote in quotes:
        if quote.get("quoteType") != "EQUITY":
            continue
        if quote.get("exchange") not in US_LISTED_EXCHANGES:
            continue

        ticker = quote.get("symbol")
        candidate_name = quote.get("longname") or quote.get("shortname") or ""
        score = name_match_score(company_name, candidate_name)
        if ticker and score >= 0.82:
            candidates.append((score, ticker))

    return max(candidates, default=(0.0, None))[1]


def map_ticker(company_name: str) -> str | None:
    if company_name in TICKER_OVERRIDES:
        return TICKER_OVERRIDES[company_name]
    return search_public_ticker(company_name)


def fetch_fundamentals(ticker: str) -> dict[str, object] | None:
    try:
        info = yf.Ticker(ticker).info
    except Exception as exc:
        print(f"  Fundamentals failed for {ticker}: {exc}")
        return None

    if info.get("quoteType") != "EQUITY":
        return None
    if info.get("exchange") not in US_LISTED_EXCHANGES:
        return None

    total_revenue = info.get("totalRevenue")
    enterprise_value = info.get("enterpriseValue")
    ebitda = info.get("ebitda")
    market_cap = info.get("marketCap")
    if not all(
        value is not None
        for value in (total_revenue, enterprise_value, ebitda, market_cap)
    ):
        print(f"  Missing required fundamentals for {ticker}")
        return None
    if total_revenue <= 0 or market_cap <= 0:
        return None

    return {
        "ticker": ticker,
        "listed_company_name": info.get("longName") or info.get("shortName"),
        "total_revenue_usd": float(total_revenue),
        "enterprise_value_usd": float(enterprise_value),
        "ebitda_usd": float(ebitda),
        "market_cap_usd": float(market_cap),
    }


def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Cleaned emissions file not found: {INPUT_PATH}")

    CACHE_DIR.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(CACHE_DIR))

    emissions = pd.read_csv(INPUT_PATH)
    top_emitters = (
        emissions.sort_values(
            "total_scope1_emissions_mtco2e", ascending=False
        )
        .head(TOP_COMPANIES)
        .copy()
    )
    top_emitters["ticker"] = top_emitters["parent_company"].apply(map_ticker)
    mapped = top_emitters.dropna(subset=["ticker"]).copy()

    print(
        f"Mapped {len(mapped)} of the top {TOP_COMPANIES} emitters "
        "to candidate public tickers."
    )

    records: list[dict[str, object]] = []
    fundamentals_cache: dict[str, dict[str, object] | None] = {}
    for row in mapped.itertuples(index=False):
        ticker = str(row.ticker)
        print(f"Fetching {ticker}: {row.parent_company}")
        if ticker not in fundamentals_cache:
            fundamentals_cache[ticker] = fetch_fundamentals(ticker)
            time.sleep(0.15)

        fundamentals = fundamentals_cache[ticker]
        if fundamentals is None:
            continue

        record = row._asdict()
        record.update(fundamentals)
        records.append(record)

    valuation = pd.DataFrame(records)
    if valuation.empty:
        raise RuntimeError("No companies returned complete Yahoo fundamentals.")

    valuation["Carbon_Intensity"] = (
        valuation["total_scope1_emissions_mtco2e"]
        / valuation["total_revenue_usd"]
    )
    valuation["ev_to_ebitda"] = (
        valuation["enterprise_value_usd"] / valuation["ebitda_usd"]
    )
    valuation["log_market_cap"] = np.log(valuation["market_cap_usd"])
    valuation["ebitda_margin"] = (
        valuation["ebitda_usd"] / valuation["total_revenue_usd"]
    )
    valuation = valuation[
        [
            "parent_company",
            "listed_company_name",
            "ticker",
            "sector",
            "total_scope1_emissions_mtco2e",
            "facility_count",
            "total_revenue_usd",
            "enterprise_value_usd",
            "ebitda_usd",
            "market_cap_usd",
            "Carbon_Intensity",
            "ev_to_ebitda",
            "log_market_cap",
            "ebitda_margin",
        ]
    ].sort_values("total_scope1_emissions_mtco2e", ascending=False)

    valuation.to_csv(OUTPUT_PATH, index=False)

    print(
        f"\nSuccessfully retained {len(valuation)} companies with complete "
        "public-market fundamentals."
    )
    print(f"Saved valuation dataset: {OUTPUT_PATH}")
    print("\nTop 5 rows:")
    print(valuation.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
