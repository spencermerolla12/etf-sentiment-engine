"""Aggregate EPA FLIGHT facility emissions to target-sector parent companies."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data for project 2 (Emissions)"
INPUT_PATH = DATA_DIR / "raw_emissions.csv"
OUTPUT_PATH = DATA_DIR / "cleaned_emissions.csv"

EPA_HEADER_ROW = 5
PARENT_COLUMN = "PARENT COMPANIES"
EMISSIONS_COLUMN = "GHG QUANTITY (METRIC TONS CO2e)"
SUBPART_COLUMN = "SUBPARTS"
FACILITY_ID_COLUMN = "GHGRP ID"

# EPA GHGRP subparts used to identify the requested industrial sectors.
SECTOR_SUBPARTS = {
    "Utilities": {"D", "DD"},
    "Energy": {"FF", "LL", "MM", "NN", "W", "Y"},
    "Chemicals/Materials": {
        "BB",
        "E",
        "EE",
        "F",
        "G",
        "L",
        "N",
        "O",
        "P",
        "U",
        "V",
        "X",
        "Z",
    },
}

PARENT_PATTERN = re.compile(r"^(.*?)\s*\(([\d.]+)%\)\s*$")
MISSING_OWNERSHIP_PATTERN = re.compile(r"^(.*?)\s*\(%\)\s*$")


def sectors_for_subparts(value: object) -> list[str]:
    subparts = {part.strip() for part in str(value).split(",")}
    return [
        sector
        for sector, sector_subparts in SECTOR_SUBPARTS.items()
        if subparts & sector_subparts
    ]


def parse_parent_companies(value: object) -> list[tuple[str, float]]:
    parents: list[tuple[str, float]] = []
    for entry in str(value).split(";"):
        entry = entry.strip()
        if not entry:
            continue

        match = PARENT_PATTERN.match(entry)
        if not match:
            missing_ownership_match = MISSING_OWNERSHIP_PATTERN.match(entry)
            if missing_ownership_match:
                parents.append((missing_ownership_match.group(1).strip(), 1.0))
                continue
            raise ValueError(f"Could not parse parent company ownership: {entry!r}")

        company = match.group(1).strip()
        ownership_share = float(match.group(2)) / 100
        parents.append((company, ownership_share))

    return parents


def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Raw EPA FLIGHT file not found: {INPUT_PATH}")

    emissions = pd.read_csv(INPUT_PATH, skiprows=EPA_HEADER_ROW, low_memory=False)

    print("Exact EPA column headers:")
    for column in emissions.columns:
        print(repr(column))

    required_columns = {
        PARENT_COLUMN,
        EMISSIONS_COLUMN,
        SUBPART_COLUMN,
        FACILITY_ID_COLUMN,
    }
    missing_columns = required_columns - set(emissions.columns)
    if missing_columns:
        raise KeyError(f"Missing required EPA columns: {sorted(missing_columns)}")

    if emissions[PARENT_COLUMN].astype(str).str.contains(r"\(%\)", regex=True).any():
        print(
            "\nWarning: EPA leaves the ownership percentage blank for "
            "'US GOVERNMENT (%)'; assuming 100% ownership."
        )

    emissions[EMISSIONS_COLUMN] = pd.to_numeric(
        emissions[EMISSIONS_COLUMN], errors="coerce"
    )
    emissions = emissions.dropna(
        subset=[PARENT_COLUMN, EMISSIONS_COLUMN, SUBPART_COLUMN]
    ).copy()
    emissions["SECTORS"] = emissions[SUBPART_COLUMN].apply(sectors_for_subparts)
    target_facilities = emissions[emissions["SECTORS"].str.len() > 0].copy()

    parent_records: list[dict[str, object]] = []
    for _, row in target_facilities.iterrows():
        facility_emissions = float(row[EMISSIONS_COLUMN])
        for parent_company, ownership_share in parse_parent_companies(
            row[PARENT_COLUMN]
        ):
            parent_records.append(
                {
                    "parent_company": parent_company,
                    "sectors": row["SECTORS"],
                    "facility_id": row[FACILITY_ID_COLUMN],
                    "ownership_adjusted_scope1_mtco2e": (
                        facility_emissions * ownership_share
                    ),
                }
            )

    parent_data = pd.DataFrame(parent_records)
    if parent_data.empty:
        raise ValueError("No facilities matched the requested target sectors.")

    cleaned = (
        parent_data.groupby("parent_company", as_index=False)
        .agg(
            sector=("sectors", lambda values: "; ".join(
                sector
                for sector in SECTOR_SUBPARTS
                if any(sector in sectors for sectors in values)
            )),
            total_scope1_emissions_mtco2e=(
                "ownership_adjusted_scope1_mtco2e",
                "sum",
            ),
            facility_count=("facility_id", "nunique"),
        )
        .sort_values(
            ["total_scope1_emissions_mtco2e", "parent_company"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )
    cleaned["total_scope1_emissions_mtco2e"] = cleaned[
        "total_scope1_emissions_mtco2e"
    ].round(2)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUTPUT_PATH, index=False)

    print(f"\nTarget facilities included: {len(target_facilities):,}")
    print(f"Unique parent companies: {len(cleaned):,}")
    print(f"Saved cleaned data: {OUTPUT_PATH}")
    print("\nTop 5 processed rows:")
    print(cleaned.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
