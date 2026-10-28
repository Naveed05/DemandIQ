"""Prepare a downloaded UCI Online Retail workbook as daily-demand CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from demandiq.uci_online_retail import prepare_online_retail

ROOT = Path(__file__).resolve().parents[1]
UCI_PAGE = "https://archive.ics.uci.edu/dataset/352/online+retail"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to the UCI .xlsx workbook")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "uci_online_retail_daily.csv",
        help="Output daily-sales CSV path",
    )
    parser.add_argument("--country", default="United Kingdom", help="Country/location proxy")
    parser.add_argument(
        "--min-history-days",
        type=int,
        default=60,
        help="Minimum inclusive calendar span per SKU (at least 29; default 60)",
    )
    args = parser.parse_args()
    try:
        transactions = pd.read_excel(args.input)
    except ImportError as exc:
        raise SystemExit(
            'Reading .xlsx requires openpyxl. Install it with: pip install -e ".[data]"'
        ) from exc
    prepared = prepare_online_retail(
        transactions, country=args.country, min_history_days=args.min_history_days
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(args.output, index=False, date_format="%Y-%m-%d")
    print(f"Source: {UCI_PAGE}")
    print(f"Country/location proxy: {args.country}")
    print(f"Prepared rows: {len(prepared):,}")
    print(f"Products: {prepared['product_id'].nunique():,}")
    print(f"Date range: {prepared['date'].min().date()} to {prepared['date'].max().date()}")
    print(f"Output: {args.output}")
    print("Note: this dataset has no store ID; country is only a location proxy.")


if __name__ == "__main__":
    main()
