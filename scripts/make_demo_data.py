"""Generate deterministic synthetic daily retail sales for local demos."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from demandiq.data import validate_sales_frame  # noqa: E402


def generate_demo_sales(days: int = 240, seed: int = 42) -> pd.DataFrame:
    """Create a deterministic multi-store, multi-product dataset."""
    if days < 60:
        raise ValueError("days must be at least 60 to support lag features and evaluation")
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")
    rows: list[dict[str, object]] = []
    for store_idx in range(1, 4):
        for product_idx in range(1, 6):
            base = 8 + product_idx * 2.2 + store_idx * 1.5
            noise = rng.normal(0, 1.8, size=days)
            weekly = 2.0 * np.sin(2 * np.pi * np.arange(days) / 7)
            yearly_slow = 1.2 * np.sin(2 * np.pi * np.arange(days) / 45)
            promo = rng.binomial(1, 0.08, size=days)
            sales = np.maximum(0, np.rint(base + weekly + yearly_slow + noise + promo * base * 0.35))
            for date, qty, is_promo in zip(dates, sales, promo, strict=True):
                rows.append({
                    "date": date,
                    "store_id": f"STORE_{store_idx:02d}",
                    "product_id": f"SKU_{product_idx:03d}",
                    "sales": int(qty),
                    "promo": int(is_promo),
                })
    return validate_sales_frame(pd.DataFrame(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=240, help="Number of daily observations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "raw" / "demo_sales.csv")
    args = parser.parse_args()

    data = generate_demo_sales(args.days, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    print(f"Wrote {len(data):,} synthetic rows to {args.output}")
    print("NOTE: This is synthetic demonstration data, not real retailer data.")


if __name__ == "__main__":
    main()
