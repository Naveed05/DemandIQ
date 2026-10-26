"""Benchmark DemandIQ against simple daily-demand baselines on a prepared CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from demandiq.benchmark import build_benchmark_report


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Prepared daily sales CSV")
    parser.add_argument(
        "--validation-days",
        type=int,
        default=28,
        help="Number of final calendar dates held out for evaluation (default: 28)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "benchmark",
        help="Directory for JSON metrics and CSV prediction reports",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    report, predictions, per_series = build_benchmark_report(
        frame, validation_days=args.validation_days
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    per_series.to_csv(args.output_dir / "per_series_metrics.csv", index=False)

    print(f"Benchmark reports written to: {args.output_dir}")
    print(json.dumps(report["metrics"], indent=2))
    print("No performance claim is made beyond this dataset and evaluation setup.")


if __name__ == "__main__":
    main()
