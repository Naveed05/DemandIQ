# DemandIQ — Demand Forecasting & Inventory Intelligence

DemandIQ is an end-to-end retail analytics project for forecasting product demand and turning forecasts into practical replenishment recommendations. It is being developed as a portfolio-grade engineering project with reproducible data, chronological evaluation, tests, and a lightweight dashboard.

> **Status:** Early development. Metrics and capabilities will be documented only after implementation and evaluation. Demo data is synthetic and must not be presented as real-world business evidence.

## Product goals
- Generate synthetic demo sales or upload a daily sales CSV with `date`, `store_id`, `product_id`, and `sales` columns.
- Build time-aware features without using future information.
- Compare a simple baseline with a machine-learning forecasting model.
- Convert forecasts into interpretable reorder recommendations.
- Generate recursive multi-day forecasts for each store/product series.
- Explore forecast quality, future demand, and inventory signals in a Streamlit dashboard.
- Add automated tests and CI as the project matures.

## Architecture (initial)
```text
Data source → Validation & features → Chronological evaluation
            → Forecasts → Inventory policy → Streamlit dashboard
```

## Quick start
Python 3.11 or newer is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
python scripts/make_demo_data.py
streamlit run app.py
```

Run tests with `pytest`.

## Uploading your own data
In the dashboard sidebar, upload a CSV with these required columns:

| Column | Meaning |
| --- | --- |
| `date` | Calendar date for the observation |
| `store_id` | Store or location identifier |
| `product_id` | Product/SKU identifier |
| `sales` | Non-negative daily units sold |

Each store/product series must have one row for every calendar day in its history. If a product was not sold on a day, include a zero-sales row when that is the correct interpretation; do not silently treat missing records as zero. Duplicate store/product/day rows are rejected. At least 29 days of history are needed to construct lag features, and more history is generally preferable.

## Running a forecasting benchmark

The benchmark command evaluates a prepared daily-sales CSV on the same chronological holdout for three methods: the Random Forest model, a last-observed-day naive forecast, and a seasonal-naive forecast using sales from seven days earlier.

```bash
python scripts/benchmark.py --input path/to/daily_sales.csv --validation-days 28
```

By default, reports are written to `reports/benchmark/`:
- `metrics.json` — dataset summary, evaluation method, and overall MAE/RMSE/WAPE for each method
- `predictions.csv` — actual sales and each method's predictions for each validation row
- `per_series_metrics.csv` — metrics split by store and product

The input must be a prepared daily dataset with `date`, `store_id`, `product_id`, and non-negative `sales` columns. Each store/product series must contain every calendar day, including explicit zero-sales rows when appropriate; duplicate store/product/day records are rejected. The tool does not download or silently transform a raw transaction dataset for you. Document the source and licence of any external dataset you use, and record any aggregation/cleaning steps separately.

The evaluation is rolling one-step-ahead: actual observations from earlier dates in the validation window are available as lag inputs for later validation dates. This is not the same as evaluating a fully recursive multi-day forecast. Do not claim that the model is better unless the measured results support that conclusion.

## Project layout
- `src/demandiq/` — reusable data, feature, forecasting, and inventory logic
- `scripts/` — reproducible demo-data generation
- `tests/` — unit tests
- `docs/` — project plan and engineering notes
- `app.py` — exploratory dashboard

## Data and evaluation principles
1. Keep train/validation ordering chronological.
2. Calculate features using only information available at prediction time.
3. Compare against a transparent baseline.
4. Treat the validation score as rolling one-step-ahead performance; future multi-day forecasts are recursive and use prior predictions as lag inputs.
5. Report the evaluation window, metric definitions, and dataset source.
6. Clearly label synthetic-data experiments; they do not establish production performance.

See [docs/roadmap.md](docs/roadmap.md) for planned milestones.

## Disclaimer
DemandIQ is an educational and portfolio project, not a substitute for a retailer's validated production planning system. Reorder recommendations depend on assumptions about lead time, service level, and inventory position.
