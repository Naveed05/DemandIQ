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
