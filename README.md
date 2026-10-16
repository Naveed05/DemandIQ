# DemandIQ — Demand Forecasting & Inventory Intelligence

DemandIQ is an end-to-end retail analytics project for forecasting product demand and turning forecasts into practical replenishment recommendations. It is being developed as a portfolio-grade engineering project with reproducible data, chronological evaluation, tests, and a lightweight dashboard.

> **Status:** Early development. Metrics and capabilities will be documented only after implementation and evaluation. Demo data is synthetic and must not be presented as real-world business evidence.

## Product goals
- Generate or ingest daily product/store sales data.
- Build time-aware features without using future information.
- Compare a simple baseline with a machine-learning forecasting model.
- Convert forecasts into interpretable reorder recommendations.
- Explore forecast quality and inventory signals in a Streamlit dashboard.
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
4. Report the evaluation window, metric definitions, and dataset source.
5. Clearly label synthetic-data experiments; they do not establish production performance.

See [docs/roadmap.md](docs/roadmap.md) for planned milestones.

## Disclaimer
DemandIQ is an educational and portfolio project, not a substitute for a retailer's validated production planning system. Reorder recommendations depend on assumptions about lead time, service level, and inventory position.
