# DemandIQ engineering roadmap

## Portfolio MVP release checklist

- [x] Package layout, installation instructions, and deterministic demo data
- [x] Input validation and leakage-aware lag/rolling feature engineering
- [x] Chronological forecast evaluation with MAE, RMSE, WAPE, last-day and seasonal-naive baselines
- [x] Croston SBA intermittent-demand benchmark with leakage-safe expanding-history predictions and unit tests\n- [x] SKU demand-pattern segmentation (smooth, erratic, intermittent, lumpy, no demand) using training-history ADI and CV²
- [x] Recursive multi-day forecasting
- [x] SKU-level reorder point, safety buffer, stock cover, and stockout-risk recommendations
- [x] Streamlit dashboard with sales/inventory uploads, forecast comparisons, and CSV exports
- [x] Public UCI Online Retail adapter with provenance, license attribution, and documented transformation assumptions
- [x] Human-readable benchmark insights report with per-metric winners and baseline comparison
- [x] Unit tests, linting, and CI on Python 3.11 and 3.12
- [x] CI end-to-end smoke test that generates demo data and produces JSON, CSV, and Markdown benchmark reports
- [x] CI Docker image build
- [x] Architecture, setup, data schemas, benchmark instructions, assumptions, and limitations documented

**MVP release status: complete.** CI verifies tests, lint, the synthetic-data benchmark workflow, and Docker image construction. The synthetic dataset is for reproducible functional validation only; it is not evidence of real-world forecasting accuracy.

## Optional extensions — outside the MVP release scope

- [x] Run and publish a measured benchmark using the downloaded UCI Online Retail workbook. The committed report covers the top 100 eligible UK products and explicitly records the preprocessing and evaluation assumptions.
- [x] Re-run the UCI Online Retail benchmark with Croston SBA and publish the four-method comparison, aggregate metrics, and per-series error distributions.
- Capture dashboard screenshots from a locally running browser session for a richer portfolio presentation.
- Add database-backed persistence and schema migrations.
- Add FastAPI prediction endpoints.
- Add experiment tracking, model artifact versioning, drift detection, and data-quality monitoring.
- Extend inventory logic for purchase orders in transit, supplier minimum order quantities, pack sizes, expiry, storage capacity, promotions, lost sales, and supplier lead-time uncertainty.

## Demand segmentation milestone\n\nThe benchmark now classifies each product using ADI and CV² calculated strictly from pre-validation training history. It exports the pattern label and numeric statistics alongside each product's model metrics, enabling future per-pattern model analysis without leaking holdout information.\n\n## Real-data benchmark milestone

The UCI Online Retail benchmark has been run on the selected top-100 product slice (35,082 daily rows, 100 series, 28-day chronological holdout). Croston SBA achieved the best aggregate MAE, RMSE, and WAPE among the four tested methods in this selected historical slice. See `reports/uci_online_retail/benchmark_summary.md` and `metrics.json`. This is a measured historical benchmark, not evidence of production readiness.

**Scope boundary:** DemandIQ is a complete, single-user, local portfolio MVP and decision-support demonstration. It is not represented as a production-validated retail replenishment platform. Any future extension should be scoped, implemented, tested, and documented before being described as complete.
