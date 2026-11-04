# DemandIQ engineering roadmap

## Portfolio MVP release checklist

- [x] Package layout, installation instructions, and deterministic demo data
- [x] Input validation and leakage-aware lag/rolling feature engineering
- [x] Chronological forecast evaluation with MAE, RMSE, WAPE, and naive baselines
- [x] Recursive multi-day forecasting
- [x] SKU-level reorder point, safety buffer, stock cover, and stockout-risk recommendations
- [x] Streamlit dashboard with sales/inventory uploads, forecast comparisons, and CSV exports
- [x] Public UCI Online Retail adapter with provenance, license attribution, and documented transformation assumptions
- [x] Unit tests, linting, and CI on Python 3.11 and 3.12
- [x] CI end-to-end smoke test that generates demo data and produces all benchmark report files
- [x] CI Docker image build
- [x] Architecture, setup, data schemas, benchmark instructions, assumptions, and limitations documented

**MVP release status: complete.** The required portfolio scope is implemented. CI verifies tests, lint, the synthetic-data benchmark workflow, and Docker image construction. The synthetic dataset is for reproducible functional validation only; it is not evidence of real-world forecasting accuracy.

## Optional extensions — outside the MVP release scope

These are future product improvements, not unfinished requirements for the portfolio MVP:

- Run and publish a measured benchmark using the downloaded UCI Online Retail workbook. This requires obtaining the external workbook and reviewing the resulting metrics; never invent or imply real-data results.
- Capture dashboard screenshots from a locally running browser session for a richer portfolio presentation.
- Add database-backed persistence and schema migrations.
- Add FastAPI prediction endpoints.
- Add experiment tracking, model artifact versioning, drift detection, and data-quality monitoring.
- Extend inventory logic for purchase orders in transit, supplier minimum order quantities, pack sizes, expiry, storage capacity, promotions, and supplier lead-time uncertainty.

**Scope boundary:** DemandIQ is a complete, single-user, local portfolio MVP and decision-support demonstration. It is not represented as a production-validated retail replenishment platform. Any future extension should be scoped, implemented, tested, and documented before being described as complete.
