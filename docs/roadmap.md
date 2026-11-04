# DemandIQ engineering roadmap

## Phase 1 — Reliable foundation
- [x] Package layout and setup instructions
- [x] Input validation and deterministic synthetic demo data
- [x] Leakage-aware lag and rolling features
- [x] Chronological holdout evaluation with baseline metrics
- [x] Reorder-point calculator and first dashboard
- [x] Unit tests for core data and inventory behavior

## Phase 2 — Evaluation quality
- [x] Run tests in CI and fix failures
- [x] Add forecast evaluation plots and segment-level benchmark metrics
- [x] Add explicit model-vs-baseline comparison
- [x] Add transparent handling of zero-demand and intermittent series (zero rows retained; interpret per-SKU metrics carefully)
- [x] Add reproducible benchmark configuration via CLI arguments and fixed model seed

## Phase 3 — Real data and persistence
- [x] Add a documented adapter for a public retail dataset (UCI Online Retail)
- [x] Document license, provenance, transformations, and limitations for the adapter
- [ ] Run and publish a benchmark on the downloaded UCI workbook
- [ ] Add database-backed storage and schema migrations
- [x] Keep synthetic data generation separate from the application package (script imported by dashboard)

## Phase 4 — Production-style engineering
- [x] Add CI, linting, and automated tests (Python 3.11 and 3.12)
- [x] Add a Docker image and local Compose workflow
- [ ] Add FastAPI prediction endpoints (not required for the Streamlit MVP)
- [ ] Add experiment tracking and model artifact versioning
- [ ] Add drift and data-quality monitoring

## Phase 5 — Portfolio polish
- [x] Add architecture diagram and reproducible demo walkthrough
- [x] Document trade-offs, assumptions, and known limitations
- [ ] Capture dashboard screenshots from a running build
- [ ] Publish only measured real-data metrics with the dataset and evaluation window stated

**Release scope:** This repository is a complete portfolio MVP for a single-user local demo. It is not a production inventory platform. Remaining unchecked items require real dataset access, a running visual capture environment, or a deliberate expansion beyond the current scope.

**Rule:** A roadmap checkbox is not evidence of implementation. Mark items complete only after the change exists and is verified.
