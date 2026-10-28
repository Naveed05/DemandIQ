# DemandIQ engineering roadmap

## Phase 1 — Reliable foundation
- [x] Package layout and setup instructions
- [x] Input validation and deterministic synthetic demo data
- [x] Leakage-aware lag and rolling features
- [x] Chronological holdout evaluation with baseline metrics
- [x] Reorder-point calculator and first dashboard
- [x] Unit tests for core data and inventory behavior

## Phase 2 — Evaluation quality
- [ ] Run tests in a clean environment and fix any failures
- [ ] Add forecast evaluation plots and segment-level metrics
- [ ] Add explicit model-vs-baseline comparison and error analysis
- [ ] Add robust handling for intermittent and zero demand
- [ ] Add reproducible experiment configuration

## Phase 3 — Real data and persistence
- [x] Add a documented adapter for a public retail dataset (UCI Online Retail)
- [x] Document license, provenance, transformations, and limitations for the adapter
- [ ] Add database-backed storage and schema migrations
- [ ] Separate demo-data generation from application runtime

## Phase 4 — Production-style engineering
- [ ] Add FastAPI prediction endpoints and request validation
- [ ] Add Docker image and local compose workflow
- [ ] Add CI, linting, and test coverage reporting
- [ ] Add experiment tracking and model artifact versioning
- [ ] Add drift and data-quality monitoring

## Phase 5 — Portfolio polish
- [ ] Add architecture diagram and reproducible demo walkthrough
- [ ] Capture dashboard screenshots from a running build
- [ ] Document trade-offs, assumptions, and known limitations
- [ ] Publish only measured metrics with the dataset and evaluation window stated

**Rule:** A roadmap checkbox is not evidence of implementation. Mark items complete only after the change exists and is verified.
