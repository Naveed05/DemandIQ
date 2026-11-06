# DemandIQ — UCI Online Retail benchmark

## Dataset and evaluation scope

- Source: [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail) (CC BY 4.0; see source page for attribution).
- Country filter: United Kingdom. The dataset has no store identifier; country is used only as a location proxy.
- Eligible products with at least 60 calendar days between first and last positive sale: 3,509.
- Selected products: top 100 by total positive units (deterministic product-code tie-break).
- Prepared daily rows: 35,082 across 100 product series.
- Date range: 2010-12-01 to 2011-12-09.
- Validation: final 28 calendar dates; 2,769 product-day predictions.
- Target: units sold (Quantity), not revenue.
- Returns/refunds (non-positive quantities) and cancellation invoices are excluded. Missing dates within each product's observed span are filled with zero; this is not evidence that inventory was available.

Evaluation is a chronological rolling one-step-ahead holdout. For later validation dates, previously observed validation sales are available to lag features. Croston SBA is recalculated at each validation timestamp using only observations strictly before that timestamp. This is not the same as a fully recursive 28-day forecast.

## Overall metrics

| Method | MAE (units) | RMSE (units) | WAPE |
| --- | ---: | ---: | ---: |
| Random Forest | 66.874 | 266.237 | 102.35% |
| Naive (last day) | 79.634 | 372.643 | 121.88% |
| Seasonal naive (7-day) | 66.499 | 362.975 | 101.78% |
| Croston SBA (alpha=0.1) | **58.921** | **264.397** | **90.18%** |

## Findings

- **Best MAE:** Croston SBA (58.921).
- **Best RMSE:** Croston SBA (264.397).
- **Best WAPE:** Croston SBA (90.18%).
- Croston SBA WAPE is **11.40% lower** than the best of the original naive baselines (seasonal naive, 101.78%).
- Croston SBA achieves lower aggregate MAE, RMSE, and WAPE than Random Forest on this selected historical holdout.
- Croston SBA has the lowest per-series WAPE on 64 of 100 selected product series (ties included).
- WAPE below 100% means aggregate absolute error is below total actual units for this holdout; it does not mean forecasts are accurate for every SKU.

## Interpretation and limitations

This is a measured benchmark on a selected top-100 product slice, not a claim about every product in the UCI dataset or a live retailer. Croston SBA is a useful benchmark candidate for this slice, but its performance must be checked across other product selections, time periods, and retailers before any production model-selection decision. It is not automatically used by the dashboard forecast.

Country is only a location proxy, not a store. The data is historical (2010–2011), returns/cancellations are excluded, and zero-filled gaps can conflate no demand with unavailable inventory. The chronological evaluation uses observed prior holdout actuals for later lag features; multi-step recursive performance may differ. Croston SBA uses a fixed alpha of 0.1 and has not been tuned.

## Reproducibility

```bash
pip install -e ".[dev,data]"
python scripts/prepare_uci_online_retail.py --input "path/to/Online Retail.xlsx" --country "United Kingdom" --min-history-days 60 --max-products 100
python scripts/benchmark.py --input data/processed/uci_online_retail_daily.csv --validation-days 28 --output-dir reports/uci_online_retail
```

The benchmark command also writes full row-level predictions and per-series metrics locally. The repository commits the compact summary and aggregate metrics, not the full row-level prediction dump or source workbook.
