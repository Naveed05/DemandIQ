"""Streamlit dashboard for DemandIQ demand forecasting and inventory planning."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demandiq.data import validate_sales_frame
from demandiq.benchmark import build_benchmark_report, build_benchmark_summary
from demandiq.forecasting import evaluate_forecast, forecast_demand
from demandiq.inventory import reorder_recommendation
from scripts.make_demo_data import generate_demo_sales


@st.cache_data(show_spinner=False)
def cached_benchmark_report(
    sales_frame: pd.DataFrame, validation_window: int
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Cache benchmark outputs for unchanged data and validation settings."""
    return build_benchmark_report(sales_frame, validation_days=validation_window)

st.set_page_config(page_title="DemandIQ | Retail Intelligence", page_icon="📦", layout="wide")
st.title("📦 DemandIQ")
st.caption("Retail demand forecasting · Model evaluation · Inventory replenishment")

with st.sidebar:
    st.header("Data")
    uploaded_file = st.file_uploader(
        "Daily sales CSV",
        type=["csv"],
        help="Required: date, store_id, product_id, sales. Include every calendar day per SKU.",
    )
    inventory_file = st.file_uploader(
        "Optional inventory CSV",
        type=["csv"],
        help="Required: store_id, product_id, current_stock. One unique row per store/SKU.",
    )
    st.divider()
    st.header("Forecast settings")
    days = st.slider("Demo history (days)", 90, 365, 240, 15)
    validation_days = st.slider("Validation window (days)", 7, 56, 28, 7)
    forecast_horizon = st.slider("Forecast horizon (days)", 7, 28, 14, 7)
    st.divider()
    st.header("Inventory policy")
    lead_time = st.slider("Supplier lead time (days)", 1, 30, 7)
    safety_days = st.slider("Extra safety buffer (days)", 0, 14, 3)
    service_level_z = st.select_slider(
        "Variability buffer multiplier",
        options=[0.0, 0.84, 1.04, 1.28, 1.65, 2.05, 2.33],
        value=1.65,
        help="Approximate z multiplier. This is a planning assumption, not a guaranteed service level.",
    )
    fallback_stock = st.number_input(
        "Fallback stock per SKU (if inventory file is absent)",
        min_value=0,
        value=50,
        step=1,
    )

try:
    if uploaded_file is not None:
        sales = validate_sales_frame(pd.read_csv(uploaded_file))
        data_label = "Uploaded sales data"
    else:
        sales = generate_demo_sales(days=days)
        data_label = "Synthetic demonstration data"

    evaluation = evaluate_forecast(sales, validation_days=validation_days)
    future_forecast = forecast_demand(sales, horizon_days=forecast_horizon)
except (ValueError, TypeError, pd.errors.ParserError) as exc:
    st.error(f"Could not prepare the analysis: {exc}")
    st.stop()

st.caption(f"Data source: **{data_label}** · {sales['store_id'].nunique()} locations · "
           f"{sales['product_id'].nunique()} products · {len(sales):,} daily rows")

model_wape = evaluation.metrics["wape"]
seasonal_wape = evaluation.metrics["seasonal_naive_wape"]
improvement = (
    (seasonal_wape - model_wape) / seasonal_wape * 100
    if seasonal_wape > 0 else 0.0
)
metric_cols = st.columns(4)
metric_cols[0].metric("Model MAE", f"{evaluation.metrics['mae']:.2f} units")
metric_cols[1].metric("Model RMSE", f"{evaluation.metrics['rmse']:.2f} units")
metric_cols[2].metric("Model WAPE", f"{model_wape:.1%}")
metric_cols[3].metric(
    "vs 7-day baseline",
    f"{improvement:+.1f}%",
    help="Positive means lower WAPE than the seasonal-naive baseline on this holdout.",
)

if uploaded_file is None:
    st.info(
        "Demo mode uses synthetic data. It is useful for exploring the workflow, "
        "not evidence of real-world forecasting accuracy."
    )
else:
    st.info(
        "Validation is a chronological rolling one-step-ahead holdout: actual earlier "
        "validation days are available as lag inputs for later validation days. "
        "The future horizon below is recursive and uses prior predictions as inputs."
    )

tab_forecast, tab_inventory, tab_method = st.tabs(
    ["Demand forecast", "Inventory intelligence", "Method & assumptions"]
)

with tab_forecast:
    trend_col, validation_col = st.columns([1.25, 1])
    with trend_col:
        st.subheader("Historical demand")
        daily = sales.groupby("date", as_index=False)["sales"].sum()
        st.plotly_chart(
            px.line(daily, x="date", y="sales", title="Total daily units sold"),
            use_container_width=True,
        )
    with validation_col:
        st.subheader("Validation performance")
        comparison = evaluation.predictions.groupby("date", as_index=False)[
            ["sales", "prediction", "seasonal_naive_prediction"]
        ].sum()
        comparison = comparison.rename(
            columns={"sales": "Actual", "prediction": "Random Forest",
                     "seasonal_naive_prediction": "7-day seasonal naive"}
        )
        long_comparison = comparison.melt(
            id_vars="date", var_name="series", value_name="units"
        )
        st.plotly_chart(
            px.line(long_comparison, x="date", y="units", color="series"),
            use_container_width=True,
        )

    st.subheader(f"Future demand · next {forecast_horizon} days")
    future_daily = future_forecast.groupby("date", as_index=False)["prediction"].sum()
    future_daily = future_daily.rename(columns={"prediction": "forecast_units"})
    st.plotly_chart(
        px.line(future_daily, x="date", y="forecast_units", markers=True,
                title="Predicted units across all store/product series"),
        use_container_width=True,
    )
    st.dataframe(future_forecast, use_container_width=True, hide_index=True)
    st.download_button(
        "Download future forecasts",
        data=future_forecast.to_csv(index=False).encode("utf-8"),
        file_name="demandiq_future_forecasts.csv",
        mime="text/csv",
    )

with tab_inventory:
    st.subheader("SKU-level replenishment recommendations")
    st.write(
        "Recommendations combine lead-time demand, a days-of-demand safety buffer, "
        "and recent demand variability. They are estimates based on the inputs below."
    )
    if inventory_file is not None:
        try:
            inventory = pd.read_csv(inventory_file)
            required_inventory = {"store_id", "product_id", "current_stock"}
            missing = required_inventory.difference(inventory.columns)
            if missing:
                raise ValueError(
                    f"Inventory CSV is missing columns: {', '.join(sorted(missing))}"
                )
            inventory = inventory[list(required_inventory)].copy()
            inventory["store_id"] = inventory["store_id"].astype("string").str.strip()
            inventory["product_id"] = inventory["product_id"].astype("string").str.strip()
            inventory["current_stock"] = pd.to_numeric(
                inventory["current_stock"], errors="raise"
            )
            if (
                inventory[["store_id", "product_id", "current_stock"]].isna().any().any()
                or (inventory["current_stock"] < 0).any()
                or not np.isfinite(inventory["current_stock"].to_numpy(dtype=float)).all()
            ):
                raise ValueError("Inventory fields must be present, finite, and non-negative.")
            if inventory.duplicated(["store_id", "product_id"]).any():
                raise ValueError("Inventory CSV has duplicate store_id/product_id rows.")
            stock_source = "uploaded inventory CSV"
        except (ValueError, TypeError, pd.errors.ParserError) as exc:
            st.error(f"Invalid inventory file: {exc}")
            st.stop()
    else:
        inventory = pd.DataFrame(columns=["store_id", "product_id", "current_stock"])
        stock_source = "fallback scenario value"

    sku_summary = sales.groupby(["store_id", "product_id"], as_index=False).agg(
        average_daily_demand=("sales", "mean"),
        demand_std_dev=("sales", lambda values: values.tail(28).std(ddof=0)),
    )
    sku_summary = sku_summary.merge(
        inventory, on=["store_id", "product_id"], how="left"
    )
    sku_summary["current_stock"] = sku_summary["current_stock"].fillna(fallback_stock)

    rows = []
    for row in sku_summary.itertuples(index=False):
        result = reorder_recommendation(
            average_daily_demand=float(row.average_daily_demand),
            lead_time_days=int(lead_time),
            current_stock=float(row.current_stock),
            safety_stock_days=float(safety_days),
            demand_std_dev=float(row.demand_std_dev or 0.0),
            service_level_z=float(service_level_z),
        )
        rows.append({
            "store_id": row.store_id,
            "product_id": row.product_id,
            "avg_daily_demand": round(row.average_daily_demand, 2),
            "recent_demand_std": round(row.demand_std_dev or 0.0, 2),
            "current_stock": row.current_stock,
            "days_of_cover": result["days_of_cover"],
            "reorder_point": result["reorder_point"],
            "suggested_order_qty": result["order_quantity"],
            "stockout_risk": result["stockout_risk"],
            "reorder_required": result["reorder_required"],
        })
    recommendations = pd.DataFrame(rows).sort_values(
        ["reorder_required", "stockout_risk", "suggested_order_qty"],
        ascending=[False, True, False],
    )
    high_risk = int(recommendations["stockout_risk"].eq("High").sum())
    reorder_count = int(recommendations["reorder_required"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("SKUs to reorder", reorder_count)
    c2.metric("High stockout-risk SKUs", high_risk)
    c3.metric("Inventory source", stock_source)
    st.dataframe(recommendations, use_container_width=True, hide_index=True)
    st.download_button(
        "Download replenishment plan",
        data=recommendations.to_csv(index=False).encode("utf-8"),
        file_name="demandiq_replenishment_plan.csv",
        mime="text/csv",
    )
    st.caption(
        "If no inventory file is supplied, the fallback stock value is applied to every SKU. "
        "Missing SKUs in an uploaded inventory file also use that fallback. Confirm actual "
        "on-hand stock, purchase orders, pack sizes, supplier constraints, and expiry before ordering."
    )

with tab_method:
    st.subheader("Forecast model comparison")
    with st.spinner("Evaluating forecasting strategies..."):
        benchmark_report, _, per_series_benchmark = cached_benchmark_report(
            sales, validation_days
        )
    benchmark_metrics = (
        pd.DataFrame(benchmark_report["metrics"])
        .rename_axis("method")
        .reset_index()
    )
    st.dataframe(benchmark_metrics, use_container_width=True, hide_index=True)
    st.caption(
        "Adaptive selection is determined from pre-validation demand patterns only. "
        "The holdout is used for evaluation, never for choosing the method."
    )
    st.subheader("Per-SKU demand pattern and selected method")
    st.dataframe(per_series_benchmark, use_container_width=True, hide_index=True)
    st.download_button(
        "Download per-SKU benchmark results",
        data=per_series_benchmark.to_csv(index=False).encode("utf-8"),
        file_name="demandiq_per_sku_benchmark.csv",
        mime="text/csv",
    )
    with st.expander("Benchmark interpretation"):
        st.markdown(build_benchmark_summary(benchmark_report))

    st.divider()
    st.subheader("How DemandIQ works")
    st.markdown(
        """
        1. **Validate:** check required fields, non-negative sales, duplicate records,
           and complete daily calendars for each store/SKU.
        2. **Engineer features:** calendar signals, 1/7/14/28-day lags, and shifted
           seven-day rolling statistics.
        3. **Evaluate:** fit a Random Forest on earlier dates and compare it with
           last-day and seven-day seasonal-naive baselines on a chronological holdout.
        4. **Forecast:** recursively predict future days, feeding each prediction
           into the next day's lag features.
        5. **Plan inventory:** calculate lead-time demand, a configurable safety
           buffer, a variability buffer, days of cover, and a suggested order quantity.
        """
    )
    st.subheader("Important assumptions")
    st.markdown(
        """
        - A missing daily row is not automatically treated as zero sales.
        - The demo dataset is synthetic; uploaded data should be checked before use.
        - WAPE can hide poor performance on individual SKUs; inspect SKU-level errors.
        - Inventory policy does not currently model purchase orders, supplier minimums,
          pack sizes, shelf life, promotions, lost sales, or capacity constraints.
        - The service-level multiplier is a planning parameter, not a calibrated guarantee.
        """
    )

st.divider()
st.caption("DemandIQ · Portfolio MVP · Forecasts and reorder quantities are decision support, not purchase instructions.")
