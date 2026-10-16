"""Streamlit exploration dashboard for the DemandIQ demo dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from demandiq.forecasting import evaluate_forecast
from demandiq.inventory import reorder_recommendation
from scripts.make_demo_data import generate_demo_sales

st.set_page_config(page_title="DemandIQ", page_icon="📦", layout="wide")
st.title("📦 DemandIQ")
st.caption("Demand forecasting and inventory intelligence · Synthetic demo data")

with st.sidebar:
    st.header("Scenario settings")
    days = st.slider("History length (days)", min_value=90, max_value=365, value=240, step=15)
    validation_days = st.slider("Validation window (days)", min_value=7, max_value=56, value=28, step=7)
    lead_time = st.slider("Supplier lead time (days)", min_value=1, max_value=30, value=7)
    safety_days = st.slider("Safety stock (days of demand)", min_value=0, max_value=14, value=3)

sales = generate_demo_sales(days=days)
try:
    evaluation = evaluate_forecast(sales, validation_days=validation_days)
except ValueError as exc:
    st.error(f"Unable to evaluate forecast: {exc}")
    st.stop()

metric_cols = st.columns(4)
metric_cols[0].metric("Validation MAE", f'{evaluation.metrics["mae"]:.2f} units')
metric_cols[1].metric("Validation RMSE", f'{evaluation.metrics["rmse"]:.2f} units')
metric_cols[2].metric("Model WAPE", f'{evaluation.metrics["wape"]:.1%}')
metric_cols[3].metric("Weekly-lag baseline WAPE", f'{evaluation.metrics["baseline_wape"]:.1%}')

st.info("These results are from deterministic synthetic data and a rolling one-step-ahead holdout. They are not evidence of real retail performance.")

left, right = st.columns([1.5, 1])
with left:
    st.subheader("Daily sales trend")
    daily = sales.groupby("date", as_index=False)["sales"].sum()
    st.plotly_chart(px.line(daily, x="date", y="sales", title="Synthetic units sold across all stores and products"), use_container_width=True)

with right:
    st.subheader("Forecast vs actual")
    comparison = evaluation.predictions.groupby("date", as_index=False)[["sales", "prediction", "baseline_prediction"]].sum()
    long_comparison = comparison.melt(id_vars="date", var_name="series", value_name="units")
    st.plotly_chart(px.line(long_comparison, x="date", y="units", color="series"), use_container_width=True)

st.subheader("Inventory recommendations")
st.caption("Current stock is a scenario input, not measured inventory from the synthetic sales data.")
stock_default = int(max(1, sales["sales"].mean() * 5))
current_stock = st.number_input("Assumed current stock per selected SKU (units)", min_value=0, value=stock_default, step=1)
sku_summary = sales.groupby(["store_id", "product_id"], as_index=False).agg(
    average_daily_demand=("sales", "mean"),
    last_28_day_units=("sales", lambda values: values.tail(28).sum()),
)
sku_summary["current_stock"] = current_stock
recommendations = []
for row in sku_summary.itertuples(index=False):
    result = reorder_recommendation(
        average_daily_demand=float(row.average_daily_demand),
        lead_time_days=int(lead_time),
        current_stock=float(row.current_stock),
        safety_stock_days=float(safety_days),
    )
    recommendations.append({
        "store_id": row.store_id,
        "product_id": row.product_id,
        "average_daily_demand": round(row.average_daily_demand, 2),
        "current_stock": row.current_stock,
        "reorder_point": result["reorder_point"],
        "suggested_order_qty": result["order_quantity"],
        "reorder_required": result["reorder_required"],
    })
recommendation_frame = pd.DataFrame(recommendations).sort_values(
    ["reorder_required", "suggested_order_qty"], ascending=[False, False]
)
st.dataframe(recommendation_frame, use_container_width=True, hide_index=True)
st.download_button(
    "Download inventory recommendations (CSV)",
    data=recommendation_frame.to_csv(index=False).encode("utf-8"),
    file_name="demandiq_inventory_recommendations.csv",
    mime="text/csv",
)
