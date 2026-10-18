"""Streamlit exploration dashboard for DemandIQ's synthetic demo dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from demandiq.data import validate_sales_frame
from demandiq.forecasting import evaluate_forecast, forecast_demand
from demandiq.inventory import reorder_recommendation
from scripts.make_demo_data import generate_demo_sales

st.set_page_config(page_title="DemandIQ", page_icon="📦", layout="wide")
st.title("📦 DemandIQ")
st.caption("Demand forecasting and inventory intelligence")

with st.sidebar:
    st.header("Sales data")
    uploaded_file = st.file_uploader(
        "Upload daily sales CSV",
        type=["csv"],
        help="Required columns: date, store_id, product_id, sales. Each series must have every calendar day.",
    )
    st.header("Scenario settings")
    days = st.slider("History length (days)", min_value=90, max_value=365, value=240, step=15)
    validation_days = st.slider("Validation window (days)", min_value=7, max_value=56, value=28, step=7)
    forecast_horizon = st.slider("Future forecast horizon (days)", min_value=7, max_value=28, value=14, step=7)
    lead_time = st.slider("Supplier lead time (days)", min_value=1, max_value=30, value=7)
    safety_days = st.slider("Safety stock (days of demand)", min_value=0, max_value=14, value=3)

try:
    if uploaded_file is not None:
        sales = validate_sales_frame(pd.read_csv(uploaded_file))
        st.caption("Data source: uploaded CSV")
    else:
        sales = generate_demo_sales(days=days)
        st.caption("Data source: synthetic demo data")
    evaluation = evaluate_forecast(sales, validation_days=validation_days)
    future_forecast = forecast_demand(sales, horizon_days=forecast_horizon)
except ValueError as exc:
    st.error(f"Unable to generate demand analysis: {exc}")
    st.stop()

metric_cols = st.columns(4)
metric_cols[0].metric("Validation MAE", f'{evaluation.metrics["mae"]:.2f} units')
metric_cols[1].metric("Validation RMSE", f'{evaluation.metrics["rmse"]:.2f} units')
metric_cols[2].metric("Model WAPE", f'{evaluation.metrics["wape"]:.1%}')
metric_cols[3].metric("Weekly-lag baseline WAPE", f'{evaluation.metrics["baseline_wape"]:.1%}')

if uploaded_file is None:
    st.info(
        "These results use deterministic synthetic data. Validation uses a rolling "
        "one-step-ahead holdout; future forecasts are recursive multi-step predictions. "
        "Synthetic results do not establish real retail performance."
    )
else:
    st.info(
        "Results use the uploaded sales CSV. Validation uses a rolling one-step-ahead "
        "holdout; future forecasts are recursive multi-step predictions. Check the "
        "dataset and evaluation window before using results for business decisions."
    )

left, right = st.columns([1.5, 1])
with left:
    st.subheader("Daily sales trend")
    daily = sales.groupby("date", as_index=False)["sales"].sum()
    st.plotly_chart(
        px.line(
            daily,
            x="date",
            y="sales",
            title="Synthetic units sold across all stores and products",
        ),
        use_container_width=True,
    )

with right:
    st.subheader("Validation: forecast vs actual")
    comparison = evaluation.predictions.groupby("date", as_index=False)[
        ["sales", "prediction", "baseline_prediction"]
    ].sum()
    long_comparison = comparison.melt(
        id_vars="date", var_name="series", value_name="units"
    )
    st.plotly_chart(
        px.line(long_comparison, x="date", y="units", color="series"),
        use_container_width=True,
    )

st.subheader(f"Future demand forecast · next {forecast_horizon} days")
future_daily = future_forecast.groupby("date", as_index=False)["prediction"].sum()
future_daily = future_daily.rename(columns={"prediction": "forecast_units"})
st.plotly_chart(
    px.line(
        future_daily,
        x="date",
        y="forecast_units",
        markers=True,
        title="Predicted units across all stores and products",
    ),
    use_container_width=True,
)
sku_forecast = future_forecast.pivot_table(
    index=["store_id", "product_id"],
    columns="forecast_step",
    values="prediction",
    aggfunc="sum",
).reset_index()
sku_forecast.columns = [
    str(column) if isinstance(column, str) else f"day_{column}"
    for column in sku_forecast.columns
]
st.dataframe(sku_forecast, use_container_width=True, hide_index=True)
st.download_button(
    "Download future forecasts (CSV)",
    data=future_forecast.to_csv(index=False).encode("utf-8"),
    file_name="demandiq_future_forecasts.csv",
    mime="text/csv",
)

st.subheader("Inventory recommendations")
st.caption("Current stock is a scenario input, not measured inventory from the synthetic sales data.")
stock_default = int(max(1, sales["sales"].mean() * 5))
current_stock = st.number_input(
    "Assumed current stock per selected SKU (units)",
    min_value=0,
    value=stock_default,
    step=1,
)
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
