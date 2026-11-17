"""Portfolio overview using current SQLite inventory observations."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.charts import availability_chart, category_stock_chart, stock_trend_chart, store_performance_chart
from app.components.layout import page_header, section_heading
from app.config import database_path
from app.services.cached_data import (
    cached_counts, cached_inventory, cached_recent_prediction_count,
)


def _latest_pairs(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return records
    return records.sort_values("observation_date").groupby(
        ["product_id", "store_id"], as_index=False, sort=False
    ).tail(1)


def render() -> None:
    page_header("Inventory overview", "A current, database-backed view of stock across products and stores.")
    db = str(database_path())
    counts = cached_counts(db)
    all_records = cached_inventory(db)
    if all_records.empty:
        st.info("No inventory observations are in the database yet. Initialize and seed it from the processed CSV to begin.")
        return
    latest = _latest_pairs(all_records)
    low_stock_limit = st.slider("Low-stock alert level (units)", min_value=1, max_value=25, value=5,
                                help="A dashboard review threshold only; the dataset does not contain a business reorder point.")
    low_stock = int(latest["stock_quantity"].between(1, low_stock_limit).sum())
    out_of_stock = int(latest["availability_status"].eq("out_of_stock").sum())
    recent_predictions = cached_recent_prediction_count(db, 7)

    metric_cols = st.columns(5)
    for col, label, value, help_text in zip(
        metric_cols,
        ["Products", "Stores", "Inventory records", f"Low stock ≤ {low_stock_limit}", "Out of stock"],
        [counts["products"], counts["stores"], counts["inventory_records"], low_stock, out_of_stock],
        ["Distinct product records", "Distinct store records", "All daily observations in SQLite",
         "Latest product/store observations only", "Latest product/store observations only"],
    ):
        col.metric(label, f"{value:,}", help=help_text)
    st.caption(f"{len(latest):,} latest product/store observations · {recent_predictions:,} predictions recorded in the past 7 days")

    left, right = st.columns(2, gap="large")
    with left:
        section_heading("Availability snapshot", "Each product/store pair contributes only its latest stored observation.")
        st.altair_chart(availability_chart(latest), width="stretch")
    with right:
        section_heading("Inventory trend", "Daily mean stock quantity across observations; this descriptive trend does not imply a cause.")
        st.altair_chart(stock_trend_chart(all_records), width="stretch")

    left, right = st.columns(2, gap="large")
    with left:
        section_heading("Stock by category", "Mean quantity in the latest observation for each product/store pair.")
        st.altair_chart(category_stock_chart(latest), width="stretch")
    with right:
        section_heading("Store overview", "Latest mean stock and out-of-stock pair counts by store.")
        st.altair_chart(store_performance_chart(latest), width="stretch")

    with st.expander("Latest stock exceptions"):
        exceptions = latest.loc[
            latest["availability_status"].eq("out_of_stock")
            | latest["stock_quantity"].between(1, low_stock_limit)
        ].sort_values(["availability_status", "stock_quantity", "product_name"])
        if exceptions.empty:
            st.success("No latest observations meet the selected low-stock or out-of-stock conditions.")
        else:
            st.dataframe(exceptions[["product_id", "product_name", "store_id", "store_location",
                                     "observation_date", "stock_quantity", "availability_status"]],
                         hide_index=True, width="stretch")
