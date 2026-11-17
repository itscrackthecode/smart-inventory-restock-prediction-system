"""Portfolio overview using current SQLite inventory observations."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.charts import availability_chart, category_stock_chart, stock_trend_chart, store_performance_chart
from app.components.layout import section_heading
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
    st.markdown(
        "<section class='stockwise-hero'><h1>Know what needs attention before the shelf is empty.</h1>"
        "<p>Review the latest stock across every product and store, spot low-stock exceptions, and explore real historical sales signals in a separate workspace.</p></section>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<a class='stockwise-cta' href='/sales-intelligence'>Explore real retail sales <span>↗</span></a>",
        unsafe_allow_html=True,
    )
    db = str(database_path())
    counts = cached_counts(db)
    all_records = cached_inventory(db)
    if all_records.empty:
        st.info("No inventory observations are in the database yet. Initialize and seed the demo inventory to begin. Public sales analysis is available separately from the workspace navigation.")
        return
    latest = _latest_pairs(all_records)
    low_stock_limit = st.slider("Low-stock alert level (units)", min_value=1, max_value=25, value=5,
                                help="A dashboard review threshold only; the dataset does not contain a business reorder point.")
    low_stock = int(latest["stock_quantity"].between(1, low_stock_limit).sum())
    out_of_stock = int(latest["availability_status"].eq("out_of_stock").sum())
    recent_predictions = cached_recent_prediction_count(db, 7)

    section_heading("Inventory briefing", "Latest recorded observation for each product/store pair. Low-stock threshold is a review setting, not a business reorder point.")
    metric_cols = st.columns(4)
    for col, label, value, help_text in zip(
        metric_cols,
        ["Products tracked", "Stores tracked", f"Low stock ≤ {low_stock_limit}", "Out of stock"],
        [counts["products"], counts["stores"], low_stock, out_of_stock],
        ["Distinct products in the inventory catalog", "Distinct stores in the inventory catalog",
         "Latest product/store observations only; adjust the review threshold above",
         "Latest product/store observations only"],
    ):
        col.metric(label, f"{value:,}", help=help_text)
    latest_timestamp = latest["observation_date"].max()
    source_labels = ", ".join(sorted(latest["data_source"].dropna().astype(str).unique())) if "data_source" in latest else "unknown"
    st.caption(
        f"{len(latest):,} product/store pairs · latest snapshot {latest_timestamp:%b %d, %Y} · "
        f"sources: {source_labels} · {recent_predictions:,} model scores recorded in the past 7 days"
    )

    exceptions = latest.loc[
        latest["availability_status"].eq("out_of_stock")
        | latest["stock_quantity"].between(1, low_stock_limit)
    ].sort_values(["availability_status", "stock_quantity", "product_name"], kind="stable")
    section_heading("Needs a stock review", "Items shown here meet the selected low-stock threshold or are currently recorded as out of stock.")
    if exceptions.empty:
        st.success("No latest observations meet the selected review conditions.")
    else:
        st.dataframe(
            exceptions[["product_name", "product_id", "store_location", "store_id", "observation_date", "stock_quantity", "availability_status"]].head(12),
            hide_index=True, width="stretch",
            column_config={
                "observation_date": st.column_config.DatetimeColumn("Last observation", format="YYYY-MM-DD"),
                "stock_quantity": st.column_config.NumberColumn("Units on hand", format="%d"),
            },
        )
        if len(exceptions) > 12:
            st.caption(f"Showing 12 of {len(exceptions):,} product/store pairs needing review.")

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

    st.caption("The restock classifier was trained and evaluated on synthetic records. Treat its scores as experimental until tested on real inventory and replenishment outcomes.")
