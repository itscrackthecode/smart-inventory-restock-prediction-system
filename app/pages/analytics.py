"""Interactive descriptive analytics over SQLite inventory observations."""

from __future__ import annotations

import streamlit as st

from app.components.charts import (
    category_stock_chart,
    category_store_heatmap,
    monthly_pattern_chart,
    product_trend_chart,
    restock_frequency_chart,
    stock_distribution_chart,
    stock_trend_chart,
    store_performance_chart,
    weekday_pattern_chart,
)
from app.components.layout import page_header, section_heading
from app.config import database_path
from app.services.cached_data import cached_catalogs, cached_inventory


def _latest_pairs(frame):
    if frame.empty:
        return frame
    return frame.sort_values("observation_date").groupby(
        ["product_id", "store_id"], as_index=False, sort=False
    ).tail(1)


def render() -> None:
    page_header("Inventory analytics", "Explore observed stock and restock patterns. Charts are descriptive and do not establish causation.")
    db = str(database_path())
    products, stores = cached_catalogs(db)
    all_data = cached_inventory(db)
    if all_data.empty:
        st.info("No inventory observations are available for analysis.")
        return

    with st.container(border=True):
        columns = st.columns(3)
        categories = columns[0].multiselect("Product category", sorted(products["category"].dropna().unique()))
        store_ids = columns[1].multiselect("Stores", stores["store_id"].tolist(),
                                          format_func=lambda key: f"{stores.loc[stores.store_id.eq(key), 'store_location'].iloc[0]} · {key}")
        products_for_filter = columns[2].multiselect("Products", products["product_id"].tolist(),
                                                    format_func=lambda key: f"{products.loc[products.product_id.eq(key), 'product_name'].iloc[0]} · {key}")
        bounds = st.columns(2)
        min_date = all_data["observation_date"].min().date()
        max_date = all_data["observation_date"].max().date()
        start_date = bounds[0].date_input("From", value=min_date, min_value=min_date, max_value=max_date)
        end_date = bounds[1].date_input("Through", value=max_date, min_value=min_date, max_value=max_date)

    if start_date > end_date:
        st.error("The start date must be on or before the end date.")
        return
    data = cached_inventory(
        db, categories=tuple(categories) if categories else None,
        store_ids=tuple(store_ids) if store_ids else None,
        product_ids=tuple(products_for_filter) if products_for_filter else None,
        start_date=start_date, end_date=end_date,
    )
    if data.empty:
        st.info("No records match the selected filters.")
        return
    latest = _latest_pairs(data)
    st.caption(f"Analysis window: {start_date:%b %d, %Y}–{end_date:%b %d, %Y} · {len(data):,} records · {len(latest):,} product/store pairs in latest snapshot")

    top_left, top_right = st.columns(2, gap="large")
    with top_left:
        section_heading("Inventory trend", "Daily mean stock uses observations within the selected filters.")
        st.altair_chart(stock_trend_chart(data), width="stretch")
    with top_right:
        section_heading("Stock distribution", "This histogram groups observed stock quantities into ranges so common levels and the shape of the distribution are easy to compare.")
        st.altair_chart(stock_distribution_chart(data), width="stretch")

    middle_left, middle_right = st.columns(2, gap="large")
    with middle_left:
        section_heading("Restocking frequency", "Counts of observed restock events in the selected records; this is not a forecast.")
        if data["restock_occurred"].any():
            st.altair_chart(restock_frequency_chart(data), width="stretch")
        else:
            st.info("No restock events appear in this filter window.")
    with middle_right:
        section_heading("Category stock analysis", "Mean quantity by category across the selected observations.")
        st.altair_chart(category_stock_chart(data), width="stretch")

    left, right = st.columns(2, gap="large")
    with left:
        section_heading("Store comparison", "Uses each pair's latest record within the selected date window.")
        st.altair_chart(store_performance_chart(latest), width="stretch")
    with right:
        section_heading("Store × category", "Color represents mean observed stock quantity.")
        st.altair_chart(category_store_heatmap(data), width="stretch")

    pattern_left, pattern_right = st.columns(2, gap="large")
    with pattern_left:
        section_heading("Weekday pattern", "Aggregated from observed dates; patterns can reflect the synthetic data design.")
        st.altair_chart(weekday_pattern_chart(data), width="stretch")
    with pattern_right:
        section_heading("Monthly pattern", "The available dataset spans only its recorded months.")
        st.altair_chart(monthly_pattern_chart(data), width="stretch")

    section_heading("Product-level trend", "Choose a product to compare its stock history across stores.")
    product_ids = sorted(data["product_id"].unique().tolist())
    selected = st.selectbox("Product for trend", product_ids,
                            format_func=lambda key: f"{data.loc[data.product_id.eq(key), 'product_name'].iloc[0]} · {key}")
    product_data = data.loc[data["product_id"].eq(selected)]
    st.altair_chart(product_trend_chart(product_data), width="stretch")
