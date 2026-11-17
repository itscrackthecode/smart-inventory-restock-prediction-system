"""Search, filter, inspect, and append inventory observations."""

from __future__ import annotations

import sqlite3
from datetime import date

import pandas as pd
import streamlit as st

from app.components.charts import product_trend_chart
from app.components.layout import page_header, section_heading
from app.config import database_path
from app.services.cached_data import cached_catalogs, cached_inventory, clear_inventory_caches
from app.services.inventory_operations import create_inventory_observation


def render() -> None:
    page_header("Inventory explorer", "Search operational observations, inspect history, and add a dated record.")
    db = str(database_path())
    products, stores = cached_catalogs(db)
    if products.empty or stores.empty:
        st.info("Product/store catalogs are empty. Import processed inventory data before adding observations.")
        return
    all_records = cached_inventory(db)
    if all_records.empty:
        st.info("No inventory records are available.")
        return

    with st.container(border=True):
        filter_cols = st.columns([1.5, 1.2, 1.2, 1.3])
        search = filter_cols[0].text_input("Search product", placeholder="Name or product ID")
        category_options = sorted(products["category"].dropna().unique().tolist())
        categories = filter_cols[1].multiselect("Category", category_options)
        store_options = stores["store_id"].tolist()
        store_labels = dict(zip(stores["store_id"], stores["store_location"]))
        selected_store_labels = filter_cols[2].multiselect(
            "Store", store_options, format_func=lambda key: f"{store_labels.get(key, key)} · {key}"
        )
        date_col = filter_cols[3]
        min_date = all_records["observation_date"].min().date()
        max_date = all_records["observation_date"].max().date()
        date_range = date_col.date_input("Observation dates", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    start_date, end_date = (date_range[0], date_range[-1]) if isinstance(date_range, (tuple, list)) else (date_range, date_range)
    filtered = cached_inventory(
        db, search=search, categories=tuple(categories) if categories else None,
        store_ids=tuple(selected_store_labels) if selected_store_labels else None,
        start_date=start_date, end_date=end_date,
    )
    st.caption(f"Showing {len(filtered):,} inventory records. Filters are applied in SQLite.")
    st.dataframe(
        filtered[["observation_date", "product_id", "product_name", "category", "store_id", "store_location",
                   "stock_quantity", "availability_status", "previous_stock_status", "restock_occurred", "data_source"]],
        hide_index=True, width="stretch",
        column_config={"observation_date": st.column_config.DatetimeColumn("Observation (UTC)", format="YYYY-MM-DD HH:mm")},
    )

    st.divider()
    section_heading("Product stock history", "Choose one product/store pair to review the stored stock timeline.")
    product_ids = products["product_id"].tolist()
    history_cols = st.columns(2)
    chosen_product = history_cols[0].selectbox(
        "Product", product_ids, format_func=lambda key: f"{products.loc[products.product_id.eq(key), 'product_name'].iloc[0]} · {key}",
        key="history_product",
    )
    chosen_store = history_cols[1].selectbox(
        "Store for history", store_options, format_func=lambda key: f"{store_labels.get(key, key)} · {key}",
        key="history_store",
    )
    history = cached_inventory(db, product_ids=(chosen_product,), store_ids=(chosen_store,))
    if not history.empty:
        st.altair_chart(product_trend_chart(history), width="stretch")
        st.caption(f"Showing {len(history)} chronological records; highlighted points expose date, stock, status, and observed restock event.")

    with st.expander("Compare latest stock across stores"):
        one_product = cached_inventory(db, product_ids=(chosen_product,))
        latest = one_product.sort_values("observation_date").groupby("store_id", as_index=False).tail(1)
        st.dataframe(latest[["store_id", "store_location", "observation_date", "stock_quantity", "availability_status"]],
                     hide_index=True, width="stretch")

    st.divider()
    section_heading("Add inventory observation", "Only existing products and stores can be selected. Duplicate dates are rejected without overwriting history.")
    render_add_observation_form(db, products, stores)


def render_add_observation_form(db: str, products: pd.DataFrame, stores: pd.DataFrame) -> None:
    with st.form("add_inventory_observation", clear_on_submit=False):
        col1, col2 = st.columns(2)
        product_id = col1.selectbox("Product", products["product_id"].tolist(),
                                    format_func=lambda key: f"{products.loc[products.product_id.eq(key), 'product_name'].iloc[0]} · {key}",
                                    key="new_product")
        store_ids = stores["store_id"].tolist()
        locations = dict(zip(stores["store_id"], stores["store_location"]))
        store_id = col2.selectbox("Store", store_ids,
                                  format_func=lambda key: f"{locations.get(key, key)} · {key}", key="new_store")
        observation_date = col1.date_input("Observation date", value=date.today(), key="new_observation_date")
        stock_quantity = col2.number_input("Stock quantity", min_value=0, max_value=1_000_000, value=10, step=1)
        availability_status = col1.selectbox("Availability status", ["in_stock", "out_of_stock"])
        restock_occurred = col2.checkbox("A restock was observed on this record")
        submitted = st.form_submit_button("Save observation", type="primary")
    if not submitted:
        return
    try:
        record = create_inventory_observation(
            db,
            product_id=product_id, store_id=store_id,
            observation_date=observation_date,
            stock_quantity=int(stock_quantity), availability_status=availability_status,
            restock_occurred=restock_occurred,
        )
    except sqlite3.IntegrityError as exc:
        if "UNIQUE constraint failed" in str(exc):
            st.warning("An observation already exists for this product, store, and timestamp. The existing record was not changed.")
        else:
            st.error(f"The inventory record could not be saved because a database integrity rule rejected it: {exc}")
        return
    except (ValueError, sqlite3.Error) as exc:
        st.error(f"The inventory record could not be saved: {exc}")
        return
    clear_inventory_caches()
    st.success(f"Observation saved as inventory record #{record['inventory_id']} ({record['data_source']}).")
