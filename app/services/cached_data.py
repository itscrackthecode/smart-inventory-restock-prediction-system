"""Short-lived Streamlit caches around read-heavy SQLite queries."""

from __future__ import annotations

from datetime import date
from typing import Sequence

import pandas as pd
import streamlit as st

from app.services.inventory_queries import (
    count_recent_predictions,
    load_catalogs,
    load_inventory_frame,
    load_prediction_history,
    table_counts,
)


@st.cache_data(ttl=60, show_spinner=False)
def cached_catalogs(database_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_catalogs(database_path)


@st.cache_data(ttl=60, show_spinner=False)
def cached_inventory(database_path: str, *, search: str = "",
                     categories: tuple[str, ...] | None = None,
                     store_ids: tuple[str, ...] | None = None,
                     product_ids: tuple[str, ...] | None = None,
                     start_date: date | None = None,
                     end_date: date | None = None) -> pd.DataFrame:
    return load_inventory_frame(
        database_path, search=search, categories=categories, store_ids=store_ids,
        product_ids=product_ids, start_date=start_date, end_date=end_date,
    )


@st.cache_data(ttl=60, show_spinner=False)
def cached_counts(database_path: str) -> dict[str, int]:
    return table_counts(database_path)


@st.cache_data(ttl=60, show_spinner=False)
def cached_recent_prediction_count(database_path: str, days: int = 7) -> int:
    return count_recent_predictions(database_path, days=days)


@st.cache_data(ttl=60, show_spinner=False)
def cached_prediction_history(database_path: str, product_id: str, store_id: str) -> pd.DataFrame:
    return load_prediction_history(database_path, product_id, store_id)


def clear_inventory_caches() -> None:
    """Refresh all DB-backed data after a record or prediction is written."""
    for cached in (cached_catalogs, cached_inventory, cached_counts,
                   cached_recent_prediction_count, cached_prediction_history):
        cached.clear()
