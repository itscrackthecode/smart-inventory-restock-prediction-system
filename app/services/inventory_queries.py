"""Read-only, parameterized database queries used by dashboards and analytics."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.database.connection import database_connection


def load_catalogs(database_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with database_connection(database_path) as conn:
        products = pd.read_sql_query(
            "SELECT product_id, product_name, category FROM products ORDER BY product_name, product_id", conn
        )
        stores = pd.read_sql_query(
            "SELECT store_id, store_location FROM stores ORDER BY store_location, store_id", conn
        )
    return products, stores


def load_inventory_frame(
    database_path: str | Path,
    *,
    search: str = "",
    categories: Sequence[str] | None = None,
    store_ids: Sequence[str] | None = None,
    product_ids: Sequence[str] | None = None,
    start_date: date | datetime | None = None,
    end_date: date | datetime | None = None,
) -> pd.DataFrame:
    """Load joined operational records with filters applied in parameterized SQL."""
    clauses: list[str] = []
    params: list[Any] = []
    if search.strip():
        clauses.append("(p.product_name LIKE ? ESCAPE '\\' OR i.product_id LIKE ? ESCAPE '\\')")
        escaped = search.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        params.extend([f"%{escaped}%", f"%{escaped}%"])
    for column, values in (("p.category", categories), ("i.store_id", store_ids), ("i.product_id", product_ids)):
        if values is not None:
            choices = list(dict.fromkeys(values))
            if not choices:
                return _empty_inventory_frame()
            clauses.append(f"{column} IN ({','.join('?' for _ in choices)})")
            params.extend(choices)
    if start_date is not None:
        clauses.append("i.observation_date >= ?")
        params.append(_date_boundary(start_date, end_of_day=False))
    if end_date is not None:
        clauses.append("i.observation_date <= ?")
        params.append(_date_boundary(end_date, end_of_day=True))
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("Start date must not be after end date")
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    query = f"""SELECT i.inventory_id, i.product_id, p.product_name, p.category,
        i.store_id, s.store_location, i.observation_date, i.stock_quantity,
        i.availability_status, i.previous_stock_status, i.restock_occurred, i.data_source
        FROM inventory_records AS i
        JOIN products AS p USING (product_id)
        JOIN stores AS s USING (store_id){where}
        ORDER BY i.observation_date, i.product_id, i.store_id"""
    with database_connection(database_path) as conn:
        frame = pd.read_sql_query(query, conn, params=params)
    if not frame.empty:
        frame["observation_date"] = pd.to_datetime(frame["observation_date"], utc=True, errors="raise")
        frame["restock_occurred"] = frame["restock_occurred"].astype(bool)
    return frame


def _date_boundary(value: date | datetime, *, end_of_day: bool) -> str:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
        if end_of_day:
            parsed = parsed.replace(hour=23, minute=59, second=59, microsecond=999999)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _empty_inventory_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "inventory_id", "product_id", "product_name", "category", "store_id", "store_location",
        "observation_date", "stock_quantity", "availability_status", "previous_stock_status",
        "restock_occurred", "data_source",
    ])


def load_product_store_timeline(
    database_path: str | Path, product_id: str, store_id: str,
) -> pd.DataFrame:
    """Load all observed history for one product/store in chronological order."""
    return load_inventory_frame(database_path, product_ids=[product_id], store_ids=[store_id])


def count_recent_predictions(database_path: str | Path, *, days: int = 7,
                             now: datetime | None = None) -> int:
    if days < 1:
        raise ValueError("days must be positive")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    cutoff = current.astimezone(timezone.utc) - pd.Timedelta(days=days)
    with database_connection(database_path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM predictions WHERE created_at >= ?",
                                (cutoff.isoformat(),)).fetchone()[0])


def load_prediction_history(database_path: str | Path, product_id: str, store_id: str) -> pd.DataFrame:
    query = """SELECT p.prediction_id, p.model_name, p.model_version, p.predicted_probability,
        p.prediction_threshold, p.predicted_restock, p.created_at,
        i.observation_date, i.stock_quantity, i.availability_status
        FROM predictions AS p JOIN inventory_records AS i USING (inventory_id)
        WHERE i.product_id = ? AND i.store_id = ? ORDER BY p.created_at DESC, p.prediction_id DESC"""
    with database_connection(database_path) as conn:
        return pd.read_sql_query(query, conn, params=[product_id, store_id])


def table_counts(database_path: str | Path) -> dict[str, int]:
    with database_connection(database_path) as conn:
        return {
            "products": int(conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]),
            "stores": int(conn.execute("SELECT COUNT(*) FROM stores").fetchone()[0]),
            "inventory_records": int(conn.execute("SELECT COUNT(*) FROM inventory_records").fetchone()[0]),
            "predictions": int(conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]),
        }
