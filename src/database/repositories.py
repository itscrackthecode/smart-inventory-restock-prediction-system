"""Validated, parameterized CRUD operations for database entities."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from src.database.connection import database_connection


VALID_STATUSES = {"in_stock", "out_of_stock"}
VALID_PREVIOUS_STATUSES = VALID_STATUSES | {"unknown"}
INVENTORY_UPDATE_FIELDS = {
    "observation_date", "stock_quantity", "availability_status",
    "previous_stock_status", "restock_occurred", "data_source",
}


def _text(value: Any, field: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"{field} must be a non-empty string")
    return str(value).strip()


def normalize_timestamp(value: str | date | datetime, *, field: str = "observation_date") -> str:
    """Return an ISO-8601 timestamp in UTC; naive dates are rejected."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    elif isinstance(value, str):
        candidate = value.strip()
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError as exc:
            raise ValueError(f"{field} must be a valid ISO-8601 timestamp") from exc
    else:
        raise ValueError(f"{field} must be a date, datetime, or ISO-8601 string")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat()


def _boolean(value: Any, field: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if value in (0, 1):
        return int(value)
    raise ValueError(f"{field} must be a boolean")


def _inventory_values(*, product_id: Any, store_id: Any, observation_date: Any,
                      stock_quantity: Any, availability_status: Any,
                      previous_stock_status: Any, restock_occurred: Any,
                      data_source: Any) -> tuple[Any, ...]:
    if isinstance(stock_quantity, bool) or not isinstance(stock_quantity, int) or stock_quantity < 0:
        raise ValueError("stock_quantity must be a non-negative integer")
    status = _text(availability_status, "availability_status").lower()
    previous = _text(previous_stock_status, "previous_stock_status").lower()
    if status not in VALID_STATUSES:
        raise ValueError(f"availability_status must be one of {sorted(VALID_STATUSES)}")
    if previous not in VALID_PREVIOUS_STATUSES:
        raise ValueError(f"previous_stock_status must be one of {sorted(VALID_PREVIOUS_STATUSES)}")
    if (stock_quantity == 0) != (status == "out_of_stock"):
        raise ValueError("stock_quantity conflicts with availability_status")
    return (
        _text(product_id, "product_id"), _text(store_id, "store_id"),
        normalize_timestamp(observation_date), stock_quantity, status, previous,
        _boolean(restock_occurred, "restock_occurred"), _text(data_source, "data_source"),
    )


def _as_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


# Products
def create_product(product_id: str, product_name: str, category: str,
                   database_path: str | Path | None = None) -> dict[str, Any]:
    values = (_text(product_id, "product_id"), _text(product_name, "product_name"), _text(category, "category"))
    with database_connection(database_path) as conn:
        conn.execute("INSERT INTO products(product_id, product_name, category) VALUES (?, ?, ?)", values)
        return dict(conn.execute("SELECT * FROM products WHERE product_id = ?", (values[0],)).fetchone())


def get_product(product_id: str, database_path: str | Path | None = None) -> dict[str, Any] | None:
    with database_connection(database_path) as conn:
        return _as_dict(conn.execute("SELECT * FROM products WHERE product_id = ?", (product_id,)).fetchone())


def list_products(database_path: str | Path | None = None) -> list[dict[str, Any]]:
    with database_connection(database_path) as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM products ORDER BY product_id")]


def update_product(product_id: str, *, product_name: str, category: str,
                   database_path: str | Path | None = None) -> dict[str, Any] | None:
    values = (_text(product_name, "product_name"), _text(category, "category"), product_id)
    with database_connection(database_path) as conn:
        cursor = conn.execute("UPDATE products SET product_name = ?, category = ? WHERE product_id = ?", values)
        return _as_dict(conn.execute("SELECT * FROM products WHERE product_id = ?", (product_id,)).fetchone()) if cursor.rowcount else None


def delete_product(product_id: str, database_path: str | Path | None = None) -> bool:
    """Delete only products without inventory references (FK restricts unsafe deletion)."""
    with database_connection(database_path) as conn:
        return conn.execute("DELETE FROM products WHERE product_id = ?", (product_id,)).rowcount > 0


# Stores
def create_store(store_id: str, store_location: str,
                 database_path: str | Path | None = None) -> dict[str, Any]:
    values = (_text(store_id, "store_id"), _text(store_location, "store_location"))
    with database_connection(database_path) as conn:
        conn.execute("INSERT INTO stores(store_id, store_location) VALUES (?, ?)", values)
        return dict(conn.execute("SELECT * FROM stores WHERE store_id = ?", (values[0],)).fetchone())


def get_store(store_id: str, database_path: str | Path | None = None) -> dict[str, Any] | None:
    with database_connection(database_path) as conn:
        return _as_dict(conn.execute("SELECT * FROM stores WHERE store_id = ?", (store_id,)).fetchone())


def list_stores(database_path: str | Path | None = None) -> list[dict[str, Any]]:
    with database_connection(database_path) as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM stores ORDER BY store_id")]


# Inventory
def insert_inventory_record(*, product_id: str, store_id: str, observation_date: str | date | datetime,
                             stock_quantity: int, availability_status: str, previous_stock_status: str,
                             restock_occurred: bool | int, data_source: str,
                             database_path: str | Path | None = None) -> dict[str, Any]:
    values = _inventory_values(product_id=product_id, store_id=store_id, observation_date=observation_date,
                               stock_quantity=stock_quantity, availability_status=availability_status,
                               previous_stock_status=previous_stock_status, restock_occurred=restock_occurred,
                               data_source=data_source)
    with database_connection(database_path) as conn:
        cursor = conn.execute("""INSERT INTO inventory_records
            (product_id, store_id, observation_date, stock_quantity, availability_status,
             previous_stock_status, restock_occurred, data_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", values)
        return dict(conn.execute("SELECT * FROM inventory_records WHERE inventory_id = ?", (cursor.lastrowid,)).fetchone())


def get_inventory_record(inventory_id: int, database_path: str | Path | None = None) -> dict[str, Any] | None:
    with database_connection(database_path) as conn:
        return _as_dict(conn.execute("SELECT * FROM inventory_records WHERE inventory_id = ?", (inventory_id,)).fetchone())


def list_inventory_records(*, product_id: str | None = None, store_id: str | None = None,
                           start_date: str | date | datetime | None = None,
                           end_date: str | date | datetime | None = None,
                           limit: int | None = None,
                           database_path: str | Path | None = None) -> list[dict[str, Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if product_id is not None:
        clauses.append("product_id = ?")
        params.append(product_id)
    if store_id is not None:
        clauses.append("store_id = ?")
        params.append(store_id)
    if start_date is not None:
        clauses.append("observation_date >= ?")
        params.append(normalize_timestamp(start_date, field="start_date"))
    if end_date is not None:
        clauses.append("observation_date <= ?")
        params.append(normalize_timestamp(end_date, field="end_date"))
    if start_date is not None and end_date is not None and params[-2] > params[-1]:
        raise ValueError("start_date must be less than or equal to end_date")
    query = "SELECT * FROM inventory_records"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY observation_date, product_id, store_id"
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        query += " LIMIT ?"
        params.append(limit)
    with database_connection(database_path) as conn:
        return [dict(row) for row in conn.execute(query, params)]


def list_inventory_by_product(product_id: str, database_path: str | Path | None = None) -> list[dict[str, Any]]:
    return list_inventory_records(product_id=product_id, database_path=database_path)


def list_inventory_by_store(store_id: str, database_path: str | Path | None = None) -> list[dict[str, Any]]:
    return list_inventory_records(store_id=store_id, database_path=database_path)


def latest_inventory_observations(*, database_path: str | Path | None = None) -> list[dict[str, Any]]:
    query = """SELECT i.* FROM inventory_records AS i
        WHERE i.observation_date = (SELECT MAX(j.observation_date) FROM inventory_records AS j
          WHERE j.product_id = i.product_id AND j.store_id = i.store_id)
        ORDER BY i.product_id, i.store_id"""
    with database_connection(database_path) as conn:
        return [dict(row) for row in conn.execute(query)]


def update_inventory_record(inventory_id: int, *, database_path: str | Path | None = None,
                            **updates: Any) -> dict[str, Any] | None:
    if not updates:
        return get_inventory_record(inventory_id, database_path)
    unknown = set(updates) - INVENTORY_UPDATE_FIELDS
    if unknown:
        raise ValueError(f"Unsupported inventory fields: {sorted(unknown)}")
    with database_connection(database_path) as conn:
        existing = conn.execute("SELECT * FROM inventory_records WHERE inventory_id = ?", (inventory_id,)).fetchone()
        if existing is None:
            return None
        current = dict(existing)
        current.update(updates)
        values = _inventory_values(product_id=current["product_id"], store_id=current["store_id"],
                                   observation_date=current["observation_date"], stock_quantity=current["stock_quantity"],
                                   availability_status=current["availability_status"],
                                   previous_stock_status=current["previous_stock_status"],
                                   restock_occurred=current["restock_occurred"], data_source=current["data_source"])
        conn.execute("""UPDATE inventory_records SET observation_date=?, stock_quantity=?, availability_status=?,
            previous_stock_status=?, restock_occurred=?, data_source=? WHERE inventory_id=?""",
            (values[2], values[3], values[4], values[5], values[6], values[7], inventory_id))
        return dict(conn.execute("SELECT * FROM inventory_records WHERE inventory_id = ?", (inventory_id,)).fetchone())


# Predictions
def save_prediction(*, inventory_id: int, model_name: str, model_version: str,
                    predicted_probability: float, prediction_threshold: float,
                    predicted_restock: bool | int, created_at: str | datetime | None = None,
                    database_path: str | Path | None = None) -> dict[str, Any]:
    probability = float(predicted_probability)
    threshold = float(prediction_threshold)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("predicted_probability must be between 0 and 1")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("prediction_threshold must be between 0 and 1")
    timestamp = normalize_timestamp(created_at or datetime.now(timezone.utc), field="created_at")
    values = (inventory_id, _text(model_name, "model_name"), _text(model_version, "model_version"),
              probability, threshold, _boolean(predicted_restock, "predicted_restock"), timestamp)
    with database_connection(database_path) as conn:
        cursor = conn.execute("""INSERT INTO predictions
            (inventory_id, model_name, model_version, predicted_probability, prediction_threshold, predicted_restock, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)""", values)
        return dict(conn.execute("SELECT * FROM predictions WHERE prediction_id = ?", (cursor.lastrowid,)).fetchone())


def list_predictions(*, inventory_id: int | None = None, model_name: str | None = None,
                     database_path: str | Path | None = None) -> list[dict[str, Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if inventory_id is not None:
        clauses.append("inventory_id = ?")
        params.append(inventory_id)
    if model_name is not None:
        clauses.append("model_name = ?")
        params.append(model_name)
    query = "SELECT * FROM predictions" + (" WHERE " + " AND ".join(clauses) if clauses else "")
    query += " ORDER BY created_at, prediction_id"
    with database_connection(database_path) as conn:
        return [dict(row) for row in conn.execute(query, params)]


def prediction_history(*, product_id: str, store_id: str,
                       database_path: str | Path | None = None) -> list[dict[str, Any]]:
    query = """SELECT p.*, i.product_id, i.store_id, i.observation_date
        FROM predictions AS p JOIN inventory_records AS i USING (inventory_id)
        WHERE i.product_id = ? AND i.store_id = ? ORDER BY p.created_at, p.prediction_id"""
    with database_connection(database_path) as conn:
        return [dict(row) for row in conn.execute(query, (product_id, store_id))]


def latest_predictions(*, database_path: str | Path | None = None) -> list[dict[str, Any]]:
    query = """SELECT p.*, i.product_id, i.store_id, i.observation_date
        FROM predictions AS p JOIN inventory_records AS i USING (inventory_id)
        WHERE p.prediction_id = (SELECT p2.prediction_id FROM predictions AS p2
          JOIN inventory_records AS i2 USING (inventory_id)
          WHERE i2.product_id = i.product_id AND i2.store_id = i.store_id
          ORDER BY p2.created_at DESC, p2.prediction_id DESC LIMIT 1)
        ORDER BY i.product_id, i.store_id"""
    with database_connection(database_path) as conn:
        return [dict(row) for row in conn.execute(query)]
