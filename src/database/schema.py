"""Idempotent schema setup for the inventory application."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.database.connection import database_connection


SCHEMA_STATEMENTS = (
    """CREATE TABLE IF NOT EXISTS products (
        product_id TEXT PRIMARY KEY CHECK (length(trim(product_id)) > 0),
        product_name TEXT NOT NULL CHECK (length(trim(product_name)) > 0),
        category TEXT NOT NULL CHECK (length(trim(category)) > 0)
    )""",
    """CREATE TABLE IF NOT EXISTS stores (
        store_id TEXT PRIMARY KEY CHECK (length(trim(store_id)) > 0),
        store_location TEXT NOT NULL CHECK (length(trim(store_location)) > 0)
    )""",
    """CREATE TABLE IF NOT EXISTS inventory_records (
        inventory_id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id TEXT NOT NULL REFERENCES products(product_id) ON UPDATE CASCADE ON DELETE RESTRICT,
        store_id TEXT NOT NULL REFERENCES stores(store_id) ON UPDATE CASCADE ON DELETE RESTRICT,
        observation_date TEXT NOT NULL,
        stock_quantity INTEGER NOT NULL CHECK (stock_quantity >= 0),
        availability_status TEXT NOT NULL CHECK (availability_status IN ('in_stock', 'out_of_stock')),
        previous_stock_status TEXT NOT NULL CHECK (previous_stock_status IN ('in_stock', 'out_of_stock', 'unknown')),
        restock_occurred INTEGER NOT NULL CHECK (restock_occurred IN (0, 1)),
        data_source TEXT NOT NULL CHECK (length(trim(data_source)) > 0),
        UNIQUE (product_id, store_id, observation_date),
        CHECK ((stock_quantity = 0 AND availability_status = 'out_of_stock') OR
               (stock_quantity > 0 AND availability_status = 'in_stock'))
    )""",
    """CREATE TABLE IF NOT EXISTS predictions (
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        inventory_id INTEGER NOT NULL REFERENCES inventory_records(inventory_id) ON UPDATE CASCADE ON DELETE RESTRICT,
        model_name TEXT NOT NULL CHECK (length(trim(model_name)) > 0),
        model_version TEXT NOT NULL CHECK (length(trim(model_version)) > 0),
        predicted_probability REAL NOT NULL CHECK (predicted_probability >= 0.0 AND predicted_probability <= 1.0),
        prediction_threshold REAL NOT NULL CHECK (prediction_threshold >= 0.0 AND prediction_threshold <= 1.0),
        predicted_restock INTEGER NOT NULL CHECK (predicted_restock IN (0, 1)),
        created_at TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_inventory_product_date ON inventory_records(product_id, observation_date)",
    "CREATE INDEX IF NOT EXISTS idx_inventory_store_date ON inventory_records(store_id, observation_date)",
    "CREATE INDEX IF NOT EXISTS idx_inventory_date ON inventory_records(observation_date)",
    "CREATE INDEX IF NOT EXISTS idx_predictions_inventory_created ON predictions(inventory_id, created_at)",
)


def initialize_database(database_path: str | Path | None = None) -> None:
    """Create tables and indexes safely; repeated calls preserve all data."""
    with database_connection(database_path) as conn:
        for statement in SCHEMA_STATEMENTS:
            conn.execute(statement)
        if conn.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
            raise sqlite3.OperationalError("SQLite foreign-key enforcement could not be enabled")
