"""Validated user-requested inventory writes using the existing repositories."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

from app.services.inventory_queries import load_product_store_timeline
from src.database.repositories import insert_inventory_record


def create_inventory_observation(
    database_path: str | Path,
    *,
    product_id: str,
    store_id: str,
    observation_date: date,
    stock_quantity: int,
    availability_status: str,
    restock_occurred: bool,
) -> dict[str, Any]:
    """Validate form fields, derive only prior status, then insert a new row."""
    if not isinstance(observation_date, date) or isinstance(observation_date, datetime):
        raise ValueError("Observation date must be a calendar date")
    if isinstance(stock_quantity, bool) or not isinstance(stock_quantity, int) or stock_quantity < 0:
        raise ValueError("Stock quantity must be a non-negative whole number")
    expected_status = "in_stock" if stock_quantity > 0 else "out_of_stock"
    if availability_status != expected_status:
        raise ValueError(f"Stock quantity {stock_quantity} requires status {expected_status}")
    if not isinstance(restock_occurred, bool):
        raise ValueError("Restock event must be a boolean")

    timeline = load_product_store_timeline(database_path, product_id, store_id)
    prior = timeline.loc[timeline["observation_date"].dt.date < observation_date]
    previous_status = str(prior.iloc[-1]["availability_status"]) if not prior.empty else "unknown"
    return insert_inventory_record(
        product_id=product_id,
        store_id=store_id,
        observation_date=f"{observation_date.isoformat()}T00:00:00+00:00",
        stock_quantity=stock_quantity,
        availability_status=availability_status,
        previous_stock_status=previous_status,
        restock_occurred=restock_occurred,
        data_source="manual",
        database_path=database_path,
    )
