"""Generate deterministic, clearly synthetic inventory observations.

This module intentionally uses only the Python standard library so it can run
before the project's analytics dependencies have been installed.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


PRODUCTS = [
    ("P001", "Arabica Coffee 500g", "Grocery"),
    ("P002", "Green Tea 100ct", "Grocery"),
    ("P003", "Olive Oil 1L", "Grocery"),
    ("P004", "Pasta 1kg", "Grocery"),
    ("P005", "Rice 5kg", "Grocery"),
    ("P006", "Dish Soap 750ml", "Household"),
    ("P007", "Laundry Detergent 2L", "Household"),
    ("P008", "Paper Towels 6-pack", "Household"),
    ("P009", "Shampoo 400ml", "Personal Care"),
    ("P010", "Toothpaste 150g", "Personal Care"),
    ("P011", "Moisturizer 250ml", "Personal Care"),
    ("P012", "Bottled Water 12-pack", "Beverages"),
    ("P013", "Orange Juice 1L", "Beverages"),
    ("P014", "Sparkling Water 6-pack", "Beverages"),
    ("P015", "USB-C Cable 2m", "Electronics"),
    ("P016", "Wireless Mouse", "Electronics"),
    ("P017", "AA Batteries 8-pack", "Electronics"),
    ("P018", "Notebook A5", "Stationery"),
    ("P019", "Ballpoint Pens 10-pack", "Stationery"),
    ("P020", "Printer Paper 500 sheets", "Stationery"),
]

STORES = [
    ("S001", "Austin, TX"),
    ("S002", "Chicago, IL"),
    ("S003", "Denver, CO"),
    ("S004", "Atlanta, GA"),
    ("S005", "Seattle, WA"),
    ("S006", "Boston, MA"),
]

FIELDS = [
    "product_id", "product_name", "category", "store_id", "store_location",
    "timestamp", "stock_quantity", "availability_status", "previous_stock_status",
    "days_since_last_restock", "historical_restock_count", "restock_occurred", "data_source",
]
DATA_SOURCE = "synthetic"


def generate_inventory_data(
    *, seed: int = 42, days: int = 90, start: datetime | None = None
) -> list[dict[str, Any]]:
    """Return one daily observation per product/store for the requested period.

    Historical fields are computed before processing the current day's possible
    restock. This strict ordering means they only summarize earlier days.
    """
    if days < 1:
        raise ValueError("days must be at least 1")

    rng = random.Random(seed)
    first_day = start or datetime(2025, 1, 1, tzinfo=timezone.utc)
    if first_day.tzinfo is None:
        first_day = first_day.replace(tzinfo=timezone.utc)

    rows: list[dict[str, Any]] = []
    for product_id, product_name, category in PRODUCTS:
        for store_id, store_location in STORES:
            # Each location has a different but repeatable demand and replenishment profile.
            demand_scale = rng.uniform(0.65, 1.45)
            reorder_point = rng.randint(5, 13)
            lead_stock = rng.randint(18, 38)
            stock = rng.randint(reorder_point + 2, lead_stock)
            last_restock_day: int | None = None
            restock_count = 0
            previous_status: str | None = None

            for day_index in range(days):
                observed_at = first_day + timedelta(days=day_index)

                # Snapshot historical features first; today's event cannot leak into them.
                days_since = (
                    day_index - last_restock_day
                    if last_restock_day is not None
                    else day_index + 1
                )
                prior_restock_count = restock_count
                prior_status = previous_status or "unknown"

                demand = max(0, round(rng.gauss(4.0 * demand_scale, 2.0)))
                stock = max(0, stock - demand)

                restocked_today = False
                if stock <= reorder_point and rng.random() < 0.74:
                    stock += lead_stock + rng.randint(0, 12)
                    last_restock_day = day_index
                    restock_count += 1
                    restocked_today = True

                # Rare delivery delays and high demand create realistic stockouts.
                if not restocked_today and stock <= reorder_point and rng.random() < 0.28:
                    stock = 0

                status = "in_stock" if stock > 0 else "out_of_stock"
                rows.append(
                    {
                        "product_id": product_id,
                        "product_name": product_name,
                        "category": category,
                        "store_id": store_id,
                        "store_location": store_location,
                        "timestamp": observed_at.isoformat(timespec="seconds"),
                        "stock_quantity": stock,
                        "availability_status": status,
                        "previous_stock_status": prior_status,
                        "days_since_last_restock": days_since,
                        "historical_restock_count": prior_restock_count,
                        "restock_occurred": restocked_today,
                        "data_source": DATA_SOURCE,
                    }
                )
                previous_status = status

    return rows


def validate_inventory_data(rows: list[dict[str, Any]], *, minimum_rows: int = 5000) -> None:
    """Raise ValueError if required schema, value, or history invariants fail."""
    if len(rows) < minimum_rows:
        raise ValueError(f"Expected at least {minimum_rows} observations; got {len(rows)}")
    if not rows:
        raise ValueError("Dataset is empty")
    expected = set(FIELDS)
    if any(set(row) != expected for row in rows):
        raise ValueError("Rows do not match the required dataset schema")

    seen: set[tuple[str, str, str]] = set()
    histories: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["product_id"], row["store_id"], row["timestamp"])
        if key in seen:
            raise ValueError(f"Duplicate product/store/timestamp observation: {key}")
        seen.add(key)
        if row["data_source"] != DATA_SOURCE:
            raise ValueError("Every observation must be labeled synthetic")
        if not isinstance(row["stock_quantity"], int) or row["stock_quantity"] < 0:
            raise ValueError("Stock quantity must be a non-negative integer")
        expected_status = "in_stock" if row["stock_quantity"] > 0 else "out_of_stock"
        if row["availability_status"] != expected_status:
            raise ValueError("Availability status conflicts with stock quantity")
        if row["days_since_last_restock"] < 1 or row["historical_restock_count"] < 0:
            raise ValueError("Historical features must be non-negative and valid")
        histories[(row["product_id"], row["store_id"])].append(row)

    for pair_rows in histories.values():
        pair_rows.sort(key=lambda row: row["timestamp"])
        restock_count = 0
        last_restock_day: int | None = None
        previous_status: str | None = None
        for day_index, row in enumerate(pair_rows):
            expected_days_since = day_index - last_restock_day if last_restock_day is not None else day_index + 1
            if row["days_since_last_restock"] != expected_days_since:
                raise ValueError("Days-since-restock feature is not based on prior history")
            if row["historical_restock_count"] != restock_count:
                raise ValueError("Restock-count feature is not based on prior history")
            if row["previous_stock_status"] != (previous_status or "unknown"):
                raise ValueError("Previous status does not match the previous observation")
            if not isinstance(row["restock_occurred"], bool):
                raise ValueError("Restock indicator must be boolean")
            # Update the running history only after checking this day's features.
            if row["restock_occurred"]:
                restock_count += 1
                last_restock_day = day_index
            previous_status = row["availability_status"]


def save_inventory_csv(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Write observations to CSV, creating parent directories as needed."""
    validate_inventory_data(rows)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_inventory.csv"))
    args = parser.parse_args()

    observations = generate_inventory_data(seed=args.seed, days=args.days)
    validate_inventory_data(observations)
    path = save_inventory_csv(observations, args.output)
    print(f"Generated and validated {len(observations):,} synthetic observations: {path}")
    for row in observations[:5]:
        print(row)


if __name__ == "__main__":
    main()
