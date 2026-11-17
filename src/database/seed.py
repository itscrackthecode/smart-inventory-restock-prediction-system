"""Repeatable import of the operational columns in processed_inventory.csv."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.database.connection import DEFAULT_DATABASE_PATH, database_connection
from src.database.repositories import _inventory_values
from src.database.schema import initialize_database


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_PATH = PROJECT_ROOT / "data" / "processed_inventory.csv"
IMPORT_COLUMNS = {
    "product_id", "product_name", "category", "store_id", "store_location",
    "timestamp", "stock_quantity", "availability_status", "previous_stock_status",
    "restock_occurred", "data_source",
}


@dataclass
class ImportReport:
    source_path: str
    input_rows: int = 0
    inserted_inventory_rows: int = 0
    duplicate_inventory_rows: int = 0
    invalid_rows: int = 0
    inserted_products: int = 0
    inserted_stores: int = 0
    final_inventory_rows: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() in {"true", "1"}:
        return True
    if str(value).strip().lower() in {"false", "0"}:
        return False
    raise ValueError(f"{field} must contain true/false")


def import_processed_inventory(
    source_path: str | Path = DEFAULT_SOURCE_PATH,
    database_path: str | Path | None = None,
) -> ImportReport:
    """Import valid rows atomically per observation and safely skip bad rows.

    Existing identical observations count as duplicates. Existing keys with
    conflicting values are reported as invalid instead of being overwritten.
    The report accounts for every source row.
    """
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"Processed inventory CSV not found: {source}")
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")
        missing = IMPORT_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Processed CSV is missing import columns: {sorted(missing)}")
        rows = list(reader)

    report = ImportReport(source_path=str(source.resolve()), input_rows=len(rows))
    initialize_database(database_path)
    with database_connection(database_path) as conn:
        before_products = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        before_stores = conn.execute("SELECT COUNT(*) FROM stores").fetchone()[0]
        # Cache existing dimensions and observation keys once. The source has
        # many repeated product/store IDs, so repeated SELECTs per day add no
        # safety and make a full import unnecessarily slow.
        known_products = {
            row["product_id"]: (row["product_name"], row["category"])
            for row in conn.execute("SELECT product_id, product_name, category FROM products")
        }
        known_stores = {
            row["store_id"]: row["store_location"]
            for row in conn.execute("SELECT store_id, store_location FROM stores")
        }
        known_observations = {
            (row["product_id"], row["store_id"], row["observation_date"]): (
                row["product_id"], row["store_id"], row["observation_date"], row["stock_quantity"],
                row["availability_status"], row["previous_stock_status"], row["restock_occurred"], row["data_source"]
            )
            for row in conn.execute("""SELECT product_id, store_id, observation_date, stock_quantity,
                availability_status, previous_stock_status, restock_occurred, data_source FROM inventory_records""")
        }
        for line_number, row in enumerate(rows, start=2):
            savepoint: str | None = None
            new_product: tuple[str, tuple[str, str]] | None = None
            new_store: tuple[str, str] | None = None
            new_observation: tuple[tuple[Any, ...], tuple[Any, ...]] | None = None
            try:
                product_id = row["product_id"].strip()
                store_id = row["store_id"].strip()
                product_values = (product_id, row["product_name"].strip(), row["category"].strip())
                store_values = (store_id, row["store_location"].strip())
                if any(not value for value in product_values + store_values):
                    raise ValueError("Product and store identity fields must be non-empty")
                values = _inventory_values(
                    product_id=product_id,
                    store_id=store_id,
                    observation_date=row["timestamp"],
                    stock_quantity=int(row["stock_quantity"]),
                    availability_status=row["availability_status"],
                    previous_stock_status=row["previous_stock_status"],
                    restock_occurred=_parse_bool(row["restock_occurred"], "restock_occurred"),
                    data_source=row["data_source"],
                )
                if product_id in known_products and known_products[product_id] != product_values[1:]:
                    raise ValueError(f"Conflicting metadata for product {product_id!r}")
                if store_id in known_stores and known_stores[store_id] != store_values[1]:
                    raise ValueError(f"Conflicting metadata for store {store_id!r}")

                observation_key = values[:3]
                existing = known_observations.get(observation_key)
                if existing is not None:
                    if existing != values:
                        raise ValueError("Conflicting data for an existing product/store/date observation")
                # A savepoint is needed only when this row may add a dimension
                # record. Most daily rows reuse known products/stores.
                if product_id not in known_products or store_id not in known_stores:
                    savepoint = f"inventory_row_{line_number}"
                    conn.execute(f"SAVEPOINT {savepoint}")
                if product_id not in known_products:
                    conn.execute("INSERT INTO products(product_id, product_name, category) VALUES (?, ?, ?)", product_values)
                    new_product = (product_id, product_values[1:])
                if store_id not in known_stores:
                    conn.execute("INSERT INTO stores(store_id, store_location) VALUES (?, ?)", store_values)
                    new_store = (store_id, store_values[1])
                if existing is None:
                    conn.execute("""INSERT INTO inventory_records
                        (product_id, store_id, observation_date, stock_quantity, availability_status,
                         previous_stock_status, restock_occurred, data_source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", values)
                    new_observation = (observation_key, values)
                if savepoint is not None:
                    conn.execute(f"RELEASE SAVEPOINT {savepoint}")
                if new_product:
                    known_products[new_product[0]] = new_product[1]
                if new_store:
                    known_stores[new_store[0]] = new_store[1]
                if new_observation:
                    known_observations[new_observation[0]] = new_observation[1]
                    report.inserted_inventory_rows += 1
                else:
                    report.duplicate_inventory_rows += 1
            except (ValueError, TypeError, sqlite3.IntegrityError, OverflowError) as exc:
                if savepoint is not None:
                    conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    conn.execute(f"RELEASE SAVEPOINT {savepoint}")
                report.invalid_rows += 1
                report.errors.append({"line": line_number, "error": str(exc)})
        report.inserted_products = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0] - before_products
        report.inserted_stores = conn.execute("SELECT COUNT(*) FROM stores").fetchone()[0] - before_stores
        report.final_inventory_rows = conn.execute("SELECT COUNT(*) FROM inventory_records").fetchone()[0]

    accounted = report.inserted_inventory_rows + report.duplicate_inventory_rows + report.invalid_rows
    if accounted != report.input_rows:
        raise RuntimeError(f"Import accounting mismatch: {accounted} outcomes for {report.input_rows} rows")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    args = parser.parse_args()
    print(json.dumps(import_processed_inventory(args.csv, args.database).to_dict(), indent=2))


if __name__ == "__main__":
    main()
