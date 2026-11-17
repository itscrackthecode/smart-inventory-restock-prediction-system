from __future__ import annotations

import csv
import hashlib
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.database.connection import database_connection
from src.database.repositories import (
    create_product,
    create_store,
    delete_product,
    get_inventory_record,
    get_product,
    get_store,
    insert_inventory_record,
    latest_inventory_observations,
    latest_predictions,
    list_inventory_by_product,
    list_inventory_by_store,
    list_inventory_records,
    list_predictions,
    list_products,
    list_stores,
    prediction_history,
    save_prediction,
    update_inventory_record,
    update_product,
)
from src.database.schema import initialize_database
from src.database.seed import import_processed_inventory


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed_inventory.csv"
SYNTHETIC = ROOT / "data" / "synthetic_inventory.csv"
ENGINEERED = ROOT / "data" / "engineered_inventory.csv"
MODEL_FILES = [ROOT / "models" / "logistic_regression_pipeline.joblib", ROOT / "models" / "random_forest_pipeline.joblib"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "test_inventory.sqlite"
        initialize_database(self.db)

    def tearDown(self):
        self.temp.cleanup()

    def seed_entities(self):
        create_product("P-1", "Coffee", "Grocery", self.db)
        create_store("S-1", "Austin, TX", self.db)

    def insert_record(self, when="2025-01-01T00:00:00+00:00", stock=7, restock=False):
        return insert_inventory_record(
            product_id="P-1", store_id="S-1", observation_date=when,
            stock_quantity=stock, availability_status="in_stock" if stock else "out_of_stock",
            previous_stock_status="unknown", restock_occurred=restock,
            data_source="synthetic", database_path=self.db,
        )

    def test_initialization_creates_required_tables_and_is_idempotent(self):
        self.seed_entities()
        initialize_database(self.db)
        with database_connection(self.db) as conn:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertTrue({"products", "stores", "inventory_records", "predictions"}.issubset(tables))
        self.assertEqual(get_product("P-1", self.db)["product_name"], "Coffee")

    def test_foreign_keys_are_enabled_and_invalid_references_rejected(self):
        with database_connection(self.db) as conn:
            self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)
        with self.assertRaises(sqlite3.IntegrityError):
            insert_inventory_record(product_id="missing", store_id="missing", observation_date="2025-01-01T00:00:00Z",
                                    stock_quantity=1, availability_status="in_stock", previous_stock_status="unknown",
                                    restock_occurred=False, data_source="synthetic", database_path=self.db)

    def test_product_store_and_inventory_crud(self):
        self.seed_entities()
        self.assertEqual(get_product("P-1", self.db)["category"], "Grocery")
        self.assertEqual(get_store("S-1", self.db)["store_location"], "Austin, TX")
        self.assertEqual(len(list_products(self.db)), 1)
        self.assertEqual(len(list_stores(self.db)), 1)
        updated = update_product("P-1", product_name="Coffee 500g", category="Grocery", database_path=self.db)
        self.assertEqual(updated["product_name"], "Coffee 500g")
        inventory = self.insert_record()
        self.assertEqual(get_inventory_record(inventory["inventory_id"], self.db)["stock_quantity"], 7)
        changed = update_inventory_record(inventory["inventory_id"], stock_quantity=0,
                                          availability_status="out_of_stock", database_path=self.db)
        self.assertEqual(changed["availability_status"], "out_of_stock")

    def test_duplicate_observation_is_prevented(self):
        self.seed_entities()
        self.insert_record()
        with self.assertRaises(sqlite3.IntegrityError):
            self.insert_record()

    def test_safe_product_delete_is_blocked_when_inventory_references_it(self):
        self.seed_entities()
        self.insert_record()
        with self.assertRaises(sqlite3.IntegrityError):
            delete_product("P-1", self.db)

    def test_invalid_stock_status_and_timestamp_are_rejected(self):
        self.seed_entities()
        with self.assertRaisesRegex(ValueError, "conflicts"):
            insert_inventory_record(product_id="P-1", store_id="S-1", observation_date="2025-01-01T00:00:00Z",
                                    stock_quantity=0, availability_status="in_stock", previous_stock_status="unknown",
                                    restock_occurred=False, data_source="synthetic", database_path=self.db)
        with self.assertRaisesRegex(ValueError, "timezone"):
            self.insert_record(when="2025-01-01T00:00:00")

    def test_filters_date_range_latest_and_order(self):
        self.seed_entities()
        first = self.insert_record("2025-01-01T02:00:00+02:00", stock=7)
        second = self.insert_record("2025-01-02T00:00:00Z", stock=9)
        self.assertEqual(first["observation_date"], "2025-01-01T00:00:00+00:00")
        self.assertEqual(len(list_inventory_by_product("P-1", self.db)), 2)
        self.assertEqual(len(list_inventory_by_store("S-1", self.db)), 2)
        ranged = list_inventory_records(start_date="2025-01-02T00:00:00Z", end_date="2025-01-02T00:00:00Z", database_path=self.db)
        self.assertEqual([row["inventory_id"] for row in ranged], [second["inventory_id"]])
        self.assertEqual(latest_inventory_observations(database_path=self.db)[0]["inventory_id"], second["inventory_id"])

    def test_parameterized_filters_do_not_treat_input_as_sql(self):
        self.seed_entities()
        self.insert_record()
        self.assertEqual(list_inventory_by_product("P-1' OR 1=1 --", self.db), [])

    def test_prediction_validation_storage_and_retrieval(self):
        self.seed_entities()
        inventory = self.insert_record()
        prediction = save_prediction(inventory_id=inventory["inventory_id"], model_name="Logistic Regression",
                                     model_version="phase6", predicted_probability=0.82, prediction_threshold=0.75,
                                     predicted_restock=True, created_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
                                     database_path=self.db)
        self.assertEqual(prediction["predicted_probability"], 0.82)
        self.assertEqual(len(list_predictions(inventory_id=inventory["inventory_id"], database_path=self.db)), 1)
        history = prediction_history(product_id="P-1", store_id="S-1", database_path=self.db)
        self.assertEqual(history[0]["predicted_restock"], 1)
        self.assertEqual(latest_predictions(database_path=self.db)[0]["prediction_id"], prediction["prediction_id"])
        later_inventory = self.insert_record("2025-01-03T00:00:00Z", stock=8)
        later_prediction = save_prediction(inventory_id=later_inventory["inventory_id"], model_name="Random Forest",
                                           model_version="phase6", predicted_probability=0.61, prediction_threshold=0.53,
                                           predicted_restock=True, created_at=datetime(2025, 1, 3, tzinfo=timezone.utc),
                                           database_path=self.db)
        self.assertEqual(len(prediction_history(product_id="P-1", store_id="S-1", database_path=self.db)), 2)
        self.assertEqual(latest_predictions(database_path=self.db)[0]["prediction_id"], later_prediction["prediction_id"])
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            save_prediction(inventory_id=inventory["inventory_id"], model_name="RF", model_version="1",
                            predicted_probability=1.1, prediction_threshold=0.5, predicted_restock=False,
                            database_path=self.db)

    def test_database_transaction_rolls_back_on_exception(self):
        with self.assertRaises(RuntimeError):
            with database_connection(self.db) as conn:
                conn.execute("INSERT INTO products VALUES (?, ?, ?)", ("P-rollback", "Temp", "Other"))
                raise RuntimeError("force rollback")
        self.assertIsNone(get_product("P-rollback", self.db))

    def test_connection_is_closed_after_context_exit(self):
        with database_connection(self.db) as conn:
            pass
        with self.assertRaises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")

    def test_import_is_idempotent_and_preserves_csv_and_models(self):
        protected = [PROCESSED, SYNTHETIC, ENGINEERED, *MODEL_FILES]
        before = {path: sha256(path) for path in protected}
        first = import_processed_inventory(PROCESSED, self.db)
        second = import_processed_inventory(PROCESSED, self.db)
        self.assertEqual(first.input_rows, 10800)
        self.assertEqual(first.inserted_inventory_rows, 10800)
        self.assertEqual(first.invalid_rows, 0)
        self.assertEqual(second.inserted_inventory_rows, 0)
        self.assertEqual(second.duplicate_inventory_rows, 10800)
        self.assertEqual(second.final_inventory_rows, 10800)
        self.assertEqual(second.inserted_products, 0)
        self.assertEqual(second.inserted_stores, 0)
        self.assertEqual(before, {path: sha256(path) for path in protected})

    def test_invalid_import_row_is_reported_and_rolled_back_per_row(self):
        with PROCESSED.open("r", newline="", encoding="utf-8") as src:
            reader = csv.DictReader(src)
            row = next(reader)
            fields = reader.fieldnames
        row["stock_quantity"] = "-4"
        fixture = Path(self.temp.name) / "invalid.csv"
        with fixture.open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)
        report = import_processed_inventory(fixture, self.db)
        self.assertEqual((report.input_rows, report.invalid_rows, report.final_inventory_rows), (1, 1, 0))
        self.assertEqual(report.errors[0]["line"], 2)
        self.assertIsNone(get_product(row["product_id"], self.db))
        self.assertIsNone(get_store(row["store_id"], self.db))

    def test_import_requires_real_operational_columns(self):
        fixture = Path(self.temp.name) / "bad_header.csv"
        fixture.write_text("product_id,product_name\nP1,Name\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing import columns"):
            import_processed_inventory(fixture, self.db)


if __name__ == "__main__":
    unittest.main()
