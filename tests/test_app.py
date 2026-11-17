from __future__ import annotations

import os
import sqlite3
import csv
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from app.config import PROJECT_ROOT, database_path
from app.dashboard import PAGE_SPECS
from app.pages.model_performance import load_phase7_report, performance_table
from app.services.inference import (
    build_inference_features,
    format_model_inputs_for_display,
    load_inference_bundle,
    predict_for_product_store,
)
from app.services.inventory_operations import create_inventory_observation
from app.services.startup import ensure_database_ready
from app.services.inventory_queries import (
    count_recent_predictions,
    load_catalogs,
    load_inventory_frame,
    load_prediction_history,
    load_product_store_timeline,
    table_counts,
)
from src.database.repositories import create_product, create_store, insert_inventory_record
from src.database.schema import initialize_database
from src.feature_engineering import MODEL_FEATURES


ROOT = Path(__file__).resolve().parents[1]


class AppServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "app_test.sqlite"
        initialize_database(self.db)
        create_product("P-1", "Arabica Coffee", "Grocery", self.db)
        create_store("S-1", "Austin, TX", self.db)

    def tearDown(self):
        self.temp.cleanup()

    def add_row(self, day: int, *, stock: int = 10, restock: bool = False,
                previous: str = "unknown") -> dict:
        return insert_inventory_record(
            product_id="P-1", store_id="S-1",
            observation_date=f"2025-01-{day:02d}T00:00:00+00:00",
            stock_quantity=stock, availability_status="in_stock" if stock else "out_of_stock",
            previous_stock_status=previous, restock_occurred=restock,
            data_source="synthetic", database_path=self.db,
        )

    def test_application_configuration_uses_project_relative_default_and_override(self):
        old = os.environ.pop("INVENTORY_DB_PATH", None)
        try:
            self.assertEqual(database_path(), PROJECT_ROOT / "data" / "inventory.db")
            os.environ["INVENTORY_DB_PATH"] = "data/custom.sqlite"
            self.assertEqual(database_path(), (PROJECT_ROOT / "data" / "custom.sqlite").resolve())
        finally:
            if old is None:
                os.environ.pop("INVENTORY_DB_PATH", None)
            else:
                os.environ["INVENTORY_DB_PATH"] = old

    def test_database_queries_and_product_store_date_filters(self):
        self.add_row(1)
        self.add_row(2, stock=0, restock=True, previous="in_stock")
        products, stores = load_catalogs(self.db)
        self.assertEqual(products.loc[0, "product_id"], "P-1")
        self.assertEqual(stores.loc[0, "store_id"], "S-1")
        filtered = load_inventory_frame(
            self.db, search="Coffee", categories=["Grocery"], store_ids=["S-1"],
            start_date=date(2025, 1, 2), end_date=date(2025, 1, 2),
        )
        self.assertEqual(filtered[["product_id", "store_id"]].values.tolist(), [["P-1", "S-1"]])
        self.assertEqual(len(load_inventory_frame(self.db, search="%' OR 1=1 --")), 0)
        self.assertEqual(load_product_store_timeline(self.db, "P-1", "S-1").shape[0], 2)
        self.assertEqual(table_counts(self.db), {
            "products": 1, "stores": 1, "inventory_records": 2, "predictions": 0,
        })

    def test_inventory_observation_creation_derives_previous_status(self):
        self.add_row(1, stock=0)
        record = create_inventory_observation(
            self.db, product_id="P-1", store_id="S-1", observation_date=date(2025, 1, 2),
            stock_quantity=8, availability_status="in_stock", restock_occurred=False,
        )
        self.assertEqual(record["previous_stock_status"], "out_of_stock")
        self.assertEqual(record["data_source"], "manual")
        with self.assertRaisesRegex(ValueError, "requires status"):
            create_inventory_observation(
                self.db, product_id="P-1", store_id="S-1", observation_date=date(2025, 1, 3),
                stock_quantity=0, availability_status="in_stock", restock_occurred=False,
            )

    def test_duplicate_observation_is_not_overwritten(self):
        self.add_row(1, stock=10)
        with self.assertRaises(sqlite3.IntegrityError):
            create_inventory_observation(
                self.db, product_id="P-1", store_id="S-1", observation_date=date(2025, 1, 1),
                stock_quantity=1, availability_status="in_stock", restock_occurred=True,
            )
        records = load_inventory_frame(self.db)
        self.assertEqual(len(records), 1)
        self.assertEqual(records.iloc[0]["stock_quantity"], 10)

    def test_inference_features_match_phase5_and_exclude_current_restock(self):
        self.add_row(1, restock=False)
        self.add_row(2, restock=True, previous="in_stock")
        current = self.add_row(3, restock=True, previous="in_stock")
        timeline = load_product_store_timeline(self.db, "P-1", "S-1")
        features, details = build_inference_features(timeline)
        self.assertEqual(features.columns.tolist(), MODEL_FEATURES)
        self.assertEqual(features.loc[0, "historical_restock_count"], 1)
        self.assertEqual(features.loc[0, "days_since_last_restock_model"], 1)
        self.assertTrue(features.loc[0, "days_since_last_restock_valid"])
        self.assertEqual(details["inventory_id"], current["inventory_id"])
        self.assertEqual(features.loc[0, "day_of_week"], 4)
        self.assertEqual(features.loc[0, "month"], 1)

    def test_initial_unknown_restock_history_is_masked(self):
        self.add_row(1, restock=True)
        timeline = load_product_store_timeline(self.db, "P-1", "S-1")
        features, _ = build_inference_features(timeline)
        self.assertFalse(features.loc[0, "days_since_last_restock_valid"])
        self.assertTrue(pd.isna(features.loc[0, "days_since_last_restock_model"]))
        self.assertEqual(features.loc[0, "historical_restock_count"], 0)

    def test_model_input_display_has_consistent_arrow_safe_values(self):
        features = pd.DataFrame([{
            "category": "Grocery", "stock_quantity": 12, "valid": True,
            "restock_age": float("nan"),
        }])
        display = format_model_inputs_for_display(features)
        self.assertEqual(display.index.tolist(), features.columns.tolist())
        self.assertEqual(str(display["value"].dtype), "string")
        self.assertEqual(display.loc["stock_quantity", "value"], "12")
        self.assertEqual(display.loc["restock_age", "value"], "Unknown")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            format_model_inputs_for_display(pd.concat([features, features], ignore_index=True))

    def test_date_gaps_use_calendar_age_and_are_reported(self):
        self.add_row(1, restock=True)
        self.add_row(4, previous="in_stock")
        timeline = load_product_store_timeline(self.db, "P-1", "S-1")
        features, details = build_inference_features(timeline)
        self.assertEqual(features.loc[0, "days_since_last_restock_model"], 3)
        self.assertTrue(details["has_observation_gap"])

    def test_selected_older_observation_does_not_use_future_events(self):
        source = self.add_row(1, restock=False)
        self.add_row(2, restock=True, previous="in_stock")
        timeline = load_product_store_timeline(self.db, "P-1", "S-1")
        features, details = build_inference_features(timeline, inventory_id=source["inventory_id"])
        self.assertEqual(details["inventory_id"], source["inventory_id"])
        self.assertEqual(features.loc[0, "historical_restock_count"], 0)
        self.assertFalse(features.loc[0, "days_since_last_restock_valid"])

    def test_missing_history_is_explained(self):
        with self.assertRaisesRegex(ValueError, "No inventory observations"):
            predict_for_product_store(self.db, product_id="P-1", store_id="S-1", model_name="random_forest")

    def test_models_load_feature_order_thresholds_and_positive_class(self):
        bundle = load_inference_bundle()
        self.assertEqual(bundle["thresholds"], {"logistic_regression": 0.75, "random_forest": 0.53})
        for pipeline in bundle["pipelines"].values():
            self.assertEqual(list(pipeline.feature_names_in_), MODEL_FEATURES)
            self.assertIn(1, pipeline.classes_)

    def test_inference_probabilities_threshold_and_prediction_persistence(self):
        # Seed an in-distribution current observation and an earlier restock.
        create_product("P001", "Arabica Coffee 500g", "Grocery", self.db)
        create_store("S001", "Austin, TX", self.db)
        insert_inventory_record(
            product_id="P001", store_id="S001", observation_date="2025-01-01T00:00:00Z",
            stock_quantity=20, availability_status="in_stock", previous_stock_status="unknown",
            restock_occurred=False, data_source="synthetic", database_path=self.db,
        )
        insert_inventory_record(
            product_id="P001", store_id="S001", observation_date="2025-01-02T00:00:00Z",
            stock_quantity=35, availability_status="in_stock", previous_stock_status="in_stock",
            restock_occurred=True, data_source="synthetic", database_path=self.db,
        )
        insert_inventory_record(
            product_id="P001", store_id="S001", observation_date="2025-01-03T00:00:00Z",
            stock_quantity=30, availability_status="in_stock", previous_stock_status="in_stock",
            restock_occurred=False, data_source="synthetic", database_path=self.db,
        )
        bundle = load_inference_bundle()
        results = [
            predict_for_product_store(self.db, product_id="P001", store_id="S001", model_name=key, bundle=bundle)
            for key in ("logistic_regression", "random_forest")
        ]
        self.assertTrue(all(0 <= result["probability"] <= 1 for result in results))
        self.assertEqual([result["threshold"] for result in results], [0.75, 0.53])
        self.assertTrue(all(result["predicted_restock"] == (result["probability"] >= result["threshold"])
                            for result in results))
        self.assertEqual(len(load_prediction_history(self.db, "P001", "S001")), 2)
        self.assertEqual(table_counts(self.db)["predictions"], 2)
        self.assertEqual(count_recent_predictions(self.db, now=datetime.now(timezone.utc)), 2)

    def test_saved_phase7_model_comparison_data_is_displayable(self):
        report = load_phase7_report()
        table = performance_table(report)
        self.assertEqual(table["Model"].tolist(), ["Logistic Regression", "Random Forest"])
        self.assertEqual(table["Threshold"].tolist(), [0.75, 0.53])
        self.assertEqual(report["evaluation"]["rows"], 2160)

    def test_application_registers_all_seven_pages(self):
        titles = [page[1] for group in PAGE_SPECS.values() for page in group]
        self.assertEqual(len(titles), 7)
        self.assertIn("Restock predictions", titles)
        self.assertIn("Sales intelligence", titles)


class DatabaseStartupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = self.root / "demo" / "inventory.sqlite"
        self.csv_path = self.root / "processed.csv"
        self.columns = [
            "product_id", "product_name", "category", "store_id", "store_location",
            "timestamp", "stock_quantity", "availability_status", "previous_stock_status",
            "restock_occurred", "data_source",
        ]
        self.rows = [
            ["P-1", "Coffee", "Grocery", "S-1", "Austin, TX", "2025-01-01T00:00:00+00:00", 8, "in_stock", "unknown", "False", "synthetic"],
            ["P-1", "Coffee", "Grocery", "S-1", "Austin, TX", "2025-01-02T00:00:00+00:00", 0, "out_of_stock", "in_stock", "False", "synthetic"],
        ]
        self.write_csv(self.rows)

    def tearDown(self):
        self.temp.cleanup()

    def write_csv(self, rows):
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.columns)
            writer.writerows(rows)

    def test_first_start_initializes_and_seeds_processed_data(self):
        report = ensure_database_ready(self.db, self.csv_path)
        self.assertEqual(report.seeded_rows, 2)
        self.assertEqual(report.total_inventory_rows, 2)
        self.assertTrue(self.db.is_file())
        self.assertEqual(table_counts(self.db)["products"], 1)

    def test_repeated_startup_is_idempotent_and_does_not_reseed_existing_rows(self):
        first = ensure_database_ready(self.db, self.csv_path)
        second = ensure_database_ready(self.db, self.csv_path)
        self.assertEqual(first.seeded_rows, 2)
        self.assertEqual(second.seeded_rows, 0)
        self.assertEqual(second.total_inventory_rows, 2)

    def test_existing_database_contents_are_preserved(self):
        initialize_database(self.db)
        create_product("P-1", "Coffee", "Grocery", self.db)
        create_store("S-1", "Austin, TX", self.db)
        insert_inventory_record(
            product_id="P-1", store_id="S-1", observation_date="2025-01-01T00:00:00+00:00",
            stock_quantity=4, availability_status="in_stock", previous_stock_status="unknown",
            restock_occurred=False, data_source="manual", database_path=self.db,
        )
        report = ensure_database_ready(self.db, self.csv_path)
        self.assertEqual(report.seeded_rows, 0)
        self.assertEqual(report.total_inventory_rows, 1)
        existing = load_inventory_frame(self.db)
        self.assertEqual(existing.loc[0, "stock_quantity"], 4)
        self.assertEqual(existing.loc[0, "data_source"], "manual")

    def test_seed_source_is_required_for_an_empty_database(self):
        with self.assertRaises(FileNotFoundError):
            ensure_database_ready(self.db, self.root / "missing.csv")


if __name__ == "__main__":
    unittest.main()
