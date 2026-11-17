import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.preprocessing import (
    load_inventory_csv,
    preprocess_file,
    preprocess_inventory_data,
    validate_required_columns,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "synthetic_inventory.csv"


class InventoryPreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load_inventory_csv(SOURCE)

    def test_csv_loading(self):
        data = load_inventory_csv(SOURCE)
        self.assertEqual(len(data), 10800)
        self.assertIn("timestamp", data.columns)

    def test_required_column_validation(self):
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            validate_required_columns(self.raw.drop(columns="store_id"))

    def test_timestamp_parsing_is_timezone_aware(self):
        clean, _ = preprocess_inventory_data(self.raw)
        self.assertIsInstance(clean["timestamp"].dtype, pd.DatetimeTZDtype)
        self.assertEqual(str(clean["timestamp"].dt.tz), "UTC")

    def test_chronological_sorting(self):
        shuffled = self.raw.sample(frac=1, random_state=5)
        clean, _ = preprocess_inventory_data(shuffled)
        for _, group in clean.groupby(["product_id", "store_id"], sort=False):
            self.assertTrue(group["timestamp"].is_monotonic_increasing)

    def test_exact_duplicate_is_reported_and_removed(self):
        duplicate = pd.concat([self.raw, self.raw.iloc[[0]]], ignore_index=True)
        clean, report = preprocess_inventory_data(duplicate)
        self.assertEqual(report.duplicate_observation_rows, 1)
        self.assertEqual(report.exact_duplicate_rows_removed, 1)
        self.assertEqual(len(clean), len(self.raw))

    def test_conflicting_duplicate_is_rejected(self):
        changed = self.raw.iloc[[0]].copy()
        changed.loc[:, "stock_quantity"] += 1
        duplicate = pd.concat([self.raw, changed], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Conflicting observations"):
            preprocess_inventory_data(duplicate)

    def test_missing_current_status_is_derived_but_history_is_preserved(self):
        incomplete = self.raw.iloc[:6].copy()
        incomplete.loc[incomplete.index[0], "availability_status"] = pd.NA
        incomplete.loc[incomplete.index[1], "previous_stock_status"] = pd.NA
        incomplete.loc[incomplete.index[2], "days_since_last_restock"] = pd.NA
        incomplete.loc[incomplete.index[3], "historical_restock_count"] = pd.NA
        clean, report = preprocess_inventory_data(incomplete)
        self.assertEqual(report.status_values_derived, 1)
        self.assertEqual(clean.loc[0, "availability_status"], "in_stock")
        self.assertTrue(pd.isna(clean.loc[1, "previous_stock_status"]))
        self.assertTrue(pd.isna(clean.loc[2, "days_since_last_restock"]))
        self.assertTrue(pd.isna(clean.loc[3, "historical_restock_count"]))
        self.assertEqual(report.missing_before["historical_restock_count"], 1)
        self.assertEqual(report.missing_after["historical_restock_count"], 1)

    def test_stock_and_status_inconsistency_is_rejected(self):
        invalid = self.raw.iloc[:10].copy()
        invalid.loc[invalid.index[0], "availability_status"] = "out_of_stock"
        with self.assertRaisesRegex(ValueError, "conflicts with stock_quantity"):
            preprocess_inventory_data(invalid)

    def test_historical_feature_leakage_is_rejected(self):
        invalid = self.raw.iloc[:10].copy()
        invalid.loc[invalid.index[0], "historical_restock_count"] = 1
        with self.assertRaisesRegex(ValueError, "historical_restock_count"):
            preprocess_inventory_data(invalid)

    def test_original_csv_is_unchanged(self):
        before = SOURCE.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "processed.csv"
            preprocess_file(SOURCE, output)
            self.assertTrue(output.is_file())
        self.assertEqual(SOURCE.read_bytes(), before)

    def test_repeated_processing_is_consistent(self):
        first, first_report = preprocess_inventory_data(self.raw)
        second, second_report = preprocess_inventory_data(self.raw)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first_report.to_dict(), second_report.to_dict())


if __name__ == "__main__":
    unittest.main()
