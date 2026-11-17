import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.feature_engineering import (
    MODEL_FEATURES,
    TARGET_COLUMN,
    create_next_observation_target,
    engineer_inventory_data,
    load_processed_inventory,
    run_feature_engineering,
    save_engineered_dataset,
    select_model_features,
    validate_engineered_dataset,
    validate_chronological_order,
)


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = ROOT / "data" / "processed_inventory.csv"
SYNTHETIC_PATH = ROOT / "data" / "synthetic_inventory.csv"


def make_series(product_id, store_id, quantities, restock_events, start=None):
    """Build small valid daily history rows for target-alignment tests."""
    first_day = start or datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = []
    last_restock_day = None
    historical_restock_count = 0
    previous_status = "unknown"
    for day_index, (quantity, restocked) in enumerate(zip(quantities, restock_events)):
        timestamp = first_day + timedelta(days=day_index)
        status = "in_stock" if quantity > 0 else "out_of_stock"
        days_since = (
            day_index - last_restock_day
            if last_restock_day is not None
            else day_index + 1
        )
        rows.append(
            {
                "product_id": product_id,
                "product_name": f"Product {product_id}",
                "category": "Grocery",
                "store_id": store_id,
                "store_location": "Austin, TX",
                "timestamp": timestamp,
                "stock_quantity": quantity,
                "availability_status": status,
                "previous_stock_status": previous_status,
                "days_since_last_restock": days_since,
                "historical_restock_count": historical_restock_count,
                "restock_occurred": restocked,
                "data_source": "synthetic",
            }
        )
        previous_status = status
        if restocked:
            historical_restock_count += 1
            last_restock_day = day_index
    return rows


def make_fixture(include_gap=False):
    product_a = make_series("A", "S1", [5, 0, 15, 12], [False, True, False, False])
    product_b = make_series("B", "S1", [8, 6, 0], [False, False, True])
    if include_gap:
        # Remove A's restock day from the observations while retaining later
        # stock history. Its Jan 1 row must not be labeled from Jan 3.
        product_a = make_series("A", "S1", [5, 0, 0], [False, False, False])
        product_a = [product_a[0], product_a[2]]
    return pd.DataFrame(product_a + product_b)


class FeatureEngineeringTests(unittest.TestCase):
    def test_target_uses_next_observation_within_each_product_store(self):
        data = make_fixture()
        labels = create_next_observation_target(data)
        self.assertEqual(int(labels.iloc[0]), 1)  # A Jan 1 -> A Jan 2 restocked
        self.assertEqual(int(labels.iloc[1]), 0)  # A Jan 2 -> A Jan 3 not restocked
        self.assertEqual(int(labels.iloc[2]), 0)  # A Jan 3 -> A Jan 4, not restocked
        self.assertTrue(pd.isna(labels.iloc[3]))  # A final row is unlabeled
        self.assertEqual(int(labels.iloc[4]), 0)  # B Jan 1 -> B Jan 2, not A
        self.assertEqual(int(labels.iloc[5]), 1)  # B Jan 2 -> B Jan 3 restocked

    def test_input_must_be_chronologically_sorted(self):
        data = make_fixture()
        with self.assertRaisesRegex(ValueError, "not chronologically sorted"):
            validate_chronological_order(data.iloc[::-1].reset_index(drop=True))

    def test_missing_daily_observation_is_not_labeled_as_negative(self):
        engineered, report = engineer_inventory_data(make_fixture(include_gap=True))
        product_a = engineered[engineered["product_id"].eq("A")]
        self.assertEqual(len(product_a), 0)
        self.assertEqual(report.continuity_gap_pairs, 1)
        self.assertEqual(report.missing_calendar_days, 1)
        self.assertEqual(report.gap_predecessors_without_label, 1)

    def test_final_observations_without_labels_are_removed(self):
        engineered, report = engineer_inventory_data(make_fixture())
        self.assertEqual(report.final_observations_without_label, 2)
        self.assertEqual(report.input_rows - report.output_rows, 2)
        self.assertFalse(engineered[TARGET_COLUMN].isna().any())

    def test_no_target_or_current_restock_event_is_a_model_feature(self):
        engineered, _ = engineer_inventory_data(make_fixture())
        selected = select_model_features(engineered)
        self.assertEqual(list(selected.columns), MODEL_FEATURES)
        self.assertNotIn(TARGET_COLUMN, selected.columns)
        self.assertNotIn("restock_occurred", selected.columns)

    def test_future_values_do_not_change_current_model_features(self):
        original = make_fixture()
        changed = original.copy(deep=True)
        changed.loc[1, "stock_quantity"] = 99
        changed.loc[1, "availability_status"] = "in_stock"
        changed.loc[1, "restock_occurred"] = False
        changed.loc[2, "historical_restock_count"] = 0
        changed.loc[2, "days_since_last_restock"] = 3
        changed.loc[2, "previous_stock_status"] = "in_stock"
        changed.loc[3, "historical_restock_count"] = 0
        changed.loc[3, "days_since_last_restock"] = 4
        original_features, _ = engineer_inventory_data(original)
        changed_features, _ = engineer_inventory_data(changed)
        first_original = original_features.query("product_id == 'A'").iloc[[0]]
        first_changed = changed_features.query("product_id == 'A'").iloc[[0]]
        pd.testing.assert_frame_equal(
            select_model_features(first_original), select_model_features(first_changed)
        )
        self.assertNotEqual(
            int(first_original[TARGET_COLUMN].iloc[0]),
            int(first_changed[TARGET_COLUMN].iloc[0]),
        )

    def test_historical_features_are_checked_and_baseline_is_masked(self):
        data = make_fixture()
        engineered, _ = engineer_inventory_data(data)
        first_a = engineered[engineered["product_id"].eq("A")].iloc[0]
        self.assertFalse(first_a["days_since_last_restock_valid"])
        self.assertTrue(pd.isna(first_a["days_since_last_restock_model"]))
        later_a = engineered[engineered["product_id"].eq("A")].iloc[2]
        self.assertTrue(later_a["days_since_last_restock_valid"])
        self.assertEqual(int(later_a["days_since_last_restock_model"]), 1)

        corrupted = data.copy()
        corrupted.loc[1, "historical_restock_count"] = 7
        with self.assertRaisesRegex(ValueError, "historical_restock_count"):
            engineer_inventory_data(corrupted)

    def test_engineering_is_reproducible(self):
        first, first_report = engineer_inventory_data(make_fixture())
        second, second_report = engineer_inventory_data(make_fixture())
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first_report.to_dict(), second_report.to_dict())

    def test_saved_csv_round_trip_validates(self):
        engineered, _ = engineer_inventory_data(make_fixture())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "engineered.csv"
            save_engineered_dataset(engineered, output)
            restored = pd.read_csv(output, parse_dates=["timestamp"])
            validate_engineered_dataset(restored)

    def test_loads_processed_dataset_and_preserves_its_order(self):
        source = load_processed_inventory(PROCESSED_PATH)
        self.assertTrue(source["timestamp"].dt.tz is not None)
        self.assertEqual(source["data_source"].unique().tolist(), ["synthetic"])

    def test_original_datasets_remain_unchanged(self):
        original_processed = PROCESSED_PATH.read_bytes()
        original_synthetic = SYNTHETIC_PATH.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "engineered.csv"
            report = run_feature_engineering(PROCESSED_PATH, output)
            self.assertEqual(report.input_rows, 10800)
            self.assertTrue(output.is_file())
        self.assertEqual(PROCESSED_PATH.read_bytes(), original_processed)
        self.assertEqual(SYNTHETIC_PATH.read_bytes(), original_synthetic)


if __name__ == "__main__":
    unittest.main()
