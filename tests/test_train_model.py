import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_engineering import MODEL_FEATURES, TARGET_COLUMN, select_model_features
from src.train_model import (
    train_and_save_models,
)
from src.model_data import chronological_train_test_split, load_engineered_dataset
from src.model_persistence import load_model_pipeline, save_model_pipelines
from src.model_preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_preprocessor
from src.model_training import build_model_pipelines, fit_model_pipelines


ROOT = Path(__file__).resolve().parents[1]
ENGINEERED_PATH = ROOT / "data" / "engineered_inventory.csv"
PROCESSED_PATH = ROOT / "data" / "processed_inventory.csv"
SYNTHETIC_PATH = ROOT / "data" / "synthetic_inventory.csv"


def tiny_training_data(rows=48):
    records = []
    for index in range(rows):
        valid_age = index % 4 != 0
        records.append(
            {
                "category": "grocery" if index % 2 else "home",
                "store_location": "Austin" if index % 3 else "Denver",
                "stock_quantity": float(index % 11),
                "availability_status": "in_stock" if index % 5 else "out_of_stock",
                "previous_stock_status": "in_stock" if index % 4 else "unknown",
                "historical_restock_count": float(index % 6),
                "days_since_last_restock_model": float(index % 9) if valid_age else np.nan,
                "days_since_last_restock_valid": valid_age,
                "day_of_week": index % 7,
                "month": index % 12 + 1,
                TARGET_COLUMN: int(index % 5 == 0),
            }
        )
    return pd.DataFrame(records)


class ModelTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.model_dir = Path(cls.temp_dir.name) / "models"
        cls.source_bytes = {
            path: path.read_bytes()
            for path in (SYNTHETIC_PATH, PROCESSED_PATH, ENGINEERED_PATH)
        }
        cls.metadata = train_and_save_models(ENGINEERED_PATH, cls.model_dir)
        cls.data = load_engineered_dataset(ENGINEERED_PATH)
        cls.split = chronological_train_test_split(cls.data)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_chronological_split_keeps_future_rows_out_of_training(self):
        split = self.split
        self.assertLess(split.train["timestamp"].max(), split.holdout_start)
        self.assertTrue(
            (split.train["timestamp"] + pd.Timedelta(days=1) < split.holdout_start).all()
        )
        self.assertGreaterEqual(split.holdout["timestamp"].min(), split.holdout_start)
        self.assertTrue(split.train["timestamp"].is_monotonic_increasing)
        self.assertTrue(split.holdout["timestamp"].is_monotonic_increasing)

    def test_boundary_labels_are_purged_and_partitions_do_not_overlap(self):
        split = self.split
        self.assertEqual(len(split.purged), 120)
        self.assertTrue(
            (split.purged["timestamp"] + pd.Timedelta(days=1) >= split.holdout_start).all()
        )
        train_keys = set(zip(split.train.product_id, split.train.store_id, split.train.timestamp))
        holdout_keys = set(zip(split.holdout.product_id, split.holdout.store_id, split.holdout.timestamp))
        self.assertFalse(train_keys.intersection(holdout_keys))
        self.assertEqual(len(split.train) + len(split.holdout) + len(split.purged), len(self.data))

    def test_reported_class_distributions_match_partitions(self):
        split_info = self.metadata["split_strategy"]
        self.assertEqual(split_info["train_rows"], len(self.split.train))
        self.assertEqual(split_info["holdout_rows"], len(self.split.holdout))
        self.assertEqual(split_info["purged_boundary_rows"], len(self.split.purged))
        for part_name, partition in (("train", self.split.train), ("holdout", self.split.holdout)):
            distribution = split_info[f"{part_name}_target_distribution" if part_name == "train" else "holdout_target_distribution_report_only"]
            self.assertEqual(distribution["0"]["count"], int(partition[TARGET_COLUMN].eq(0).sum()))
            self.assertEqual(distribution["1"]["count"], int(partition[TARGET_COLUMN].eq(1).sum()))
        self.assertFalse(self.metadata["holdout_used_for_fit_or_selection"])

    def test_feature_selection_excludes_target_and_forbidden_columns(self):
        model_features = select_model_features(self.split.train)
        self.assertEqual(list(model_features.columns), MODEL_FEATURES)
        for excluded in (
            TARGET_COLUMN, "restock_occurred", "product_id", "store_id", "product_name",
            "timestamp", "data_source", "days_since_last_restock",
        ):
            self.assertNotIn(excluded, model_features.columns)

    def test_imputer_statistics_and_encoder_categories_use_training_only(self):
        train = tiny_training_data(8)
        holdout = tiny_training_data(2).iloc[:1].copy()
        holdout.loc[:, "category"] = "holdout-only-category"
        holdout.loc[:, "store_location"] = "holdout-only-location"
        preprocessor = build_preprocessor(scale_numeric=False)
        train_features = select_model_features(train)
        preprocessor.fit(train_features)

        numeric_imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
        expected_stock_median = train_features["stock_quantity"].median()
        self.assertEqual(numeric_imputer.statistics_[NUMERIC_FEATURES.index("stock_quantity")], expected_stock_median)

        cat_pipeline = preprocessor.named_transformers_["categorical"]
        encoder = cat_pipeline.named_steps["one_hot"]
        categories = dict(zip(CATEGORICAL_FEATURES, encoder.categories_))
        self.assertNotIn("holdout-only-category", categories["category"])
        self.assertNotIn("holdout-only-location", categories["store_location"])
        transformed = preprocessor.transform(select_model_features(holdout))
        self.assertEqual(transformed.shape[0], 1)

    def test_missing_values_and_unknown_categories_are_supported(self):
        train = tiny_training_data(12)
        train.loc[0, "category"] = np.nan
        train.loc[1, "stock_quantity"] = np.nan
        train.loc[2, "days_since_last_restock_model"] = np.nan
        preprocessor = build_preprocessor()
        train_features = select_model_features(train)
        preprocessor.fit(train_features)

        unknown = train_features.iloc[[0]].copy()
        unknown.loc[:, "category"] = "new-category"
        unknown.loc[:, "store_location"] = "new-store"
        transformed = preprocessor.transform(unknown)
        values = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
        self.assertTrue(np.isfinite(values).all())

    def test_both_models_train_as_balanced_pipelines(self):
        models = build_model_pipelines()
        self.assertEqual(set(models), {"logistic_regression", "random_forest"})
        trained = fit_model_pipelines(tiny_training_data())
        for model in trained.values():
            self.assertIn("preprocessor", model.named_steps)
            self.assertEqual(model.named_steps["classifier"].class_weight, "balanced")
            self.assertEqual(set(model.classes_), {0, 1})

    def test_training_is_reproducible(self):
        training = tiny_training_data()
        features = select_model_features(training)
        first = fit_model_pipelines(training)
        second = fit_model_pipelines(training)
        for name in first:
            np.testing.assert_allclose(
                first[name].predict_proba(features), second[name].predict_proba(features)
            )

    def test_saved_pipeline_loads_and_predicts(self):
        train_sample = self.split.train.head(5)
        features = select_model_features(train_sample)
        for name in ("logistic_regression", "random_forest"):
            loaded = load_model_pipeline(name, self.model_dir)
            probabilities = loaded.predict_proba(features)
            self.assertEqual(probabilities.shape, (5, 2))
            self.assertTrue(np.isfinite(probabilities).all())

    def test_model_metadata_and_original_csv_preservation(self):
        for path, original_bytes in self.source_bytes.items():
            self.assertEqual(path.read_bytes(), original_bytes)
        self.assertEqual(self.metadata["fit_rows"], len(self.split.train))
        self.assertEqual(self.metadata["feature_columns"], MODEL_FEATURES)
        self.assertTrue((self.model_dir / "model_metadata.json").is_file())


if __name__ == "__main__":
    unittest.main()
