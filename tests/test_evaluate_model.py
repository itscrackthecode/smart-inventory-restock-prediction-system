import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate_model import (
    DEFAULT_THRESHOLD,
    evaluate_binary_predictions,
    extract_logistic_coefficients,
    extract_random_forest_importance,
    run_model_evaluation,
    select_validation_threshold,
)
from src.feature_engineering import MODEL_FEATURES, TARGET_COLUMN, select_model_features
from src.model_data import chronological_train_test_split, load_engineered_dataset
from src.model_persistence import load_model_pipeline


ROOT = Path(__file__).resolve().parents[1]
ENGINEERED = ROOT / "data" / "engineered_inventory.csv"
SOURCE_CSVS = [
    ROOT / "data" / "synthetic_inventory.csv",
    ROOT / "data" / "processed_inventory.csv",
    ENGINEERED,
]
MODEL_FILES = [
    ROOT / "models" / "logistic_regression_pipeline.joblib",
    ROOT / "models" / "random_forest_pipeline.joblib",
    ROOT / "models" / "model_metadata.json",
]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ModelEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.report_dir = Path(cls.temp_dir.name) / "reports"
        cls.input_hashes = {path: file_hash(path) for path in SOURCE_CSVS + MODEL_FILES}
        cls.report = run_model_evaluation(report_dir=cls.report_dir)
        cls.data = load_engineered_dataset(ENGINEERED)
        cls.split = chronological_train_test_split(cls.data)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_exact_phase6_chronological_holdout_reconstruction(self):
        evaluation = self.report["evaluation"]
        self.assertEqual(evaluation["holdout_start"], "2025-03-13T00:00:00+00:00")
        self.assertEqual(evaluation["rows"], 2160)
        self.assertEqual(len(self.split.train), 8400)
        self.assertEqual(len(self.split.holdout), 2160)
        self.assertEqual(len(self.split.purged), 120)

    def test_boundary_purge_is_preserved(self):
        cutoff = self.split.holdout_start
        self.assertTrue((self.split.train.timestamp + pd.Timedelta(days=1) < cutoff).all())
        self.assertTrue((self.split.purged.timestamp + pd.Timedelta(days=1) >= cutoff).all())
        self.assertTrue((self.split.holdout.timestamp >= cutoff).all())
        self.assertEqual(self.report["evaluation"]["purged_boundary_rows"], 120)
        self.assertEqual(self.report["evaluation"]["training_label_date_end"], "2025-03-12T00:00:00+00:00")

    def test_metric_calculations_include_positive_class_and_confusion_matrix(self):
        truth = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.8, 0.6, 0.4])
        result = evaluate_binary_predictions(truth, probabilities, threshold=0.5)
        self.assertAlmostEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["f1"], 0.5)
        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertAlmostEqual(result["roc_auc"], 0.5)
        self.assertIn("restock", result["classification_report"])

    def test_undefined_positive_metrics_are_zero_not_exceptions(self):
        result = evaluate_binary_predictions(
            np.array([0, 1, 1]), np.array([0.1, 0.2, 0.3]), threshold=0.9
        )
        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["f1"], 0.0)

    def test_probability_prediction_from_saved_pipelines(self):
        holdout_sample = self.split.holdout.head(8)
        features = select_model_features(holdout_sample)
        for name in ("logistic_regression", "random_forest"):
            model = load_model_pipeline(name)
            probability = model.predict_proba(features)
            self.assertEqual(probability.shape, (8, 2))
            self.assertTrue(np.isfinite(probability).all())
            np.testing.assert_allclose(probability.sum(axis=1), 1.0)

    def test_feature_importance_and_coefficients_map_to_encoded_inputs(self):
        forest = load_model_pipeline("random_forest")
        logistic = load_model_pipeline("logistic_regression")
        importance = extract_random_forest_importance(forest)
        coefficients = extract_logistic_coefficients(logistic)
        self.assertAlmostEqual(importance["importance"].sum(), 1.0)
        self.assertTrue(set(importance["source_feature"]).issubset(set(MODEL_FEATURES)))
        self.assertTrue(set(coefficients["source_feature"]).issubset(set(MODEL_FEATURES)))
        self.assertIn("categorical__category_", " ".join(importance["transformed_feature"]))
        self.assertTrue(np.isfinite(coefficients["coefficient"]).all())

    def test_threshold_selection_uses_only_validation_inputs(self):
        validation_y = np.array([0, 0, 1, 1, 1, 0])
        validation_probability = np.array([0.1, 0.45, 0.4, 0.65, 0.8, 0.3])
        first = select_validation_threshold(validation_y, validation_probability)
        second = select_validation_threshold(validation_y, validation_probability)
        self.assertEqual(first, second)
        self.assertEqual(
            set(inspect.signature(select_validation_threshold).parameters),
            {"y_validation", "validation_probabilities", "candidates"},
        )
        self.assertEqual(first["selection_metric"], "validation F1")
        self.assertEqual(first["candidate_count"], 91)
        self.assertFalse(self.report["evaluation"]["threshold_validation"]["holdout_used_for_threshold_selection"])
        self.assertTrue(
            pd.Timestamp(self.report["evaluation"]["threshold_validation"]["feature_date_end"])
            < self.split.holdout_start
        )

    def test_default_baseline_and_selected_threshold_results_are_reported(self):
        for name in ("logistic_regression", "random_forest"):
            model_report = self.report["models"][name]
            self.assertEqual(model_report["holdout_default_threshold"]["threshold"], DEFAULT_THRESHOLD)
            self.assertEqual(
                model_report["holdout_selected_threshold"]["threshold"],
                model_report["threshold_selection"]["threshold"],
            )
            self.assertEqual(
                sum(map(sum, model_report["holdout_selected_threshold"]["confusion_matrix"])), 2160
            )

    def test_report_and_all_figure_files_are_generated(self):
        self.assertTrue((self.report_dir / "model_evaluation.json").is_file())
        self.assertTrue((self.report_dir / "model_comparison.csv").is_file())
        self.assertTrue((self.report_dir / "random_forest_feature_importance.csv").is_file())
        self.assertTrue((self.report_dir / "logistic_regression_coefficients.csv").is_file())
        self.assertEqual(len(list((self.report_dir / "figures").glob("*.png"))), 7)
        saved = json.loads((self.report_dir / "model_evaluation.json").read_text(encoding="utf-8"))
        self.assertFalse(saved["evaluation"]["holdout_used_for_fit_or_tuning"])

    def test_saved_models_and_all_source_csvs_remain_unchanged(self):
        for path, old_hash in self.input_hashes.items():
            self.assertEqual(file_hash(path), old_hash, str(path))


if __name__ == "__main__":
    unittest.main()
