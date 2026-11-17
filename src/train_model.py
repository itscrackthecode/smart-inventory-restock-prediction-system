"""Orchestrate chronological splitting, model fitting, and persistence."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import pandas as pd
import sklearn

from src.feature_engineering import FORBIDDEN_MODEL_FEATURES, MODEL_FEATURES, TARGET_COLUMN
from src.model_data import (
    DEFAULT_DATA_PATH,
    TEST_SIZE,
    chronological_train_test_split,
    load_engineered_dataset,
    target_distribution,
)
from src.model_persistence import DEFAULT_MODEL_DIR, save_model_pipelines
from src.model_preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.model_training import RANDOM_STATE, fit_model_pipelines


def train_and_save_models(
    input_path: str | Path = DEFAULT_DATA_PATH,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    *,
    test_size: float = TEST_SIZE,
) -> dict[str, Any]:
    """Fit models on the training partition and save pipelines plus metadata.

    Holdout rows are inspected only for row/date/class-count reporting. They are
    never passed through the fitted preprocessor or model in this phase.
    """
    data = load_engineered_dataset(input_path)
    split = chronological_train_test_split(data, test_size=test_size)
    models = fit_model_pipelines(split.train)

    metadata: dict[str, Any] = {
        "project": "Smart Inventory & Restock Prediction System",
        "target": TARGET_COLUMN,
        "target_definition": "restock_occurred on the next daily observation for the same product/store",
        "split_strategy": {
            "method": "global chronological holdout by unique feature dates",
            "test_size": test_size,
            "holdout_start": split.holdout_start.isoformat(),
            "training_feature_start": split.train_feature_start.isoformat(),
            "training_feature_end": split.train_feature_end.isoformat(),
            "boundary_rule": "A training feature row is retained only when its next-day label date is strictly before the holdout start; crossing rows are purged.",
            "train_rows": len(split.train),
            "holdout_rows": len(split.holdout),
            "purged_boundary_rows": len(split.purged),
            "train_target_distribution": target_distribution(split.train),
            "holdout_target_distribution_report_only": target_distribution(split.holdout),
        },
        "feature_columns": list(MODEL_FEATURES),
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "excluded_columns": sorted(
            FORBIDDEN_MODEL_FEATURES | {"product_id", "store_id", "product_name", "data_source"}
        ),
        "preprocessing": {
            "categorical": "Most-frequent imputation, then OneHotEncoder(handle_unknown='ignore').",
            "numeric": "Median imputation fitted on training rows; Logistic Regression scales values; Random Forest leaves values unscaled.",
            "restock_age": "Unverified baseline values are masked in the Phase 5 model feature; age is median-imputed from training rows and its validity flag is one-hot encoded.",
        },
        "models": {
            "logistic_regression": {
                "class_weight": "balanced", "random_state": RANDOM_STATE, "max_iter": 1000,
            },
            "random_forest": {
                "class_weight": "balanced", "random_state": RANDOM_STATE,
                "n_estimators": 250, "max_features": "sqrt", "min_samples_leaf": 2,
                "n_jobs": -1,
            },
        },
        "fit_rows": len(split.train),
        "holdout_used_for_fit_or_selection": False,
        "phase_note": "Phase 6 training only; holdout performance evaluation is deferred to Phase 7.",
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
    }
    paths = save_model_pipelines(models, metadata, model_dir)
    metadata["artifacts"] = {name: str(path) for name, path in paths.items() if name != "metadata"}
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    args = parser.parse_args()
    metadata = train_and_save_models(args.input, args.model_dir, test_size=args.test_size)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
