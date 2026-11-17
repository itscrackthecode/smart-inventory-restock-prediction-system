"""Build and fit the two Phase 6 classification pipelines."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.feature_engineering import (
    FORBIDDEN_MODEL_FEATURES,
    MODEL_FEATURES,
    TARGET_COLUMN,
    select_model_features,
)
from src.model_preprocessing import build_preprocessor


RANDOM_STATE = 42


def build_model_pipelines() -> dict[str, Pipeline]:
    """Create Logistic Regression and Random Forest pipelines."""
    logistic = Pipeline(
        [
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    max_iter=1000,
                ),
            ),
        ]
    )
    forest = Pipeline(
        [
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return {"logistic_regression": logistic, "random_forest": forest}


def fit_model_pipelines(training_data):
    """Fit both pipelines using only the supplied training rows."""
    missing = set(MODEL_FEATURES + [TARGET_COLUMN]) - set(training_data.columns)
    if missing:
        raise ValueError(f"Training data is missing required fields: {sorted(missing)}")
    if training_data[TARGET_COLUMN].isna().any() or training_data[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Training data must contain both binary target classes")
    forbidden = FORBIDDEN_MODEL_FEATURES.intersection(MODEL_FEATURES)
    if forbidden:
        raise ValueError(f"Forbidden model inputs were selected: {sorted(forbidden)}")

    training_features = select_model_features(training_data)
    training_target = training_data[TARGET_COLUMN].astype("int8")
    models = build_model_pipelines()
    for pipeline in models.values():
        # No holdout rows are accepted by this function or passed to fit.
        pipeline.fit(training_features, training_target)
    return models
