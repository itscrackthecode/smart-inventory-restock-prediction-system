"""Scikit-learn preprocessing for the selected Phase 5 features."""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_FEATURES = [
    "category",
    "store_location",
    "availability_status",
    "previous_stock_status",
    "days_since_last_restock_valid",
]
NUMERIC_FEATURES = [
    "stock_quantity",
    "historical_restock_count",
    "days_since_last_restock_model",
    "day_of_week",
    "month",
]


def build_preprocessor(*, scale_numeric: bool = True) -> ColumnTransformer:
    """Build imputers, encoding, and optional numeric scaling.

    This object is always placed inside a model Pipeline, so ``fit`` learns
    imputation statistics and categories from the training partition only.
    """
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
