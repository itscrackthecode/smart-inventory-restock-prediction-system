"""Leakage-aware feature building and inference using Phase 6 model pipelines."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

from app.config import MODEL_DIR, PROJECT_ROOT, REPORTS_DIR
from app.services.inventory_queries import load_product_store_timeline
from src.database.repositories import save_prediction
from src.feature_engineering import MODEL_FEATURES


MODEL_FILE_NAMES = {
    "logistic_regression": "logistic_regression_pipeline.joblib",
    "random_forest": "random_forest_pipeline.joblib",
}
DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}


def build_inference_features(
    timeline: pd.DataFrame,
    *,
    inventory_id: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the exact Phase 5 inputs using only history up to the source row.

    Phase 5's `days_since_last_restock` is the calendar-day distance to the
    most recent *earlier* restock observation. It is masked until an earlier
    restock is present. Calendar distance remains meaningful across missing
    observations; the model target itself still describes the next observed
    chronological row, not necessarily tomorrow.
    """
    required = {
        "inventory_id", "product_id", "store_id", "product_name", "category", "store_location",
        "observation_date", "stock_quantity", "availability_status", "previous_stock_status",
        "restock_occurred",
    }
    missing = required - set(timeline.columns)
    if missing:
        raise ValueError(f"Inventory history is missing required fields: {sorted(missing)}")
    if timeline.empty:
        raise ValueError("No inventory observations are available for this product/store")

    history = timeline.copy(deep=True)
    history["observation_date"] = pd.to_datetime(history["observation_date"], errors="coerce", utc=True)
    if history["observation_date"].isna().any():
        raise ValueError("Inventory history contains an invalid observation timestamp")
    if history[["product_id", "store_id"]].drop_duplicates().shape[0] != 1:
        raise ValueError("History must contain exactly one product/store combination")
    history = history.sort_values(["observation_date", "inventory_id"], kind="stable").reset_index(drop=True)
    if history["observation_date"].duplicated().any():
        raise ValueError("Product/store history contains duplicate observation timestamps")
    if inventory_id is None:
        selected_index = len(history) - 1
    else:
        matches = history.index[history["inventory_id"].eq(inventory_id)].tolist()
        if len(matches) != 1:
            raise ValueError("Selected inventory observation is missing or ambiguous")
        selected_index = matches[0]
    source = history.iloc[selected_index]
    # Truncating here makes future-row use impossible even if callers pass a
    # timeline containing observations after the selected source row.
    prior = history.iloc[:selected_index]
    if int(source["stock_quantity"]) < 0:
        raise ValueError("Source observation has a negative stock quantity")
    expected_status = "in_stock" if int(source["stock_quantity"]) > 0 else "out_of_stock"
    if source["availability_status"] != expected_status:
        raise ValueError("Source observation stock and availability status conflict")

    earlier_restocks = prior.loc[prior["restock_occurred"].astype(bool)]
    history_valid = not earlier_restocks.empty
    if history_valid:
        last_restock_date = earlier_restocks.iloc[-1]["observation_date"]
        restock_age: int | None = int((source["observation_date"].normalize() - last_restock_date.normalize()).days)
        if restock_age < 1:
            raise ValueError("Earlier restock must precede the source observation by at least one calendar day")
    else:
        restock_age = None

    feature_values = {
        "category": str(source["category"]),
        "store_location": str(source["store_location"]),
        "stock_quantity": int(source["stock_quantity"]),
        "availability_status": str(source["availability_status"]),
        "previous_stock_status": str(source["previous_stock_status"]),
        "historical_restock_count": int(prior["restock_occurred"].astype(bool).sum()),
        "days_since_last_restock_model": restock_age,
        "days_since_last_restock_valid": bool(history_valid),
        "day_of_week": int(source["observation_date"].day_of_week),
        "month": int(source["observation_date"].month),
    }
    features = pd.DataFrame([[feature_values[column] for column in MODEL_FEATURES]], columns=MODEL_FEATURES)
    if features.columns.tolist() != MODEL_FEATURES:
        raise ValueError("Inference feature order differs from the Phase 5 training order")
    if source["observation_date"] > history["observation_date"].max():
        raise ValueError("Source observation is later than the available as-of history")
    as_of_history = history.iloc[:selected_index + 1]
    details = {
        "inventory_id": int(source["inventory_id"]),
        "product_id": str(source["product_id"]),
        "product_name": str(source["product_name"]),
        "category": str(source["category"]),
        "store_id": str(source["store_id"]),
        "store_location": str(source["store_location"]),
        "observation_date": source["observation_date"].isoformat(),
        "stock_quantity": int(source["stock_quantity"]),
        "availability_status": str(source["availability_status"]),
        "historical_restock_count": feature_values["historical_restock_count"],
        "days_since_last_restock_model": restock_age,
        "days_since_last_restock_valid": history_valid,
        "history_observation_count": selected_index + 1,
        "has_observation_gap": bool(
            as_of_history["observation_date"].diff().dropna().gt(pd.Timedelta(days=1)).any()
        ),
    }
    return features, details


def format_model_inputs_for_display(features: pd.DataFrame) -> pd.DataFrame:
    """Return a one-column, Arrow-safe view of a single model input row."""
    if len(features) != 1:
        raise ValueError("Model input display expects exactly one observation row")
    display = features.iloc[0].to_frame(name="value")
    display["value"] = display["value"].astype("string").fillna("Unknown")
    return display


@st.cache_resource(show_spinner="Loading trained inventory models…")
def load_inference_bundle(model_dir: str | Path | None = None,
                          report_path: str | Path | None = None) -> dict[str, Any]:
    """Load the fitted pipelines and Phase 7 thresholds once per app process."""
    models_path = Path(model_dir) if model_dir is not None else MODEL_DIR
    report_file = Path(report_path) if report_path is not None else REPORTS_DIR / "model_evaluation.json"
    metadata_file = models_path / "model_metadata.json"
    if not report_file.is_file() or not metadata_file.is_file():
        raise FileNotFoundError("Model metadata or Phase 7 evaluation report is missing")
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    report = json.loads(report_file.read_text(encoding="utf-8"))
    expected_features = metadata.get("feature_columns")
    if expected_features != MODEL_FEATURES:
        raise ValueError("Model metadata feature list does not match the Phase 5 feature order")
    thresholds: dict[str, float] = {}
    pipelines: dict[str, Any] = {}
    versions: dict[str, str] = {}
    for key, filename in MODEL_FILE_NAMES.items():
        model_file = models_path / filename
        if not model_file.is_file():
            raise FileNotFoundError(f"Trained model pipeline is missing: {model_file}")
        metric = report.get("models", {}).get(key, {}).get("threshold_selection", {})
        threshold = metric.get("threshold")
        if threshold is None or not 0 <= float(threshold) <= 1:
            raise ValueError(f"Phase 7 threshold is missing or invalid for {key}")
        pipeline = joblib.load(model_file)
        if list(getattr(pipeline, "feature_names_in_", [])) != MODEL_FEATURES:
            raise ValueError(f"Saved {key} pipeline input columns do not match Phase 5")
        if 1 not in list(pipeline.classes_):
            raise ValueError(f"Saved {key} pipeline does not contain positive class 1")
        pipelines[key] = pipeline
        thresholds[key] = float(threshold)
        versions[key] = hashlib.sha256(model_file.read_bytes()).hexdigest()[:12]
    return {
        "pipelines": pipelines,
        "thresholds": thresholds,
        "versions": versions,
        "metadata": metadata,
        "report": report,
    }


def predict_for_product_store(database_path: str | Path, *, product_id: str,
                              store_id: str, model_name: str,
                              bundle: dict[str, Any] | None = None) -> dict[str, Any]:
    """Predict and persist one selected model's next-observation probability."""
    if model_name not in MODEL_FILE_NAMES:
        raise ValueError(f"Unsupported model: {model_name}")
    loaded = bundle or load_inference_bundle()
    timeline = load_product_store_timeline(database_path, product_id, store_id)
    features, source_details = build_inference_features(timeline)
    pipeline = loaded["pipelines"][model_name]
    probabilities = pipeline.predict_proba(features)
    classes = list(pipeline.classes_)
    positive_indices = [index for index, label in enumerate(classes) if label == 1]
    if len(positive_indices) != 1:
        raise ValueError("Could not identify exactly one positive restock class")
    probability = float(probabilities[0, positive_indices[0]])
    threshold = float(loaded["thresholds"][model_name])
    predicted = probability >= threshold
    persisted = save_prediction(
        inventory_id=source_details["inventory_id"],
        model_name=DISPLAY_NAMES[model_name],
        model_version=loaded["versions"][model_name],
        predicted_probability=probability,
        prediction_threshold=threshold,
        predicted_restock=predicted,
        created_at=datetime.now(timezone.utc),
        database_path=database_path,
    )
    return {
        **source_details,
        "model_key": model_name,
        "model_name": DISPLAY_NAMES[model_name],
        "model_version": loaded["versions"][model_name],
        "probability": probability,
        "threshold": threshold,
        "predicted_restock": bool(predicted),
        "prediction_id": persisted["prediction_id"],
        "created_at": persisted["created_at"],
        "features": features,
    }
