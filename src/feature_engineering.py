"""Create leakage-aware model features and next-observation restock labels."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.preprocessing import (
    KEY_COLUMNS,
    REQUIRED_COLUMNS,
    load_inventory_csv,
    preprocess_inventory_data,
    validate_required_columns,
)


GROUP_COLUMNS = ["product_id", "store_id"]
TARGET_COLUMN = "target_next_day_restock"
MODEL_FEATURES = [
    "category",
    "store_location",
    "stock_quantity",
    "availability_status",
    "previous_stock_status",
    "historical_restock_count",
    "days_since_last_restock_model",
    "days_since_last_restock_valid",
    "day_of_week",
    "month",
]
FORBIDDEN_MODEL_FEATURES = {
    TARGET_COLUMN,
    "restock_occurred",
    "timestamp",
    "product_id",
    "store_id",
    "product_name",
    "data_source",
    "days_since_last_restock",
}
NEXT_DAY = pd.Timedelta(days=1)


@dataclass
class ContinuityReport:
    """Non-daily gaps found within product/store observation timelines."""

    product_store_groups: int
    gap_pairs: list[dict[str, Any]] = field(default_factory=list)
    missing_calendar_days: int = 0

    @property
    def gap_pair_count(self) -> int:
        return len(self.gap_pairs)


@dataclass
class FeatureEngineeringReport:
    """Summary of target creation and rows excluded for missing labels."""

    input_rows: int
    output_rows: int = 0
    positive_targets: int = 0
    negative_targets: int = 0
    final_observations_without_label: int = 0
    gap_predecessors_without_label: int = 0
    continuity_gap_pairs: int = 0
    missing_calendar_days: int = 0
    baseline_unknown_rows_in_input: int = 0
    baseline_unknown_rows_in_output: int = 0
    selected_features: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary."""
        return asdict(self)


def validate_chronological_order(data: pd.DataFrame) -> None:
    """Require rows sorted by product, store, then timezone-aware timestamp."""
    missing = set(GROUP_COLUMNS + ["timestamp"]) - set(data.columns)
    if missing:
        raise ValueError(f"Missing ordering columns: {sorted(missing)}")
    timestamp = data["timestamp"]
    if not isinstance(timestamp.dtype, pd.DatetimeTZDtype):
        raise ValueError("timestamp must be parsed and timezone-aware before ordering checks")
    order = pd.MultiIndex.from_frame(data[GROUP_COLUMNS + ["timestamp"]])
    if not order.is_monotonic_increasing:
        raise ValueError("Input rows are not chronologically sorted by product and store")
    if data.duplicated(KEY_COLUMNS).any():
        raise ValueError("Duplicate product/store/timestamp observations are not allowed")


def validate_product_store_continuity(data: pd.DataFrame) -> ContinuityReport:
    """Report missing daily observations without treating them as negative labels."""
    validate_chronological_order(data)
    gaps: list[dict[str, Any]] = []
    for (product_id, store_id), group in data.groupby(GROUP_COLUMNS, sort=False):
        timestamps = group["timestamp"].reset_index(drop=True)
        for current, following in zip(timestamps.iloc[:-1], timestamps.iloc[1:]):
            elapsed = following - current
            if elapsed != NEXT_DAY:
                missing_days = max(int(elapsed / NEXT_DAY) - 1, 0)
                gaps.append(
                    {
                        "product_id": str(product_id),
                        "store_id": str(store_id),
                        "current_timestamp": current.isoformat(),
                        "next_timestamp": following.isoformat(),
                        "missing_calendar_days": missing_days,
                    }
                )
    return ContinuityReport(
        product_store_groups=int(data.groupby(GROUP_COLUMNS, sort=False).ngroups),
        gap_pairs=gaps,
        missing_calendar_days=sum(gap["missing_calendar_days"] for gap in gaps),
    )


def _prepare_source_data(data: pd.DataFrame) -> pd.DataFrame:
    """Parse, validate, and preserve the input's ordering before feature work."""
    validate_required_columns(data)
    parsed = data.copy(deep=True)
    parsed["timestamp"] = pd.to_datetime(parsed["timestamp"], errors="coerce", utc=True)
    if parsed["timestamp"].isna().any():
        raise ValueError("Input contains a missing or invalid timestamp")
    validate_chronological_order(parsed)

    # Reuse preprocessing's domain checks for status, restock history, and source labels.
    clean, _ = preprocess_inventory_data(parsed)
    validate_chronological_order(clean)
    if clean["restock_occurred"].isna().any():
        raise ValueError("restock_occurred cannot be missing when constructing targets")
    return clean


def load_processed_inventory(input_path: str | Path = "data/processed_inventory.csv") -> pd.DataFrame:
    """Load and validate the processed inventory dataset without modifying it."""
    return _prepare_source_data(load_inventory_csv(input_path))


def create_next_observation_target(data: pd.DataFrame) -> pd.Series:
    """Label restock on the next calendar-day observation for the same pair.

    A later observed row separated by a gap is not considered a next-day label.
    The returned nullable integer series leaves those rows unlabeled for explicit
    handling by :func:`engineer_inventory_data`.
    """
    validate_chronological_order(data)
    if data["restock_occurred"].isna().any():
        raise ValueError("restock_occurred cannot be missing when constructing targets")

    grouped = data.groupby(GROUP_COLUMNS, sort=False)
    next_timestamp = grouped["timestamp"].shift(-1)
    next_restock = grouped["restock_occurred"].shift(-1)
    has_next_daily_observation = next_timestamp.sub(data["timestamp"]).eq(NEXT_DAY)
    target = next_restock.astype("Int8").where(has_next_daily_observation, pd.NA)
    target.name = TARGET_COLUMN
    return target


def create_historical_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add safe calendar and restock-age features to a validated input copy."""
    validate_chronological_order(data)
    if data["restock_occurred"].isna().any():
        raise ValueError("restock_occurred history must be complete to validate recency")

    features = data.copy(deep=True)
    prior_observed_restocks = (
        features.groupby(GROUP_COLUMNS, sort=False)["restock_occurred"].cumsum().astype("int64")
        - features["restock_occurred"].astype("int64")
    )
    features["days_since_last_restock_valid"] = prior_observed_restocks.gt(0).astype("boolean")

    # The source value is assumed before the first observed restock. Mask it so
    # downstream models cannot mistake the synthetic baseline for verified history.
    features["days_since_last_restock_model"] = features[
        "days_since_last_restock"
    ].where(features["days_since_last_restock_valid"], pd.NA).astype("Int64")
    features["day_of_week"] = features["timestamp"].dt.day_of_week.astype("int8")
    features["month"] = features["timestamp"].dt.month.astype("int8")
    return features


def select_model_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return only the declared, prediction-time model inputs."""
    missing = set(MODEL_FEATURES) - set(data.columns)
    if missing:
        raise ValueError(f"Engineered model features are missing: {sorted(missing)}")
    forbidden = FORBIDDEN_MODEL_FEATURES.intersection(MODEL_FEATURES)
    if forbidden:
        raise ValueError(f"Leaking or trace-only columns selected as model features: {sorted(forbidden)}")
    return data.loc[:, MODEL_FEATURES].copy()


def validate_engineered_dataset(data: pd.DataFrame) -> None:
    """Check labels, feature selection, source labels, and baseline masking."""
    required_output = set(REQUIRED_COLUMNS) | set(MODEL_FEATURES) | {TARGET_COLUMN}
    missing = required_output - set(data.columns)
    if missing:
        raise ValueError(f"Engineered dataset is missing columns: {sorted(missing)}")
    validate_chronological_order(data)
    if data[TARGET_COLUMN].isna().any() or not data[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("Every engineered row must have a binary next-day target")
    if not data["data_source"].eq("synthetic").all():
        raise ValueError("Synthetic source labels must be preserved")
    if not pd.api.types.is_bool_dtype(data["days_since_last_restock_valid"].dtype):
        raise ValueError("days_since_last_restock_valid must be boolean")
    invalid_recency = ~data["days_since_last_restock_valid"]
    if data.loc[invalid_recency, "days_since_last_restock_model"].notna().any():
        raise ValueError("Unverified restock-age values must be masked from model features")
    select_model_features(data)


def engineer_inventory_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, FeatureEngineeringReport]:
    """Build a leakage-aware labeled dataset and report rows without next-day labels."""
    source = _prepare_source_data(data)
    continuity = validate_product_store_continuity(source)
    result = create_historical_features(source)
    result[TARGET_COLUMN] = create_next_observation_target(source)

    grouped = source.groupby(GROUP_COLUMNS, sort=False)
    next_timestamp = grouped["timestamp"].shift(-1)
    has_following_row = next_timestamp.notna()
    is_daily_pair = next_timestamp.sub(source["timestamp"]).eq(NEXT_DAY)
    final_mask = ~has_following_row
    gap_mask = has_following_row & ~is_daily_pair
    unlabeled_mask = result[TARGET_COLUMN].isna()

    report = FeatureEngineeringReport(
        input_rows=len(source),
        final_observations_without_label=int(final_mask.sum()),
        gap_predecessors_without_label=int(gap_mask.sum()),
        continuity_gap_pairs=continuity.gap_pair_count,
        missing_calendar_days=continuity.missing_calendar_days,
        baseline_unknown_rows_in_input=int((~result["days_since_last_restock_valid"]).sum()),
        selected_features=list(MODEL_FEATURES),
    )
    if not unlabeled_mask.equals(final_mask | gap_mask):
        raise ValueError("Unlabeled rows do not match final observations and daily gaps")

    result = result.loc[~unlabeled_mask].copy()
    result[TARGET_COLUMN] = result[TARGET_COLUMN].astype("int8")
    result = result.sort_values(KEY_COLUMNS, kind="mergesort").reset_index(drop=True)
    report.output_rows = len(result)
    report.positive_targets = int(result[TARGET_COLUMN].eq(1).sum())
    report.negative_targets = int(result[TARGET_COLUMN].eq(0).sum())
    report.baseline_unknown_rows_in_output = int(
        (~result["days_since_last_restock_valid"]).sum()
    )

    validate_engineered_dataset(result)
    if report.output_rows != report.positive_targets + report.negative_targets:
        raise ValueError("Target class counts do not match engineered row count")
    return result, report


def save_engineered_dataset(
    data: pd.DataFrame, output_path: str | Path = "data/engineered_inventory.csv"
) -> Path:
    """Validate and save engineered observations to a separate CSV file."""
    validate_engineered_dataset(data)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(destination, index=False, date_format="%Y-%m-%dT%H:%M:%S%z")
    return destination


def run_feature_engineering(
    input_path: str | Path = "data/processed_inventory.csv",
    output_path: str | Path = "data/engineered_inventory.csv",
) -> FeatureEngineeringReport:
    """Load, engineer, validate, and save the feature dataset."""
    source_path = Path(input_path)
    destination = Path(output_path)
    if source_path.resolve() == destination.resolve():
        raise ValueError("Output path must differ from the processed input dataset")
    source = load_processed_inventory(source_path)
    engineered, report = engineer_inventory_data(source)
    save_engineered_dataset(engineered, destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed_inventory.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/engineered_inventory.csv"))
    args = parser.parse_args()
    report = run_feature_engineering(args.input, args.output)
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
