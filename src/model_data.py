"""Load engineered data and create a leakage-safe chronological holdout."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.feature_engineering import TARGET_COLUMN, validate_engineered_dataset


DEFAULT_DATA_PATH = Path("data/engineered_inventory.csv")
TEST_SIZE = 0.20
NEXT_LABEL_HORIZON = pd.Timedelta(days=1)


@dataclass
class ChronologicalSplit:
    """Training rows, untouched holdout rows, and purged boundary rows."""

    train: pd.DataFrame
    holdout: pd.DataFrame
    purged: pd.DataFrame
    holdout_start: pd.Timestamp
    train_feature_start: pd.Timestamp
    train_feature_end: pd.Timestamp


def load_engineered_dataset(input_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read the Phase 5 CSV and validate its saved schema and labels."""
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Engineered inventory dataset not found: {path}")
    data = pd.read_csv(path)
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise", utc=True)
    validate_engineered_dataset(data)
    return data


def chronological_train_test_split(
    data: pd.DataFrame, *, test_size: float = TEST_SIZE
) -> ChronologicalSplit:
    """Hold out latest dates and purge training labels crossing the boundary.

    The label on feature date ``t`` describes the restock on ``t + 1 day``.
    Training rows are kept only if their label date is strictly before the first
    holdout feature date. Rows on the final pre-holdout day are purged so their
    labels cannot include an event in the holdout period.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    validate_engineered_dataset(data)

    unique_dates = pd.DatetimeIndex(data["timestamp"].drop_duplicates().sort_values())
    if len(unique_dates) < 3:
        raise ValueError("At least three unique observation dates are needed for a safe split")
    holdout_days = max(1, math.ceil(len(unique_dates) * test_size))
    if holdout_days >= len(unique_dates) - 1:
        raise ValueError("test_size leaves too few chronological dates for training")
    holdout_start = unique_dates[-holdout_days]

    feature_dates = data["timestamp"]
    label_dates = feature_dates + NEXT_LABEL_HORIZON
    train_mask = (feature_dates < holdout_start) & (label_dates < holdout_start)
    holdout_mask = feature_dates >= holdout_start
    purged_mask = (feature_dates < holdout_start) & ~train_mask

    sort_columns = ["timestamp", "product_id", "store_id"]
    train = data.loc[train_mask].sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    holdout = data.loc[holdout_mask].sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    purged = data.loc[purged_mask].sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    if train.empty or holdout.empty:
        raise ValueError("Chronological split produced an empty training or holdout partition")
    if not (train["timestamp"] + NEXT_LABEL_HORIZON < holdout_start).all():
        raise ValueError("A training label reaches the holdout boundary")
    if not (holdout["timestamp"] >= holdout_start).all():
        raise ValueError("Holdout contains a feature date before its boundary")
    return ChronologicalSplit(
        train=train,
        holdout=holdout,
        purged=purged,
        holdout_start=holdout_start,
        train_feature_start=train["timestamp"].min(),
        train_feature_end=train["timestamp"].max(),
    )


def target_distribution(data: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """Summarize binary labels for reporting without fitting or scoring a model."""
    counts = data[TARGET_COLUMN].value_counts().reindex([0, 1], fill_value=0)
    return {
        str(label): {
            "count": int(count),
            "percentage": round(float(100 * count / len(data)), 2) if len(data) else 0.0,
        }
        for label, count in counts.items()
    }
