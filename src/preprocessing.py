"""Load, validate, and clean the synthetic inventory CSV.

Preprocessing never reconstructs missing historical features. It only derives
availability_status when that current-state value is missing and stock quantity
is present, because those two fields are deterministically equivalent here.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {
    "product_id",
    "product_name",
    "category",
    "store_id",
    "store_location",
    "timestamp",
    "stock_quantity",
    "availability_status",
    "previous_stock_status",
    "days_since_last_restock",
    "historical_restock_count",
    "restock_occurred",
    "data_source",
}
KEY_COLUMNS = ["product_id", "store_id", "timestamp"]
STRING_COLUMNS = [
    "product_id", "product_name", "category", "store_id", "store_location",
    "availability_status", "previous_stock_status", "data_source",
]
HISTORICAL_COLUMNS = [
    "previous_stock_status", "days_since_last_restock", "historical_restock_count",
]
VALID_STATUSES = {"in_stock", "out_of_stock"}
FIRST_OBSERVATION_NOTE = (
    "The generator initializes days_since_last_restock to 1 on the first row "
    "and increments it until an observed restock. No pre-window restock date is "
    "recorded, so those initial values represent an assumed baseline, not a "
    "known elapsed time since an actual restock. Preserve them for now; a future "
    "data-version change should record a pre-window restock timestamp or mark "
    "this feature unknown until the first observed restock."
)


@dataclass
class PreprocessingReport:
    """Counts and notes produced while preprocessing one dataset."""

    input_rows: int
    output_rows: int = 0
    missing_before: dict[str, int] = field(default_factory=dict)
    missing_after: dict[str, int] = field(default_factory=dict)
    duplicate_observation_rows: int = 0
    exact_duplicate_rows_removed: int = 0
    status_values_derived: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary."""
        return asdict(self)


def load_inventory_csv(input_path: str | Path) -> pd.DataFrame:
    """Load the CSV without changing its source file or parsing timestamps yet."""
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Inventory CSV not found: {path}")
    return pd.read_csv(path, keep_default_na=True)


def validate_required_columns(data: pd.DataFrame) -> None:
    """Check that the dataset contains every field required by this project."""
    missing_columns = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")


def _normalize_values(data: pd.DataFrame, report: PreprocessingReport) -> pd.DataFrame:
    """Normalize CSV representations and use nullable types for history fields."""
    clean = data.copy(deep=True)
    clean.columns = clean.columns.str.strip()
    validate_required_columns(clean)

    # Treat whitespace-only text as missing without filling unknown information.
    for column in clean.select_dtypes(include=["object", "string"]).columns:
        clean[column] = clean[column].map(
            lambda value: pd.NA if isinstance(value, str) and not value.strip() else value
        )
    report.missing_before = clean.isna().sum().astype(int).to_dict()

    for column in STRING_COLUMNS:
        clean[column] = clean[column].astype("string").str.strip()
    clean["availability_status"] = clean["availability_status"].str.lower()
    clean["previous_stock_status"] = clean["previous_stock_status"].str.lower()
    clean["data_source"] = clean["data_source"].str.lower()

    clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="coerce", utc=True)
    if clean["timestamp"].isna().any():
        bad_count = int(clean["timestamp"].isna().sum())
        raise ValueError(f"Found {bad_count} missing or invalid timestamps")

    for column in ("stock_quantity", "days_since_last_restock", "historical_restock_count"):
        original = clean[column]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & numeric.isna()
        if invalid.any():
            raise ValueError(f"Column {column!r} contains non-numeric values")
        fractional = numeric.notna() & (numeric % 1 != 0)
        if fractional.any():
            raise ValueError(f"Column {column!r} must contain whole numbers")
        clean[column] = numeric.astype("Int64")

    restock_values = clean["restock_occurred"].map(
        {True: True, False: False, 1: True, 0: False, "True": True, "False": False,
         "true": True, "false": False, "1": True, "0": False}
    )
    invalid_restock = clean["restock_occurred"].notna() & restock_values.isna()
    if invalid_restock.any():
        raise ValueError("restock_occurred must contain true/false values")
    clean["restock_occurred"] = restock_values.astype("boolean")

    # Only this present-time status is safely derivable from current stock.
    missing_status = clean["availability_status"].isna()
    report.status_values_derived = int(missing_status.sum())
    clean.loc[missing_status, "availability_status"] = clean.loc[
        missing_status, "stock_quantity"
    ].map(lambda quantity: "in_stock" if pd.notna(quantity) and quantity > 0 else "out_of_stock")

    # Rows without identifiers, source labels, or current quantity cannot be
    # reliably interpreted; fail explicitly instead of fabricating values.
    essential = ["product_id", "store_id", "timestamp", "stock_quantity", "data_source"]
    missing_essential = clean[essential].isna().sum()
    missing_essential = missing_essential[missing_essential > 0]
    if not missing_essential.empty:
        details = ", ".join(f"{name}={count}" for name, count in missing_essential.items())
        raise ValueError(f"Missing essential values cannot be inferred: {details}")

    if not clean["data_source"].eq("synthetic").all():
        raise ValueError("Every row must retain data_source='synthetic'")

    return clean


def _remove_only_exact_duplicates(
    data: pd.DataFrame, report: PreprocessingReport
) -> pd.DataFrame:
    """Drop exact repeated rows, report them, and reject conflicting key repeats."""
    clean = data.copy()
    report.duplicate_observation_rows = int(clean.duplicated(KEY_COLUMNS).sum())

    exact_mask = clean.duplicated(keep="first")
    report.exact_duplicate_rows_removed = int(exact_mask.sum())
    clean = clean.loc[~exact_mask].copy()
    conflicts = clean.duplicated(KEY_COLUMNS, keep=False)
    if conflicts.any():
        examples = clean.loc[conflicts, KEY_COLUMNS].head(3).to_dict("records")
        raise ValueError(
            "Conflicting observations share a product/store/timestamp key; "
            f"no rows were selected or removed. Examples: {examples}"
        )

    return clean


def _validate_inventory_values(data: pd.DataFrame) -> None:
    """Validate stock/status and historical fields without filling unknowns."""
    if (data["stock_quantity"].dropna() < 0).any():
        raise ValueError("stock_quantity cannot be negative")
    if data["stock_quantity"].isna().any():
        raise ValueError("stock_quantity cannot be missing")
    if not data["availability_status"].dropna().isin(VALID_STATUSES).all():
        raise ValueError("availability_status must be in_stock or out_of_stock")
    expected_status = data["stock_quantity"].gt(0).map(
        {True: "in_stock", False: "out_of_stock"}
    )
    mismatched_status = data["availability_status"].notna() & data[
        "availability_status"
    ].ne(expected_status)
    if mismatched_status.any():
        raise ValueError("availability_status conflicts with stock_quantity")

    if not data["previous_stock_status"].dropna().isin(
        VALID_STATUSES | {"unknown"}
    ).all():
        raise ValueError("previous_stock_status contains an unsupported value")
    for column in ("days_since_last_restock", "historical_restock_count"):
        values = data[column].dropna()
        if (values < 0).any():
            raise ValueError(f"{column} cannot be negative")
    if (data["days_since_last_restock"].dropna() < 1).any():
        raise ValueError("days_since_last_restock must be at least 1")

    # Compare each feature only with information from earlier rows in the same
    # product/store history. A missing past event indicator makes later history
    # unknowable, so checks stop instead of guessing.
    for _, group in data.groupby(["product_id", "store_id"], sort=False, dropna=False):
        group = group.sort_values("timestamp", kind="mergesort")
        prior_status: str | None = None
        prior_restock_count = 0
        count_known = True
        first_timestamp: pd.Timestamp | None = None
        prior_timestamp: pd.Timestamp | None = None
        last_restock_timestamp: pd.Timestamp | None = None
        recency_known = True

        for row in group.itertuples(index=False):
            observed_at = row.timestamp
            if first_timestamp is None:
                first_timestamp = observed_at
            previous_value = row.previous_stock_status
            if pd.notna(previous_value):
                # Only compare against the prior row when it is truly the
                # previous daily observation. A gap means that day's status is
                # missing, so the last available row is not a safe substitute.
                can_validate_previous = (
                    prior_timestamp is None or observed_at - prior_timestamp == pd.Timedelta(days=1)
                )
                expected_previous = prior_status if prior_status is not None else "unknown"
                if can_validate_previous and previous_value != expected_previous:
                    raise ValueError(
                        "previous_stock_status does not match the prior observation "
                        f"for {row.product_id}/{row.store_id} at {row.timestamp}"
                    )

            count_value = row.historical_restock_count
            if count_known and pd.notna(count_value) and count_value != prior_restock_count:
                raise ValueError(
                    "historical_restock_count includes an incorrect or current/future "
                    f"restock for {row.product_id}/{row.store_id} at {row.timestamp}"
                )

            days_value = row.days_since_last_restock
            if recency_known and pd.notna(days_value):
                expected_days = (
                    max(1, (observed_at - last_restock_timestamp).days)
                    if last_restock_timestamp is not None
                    else (observed_at - first_timestamp).days + 1
                )
                if days_value != expected_days:
                    raise ValueError(
                        "days_since_last_restock does not match prior restock history "
                        f"for {row.product_id}/{row.store_id} at {row.timestamp}"
                    )

            prior_status = row.availability_status
            prior_timestamp = observed_at
            if pd.isna(row.restock_occurred):
                count_known = False
                recency_known = False
            elif row.restock_occurred:
                prior_restock_count += 1
                last_restock_timestamp = observed_at


def preprocess_inventory_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, PreprocessingReport]:
    """Validate and return a cleaned, chronological copy plus its report."""
    report = PreprocessingReport(input_rows=len(data))
    clean = _normalize_values(data, report)
    clean = _remove_only_exact_duplicates(clean, report)
    clean = clean.sort_values(KEY_COLUMNS, kind="mergesort").reset_index(drop=True)
    _validate_inventory_values(clean)

    report.output_rows = len(clean)
    report.missing_after = clean.isna().sum().astype(int).to_dict()
    report.warnings.append(FIRST_OBSERVATION_NOTE)
    return clean, report


def preprocess_file(
    input_path: str | Path = "data/synthetic_inventory.csv",
    output_path: str | Path = "data/processed_inventory.csv",
) -> PreprocessingReport:
    """Load, preprocess, and save to a separate CSV; never modify the source."""
    source = Path(input_path)
    destination = Path(output_path)
    if source.resolve() == destination.resolve():
        raise ValueError("Output path must differ from the original synthetic CSV")

    raw = load_inventory_csv(source)
    cleaned, report = preprocess_inventory_data(raw)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(destination, index=False, date_format="%Y-%m-%dT%H:%M:%S%z")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/synthetic_inventory.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed_inventory.csv"))
    args = parser.parse_args()
    report = preprocess_file(args.input, args.output)
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
