"""Fetch and prepare UCI Online Retail II for descriptive sales analysis.

This is transaction data, not inventory-on-hand or replenishment data. The
module keeps the source workbook and derived daily sales file separate from the
project's synthetic inventory datasets.
"""

from __future__ import annotations

import json
import os
import tempfile
import urllib.request
import zipfile
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "external"
DATASET_URL = "https://archive.ics.uci.edu/static/public/502/online%2Bretail%2Bii.zip"
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/502/online+retail+ii"
WORKBOOK_NAME = "online_retail_II.xlsx"
DAILY_SALES_NAME = "uci_online_retail_daily.csv"
METADATA_NAME = "uci_online_retail_metadata.json"
WORKSHEETS = ("Year 2009-2010", "Year 2010-2011")
REQUIRED_COLUMNS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Country"]


def external_data_paths(data_dir: str | Path = DEFAULT_DATA_DIR) -> dict[str, Path]:
    """Return paths for the downloaded workbook and derived artifacts."""
    root = Path(data_dir)
    return {
        "directory": root,
        "workbook": root / WORKBOOK_NAME,
        "daily_sales": root / DAILY_SALES_NAME,
        "metadata": root / METADATA_NAME,
    }


def _download_workbook(data_dir: Path) -> Path:
    """Download the official UCI archive and safely extract its workbook."""
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_fd, archive_name = tempfile.mkstemp(prefix="uci-retail-", suffix=".zip", dir=data_dir)
    os.close(archive_fd)
    archive_path = Path(archive_name)
    try:
        request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "StockwiseInventoryResearch/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response, archive_path.open("wb") as destination:
            while chunk := response.read(1024 * 1024):
                destination.write(chunk)
        with zipfile.ZipFile(archive_path) as archive:
            bad_member = archive.testzip()
            if bad_member:
                raise ValueError(f"Downloaded UCI archive failed its integrity check: {bad_member}")
            member = next((name for name in archive.namelist() if Path(name).name == WORKBOOK_NAME), None)
            if member is None:
                raise ValueError(f"UCI archive does not contain {WORKBOOK_NAME}")
            destination = data_dir / WORKBOOK_NAME
            temporary_workbook = destination.with_suffix(".xlsx.tmp")
            with archive.open(member) as source, temporary_workbook.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
            temporary_workbook.replace(destination)
        return data_dir / WORKBOOK_NAME
    finally:
        archive_path.unlink(missing_ok=True)


def _prepare_sheet(workbook: pd.ExcelFile, sheet_name: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean one source sheet and aggregate qualifying positive sale lines."""
    raw = pd.read_excel(
        workbook,
        sheet_name=sheet_name,
        usecols=REQUIRED_COLUMNS,
        dtype={"Invoice": "string", "StockCode": "string", "Country": "string"},
    )
    input_rows = len(raw)
    invoice = raw["Invoice"].astype("string").str.strip()
    cancelled = invoice.str.upper().str.startswith("C", na=False)
    dates = pd.to_datetime(raw["InvoiceDate"], errors="coerce")
    quantity = pd.to_numeric(raw["Quantity"], errors="coerce")
    price = pd.to_numeric(raw["Price"], errors="coerce")
    valid_date = dates.notna()
    valid_numbers = quantity.notna() & price.notna() & quantity.gt(0) & price.gt(0)
    valid_product = raw["StockCode"].notna() & raw["StockCode"].astype("string").str.strip().ne("")
    valid_market = raw["Country"].notna() & raw["Country"].astype("string").str.strip().ne("")
    included = ~cancelled & valid_date & valid_numbers & valid_product & valid_market

    kept = pd.DataFrame({
        "date": dates.loc[included].dt.date.astype("string"),
        "stock_code": raw.loc[included, "StockCode"].astype("string").str.strip(),
        "description": raw.loc[included, "Description"].astype("string").str.strip(),
        "country": raw.loc[included, "Country"].astype("string").str.strip(),
        "units_sold": quantity.loc[included].astype("int64"),
        "sales_value_gbp": (quantity.loc[included] * price.loc[included]).astype("float64"),
        "invoice": invoice.loc[included],
    })
    daily = kept.groupby(["date", "stock_code", "country"], as_index=False, dropna=False).agg(
        description=("description", "first"),
        units_sold=("units_sold", "sum"),
        sales_value_gbp=("sales_value_gbp", "sum"),
        invoice_count=("invoice", "nunique"),
        transaction_lines=("invoice", "size"),
    )
    stats = {
        "input_rows": input_rows,
        "cancelled_invoice_lines": int(cancelled.sum()),
        "invalid_non_cancelled_lines": int((~cancelled & (~valid_date | ~valid_numbers | ~valid_product | ~valid_market)).sum()),
        "included_sale_lines": int(included.sum()),
    }
    return daily, stats


def build_daily_sales(
    workbook_path: str | Path,
    *,
    data_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a daily product/market sales dataset from the original workbook."""
    workbook_path = Path(workbook_path)
    if not workbook_path.is_file():
        raise FileNotFoundError(f"UCI workbook not found: {workbook_path}")
    output_dir = Path(data_dir) if data_dir is not None else workbook_path.parent
    sheet_frames: list[pd.DataFrame] = []
    sheet_stats: dict[str, dict[str, int]] = {}
    with pd.ExcelFile(workbook_path, engine="openpyxl") as workbook:
        missing = set(WORKSHEETS) - set(workbook.sheet_names)
        if missing:
            raise ValueError(f"UCI workbook is missing expected sheets: {sorted(missing)}")
        for sheet in WORKSHEETS:
            daily, stats = _prepare_sheet(workbook, sheet)
            sheet_frames.append(daily)
            sheet_stats[sheet] = stats

    combined = pd.concat(sheet_frames, ignore_index=True)
    # Re-aggregate at the worksheet boundary in case the same daily key appears in both.
    combined = combined.groupby(["date", "stock_code", "country"], as_index=False, dropna=False).agg(
        description=("description", "first"),
        units_sold=("units_sold", "sum"),
        sales_value_gbp=("sales_value_gbp", "sum"),
        invoice_count=("invoice_count", "sum"),
        transaction_lines=("transaction_lines", "sum"),
    ).sort_values(["date", "stock_code", "country"], kind="stable").reset_index(drop=True)
    combined["sales_value_gbp"] = combined["sales_value_gbp"].round(2)

    metadata: dict[str, Any] = {
        "source_name": "UCI Online Retail II",
        "source_url": DATASET_PAGE,
        "download_url": DATASET_URL,
        "citation": "Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D.",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "source_period": "2009-12-01 through 2011-12-09",
        "source_rows": sum(item["input_rows"] for item in sheet_stats.values()),
        "cancelled_invoice_lines_excluded": sum(item["cancelled_invoice_lines"] for item in sheet_stats.values()),
        "invalid_non_cancelled_lines_excluded": sum(item["invalid_non_cancelled_lines"] for item in sheet_stats.values()),
        "included_sale_lines": sum(item["included_sale_lines"] for item in sheet_stats.values()),
        "daily_product_market_rows": int(len(combined)),
        "first_sale_date": str(combined["date"].min()) if not combined.empty else None,
        "last_sale_date": str(combined["date"].max()) if not combined.empty else None,
        "worksheets": sheet_stats,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_scope": "Positive retail transactions aggregated by calendar date, product stock code, and customer country; not inventory-on-hand or restock records.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = external_data_paths(output_dir)
    temp_csv = paths["daily_sales"].with_suffix(".csv.tmp")
    combined.to_csv(temp_csv, index=False)
    temp_csv.replace(paths["daily_sales"])
    temp_metadata = paths["metadata"].with_suffix(".json.tmp")
    temp_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temp_metadata.replace(paths["metadata"])
    return combined, metadata


def load_or_fetch_daily_sales(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    refresh: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load cached UCI sales data, fetching and preparing it when needed."""
    paths = external_data_paths(data_dir)
    if refresh:
        paths["daily_sales"].unlink(missing_ok=True)
        paths["metadata"].unlink(missing_ok=True)
    if paths["daily_sales"].is_file() and paths["metadata"].is_file():
        frame = pd.read_csv(paths["daily_sales"], dtype={"stock_code": "string", "country": "string"})
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        return frame, metadata
    workbook = paths["workbook"]
    if not workbook.is_file():
        workbook = _download_workbook(paths["directory"])
    return build_daily_sales(workbook, data_dir=paths["directory"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Rebuild the derived summary from the local source workbook")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory for downloaded and prepared data")
    args = parser.parse_args()
    frame, metadata = load_or_fetch_daily_sales(args.data_dir, refresh=args.refresh)
    print(f"Prepared {len(frame):,} daily product-market summaries")
    print(f"Included {metadata['included_sale_lines']:,} of {metadata['source_rows']:,} source transaction lines")
    print(f"Saved to {external_data_paths(args.data_dir)['daily_sales']}")


if __name__ == "__main__":
    main()
