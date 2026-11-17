"""Prepare an application database on first start without replacing existing data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import database_path as configured_database_path
from src.database.connection import database_connection
from src.database.schema import initialize_database
from src.database.seed import DEFAULT_SOURCE_PATH, import_processed_inventory


@dataclass(frozen=True)
class StartupReport:
    """Summary of safe schema initialization and optional first seed."""

    database_path: Path
    seeded_rows: int
    total_inventory_rows: int


def ensure_database_ready(
    database_file: str | Path | None = None,
    source_csv: str | Path = DEFAULT_SOURCE_PATH,
) -> StartupReport:
    """Create schema and seed only an empty inventory database.

    The database is never cleared or replaced. Existing inventory observations
    are retained, and importing into an empty database uses the project's
    idempotent processed-CSV importer.
    """
    path = Path(database_file) if database_file is not None else configured_database_path()
    path = path.expanduser().resolve()
    initialize_database(path)

    with database_connection(path) as connection:
        inventory_rows = int(connection.execute(
            "SELECT COUNT(*) FROM inventory_records"
        ).fetchone()[0])

    seeded_rows = 0
    if inventory_rows == 0:
        report = import_processed_inventory(source_csv, path)
        if report.invalid_rows:
            raise ValueError(
                "Database seed found invalid rows in the processed inventory CSV: "
                f"{report.invalid_rows} invalid of {report.input_rows}."
            )
        seeded_rows = report.inserted_inventory_rows
        inventory_rows = report.final_inventory_rows

    return StartupReport(
        database_path=path,
        seeded_rows=seeded_rows,
        total_inventory_rows=inventory_rows,
    )
