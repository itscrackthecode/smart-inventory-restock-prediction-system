"""SQLite connection lifecycle and transaction management."""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "inventory.db"


def resolve_database_path(database_path: str | Path | None = None) -> str:
    """Resolve an explicit path, environment override, or project default."""
    configured = database_path or os.environ.get("INVENTORY_DB_PATH") or DEFAULT_DATABASE_PATH
    if str(configured) == ":memory:":
        return ":memory:"
    return str(Path(configured).expanduser().resolve())


@contextmanager
def database_connection(
    database_path: str | Path | None = None,
) -> Iterator[sqlite3.Connection]:
    """Open a connection, enforce foreign keys, and commit/rollback atomically.

    The connection is always closed on exit. Use a file-backed database for
    operations that open more than one connection; `:memory:` is scoped to one
    connection.
    """
    resolved = resolve_database_path(database_path)
    if resolved != ":memory:":
        Path(resolved).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(resolved, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        with conn:
            yield conn
    finally:
        conn.close()
