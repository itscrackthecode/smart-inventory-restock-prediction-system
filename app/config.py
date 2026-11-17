"""Portable project paths and application configuration."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def database_path() -> Path:
    """Resolve the project-relative database path with an optional override."""
    configured = os.environ.get("INVENTORY_DB_PATH")
    if not configured:
        return DATA_DIR / "inventory.db"
    path = Path(configured).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
