"""Command-line helpers for initializing and seeding the local database."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.database.connection import DEFAULT_DATABASE_PATH
from src.database.schema import initialize_database
from src.database.seed import DEFAULT_SOURCE_PATH, import_processed_inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init", help="create tables and indexes safely")
    init_parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    seed_parser = subparsers.add_parser("seed", help="import processed inventory CSV")
    seed_parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    seed_parser.add_argument("--csv", type=Path, default=DEFAULT_SOURCE_PATH)
    args = parser.parse_args()
    if args.command == "init":
        initialize_database(args.database)
        print(f"Initialized SQLite database at {args.database}")
    else:
        print(json.dumps(import_processed_inventory(args.csv, args.database).to_dict(), indent=2))


if __name__ == "__main__":
    main()
