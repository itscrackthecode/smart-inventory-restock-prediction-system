# SQLite database

This module stores operational product, store, daily inventory, and model prediction records. It uses Python's built-in `sqlite3`; no database package is required.

## Location and setup

By default the file is `data/inventory.db` (created only when initialization or import is run). Set `INVENTORY_DB_PATH` to use another location, or pass an explicit path to the Python functions and CLI commands. Parent directories are created as needed.

From the project root, using the project's virtual environment:

```powershell
.\.venv\Scripts\python.exe -m src.database init
.\.venv\Scripts\python.exe -m src.database seed
```

Both commands accept `--database PATH`; `seed` also accepts `--csv PATH`. Initialization is repeatable and does not drop data. The importer reads `data/processed_inventory.csv`, stores timestamps in UTC ISO-8601 format, and only imports operational columns. It does not store feature-engineering history fields or model targets. Identical observations are counted as duplicates; invalid rows and conflicting existing observations are reported without overwriting the stored record. The import runs in one transaction and uses savepoints when a row also introduces new product or store records.

## Schema and relationships

- `products(product_id PK, product_name, category)` stores product metadata.
- `stores(store_id PK, store_location)` stores store metadata.
- `inventory_records(inventory_id PK, product_id FK, store_id FK, observation_date, stock_quantity, availability_status, previous_stock_status, restock_occurred, data_source)` stores daily operational observations. `(product_id, store_id, observation_date)` is unique.
- `predictions(prediction_id PK, inventory_id FK, model_name, model_version, predicted_probability, prediction_threshold, predicted_restock, created_at)` records the pipeline/version and threshold used for each prediction.

Foreign keys are enabled on every connection. Database checks enforce nonnegative whole-number stock, stock/status consistency, valid booleans, and probabilities and thresholds in `[0, 1]`. Foreign-key delete restrictions prevent removing products, stores, or inventory rows still referenced by observations or predictions.

## CRUD examples

```python
from src.database import initialize_database
from src.database.repositories import (
    create_product, create_store, insert_inventory_record,
    list_inventory_records, save_prediction,
)

initialize_database()
create_product("P100", "Coffee 500g", "Grocery")
create_store("S100", "Austin, TX")
record = insert_inventory_record(
    product_id="P100", store_id="S100",
    observation_date="2026-11-17T09:00:00-06:00", stock_quantity=12,
    availability_status="in_stock", previous_stock_status="in_stock",
    restock_occurred=False, data_source="synthetic",
)
save_prediction(
    inventory_id=record["inventory_id"], model_name="Logistic Regression",
    model_version="phase6", predicted_probability=0.72,
    prediction_threshold=0.75, predicted_restock=False,
)
records = list_inventory_records(product_id="P100", store_id="S100")
```

Repository operations accept an optional `database_path=` argument. Inventory list queries accept product/store filters, timezone-aware `start_date` and `end_date`, and a positive `limit`. Prediction queries include `list_predictions`, `prediction_history(product_id=..., store_id=...)`, and `latest_predictions`.

## Example SQL

```sql
-- Latest stock observation per product/store
SELECT i.product_id, i.store_id, i.observation_date, i.stock_quantity,
       i.availability_status
FROM inventory_records AS i
WHERE i.observation_date = (
    SELECT MAX(j.observation_date)
    FROM inventory_records AS j
    WHERE j.product_id = i.product_id AND j.store_id = i.store_id
);

-- Prediction history for one product/store
SELECT p.model_name, p.model_version, p.predicted_probability,
       p.prediction_threshold, p.predicted_restock, p.created_at
FROM predictions AS p
JOIN inventory_records AS i USING (inventory_id)
WHERE i.product_id = 'P100' AND i.store_id = 'S100'
ORDER BY p.created_at;
```

## Tests

Tests use temporary file-backed SQLite databases and do not create or modify `data/inventory.db`:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_database -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
