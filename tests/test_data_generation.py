import unittest

from src.data_generation import generate_inventory_data, validate_inventory_data


class InventoryGenerationTests(unittest.TestCase):
    def test_generation_is_reproducible_and_large_enough(self):
        first = generate_inventory_data(seed=7)
        second = generate_inventory_data(seed=7)
        self.assertEqual(first, second)
        self.assertGreaterEqual(len(first), 5000)
        validate_inventory_data(first)

    def test_history_features_exclude_current_restock(self):
        rows = generate_inventory_data(seed=42)
        pair = [r for r in rows if r["product_id"] == "P001" and r["store_id"] == "S001"]
        for index, row in enumerate(pair):
            earlier_restock_count = sum(r["restock_occurred"] for r in pair[:index])
            self.assertEqual(row["historical_restock_count"], earlier_restock_count)
            if row["restock_occurred"]:
                self.assertEqual(row["historical_restock_count"], earlier_restock_count)

    def test_validator_rejects_duplicate_observations(self):
        rows = generate_inventory_data(seed=3)
        rows.append(rows[0].copy())
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_inventory_data(rows)


if __name__ == "__main__":
    unittest.main()
