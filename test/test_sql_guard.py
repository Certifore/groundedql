"""
Unit tests for intentql.sql_guard (requires sqlglot: pip install -e '.[guided]').
"""
from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SCHEMA_PATH = os.path.join(ROOT, "config", "schema.yaml")


@unittest.skipUnless(os.path.isfile(SCHEMA_PATH), "config/schema.yaml not present")
class TestSqlGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import sqlglot  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("sqlglot not installed; pip install -e '.[guided]'")

        from intentql.schema_catalog import load_schema_catalog

        cls.catalog = load_schema_catalog(SCHEMA_PATH)

    def test_allows_select_physical_table(self):
        from intentql.sql_guard import validate_sql

        sql = "SELECT COUNT(*) AS n FROM customers"
        r = validate_sql(sql, self.catalog)
        self.assertTrue(r.ok, r.message)

    def test_rejects_unknown_table(self):
        from intentql.sql_guard import validate_sql

        sql = "SELECT * FROM nowhere_table"
        r = validate_sql(sql, self.catalog)
        self.assertFalse(r.ok)

    def test_rejects_insert(self):
        from intentql.sql_guard import validate_sql

        sql = "INSERT INTO customers (customer_id) VALUES ('x')"
        r = validate_sql(sql, self.catalog)
        self.assertFalse(r.ok)

    def test_apply_row_limit(self):
        from intentql.sql_guard import apply_row_limit

        sql = "SELECT 1 AS x"
        out = apply_row_limit(sql, 100)
        self.assertIn("LIMIT 100", out)


if __name__ == "__main__":
    unittest.main()
