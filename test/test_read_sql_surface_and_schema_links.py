"""
Tests: read_sql_surface contract + schema link.on column validation.

Run from repo root:
  python test/test_read_sql_surface_and_schema_links.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from groundedql.read_sql_surface import (
    READ_SQL_SURFACE_VERSION,
    read_sql_surface_capabilities,
    read_sql_surface_summary_for_spec,
)
from groundedql.schema_validator import validate_schema
from groundedql.exceptions import SchemaError


def test_read_sql_surface_version_stable() -> None:
    caps = read_sql_surface_capabilities()
    assert caps["version"] == READ_SQL_SURFACE_VERSION
    assert "postgresql" in caps["dialect"]
    summary = read_sql_surface_summary_for_spec()
    assert f"READ_SQL_SURFACE_VERSION={READ_SQL_SURFACE_VERSION}" in summary


def test_validate_schema_link_columns_good() -> None:
    schema = {
        "tables": [
            {
                "name": "orders",
                "db_table": "orders",
                "columns": [
                    {"name": "order_id", "db_column": "order_id", "type": "int"},
                    {"name": "customer_id", "db_column": "customer_id", "type": "int"},
                ],
            },
            {
                "name": "customers",
                "db_table": "customers",
                "columns": [
                    {"name": "customer_id", "db_column": "customer_id", "type": "int"},
                ],
            },
        ],
        "links": [
            {
                "name": "orders_to_customers",
                "from_table": "orders",
                "to_table": "customers",
                "join_type": "left",
                "on": [{"left": "orders.customer_id", "right": "customers.customer_id"}],
            }
        ],
    }
    validate_schema(schema)


def test_validate_schema_link_bad_column() -> None:
    schema = {
        "tables": [
            {
                "name": "orders",
                "db_table": "orders",
                "columns": [
                    {"name": "order_id", "db_column": "order_id", "type": "int"},
                    {"name": "customer_id", "db_column": "customer_id", "type": "int"},
                ],
            },
            {
                "name": "customers",
                "db_table": "customers",
                "columns": [
                    {"name": "customer_id", "db_column": "customer_id", "type": "int"},
                ],
            },
        ],
        "links": [
            {
                "name": "orders_to_customers",
                "from_table": "orders",
                "to_table": "customers",
                "join_type": "left",
                "on": [{"left": "orders.typo_id", "right": "customers.customer_id"}],
            }
        ],
    }
    try:
        validate_schema(schema)
    except SchemaError as e:
        assert "typo_id" in str(e) or "not found" in str(e).lower()
    else:
        raise AssertionError("expected SchemaError")


def main() -> None:
    test_read_sql_surface_version_stable()
    test_validate_schema_link_columns_good()
    test_validate_schema_link_bad_column()
    print("ok: read_sql_surface + schema link validation")


if __name__ == "__main__":
    main()
