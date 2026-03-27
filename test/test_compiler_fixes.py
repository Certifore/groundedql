"""Regression tests for compiler hardening (joins, func, set_op, IN, schema links)."""

from __future__ import annotations

import pytest

from dsl_compiler.compiler import Compiler, QueryPlanError
from dsl_compiler.exceptions import SchemaError
from dsl_compiler.schema_validator import validate_schema


def _minimal_schema(*, bad_link_join_type: str | None = None) -> dict:
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
                    {"name": "name", "db_column": "name", "type": "varchar"},
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
    if bad_link_join_type is not None:
        schema["links"][0]["join_type"] = bad_link_join_type
    return schema


def test_schema_validator_rejects_bad_link_join_type():
    s = _minimal_schema(bad_link_join_type="right")
    with pytest.raises(SchemaError, match="join_type"):
        validate_schema(s)


def test_compiler_rejects_bad_link_join_type_without_prior_validation():
    s = _minimal_schema(bad_link_join_type="full")
    with pytest.raises(SchemaError, match="join_type"):
        Compiler(s)


def test_duplicate_explicit_join_requires_as():
    schema = _minimal_schema()
    c = Compiler(schema)
    plan = {
        "dataset": "orders",
        "joins": [
            {
                "dataset": "customers",
                "type": "inner",
                "on": {"cmp": {"left": {"col": "orders.customer_id"}, "op": "=", "right": {"col": "customers.customer_id"}}},
            },
            {
                "dataset": "customers",
                "type": "inner",
                "on": {"cmp": {"left": {"col": "orders.customer_id"}, "op": "=", "right": {"col": "customers.customer_id"}}},
            },
        ],
        "select": [{"expr": {"col": "orders.order_id"}, "alias": "order_id"}],
        "limit": 5,
    }
    with pytest.raises(QueryPlanError, match="join.as"):
        c.compile(plan)


def test_self_join_with_as_compiles():
    schema = _minimal_schema()
    c = Compiler(schema)
    plan = {
        "dataset": "orders",
        "joins": [
            {
                "dataset": "orders",
                "as": "o2",
                "type": "inner",
                "on": {"cmp": {"left": {"col": "orders.order_id"}, "op": "=", "right": {"col": "o2.order_id"}}},
            }
        ],
        "select": [{"expr": {"col": "orders.order_id"}, "alias": "a"}],
        "limit": 1,
    }
    sql, _params = c.compile(plan)
    assert "JOIN" in sql.upper()
    assert "orders" in sql.lower()


def test_sql_function_wrong_arity_raises_query_plan_error():
    """sqlalchemy.func accepts any name; bad arity should become a clear QueryPlanError."""
    schema = _minimal_schema()
    c = Compiler(schema)
    plan = {
        "dataset": "orders",
        "select": [
            {
                "expr": {"func": "count", "args": [{"col": "order_id"}, {"col": "customer_id"}]},
                "alias": "x",
            }
        ],
        "limit": 1,
    }
    with pytest.raises(QueryPlanError, match="does not accept"):
        c.compile(plan)


def test_set_op_mismatched_column_count():
    schema = _minimal_schema()
    c = Compiler(schema)
    plan = {
        "set_op": {
            "op": "union_all",
            "left": {"dataset": "orders", "select": [{"expr": {"col": "order_id"}, "alias": "a"}], "limit": 1},
            "right": {
                "dataset": "orders",
                "select": [
                    {"expr": {"col": "order_id"}, "alias": "a"},
                    {"expr": {"col": "customer_id"}, "alias": "b"},
                ],
                "limit": 1,
            },
        }
    }
    with pytest.raises(QueryPlanError, match="same number of select columns"):
        c.compile(plan)


def test_empty_in_list_rejected():
    schema = _minimal_schema()
    c = Compiler(schema)
    plan = {
        "dataset": "orders",
        "select": [{"expr": {"col": "order_id"}, "alias": "x"}],
        "where": {"cmp": {"left": {"col": "order_id"}, "op": "in", "right": []}},
        "limit": 1,
    }
    with pytest.raises(QueryPlanError, match="non-empty"):
        c.compile(plan)


def test_dotted_logical_table_column_ref():
    """Longest known-table prefix wins for col refs."""
    schema = {
        "tables": [
            {
                "name": "a.b",
                "db_table": "ab_t",
                "columns": [{"name": "c", "db_column": "c", "type": "int"}],
            },
            {
                "name": "a",
                "db_table": "a_t",
                "columns": [{"name": "b", "db_column": "b", "type": "int"}],
            },
        ],
        "links": [],
    }
    c = Compiler(schema)
    plan = {
        "dataset": "a.b",
        "select": [{"expr": {"col": "a.b.c"}, "alias": "x"}],
        "limit": 1,
    }
    sql, _ = c.compile(plan)
    assert "ab_t" in sql or "c" in sql
