"""
Frozen read-SQL surface for IntentQL (library contract, not user documentation).

Defines which read-only Postgres-shaped constructs the plan JSON + compiler
target for a major line of development. Version bumps when the set of
supported constructs changes incompatibly.

Used by:
  - spec_builder (injects capability summary into generated queryplan spec)
  - tests / callers that need a single source of truth for feature flags
"""
from __future__ import annotations

from typing import Any, Dict, List

# Bump when supported constructs or defaults change incompatibly.
READ_SQL_SURFACE_VERSION: int = 1

# Declarative contract — keep in sync with compiler.py capabilities.
READ_SQL_SURFACE: Dict[str, Any] = {
    "version": READ_SQL_SURFACE_VERSION,
    "dialect": "postgresql",
    "statement_class": "read_only_select",
    "composition": [
        "single_select",
        "with_cte_recursive_false",
        "set_ops_union_union_all_intersect_except",
    ],
    "from_clause": [
        "single_dataset",
        "explicit_joins_inner_left",
        "link_declared_joins",
        "self_join_via_join_as",
    ],
    "predicates": [
        "legacy_filters_and",
        "where_boolean_tree_and_or_not_cmp",
        "relative_date_sentinels",
    ],
    "projection": [
        "dimensions_metrics_legacy",
        "select_expr_list",
        "time_bucket_on_dimensions",
    ],
    "aggregation": [
        "group_by",
        "having",
        "rollup_outer_metrics",
        "filtered_aggregates_via_expr",
    ],
    "windows": [
        "func_over_partition_order",
    ],
    "subqueries": [
        "scalar_subquery_in_expr",
        "nested_plans_in_cte",
    ],
    "ordering_pagination": ["order_by", "limit", "offset"],
    "limits": {
        "max_joins_default": 8,
        "max_select_default": 200,
        "max_predicates_default": 200,
        "max_limit_clamp_default": 1000,
    },
    "out_of_scope_examples": [
        "mutating statements (INSERT/UPDATE/DELETE/DDL)",
        "arbitrary raw SQL strings from the LLM",
        "queries referencing tables/columns not in schema.yaml",
    ],
}


def read_sql_surface_capabilities() -> Dict[str, Any]:
    """Return a copy of the frozen surface dict (safe to mutate by callers)."""
    import copy

    return copy.deepcopy(READ_SQL_SURFACE)


def read_sql_surface_summary_for_spec() -> str:
    """Short prose block embedded in auto-generated queryplan specs."""
    caps = READ_SQL_SURFACE
    lines: List[str] = [
        f"READ_SQL_SURFACE_VERSION={caps['version']} (internal contract).",
        "The compiler targets read-only SELECT-shaped plans over schema.yaml only.",
        f"Composition: {', '.join(caps['composition'])}.",
        f"FROM / joins: {', '.join(caps['from_clause'])}.",
        f"Predicates: {', '.join(caps['predicates'])}.",
        f"Aggregation: {', '.join(caps['aggregation'])}.",
        f"Windows: {', '.join(caps['windows'])}.",
        f"Subqueries: {', '.join(caps['subqueries'])}.",
        f"Hard caps (defaults): {caps['limits']}.",
        "Out of scope for NL→plan→SQL path: "
        + "; ".join(caps["out_of_scope_examples"]),
    ]
    return "\n".join(lines)
