from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, ConfigDict


# Keep this strict. Extra fields from the LLM should be rejected.
STRICT = ConfigDict(extra="forbid")


Operator = Literal[
    "=", "!=", ">", ">=", "<", "<=",
    "in", "not_in",
    "contains", "not_contains", "starts_with", "ends_with",
    "is_null", "is_not_null",
    "between",
]

Agg = Literal["count", "count_distinct", "sum", "avg", "min", "max"]

TimeBucket = Literal["day", "month", "quarter", "year"]


class QueryFilter(BaseModel):
    model_config = STRICT

    field: str = Field(..., description="Logical column name (snake_case) from schema.yaml")
    op: Operator
    value: Optional[Union[str, int, float, List[Any]]] = None


class QueryDimension(BaseModel):
    model_config = STRICT

    field: str = Field(..., description="Logical column name (snake_case) from schema.yaml")
    alias: Optional[str] = Field(None, description="Optional alias for output column name")
    time_bucket: Optional[TimeBucket] = Field(
        None,
        description="If set, compiler groups by date_trunc(bucket, field) instead of raw timestamp.",
    )


class QueryMetric(BaseModel):
    model_config = STRICT

    agg: Agg
    field: Optional[str] = Field(
        None,
        description="Logical column name (snake_case) from schema.yaml, or '*' for count",
    )
    alias: str = Field(..., description="Output alias for this metric")


class OrderBy(BaseModel):
    model_config = STRICT

    by: str = Field(..., description="Metric alias or dimension alias/field")
    dir: Literal["asc", "desc"] = "asc"


class Rollup(BaseModel):
    model_config = STRICT

    metrics: List[QueryMetric] = Field(..., min_length=1, description="Outer aggregation over inner output")
    limit: Optional[int] = 1
    offset: int = 0


class CteDef(BaseModel):
    """Named sub-plan compiled as WITH name AS (...)."""

    model_config = STRICT

    name: str = Field(..., description="CTE name (SQL identifier, not a schema table name)")
    plan: Dict[str, Any] = Field(
        ...,
        description="Nested plan: legacy 1.0 shape and/or select/set_op/with accepted by Compiler.",
    )


class QueryPlan(BaseModel):
    """
    Legacy QueryPlan shape plus optional advanced ``where`` tree (boolean ``and`` / ``or`` / ``cmp``).
    When ``keyword_search_or`` is set in schema.yaml for a table, combine legacy ``filters`` (AND)
    with ``where`` for OR-of-contains across those columns — see queryplan spec examples.
    """
    model_config = STRICT

    version: Literal["1.0"] = "1.0"
    dataset: str = Field(..., description="Logical dataset name, e.g. 'assets' or 'work_orders'")

    filters: List[QueryFilter] = Field(default_factory=list)
    dimensions: List[QueryDimension] = Field(default_factory=list)
    metrics: List[QueryMetric] = Field(default_factory=list)

    # Advanced predicate tree merged with legacy filters in the compiler (AND).
    where: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional boolean tree: {and: [...]}, {or: [...]}, {cmp: {left, op, right}}, {not: ...}",
    )

    order_by: List[OrderBy] = Field(default_factory=list)
    limit: Optional[int] = 100
    offset: int = 0

    rollup: Optional[Rollup] = None

    # Non-legacy top-level select (e.g. ratio = scalar exprs); compiler lowers when present.
    select: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Direct select list (expr/alias); used when dimensions/metrics are empty.",
    )

    # JSON key "with" — SQL CTEs, composed before the main dataset query (see compiler _build_selectable).
    ctes: Optional[List[CteDef]] = Field(
        default=None,
        alias="with",
        description="Optional WITH clauses; each plan is compiled recursively.",
    )


def queryplan_json_schema() -> Dict[str, Any]:
    """JSON Schema that you can pass to structured-output capable LLMs."""
    return QueryPlan.model_json_schema()