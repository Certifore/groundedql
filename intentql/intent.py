"""
intent.py — Lightweight structured intent extracted from a natural-language question.

The LLM fills this instead of a full QueryPlan.  A deterministic
``build_plan_from_intent`` then converts it into the QueryPlan that the
compiler expects.  This reduces LLM freedom (and errors) dramatically.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

STRICT = ConfigDict(extra="forbid")

TimeRange = Literal[
    "last_year", "this_year",
    "last_month", "this_month",
    "last_7_days", "last_30_days", "last_90_days",
    "last_12_months",
    "last_6_months",
    "last_2_years",
    "last_3_years",
    "yesterday", "today",
]

AggType = Literal["count", "list", "sum", "avg", "min", "max", "ratio"]
SortDir = Literal["asc", "desc"]
TimeBucket = Literal["day", "month", "quarter", "year"]


class IntentFilter(BaseModel):
    model_config = STRICT
    column: str = Field(..., description="Logical column name from schema.yaml (snake_case)")
    values: List[str] = Field(
        ...,
        min_length=1,
        description="One or more values. Use UPPER CASE when schema says values are stored that way.",
    )


class QueryIntent(BaseModel):
    """Structured intent that an LLM extracts from a question + schema context."""
    model_config = STRICT

    dataset: str = Field(..., description="Table name from schema.yaml (e.g. 'work_orders', 'assets')")
    keyword: Optional[str] = Field(
        None,
        description=(
            "Free-text keyword to search across text columns (e.g. 'plumbing', 'electrical'). "
            "Leave null if the question has no topic/keyword constraint."
        ),
    )
    filters: List[IntentFilter] = Field(
        default_factory=list,
        description="Column-value constraints extracted from the question.",
    )
    time_range: Optional[TimeRange] = Field(
        None,
        description="Time period mentioned in the question, or null if none.",
    )
    aggregation: AggType = Field(
        "count",
        description=(
            "'count' for how many / total; 'list' for show/display/enumerate; "
            "'sum'/'avg'/'min'/'max' for numeric aggregation; "
            "'ratio' for percentages ('what % of X are Y?') — requires keyword and uses count/total."
        ),
    )
    aggregation_field: Optional[str] = Field(
        None,
        description="Column to aggregate (for sum/avg/min/max). Not needed for count or list.",
    )
    group_by: List[str] = Field(
        default_factory=list,
        description=(
            "Columns to group results by. "
            "If the question says 'per building', 'for each house', 'by month' etc., "
            "put the column name here."
        ),
    )
    time_bucket: Optional[TimeBucket] = Field(
        None,
        description=(
            "For trends over time: bucket the primary date column by day/month/quarter/year "
            "(compiler uses date_trunc). Set when the user asks for a trend, over time, or by month/year."
        ),
    )
    sort_direction: Optional[SortDir] = Field(
        None,
        description="'desc' for most/highest first, 'asc' for least/lowest first, null if unspecified.",
    )
    sort_column: Optional[str] = Field(
        None,
        description=(
            "For list queries: ORDER BY this column (e.g. primary_date for 'most recent work order'). "
            "Use with sort_direction."
        ),
    )
    limit: Optional[int] = Field(
        None,
        description="Number of results if the question says 'top N' or 'first N'. Null otherwise.",
    )
    output_columns: List[str] = Field(
        default_factory=list,
        description="For 'list' aggregation: which columns to include in the output.",
    )


def intent_json_schema() -> Dict[str, Any]:
    """JSON Schema for LLM structured output."""
    return QueryIntent.model_json_schema()
