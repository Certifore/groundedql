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

FilterOp = Literal[
    "=", "!=", ">", ">=", "<", "<=",
    "in", "not_in",
    "contains", "not_contains", "starts_with", "ends_with",
    "between",
    "is_null", "is_not_null",
]
AggType = Literal["count", "list", "sum", "avg", "min", "max", "ratio", "difference"]
MetricAggType = Literal["count", "sum", "avg", "min", "max"]
ComparisonOp = Literal["ratio", "difference"]
RatioScale = Literal["raw", "percent"]
FormulaOp = Literal["+", "-", "*", "/"]
SortDir = Literal["asc", "desc"]
TimeBucket = Literal["day", "month", "quarter", "year"]


class IntentFilter(BaseModel):
    model_config = STRICT
    column: str = Field(
        ...,
        description=(
            "Logical column name from schema.yaml. Use table.column when the "
            "constraint belongs to a linked table rather than the selected dataset."
        ),
    )
    op: Optional[FilterOp] = Field(
        None,
        description=(
            "Optional comparison operator. Leave null to infer contains/equality/in "
            "from values and column metadata. Use between with exactly two values."
        ),
    )
    values: List[Any] = Field(
        default_factory=list,
        description=(
            "Zero or more literal values. is_null/is_not_null use no values; "
            "between uses exactly two values. Use exact database casing when known."
        ),
    )


class IntentSegment(BaseModel):
    model_config = STRICT
    name: Optional[str] = Field(None, description="Optional human label for this side of a comparison.")
    filters: List[IntentFilter] = Field(
        default_factory=list,
        description="Filters that define this comparison segment.",
    )


class IntentComparison(BaseModel):
    model_config = STRICT

    operator: ComparisonOp = Field(
        ...,
        description="ratio for A/B; difference for A-B.",
    )
    left: IntentSegment = Field(..., description="Numerator/minuend segment.")
    right: IntentSegment = Field(..., description="Denominator/subtrahend segment.")
    metric_aggregation: Optional[MetricAggType] = Field(
        None,
        description="Metric to compute inside each segment. Defaults to count.",
    )
    metric_field: Optional[str] = Field(
        None,
        description="Column to aggregate for sum/avg/min/max. May be table.column.",
    )
    scale: Optional[RatioScale] = Field(
        None,
        description="For ratio only: raw returns A/B; percent returns A/B*100.",
    )


class IntentConditionalMetric(BaseModel):
    model_config = STRICT

    alias: str = Field(..., description="Output/reference alias for this metric.")
    aggregation: MetricAggType = Field(
        ...,
        description="Aggregate to compute after applying filters.",
    )
    field: Optional[str] = Field(
        None,
        description="Column to aggregate for sum/avg/min/max. May be table.column. Not needed for count.",
    )
    filters: List[IntentFilter] = Field(
        default_factory=list,
        description="Filters that define the rows included in this metric.",
    )
    include: bool = Field(
        True,
        description="Whether to include this metric in the final output.",
    )


class IntentFormulaMetric(BaseModel):
    model_config = STRICT

    alias: str = Field(..., description="Output/reference alias for this computed metric.")
    op: FormulaOp = Field(..., description="Binary arithmetic operator.")
    left: Any = Field(..., description="Metric alias or numeric literal.")
    right: Any = Field(..., description="Metric alias or numeric literal.")
    nullif_right: bool = Field(
        False,
        description="If true, divide/operate against NULLIF(right, 0). Useful for ratios.",
    )
    scale: Optional[float] = Field(
        None,
        description="Optional multiplier applied after the binary operation, e.g. 100 for percent.",
    )
    include: bool = Field(
        True,
        description="Whether to include this formula metric in the final output.",
    )


class QueryIntent(BaseModel):
    """Structured intent that an LLM extracts from a question + schema context."""
    model_config = STRICT

    dataset: str = Field(..., description="Table name from schema.yaml")
    keyword: Optional[str] = Field(
        None,
        description=(
            "Free-text keyword to search across configured text columns. "
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
            "'ratio' for A/B or percentages; 'difference' for A-B comparisons."
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
    comparison: Optional[IntentComparison] = Field(
        None,
        description=(
            "Use for questions comparing two filtered segments, such as a ratio, "
            "percentage, or difference between A and B."
        ),
    )
    conditional_metrics: List[IntentConditionalMetric] = Field(
        default_factory=list,
        description=(
            "Named aggregate metrics over filtered row subsets. Use for conditional "
            "aggregates such as sum of amount where status='open' or year=2020."
        ),
    )
    formula_metrics: List[IntentFormulaMetric] = Field(
        default_factory=list,
        description=(
            "Named arithmetic formulas over conditional_metrics or earlier formulas. "
            "Use for derived values such as differences, ratios, percentages, and "
            "average-per-period calculations."
        ),
    )


def intent_json_schema() -> Dict[str, Any]:
    """JSON Schema for LLM structured output."""
    return QueryIntent.model_json_schema()
