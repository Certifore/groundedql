from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

import yaml
from sqlalchemy.engine import Engine

from ..compiler import Compiler
from ..executor import Executor
from ..exceptions import QueryPlanError, DatabaseExecutionError, SchemaError
from ..join_planner import auto_inject_joins
from ..schema_validator import validate_schema
from ..validation import validate_query_plan_dict


def _resolve_relative_dates(plan: Any) -> Any:
    """
    Recursively walk the plan and replace relative date sentinels with
    concrete ISO-8601 UTC timestamps.

    Supported value shapes in filter/cmp nodes:
      {"$relative_date": {"op": "now_minus_days", "days": 7}}
      -> replaced with "2024-01-15T10:30:00+00:00" (UTC ISO string)

    This allows the LLM to express date-relative intent without generating
    SQL expressions as string values (which fail bindparam type checking).
    """
    if isinstance(plan, dict):
        # Resolve relative date sentinel at this node
        if "$relative_date" in plan:
            spec = plan["$relative_date"]
            op = spec.get("op")
            if op == "now_minus_days":
                days = int(spec.get("days", 0))
                dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
                return dt.isoformat()
            if op == "now_minus_hours":
                hours = int(spec.get("hours", 0))
                dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours)
                return dt.isoformat()
            if op == "today":
                return datetime.date.today().isoformat()
            # Unknown op — leave as-is so validation catches it
            return plan

        return {k: _resolve_relative_dates(v) for k, v in plan.items()}

    if isinstance(plan, list):
        return [_resolve_relative_dates(item) for item in plan]

    return plan


def load_and_validate_schema(schema_path: str) -> Dict[str, Any]:
    """
    Load schema.yaml and run load-time validation.
    Prints any non-fatal warnings. Raises SchemaError on fatal issues.
    """
    try:
        with open(schema_path, "r") as f:
            schema = yaml.safe_load(f) or {}
    except Exception as e:
        raise SchemaError(f"Failed to load schema from '{schema_path}': {e}") from e

    warnings = validate_schema(schema)
    for w in warnings:
        print(f"[QCE schema] {w}")

    return schema


def validate_query_plan(
    query_plan: Dict[str, Any],
    schema_path: str,
) -> List[str]:
    """
    Validate a QueryPlan without executing it.

    Returns a list of error strings. Empty list = valid.
    Does NOT require a database connection.

    Args:
        query_plan: The QueryPlan dict to validate.
        schema_path: Path to schema.yaml.

    Example:
        errors = validate_query_plan(plan, "config/schema.yaml")
        if errors:
            print("Plan is invalid:", errors)
    """
    schema = load_and_validate_schema(schema_path)

    # Strip meta before validation
    clean_plan = {k: v for k, v in query_plan.items() if k != "meta"}
    clean_plan = _resolve_relative_dates(clean_plan)

    _, errors = validate_query_plan_dict(clean_plan, schema_path)
    return [f"{e.path}: {e.message}" for e in errors]


def execute_query_plan(
    *,
    engine: Engine,
    schema_path: str,
    query_plan: Dict[str, Any],
    raise_on_error: bool = False,
    statement_timeout_ms: int = 30_000,
) -> Dict[str, Any]:
    """
    Compile and execute a QueryPlan.

    Args:
        engine: SQLAlchemy engine.
        schema_path: Path to schema.yaml.
        query_plan: The QueryPlan dict (from LLM or hand-written).
        raise_on_error: If True, raises typed exceptions instead of returning
                        {"error": ...}. Default False for backward compatibility.
        statement_timeout_ms: Per-query statement timeout in milliseconds.
                              Default 30000 (30 seconds).

    Returns:
        Dict with keys: rows, row_count, columns, sql, params, meta (if present)
        On failure (raise_on_error=False): {"error": {"message": ...}}
    """
    try:
        schema = load_and_validate_schema(schema_path)

        # Strip meta early — before any processing
        meta = query_plan.get("meta")
        clean_plan = {k: v for k, v in query_plan.items() if k != "meta"}

        resolved_plan = _resolve_relative_dates(clean_plan)
        resolved_plan = auto_inject_joins(resolved_plan, schema)

        compiler = Compiler(schema)
        sql, params = compiler.compile(resolved_plan)

        executor = Executor(engine, statement_timeout_ms=statement_timeout_ms)
        result = executor.execute(sql, params)

        if "error" in result:
            raise DatabaseExecutionError(
                result["error"]["message"],
                sql=sql,
            )

        result["sql"] = sql
        result["params"] = params

        # Forward meta from planner if originally present
        if meta is not None:
            result["meta"] = meta

        return result

    except (QueryPlanError, DatabaseExecutionError, SchemaError):
        if raise_on_error:
            raise
        import traceback
        return {"error": {"message": traceback.format_exc(limit=3)}}
    except Exception as e:
        if raise_on_error:
            raise DatabaseExecutionError(str(e)) from e
        return {"error": {"message": str(e)}}