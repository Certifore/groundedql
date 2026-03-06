from __future__ import annotations

import datetime
from typing import Any, Dict

import yaml
from sqlalchemy.engine import Engine

from ..compiler import Compiler
from ..executor import Executor
from ..exceptions import QueryPlanError, DatabaseExecutionError, SchemaError


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


def execute_query_plan(
    *,
    engine: Engine,
    schema_path: str,
    query_plan: Dict[str, Any],
    raise_on_error: bool = False,
) -> Dict[str, Any]:
    """
    Compile and execute a QueryPlan.

    Args:
        engine: SQLAlchemy engine.
        schema_path: Path to schema.yaml.
        query_plan: The QueryPlan dict (from LLM or hand-written).
        raise_on_error: If True, raises QueryPlanError / DatabaseExecutionError
                        instead of returning {"error": ...}. Default False for
                        backward compatibility.

    Returns:
        Dict with keys: rows, row_count, columns, sql, params
        On failure (raise_on_error=False): {"error": {"message": ...}}
    """
    try:
        try:
            with open(schema_path, "r") as f:
                schema = yaml.safe_load(f) or {}
        except Exception as e:
            raise SchemaError(f"Failed to load schema from '{schema_path}': {e}") from e

        resolved_plan = _resolve_relative_dates(query_plan)

        compiler = Compiler(schema)
        sql, params = compiler.compile(resolved_plan)

        executor = Executor(engine)
        result = executor.execute(sql, params)

        if "error" in result:
            raise DatabaseExecutionError(
                result["error"]["message"],
                sql=sql,
            )

        result["sql"] = sql
        result["params"] = params
        return result

    except (QueryPlanError, DatabaseExecutionError, SchemaError):
        if raise_on_error:
            raise
        # Backward-compatible: return error dict
        import traceback
        return {"error": {"message": traceback.format_exc(limit=3)}}
    except Exception as e:
        if raise_on_error:
            raise DatabaseExecutionError(str(e)) from e
        return {"error": {"message": str(e)}}