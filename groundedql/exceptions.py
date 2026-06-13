"""
Public exception hierarchy for GroundedQL.

Consuming code should catch these explicitly:

    from groundedql.exceptions import QueryPlanError, DatabaseExecutionError

    try:
        result = execute_query_plan(...)
    except QueryPlanError as e:
        # invalid plan — LLM output was bad, can retry
    except DatabaseExecutionError as e:
        # valid plan but DB rejected it
    except DSLCompilerError as e:
        # catch-all for any library error
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional


class DSLCompilerError(Exception):
    """Base class for all GroundedQL errors."""


class SchemaError(DSLCompilerError):
    """Raised when schema.yaml is missing, malformed, or references invalid identifiers."""


class QueryPlanError(DSLCompilerError):
    """
    Raised when a QueryPlan is structurally or semantically invalid.
    Suitable to catch and feed back to the LLM for a retry.
    """
    def __init__(
        self,
        message: str,
        *,
        code: str = "INVALID_PLAN",
        path: str = "$",
        suggestion: Optional[str] = None,
        validation_errors: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.path = path
        self.suggestion = suggestion
        self.validation_errors = validation_errors or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "path": self.path,
            "suggestion": self.suggestion,
            "validation_errors": self.validation_errors,
        }


class AmbiguousColumnError(QueryPlanError):
    """
    Raised when an unqualified column reference is ambiguous across
    multiple tables in the current FROM/JOIN scope.
    """
    def __init__(self, column: str, tables: List[str], path: str = "$"):
        super().__init__(
            f"Column '{column}' is ambiguous — it exists in multiple tables: "
            f"{tables}. Use a fully qualified reference like 'table.{column}'.",
            code="AMBIGUOUS_COLUMN",
            path=path,
        )
        self.column = column
        self.tables = tables


class DatabaseExecutionError(DSLCompilerError):
    """
    Raised when a valid, compiled query fails at the database level.
    Wraps the original DB exception.
    """
    def __init__(self, message: str, *, sql: Optional[str] = None, original: Optional[Exception] = None):
        super().__init__(message)
        self.sql = sql
        self.original = original


class QueryCostError(DSLCompilerError):
    """
    Raised when a query exceeds the configured cost/complexity threshold
    before execution (pre-execution safety check).
    """
    def __init__(self, message: str, *, estimated_cost: Optional[float] = None):
        super().__init__(message)
        self.estimated_cost = estimated_cost
