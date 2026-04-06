from importlib.metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version("intentql")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from .api.api import execute_query_plan, validate_query_plan, load_and_validate_schema
from .planner import QueryPlanPlanner
from .validation import validate_query_plan_dict, ValidationErrorItem
from .queryplan_models import CteDef, QueryPlan, queryplan_json_schema
from .agent import QueryAgent
from .schema_catalog import SchemaCatalog, load_schema_catalog
from .sql_guard import SqlGuardResult, validate_sql, apply_row_limit
from .guided_sql import run_guided_sql
from .semantic_lint import semantic_lint
from .join_planner import auto_inject_joins, build_link_graph, shortest_join_path
from .plan_canonical import canonicalize_query_plan, plan_fingerprint
from .spec_builder import build_spec, write_spec
from .cli import introspect_database
from .decompose import is_compound, split_compound, SubQuestion
from .plan_autofix import autofix_plan
from .exceptions import (
    DSLCompilerError,
    SchemaError,
    QueryPlanError,
    AmbiguousColumnError,
    DatabaseExecutionError,
    QueryCostError,
)

__all__ = [
    "execute_query_plan",
    "validate_query_plan",
    "load_and_validate_schema",
    "QueryPlanPlanner",
    "validate_query_plan_dict",
    "ValidationErrorItem",
    "CteDef",
    "QueryPlan",
    "queryplan_json_schema",
    "QueryAgent",
    "SchemaCatalog",
    "load_schema_catalog",
    "SqlGuardResult",
    "validate_sql",
    "apply_row_limit",
    "run_guided_sql",
    "semantic_lint",
    "auto_inject_joins",
    "build_link_graph",
    "shortest_join_path",
    "canonicalize_query_plan",
    "plan_fingerprint",
    "is_compound",
    "split_compound",
    "SubQuestion",
    "autofix_plan",
    "DSLCompilerError",
    "SchemaError",
    "QueryPlanError",
    "AmbiguousColumnError",
    "DatabaseExecutionError",
    "QueryCostError",
    "build_spec",
    "write_spec",
    "introspect_database",
]