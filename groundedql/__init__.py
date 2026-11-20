from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("groundedql")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from .agent import QueryAgent
from .api.api import (
    compile_query_plan,
    execute_query_plan,
    load_and_validate_schema,
    plan_and_compile_query,
    plan_query,
    validate_query_plan,
)
from .cli import introspect_database
from .decompose import SubQuestion, is_compound, split_compound
from .exceptions import (
    AmbiguousColumnError,
    DatabaseExecutionError,
    DSLCompilerError,
    QueryCostError,
    QueryPlanError,
    SchemaError,
)
from .join_planner import auto_inject_joins, build_link_graph, shortest_join_path
from .llm_adapters import MistralChatJSONAdapter, OllamaChatJSONAdapter
from .plan_autofix import autofix_plan
from .plan_canonical import canonicalize_query_plan, plan_fingerprint
from .planner import QueryPlanPlanner
from .queryplan_models import CteDef, QueryPlan, queryplan_json_schema
from .read_sql_surface import (
    READ_SQL_SURFACE,
    READ_SQL_SURFACE_VERSION,
    read_sql_surface_capabilities,
    read_sql_surface_summary_for_spec,
)
from .semantic_lint import semantic_lint
from .spec_builder import build_spec, write_spec
from .validation import ValidationErrorItem, validate_query_plan_dict

__all__ = [
    "READ_SQL_SURFACE",
    "READ_SQL_SURFACE_VERSION",
    "AmbiguousColumnError",
    "CteDef",
    "DSLCompilerError",
    "DatabaseExecutionError",
    "MistralChatJSONAdapter",
    "OllamaChatJSONAdapter",
    "QueryAgent",
    "QueryCostError",
    "QueryPlan",
    "QueryPlanError",
    "QueryPlanPlanner",
    "SchemaError",
    "SubQuestion",
    "ValidationErrorItem",
    "auto_inject_joins",
    "autofix_plan",
    "build_link_graph",
    "build_spec",
    "canonicalize_query_plan",
    "compile_query_plan",
    "execute_query_plan",
    "introspect_database",
    "is_compound",
    "load_and_validate_schema",
    "plan_and_compile_query",
    "plan_fingerprint",
    "plan_query",
    "queryplan_json_schema",
    "read_sql_surface_capabilities",
    "read_sql_surface_summary_for_spec",
    "semantic_lint",
    "shortest_join_path",
    "split_compound",
    "validate_query_plan",
    "validate_query_plan_dict",
    "write_spec",
]
