from importlib.metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version("intentql")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from .agent import QueryAgent
from .schema_api import load_and_validate_schema
from .schema_catalog import SchemaCatalog, load_schema_catalog
from .sql_guard import SqlGuardResult, apply_row_limit, validate_sql
from .sql_canonicalize import canonicalize_sql
from .guided_sql import run_guided_sql
from .exceptions import DSLCompilerError, SchemaError, DatabaseExecutionError

__all__ = [
    "__version__",
    "QueryAgent",
    "load_and_validate_schema",
    "SchemaCatalog",
    "load_schema_catalog",
    "SqlGuardResult",
    "validate_sql",
    "apply_row_limit",
    "canonicalize_sql",
    "run_guided_sql",
    "DSLCompilerError",
    "SchemaError",
    "DatabaseExecutionError",
]
