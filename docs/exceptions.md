# Exception Reference

QCE uses a structured exception hierarchy. All exceptions inherit from `DSLCompilerError`, making it easy to catch all library errors in one place or handle each type separately.

```python
from dsl_compiler import (
    DSLCompilerError,
    SchemaError,
    QueryPlanError,
    AmbiguousColumnError,
    DatabaseExecutionError,
    QueryCostError,
)
```

---

## Hierarchy

```
Exception
└── DSLCompilerError
    ├── SchemaError              ← bad schema.yaml configuration
    ├── QueryPlanError           ← invalid or unresolvable query plan
    │   └── AmbiguousColumnError ← unqualified column matches multiple tables
    ├── DatabaseExecutionError   ← valid SQL, but Postgres rejected it
    └── QueryCostError           ← reserved for future cost-guard use
```

---

## `DSLCompilerError`

Base class for all QCE exceptions. Catch this to handle any library error in one place:

```python
try:
    result = qce.execute_query_plan(...)
except qce.DSLCompilerError as e:
    print(f"QCE error: {e}")
```

---

## `SchemaError`

Raised when `schema.yaml` is missing, malformed, or contains invalid configuration.

**When you'll see it:**

- `schema.yaml` does not exist or is not valid YAML
- A table entry is missing `name`, `db_table`, or `columns`
- A column entry is missing `name` or `db_column`
- A link references a table not present in `tables`
- A link's `"on"` field was parsed as a boolean (YAML interprets bare `on:` as `true`)

```python
try:
    schema = qce.load_and_validate_schema("config/schema.yaml")
except qce.SchemaError as e:
    print(f"Fix your schema.yaml: {e}")
```

!!! warning "Common YAML pitfall: bare `on:` key"
    YAML parses `on` as boolean `true` unless quoted. Always quote it:

    ```yaml
    # ❌ "on" parsed as boolean true — causes SchemaError
    links:
      - name: orders_to_customers
        on:
          - from_col: customer_id

    # ✅ correct
    links:
      - name: orders_to_customers
        "on":
          - from_col: customer_id
    ```

---

## `QueryPlanError`

Raised when a QueryPlan is structurally or semantically invalid. This is the most common exception during development.

**Constructor:**

```python
QueryPlanError(
    message: str,
    *,
    code: str = "INVALID_PLAN",
    path: str = "$",
    suggestion: str | None = None,
    validation_errors: list | None = None,
)
```

**Attributes:**

| Attribute | Type | Description |
|---|---|---|
| `message` | `str` | Human-readable error description |
| `code` | `str` | Machine-readable error code |
| `path` | `str` | JSONPath error location, e.g. `"$.filters[0].field"` |
| `suggestion` | `str \| None` | Suggested fix (when available) |
| `validation_errors` | `list \| None` | Structured validation error list |

**Method:**

```python
error.to_dict()
# {"code": "...", "message": "...", "path": "...", "suggestion": "...", "validation_errors": [...]}
```

**When you'll see it:**

- `dataset` references a table not in `schema.yaml`
- A `field` references a column not in `schema.yaml`
- A `rollup.metrics[*].field` references a raw column instead of an inner metric alias
- Metric aliases are not unique
- Unknown fields are present in the plan (`extra="forbid"`)
- `agg` requires a field but none was provided

```python
try:
    sql, params = Compiler(schema).compile({"dataset": "nonexistent", ...})
except qce.QueryPlanError as e:
    print(e.code)       # "INVALID_PLAN"
    print(e.path)       # "$.dataset"
    print(e.message)    # "Unknown dataset: nonexistent"
```

!!! tip "Designed for LLM retry feedback"
    `QueryPlanError` is serializable by design. When using `plan_with_retry`, QCE feeds the error dict back to the LLM automatically. In a custom retry loop:

    ```python
    try:
        sql, params = compiler.compile(plan)
    except qce.QueryPlanError as e:
        error_context = e.to_dict()
        # Append error_context to your next LLM message
    ```

---

## `AmbiguousColumnError`

A subclass of `QueryPlanError` raised when an unqualified column reference matches columns in multiple joined tables.

**Constructor:**

```python
AmbiguousColumnError(column: str, tables: List[str], path: str = "$")
```

**Attributes:**

| Attribute | Type | Description |
|---|---|---|
| `column` | `str` | The ambiguous column name |
| `tables` | `list[str]` | All tables that contain this column |

**Example:** When `orders` is joined with `customers` and both have `customer_id`:

```json
{
  "dataset": "orders",
  "dimensions": [{"field": "customer_id"}]
}
```

This raises `AmbiguousColumnError`. **Fix:** qualify the reference:

```json
{"field": "orders.customer_id"}
```

---

## `DatabaseExecutionError`

Raised when a valid, compiled query fails at the Postgres level.

**Constructor:**

```python
DatabaseExecutionError(
    message: str,
    *,
    sql: str | None = None,
    original: Exception | None = None,
)
```

**Attributes:**

| Attribute | Type | Description |
|---|---|---|
| `sql` | `str \| None` | The SQL that triggered the error |
| `original` | `Exception \| None` | The underlying psycopg2/SQLAlchemy exception |

**When you'll see it:**

- Statement timeout exceeded (`statement_timeout_ms`)
- Postgres connection failure
- Permission denied on a table
- Type mismatch between a filter value and the column type

```python
try:
    result = qce.execute_query_plan(engine=engine, ..., raise_on_error=True)
except qce.DatabaseExecutionError as e:
    print(f"DB error: {e.message}")
    print(f"SQL: {e.sql}")
    print(f"Caused by: {e.original}")
```

---

## `QueryCostError`

Raised when a query exceeds a configured complexity threshold before execution.

**Constructor:**

```python
QueryCostError(message: str, *, estimated_cost: float | None = None)
```

!!! note "Reserved for future use"
    `QueryCostError` is defined in the exception hierarchy but is not currently raised by compiler internals. It is reserved for a future cost-based query guard feature.

---

## Production Error Handling Pattern

Recommended pattern for a web API endpoint using `raise_on_error=True`:

```python
import dsl_compiler as qce

def handle_question(question: str) -> dict:
    try:
        plan = planner.plan_with_retry(question)
        return qce.execute_query_plan(
            engine=engine,
            schema_path="config/schema.yaml",
            query_plan=plan,
            raise_on_error=True,
        )

    except qce.QueryPlanError as e:
        # LLM generated an invalid plan; all retries exhausted
        # Safe to surface the error code to the client
        return {"error": "invalid_query", "detail": e.message, "code": e.code}

    except qce.DatabaseExecutionError as e:
        # Valid SQL but DB rejected it (timeout, permissions, type error)
        # Log e.sql for debugging; do NOT surface raw SQL to end users
        logger.error("DB execution failed", sql=e.sql, cause=str(e.original))
        return {"error": "execution_failed", "detail": "Query failed. Please try again."}

    except qce.SchemaError as e:
        # Configuration error — should never reach production if schema is tested at startup
        logger.critical("Schema configuration error", error=str(e))
        return {"error": "configuration_error", "detail": "Internal error"}
```
