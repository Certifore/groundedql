# API Reference

All public symbols are importable directly from `dsl_compiler`:

```python
import dsl_compiler as qce

# or decompose individually:
from dsl_compiler import execute_query_plan, QueryPlanPlanner, QueryAgent
```

---

## Quick Reference

| Symbol | Kind | Summary |
|---|---|---|
| [`execute_query_plan`](#execute_query_plan) | function | Compile + execute a QueryPlan. Primary entrypoint. |
| [`validate_query_plan`](#validate_query_plan) | function | Validate a plan offline. Returns list of error strings. |
| [`validate_query_plan_dict`](#validate_query_plan_dict) | function | Structured validation. Returns typed error objects. |
| [`load_and_validate_schema`](#load_and_validate_schema) | function | Parse and validate `schema.yaml`. |
| [`QueryPlanPlanner`](#queryplanplanner) | class | LLM → QueryPlan orchestration with retry loop. |
| [`QueryAgent`](#queryagent) | class | Natural language → executed result in one call. |
| [`QueryPlan`](#queryplan) | Pydantic model | Type-safe QueryPlan construction. |
| [`queryplan_json_schema`](#queryplan_json_schema) | function | JSON Schema for structured LLM output. |
| [`semantic_lint`](#semantic_lint) | function | Run semantic checks without execution. |
| [`auto_inject_joins`](#auto_inject_joins) | function | Auto-inject join paths into a plan. |
| [`build_spec` / `write_spec`](#build_spec) | functions | Programmatic spec builder. |
| [`get_queryplan_instructions`](#get_queryplan_instructions) | function | Build full LLM system prompt string. |
| [`Compiler`](#compiler-low-level) | class | Low-level: compile a plan to `(sql, params)`. |

---

## `execute_query_plan`

The primary entrypoint. Compiles and executes a QueryPlan against a live Postgres database.

```python
def execute_query_plan(
    *,
    engine: Engine,
    schema_path: str,
    query_plan: Dict[str, Any],
    raise_on_error: bool = False,
    statement_timeout_ms: int = 30_000,
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `engine` | `sqlalchemy.Engine` | required | A SQLAlchemy engine connected to your Postgres DB |
| `schema_path` | `str` | required | Path to your `schema.yaml` file |
| `query_plan` | `dict` | required | A QueryPlan dict (see [QueryPlan Reference](query-plan-reference.md)) |
| `raise_on_error` | `bool` | `False` | If `True`, raises typed exceptions instead of returning an error dict |
| `statement_timeout_ms` | `int` | `30_000` | Per-query timeout in milliseconds |

**Returns — success:**

```python
{
    "rows":      [{"col": val, ...}, ...],  # list of row dicts
    "row_count": 42,                        # number of rows returned
    "columns":   ["col1", "col2", ...],     # column names in order
    "sql":       "SELECT ...",              # compiled SQL (for logging/debug)
    "params":    {"p0": "Germany", ...},    # bind parameter values
    "meta":      {...},                     # forwarded from plan["meta"] if present
}
```

**Returns — error** (when `raise_on_error=False`):

```python
{"error": {"message": "..."}}
```

**Execution steps:**

1. Load and validate `schema.yaml`
2. Strip `meta` key from plan (forwarded to output, not compiled)
3. Resolve `$relative_date` sentinels to concrete UTC timestamps
4. `auto_inject_joins` — insert shortest join path for multi-table refs
5. `Compiler(schema).compile(plan)` → `(sql, params)`
6. `Executor(engine, timeout).execute(sql, params)` → result dict

**Example:**

```python
from sqlalchemy import create_engine
import dsl_compiler as qce

engine = create_engine("postgresql+psycopg2://user:pass@localhost/mydb")

result = qce.execute_query_plan(
    engine=engine,
    schema_path="config/schema.yaml",
    query_plan={
        "dataset": "orders",
        "filters": [{"field": "ship_country", "op": "=", "value": "Germany"}],
        "metrics": [{"agg": "count", "field": "*", "alias": "n"}],
    },
    raise_on_error=True,
    statement_timeout_ms=10_000,
)
print(result["rows"])
```

---

## `validate_query_plan`

Validates a QueryPlan **offline** — no database connection required.

```python
def validate_query_plan(
    query_plan: Dict[str, Any],
    schema_path: str,
) -> List[str]
```

**Returns:** A list of error strings. Empty list means valid.

```python
errors = qce.validate_query_plan(plan, "config/schema.yaml")
if errors:
    for e in errors:
        print(e)
```

Validation runs three checks in order:

1. **Pydantic** — structural schema (required fields, valid operators/aggs, `extra="forbid"`)
2. **Allowlist** — every table and column present in `schema.yaml`
3. **Semantic rules** — rollup `field` references inner aliases, metric aliases are unique

---

## `load_and_validate_schema`

Load `schema.yaml` and run validation.

```python
def load_and_validate_schema(schema_path: str) -> Dict[str, Any]
```

**Returns:** Parsed schema dict.  
**Raises:** `SchemaError` on fatal issues. Prints warnings to stdout for non-fatal issues.

```python
schema = qce.load_and_validate_schema("config/schema.yaml")
```

---

## `QueryPlanPlanner`

LLM → QueryPlan orchestration with validation and retry.

```python
class QueryPlanPlanner:
    def __init__(
        self,
        *,
        llm: Any,
        schema_path: str,
        spec_path: str | None = None,
        spec_dict: dict | None = None,
        temperature: float = 0.0,
        model: str | None = None,
    )
```

**Parameters:**

| Parameter | Description |
|---|---|
| `llm` | Any LLM — OpenAI client, LangChain model, callable, or `LLMClient` instance |
| `schema_path` | Path to `schema.yaml` |
| `spec_path` | Path to `queryplan_spec_generated.yaml`. Auto-generates from schema if omitted. |
| `spec_dict` | Pre-parsed spec dict. Takes priority over `spec_path`. |
| `temperature` | LLM temperature. Use `0.0` for maximum determinism. |
| `model` | Model name hint (used only by the OpenAI adapter). |

### `plan(question)` — Single Shot

One LLM call, auto-fixes applied. No validation, no retry. Fastest path.

```python
plan = planner.plan("How many customers are in France?")
```

### `plan_with_retry(question, max_retries=1)` — Full Pipeline

LLM call → auto-fix → validate → semantic lint → retry with structured feedback if needed.

```python
plan = planner.plan_with_retry(
    "Top 5 products by revenue last 30 days",
    max_retries=2,
)
```

The returned plan includes a `meta` dict:

```python
plan["meta"] == {
    "plan_hash":          "a3f1b2c9...",  # SHA-256 fingerprint (stable across calls)
    "retry_count":        0,              # 0 = first attempt succeeded
    "auto_fixes_applied": [],             # list of applied auto-fix descriptions
    "validation_errors":  [],             # empty = valid
    "lint_errors":        [],             # semantic lint; QueryAgent refuses to execute if non-empty (when enforce_semantic_lint=True)
}
```

---

## `QueryAgent`

High-level wrapper: natural language question → executed result in one call.

```python
class QueryAgent:
    def __init__(
        self,
        *,
        engine: Engine,
        schema_path: str,
        spec_path: str,
        llm: Any,
        max_plan_retries: int = 2,
        enforce_semantic_lint: bool = True,
    )
```

`max_plan_retries` is the number of **extra** LLM attempts after the first plan when structural or semantic checks fail. `enforce_semantic_lint` (default True) prevents execution when `semantic_lint` still reports errors after retries; set False only for debugging.

### `ask(question)`

```python
result = agent.ask("What are the top 10 countries by number of orders?")
```

Returns the same structure as `execute_query_plan` on success. On plan validation failure after all retries:

```python
{
    "error": {
        "message":          "validation failed after retries",
        "validation_errors": [{"path": "$.metrics[0].field", "message": "..."}],
        "plan":             {...},
    }
}
```

When semantic lint still fails (and `enforce_semantic_lint` is True):

```python
{
    "error": {
        "message":     "QueryPlan failed semantic lint after retries — plan does not match the question.",
        "lint_errors": ["Lint: ...", ...],
        "plan":        {...},
    }
}
```

---

## `validate_query_plan_dict`

Lower-level validation that returns structured error objects.

```python
def validate_query_plan_dict(
    plan_dict: Dict[str, Any],
    schema_path: str,
) -> Tuple[Optional[QueryPlan], List[ValidationErrorItem]]
```

**Returns:** `(parsed_plan or None, list of ValidationErrorItem)`

```python
from dsl_compiler import validate_query_plan_dict

plan, errors = validate_query_plan_dict(plan_dict, "config/schema.yaml")
for e in errors:
    print(f"{e.path}: {e.message}")
```

`ValidationErrorItem` fields:

| Field | Type | Description |
|---|---|---|
| `path` | `str` | JSONPath-style location, e.g. `"$.filters[0].field"` |
| `message` | `str` | Human-readable error description |

---

## `QueryPlan`

The Pydantic model for type-safe plan construction:

```python
from dsl_compiler import QueryPlan

plan = QueryPlan(
    dataset="orders",
    filters=[{"field": "ship_country", "op": "=", "value": "Germany"}],
    metrics=[{"agg": "count", "field": "*", "alias": "n"}],
)

result = qce.execute_query_plan(
    engine=engine,
    schema_path="config/schema.yaml",
    query_plan=plan.model_dump(),
)
```

---

## `queryplan_json_schema`

Returns the JSON Schema for the QueryPlan model. Pass this to any LLM that supports structured output natively:

```python
from dsl_compiler import queryplan_json_schema

schema = queryplan_json_schema()
# {"type": "object", "properties": {...}, ...}
```

---

## `semantic_lint`

Run semantic lint checks against a plan without executing it:

```python
def semantic_lint(
    question: str,
    plan: Dict[str, Any],
    schema: Dict[str, Any] | None = None,
) -> List[str]
```

```python
from dsl_compiler import semantic_lint

errors = semantic_lint(
    question="Top 10 customers by revenue",
    plan=plan_dict,
    schema=schema_dict,  # optional — enables grain checks
)
for e in errors:
    print(e)
```

---

## `auto_inject_joins`

Lower-level join utilities (`auto_inject_joins`, `build_link_graph`, `shortest_join_path`) for direct pipeline control:

```python
from dsl_compiler import auto_inject_joins, build_link_graph, shortest_join_path

# Build the adjacency graph from schema links
graph = build_link_graph(schema)

# Find shortest path between two tables
path = shortest_join_path(graph, start="order_details", end="customers")
# Returns list of link defs in traversal order, or None if no path

# Auto-inject joins into a plan
enriched_plan = auto_inject_joins(plan_dict, schema)
```

---

## `build_spec`

Programmatic spec builder (equivalent to the CLI command):

```python
from dsl_compiler import build_spec, write_spec

spec = build_spec("config/schema.yaml")          # returns dict
write_spec(spec, "config/queryplan_spec_generated.yaml")
```

---

## `get_queryplan_instructions`

Build the full LLM system prompt string from spec + schema:

```python
from dsl_compiler.api.spec_api import get_queryplan_instructions

prompt = get_queryplan_instructions(
    schema_path="config/schema.yaml",
    spec_path="config/queryplan_spec_generated.yaml",
    include_schema_yaml=True,   # append raw schema.yaml — recommended
)
```

Useful when building your own message lists instead of using `QueryPlanPlanner`.

---

## `Compiler` (low-level)

Direct access to the compiler if you manage the pipeline yourself:

```python
from dsl_compiler.compiler import Compiler

schema = qce.load_and_validate_schema("config/schema.yaml")
compiler = Compiler(
    schema,
    default_limit=100,
    max_limit=1000,
    max_joins=8,
)

sql, params = compiler.compile(plan_dict)
print(sql)
print(params)
```

`compile()` returns `(sql_string, params_dict)` using named bind parameters compatible with psycopg2 (`%(name)s` style).

---
