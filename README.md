# DSL-to-SQL Compiler for LLMs

Deterministic, schema-validated JSON → SQL for Postgres.  
Instead of letting an LLM generate free-form SQL, the LLM outputs a **QueryPlan JSON** (DSL), and this library compiles it into parameterized SQL and executes it safely.

> **Status:** Not published to PyPI yet. Install from source (see below).

---

## Why This Exists

LLM-generated SQL is often:
- **Inconsistent**: Same question → different SQL on every call.
- **Unsafe**: Susceptible to injection and unauthorized schema traversal.
- **Brittle**: Breaks when schema or column names change.

This library fixes that by splitting the problem in two:

| Responsibility | Who does it |
|---|---|
| Extract intent + entities from natural language | LLM |
| Generate deterministic, safe SQL | This library (compiler) |

The LLM's only job is to produce a **QueryPlan JSON object**. The compiler handles everything else.

---

## Features

- ✅ **Deterministic Compilation**: Same JSON → same SQL, every time.
- ✅ **Schema Allowlist**: Only tables and columns defined in `schema.yaml` are accessible.
- ✅ **Fully Parameterized**: All values use `bindparams` — no string concatenation.
- ✅ **Auto-Fix Layer**: Scalar aggregate queries (`COUNT`, `AVG`, etc. with no dimensions) are automatically clamped to `limit=1`.
- ✅ **Optional Retry Loop**: If the LLM produces an invalid QueryPlan, the planner sends validation errors back and retries once.
- ✅ **Advanced SQL Support**:
  - **Rollups**: Multi-level aggregations (e.g., average of per-building counts) via subquery.
  - **CTEs**: `WITH` clauses for multi-step logic.
  - **Set Operations**: `UNION`, `INTERSECT`, `EXCEPT`.
  - **Expressions**: `CASE`, `CAST`, `COALESCE`, `EXISTS`, Window functions (`OVER`).
- ✅ **LLM-Agnostic**: Works with OpenAI, LangChain, or any callable.
- ✅ **Library-First**: Use `execute_query_plan` directly in any agent/router — no magic.

---

## Repository Layout

```text
config/
  schema.yaml               # Logical → physical DB mapping (tables, columns, links)
  queryplan_spec.yaml       # DSL spec + examples fed to the LLM as system context

dsl_compiler/
  __init__.py               # Public exports
  compiler.py               # Core: QueryPlan JSON → parameterized SQL (SQLAlchemy Core)
  executor.py               # Runs SQL against the DB, serializes results (dates, decimals)
  queryplan_models.py       # Pydantic models + JSON Schema for QueryPlan validation
  validation.py             # Semantic validation: dataset/column allowlist, rollup rules
  planner.py                # LLM → QueryPlan JSON, with auto-fix + retry loop
  llm_adapters.py           # LLM adapter factory (OpenAI, LangChain, callable)
  llm_client.py             # Base LLMClient protocol + CallableLLMClient
  agent.py                  # Optional: QueryAgent convenience wrapper (demo/integration)
  api/
    api.py                  # execute_query_plan — the main library entrypoint
    spec_api.py             # get_queryplan_instructions — builds LLM system prompt

test/
  test_main.py              # Regression test runner
  regression_test/
    test_qs.json            # Test questions + expected outputs
    suite_results.json      # Written by test runner
```

---

## Installation

> Not on PyPI yet. Install from source.

### Recommended: Editable Install

Changes to the library are immediately reflected — no reinstall needed.

```bash
git clone <YOUR_REPO_URL>
cd dsl_compiler
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

**Using from another project** (e.g. your backend repo):

```bash
# Activate your backend's virtualenv, then:
pip install -e /absolute/path/to/dsl_compiler
```

### Stable Snapshot

```bash
pip install .
# To pick up updates later:
pip install --upgrade .
```

### From Git (no clone)

```bash
pip install git+https://github.com/<user>/<repo>.git
```

---

## Configuration

### 1. `config/schema.yaml` — Logical → Physical Mapping

Define the logical names your QueryPlan JSON uses and map them to the actual Postgres identifiers.
The compiler enforces this allowlist — no other tables or columns can be queried.

```yaml
tables:
  - name: assets                       # logical name used in QueryPlan JSON
    db_table: '"finalAssets"'          # physical Postgres table (quoted = case-sensitive)
    columns:
      - name: asset_tag                # logical column name
        db_column: asset_tag           # physical column name
        type: varchar
      - name: building_name
        db_column: building_name
        type: varchar

  - name: work_orders
    db_table: '"finalWorkOrder"'
    columns:
      - name: work_order_id
        db_column: '"workOrderId"'     # quoted physical name
        type: varchar
      - name: building_id
        db_column: '"buildingId"'
        type: varchar

# Optional: define how tables can be joined
links:
  - name: work_orders_to_assets_by_asset_tag
    from_table: work_orders
    to_table: assets
    join_type: left
    on:
      - left: work_orders.asset_tag
        op: "="
        right: assets.asset_tag
```

> **Important**: Your QueryPlan JSON must always use **logical names** (`work_orders.building_id`), never physical DB names (`"buildingId"`).

---

### 2. `config/queryplan_spec.yaml` — LLM Instructions

This file contains the DSL specification, structural rules, semantic rules, and examples.
Feed it to your LLM as system context using `get_queryplan_instructions`.
The richer this file, the more consistent the LLM's output.

---

## Core Usage

### Execute a QueryPlan Directly

The primary library interface. Use this inside any agent, router, or tool.

```python
from sqlalchemy import create_engine
from dsl_compiler import execute_query_plan

engine = create_engine("postgresql+psycopg2://user:pass@host:port/db?sslmode=require")

plan = {
    "version": "1.0",
    "dataset": "assets",
    "filters": [
        {"field": "building_name", "op": "contains", "value": "CHEN"},
        {"field": "keyword_of_asset", "op": "contains", "value": "FIRE EXTINGUISHER"}
    ],
    "limit": 20,
    "offset": 0
}

result = execute_query_plan(
    engine=engine,
    schema_path="config/schema.yaml",
    query_plan=plan
)
# result keys: rows, row_count, columns, sql, params  (or "error")
```

---

### Build the LLM System Prompt

```python
from dsl_compiler import get_queryplan_instructions

prompt = get_queryplan_instructions(
    schema_path="config/schema.yaml",
    spec_path="config/queryplan_spec.yaml",
    include_schema_yaml=True,   # includes table/column descriptions inline
)
# Feed `prompt` to your LLM as the system message.
# Tell the LLM: "Output ONLY a JSON QueryPlan object."
```

---

### Optional: QueryAgent (Demo / Integration Wrapper)

`QueryAgent` wires the planner + compiler + executor together in one call.
Useful for demos and integrations. In production, most users call `execute_query_plan` directly.

```python
from dsl_compiler import QueryAgent
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://user:pass@host:port/db?sslmode=require")

agent = QueryAgent(
    engine=engine,
    schema_path="config/schema.yaml",
    spec_path="config/queryplan_spec.yaml",
    llm=your_langchain_or_openai_llm,   # see LLM Integration below
    max_plan_retries=1,
)

result = agent.ask("How many distinct buildings have work orders?")
# Returns the execute_query_plan result dict directly
```

---

## LLM Integration

Pass any of the following as the `llm` argument to `QueryAgent` or `QueryPlanPlanner`:

| What you pass | Adapter used |
|---|---|
| Object with `.generate_json(...)` | Used directly |
| `openai.OpenAI()` instance | `OpenAIResponsesJSONAdapter` |
| LangChain chat model (has `.invoke`) | `LangChainJSONAdapter` |
| Plain `callable(json_schema, messages, temperature)` | `CallableLLMClient` |

> **LangChain note**: LangChain models implement both `__call__` (deprecated) and `.invoke`. The adapter factory always routes LangChain models through `.invoke` with proper `BaseMessage` objects to avoid deprecation warnings.

---

## QueryPlan Format

### Legacy Format (recommended for most queries)

```json
{
  "version": "1.0",
  "dataset": "work_orders",
  "dimensions": [{"field": "building_id", "alias": "building_id"}],
  "metrics": [{"agg": "count_distinct", "field": "work_order_id", "alias": "wo_count"}],
  "filters": [{"field": "building_id", "op": "is_not_null", "value": null}],
  "order_by": [{"by": "wo_count", "dir": "desc"}],
  "limit": 10,
  "offset": 0
}
```

### With Rollup (aggregate of grouped values)

When a query needs an aggregate *over* grouped results (e.g., "average work orders **per building**"), use `rollup`. The inner query groups and counts; the rollup outer query aggregates those counts.

```json
{
  "version": "1.0",
  "dataset": "work_orders",
  "dimensions": [{"field": "building_id", "alias": "building_id"}],
  "metrics": [{"agg": "count_distinct", "field": "work_order_id", "alias": "wo_count"}],
  "filters": [{"field": "building_id", "op": "is_not_null", "value": null}],
  "order_by": [],
  "offset": 0,
  "rollup": {
    "metrics": [{"agg": "avg", "field": "wo_count", "alias": "avg_wo_per_building"}],
    "limit": 1,
    "offset": 0
  }
}
```

> `limit` is **omitted** from the inner plan so all buildings are included before the average is computed.

### Supported Operators

| Category | Operators |
|---|---|
| Comparison | `=` `!=` `>` `>=` `<` `<=` |
| Membership | `in` `not_in` |
| Text (case-insensitive) | `contains` `not_contains` `starts_with` `ends_with` |
| Null checks | `is_null` `is_not_null` |

### Supported Aggregations

`count` · `count_distinct` · `sum` · `avg` · `min` · `max`

---

## Auto-Fix Behaviour

The planner applies deterministic fixes after the LLM responds:

| Condition | Fix applied |
|---|---|
| No dimensions, all metrics are aggregations, no rollup | `limit` forced to `1`, `offset` forced to `0` |

This prevents unnecessary `LIMIT 100` clauses on scalar queries like `COUNT(*)`.

---

## Semantic Lint

In addition to JSON-schema and Pydantic validation, the planner runs a **semantic lint** pass that compares the user's raw question against the generated plan. This catches plans that are structurally valid but semantically wrong — cases where the plan would execute without error but answer a different question than what was asked.

### What it checks

| Rule | Signal words in question | Plan invariant enforced |
|---|---|---|
| **Distinct** | `distinct`, `unique`, `different` | Any metric counting a named field must use `count_distinct`, not `count` |
| **Grouping** | `per X`, `by X`, `each X`, `for each`, `grouped by` | `dimensions` must be non-empty when metrics are present |
| **Top-N** | `top N`, `most`, `least`, `highest`, `lowest`, `ranked` | `order_by` must be set and `limit` must be present |
| **Two-step aggregation** | `average per`, `stddev per`, `median per`, `variance per`, `average number of X per` | `rollup` must be present |

### How it works

- Runs after LLM generation and `_auto_fix_plan`, before Pydantic validation
- Returns a list of plain-English error strings (empty = clean)
- On failure, errors are appended to validation feedback and sent to the LLM as part of the retry message
- Never modifies the plan — only reports issues
- False positives waste one retry but don't break anything; false negatives are harmless

### Using it directly

```python
from dsl_compiler import semantic_lint

errors = semantic_lint(
    "What is the average number of work orders per building?",
    plan_dict
)
# errors = ["Lint: question implies an aggregate-of-aggregates ('average number of work orders per') but plan has no rollup..."]
```

---

## Statistical Functions

The compiler supports **any Postgres aggregate or statistical function** — you are not limited to `count`, `sum`, `avg`, `min`, `max`. Use the advanced format `func` expression node with the Postgres function name:

| Function | Use case |
|---|---|
| `stddev` / `stddev_pop` | Standard deviation |
| `variance` / `var_pop` | Variance |
| `corr` | Pearson correlation (two args) |
| `regr_slope` / `regr_intercept` | Linear regression |
| `percentile_cont` / `percentile_disc` | Percentiles / median |
| `mode` | Most frequent value |
| `row_number`, `rank`, `dense_rank` | Window ranking |
| `lag`, `lead`, `first_value`, `last_value` | Window offset |

**Single-step** (aggregating raw column values directly):
```json
{
  "dataset": "assets",
  "select": [
    {"expr": {"func": "stddev", "args": [{"cast": {"expr": {"col": "assets.replacement_year"}, "type": "integer"}}]}, "alias": "stddev_replacement_year"}
  ],
  "limit": 1, "offset": 0
}
```

**Two-step** (aggregating over grouped values — requires rollup):
```json
{
  "dataset": "work_orders",
  "dimensions": [{"field": "building_id", "alias": "building_id"}],
  "metrics": [{"agg": "count_distinct", "field": "work_order_id", "alias": "wo_count"}],
  "filters": [{"field": "building_id", "op": "is_not_null", "value": null}],
  "order_by": [], "offset": 0,
  "rollup": {
    "metrics": [{"agg": "stddev", "field": "wo_count", "alias": "stddev_wo_per_building"}],
    "limit": 1, "offset": 0
  }
}
```

---

## Relative Date Filters

Filter values must be plain scalars — never SQL expressions. For time-relative filters use the `$relative_date` sentinel, which the library resolves to a concrete ISO-8601 UTC timestamp before compilation:

```json
{"field": "edit_date", "op": "<", "value": {"$relative_date": {"op": "now_minus_days", "days": 7}}}
{"field": "entry_date", "op": ">=", "value": {"$relative_date": {"op": "now_minus_hours", "hours": 24}}}
{"field": "edit_date", "op": ">=", "value": {"$relative_date": {"op": "today"}}}
```

Supported ops: `now_minus_days`, `now_minus_hours`, `today`.

---

## Error Handling

The library raises typed exceptions instead of returning error dicts when `raise_on_error=True`:

```python
from dsl_compiler.exceptions import QueryPlanError, DatabaseExecutionError, SchemaError

result = execute_query_plan(
    engine=engine,
    schema_path="config/schema.yaml",
    query_plan=plan,
    raise_on_error=True,   # default False for backward compatibility
)
```

| Exception | When raised |
|---|---|
| `SchemaError` | `schema.yaml` is missing or malformed |
| `QueryPlanError` | Plan is structurally or semantically invalid |
| `AmbiguousColumnError` | Unqualified column name exists in multiple joined tables |
| `DatabaseExecutionError` | Valid plan but Postgres rejected the query |
| `QueryCostError` | Plan exceeds configured `max_cost` complexity threshold |

All exceptions inherit from `DSLCompilerError` for a single catch-all.

---

## Regression Tests

The test suite validates the compiler end-to-end against a real Postgres database.
It covers legacy format, rollups, CTEs, set operations, CASE expressions, EXISTS, and window functions.

### Setup

The test runner reads DB credentials from the repo root `.env` file:

```bash
# .env (repo root)
DB_HOST=your-host
DB_PORT=5432        # or 6543 for Supabase/PgBouncer pooler
DB_NAME=your-db
DB_USER=your-user
DB_PASSWORD=your-password
```

### Modes

| Command | What it does |
|---|---|
| `python test/test_main.py` | Run all tests, print results. No baseline written. |
| `python test/test_main.py update` | Run all tests and **overwrite** the saved baseline. |
| `python test/test_main.py check` | Run all tests and **compare against baseline**. Exits `1` if any regression is found. |

### First-Time Setup

```bash
# Create the baseline before your first push
python test/test_main.py update
```

### Running Regression Checks (CI / Before Pushing)

```bash
python test/test_main.py check
```

Output:

```
[test] Connected: db=postgres, user=postgres
[test] Mode=check  Tests=10  Schema=...

  [PASS] cte_top_buildings
  [PASS] setop_union_assets_chen_beckman
  [PASS] case_asset_category
  [PASS] exists_any_chen_workorders
  [PASS] window_row_number_recent_workorders
  [PASS] count_all_assets
  [PASS] count_distinct_buildings_with_work_orders
  [PASS] avg_work_orders_per_building
  [PASS] top_10_buildings_by_work_orders
  [PASS] fire_extinguishers_chen

[test] Results: 10 passed, 0 failed, 0 errors out of 10 tests
```

### Adding a New Test

1. Open `test/regression_test/test_qs.json`.
2. Add a new entry with a `name`, `question`, and `plan`:

```json
{
  "name": "my_new_test",
  "question": "How many closed work orders are there?",
  "plan": {
    "version": "1.0",
    "dataset": "work_orders",
    "dimensions": [],
    "metrics": [{"agg": "count_distinct", "field": "work_order_id", "alias": "closed_wo_count"}],
    "filters": [{"field": "status_code", "op": "=", "value": "CLOSED"}],
    "order_by": [],
    "limit": 1,
    "offset": 0
  }
}
```

3. Update the baseline to include the new test:

```bash
python test/test_main.py update
```

4. Verify everything passes:

```bash
python test/test_main.py check
```

### What the Check Compares

For each test, `check` mode compares:
- `row_count` — number of rows returned
- `first_row` — the first row of results (as a dict)
- `error` — whether an error occurred

The `sql` field is saved in the baseline for human review but is **not** part of the regression comparison (SQL formatting can change without affecting correctness).

### Test Suite Coverage

| Test Name | Feature Covered |
|---|---|
| `count_all_assets` | Scalar aggregate, legacy format |
| `count_distinct_buildings_with_work_orders` | `COUNT DISTINCT`, scalar aggregate |
| `avg_work_orders_per_building` | Rollup (avg of per-group counts), unlimited inner query |
| `top_10_buildings_by_work_orders` | `GROUP BY`, `ORDER BY`, `LIMIT` |
| `fire_extinguishers_chen` | Filter-only query, auto select-all columns |
| `cte_top_buildings` | CTE (`WITH`), advanced format |
| `setop_union_assets_chen_beckman` | Set operation (`UNION ALL`) |
| `case_asset_category` | `CASE` expression |
| `exists_any_chen_workorders` | `EXISTS` subquery |
| `window_row_number_recent_workorders` | Window function (`ROW_NUMBER OVER`) |

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `SyntaxError` or `unmatched ')'` on import | File corruption from a bad merge — rewrite the file cleanly |
| `'str' object has no attribute 'content'` | LangChain model hit the deprecated `__call__` path — ensure `make_llm_client` routes it to `LangChainJSONAdapter` via `.invoke` |
| Wrong port for Supabase/PgBouncer | Use the transaction pooler port (usually `6543`) with `sslmode=require` |
| Physical table names not quoted | Wrap `CamelCase` names in double quotes in `schema.yaml`: `db_table: '"MyTable"'` |
| `LIMIT 100` on a scalar aggregate | Handled automatically by `_auto_fix_plan` in `planner.py` |

---

## Security Notes

- The compiler only allows tables/columns defined in `schema.yaml` — unknown references raise `QueryPlanError`.
- All user-supplied values go through SQLAlchemy `bindparams` — no string interpolation.
- In production, connect with a **read-only database user**.

---

## Roadmap

- [ ] Publish to PyPI
- [ ] First-class `validate_query_plan` API (pre-execution, callable independently)
- [ ] DB introspection to auto-generate `schema.yaml`
- [ ] Richer join inference (multi-hop links)
- [ ] Correlated subqueries

---

## License

MIT — add a `LICENSE` file and update `pyproject.toml` accordingly.