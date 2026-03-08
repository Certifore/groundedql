# QCE — Query Compiler Engine

Deterministic, schema-validated JSON → SQL for Postgres.  
Instead of letting an LLM generate free-form SQL, the LLM outputs a **QueryPlan JSON** (DSL), and QCE compiles it into parameterized SQL and executes it safely.

> **Status:** Not published to PyPI yet. Install from source (see below).

---

## Why QCE Exists

LLM-generated SQL is often:
- **Inconsistent**: Same question → different SQL on every call.
- **Unsafe**: Susceptible to injection and unauthorized schema traversal.
- **Brittle**: Breaks when schema or column names change.

QCE fixes that by splitting the problem in two:

| Responsibility | Who does it |
|---|---|
| Extract intent + entities from natural language | LLM |
| Generate deterministic, safe SQL | QCE (compiler) |

The LLM's only job is to produce a **QueryPlan JSON object**. QCE handles everything else.

**QCE works with any Postgres database and any domain** — e-commerce, finance, healthcare, logistics, SaaS analytics, facility management, or anything else. You define the schema mapping once in `schema.yaml` and QCE enforces it on every query.

---

## Features

- ✅ **Deterministic Compilation**: Same JSON → same SQL, every time.
- ✅ **Any Postgres Schema**: Works with any domain — e-commerce, finance, SaaS, logistics, etc.
- ✅ **Schema Allowlist**: Only tables and columns defined in `schema.yaml` are accessible.
- ✅ **Fully Parameterized**: All values use `bindparams` — no string concatenation, no SQL injection surface.
- ✅ **Schema Load-Time Validation**: Catches misconfigured `schema.yaml` before any query runs.
- ✅ **Statement Timeout**: Every query has a configurable timeout — no runaway queries.
- ✅ **Standalone Plan Validation**: Validate a QueryPlan without a DB connection.
- ✅ **Auto-Fix Layer**: Common LLM mistakes (wrong LIMIT on scalar aggregates, missing inner rollup limits) are fixed automatically.
- ✅ **Optional Retry Loop**: Invalid plans are sent back to the LLM with structured error feedback.
- ✅ **Advanced SQL Support**:
  - **Rollups**: Multi-level aggregations via subquery (e.g. average of per-group counts).
  - **CTEs**: `WITH` clauses for multi-step logic.
  - **Set Operations**: `UNION`, `INTERSECT`, `EXCEPT`.
  - **Expressions**: `CASE`, `CAST`, `COALESCE`, `EXISTS`, scalar subqueries, window functions (`OVER`).
  - **Statistical Functions**: Any Postgres aggregate — `stddev`, `variance`, `corr`, `percentile_cont`, etc.
- ✅ **Join-Path Planning**: Automatically injects joins when a plan references multiple tables.
- ✅ **Semantic Lint**: Catches plans that compile correctly but answer the wrong question.
- ✅ **LLM-Agnostic**: Works with OpenAI, LangChain, Google Gemini, or any callable.
- ✅ **Library-First**: Drop `execute_query_plan` into any agent, router, or API — no framework lock-in.

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
**This is the only place you configure QCE for your specific database.**
The compiler enforces this allowlist — no other tables or columns can be queried.

```yaml
# Works for any Postgres schema — e-commerce, SaaS, finance, logistics, etc.
tables:
  - name: orders                       # logical name used in QueryPlan JSON
    db_table: '"Orders"'               # physical Postgres table (quote = case-sensitive)
    primary_id: order_id               # optional — enforces count_distinct on "how many" questions
    description: Customer orders
    columns:
      - name: order_id
        db_column: order_id
        type: varchar
      - name: customer_id
        db_column: customer_id
        type: varchar
      - name: total_amount
        db_column: total_amount
        type: numeric
      - name: created_at
        db_column: created_at
        type: timestamp

  - name: customers
    db_table: customers
    primary_id: customer_id
    columns:
      - name: customer_id
        db_column: customer_id
        type: varchar
      - name: region
        db_column: region
        type: varchar

links:
  - name: orders_to_customers
    from_table: orders
    to_table: customers
    join_type: left
    "on":
      - left: orders.customer_id
        op: "="
        right: customers.customer_id
```

> **`primary_id` is optional.** When declared, the semantic linter enforces that
> "how many X" questions use `count_distinct(primary_id)` rather than counting
> a non-identifying field. If omitted, the grain check is silently skipped.

> **Important**: Always use **logical names** in QueryPlan JSON, never physical DB names.

> **YAML gotcha**: The `on` key in links must be quoted (`"on":`) because `on` is a reserved
> boolean keyword in YAML 1.1. QCE will raise a clear `SchemaError` if this is forgotten.

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

result = execute_query_plan(
    engine=engine,
    schema_path="config/schema.yaml",
    query_plan=plan,
    statement_timeout_ms=30_000,   # optional, default 30s
)
```

---

### Validate a QueryPlan Without Executing

```python
from dsl_compiler import validate_query_plan

errors = validate_query_plan(plan, "config/schema.yaml")
if errors:
    print("Invalid plan:", errors)
else:
    print("Plan is valid")
```

No database connection required. Use this before sending a plan to the LLM retry loop,
or to pre-check hand-written plans.

---

### Validate schema.yaml at Load Time

```python
from dsl_compiler import load_and_validate_schema

schema = load_and_validate_schema("config/schema.yaml")
# Raises SchemaError immediately on fatal issues (missing db_table, unknown link tables, etc.)
# Prints warnings for non-fatal issues (primary_id pointing to unknown column)
```

---

## Statement Timeout

Every query executed by QCE automatically sets a per-query Postgres statement timeout
before running the SQL. This prevents runaway queries from blocking indefinitely.

Default: **30 seconds**. Override per call:

```python
result = execute_query_plan(
    engine=engine,
    schema_path="config/schema.yaml",
    query_plan=plan,
    statement_timeout_ms=10_000,   # 10 seconds
)
```

If the query exceeds the timeout, Postgres cancels it and QCE raises `DatabaseExecutionError`.

---

## Schema Validation

QCE validates `schema.yaml` at load time on every call to `execute_query_plan` or
`load_and_validate_schema`. Fatal errors raise `SchemaError` immediately. Non-fatal
issues (like `primary_id` pointing to a non-existent column) print a warning and
continue — the grain check is silently skipped for that table.

| Issue | Severity | Behavior |
|---|---|---|
| Missing `tables` list | Fatal | Raises `SchemaError` |
| Table missing `db_table` | Fatal | Raises `SchemaError` |
| Table missing `columns` | Fatal | Raises `SchemaError` |
| Column missing `db_column` | Fatal | Raises `SchemaError` |
| `primary_id` references unknown column | Warning | Prints warning, grain check skipped |
| Link `from_table`/`to_table` unknown | Fatal | Raises `SchemaError` |
| Link missing `on` conditions | Fatal | Raises `SchemaError` |

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

The planner applies deterministic fixes after the LLM responds and records every fix applied:

| Condition | Fix applied | `meta.auto_fixes_applied` value |
|---|---|---|
| No dimensions, all metrics are aggregations, no rollup | `limit` forced to `1`, `offset` forced to `0` | `scalar_aggregate_limit_clamped_to_1` |
| Dimensions present, rollup present, no top-N signal in question | `limit` removed from inner plan so all groups are included | `inner_rollup_limit_removed_for_full_aggregation` |
| Advanced format plan references columns from multiple tables but no joins declared | Shortest join path injected automatically from `links` in `schema.yaml` | `joins_auto_injected_from_link_graph` |

---

## Join-Path Planning

When an advanced format plan references columns from multiple tables but declares no explicit joins, the planner automatically resolves the shortest join path using the `links` graph from `schema.yaml` and injects it before compilation.

**Example** — LLM generates this plan (no joins declared):

```json
{
  "dataset": "work_orders",
  "select": [
    {"expr": {"col": "work_orders.work_order_id"}, "alias": "work_order_id"},
    {"expr": {"col": "assets.keyword_of_asset"}, "alias": "keyword_of_asset"}
  ],
  "limit": 10, "offset": 0
}
```

The planner detects that `assets` is referenced, finds the shortest path via the `work_orders_to_assets_by_asset_tag` link, and injects:

```json
"joins": [{"link": "work_orders_to_assets_by_asset_tag"}]
```

The `meta.auto_fixes_applied` field records `"joins_auto_injected_from_link_graph"`.

**Using join planning directly:**

```python
from dsl_compiler import auto_inject_joins, build_link_graph, shortest_join_path
import yaml

with open("config/schema.yaml") as f:
    schema = yaml.safe_load(f)

# Find shortest path between two tables
graph = build_link_graph(schema)
path = shortest_join_path(graph, "work_orders", "assets")
# [{"name": "work_orders_to_assets_by_asset_tag", "from_table": "work_orders", ...}]

# Auto-inject joins into a plan
fixed_plan = auto_inject_joins(plan, schema)
```

---

## Semantic Lint

In addition to JSON-schema and Pydantic validation, the planner runs a **semantic lint** pass that compares the user's raw question against the generated plan. This catches plans that are structurally valid but semantically wrong — cases where the plan would execute without error but answer a different question than what was asked.

### What it checks

| Rule | Signal words in question | Plan invariant enforced |
|---|---|---|
| **Distinct** | `distinct`, `unique`, `different` | Any metric counting a named field must use `count_distinct`, not `count` |
| **Grouping** | `per X`, `by X`, `each X`, `for each`, `grouped by` | `dimensions` must be non-empty when metrics are present |
| **Top-N** | `top N`, `most`, `least`, `highest`, `lowest`, `ranked` | `order_by` must be set and `limit` must be present |
| **Two-step aggregation** | `average per`, `stddev per`, `median per`, `variance per` | `rollup` must be present |
| **Grain (legacy)** | `how many`, `count of`, `number of`, `total X` | Legacy metrics must use `primary_id` from `schema.yaml` when declared |
| **Grain (advanced)** | `how many`, `count of`, `number of`, `total X` | Advanced format `func: count/count_distinct` nodes must use `primary_id` |

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
[test] Mode=check  Tests=30  Schema=...

  [PASS] cte_top_buildings
  ...
  [PASS] join_path_auto_inject_work_orders_to_assets

[test] Results: 30 passed, 0 failed, 0 errors out of 30 tests
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
| `lint_distinct_*` | Lint: distinct rule (fires + clean + count(*) no FP) |
| `lint_grouping_*` | Lint: grouping rule (fires + clean + percent no FP) |
| `lint_top_n_*` | Lint: top-N rule (fires on missing order_by, limit, clean) |
| `lint_two_step_*` | Lint: two-step aggregation rule (fires + clean + stddev) |
| `lint_grain_*` | Lint: grain rule legacy format (fires + clean) |
| `lint_grain_advanced_format_*` | Lint: grain rule advanced format (fires + clean) |
| `lint_meta_auto_fix_scalar` | Auto-fix: scalar aggregate limit clamped |
| `lint_limit_policy_rollup_no_top_n` | Auto-fix: rollup inner limit removed |
| `lint_targeted_retry_schema_error_fragment` | Lint: targeted retry grain fragment |
| `join_path_auto_inject_work_orders_to_assets` | Join-path: auto-inject from link graph |

---

## Who Is QCE For?

QCE is a general-purpose compiler. It is not tied to any industry or domain. If your system:

- Has a Postgres database
- Wants to let users (or LLMs) query it in natural language
- Needs deterministic, safe, auditable SQL generation

...then QCE is the right tool. The only customization required is writing `schema.yaml` for your database and `queryplan_spec.yaml` with domain-appropriate examples for the LLM.

**Example domains where QCE applies directly:**

| Domain | Example questions |
|---|---|
| E-commerce | "What are the top 10 products by revenue this month?" |
| SaaS analytics | "How many active users per plan tier?" |
| Finance | "What is the average transaction value by region?" |
| Healthcare | "How many patients were admitted per ward last week?" |
| Logistics | "Which routes have the most delayed shipments?" |
| Facility management | "How many open work orders per building?" |
| HR | "What is the average tenure per department?" |

---

## Roadmap

- [ ] Publish to PyPI
- [x] First-class `validate_query_plan` API (pre-execution, callable independently)
- [ ] DB introspection to auto-generate `schema.yaml`
- [ ] Richer join inference (multi-hop links)
- [ ] Correlated subqueries

---

## License

MIT — add a `LICENSE` file and update `pyproject.toml` accordingly.