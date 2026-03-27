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

**QCE works with any Postgres database and any domain** — e-commerce, finance, healthcare, logistics, SaaS analytics, or anything else. You define the schema mapping once in `schema.yaml` and QCE enforces it on every query.

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
- ✅ **LLM-Agnostic**: Works with OpenAI, Anthropic, Google Gemini, self-hosted models, or any callable.
- ✅ **Library-First**: Drop `execute_query_plan` into any agent, router, or API — no framework lock-in.

### Recent Compiler Hardening

- Unified exception handling: `Compiler` and `execute_query_plan` now use the same public `QueryPlanError` type.
- Explicit join hardening: repeated/self joins require `join.as` aliases; duplicate aliases are rejected early.
- Set-operation safety: `UNION` / `INTERSECT` / `EXCEPT` branches are checked for matching select-column counts.
- Membership safety: empty `in` / `not_in` lists are rejected at compile time.
- Schema-link validation: invalid link `join_type` values now raise `SchemaError` (`left` / `inner` only).
- Type mapping coverage: added support for more cast/schema hints (`smallint`, `bigint`, `time`, `interval`, `uuid`, `json/jsonb`, `text`).

---

## Repository Layout

```text
config/
  schema.yaml                       # Logical → physical DB mapping (tables, columns, links)
  queryplan_spec.yaml               # Full DSL spec + examples (development mode)
  queryplan_spec_generated.yaml     # Auto-generated minimal spec (production mode)

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
  spec_builder.py           # Auto-generates minimal queryplan_spec from schema.yaml
  api/
    api.py                  # execute_query_plan — the main library entrypoint
    spec_api.py             # get_queryplan_instructions — builds LLM system prompt

test/
  test_main.py              # Regression test runner
  regression_test/
    test_qs.json            # Test questions + expected outputs (Northwind schema)
    suite_results.json      # Written by test runner
```

---

## Installation

> Not on PyPI yet. Install from source.

### Recommended: Editable Install

```bash
git clone https://github.com/Certifore/dsl_compiler.git
cd dsl_compiler
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

**Using from another project:**

```bash
pip install -e /absolute/path/to/dsl_compiler
```

---

## Configuration

### 1. `config/schema.yaml` — Logical → Physical Mapping

Define the logical names your QueryPlan JSON uses and map them to actual Postgres identifiers.
The compiler enforces this allowlist — no other tables or columns can be queried.

```yaml
tables:
  - name: orders
    db_table: orders
    primary_id: order_id
    description: Customer orders
    columns:
      - name: order_id
        db_column: order_id
        type: integer
      - name: customer_id
        db_column: customer_id
        type: varchar
      - name: freight
        db_column: freight
        type: numeric

  - name: customers
    db_table: customers
    primary_id: customer_id
    columns:
      - name: customer_id
        db_column: customer_id
        type: varchar
      - name: company_name
        db_column: company_name
        type: varchar
      - name: country
        db_column: country
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
> "how many X" questions use `count_distinct(primary_id)` rather than a non-identifying field.

> **YAML gotcha**: The `on` key in links must be quoted (`"on":`) — `on` is a reserved
> keyword in YAML 1.1.

### 2. Generate the LLM Spec

```bash
python -m dsl_compiler.spec_builder \
    --schema config/schema.yaml \
    --output config/queryplan_spec_generated.yaml
```

This generates a minimal spec containing only the tables, columns, and examples relevant
to your schema. Regenerate whenever `schema.yaml` changes.

---

## Core Usage

### Execute a QueryPlan Directly

```python
from sqlalchemy import create_engine
from dsl_compiler import execute_query_plan

engine = create_engine("postgresql+psycopg2://user:pass@host:port/db?sslmode=require")

result = execute_query_plan(
    engine=engine,
    schema_path="config/schema.yaml",
    query_plan=plan,
    statement_timeout_ms=30_000,
)
```

### Full Pipeline: Natural Language → Results

```python
from langchain_openai import ChatOpenAI
from dsl_compiler import QueryPlanPlanner, execute_query_plan

llm = ChatOpenAI(model="gpt-4o", temperature=0)
planner = QueryPlanPlanner(
    llm=llm,
    schema_path="config/schema.yaml",
    spec_path="config/queryplan_spec_generated.yaml",
)

plan = planner.plan_with_retry("How many orders per country?")
result = execute_query_plan(engine=engine, schema_path="config/schema.yaml", query_plan=plan)
```

### Validate a QueryPlan Without Executing

```python
from dsl_compiler import validate_query_plan

errors = validate_query_plan(plan, "config/schema.yaml")
if errors:
    print("Invalid plan:", errors)
```

---

## LLM Integration

Pass any of the following as the `llm` argument:

| What you pass | Adapter used |
|---|---|
| `openai.OpenAI()` instance | `OpenAIResponsesJSONAdapter` |
| LangChain chat model (has `.invoke`) | `LangChainJSONAdapter` |
| Plain `callable(json_schema, messages, temperature)` | `CallableLLMClient` |
| Self-hosted model via callable | `CallableLLMClient` |

---

## QueryPlan Format

### Legacy Format (recommended for most queries)

```json
{
  "version": "1.0",
  "dataset": "orders",
  "dimensions": [{"field": "ship_country", "alias": "ship_country"}],
  "metrics": [{"agg": "count_distinct", "field": "order_id", "alias": "order_count"}],
  "filters": [{"field": "ship_country", "op": "is_not_null", "value": null}],
  "order_by": [{"by": "order_count", "dir": "desc"}],
  "limit": 100,
  "offset": 0
}
```

### With Rollup (aggregate of grouped values)

```json
{
  "version": "1.0",
  "dataset": "orders",
  "dimensions": [{"field": "customer_id", "alias": "customer_id"}],
  "metrics": [{"agg": "count_distinct", "field": "order_id", "alias": "orders_per_customer"}],
  "filters": [{"field": "customer_id", "op": "is_not_null", "value": null}],
  "order_by": [],
  "offset": 0,
  "rollup": {
    "metrics": [{"agg": "avg", "field": "orders_per_customer", "alias": "avg_orders_per_customer"}],
    "limit": 1,
    "offset": 0
  }
}
```

> `limit` is omitted from the inner plan so all customers are included before the average is computed.

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

| Condition | Fix applied | `meta.auto_fixes_applied` value |
|---|---|---|
| No dimensions, all metrics are aggregations, no rollup | `limit` forced to `1` | `scalar_aggregate_limit_clamped_to_1` |
| Dimensions present, rollup present, no top-N signal | `limit` removed from inner plan | `inner_rollup_limit_removed_for_full_aggregation` |
| Multi-table plan with no joins declared | Shortest join path injected from `links` | `joins_auto_injected_from_link_graph` |

---

## Semantic Lint

The planner runs a lint pass comparing the question against the plan before execution.

| Rule | Signal words | Invariant enforced |
|---|---|---|
| **Distinct** | `distinct`, `unique`, `different` | Metric must use `count_distinct` |
| **Grouping** | `per X`, `by X`, `each X` | `dimensions` must be non-empty |
| **Top-N** | `top N`, `most`, `least`, `highest`, `lowest` | `order_by` and `limit` must be set |
| **Two-step aggregation** | `average per`, `stddev per`, `median per` | `rollup` must be present |
| **Grain** | `how many`, `count of`, `number of` | Must use `primary_id` when declared |

---

## Error Handling

```python
from dsl_compiler.exceptions import QueryPlanError, DatabaseExecutionError, SchemaError

result = execute_query_plan(
    engine=engine,
    schema_path="config/schema.yaml",
    query_plan=plan,
    raise_on_error=True,
)
```

| Exception | When raised |
|---|---|
| `SchemaError` | `schema.yaml` is missing or malformed |
| `QueryPlanError` | Plan is structurally or semantically invalid |
| `AmbiguousColumnError` | Unqualified column name exists in multiple joined tables |
| `DatabaseExecutionError` | Valid plan but Postgres rejected the query |
| `QueryCostError` | Plan exceeds configured `max_cost` complexity threshold |

---

## Regression Tests

The test suite runs against the **Northwind** database (open source, publicly reproducible).
28 tests covering: scalar aggregates, rollups, CTEs, set operations, CASE expressions,
window functions, lint rules, auto-fix rules, and join-path injection.

### Setup

```bash
# .env (repo root)
DB_USER=your-user
DB_PASSWORD=your-password
DB_HOST=your-host
DB_PORT=5432
DB_NAME=postgres
```

Load Northwind into your Postgres instance:

```bash
curl -o /tmp/northwind.sql https://raw.githubusercontent.com/pthom/northwind_psql/master/northwind.sql
psql -h your-host -U your-user -d postgres -f /tmp/northwind.sql
```

### Modes

```bash
python test/test_main.py          # run all, print results
python test/test_main.py update   # run all and overwrite baseline
python test/test_main.py check    # run all and compare against baseline (CI)
```

### Test Suite Coverage

| Test Name | Feature Covered |
|---|---|
| `count_all_customers` | Scalar aggregate, legacy format |
| `count_all_orders` | Scalar aggregate |
| `count_all_products` | Scalar aggregate |
| `orders_by_country` | `GROUP BY`, `ORDER BY` |
| `top_10_customers_by_orders` | Top-N, `COUNT DISTINCT` |
| `products_in_beverages` | Filter-only query |
| `discontinued_products` | Scalar aggregate with filter |
| `avg_orders_per_customer` | Rollup (avg of per-group counts) |
| `total_freight_cost` | `SUM` aggregate |
| `products_by_category` | Group-by with count |
| `cte_top_customers` | CTE (`WITH`), advanced format |
| `setop_union_customers_germany_france` | Set operation (`UNION ALL`) |
| `case_product_price_tier` | `CASE` expression |
| `exists_customers_with_orders` | Customers with at least one order |
| `window_row_number_orders_by_freight` | Window function (`ROW_NUMBER OVER`) |
| `lint_distinct_*` | Lint: distinct rule |
| `lint_grouping_*` | Lint: grouping rule |
| `lint_top_n_*` | Lint: top-N rule |
| `lint_two_step_*` | Lint: two-step aggregation rule |
| `lint_grain_*` | Lint: grain rule |
| `lint_meta_auto_fix_scalar` | Auto-fix: scalar aggregate limit clamped |
| `lint_limit_policy_rollup_no_top_n` | Auto-fix: rollup inner limit removed |
| `join_path_auto_inject_orders_to_customers` | Join-path: auto-inject from link graph |

---

## Spec Builder

Generates a minimal `queryplan_spec_generated.yaml` from your `schema.yaml`.
Reduces token cost vs the full development spec.

```bash
python -m dsl_compiler.spec_builder \
    --schema config/schema.yaml \
    --output config/queryplan_spec_generated.yaml
```

Regenerate whenever `schema.yaml` changes.

---

## Roadmap

- [ ] Publish to PyPI
- [x] Standalone `validate_query_plan` API
- [x] Auto-generated domain spec (`spec_builder`)
- [ ] DB Introspection Wizard (`qce init <db_url>`)

### Not done yet (planned hardening)

These are known limits today; we expect to chip away at them in follow-up releases.

- [ ] **Scalar subqueries** — The compiler does not prove that a `scalar_subquery` returns at most one row; Postgres errors at runtime if not. Optional lint or documented patterns (`LIMIT 1`, aggregates, unique keys) could tighten this.
- [ ] **`func` names** — Any name is passed through to SQLAlchemy; typos become invalid SQL that fails in Postgres. An optional allowlist or validation mode would catch mistakes earlier.
- [ ] **Set operations** — Branch **column counts** are checked; per-column **type compatibility** for `UNION` / `INTERSECT` / `EXCEPT` is still left to Postgres.
- [ ] **Aggregates / `GROUP BY`** — Invalid grouping (e.g. non-aggregated columns) is not fully modeled in the compiler; the database rejects bad plans.
- [ ] **Join inference + dotted names** — Auto-inject walks `table.column` refs; logical table names containing dots need extra care in that path.
- [ ] **Drivers** — Execution is oriented around SQLAlchemy + the shipped `psycopg2-binary` stack; locked-down environments may prefer building `psycopg2` from source, and async drivers (`asyncpg`, etc.) are not first-class yet.

---

## License

MIT