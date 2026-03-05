```markdown
# DSL-to-SQL Compiler for LLMs

Deterministic, schema-validated JSON → SQL for Postgres.  
Instead of letting an LLM generate free-form SQL, the LLM outputs a **QueryPlan JSON** (DSL), and this library compiles it into parameterized SQL and executes it safely.

> **Status:** Not published to PyPI yet.

---

## Why this exists

LLM-generated SQL is often:
- inconsistent (same question → different SQL),
- unsafe (injection / unexpected joins),
- brittle (minor schema changes break prompts).

This library makes the LLM do a simpler job:
- extract intent + entities → generate a structured QueryPlan JSON
- the compiler does deterministic SQL generation

---

## Features

- ✅ Deterministic JSON → SQL compilation (Postgres)
- ✅ Schema allowlist: only tables/columns defined in YAML are usable
- ✅ Fully parameterized queries (no string concatenation)
- ✅ Supports advanced SQL constructs in the DSL:
  - rollups (avg of grouped metrics via subquery)
  - CTEs (`WITH`)
  - set operations (`UNION`, `INTERSECT`, `EXCEPT`)
  - `CASE` expressions
  - `EXISTS`
  - window functions (`OVER`)
- ✅ Library-first design: plug into any agent/router
- ✅ Optional convenience `QueryAgent` for demos

---

## Repository Layout

```

config/
schema.yaml               # DB schema (logical -> physical mapping)
queryplan_spec.yaml       # instructions/examples for LLM to output QueryPlan JSON

dsl_compiler/
compiler.py               # QueryPlan -> SQL compiler
executor.py               # query execution helper
api/
api.py                  # execute_query_plan entrypoint
spec_api.py             # QueryPlan authoring spec helpers
agent.py                  # optional: simple agent wrapper (demo)
llm_integration.py        # optional: example LLM integration

test/
regression_test/
test_qs.json
suite_results.json
test_main.py

````

---

## Installation (from source)

> Not on PyPI yet — for now users install from source.

### Option A (recommended for development): Editable install (changes stay in sync)

An editable install links the installed package to your local working copy.  
That means if you edit the library code, your project will pick up changes immediately (restart your Python process/server).

1) Clone the repo:

```bash
git clone <YOUR_REPO_URL>
cd dsl_compiler
````

2. Create a virtualenv and install in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

✅ Result: `import dsl_compiler` works from anywhere inside that virtualenv, and code changes in this repo are reflected immediately.

**Using it from another project (recommended dev workflow):**

* Activate your other project’s virtualenv
* Install this repo by path in editable mode:

```bash
pip install -e /absolute/path/to/dsl_compiler
```

Now updates to `dsl_compiler` are automatically “in sync” for that project too.

---

### Option B: Non-editable install (stable snapshot)

Use this if you want a fixed copy of the code (changes to the repo won’t affect your environment until you reinstall):

```bash
pip install .
```

To pick up updates later:

```bash
pip install --upgrade .
```

---

### Option C: Install from Git (no PyPI)

If the repo is hosted on GitHub, users can install directly:

```bash
pip install git+https://github.com/<user>/<repo>.git
```

This installs a snapshot. To get updates later, rerun the command.
For reproducibility, pin to a tag/commit (recommended once you start releases).

---

## Configuration

### 1) Database schema: `config/schema.yaml`

You define **logical table/column names** and map them to the actual DB identifiers.

Example (simplified):

```yaml
tables:
  - name: assets
    db_table: '"finalAssets"'        # physical table name in Postgres
    columns:
      - name: asset_tag              # logical name used in QueryPlan JSON
        db_column: asset_tag         # physical column name in DB
        type: varchar

  - name: work_orders
    db_table: '"finalWorkOrder"'
    columns:
      - name: work_order_id
        db_column: '"workOrderId"'
        type: varchar
      - name: building_id
        db_column: '"buildingId"'
        type: varchar
```

**Important:** Your QueryPlan JSON must use **logical** names only
(e.g., `work_orders.building_id`), not physical DB names.

---

### 2) QueryPlan authoring spec: `config/queryplan_spec.yaml`

This file explains to an LLM how to structure QueryPlan JSON consistently.
The library provides APIs to fetch it as raw YAML or as a prompt-ready string.

---

## Core Usage (recommended)

### Execute a QueryPlan (JSON DSL) directly

This is the main library interface. Use it inside any agent/router.

```python
import json
from sqlalchemy import create_engine
from dsl_compiler import execute_query_plan

engine = create_engine("postgresql+psycopg2://USER:PASSWORD@HOST:PORT/DB?sslmode=require")

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

print(json.dumps(result, indent=2, default=str))
```

Returned result includes:

* `rows` (list of dicts)
* `row_count`
* `columns`
* `sql` (compiled SQL)
* `params` (bind params)
* or `error`

---

## Give your LLM the “how to write QueryPlans” instructions

Use this helper to generate a prompt string that includes:

* QueryPlan spec rules
* your DB schema (optional)

```python
from dsl_compiler import get_queryplan_instructions

prompt = get_queryplan_instructions(
    schema_path="config/schema.yaml",
    spec_path="config/queryplan_spec.yaml",
    include_schema_yaml=True,
)

# Feed `prompt` into your LLM as system context.
# The LLM should output ONLY a JSON QueryPlan object.
```

---

## Optional: Using QueryAgent (demo)

`QueryAgent` is a convenience wrapper for demos.
In production, most users will call `execute_query_plan` directly.

```python
from dsl_compiler.agent import QueryAgent
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://USER:PASSWORD@HOST:PORT/DB?sslmode=require")

agent = QueryAgent(
    schema_path="config/schema.yaml",
    engine=engine,
    llm_config={"provider": "openai", "api_key": "YOUR_API_KEY"}
)

print(agent.ask("How many distinct work orders per building?"))
```

---

## Example: Average work orders per building (SQL rollup, no Python math)

```json
{
  "version": "1.0",
  "dataset": "work_orders",
  "dimensions": [{"field": "building_id", "alias": "building_id"}],
  "metrics": [{"agg": "count_distinct", "field": "work_order_id", "alias": "work_orders_per_building"}],
  "filters": [{"field": "building_id", "op": "is_not_null", "value": null}],
  "rollup": {
    "metrics": [{"agg": "avg", "field": "work_orders_per_building", "alias": "avg_work_orders_per_building"}],
    "limit": 1,
    "offset": 0
  }
}
```

---

## Regression Tests

```bash
python test/test_main.py
```

Reads:

* `test/regression_test/test_qs.json`

Writes:

* `test/regression_test/suite_results.json`

---

## Roadmap

* [ ] Publish to PyPI
* [ ] Add first-class QueryPlan validation API (pre-execution)
* [ ] Add more SQL features (joins inference, correlated subqueries, richer windows)
* [ ] Add DB introspection to generate `schema.yaml` automatically

---

## Security Notes

* The compiler only allows tables/columns defined in `schema.yaml`
* Values are parameterized (bind params), reducing injection risk
* In production, use a read-only DB user where possible

---

## License

MIT (recommended). Add a `LICENSE` file and update `pyproject.toml` accordingly.

```
```