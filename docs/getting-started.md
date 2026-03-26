# Getting Started

Get from zero to a running QCE query in under five minutes.

---

## Prerequisites

- Python ≥ 3.10
- A running Postgres database (local or remote)
- Optional: an LLM API key (OpenAI, Gemini, Groq, etc.) — only needed if you want natural language input

---

## 1. Install

```bash
pip install qce
```

Install with LLM extras if you plan to use natural language input:

```bash
pip install "qce[openai]"                   # OpenAI SDK
pip install langchain-openai                # LangChain + OpenAI
pip install langchain-google-genai          # Gemini (free tier available)
pip install langchain-groq                  # Groq (free, fast)
```

??? note "Install from source"
    ```bash
    git clone https://github.com/alexanderabakah/dsl_compiler
    cd dsl_compiler
    pip install -e ".[dev]"
    ```

---

## 2. Define Your Schema

`schema.yaml` is the **allowlist** that tells QCE which tables and columns exist. The LLM only sees logical names; the compiler maps them to physical Postgres identifiers.

Create `config/schema.yaml`:

```yaml
version: 1
dialect: postgres

tables:
  - name: orders           # (1)
    db_table: orders       # (2)
    description: "Customer orders"
    primary_id: order_id   # (3)
    columns:
      - { name: order_id,    db_column: order_id,    type: integer }
      - { name: customer_id, db_column: customer_id, type: varchar }
      - { name: order_date,  db_column: order_date,  type: date    }
      - { name: freight,     db_column: freight,     type: numeric }
      - { name: ship_country,db_column: ship_country,type: varchar }

  - name: customers
    db_table: customers
    primary_id: customer_id
    columns:
      - { name: customer_id, db_column: customer_id, type: varchar }
      - { name: company_name,db_column: company_name,type: varchar }
      - { name: country,     db_column: country,     type: varchar }

links:
  - name: orders_to_customers
    from_table: orders
    to_table: customers
    join_type: left
    "on": # (4)
      - { from_col: customer_id, to_col: customer_id }
```

1. **Logical name** — what appears in QueryPlan `dataset` and field refs
2. **Physical name** — the actual Postgres table name
3. **Primary key** — used by semantic lint to suggest `count_distinct` on the right column
4. **Quote `"on":`** — `on` is a YAML reserved word; always quote it

!!! warning "Always quote `\"on\":`"
    Forgetting the quotes causes the YAML parser to read `on: [...]` as `true: [...]`, which QCE catches with a clear `SchemaError` — but save yourself the confusion and quote it from the start.

See the full [Schema Reference](schema-reference.md) for all options.

---

## 3. Generate the LLM Spec

QCE auto-generates a compact prompt file from your schema. Run this once, then regenerate whenever `schema.yaml` changes:

```bash
python -m dsl_compiler.spec_builder \
    --schema config/schema.yaml \
    --output config/queryplan_spec_generated.yaml
```

This creates the system prompt that tells the LLM what tables and columns exist, what the QueryPlan shape looks like, and how to construct valid plans. You never have to write it by hand.

---

## 4. Run Your First Query

### Option A — Hand-written plan (no LLM needed)

```python title="quickstart.py"
from sqlalchemy import create_engine
import dsl_compiler as qce

engine = create_engine("postgresql+psycopg2://user:pass@localhost/mydb")

result = qce.execute_query_plan(
    engine=engine,
    schema_path="config/schema.yaml",
    query_plan={
        "dataset": "orders",
        "filters": [
            {"field": "ship_country", "op": "=", "value": "Germany"}
        ],
        "metrics": [
            {"agg": "count", "field": "*", "alias": "total_orders"},
            {"agg": "sum",   "field": "freight", "alias": "total_freight"},
        ],
        "limit": 1,
    },
)

print(result["rows"])
# [{"total_orders": 122, "total_freight": 4891.5}]

print(result["sql"])
# SELECT count(*) AS total_orders, sum(freight) AS total_freight
# FROM orders WHERE ship_country = %(p0)s
```

### Option B — Natural language via `QueryAgent`

```python title="agent_quickstart.py"
from sqlalchemy import create_engine
from openai import OpenAI
import dsl_compiler as qce

engine = create_engine("postgresql+psycopg2://user:pass@localhost/mydb")

agent = qce.QueryAgent(
    engine=engine,
    schema_path="config/schema.yaml",
    spec_path="config/queryplan_spec_generated.yaml",
    llm=OpenAI(api_key="sk-..."),
    max_plan_retries=1,
)

result = agent.ask("Which country had the most orders last month?")
print(result["rows"])    # [{"ship_country": "Germany", "n": 34}]
print(result["sql"])
```

### Option C — LLM planner, execute yourself

```python title="planner_quickstart.py"
from sqlalchemy import create_engine
from openai import OpenAI
import dsl_compiler as qce

engine = create_engine("postgresql+psycopg2://user:pass@localhost/mydb")

planner = qce.QueryPlanPlanner(
    llm=OpenAI(api_key="sk-..."),
    schema_path="config/schema.yaml",
    spec_path="config/queryplan_spec_generated.yaml",
)

# Step 1: generate a QueryPlan from natural language
plan = planner.plan_with_retry("Top 5 customers by freight spend", max_retries=2)
print(plan["meta"]["retry_count"])       # 0 — succeeded on first attempt
print(plan["meta"]["auto_fixes_applied"])

# Step 2: validate independently (optional — execute_query_plan does this too)
errors = qce.validate_query_plan(plan, "config/schema.yaml")
assert not errors

# Step 3: execute
result = qce.execute_query_plan(engine=engine, schema_path="config/schema.yaml", query_plan=plan)
print(result["rows"])
```

---

## 5. Validate Without a Database

Use `validate_query_plan` to check plans in unit tests, CI, or a plan preview endpoint — no database connection required:

```python
errors = qce.validate_query_plan(
    query_plan=plan,
    schema_path="config/schema.yaml",
)

if errors:
    for e in errors:
        print(e)
    # "$.filters[0].field: unknown column 'revenue' on table 'orders'"
else:
    print("✓ Plan is valid")
```

---

## Next Steps

| I want to… | Go to |
|---|---|
| Understand joins, rollup, relative dates | [QueryPlan Reference](query-plan-reference.md) |
| Use a free LLM (Gemini, Groq) | [LLM Integration](llm-integration.md) |
| Learn the full compilation pipeline | [Core Concepts](concepts.md) |
| See every public function | [API Reference](api-reference.md) |
| Handle errors in production | [Exception Reference](exceptions.md) |


---

## 1. Installation

```bash
pip install qce
```

**Optional extras** — install only what you need for your LLM stack:

```bash
pip install "qce[openai]"    # adds openai>=1.0.0
pip install "qce[google]"    # adds google-generativeai>=0.5.0
pip install "qce[dev]"       # adds pytest, ruff, build, twine
```

!!! note "Installing from source"
    ```bash
    git clone https://github.com/alexanderabakah/dsl_compiler
    cd dsl_compiler
    pip install -e ".[dev]"
    ```

---

## 2. Define Your Schema

QCE needs a `schema.yaml` file that maps your database's physical tables and columns to logical names the LLM will use. This file acts as the **compiler allowlist** — nothing outside it is ever accessible.

Create `config/schema.yaml`:

```yaml
version: 1
dialect: postgres

tables:
  - name: orders           # logical name (what the LLM uses)
    db_table: orders       # physical table name in Postgres
    description: "Customer orders"
    primary_id: order_id   # used for count_distinct grain checks
    columns:
      - name: order_id
        db_column: order_id
        type: integer
      - name: customer_id
        db_column: customer_id
        type: varchar
      - name: order_date
        db_column: order_date
        type: date
      - name: freight
        db_column: freight
        type: numeric
        description: "Shipping freight cost"
      - name: ship_country
        db_column: ship_country
        type: varchar

  - name: customers
    db_table: customers
    description: "Customer master"
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

# Join paths between tables
links:
  - name: orders_to_customers
    from_table: orders
    to_table: customers
    join_type: left
    "on":                            # must be quoted — "on" is a reserved word in YAML
      - from_col: customer_id
        to_col: customer_id
```

!!! warning "`on:` must be quoted"
    `on` is a reserved boolean in YAML 1.1. Always write `"on":` (with quotes) in your `links` section, otherwise the YAML parser interprets it as `true`.

See the [Schema Reference](schema-reference.md) for the full format.

---

## 3. Generate the LLM Spec

QCE auto-generates a compact LLM instruction file from your schema. This file tells the LLM what tables and columns exist and how to structure a valid QueryPlan.

```bash
python -m dsl_compiler.spec_builder \
    --schema config/schema.yaml \
    --output config/queryplan_spec_generated.yaml
```

Regenerate this file whenever `schema.yaml` changes.

---

## 4. Run Your First Query

### Without an LLM — hand-written plan

```python
from sqlalchemy import create_engine
import dsl_compiler as qce

engine = create_engine("postgresql+psycopg2://user:pass@localhost/mydb")

plan = {
    "dataset": "orders",
    "filters": [
        {"field": "ship_country", "op": "=", "value": "Germany"}
    ],
    "metrics": [
        {"agg": "count", "field": "*", "alias": "total_orders"}
    ],
}

result = qce.execute_query_plan(
    engine=engine,
    schema_path="config/schema.yaml",
    query_plan=plan,
)

print(result["rows"])       # [{"total_orders": 122}]
print(result["sql"])        # SELECT count(*) AS total_orders FROM orders WHERE ...
print(result["row_count"])  # 1
```

### With an LLM — natural language → QueryPlan → SQL

```python
from sqlalchemy import create_engine
from openai import OpenAI
import dsl_compiler as qce

engine = create_engine("postgresql+psycopg2://user:pass@localhost/mydb")
openai_client = OpenAI(api_key="sk-...")

agent = qce.QueryAgent(
    engine=engine,
    schema_path="config/schema.yaml",
    spec_path="config/queryplan_spec_generated.yaml",
    llm=openai_client,
)

result = agent.ask("How many orders were shipped to Germany?")
print(result["rows"])
```

---

## 5. Validate Without Executing

To check a plan is valid without a database connection:

```python
import dsl_compiler as qce

errors = qce.validate_query_plan(
    query_plan=plan,
    schema_path="config/schema.yaml",
)

if errors:
    for e in errors:
        print(e)
else:
    print("Plan is valid")
```

---

## 6. What's Next?

| Topic | Description |
|---|---|
| [Core Concepts](concepts.md) | Understand the full compilation pipeline |
| [QueryPlan Reference](query-plan-reference.md) | Filters, aggregations, rollup, relative dates |
| [LLM Integration](llm-integration.md) | OpenAI, LangChain, Gemini, Groq, custom callable |
| [Schema Reference](schema-reference.md) | Full schema.yaml format |
| [API Reference](api-reference.md) | All public functions and classes |
