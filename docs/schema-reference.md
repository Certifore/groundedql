# Schema Reference

`schema.yaml` is the **allowlist** that the QCE compiler enforces. Only tables and columns declared here are reachable by the LLM or any QueryPlan. Anything not listed causes the compiler to raise a `QueryPlanError` — there is no silent fallthrough, no dynamic DB introspection.

!!! tip "Quick start"
    Copy the [Northwind example](getting-started.md) in Getting Started and adapt it to your database. You only need `tables` and `links` to get going.

---

## Top-Level Structure

```yaml
version: 1          # always 1
dialect: postgres   # always postgres for now

tables:
  - ...

links:
  - ...
```

---

## Tables

Each entry under `tables` maps a **logical name** (what the LLM and QueryPlan use) to a **physical name** (what Postgres actually has).

```yaml
tables:
  - name: orders               # (required) logical name — used in QueryPlan "dataset" and field refs
    db_table: orders           # (required) physical table name in Postgres
    description: "Customer orders placed through the storefront"  # (optional) shown to LLM
    primary_id: order_id       # (optional) primary key logical name — used by semantic lint Rule 5

    columns:
      - name: order_id         # (required) logical column name
        db_column: order_id    # (required) physical column name in Postgres
        type: integer          # (optional) type hint — see supported types below
        description: "Unique order identifier"  # (optional) shown to LLM
```

### Supported `type` Values

| Type string | Postgres type |
|---|---|
| `integer`, `int` | `INTEGER` |
| `bigint` | `BIGINT` |
| `float`, `double` | `FLOAT` |
| `numeric`, `decimal` | `NUMERIC` |
| `varchar`, `text`, `string` | `TEXT` |
| `boolean`, `bool` | `BOOLEAN` |
| `date` | `DATE` |
| `timestamp`, `datetime` | `TIMESTAMP` |
| `uuid` | `UUID` |
| `json`, `jsonb` | `JSONB` |

Unknown type strings fall back to `TEXT`.

### Quoted Identifiers

If your physical table or column name contains uppercase letters, spaces, or reserved words, QCE will automatically double-quote it in the generated SQL. You can also force quoting by wrapping the name in the YAML string:

```yaml
- name: user_events
  db_table: '"UserEvents"'   # becomes FROM "UserEvents" in SQL
  columns:
    - name: event_type
      db_column: '"EventType"'
```

### `primary_id`

The `primary_id` field names the primary key column's **logical name**. It is used by:

1. **Semantic lint Rule 5** — if the question says "how many customers", the lint check verifies that the plan uses `count_distinct("customer_id")` (the primary_id) rather than counting a non-key column.
2. **Spec builder** — auto-generates correct `count_distinct(primary_id)` examples in the LLM prompt.

It is not required, but recommended for any table you expect users to ask "how many X" questions about.

---

## Links

Links declare join relationships between tables. The compiler uses these when `auto_inject_joins` is active, or when the QueryPlan explicitly names a link.

```yaml
links:
  - name: orders_to_customers          # (required) unique link name
    from_table: orders                 # (required) logical table name — must exist in tables
    to_table: customers                # (required) logical table name — must exist in tables
    join_type: left                    # (optional) "left" (default) or "inner"
    "on":                              # (required) must be quoted! see warning below
      - left: orders.customer_id       # logical qualified column from `from_table`
        op: "="                        # currently only "=" is supported
        right: customers.customer_id   # logical qualified column from `to_table`
    optional: true                     # (optional) default true — whether the join is LEFT or INNER
```

!!! warning "`on:` must always be quoted"
    `on` is a YAML 1.1 reserved word that parses as boolean `true`. Always write `"on":` (with double quotes). QCE's schema validator will raise a `SchemaError` with a helpful message if it detects this issue.

    ```yaml
    # WRONG — parsed as boolean true
    links:
      - name: orders_to_customers
        on:
          - left: orders.customer_id

    # CORRECT
    links:
      - name: orders_to_customers
        "on":
          - left: orders.customer_id
    ```

### Multi-Column Joins

For composite keys, list multiple conditions under `"on":`:

```yaml
links:
  - name: order_details_join
    from_table: order_details
    to_table: orders
    join_type: inner
    "on":
      - left: order_details.order_id
        op: "="
        right: orders.order_id
```

### Bidirectional Traversal

Links are stored bidirectionally — `auto_inject_joins` can traverse them in either direction to find the shortest path between any two tables.

---

## Full Example

```yaml
version: 1
dialect: postgres

tables:
  - name: customers
    db_table: customers
    description: "Customer master records"
    primary_id: customer_id
    columns:
      - {name: customer_id, db_column: customer_id, type: varchar}
      - {name: company_name, db_column: company_name, type: varchar}
      - {name: city, db_column: city, type: varchar}
      - {name: country, db_column: country, type: varchar}

  - name: orders
    db_table: orders
    description: "Orders placed by customers"
    primary_id: order_id
    columns:
      - {name: order_id, db_column: order_id, type: integer}
      - {name: customer_id, db_column: customer_id, type: varchar}
      - {name: order_date, db_column: order_date, type: date}
      - {name: freight, db_column: freight, type: numeric, description: "Shipping cost"}
      - {name: ship_country, db_column: ship_country, type: varchar}

  - name: order_details
    db_table: order_details
    description: "Line items within each order"
    columns:
      - {name: order_id, db_column: order_id, type: integer}
      - {name: product_id, db_column: product_id, type: integer}
      - {name: unit_price, db_column: unit_price, type: numeric}
      - {name: quantity, db_column: quantity, type: smallint}
      - {name: discount, db_column: discount, type: numeric}

links:
  - name: orders_to_customers
    from_table: orders
    to_table: customers
    join_type: left
    "on":
      - left: orders.customer_id
        op: "="
        right: customers.customer_id

  - name: order_details_to_orders
    from_table: order_details
    to_table: orders
    join_type: left
    "on":
      - left: order_details.order_id
        op: "="
        right: orders.order_id
```

---

## Schema Validation

When `execute_query_plan` or `load_and_validate_schema` is called, QCE validates the schema file and:

- **Raises `SchemaError`** for fatal problems (missing required fields, unknown link tables, invalid `join_type`, `"on"` parsed as boolean)
- **Prints warnings** for non-fatal issues (e.g. `primary_id` declared but not in the columns list)

You can run schema validation standalone:

```python
from dsl_compiler.api.api import load_and_validate_schema

schema = load_and_validate_schema("config/schema.yaml")
# raises SchemaError on fatal issues
# prints warnings to stdout for non-fatal issues
```
