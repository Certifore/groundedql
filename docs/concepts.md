# Core Concepts

Understanding how QCE works end-to-end will help you configure it correctly and debug issues when they arise.

<div class="qce-path-grid" markdown>
<a class="qce-path-card" href="getting-started/">
  <span class="qce-path-card__kicker">Step 1</span>
  <span class="qce-path-card__title">Getting Started</span>
  <span class="qce-path-card__desc">Install QCE, define schema allowlists, and run your first execution path.</span>
  <span class="qce-path-card__cta">Open guide -></span>
</a>
<a class="qce-path-card" href="query-plan-reference/">
  <span class="qce-path-card__kicker">Step 3</span>
  <span class="qce-path-card__title">QueryPlan Reference</span>
  <span class="qce-path-card__desc">Use the full plan grammar, operators, rollup, joins, and advanced expressions.</span>
  <span class="qce-path-card__cta">Read reference -></span>
</a>
<a class="qce-path-card" href="api-reference/">
  <span class="qce-path-card__kicker">Production</span>
  <span class="qce-path-card__title">API Contracts</span>
  <span class="qce-path-card__desc">Integrate planner/compiler/executor entry points with typed error handling.</span>
  <span class="qce-path-card__cta">Open API docs -></span>
</a>
</div>

!!! abstract "TL;DR"
    QCE separates **intent** (what data you want) from **execution** (how to fetch it). The LLM produces a structured `QueryPlan`; the compiler turns that into parameterized SQL. The two never mix.

---

## The Compilation Pipeline

Every QCE query goes through the same deterministic stages:

```
Natural language question
        │
        ▼  ── optional: skip if you write plans by hand ──
 ┌─────────────────────────────────────────────────────┐
 │  LLM Planner                                         │
 │  QueryPlanPlanner.plan_with_retry()                  │
 │                                                      │
 │  • sends: question + schema + spec YAML              │
 │  • receives: QueryPlan JSON dict                     │
 │  • applies: auto-fixes (relative dates, limits...)   │
 │  • validates & retries with structured error feedback │
 └─────────────────────────────────────────────────────┘
        │
        ▼  QueryPlan dict  ── pure JSON, zero SQL ──
 ┌─────────────────────────────────────────────────────┐
 │  Validation   validate_query_plan_dict()             │
 │                                                      │
 │  1. Pydantic — structural check (extra="forbid")     │
 │  2. Allowlist — every table/column in schema.yaml    │
 │  3. Semantic — rollup integrity, alias uniqueness    │
 └─────────────────────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────────────────┐
 │  Auto Join Inject   auto_inject_joins()              │
 │                                                      │
 │  BFS over schema links → inject shortest join path  │
 └─────────────────────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────────────────┐
 │  Compiler   Compiler.compile()                       │
 │                                                      │
 │  • logical names → physical names (schema.yaml)      │
 │  • all values → %(p0)s bind parameters               │
 │  • builds SQLAlchemy expression tree                 │
 │  • renders: parameterized SQL string                 │
 └─────────────────────────────────────────────────────┘
        │
        ▼  (sql_string, params_dict)
 ┌─────────────────────────────────────────────────────┐
 │  Executor   Executor.execute()                       │
 │                                                      │
 │  • SET LOCAL statement_timeout first                 │
 │  • executes with bind params                         │
 │  • serializes: datetime → ISO, Decimal → float      │
 └─────────────────────────────────────────────────────┘
        │
        ▼
   {"rows": [...], "row_count": N, "columns": [...], "sql": "..."}
```

---

## The Schema: Your Allowlist

`schema.yaml` is the **single source of truth** for everything the compiler is allowed to touch.

!!! danger "Security guarantee"
    If a table or column is not declared in `schema.yaml`, it **cannot** appear in compiled SQL — not from the LLM, not from a hand-written plan, not from user input of any kind. The compiler raises `QueryPlanError` and execution never happens.

The schema defines three things:

=== "Tables & Columns"

    ```yaml
    tables:
      - name: orders               # logical name — the LLM uses this
        db_table: '"Orders"'       # physical name — the DB sees this
        description: "Customer orders"
        columns:
          - name: order_id
            db_column: '"OrderID"'
            type: integer
          - name: freight
            db_column: '"Freight"'
            type: numeric
    ```

=== "Links (Joins)"

    ```yaml
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

    Links are used by the auto-join injector to find join paths automatically.

=== "Primary Keys"

    ```yaml
    tables:
      - name: orders
        primary_id: order_id
    ```

    Primary keys help the semantic linter suggest `count_distinct` over `count` when appropriate.

---

## The QueryPlan: Structured Intent

A `QueryPlan` expresses *what data to fetch*, not *how to fetch it as SQL*. The LLM produces this structure; the compiler handles all SQL decisions.

```json
{
  "dataset": "orders",
  "filters": [{"field": "ship_country", "op": "=", "value": "Germany"}],
  "metrics": [{"agg": "count", "field": "*", "alias": "total_orders"}]
}
```

Compiles to:

```sql
SELECT count(*) AS total_orders
FROM "Orders"
WHERE "ShipCountry" = %(p0)s
```

!!! note "No SQL in the plan"
    The plan has no `FROM`, `WHERE`, `GROUP BY`, or SQL keywords — those are the compiler's responsibility. The LLM only declares intent.

See the [QueryPlan Reference](query-plan-reference.md) for the full spec.

---

## Determinism

The compiler is a **pure function**: identical input always produces identical output.

- No randomness in compilation (LLM temperature is isolated to the planning stage)
- Bind params (`p0`, `p1`, …) are generated from a reset counter on each call
- SHA-256 `plan_hash` attached to every generated plan

This enables cache-by-hash patterns:

```python
plan = planner.plan_with_retry(question)
cache_key = plan["meta"]["plan_hash"]

if cache_key not in result_cache:
    result_cache[cache_key] = qce.execute_query_plan(
        engine=engine, schema_path=..., query_plan=plan
    )
return result_cache[cache_key]
```

---

## Parameterized Values

Every filter value becomes a **named bind parameter** — it is never string-interpolated into SQL:

```python
plan = {"dataset": "orders", "filters": [{"field": "freight", "op": ">", "value": 100}]}
sql, params = Compiler(schema).compile(plan)
# sql:    "SELECT ... FROM orders WHERE freight > %(p0)s"
# params: {"p0": 100}
```

This is identical to what `cursor.execute(sql, params)` requires. There is no mechanism by which a filter value can escape into SQL syntax, making SQL injection structurally impossible regardless of what the LLM (or a user) provides.

---

## Rollup: Two-Level Aggregation

Some questions require aggregating an already-aggregated result — for example: *"What is the average order value per customer?"*

This needs two layers:

1. **Inner query** — sum order value per customer
2. **Outer query** — average those sums

QCE models this with `rollup`:

```json
{
  "dataset": "order_details",
  "dimensions": [{"field": "order_details.order_id"}],
  "metrics": [{"agg": "sum", "field": "order_details.unit_price", "alias": "revenue"}],
  "rollup": {
    "metrics": [{"agg": "avg", "field": "revenue", "alias": "avg_revenue"}],
    "limit": 1
  }
}
```

The compiler wraps the inner query in a CTE and applies the outer aggregation over it. The `rollup.field` values **must match** aliases from the inner `metrics` — the semantic validator enforces this.

---

## Auto Join Injection

When a plan references columns from multiple tables, QCE automatically computes and injects the join path:

```json
{
  "dataset": "orders",
  "dimensions": [
    {"field": "orders.order_date"},
    {"field": "customers.country"}
  ],
  "metrics": [{"agg": "count", "field": "*", "alias": "n"}]
}
```

QCE performs BFS over the `links` graph in `schema.yaml`, finds the shortest path from `orders` to `customers`, and injects the join definition into the plan — no manual join declaration needed.

!!! tip "Explicit joins override auto-injection"
    If you declare joins in the plan's `joins` array, auto-injection is skipped for that plan. Use explicit joins to control join type (INNER vs LEFT) or when the BFS path is ambiguous.

---

## Semantic Lint

Before retrying with the LLM, `semantic_lint` compares the user's question against the plan and flags common model mistakes:

| Question signal | Expected plan property | Rule |
|---|---|---|
| "distinct values", "unique" | `count_distinct` not `count` | 1 |
| "per X", "by X", "each X" | non-empty `dimensions` | 2 |
| "top N", "most", "highest" | `order_by` + `limit` set | 3 |
| "average per", "avg per" | `rollup` present | 4 |
| "how many", "count of" | `count_distinct(primary_id)` | 5 |

Lint errors are appended to the structured retry message sent back to the LLM, giving it an explicit chance to self-correct.

---

## The Spec Builder

The spec builder auto-generates the LLM's system prompt from your `schema.yaml`:

```bash
python -m dsl_compiler.spec_builder \
  --schema config/schema.yaml \
  --output config/queryplan_spec_generated.yaml
```

The generated spec contains:

- System instructions (produce JSON only, never SQL, use logical names)
- Schema summary (tables, columns, types, descriptions)  
- The exact QueryPlan JSON shape with field descriptions
- Operator and aggregation reference
- Concrete examples auto-derived from your schema

!!! warning "Regenerate after every schema change"
    The spec is the LLM's only knowledge of your database structure. If you add a table and don't regenerate, the LLM cannot reference it — and the compiler will reject any column it doesn't know about regardless.
