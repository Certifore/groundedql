# QueryPlan Reference

A QueryPlan is a JSON/dict document that describes *what data to fetch*. The LLM produces it; the compiler turns it into SQL. You can also write plans by hand — the compiler doesn't care about the source.

!!! abstract "Mental model"
    Think of a QueryPlan as a **structured SELECT intent**: which table, which filters, which aggregations, how to sort, and how many rows. The compiler fills in all the SQL syntax — the plan has zero SQL keywords.

---

## Top-Level Structure

```json
{
  "version": "1.0",
  "dataset": "orders",
  "filters":    [...],
  "dimensions": [...],
  "metrics":    [...],
  "order_by":   [...],
  "limit":      100,
  "offset":     0,
  "rollup":     {...}
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `version` | `"1.0"` | no | `"1.0"` | Schema version — always `"1.0"` |
| `dataset` | string | **yes** | — | Logical table name (must be in `schema.yaml`) |
| `filters` | array | no | `[]` | WHERE conditions |
| `dimensions` | array | no | `[]` | GROUP BY columns + SELECT columns |
| `metrics` | array | no | `[]` | Aggregate expressions |
| `order_by` | array | no | `[]` | ORDER BY clauses |
| `limit` | integer \| null | no | `100` | Row limit (max 1000) |
| `offset` | integer | no | `0` | Row offset for pagination |
| `rollup` | object | no | — | Two-level aggregation (see [Rollup](#rollup)) |

!!! note "Unknown fields are rejected"
    The QueryPlan model uses `extra="forbid"`. Any field not listed above will cause a `QueryPlanError` immediately. This prevents prompt injection payloads that try to smuggle extra data through the plan.

---

## Filters

```json
{"field": "ship_country", "op": "=", "value": "Germany"}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `field` | string | **yes** | Logical column name (optionally qualified: `"orders.ship_country"`) |
| `op` | Operator | **yes** | Comparison operator — see table below |
| `value` | any | depends on `op` | Filter value — becomes a bind parameter |

### Operators

#### Comparison

| Operator | SQL equivalent | `value` |
|---|---|---|
| `=` | `col = :p` | scalar |
| `!=` | `col != :p` | scalar |
| `>` | `col > :p` | scalar (numeric / date) |
| `>=` | `col >= :p` | scalar |
| `<` | `col < :p` | scalar |
| `<=` | `col <= :p` | scalar |

#### Membership

| Operator | SQL equivalent | `value` |
|---|---|---|
| `in` | `col IN (:p0, :p1, ...)` | array |
| `not_in` | `col NOT IN (...)` | array |

#### Text

| Operator | SQL equivalent | `value` |
|---|---|---|
| `contains` | `col ILIKE '%value%'` | string |
| `not_contains` | `col NOT ILIKE '%value%'` | string |
| `starts_with` | `col ILIKE 'value%'` | string |
| `ends_with` | `col ILIKE '%value'` | string |

!!! info "`contains` uses `ILIKE`"
    Text operators are case-insensitive (`ILIKE`). This is intentional — LLMs should not need to know the case of values in the database.

#### Null Checks

| Operator | SQL equivalent | `value` |
|---|---|---|
| `is_null` | `col IS NULL` | omit or `null` |
| `is_not_null` | `col IS NOT NULL` | omit or `null` |

#### Range

| Operator | SQL equivalent | `value` |
|---|---|---|
| `between` | `col BETWEEN :p0 AND :p1` | `[low, high]` array |

### Relative Dates

Instead of a literal date string, filters can use a `$relative_date` sentinel that resolves to a concrete UTC timestamp at query time:

```json
{
  "field": "order_date",
  "op": ">=",
  "value": {"$relative_date": {"op": "now_minus_days", "days": 30}}
}
```

| Sentinel op | Resolves to |
|---|---|
| `now_minus_days` | `datetime.utcnow() - timedelta(days=N)` |
| `now_minus_hours` | `datetime.utcnow() - timedelta(hours=N)` |
| `today` | `date.today().isoformat()` |

---

## Dimensions

Dimensions map to `SELECT` + `GROUP BY`:

```json
{"field": "ship_country", "alias": "country"}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `field` | string | **yes** | Logical column name (optionally qualified) |
| `alias` | string | no | AS alias in SELECT — defaults to the column name |

---

## Metrics

Metrics map to aggregate `SELECT` expressions:

```json
{"agg": "sum", "field": "freight", "alias": "total_freight"}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `agg` | Aggregation | **yes** | Aggregation function — see table below |
| `field` | string | depends | Logical column name — required for all aggs except `count(*)` |
| `alias` | string | **yes** | AS alias — must be unique within the plan |

### Aggregations

| Agg | SQL | Field required? |
|---|---|---|
| `count` | `count(*)` or `count(col)` | no (use `"*"` for `count(*)`) |
| `count_distinct` | `count(DISTINCT col)` | **yes** |
| `sum` | `sum(col)` | **yes** |
| `avg` | `avg(col)` | **yes** |
| `min` | `min(col)` | **yes** |
| `max` | `max(col)` | **yes** |

!!! tip "Counting entities"
    To count distinct customers: `{"agg": "count_distinct", "field": "customer_id", "alias": "n_customers"}`.  
    To count rows: `{"agg": "count", "field": "*", "alias": "n_rows"}`.

---

## Order By

```json
{"by": "total_freight", "dir": "desc"}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `by` | string | **yes** | — | Metric alias or dimension alias/field name |
| `dir` | `"asc"` \| `"desc"` | no | `"asc"` | Sort direction |

---

## Rollup

Rollup implements two-level aggregation — aggregate an already-aggregated inner query:

```json
{
  "dataset": "order_details",
  "dimensions": [{"field": "order_details.order_id"}],
  "metrics": [{"agg": "sum", "field": "order_details.unit_price", "alias": "revenue"}],
  "rollup": {
    "metrics": [{"agg": "avg", "field": "revenue", "alias": "avg_revenue"}],
    "limit": 1,
    "offset": 0
  }
}
```

The compiler generates:

```sql
WITH _inner AS (
    SELECT order_id, sum(unit_price) AS revenue
    FROM order_details
    GROUP BY order_id
)
SELECT avg(revenue) AS avg_revenue
FROM _inner
LIMIT 1
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `metrics` | array | **yes** | — | Outer aggregations — `field` must reference an **inner metric alias** |
| `limit` | integer | no | `1` | Outer LIMIT — minimum 1 |
| `offset` | integer | no | `0` | Outer OFFSET |

!!! warning "Rollup `field` references inner aliases, not columns"
    In the rollup `metrics` array, `"field"` must be the **alias** of a metric in the inner plan (e.g. `"revenue"`), not a raw column name. The compiler will raise a `QueryPlanError` otherwise.

### When to use Rollup

Use rollup when the question asks for an aggregate *of an aggregate*:

| Question pattern | Needs rollup? |
|---|---|
| "total revenue" | no — single-level `sum` |
| "revenue per customer" | no — `sum` grouped by `customer_id` |
| "average revenue per customer" | **yes** — `avg(sum(revenue) per customer)` |
| "standard deviation of orders per day" | **yes** |
| "top customer by number of orders" | no — `count` grouped by `customer_id` + `ORDER BY` + `LIMIT` |

---

## Column Qualification

When a plan joins multiple tables, column references must be qualified to avoid ambiguity:

```json
{"field": "orders.freight"}        // qualified
{"field": "freight"}               // unqualified — only valid if unambiguous
```

Use the logical table name, not the physical `db_table` name.

If an unqualified column exists in multiple joined tables, the compiler raises `AmbiguousColumnError`.

---

## Examples

### Scalar aggregate

```json
{
  "dataset": "orders",
  "metrics": [{"agg": "count", "field": "*", "alias": "total_orders"}],
  "limit": 1
}
```

### Filtered list

```json
{
  "dataset": "products",
  "filters": [
    {"field": "discontinued", "op": "=", "value": false},
    {"field": "unit_price", "op": ">=", "value": 20}
  ],
  "dimensions": [
    {"field": "product_name"},
    {"field": "unit_price"}
  ],
  "order_by": [{"by": "unit_price", "dir": "desc"}],
  "limit": 20
}
```

### Group-by with multiple metrics

```json
{
  "dataset": "orders",
  "dimensions": [{"field": "ship_country", "alias": "country"}],
  "metrics": [
    {"agg": "count", "field": "*", "alias": "n_orders"},
    {"agg": "sum", "field": "freight", "alias": "total_freight"},
    {"agg": "avg", "field": "freight", "alias": "avg_freight"}
  ],
  "order_by": [{"by": "n_orders", "dir": "desc"}],
  "limit": 10
}
```

### Cross-table query (auto join)

```json
{
  "dataset": "orders",
  "filters": [{"field": "customers.country", "op": "=", "value": "UK"}],
  "dimensions": [
    {"field": "customers.company_name", "alias": "customer"},
    {"field": "orders.order_date"}
  ],
  "metrics": [{"agg": "sum", "field": "orders.freight", "alias": "freight"}],
  "order_by": [{"by": "freight", "dir": "desc"}],
  "limit": 25
}
```

`auto_inject_joins` will resolve the `orders` → `customers` join automatically.

### Relative date filter

```json
{
  "dataset": "orders",
  "filters": [
    {
      "field": "order_date",
      "op": ">=",
      "value": {"$relative_date": {"op": "now_minus_days", "days": 90}}
    }
  ],
  "metrics": [{"agg": "count", "field": "*", "alias": "recent_orders"}],
  "limit": 1
}
```
