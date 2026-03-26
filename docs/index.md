<div class="qce-home" markdown>

<div class="qce-hero" markdown>

<div class="qce-hero__eyebrow">⚡ Query Compiler Engine</div>

<h1 class="qce-hero__title">LLM-powered queries.<br>Deterministic SQL. Zero trust.</h1>

<p class="qce-hero__subtitle">
QCE compiles structured <strong>QueryPlan JSON</strong> into parameterized Postgres SQL —
enforcing your schema allowlist, neutralizing injections, and guaranteeing identical
output on every run. Let your LLM own <em>intent</em>. Let QCE own <em>SQL</em>.
</p>

<div class="qce-hero__actions">
  <a href="getting-started/" class="qce-btn qce-btn--primary">→ Get Started</a>
  <a href="concepts/" class="qce-btn qce-btn--outline">How it works</a>
  <a href="https://github.com/alexanderabakah/dsl_compiler" class="qce-btn qce-btn--outline">GitHub</a>
</div>

<div class="qce-badges">
  <span class="qce-badge qce-badge--purple">Python ≥ 3.10</span>
  <span class="qce-badge qce-badge--blue">Postgres</span>
  <span class="qce-badge qce-badge--green">MIT License</span>
  <span class="qce-badge qce-badge--gray">SQLAlchemy 2</span>
  <span class="qce-badge qce-badge--green">v0.1.0</span>
</div>

</div>

---

## The Problem with LLM-Generated SQL

Every production AI data feature eventually hits the same three walls:

<div class="qce-comparison" markdown>

|  | Raw LLM SQL | **QCE** |
|---|:---:|:---:|
| SQL injection via prompt | <span class="cross">✗ Never safe</span> | <span class="check">✓ Bind params always</span> |
| Hallucinated table names | <span class="cross">✗ Silent wrong answer</span> | <span class="check">✓ Hard error, allowlist enforced</span> |
| Non-deterministic output | <span class="cross">✗ Varies per call</span> | <span class="check">✓ Same plan → same SQL</span> |
| LLM picks JOIN strategy | <span class="cross">✗ Unpredictable</span> | <span class="check">✓ BFS shortest path, always</span> |
| Needs DB introspection | <span class="cross">✗ Exposes full schema</span> | <span class="check">✓ LLM sees logical names only</span> |

</div>

QCE solves this by **separating intent from execution**. The LLM produces a structured
`QueryPlan` JSON — no SQL, no table names, no dialect concerns. The compiler takes that
plan and produces correct, safe, parameterized SQL every time.

---

## How It Works

```
  User question
       │
       ▼
  ┌──────────────────────────────────────────────┐
  │  LLM  (OpenAI · Gemini · Groq · any model)  │
  │  Produces: QueryPlan JSON                    │  ← structured intent, not SQL
  └──────────────────────────────────────────────┘
       │
       ▼  { "dataset": "orders", "filters": [...], "metrics": [...] }
  ┌──────────────────────────────────────────────┐
  │  QCE Compiler                                │
  │  · Validates against schema.yaml allowlist   │
  │  · All values → bind parameters              │
  │  · Auto-injects JOIN paths via BFS           │
  │  · Resolves $relative_date sentinels         │
  └──────────────────────────────────────────────┘
       │
       ▼  SELECT count(*) AS n FROM orders WHERE ship_country = %(p0)s
  ┌──────────────────────────────────────────────┐
  │  Postgres                                    │
  └──────────────────────────────────────────────┘
       │
       ▼  { "rows": [...], "row_count": 122, "sql": "...", "params": {...} }
```

---

## Quick Start

=== "With an LLM"

    ```python
    from sqlalchemy import create_engine
    from openai import OpenAI
    import dsl_compiler as qce

    engine = create_engine("postgresql+psycopg2://user:pass@host/db")

    agent = qce.QueryAgent(
        engine=engine,
        schema_path="config/schema.yaml",
        spec_path="config/queryplan_spec_generated.yaml",
        llm=OpenAI(api_key="sk-..."),
    )

    result = agent.ask("Top 10 customers by total order value last 90 days")
    print(result["rows"])
    print(result["sql"])  # always inspect the generated SQL
    ```

=== "Hand-written plan"

    ```python
    from sqlalchemy import create_engine
    import dsl_compiler as qce

    engine = create_engine("postgresql+psycopg2://user:pass@host/db")

    result = qce.execute_query_plan(
        engine=engine,
        schema_path="config/schema.yaml",
        query_plan={
            "dataset": "orders",
            "filters": [
                {"field": "ship_country", "op": "=", "value": "Germany"},
                {
                    "field": "order_date",
                    "op": ">=",
                    "value": {"$relative_date": {"op": "now_minus_days", "days": 30}},
                },
            ],
            "metrics": [{"agg": "sum", "field": "freight", "alias": "total_freight"}],
            "limit": 1,
        },
    )
    print(result["rows"])   # [{"total_freight": 3847.25}]
    print(result["sql"])    # SELECT sum(freight) AS total_freight FROM orders
                            # WHERE ship_country = %(p0)s AND order_date >= %(p1)s
                            # LIMIT 1
    ```

=== "Validate offline"

    ```python
    import dsl_compiler as qce

    errors = qce.validate_query_plan(
        query_plan=plan,
        schema_path="config/schema.yaml",
    )

    if errors:
        for e in errors:
            print(e)   # "$.filters[0].field: unknown column 'revenue' on table 'orders'"
    else:
        print("✓ Plan is valid")
    ```

---

## Feature Highlights

<div class="qce-features" markdown>
<div class="qce-feature" markdown>
<div class="qce-feature__icon">🔒</div>
<p class="qce-feature__title">Injection-Proof by Design</p>
<p class="qce-feature__desc">Every filter value becomes a named bind parameter. There is no code path that interpolates user input into SQL text — ever.</p>
</div>
<div class="qce-feature" markdown>
<div class="qce-feature__icon">📋</div>
<p class="qce-feature__title">Schema Allowlist Enforcement</p>
<p class="qce-feature__desc">Only tables and columns declared in <code>schema.yaml</code> are reachable. Unknown names raise a hard <code>QueryPlanError</code> — no silent degradation.</p>
</div>
<div class="qce-feature" markdown>
<div class="qce-feature__icon">⚙️</div>
<p class="qce-feature__title">Deterministic Output</p>
<p class="qce-feature__desc">The compiler is a pure function. Same <code>QueryPlan</code> → same SQL every time. Cache by plan hash, regression-test against a corpus.</p>
</div>
<div class="qce-feature" markdown>
<div class="qce-feature__icon">🔗</div>
<p class="qce-feature__title">Auto Join Injection</p>
<p class="qce-feature__desc">Reference columns from multiple tables. QCE runs BFS over your schema's link graph and injects the shortest join path automatically.</p>
</div>
<div class="qce-feature" markdown>
<div class="qce-feature__icon">🔄</div>
<p class="qce-feature__title">LLM Retry Loop</p>
<p class="qce-feature__desc">Validation errors are structured and fed back to the LLM as focused correction prompts — not stack traces. Models self-correct reliably.</p>
</div>
<div class="qce-feature" markdown>
<div class="qce-feature__icon">🌐</div>
<p class="qce-feature__title">LLM-Agnostic</p>
<p class="qce-feature__desc">One factory function — <code>make_llm_client()</code> — adapts OpenAI, LangChain, Gemini, Groq, Ollama, or any callable automatically.</p>
</div>
</div>

---

## Install

```bash
pip install qce
```

```bash
# With LLM extras
pip install "qce[openai]"    # + openai SDK
pip install "qce[google]"    # + google-generativeai
```

!!! tip "Installing from source"
    ```bash
    git clone https://github.com/alexanderabakah/dsl_compiler
    cd dsl_compiler
    pip install -e ".[dev]"
    ```

---

## Next Steps

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Get Started in 5 Minutes**

    ---

    Install QCE, define your schema, and run your first query against a live database.

    [:octicons-arrow-right-24: Getting Started](getting-started.md)

-   :material-lightbulb-on:{ .lg .middle } **Understand the Architecture**

    ---

    Learn the compilation pipeline, the security model, and how determinism is guaranteed.

    [:octicons-arrow-right-24: Core Concepts](concepts.md)

-   :material-code-json:{ .lg .middle } **QueryPlan Reference**

    ---

    All operators, aggregations, rollup, relative dates, and cross-table query examples.

    [:octicons-arrow-right-24: QueryPlan Reference](query-plan-reference.md)

-   :material-robot:{ .lg .middle } **Connect Your LLM**

    ---

    OpenAI, LangChain, Gemini, Groq, Ollama — or any callable function.

    [:octicons-arrow-right-24: LLM Integration](llm-integration.md)

</div>

</div>
