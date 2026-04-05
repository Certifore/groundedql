# IntentQL

Intent-driven, deterministic natural language to SQL for Postgres.  
Instead of letting an LLM generate free-form SQL, the LLM extracts a lightweight **QueryIntent**, and IntentQL deterministically compiles it into parameterized SQL and executes it safely.

## Install

```bash
pip install intentql
```

With optional few-shot memory (recommended for production):

```bash
pip install "intentql[memory]"
```

<details>
<summary>Install from source</summary>

```bash
git clone https://github.com/Certifore/intentql
cd intentql
pip install -e ".[dev]"
```
</details>

## Quick Start

### 1. Generate your schema from the database

```bash
intentql init --db "postgresql://user:pass@host/db"
# → config/schema.yaml  (tables, columns, types, PKs, links — all auto-detected)
```

### 2. Enrich with LLM-generated descriptions (optional, recommended)

```bash
export LLM_API_KEY=sk-...   # works with any OpenAI-compatible provider
intentql describe --schema config/schema.yaml --db "postgresql://user:pass@host/db"
# → Adds table + column descriptions using sample data for context
```

### 3. Ask questions

```python
from sqlalchemy import create_engine
from openai import OpenAI
from intentql.agent import QueryAgent

engine = create_engine("postgresql+psycopg2://user:pass@host/db")

agent = QueryAgent(
    engine=engine,
    schema_path="config/schema.yaml",
    llm=OpenAI(api_key="sk-..."),
)

result = agent.ask("how many plumbing issues last year?")
print(result["rows"])
print(result["sql"])
```

## CLI Reference

| Command | Description |
|---|---|
| `intentql init --db URL` | Introspect Postgres and generate `schema.yaml` |
| `intentql describe --schema PATH --db URL` | Enrich schema with LLM-generated descriptions |

Run `intentql --help` for full options.

## Documentation

Full documentation, benchmarks, and guides are in the [intentql_docs](https://github.com/Certifore/intentql_docs) repository.
