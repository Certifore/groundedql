# IntentQL

Guided natural language to **Postgres SQL**: an LLM proposes a read-only query using **schema.yaml** as the single source of truth; **sqlglot** validates identifiers against that schema; then the query runs with a statement timeout.

There is no QueryPlan compiler or JSON plan layer on this branch — only guided SQL + validation + execution.

## Install

```bash
pip install intentql
pip install "intentql[guided]"   # sqlglot (required for validation)
pip install "intentql[memory]"    # optional: ChromaDB few-shot memory
```

## Quick start

```python
from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI
from intentql.agent import QueryAgent

engine = create_engine("postgresql+psycopg2://user:pass@host/db")

agent = QueryAgent(
    engine=engine,
    schema_path="config/schema.yaml",
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
)

out = agent.ask("How many customers are in London?")
print(out["rows"])
print(out.get("sql"))
```

## CLI

```bash
intentql init --db "postgresql://..."   # writes config/schema.yaml
intentql describe --schema config/schema.yaml  # LLM descriptions (needs API key)
```

## Tests

```bash
pip install -e ".[guided]"
python test/test_main.py
```

## Documentation

See the [intentql_docs](https://github.com/Certifore/intentql_docs) repository for broader context (some pages may describe the legacy QueryPlan stack — this branch does not include it).
