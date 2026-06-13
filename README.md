# GroundedQL

![GroundedQL - Reliable natural-language analytics through deterministic SQL compilation](assets/groundedql-social-preview.png)

Intent-driven, deterministic natural language to SQL for Postgres.  
Instead of letting an LLM generate free-form SQL, the LLM extracts a lightweight **QueryIntent**, and GroundedQL deterministically compiles it into parameterized SQL and executes it safely.

## Install

```bash
pip install groundedql
```

To use the OpenAI SDK adapter shown below:

```bash
pip install "groundedql[openai]"
```

With optional few-shot memory (recommended for production):

```bash
pip install "groundedql[memory]"
```

<details>
<summary>Install from source</summary>

```bash
git clone https://github.com/Certifore/groundedql
cd groundedql
pip install -e ".[dev]"
```
</details>

## Quick Start

### 1. Generate your schema from the database

```bash
groundedql init --db "postgresql://user:pass@host/db"
# → config/schema.yaml  (tables, columns, types, PKs, links — all auto-detected)
```

### 2. Enrich with LLM-generated descriptions (optional, recommended)

```bash
export LLM_API_KEY=sk-...   # works with any OpenAI-compatible provider
groundedql describe --schema config/schema.yaml --db "postgresql://user:pass@host/db"
# → Adds table + column descriptions using sample data for context
```

### 3. Ask questions

```python
from sqlalchemy import create_engine
from openai import OpenAI
from groundedql.agent import QueryAgent

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

### Use Mistral instead of OpenAI

```bash
export MISTRAL_AI=...
# optional
export MISTRAL_MODEL=mistral-small-latest
```

```python
agent = QueryAgent(
    engine=engine,
    schema_path="config/schema.yaml",
    llm="mistral",  # or "mistral:mistral-small-latest"
)
```

### Use a local Ollama model

```bash
export OLLAMA_MODEL=groundedql-gemma4
# optional
export OLLAMA_BASE_URL=http://127.0.0.1:11434
export OLLAMA_NUM_CTX=8192
```

```python
agent = QueryAgent(
    engine=engine,
    schema_path="config/schema.yaml",
    llm="ollama",  # or "ollama:groundedql-gemma4"
)
```

## CLI Reference

| Command | Description |
|---|---|
| `groundedql init --db URL` | Introspect Postgres and generate `schema.yaml` |
| `groundedql describe --schema PATH --db URL` | Enrich schema with LLM-generated descriptions |

Run `groundedql --help` for full options.

## Documentation

Full documentation, benchmarks, and guides are at [certifore.github.io/groundedql_docs](https://certifore.github.io/groundedql_docs) ([source](https://github.com/Certifore/groundedql_docs)).

See [ROADMAP.md](ROADMAP.md) for planned capabilities (lookup, trends, ratios, multi-step NL, and expressiveness goals).

Existing users of the previous package name should read [MIGRATION.md](MIGRATION.md).

## Contributing

Contributions are welcome through pull requests. Please read
[CONTRIBUTING.md](CONTRIBUTING.md) before submitting changes. The lead maintainer reviews
and merges all changes into the official repository and is the only person who publishes
official releases.

## License

GroundedQL is licensed under the [Apache License 2.0](LICENSE).
