# IntentQL

Intent-driven, deterministic natural language to SQL for Postgres.  
Instead of letting an LLM generate free-form SQL, the LLM extracts a lightweight **QueryIntent**, and IntentQL deterministically compiles it into parameterized SQL and executes it safely.

## Quick Start

```python
from intentql.agent import QueryAgent

agent = QueryAgent(
    engine=your_sqlalchemy_engine,
    schema_path="path/to/schema.yaml",
    spec_path="path/to/queryplan_spec.yaml",
    llm=your_llm_instance,
)

result = agent.ask("how many plumbing issues last year?")
```

## Documentation

Full documentation, benchmarks, and guides are in the [intentql_docs](https://github.com/Certifore/intentql_docs) repository.

## Install

```bash
pip install intentql @ git+ssh://git@github.com/Certifore/intentql.git
```
