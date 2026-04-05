# QCE — Query Compiler Engine

Deterministic, schema-validated JSON → SQL for Postgres.  
Instead of letting an LLM generate free-form SQL, the LLM extracts a lightweight **QueryIntent**, and QCE deterministically compiles it into parameterized SQL and executes it safely.

## Quick Start

```python
from dsl_compiler.agent import QueryAgent

agent = QueryAgent(
    engine=your_sqlalchemy_engine,
    schema_path="path/to/schema.yaml",
    spec_path="path/to/queryplan_spec.yaml",
    llm=your_llm_instance,
)

result = agent.ask("how many plumbing issues last year?")
```

## Documentation

Full documentation, benchmarks, and guides are in the [qce_docs](https://github.com/Certifore/qce_docs) repository.

## Install

```bash
pip install qce @ git+ssh://git@github.com/Certifore/dsl_compiler.git
```
