# Contributing

Thanks for your interest in contributing to QCE!

!!! tip "New to the codebase?"
    Read [Core Concepts](concepts.md) first to understand the pipeline, then come back here. The best first contribution is usually a new operator or aggregation — both require only a few targeted edits.

---

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Certifore/dsl_compiler
cd dsl_compiler
```

### 2. Create a Virtual Environment

```bash
python3 -m venv env_dsl
source env_dsl/bin/activate   # Linux/macOS
# or: env_dsl\Scripts\activate  # Windows
```

### 3. Install in Editable Mode

```bash
pip install -e ".[dev]"
```

This installs `dsl_compiler`/`qce` in editable mode (source changes are reflected immediately), plus `ruff`, `build`, and `twine`.

### 4. Install LLM Extras (optional)

```bash
pip install -e ".[dev,openai]"          # OpenAI SDK
pip install langchain-google-genai      # Gemini via LangChain
pip install langchain-groq              # Groq via LangChain
pip install langchain-ollama            # Ollama (local models)
```

---

## Project Structure

```
dsl_compiler/
  __init__.py           # public API — everything exported from here
  compiler.py           # core SQL compiler: logical → physical, values → bind params
  executor.py           # executes compiled SQL against Postgres
  planner.py            # LLM → QueryPlan orchestration + retry loop
  validation.py         # three-layer validation: Pydantic + allowlist + semantic
  join_planner.py       # BFS auto-join injection
  semantic_lint.py      # question-aware lint checks (5 rules)
  spec_builder.py       # auto-generates LLM prompt spec from schema.yaml
  queryplan_models.py   # Pydantic models for the QueryPlan DSL
  llm_client.py         # LLMClient protocol + CallableLLMClient
  llm_adapters.py       # OpenAI and LangChain adapters + make_llm_client() factory
  exceptions.py         # structured exception hierarchy
  agent.py              # high-level QueryAgent (planner + executor combined)
  api/
    api.py              # execute_query_plan, validate_query_plan, load_and_validate_schema
    spec_api.py         # get_queryplan_instructions

config/
  schema.yaml                     # demo Northwind schema
  queryplan_spec.yaml             # hand-authored reference spec
  queryplan_spec_generated.yaml   # auto-generated — regenerate after schema changes

test/
  test_main.py          # unit and integration tests
  regression_test/      # 28-question Northwind regression suite

benchmark/              # benchmarks: QCE vs LangChain vs GPT-4 Direct
docs/                   # this documentation site (MkDocs)
```

---

## Running Tests

All automated checks go through `test/test_main.py` (no separate pytest modules).

**Compiler + validation (no DB):**

```bash
python test/test_main.py selfcheck
```

**Regression suite** — executes plans from `test/regression_test/test_qs.json` against Postgres:

```bash
cp test/.env.example test/.env
# fill in: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
```

```bash
python test/test_main.py run      # execute and print results
python test/test_main.py check    # compare to saved baseline (CI)
```

---

## Linting and Formatting

QCE uses [Ruff](https://docs.astral.sh/ruff/):

```bash
ruff check dsl_compiler/         # check for issues
ruff check --fix dsl_compiler/   # auto-fix
ruff format dsl_compiler/        # format
```

All configuration lives in `pyproject.toml`.

---

## Regenerating the Spec

After any change to `schema.yaml`:

```bash
python -m dsl_compiler.spec_builder \
    --schema config/schema.yaml \
    --output config/queryplan_spec_generated.yaml
```

---

## Common Extension Patterns

=== "New Operator"

    1. Add the literal to `Operator` in `queryplan_models.py`
    2. Handle it in `Compiler._compile_bool_expr` in `compiler.py`
    3. Add it to `_operators_block()` in `spec_builder.py`
    4. Add a test case in `test/test_main.py`

=== "New Aggregation"

    1. Add the literal to `Agg` in `queryplan_models.py`
    2. Handle it in `Compiler._build_select_query` in `compiler.py`
    3. Add a test case in `test/test_main.py`

=== "New Semantic Lint Rule"

    1. Add a `_check_*` function in `semantic_lint.py`
    2. Call it from `semantic_lint()`
    3. Add a test that provides a question with the pattern and a plan that violates the rule

---

## Submitting a Pull Request

1. **Fork** and create a feature branch:

    ```bash
    git checkout -b feat/my-feature
    ```

2. **Write tests** for your changes — all PRs require coverage for new behaviour.

3. **Run checks:**

    ```bash
    ruff check dsl_compiler/ && python test/test_main.py selfcheck
    ```

4. **Commit** with a clear message:

    ```bash
    git commit -m "feat: add support for BETWEEN operator in compiler"
    ```

5. **Open a PR** against `main`. Describe what the change does and why.

---

## Building the Package

```bash
python -m build
# output: dist/
```

## Publishing to PyPI

```bash
python -m twine upload dist/*
```

Requires a PyPI API token set as `TWINE_PASSWORD` or in `~/.pypirc`.

---

## Docs Locally

```bash
pip install -r requirements-docs.txt
mkdocs serve --dev-addr 127.0.0.1:8001
```

Open [http://127.0.0.1:8001](http://127.0.0.1:8001) to preview.

```bash
mkdocs build   # static HTML → site/
```

---

## Code Style Guidelines

- Type annotations on all public functions and constructors
- Keep functions focused — the compiler's `_compile_*` methods are a good model
- Raise typed exceptions (`QueryPlanError`, `SchemaError`, …) — never bare `Exception` from public paths
- All filter values must go through bind parameters — never `str.format()` or f-strings into SQL
- Tests should be self-contained and not require a live DB unless they're in `regression_test/`
