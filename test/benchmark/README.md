# IntentQL BIRD Mini-Dev Benchmark

This folder is for evaluating IntentQL against BIRD-SQL Mini-Dev.

BIRD Mini-Dev provides:

- natural-language `question`
- optional `evidence`
- `db_id`
- gold `SQL`
- database files / setup scripts

The benchmark runner executes the gold SQL to produce the expected answer, asks
IntentQL the natural-language question, executes IntentQL's SQL, and compares the
two result tables.

## Expected Layout

Download BIRD Mini-Dev separately. Do not commit the dataset here.

```text
test/benchmark/bird_minidev/
  minidev/
    MINIDEV/
      mini_dev_postgresql.json
      dev_tables.json
      dev_databases/
        ...
    MINIDEV_postgresql/
      BIRD_dev.sql
```

IntentQL also needs one `schema.yaml` per BIRD `db_id`:

```text
test/benchmark/schemas/
  financial/
    schema.yaml
  superhero/
    schema.yaml
```

or:

```text
test/benchmark/schemas/
  financial.yaml
  superhero.yaml
```

Generate these once the BIRD Postgres dump is loaded:

```bash
python test/benchmark/generate_bird_schemas.py
```

Keep the generated schemas if you want reproducible benchmark runs.

## Local Postgres Setup

One simple local setup is a dedicated Docker Postgres container:

```bash
docker run --name intentql-bird-pg \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=BIRD \
  -p 55432:5432 \
  -d postgres:14
```

The official dump expects a role named `xiaolongli`:

```bash
psql postgresql://postgres:postgres@localhost:55432/postgres \
  -c "CREATE ROLE xiaolongli;"
```

Import the Postgres dump:

```bash
psql postgresql://postgres:postgres@localhost:55432/BIRD \
  -v ON_ERROR_STOP=1 \
  -f test/benchmark/bird_minidev/minidev/MINIDEV_postgresql/BIRD_dev.sql
```

## Environment

At minimum:

```env
MISTRAL_AI=...
BIRD_DB_URL=postgresql+psycopg2://postgres:postgres@localhost:55432/BIRD
BIRD_MINIDEV_ROOT=test/benchmark/bird_minidev
```

Optional:

```env
MISTRAL_MODEL=mistral-small-latest
OLLAMA_MODEL=intentql-gemma4
OLLAMA_TIMEOUT=240
OLLAMA_NUM_CTX=8192
```

If each BIRD `db_id` is in a separate Postgres database, use a template:

```env
BIRD_DB_URL_TEMPLATE=postgresql+psycopg2://user:pass@host:5432/{db_id}
```

## Dry Run

Validate that the dataset can be found and count examples without calling an LLM:

```bash
python test/benchmark/bird_minidev.py --dry-run --limit 10
```

## Run

```bash
python test/benchmark/bird_minidev.py --limit 25 --llm mistral
```

To run through local Ollama/Gemma instead:

```bash
python test/benchmark/bird_minidev.py --limit 25 --llm ollama
```

or with an explicit Ollama model name:

```bash
python test/benchmark/bird_minidev.py --limit 25 --llm ollama:intentql-gemma4
```

Results are written to:

```text
test/benchmark/results/bird_minidev_latest.json
```

## Notes

- The runner only executes gold SQL that starts with `SELECT` or `WITH` by default.
- The default comparison ignores row order and compares row values, not SQL text.
- Unsupported cases are expected early on. The important report is:
  attempted, correct, wrong, skipped, and failure categories.
