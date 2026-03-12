# QCE Benchmarks

Three compiler benchmarks that demonstrate QCE's core safety and reliability properties.
No LLM required. No database required. Pure compiler tests.

## Run All Benchmarks

```bash
cd /home/alexander/git_repos/dsl_compiler
python3 benchmark/run_benchmarks.py
```

Results are printed to the terminal and written to `benchmark/results/`.

## What Each Benchmark Tests

### Benchmark 1 — Injection Resistance
Attempts 50 adversarial inputs across two categories:

**Schema-based attacks (46 inputs):** attempts to access tables or columns not in
`schema.yaml` (e.g. `pg_shadow`, `passwords`, `email`). QCE must reject every one
with a `QueryPlanError` or `SchemaError` before any SQL is generated.

**Value-based injection attempts (4 inputs):** SQL injection strings passed as filter
values (e.g. `'; DROP TABLE assets; --`). QCE correctly compiles these — the value
is passed as a SQLAlchemy `bindparam` and reaches Postgres as a plain string, never
as SQL. This is the correct behavior. Parameterization neutralizes all value-based
injection attempts by design.

**Expected result:** 50/50 neutralized (46 rejected at compile time + 4 neutralized by parameterization)

### Benchmark 2 — Determinism
Compiles 20 valid QueryPlans 10 times each and compares the SQL output across runs.
QCE must produce byte-for-byte identical SQL every time for the same input.

**Expected result:** 20/20 questions produce identical SQL across all 10 runs

### Benchmark 3 — Hallucination Rejection
Attempts 30 QueryPlans that reference columns or tables that do not exist in the schema.
QCE must reject every one with `UNKNOWN_COLUMN` or `UNKNOWN_DATASET` error codes.

**Expected result:** 30/30 hallucinations rejected cleanly

## Setup

No `.env` or DB connection needed. The benchmarks use the schema at
`config/schema.yaml` and compile plans without executing them.

```bash
pip install -e /home/alexander/git_repos/dsl_compiler
```
