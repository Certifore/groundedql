"""
Regression test runner for dsl_compiler.

Modes (set via TEST_MODE env var or command-line arg):
  run     — Execute all plans, print results. Does NOT write baseline. (default)
  update  — Execute all plans and OVERWRITE the baseline file.
  check   — Execute all plans and COMPARE against the saved baseline.
             Exits with code 1 if any row_count or first-row values differ.
  pipeline — Benchmark 4 only: NL → LLM planner → compile → execute (see benchmark/.env).
  selfcheck — No DB: compiler + schema + validation checks (former pytest cases).

Usage:
  python test/test_main.py              # run (default)
  python test/test_main.py update       # overwrite baseline
  python test/test_main.py check        # regression check (for CI)
  python test/test_main.py pipeline     # full pipeline benchmark (loads benchmark/.env)
  python test/test_main.py selfcheck    # compiler hardening + validate_query_plan meta

Test types in test_qs.json:
  (default) db   — execute against Postgres
  lint           — semantic_lint only (no DB)
  canonical      — structural plan_fingerprint / canonicalize checks (no DB)
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
import yaml

from dsl_compiler import execute_query_plan
from dsl_compiler.join_planner import auto_inject_joins
from dsl_compiler.plan_canonical import canonicalize_query_plan, plan_fingerprint
from dsl_compiler.semantic_lint import semantic_lint

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

ENV_PATH = ROOT / ".env"
SCHEMA_PATH = ROOT / "config" / "schema.yaml"
REG_DIR = HERE / "regression_test"
SUITE_PATH = REG_DIR / "test_qs.json"
BASELINE_PATH = REG_DIR / "suite_results.json"

REG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Mode
# ---------------------------------------------------------------------------
MODE = (sys.argv[1] if len(sys.argv) > 1 else os.getenv("TEST_MODE", "run")).lower()
assert MODE in {"run", "update", "check", "lint", "pipeline", "selfcheck"}, (
    f"Unknown mode '{MODE}'. Use: run | update | check | lint | pipeline | selfcheck"
)


def _run_pipeline_mode() -> None:
    """
    Benchmark 4 — NL → planner → compile → execute.
    Loads repo .env then benchmark/.env; writes benchmark/results/pipeline_latest.json.
    """
    import json as _json
    from datetime import datetime, timezone

    load_dotenv(dotenv_path=ENV_PATH, override=True)
    load_dotenv(dotenv_path=ROOT / "benchmark" / ".env", override=True)
    sys.path.insert(0, str(ROOT / "benchmark" / "compare"))
    from run_comparison import (
        _check_env,
        _check_env_pipeline_flexible,
        _db_url,
        _print_table,
        _print_token_table,
        bench_pipeline_qce,
        bench_pipeline_gpt4,
        bench_pipeline_langchain,
        make_client,
        make_agent,
        RESULTS_DIR,
        DATA_DIR,
        SCHEMA_PATH,
        SPEC_PATH,
    )

    def _print_first_error_hint(pipe: list) -> None:
        for r in pipe:
            for d in r.get("details") or []:
                err = d.get("error")
                if not err:
                    continue
                qid = d.get("id", "?")
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    print(
                        f"\n  First failure (question {qid}): Gemini API quota/rate limit (429). "
                        "See https://ai.google.dev/gemini-api/docs/rate-limits — full text in results JSON."
                    )
                else:
                    line = err.strip().split("\n")[0]
                    if len(line) > 200:
                        line = line[:200] + "…"
                    print(f"\n  First failure (question {qid}): {line}")
                return

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

    if openai_key:
        _check_env()
    else:
        _check_env_pipeline_flexible()

    print("=" * 75)
    print("  Pipeline benchmark (Benchmark 4)")
    print(f"  Spec: {SPEC_PATH}")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 75)

    with open(SCHEMA_PATH) as f:
        schema = yaml.safe_load(f)
    with open(DATA_DIR / "pipeline_questions.json") as f:
        pipeline_questions = _json.load(f)

    db_url = _db_url()

    if not openai_key and gemini_key:
        print("\n[setup] OPENAI_API_KEY not set — running QCE full pipeline with Gemini only.")
        print(f"\n[1/1] QCE full pipeline ({len(pipeline_questions)} questions)...")
        qce = bench_pipeline_qce(schema, pipeline_questions, db_url)
        pipe = [qce]
    else:
        print("\n[setup] Initialising competitors...")
        gpt4_client = make_client(openai_key)
        langchain_agent = make_agent(db_url, openai_key)

        print(f"\n[1/3] QCE full pipeline ({len(pipeline_questions)} questions)...")
        qce = bench_pipeline_qce(schema, pipeline_questions, db_url)

        print(f"\n[2/3] LangChain full pipeline ({len(pipeline_questions)} questions)...")
        lc = bench_pipeline_langchain(langchain_agent, pipeline_questions)

        print(f"\n[3/3] GPT-4 Direct full pipeline ({len(pipeline_questions)} questions)...")
        gpt4 = bench_pipeline_gpt4(gpt4_client, schema, pipeline_questions, db_url)

        pipe = [qce, lc, gpt4]

    _print_table(f"Benchmark 4 — Full Pipeline ({len(pipeline_questions)} questions)", pipe)
    _print_token_table(pipe)

    if pipe and pipe[0].get("correct", -1) == 0 and pipe[0].get("total", 0) > 0:
        _print_first_error_hint(pipe)

    out_path = RESULTS_DIR / "pipeline_latest.json"
    out_path.write_text(
        _json.dumps(
            {
                "run_at": datetime.now(timezone.utc).isoformat(),
                "spec_used": str(SPEC_PATH),
                "results": pipe,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\n  Results saved → {out_path}")

    qce = pipe[0]
    ok = qce.get("correct", 0) == qce.get("total", 0) and qce.get("total", 0) > 0
    sys.exit(0 if ok else 1)


def _run_selfcheck() -> None:
    """Compiler/schema/validation checks (no DB). Replaces former test_compiler_fixes + test_validation_meta."""
    import tempfile
    from pathlib import Path as P

    from dsl_compiler.compiler import Compiler, QueryPlanError
    from dsl_compiler.exceptions import SchemaError
    from dsl_compiler.schema_validator import validate_schema
    from dsl_compiler.validation import validate_query_plan_dict

    failures: list[str] = []

    def _minimal_schema(*, bad_link_join_type: str | None = None) -> dict:
        schema = {
            "tables": [
                {
                    "name": "orders",
                    "db_table": "orders",
                    "columns": [
                        {"name": "order_id", "db_column": "order_id", "type": "int"},
                        {"name": "customer_id", "db_column": "customer_id", "type": "int"},
                    ],
                },
                {
                    "name": "customers",
                    "db_table": "customers",
                    "columns": [
                        {"name": "customer_id", "db_column": "customer_id", "type": "int"},
                        {"name": "name", "db_column": "name", "type": "varchar"},
                    ],
                },
            ],
            "links": [
                {
                    "name": "orders_to_customers",
                    "from_table": "orders",
                    "to_table": "customers",
                    "join_type": "left",
                    "on": [{"left": "orders.customer_id", "right": "customers.customer_id"}],
                }
            ],
        }
        if bad_link_join_type is not None:
            schema["links"][0]["join_type"] = bad_link_join_type
        return schema

    # --- validate_query_plan_dict ignores planner meta ---
    with tempfile.TemporaryDirectory() as td:
        sp = P(td) / "schema.yaml"
        sp.write_text(
            yaml.safe_dump(
                {
                    "tables": [
                        {
                            "name": "work_orders",
                            "columns": [{"name": "asset_tag"}, {"name": "work_order_id"}],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        plan = {
            "version": "1.0",
            "dataset": "work_orders",
            "dimensions": [{"field": "asset_tag", "alias": "asset_tag"}],
            "metrics": [{"agg": "count", "field": "*", "alias": "work_order_count"}],
            "order_by": [{"by": "work_order_count", "dir": "desc"}],
            "limit": 1,
            "offset": 0,
            "filters": [],
            "meta": {
                "plan_hash": "deadbeef",
                "retry_count": 1,
                "auto_fixes_applied": [],
                "validation_errors": [],
                "lint_errors": [],
            },
        }
        parsed, errs = validate_query_plan_dict(plan, str(sp))
        if parsed is None or errs:
            failures.append(f"validate_query_plan_dict+meta: {errs}")

    # --- schema validator: bad link join type ---
    try:
        validate_schema(_minimal_schema(bad_link_join_type="right"))
        failures.append("validate_schema should reject bad join_type")
    except SchemaError as e:
        if "join_type" not in str(e).lower():
            failures.append(f"validate_schema wrong message: {e}")
    except Exception as e:
        failures.append(f"validate_schema wrong exc: {e}")

    # --- Compiler rejects bad link without prior validate_schema ---
    try:
        Compiler(_minimal_schema(bad_link_join_type="full"))
        failures.append("Compiler should reject bad join_type at init")
    except SchemaError as e:
        if "join_type" not in str(e).lower():
            failures.append(f"Compiler schema wrong message: {e}")
    except Exception as e:
        failures.append(f"Compiler schema wrong exc: {e}")

    # --- duplicate explicit join requires as ---
    try:
        c = Compiler(_minimal_schema())
        c.compile(
            {
                "dataset": "orders",
                "joins": [
                    {
                        "dataset": "customers",
                        "type": "inner",
                        "on": {
                            "cmp": {
                                "left": {"col": "orders.customer_id"},
                                "op": "=",
                                "right": {"col": "customers.customer_id"},
                            }
                        },
                    },
                    {
                        "dataset": "customers",
                        "type": "inner",
                        "on": {
                            "cmp": {
                                "left": {"col": "orders.customer_id"},
                                "op": "=",
                                "right": {"col": "customers.customer_id"},
                            }
                        },
                    },
                ],
                "select": [{"expr": {"col": "orders.order_id"}, "alias": "order_id"}],
                "limit": 5,
            }
        )
        failures.append("duplicate join should require join.as")
    except QueryPlanError as e:
        if "as" not in str(e).lower():
            failures.append(f"duplicate join message: {e}")
    except Exception as e:
        failures.append(f"duplicate join wrong exc: {e}")

    # --- self-join with as compiles ---
    try:
        c = Compiler(_minimal_schema())
        sql, _ = c.compile(
            {
                "dataset": "orders",
                "joins": [
                    {
                        "dataset": "orders",
                        "as": "o2",
                        "type": "inner",
                        "on": {
                            "cmp": {
                                "left": {"col": "orders.order_id"},
                                "op": "=",
                                "right": {"col": "o2.order_id"},
                            }
                        },
                    }
                ],
                "select": [{"expr": {"col": "orders.order_id"}, "alias": "a"}],
                "limit": 1,
            }
        )
        if "JOIN" not in sql.upper() or "orders" not in sql.lower():
            failures.append(f"self-join SQL unexpected: {sql[:120]}")
    except Exception as e:
        failures.append(f"self-join compile: {e}")

    # --- sql function wrong arity ---
    try:
        c = Compiler(_minimal_schema())
        c.compile(
            {
                "dataset": "orders",
                "select": [
                    {
                        "expr": {"func": "count", "args": [{"col": "order_id"}, {"col": "customer_id"}]},
                        "alias": "x",
                    }
                ],
                "limit": 1,
            }
        )
        failures.append("wrong func arity should error")
    except QueryPlanError as e:
        if "does not accept" not in str(e).lower():
            failures.append(f"func arity message: {e}")
    except Exception as e:
        failures.append(f"func arity wrong exc: {e}")

    # --- set_op column mismatch ---
    try:
        c = Compiler(_minimal_schema())
        c.compile(
            {
                "set_op": {
                    "op": "union_all",
                    "left": {"dataset": "orders", "select": [{"expr": {"col": "order_id"}, "alias": "a"}], "limit": 1},
                    "right": {
                        "dataset": "orders",
                        "select": [
                            {"expr": {"col": "order_id"}, "alias": "a"},
                            {"expr": {"col": "customer_id"}, "alias": "b"},
                        ],
                        "limit": 1,
                    },
                }
            }
        )
        failures.append("set_op mismatch should error")
    except QueryPlanError as e:
        if "same number" not in str(e).lower():
            failures.append(f"set_op message: {e}")
    except Exception as e:
        failures.append(f"set_op wrong exc: {e}")

    # --- empty IN list ---
    try:
        c = Compiler(_minimal_schema())
        c.compile(
            {
                "dataset": "orders",
                "select": [{"expr": {"col": "order_id"}, "alias": "x"}],
                "where": {"cmp": {"left": {"col": "order_id"}, "op": "in", "right": []}},
                "limit": 1,
            }
        )
        failures.append("empty IN should error")
    except QueryPlanError as e:
        if "non-empty" not in str(e).lower():
            failures.append(f"empty IN message: {e}")
    except Exception as e:
        failures.append(f"empty IN wrong exc: {e}")

    # --- dotted logical table name ---
    try:
        schema = {
            "tables": [
                {
                    "name": "a.b",
                    "db_table": "ab_t",
                    "columns": [{"name": "c", "db_column": "c", "type": "int"}],
                },
                {
                    "name": "a",
                    "db_table": "a_t",
                    "columns": [{"name": "b", "db_column": "b", "type": "int"}],
                },
            ],
            "links": [],
        }
        c = Compiler(schema)
        sql, _ = c.compile(
            {
                "dataset": "a.b",
                "select": [{"expr": {"col": "a.b.c"}, "alias": "x"}],
                "limit": 1,
            }
        )
        if "ab_t" not in sql and "c" not in sql:
            failures.append(f"dotted table SQL: {sql[:120]}")
    except Exception as e:
        failures.append(f"dotted table: {e}")

    if failures:
        print("[selfcheck] failures:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("[selfcheck] all checks passed")
    sys.exit(0)


if MODE == "pipeline":
    _run_pipeline_mode()

if MODE == "selfcheck":
    _run_selfcheck()

# ---------------------------------------------------------------------------
# DB connection (skipped in lint, pipeline, and selfcheck modes)
# ---------------------------------------------------------------------------
load_dotenv(dotenv_path=ENV_PATH, override=True)

def _must(k: str) -> str:
    v = os.getenv(k, "").strip()
    if not v:
        raise SystemExit(f"[test] Missing env var '{k}' (looked in {ENV_PATH})")
    return v

engine = None
if MODE not in ("lint", "pipeline", "selfcheck"):
    engine = create_engine(
        "postgresql+psycopg2://",
        creator=lambda: psycopg2.connect(
            host=_must("DB_HOST"),
            port=int(_must("DB_PORT")),
            dbname=_must("DB_NAME"),
            user=_must("DB_USER"),
            password=_must("DB_PASSWORD"),
            sslmode="require",
        ),
        poolclass=NullPool,
        pool_pre_ping=True,
    )

    # Quick connectivity check
    with engine.connect() as conn:
        row = conn.execute(text("select current_database(), current_user")).fetchone()
        print(f"[test] Connected: db={row[0]}, user={row[1]}")

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
for p in [SCHEMA_PATH, SUITE_PATH]:
    if not p.exists():
        raise SystemExit(f"[test] Required file not found: {p}")

# ---------------------------------------------------------------------------
# Load schema for lint tests
# ---------------------------------------------------------------------------
with open(SCHEMA_PATH) as f:
    _schema_for_lint = yaml.safe_load(f) or {}

# ---------------------------------------------------------------------------
# Run suite — handles both DB tests and lint tests
# ---------------------------------------------------------------------------
with open(SUITE_PATH) as f:
    suite: list = json.load(f)

print(f"[test] Mode={MODE}  Tests={len(suite)}  Schema={SCHEMA_PATH}\n")

results = []
errors = 0

def _lint_fires(question: str, plan: dict, fragment: str) -> tuple[bool, str]:
    errors = semantic_lint(question, plan, _schema_for_lint)   # <-- pass schema
    matched = any(fragment.lower() in e.lower() for e in errors)
    msg = errors[0][:100] if errors else "(no errors)"
    return matched, msg

def _lint_clean(question: str, plan: dict) -> tuple[bool, str]:
    errors = semantic_lint(question, plan, _schema_for_lint)   # <-- pass schema
    return (not errors), (str(errors) if errors else "clean")


def _fingerprint_after_pipeline(plan: dict) -> str:
    """Match execute_query_plan: auto_inject_joins → canonicalize → fingerprint."""
    p = copy.deepcopy(plan)
    p = auto_inject_joins(p, _schema_for_lint)
    p = canonicalize_query_plan(p)
    return plan_fingerprint(p)


def _run_canonical_test(test: dict) -> tuple[bool, str | None]:
    """
    canonical.kind:
      pair      — plan_fingerprint(plan_a) == plan_fingerprint(plan_b)
      order_by  — order_by clause order unchanged after canonicalize
      idempotent — canonicalize twice equals once (JSON-stable)
    """
    spec = test.get("canonical") or {}
    kind = spec.get("kind", "")

    if kind == "pair":
        pa = spec.get("plan_a")
        pb = spec.get("plan_b")
        if not isinstance(pa, dict) or not isinstance(pb, dict):
            return False, "canonical.pair requires plan_a and plan_b objects"
        fa = _fingerprint_after_pipeline(pa)
        fb = _fingerprint_after_pipeline(pb)
        if fa != fb:
            return False, f"fingerprint mismatch: {fa} vs {fb}"
        return True, None

    if kind == "order_by":
        pl = spec.get("plan")
        if not isinstance(pl, dict):
            return False, "canonical.order_by requires plan object"
        want = pl.get("order_by") or []
        if not isinstance(want, list) or len(want) < 2:
            return False, "plan.order_by must be a list with at least 2 items"
        c = canonicalize_query_plan(copy.deepcopy(auto_inject_joins(copy.deepcopy(pl), _schema_for_lint)))
        got = c.get("order_by") or []
        if len(got) != len(want):
            return False, f"order_by length {len(got)} != {len(want)}"
        for i, w in enumerate(want):
            if (got[i].get("by") if isinstance(got[i], dict) else None) != (
                w.get("by") if isinstance(w, dict) else None
            ):
                return False, f"order_by[{i}] by= mismatch after canonicalize"
        return True, None

    if kind == "idempotent":
        pl = spec.get("plan")
        if not isinstance(pl, dict):
            return False, "canonical.idempotent requires plan object"
        once = canonicalize_query_plan(copy.deepcopy(pl))
        twice = canonicalize_query_plan(copy.deepcopy(once))
        j1 = json.dumps(once, sort_keys=True, default=str)
        j2 = json.dumps(twice, sort_keys=True, default=str)
        if j1 != j2:
            return False, "second canonicalize changed JSON"
        return True, None

    return False, f"unknown canonical.kind {kind!r}"


for i, test in enumerate(suite):
    name = test.get("name", f"test_{i}")
    question = test.get("question", "")
    plan = test.get("plan")
    test_type = test.get("type", "db")  # "db" | "lint" | "canonical"

    if test_type == "canonical":
        ok, err = _run_canonical_test(test)
        result_entry = {
            "name": name,
            "question": question,
            "type": "canonical",
            "passed": ok,
            "error": err,
        }
        results.append(result_entry)
        status = "PASS" if ok else "FAIL"
        print(f"  [{i+1}/{len(suite)}] {name}: [{status}]")
        if not ok:
            print(f"         {err}")
            errors += 1
        continue

    if test_type == "lint":
        lint_spec = test.get("lint", {})
        expect = lint_spec.get("expect")        # "fires" or "clean"
        fragment = lint_spec.get("fragment", "")

        if expect == "fires":
            ok, msg = _lint_fires(question, plan, fragment)
            result_entry = {
                "name": name,
                "question": question,
                "type": "lint",
                "plan": plan,
                "lint_expect": expect,
                "lint_fragment": fragment,
                "lint_errors": msg,
                "passed": ok,
                "error": None if ok else f"Expected lint to fire with fragment '{fragment}' but got: {msg}",
            }
        else:  # "clean"
            ok, msg = _lint_clean(question, plan)
            result_entry = {
                "name": name,
                "question": question,
                "type": "lint",
                "plan": plan,
                "lint_expect": expect,
                "lint_fragment": fragment,
                "lint_errors": msg,
                "passed": ok,
                "error": None if ok else f"Expected lint clean but got: {msg}",
            }

        results.append(result_entry)
        status = "PASS" if ok else "FAIL"
        print(f"  [{i+1}/{len(suite)}] {name}: [{status}]")
        if not ok:
            print(f"         {result_entry['error']}")
            errors += 1

    else:
        # Standard DB test — skipped in lint mode
        if MODE == "lint":
            print(f"  [{i+1}/{len(suite)}] {name}: [SKIP] (db test, lint mode)")
            continue

        try:
            res = execute_query_plan(
                engine=engine,
                schema_path=str(SCHEMA_PATH),
                query_plan=plan,
                statement_timeout_ms=120_000,   # 2 minutes — generous for test/dev DB latency
            )
        except Exception as e:
            res = {"error": {"message": str(e)}}

        entry = {
            "name": name,
            "question": question,
            "type": "db",
            "plan": plan,
            "row_count": res.get("row_count"),
            "columns": res.get("columns"),
            "first_row": res["rows"][0] if res.get("rows") else None,
            "error": res.get("error"),
            "sql": res.get("sql"),
        }
        results.append(entry)

        status = "ERROR" if entry["error"] else "OK"
        print(f"  [{i+1}/{len(suite)}] {name}: {status}  row_count={entry['row_count']}")
        if entry["error"]:
            print(f"         ERROR: {entry['error']['message']}")
            errors += 1

print()

# ---------------------------------------------------------------------------
# Mode: lint — only lint tests, no DB needed
# ---------------------------------------------------------------------------
if MODE == "lint":
    print(f"\n[test] Lint + canonical (no DB): {len(results)} tests, {errors} failure(s)")
    sys.exit(0 if errors == 0 else 1)

# ---------------------------------------------------------------------------
# Mode: update — save baseline
# ---------------------------------------------------------------------------
if MODE == "update":
    payload = {"count": len(results), "results": results}
    BASELINE_PATH.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[test] Baseline updated → {BASELINE_PATH}")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Mode: run — just print, no comparison
# ---------------------------------------------------------------------------
if MODE == "run":
    payload = {"count": len(results), "results": results}
    print(json.dumps(payload, indent=2, default=str))
    sys.exit(0 if errors == 0 else 1)

# ---------------------------------------------------------------------------
# Mode: check — compare against baseline
# ---------------------------------------------------------------------------
if not BASELINE_PATH.exists():
    raise SystemExit(
        f"[test] No baseline found at {BASELINE_PATH}.\n"
        "Run `python test/test_main.py update` first to create it."
    )

with open(BASELINE_PATH) as f:
    baseline = json.load(f)

baseline_by_name = {r["name"]: r for r in baseline.get("results", [])}

regression_failures = []
passed = failed = 0

for entry in results:
    name = entry["name"]
    base = baseline_by_name.get(name)

    if base is None:
        print(f"  [NEW]  {name} — not in baseline (run 'update' to add)")
        continue

    if entry.get("type") == "lint":
        if base.get("type") != "lint" or "passed" not in base:
            failed += 1
            regression_failures.append(name)
            print(f"  [STALE] {name} — baseline has no lint 'passed' field (legacy db row?).")
            print("         Run: python test/test_main.py update")
            continue
        ok = entry["passed"] == base["passed"]
        if ok:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            failed += 1
            regression_failures.append(name)
            print(f"  [FAIL] {name}")
            print(f"         passed baseline={base['passed']}  current={entry['passed']}")
    elif entry.get("type") == "canonical":
        if base.get("type") != "canonical" or "passed" not in base:
            failed += 1
            regression_failures.append(name)
            print(f"  [STALE] {name} — baseline has no canonical 'passed' field.")
            print("         Run: python test/test_main.py update")
            continue
        ok = entry["passed"] == base["passed"]
        if ok:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            failed += 1
            regression_failures.append(name)
            print(f"  [FAIL] {name}")
            print(f"         passed baseline={base['passed']}  current={entry['passed']}")
    else:
        row_count_match = entry["row_count"] == base["row_count"]
        first_row_match = entry["first_row"] == base["first_row"]
        error_match = entry["error"] == base["error"]

        if row_count_match and first_row_match and error_match:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            failed += 1
            regression_failures.append(name)
            print(f"  [FAIL] {name}")
            if not row_count_match:
                print(f"         row_count: baseline={base['row_count']}  current={entry['row_count']}")
            if not first_row_match:
                print(f"         first_row baseline: {base['first_row']}")
                print(f"         first_row current:  {entry['first_row']}")
            if not error_match:
                print(f"         error baseline: {base['error']}")
                print(f"         error current:  {entry['error']}")

print(f"\n[test] Results: {passed} passed, {failed} failed, {errors} errors out of {len(results)} tests")

if regression_failures:
    print(f"[test] REGRESSIONS: {regression_failures}")
    sys.exit(1)

sys.exit(0)