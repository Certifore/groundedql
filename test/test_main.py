"""
Regression test runner for IntentQL.

Modes (set via TEST_MODE env var or command-line arg):
  run     — Full regression suite from test_qs.json (default)
  update  — Execute all plans and OVERWRITE the baseline file
  check   — Compare against baseline (exits 1 on regression; CI)
  pipeline — Compile preflight from test_qs.json then Benchmark 4 (NL → planner → DB; see benchmark/.env)
  lint    — No DB: lint + canonical + compile rows from test_qs.json

Usage:
  python test/test_main.py              # full suite (default)
  python test/test_main.py update       # overwrite baseline
  python test/test_main.py check        # regression check (for CI)
  python test/test_main.py pipeline     # compile preflight + pipeline benchmark
  python test/test_main.py lint        # no-DB tests only

Test types in test_qs.json:
  (default) db   — execute against Postgres
  lint           — semantic_lint only (no DB)
  canonical      — structural plan_fingerprint / canonicalize checks (no DB)
  compile        — compiler / schema / validate_query_plan_dict / $relative_date checks (no DB); see compile.kind
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

from intentql import execute_query_plan
from intentql.join_planner import auto_inject_joins
from intentql.plan_canonical import canonicalize_query_plan, plan_fingerprint
from intentql.semantic_lint import semantic_lint

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
assert MODE in {"run", "update", "check", "lint", "pipeline"}, (
    f"Unknown mode '{MODE}'. Use: run | update | check | lint | pipeline"
)


def _minimal_schema_for_compile(*, bad_link_join_type: str | None = None) -> dict:
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


def _run_compile_test(test: dict) -> tuple[bool, str | None]:
    """Dispatch compile.kind from test_qs.json (no DB)."""
    import tempfile
    from pathlib import Path as P

    from intentql.compiler import Compiler, QueryPlanError
    from intentql.exceptions import SchemaError
    from intentql.schema_validator import validate_schema
    from intentql.validation import validate_query_plan_dict

    spec = test.get("compile") or {}
    kind = spec.get("kind", "")

    if kind == "validate_meta":
        schema = spec.get("schema")
        plan = spec.get("plan")
        if not isinstance(schema, dict) or not isinstance(plan, dict):
            return False, "validate_meta requires compile.schema and compile.plan objects"
        with tempfile.TemporaryDirectory() as td:
            sp = P(td) / "schema.yaml"
            sp.write_text(yaml.safe_dump(schema), encoding="utf-8")
            parsed, errs = validate_query_plan_dict(plan, str(sp))
            if parsed is None or errs:
                return False, f"validate_query_plan_dict+meta: {errs}"
            return True, None

    if kind == "schema_rejects_bad_link":
        jt = spec.get("bad_link_join_type")
        if not isinstance(jt, str):
            return False, "schema_rejects_bad_link requires compile.bad_link_join_type"
        try:
            validate_schema(_minimal_schema_for_compile(bad_link_join_type=jt))
            return False, "validate_schema should reject bad join_type"
        except SchemaError as e:
            if "join_type" not in str(e).lower():
                return False, f"validate_schema wrong message: {e}"
            return True, None
        except Exception as e:
            return False, f"validate_schema wrong exc: {e}"

    if kind == "compiler_rejects_bad_schema":
        jt = spec.get("bad_link_join_type")
        if not isinstance(jt, str):
            return False, "compiler_rejects_bad_schema requires compile.bad_link_join_type"
        try:
            Compiler(_minimal_schema_for_compile(bad_link_join_type=jt))
            return False, "Compiler should reject bad join_type at init"
        except SchemaError as e:
            if "join_type" not in str(e).lower():
                return False, f"Compiler schema wrong message: {e}"
            return True, None
        except Exception as e:
            return False, f"Compiler schema wrong exc: {e}"

    if kind == "duplicate_join_requires_as":
        plan = spec.get("plan")
        if not isinstance(plan, dict):
            return False, "duplicate_join_requires_as requires compile.plan"
        try:
            c = Compiler(_minimal_schema_for_compile())
            c.compile(plan)
            return False, "duplicate join should require join.as"
        except QueryPlanError as e:
            if "as" not in str(e).lower():
                return False, f"duplicate join message: {e}"
            return True, None
        except Exception as e:
            return False, f"duplicate join wrong exc: {e}"

    if kind == "self_join_compiles":
        plan = spec.get("plan")
        if not isinstance(plan, dict):
            return False, "self_join_compiles requires compile.plan"
        try:
            c = Compiler(_minimal_schema_for_compile())
            sql, _ = c.compile(plan)
            if "JOIN" not in sql.upper() or "orders" not in sql.lower():
                return False, f"self-join SQL unexpected: {sql[:120]}"
            return True, None
        except Exception as e:
            return False, f"self-join compile: {e}"

    if kind == "func_wrong_arity":
        plan = spec.get("plan")
        if not isinstance(plan, dict):
            return False, "func_wrong_arity requires compile.plan"
        try:
            c = Compiler(_minimal_schema_for_compile())
            c.compile(plan)
            return False, "wrong func arity should error"
        except QueryPlanError as e:
            if "does not accept" not in str(e).lower():
                return False, f"func arity message: {e}"
            return True, None
        except Exception as e:
            return False, f"func arity wrong exc: {e}"

    if kind == "set_op_mismatch":
        plan = spec.get("plan")
        if not isinstance(plan, dict):
            return False, "set_op_mismatch requires compile.plan"
        try:
            c = Compiler(_minimal_schema_for_compile())
            c.compile(plan)
            return False, "set_op mismatch should error"
        except QueryPlanError as e:
            if "same number" not in str(e).lower():
                return False, f"set_op message: {e}"
            return True, None
        except Exception as e:
            return False, f"set_op wrong exc: {e}"

    if kind == "empty_in":
        plan = spec.get("plan")
        if not isinstance(plan, dict):
            return False, "empty_in requires compile.plan"
        try:
            c = Compiler(_minimal_schema_for_compile())
            c.compile(plan)
            return False, "empty IN should error"
        except QueryPlanError as e:
            if "non-empty" not in str(e).lower():
                return False, f"empty IN message: {e}"
            return True, None
        except Exception as e:
            return False, f"empty IN wrong exc: {e}"

    if kind == "dotted_table":
        schema = spec.get("schema")
        plan = spec.get("plan")
        if not isinstance(schema, dict) or not isinstance(plan, dict):
            return False, "dotted_table requires compile.schema and compile.plan"
        try:
            c = Compiler(schema)
            sql, _ = c.compile(plan)
            if "ab_t" not in sql and "c" not in sql:
                return False, f"dotted table SQL: {sql[:120]}"
            return True, None
        except Exception as e:
            return False, f"dotted table: {e}"

    if kind == "legacy_order_by_string":
        plan = spec.get("plan")
        if not isinstance(plan, dict):
            return False, "legacy_order_by_string requires compile.plan"
        try:
            c = Compiler(_minimal_schema_for_compile())
            sql, _ = c.compile(plan)
            if "ORDER BY" not in sql.upper():
                return False, "legacy order_by: expected ORDER BY in SQL"
            return True, None
        except Exception as e:
            if "Expression must be an object" in str(e):
                return False, f"legacy order_by string column: {e}"
            return False, f"legacy order_by: {e}"

    if kind == "compound_cte_compiles":
        plan = spec.get("plan")
        if not isinstance(plan, dict):
            return False, "compound_cte_compiles requires compile.plan"
        try:
            c = Compiler(_minimal_schema_for_compile())
            sql, _ = c.compile(plan)
            sql_u = sql.upper()
            if "WITH" not in sql_u:
                return False, "compound plan: expected WITH in SQL (CTE pipeline)"
            if "FIRST_ORDERS" not in sql_u and "first_orders" not in sql.lower():
                return False, "compound plan: expected CTE name in SQL"
            return True, None
        except Exception as e:
            return False, f"compound plan compile: {e}"

    if kind in ("compound_cte_validates", "chained_cte_validates"):
        schema = spec.get("schema")
        plan = spec.get("plan")
        if not isinstance(schema, dict) or not isinstance(plan, dict):
            return False, f"{kind} requires compile.schema and compile.plan"
        try:
            with tempfile.TemporaryDirectory() as td:
                sp2 = P(td) / "schema.yaml"
                sp2.write_text(yaml.safe_dump(schema), encoding="utf-8")
                _, verr = validate_query_plan_dict(plan, str(sp2))
                if verr:
                    return False, f"compound plan validation: {verr}"
                return True, None
        except Exception as e:
            return False, f"compound plan validate_query_plan_dict: {e}"

    if kind == "cte_outer_count_distinct_compiles":
        schema = spec.get("schema")
        plan = spec.get("plan")
        if not isinstance(schema, dict) or not isinstance(plan, dict):
            return False, "cte_outer_count_distinct_compiles requires compile.schema and compile.plan"
        try:
            with tempfile.TemporaryDirectory() as td:
                sp2 = P(td) / "schema.yaml"
                sp2.write_text(yaml.safe_dump(schema), encoding="utf-8")
                _, verr = validate_query_plan_dict(plan, str(sp2))
                if verr:
                    return False, f"cte outer metric validation: {verr}"
            c = Compiler(schema)
            sql, _ = c.compile(plan)
            sql_l = sql.lower()
            if "with" not in sql_l or "filtered_work_orders" not in sql_l:
                return False, f"expected WITH CTE in SQL: {sql[:240]}"
            if "count(distinct" not in sql_l.replace(" ", ""):
                return False, f"expected COUNT(DISTINCT ...) in SQL: {sql[:240]}"
            # Outer plan includes order_by by edit_date; scalar aggregate must drop it (no inner ORDER BY).
            if "order by" in sql_l:
                return False, f"invalid outer order_by should be stripped: {sql[:320]}"
            return True, None
        except Exception as e:
            return False, f"cte outer count_distinct compile: {e}"

    if kind == "scalar_aggregate_strips_order_by_compiles":
        schema = spec.get("schema")
        plan = spec.get("plan")
        if not isinstance(schema, dict) or not isinstance(plan, dict):
            return False, "scalar_aggregate_strips_order_by_compiles requires compile.schema and compile.plan"
        try:
            with tempfile.TemporaryDirectory() as td:
                sp2 = P(td) / "schema.yaml"
                sp2.write_text(yaml.safe_dump(schema), encoding="utf-8")
                _, verr = validate_query_plan_dict(plan, str(sp2))
                if verr:
                    return False, f"scalar aggregate validation: {verr}"
            c = Compiler(schema)
            sql, _ = c.compile(plan)
            sql_l = sql.lower()
            if "count(" not in sql_l:
                return False, f"expected aggregate in SQL: {sql[:240]}"
            if "order by" in sql_l:
                return False, f"scalar aggregate must drop ORDER BY on non-metric columns: {sql[:320]}"
            return True, None
        except Exception as e:
            return False, f"scalar aggregate strip order_by: {e}"

    if kind == "relative_date_calendar_year":
        import datetime as dt

        from intentql.api.api import _resolve_relative_dates

        lo = _resolve_relative_dates(
            {"$relative_date": {"op": "calendar_year_start", "year_offset": -1}}
        )
        hi = _resolve_relative_dates(
            {"$relative_date": {"op": "calendar_year_start", "year_offset": 0}}
        )
        d_lo = dt.datetime.fromisoformat(lo)
        d_hi = dt.datetime.fromisoformat(hi)
        if d_lo.month != 1 or d_lo.day != 1 or d_hi.month != 1 or d_hi.day != 1:
            return False, f"calendar_year_start expected Jan 1: lo={lo!r} hi={hi!r}"
        if d_hi.year != d_lo.year + 1:
            return False, f"calendar_year_start year gap: lo={lo!r} hi={hi!r}"
        if d_lo.tzinfo != dt.timezone.utc or d_hi.tzinfo != dt.timezone.utc:
            return False, f"calendar_year_start expected UTC: lo={lo!r} hi={hi!r}"
        y = dt.datetime.now(dt.timezone.utc).year
        expected_default = dt.datetime(
            y, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc
        ).isoformat()
        z = _resolve_relative_dates({"$relative_date": {"op": "calendar_year_start"}})
        if z != expected_default:
            return (
                False,
                f"calendar_year_start default year_offset: got {z!r} want {expected_default!r}",
            )
        return True, None

    return False, f"unknown compile.kind {kind!r}"


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
        print("\n[setup] OPENAI_API_KEY not set — running IntentQL full pipeline with Gemini only.")
        print(f"\n[1/1] IntentQL full pipeline ({len(pipeline_questions)} questions)...")
        qce = bench_pipeline_qce(schema, pipeline_questions, db_url)
        pipe = [qce]
    else:
        print("\n[setup] Initialising competitors...")
        gpt4_client = make_client(openai_key)
        langchain_agent = make_agent(db_url, openai_key)

        print(f"\n[1/3] IntentQL full pipeline ({len(pipeline_questions)} questions)...")
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


if MODE == "pipeline":
    if not SUITE_PATH.exists():
        raise SystemExit(f"[test] Required file not found: {SUITE_PATH}")
    with open(SUITE_PATH) as f:
        _suite_pipeline: list = json.load(f)
    _pf_errs = 0
    for _t in _suite_pipeline:
        if _t.get("type") != "compile":
            continue
        _ok, _err = _run_compile_test(_t)
        if not _ok:
            print(f"[compile] {_t.get('name')}: {_err}")
            _pf_errs += 1
    if _pf_errs:
        print(f"[compile] {_pf_errs} failure(s) — fix before pipeline")
        sys.exit(1)
    print("[compile] OK (pipeline preflight)\n")
    _run_pipeline_mode()

# ---------------------------------------------------------------------------
# DB connection (skipped in lint and pipeline modes)
# ---------------------------------------------------------------------------
load_dotenv(dotenv_path=ENV_PATH, override=True)

def _must(k: str) -> str:
    v = os.getenv(k, "").strip()
    if not v:
        raise SystemExit(f"[test] Missing env var '{k}' (looked in {ENV_PATH})")
    return v

engine = None
if MODE not in ("lint", "pipeline"):
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

def _lint_fires(question: str, plan: dict, fragment: str, schema: dict | None = None) -> tuple[bool, str]:
    sch = schema if schema is not None else _schema_for_lint
    errors = semantic_lint(question, plan, sch)
    matched = any(fragment.lower() in e.lower() for e in errors)
    msg = errors[0][:100] if errors else "(no errors)"
    return matched, msg


def _lint_clean(question: str, plan: dict, schema: dict | None = None) -> tuple[bool, str]:
    sch = schema if schema is not None else _schema_for_lint
    errors = semantic_lint(question, plan, sch)
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
    test_type = test.get("type", "db")  # "db" | "lint" | "canonical" | "compile"

    if test_type == "compile":
        ok, err = _run_compile_test(test)
        result_entry = {
            "name": name,
            "question": question,
            "type": "compile",
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
        lint_schema = lint_spec.get("schema")

        if expect == "fires":
            ok, msg = _lint_fires(question, plan, fragment, lint_schema)
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
            ok, msg = _lint_clean(question, plan, lint_schema)
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
    print(f"\n[test] Lint + canonical + compile (no DB): {len(results)} tests, {errors} failure(s)")
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
    elif entry.get("type") == "compile":
        if base.get("type") != "compile" or "passed" not in base:
            failed += 1
            regression_failures.append(name)
            print(f"  [STALE] {name} — baseline has no compile 'passed' field.")
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