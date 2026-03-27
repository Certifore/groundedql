"""
Regression test runner for dsl_compiler.

Modes (set via TEST_MODE env var or command-line arg):
  run     — Execute all plans, print results. Does NOT write baseline. (default)
  update  — Execute all plans and OVERWRITE the baseline file.
  check   — Execute all plans and COMPARE against the saved baseline.
             Exits with code 1 if any row_count or first-row values differ.

Usage:
  python test/test_main.py              # run (default)
  python test/test_main.py update       # overwrite baseline
  python test/test_main.py check        # regression check (for CI)

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
assert MODE in {"run", "update", "check", "lint"}, f"Unknown mode '{MODE}'. Use: run | update | check | lint"

# ---------------------------------------------------------------------------
# DB connection (skipped in lint mode)
# ---------------------------------------------------------------------------
load_dotenv(dotenv_path=ENV_PATH, override=True)

def _must(k: str) -> str:
    v = os.getenv(k, "").strip()
    if not v:
        raise SystemExit(f"[test] Missing env var '{k}' (looked in {ENV_PATH})")
    return v

engine = None
if MODE != "lint":
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