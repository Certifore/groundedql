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
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from dsl_compiler import execute_query_plan

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
assert MODE in {"run", "update", "check"}, f"Unknown mode '{MODE}'. Use: run | update | check"

# ---------------------------------------------------------------------------
# DB connection
# ---------------------------------------------------------------------------
load_dotenv(dotenv_path=ENV_PATH, override=True)

def _must(k: str) -> str:
    v = os.getenv(k, "").strip()
    if not v:
        raise SystemExit(f"[test] Missing env var '{k}' (looked in {ENV_PATH})")
    return v

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
# Run suite
# ---------------------------------------------------------------------------
with open(SUITE_PATH) as f:
    suite: list = json.load(f)

print(f"[test] Mode={MODE}  Tests={len(suite)}  Schema={SCHEMA_PATH}\n")

results = []
passed = 0
failed = 0
errors = 0

for i, test in enumerate(suite):
    name = test.get("name", f"test_{i}")
    question = test.get("question", "")
    plan = test.get("plan")

    try:
        res = execute_query_plan(
            engine=engine,
            schema_path=str(SCHEMA_PATH),
            query_plan=plan,
        )
    except Exception as e:
        res = {"error": {"message": str(e)}}

    entry = {
        "name": name,
        "question": question,      # <-- added for easy review
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

for entry in results:
    name = entry["name"]
    base = baseline_by_name.get(name)

    if base is None:
        print(f"  [NEW]  {name} — not in baseline (run 'update' to add)")
        continue

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