"""
IntentQL test runner (guided SQL + sql_guard).

Usage:
  python test/test_main.py              # default: sql_guard checks (no DB)
  python test/test_main.py sqlguard     # same as default

Environment:
  TEST_MODE — same as argv[1] if argv omitted (default: run)

Requires: pip install -e ".[guided]" for sqlglot.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
_rp = str(ROOT)
if _rp not in sys.path:
    sys.path.insert(0, _rp)

SCHEMA_PATH = ROOT / "config" / "schema.yaml"

MODE = (sys.argv[1] if len(sys.argv) > 1 else os.getenv("TEST_MODE", "run")).lower()


def _require_sqlglot() -> None:
    try:
        import sqlglot  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "sqlglot is required. Install: pip install -e '.[guided]'"
        ) from e


def run_sql_guard_suite() -> None:
    """Validate sql_guard against bundled config/schema.yaml (no database)."""
    _require_sqlglot()

    if not SCHEMA_PATH.is_file():
        raise SystemExit(f"Missing {SCHEMA_PATH}")

    from intentql.schema_catalog import load_schema_catalog
    from intentql.sql_guard import apply_row_limit, validate_sql

    catalog = load_schema_catalog(str(SCHEMA_PATH))

    sql = "SELECT COUNT(*) AS n FROM customers"
    r = validate_sql(sql, catalog)
    assert r.ok, r.message

    sql_bad = "SELECT * FROM not_a_real_table_xyz"
    r2 = validate_sql(sql_bad, catalog)
    assert not r2.ok

    lim = apply_row_limit("SELECT 1 AS x", 100)
    assert "LIMIT 100" in lim

    # ORDER BY may use a SELECT alias (not a physical column name).
    sql_alias_order = (
        "SELECT customer_id, COUNT(*) AS row_cnt FROM orders "
        "GROUP BY customer_id ORDER BY row_cnt DESC LIMIT 1"
    )
    r3 = validate_sql(sql_alias_order, catalog)
    assert r3.ok, r3.message

    print("test_main: sql_guard suite OK")


def main() -> None:
    if MODE in ("run", "sqlguard", ""):
        run_sql_guard_suite()
        return
    print(f"Unknown TEST_MODE / argv: {MODE!r}. Use: run | sqlguard", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
