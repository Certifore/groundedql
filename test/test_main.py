import json
import os
from pathlib import Path

from dotenv import load_dotenv
import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from dsl_compiler import execute_query_plan

# ----------------------------
# Resolve paths deterministically
# ----------------------------
HERE = Path(__file__).resolve().parent          # .../test
ROOT = HERE.parents[0]                         # repo root (../.. from this file)

ENV_PATH = ROOT / ".env"
SCHEMA_PATH = ROOT / "config" / "schema.yaml"

REG_DIR = HERE / "regression_test"
SUITE_PATH = REG_DIR / "test_qs.json"
OUT_PATH = REG_DIR / "suite_results.json"

# Ensure output directory exists
REG_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Load env reliably (always the repo root .env)
# ----------------------------
load_dotenv(dotenv_path=ENV_PATH, override=True)

def must_env(k: str) -> str:
    v = os.getenv(k)
    if not v:
        raise SystemExit(f"Missing {k} in {ENV_PATH}")
    return v.strip()

DB_USER = must_env("DB_USER")
DB_PASSWORD = must_env("DB_PASSWORD")
DB_HOST = must_env("DB_HOST")
DB_PORT = must_env("DB_PORT")   # Supabase pooler typically 6543
DB_NAME = must_env("DB_NAME")

def creator():
    conninfo = (
        f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} "
        f"user={DB_USER} password={DB_PASSWORD} sslmode=require"
    )
    return psycopg2.connect(conninfo)

engine = create_engine(
    "postgresql+psycopg2://",
    creator=creator,
    poolclass=NullPool,
    pool_pre_ping=True,
)

# ----------------------------
# Sanity checks + debug context
# ----------------------------
with engine.connect() as conn:
    conn.execute(text("select 1"))
    ctx = conn.execute(text("select current_database(), current_user, current_schema()")).fetchone()
    print("DB context:", ctx)

print("CWD:", Path.cwd())
print("ENV_PATH:", ENV_PATH)
print("SCHEMA_PATH:", SCHEMA_PATH)
print("SUITE_PATH:", SUITE_PATH)
print("OUT_PATH:", OUT_PATH)

if not SCHEMA_PATH.exists():
    raise SystemExit(f"Schema file not found: {SCHEMA_PATH}")

if not SUITE_PATH.exists():
    raise SystemExit(f"Suite file not found: {SUITE_PATH}")

# ----------------------------
# Run the suite
# ----------------------------
with open(SUITE_PATH, "r") as f:
    suite = json.load(f)

results = []
for i, test in enumerate(suite):
    name = test.get("name", f"test_{i}")
    question = test.get("question", "")
    plan = test.get("plan")

    out_obj = {
        "name": name,
        "question": question,
        "plan": plan,
        "result": None,
    }

    try:
        res = execute_query_plan(engine=engine, schema_path=str(SCHEMA_PATH), query_plan=plan)
        out_obj["result"] = res
    except Exception as e:
        out_obj["result"] = {"error": {"message": str(e)}}

    results.append(out_obj)

final = {
    "suite": str(SUITE_PATH),
    "count": len(results),
    "results": results,
}

# Print combined JSON
print(json.dumps(final, indent=2, default=str))

# Save to regression_test/
OUT_PATH.write_text(json.dumps(final, indent=2, default=str))
print(f"\nSaved: {OUT_PATH}")