"""
Pipeline-only benchmark — re-runs Benchmark 4 only.
Use this to test latency/cost impact of spec changes without re-running all benchmarks.

Usage:
    cd /home/alexander/git_repos/dsl_compiler
    python3 benchmark/compare/run_pipeline_only.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import everything from run_comparison — no duplication
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_comparison import (
    _check_env, _db_url, _print_table, _print_token_table,
    bench_pipeline_qce, bench_pipeline_gpt4, bench_pipeline_langchain,
    make_client, make_agent, RESULTS_DIR, DATA_DIR, SCHEMA_PATH, SPEC_PATH,
)

import yaml


def main() -> None:
    _check_env()

    print("=" * 75)
    print("  Pipeline-Only Benchmark (Benchmark 4)")
    print(f"  Spec: {SPEC_PATH}")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 75)

    with open(SCHEMA_PATH) as f:
        schema = yaml.safe_load(f)
    with open(DATA_DIR / "pipeline_questions.json") as f:
        pipeline_questions = json.load(f)

    openai_key = os.environ["OPENAI_API_KEY"]
    db_url = _db_url()

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

    out_path = RESULTS_DIR / "pipeline_latest.json"
    out_path.write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).isoformat(),
        "spec_used": str(SPEC_PATH),
        "results": pipe,
    }, indent=2, default=str))
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
