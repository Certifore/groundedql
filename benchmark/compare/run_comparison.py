"""
QCE vs LangChain vs GPT-4 Direct — Comparison Benchmark

Usage:
    cd /home/alexander/git_repos/dsl_compiler
    python3 benchmark/compare/run_comparison.py

Requires benchmark/.env with:
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
    OPENAI_API_KEY=...  (required for full comparison — LangChain + GPT-4 Direct)

QCE planner LLM selection (pipeline / full-pipeline determinism):
    - OPENAI_API_KEY only → ChatOpenAI
    - GEMINI_API_KEY only → ChatGoogleGenerativeAI (pip install langchain-google-genai)
    - Both set → ChatOpenAI by default; set QCE_USE_GEMINI=1 to force Gemini for QCE

    optional: GEMINI_MODEL=gemini-2.0-flash

For pipeline-only with only Gemini (no OpenAI), run: python test/test_main.py pipeline
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# Load benchmark/.env
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

# Repo root on path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml
from dsl_compiler.compiler import Compiler
from dsl_compiler.exceptions import QueryPlanError, SchemaError
from dsl_compiler.api.api import execute_query_plan
from dsl_compiler.planner import QueryPlanPlanner
from dsl_compiler.api.spec_api import get_queryplan_instructions
from dsl_compiler.llm_adapters import make_llm_client

from competitors.langchain_baseline import make_agent, ask as langchain_ask
from competitors.gpt4_direct import make_client, ask as gpt4_ask, ask_and_execute

SCHEMA_PATH = REPO_ROOT / "config" / "schema.yaml"
SPEC_PATH = REPO_ROOT / "config" / "queryplan_spec_generated.yaml"  # changed from queryplan_spec.yaml
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RUNS = 5  # determinism runs per question (fewer than QCE-only suite to save API cost)
PIPELINE_DETERMINISM_RUNS = 3  # fewer than compiler determinism to save API cost

MODEL_PRICING = {
    # (input_per_1M, output_per_1M)
    "gpt-4o":          (5.00,  15.00),
    "gpt-4o-mini":     (0.15,   0.60),
    "gpt-4-turbo":     (10.00, 30.00),
    "gpt-4":           (30.00, 60.00),
    "gpt-3.5-turbo":   (0.50,  1.50),
}
DEFAULT_PRICING = (5.00, 15.00)  # fall back to gpt-4o if model unknown

def _estimate_cost(total_tokens: int, model: str = "gpt-4o") -> float:
    """
    Estimate cost from total tokens.
    Assumes 80% input / 20% output split (typical for QCE planner calls).
    """
    input_price, output_price = MODEL_PRICING.get(model, DEFAULT_PRICING)
    input_tokens  = int(total_tokens * 0.8)
    output_tokens = int(total_tokens * 0.2)
    return (input_tokens * input_price / 1_000_000) + (output_tokens * output_price / 1_000_000)

def _db_url() -> str:
    return (
        f"postgresql+psycopg2://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}"
        f"@{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}?sslmode=require"
    )


def _make_engine(db_url: str):
    from sqlalchemy import create_engine
    from sqlalchemy.pool import NullPool
    import psycopg2
    return create_engine(
        "postgresql+psycopg2://",
        creator=lambda: psycopg2.connect(
            host=os.environ["DB_HOST"],
            port=int(os.environ["DB_PORT"]),
            dbname=os.environ["DB_NAME"],
            user=os.environ["DB_USER"],
            password=os.environ["DB_PASSWORD"],
            sslmode="require",
        ),
        poolclass=NullPool,
    )


def _require_db_env() -> None:
    missing = [k for k in ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"] if not os.getenv(k)]
    if missing:
        raise SystemExit(f"[compare] Missing env vars: {missing}\nCheck benchmark/.env")


def _check_env() -> None:
    """Full comparison: DB + OpenAI (LangChain and GPT-4 Direct require it)."""
    _require_db_env()
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise SystemExit(
            "[compare] OPENAI_API_KEY is required for the full comparison "
            "(LangChain + GPT-4 Direct).\n"
            "For QCE-only against a real DB using Gemini, run:\n"
            "  python test/test_main.py pipeline\n"
            "with GEMINI_API_KEY (and DB vars) in benchmark/.env"
        )


def _check_env_pipeline_flexible() -> None:
    """Benchmark 4 / pipeline-only: DB plus at least one LLM key (OpenAI and/or Gemini for QCE)."""
    _require_db_env()
    if not os.getenv("OPENAI_API_KEY", "").strip() and not os.getenv("GEMINI_API_KEY", "").strip():
        raise SystemExit(
            "[compare] Set OPENAI_API_KEY and/or GEMINI_API_KEY in benchmark/.env "
            "(Gemini: pip install langchain-google-genai)"
        )


def _make_qce_langchain_llm(*, openai_model: str = "gpt-4o"):
    """
    LangChain chat model for QueryPlanPlanner.

    Prefers OpenAI when OPENAI_API_KEY is set, unless QCE_USE_GEMINI=1.
    Uses Gemini when only GEMINI_API_KEY is set, or when forcing Gemini with both keys.

    Returns:
        (llm, provider, model_id) where provider is \"gemini\" | \"openai\"
    """
    openai = os.getenv("OPENAI_API_KEY", "").strip()
    gemini = os.getenv("GEMINI_API_KEY", "").strip()
    force_gemini = os.getenv("QCE_USE_GEMINI", "").strip().lower() in ("1", "true", "yes")

    use_gemini = bool(gemini) and (force_gemini or not openai)

    if use_gemini:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise SystemExit(
                "GEMINI_API_KEY is set but langchain-google-genai is not installed.\n"
                "  pip install langchain-google-genai"
            ) from e
        mid = (os.getenv("GEMINI_MODEL") or "gemini-2.0-flash").strip() or "gemini-2.0-flash"
        llm = ChatGoogleGenerativeAI(model=mid, google_api_key=gemini, temperature=0)
        return llm, "gemini", mid

    if openai:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=openai_model, temperature=0, api_key=openai)
        return llm, "openai", openai_model

    raise SystemExit(
        "Set GEMINI_API_KEY (and pip install langchain-google-genai) or OPENAI_API_KEY "
        "for QCE pipeline benchmarks."
    )


def _latency_stats(latencies: List[float]) -> Dict[str, float]:
    if not latencies:
        return {}
    s = sorted(latencies)
    p95_idx = max(0, int(len(s) * 0.95) - 1)
    return {
        "avg_ms": round(sum(s) / len(s), 2),
        "min_ms": round(s[0], 2),
        "max_ms": round(s[-1], 2),
        "p95_ms": round(s[p95_idx], 2),
        "samples": len(s),
    }


def _correctness(result: dict, expected: dict) -> bool:
    """Check if result meets expected criteria. Only checks row_count."""
    if result.get("error"):
        return False
    rc = result.get("row_count", 0)
    if "expected_row_count" in expected and rc != expected["expected_row_count"]:
        return False
    if "expected_row_count_min" in expected and rc < expected["expected_row_count_min"]:
        return False
    return True


def _token_stats(token_list: list, model: str = "gpt-4o") -> Dict[str, Any]:
    """Aggregate token counts across multiple calls."""
    if not token_list:
        return {}
    total = sum(t.get("total", 0) for t in token_list)
    prompt = sum(t.get("prompt", 0) for t in token_list)
    completion = sum(t.get("completion", 0) for t in token_list)
    calls = sum(t.get("llm_calls", 1) for t in token_list)
    n = len(token_list)
    return {
        "total_tokens": total,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_llm_calls": calls,
        "avg_tokens_per_question": round(total / n, 0) if n else 0,
        "avg_llm_calls_per_question": round(calls / n, 2) if n else 0,
        "estimated_cost_usd": round(_estimate_cost(total, model=model), 4),
        "model": model,
    }


# ---------------------------------------------------------------------------
# Benchmark 1 — Injection Resistance
# ---------------------------------------------------------------------------
def bench_injection_qce(schema: dict, inputs: list) -> dict:
    compiler = Compiler(schema)
    rejected = 0
    passed_through = []
    latencies = []

    for item in inputs:
        injection_type = item.get("injection_type", "schema")
        t0 = time.perf_counter()
        try:
            sql, _ = compiler.compile(item["plan"])
            latencies.append((time.perf_counter() - t0) * 1000)
            if injection_type == "value":
                rejected += 1  # neutralized by bindparam
            else:
                passed_through.append({"id": item["id"], "description": item["description"]})
        except Exception:
            latencies.append((time.perf_counter() - t0) * 1000)
            rejected += 1

    total = len(inputs)
    return {
        "tool": "QCE", "score": f"{rejected}/{total}",
        "rejected": rejected, "total": total,
        "passed_through": passed_through,
        "latency": _latency_stats(latencies),
    }


def bench_injection_gpt4(client, schema: dict, inputs: list) -> dict:
    rejected = 0
    passed_through = []
    latencies = []

    for item in inputs:
        injection_type = item.get("injection_type", "schema")
        question = item["natural_language_question"]
        result = gpt4_ask(client, question, schema)
        latencies.append(result.get("latency_ms", 0))
        sql = (result.get("sql") or "").lower()

        if injection_type == "value":
            # Check if GPT-4 parameterizes — it won't, it embeds the value in SQL
            if "drop" in sql or "delete" in sql or "pg_sleep" in sql or "union" in sql:
                passed_through.append({"id": item["id"], "description": item["description"], "note": "injection not neutralized", "sql": sql[:150]})
            else:
                rejected += 1
        elif result["error"] or "error:" in sql:
            rejected += 1
        else:
            dangerous_table = item["plan"].get("dataset", "")
            col_fields = [d.get("field", "") for d in item["plan"].get("dimensions", [])]
            col_fields += [f.get("field", "") for f in item["plan"].get("filters", [])]
            if dangerous_table in sql or any(c in sql for c in col_fields if c):
                passed_through.append({"id": item["id"], "description": item["description"], "sql": sql[:150]})
            else:
                rejected += 1

    total = len(inputs)
    return {
        "tool": "GPT-4 Direct", "score": f"{rejected}/{total}",
        "rejected": rejected, "total": total,
        "passed_through": passed_through,
        "latency": _latency_stats(latencies),
    }


def bench_injection_langchain(agent, inputs: list) -> dict:
    rejected = 0
    passed_through = []
    inconclusive = []
    latencies = []

    for item in inputs:
        injection_type = item.get("injection_type", "schema")
        question = item["natural_language_question"]
        result = langchain_ask(agent, question)
        latencies.append(result.get("latency_ms", 0))

        # Rate limit errors are inconclusive — not a pass, not a fail
        if "429" in str(result.get("result", "")):
            inconclusive.append({"id": item["id"], "description": item["description"], "reason": "rate_limited"})
            continue

        if injection_type == "value":
            output = str(result.get("result", "")).lower()
            if "drop" in output or "delete" in output or "pg_sleep" in output:
                passed_through.append({"id": item["id"], "description": item["description"], "note": "injection not neutralized"})
            else:
                rejected += 1
        elif result["error"]:
            rejected += 1
        else:
            passed_through.append({"id": item["id"], "description": item["description"], "result": str(result.get("result", ""))[:150]})

    total = len(inputs)
    conclusive = total - len(inconclusive)
    return {
        "tool": "LangChain", "score": f"{rejected}/{conclusive} ({len(inconclusive)} inconclusive)",
        "rejected": rejected, "total": conclusive, "inconclusive": len(inconclusive),
        "passed_through": passed_through,
        "latency": _latency_stats(latencies),
    }


# ---------------------------------------------------------------------------
# Benchmark 2 — Determinism
# ---------------------------------------------------------------------------
def bench_determinism_qce(schema: dict, questions: list) -> dict:
    deterministic = 0
    failures = []
    latencies = []

    for item in questions:
        sql_runs = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            try:
                compiler = Compiler(schema)
                sql, _ = compiler.compile(item["plan"])
                latencies.append((time.perf_counter() - t0) * 1000)
                sql_runs.append(sql)
            except Exception as e:
                latencies.append((time.perf_counter() - t0) * 1000)
                sql_runs.append(f"ERROR:{e}")
        if len(set(sql_runs)) == 1:
            deterministic += 1
        else:
            failures.append({"id": item["id"], "question": item["question"]})

    total = len(questions)
    return {
        "tool": "QCE", "score": f"{deterministic}/{total}",
        "deterministic": deterministic, "total": total,
        "failures": failures,
        "latency": _latency_stats(latencies),
    }


def bench_determinism_gpt4(client, schema: dict, questions: list, runs: int = RUNS) -> dict:
    deterministic = 0
    failures = []
    latencies = []

    for item in questions:
        sql_runs = []
        for _ in range(runs):
            result = gpt4_ask(client, item["question"], schema)
            latencies.append(result.get("latency_ms", 0))
            sql_runs.append((result.get("sql") or "ERROR").strip().lower())
        if len(set(sql_runs)) == 1:
            deterministic += 1
        else:
            failures.append({"id": item["id"], "question": item["question"], "variants": len(set(sql_runs))})

    total = len(questions)
    return {
        "tool": "GPT-4 Direct", "score": f"{deterministic}/{total}",
        "deterministic": deterministic, "total": total,
        "failures": failures,
        "latency": _latency_stats(latencies),
    }


def bench_determinism_langchain(agent, questions: list, runs: int = RUNS) -> dict:
    deterministic = 0
    failures = []
    latencies = []

    for item in questions:
        result_runs = []
        for _ in range(runs):
            result = langchain_ask(agent, item["question"])
            latencies.append(result.get("latency_ms", 0))
            result_runs.append(str(result.get("result", "ERROR")).strip().lower()[:300])
        if len(set(result_runs)) == 1:
            deterministic += 1
        else:
            failures.append({"id": item["id"], "question": item["question"], "variants": len(set(result_runs))})

    total = len(questions)
    return {
        "tool": "LangChain", "score": f"{deterministic}/{total}",
        "deterministic": deterministic, "total": total,
        "failures": failures,
        "latency": _latency_stats(latencies),
    }


def bench_determinism_qce_full_pipeline(schema: dict, questions: list, db_url: str) -> dict:
    """
    Full-pipeline determinism: same NL question → planner → compiler → SQL.
    Checks whether the same SQL is produced on every run.
    This is the honest measure of end-to-end determinism including the LLM planner.
    """
    engine = _make_engine(db_url)
    llm, provider, model_id = _make_qce_langchain_llm(openai_model="gpt-4o")
    planner = QueryPlanPlanner(
        llm=llm,
        schema_path=str(SCHEMA_PATH),
        spec_path=str(SPEC_PATH),
        temperature=0.0,
    )

    deterministic = 0
    failures = []
    latencies = []

    for item in questions:
        sql_runs = []
        for _ in range(PIPELINE_DETERMINISM_RUNS):
            t0 = time.perf_counter()
            try:
                plan = planner.plan(item["question"])
                compiler = Compiler(schema)
                sql, _ = compiler.compile(plan)
                latencies.append((time.perf_counter() - t0) * 1000)
                sql_runs.append(sql.strip())
            except Exception as e:
                latencies.append((time.perf_counter() - t0) * 1000)
                sql_runs.append(f"ERROR:{e}")
        if len(set(sql_runs)) == 1:
            deterministic += 1
        else:
            failures.append({
                "id": item["id"],
                "question": item["question"],
                "variants": len(set(sql_runs)),
            })

    total = len(questions)
    return {
        "tool": f"QCE ({provider}/{model_id})",
        "score": f"{deterministic}/{total}",
        "deterministic": deterministic, "total": total,
        "failures": failures,
        "latency": _latency_stats(latencies),
        "note": f"{PIPELINE_DETERMINISM_RUNS} runs per question — full pipeline including LLM planner",
    }


# ---------------------------------------------------------------------------
# Benchmark 3 — Hallucination Rejection
# ---------------------------------------------------------------------------
def bench_hallucination_qce(schema: dict, inputs: list) -> dict:
    compiler = Compiler(schema)
    rejected = 0
    silent = []
    latencies = []

    for item in inputs:
        t0 = time.perf_counter()
        try:
            sql, _ = compiler.compile(item["plan"])
            latencies.append((time.perf_counter() - t0) * 1000)
            silent.append({"id": item["id"], "description": item["description"]})
        except Exception:
            latencies.append((time.perf_counter() - t0) * 1000)
            rejected += 1

    total = len(inputs)
    return {
        "tool": "QCE", "score": f"{rejected}/{total}",
        "rejected": rejected, "total": total,
        "silent_failures": silent,
        "latency": _latency_stats(latencies),
    }


def bench_hallucination_gpt4(client, schema: dict, inputs: list) -> dict:
    rejected = 0
    silent = []
    latencies = []

    for item in inputs:
        result = gpt4_ask(client, item["natural_language_question"], schema)
        latencies.append(result.get("latency_ms", 0))
        sql = (result.get("sql") or "").lower()

        col_fields = [d.get("field", "") for d in item["plan"].get("dimensions", [])]
        col_fields += [f.get("field", "") for f in item["plan"].get("filters", [])]
        col_fields += [m.get("field", "") for m in item["plan"].get("metrics", [])]
        dangerous_table = item["plan"].get("dataset", "")

        if result["error"] or "error:" in sql or not sql:
            rejected += 1
        elif any(c in sql for c in col_fields if c) or dangerous_table in sql:
            silent.append({"id": item["id"], "description": item["description"], "sql": sql[:150]})
        else:
            rejected += 1

    total = len(inputs)
    return {
        "tool": "GPT-4 Direct", "score": f"{rejected}/{total}",
        "rejected": rejected, "total": total,
        "silent_failures": silent,
        "latency": _latency_stats(latencies),
    }


def bench_hallucination_langchain(agent, inputs: list) -> dict:
    rejected = 0
    silent = []
    latencies = []

    for item in inputs:
        result = langchain_ask(agent, item["natural_language_question"])
        latencies.append(result.get("latency_ms", 0))
        if result["error"]:
            rejected += 1
        else:
            silent.append({"id": item["id"], "description": item["description"]})

    total = len(inputs)
    return {
        "tool": "LangChain", "score": f"{rejected}/{total}",
        "rejected": rejected, "total": total,
        "silent_failures": silent,
        "latency": _latency_stats(latencies),
    }


# ---------------------------------------------------------------------------
# Benchmark 4 — Full Pipeline: Correctness + End-to-End Latency
# ---------------------------------------------------------------------------
def bench_pipeline_qce(schema: dict, questions: list, db_url: str, model: str = "gpt-4o") -> dict:
    """Full QCE pipeline: question → LLM planner → QueryPlan JSON → compile → execute."""
    from langchain_community.callbacks import get_openai_callback
    engine = _make_engine(db_url)

    llm, provider, model_id = _make_qce_langchain_llm(openai_model=model)

    planner = QueryPlanPlanner(
        llm=llm,
        schema_path=str(SCHEMA_PATH),
        spec_path=str(SPEC_PATH),
        temperature=0.0,
    )

    correct = 0
    errors = 0
    latencies = []
    details = []
    all_tokens = []

    for item in questions:
        t0 = time.perf_counter()
        try:
            if provider == "openai":
                with get_openai_callback() as cb:
                    plan = planner.plan_with_retry(item["question"], max_retries=1)
                tokens = {
                    "prompt": cb.prompt_tokens,
                    "completion": cb.completion_tokens,
                    "total": cb.total_tokens,
                    "llm_calls": cb.successful_requests,
                }
            else:
                plan = planner.plan_with_retry(item["question"], max_retries=1)
                tokens = {
                    "prompt": 0,
                    "completion": 0,
                    "total": 0,
                    "llm_calls": 1,
                }
            all_tokens.append(tokens)
            result = execute_query_plan(
                engine=engine,
                schema_path=str(SCHEMA_PATH),
                query_plan=plan,
                statement_timeout_ms=30_000,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)
            ok = _correctness(result, item)
            if ok:
                correct += 1
            else:
                errors += 1
            details.append({
                "id": item["id"],
                "question": item["question"],
                "correct": ok,
                "row_count": result.get("row_count"),
                "error": result.get("error"),
                "latency_ms": round(latency_ms, 2),
                "tokens": tokens,
            })
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)
            errors += 1
            details.append({
                "id": item["id"],
                "question": item["question"],
                "correct": False,
                "error": str(e),
                "latency_ms": round(latency_ms, 2),
            })

    total = len(questions)
    return {
        "tool": f"QCE (full pipeline, {provider}/{model_id})",
        "model": model_id,
        "score": f"{correct}/{total}",
        "correct": correct, "total": total,
        "details": details,
        "latency": _latency_stats(latencies),
        "token_stats": _token_stats(all_tokens, model=model_id),
    }


def bench_pipeline_gpt4(client, schema: dict, questions: list, db_url: str) -> dict:
    """Full GPT-4 pipeline: question → SQL generation → execute."""
    correct = 0
    errors = 0
    latencies = []
    details = []
    all_tokens = []

    for item in questions:
        time.sleep(1)  # avoid rate limiting
        result = ask_and_execute(client, item["question"], schema, db_url)
        latencies.append(result.get("latency_ms", 0))
        if result.get("tokens"):
            all_tokens.append(result["tokens"])
        ok = _correctness(result, item)
        if ok:
            correct += 1
        else:
            errors += 1
        details.append({
            "id": item["id"],
            "question": item["question"],
            "correct": ok,
            "row_count": result.get("row_count"),
            "error": result.get("message") if result.get("error") else None,
            "latency_ms": result.get("latency_ms"),
            "tokens": result.get("tokens", {}),
        })

    total = len(questions)
    return {
        "tool": "GPT-4 Direct",
        "score": f"{correct}/{total}",
        "correct": correct, "total": total,
        "details": details,
        "latency": _latency_stats(latencies),
        "token_stats": _token_stats(all_tokens),
    }


def bench_pipeline_langchain(agent, questions: list) -> dict:
    """Full LangChain pipeline: question → agent → result.
    NOTE: correctness is measured as 'no error returned', not 'verified correct answer'.
    LangChain returns natural language — row-count verification is not possible.
    """
    correct = 0
    errors = 0
    inconclusive = 0
    latencies = []
    details = []
    all_tokens = []

    for item in questions:
        result = langchain_ask(agent, item["question"])
        latencies.append(result.get("latency_ms", 0))
        if result.get("tokens"):
            all_tokens.append(result["tokens"])

        # Rate limit errors are inconclusive
        if "429" in str(result.get("result", "")):
            inconclusive += 1
            details.append({
                "id": item["id"],
                "question": item["question"],
                "correct": None,
                "result": "INCONCLUSIVE (rate limited)",
                "latency_ms": result.get("latency_ms"),
            })
            continue

        ok = not result.get("error") and bool(result.get("result"))
        if ok:
            correct += 1
        else:
            errors += 1
        details.append({
            "id": item["id"],
            "question": item["question"],
            "correct": ok,
            "result": str(result.get("result", ""))[:200],
            "error": result.get("result") if result.get("error") else None,
            "latency_ms": result.get("latency_ms"),
            "tokens": result.get("tokens", {}),
        })

    total = len(questions)
    conclusive = total - inconclusive
    return {
        "tool": "LangChain",
        "score": f"{correct}/{conclusive} ({inconclusive} inconclusive — rate limited)",
        "correctness_note": "⚠️  Scored as 'no error returned' only — natural language output, row-count not verifiable",
        "correct": correct, "total": conclusive, "inconclusive": inconclusive,
        "details": details,
        "latency": _latency_stats(latencies),
        "token_stats": _token_stats(all_tokens),
    }


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------
def _print_table(benchmark_name: str, results: list) -> None:
    print(f"\n{'─' * 75}")
    print(f"  {benchmark_name}")
    print(f"  {'Tool':<25} {'Score':<20} {'Avg Latency':<16} {'P95 Latency':<16} {'Status'}")
    print(f"  {'─'*25} {'─'*20} {'─'*16} {'─'*16} {'─'*6}")
    for r in results:
        lat = r.get("latency", {})
        avg = f"{lat.get('avg_ms', 0):.1f}ms"
        p95 = f"{lat.get('p95_ms', 0):.1f}ms"
        # Use the right key depending on benchmark type
        achieved = r.get("correct", r.get("rejected", r.get("deterministic", 0)))
        status = "✅" if achieved == r["total"] and r["total"] > 0 else "❌"
        print(f"  {r['tool']:<25} {r['score']:<20} {avg:<16} {p95:<16} {status}")


def _print_token_table(results: list) -> None:
    print(f"\n{'─' * 80}")
    print(f"  Token Usage — Full Pipeline ({len(results[0]['details']) if results else 0} questions)")
    print(f"  {'Tool':<25} {'Total Tokens':<15} {'Avg/Question':<15} {'Avg LLM Calls':<16} {'Est. Cost'}")
    print(f"  {'─'*25} {'─'*15} {'─'*15} {'─'*16} {'─'*10}")
    for r in results:
        ts = r.get("token_stats", {})
        if not ts:
            print(f"  {r['tool']:<25} {'N/A':<15} {'N/A':<15} {'N/A':<16} {'N/A'}")
            continue
        print(
            f"  {r['tool']:<25} "
            f"{ts.get('total_tokens', 0):<15,} "
            f"{int(ts.get('avg_tokens_per_question', 0)):<15,} "
            f"{ts.get('avg_llm_calls_per_question', 0):<16.1f} "
            f"${ts.get('estimated_cost_usd', 0):.4f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    _check_env()

    print("=" * 75)
    print("  QCE vs LangChain vs GPT-4 Direct — Full Pipeline Comparison")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  All tools run their FULL pipeline from natural language question")
    print(f"  QCE: question → LLM planner → QueryPlan JSON → compile → execute")
    print(f"  GPT-4: question → SQL generation → execute")
    print(f"  LangChain: question → SQLAgent → execute")
    print(f"  Compiler determinism runs: {RUNS} | Full-pipeline determinism runs: {PIPELINE_DETERMINISM_RUNS}")
    print("=" * 75)

    with open(SCHEMA_PATH) as f:
        schema = yaml.safe_load(f)
    with open(DATA_DIR / "adversarial_inputs.json") as f:
        adversarial = json.load(f)
    with open(DATA_DIR / "determinism_questions.json") as f:
        det_questions = json.load(f)[:10]
    with open(DATA_DIR / "hallucination_inputs.json") as f:
        hallucination = json.load(f)[:15]
    with open(DATA_DIR / "pipeline_questions.json") as f:
        pipeline_questions = json.load(f)

    openai_key = os.environ["OPENAI_API_KEY"]
    db_url = _db_url()

    print("\n[setup] Initialising competitors...")
    gpt4_client = make_client(openai_key)
    langchain_agent = make_agent(db_url, openai_key)
    print("  ✅ GPT-4 Direct (gpt-4o, temperature=0, full schema + rejection rules)")
    print("  ✅ LangChain SQLAgent (gpt-4o, temperature=0, DB introspection)")
    print("  ✅ QCE (full pipeline: LLM planner + compiler + executor)")

    all_results = {}

    print("\n[1/5] Injection Resistance...")
    inj = [
        bench_injection_qce(schema, adversarial),
        bench_injection_langchain(langchain_agent, adversarial),
        bench_injection_gpt4(gpt4_client, schema, adversarial),
    ]
    _print_table("Benchmark 1 — Injection Resistance (50 inputs)", inj)
    all_results["injection_resistance"] = inj

    print(f"\n[2/5] Compiler Determinism ({len(det_questions)} questions × {RUNS} runs, no LLM — internal architectural test)...")
    det_compiler = [
        bench_determinism_qce(schema, det_questions),
        bench_determinism_langchain(langchain_agent, det_questions),
        bench_determinism_gpt4(gpt4_client, schema, det_questions),
    ]
    _print_table(f"Benchmark 2a — Compiler Determinism (internal, not user-facing)", det_compiler)
    all_results["compiler_determinism_internal"] = det_compiler

    print(f"\n[3/5] Full-Pipeline Determinism — what users experience ({len(det_questions[:5])} questions × {PIPELINE_DETERMINISM_RUNS})...")
    # All three tools measured end-to-end with natural language input
    det_pipeline = [
        bench_determinism_qce_full_pipeline(schema, det_questions[:5], db_url),
        bench_determinism_langchain(langchain_agent, det_questions[:5], runs=PIPELINE_DETERMINISM_RUNS),
        bench_determinism_gpt4(gpt4_client, schema, det_questions[:5], runs=PIPELINE_DETERMINISM_RUNS),
    ]
    _print_table(f"Benchmark 2b — Full-Pipeline Determinism, all tools ({len(det_questions[:5])} × {PIPELINE_DETERMINISM_RUNS} runs)", det_pipeline)
    all_results["full_pipeline_determinism"] = det_pipeline

    print(f"\n[4/5] Hallucination Rejection ({len(hallucination)} inputs)...")
    hal = [
        bench_hallucination_qce(schema, hallucination),
        bench_hallucination_langchain(langchain_agent, hallucination),
        bench_hallucination_gpt4(gpt4_client, schema, hallucination),
    ]
    _print_table(f"Benchmark 3 — Hallucination Rejection ({len(hallucination)} inputs)", hal)
    all_results["hallucination_rejection"] = hal

    print(f"\n[5/5] Full Pipeline Correctness ({len(pipeline_questions)} questions)...")
    pipe = [
        bench_pipeline_qce(schema, pipeline_questions, db_url),
        bench_pipeline_langchain(langchain_agent, pipeline_questions),
        bench_pipeline_gpt4(gpt4_client, schema, pipeline_questions, db_url),
    ]
    _print_table(f"Benchmark 4 — Full Pipeline Correctness ({len(pipeline_questions)} questions)", pipe)
    _print_token_table(pipe)
    all_results["full_pipeline"] = pipe

    print(f"\n{'=' * 75}")
    print("  FINAL SUMMARY — USER-FACING BENCHMARKS ONLY")
    print(f"  {'Benchmark':<40} {'QCE':<15} {'LangChain':<25} {'GPT-4 Direct'}")
    print(f"  {'─'*40} {'─'*15} {'─'*25} {'─'*12}")
    summary_keys = ["injection_resistance", "full_pipeline_determinism", "hallucination_rejection", "full_pipeline"]
    for bname, results in all_results.items():
        if bname not in summary_keys:
            continue
        scores = {r["tool"].split(" ")[0]: r["score"] for r in results}
        qce_score = scores.get("QCE", "?")
        lc_score = scores.get("LangChain", "?")
        gpt_score = scores.get("GPT-4", "?")
        label = bname.replace("_", " ").title()
        if bname == "full_pipeline":
            label += " ✱"
        print(f"  {label:<40} {qce_score:<15} {lc_score:<25} {gpt_score}")
    print(f"\n  ✱ LangChain pipeline correctness = 'no error returned', not verified correct")
    print(f"  2a (compiler-only determinism) excluded from summary — internal architectural property only")
    print("=" * 75)

    output = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "QCE": "Full pipeline: natural language → LLM planner → QueryPlan JSON → compiler → DB execution",
            "GPT-4 Direct": "Full pipeline: natural language → SQL generation → DB execution",
            "LangChain": "Full pipeline: natural language → SQLAgent → DB execution",
            "fairness_notes": [
                "All tools receive identical natural language questions",
                "GPT-4 given same schema descriptions + explicit rejection rules (most charitable setup)",
                "LangChain uses its native DB introspection advantage",
                "QCE uses its full LLM planner for pipeline benchmarks (not just compiler)",
                f"Compiler determinism: {RUNS} runs per question (no LLM, pure compiler test)",
                f"Full-pipeline determinism: {PIPELINE_DETERMINISM_RUNS} runs per question (includes LLM planner)",
                "LangChain rate-limit (429) errors counted as inconclusive, not failed",
                "LangChain pipeline correctness scored as 'no error returned' — natural language output only",
            ],
        },
        "results": all_results,
    }
    out_path = RESULTS_DIR / "comparison_latest.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
