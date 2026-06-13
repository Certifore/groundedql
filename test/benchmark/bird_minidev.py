"""
Run GroundedQL against BIRD-SQL Mini-Dev.

This harness intentionally lives under test/benchmark so it is separate from the
normal regression tests. It expects the BIRD dataset and databases to be provided
locally; it does not download data.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

import groundedql
from groundedql.llm_adapters import env_value


DEFAULT_RESULT_PATH = Path(__file__).resolve().parent / "results" / "bird_minidev_latest.json"


@dataclass(frozen=True)
class BirdExample:
    idx: int
    db_id: str
    question: str
    sql: str
    evidence: str = ""


def _candidate_data_files(root: Path) -> list[Path]:
    return [
        root / "mini_dev_postgresql.json",
        root / "mini_dev_pg.json",
        root / "mini_dev_postgres.json",
        root / "mini_dev_sqlite.json",
        root / "MINIDEV" / "mini_dev_postgresql.json",
        root / "MINIDEV" / "mini_dev_pg.json",
        root / "MINIDEV" / "mini_dev_postgres.json",
        root / "MINIDEV" / "mini_dev_sqlite.json",
        root / "minidev" / "MINIDEV" / "mini_dev_postgresql.json",
        root / "minidev" / "MINIDEV" / "mini_dev_pg.json",
        root / "minidev" / "MINIDEV" / "mini_dev_postgres.json",
        root / "minidev" / "MINIDEV" / "mini_dev_sqlite.json",
        root / "mini_dev_data" / "mini_dev_postgresql.json",
        root / "mini_dev_data" / "mini_dev_pg.json",
        root / "mini_dev_data" / "mini_dev_postgres.json",
        root / "mini_dev_data" / "mini_dev_sqlite.json",
    ]


def find_data_file(root: Path, explicit: str | None = None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"BIRD data file not found: {path}")
        return path

    for path in _candidate_data_files(root):
        if path.exists():
            return path

    tried = "\n  ".join(str(p) for p in _candidate_data_files(root))
    raise FileNotFoundError(f"No BIRD Mini-Dev JSON file found. Tried:\n  {tried}")


def load_examples(data_file: Path) -> list[BirdExample]:
    raw = json.loads(data_file.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        # Some dataset wrappers expose a top-level split key.
        for key in ("mini_dev_pg", "mini_dev_postgresql", "data", "examples"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break

    if not isinstance(raw, list):
        raise ValueError(f"Expected a list of BIRD examples in {data_file}")

    examples = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        db_id = str(item.get("db_id") or item.get("database_id") or "").strip()
        question = str(item.get("question") or "").strip()
        sql = str(item.get("SQL") or item.get("sql") or item.get("query") or "").strip()
        evidence = str(item.get("evidence") or "").strip()
        if not db_id or not question or not sql:
            continue
        examples.append(BirdExample(idx=idx, db_id=db_id, question=question, sql=sql, evidence=evidence))

    return examples


def schema_path_for(db_id: str, schema_dir: Path) -> Path | None:
    candidates = [
        schema_dir / db_id / "schema.yaml",
        schema_dir / f"{db_id}.yaml",
        schema_dir / f"{db_id}.yml",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def db_url_for(db_id: str, args: argparse.Namespace) -> str:
    template = args.db_url_template or env_value("BIRD_DB_URL_TEMPLATE")
    if template:
        return template.format(db_id=db_id)

    url = args.db_url or env_value("BIRD_DB_URL")
    if not url:
        raise ValueError("Set BIRD_DB_URL or BIRD_DB_URL_TEMPLATE, or pass --db-url.")
    return url


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def normalize_rows(rows: Iterable[Any], *, ordered: bool = False) -> list[tuple[Any, ...]]:
    normalized = []
    for row in rows:
        values = tuple(_jsonable(v) for v in tuple(row))
        normalized.append(values)
    if ordered:
        return normalized
    return sorted(normalized, key=lambda r: json.dumps(r, sort_keys=True, default=str))


def rows_match(gold_rows: Iterable[Any], actual_rows: Iterable[Any], *, ordered: bool = False) -> bool:
    gold = normalize_rows(gold_rows, ordered=ordered)
    actual = normalize_rows(actual_rows, ordered=ordered)
    if len(gold) != len(actual):
        return False
    return all(_row_values_match(g, a) for g, a in zip(gold, actual))


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _values_match(left: Any, right: Any) -> bool:
    left_num = _numeric_value(left)
    right_num = _numeric_value(right)
    if left_num is not None and right_num is not None:
        return math.isclose(left_num, right_num, rel_tol=1e-4, abs_tol=1e-4)
    return left == right


def _row_values_match(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    if len(left) != len(right):
        return False
    return all(_values_match(lv, rv) for lv, rv in zip(left, right))


def rows_sample(rows: Iterable[Any], *, limit: int) -> list[list[Any]]:
    if limit <= 0:
        return []
    sample = []
    for row in rows:
        sample.append([_jsonable(v) for v in tuple(row)])
        if len(sample) >= limit:
            break
    return sample


def execute_sql(engine: Engine, sql: str) -> list[Any]:
    if not sql.strip().lower().startswith(("select", "with")):
        raise ValueError("Refusing to execute non-SELECT gold SQL.")
    sql = _escape_dbapi_percent_in_string_literals(sql)
    with engine.connect() as conn:
        return conn.exec_driver_sql(sql).fetchall()


def _escape_dbapi_percent_in_string_literals(sql: str) -> str:
    """Escape literal % inside SQL string literals for pyformat DBAPI drivers."""
    out: list[str] = []
    in_single = False
    i = 0
    while i < len(sql):
        ch = sql[i]
        if ch == "'":
            out.append(ch)
            if in_single and i + 1 < len(sql) and sql[i + 1] == "'":
                out.append(sql[i + 1])
                i += 2
                continue
            in_single = not in_single
            i += 1
            continue
        if ch == "%" and in_single:
            out.append("%%")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def question_with_evidence(example: BirdExample, *, include_evidence: bool) -> str:
    if include_evidence and example.evidence:
        return f"{example.question}\n\nEvidence: {example.evidence}"
    return example.question


def build_report(
    *,
    data_file: Path,
    schema_dir: Path,
    llm: str,
    planned_total: int,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = Counter(r["status"] for r in results)
    attempted = counts["correct"] + counts["wrong"] + counts["error"]
    accuracy = (counts["correct"] / attempted) if attempted else 0.0
    return {
        "data_file": str(data_file),
        "schema_dir": str(schema_dir),
        "llm": llm,
        "planned_total": planned_total,
        "total": len(results),
        "attempted": attempted,
        "correct": counts["correct"],
        "wrong": counts["wrong"],
        "errors": counts["error"],
        "skipped": counts["skipped"],
        "execution_accuracy": round(accuracy, 4),
        "results": results,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def load_existing_results(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    existing = {}
    for result in raw.get("results", []):
        db_id = result.get("db_id")
        idx = result.get("id")
        if isinstance(db_id, str) and isinstance(idx, int):
            existing[(db_id, idx)] = result
    return existing


def run_example(
    example: BirdExample,
    *,
    engine: Engine,
    agent: groundedql.QueryAgent,
    include_evidence: bool,
    ordered: bool,
    sample_rows: int,
    quiet_agent: bool,
) -> dict[str, Any]:
    started = time.perf_counter()

    gold_rows = execute_sql(engine, example.sql)
    question = question_with_evidence(example, include_evidence=include_evidence)
    if quiet_agent:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            actual = agent.ask(question)
    else:
        actual = agent.ask(question)
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    if actual.get("error"):
        return {
            "id": example.idx,
            "db_id": example.db_id,
            "status": "error",
            "question": example.question,
            "error": actual["error"],
            "gold_sql": example.sql,
            "gold_row_count": len(gold_rows),
            "gold_rows_sample": rows_sample(gold_rows, limit=sample_rows),
            "latency_ms": latency_ms,
        }

    actual_rows = [tuple(row.values()) for row in actual.get("rows", [])]
    actual_columns = list(actual.get("rows", [{}])[0].keys()) if actual.get("rows") else []
    correct = rows_match(gold_rows, actual_rows, ordered=ordered)
    return {
        "id": example.idx,
        "db_id": example.db_id,
        "status": "correct" if correct else "wrong",
        "question": example.question,
        "evidence": example.evidence,
        "gold_sql": example.sql,
        "gold_row_count": len(gold_rows),
        "gold_rows_sample": rows_sample(gold_rows, limit=sample_rows),
        "actual_row_count": actual.get("row_count"),
        "actual_columns": actual_columns,
        "actual_rows_sample": rows_sample(actual_rows, limit=sample_rows),
        "sql": actual.get("sql"),
        "latency_ms": latency_ms,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GroundedQL on BIRD Mini-Dev.")
    parser.add_argument("--bird-root", default=env_value("BIRD_MINIDEV_ROOT") or "test/benchmark/bird_minidev")
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--schema-dir", default="test/benchmark/schemas")
    parser.add_argument("--db-url", default=None)
    parser.add_argument("--db-url-template", default=None)
    parser.add_argument("--llm", default="mistral")
    parser.add_argument("--db-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--ordered", action="store_true", help="Require row order to match.")
    parser.add_argument("--no-evidence", action="store_true", help="Do not append BIRD evidence to the question.")
    parser.add_argument("--dry-run", action="store_true", help="Load examples and schemas only; do not call DB or LLM.")
    parser.add_argument("--delay-seconds", type=float, default=float(env_value("BIRD_BENCHMARK_DELAY_SECONDS") or 0))
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true", help="Reuse completed examples from the output report.")
    parser.add_argument("--quiet-agent", action="store_true", help="Suppress GroundedQL planner logs during each example.")
    parser.add_argument("--sample-rows", type=int, default=5, help="Rows to include in diagnostic report samples.")
    parser.add_argument("--output", default=str(DEFAULT_RESULT_PATH))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    bird_root = Path(args.bird_root)
    schema_dir = Path(args.schema_dir)
    data_file = find_data_file(bird_root, args.data_file)
    examples = load_examples(data_file)

    if args.db_id:
        allowed = set(args.db_id)
        examples = [e for e in examples if e.db_id in allowed]
    if args.offset:
        examples = examples[args.offset:]
    if args.limit:
        examples = examples[:args.limit]

    by_db = Counter(e.db_id for e in examples)
    print(f"[bird] data_file={data_file}")
    print(f"[bird] examples={len(examples)} dbs={len(by_db)}")
    for db_id, count in sorted(by_db.items()):
        marker = "schema" if schema_path_for(db_id, schema_dir) else "missing schema"
        print(f"  {db_id}: {count} ({marker})")

    if args.dry_run:
        return 0

    results = []
    engines: dict[str, Engine] = {}
    agents: dict[tuple[str, str, str], groundedql.QueryAgent] = {}
    out_path = Path(args.output)
    existing_results = load_existing_results(out_path) if args.resume else {}
    if existing_results:
        print(f"[bird] resume={len(existing_results)} existing results from {out_path}")

    for i, example in enumerate(examples, start=1):
        existing = existing_results.get((example.db_id, example.idx))
        if existing is not None:
            results.append(existing)
            print(f"[{i}/{len(examples)}] {example.db_id} reused: {example.question[:90]}")
            continue

        schema_path = schema_path_for(example.db_id, schema_dir)
        if schema_path is None:
            result = {
                "id": example.idx,
                "db_id": example.db_id,
                "status": "skipped",
                "question": example.question,
                "reason": f"Missing schema for db_id '{example.db_id}' in {schema_dir}",
            }
            results.append(result)
            print(f"[{i}/{len(examples)}] {example.db_id} skipped: missing schema")
            continue

        try:
            db_url = db_url_for(example.db_id, args)
            engine = engines.setdefault(db_url, create_engine(db_url))
            agent_key = (db_url, str(schema_path), args.llm)
            if agent_key not in agents:
                if args.quiet_agent:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        agents[agent_key] = groundedql.QueryAgent(
                            engine=engine,
                            schema_path=str(schema_path),
                            llm=args.llm,
                        )
                else:
                    agents[agent_key] = groundedql.QueryAgent(
                        engine=engine,
                        schema_path=str(schema_path),
                        llm=args.llm,
                    )
            result = run_example(
                example,
                engine=engine,
                agent=agents[agent_key],
                include_evidence=not args.no_evidence,
                ordered=args.ordered,
                sample_rows=args.sample_rows,
                quiet_agent=args.quiet_agent,
            )
        except Exception as exc:
            result = {
                "id": example.idx,
                "db_id": example.db_id,
                "status": "error",
                "question": example.question,
                "gold_sql": example.sql,
                "error": str(exc),
            }

        results.append(result)
        print(f"[{i}/{len(examples)}] {example.db_id} {result['status']}: {example.question[:90]}")
        if args.delay_seconds > 0:
            time.sleep(args.delay_seconds)
        if args.checkpoint_every > 0 and len(results) % args.checkpoint_every == 0:
            report = build_report(
                data_file=data_file,
                schema_dir=schema_dir,
                llm=args.llm,
                planned_total=len(examples),
                results=results,
            )
            write_report(out_path, report)
            print(f"[bird] checkpoint wrote {out_path} ({len(results)}/{len(examples)})")

    report = build_report(
        data_file=data_file,
        schema_dir=schema_dir,
        llm=args.llm,
        planned_total=len(examples),
        results=results,
    )
    write_report(out_path, report)
    print(f"[bird] accuracy={report['execution_accuracy']} correct={report['correct']} attempted={report['attempted']}")
    print(f"[bird] wrote {out_path}")
    return 0 if report["wrong"] == 0 and report["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
