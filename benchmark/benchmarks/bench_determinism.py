from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml

from dsl_compiler.compiler import Compiler

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "determinism_questions.json"
SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "schema.yaml"
RUNS = 10


def run() -> Dict[str, Any]:
    with open(SCHEMA_PATH) as f:
        schema = yaml.safe_load(f)

    with open(DATA_PATH) as f:
        questions = json.load(f)

    total = len(questions)
    deterministic = 0
    failures = []

    for item in questions:
        plan = item["plan"]
        sql_runs = []

        for _ in range(RUNS):
            compiler = Compiler(schema)
            try:
                sql, _ = compiler.compile(plan)
                sql_runs.append(sql)
            except Exception as e:
                sql_runs.append(f"ERROR: {e}")

        unique = set(sql_runs)
        if len(unique) == 1:
            deterministic += 1
        else:
            failures.append({
                "id": item["id"],
                "question": item["question"],
                "unique_sql_variants": len(unique),
                "samples": list(unique)[:2],
            })

    return {
        "benchmark": "determinism",
        "total": total,
        "runs_per_question": RUNS,
        "deterministic": deterministic,
        "failures": failures,
        "score": f"{deterministic}/{total}",
        "passed": deterministic == total,
    }
