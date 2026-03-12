from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml

from dsl_compiler.compiler import Compiler
from dsl_compiler.exceptions import QueryPlanError, SchemaError

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "hallucination_inputs.json"
SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "schema.yaml"


def run() -> Dict[str, Any]:
    with open(SCHEMA_PATH) as f:
        schema = yaml.safe_load(f)

    with open(DATA_PATH) as f:
        inputs = json.load(f)

    compiler = Compiler(schema)
    total = len(inputs)
    rejected = 0
    silent_failures = []

    for item in inputs:
        plan = item["plan"]
        try:
            sql, _ = compiler.compile(plan)
            # Compiled successfully — hallucination was NOT caught
            silent_failures.append({
                "id": item["id"],
                "description": item["description"],
                "sql": sql[:200],
            })
        except (QueryPlanError, SchemaError, Exception):
            rejected += 1

    return {
        "benchmark": "hallucination_rejection",
        "total": total,
        "rejected": rejected,
        "silent_failures": silent_failures,
        "score": f"{rejected}/{total}",
        "passed": rejected == total,
    }
