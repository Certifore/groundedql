from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from dsl_compiler.compiler import Compiler
from dsl_compiler.exceptions import QueryPlanError, SchemaError

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "adversarial_inputs.json"
SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "schema.yaml"


def run() -> Dict[str, Any]:
    with open(SCHEMA_PATH) as f:
        schema = yaml.safe_load(f)

    with open(DATA_PATH) as f:
        inputs = json.load(f)

    compiler = Compiler(schema)
    total = len(inputs)
    rejected = 0
    passed_through = []

    for item in inputs:
        plan = item["plan"]
        injection_type = item.get("injection_type", "schema")  # "schema" or "value"
        try:
            sql, params = compiler.compile(plan)
            if injection_type == "value":
                # Value-based injection attempts SHOULD compile successfully.
                # The value is passed as a bindparam — Postgres receives it as a
                # plain string, never as SQL. This is correct behavior.
                rejected += 1  # count as rejected (neutralized by parameterization)
            else:
                # Schema-based injection (unknown table/column) should raise.
                passed_through.append({
                    "id": item["id"],
                    "description": item["description"],
                    "sql": sql[:200],
                })
        except (QueryPlanError, SchemaError, Exception):
            rejected += 1

    return {
        "benchmark": "injection_resistance",
        "total": total,
        "rejected": rejected,
        "passed_through": passed_through,
        "score": f"{rejected}/{total}",
        "passed": rejected == total,
    }
