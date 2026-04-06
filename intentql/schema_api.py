"""Load and validate schema.yaml (no QueryPlan layer)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .exceptions import SchemaError
from .schema_validator import validate_schema


def load_and_validate_schema(schema_path: str) -> Dict[str, Any]:
    """
    Load schema.yaml and run load-time validation.
    Prints any non-fatal warnings. Raises SchemaError on fatal issues.
    """
    try:
        raw = Path(schema_path).read_text(encoding="utf-8")
        schema = yaml.safe_load(raw) or {}
    except Exception as e:
        raise SchemaError(f"Failed to load schema from '{schema_path}': {e}") from e

    warnings = validate_schema(schema)
    for w in warnings:
        print(f"[IntentQL schema] {w}")

    return schema
