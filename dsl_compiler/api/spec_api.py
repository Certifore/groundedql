from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    data = yaml.safe_load(p.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {p} must be a mapping/object.")
    return data


def get_queryplan_spec(
    *,
    spec_path: str | Path = "config/queryplan_spec.yaml",
) -> Dict[str, Any]:
    """
    Returns the raw spec dict (programmatic use).
    """
    return load_yaml(spec_path)


def get_queryplan_instructions(
    *,
    schema_path: str | Path,
    spec_path: str | Path = "config/queryplan_spec.yaml",
    include_schema_yaml: bool = True,
) -> str:
    """
    Returns a ready-to-use prompt string for an LLM:
    - high-level instructions from queryplan_spec.yaml
    - optionally appends the DB schema YAML so the model knows allowed tables/columns
    """
    spec = load_yaml(spec_path)
    parts: list[str] = []

    # Core instructions
    parts.append(spec.get("system_instructions", "").strip())
    parts.append("\n---\n")
    parts.append("QUERYPLAN SPEC (authoring rules):\n")
    parts.append(yaml.safe_dump(spec, sort_keys=False))

    if include_schema_yaml:
        schema = Path(schema_path).read_text()
        parts.append("\n---\n")
        parts.append("DB SCHEMA (logical names to use):\n")
        parts.append(schema)

    return "\n".join(parts).strip()