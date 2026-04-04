"""
value_index.py — Database value index for constraining LLM intent extraction.

Queries the live database at startup for distinct values of key columns
(building names, asset keywords, status codes, etc.) and provides:

1. Pick-lists to inject into the LLM prompt (enum-style constraints)
2. Fuzzy matching to resolve LLM-extracted values against real data
"""

from __future__ import annotations

import sys
from difflib import get_close_matches
from typing import Any, Dict, List, Optional

import yaml
from sqlalchemy import text as sqla_text
from sqlalchemy.engine import Engine


def _load_schema(schema_path: str) -> Dict[str, Any]:
    from pathlib import Path
    return yaml.safe_load(Path(schema_path).read_text()) or {}


def build_value_index(
    engine: Engine,
    schema_path: str,
    max_distinct: int = 500,
) -> Dict[str, Dict[str, List[str]]]:
    """Query the DB for distinct values of indexable columns.

    Returns::

        {
            "work_orders": {
                "building_name": ["ATHENAEUM", "BECHTEL RESIDENCE", ...],
                "status_code": ["CLOSED", "COMPLETE", ...],
                ...
            },
            "assets": {
                "building_name": ["ATHENAEUM", "BECHTEL MALL", ...],
                "keyword_of_asset": ["AIR HANDLER", "FIRE EXTINGUISHER", ...],
                ...
            },
        }
    """
    schema = _load_schema(schema_path)
    index: Dict[str, Dict[str, List[str]]] = {}

    for table in schema.get("tables", []):
        logical_name = table.get("name", "")
        db_table = table.get("db_table", logical_name)
        indexable = _get_indexable_columns(table)
        if not indexable:
            continue

        table_index: Dict[str, List[str]] = {}
        for col_meta in indexable:
            col_logical = col_meta["name"]
            col_db = col_meta.get("db_column", col_logical)
            try:
                sql = sqla_text(
                    f'SELECT DISTINCT {col_db} FROM {db_table} '
                    f'WHERE {col_db} IS NOT NULL '
                    f'ORDER BY {col_db} LIMIT :lim'
                )
                with engine.connect() as conn:
                    rows = conn.execute(sql, {"lim": max_distinct}).fetchall()
                values = [str(r[0]).strip() for r in rows if r[0] and str(r[0]).strip()]
                if values:
                    table_index[col_logical] = values
                    print(
                        f"[ValueIndex] {logical_name}.{col_logical}: {len(values)} values",
                        file=sys.stderr,
                    )
            except Exception as exc:
                print(
                    f"[ValueIndex] Failed to index {logical_name}.{col_logical}: {exc}",
                    file=sys.stderr,
                )

        if table_index:
            index[logical_name] = table_index

    return index


def _get_indexable_columns(table: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Decide which columns are worth indexing for pick-lists.

    Index columns that are categorical (varchar with limited distinct values):
    building names, status codes, priority codes, asset keywords, etc.
    Skip free-text columns and high-cardinality ID columns.
    """
    skip_patterns = {
        "description", "long_desc", "desc", "comment", "note",
        "work_order_id", "worker_id", "worker_name",
        "asset_tag", "tag_number", "phase_id",
        "shop_id", "building_id", "floor_id", "location_code",
    }
    indexable = []
    for col in table.get("columns", []):
        cname = col.get("name", "")
        ctype = (col.get("type") or "").lower()
        if ctype not in ("varchar", "text"):
            continue
        if cname.lower() in skip_patterns:
            continue
        if any(pat in cname.lower() for pat in {"description", "long_desc", "desc"}):
            continue
        indexable.append(col)
    return indexable


def format_value_index_for_prompt(
    value_index: Dict[str, Dict[str, List[str]]],
    max_values_per_column: int = 80,
) -> str:
    """Format the value index as a compact string for the LLM prompt.

    For columns with many values, show a representative sample.
    """
    parts: List[str] = []
    parts.append("KNOWN DATABASE VALUES (use these exact values in filters):")

    for table_name, columns in sorted(value_index.items()):
        parts.append(f"\n  Table: {table_name}")
        for col_name, values in sorted(columns.items()):
            if len(values) <= max_values_per_column:
                display = values
            else:
                display = values[:max_values_per_column]
            vals_str = ", ".join(display)
            suffix = f" ... ({len(values)} total)" if len(values) > max_values_per_column else ""
            parts.append(f"    {col_name}: [{vals_str}{suffix}]")

    return "\n".join(parts)


def fuzzy_resolve(
    value: str,
    known_values: List[str],
    cutoff: float = 0.4,
) -> Optional[str]:
    """Find the best match for *value* in *known_values*.

    Returns the matched value, or None if no close match is found.
    Uses case-insensitive comparison.
    """
    if not known_values or not value:
        return None

    upper_val = value.upper().strip()
    upper_known = {v.upper().strip(): v for v in known_values}

    if upper_val in upper_known:
        return upper_known[upper_val]

    for k, original in upper_known.items():
        if upper_val in k or k in upper_val:
            return original

    matches = get_close_matches(
        upper_val, list(upper_known.keys()), n=1, cutoff=cutoff,
    )
    if matches:
        return upper_known[matches[0]]

    return None


def resolve_intent_values(
    intent: Dict[str, Any],
    value_index: Dict[str, Dict[str, List[str]]],
) -> Dict[str, Any]:
    """Resolve intent filter values and keyword against the value index.

    Modifies the intent in-place and returns it with corrections applied.
    Also returns a list of corrections made for logging.
    """
    dataset = intent.get("dataset", "")
    table_values = value_index.get(dataset, {})
    corrections: List[str] = []

    for filt in intent.get("filters") or []:
        col = filt.get("column", "")
        known = table_values.get(col)
        if not known:
            continue

        resolved_vals = []
        for v in filt.get("values", []):
            match = fuzzy_resolve(v, known)
            if match and match != v:
                corrections.append(f"filter {col}: '{v}' → '{match}'")
                resolved_vals.append(match)
            elif match:
                resolved_vals.append(match)
            else:
                resolved_vals.append(v)
        filt["values"] = resolved_vals

    keyword = intent.get("keyword")
    if keyword:
        kw_col = table_values.get("keyword_of_asset")
        if kw_col:
            match = fuzzy_resolve(keyword, kw_col)
            if match and match.upper() != keyword.upper():
                corrections.append(f"keyword: '{keyword}' → '{match}'")
                intent["keyword"] = match

    if corrections:
        print(
            f"[ValueIndex] Resolved: {'; '.join(corrections)}",
            file=sys.stderr,
        )

    return intent


def validate_intent_against_index(
    intent: Dict[str, Any],
    value_index: Dict[str, Dict[str, List[str]]],
) -> List[str]:
    """Check if intent values exist in the value index.

    Returns a list of issues found (empty = all good).
    """
    dataset = intent.get("dataset", "")
    table_values = value_index.get(dataset, {})
    issues: List[str] = []

    for filt in intent.get("filters") or []:
        col = filt.get("column", "")
        known = table_values.get(col)
        if not known:
            continue
        known_upper = {v.upper().strip() for v in known}
        for v in filt.get("values", []):
            if v.upper().strip() not in known_upper:
                close = get_close_matches(v.upper(), list(known_upper), n=3, cutoff=0.4)
                suggestion = f" Did you mean: {close}" if close else ""
                issues.append(
                    f"Filter value '{v}' for column '{col}' not found in database.{suggestion}"
                )

    return issues
