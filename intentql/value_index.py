"""
Database value index — DISTINCT pick-lists for guided SQL and (optionally) intent resolution.

- **Auto mode** (``value_index: auto`` in schema.yaml): heuristically chooses string columns
  to index; skips typical ID / free-text fields. Results are **cached** per
  ``(engine URL, schema path mtime, max_distinct)`` to avoid repeated queries every turn.

- **Explicit mode**: list columns under ``value_index:`` (see :mod:`intentql.guided_sql`).

- **Intent helpers** ``resolve_intent_values`` / ``validate_intent_against_index`` are for
  structured QueryPlan-style intents (optional).
"""

from __future__ import annotations

import sys
import time
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from sqlalchemy import text as sqla_text
from sqlalchemy.engine import Engine

# Process-local cache: key -> (built index, monotonic time)
_cache: Dict[Tuple[str, str, int, int], Tuple[Dict[str, Dict[str, List[str]]], float]] = {}
_CACHE_TTL_SEC = 3600.0
_CACHE_MAX_ENTRIES = 32


def _load_schema(schema_path: str) -> Dict[str, Any]:
    return yaml.safe_load(Path(schema_path).read_text(encoding="utf-8")) or {}


def _cache_prune() -> None:
    if len(_cache) <= _CACHE_MAX_ENTRIES:
        return
    # Drop oldest by insertion time (approximate: by value tuple time)
    sorted_keys = sorted(_cache.keys(), key=lambda k: _cache[k][1])
    for k in sorted_keys[: len(_cache) - _CACHE_MAX_ENTRIES + 8]:
        _cache.pop(k, None)


def get_cached_value_index(
    engine: Engine,
    schema_path: str,
    max_distinct: int,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Build (or return cached) index for :func:`build_value_index`.

    Cache invalidates when ``schema.yaml`` mtime changes or ``max_distinct`` changes.
    """
    p = Path(schema_path).resolve()
    try:
        mtime_ns = p.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    url = str(engine.url)
    key = (url, str(p), mtime_ns, max_distinct)
    now = time.monotonic()
    ent = _cache.get(key)
    if ent is not None:
        idx, t0 = ent
        if now - t0 < _CACHE_TTL_SEC:
            return idx
    idx = build_value_index(engine, str(p), max_distinct=max_distinct)
    _cache[key] = (idx, now)
    _cache_prune()
    return idx


def build_value_index(
    engine: Engine,
    schema_path: str,
    max_distinct: int = 500,
) -> Dict[str, Dict[str, List[str]]]:
    """Query the DB for DISTINCT values of indexable columns (see :func:`_get_indexable_columns`)."""
    schema = _load_schema(schema_path)
    index: Dict[str, Dict[str, List[str]]] = {}

    for table in schema.get("tables", []) or []:
        logical_name = str(table.get("name", "") or "").strip()
        if not logical_name:
            continue
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
                    f"SELECT DISTINCT {col_db} FROM {db_table} "
                    f"WHERE {col_db} IS NOT NULL "
                    f"ORDER BY {col_db} LIMIT :lim"
                )
                with engine.connect() as conn:
                    rows = conn.execute(sql, {"lim": max_distinct}).fetchall()
                values = [str(r[0]).strip() for r in rows if r[0] and str(r[0]).strip()]
                if values:
                    table_index[str(col_logical)] = values
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
    """Choose varchar/text columns suitable for DISTINCT pick-lists; skip IDs and long text."""
    skip_exact = {
        "description",
        "long_desc",
        "desc",
        "comment",
        "note",
        "work_order_id",
        "worker_id",
        "worker_name",
        "asset_tag",
        "tag_number",
        "phase_id",
        "shop_id",
        "building_id",
        "floor_id",
        "location_code",
    }
    indexable: List[Dict[str, Any]] = []
    for col in table.get("columns", []) or []:
        if not isinstance(col, dict):
            continue
        cname = str(col.get("name", "") or "").strip()
        if not cname:
            continue
        if col.get("index_values") is False:
            continue
        if col.get("index_values") is True:
            indexable.append(col)
            continue

        ctype = (col.get("type") or "").lower()
        if ctype not in ("varchar", "text"):
            continue
        if cname.lower() in skip_exact:
            continue
        if any(pat in cname.lower() for pat in ("description", "long_desc", "desc")):
            continue
        indexable.append(col)
    return indexable


def format_auto_value_index_for_guided_prompt(
    index: Dict[str, Dict[str, List[str]]],
    *,
    max_values_per_column: int,
) -> str:
    """Markdown block matching guided SQL VALUE INDEX style (bullet lists)."""
    if not index:
        return ""

    sections: List[str] = [
        "### VALUE INDEX (distinct values from the database)",
        "Auto mode: string columns were chosen heuristically; IDs and long-text fields are "
        "skipped. Set per-column `index_values: false` in schema.yaml to exclude, or "
        "`index_values: true` to include a column that was skipped. Use explicit `value_index` "
        "instead of `auto` for full control.",
        "",
    ]
    any_v = False
    for ltable in sorted(index.keys()):
        cols = index[ltable]
        for lcname in sorted(cols.keys()):
            vals = cols[lcname]
            if not vals:
                continue
            any_v = True
            show = vals[:max_values_per_column]
            sections.append(
                f"**{ltable}.{lcname}**, up to {len(show)} values shown "
                f"({len(vals)} distinct in sample):"
            )
            sections.extend(f"  - {v}" for v in show)
            if len(vals) > len(show):
                sections.append(f"  (... truncated; cap={max_values_per_column})")
            sections.append("")

    if not any_v:
        return ""
    return "\n".join(sections).rstrip() + "\n\n"


def format_value_index_for_prompt(
    value_index: Dict[str, Dict[str, List[str]]],
    max_values_per_column: int = 80,
) -> str:
    """Compact one-line-per-column format (legacy / intent tooling)."""
    parts: List[str] = []
    parts.append("KNOWN DATABASE VALUES (exact strings for filters):")
    for table_name in sorted(value_index.keys()):
        parts.append(f"\n  Table: {table_name}")
        columns = value_index[table_name]
        for col_name in sorted(columns.keys()):
            values = columns[col_name]
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
    """Best-effort match for *value* in *known_values* (case-insensitive)."""
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
        upper_val,
        list(upper_known.keys()),
        n=1,
        cutoff=cutoff,
    )
    if matches:
        return upper_known[matches[0]]

    return None


def resolve_intent_values(
    intent: Dict[str, Any],
    value_index: Dict[str, Dict[str, List[str]]],
) -> Dict[str, Any]:
    """Resolve intent filter values against the value index (QueryPlan path)."""
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
    """Return human-readable issues if filter values are far from known DISTINCT values."""
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
