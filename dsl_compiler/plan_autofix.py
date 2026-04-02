"""
plan_autofix.py — Deterministic fixes for common LLM planning mistakes.

Applied *before* execution so the plan runs correctly without burning
extra LLM retries.  Each fix is driven by schema declarations
(primary_id, keyword_search_or) — nothing domain-specific.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Set


def autofix_plan(
    plan: Dict[str, Any],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Apply all deterministic fixes to *plan* in-place and return it.

    Fixes applied:
    1. count/count(*) → count_distinct(primary_id) when primary_id is declared.
    2. Duplicate keyword_search_or columns in both filters AND where.or → strip from filters.
    """
    table_meta = _build_table_meta(schema)
    _fix_all_subplans(plan, table_meta)
    return plan


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_table_meta(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build a lookup of primary_id and keyword_search_or by table name."""
    meta: Dict[str, Dict[str, Any]] = {}
    for t in schema.get("tables", []):
        name = t.get("name")
        if not isinstance(name, str):
            continue
        entry: Dict[str, Any] = {}
        pid = t.get("primary_id")
        if pid:
            entry["primary_id"] = pid
        kso = t.get("keyword_search_or")
        if isinstance(kso, list) and len(kso) >= 2:
            entry["keyword_search_or"] = set(c for c in kso if isinstance(c, str))
        meta[name] = entry
    return meta


def _iter_subplans(plan: Dict[str, Any]):
    """Yield this plan and every nested CTE inner plan."""
    yield plan
    for w in plan.get("with") or []:
        if isinstance(w, dict) and isinstance(w.get("plan"), dict):
            yield from _iter_subplans(w["plan"])


def _fix_all_subplans(plan: Dict[str, Any], table_meta: Dict[str, Dict[str, Any]]) -> None:
    for sub in _iter_subplans(plan):
        ds = sub.get("dataset")
        if not isinstance(ds, str) or ds not in table_meta:
            continue
        meta = table_meta[ds]
        _fix_count_to_count_distinct(sub, meta)
        _fix_duplicate_keyword_filters(sub, meta)


def _fix_count_to_count_distinct(plan: Dict[str, Any], meta: Dict[str, Any]) -> None:
    """Fix count(*) or count(primary_id) → count_distinct(primary_id)."""
    primary_id = meta.get("primary_id")
    if not primary_id:
        return

    for m in plan.get("metrics") or []:
        agg = (m.get("agg") or "").lower()
        field = m.get("field", "")

        if agg == "count" and field in ("*", primary_id, ""):
            print(
                f"[QCE autofix] count({field or '*'}) → count_distinct({primary_id})",
                file=sys.stderr,
            )
            m["agg"] = "count_distinct"
            m["field"] = primary_id
        elif agg == "count_distinct" and field == "*":
            print(
                f"[QCE autofix] count_distinct(*) → count_distinct({primary_id})",
                file=sys.stderr,
            )
            m["field"] = primary_id


def _or_tree_columns(where: Any) -> Set[str]:
    """Extract column names from a where.or tree's cmp contains nodes."""
    cols: Set[str] = set()
    if not isinstance(where, dict):
        return cols
    if "or" in where:
        for item in where.get("or") or []:
            if not isinstance(item, dict) or "cmp" not in item:
                continue
            c = item["cmp"]
            left = c.get("left") or {}
            col = left.get("col") if isinstance(left, dict) else None
            if isinstance(col, str):
                cols.add(col.split(".")[-1])
    if "and" in where:
        for item in where.get("and") or []:
            cols.update(_or_tree_columns(item))
    return cols


def _extract_keyword_value(plan: Dict[str, Any], kso: Set[str]) -> Optional[str]:
    """Find the search term from filters or where.or that targets keyword_search_or columns."""
    for f in plan.get("filters") or []:
        if not isinstance(f, dict):
            continue
        if (f.get("op") or "").lower() == "contains":
            field = (f.get("field") or "").split(".")[-1]
            if field in kso and f.get("value"):
                return str(f["value"])

    where = plan.get("where")
    if isinstance(where, dict) and "or" in where:
        for item in where.get("or") or []:
            if not isinstance(item, dict) or "cmp" not in item:
                continue
            c = item["cmp"]
            if (c.get("op") or "").lower() == "contains" and c.get("right"):
                return str(c["right"])
    return None


def _build_keyword_or(kso: Set[str], value: str) -> Dict[str, Any]:
    """Build a correct where.or tree covering all keyword_search_or columns."""
    return {"or": [
        {"cmp": {"left": {"col": col}, "op": "contains", "right": value}}
        for col in sorted(kso)
    ]}


def _fix_duplicate_keyword_filters(plan: Dict[str, Any], meta: Dict[str, Any]) -> None:
    """Ensure keyword_search_or columns use where.or (not filters) with all columns covered."""
    kso: Optional[Set[str]] = meta.get("keyword_search_or")
    if not kso:
        return

    keyword_value = _extract_keyword_value(plan, kso)
    if not keyword_value:
        return

    where = plan.get("where")
    or_cols = _or_tree_columns(where) if where else set()

    if not kso.issubset(or_cols):
        correct_or = _build_keyword_or(kso, keyword_value)
        if where is None:
            plan["where"] = correct_or
        elif isinstance(where, dict) and "or" in where:
            plan["where"] = correct_or
        else:
            plan["where"] = {"and": [where, correct_or]}
        print(
            f"[QCE autofix] Built/completed where.or for keyword_search_or columns: {sorted(kso)}",
            file=sys.stderr,
        )

    filters = plan.get("filters")
    if not filters:
        return

    cleaned = []
    removed = []
    for f in filters:
        if not isinstance(f, dict):
            cleaned.append(f)
            continue
        op = (f.get("op") or "").lower()
        field = (f.get("field") or "").split(".")[-1]
        if op == "contains" and field in kso:
            removed.append(field)
            continue
        cleaned.append(f)

    if removed:
        print(
            f"[QCE autofix] Stripped duplicate keyword filters: {sorted(set(removed))}",
            file=sys.stderr,
        )
        plan["filters"] = cleaned
