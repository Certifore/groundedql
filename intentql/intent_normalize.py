"""
intent_normalize.py — Deterministic normalization of extracted intents.

No matter how the LLM structures the intent, this module canonicalizes it
so that semantically equivalent questions always produce the same intent
structure.  This eliminates a major source of inconsistency.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Set


def normalize_intent(
    intent: Dict[str, Any],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Canonicalize an extracted intent dict.

    Applies deterministic rules to resolve structural ambiguities
    the LLM might produce differently across rephrasings.
    """
    dataset = intent.get("dataset", "")
    table = _table_meta(schema, dataset)
    kso_cols = set(table.get("keyword_search_or") or [])

    intent = _absorb_keyword_filters(intent, kso_cols)
    intent = _ensure_group_by_is_list(intent)
    intent = _normalize_group_by_for_multi_value_filters(intent)

    return intent


def _table_meta(schema: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    for t in schema.get("tables", []):
        if t.get("name") == dataset:
            return t
    return {}


def _absorb_keyword_filters(
    intent: Dict[str, Any],
    kso_cols: Set[str],
) -> Dict[str, Any]:
    """If the LLM put the keyword as both `keyword` and a filter on a kso column,
    remove the filter and keep only the keyword.

    The keyword generates a broad OR across all kso columns.  A filter on one
    kso column is strictly narrower, so the keyword subsumes it.
    """
    keyword = intent.get("keyword")
    if not keyword or not kso_cols:
        return intent

    kw_upper = keyword.upper().strip()
    filters = intent.get("filters") or []
    kept: List[Dict[str, Any]] = []
    absorbed = False

    for f in filters:
        col = f.get("column", "")
        vals = f.get("values") or []
        if col in kso_cols and any(kw_upper in v.upper() for v in vals):
            absorbed = True
            print(
                f"[Normalize] Absorbed redundant filter {col}={vals} "
                f"(subsumed by keyword '{keyword}')",
                file=sys.stderr,
            )
            continue
        kept.append(f)

    if absorbed:
        intent["filters"] = kept

    return intent


def _ensure_group_by_is_list(intent: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize group_by to always be a list."""
    gb = intent.get("group_by")
    if gb is None:
        intent["group_by"] = []
    elif isinstance(gb, str):
        intent["group_by"] = [gb]
    return intent


def _normalize_group_by_for_multi_value_filters(
    intent: Dict[str, Any],
) -> Dict[str, Any]:
    """If there are multi-value filters and aggregation is count,
    ensure the filtered column is in group_by.
    """
    agg = intent.get("aggregation", "count")
    if agg not in ("count", "sum", "avg"):
        return intent

    group_by = intent.get("group_by") or []
    for f in intent.get("filters") or []:
        vals = f.get("values") or []
        col = f.get("column", "")
        if len(vals) > 1 and col not in group_by:
            group_by.append(col)
            print(
                f"[Normalize] Auto-added group_by '{col}' for multi-value filter",
                file=sys.stderr,
            )

    intent["group_by"] = group_by
    return intent
