"""
intent_normalize.py — Deterministic normalization of extracted intents.

No matter how the LLM structures the intent, this module canonicalizes it
so that semantically equivalent questions always produce the same intent
structure.  This eliminates a major source of inconsistency.

Optional ``intent_id_patterns`` on a table in schema.yaml (list of regex strings)
may be used to add a primary_id filter from the user question; patterns are
application-defined — IntentQL does not ship domain-specific ID heuristics.
"""
from __future__ import annotations

import re
import sys
from typing import Any, Dict, List, Optional, Set


def normalize_intent(
    intent: Dict[str, Any],
    schema: Dict[str, Any],
    question: Optional[str] = None,
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
    intent = _strip_spurious_work_token_as_primary_id(intent, question, schema)
    intent = _inject_primary_id_from_question(intent, question, schema)
    intent = _infer_time_bucket_for_trends(intent, question, schema)
    intent = _maybe_coerce_list_for_detail_lookup(intent, question, schema)
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


# --- Phase 1: optional ID hints from question, trend bucketing ----------------


def _strip_spurious_work_token_as_primary_id(
    intent: Dict[str, Any],
    question: Optional[str],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Remove primary_id filter value WORK when the user said 'work order(s)'.

    This is a common structured-output mistake: the model treats the word **work**
    as a literal ID. Not domain-specific to WO formats — English phrase guard only.
    """
    if not question:
        return intent
    q = question.lower()
    if "work order" not in q:
        return intent
    dataset = intent.get("dataset") or ""
    table = _table_meta(schema, dataset)
    pid = table.get("primary_id")
    if not pid:
        return intent
    kept: List[Dict[str, Any]] = []
    for f in intent.get("filters") or []:
        if f.get("column") != pid:
            kept.append(f)
            continue
        vals = f.get("values") or []
        if len(vals) == 1 and str(vals[0]).strip().upper() == "WORK":
            print(
                "[Normalize] Dropped spurious primary_id filter WORK "
                "(misread from 'work order(s)' in the question).",
                file=sys.stderr,
            )
            continue
        kept.append(f)
    intent["filters"] = kept
    return intent


def _compiled_intent_id_patterns(table: Dict[str, Any]) -> List[re.Pattern]:
    """Optional per-table regex list from schema (`intent_id_patterns`). Domain-agnostic."""
    raw = table.get("intent_id_patterns") or []
    if not isinstance(raw, list):
        return []
    out: List[re.Pattern] = []
    for pat in raw:
        if not isinstance(pat, str) or not pat.strip():
            continue
        try:
            out.append(re.compile(pat, re.IGNORECASE))
        except re.error:
            continue
    return out


def _match_value_from_regex(m: re.Match) -> str:
    if m.lastindex and m.lastindex >= 1:
        return m.group(1).strip()
    return m.group(0).strip()


def _extract_id_candidates_from_question(text: str, patterns: List[re.Pattern]) -> List[str]:
    if not text or not patterns:
        return []
    out: List[str] = []
    for pat in patterns:
        for m in pat.finditer(text):
            out.append(_match_value_from_regex(m))
    return out


def _inject_primary_id_from_question(
    intent: Dict[str, Any],
    question: Optional[str],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """If schema declares ``intent_id_patterns`` for this table, add a primary_id filter from the first match."""
    if not question:
        return intent
    dataset = intent.get("dataset") or ""
    table = _table_meta(schema, dataset)
    pid = table.get("primary_id")
    if not pid:
        return intent
    patterns = _compiled_intent_id_patterns(table)
    if not patterns:
        return intent
    existing_cols = {f.get("column") for f in intent.get("filters") or []}
    if pid in existing_cols:
        return intent
    for cand in _extract_id_candidates_from_question(question, patterns):
        intent.setdefault("filters", []).append({"column": pid, "values": [cand]})
        print(
            f"[Normalize] Injected primary_id filter {pid}={cand} from question text (intent_id_patterns)",
            file=sys.stderr,
        )
        break
    return intent


def _infer_time_bucket_for_trends(
    intent: Dict[str, Any],
    question: Optional[str],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """When the user asks for a trend / over time and groups by the date column, set time_bucket."""
    if intent.get("time_bucket"):
        return intent
    q = (question or "").lower()
    trendish = any(
        w in q
        for w in (
            "trend",
            "over time",
            "over the",
            "by month",
            "by year",
            "by quarter",
            "monthly",
            "yearly",
            "quarterly",
        )
    )
    if not trendish:
        return intent
    dataset = intent.get("dataset") or ""
    table = _table_meta(schema, dataset)
    pd = table.get("primary_date")
    if not pd:
        return intent
    group_by = intent.get("group_by") or []
    if pd not in group_by:
        return intent
    if "year" in q and "month" not in q:
        bucket = "year"
    elif "quarter" in q:
        bucket = "quarter"
    elif "day" in q or "daily" in q:
        bucket = "day"
    else:
        bucket = "month"
    intent["time_bucket"] = bucket
    print(
        f"[Normalize] Set time_bucket={bucket} for trend on {pd}",
        file=sys.stderr,
    )
    return intent


def _maybe_coerce_list_for_detail_lookup(
    intent: Dict[str, Any],
    question: Optional[str],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Prefer list aggregation when the user asks for details/lookup and primary_id is filtered."""
    if not question:
        return intent
    q = question.lower()
    detailish = any(
        w in q
        for w in (
            "detail",
            "details",
            "show me the",
            "lookup",
            "information about",
            "full record",
        )
    )
    if not detailish:
        return intent
    if intent.get("aggregation") == "list":
        return intent
    table = _table_meta(schema, intent.get("dataset") or "")
    pid = table.get("primary_id")
    if not pid:
        return intent
    for f in intent.get("filters") or []:
        if f.get("column") != pid:
            continue
        vals = f.get("values") or []
        if not vals:
            continue
        intent["aggregation"] = "list"
        print(
            "[Normalize] Coerced aggregation to 'list' for detail-style question with primary_id filter",
            file=sys.stderr,
        )
        return intent
    return intent
