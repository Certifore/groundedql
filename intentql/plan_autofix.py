"""
plan_autofix.py — Deterministic fixes for common LLM planning mistakes.

Applied *before* execution so the plan runs correctly without burning
extra LLM retries.  Each fix is driven by schema declarations
(primary_id, keyword_search_or) — nothing domain-specific.
"""
from __future__ import annotations

import re
import sys
from typing import Any, Dict, List, Optional, Set


def autofix_plan(
    plan: Dict[str, Any],
    schema: Dict[str, Any],
    question: str = "",
) -> Dict[str, Any]:
    """
    Apply all deterministic fixes to *plan* in-place and return it.

    Fixes applied:
    1. count/count(*) → count_distinct(primary_id) when primary_id is declared.
    2. Duplicate keyword_search_or columns in both filters AND where.or → strip from filters.
    3. Multiple contains on same column → where.or.
    4. Missing dimension for multi-value IN/OR filters.
    5. Missing date filter when question mentions a temporal phrase.
    """
    table_meta = _build_table_meta(schema)
    _fix_all_subplans(plan, table_meta, question)
    return plan


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_table_meta(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build a lookup of primary_id, keyword_search_or, and date columns by table name."""
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
        pdate = t.get("primary_date")
        if pdate:
            entry["primary_date"] = pdate
        else:
            for col in t.get("columns", []):
                ctype = (col.get("type") or "").lower()
                if ctype in ("date", "timestamp", "datetime", "timestamptz"):
                    entry["primary_date"] = col.get("name")
                    break
        meta[name] = entry
    return meta


def _iter_subplans(plan: Dict[str, Any]):
    """Yield this plan and every nested CTE inner plan."""
    yield plan
    for w in plan.get("with") or []:
        if isinstance(w, dict) and isinstance(w.get("plan"), dict):
            yield from _iter_subplans(w["plan"])


def _fix_all_subplans(plan: Dict[str, Any], table_meta: Dict[str, Dict[str, Any]], question: str = "") -> None:
    for sub in _iter_subplans(plan):
        ds = sub.get("dataset")
        if not isinstance(ds, str) or ds not in table_meta:
            continue
        meta = table_meta[ds]
        _fix_count_to_count_distinct(sub, meta)
        _fix_multi_contains_same_column(sub)
        _fix_duplicate_keyword_filters(sub, meta, question)
        _fix_missing_date_filter(sub, meta, question)
    _fix_missing_dimension_for_multi_value(plan, table_meta)


def _fix_multi_contains_same_column(plan: Dict[str, Any]) -> None:
    """Convert multiple AND-ed `contains` filters on the same column into a where.or.

    When the LLM generates e.g.:
      filters: [{field: entity_name, op: contains, value: "alpha"},
                {field: entity_name, op: contains, value: "beta"}]
    these are AND-ed (impossible).  This fix moves them into an OR tree.
    """
    filters = plan.get("filters")
    if not isinstance(filters, list) or len(filters) < 2:
        return

    from collections import defaultdict
    contains_by_col: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    eq_by_col: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for f in filters:
        if not isinstance(f, dict):
            continue
        op = (f.get("op") or "").lower()
        field = (f.get("field") or "").split(".")[-1]
        if not field:
            continue
        if op == "contains":
            contains_by_col[field].append(f)
        elif op == "=":
            eq_by_col[field].append(f)

    cols_to_fix = {col: flist for col, flist in contains_by_col.items() if len(flist) >= 2}
    eq_cols_to_fix = {col: flist for col, flist in eq_by_col.items() if len(flist) >= 2}
    if not cols_to_fix and not eq_cols_to_fix:
        return

    filters_to_remove: set = set()
    or_groups: List[Dict[str, Any]] = []

    for col, flist in cols_to_fix.items():
        or_branches = []
        for f in flist:
            or_branches.append(
                {"cmp": {"left": {"col": col}, "op": "contains", "right": f["value"]}}
            )
            filters_to_remove.add(id(f))
        or_groups.append({"or": or_branches})

        print(
            f"[IntentQL autofix] Converted {len(flist)} AND-ed contains filters on '{col}' → where.or",
            file=sys.stderr,
        )

    for col, flist in eq_cols_to_fix.items():
        values = [f["value"] for f in flist]
        for f in flist:
            filters_to_remove.add(id(f))
        or_branches = [
            {"cmp": {"left": {"col": col}, "op": "=", "right": v}}
            for v in values
        ]
        or_groups.append({"or": or_branches})
        print(
            f"[IntentQL autofix] Converted {len(flist)} AND-ed '=' filters on '{col}' → where.or",
            file=sys.stderr,
        )

    cleaned = [f for f in filters if id(f) not in filters_to_remove]
    plan["filters"] = cleaned

    where = plan.get("where")
    new_clauses = or_groups if len(or_groups) == 1 else or_groups
    if where is None:
        if len(new_clauses) == 1:
            plan["where"] = new_clauses[0]
        else:
            plan["where"] = {"and": new_clauses}
    else:
        existing = [where] if not isinstance(where, list) else where
        plan["where"] = {"and": existing + new_clauses}


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
                f"[IntentQL autofix] count({field or '*'}) → count_distinct({primary_id})",
                file=sys.stderr,
            )
            m["agg"] = "count_distinct"
            m["field"] = primary_id
        elif agg == "count_distinct" and field == "*":
            print(
                f"[IntentQL autofix] count_distinct(*) → count_distinct({primary_id})",
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
    """Find the search term from filters or where tree that targets keyword_search_or columns."""
    for f in plan.get("filters") or []:
        if not isinstance(f, dict):
            continue
        if (f.get("op") or "").lower() == "contains":
            field = (f.get("field") or "").split(".")[-1]
            if field in kso and f.get("value"):
                return str(f["value"])

    where = plan.get("where")
    val = _extract_keyword_from_where(where, kso)
    if val:
        return val
    return None


def _extract_keyword_from_where(where: Any, kso: Set[str]) -> Optional[str]:
    """Recursively search where tree for a keyword value targeting kso columns."""
    if not isinstance(where, dict):
        return None
    if "or" in where:
        for item in where.get("or") or []:
            if not isinstance(item, dict) or "cmp" not in item:
                continue
            c = item["cmp"]
            left = c.get("left") or {}
            col = left.get("col") if isinstance(left, dict) else None
            if isinstance(col, str) and col.split(".")[-1] in kso:
                if (c.get("op") or "").lower() == "contains" and c.get("right"):
                    return str(c["right"])
    if "and" in where:
        for item in where.get("and") or []:
            val = _extract_keyword_from_where(item, kso)
            if val:
                return val
    return None


def _build_keyword_or(kso: Set[str], value: str) -> Dict[str, Any]:
    """Build a correct where.or tree covering all keyword_search_or columns."""
    return {"or": [
        {"cmp": {"left": {"col": col}, "op": "contains", "right": value}}
        for col in sorted(kso)
    ]}


_QUESTION_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "their", "its", "this", "that", "these", "those",
    "and", "or", "but", "not", "no", "nor", "if", "then", "than",
    "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "up", "out", "about", "into", "over", "after", "before",
    "how", "many", "much", "what", "which", "who", "where", "when",
    "each", "every", "all", "any", "both", "few", "more", "most",
    "some", "such", "than", "too", "very", "just", "also",
    "give", "get", "got", "show", "list", "tell", "find", "look",
    "display", "fetch", "pull", "query", "search", "check", "see",
    "provide", "report", "reported", "report", "tally", "log", "logged",
    "need", "want", "like", "know", "compare", "submit", "submitted",
    "create", "created", "open", "opened", "place", "placed", "raise",
    "raised", "file", "filed", "record", "recorded", "happen", "happened",
    "count", "number", "total", "per", "breakdown", "summary", "volume",
    "last", "year", "years", "month", "months", "week", "day", "date",
    "issue", "issues", "order", "orders", "work", "request", "requests",
    "building", "buildings", "house", "center", "hall", "lab",
    "break", "down", "across", "among", "between", "within", "through",
    "detail", "details", "detailed", "specific", "separately", "individual",
    "individually", "respective", "respectively",
    "during", "since", "until", "past", "previous", "recent", "recently",
    "annual", "annually", "quarterly", "monthly", "weekly", "daily",
    "maintenance", "repair", "service", "facility", "facilities",
})


def _extract_keyword_from_question(question: str, plan: Dict[str, Any]) -> Optional[str]:
    """Try to find a keyword from the question that isn't a value already in the plan.

    Extracts the first substantive word from the question that is not a stop word
    and not already used as a filter value.  This allows the autofix to inject
    a keyword_search_or filter when the LLM forgot to include one.
    """
    if not question:
        return None
    q_lower = question.lower()
    existing_values: set = set()
    for f in plan.get("filters") or []:
        if isinstance(f, dict) and f.get("value"):
            val = f["value"]
            if isinstance(val, str):
                for part in val.lower().split():
                    existing_values.add(part.strip(",.?!;:'\""))
            elif isinstance(val, list):
                for v in val:
                    if isinstance(v, str):
                        for part in v.lower().split():
                            existing_values.add(part.strip(",.?!;:'\""))

    for word in q_lower.split():
        clean = word.strip(",.?!;:'\"")
        if len(clean) < 3:
            continue
        if clean in _QUESTION_STOP_WORDS:
            continue
        if clean in existing_values:
            continue
        return clean
    return None


def _strip_partial_keyword_ors(where: Any, kso: Set[str]) -> Any:
    """Remove OR nodes from the where tree that only reference keyword_search_or columns.

    When the LLM produces a partial keyword OR (e.g. 2 of 3 kso columns) AND the autofix
    is about to add a complete one, the partial becomes redundant and makes the query
    incorrectly restrictive.  Returns the cleaned where, or None if nothing remains.
    """
    if not isinstance(where, dict):
        return where

    if "or" in where:
        or_cols = set()
        for item in where.get("or") or []:
            if isinstance(item, dict) and "cmp" in item:
                c = item["cmp"]
                left = c.get("left") or {}
                col = left.get("col") if isinstance(left, dict) else None
                if isinstance(col, str):
                    or_cols.add(col.split(".")[-1])
        if or_cols and or_cols.issubset(kso):
            return None
        return where

    if "and" in where:
        cleaned = []
        for item in where.get("and") or []:
            result = _strip_partial_keyword_ors(item, kso)
            if result is not None:
                cleaned.append(result)
        if not cleaned:
            return None
        if len(cleaned) == 1:
            return cleaned[0]
        return {"and": cleaned}

    return where


def _fix_duplicate_keyword_filters(plan: Dict[str, Any], meta: Dict[str, Any], question: str = "") -> None:
    """Ensure keyword_search_or columns use where.or (not filters) with all columns covered."""
    kso: Optional[Set[str]] = meta.get("keyword_search_or")
    if not kso:
        return

    keyword_value = _extract_keyword_value(plan, kso)
    if not keyword_value:
        keyword_value = _extract_keyword_from_question(question, plan)
    if not keyword_value:
        return

    where = plan.get("where")
    or_cols = _or_tree_columns(where) if where else set()

    if not kso.issubset(or_cols):
        correct_or = _build_keyword_or(kso, keyword_value)
        if where is None:
            plan["where"] = correct_or
        else:
            cleaned_where = _strip_partial_keyword_ors(where, kso)
            if cleaned_where is None:
                plan["where"] = correct_or
            else:
                plan["where"] = {"and": [cleaned_where, correct_or]}
        print(
            f"[IntentQL autofix] Built/completed where.or for keyword_search_or columns: {sorted(kso)}",
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
            f"[IntentQL autofix] Stripped duplicate keyword filters: {sorted(set(removed))}",
            file=sys.stderr,
        )
        plan["filters"] = cleaned


_TEMPORAL_PATTERNS = [
    (re.compile(r"\blast\s+year\b", re.I), {"year_offset": -1}),
    (re.compile(r"\bprevious\s+year\b", re.I), {"year_offset": -1}),
    (re.compile(r"\bthis\s+year\b", re.I), {"year_offset": 0}),
    (re.compile(r"\bcurrent\s+year\b", re.I), {"year_offset": 0}),
    (re.compile(r"\bin\s+(\d{4})\b", re.I), "explicit_year"),
    (re.compile(r"\bfor\s+(\d{4})\b", re.I), "explicit_year"),
    (re.compile(r"\bduring\s+(\d{4})\b", re.I), "explicit_year"),
]


def _fix_missing_date_filter(plan: Dict[str, Any], meta: Dict[str, Any], question: str) -> None:
    """Add date filters when the question mentions a time period but the plan has none."""
    if not question:
        return
    date_col = meta.get("primary_date")
    if not date_col:
        return

    filters = plan.get("filters") or []
    for f in filters:
        if isinstance(f, dict) and (f.get("field") or "").split(".")[-1] == date_col:
            return

    year_offset = None
    explicit_year = None
    for pattern, info in _TEMPORAL_PATTERNS:
        m = pattern.search(question)
        if m:
            if info == "explicit_year":
                explicit_year = int(m.group(1))
            else:
                year_offset = info["year_offset"]
            break

    if year_offset is not None:
        gte = {"$relative_date": {"op": "calendar_year_start", "year_offset": year_offset}}
        lt = {"$relative_date": {"op": "calendar_year_start", "year_offset": year_offset + 1}}
    elif explicit_year is not None:
        gte = f"{explicit_year}-01-01T00:00:00+00:00"
        lt = f"{explicit_year + 1}-01-01T00:00:00+00:00"
    else:
        return

    if not isinstance(filters, list):
        filters = []
        plan["filters"] = filters
    filters.append({"field": date_col, "op": ">=", "value": gte})
    filters.append({"field": date_col, "op": "<", "value": lt})
    print(
        f"[IntentQL autofix] Added missing date filter on '{date_col}' from question",
        file=sys.stderr,
    )


def _where_or_multi_value_columns(where: Any, kso: Set[str]) -> Set[str]:
    """Find columns in where that have multiple OR-ed values (excluding keyword_search_or cols)."""
    cols_with_multi: Set[str] = set()
    if not isinstance(where, dict):
        return cols_with_multi

    if "or" in where:
        from collections import Counter
        col_counter: Counter = Counter()
        for item in where.get("or") or []:
            if not isinstance(item, dict) or "cmp" not in item:
                continue
            c = item["cmp"]
            left = c.get("left") or {}
            col = left.get("col") if isinstance(left, dict) else None
            if isinstance(col, str):
                col_counter[col.split(".")[-1]] += 1
        for col, cnt in col_counter.items():
            if cnt >= 2 and col not in kso:
                cols_with_multi.add(col)

    if "and" in where:
        for item in where.get("and") or []:
            cols_with_multi.update(_where_or_multi_value_columns(item, kso))

    return cols_with_multi


def _fix_missing_dimension_for_multi_value(
    plan: Dict[str, Any], table_meta: Dict[str, Dict[str, Any]]
) -> None:
    """Add missing dimension when plan filters by multiple values of a column without grouping.

    When the user asks "how many X for A, B, C — give a total for each",
    the LLM often produces dimensions: [] with a where.or on the entity column.
    This fix detects that pattern and adds the column as a dimension so the
    result includes per-value breakdowns.

    Only runs on the outermost plan (not CTE sub-plans).
    """
    ds = plan.get("dataset")
    if not isinstance(ds, str):
        return

    meta = table_meta.get(ds, {})
    kso = meta.get("keyword_search_or") or set()

    existing_dims = set()
    for d in plan.get("dimensions") or []:
        if isinstance(d, dict):
            existing_dims.add((d.get("field") or "").split(".")[-1])

    multi_val_cols: Set[str] = set()

    for f in plan.get("filters") or []:
        if not isinstance(f, dict):
            continue
        op = (f.get("op") or "").lower()
        field = (f.get("field") or "").split(".")[-1]
        if op == "in" and field and field not in kso:
            val = f.get("value")
            if isinstance(val, list) and len(val) >= 2:
                multi_val_cols.add(field)

    where = plan.get("where")
    if where:
        multi_val_cols.update(_where_or_multi_value_columns(where, kso))

    has_metrics = bool(plan.get("metrics"))
    if not has_metrics:
        return

    if not multi_val_cols:
        return

    filter_cols = set()
    for f in plan.get("filters") or []:
        if isinstance(f, dict):
            fc = (f.get("field") or "").split(".")[-1]
            if fc:
                filter_cols.add(fc)

    for col in sorted(multi_val_cols):
        if col not in existing_dims:
            dims = plan.get("dimensions")
            if dims is None:
                dims = []
                plan["dimensions"] = dims
            dims.append({"field": col, "alias": col})
            if plan.get("limit") == 1:
                plan["limit"] = 100
            print(
                f"[IntentQL autofix] Added missing dimension '{col}' for multi-value filter",
                file=sys.stderr,
            )

    dims = plan.get("dimensions") or []
    if len(dims) > 1 and multi_val_cols:
        keep = []
        removed_names = []
        for d in dims:
            if not isinstance(d, dict):
                keep.append(d)
                continue
            field = (d.get("field") or "").split(".")[-1]
            if field in multi_val_cols or field in filter_cols:
                keep.append(d)
            else:
                removed_names.append(field)
        if removed_names:
            plan["dimensions"] = keep
            print(
                f"[IntentQL autofix] Removed unneeded dimensions {removed_names} "
                f"(not in filters/multi-value columns)",
                file=sys.stderr,
            )
