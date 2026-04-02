"""
semantic_lint.py — Question-aware plan linter.

Compares the user's raw question against the generated plan and returns
a list of actionable lint error strings. Empty list = clean.

Rules are intentionally conservative: only fire on high-confidence,
unambiguous signal words. False negatives are harmless; false positives
waste one retry but don't break anything.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set


# Helpers

def _q(question: str) -> str:
    """Lowercase + normalise whitespace for matching."""
    return re.sub(r"\s+", " ", question.lower().strip())


def _has(pattern: str, text: str) -> bool:
    """Case-insensitive word-boundary regex match."""
    return bool(re.search(pattern, text, re.IGNORECASE))


def _metrics(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    return plan.get("metrics") or []


def _dimensions(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    return plan.get("dimensions") or []


def _order_by(plan: Dict[str, Any]) -> List[Any]:
    return plan.get("order_by") or []


def _is_multi_part_deliverables_question(q: str) -> bool:
    """Count + list/rank + which/what detail — needs compound CTE plan, not one grouped query."""
    has_count = _has(r"\bhow many\b", q) or _has(r"\bnumber of\b", q)
    has_list = (
        _has(r"\blist\b", q)
        or _has(r"\bmost recent\b", q)
        or _has(r"\btop\s+\d+\b", q)
        or _has(r"\bshow (me )?the\b", q)
    )
    has_detail = _has(r"\bwhich\b", q) or _has(r"\bwhat are the\b", q) or _has(r"\bwhat is the\b", q)
    return bool(has_count and has_list and has_detail)


# Public API

def semantic_lint(
    question: str,
    plan: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Run all semantic lint checks.

    Args:
        question: The raw user question string.
        plan:     The resolved QueryPlan dict (after _auto_fix_plan).
        schema:   Parsed schema dict (optional). When provided, enables grain checks.

    Returns:
        List of lint error strings (empty = no issues found).
    """
    q = _q(question)
    errors: List[str] = []

    _check_multi_part_compound(q, plan, errors)
    _check_trade_on_workflow_columns(q, plan, errors)
    _check_compound_outer_uses_filtered_cte(q, plan, errors)
    _check_distinct(q, plan, errors)
    _check_grouping(q, plan, errors)
    _check_top_n(q, plan, errors)
    _check_two_step_aggregation(q, plan, errors)

    if schema is not None:
        _check_grain(q, plan, schema, errors)
        _check_keyword_search_or(plan, schema, errors)

    return errors


# ---------------------------------------------------------------------------
# Trade keywords vs workflow columns (order_type / order_category)
# ---------------------------------------------------------------------------

_TRADE_IN_QUESTION = re.compile(
    r"\b(plumbing|plumb|hvac|electrical|electric|mechanical|carpentry|steam|pipefit)\b",
    re.I,
)

_WORKFLOW_ENUM_FIELDS = frozenset({"order_type", "order_category"})

# Values that look like a trade/craft encoded as a filter (not PLANNED / PREVENTIVE / …)
_TRADE_VALUE_HINT = re.compile(r"plumb|hvac|electric|carpent|mechanic|steamfit|pipe", re.I)


def _iter_plans_depth_first(plan: Dict[str, Any]):
    """Yield this plan and every nested CTE inner plan."""
    yield plan
    for w in plan.get("with") or []:
        if isinstance(w, dict) and isinstance(w.get("plan"), dict):
            yield from _iter_plans_depth_first(w["plan"])


def _collect_filters_from_plan_tree(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in _iter_plans_depth_first(plan):
        out.extend(p.get("filters") or [])
    return out


def _check_trade_on_workflow_columns(q: str, plan: Dict[str, Any], errors: List[str]) -> None:
    """
    Fire when the question mentions a trade but filters order_type/order_category with a
    trade-shaped value — common LLM mistake; trades usually live in free-text fields.
    """
    if not _TRADE_IN_QUESTION.search(q):
        return
    for f in _collect_filters_from_plan_tree(plan):
        field = f.get("field")
        if not isinstance(field, str):
            continue
        field = field.split(".")[-1]
        if field not in _WORKFLOW_ENUM_FIELDS:
            continue
        val = f.get("value")
        if not isinstance(val, str) or not _TRADE_VALUE_HINT.search(val):
            continue
        op = (f.get("op") or "").lower()
        if op not in ("contains", "=", "starts_with", "ends_with"):
            continue
        errors.append(
            "Lint: question mentions a trade/skill; "
            f"filter on '{field}' with value {val!r} looks like a trade term. "
            "Columns like order_type and order_category usually hold workflow values "
            "(PLANNED, HOUSING, PREVENTIVE, …), not trade names. "
            "Use op 'contains' on description/component/shop free-text columns (often OR several), "
            "unless the schema states that trade appears in this column."
        )
        return


def _substantive_outer_filters(filters: List[Any]) -> List[Dict[str, Any]]:
    """
    Filters that actually restrict rows (building, dates, text search).
    LLMs often add `work_order_id is_not_null` on the outer query to satisfy linters
    while still selecting from an unfiltered base table — treat those as non-substantive.
    """
    out: List[Dict[str, Any]] = []
    for f in filters or []:
        if not isinstance(f, dict):
            continue
        op = (f.get("op") or "").lower()
        field = (f.get("field") or "").split(".")[-1]
        if op in {"is_null", "is_not_null"} and field.endswith("id"):
            continue
        out.append(f)
    return out


def _check_compound_outer_uses_filtered_cte(q: str, plan: Dict[str, Any], errors: List[str]) -> None:
    """
    Fire when a multi-deliverable question uses WITH, a CTE applies filters, but the outer
    query selects from a base dataset with empty filters (ignores the CTE predicates).
    """
    if not _is_multi_part_deliverables_question(q):
        return
    ctes = plan.get("with") or []
    if not ctes:
        return
    cte_names: set[str] = set()
    inner_has_filters = False
    for w in ctes:
        if not isinstance(w, dict):
            continue
        n = w.get("name")
        if isinstance(n, str):
            cte_names.add(n)
        inner = w.get("plan") or {}
        if inner.get("filters") or inner.get("where"):
            inner_has_filters = True
    if not inner_has_filters:
        return
    if _substantive_outer_filters(plan.get("filters") or []):
        return
    # Outer query may use only advanced `where` (merged with filters in the compiler).
    if plan.get("where") is not None:
        return
    ds = plan.get("dataset")
    if not isinstance(ds, str):
        return
    if ds in cte_names:
        return
    errors.append(
        "Lint: compound plan has a CTE with filters but the outer query uses dataset "
        f"'{ds}' with empty filters. Point the outer FROM at the CTE that applies the same "
        "building/keyword/date filters, or repeat those filters on the outer query."
    )


def _or_tree_contains_all_keyword_cols(where: Any, cols: Set[str]) -> bool:
    """True if `where` has an `or` branch whose children are `contains` on each logical column in cols."""
    if not isinstance(where, dict):
        return False
    if "or" in where:
        found: set[str] = set()
        for item in where.get("or") or []:
            if not isinstance(item, dict) or "cmp" not in item:
                continue
            c = item["cmp"]
            left = c.get("left") or {}
            col = left.get("col") if isinstance(left, dict) else None
            if isinstance(col, str):
                col = col.split(".")[-1]
            op = (c.get("op") or "").lower()
            if col in cols and op == "contains":
                found.add(col)
        if cols.issubset(found):
            return True
    if "and" in where:
        for item in where.get("and") or []:
            if _or_tree_contains_all_keyword_cols(item, cols):
                return True
    return False


def _check_keyword_search_or(plan: Dict[str, Any], schema: Dict[str, Any], errors: List[str]) -> None:
    """
    Optional schema.yaml per table: keyword_search_or: [col, ...] (>=2 logical columns).
    If set, legacy `contains` on any of those columns requires advanced `where.or` across all of them
    for OR semantics; otherwise the model often emits a single column only.
    """
    for t in schema.get("tables", []) or []:
        name = t.get("name")
        if not isinstance(name, str):
            continue
        kso = t.get("keyword_search_or")
        if not isinstance(kso, list) or len(kso) < 2:
            continue
        cols = [c for c in kso if isinstance(c, str)]
        if len(cols) < 2:
            continue
        colset = set(cols)
        for sub in _iter_plans_depth_first(plan):
            if sub.get("dataset") != name:
                continue
            has_correct_or = _or_tree_contains_all_keyword_cols(sub.get("where"), colset)

            legacy_kw: List[str] = []
            for f in sub.get("filters") or []:
                if (f.get("op") or "").lower() != "contains":
                    continue
                field = (f.get("field") or "").split(".")[-1]
                if field in colset:
                    legacy_kw.append(field)

            if has_correct_or and legacy_kw:
                keep = []
                for f in sub.get("filters") or []:
                    field = (f.get("field") or "").split(".")[-1]
                    if field not in colset or (f.get("op") or "").lower() != "contains":
                        keep.append(f)
                keep_desc = ", ".join(
                    f"{f['field']} {f['op']} {f.get('value','')!r}" for f in keep
                ) or "(none)"
                errors.append(
                    f"Lint: table '{name}' keyword_search_or {cols!r}: the `where` OR tree is correct "
                    f"but `filters` also contain AND-ed `contains` on {sorted(set(legacy_kw))!r}. "
                    "Those AND filters defeat the OR intent. "
                    f"Remove ONLY these fields from `filters`: {sorted(set(legacy_kw))!r}. "
                    f"KEEP all other filters unchanged: [{keep_desc}]. "
                    "The `where.or` already handles the keyword search correctly."
                )
                return

            if has_correct_or:
                continue

            if not legacy_kw:
                continue
            uniq_kw = set(legacy_kw)
            if len(uniq_kw) >= 2:
                errors.append(
                    f"Lint: table '{name}' keyword_search_or {cols!r}: legacy filters AND "
                    f"`contains` on {sorted(uniq_kw)!r} — every row must match all of them, "
                    "which is narrower than OR across keyword columns (and omits OR branches for "
                    "columns not listed). Use advanced `where` with `or` of `cmp` contains nodes "
                    f"covering every column in keyword_search_or {cols!r}."
                )
            else:
                errors.append(
                    f"Lint: table '{name}' declares keyword_search_or {cols!r} in schema.yaml — "
                    "use OR of `contains` on those columns via advanced `where` with `or` of `cmp` nodes; "
                    "legacy filters are ANDed. Omit or narrow keyword_search_or if single-column search is intended."
                )
            return


# ---------------------------------------------------------------------------
# Rule 1: Distinct intent → count_distinct
# ---------------------------------------------------------------------------

_DISTINCT_PATTERNS = [
    r"\bdistinct\b",
    r"\bunique\b",
    r"\bdifferent\b",
    r"\bhow many different\b",
    r"\bhow many unique\b",
]

def _check_distinct(q: str, plan: Dict[str, Any], errors: List[str]) -> None:
    if not any(_has(p, q) for p in _DISTINCT_PATTERNS):
        return

    # Check if any metric uses plain `count` on a non-star field
    for m in _metrics(plan):
        agg = (m.get("agg") or "").lower()
        field = m.get("field", "*")
        if agg == "count" and field != "*":
            errors.append(
                f"Lint: question implies distinct counting ('{_first_match(_DISTINCT_PATTERNS, q)}') "
                f"but metric '{m.get('alias', field)}' uses agg='count' — "
                f"did you mean agg='count_distinct'?"
            )


# ---------------------------------------------------------------------------
# Rule 2: Grouping intent → dimensions must be non-empty
# ---------------------------------------------------------------------------

_GROUPING_PATTERNS = [
    r"\bper\s+\w+\b",        # "per building" — but NOT "percent"
    r"\bby\s+\w+\b",         # "by building"
    r"\beach\s+\w+\b",       # "each building"
    r"\bfor each\b",
    r"\bbroken down by\b",
    r"\bgrouped by\b",
    r"\bgroup by\b",
]

_GROUPING_EXCLUSIONS = [
    r"\bpercent\b",
    r"\bpercentage\b",
    r"\bpercentile\b",
]

def _check_grouping(q: str, plan: Dict[str, Any], errors: List[str]) -> None:
    # Skip if question contains excluded words that match "per" accidentally
    if any(_has(p, q) for p in _GROUPING_EXCLUSIONS):
        return
    if not any(_has(p, q) for p in _GROUPING_PATTERNS):
        return

    # Only fire if there are metrics (pure list queries legitimately have no dimensions)
    if not _metrics(plan):
        return

    if not _dimensions(plan):
        match = _first_match(_GROUPING_PATTERNS, q)
        errors.append(
            f"Lint: question implies grouping ('{match}') "
            f"but plan has no dimensions — add the grouping field to dimensions."
        )


# ---------------------------------------------------------------------------
# Rule 3: Top-N intent → order_by + limit must be set
# ---------------------------------------------------------------------------

_TOP_N_PATTERNS = [
    r"\btop\s+\d+\b",
    r"\bbottom\s+\d+\b",
    r"\bmost\b",
    r"\bleast\b",
    r"\bhighest\b",
    r"\blowest\b",
    r"\branked\b",
    r"\bbest\b",
    r"\bworst\b",
]

def _check_top_n(q: str, plan: Dict[str, Any], errors: List[str]) -> None:
    if not any(_has(p, q) for p in _TOP_N_PATTERNS):
        return
    # "most recent" / "top N" inside a multi-clause question is addressed by compound "with", not order_by on a single legacy plan
    if _is_multi_part_deliverables_question(q):
        return

    match = _first_match(_TOP_N_PATTERNS, q)
    missing = []

    if not _order_by(plan):
        missing.append("order_by")
    if plan.get("limit") is None:
        missing.append("limit")

    if missing:
        errors.append(
            f"Lint: question implies ranking ('{match}') "
            f"but plan is missing: {', '.join(missing)}. "
            f"Add order_by (desc for most/highest/best, asc for least/lowest/worst) and a limit."
        )


# ---------------------------------------------------------------------------
# Rule 4: Multi-part deliverables (count + list + detail) → compound "with"
# ---------------------------------------------------------------------------

def _check_multi_part_compound(q: str, plan: Dict[str, Any], errors: List[str]) -> None:
    if not _is_multi_part_deliverables_question(q):
        return
    if plan.get("with"):
        return
    if plan.get("set_op") or plan.get("rollup"):
        return
    # Single-query escape hatch: substantive legacy filters + advanced `where` (e.g. OR across
    # keyword_search_or). Prefer separate CTEs for count vs list when grains differ, but allow
    # execution when the planner already combined building/date filters with a `where.or` tree.
    if _substantive_outer_filters(plan.get("filters") or []) and plan.get("where") is not None:
        return

    errors.append(
        "Lint: the question asks for multiple deliverables (e.g. a count, a ranked list, and detail). "
        'Prefer a compound plan with top-level "with" (CTEs): pipeline filters in early CTEs, '
        "outer SELECT for listing or metrics; use separate CTEs when counts and lists need different grains. "
        "Avoid one grouped legacy query that mixes unrelated grains."
    )


# ---------------------------------------------------------------------------
# Rule 5: Two-step aggregation intent → rollup must be present
# ---------------------------------------------------------------------------

_TWO_STEP_PATTERNS = [
    r"\baverage\s+per\b",
    r"\bavg\s+per\b",
    r"\bmean\s+per\b",
    # Allow any words between the stat function and "per" (e.g. "average number of work orders per")
    r"\baverage\s+\w+.*?\bper\b",
    r"\bmean\s+\w+.*?\bper\b",
    r"\bstandard\s+deviation\s+\w+.*?\bper\b",
    r"\bstddev\s+\w+.*?\bper\b",
    r"\bvariance\s+\w+.*?\bper\b",
    r"\bmedian\s+\w+.*?\bper\b",
    r"\bpercentile\s+\w+.*?\bper\b",
    # Catch "X per Y" where X is a stat function name directly before "per"
    r"\bstandard\s+deviation\s+per\b",
    r"\bstddev\s+per\b",
    r"\bvariance\s+per\b",
    r"\bmedian\s+per\b",
    r"\bpercentile\s+per\b",
]

def _check_two_step_aggregation(q: str, plan: Dict[str, Any], errors: List[str]) -> None:
    if not any(_has(p, q) for p in _TWO_STEP_PATTERNS):
        return

    if not plan.get("rollup"):
        match = _first_match(_TWO_STEP_PATTERNS, q)
        errors.append(
            f"Lint: question implies an aggregate-of-aggregates ('{match}') "
            f"but plan has no rollup. "
            f"Use a two-step plan: inner query groups and computes per-group metric, "
            f"outer rollup computes the aggregate over those grouped values."
        )


# ---------------------------------------------------------------------------
# Rule 6: Grain — count metrics should use primary_id when counting entities
# ---------------------------------------------------------------------------

_COUNT_QUESTION_PATTERNS = [
    r"\bhow many\b",
    r"\bcount of\b",
    r"\bnumber of\b",
    r"\btotal\s+\w+\b",
]

def _check_grain(q: str, plan: Dict[str, Any], schema: Dict[str, Any], errors: List[str]) -> None:
    """
    If the question asks "how many X" and the dataset has a primary_id declared,
    ensure that count/count_distinct metrics use the primary_id field, not another field.
    """
    if not any(_has(p, q) for p in _COUNT_QUESTION_PATTERNS):
        return

    # If the user explicitly says "distinct X", they are intentionally counting
    # a dimension — not the dataset's primary entity. Rule 1 (distinct check)
    # already handles correctness for that case. Skip grain check.
    if any(_has(p, q) for p in _DISTINCT_PATTERNS):
        return

    dataset = plan.get("dataset")
    if not dataset:
        return

    # Build a lookup of primary_id by table name from the schema
    primary_ids: Dict[str, str] = {}
    for t in schema.get("tables", []):
        pid = t.get("primary_id")
        if pid:
            primary_ids[t["name"]] = pid

    primary_id = primary_ids.get(dataset)
    if not primary_id:
        return  # no grain declared for this table — skip

    # Check legacy metrics
    for m in _metrics(plan):
        agg = (m.get("agg") or "").lower()
        field = m.get("field", "*")
        alias = m.get("alias", field)

        if agg in {"count", "count_distinct"} and field == "*":
            errors.append(
                f"Lint: question implies counting {dataset} entities ('{_first_match(_COUNT_QUESTION_PATTERNS, q)}') "
                f"but metric '{alias}' uses count(*) which counts rows, not distinct entities. "
                f"The declared primary identifier for '{dataset}' is '{primary_id}'. "
                f"Use agg='count_distinct', field='{primary_id}' for an accurate entity count."
            )
        elif agg == "count" and field == primary_id:
            errors.append(
                f"Lint: question implies counting {dataset} entities ('{_first_match(_COUNT_QUESTION_PATTERNS, q)}') "
                f"but metric '{alias}' uses agg='count' on '{primary_id}' — "
                f"COUNT({primary_id}) counts rows (including duplicates from joins/phases). "
                f"Use agg='count_distinct', field='{primary_id}' for an accurate entity count."
            )
        elif agg in {"count", "count_distinct"} and field != primary_id:
            errors.append(
                f"Lint: question implies counting {dataset} entities ('{_first_match(_COUNT_QUESTION_PATTERNS, q)}') "
                f"but metric '{alias}' counts '{field}' — "
                f"the declared primary identifier for '{dataset}' is '{primary_id}'. "
                f"Use agg='count_distinct', field='{primary_id}' for an accurate count."
            )

    # Check advanced format select items for count/count_distinct func nodes
    for item in (plan.get("select") or []):
        if not isinstance(item, dict):
            continue
        expr = item.get("expr") or {}
        alias = item.get("alias", "")
        _check_grain_expr(expr, dataset, primary_id, alias, q, errors)


def _check_grain_expr(
    expr: Dict[str, Any],
    dataset: str,
    primary_id: str,
    alias: str,
    q: str,
    errors: List[str],
) -> None:
    """Recursively check advanced format expression nodes for grain violations."""
    if not isinstance(expr, dict):
        return

    fn = (expr.get("func") or "").lower()
    if fn not in {"count", "count_distinct", "countdistinct"}:
        return

    args = expr.get("args") or []
    if not args:
        errors.append(
            f"Lint: question implies counting {dataset} entities ('{_first_match(_COUNT_QUESTION_PATTERNS, q)}') "
            f"but advanced select '{alias}' uses count() with no args (= count(*)). "
            f"The declared primary identifier for '{dataset}' is '{primary_id}'. "
            f"Use count_distinct('{primary_id}') for an accurate entity count."
        )
        return

    # Extract the column reference from the first arg
    first_arg = args[0] if args else {}
    if not isinstance(first_arg, dict):
        return

    col_ref = first_arg.get("col", "")
    if not col_ref:
        return

    # Strip table prefix if present (e.g. "work_orders.phase_id" -> "phase_id")
    col_name = col_ref.split(".")[-1] if "." in col_ref else col_ref

    if col_name == "*":
        errors.append(
            f"Lint: question implies counting {dataset} entities ('{_first_match(_COUNT_QUESTION_PATTERNS, q)}') "
            f"but advanced select '{alias}' uses count(*) which counts rows. "
            f"The declared primary identifier for '{dataset}' is '{primary_id}'. "
            f"Use count_distinct('{primary_id}') for an accurate entity count."
        )
        return

    if col_name != primary_id:
        errors.append(
            f"Lint: question implies counting {dataset} entities ('{_first_match(_COUNT_QUESTION_PATTERNS, q)}') "
            f"but advanced select '{alias}' counts '{col_name}' — "
            f"the declared primary identifier for '{dataset}' is '{primary_id}'. "
            f"Use count_distinct('{primary_id}') for an accurate count."
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _first_match(patterns: List[str], text: str) -> str:
    """Return the first matching substring for display in error messages."""
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(0)
    return ""
