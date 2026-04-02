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
from typing import Any, Dict, List, Optional


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
    _check_distinct(q, plan, errors)
    _check_grouping(q, plan, errors)
    _check_top_n(q, plan, errors)
    _check_two_step_aggregation(q, plan, errors)

    if schema is not None:
        _check_grain(q, plan, schema, errors)

    return errors


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

        if agg in {"count", "count_distinct"} and field not in {"*", primary_id}:
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
        return  # count() with no args = count(*) — fine

    # Extract the column reference from the first arg
    first_arg = args[0] if args else {}
    if not isinstance(first_arg, dict):
        return

    col_ref = first_arg.get("col", "")
    if not col_ref:
        return

    # Strip table prefix if present (e.g. "work_orders.phase_id" -> "phase_id")
    col_name = col_ref.split(".")[-1] if "." in col_ref else col_ref

    if col_name not in {"*", primary_id}:
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
