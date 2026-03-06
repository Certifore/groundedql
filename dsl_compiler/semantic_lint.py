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
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def semantic_lint(question: str, plan: Dict[str, Any]) -> List[str]:
    """
    Run all semantic lint checks.

    Args:
        question: The raw user question string.
        plan:     The resolved QueryPlan dict (after _auto_fix_plan).

    Returns:
        List of lint error strings (empty = no issues found).
    """
    q = _q(question)
    errors: List[str] = []

    _check_distinct(q, plan, errors)
    _check_grouping(q, plan, errors)
    _check_top_n(q, plan, errors)
    _check_two_step_aggregation(q, plan, errors)

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
# Rule 4: Two-step aggregation intent → rollup must be present
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
# Utility
# ---------------------------------------------------------------------------

def _first_match(patterns: List[str], text: str) -> str:
    """Return the first matching substring for display in error messages."""
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(0)
    return ""
