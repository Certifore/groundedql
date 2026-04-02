"""
decompose.py — Detect and split compound natural-language questions.

A "compound" question asks for multiple deliverables at different grains
in one sentence, e.g.:

    "How many plumbing work orders were in Page House last year
     and list the most recent 10 and which rooms?"

This requires a global COUNT *and* a detail LIST — two incompatible query
shapes.  Rather than forcing the LLM planner to produce a single plan that
satisfies both (which it routinely gets wrong), we split the question into
focused sub-questions and run each independently.

Public API
----------
is_compound(question)          → bool
split_compound(question)       → list[SubQuestion]
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SubQuestion:
    """One focused sub-question derived from a compound question."""
    text: str
    role: str  # "count" | "list"


_COMPOUND_RE = re.compile(
    r"(?=.*\bhow many\b)"
    r"(?=.*\b(?:list|most recent|top\s+\d+|show)\b)"
    r"(?=.*\b(?:which|what)\b)",
    re.I,
)


def is_compound(question: str) -> bool:
    """True when *question* asks for a count AND a ranked listing AND detail columns."""
    return bool(_COMPOUND_RE.search(question))


def split_compound(question: str) -> List[SubQuestion]:
    """
    Split a compound count+list+detail question into focused sub-questions.

    Returns a list of :class:`SubQuestion` (typically two: one ``"count"``
    and one ``"list"``).  If the question is not compound, returns a single
    ``SubQuestion`` with the original text and ``role="single"``.
    """
    if not is_compound(question):
        return [SubQuestion(text=question, role="single")]

    count_q = re.sub(
        r"\b(and\s+)?(list|show)\s+(me\s+)?(the\s+)?(most recent|latest|top)\s+\d+.*$",
        "", question, flags=re.I,
    ).strip().rstrip("?.,;:") + "?"

    if count_q.lower().strip("? ") == question.lower().strip("? "):
        count_q = question.split(" and ")[0].strip() + "?"

    base = re.sub(r"^how many\s+", "", count_q, flags=re.I).strip().rstrip("?")

    top_n_match = re.search(r"\b(?:top|most recent|latest)\s+(\d+)\b", question, re.I)
    limit_n = int(top_n_match.group(1)) if top_n_match else 10

    detail_match = re.search(r"\b(?:which|what)\s+(\w+)", question, re.I)
    detail_col = detail_match.group(1) if detail_match else ""

    sort_key = "most recent" if re.search(r"\bmost recent\b", question, re.I) else "top"
    list_q = f"List the {sort_key} {limit_n} {base}"
    if detail_col:
        list_q += f" and show which {detail_col}"
    list_q += "?"

    return [
        SubQuestion(text=count_q, role="count"),
        SubQuestion(text=list_q, role="list"),
    ]
