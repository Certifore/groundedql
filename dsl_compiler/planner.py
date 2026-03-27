from __future__ import annotations

from typing import Any, Dict, List, Optional

import yaml
from pathlib import Path

from .queryplan_models import queryplan_json_schema
from .validation import validate_query_plan_dict, ValidationErrorItem
from .llm_adapters import make_llm_client
from .api.api import _resolve_relative_dates
from .semantic_lint import semantic_lint
from .join_planner import auto_inject_joins
from .plan_canonical import canonicalize_query_plan, plan_fingerprint
from .spec_builder import build_minimal_queryplan_spec


SCALAR_AGGS = {"count", "count_distinct", "sum", "avg", "min", "max"}

_TOP_N_QUESTION_SIGNALS = [
    r"\btop\s+\d+\b", r"\bbottom\s+\d+\b", r"\bmost\b", r"\bleast\b",
    r"\bhighest\b", r"\blowest\b", r"\branked\b", r"\bbest\b", r"\bworst\b",
]

import re

def _has_top_n_signal(question: str) -> bool:
    q = question.lower()
    return any(re.search(p, q) for p in _TOP_N_QUESTION_SIGNALS)


def _is_scalar_aggregate(plan_dict: dict) -> bool:
    """
    Returns True if the plan has no dimensions and all metrics are pure aggregations.
    These queries always return exactly one row, so limit=1 is correct.
    """
    if not isinstance(plan_dict, dict):
        return False
    dimensions = plan_dict.get("dimensions", []) or []
    metrics = plan_dict.get("metrics", []) or []
    rollup = plan_dict.get("rollup")

    if rollup is not None:
        return False  # rollup handles its own limit
    if dimensions:
        return False
    if not metrics:
        return False
    return all(
        isinstance(m, dict) and (m.get("agg") or "").lower() in SCALAR_AGGS
        for m in metrics
    )


def _auto_fix_plan(plan_dict: dict, question: str = "", schema: Optional[Dict[str, Any]] = None) -> tuple[dict, list[str]]:
    """
    Apply deterministic post-generation fixes to the plan.
    Returns (fixed_plan, list_of_fixes_applied).
    """
    fixes = []
    plan_dict = _resolve_relative_dates(plan_dict)

    if _is_scalar_aggregate(plan_dict):
        plan_dict = {**plan_dict, "limit": 1, "offset": 0}
        fixes.append("scalar_aggregate_limit_clamped_to_1")

    # LIMIT policy: grouped queries without a top-N signal should not have
    # a default limit that silently truncates grouped results.
    dimensions = plan_dict.get("dimensions", []) or []
    rollup = plan_dict.get("rollup")
    if (
        dimensions
        and rollup is not None
        and plan_dict.get("limit") is not None
        and not _has_top_n_signal(question)
    ):
        plan_dict = {k: v for k, v in plan_dict.items() if k != "limit"}
        fixes.append("inner_rollup_limit_removed_for_full_aggregation")

    # Auto-inject joins when multiple tables are referenced but no joins declared
    if schema is not None:
        original = plan_dict
        plan_dict = auto_inject_joins(plan_dict, schema)
        if plan_dict is not original:
            fixes.append("joins_auto_injected_from_link_graph")

    plan_dict = canonicalize_query_plan(plan_dict)
    fixes.append("plan_canonicalized")

    return plan_dict, fixes


def _plan_hash(plan_dict: dict) -> str:
    return plan_fingerprint(plan_dict)[:12]


def _format_errors(errors: List[ValidationErrorItem]) -> str:
    lines = []
    for e in errors:
        lines.append(f"- {e.path}: {e.message}")
    return "\n".join(lines)


def build_planner_messages(*, question: str, schema_yaml_text: str, spec_text: str) -> List[Dict[str, str]]:
    system = (
        "You are a QueryPlan generator.\n"
        "Return ONLY a JSON object that conforms to the provided JSON Schema.\n"
        "Use ONLY logical snake_case field names from schema.yaml.\n"
        "Do NOT output SQL.\n"
    )

    context = (
        "SCHEMA.YAML (logical datasets + fields):\n"
        f"{schema_yaml_text}\n\n"
        "QUERYPLAN SPEC:\n"
        f"{spec_text}\n"
    )

    return [
        {"role": "system", "content": system},
        {"role": "system", "content": context},
        {"role": "user", "content": question},
    ]


class QueryPlanPlanner:
    def __init__(
        self,
        *,
        llm: Any,
        schema_path: str,
        spec_path: str | None = None,
        spec_dict: dict | None = None,
        temperature: float = 0.0,
        model: Optional[str] = None,
    ):
        self.llm = make_llm_client(llm, model=model)
        self.schema_path = schema_path
        self.temperature = temperature

        if spec_dict is not None:
            self.spec = spec_dict
        elif spec_path is not None:
            self.spec = yaml.safe_load(Path(spec_path).read_text())
        else:
            # Fallback: auto-generate a minimal spec from the schema
            self.spec = build_minimal_queryplan_spec(schema_path)

    def _read_text(self, path: str) -> str:
        with open(path, "r") as f:
            return f.read()

    def _read_schema(self) -> Dict[str, Any]:
        with open(self.schema_path, "r") as f:
            return yaml.safe_load(f) or {}

    def plan(self, question: str) -> Dict[str, Any]:
        schema_text = self._read_text(self.schema_path)
        spec_text = yaml.dump(self.spec)
        schema = self._read_schema()

        messages = build_planner_messages(
            question=question,
            schema_yaml_text=schema_text,
            spec_text=spec_text,
        )

        plan_dict = self.llm.generate_json(
            json_schema=queryplan_json_schema(),
            messages=messages,
            temperature=self.temperature,
        )
        plan_dict, _ = _auto_fix_plan(plan_dict, question, schema)
        print(f"[DSL] LLM QueryPlan: {plan_dict}")
        return plan_dict

    def plan_with_retry(self, question: str, max_retries: int = 1) -> Dict[str, Any]:
        schema_text = self._read_text(self.schema_path)
        spec_text = yaml.dump(self.spec)
        schema = self._read_schema()

        base_messages = build_planner_messages(
            question=question,
            schema_yaml_text=schema_text,
            spec_text=spec_text,
        )

        all_auto_fixes: list[str] = []
        retry_count = 0

        plan_dict = self.llm.generate_json(
            json_schema=queryplan_json_schema(),
            messages=base_messages,
            temperature=self.temperature,
        )
        plan_dict, fixes = _auto_fix_plan(plan_dict, question, schema)
        all_auto_fixes.extend(fixes)
        print(f"[DSL] LLM QueryPlan: {plan_dict}")

        _, errs = validate_query_plan_dict(plan_dict, self.schema_path)
        lint_errs = semantic_lint(question, plan_dict, schema)

        while (errs or lint_errs) and retry_count < max_retries:
            retry_count += 1

            # Targeted retry: schema errors and lint errors get separate sections
            # so the LLM knows exactly what type of fix is needed.
            feedback_parts = ["The previous JSON did NOT pass checks. Fix and output a corrected QueryPlan.\n"]

            if errs:
                feedback_parts.append(
                    "STRUCTURAL ERRORS (fix the JSON shape/field references):\n"
                    + _format_errors(errs)
                )
            if lint_errs:
                feedback_parts.append(
                    "SEMANTIC ERRORS (plan does not match question intent — fix the logic):\n"
                    + "\n".join(f"- {e}" for e in lint_errs)
                )

            messages = base_messages + [{"role": "system", "content": "\n\n".join(feedback_parts)}]

            plan_dict = self.llm.generate_json(
                json_schema=queryplan_json_schema(),
                messages=messages,
                temperature=self.temperature,
            )
            plan_dict, fixes = _auto_fix_plan(plan_dict, question, schema)    # pass schema
            all_auto_fixes.extend(fixes)
            print(f"[DSL] LLM QueryPlan (retry {retry_count}): {plan_dict}")

            _, errs = validate_query_plan_dict(plan_dict, self.schema_path)
            lint_errs = semantic_lint(question, plan_dict, schema)   # <-- pass schema

        # Attach explainability metadata
        plan_dict["meta"] = {
            "plan_hash": _plan_hash({k: v for k, v in plan_dict.items() if k != "meta"}),
            "retry_count": retry_count,
            "auto_fixes_applied": all_auto_fixes,
            "validation_errors": [{"path": e.path, "message": e.message} for e in errs],
            "lint_errors": lint_errs,
        }

        return plan_dict