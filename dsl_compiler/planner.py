from __future__ import annotations

from typing import Any, Dict, List, Optional

from .queryplan_models import queryplan_json_schema
from .validation import validate_query_plan_dict, ValidationErrorItem
from .llm_adapters import make_llm_client
from .api.api import _resolve_relative_dates
from .semantic_lint import semantic_lint


SCALAR_AGGS = {"count", "count_distinct", "sum", "avg", "min", "max"}


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


def _auto_fix_plan(plan_dict: dict) -> dict:
    """Apply deterministic post-generation fixes to the plan."""
    # Resolve relative date sentinels BEFORE Pydantic validation so
    # $relative_date dicts don't fail the Union[str, int, float, ...] type check.
    plan_dict = _resolve_relative_dates(plan_dict)

    if _is_scalar_aggregate(plan_dict):
        plan_dict = {**plan_dict, "limit": 1, "offset": 0}
    return plan_dict


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
        llm,
        schema_path: str,
        spec_path: str,
        temperature: float = 0.0,
        model: Optional[str] = None,
    ):
        self.llm = make_llm_client(llm, model=model)
        self.schema_path = schema_path
        self.spec_path = spec_path
        self.temperature = temperature

    def _read_text(self, path: str) -> str:
        with open(path, "r") as f:
            return f.read()

    def plan(self, question: str) -> Dict[str, Any]:
        schema_text = self._read_text(self.schema_path)
        spec_text = self._read_text(self.spec_path)

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
        plan_dict = _auto_fix_plan(plan_dict)
        print(f"[DSL] LLM QueryPlan: {plan_dict}")
        return plan_dict

    def plan_with_retry(self, question: str, max_retries: int = 1) -> Dict[str, Any]:
        """
        Attempt to generate a valid QueryPlan, retrying once with validation
        error feedback if the first attempt fails validation.
        """
        schema_text = self._read_text(self.schema_path)
        spec_text = self._read_text(self.spec_path)

        base_messages = build_planner_messages(
            question=question,
            schema_yaml_text=schema_text,
            spec_text=spec_text,
        )

        # attempt 1
        plan_dict = self.llm.generate_json(
            json_schema=queryplan_json_schema(),
            messages=base_messages,
            temperature=self.temperature,
        )
        plan_dict = _auto_fix_plan(plan_dict)
        print(f"[DSL] LLM QueryPlan: {plan_dict}")

        _, errs = validate_query_plan_dict(plan_dict, self.schema_path)
        lint_errs = semantic_lint(question, plan_dict)

        if not errs and not lint_errs:
            return plan_dict

        retries = 0
        while (errs or lint_errs) and retries < max_retries:
            retries += 1

            feedback_parts = []
            if errs:
                feedback_parts.append(
                    "Validation errors:\n" + _format_errors(errs)
                )
            if lint_errs:
                feedback_parts.append(
                    "Semantic lint warnings (plan does not match question intent):\n"
                    + "\n".join(f"- {e}" for e in lint_errs)
                )

            feedback = (
                "The previous JSON did NOT pass validation.\n"
                "Fix the JSON and output a corrected QueryPlan.\n\n"
                + "\n\n".join(feedback_parts)
            )
            messages = base_messages + [{"role": "system", "content": feedback}]

            plan_dict = self.llm.generate_json(
                json_schema=queryplan_json_schema(),
                messages=messages,
                temperature=self.temperature,
            )
            plan_dict = _auto_fix_plan(plan_dict)
            print(f"[DSL] LLM QueryPlan (retry {retries}): {plan_dict}")

            _, errs = validate_query_plan_dict(plan_dict, self.schema_path)
            lint_errs = semantic_lint(question, plan_dict)

        return plan_dict