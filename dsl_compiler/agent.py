from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from sqlalchemy.engine import Engine

from .planner import QueryPlanPlanner
from .validation import validate_query_plan_dict
from .api.api import execute_query_plan
from .llm_adapters import make_llm_client
from .semantic_lint import semantic_lint
from .decompose import is_compound, split_compound, SubQuestion
from .plan_autofix import autofix_plan


class QueryAgent:
    def __init__(
        self,
        *,
        engine: Engine,
        schema_path: str,
        spec_path: str,
        llm: Any,
        max_plan_retries: int = 2,
        enforce_semantic_lint: bool = True,
    ):
        """
        Args:
            max_plan_retries: Extra LLM attempts after the first plan when structural
                or semantic checks fail. Default 2 so multi-step fixes (compound CTE + filters)
                can succeed after feedback.
            enforce_semantic_lint: If True (default), do not execute when
                :func:`semantic_lint` still reports errors after retries. Set False only
                for debugging or custom callers that handle ``meta`` themselves.
        """
        self.engine = engine
        self.schema_path = schema_path
        self.spec_path = spec_path
        self.max_plan_retries = max_plan_retries
        self.enforce_semantic_lint = enforce_semantic_lint
        llm_client = make_llm_client(llm)  # auto-picks OpenAI/LangChain/callable adapter
        self.planner = QueryPlanPlanner(
            llm=llm_client,
            schema_path=schema_path,
            spec_path=spec_path,
        )

    def ask(self, question: str) -> Dict[str, Any]:
        plan_dict = self.planner.plan_with_retry(question, max_retries=self.max_plan_retries)

        parsed, errors = validate_query_plan_dict(plan_dict, self.schema_path)
        if errors:
            return {
                "error": {
                    "message": "QueryPlan failed validation after retries.",
                    "validation_errors": [{"path": e.path, "message": e.message} for e in errors],
                    "plan": plan_dict,
                }
            }

        schema_data = yaml.safe_load(
            Path(self.schema_path).read_text(encoding="utf-8")
        ) or {}

        plan_body = {k: v for k, v in plan_dict.items() if k != "meta"}
        autofix_plan(plan_body, schema_data)
        for k, v in plan_body.items():
            plan_dict[k] = v

        if self.enforce_semantic_lint:
            lint_errs = semantic_lint(question, plan_body, schema_data)
            if lint_errs:
                return {
                    "error": {
                        "message": "QueryPlan failed semantic lint after retries — plan does not match the question.",
                        "lint_errors": lint_errs,
                        "plan": plan_dict,
                    }
                }

        return execute_query_plan(
            engine=self.engine,
            schema_path=self.schema_path,
            query_plan=plan_dict,
        )

    def ask_compound(self, question: str) -> Dict[str, Any]:
        """
        Smart entry point that handles compound questions automatically.

        If the question asks for multiple deliverables (e.g. a count AND a
        ranked list with detail columns), it splits the question into focused
        sub-questions, runs each through :meth:`ask`, and merges the results.

        For simple questions it delegates to :meth:`ask` directly.

        Returns a dict with:
            - ``"compound": False, ...`` for simple questions (same as ``ask()``)
            - ``"compound": True, "parts": [...]`` for compound questions, where
              each part is ``{"role": str, "question": str, "result": dict}``
        """
        if not is_compound(question):
            result = self.ask(question)
            result["compound"] = False
            return result

        subs = split_compound(question)
        print(
            f"[QCE] Compound question detected — splitting into {len(subs)} sub-questions.",
            file=sys.stderr,
        )

        parts: List[Dict[str, Any]] = []
        has_success = False

        for sq in subs:
            print(f"[QCE]   {sq.role}: {sq.text!r}", file=sys.stderr)
            try:
                r = self.ask(sq.text)
                is_error = isinstance(r, dict) and bool(r.get("error"))
                if not is_error:
                    has_success = True
                parts.append({
                    "role": sq.role,
                    "question": sq.text,
                    "result": r,
                })
            except Exception as exc:
                print(f"[QCE]   {sq.role} failed: {exc}", file=sys.stderr)
                parts.append({
                    "role": sq.role,
                    "question": sq.text,
                    "result": {"error": {"message": str(exc)}},
                })

        if not has_success:
            first_err = next(
                (p["result"] for p in parts if p["result"].get("error")),
                {"error": {"message": "All sub-questions failed."}},
            )
            return first_err

        return {"compound": True, "parts": parts}