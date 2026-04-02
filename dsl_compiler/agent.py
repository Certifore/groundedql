from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from sqlalchemy.engine import Engine

from .planner import QueryPlanPlanner
from .validation import validate_query_plan_dict
from .api.api import execute_query_plan
from .llm_adapters import make_llm_client
from .semantic_lint import semantic_lint


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

        if self.enforce_semantic_lint:
            schema_data = yaml.safe_load(
                Path(self.schema_path).read_text(encoding="utf-8")
            ) or {}
            plan_body = {k: v for k, v in plan_dict.items() if k != "meta"}
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