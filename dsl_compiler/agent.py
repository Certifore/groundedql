from __future__ import annotations

from typing import Any, Dict, Optional

from sqlalchemy.engine import Engine

from .planner import QueryPlanPlanner
from .validation import validate_query_plan_dict
from .api.api import execute_query_plan  # adjust import if your path differs
from .llm_adapters import make_llm_client


class QueryAgent:
    def __init__(
        self,
        *,
        engine: Engine,
        schema_path: str,
        spec_path: str,
        llm: Any,
        max_plan_retries: int = 1,
    ):
        self.engine = engine
        self.schema_path = schema_path
        self.spec_path = spec_path
        self.max_plan_retries = max_plan_retries
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

        # execute
        return execute_query_plan(
            engine=self.engine,
            schema_path=self.schema_path,
            query_plan=plan_dict,
        )