from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from sqlalchemy.engine import Engine

from .planner import QueryPlanPlanner
from .intent_planner import IntentPlanner
from .validation import validate_query_plan_dict
from .api.api import execute_query_plan
from .llm_adapters import make_llm_client
from .semantic_lint import semantic_lint
from .decompose import is_compound, split_compound, SubQuestion
from .spec_builder import build_spec, write_spec
from .value_index import build_value_index
from .intent_memory import IntentMemory


def _ensure_spec(schema_path: str, spec_path: Optional[str]) -> str:
    """Auto-generate the spec file from schema.yaml if not provided or missing."""
    if spec_path:
        p = Path(spec_path)
        if p.exists():
            return spec_path
        print(
            f"[IntentQL] Spec file not found at {spec_path}, generating from schema...",
            file=sys.stderr,
        )
    else:
        p = Path(schema_path).parent / "queryplan_spec_generated.yaml"
        spec_path = str(p)

    if p.exists():
        schema_mtime = Path(schema_path).stat().st_mtime
        spec_mtime = p.stat().st_mtime
        if spec_mtime >= schema_mtime:
            return spec_path
        print(
            "[IntentQL] Spec is older than schema.yaml, regenerating...",
            file=sys.stderr,
        )

    spec = build_spec(schema_path)
    write_spec(spec, spec_path)
    return spec_path


class QueryAgent:
    def __init__(
        self,
        *,
        engine: Engine,
        schema_path: str,
        spec_path: Optional[str] = None,
        llm: Any,
        max_plan_retries: int = 2,
        enforce_semantic_lint: bool = True,
        use_intent_pipeline: bool = True,
        use_guided_sql: bool = False,
    ):
        """
        Args:
            schema_path: Path to your schema.yaml — the only required config file.
            spec_path: Path to the LLM spec file. If omitted or missing, it is
                auto-generated from schema.yaml at startup.
            max_plan_retries: Extra LLM attempts after the first plan when structural
                or semantic checks fail.
            enforce_semantic_lint: If True (default), do not execute when
                :func:`semantic_lint` still reports errors after retries.
            use_intent_pipeline: If True (default), use the two-stage intent
                extraction + deterministic plan builder instead of direct LLM
                QueryPlan generation. Falls back to legacy pipeline on error.
            use_guided_sql: If True, :meth:`ask` uses LLM→Postgres SQL with
                schema-backed validation (:mod:`sql_guard`) instead of QueryPlan
                compilation. Requires ``pip install 'intentql[guided]'`` and a
                LangChain-compatible ``llm`` with ``.invoke()``.
        """
        self.engine = engine
        self.schema_path = schema_path
        self.use_guided_sql = use_guided_sql
        self._llm_raw = llm
        self.spec_path = _ensure_spec(schema_path, spec_path)
        self.max_plan_retries = max_plan_retries
        self.enforce_semantic_lint = enforce_semantic_lint
        self.use_intent_pipeline = use_intent_pipeline
        llm_client = make_llm_client(llm)
        self.planner = QueryPlanPlanner(
            llm=llm_client,
            schema_path=schema_path,
            spec_path=self.spec_path,
        )

        self.value_index = {}
        try:
            self.value_index = build_value_index(engine, schema_path)
            print(
                f"[IntentQL] Value index built: "
                f"{sum(len(cols) for cols in self.value_index.values())} columns indexed",
                file=sys.stderr,
            )
        except Exception as exc:
            print(f"[IntentQL] Value index build failed ({exc}), continuing without it.", file=sys.stderr)

        memory_dir = str(Path(schema_path).parent / ".intent_memory")
        self.intent_memory = IntentMemory(persist_directory=memory_dir)

        self.intent_planner = IntentPlanner(
            llm=llm_client,
            schema_path=schema_path,
            value_index=self.value_index or None,
            memory=self.intent_memory,
        )

    def ask(self, question: str) -> Dict[str, Any]:
        if self.use_guided_sql:
            from .guided_sql import run_guided_sql

            return run_guided_sql(
                engine=self.engine,
                schema_path=self.schema_path,
                llm=self._llm_raw,
                question=question,
                intent_memory=self.intent_memory,
            )
        if self.use_intent_pipeline:
            return self._ask_intent(question)
        return self._ask_legacy(question)

    def _ask_intent(self, question: str) -> Dict[str, Any]:
        """Two-stage pipeline: intent extraction → deterministic plan builder."""
        try:
            plan_dict = self.intent_planner.plan(question)
        except Exception as exc:
            print(
                f"[IntentQL] Intent pipeline failed ({exc}), falling back to legacy.",
                file=sys.stderr,
            )
            return self._ask_legacy(question)

        parsed, errors = validate_query_plan_dict(plan_dict, self.schema_path)
        if errors:
            print(
                f"[IntentQL] Intent plan failed validation, falling back to legacy.",
                file=sys.stderr,
            )
            return self._ask_legacy(question)

        return execute_query_plan(
            engine=self.engine,
            schema_path=self.schema_path,
            query_plan=plan_dict,
        )

    def _ask_legacy(self, question: str) -> Dict[str, Any]:
        """Original full-plan LLM generation with retries + autofix."""
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
            f"[IntentQL] Compound question detected — splitting into {len(subs)} sub-questions.",
            file=sys.stderr,
        )

        parts: List[Dict[str, Any]] = []
        has_success = False

        for sq in subs:
            print(f"[IntentQL]   {sq.role}: {sq.text!r}", file=sys.stderr)
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
                print(f"[IntentQL]   {sq.role} failed: {exc}", file=sys.stderr)
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