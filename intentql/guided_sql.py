"""
Guided SQL — LLM emits Postgres using schema.yaml as context, then :mod:`sql_guard` validates before execution.

Used by :class:`intentql.agent.QueryAgent`.

Requires: ``pip install 'intentql[guided]'`` (brings in sqlglot).

Environment:
  INTENTQL_GUIDED_MAX_ROWS — append LIMIT if missing (default 5000)
  INTENTQL_GUIDED_REPAIR_ATTEMPTS — extra LLM turns after validation/DB errors (default 1)
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

from sqlalchemy.engine import Engine

from .exceptions import DatabaseExecutionError
from .executor import Executor
from .schema_catalog import SchemaCatalog, load_schema_catalog
from .sql_canonicalize import canonicalize_sql
from .sql_guard import apply_row_limit, validate_sql

_SQL_FENCE = re.compile(r"```sql\s*([\s\S]*?)```", re.IGNORECASE)
_READ_START = re.compile(r"(?is)^\s*(WITH|SELECT)\b")

_GUIDED_SYSTEM = """You are a careful PostgreSQL analyst.

Rules:
- Produce exactly ONE read-only query: SELECT or WITH … SELECT. No INSERT/UPDATE/DELETE/DDL.
- Use ONLY tables and columns from the SCHEMA block.
- Quote mixed-case identifiers when required by Postgres.
- Output ONLY a Markdown fenced block: ```sql ... ``` — no prose outside the fence."""


def _extract_sql(response: str) -> Optional[str]:
    raw = (response or "").strip()
    m = _SQL_FENCE.search(raw)
    if m:
        return m.group(1).strip().rstrip(";")
    if _READ_START.search(raw):
        return raw.rstrip().rstrip(";")
    return None


def _invoke_text(llm: Any, system: str, user: str) -> str:
    """Invoke a chat model for free text (LangChain-style .invoke preferred)."""
    if hasattr(llm, "invoke"):
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            res = llm.invoke(
                [SystemMessage(content=system), HumanMessage(content=user)]
            )
            return (getattr(res, "content", None) or str(res)).strip()
        except ImportError:
            pass
        except Exception:
            raise
    raise ValueError(
        "Guided SQL needs a LangChain-compatible chat model with .invoke() "
        "(e.g. langchain_openai.ChatOpenAI)."
    )


def _intent_memory_block(memory: Any, question: str) -> str:
    """Optional few-shot from IntentMemory (intent JSON shapes, not SQL)."""
    if memory is None:
        return ""
    try:
        examples = memory.retrieve(question, top_k=2)
    except Exception:
        return ""
    if not examples:
        return ""
    lines: List[str] = []
    for ex in examples:
        q = ex.get("question", "")
        intent = ex.get("intent")
        sim = ex.get("similarity", 0.0)
        blob = json.dumps(intent, default=str)[:1200] if intent else ""
        lines.append(f"- (sim={sim:.2f}) Q: {q}\n  Past intent: {blob}")
    return "### Similar past questions (intent shapes — adapt to SQL as needed)\n" + "\n".join(lines)


def run_guided_sql(
    *,
    engine: Engine,
    schema_path: str,
    llm: Any,
    question: str,
    intent_memory: Any = None,
    statement_timeout_ms: int = 30_000,
) -> Dict[str, Any]:
    """
    Generate SQL with the LLM, validate against schema, execute with :class:`Executor`.

    Returns the same shape as :func:`intentql.api.api.execute_query_plan` on success:
    ``rows``, ``row_count``, ``columns``, ``sql``, ``params`` (empty dict).
    """
    q = (question or "").strip()
    if not q:
        return {"error": {"message": "Empty question"}, "rows": [], "row_count": 0}

    try:
        catalog: SchemaCatalog = load_schema_catalog(schema_path)
    except Exception as e:
        return {"error": {"message": f"Failed to load schema: {e}"}}

    max_rows = int(os.environ.get("INTENTQL_GUIDED_MAX_ROWS", "5000"))
    repair_attempts = int(os.environ.get("INTENTQL_GUIDED_REPAIR_ATTEMPTS", "1"))

    mem_block = _intent_memory_block(intent_memory, q)
    user_body = f"### USER QUESTION\n{q}\n\n### SCHEMA\n{catalog.schema_prompt_block}\n"
    if mem_block:
        user_body += f"\n{mem_block}\n"

    messages_tail: List[str] = []
    last_err: Optional[str] = None
    attempts = max(1, repair_attempts + 1)

    for attempt in range(attempts):
        user = user_body
        if messages_tail:
            user = user_body + "\n\n" + "\n".join(messages_tail)

        try:
            content = _invoke_text(llm, _GUIDED_SYSTEM, user)
        except Exception as e:
            return {"error": {"message": f"LLM error: {e}"}}

        sql = _extract_sql(content)
        if not sql:
            last_err = "Model did not return a ```sql``` fenced SELECT."
            messages_tail.append(f"Fix: {last_err}")
            continue

        sql = canonicalize_sql(sql, catalog)

        vr = validate_sql(sql, catalog)
        if not vr.ok:
            last_err = vr.message
            messages_tail.append(f"Validation error: {last_err}")
            continue

        limited = apply_row_limit(sql, max_rows)
        executor = Executor(engine, statement_timeout_ms=statement_timeout_ms)
        try:
            result = executor.execute(limited, {})
        except DatabaseExecutionError as e:
            last_err = str(e)
            messages_tail.append(last_err)
            continue

        result["sql"] = limited
        result["params"] = {}
        print(
            f"[IntentQL] Guided SQL executed ({result.get('row_count', 0)} rows).",
            file=sys.stderr,
        )
        return result

    return {
        "error": {"message": last_err or "Guided SQL failed after retries"},
        "rows": [],
        "row_count": 0,
    }
