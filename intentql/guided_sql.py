"""
Guided SQL — LLM emits Postgres using schema.yaml as context, then :mod:`sql_guard` validates before execution.

Used by :class:`intentql.agent.QueryAgent`.

Requires: ``pip install 'intentql[guided]'`` (brings in sqlglot).

Environment:
  INTENTQL_GUIDED_MAX_ROWS — append LIMIT if missing (default 5000)
  INTENTQL_GUIDED_REPAIR_ATTEMPTS — extra LLM turns after validation/DB errors (default 1)
  INTENTQL_DISABLE_VALUE_INDEX — if 1/true, skip loading DISTINCT value lists (default off)
  INTENTQL_VALUE_INDEX_HARD_CAP — max DISTINCT rows per column from DB (default 800)
  INTENTQL_GUIDED_MEMORY_MIN_SIMILARITY — retrieval threshold for prior guided SQL (default 0.62)

``value_index`` in schema.yaml:

- Omit or set ``false`` / ``none`` — no DISTINCT pick-list (smallest prompts).
- ``auto`` — heuristic string columns + DB DISTINCT (cached; see :mod:`intentql.value_index`).
- Explicit table → column list — full control.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from sqlalchemy import text as sqla_text
from sqlalchemy.engine import Engine

from .exceptions import DatabaseExecutionError
from .executor import Executor
from .schema_catalog import SchemaCatalog, load_schema_catalog
from .sql_canonicalize import canonicalize_sql
from .sql_guard import apply_row_limit, validate_sql
from .value_index import (
    format_auto_value_index_for_guided_prompt,
    get_cached_value_index,
)

_SQL_FENCE = re.compile(r"```sql\s*([\s\S]*?)```", re.IGNORECASE)
_READ_START = re.compile(r"(?is)^\s*(WITH|SELECT)\b")

_GUIDED_SYSTEM = """You are a careful PostgreSQL analyst.

Rules:
- Produce exactly ONE read-only query: SELECT or WITH … SELECT. No INSERT/UPDATE/DELETE/DDL.
- Use ONLY tables and columns from the SCHEMA block.
- If a VALUE INDEX section lists allowed values for a column, filter using those exact strings only; map user typos or informal names to the closest listed value.
- For relative time phrases ("last year", "this month", "YTD", etc.), use the CURRENT DATE (UTC) block in the user message — do not guess years.
- Quote mixed-case identifiers when required by Postgres.
- Output ONLY a Markdown fenced block: ```sql ... ``` — no prose outside the fence."""


def _current_date_context_block() -> str:
    """Anchor relative dates so the model does not invent calendar years."""
    now = datetime.now(timezone.utc)
    y = now.year
    d = now.date().isoformat()
    prev = y - 1
    return (
        "### CURRENT DATE (UTC)\n"
        f"- Today: {d}\n"
        f"- Current calendar year: {y}\n"
        f"- \"Last calendar year\" / \"last year\" (when meaning the prior full year): "
        f"{prev}-01-01 through {prev}-12-31 inclusive on the relevant date column.\n"
        f"- \"This year\" (year-to-date): {y}-01-01 through {d} (or through end of query range as appropriate).\n\n"
    )


def _load_schema_doc(schema_path: str) -> Dict[str, Any]:
    with open(Path(schema_path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _distinct_column_values(
    engine: Engine, db_table_sql: str, db_column_sql: str, limit: int
) -> List[str]:
    """Fetch DISTINCT non-null values. Identifiers come from schema YAML only (trusted)."""
    if limit <= 0:
        return []
    q = (
        f"SELECT DISTINCT {db_column_sql} AS _v FROM {db_table_sql} "
        f"WHERE {db_column_sql} IS NOT NULL ORDER BY 1 LIMIT :lim"
    )
    try:
        with engine.connect() as conn:
            rows = conn.execute(sqla_text(q), {"lim": limit}).fetchall()
        out: List[str] = []
        for r in rows:
            if r[0] is None:
                continue
            s = str(r[0]).strip()
            if s:
                out.append(s)
        return out
    except Exception:
        return []


def _value_index_is_auto_mode(raw: Any) -> bool:
    if isinstance(raw, str) and raw.strip().lower() == "auto":
        return True
    if isinstance(raw, dict):
        mode = raw.get("mode")
        if isinstance(mode, str) and mode.strip().lower() == "auto":
            if len(raw) == 1:
                return True
            raise ValueError(
                "value_index: `mode: auto` must be the only key, or use explicit table entries."
            )
    return False


def _value_index_block(engine: Engine, schema_path: str) -> str:
    """
    Schema top-level ``value_index``:

    - **Omit** or ``false`` / ``none`` — no pick-list.
    - **``auto``** — :func:`intentql.value_index.build_value_index` heuristics + cache.
    - **Explicit** dict — per-table column lists (list or legacy int limits).

    Row caps use ``INTENTQL_VALUE_INDEX_HARD_CAP`` (and env).
    """
    flag = os.environ.get("INTENTQL_DISABLE_VALUE_INDEX", "").strip().lower()
    if flag in ("1", "true", "yes", "on"):
        return ""
    try:
        hard_cap = int(os.environ.get("INTENTQL_VALUE_INDEX_HARD_CAP", "800"))
    except ValueError:
        hard_cap = 800
    hard_cap = max(1, min(hard_cap, 50_000))

    doc = _load_schema_doc(schema_path)
    raw = doc.get("value_index")

    if raw is False:
        return ""
    if isinstance(raw, str) and raw.strip().lower() in ("none", "off", "false"):
        return ""
    if raw is None:
        return ""

    try:
        if _value_index_is_auto_mode(raw):
            idx = get_cached_value_index(engine, schema_path, max_distinct=hard_cap)
            return format_auto_value_index_for_guided_prompt(
                idx,
                max_values_per_column=min(80, hard_cap),
            )
    except ValueError as e:
        print(f"[IntentQL] value_index auto: {e}", file=sys.stderr)
        return ""
    except Exception as e:
        print(f"[IntentQL] value_index auto failed: {e}", file=sys.stderr)
        return ""

    if not isinstance(raw, dict) or not raw:
        return ""

    tables_by_name = {
        str(t.get("name", "")).strip(): t
        for t in doc.get("tables", []) or []
        if isinstance(t, dict) and str(t.get("name", "")).strip()
    }

    sections: List[str] = [
        "### VALUE INDEX (distinct values from the database)",
        "Use these exact strings in SQL filters when the question names entities that appear below. "
        "If the user's spelling differs slightly, pick the closest matching line.",
        "",
    ]
    any_values = False

    for ltable, cmap in raw.items():
        ltable = str(ltable).strip()
        t = tables_by_name.get(ltable)
        if not t:
            continue
        # value_index[table] may be a list of logical columns (limits = INTENTQL_VALUE_INDEX_HARD_CAP only)
        # or a mapping column_name -> positive int (legacy).
        col_by_name = {
            str(c.get("name", "")).strip(): c
            for c in t.get("columns", []) or []
            if isinstance(c, dict) and str(c.get("name", "")).strip()
        }
        entries: List[tuple[str, int]] = []
        if isinstance(cmap, list):
            for item in cmap:
                lc = str(item).strip()
                if lc:
                    entries.append((lc, hard_cap))
        elif isinstance(cmap, dict):
            for lcname, lim_raw in cmap.items():
                lcname = str(lcname).strip()
                try:
                    lim = int(lim_raw)
                except (TypeError, ValueError):
                    continue
                if lim > 0:
                    entries.append((lcname, max(1, min(lim, hard_cap))))
        else:
            continue

        db_table = str(t.get("db_table", "")).strip()
        if not db_table:
            continue

        for lcname, lim in entries:
            col = col_by_name.get(lcname)
            if not col:
                continue
            db_col = str(col.get("db_column", "")).strip()
            if not db_col:
                continue
            vals = _distinct_column_values(engine, db_table, db_col, lim)
            if not vals:
                continue
            any_values = True
            sections.append(f"**{ltable}.{lcname}** (`{db_col}`), up to {len(vals)} values:")
            sections.extend(f"  - {v}" for v in vals)
            sections.append("")

    if not any_values:
        return ""
    return "\n".join(sections).rstrip() + "\n\n"


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


def _guided_sql_memory_block(memory: Any, question: str) -> str:
    """Optional: prior (question, SQL) pairs from :meth:`IntentMemory.retrieve_guided_sql`."""
    if memory is None:
        return ""
    try:
        retrieve = getattr(memory, "retrieve_guided_sql", None)
        if not callable(retrieve):
            return ""
        examples = retrieve(question, top_k=2)
    except Exception:
        return ""
    if not examples:
        return ""
    lines: List[str] = [
        "### Guided SQL memory",
        "",
    ]
    for ex in examples:
        q = ex.get("question", "")
        sql = (ex.get("sql") or "").strip()
        sim = ex.get("similarity", 0.0)
        if not sql:
            continue
        lines.append(f"- (sim={sim:.2f}) Q: {q}")
        lines.append("```sql")
        lines.append(sql[:12_000])
        lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


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

    guided_mem = _guided_sql_memory_block(intent_memory, q)
    intent_mem = _intent_memory_block(intent_memory, q)
    vi_block = ""
    try:
        vi_block = _value_index_block(engine, schema_path)
    except Exception:
        vi_block = ""
    user_body = (
        _current_date_context_block()
        + f"### USER QUESTION\n{q}\n\n### SCHEMA\n{catalog.schema_prompt_block}\n"
    )
    if vi_block:
        user_body += "\n" + vi_block
    memory_extra = ""
    if guided_mem:
        memory_extra += guided_mem + "\n"
    if intent_mem:
        memory_extra += intent_mem + "\n"
    if memory_extra.strip():
        user_body += "\n" + memory_extra.strip() + "\n"

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
        if intent_memory is not None:
            try:
                sg = getattr(intent_memory, "store_guided_sql", None)
                if callable(sg):
                    sg(q, limited)
            except Exception:
                pass
        return result

    return {
        "error": {"message": last_err or "Guided SQL failed after retries"},
        "rows": [],
        "row_count": 0,
    }
