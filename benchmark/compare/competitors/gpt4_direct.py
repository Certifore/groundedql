"""
Direct GPT-4 SQL generation baseline.
Given a natural language question and schema description, asks GPT-4 to write SQL directly.
No compiler, no allowlist, no validation layer.
"""
from __future__ import annotations

import time
from typing import Any, Dict

from openai import OpenAI

# Same schema information QCE uses — loaded dynamically so it stays in sync
def _build_system_prompt(schema: dict) -> str:
    lines = [
        "You are a SQL expert for a PostgreSQL database.",
        "Your job is to write a single SELECT query to answer the user's question.",
        "",
        "STRICT RULES:",
        "1. You may ONLY use the tables and columns listed below.",
        "2. If the question references a table or column not listed below, you MUST respond",
        "   with exactly: ERROR: unknown table or column referenced",
        "3. Do NOT guess or invent column names.",
        "4. Output ONLY the raw SQL query or the ERROR message.",
        "5. Do NOT wrap output in markdown, code blocks, or backticks.",
        "6. Always use the exact physical table name shown below, including double quotes",
        "   if the name contains uppercase letters. Example: SELECT * FROM \"MyTable\"",
        "",
        "AVAILABLE SCHEMA:",
    ]
    for table in schema.get("tables", []):
        lines.append(f"\nTable: {table['name']} (physical name to use in SQL: {table['db_table']})")
        if table.get("description"):
            lines.append(f"  Description: {table['description']}")
        lines.append("  Columns (use the physical db_column name in SQL):")
        for col in table.get("columns", []):
            desc = f" — {col['description']}" if col.get("description") else ""
            lines.append(f"    - logical: {col['name']} | physical: {col['db_column']} ({col.get('type','')}{desc})")
    return "\n".join(lines)


def make_client(openai_api_key: str) -> OpenAI:
    return OpenAI(api_key=openai_api_key)


def ask(client: OpenAI, question: str, schema: dict, model: str = "gpt-4o") -> Dict[str, Any]:
    t0 = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": _build_system_prompt(schema)},
                {"role": "user", "content": question},
            ],
        )
        sql = response.choices[0].message.content.strip()
        latency_ms = (time.perf_counter() - t0) * 1000
        usage = response.usage
        return {
            "sql": sql,
            "error": False,
            "latency_ms": round(latency_ms, 2),
            "tokens": {
                "prompt": usage.prompt_tokens,
                "completion": usage.completion_tokens,
                "total": usage.total_tokens,
            },
        }
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        return {"sql": None, "error": True, "message": str(e), "latency_ms": round(latency_ms, 2), "tokens": {}}


def execute_sql(sql: str, db_url: str) -> Dict[str, Any]:
    """Execute raw SQL against the DB and return rows + row_count."""
    import psycopg2
    from urllib.parse import urlparse
    t0 = time.perf_counter()
    try:
        p = urlparse(db_url.replace("postgresql+psycopg2://", "postgresql://"))
        conn = psycopg2.connect(
            host=p.hostname, port=p.port or 5432,
            dbname=p.path.lstrip("/"),
            user=p.username, password=p.password,
            sslmode="require",
            options="-c statement_timeout=30000",
        )
        cur = conn.cursor()
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        cur.close()
        conn.close()
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "rows": rows,
            "row_count": len(rows),
            "columns": cols,
            "error": False,
            "latency_ms": round(latency_ms, 2),
        }
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        return {"rows": [], "row_count": 0, "error": True, "message": str(e), "latency_ms": round(latency_ms, 2)}


def ask_and_execute(client: OpenAI, question: str, schema: dict, db_url: str, model: str = "gpt-4o") -> Dict[str, Any]:
    """Full pipeline: question → SQL → execute → result."""
    gen = ask(client, question, schema, model)
    total_latency = gen.get("latency_ms", 0)
    if gen["error"] or not gen.get("sql") or gen["sql"].lower().startswith("error"):
        return {
            "sql": gen.get("sql"),
            "rows": [], "row_count": 0,
            "error": True,
            "message": gen.get("sql") or gen.get("message", "generation failed"),
            "latency_ms": total_latency,
            "tokens": gen.get("tokens", {}),
        }
    exec_result = execute_sql(gen["sql"], db_url)
    exec_result["sql"] = gen["sql"]
    exec_result["latency_ms"] = round(total_latency + exec_result.get("latency_ms", 0), 2)
    exec_result["tokens"] = gen.get("tokens", {})
    return exec_result
