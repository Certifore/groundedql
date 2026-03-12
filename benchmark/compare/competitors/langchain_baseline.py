"""
LangChain SQLDatabaseChain baseline competitor.
Given a natural language question, generates and executes SQL via LangChain.
Used to compare against QCE on injection resistance and determinism.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback


def make_agent(db_url: str, openai_api_key: str, model: str = "gpt-4o") -> Any:
    db = SQLDatabase.from_uri(db_url)
    llm = ChatOpenAI(model=model, temperature=0, api_key=openai_api_key)
    agent = create_sql_agent(llm=llm, db=db, verbose=False)
    return agent


def ask(agent: Any, question: str) -> Dict[str, Any]:
    """
    Ask a natural language question. Returns dict with:
    - sql: the SQL generated (if extractable)
    - result: the result or error string
    - error: True if an exception was raised
    - latency_ms: the time taken to process the request in milliseconds
    """
    t0 = time.perf_counter()
    try:
        with get_openai_callback() as cb:
            result = agent.invoke({"input": question})
            output = result.get("output", str(result))
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "sql": None,
            "result": output,
            "error": False,
            "latency_ms": round(latency_ms, 2),
            "tokens": {
                "prompt": cb.prompt_tokens,
                "completion": cb.completion_tokens,
                "total": cb.total_tokens,
                "llm_calls": cb.successful_requests,
            },
        }
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        return {"sql": None, "result": str(e), "error": True, "latency_ms": round(latency_ms, 2), "tokens": {}}
