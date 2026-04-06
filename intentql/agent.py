"""QueryAgent — guided LLM → Postgres SQL with schema-backed validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy.engine import Engine

from .guided_sql import run_guided_sql
from .intent_memory import IntentMemory


class QueryAgent:
    def __init__(
        self,
        *,
        engine: Engine,
        schema_path: str,
        llm: Any,
        memory_persist_directory: Optional[str] = None,
    ):
        """
        Args:
            engine: SQLAlchemy engine (Postgres).
            schema_path: Path to schema.yaml.
            llm: LangChain-compatible chat model with ``.invoke()`` (e.g. ChatOpenAI).
            memory_persist_directory: Optional ChromaDB path for :class:`IntentMemory`
                (default: ``<schema_dir>/.intent_memory``).
        """
        self.engine = engine
        self.schema_path = schema_path
        self._llm_raw = llm
        mem_dir = memory_persist_directory or str(Path(schema_path).parent / ".intent_memory")
        self.intent_memory = IntentMemory(persist_directory=mem_dir)

    def ask(self, question: str) -> Dict[str, Any]:
        return run_guided_sql(
            engine=self.engine,
            schema_path=self.schema_path,
            llm=self._llm_raw,
            question=question,
            intent_memory=self.intent_memory,
        )

    def ask_compound(self, question: str) -> Dict[str, Any]:
        """Single-path execution; compound splitting is not used."""
        result = self.ask(question)
        result["compound"] = False
        return result
