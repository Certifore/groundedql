"""Public exceptions for IntentQL (guided SQL)."""

from __future__ import annotations

from typing import Optional


class DSLCompilerError(Exception):
    """Base class for all IntentQL errors."""


class SchemaError(DSLCompilerError):
    """Raised when schema.yaml is missing, malformed, or invalid."""


class DatabaseExecutionError(DSLCompilerError):
    """Raised when validated SQL fails at the database."""

    def __init__(self, message: str, *, sql: Optional[str] = None, original: Optional[Exception] = None):
        super().__init__(message)
        self.sql = sql
        self.original = original
