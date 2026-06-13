from __future__ import annotations

from typing import Any, Dict, List, Protocol, Callable, Optional


class LLMClient(Protocol):
    """
    Minimal contract:
    - You give: json_schema + messages
    - You get: a Python dict that matches the schema
    """

    def generate_json(
        self,
        *,
        json_schema: Dict[str, Any],
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        ...


GenerateJsonFn = Callable[[Dict[str, Any], List[Dict[str, str]], float], Dict[str, Any]]


class CallableLLMClient:
    """
    Adapter for ANY user stack.
    Users pass a function that returns a dict.
    """

    def __init__(self, fn: GenerateJsonFn):
        self._fn = fn

    def generate_json(
        self,
        *,
        json_schema: Dict[str, Any],
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        return self._fn(json_schema, messages, temperature)