from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .llm_client import LLMClient, CallableLLMClient


class OpenAIResponsesJSONAdapter:
    """
    OpenAI SDK adapter (Responses API).
    Requirements:
      - openai package installed
      - user passes an OpenAI client instance: openai.OpenAI()
    """

    def __init__(self, client: Any, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model

    def generate_json(
        self,
        *,
        json_schema: Dict[str, Any],
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        resp = self.client.responses.create(
            model=self.model,
            input=messages,
            temperature=temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "QueryPlan",
                    "schema": json_schema,
                    "strict": True,
                },
            },
        )

        text = getattr(resp, "output_text", None)
        if not text:
            try:
                out = resp.output  # type: ignore[attr-defined]
                chunks = []
                for item in out:
                    for c in getattr(item, "content", []) or []:
                        if getattr(c, "type", None) in ("output_text", "text"):
                            chunks.append(getattr(c, "text", ""))
                text = "".join(chunks).strip()
            except Exception:
                text = None

        if not text:
            raise ValueError("OpenAI response missing JSON text output.")

        try:
            return json.loads(text)
        except Exception as e:
            raise ValueError(f"OpenAI returned non-JSON text: {e}\nRaw: {text[:500]}") from e


class LangChainJSONAdapter:
    """
    LangChain adapter.
    Works with ChatOpenAI or any LC chat model that supports .invoke(messages).
    Does not rely on tool calling — parses JSON from the model's text output directly.
    """

    def __init__(self, llm: Any):
        self.llm = llm

    def generate_json(
        self,
        *,
        json_schema: Dict[str, Any],
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        if hasattr(self.llm, "temperature"):
            try:
                self.llm.temperature = temperature
            except Exception:
                pass

        schema_hint = (
            "Return ONLY valid JSON with no markdown, no prose, no code fences.\n"
            "Your JSON MUST conform to this JSON Schema (strict):\n"
            f"{json.dumps(json_schema)}\n"
        )

        try:
            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

            lc_messages = [SystemMessage(content=schema_hint)]
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                if role == "system":
                    lc_messages.append(SystemMessage(content=content))
                elif role == "assistant":
                    lc_messages.append(AIMessage(content=content))
                else:
                    lc_messages.append(HumanMessage(content=content))
        except ImportError:
            # Fallback: pass dicts directly
            lc_messages = [{"role": "system", "content": schema_hint}] + list(messages)

        res = self.llm.invoke(lc_messages)

        text = getattr(res, "content", None)
        if text is None:
            text = str(res)

        text = text.strip()

        # Strip markdown code fences if model adds them despite instructions
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text).rstrip("`").strip()

        try:
            return json.loads(text)
        except Exception as e:
            raise ValueError(f"LangChain model returned non-JSON text: {e}\nRaw: {text[:500]}") from e


def make_llm_client(obj: Any, *, model: Optional[str] = None) -> LLMClient:
    """
    Factory that returns an LLMClient adapter based on what the user passes.

    Accepted inputs:
      A) already an LLMClient (has generate_json) -> returned as-is
      B) OpenAI client (duck-typed: has .responses.create) -> OpenAI adapter
      C) LangChain chat model (duck-typed: has .invoke) -> LangChain adapter
      D) plain callable(json_schema, messages, temperature) -> CallableLLMClient
    """
    if obj is None:
        raise ValueError("make_llm_client(obj): obj cannot be None")

    # A) already implements generate_json
    if hasattr(obj, "generate_json") and callable(getattr(obj, "generate_json")):
        return obj  # type: ignore[return-value]

    # B) OpenAI client (check before callable since openai client is also callable-ish)
    if hasattr(obj, "responses") and hasattr(obj.responses, "create"):
        return OpenAIResponsesJSONAdapter(obj, model=model or "gpt-4o-mini")

    # C) LangChain-like model — MUST come before callable() check because
    #    LangChain chat models implement __call__ (deprecated) AND .invoke().
    #    Calling them as raw callables triggers the deprecated __call__ path
    #    which expects BaseMessage objects, not plain dicts/strings.
    if hasattr(obj, "invoke") and callable(getattr(obj, "invoke")):
        return LangChainJSONAdapter(obj)

    # D) plain callable (last resort)
    if callable(obj):
        return CallableLLMClient(obj)

    raise ValueError(
        "Unsupported LLM object. Pass one of:\n"
        "- an object implementing generate_json(...)\n"
        "- an OpenAI client (openai.OpenAI())\n"
        "- a LangChain chat model (has .invoke)\n"
        "- a callable(json_schema, messages, temperature)->dict\n"
    )