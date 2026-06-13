from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from .llm_client import LLMClient, CallableLLMClient


def env_value(*names: str) -> str:
    """Read an environment value, falling back to a simple cwd/.env lookup."""
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value

    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return ""

    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() in names:
                return value.strip().strip("\"'")
    except OSError:
        return ""

    return ""


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text).rstrip("`").strip()
    return text


def _parse_json_object(text: str, *, provider: str) -> Dict[str, Any]:
    text = _strip_json_fences(text)
    try:
        value = json.loads(text)
    except Exception as primary_error:
        decoder = json.JSONDecoder()
        candidates: List[Any] = []
        for idx, char in enumerate(text):
            if char not in "{[":
                continue
            try:
                candidate, _ = decoder.raw_decode(text[idx:])
            except json.JSONDecodeError:
                continue
            candidates.append(candidate)
        if not candidates:
            raise ValueError(f"{provider} returned non-JSON text: {primary_error}\nRaw: {text[:500]}") from primary_error
        value = next(
            (
                c
                for c in candidates
                if isinstance(c, dict) and ("dataset" in c or "aggregation" in c)
            ),
            candidates[0],
        )

    if isinstance(value, str):
        return _parse_json_object(value, provider=provider)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        return value[0]
    if not isinstance(value, dict):
        raise ValueError(f"{provider} returned JSON {type(value).__name__}, expected object.\nRaw: {text[:500]}")
    return value


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

        return _parse_json_object(text, provider="OpenAI")


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

        return _parse_json_object(text, provider="LangChain model")


class MistralChatJSONAdapter:
    """
    Minimal Mistral API adapter using the chat completions endpoint.

    Reads the API key from MISTRAL_AI or MISTRAL_API_KEY when api_key is not
    provided. Mistral JSON mode guarantees syntactically valid JSON; the
    QueryPlan/QueryIntent schema is still included in the prompt so the model
    knows the expected shape.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: Optional[int] = None,
        retry_base_seconds: Optional[float] = None,
    ):
        self.api_key = (
            api_key
            or env_value("MISTRAL_AI", "MISTRAL_API_KEY")
        )
        if not self.api_key:
            raise ValueError("Set MISTRAL_AI or MISTRAL_API_KEY to use Mistral.")

        self.model = (
            model
            or env_value("MISTRAL_MODEL")
            or "mistral-small-latest"
        )
        self.base_url = (
            base_url
            or env_value("MISTRAL_BASE_URL")
            or "https://api.mistral.ai/v1"
        )
        self.timeout = timeout
        self.max_retries = (
            max_retries
            if max_retries is not None
            else int(env_value("MISTRAL_MAX_RETRIES") or "3")
        )
        self.retry_base_seconds = (
            retry_base_seconds
            if retry_base_seconds is not None
            else float(env_value("MISTRAL_RETRY_BASE_SECONDS") or "5")
        )

    def generate_json(
        self,
        *,
        json_schema: Dict[str, Any],
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        schema_hint = (
            "Return ONLY valid JSON with no markdown, no prose, no code fences.\n"
            "Your JSON MUST conform to this JSON Schema:\n"
            f"{json.dumps(json_schema)}\n"
        )
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": schema_hint}] + list(messages),
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                last_error = e
                if e.code != 429 or attempt >= self.max_retries:
                    raise ValueError(f"Mistral chat completion failed: {e}") from e
                retry_after = e.headers.get("Retry-After") if e.headers else None
                try:
                    delay = float(retry_after) if retry_after else self.retry_base_seconds * (2 ** attempt)
                except ValueError:
                    delay = self.retry_base_seconds * (2 ** attempt)
                time.sleep(delay)
            except Exception as e:
                last_error = e
                raise ValueError(f"Mistral chat completion failed: {e}") from e
        else:
            raise ValueError(f"Mistral chat completion failed: {last_error}") from last_error

        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as e:
            raise ValueError(f"Mistral response missing message content: {data}") from e

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item))
            content = "".join(parts)

        return _parse_json_object(str(content), provider="Mistral")


class OllamaChatJSONAdapter:
    """
    Minimal Ollama adapter for local models.

    Defaults to the local Gemma model imported as groundedql-gemma4. Override with
    OLLAMA_MODEL or pass --llm ollama:<model-name> in the benchmark runner.
    """

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        num_ctx: Optional[int] = None,
    ):
        self.model = model or env_value("OLLAMA_MODEL") or "groundedql-gemma4"
        self.base_url = (
            base_url
            or env_value("OLLAMA_BASE_URL")
            or "http://127.0.0.1:11434"
        ).rstrip("/")
        self.timeout = (
            timeout
            if timeout is not None
            else int(env_value("OLLAMA_TIMEOUT") or "240")
        )
        self.num_ctx = (
            num_ctx
            if num_ctx is not None
            else int(env_value("OLLAMA_NUM_CTX") or "8192")
        )

    def generate_json(
        self,
        *,
        json_schema: Dict[str, Any],
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        schema_hint = (
            "Return ONLY valid JSON with no markdown, no prose, no code fences.\n"
            "Your JSON MUST conform to this JSON Schema:\n"
            f"{json.dumps(json_schema)}\n"
        )
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": schema_hint}] + list(messages),
            "stream": False,
            "format": "json",
            "options": {
                "temperature": temperature,
                "num_ctx": self.num_ctx,
            },
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            raise ValueError(f"Ollama chat completion failed for model '{self.model}': {e}") from e

        try:
            content = data["message"]["content"]
        except Exception as e:
            raise ValueError(f"Ollama response missing message content: {data}") from e

        return _parse_json_object(str(content), provider="Ollama")


def make_llm_client(obj: Any, *, model: Optional[str] = None) -> LLMClient:
    """
    Factory that returns an LLMClient adapter based on what the user passes.

    Accepted inputs:
      A) already an LLMClient (has generate_json) -> returned as-is
      B) OpenAI client (duck-typed: has .responses.create) -> OpenAI adapter
      C) LangChain chat model (duck-typed: has .invoke) -> LangChain adapter
      D) provider string: "mistral", "ollama", "gemma", or "local"
      E) plain callable(json_schema, messages, temperature) -> CallableLLMClient
    """
    if obj is None:
        raise ValueError("make_llm_client(obj): obj cannot be None")

    if isinstance(obj, str):
        name, _, inline_model = obj.partition(":")
        provider = name.strip().lower()
        resolved_model = model or inline_model.strip() or None
        if provider in {"mistral", "mistralai", "mistral_ai"}:
            return MistralChatJSONAdapter(model=resolved_model)
        if provider in {"ollama", "gemma", "gemma4", "local"}:
            return OllamaChatJSONAdapter(model=resolved_model)

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
        "- a provider string: 'mistral', 'ollama', 'gemma', or 'local'\n"
        "- a callable(json_schema, messages, temperature)->dict\n"
    )
