"""
intent_memory.py — Few-shot memory for intent extraction consistency.

Stores successful (question, normalized_intent) pairs and retrieves
semantically similar examples to inject into the extraction prompt.
This is what makes the LLM produce consistent intents across rephrasings.

Uses OpenAI embeddings + cosine similarity for retrieval (no extra DB needed).
Falls back gracefully if embeddings are unavailable.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _get_embedding(text: str, _client: Any = None) -> Optional[List[float]]:
    """Get an embedding vector for *text* using OpenAI's API."""
    try:
        import openai
        client = _client or openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        resp = client.embeddings.create(
            input=text,
            model="text-embedding-3-small",
        )
        return resp.data[0].embedding
    except Exception as exc:
        print(f"[IntentMemory] Embedding failed: {exc}", file=sys.stderr)
        return None


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    va = np.array(a)
    vb = np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


class IntentMemory:
    """Stores and retrieves (question, intent) pairs for few-shot prompting.

    Persists to a JSON file so examples survive restarts.
    Uses embedding-based similarity for retrieval.
    """

    def __init__(
        self,
        persist_path: Optional[str] = None,
        max_examples: int = 200,
    ):
        self.persist_path = persist_path
        self.max_examples = max_examples
        self._entries: List[Dict[str, Any]] = []
        self._openai_client: Any = None

        if persist_path:
            self._load()

    def _get_client(self) -> Any:
        if self._openai_client is None:
            import openai
            self._openai_client = openai.OpenAI(
                api_key=os.getenv("OPENAI_API_KEY", ""),
            )
        return self._openai_client

    def _load(self) -> None:
        if not self.persist_path:
            return
        p = Path(self.persist_path)
        if p.exists():
            try:
                self._entries = json.loads(p.read_text()) or []
                print(
                    f"[IntentMemory] Loaded {len(self._entries)} examples from {p.name}",
                    file=sys.stderr,
                )
            except Exception as exc:
                print(f"[IntentMemory] Load failed: {exc}", file=sys.stderr)
                self._entries = []

    def _save(self) -> None:
        if not self.persist_path:
            return
        p = Path(self.persist_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self._entries, indent=2, default=str))

    def store(self, question: str, intent: Dict[str, Any]) -> None:
        """Store a successful (question, intent) pair."""
        embedding = _get_embedding(question, self._get_client())
        if embedding is None:
            return

        self._entries.append({
            "question": question,
            "intent": intent,
            "embedding": embedding,
        })

        if len(self._entries) > self.max_examples:
            self._entries = self._entries[-self.max_examples:]

        self._save()
        print(
            f"[IntentMemory] Stored example ({len(self._entries)} total)",
            file=sys.stderr,
        )

    def retrieve(
        self,
        question: str,
        top_k: int = 3,
        min_similarity: float = 0.75,
    ) -> List[Dict[str, Any]]:
        """Find the most similar past questions and return their intents.

        Returns a list of {"question": str, "intent": dict, "similarity": float}
        sorted by similarity descending.
        """
        if not self._entries:
            return []

        q_embedding = _get_embedding(question, self._get_client())
        if q_embedding is None:
            return []

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for entry in self._entries:
            emb = entry.get("embedding")
            if not emb:
                continue
            sim = _cosine_similarity(q_embedding, emb)
            if sim >= min_similarity:
                scored.append((sim, entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for sim, entry in scored[:top_k]:
            results.append({
                "question": entry["question"],
                "intent": entry["intent"],
                "similarity": round(sim, 3),
            })

        if results:
            print(
                f"[IntentMemory] Found {len(results)} similar examples "
                f"(best: {results[0]['similarity']})",
                file=sys.stderr,
            )

        return results

    def format_few_shot_examples(self, examples: List[Dict[str, Any]]) -> str:
        """Format retrieved examples as a prompt section."""
        if not examples:
            return ""

        parts = [
            "EXAMPLES — For similar questions, here are the intents that were "
            "previously extracted and verified correct. Follow the same structure:"
        ]

        for i, ex in enumerate(examples, 1):
            intent_clean = {
                k: v for k, v in ex["intent"].items()
                if v is not None and v != [] and v != ""
            }
            parts.append(
                f"\n  Example {i} (similarity: {ex['similarity']}):"
                f"\n    Question: {ex['question']}"
                f"\n    Intent: {json.dumps(intent_clean, default=str)}"
            )

        return "\n".join(parts)
