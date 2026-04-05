"""
intent_memory.py — Few-shot memory for intent extraction consistency.

Stores successful (question, normalized_intent) pairs and retrieves
semantically similar examples to inject into the extraction prompt.
This is what makes the LLM produce consistent intents across rephrasings.

Uses ChromaDB for persistent, embedding-indexed storage with fast
similarity search.  Falls back gracefully if ChromaDB is unavailable.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


class IntentMemory:
    """Stores and retrieves (question, intent) pairs for few-shot prompting.

    Persists to a ChromaDB collection so examples survive restarts
    and similarity search is indexed (not brute-force).
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "intent_memory",
        max_examples: int = 500,
    ):
        self.max_examples = max_examples
        self._collection = None

        try:
            import chromadb
            from chromadb.config import Settings

            persist_dir = persist_directory or str(
                Path.home() / ".intentql" / "intent_memory"
            )
            Path(persist_dir).mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            count = self._collection.count()
            if count > 0:
                print(
                    f"[IntentMemory] Loaded {count} examples from ChromaDB",
                    file=sys.stderr,
                )
        except ImportError:
            print(
                "[IntentMemory] chromadb not installed, memory disabled.",
                file=sys.stderr,
            )
        except Exception as exc:
            print(
                f"[IntentMemory] ChromaDB init failed ({exc}), memory disabled.",
                file=sys.stderr,
            )

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        try:
            import openai
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
            resp = client.embeddings.create(
                input=text,
                model="text-embedding-3-small",
            )
            return resp.data[0].embedding
        except Exception as exc:
            print(f"[IntentMemory] Embedding failed: {exc}", file=sys.stderr)
            return None

    def store(self, question: str, intent: Dict[str, Any]) -> None:
        """Store a successful (question, intent) pair."""
        if self._collection is None:
            return

        embedding = self._get_embedding(question)
        if embedding is None:
            return

        import hashlib
        doc_id = hashlib.md5(question.lower().strip().encode()).hexdigest()

        existing = self._collection.get(ids=[doc_id])
        if existing and existing["ids"]:
            return

        self._collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[question],
            metadatas=[{"intent": json.dumps(intent, default=str)}],
        )

        count = self._collection.count()

        if count > self.max_examples:
            all_docs = self._collection.get(limit=count - self.max_examples)
            if all_docs["ids"]:
                self._collection.delete(ids=all_docs["ids"])

        print(
            f"[IntentMemory] Stored example ({count} total)",
            file=sys.stderr,
        )

    def retrieve(
        self,
        question: str,
        top_k: int = 3,
        min_similarity: float = 0.60,
    ) -> List[Dict[str, Any]]:
        """Find the most similar past questions and return their intents.

        Returns a list of {"question": str, "intent": dict, "similarity": float}
        sorted by similarity descending.
        """
        if self._collection is None or self._collection.count() == 0:
            return []

        embedding = self._get_embedding(question)
        if embedding is None:
            return []

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self._collection.count()),
        )

        if not results or not results["documents"] or not results["documents"][0]:
            return []

        matched: List[Dict[str, Any]] = []
        for i, doc in enumerate(results["documents"][0]):
            distance = results["distances"][0][i] if results.get("distances") else 1.0
            similarity = 1.0 - distance
            if similarity < min_similarity:
                continue

            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            try:
                intent = json.loads(meta.get("intent", "{}"))
            except (json.JSONDecodeError, TypeError):
                continue

            matched.append({
                "question": doc,
                "intent": intent,
                "similarity": round(similarity, 3),
            })

        if matched:
            print(
                f"[IntentMemory] Found {len(matched)} similar examples "
                f"(best: {matched[0]['similarity']})",
                file=sys.stderr,
            )

        return matched

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
