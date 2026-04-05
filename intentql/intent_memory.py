"""
intent_memory.py — Few-shot memory for intent extraction consistency.

Stores successful (question, normalized_intent) pairs and retrieves
semantically similar examples to inject into the extraction prompt.
This is what makes the LLM produce consistent intents across rephrasings.

Uses ChromaDB for persistent, embedding-indexed storage with fast
similarity search.  Falls back gracefully if ChromaDB is unavailable.

Embeddings are computed locally by ChromaDB's default embedding function
(all-MiniLM-L6-v2 via sentence-transformers).  No API key or external
service required.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Coarse task bucket for few-shot gating. Sentence embeddings conflate shared vocabulary
# ("work order") with the same *intent shape*; we only inject examples whose bucket
# matches the current question's bucket (see retrieve()).
_TASK_CLASS_INTENT = "task_class"


def _task_class_from_intent(intent: Dict[str, Any]) -> str:
    """Structural class stored with each example (normalized intent)."""
    agg = intent.get("aggregation") or "count"
    if agg == "ratio":
        return "ratio"
    if agg == "list":
        if intent.get("sort_column"):
            return "sorted_list"
        return "detail_list"
    if agg == "count":
        if intent.get("time_bucket"):
            return "trend"
        gb = intent.get("group_by") or []
        if isinstance(gb, str):
            gb = [gb]
        if gb:
            return "group_rank"
        return "scalar_count"
    if agg in ("sum", "avg", "min", "max"):
        return "aggregate_metric"
    return "generic"


def _task_class_from_question(question: str) -> str:
    """Same buckets as _task_class_from_intent, inferred from wording only (pre-LLM)."""
    q = question.lower().strip()
    if re.search(r"\b(most\s+recent|latest|newest)\b", q):
        return "sorted_list"
    if re.search(r"\b(what percent|what %|proportion|what share)\b", q) or (
        "%" in q and "work order" in q
    ):
        return "ratio"
    if any(
        p in q
        for p in (
            "trend",
            "over time",
            "by month",
            "by year",
            "by quarter",
            "monthly",
            "yearly",
            "quarterly",
        )
    ):
        return "trend"
    if re.search(r"\b(details?|show me|look\s*up|lookup)\b", q) and (
        re.search(r"\b(id|number)\b", q) or re.search(r"\bwo[-\s]?\d", q, re.I)
    ):
        return "detail_list"
    if re.search(
        r"\b(which|what)\s+.+\b(most|least|highest|lowest|top|fewest)\b",
        q,
        re.I,
    ) or re.search(
        r"\b(most|least|highest|lowest|fewest)\s+.+\b(per|each|every)\b",
        q,
        re.I,
    ):
        return "group_rank"
    if re.search(r"\b(how many|count|total|number of)\b", q) and not re.search(
        r"\b(per|each|every|by\s+(asset|building|location))\b",
        q,
    ):
        return "scalar_count"
    return "generic"


class IntentMemory:
    """Stores and retrieves (question, intent) pairs for few-shot prompting.

    Persists to a ChromaDB collection so examples survive restarts
    and similarity search is indexed (not brute-force).

    Embeddings are computed locally — no API key needed.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "intent_memory",
        max_examples: int = 500,
    ):
        self.max_examples = max_examples
        self._collection = None

        if os.environ.get("INTENTQL_DISABLE_MEMORY", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            print(
                "[IntentMemory] Disabled via INTENTQL_DISABLE_MEMORY.",
                file=sys.stderr,
            )
            return

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

            # Detect embedding dimension mismatch (e.g. migrating from
            # OpenAI 1536-dim to local 384-dim).  If mismatched, delete
            # and recreate so queries don't crash at runtime.
            count = self._collection.count()
            if count > 0:
                try:
                    self._collection.query(query_texts=["test"], n_results=1)
                except Exception as dim_exc:
                    if "dimension" in str(dim_exc).lower():
                        print(
                            f"[IntentMemory] Embedding dimension changed, "
                            f"resetting collection ({count} old examples removed).",
                            file=sys.stderr,
                        )
                        self._client.delete_collection(collection_name)
                        self._collection = self._client.get_or_create_collection(
                            name=collection_name,
                            metadata={"hnsw:space": "cosine"},
                        )
                        count = 0
                    else:
                        raise

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

    def store(self, question: str, intent: Dict[str, Any]) -> None:
        """Store a successful (question, intent) pair."""
        if self._collection is None:
            return

        import hashlib
        doc_id = hashlib.md5(question.lower().strip().encode()).hexdigest()

        existing = self._collection.get(ids=[doc_id])
        if existing and existing["ids"]:
            return

        tc = _task_class_from_intent(intent)
        self._collection.add(
            ids=[doc_id],
            documents=[question],
            metadatas=[
                {
                    "intent": json.dumps(intent, default=str),
                    _TASK_CLASS_INTENT: tc,
                }
            ],
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
        min_similarity: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Find the most similar past questions and return their intents.

        Returns a list of {"question": str, "intent": dict, "similarity": float}
        sorted by similarity descending.

        Default min_similarity can be overridden with env INTENTQL_MEMORY_MIN_SIMILARITY
        (default 0.78 — 0.60 treated too many topic-overlap pairs as similar).

        Set INTENTQL_MEMORY_TASK_FILTER=0 to disable task_class filtering (debug only).
        """
        if min_similarity is None:
            raw = os.environ.get("INTENTQL_MEMORY_MIN_SIMILARITY", "0.78")
            try:
                min_similarity = float(raw)
            except ValueError:
                min_similarity = 0.78

        task_filter_on = os.environ.get(
            "INTENTQL_MEMORY_TASK_FILTER", "1"
        ).strip().lower() not in ("0", "false", "no", "off")

        if self._collection is None or self._collection.count() == 0:
            return []

        n_docs = self._collection.count()
        # Over-fetch: task_class filter drops most neighbors for unrelated intent shapes.
        fetch_n = min(max(top_k * 10, top_k), n_docs)

        results = self._collection.query(
            query_texts=[question],
            n_results=fetch_n,
        )

        if not results or not results["documents"] or not results["documents"][0]:
            return []

        query_tc = _task_class_from_question(question)

        matched: List[Dict[str, Any]] = []
        skipped_tc = 0
        skipped_legacy = 0
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

            if task_filter_on:
                stored_tc = meta.get(_TASK_CLASS_INTENT)
                if not stored_tc:
                    skipped_legacy += 1
                    continue
                if stored_tc != query_tc:
                    skipped_tc += 1
                    continue

            matched.append({
                "question": doc,
                "intent": intent,
                "similarity": round(similarity, 3),
            })
            if len(matched) >= top_k:
                break

        if skipped_legacy and task_filter_on:
            print(
                f"[IntentMemory] Skipped {skipped_legacy} legacy example(s) without "
                f"{_TASK_CLASS_INTENT} — delete .intent_memory once to re-store with classes.",
                file=sys.stderr,
            )

        if matched:
            print(
                f"[IntentMemory] Found {len(matched)} similar example(s) "
                f"(best: {matched[0]['similarity']}, "
                f"question_task={query_tc}"
                + (f", dropped {skipped_tc} wrong-task" if skipped_tc else "")
                + ")",
                file=sys.stderr,
            )
        elif task_filter_on and skipped_tc + skipped_legacy > 0:
            print(
                f"[IntentMemory] No examples after filter "
                f"(question_task={query_tc}, "
                f"dropped_wrong_task={skipped_tc}, legacy_no_class={skipped_legacy})",
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
