"""
intent_planner.py — Two-stage query pipeline.

Stage 1 (LLM):  Extract a lightweight ``QueryIntent`` from the user's
                 natural-language question + schema context.
Stage 2 (deterministic):  Build a correct ``QueryPlan`` dict from the
                          intent + schema metadata.

This replaces the "LLM generates full QueryPlan → reactive autofix" loop
with a simpler LLM task (intent extraction) and a deterministic plan
builder that cannot produce structurally wrong plans.
"""
from __future__ import annotations

import sys
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

from .intent import QueryIntent, intent_json_schema
from .llm_adapters import make_llm_client
from .plan_canonical import canonicalize_query_plan
from .api.api import _resolve_relative_dates


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _table_meta(schema: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    """Return schema metadata for *dataset*, or empty dict."""
    for t in schema.get("tables", []):
        if t.get("name") == dataset:
            return t
    return {}


def _column_names(table: Dict[str, Any]) -> set[str]:
    return {c["name"] for c in table.get("columns", []) if isinstance(c, dict) and "name" in c}


def _schema_summary_for_prompt(schema: Dict[str, Any]) -> str:
    """Compact schema description injected into the intent-extraction prompt."""
    parts: list[str] = []

    ctx = schema.get("context")
    if ctx:
        parts.append(f"Context: {ctx.strip()}")

    for t in schema.get("tables", []):
        tname = t.get("name", "?")
        cols = []
        for c in t.get("columns", []):
            ctype = c.get("type", "")
            desc = (c.get("description") or "").strip().replace("\n", " ")
            cols.append(f"  - {c['name']} ({ctype}): {desc}")
        hdr = f"Table: {tname}"
        pid = t.get("primary_id")
        if pid:
            hdr += f"  [primary_id={pid}]"
        pdate = t.get("primary_date")
        if pdate:
            hdr += f"  [primary_date={pdate}]"
        kso = t.get("keyword_search_or")
        if kso:
            hdr += f"  [keyword_search_or={kso}]"
        parts.append(hdr)
        parts.extend(cols)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Stage 1 — LLM intent extraction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an intent extractor.  Given a database schema and a user question,
output a JSON object capturing WHAT the user wants — not HOW to query it.

Rules:
- dataset: pick the ONE table that has the data.
- keyword: the topic/trade word the user is asking about (e.g. "plumbing",
  "electrical", "HVAC").  Leave null if the question has no free-text topic.
  Use the root form of the word (e.g. "plumbing" not "plumb").
- filters: column + values.  Use UPPER CASE for values that the schema says
  are stored in upper case.  When the user mentions building names, always
  use column "building_name" — never "asset_tag".
  CRITICAL: if the question names specific entities (buildings, workers, etc.),
  you MUST include them as filter values.  Never omit explicit entity names.
- time_range: pick from the enum if the question mentions a time period.
  "last year" → "last_year", "this year" → "this_year", etc.
- aggregation: "count" for "how many / total / tally / numbers / breakdown /
  report / summary / give me".  Use "list" ONLY when the user explicitly asks
  to see individual records, details, or specific IDs.  When in doubt and
  the question names multiple entities, prefer "count".
- group_by: if the question says "per X", "for each X", "by X", or lists
  multiple specific values and wants a total for each, put X's column here.
  IMPORTANT: when the question names multiple specific values of a column
  (e.g. "for house A, house B, house C"), ALWAYS put that column in group_by.
- sort_direction: "desc" for most/highest, "asc" for least/lowest.
- limit: integer if "top N" or "first N", otherwise null.
- output_columns: for "list" queries, which columns to show.
"""


def extract_intent(
    *,
    llm: Any,
    question: str,
    schema: Dict[str, Any],
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Call the LLM to extract a QueryIntent dict from *question*."""
    client = make_llm_client(llm) if not hasattr(llm, "generate_json") else llm

    schema_text = _schema_summary_for_prompt(schema)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "system", "content": f"DATABASE SCHEMA:\n{schema_text}"},
        {"role": "user", "content": question},
    ]

    raw = client.generate_json(
        json_schema=intent_json_schema(),
        messages=messages,
        temperature=temperature,
    )
    print(f"[DSL] Intent: {raw}", file=sys.stderr)
    return raw


# ---------------------------------------------------------------------------
# Stage 2 — deterministic plan builder
# ---------------------------------------------------------------------------

_DATE_RANGE_MAP: Dict[str, Dict[str, Any]] = {
    "last_year": {
        "gte": {"$relative_date": {"op": "calendar_year_start", "year_offset": -1}},
        "lt":  {"$relative_date": {"op": "calendar_year_start", "year_offset": 0}},
    },
    "this_year": {
        "gte": {"$relative_date": {"op": "calendar_year_start", "year_offset": 0}},
        "lt":  {"$relative_date": {"op": "calendar_year_start", "year_offset": 1}},
    },
    "last_month": {
        "gte": {"$relative_date": {"op": "now_minus_days", "days": 30}},
    },
    "this_month": {
        "gte": {"$relative_date": {"op": "now_minus_days", "days": 30}},
    },
    "last_7_days": {
        "gte": {"$relative_date": {"op": "now_minus_days", "days": 7}},
    },
    "last_30_days": {
        "gte": {"$relative_date": {"op": "now_minus_days", "days": 30}},
    },
    "last_90_days": {
        "gte": {"$relative_date": {"op": "now_minus_days", "days": 90}},
    },
    "last_12_months": {
        "gte": {"$relative_date": {"op": "now_minus_days", "days": 365}},
    },
    "yesterday": {
        "gte": {"$relative_date": {"op": "now_minus_days", "days": 1}},
        "lt":  {"$relative_date": {"op": "today"}},
    },
    "today": {
        "gte": {"$relative_date": {"op": "today"}},
    },
}


def build_plan_from_intent(
    intent: Dict[str, Any],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Deterministically convert an intent dict into a QueryPlan dict.

    Every structural decision (keyword OR, date sentinel, count_distinct on
    primary_id, etc.) is made here based on schema metadata — not by the LLM.
    """
    dataset = intent.get("dataset", "")
    table = _table_meta(schema, dataset)
    valid_cols = _column_names(table)
    primary_id = table.get("primary_id")
    primary_date = table.get("primary_date")
    kso_cols = table.get("keyword_search_or") or []

    plan: Dict[str, Any] = {
        "version": "1.0",
        "dataset": dataset,
        "filters": [],
        "dimensions": [],
        "metrics": [],
        "order_by": [],
        "limit": 100,
        "offset": 0,
    }

    # --- entity filters ---
    for f in intent.get("filters") or []:
        col = f.get("column", "")
        vals = f.get("values") or []
        if col not in valid_cols:
            print(
                f"[DSL intent] Ignoring unknown filter column '{col}'",
                file=sys.stderr,
            )
            continue
        if len(vals) == 1:
            plan["filters"].append({"field": col, "op": "contains", "value": vals[0]})
        else:
            plan["filters"].append({"field": col, "op": "in", "value": vals})

    # --- keyword search (OR across keyword_search_or columns) ---
    keyword = intent.get("keyword")
    if keyword and kso_cols:
        plan["where"] = {
            "or": [
                {"cmp": {"left": {"col": c}, "op": "contains", "right": keyword}}
                for c in sorted(kso_cols)
            ]
        }
    elif keyword:
        # no keyword_search_or declared — best-effort contains on dataset
        text_cols = [
            c["name"]
            for c in table.get("columns", [])
            if (c.get("type") or "").lower() == "varchar"
            and c["name"] != primary_id
        ]
        if text_cols:
            plan["where"] = {
                "or": [
                    {"cmp": {"left": {"col": c}, "op": "contains", "right": keyword}}
                    for c in text_cols[:3]
                ]
            }

    # --- time range ---
    if intent.get("time_range") and primary_date:
        dr = _DATE_RANGE_MAP.get(intent["time_range"])
        if dr:
            if "gte" in dr:
                plan["filters"].append({"field": primary_date, "op": ">=", "value": dr["gte"]})
            if "lt" in dr:
                plan["filters"].append({"field": primary_date, "op": "<", "value": dr["lt"]})

    # --- aggregation + metrics ---
    agg = intent.get("aggregation", "count")
    if agg == "count":
        field = primary_id or "*"
        agg_func = "count_distinct" if primary_id else "count"
        plan["metrics"] = [{"agg": agg_func, "field": field, "alias": "total"}]
    elif agg == "list":
        plan["metrics"] = []
    elif agg in ("sum", "avg", "min", "max"):
        agg_field = intent.get("aggregation_field")
        if agg_field and agg_field in valid_cols:
            plan["metrics"] = [{"agg": agg, "field": agg_field, "alias": f"{agg}_{agg_field}"}]
        else:
            field = primary_id or "*"
            plan["metrics"] = [{"agg": "count_distinct" if primary_id else "count", "field": field, "alias": "total"}]

    # --- dimensions (group by) ---
    group_by = intent.get("group_by") or []
    if isinstance(group_by, str):
        group_by = [group_by]
    for col in group_by:
        if col in valid_cols:
            plan["dimensions"].append({"field": col, "alias": col})

    # --- output columns (for list queries) ---
    if agg == "list":
        existing = {d["field"] for d in plan["dimensions"]}
        out_cols = intent.get("output_columns") or []
        if isinstance(out_cols, str):
            out_cols = [out_cols]
        for col in out_cols:
            if col in valid_cols and col not in existing:
                plan["dimensions"].append({"field": col, "alias": col})

    # --- safety: multi-value IN filter without grouping → add dimension ---
    if plan["metrics"] and not plan["dimensions"]:
        for f in plan["filters"]:
            if f.get("op") == "in" and f.get("field") != primary_date:
                col = f["field"]
                plan["dimensions"].append({"field": col, "alias": col})
                print(
                    f"[DSL intent] Auto-added dimension '{col}' for multi-value filter",
                    file=sys.stderr,
                )

    # --- sort & limit ---
    sort_dir = intent.get("sort_direction")
    if sort_dir and plan["metrics"]:
        plan["order_by"] = [{"by": plan["metrics"][0]["alias"], "dir": sort_dir}]

    user_limit = intent.get("limit")
    if user_limit:
        plan["limit"] = user_limit
    elif agg == "count" and not plan["dimensions"]:
        plan["limit"] = 1

    # --- resolve $relative_date sentinels ---
    plan = _resolve_relative_dates(plan)
    plan = canonicalize_query_plan(plan)

    return plan


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------

class IntentPlanner:
    """Two-stage planner: LLM intent extraction → deterministic plan builder."""

    def __init__(
        self,
        *,
        llm: Any,
        schema_path: str,
        temperature: float = 0.0,
        model: Optional[str] = None,
    ):
        self.llm = make_llm_client(llm, model=model)
        self.schema_path = schema_path
        self.temperature = temperature

    def _read_schema(self) -> Dict[str, Any]:
        return yaml.safe_load(Path(self.schema_path).read_text()) or {}

    def plan(self, question: str) -> Dict[str, Any]:
        """Extract intent and build a QueryPlan deterministically."""
        schema = self._read_schema()
        intent = extract_intent(
            llm=self.llm,
            question=question,
            schema=schema,
            temperature=self.temperature,
        )
        plan = build_plan_from_intent(intent, schema)

        plan["meta"] = {
            "pipeline": "intent",
            "intent": intent,
            "retry_count": 0,
            "auto_fixes_applied": [],
            "validation_errors": [],
            "lint_errors": [],
        }

        print(f"[DSL] Built plan from intent: {plan}", file=sys.stderr)
        return plan
