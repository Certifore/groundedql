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

import re
import sys
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

from .intent import QueryIntent, intent_json_schema
from .llm_adapters import make_llm_client
from .plan_canonical import canonicalize_query_plan
from .api.api import _resolve_relative_dates
from .value_index import (
    format_value_index_for_prompt,
    resolve_intent_values,
    validate_intent_against_index,
)
from .intent_normalize import normalize_intent
from .intent_memory import IntentMemory


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
- dataset: pick the ONE table that has the data. If the question is about how many
  **work orders** each asset has (or which asset has the most work orders), the rows to
  count live on the **work_orders** (or similarly named) table — not on an **assets** /
  catalog table that describes equipment. Count work_order rows grouped by asset_tag.
- keyword: the topic/trade word the user is asking about (e.g. "plumbing",
  "electrical", "HVAC").  Leave null if the question has no free-text topic.
  Use the root form of the word (e.g. "plumbing" not "plumb").
- filters: column + values.  Use UPPER CASE for values that the schema says
  are stored in upper case.  When the user mentions building names, always
  use column "building_name" — never "asset_tag".
  CRITICAL: if the question names specific entities (buildings, workers, etc.),
  you MUST include them as filter values.  Never omit explicit entity names.
  CRITICAL: if a KNOWN DATABASE VALUES list is provided, you MUST pick values
  from that list.  Map user input to the closest matching known value.
  For example, if the user says "bechtel" and the known building names include
  "BECHTEL RESIDENCE", use "BECHTEL RESIDENCE".  If a user says "page house"
  and the list has "PAGE HOUSE", use "PAGE HOUSE" exactly.
  If the user names a specific record ID, put it in filters on the table's
  primary_id column — do not rely on list mode alone.
  Never use the substring "WORK" or similar as a primary_id value just because
  the user said "work orders" — that names the entity type, not a row id.
- time_range: pick from the enum if the question mentions a time period.
  "last year" → "last_year", "this year" → "this_year",
  "last 3 years" / "past three years" → "last_3_years",
  "last 2 years" → "last_2_years",
  "last 6 months" / "past half year" → "last_6_months", etc.
- aggregation: "count" for "how many / total / tally / numbers / breakdown /
  report / summary / give me".  Use "list" ONLY when the user explicitly asks
  to see individual records, details, or specific IDs.  When in doubt and
  the question names multiple entities, prefer "count".
  Use "ratio" for "what percent / what % / proportion / share" of rows match
  a keyword (e.g. "what % of work orders are plumbing?") — set keyword to the
  topic and aggregation to "ratio".
- group_by: if the question says "per X", "for each X", "by X", or lists
  multiple specific values and wants a total for each, put X's column here.
  IMPORTANT: when the question names multiple specific values of a column
  (e.g. "for house A, house B, house C"), ALWAYS put that column in group_by.
  For trends ("trend over …", "over time", "by month"), include the primary
  date column in group_by; normalization may add time_bucket (month/year).
- time_bucket: optional — "month", "year", "quarter", or "day" when bucketing
  a trend over the primary date (usually inferred from the question).
- sort_direction: "desc" for most/highest, "asc" for least/lowest.
- sort_column: for "most recent / latest / newest" ONE record (e.g. one work order), use
  aggregation "list", group_by [], the table's primary_date column as sort_column,
  sort_direction "desc", limit 1. Do NOT use count+group_by(primary_id) for that —
  that ranks by row count, not by date.
- limit: integer if "top N" or "first N", otherwise null.
- output_columns: for "list" queries, which columns to show.
"""


def extract_intent(
    *,
    llm: Any,
    question: str,
    schema: Dict[str, Any],
    temperature: float = 0.0,
    value_index: Optional[Dict[str, Dict[str, List[str]]]] = None,
    feedback: Optional[str] = None,
    few_shot_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Call the LLM to extract a QueryIntent dict from *question*.

    Args:
        value_index: If provided, known DB values are injected into the prompt
            so the LLM picks from real values instead of guessing.
        feedback: If provided (on retry), appended as a correction hint.
        few_shot_prompt: If provided, similar past examples injected into prompt.
    """
    client = make_llm_client(llm) if not hasattr(llm, "generate_json") else llm

    schema_text = _schema_summary_for_prompt(schema)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "system", "content": f"DATABASE SCHEMA:\n{schema_text}"},
    ]

    if value_index:
        vi_text = format_value_index_for_prompt(value_index)
        messages.append({"role": "system", "content": vi_text})

    if few_shot_prompt:
        messages.append({"role": "system", "content": few_shot_prompt})

    if feedback:
        messages.append({
            "role": "system",
            "content": (
                f"CORRECTION — your previous extraction had issues:\n{feedback}\n"
                "Please fix these issues in your response."
            ),
        })

    messages.append({"role": "user", "content": question})

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
    "last_6_months": {
        "gte": {"$relative_date": {"op": "now_minus_days", "days": 182}},
    },
    "last_2_years": {
        "gte": {"$relative_date": {"op": "now_minus_days", "days": 730}},
    },
    "last_3_years": {
        "gte": {"$relative_date": {"op": "now_minus_days", "days": 1095}},
    },
    "yesterday": {
        "gte": {"$relative_date": {"op": "now_minus_days", "days": 1}},
        "lt":  {"$relative_date": {"op": "today"}},
    },
    "today": {
        "gte": {"$relative_date": {"op": "today"}},
    },
}


def _should_use_equals_filter(
    col: str,
    vals: List[str],
    primary_id: Optional[str],
) -> bool:
    """Single-value filters on IDs or id-shaped tokens use equality, not substring match."""
    if len(vals) != 1:
        return False
    v = str(vals[0]).strip()
    if primary_id and col == primary_id:
        return True
    if col.endswith("_id"):
        return True
    if len(v) >= 8 and re.match(r"^[A-Z0-9\-_]+$", v, re.I):
        return True
    return False


def _ratio_percentage_plan(
    dataset: str,
    primary_id: str,
    base_filters: List[Dict[str, Any]],
    keyword_where: Dict[str, Any],
) -> Dict[str, Any]:
    """Scalar subqueries: count with keyword OR vs count overall, then pct = num/den*100."""
    num_sub: Dict[str, Any] = {
        "dataset": dataset,
        "metrics": [{"agg": "count_distinct", "field": primary_id, "alias": "c"}],
        "filters": list(base_filters),
        "dimensions": [],
        "where": keyword_where,
        "limit": 1,
        "offset": 0,
    }
    den_sub: Dict[str, Any] = {
        "dataset": dataset,
        "metrics": [{"agg": "count_distinct", "field": primary_id, "alias": "c"}],
        "filters": list(base_filters),
        "dimensions": [],
        "limit": 1,
        "offset": 0,
    }
    pct_expr: Dict[str, Any] = {
        "op": "/",
        "args": [
            {
                "op": "*",
                "args": [
                    {"scalar_subquery": {"plan": num_sub}},
                    {"lit": 100.0},
                ],
            },
            {
                "func": "nullif",
                "args": [
                    {"scalar_subquery": {"plan": den_sub}},
                    {"lit": 0},
                ],
            },
        ],
    }
    return {
        "version": "1.0",
        "dataset": dataset,
        "select": [{"expr": pct_expr, "alias": "pct"}],
        "limit": 1,
        "offset": 0,
    }


def build_plan_from_intent(
    intent: Dict[str, Any],
    schema: Dict[str, Any],
    value_index: Optional[Dict[str, Dict[str, List[str]]]] = None,
) -> Dict[str, Any]:
    """Deterministically convert an intent dict into a QueryPlan dict.

    Every structural decision (keyword OR, date sentinel, count_distinct on
    primary_id, etc.) is made here based on schema metadata — not by the LLM.
    If *value_index* is provided, filter values are fuzzy-matched against
    known DB values before plan construction.
    """
    if value_index:
        intent = resolve_intent_values(intent, value_index)

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
            if _should_use_equals_filter(col, vals, primary_id):
                plan["filters"].append({"field": col, "op": "=", "value": vals[0]})
            else:
                plan["filters"].append({"field": col, "op": "contains", "value": vals[0]})
        else:
            plan["filters"].append({"field": col, "op": "in", "value": vals})

    # --- keyword search (OR across keyword_search_or columns) ---
    # Normalization already removed any redundant keyword filters on kso columns,
    # so we always generate the keyword OR clause if a keyword is present.
    keyword = intent.get("keyword")
    if keyword and kso_cols:
        plan["where"] = {
            "or": [
                {"cmp": {"left": {"col": c}, "op": "contains", "right": keyword}}
                for c in sorted(kso_cols)
            ]
        }
    elif keyword:
        filter_cols = {f.get("column", "") for f in intent.get("filters") or []}
        text_cols = [
            c["name"]
            for c in table.get("columns", [])
            if (c.get("type") or "").lower() == "varchar"
            and c["name"] != primary_id
            and c["name"] not in filter_cols
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

    # --- ratio (% with keyword as numerator) → scalar subquery plan ---
    agg = intent.get("aggregation", "count")
    if agg == "ratio":
        if not primary_id:
            print(
                "[DSL intent] aggregation=ratio requires primary_id; using count instead",
                file=sys.stderr,
            )
            agg = "count"
        elif not plan.get("where"):
            print(
                "[DSL intent] aggregation=ratio requires a keyword search; using count instead",
                file=sys.stderr,
            )
            agg = "count"
        else:
            ratio_plan = _ratio_percentage_plan(
                dataset,
                primary_id,
                plan["filters"],
                plan["where"],
            )
            ratio_plan = _resolve_relative_dates(ratio_plan)
            return canonicalize_query_plan(ratio_plan)

    # --- aggregation + metrics ---
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
    tb = intent.get("time_bucket")
    for col in group_by:
        if col not in valid_cols:
            continue
        if primary_date and col == primary_date and tb:
            dim_alias = f"{col}_{tb}"
            plan["dimensions"].append({"field": col, "alias": dim_alias, "time_bucket": tb})
        else:
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

    # --- safety: exclude null/empty values on group-by columns (aggregates only) ---
    # When grouping by a column (e.g. asset_tag to find "which asset has the most"),
    # null/empty values form a giant catch-all bucket that dominates the results.
    # Do NOT apply to pure "list" queries: output columns are often optional text fields
    # that may be null without invalidating the row.
    if plan["metrics"]:
        existing_filter_fields = {f.get("field") for f in plan["filters"]}
        for dim in plan["dimensions"]:
            dim_field = dim.get("field", "")
            if dim_field and dim_field not in existing_filter_fields and dim_field != primary_date:
                plan["filters"].append({"field": dim_field, "op": "!=", "value": ""})
                plan["filters"].append({"field": dim_field, "op": "is_not_null", "value": True})
                print(
                    f"[DSL intent] Auto-added null/empty exclusion for group-by column '{dim_field}'",
                    file=sys.stderr,
                )

    # --- sort & limit ---
    sort_dir = intent.get("sort_direction")
    sort_col = intent.get("sort_column")
    if agg == "list" and sort_col and sort_col in valid_cols:
        # Order by primary_date (etc.) without adding it as a dimension — list/detail
        # rows must not be GROUP BY date buckets; empty dimensions + no metrics → SELECT *.
        plan["order_by"] = [{"by": sort_col, "dir": sort_dir or "desc"}]
    elif sort_dir and plan["metrics"]:
        plan["order_by"] = [{"by": plan["metrics"][0]["alias"], "dir": sort_dir}]
    elif any(d.get("time_bucket") for d in plan["dimensions"]) and plan["metrics"]:
        for d in plan["dimensions"]:
            if d.get("time_bucket"):
                plan["order_by"] = [{"by": d["alias"], "dir": "asc"}]
                break

    user_limit = intent.get("limit")
    if user_limit:
        plan["limit"] = user_limit
    elif agg == "count" and not plan["dimensions"]:
        plan["limit"] = 1

    if agg == "list" and primary_id:
        for f in plan["filters"]:
            if f.get("field") == primary_id and f.get("op") == "=":
                plan["limit"] = min(int(plan.get("limit", 100)), 25)
                break

    # --- resolve $relative_date sentinels ---
    plan = _resolve_relative_dates(plan)
    plan = canonicalize_query_plan(plan)

    return plan


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------

class IntentPlanner:
    """Two-stage planner: LLM intent extraction → deterministic plan builder.

    Pipeline:
    1. Retrieve similar past questions from memory (few-shot examples)
    2. LLM extracts intent (with value index + few-shot examples in prompt)
    3. Normalize intent (absorb redundant keyword filters, fix group_by, etc.)
    4. Validate intent against value index, retry with feedback if needed
    5. Fuzzy-resolve values and build QueryPlan deterministically
    6. Store successful (question, intent) pair in memory for future use
    """

    def __init__(
        self,
        *,
        llm: Any,
        schema_path: str,
        temperature: float = 0.0,
        model: Optional[str] = None,
        value_index: Optional[Dict[str, Dict[str, List[str]]]] = None,
        max_intent_retries: int = 2,
        memory: Optional[IntentMemory] = None,
    ):
        self.llm = make_llm_client(llm, model=model)
        self.schema_path = schema_path
        self.temperature = temperature
        self.value_index = value_index
        self.max_intent_retries = max_intent_retries
        self.memory = memory

    def _read_schema(self) -> Dict[str, Any]:
        return yaml.safe_load(Path(self.schema_path).read_text()) or {}

    def plan(self, question: str) -> Dict[str, Any]:
        """Extract intent and build a QueryPlan deterministically.

        Uses few-shot memory for consistency and validates extracted intent
        values against the value index with retry.
        """
        schema = self._read_schema()
        retry_count = 0
        feedback = None

        # Step 1: Retrieve similar past examples for few-shot prompting
        few_shot_prompt = None
        if self.memory:
            examples = self.memory.retrieve(question)
            if examples:
                few_shot_prompt = self.memory.format_few_shot_examples(examples)

        for attempt in range(1 + self.max_intent_retries):
            # Step 2: LLM extracts intent
            intent = extract_intent(
                llm=self.llm,
                question=question,
                schema=schema,
                temperature=self.temperature,
                value_index=self.value_index,
                feedback=feedback,
                few_shot_prompt=few_shot_prompt,
            )

            # Step 3: Deterministic normalization
            intent = normalize_intent(intent, schema, question=question)

            # Step 4: Validate against value index
            if self.value_index:
                issues = validate_intent_against_index(intent, self.value_index)
                if issues and attempt < self.max_intent_retries:
                    feedback = "\n".join(issues)
                    retry_count += 1
                    print(
                        f"[DSL] Intent validation issues (attempt {attempt + 1}), "
                        f"retrying: {feedback}",
                        file=sys.stderr,
                    )
                    continue

            break

        # Step 5: Build plan deterministically
        plan = build_plan_from_intent(intent, schema, value_index=self.value_index)

        plan["meta"] = {
            "pipeline": "intent",
            "intent": intent,
            "retry_count": retry_count,
            "auto_fixes_applied": [],
            "validation_errors": [],
            "lint_errors": [],
        }

        # Step 6: Store successful example in memory
        if self.memory:
            self.memory.store(question, intent)

        print(f"[DSL] Built plan from intent: {plan}", file=sys.stderr)
        return plan
