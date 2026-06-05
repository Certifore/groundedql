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
from .evidence_planner import build_evidence_plan


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


def _schema_table_map(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        t["name"]: t
        for t in schema.get("tables", []) or []
        if isinstance(t, dict) and isinstance(t.get("name"), str)
    }


def _split_column_ref(schema: Dict[str, Any], dataset: str, ref: str) -> tuple[str, str]:
    """Split a logical column ref into (table, column), preserving table names with dots."""
    if not isinstance(ref, str) or not ref:
        return dataset, ""
    if "." not in ref:
        return dataset, ref
    known = set(_schema_table_map(schema))
    parts = ref.split(".")
    for n in range(len(parts) - 1, 0, -1):
        tname = ".".join(parts[:n])
        if tname in known:
            return tname, ".".join(parts[n:])
    return ref.split(".", 1)


def _table_for_ref(schema: Dict[str, Any], dataset: str, ref: str) -> Dict[str, Any]:
    table_name, _col = _split_column_ref(schema, dataset, ref)
    return _table_meta(schema, table_name)


def _local_col_for_ref(schema: Dict[str, Any], dataset: str, ref: str) -> str:
    _table_name, col = _split_column_ref(schema, dataset, ref)
    return col


def _column_ref_valid(schema: Dict[str, Any], dataset: str, ref: str) -> bool:
    table_name, col = _split_column_ref(schema, dataset, ref)
    table = _table_meta(schema, table_name)
    return bool(table and col in _column_names(table))


def _alias_for_ref(ref: str) -> str:
    return ref.replace(".", "__")


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

    links = schema.get("links") or []
    if links:
        parts.append("Links:")
        for link in links:
            name = link.get("name", "?")
            frm = link.get("from_table", "?")
            to = link.get("to_table", "?")
            ons = []
            for on in link.get("on") or []:
                left = on.get("left", "?")
                right = on.get("right", "?")
                op = on.get("op", "=")
                ons.append(f"{left} {op} {right}")
            on_text = "; ".join(ons) if ons else "(no on clause)"
            parts.append(f"  - {name}: {frm} -> {to} on {on_text}")

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
- filters: column + optional op + values.  Use UPPER CASE for values that the schema says
  are stored in upper case.  If a filter column belongs to a linked table, write
  it as table.column (for example customers.country) so the deterministic link
  graph can join it.  Use unqualified columns only when they belong to dataset.
  Supported ops: =, !=, >, >=, <, <=, in, not_in, contains, not_contains,
  starts_with, ends_with, between, is_null, is_not_null.  Leave op null when a
  simple equality/contains/in choice is enough.
  When the user mentions building names, always use column "building_name" — never "asset_tag".
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
  For "ratio of A against B", "percentage of A compared with B", or
  "difference between A and B", also fill comparison:
  comparison.operator = "ratio" or "difference"; comparison.left.filters define
  A; comparison.right.filters define B.  Use comparison.scale="percent" only for
  percent/% questions; for ordinary "ratio" use scale="raw".  If the comparison
  is over a numeric measure, set comparison.metric_aggregation to sum/avg/min/max
  and comparison.metric_field to that numeric column; otherwise leave it as count.
  For conditional aggregate questions, fill conditional_metrics and formula_metrics
  instead of trying to encode the calculation as one keyword. Examples:
    * "sum of sales in 2020 vs 2021" → conditional_metrics sum_2020 and sum_2021.
    * "percentage increase from 2020 to 2021" → conditional_metrics for both years,
      then formula_metrics diff = sum_2021 - sum_2020, pct = diff / sum_2020 with
      nullif_right=true and scale=100.
    * "average per month" → aggregate metric, then formula metric dividing by the
      period count if the question explicitly gives the period count.
  Set include=false on intermediate conditional/formula metrics that are only used
  to compute a final formula. For multi-part questions, include each requested
  final formula metric.
- group_by: if the question says "per X", "for each X", "by X", or lists
  multiple specific values and wants a total for each, put X's column here.
  IMPORTANT: when the question names multiple specific values of a column
  (e.g. "for house A, house B, house C"), ALWAYS put that column in group_by.
  For trends ("trend over …", "over time", "by month"), include the primary
  date column in group_by; normalization may add time_bucket (month/year).
  For "which year/month had the most/highest/peak <numeric measure>", group by
  the date/period column, set time_bucket to year/month, aggregate the numeric
  measure with sum unless the question clearly asks for average/min/max, sort
  desc, and limit 1.
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
    if isinstance(raw, dict) and raw.get("comparison") is None:
        left = raw.get("left")
        right = raw.get("right")
        if raw.get("aggregation") in {"ratio", "difference"} and isinstance(left, dict) and isinstance(right, dict):
            raw["comparison"] = {
                "operator": raw.get("aggregation"),
                "left": left,
                "right": right,
            }

    allowed = set(QueryIntent.model_fields)
    cleaned = {k: v for k, v in raw.items() if k in allowed}
    for list_field in ("filters", "group_by", "output_columns", "conditional_metrics", "formula_metrics"):
        if cleaned.get(list_field) is None:
            cleaned[list_field] = []

    try:
        validated = QueryIntent.model_validate(cleaned).model_dump(exclude_none=True)
    except Exception as exc:
        raise ValueError(f"LLM returned invalid QueryIntent: {exc}; raw={str(raw)[:500]}") from exc

    print(f"[DSL] Intent: {validated}", file=sys.stderr)
    return validated


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
    if col.endswith("_id") or col.endswith("id"):
        return True
    if len(v) >= 8 and re.match(r"^[A-Z0-9\-_]+$", v, re.I):
        return True
    return False


def _column_type(table: Dict[str, Any], col: str) -> str:
    for c in table.get("columns", []) or []:
        if c.get("name") == col:
            return (c.get("type") or "").lower()
    return ""


def _column_type_for_ref(schema: Dict[str, Any], dataset: str, ref: str) -> str:
    table = _table_for_ref(schema, dataset, ref)
    col = _local_col_for_ref(schema, dataset, ref)
    return _column_type(table, col)


def _primary_id_for_ref(schema: Dict[str, Any], dataset: str, ref: str) -> Optional[str]:
    table = _table_for_ref(schema, dataset, ref)
    return table.get("primary_id")


def _numeric_type(ctype: str) -> bool:
    return any(tok in ctype for tok in ("int", "numeric", "float", "double", "real", "decimal"))


def _date_like_column(col: str) -> bool:
    n = (col or "").lower()
    return n == "date" or n.endswith("_date") or n.endswith("date")


def _primary_date_column(table: Dict[str, Any]) -> Optional[str]:
    primary_date = table.get("primary_date")
    if isinstance(primary_date, str) and primary_date:
        return primary_date
    for col in table.get("columns", []) or []:
        name = col.get("name") if isinstance(col, dict) else None
        if isinstance(name, str) and _date_like_column(name):
            return name
    return None


def _maybe_compact_yyyymm_range(vals: List[Any]) -> List[str] | None:
    text_vals = [str(v).strip() for v in vals if str(v).strip()]
    if len(text_vals) < 2 or not all(re.fullmatch(r"\d{6}", v) for v in text_vals):
        return None
    ordered = sorted(dict.fromkeys(text_vals))
    return [ordered[0], ordered[-1]]


_FILTER_OP_ALIASES = {
    "eq": "=",
    "equals": "=",
    "equal": "=",
    "ne": "!=",
    "not_equals": "!=",
    "not equal": "!=",
    "gt": ">",
    "gte": ">=",
    "ge": ">=",
    "lt": "<",
    "lte": "<=",
    "le": "<=",
    "not in": "not_in",
    "not-in": "not_in",
    "not contains": "not_contains",
    "starts with": "starts_with",
    "ends with": "ends_with",
}

_FILTER_OPS = {
    "=", "!=", ">", ">=", "<", "<=",
    "in", "not_in",
    "contains", "not_contains", "starts_with", "ends_with",
    "between",
    "is_null", "is_not_null",
}


def _normalize_filter_op(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    op = str(raw).strip().lower()
    op = _FILTER_OP_ALIASES.get(op, op)
    return op if op in _FILTER_OPS else None


def _single_value(vals: List[Any]) -> Any:
    return vals[0] if vals else None


def _build_filter_clause(
    filt: Dict[str, Any],
    *,
    schema: Dict[str, Any],
    dataset: str,
) -> Optional[Dict[str, Any]]:
    col_ref = str(filt.get("column", "")).strip()
    if not col_ref:
        return None
    if not _column_ref_valid(schema, dataset, col_ref):
        print(
            f"[DSL intent] Ignoring unknown filter column '{col_ref}'",
            file=sys.stderr,
        )
        return None

    vals = filt.get("values") or []
    if not isinstance(vals, list):
        vals = [vals]
    vals = [v for v in vals if v is not None and str(v).strip() != ""]

    local_col = _local_col_for_ref(schema, dataset, col_ref)
    ctype = _column_type_for_ref(schema, dataset, col_ref)
    primary_id = _primary_id_for_ref(schema, dataset, col_ref)

    op = _normalize_filter_op(filt.get("op"))
    compact_range = _maybe_compact_yyyymm_range(vals) if _date_like_column(local_col) else None
    if compact_range and op in (None, "between", "in"):
        return {"field": col_ref, "op": "between", "value": compact_range}

    if _date_like_column(local_col) and len(vals) == 1:
        value_text = str(vals[0]).strip()
        if re.fullmatch(r"(?:19|20)\d{2}", value_text) and op in (None, "=", "contains", "starts_with"):
            return {"field": col_ref, "op": "starts_with", "value": value_text}

    if op in {"is_null", "is_not_null"}:
        return {"field": col_ref, "op": op, "value": True}

    if not vals:
        print(
            f"[DSL intent] Ignoring empty filter values for column '{col_ref}'",
            file=sys.stderr,
        )
        return None

    if op == "between":
        if len(vals) != 2:
            print(
                f"[DSL intent] Ignoring invalid between filter for column '{col_ref}'",
                file=sys.stderr,
            )
            return None
        return {"field": col_ref, "op": "between", "value": vals}

    if op in {"in", "not_in"}:
        return {"field": col_ref, "op": op, "value": vals}

    if op in {"contains", "not_contains", "starts_with", "ends_with"}:
        return {"field": col_ref, "op": op, "value": str(_single_value(vals))}

    if op in {"=", "!=", ">", ">=", "<", "<="}:
        if len(vals) > 1 and op in {"=", "!="}:
            return {"field": col_ref, "op": "in" if op == "=" else "not_in", "value": vals}
        return {"field": col_ref, "op": op, "value": _single_value(vals)}

    if len(vals) == 1:
        value_text = str(vals[0]).strip()
        if (
            _should_use_equals_filter(local_col, [value_text], primary_id)
            or (_numeric_type(ctype) and re.fullmatch(r"-?\d+(\.\d+)?", value_text))
        ):
            return {"field": col_ref, "op": "=", "value": vals[0]}
        return {"field": col_ref, "op": "contains", "value": vals[0]}

    return {"field": col_ref, "op": "in", "value": vals}


def _build_filter_clauses(
    filters: List[Dict[str, Any]],
    *,
    schema: Dict[str, Any],
    dataset: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for filt in filters or []:
        if not isinstance(filt, dict):
            continue
        clause = _build_filter_clause(filt, schema=schema, dataset=dataset)
        if clause is not None:
            out.append(clause)
    return out


def _freeze_for_key(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze_for_key(v)) for k, v in value.items()))
    if isinstance(value, list):
        return tuple(_freeze_for_key(v) for v in value)
    return value


def _dedupe_filter_clauses(filters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for filt in filters:
        key = (
            filt.get("field"),
            filt.get("op"),
            _freeze_for_key(filt.get("value")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(filt)
    return out


def _safe_alias(raw: Any, fallback: str) -> str:
    alias = re.sub(r"[^A-Za-z0-9_]+", "_", str(raw or "").strip())
    alias = alias.strip("_").lower()
    if not alias:
        alias = fallback
    if not re.match(r"^[A-Za-z_]", alias):
        alias = f"m_{alias}"
    return alias


def _filter_clause_to_cmp(filt: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "cmp": {
            "left": {"col": filt["field"]},
            "op": filt["op"],
            "right": filt.get("value"),
        }
    }


def _filters_to_where(filters: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    clauses = [_filter_clause_to_cmp(f) for f in filters if f.get("field") and f.get("op")]
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"and": clauses}


def _always_true_bool() -> Dict[str, Any]:
    return {"cmp": {"left": {"lit": 1}, "op": "=", "right": 1}}


def _time_bucket_expr_for_ref(
    schema: Dict[str, Any],
    dataset: str,
    field: str,
    bucket: Optional[str],
) -> Dict[str, Any]:
    if bucket not in {"day", "month", "quarter", "year"}:
        return {"col": field}

    local_col = _local_col_for_ref(schema, dataset, field)
    ctype = _column_type_for_ref(schema, dataset, field)
    cname = local_col.lower()
    textish = any(tok in ctype for tok in ("char", "text", "string"))
    dateish = cname == "date" or cname.endswith("_date") or cname.endswith("date") or cname.endswith("month")
    if textish and dateish:
        if bucket == "year":
            return {"func": "substr", "args": [{"col": field}, {"lit": 1}, {"lit": 4}]}
        if bucket == "month":
            return {
                "case": {
                    "whens": [
                        {
                            "when": {
                                "cmp": {
                                    "left": {"func": "length", "args": [{"col": field}]},
                                    "op": "=",
                                    "right": 6,
                                }
                            },
                            "then": {"func": "substr", "args": [{"col": field}, {"lit": 5}, {"lit": 2}]},
                        }
                    ],
                    "else": {"func": "substr", "args": [{"col": field}, {"lit": 6}, {"lit": 2}]},
                }
            }
    return {"func": "date_trunc", "args": [{"lit": bucket}, {"col": field}]}


def _dimension_exprs_from_intent(
    *,
    schema: Dict[str, Any],
    dataset: str,
    group_by: Any,
    time_bucket: Optional[str],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if isinstance(group_by, str):
        group_cols = [group_by]
    else:
        group_cols = list(group_by or [])

    select_items: List[Dict[str, Any]] = []
    group_exprs: List[Dict[str, Any]] = []
    for col in group_cols:
        if not _column_ref_valid(schema, dataset, col):
            continue
        ref_table = _table_for_ref(schema, dataset, col)
        local_col = _local_col_for_ref(schema, dataset, col)
        ref_primary_date = _primary_date_column(ref_table)
        bucket = time_bucket if ref_primary_date and local_col == ref_primary_date else None
        expr = _time_bucket_expr_for_ref(schema, dataset, col, bucket)
        alias = f"{_alias_for_ref(col)}_{bucket}" if bucket else _alias_for_ref(col)
        select_items.append({"expr": expr, "alias": alias})
        group_exprs.append(expr)
    return select_items, group_exprs


def _conditional_metric_expr(
    metric: Dict[str, Any],
    *,
    schema: Dict[str, Any],
    dataset: str,
) -> Optional[Dict[str, Any]]:
    agg = str(metric.get("aggregation") or "").lower()
    if agg not in {"count", "sum", "avg", "min", "max"}:
        return None

    filters = _build_filter_clauses(metric.get("filters") or [], schema=schema, dataset=dataset)
    cond = _filters_to_where(filters) or _always_true_bool()

    field = metric.get("field")
    if agg == "count":
        value_expr = {"lit": 1}
        else_expr = {"lit": 0}
        return {
            "func": "sum",
            "args": [
                {"case": {"whens": [{"when": cond, "then": value_expr}], "else": else_expr}}
            ],
        }

    if not field or not _column_ref_valid(schema, dataset, field):
        print(
            f"[DSL intent] Ignoring conditional metric with invalid field '{field}'",
            file=sys.stderr,
        )
        return None

    then_expr = {"col": field}
    else_expr = {"lit": 0} if agg == "sum" else {"lit": None}
    return {
        "func": agg,
        "args": [
            {"case": {"whens": [{"when": cond, "then": then_expr}], "else": else_expr}}
        ],
    }


def _metric_operand_expr(operand: Any, exprs_by_alias: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if isinstance(operand, (int, float)):
        return {"lit": operand}
    if isinstance(operand, str):
        key = _safe_alias(operand, "metric")
        if key in exprs_by_alias:
            return exprs_by_alias[key]
        try:
            return {"lit": float(operand)}
        except ValueError:
            return None
    if isinstance(operand, dict):
        if "alias" in operand:
            return _metric_operand_expr(operand.get("alias"), exprs_by_alias)
        if "metric" in operand:
            return _metric_operand_expr(operand.get("metric"), exprs_by_alias)
        if "literal" in operand:
            return _metric_operand_expr(operand.get("literal"), exprs_by_alias)
    return None


def _formula_metric_expr(
    metric: Dict[str, Any],
    *,
    exprs_by_alias: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    op = str(metric.get("op") or "").strip()
    if op not in {"+", "-", "*", "/"}:
        return None
    left = _metric_operand_expr(metric.get("left"), exprs_by_alias)
    right = _metric_operand_expr(metric.get("right"), exprs_by_alias)
    if left is None or right is None:
        print(
            f"[DSL intent] Ignoring formula metric with unresolved operands: {metric}",
            file=sys.stderr,
        )
        return None
    if metric.get("nullif_right"):
        right = {"func": "nullif", "args": [right, {"lit": 0}]}
    if op == "/":
        left = {"op": "*", "args": [left, {"lit": 1.0}]}
    expr: Dict[str, Any] = {"op": op, "args": [left, right]}
    scale = metric.get("scale")
    if scale is not None:
        try:
            expr = {"op": "*", "args": [expr, {"lit": float(scale)}]}
        except (TypeError, ValueError):
            pass
    return expr


def _conditional_formula_plan(
    *,
    dataset: str,
    schema: Dict[str, Any],
    common_filters: List[Dict[str, Any]],
    intent: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    conditional_metrics = intent.get("conditional_metrics") or []
    formula_metrics = intent.get("formula_metrics") or []
    if not conditional_metrics and not formula_metrics:
        return None

    select_items, group_exprs = _dimension_exprs_from_intent(
        schema=schema,
        dataset=dataset,
        group_by=intent.get("group_by") or [],
        time_bucket=intent.get("time_bucket"),
    )

    exprs_by_alias: Dict[str, Dict[str, Any]] = {}
    included_metric_count = 0
    for i, metric in enumerate(conditional_metrics):
        if not isinstance(metric, dict):
            continue
        alias = _safe_alias(metric.get("alias"), f"metric_{i + 1}")
        expr = _conditional_metric_expr(metric, schema=schema, dataset=dataset)
        if expr is None:
            continue
        exprs_by_alias[alias] = expr
        if metric.get("include", True):
            select_items.append({"expr": expr, "alias": alias})
            included_metric_count += 1

    for i, metric in enumerate(formula_metrics):
        if not isinstance(metric, dict):
            continue
        alias = _safe_alias(metric.get("alias"), f"formula_{i + 1}")
        expr = _formula_metric_expr(metric, exprs_by_alias=exprs_by_alias)
        if expr is None:
            continue
        exprs_by_alias[alias] = expr
        if metric.get("include", True):
            select_items.append({"expr": expr, "alias": alias})
            included_metric_count += 1

    if not select_items:
        return None
    if included_metric_count == 0:
        print(
            "[DSL intent] Conditional/formula intent produced no included metric outputs",
            file=sys.stderr,
        )
        return None

    where = _filters_to_where(common_filters)
    extreme = intent.get("extreme_measure_filter")
    if isinstance(extreme, dict):
        field = extreme.get("field")
        agg = str(extreme.get("agg") or "").lower()
        if agg in {"min", "max"} and isinstance(field, str) and _column_ref_valid(schema, dataset, field):
            extreme_plan = {
                "dataset": dataset,
                "metrics": [{"agg": agg, "field": field, "alias": "value"}],
                "dimensions": [],
                "filters": [],
                "limit": 1,
                "offset": 0,
            }
            extreme_where = {
                "cmp": {
                    "left": {"col": field},
                    "op": "=",
                    "right": {"scalar_subquery": {"plan": extreme_plan}},
                }
            }
            where = extreme_where if where is None else {"and": [where, extreme_where]}

    plan: Dict[str, Any] = {
        "version": "1.0",
        "dataset": dataset,
        "select": select_items,
        "where": where,
        "group_by": group_exprs,
        "order_by": [],
        "limit": intent.get("limit") or 100,
        "offset": 0,
    }

    sort_dir = intent.get("sort_direction")
    sort_col = intent.get("sort_column")
    group_cols = intent.get("group_by") or []
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    if sort_dir and sort_col:
        alias = _safe_alias(sort_col, "sort")
        if alias in exprs_by_alias:
            plan["order_by"] = [{"by": exprs_by_alias[alias], "dir": sort_dir}]
        elif _column_ref_valid(schema, dataset, sort_col) and (not group_exprs or sort_col in group_cols):
            plan["order_by"] = [{"by": sort_col, "dir": sort_dir}]
    elif sort_dir and exprs_by_alias:
        # Sort by the first included formula/metric if the user asked for highest/lowest.
        first_metric_alias = None
        for item in select_items:
            alias = item.get("alias")
            if alias in exprs_by_alias:
                first_metric_alias = alias
                break
        if first_metric_alias:
            plan["order_by"] = [{"by": exprs_by_alias[first_metric_alias], "dir": sort_dir}]

    return plan


def _metric_for_comparison_subplan(
    *,
    dataset: str,
    schema: Dict[str, Any],
    metric_aggregation: Optional[str],
    metric_field: Optional[str],
) -> Optional[Dict[str, Any]]:
    table = _table_meta(schema, dataset)
    primary_id = table.get("primary_id")
    agg = (metric_aggregation or "count").lower()
    if agg == "count":
        return {
            "agg": "count_distinct" if primary_id else "count",
            "field": primary_id or "*",
            "alias": "value",
        }
    if agg in {"sum", "avg", "min", "max"} and metric_field:
        if _column_ref_valid(schema, dataset, metric_field):
            return {"agg": agg, "field": metric_field, "alias": "value"}
    print(
        "[DSL intent] Comparison metric was invalid; using count instead",
        file=sys.stderr,
    )
    return {
        "agg": "count_distinct" if primary_id else "count",
        "field": primary_id or "*",
        "alias": "value",
    }


def _comparison_segment_subplan(
    *,
    dataset: str,
    schema: Dict[str, Any],
    common_filters: List[Dict[str, Any]],
    segment: Dict[str, Any],
    metric: Dict[str, Any],
) -> Dict[str, Any]:
    segment_filters = _build_filter_clauses(
        segment.get("filters") or [],
        schema=schema,
        dataset=dataset,
    )
    return {
        "dataset": dataset,
        "metrics": [metric],
        "filters": _dedupe_filter_clauses(list(common_filters) + segment_filters),
        "dimensions": [],
        "limit": 1,
        "offset": 0,
    }


def _comparison_plan(
    *,
    dataset: str,
    schema: Dict[str, Any],
    common_filters: List[Dict[str, Any]],
    comparison: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    op = str(comparison.get("operator") or "").lower()
    if op not in {"ratio", "difference"}:
        return None

    left = comparison.get("left") or {}
    right = comparison.get("right") or {}
    if not isinstance(left, dict) or not isinstance(right, dict):
        return None

    metric = _metric_for_comparison_subplan(
        dataset=dataset,
        schema=schema,
        metric_aggregation=comparison.get("metric_aggregation"),
        metric_field=comparison.get("metric_field"),
    )
    if metric is None:
        return None

    left_sub = _comparison_segment_subplan(
        dataset=dataset,
        schema=schema,
        common_filters=common_filters,
        segment=left,
        metric=metric,
    )
    right_sub = _comparison_segment_subplan(
        dataset=dataset,
        schema=schema,
        common_filters=common_filters,
        segment=right,
        metric=metric,
    )
    left_expr: Dict[str, Any] = {"scalar_subquery": {"plan": left_sub}}
    right_expr: Dict[str, Any] = {"scalar_subquery": {"plan": right_sub}}

    if op == "difference":
        expr: Dict[str, Any] = {"op": "-", "args": [left_expr, right_expr]}
        alias = "difference"
    else:
        denom = {"func": "nullif", "args": [right_expr, {"lit": 0}]}
        expr = {"op": "/", "args": [{"op": "*", "args": [left_expr, {"lit": 1.0}]}, denom]}
        if comparison.get("scale") == "percent":
            expr = {"op": "*", "args": [expr, {"lit": 100.0}]}
            alias = "pct"
        else:
            alias = "ratio"

    return {
        "version": "1.0",
        "dataset": dataset,
        "select": [{"expr": expr, "alias": alias}],
        "limit": 1,
        "offset": 0,
    }


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
    primary_date = _primary_date_column(table)
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
    plan["filters"].extend(
        _build_filter_clauses(intent.get("filters") or [], schema=schema, dataset=dataset)
    )

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
    plan["filters"] = _dedupe_filter_clauses(plan["filters"])

    # --- conditional aggregates + formula metrics ---
    conditional_plan = _conditional_formula_plan(
        dataset=dataset,
        schema=schema,
        common_filters=plan["filters"],
        intent=intent,
    )
    if conditional_plan is not None:
        conditional_plan = _resolve_relative_dates(conditional_plan)
        return canonicalize_query_plan(conditional_plan)

    # --- generic comparison: ratio/difference of two filtered segments ---
    comparison = intent.get("comparison")
    if isinstance(comparison, dict):
        comparison_plan = _comparison_plan(
            dataset=dataset,
            schema=schema,
            common_filters=plan["filters"],
            comparison=comparison,
        )
        if comparison_plan is not None:
            comparison_plan = _resolve_relative_dates(comparison_plan)
            return canonicalize_query_plan(comparison_plan)

    # --- ratio (% with keyword as numerator) → scalar subquery plan ---
    agg = intent.get("aggregation", "count")
    if agg == "difference":
        print(
            "[DSL intent] aggregation=difference requires comparison; using count instead",
            file=sys.stderr,
        )
        agg = "count"
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
        if agg_field and _column_ref_valid(schema, dataset, agg_field):
            plan["metrics"] = [{"agg": agg, "field": agg_field, "alias": f"{agg}_{_alias_for_ref(agg_field)}"}]
        else:
            field = primary_id or "*"
            plan["metrics"] = [{"agg": "count_distinct" if primary_id else "count", "field": field, "alias": "total"}]

    # --- dimensions (group by) ---
    group_by = intent.get("group_by") or []
    if isinstance(group_by, str):
        group_by = [group_by]
    tb = intent.get("time_bucket")
    for col in group_by:
        if not _column_ref_valid(schema, dataset, col):
            continue
        ref_table = _table_for_ref(schema, dataset, col)
        local_col = _local_col_for_ref(schema, dataset, col)
        ref_primary_date = _primary_date_column(ref_table)
        if ref_primary_date and local_col == ref_primary_date and tb:
            dim_alias = f"{_alias_for_ref(col)}_{tb}"
            plan["dimensions"].append({"field": col, "alias": dim_alias, "time_bucket": tb})
        else:
            plan["dimensions"].append({"field": col, "alias": _alias_for_ref(col)})

    # --- output columns (for list queries) ---
    if agg == "list":
        existing = {d["field"] for d in plan["dimensions"]}
        out_cols = intent.get("output_columns") or []
        if isinstance(out_cols, str):
            out_cols = [out_cols]
        for col in out_cols:
            if _column_ref_valid(schema, dataset, col) and col not in existing:
                plan["dimensions"].append({"field": col, "alias": _alias_for_ref(col)})

    # --- output projection for ranked aggregate questions ---
    requested_outputs = intent.get("output_columns") or []
    if isinstance(requested_outputs, str):
        requested_outputs = [requested_outputs]
    if agg != "list" and plan["metrics"] and plan["dimensions"] and requested_outputs:
        dim_refs = {d.get("field") for d in plan["dimensions"]}
        dim_aliases = {d.get("alias") for d in plan["dimensions"]}
        requested = set(requested_outputs)
        if requested and requested.issubset(dim_refs | dim_aliases):
            for metric in plan["metrics"]:
                metric["include"] = False

    # --- safety: multi-value IN filter without grouping → add dimension ---
    if plan["metrics"] and not plan["dimensions"]:
        for f in plan["filters"]:
            if f.get("op") == "in" and f.get("field") != primary_date:
                col = f["field"]
                plan["dimensions"].append({"field": col, "alias": _alias_for_ref(col)})
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
            local_dim = _local_col_for_ref(schema, dataset, dim_field)
            if dim_field and dim_field not in existing_filter_fields and local_dim != primary_date:
                ctype = _column_type_for_ref(schema, dataset, dim_field)
                if not _numeric_type(ctype):
                    plan["filters"].append({"field": dim_field, "op": "!=", "value": ""})
                plan["filters"].append({"field": dim_field, "op": "is_not_null", "value": True})
                print(
                    f"[DSL intent] Auto-added null/empty exclusion for group-by column '{dim_field}'",
                    file=sys.stderr,
                )

    # --- sort & limit ---
    sort_dir = intent.get("sort_direction")
    sort_col = intent.get("sort_column")
    if agg == "list" and sort_col and _column_ref_valid(schema, dataset, sort_col):
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

        evidence_plan = build_evidence_plan(question, schema, value_index=self.value_index)
        if evidence_plan is not None:
            print(f"[DSL] Built plan from evidence: {evidence_plan}", file=sys.stderr)
            return evidence_plan

        # Step 1: Retrieve similar past examples for few-shot prompting
        few_shot_prompt = None
        if self.memory:
            examples = self.memory.retrieve(question)
            if examples:
                few_shot_prompt = self.memory.format_few_shot_examples(examples)

        for attempt in range(1 + self.max_intent_retries):
            # Step 2: LLM extracts intent
            try:
                intent = extract_intent(
                    llm=self.llm,
                    question=question,
                    schema=schema,
                    temperature=self.temperature,
                    value_index=self.value_index,
                    feedback=feedback,
                    few_shot_prompt=few_shot_prompt,
                )
            except Exception as exc:
                if attempt < self.max_intent_retries:
                    feedback = (
                        f"{exc}\n"
                        "Return one complete QueryIntent JSON object with dataset, filters, "
                        "aggregation, and any comparison/conditional metrics needed."
                    )
                    retry_count += 1
                    print(
                        f"[DSL] Intent extraction failed (attempt {attempt + 1}), retrying: {exc}",
                        file=sys.stderr,
                    )
                    continue
                raise

            # Step 3: Deterministic normalization
            intent = normalize_intent(
                intent,
                schema,
                question=question,
                value_index=self.value_index,
            )

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
