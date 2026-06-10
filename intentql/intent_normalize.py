"""
intent_normalize.py — Deterministic normalization of extracted intents.

No matter how the LLM structures the intent, this module canonicalizes it
so that semantically equivalent questions always produce the same intent
structure.  This eliminates a major source of inconsistency.

Optional ``intent_id_patterns`` on a table in schema.yaml (list of regex strings)
may be used to add a primary_id filter from the user question; patterns are
application-defined — IntentQL does not ship domain-specific ID heuristics.
"""
from __future__ import annotations

import re
import sys
from typing import Any, Dict, List, Optional, Set


def normalize_intent(
    intent: Dict[str, Any],
    schema: Dict[str, Any],
    question: Optional[str] = None,
    value_index: Optional[Dict[str, Dict[str, List[str]]]] = None,
) -> Dict[str, Any]:
    """Canonicalize an extracted intent dict.

    Applies deterministic rules to resolve structural ambiguities
    the LLM might produce differently across rephrasings.
    """
    dataset = intent.get("dataset", "")
    table = _table_meta(schema, dataset)
    kso_cols = set(table.get("keyword_search_or") or [])

    intent = _absorb_keyword_filters(intent, kso_cols)
    intent = _ensure_group_by_is_list(intent)
    intent = _inject_primary_id_from_question(intent, question, schema)
    intent = _inject_single_year_filter_from_question(intent, question, schema)
    intent = _infer_time_bucket_for_trends(intent, question, schema)
    intent = _qualify_linked_column_refs(intent, schema)
    intent = _inject_known_value_filters_from_question(intent, question, schema, value_index)
    intent = _drop_relative_time_range_with_absolute_dates(intent, question, schema)
    intent = _maybe_coerce_list_for_detail_lookup(intent, question, schema)
    intent = _coerce_latest_row_intent(intent, question, schema)
    intent = _normalize_group_by_for_multi_value_filters(intent, question, schema)
    intent = _coerce_comparison_measure_intent(intent, question, schema)
    intent = _coerce_extreme_measure_filter(intent, question, schema)
    intent = _synthesize_average_difference_from_pairs(intent, question, schema)
    intent = _repair_average_difference_metrics(intent, question, schema)
    intent = _synthesize_percentage_change_from_grouped_comparison(intent, question, schema)
    intent = _repair_percentage_change_metrics(intent, question, schema)
    intent = _coerce_ranked_measure_intent(intent, question, schema)
    intent = _coerce_average_per_period_intent(intent, question, schema)
    intent = _infer_dimension_only_answer(intent, question)

    return intent


def _table_meta(schema: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    for t in schema.get("tables", []):
        if t.get("name") == dataset:
            return t
    return {}


def _column_names_set(table: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    for c in table.get("columns", []) or []:
        if isinstance(c, dict) and c.get("name"):
            out.add(c["name"])
    return out


def _schema_table_map(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        t["name"]: t
        for t in schema.get("tables", []) or []
        if isinstance(t, dict) and isinstance(t.get("name"), str)
    }


def _split_column_ref(schema: Dict[str, Any], dataset: str, ref: str) -> tuple[str, str]:
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
    return bool(table and col in _column_names_set(table))


def _column_type(table: Dict[str, Any], col: str) -> str:
    for c in table.get("columns", []) or []:
        if isinstance(c, dict) and c.get("name") == col:
            return (c.get("type") or "").lower()
    return ""


def _numeric_type(ctype: str) -> bool:
    return any(tok in ctype for tok in ("int", "numeric", "float", "double", "real", "decimal"))


def _numeric_measure_columns(table: Dict[str, Any]) -> List[str]:
    primary_id = table.get("primary_id")
    out: List[str] = []
    for c in table.get("columns", []) or []:
        if not isinstance(c, dict):
            continue
        name = c.get("name")
        if not isinstance(name, str) or name == primary_id:
            continue
        if _numeric_type((c.get("type") or "").lower()):
            out.append(name)
    return out


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


def _absorb_keyword_filters(
    intent: Dict[str, Any],
    kso_cols: Set[str],
) -> Dict[str, Any]:
    """If the LLM put the keyword as both `keyword` and a filter on a kso column,
    remove the filter and keep only the keyword.

    The keyword generates a broad OR across all kso columns.  A filter on one
    kso column is strictly narrower, so the keyword subsumes it.
    """
    keyword = intent.get("keyword")
    if not keyword or not kso_cols:
        return intent

    kw_upper = keyword.upper().strip()
    filters = intent.get("filters") or []
    kept: List[Dict[str, Any]] = []
    absorbed = False

    for f in filters:
        col = f.get("column", "")
        vals = f.get("values") or []
        if col in kso_cols and any(kw_upper in str(v).upper() for v in vals):
            absorbed = True
            print(
                f"[Normalize] Absorbed redundant filter {col}={vals} "
                f"(subsumed by keyword '{keyword}')",
                file=sys.stderr,
            )
            continue
        kept.append(f)

    if absorbed:
        intent["filters"] = kept

    return intent


def _ensure_group_by_is_list(intent: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize group_by to always be a list."""
    gb = intent.get("group_by")
    if gb is None:
        intent["group_by"] = []
    elif isinstance(gb, str):
        intent["group_by"] = [gb]
    return intent


def _normalize_group_by_for_multi_value_filters(
    intent: Dict[str, Any],
    question: Optional[str],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """If there are multi-value filters and aggregation is count,
    ensure the filtered column is in group_by.
    """
    agg = intent.get("aggregation", "count")
    if agg not in ("count", "sum", "avg"):
        return intent

    group_by = intent.get("group_by") or []
    q = (question or "").lower()
    for f in intent.get("filters") or []:
        vals = f.get("values") or []
        col = f.get("column", "")
        op = (f.get("op") or "").lower() if isinstance(f.get("op"), str) else ""
        if op == "between":
            continue
        local_col = _local_col_for_ref(schema, intent.get("dataset") or "", col)
        if _date_like_column(local_col) and not re.search(r"\b(?:by|per|each|for each)\s+(?:date|day|month|quarter|year)\b", q):
            continue
        if len(vals) > 1 and col not in group_by:
            group_by.append(col)
            print(
                f"[Normalize] Auto-added group_by '{col}' for multi-value filter",
                file=sys.stderr,
            )

    intent["group_by"] = group_by
    return intent


def _infer_dimension_only_answer(
    intent: Dict[str, Any],
    question: Optional[str],
) -> Dict[str, Any]:
    """For "which/who had most X" questions, return the ranked dimension only."""
    if not question:
        return intent
    if intent.get("aggregation") == "list":
        return intent
    group_by = intent.get("group_by") or []
    if isinstance(group_by, str):
        group_by = [group_by]
    if not group_by:
        return intent
    if intent.get("output_columns"):
        return intent
    q = question.lower()
    asks_ranked_entity = bool(
        re.search(r"\bwhich\b.*\b(?:most|least|highest|lowest|biggest|smallest|peak|recorded)\b", q)
        or re.search(r"\bwho\b.*\b(?:most|least|highest|lowest|biggest|smallest)\b", q)
        or re.search(r"\bwhat\b.*\b(?:most|least|highest|lowest|peak|recorded)\b", q)
        or re.search(r"\bwhat\b.*\bpeak\s+(?:year|month|quarter)\b", q)
    )
    if not asks_ranked_entity:
        return intent
    intent["output_columns"] = list(group_by)
    if not intent.get("limit"):
        intent["limit"] = 1
    print(
        f"[Normalize] Set output_columns={group_by} for ranked dimension answer",
        file=sys.stderr,
    )
    return intent


def _directly_linked_tables(schema: Dict[str, Any], dataset: str) -> List[str]:
    linked: List[str] = []
    for link in schema.get("links") or []:
        if not isinstance(link, dict):
            continue
        frm = link.get("from_table")
        to = link.get("to_table")
        if frm == dataset and isinstance(to, str):
            linked.append(to)
        elif to == dataset and isinstance(frm, str):
            linked.append(frm)
    return linked


def _unique_linked_ref_for_column(schema: Dict[str, Any], dataset: str, col: str) -> Optional[str]:
    if not isinstance(col, str) or not col:
        return None
    if "." in col:
        if _column_ref_valid(schema, dataset, col):
            return col
        _hinted_table, local_col = _split_column_ref(schema, dataset, col)
        matches = [
            f"{table_name}.{local_col}"
            for table_name in _directly_linked_tables(schema, dataset)
            if local_col in _column_names_set(_table_meta(schema, table_name))
        ]
        return matches[0] if len(matches) == 1 else None
    if _column_ref_valid(schema, dataset, col):
        for table_name in _directly_linked_tables(schema, dataset):
            table = _table_meta(schema, table_name)
            if col in _column_names_set(table):
                return f"{dataset}.{col}"
        return col
    matches: List[str] = []
    for table_name in _directly_linked_tables(schema, dataset):
        table = _table_meta(schema, table_name)
        if col in _column_names_set(table):
            matches.append(f"{table_name}.{col}")
    if len(matches) == 1:
        return matches[0]
    return None


def _qualify_filter_refs(filters: Any, schema: Dict[str, Any], dataset: str) -> None:
    if not isinstance(filters, list):
        return
    for filt in filters:
        if not isinstance(filt, dict):
            continue
        col = filt.get("column")
        qualified = _unique_linked_ref_for_column(schema, dataset, col)
        if qualified and qualified != col:
            filt["column"] = qualified
            print(
                f"[Normalize] Qualified linked filter {col!r} -> {qualified!r}",
                file=sys.stderr,
            )


def _qualify_ref_list(refs: Any, schema: Dict[str, Any], dataset: str, *, label: str) -> Any:
    if isinstance(refs, str):
        refs_list = [refs]
        was_str = True
    elif isinstance(refs, list):
        refs_list = refs
        was_str = False
    else:
        return refs
    out: List[Any] = []
    for ref in refs_list:
        qualified = _unique_linked_ref_for_column(schema, dataset, ref)
        if qualified and qualified != ref:
            print(
                f"[Normalize] Qualified linked {label} {ref!r} -> {qualified!r}",
                file=sys.stderr,
            )
            out.append(qualified)
        else:
            out.append(ref)
    return out[0] if was_str and out else out


def _qualify_linked_column_refs(intent: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """Qualify unqualified references that belong to a unique directly linked table."""
    dataset = intent.get("dataset") or ""
    if not dataset:
        return intent

    _qualify_filter_refs(intent.get("filters") or [], schema, dataset)
    intent["group_by"] = _qualify_ref_list(intent.get("group_by") or [], schema, dataset, label="group_by")
    intent["output_columns"] = _qualify_ref_list(
        intent.get("output_columns") or [],
        schema,
        dataset,
        label="output_column",
    )
    sort_col = intent.get("sort_column")
    qualified_sort = _unique_linked_ref_for_column(schema, dataset, sort_col)
    if qualified_sort and qualified_sort != sort_col:
        intent["sort_column"] = qualified_sort
        print(
            f"[Normalize] Qualified linked sort_column {sort_col!r} -> {qualified_sort!r}",
            file=sys.stderr,
        )

    comparison = intent.get("comparison")
    if isinstance(comparison, dict):
        for side in ("left", "right"):
            segment = comparison.get(side)
            if isinstance(segment, dict):
                _qualify_filter_refs(segment.get("filters") or [], schema, dataset)
        metric_field = comparison.get("metric_field")
        qualified_metric = _unique_linked_ref_for_column(schema, dataset, metric_field)
        if qualified_metric and qualified_metric != metric_field:
            comparison["metric_field"] = qualified_metric

    for metric in intent.get("conditional_metrics") or []:
        if not isinstance(metric, dict):
            continue
        _qualify_filter_refs(metric.get("filters") or [], schema, dataset)
        field = metric.get("field")
        qualified_field = _unique_linked_ref_for_column(schema, dataset, field)
        if qualified_field and qualified_field != field:
            metric["field"] = qualified_field

    return intent


def _absolute_date_filter_present(intent: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    dataset = intent.get("dataset") or ""
    for filt in intent.get("filters") or []:
        if not isinstance(filt, dict):
            continue
        col = filt.get("column")
        if not isinstance(col, str):
            continue
        local_col = _local_col_for_ref(schema, dataset, col)
        table = _table_for_ref(schema, dataset, col)
        if not _date_like_column(local_col) and _primary_date_column(table) != local_col:
            continue
        vals = filt.get("values") or []
        if not isinstance(vals, list):
            vals = [vals]
        if any(re.fullmatch(r"(?:19|20)\d{2}(?:[-/]?\d{2}){0,2}", str(v).strip()) for v in vals):
            return True
    return False


def _drop_relative_time_range_with_absolute_dates(
    intent: Dict[str, Any],
    question: Optional[str],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Prefer explicit year/month filters over relative enums like last_year."""
    if not intent.get("time_range"):
        return intent
    if not _absolute_date_filter_present(intent, schema):
        return intent
    old = intent.get("time_range")
    intent["time_range"] = None
    print(
        f"[Normalize] Dropped time_range={old!r}; explicit absolute date filter is present.",
        file=sys.stderr,
    )
    return intent


def _intent_filter_columns(intent: Dict[str, Any]) -> Set[str]:
    cols: Set[str] = set()
    for filt in intent.get("filters") or []:
        if isinstance(filt, dict) and isinstance(filt.get("column"), str):
            cols.add(filt["column"])
    return cols


def _question_mentions_value(question: str, value: str) -> bool:
    if not value:
        return False
    pattern = r"(?<![A-Za-z0-9_])" + re.escape(value.lower()) + r"(?![A-Za-z0-9_])"
    return bool(re.search(pattern, question.lower()))


def _inject_known_value_filters_from_question(
    intent: Dict[str, Any],
    question: Optional[str],
    schema: Dict[str, Any],
    value_index: Optional[Dict[str, Dict[str, List[str]]]],
) -> Dict[str, Any]:
    if not question or not value_index:
        return intent
    dataset = intent.get("dataset") or ""
    candidate_tables = [dataset] + _directly_linked_tables(schema, dataset)
    existing_cols = _intent_filter_columns(intent)
    for table_name in candidate_tables:
        for col, values in (value_index.get(table_name) or {}).items():
            ref = col if table_name == dataset else f"{table_name}.{col}"
            if ref in existing_cols or col in existing_cols:
                continue
            hits = [v for v in values if _question_mentions_value(question, str(v))]
            if not hits:
                continue
            vals = hits if len(hits) > 1 else [hits[0]]
            intent.setdefault("filters", []).append({"column": ref, "op": "in" if len(vals) > 1 else "=", "values": vals})
            existing_cols.add(ref)
            print(
                f"[Normalize] Injected known-value filter {ref}={vals} from question text",
                file=sys.stderr,
            )
    return intent


def _find_numeric_measure_for_question(
    intent: Dict[str, Any],
    question: Optional[str],
    schema: Dict[str, Any],
) -> Optional[str]:
    dataset = intent.get("dataset") or ""
    table = _table_meta(schema, dataset)
    agg_field = intent.get("aggregation_field")
    if isinstance(agg_field, str) and _column_ref_valid(schema, dataset, agg_field):
        return agg_field

    q = (question or "").lower()
    measures = _numeric_measure_columns(table)
    if len(measures) == 1:
        return measures[0]
    for col in measures:
        token = col.lower().replace("_", " ")
        if token and token in q:
            return col
    return measures[0] if measures else None


def _coerce_extreme_measure_filter(
    intent: Dict[str, Any],
    question: Optional[str],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    if not question:
        return intent
    q = question.lower()
    measure = _find_numeric_measure_for_question(intent, question, schema)
    if not measure:
        return intent
    measure_text = re.escape(measure.lower().replace("_", " "))
    amount_pattern = (
        rf"\b(?:least|lowest|minimum|smallest|most|highest|maximum|largest)\s+"
        rf"(?:amount\s+of\s+)?{measure_text}\b"
    )
    with_amount_pattern = (
        rf"\bwith\s+the\s+"
        rf"(?:least|lowest|minimum|smallest|most|highest|maximum|largest)\s+"
        rf"(?:amount\s+of\s+)?{measure_text}\b"
    )
    if not (re.search(amount_pattern, q) or re.search(with_amount_pattern, q)):
        return intent
    agg = None
    if re.search(r"\b(?:least|lowest|minimum|smallest)\b", q):
        agg = "min"
    elif re.search(r"\b(?:most|highest|maximum|largest)\b", q):
        agg = "max"
    if not agg:
        return intent
    intent["extreme_measure_filter"] = {"field": measure, "agg": agg}
    print(
        f"[Normalize] Added extreme measure filter {measure} = global {agg}({measure})",
        file=sys.stderr,
    )
    return intent


def _question_sort_direction(question: str) -> Optional[str]:
    q = question.lower()
    if re.search(r"\b(?:least|lowest|smallest|minimum|fewest)\b", q):
        return "asc"
    if re.search(r"\b(?:most|highest|largest|biggest|maximum|peak|recorded|top)\b", q):
        return "desc"
    return None


def _coerce_ranked_measure_intent(
    intent: Dict[str, Any],
    question: Optional[str],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Turn mistaken list/max rows into aggregate rankings by a numeric measure."""
    if not question:
        return intent
    q = question.lower()
    if not re.search(r"\b(?:which|who|what)\b", q):
        return intent
    if not re.search(r"\b(?:most|least|highest|lowest|biggest|smallest|peak|recorded|top|minimum|maximum)\b", q):
        return intent

    group_by = intent.get("group_by") or []
    if isinstance(group_by, str):
        group_by = [group_by]
    if not group_by:
        return intent

    measure = _find_numeric_measure_for_question(intent, question, schema)
    if not measure:
        return intent

    agg = intent.get("aggregation") or "count"
    if agg not in {"list", "min", "max", "count"}:
        return intent

    metric_agg = "avg" if re.search(r"\b(?:average|avg|mean)\b", q) else "sum"
    intent["aggregation"] = metric_agg
    intent["aggregation_field"] = measure
    if not intent.get("sort_direction"):
        intent["sort_direction"] = _question_sort_direction(question) or "desc"
    if not intent.get("limit"):
        intent["limit"] = 1
    intent["output_columns"] = list(group_by)
    if intent.get("sort_column") in group_by or intent.get("sort_column") == measure:
        intent["sort_column"] = None
    print(
        f"[Normalize] Coerced ranked measure question to {metric_agg}({measure}) grouped by {group_by}",
        file=sys.stderr,
    )
    return intent


def _coerce_comparison_measure_intent(
    intent: Dict[str, Any],
    question: Optional[str],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    comparison = intent.get("comparison")
    if not isinstance(comparison, dict):
        left = intent.get("left")
        right = intent.get("right")
        if intent.get("aggregation") in {"ratio", "difference"} and isinstance(left, dict) and isinstance(right, dict):
            comparison = {
                "operator": intent.get("aggregation"),
                "left": left,
                "right": right,
            }
            intent["comparison"] = comparison
        else:
            return intent

    q = (question or "").lower()
    if comparison.get("operator") == "ratio":
        comparison["scale"] = "percent" if re.search(r"\b(?:percent|percentage)\b|%", q) else "raw"
    elif not comparison.get("scale") and intent.get("scale") in {"raw", "percent"}:
        comparison["scale"] = intent.get("scale")

    dataset = intent.get("dataset") or ""
    metric_field = comparison.get("metric_field")
    if not (isinstance(metric_field, str) and _column_ref_valid(schema, dataset, metric_field)):
        measure = _find_numeric_measure_for_question(intent, question, schema)
        if measure:
            comparison["metric_field"] = measure
            metric_field = measure

    if metric_field and not comparison.get("metric_aggregation"):
        if re.search(r"\b(?:average|avg|mean)\b", q):
            metric_agg = "avg"
        elif re.search(r"\b(?:minimum|min|least individual|lowest individual)\b", q):
            metric_agg = "min"
        elif re.search(r"\b(?:maximum|max|highest individual)\b", q):
            metric_agg = "max"
        else:
            metric_agg = "sum"
        comparison["metric_aggregation"] = metric_agg
        print(
            f"[Normalize] Set comparison metric to {metric_agg}({metric_field})",
            file=sys.stderr,
        )
    return intent


def _metric_has_filter(metric: Dict[str, Any], column: str) -> bool:
    for filt in metric.get("filters") or []:
        if isinstance(filt, dict) and filt.get("column") == column:
            return True
    return False


def _alias_category_year(alias: Any) -> tuple[Optional[str], Optional[str]]:
    if not isinstance(alias, str):
        return None, None
    m = re.match(r"^([A-Za-z][A-Za-z0-9]*)[_\-\s]+((?:19|20)\d{2})$", alias.strip())
    if not m:
        return None, None
    return m.group(1), m.group(2)


def _repair_percentage_change_metrics(
    intent: Dict[str, Any],
    question: Optional[str],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Repair common old/new percentage-change metric shapes.

    Small models often emit metrics like SME_2012 and SME_2013 but then divide
    2012/2013. For "percentage increase/decrease", the canonical formula is
    (new - old) / old * 100. If category aliases map cleanly to a grouped
    category column, make each category its own scalar output metric.
    """
    if not question:
        return intent
    q = question.lower()
    if not re.search(r"\bpercentage\s+(?:increase|decrease|change)|\bpercent\s+(?:increase|decrease|change)|%", q):
        return intent
    conditional_metrics = intent.get("conditional_metrics") or []
    formula_metrics = intent.get("formula_metrics") or []
    if not conditional_metrics or not formula_metrics:
        return intent

    group_by = intent.get("group_by") or []
    if isinstance(group_by, str):
        group_by = [group_by]
    category_col = None
    dataset = intent.get("dataset") or ""
    for col in group_by:
        local = _local_col_for_ref(schema, dataset, col)
        if local and not _date_like_column(local):
            category_col = col
            break
    if not category_col:
        return intent

    by_category: Dict[str, Dict[str, str]] = {}
    measure_names = {
        c.lower()
        for c in _numeric_measure_columns(_table_meta(schema, dataset))
    }
    for metric in conditional_metrics:
        if not isinstance(metric, dict):
            continue
        category, year = _alias_category_year(metric.get("alias"))
        if not category or not year:
            continue
        if category.lower() in measure_names:
            continue
        by_category.setdefault(category, {})[year] = str(metric.get("alias"))
        if not _metric_has_filter(metric, category_col):
            metric.setdefault("filters", []).append(
                {"column": category_col, "op": "=", "values": [category]}
            )
            print(
                f"[Normalize] Added category filter {category_col}={category} for metric {metric.get('alias')}",
                file=sys.stderr,
            )

    if not by_category:
        return intent

    repaired: List[Dict[str, Any]] = []
    used_categories: Set[str] = set()
    for formula in formula_metrics:
        if not isinstance(formula, dict):
            continue
        alias = str(formula.get("alias") or "")
        category = alias.split("_", 1)[0] if "_" in alias else ""
        if category not in by_category or category in used_categories:
            continue
        years = sorted(by_category[category])
        if len(years) < 2:
            continue
        old_alias = by_category[category][years[0]]
        new_alias = by_category[category][years[-1]]
        diff_alias = f"{category}_change"
        repaired.append(
            {
                "alias": diff_alias,
                "op": "-",
                "left": new_alias,
                "right": old_alias,
                "include": False,
            }
        )
        repaired.append(
            {
                "alias": alias or f"{category}_pct_change",
                "op": "/",
                "left": diff_alias,
                "right": old_alias,
                "nullif_right": True,
                "scale": 100.0,
                "include": True,
            }
        )
        used_categories.add(category)

    if repaired:
        intent["formula_metrics"] = repaired
        intent["group_by"] = []
        intent["output_columns"] = []
        intent["sort_column"] = None
        intent["sort_direction"] = None
        intent["limit"] = 1
        print(
            f"[Normalize] Repaired percentage-change formulas for categories {sorted(used_categories)}",
            file=sys.stderr,
        )
    return intent


def _category_alias_map_from_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        alias = metric.get("alias")
        if not isinstance(alias, str) or "_" not in alias:
            continue
        category = alias.split("_", 1)[0]
        if category and category.upper() == category:
            out.setdefault(category, alias)
    return out


def _question_category_pairs(question: str) -> List[tuple[str, str]]:
    pairs: List[tuple[str, str]] = []
    for left, right in re.findall(r"\b([A-Z][A-Z0-9]{1,})\b\s+and\s+\b([A-Z][A-Z0-9]{1,})\b", question):
        pairs.append((left, right))
    return pairs


def _question_upper_tokens(question: str) -> List[str]:
    out: List[str] = []
    for token in re.findall(r"\b[A-Z][A-Z0-9]{1,}\b", question or ""):
        if token not in out:
            out.append(token)
    return out


def _ordered_values_from_question(question: str, allowed: List[str]) -> List[str]:
    if not allowed:
        return []
    allowed_by_lower = {str(v).lower(): str(v) for v in allowed}
    ordered: List[str] = []
    for token in _question_upper_tokens(question):
        val = allowed_by_lower.get(token.lower())
        if val is not None and val not in ordered:
            ordered.append(val)
    for val in allowed:
        sval = str(val)
        if sval not in ordered:
            ordered.append(sval)
    return ordered


def _ordered_categories_from_pairs(pairs: List[tuple[str, str]]) -> List[str]:
    out: List[str] = []
    for left, right in pairs:
        for val in (left, right):
            if val not in out:
                out.append(val)
    return out


def _non_date_text_filter_values(
    intent: Dict[str, Any],
    question: str,
    schema: Dict[str, Any],
) -> tuple[Optional[str], List[str]]:
    """Find a categorical column constrained by multiple named values."""
    dataset = intent.get("dataset") or ""
    best_col = None
    best_vals: List[str] = []

    for filt in intent.get("filters") or []:
        if not isinstance(filt, dict):
            continue
        col = filt.get("column")
        if not isinstance(col, str):
            continue
        local = _local_col_for_ref(schema, dataset, col)
        if not local or _date_like_column(local):
            continue
        vals = filt.get("values") or []
        if not isinstance(vals, list):
            vals = [vals]
        text_vals = [str(v) for v in vals if str(v)]
        if len(text_vals) < 2:
            continue
        table = _table_for_ref(schema, dataset, col)
        ctype = _column_type(table, local)
        if _numeric_type(ctype):
            continue
        ordered = _ordered_values_from_question(question, text_vals)
        if len(ordered) > len(best_vals):
            best_col = col
            best_vals = ordered

    return best_col, best_vals


def _category_column_and_values_from_intent(
    intent: Dict[str, Any],
    question: str,
    schema: Dict[str, Any],
) -> tuple[Optional[str], List[str]]:
    category_col, category_vals = _non_date_text_filter_values(intent, question, schema)
    if category_col and category_vals:
        return category_col, category_vals

    dataset = intent.get("dataset") or ""
    group_by = intent.get("group_by") or []
    if isinstance(group_by, str):
        group_by = [group_by]
    pair_vals = _ordered_categories_from_pairs(_question_category_pairs(question))
    upper_vals = _question_upper_tokens(question)
    for col in group_by:
        if not isinstance(col, str):
            continue
        local = _local_col_for_ref(schema, dataset, col)
        if not local or _date_like_column(local):
            continue
        table = _table_for_ref(schema, dataset, col)
        ctype = _column_type(table, local)
        if _numeric_type(ctype):
            continue
        vals = pair_vals or upper_vals
        return col, vals

    return None, []


def _metric_alias_for_category(category: str, suffix: str) -> str:
    raw = re.sub(r"[^A-Za-z0-9_]+", "_", str(category)).strip("_")
    if not raw:
        raw = "category"
    return f"{raw}_{suffix}"


def _synthesize_average_difference_from_pairs(
    intent: Dict[str, Any],
    question: Optional[str],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Build category average-difference formulas from question pairs.

    This covers a broad pattern: "difference in average X between A and B, B
    and C". The compiler derives category filters, shared denominator, and
    output formulas instead of relying on the model to spell out every metric.
    """
    if not question:
        return intent
    q = question.lower()
    if not re.search(r"\b(?:average|avg|mean)\b", q) or "between" not in q:
        return intent
    pairs = _question_category_pairs(question)
    if not pairs:
        return intent

    category_col, category_vals = _category_column_and_values_from_intent(intent, question, schema)
    if not category_col:
        return intent
    ordered_from_pairs = _ordered_categories_from_pairs(pairs)
    if category_vals:
        allowed = {v.lower(): v for v in category_vals}
        categories = [allowed.get(v.lower(), v) for v in ordered_from_pairs if v.lower() in allowed]
        for val in category_vals:
            if val not in categories:
                categories.append(val)
    else:
        categories = ordered_from_pairs
    if len(categories) < 2:
        return intent

    measure = _find_numeric_measure_for_question(intent, question, schema)
    if not measure:
        return intent

    conditionals: List[Dict[str, Any]] = [
        {
            "alias": "common_count",
            "aggregation": "count",
            "filters": [],
            "include": False,
        }
    ]
    formulas: List[Dict[str, Any]] = []
    avg_alias_by_category: Dict[str, str] = {}
    for category in categories:
        sum_alias = _metric_alias_for_category(category, "sum")
        avg_alias = _metric_alias_for_category(category, "avg")
        avg_alias_by_category[category.lower()] = avg_alias
        conditionals.append(
            {
                "alias": sum_alias,
                "aggregation": "sum",
                "field": measure,
                "filters": [{"column": category_col, "op": "=", "values": [category]}],
                "include": False,
            }
        )
        formulas.append(
            {
                "alias": avg_alias,
                "op": "/",
                "left": sum_alias,
                "right": "common_count",
                "nullif_right": True,
                "include": False,
            }
        )

    for left, right in pairs:
        left_alias = avg_alias_by_category.get(left.lower())
        right_alias = avg_alias_by_category.get(right.lower())
        if not left_alias or not right_alias:
            continue
        formulas.append(
            {
                "alias": f"{left}_minus_{right}",
                "op": "-",
                "left": left_alias,
                "right": right_alias,
                "include": True,
            }
        )

    if not any(f.get("include") for f in formulas):
        return intent

    intent["conditional_metrics"] = conditionals
    intent["formula_metrics"] = formulas
    intent["comparison"] = None
    intent["aggregation"] = "difference"
    intent["group_by"] = []
    intent["output_columns"] = []
    intent["sort_column"] = None
    intent["sort_direction"] = None
    intent["limit"] = 1
    print(
        f"[Normalize] Synthesized average-difference formulas for {category_col} pairs {pairs}",
        file=sys.stderr,
    )
    return intent


def _years_from_question(question: str) -> List[str]:
    out: List[str] = []
    for year in re.findall(r"\b(?:19|20)\d{2}\b", question or ""):
        if year not in out:
            out.append(year)
    return out


def _date_field_for_year_filters(intent: Dict[str, Any], schema: Dict[str, Any]) -> Optional[str]:
    dataset = intent.get("dataset") or ""
    table = _table_meta(schema, dataset)
    primary_date = _primary_date_column(table)
    if primary_date:
        return primary_date
    for filt in intent.get("filters") or []:
        if not isinstance(filt, dict):
            continue
        col = filt.get("column")
        if not isinstance(col, str):
            continue
        local = _local_col_for_ref(schema, dataset, col)
        ref_table = _table_for_ref(schema, dataset, col)
        if _date_like_column(local) or _primary_date_column(ref_table) == local:
            return col
    return None


def _evidence_text(question: str) -> str:
    m = re.search(r"\bEvidence\s*:\s*(.*)$", question or "", re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def _text_mentions_filter(filt: Dict[str, Any], text: str, schema: Dict[str, Any], dataset: str) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    col = filt.get("column")
    if isinstance(col, str):
        local = _local_col_for_ref(schema, dataset, col)
        if local and re.search(r"(?<![a-z0-9_])" + re.escape(local.lower()) + r"(?![a-z0-9_])", text_lower):
            return True
    vals = filt.get("values") or []
    if not isinstance(vals, list):
        vals = [vals]
    return any(_question_mentions_value(text, str(v)) for v in vals)


def _prune_common_filters_absent_from_formula_evidence(
    intent: Dict[str, Any],
    question: str,
    schema: Dict[str, Any],
    *,
    category_col: str,
    date_field: str,
) -> None:
    """Let an explicit evidence block define which common filters belong to a formula."""
    evidence = _evidence_text(question)
    if not evidence:
        return
    if not re.search(r"\b(?:percentage|percent|increase|decrease|change)\b", evidence, re.IGNORECASE):
        return

    dataset = intent.get("dataset") or ""
    kept: List[Dict[str, Any]] = []
    dropped: List[str] = []
    for filt in intent.get("filters") or []:
        if not isinstance(filt, dict):
            kept.append(filt)
            continue
        col = filt.get("column")
        if col in {category_col, date_field} or _text_mentions_filter(filt, evidence, schema, dataset):
            kept.append(filt)
            continue
        vals = filt.get("values") or []
        dropped.append(f"{col}={vals}")

    if dropped:
        intent["filters"] = kept
        print(
            f"[Normalize] Pruned evidence-absent common formula filters: {dropped}",
            file=sys.stderr,
        )


def _synthesize_percentage_change_from_grouped_comparison(
    intent: Dict[str, Any],
    question: Optional[str],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Build per-category old/new percentage-change formulas.

    The generic shape is "percentage change/increase from YEAR1 to YEAR2 for
    A, B, C".  The compiler chooses the category column and numeric measure
    from normalized filters/schema and emits one scalar percentage per category.
    """
    if not question:
        return intent
    q = question.lower()
    if not re.search(r"\b(?:percentage|percent)\s+(?:increase|decrease|change)\b|%", q):
        return intent
    years = _years_from_question(question)
    if len(years) < 2:
        return intent
    ordered_years = sorted(years)
    old_year, new_year = ordered_years[0], ordered_years[-1]

    category_col, category_vals = _category_column_and_values_from_intent(intent, question, schema)
    if not category_col or len(category_vals) < 2:
        return intent
    categories = _ordered_values_from_question(question, category_vals)

    measure = None
    comparison = intent.get("comparison")
    if isinstance(comparison, dict) and isinstance(comparison.get("metric_field"), str):
        measure = comparison.get("metric_field")
    if not measure:
        measure = _find_numeric_measure_for_question(intent, question, schema)
    if not measure:
        return intent

    date_field = _date_field_for_year_filters(intent, schema)
    if not date_field:
        return intent

    metric_agg = "sum"
    if re.search(r"\b(?:average|avg|mean)\b", q):
        metric_agg = "avg"
    elif isinstance(comparison, dict) and comparison.get("metric_aggregation") in {"sum", "avg", "min", "max"}:
        metric_agg = comparison["metric_aggregation"]
    elif isinstance(comparison, dict) and comparison.get("metric_aggregation") == "count":
        if not re.search(r"\b(?:count|number|how many)\b", q):
            metric_agg = "sum"
        else:
            metric_agg = "count"

    conditionals: List[Dict[str, Any]] = []
    formulas: List[Dict[str, Any]] = []
    for category in categories:
        old_alias = _metric_alias_for_category(category, old_year)
        new_alias = _metric_alias_for_category(category, new_year)
        conditionals.append(
            {
                "alias": old_alias,
                "aggregation": metric_agg,
                "field": measure,
                "filters": [
                    {"column": category_col, "op": "=", "values": [category]},
                    {"column": date_field, "op": "starts_with", "values": [old_year]},
                ],
                "include": False,
            }
        )
        conditionals.append(
            {
                "alias": new_alias,
                "aggregation": metric_agg,
                "field": measure,
                "filters": [
                    {"column": category_col, "op": "=", "values": [category]},
                    {"column": date_field, "op": "starts_with", "values": [new_year]},
                ],
                "include": False,
            }
        )
        diff_alias = _metric_alias_for_category(category, "change")
        formulas.append(
            {
                "alias": diff_alias,
                "op": "-",
                "left": new_alias,
                "right": old_alias,
                "include": False,
            }
        )
        formulas.append(
            {
                "alias": _metric_alias_for_category(category, "pct_change"),
                "op": "/",
                "left": diff_alias,
                "right": old_alias,
                "nullif_right": True,
                "scale": 100.0,
                "include": True,
            }
        )

    intent["conditional_metrics"] = conditionals
    intent["formula_metrics"] = formulas
    intent["comparison"] = None
    intent["aggregation"] = "ratio"
    intent["group_by"] = []
    intent["output_columns"] = []
    intent["sort_column"] = None
    intent["sort_direction"] = None
    intent["limit"] = 1
    _prune_common_filters_absent_from_formula_evidence(
        intent,
        question,
        schema,
        category_col=category_col,
        date_field=date_field,
    )
    print(
        f"[Normalize] Synthesized percentage-change formulas for {category_col} over {old_year}->{new_year}",
        file=sys.stderr,
    )
    return intent


def _repair_average_difference_metrics(
    intent: Dict[str, Any],
    question: Optional[str],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    if not question:
        return intent
    q = question.lower()
    if "average" not in q or "between" not in q:
        return intent
    conditional_metrics = intent.get("conditional_metrics") or []
    formula_metrics = intent.get("formula_metrics") or []
    if not conditional_metrics or not formula_metrics:
        return intent

    avg_metrics = [
        m for m in conditional_metrics
        if isinstance(m, dict) and str(m.get("aggregation") or "").lower() == "avg"
    ]
    if not avg_metrics:
        return intent

    denominator_alias = "common_count"
    rewritten_conditionals: List[Dict[str, Any]] = [
        {
            "alias": denominator_alias,
            "aggregation": "count",
            "filters": [],
            "include": False,
        }
    ]
    avg_aliases: Set[str] = set()
    avg_formulas: List[Dict[str, Any]] = []
    for metric in conditional_metrics:
        if not isinstance(metric, dict):
            continue
        alias = str(metric.get("alias") or "")
        if str(metric.get("aggregation") or "").lower() != "avg":
            rewritten_conditionals.append(metric)
            continue
        sum_alias = f"{alias}_sum"
        rewritten = dict(metric)
        rewritten["alias"] = sum_alias
        rewritten["aggregation"] = "sum"
        rewritten["include"] = False
        rewritten_conditionals.append(rewritten)
        avg_aliases.add(alias)
        avg_formulas.append(
            {
                "alias": alias,
                "op": "/",
                "left": sum_alias,
                "right": denominator_alias,
                "nullif_right": True,
                "include": False,
            }
        )

    category_aliases = _category_alias_map_from_metrics(avg_metrics)
    existing_formula_aliases = {
        str(m.get("alias") or "").lower()
        for m in formula_metrics
        if isinstance(m, dict)
    }
    extra_pair_formulas: List[Dict[str, Any]] = []
    for left, right in _question_category_pairs(question):
        left_alias = category_aliases.get(left)
        right_alias = category_aliases.get(right)
        if not left_alias or not right_alias:
            continue
        alias = f"{left}_minus_{right}"
        if alias.lower() in existing_formula_aliases:
            continue
        extra_pair_formulas.append(
            {
                "alias": alias,
                "op": "-",
                "left": left_alias,
                "right": right_alias,
                "include": True,
            }
        )

    intent["conditional_metrics"] = rewritten_conditionals
    intent["formula_metrics"] = avg_formulas + list(formula_metrics) + extra_pair_formulas
    intent["group_by"] = []
    intent["output_columns"] = []
    intent["sort_column"] = None
    intent["sort_direction"] = None
    intent["limit"] = 1
    print(
        "[Normalize] Repaired average-difference metrics to use a shared denominator",
        file=sys.stderr,
    )
    return intent


def _coerce_average_per_period_intent(
    intent: Dict[str, Any],
    question: Optional[str],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Represent 'average monthly X for a year' as an aggregate divided by period count."""
    if not question:
        return intent
    if intent.get("conditional_metrics") or intent.get("formula_metrics"):
        return intent
    q = question.lower()
    divisor = None
    label = None
    if re.search(r"\baverage\s+monthly\b|\bmonthly\s+average\b", q) and re.search(r"\b(?:19|20)\d{2}\b", q):
        divisor = 12
        label = "monthly"
    elif re.search(r"\baverage\s+quarterly\b|\bquarterly\s+average\b", q) and re.search(r"\b(?:19|20)\d{2}\b", q):
        divisor = 4
        label = "quarterly"
    if divisor is None:
        return intent

    measure = _find_numeric_measure_for_question(intent, question, schema)
    if not measure:
        return intent

    base_alias = f"avg_{measure}"
    final_alias = f"avg_{label}_{measure}"
    intent["conditional_metrics"] = [
        {
            "alias": base_alias,
            "aggregation": "avg",
            "field": measure,
            "filters": [],
            "include": False,
        }
    ]
    intent["formula_metrics"] = [
        {
            "alias": final_alias,
            "op": "/",
            "left": base_alias,
            "right": divisor,
            "nullif_right": True,
            "scale": None,
            "include": True,
        }
    ]
    intent["group_by"] = []
    intent["output_columns"] = []
    intent["sort_direction"] = None
    intent["sort_column"] = None
    print(
        f"[Normalize] Coerced average-per-{label} question to avg({measure})/{divisor}",
        file=sys.stderr,
    )
    return intent


# --- Phase 1: optional ID hints from question, trend bucketing ----------------


def _compiled_intent_id_patterns(table: Dict[str, Any]) -> List[re.Pattern]:
    """Optional per-table regex list from schema (`intent_id_patterns`). Domain-agnostic."""
    raw = table.get("intent_id_patterns") or []
    if not isinstance(raw, list):
        return []
    out: List[re.Pattern] = []
    for pat in raw:
        if not isinstance(pat, str) or not pat.strip():
            continue
        try:
            out.append(re.compile(pat, re.IGNORECASE))
        except re.error:
            continue
    return out


def _match_value_from_regex(m: re.Match) -> str:
    if m.lastindex and m.lastindex >= 1:
        return m.group(1).strip()
    return m.group(0).strip()


def _extract_id_candidates_from_question(text: str, patterns: List[re.Pattern]) -> List[str]:
    if not text or not patterns:
        return []
    out: List[str] = []
    for pat in patterns:
        for m in pat.finditer(text):
            out.append(_match_value_from_regex(m))
    return out


def _inject_primary_id_from_question(
    intent: Dict[str, Any],
    question: Optional[str],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """If schema declares ``intent_id_patterns`` for this table, add a primary_id filter from the first match."""
    if not question:
        return intent
    dataset = intent.get("dataset") or ""
    table = _table_meta(schema, dataset)
    pid = table.get("primary_id")
    if not pid:
        return intent
    patterns = _compiled_intent_id_patterns(table)
    if not patterns:
        return intent
    existing_cols = {f.get("column") for f in intent.get("filters") or []}
    if pid in existing_cols:
        return intent
    for cand in _extract_id_candidates_from_question(question, patterns):
        intent.setdefault("filters", []).append({"column": pid, "values": [cand]})
        print(
            f"[Normalize] Injected primary_id filter {pid}={cand} from question text (intent_id_patterns)",
            file=sys.stderr,
        )
        break
    return intent


def _has_filter_on_column(intent: Dict[str, Any], col: str) -> bool:
    for filt in intent.get("filters") or []:
        ref = filt.get("column")
        if ref == col or (isinstance(ref, str) and ref.endswith(f".{col}")):
            return True
    return False


def _inject_single_year_filter_from_question(
    intent: Dict[str, Any],
    question: Optional[str],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Add a simple date-prefix filter when a question names exactly one year."""
    if not question:
        return intent
    years = sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", question)))
    if len(years) != 1:
        return intent
    dataset = intent.get("dataset") or ""
    table = _table_meta(schema, dataset)
    primary_date = _primary_date_column(table)
    if not primary_date or _has_filter_on_column(intent, primary_date):
        return intent
    year = years[0]
    intent.setdefault("filters", []).append(
        {"column": primary_date, "op": "starts_with", "values": [year]}
    )
    print(
        f"[Normalize] Injected {primary_date} starts_with {year} from question year",
        file=sys.stderr,
    )
    return intent


def _infer_time_bucket_for_trends(
    intent: Dict[str, Any],
    question: Optional[str],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """When the user asks for a trend / over time and groups by the date column, set time_bucket."""
    q = (question or "").lower()
    dataset = intent.get("dataset") or ""
    table = _table_meta(schema, dataset)
    pd = _primary_date_column(table)
    if not pd:
        return intent
    wants_year = bool(re.search(r"\b(?:which|what)\s+year\b|\bby\s+year\b|\byearly\b|\byear\s+recorded\b", q))
    wants_month = bool(re.search(r"\b(?:which|what)\s+month\b|\bpeak\s+month\b|\bby\s+month\b|\bmonthly\b", q))
    wants_quarter = bool(re.search(r"\b(?:which|what)\s+quarter\b|\bby\s+quarter\b|\bquarterly\b", q))
    trendish = any(
        w in q
        for w in (
            "trend",
            "over time",
            "over the",
            "by month",
            "by year",
            "by quarter",
            "monthly",
            "yearly",
            "quarterly",
        )
    ) or wants_year or wants_month or wants_quarter
    if not trendish:
        return intent
    group_by = intent.get("group_by") or []
    if pd not in group_by:
        group_by.append(pd)
        intent["group_by"] = group_by
        print(
            f"[Normalize] Added group_by {pd} for date-part question",
            file=sys.stderr,
        )
    if intent.get("time_bucket"):
        return intent

    if wants_year or ("year" in q and "month" not in q):
        bucket = "year"
    elif wants_quarter or "quarter" in q:
        bucket = "quarter"
    elif "day" in q or "daily" in q:
        bucket = "day"
    else:
        bucket = "month"
    intent["time_bucket"] = bucket
    print(
        f"[Normalize] Set time_bucket={bucket} for trend on {pd}",
        file=sys.stderr,
    )
    return intent


def _looks_like_latest_single_row_question(question_lower: str) -> bool:
    """True when the user wants the single latest/most recent row (not a time-window phrase)."""
    if re.search(r"\blast\s+\d+\s+(?:day|days|week|weeks|month|months|year|years)\b", question_lower):
        return False
    if any(
        p in question_lower
        for p in ("last year", "last month", "last week", "last quarter", "past year", "past month")
    ):
        return False
    if re.search(
        r"\b(?:most\s+recent|latest|newest)\b.*\b(?:work\s+)?orders?\b",
        question_lower,
    ):
        return True
    if re.search(
        r"\bwhat(?:'s|s| is)\s+the\s+(?:most\s+recent|latest|newest)\b",
        question_lower,
    ):
        return True
    if re.search(r"\b(?:the\s+)?last\s+(?:work\s+)?order\b", question_lower):
        return True
    return False


def _coerce_latest_row_intent(
    intent: Dict[str, Any],
    question: Optional[str],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Turn mistaken count+group_by(primary_id) into list ordered by primary_date desc.

    The LLM often copies the 'top 1 by count' template; 'most recent' needs ORDER BY date.
    """
    if not question:
        return intent
    q = question.lower()
    if not _looks_like_latest_single_row_question(q):
        return intent

    dataset = intent.get("dataset") or ""
    table = _table_meta(schema, dataset)
    primary_id = table.get("primary_id")
    primary_date = table.get("primary_date")
    if not primary_id or not primary_date:
        return intent

    if intent.get("aggregation") != "count":
        return intent

    group_by = intent.get("group_by") or []
    if isinstance(group_by, str):
        group_by = [group_by]
    if group_by != [primary_id]:
        return intent

    intent["aggregation"] = "list"
    intent["group_by"] = []
    intent["sort_column"] = primary_date
    intent["sort_direction"] = "desc"
    if not intent.get("limit"):
        intent["limit"] = 1
    print(
        "[Normalize] Coerced 'latest/most recent' intent: list ordered by "
        f"{primary_date} desc (was count grouped by {primary_id})",
        file=sys.stderr,
    )
    return intent


def _maybe_coerce_list_for_detail_lookup(
    intent: Dict[str, Any],
    question: Optional[str],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Prefer list aggregation when the user asks for details/lookup and primary_id is filtered."""
    if not question:
        return intent
    q = question.lower()
    detailish = any(
        w in q
        for w in (
            "detail",
            "details",
            "show me the",
            "lookup",
            "information about",
            "full record",
        )
    )
    if not detailish:
        return intent
    if intent.get("aggregation") == "list":
        return intent
    table = _table_meta(schema, intent.get("dataset") or "")
    pid = table.get("primary_id")
    if not pid:
        return intent
    for f in intent.get("filters") or []:
        if f.get("column") != pid:
            continue
        vals = f.get("values") or []
        if not vals:
            continue
        intent["aggregation"] = "list"
        print(
            "[Normalize] Coerced aggregation to 'list' for detail-style question with primary_id filter",
            file=sys.stderr,
        )
        return intent
    return intent
