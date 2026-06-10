"""Generic evidence-guided deterministic planning.

The evidence block in a benchmark or application prompt often contains useful
compiler-grade facts: column mappings, literal predicates, simple formulas, and
date/range hints.  This module converts only those explicit, schema-grounded
facts into QueryPlan dictionaries.  It intentionally does not branch on a
particular database, domain, benchmark, table set, or question id.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


MONTHS = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
}


@dataclass(frozen=True)
class ColumnInfo:
    table: str
    name: str
    type: str
    aliases: Tuple[str, ...]

    @property
    def ref(self) -> str:
        return f"{self.table}.{self.name}"


@dataclass
class EvidenceContext:
    schema: Dict[str, Any]
    tables: List[str]
    columns: List[ColumnInfo]
    primary_ids: Dict[str, str]
    primary_dates: Dict[str, str]
    value_index: Dict[str, Dict[str, List[str]]]


def build_evidence_plan(
    question: str,
    schema: Dict[str, Any],
    value_index: Optional[Dict[str, Dict[str, List[str]]]] = None,
) -> Optional[Dict[str, Any]]:
    """Return a deterministic QueryPlan when evidence is explicit enough."""
    q, evidence = _split_question_evidence(question)
    if not evidence and not _question_has_deterministic_surface(q):
        return None

    ctx = _build_context(schema, value_index or {})
    plan = _generic_evidence_plan(q, evidence, ctx)
    if plan is None:
        return None
    plan = _add_intent_aware_joins(plan, f"{q}\n{evidence}", ctx)
    return _with_meta(plan, "evidence")


def _question_has_deterministic_surface(question: str) -> bool:
    return bool(
        re.search(r"\b(?:percent|percentage|ratio)\b", question, re.I)
        and re.search(r"(?:=|>=|<=|>|<|\boverall\b|\"[^\"]+\"|'[^']+')", question)
        or re.search(
            r"\b\d+(?:\.\d+)?\s*%\s+(?:higher|lower|above|below|more|less)\s+than\s+(?:the\s+)?average\b",
            question,
            re.I,
        )
    )


def _with_meta(plan: Dict[str, Any], source: str) -> Dict[str, Any]:
    plan = dict(plan)
    plan["meta"] = {
        "pipeline": source,
        "intent": {},
        "retry_count": 0,
        "auto_fixes_applied": ["deterministic_text_plan"],
        "validation_errors": [],
        "lint_errors": [],
    }
    return plan


def _split_question_evidence(question: str) -> Tuple[str, str]:
    parts = re.split(r"\bEvidence\s*:\s*", question or "", maxsplit=1, flags=re.I | re.S)
    if len(parts) == 1:
        return question or "", ""
    return parts[0].strip(), parts[1].strip()


def _build_context(
    schema: Dict[str, Any],
    value_index: Dict[str, Dict[str, List[str]]],
) -> EvidenceContext:
    tables: List[str] = []
    columns: List[ColumnInfo] = []
    primary_ids: Dict[str, str] = {}
    primary_dates: Dict[str, str] = {}

    for table in schema.get("tables", []) or []:
        if not isinstance(table, dict) or not isinstance(table.get("name"), str):
            continue
        table_name = table["name"]
        tables.append(table_name)
        if isinstance(table.get("primary_id"), str):
            primary_ids[table_name] = table["primary_id"]
        if isinstance(table.get("primary_date"), str):
            primary_dates[table_name] = table["primary_date"]
        for col in table.get("columns", []) or []:
            if not isinstance(col, dict) or not isinstance(col.get("name"), str):
                continue
            name = col["name"]
            db_column = str(col.get("db_column") or name)
            aliases = _column_aliases(table_name, name, db_column)
            columns.append(
                ColumnInfo(
                    table=table_name,
                    name=name,
                    type=str(col.get("type") or ""),
                    aliases=aliases,
                )
            )

    return EvidenceContext(
        schema=schema,
        tables=tables,
        columns=columns,
        primary_ids=primary_ids,
        primary_dates=primary_dates,
        value_index=value_index,
    )


def _column_aliases(table: str, name: str, db_column: str) -> Tuple[str, ...]:
    name_no_id = name[:-2] if name.lower().endswith("id") and len(name) > 2 else ""
    db_no_id = db_column[:-2] if db_column.lower().endswith("id") and len(db_column) > 2 else ""
    raw = {
        name,
        name.replace("_", " "),
        _strip_db_identifier(db_column),
        _strip_db_identifier(db_column).replace("_", " "),
        f"{table}.{name}",
        f"{table} {name.replace('_', ' ')}",
    }
    if name_no_id:
        raw.update({f"{name_no_id} id", f"{name_no_id} number", f"{name_no_id} code"})
    if db_no_id:
        raw.update({f"{db_no_id} id", f"{db_no_id} number", f"{db_no_id} code"})
    aliases = {_norm(a) for a in raw if a}
    return tuple(sorted(aliases, key=len, reverse=True))


def _strip_db_identifier(value: str) -> str:
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return value


def _generic_evidence_plan(question: str, evidence: str, ctx: EvidenceContext) -> Optional[Dict[str, Any]]:
    all_text = f"{question}\n{evidence}"
    base_evidence = _strip_value_definition_clauses(_strip_formula_clauses(evidence))
    base_text = f"{question}\n{_strip_bool_filter_clauses(base_evidence)}"
    filters = _extract_filters(base_text, ctx)
    filters.extend(_extract_mapping_condition_filters(evidence, ctx))
    filters.extend(_extract_contextual_mapping_where_filters(question, evidence, ctx))
    filters.extend(_extract_standalone_condition_filters(evidence, ctx))
    filters.extend(_extract_represented_filters(question, evidence, ctx))
    filters.extend(_extract_question_temporal_filters(question, evidence, ctx, filters))
    filters.extend(_extract_question_year_filters(question, ctx, filters))
    filters.extend(_extract_is_a_filters(question, evidence, ctx))
    filters.extend(_extract_full_name_filters(question, evidence, ctx))
    filters.extend(_extract_percentage_question_filters(question, ctx))
    filters.extend(_extract_question_literal_filters(question, evidence, ctx))
    filters.extend(_extract_per_unit_filters(question, evidence, ctx))
    filters.extend(_extract_natural_language_filters(question, ctx))
    filters.extend(_extract_natural_range_filters(question, ctx))
    filters.extend(_extract_named_entity_filters(question, ctx, filters))
    filters.extend(_extract_literal_value_filters(question, evidence, ctx, existing_filters=filters))
    filters = _rewrite_filter_value_aliases(filters, evidence, ctx)
    filters = _dedupe_bool(filters)
    filters = _rewrite_open_age_filters(filters, ctx)
    filters = _prune_raw_age_year_filters(filters)
    filters = _dedupe_bool(filters)
    filters = _resolve_filter_values_with_index(filters, ctx)
    filters = _prune_type_incompatible_filters(filters, ctx)
    filters = _prune_equalities_covered_by_in(filters)
    filters = _prune_conflicting_equalities(filters)
    relative_average = _extract_relative_average_filter(question, ctx, filters, evidence=evidence)
    if relative_average is None:
        relative_average = _extract_above_below_average_filter(question, ctx, filters)
    if relative_average is not None:
        filters.append(relative_average)
        filters = _dedupe_bool(filters)
    mapping_select = _extract_mapped_selects(question, evidence, ctx, filters)
    boolean_answer = _extract_boolean_answer(question, evidence, filters, ctx)
    count_threshold_hint = _extract_count_threshold(evidence, ctx)

    extreme_comparison = _extract_extreme_group_comparison_plan(question, evidence, ctx, filters)
    if extreme_comparison is not None:
        return extreme_comparison

    entity_extremum = _extract_entity_extremum_plan(question, evidence, ctx, filters)
    if entity_extremum is not None:
        return entity_extremum

    ranked_metric_values = _extract_ranked_metric_values_plan(question, evidence, ctx, filters, mapping_select)
    if ranked_metric_values is not None:
        return ranked_metric_values
    formula = None
    defer_formula = _defer_scalar_formula(question, evidence, count_threshold_hint)
    if not defer_formula:
        formula = _extract_formula_select(question, evidence, ctx, filters)
    if formula is None and not defer_formula:
        formula = _extract_infix_aggregate_ratio_select(question, evidence, ctx, filters)
    if formula is None and not defer_formula:
        formula = _extract_year_change_formula(question, evidence, ctx, filters)
    if formula is None and not defer_formula:
        formula = _extract_year_difference_formula(question, evidence, ctx, filters)
    if formula is None and not defer_formula:
        formula = _extract_spent_period_formula(question, evidence, ctx, filters)

    if formula is not None:
        ranked_dimension = _extract_ranked_dimension_plan(
            question,
            evidence,
            ctx,
            filters,
            mapping_select,
            formula["select"]["expr"] if isinstance(formula.get("select"), dict) else None,
            formula.get("metric_refs") or formula["refs"],
        )
        if ranked_dimension is not None:
            return ranked_dimension
        formula_filter_keys = formula.get("filter_keys") or set()
        formula_condition_refs = set(formula.get("condition_refs") or set())
        outer_filters = [
            f
            for f in filters
            if _bool_key(f) not in formula_filter_keys
            and not (_filter_refs([f]) and _filter_refs([f]) <= formula_condition_refs)
        ]
        metric_refs = formula.get("metric_refs") or formula["refs"]
        scoped = _entity_scoped_formula_dataset(ctx, metric_refs, outer_filters, text=all_text)
        if scoped is not None:
            dataset, outer_filters = scoped
        else:
            dataset = _choose_dataset(ctx, formula["refs"], outer_filters, text=all_text)
            if dataset is None:
                return None
        select_items = formula["select"] if isinstance(formula.get("select"), list) else [formula["select"]]
        if re.search(r"\b(?:age|how old)\b", question, re.I) and not any(
            _expr_has_aggregate(item.get("expr")) for item in select_items if isinstance(item, dict)
        ):
            return _advanced(
                dataset,
                select_items,
                where=_and(outer_filters) if outer_filters else None,
                distinct=True,
                limit=None,
            )
        return _advanced(
            dataset,
            select_items,
            where=_and(outer_filters) if outer_filters else None,
            limit=1,
        )

    common_dimension = _extract_most_common_dimension_plan(question, evidence, ctx, filters)
    if common_dimension is not None:
        return common_dimension

    ranked_computed = _extract_ranked_computed_report(question, evidence, ctx, filters)
    if ranked_computed is not None:
        return ranked_computed

    ranked_report = _extract_ranked_entity_report(question, evidence, ctx, filters)
    if ranked_report is not None:
        return ranked_report

    ranked_temporal = _extract_ranked_temporal_answer(question, evidence, ctx, filters)
    if ranked_temporal is not None:
        return ranked_temporal

    explicit_ranked_aggregate = _extract_explicit_ranked_aggregate_plan(
        question,
        evidence,
        ctx,
        filters,
        mapping_select,
    )
    if explicit_ranked_aggregate is not None:
        return explicit_ranked_aggregate

    percentage = _extract_percentage_from_filters(question, filters, ctx)
    if percentage is not None:
        dataset = _choose_dataset(ctx, list(_filter_refs(filters)), filters, text=all_text)
        if dataset is None:
            return None
        return _advanced(
            dataset,
            [percentage["select"]],
            where=percentage.get("where"),
            limit=1,
        )

    aggregate = None
    if relative_average is None and not _has_filter_only_aggregate(question, evidence):
        aggregate = _extract_average_age_aggregate(question, evidence, ctx) or _extract_simple_aggregate(question, evidence, ctx)
    if aggregate is not None:
        ranked_dimension = _extract_ranked_dimension_plan(
            question,
            evidence,
            ctx,
            filters,
            mapping_select,
            {"col": aggregate["ref"]},
            [aggregate["ref"]],
        )
        if ranked_dimension is not None:
            return ranked_dimension
        aggregate_dimension = _extract_aggregate_dimension_plan(
            question,
            evidence,
            ctx,
            filters,
            mapping_select,
            aggregate,
        )
        if aggregate_dimension is not None:
            return aggregate_dimension
        refs = list(aggregate.get("refs") or [aggregate["ref"]])
        dataset = _choose_dataset(ctx, refs, filters, text=all_text)
        if dataset is None:
            return None
        return _advanced(
            dataset,
            [aggregate["select"]],
            where=_and(filters) if filters else None,
            limit=1,
        )

    if _asks_for_count(question) and not _asks_to_list_answers_with_count(question):
        dataset = _choose_dataset(ctx, [], filters, text=all_text)
        if dataset is None:
            return None
        threshold = count_threshold_hint
        count_ref = _count_ref_for_question(question, dataset, ctx, filters)
        if threshold is not None:
            count_ref = threshold["col"].ref
        if count_ref and _count_should_use_distinct(question, all_text):
            count_expr = _func("count_distinct", {"col": count_ref})
        else:
            count_expr = _func("count", {"col": count_ref}) if count_ref else _func("count")
        if threshold is not None and count_ref:
            threshold_ref = _threshold_count_ref(threshold["col"], all_text, ctx) or threshold["col"].ref
            plan = _advanced(
                dataset,
                [{"expr": count_expr, "alias": "total"}],
                where=_and(filters) if filters else None,
                limit=None,
            )
            plan["group_by"] = [{"col": count_ref}]
            plan["having"] = _cmp(_func("count", {"col": threshold_ref}), threshold["op"], threshold["value"])
            return plan
        return _advanced(
            dataset,
            [{"expr": count_expr, "alias": "total"}],
            where=_and(filters) if filters else None,
            limit=1,
        )

    question_selects = (
        _extract_explicit_question_selects(question, ctx, filters)
        if mapping_select
        else _extract_question_selects(question, ctx, filters)
    )
    selects = _dedupe_selects(list(mapping_select) + list(question_selects)) if mapping_select else question_selects
    selects = _prefer_mapped_selects(selects, mapping_select, ctx)
    selects = _dedupe_selects(list(selects) + _extract_requested_filter_selects(question, ctx, filters))
    selects = _prefer_filter_backed_selects(selects, filters, ctx)
    selects = _sort_selects_by_question(selects, question, ctx)
    selects = _expand_sibling_measure_selects(selects, question, ctx)
    selects = _prefer_order_context_selects(selects, _extract_ordering(question, evidence, ctx), ctx)
    computed_selects = _extract_computed_selects(question, evidence, ctx)
    if computed_selects:
        selects = _add_ordered_computed_sources(selects, computed_selects, question, _extract_ordering(question, evidence, ctx), ctx)
        selects = _drop_computed_source_selects(
            selects,
            computed_selects,
            question,
            filters,
            ctx,
            _extract_ordering(question, evidence, ctx),
        )
    if not selects and boolean_answer is not None:
        boolean_filter_keys = boolean_answer.get("filter_keys") or set()
        outer_filters = [f for f in filters if _bool_key(f) not in boolean_filter_keys]
        dataset = _choose_dataset(ctx, boolean_answer["refs"], outer_filters, text=all_text)
        if dataset is None:
            return None
        return _advanced(
            dataset,
            [boolean_answer["select"]],
            where=_and(outer_filters) if outer_filters else None,
            limit=None if boolean_answer.get("rowwise") else 1,
        )
    if not selects and not computed_selects:
        selects = _extract_primary_id_selects(question, ctx, filters)
    if not selects and not computed_selects:
        return None

    refs = [s["ref"] for s in selects if isinstance(s.get("ref"), str)]
    refs.extend(ref for item in computed_selects for ref in item.get("refs", []))
    dataset = _choose_dataset(ctx, refs, filters, text=all_text)
    if dataset is None:
        return None

    threshold = count_threshold_hint
    if threshold is not None and not _asks_for_count(question):
        threshold_ref = _threshold_count_ref(threshold["col"], all_text, ctx) or threshold["col"].ref
        group_by = [{"col": s["ref"]} for s in selects if isinstance(s.get("ref"), str)]
        group_by.extend(_group_identity_refs(selects, ctx))
        order_by = _extract_ordering(question, evidence, ctx)
        distinct = _asks_for_distinct(all_text) or _entity_select_needs_distinct(selects, filters, ctx)
        distinct = _valid_distinct_with_ordering(distinct, selects, order_by)
        plan = _advanced(
            dataset,
            [{"expr": {"col": s["ref"]}, "alias": s["alias"]} for s in selects]
            + [{"expr": item["expr"], "alias": item["alias"]} for item in computed_selects],
            where=_and(filters) if filters else None,
            distinct=distinct,
            order_by=order_by,
            limit=_limit_for_question(question),
        )
        plan["group_by"] = _dedupe_exprs(group_by)
        plan["group_by"].extend(
            item for item in _dedupe_exprs({"col": ref} for computed in computed_selects for ref in computed.get("refs", []))
            if item not in plan["group_by"]
        )
        plan["having"] = _cmp(_func("count", {"col": threshold_ref}), threshold["op"], threshold["value"])
        return plan

    order_by = _extract_ordering(question, evidence, ctx)
    distinct = (
        _asks_for_distinct(all_text)
        or _entity_select_needs_distinct(selects, filters, ctx)
        or _identifier_values_need_distinct(question, selects, ctx)
    )
    if _preserve_rowwise_selects(question):
        distinct = False
    distinct_on = _ordered_distinct_keys(distinct, selects, order_by)
    distinct = _valid_distinct_with_ordering(distinct, selects, order_by)
    final_filters = list(filters)
    if distinct:
        final_filters.extend(_answer_not_null_filters(selects, filters, ctx))
    front_selects, tail_selects = _split_computed_source_selects(selects, computed_selects, order_by)
    select_payload = (
        [{"expr": {"col": s["ref"]}, "alias": s["alias"]} for s in front_selects]
        + [{"expr": item["expr"], "alias": item["alias"]} for item in computed_selects]
        + [{"expr": {"col": s["ref"]}, "alias": s["alias"]} for s in tail_selects]
    )
    return _advanced(
        dataset,
        select_payload,
        where=_and(final_filters) if final_filters else None,
        distinct=distinct,
        distinct_on=distinct_on,
        order_by=order_by,
        limit=_limit_for_question(question),
    )


def _strip_formula_clauses(evidence: str) -> str:
    kept: List[str] = []
    for clause in re.split(r";|\n", evidence or ""):
        if re.search(r"\b(?:DIVIDE|DIVISION|SUBTRACT|MULTIPLY)\s*\(", clause, re.I) and not re.search(r"\)\s*(?:=|>=|<=|>|<)\s*'?[-+]?\d", clause):
            continue
        if re.search(r"\bcalculation\s*=", clause, re.I):
            continue
        if re.search(r"\b(?:percentage|ratio|difference)\s*=", clause, re.I):
            continue
        kept.append(clause)
    return "; ".join(kept)


def _strip_value_definition_clauses(evidence: str) -> str:
    """Remove clauses that define literal aliases rather than row predicates."""
    kept: List[str] = []
    for clause in re.split(r";|\n", evidence or ""):
        if re.search(
            r"(?:^|\b[A-Za-z_][A-Za-z0-9_.\"` -]*\s*=\s*)"
            r"(?:'[^']*'|\"[^\"]*\")\s+(?:refers?\s*(?:to)?|means?)\s+"
            r"(?:'[^']*'|\"[^\"]*\")\s*$",
            clause.strip(),
            re.I,
        ):
            continue
        kept.append(clause)
    return "; ".join(kept)


def _strip_bool_filter_clauses(evidence: str) -> str:
    kept: List[str] = []
    for clause in re.split(r";|\n", evidence or ""):
        check = re.sub(
            r"\bbetween\s+'?[^;]+?'?\s+and\s+'?[^;]+?'?",
            "between_range",
            clause,
            flags=re.I,
        )
        has_bool = re.search(r"\b(?:or|and)\b", check, re.I)
        has_cmp = re.search(r"(?:=|>=|<=|>|<|\bis\s+not\s+null\b|\bis\s+null\b)", check, re.I)
        if has_bool and has_cmp:
            continue
        kept.append(clause)
    return "; ".join(kept)


def _extract_filters(text: str, ctx: EvidenceContext) -> List[Dict[str, Any]]:
    filters: List[Dict[str, Any]] = []
    text = _normalize_condition_spacing(text)
    list_filters, text = _extract_list_conditions_and_mask(text, ctx)
    filters.extend(list_filters)

    for m in re.finditer(
        r"\b([A-Za-z_][A-Za-z0-9_ -]{0,40})\s*=\s*(?:'([^']*)'|\"([^\"]*)\")",
        text,
        flags=re.I,
    ):
        lhs = _comparison_lhs(m.group(1))
        col = _resolve_column(lhs, ctx) or _resolve_column_for_context(lhs, ctx, filters, text)
        if col is not None:
            raw = m.group(0).split("=", 1)[1].strip()
            value = _parse_literal_for_column(raw, col, "=")
            if _looks_like_compact_period(value) and not _text_date_like(col):
                alt = _resolve_text_period_column(lhs, ctx)
                if alt is not None:
                    col = alt
                    value = _parse_literal_for_column(raw, col, "=")
            filters.append(_column_cmp(col, "=", value))

    for m in re.finditer(
        r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:'([^']*)'|\"([^\"]*)\")",
        text,
    ):
        if _has_identifier_word_before(text, m.start()):
            continue
        lhs = _comparison_lhs(m.group(1))
        col = _resolve_column(lhs, ctx) or _resolve_column_for_context(lhs, ctx, filters, text)
        if col is not None:
            raw = m.group(0).split("=", 1)[1].strip()
            value = _parse_literal_for_column(raw, col, "=")
            if _looks_like_compact_period(value) and not _text_date_like(col):
                alt = _resolve_text_period_column(lhs, ctx)
                if alt is not None:
                    col = alt
                    value = _parse_literal_for_column(raw, col, "=")
            filters.append(_column_cmp(col, "=", value))

    for m in re.finditer(
        r"\byear\s*\(\s*([^)]+?)\s*\)\s*(=|>=|<=|>|<|between)\s*('?[\w./:-]+'?(?:\s+and\s+'?[\w./:-]+'?)?)",
        text,
        flags=re.I,
    ):
        col = _resolve_column(m.group(1), ctx)
        if col is None:
            continue
        value = _year_comparison_value(m.group(3), col)
        age_cmp = _age_like_year_comparison(col, m.group(2).lower(), value, text, ctx)
        if age_cmp is not None:
            filters.append(age_cmp)
            continue
        filters.append(_cmp(_year_expr(col), m.group(2).lower(), value))

    for m in re.finditer(
        r"\bmonth\s*\(\s*([^)]+?)\s*\)\s*(=|>=|<=|>|<|between)\s*('?[\w./:-]+'?(?:\s+and\s+'?[\w./:-]+'?)?)",
        text,
        flags=re.I,
    ):
        col = _resolve_column(m.group(1), ctx)
        if col is None:
            continue
        value = _parse_comparison_value(m.group(3), force_year=False)
        if _text_date_like(col):
            value = _month_value(value)
        filters.append(_cmp(_month_expr(col), m.group(2).lower(), value))

    for m in re.finditer(
        r"([A-Za-z0-9_.\"` -]+?)\s+between\s+'?([0-9A-Za-z./:-]+)'?\s+and\s+'?([0-9A-Za-z./:-]+)'?",
        text,
        flags=re.I,
    ):
        col = _resolve_column(m.group(1), ctx)
        if col is None:
            continue
        filters.append(_cmp({"col": col.ref}, "between", [_parse_literal(m.group(2)), _parse_literal(m.group(3))]))

    for m in re.finditer(r"([A-Za-z0-9_.\"` -]+?)\s+like\s+'([^']+)'", text, flags=re.I):
        col = _resolve_column(m.group(1), ctx)
        if col is None:
            continue
        op, value = _like_to_op(m.group(2))
        filters.append(_cmp({"col": col.ref}, op, value))

    for m in re.finditer(
        r"([A-Za-z0-9_.\"` -]+?)\s*(=|>=|<=|>|<)\s*('(?:[^']*)'?|\"(?:[^\"]*)\"?|[+-]?\d+(?:\.\d+)?|[+-])",
        text,
        flags=re.I,
    ):
        lhs = _comparison_lhs(m.group(1))
        if _norm(lhs).endswith(("calculation", "percentage", "ratio")):
            continue
        col = _resolve_column(lhs, ctx) or _resolve_column_for_context(lhs, ctx, filters, text)
        if col is None:
            continue
        value = _parse_literal_for_column(m.group(3), col, m.group(2))
        if _looks_like_compact_period(value) and not _text_date_like(col):
            alt = _resolve_text_period_column(lhs, ctx)
            if alt is not None:
                col = alt
        filters.append(_column_cmp(col, m.group(2), value))

    return _dedupe_bool(filters)


def _extract_list_conditions_and_mask(
    text: str,
    ctx: EvidenceContext,
) -> Tuple[List[Dict[str, Any]], str]:
    filters: List[Dict[str, Any]] = []
    spans: List[Tuple[int, int]] = []
    patterns = [
        re.compile(
            r"([A-Za-z0-9_.\"` -]+?)\s+(NOT\s+)?IN\s*\(([^)]*)\)",
            re.I,
        ),
        re.compile(
            r"([A-Za-z0-9_.\"` -]+?)\s*=\s*"
            r"((?:'[^']*'|\"[^\"]*\")\s*,\s*(?:'[^']*'|\"[^\"]*\")"
            r"(?:\s*,\s*(?:'[^']*'|\"[^\"]*\"))*)",
            re.I,
        ),
    ]
    for pattern in patterns:
        for match in pattern.finditer(text):
            if any(start <= match.start() < end for start, end in spans):
                continue
            lhs = _comparison_lhs(match.group(1))
            col = _resolve_column(lhs, ctx) or _resolve_column_for_context(lhs, ctx, filters, text)
            if col is None:
                continue
            raw_values = match.group(3) if match.lastindex and match.lastindex >= 3 else match.group(2)
            values = [
                _parse_literal_for_column(raw, col, "=")
                for raw in _split_top_level_args(raw_values)
                if raw.strip()
            ]
            if len(values) < 2:
                continue
            negated = bool(match.lastindex and match.lastindex >= 2 and match.group(2) and re.search(r"\bNOT\b", match.group(2), re.I))
            filters.append(_cmp({"col": col.ref}, "not_in" if negated else "in", values))
            spans.append(match.span())

    if not spans:
        return filters, text
    chars = list(text)
    for start, end in spans:
        chars[start:end] = " " * (end - start)
    return filters, "".join(chars)


def _extract_natural_language_filters(question: str, ctx: EvidenceContext) -> List[Dict[str, Any]]:
    """Extract simple typed attribute/value predicates stated without SQL operators."""
    text = _norm(question)
    matches: List[Tuple[int, int, Dict[str, Any]]] = []
    seen_spans: List[Tuple[int, int]] = []
    columns = sorted(ctx.columns, key=lambda col: max((len(alias) for alias in col.aliases), default=0), reverse=True)

    for col in columns:
        aliases = [alias for alias in col.aliases if len(alias) > 2 and "." not in alias]
        for alias in aliases:
            pattern = (
                rf"\b{re.escape(alias)}\b"
                r"(?:"
                r"\s+(?:level|value|degree)(?:\s+of)?(?:\s+only)?"
                r"|\s+of\s+only"
                r"|\s+(?:is|equals|equal\s+to)"
                r")"
                r"\s+([a-z][a-z0-9_.+-]*|[+-]?\d+(?:\.\d+)?)\b"
            )
            for match in re.finditer(pattern, text, re.I):
                value_text = match.group(1)
                tail = text[match.end():].lstrip()
                if re.match(r"^(?:percent|higher|lower|above|below|more|less)\b", tail, re.I):
                    continue
                value = _parse_literal(value_text)
                if not _literal_type_fits_column(col, value):
                    continue
                span = match.span()
                if any(start <= span[0] and span[1] <= end for start, end in seen_spans):
                    continue
                matches.append((span[0], len(alias), _column_cmp(col, "=", value)))
                seen_spans.append(span)

    matches.sort(key=lambda item: (item[0], -item[1]))
    return _dedupe_bool([item[2] for item in matches])


def _extract_natural_range_filters(question: str, ctx: EvidenceContext) -> List[Dict[str, Any]]:
    filters: List[Dict[str, Any]] = []
    pattern = re.compile(
        r"\b(?:more|greater|higher)\s+than\s+([+-]?\d+(?:\.\d+)?)\s+"
        r"(?:but\s+)?(?:and\s+)?(?:less|lower|smaller)\s+than\s+([+-]?\d+(?:\.\d+)?)\s+"
        r"([A-Za-z][A-Za-z0-9_ -]{1,70}?)(?:[?.;,]|$)",
        re.I,
    )
    for match in pattern.finditer(question):
        col = _resolve_column_for_context(match.group(3), ctx, filters, question)
        if col is None or not _numeric_column(col):
            continue
        filters.extend(
            [
                _column_cmp(col, ">", _parse_literal(match.group(1))),
                _column_cmp(col, "<", _parse_literal(match.group(2))),
            ]
        )
    return _dedupe_bool(filters)


def _extract_named_entity_filters(
    question: str,
    ctx: EvidenceContext,
    existing_filters: Sequence[Dict[str, Any]] = (),
) -> List[Dict[str, Any]]:
    """Bind proper-name comparisons to a mentioned entity's name-like column."""
    existing_values = {
        _norm(value)
        for filt in existing_filters
        for value in [_single_filter_value(filt)]
        if value is not None
    }
    phrases = [
        phrase.strip()
        for phrase in re.findall(r"\b[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+)+\b", question)
        if not re.match(r"^(?:What|Which|Who|Give|List|Find|Calculate|Among|From|At|In|On|For|With|By)\b", phrase)
        and not _phrase_is_quoted(question, phrase)
        and _norm(phrase) not in existing_values
        and _resolve_column_for_context(phrase, ctx, [], question) is None
        and not any(
            _compact_identifier(alias).endswith(_compact_identifier(phrase))
            for col in ctx.columns
            for alias in col.aliases
            if _compact_identifier(phrase)
        )
    ]
    phrases = _dedupe_strings(phrases)
    if not phrases:
        return []

    q_norm = _norm(question)
    comparative = len(phrases) > 1 and re.search(r"\b(?:older|younger|earlier|later)\b", question, re.I) is not None
    candidates: List[Tuple[int, ColumnInfo]] = []
    for col in ctx.columns:
        name_norm = _norm(col.name)
        if col.type not in {"text", "string", "varchar"} and "char" not in col.type.lower():
            continue
        if name_norm != "name" and not name_norm.endswith(" name"):
            continue
        table_norm = _norm(col.table)
        singular = table_norm[:-1] if table_norm.endswith("s") else table_norm
        plural = table_norm if table_norm.endswith("s") else f"{table_norm}s"
        near_entity_mention = any(
            re.search(
                rf"\b(?:{re.escape(table_norm)}|{re.escape(singular)}|{re.escape(plural)})\b[^?.;,]{{0,40}}"
                rf"{re.escape(_norm(phrase))}\b",
                q_norm,
            )
            for phrase in phrases
        )
        if not comparative and not near_entity_mention:
            continue
        score = 0
        if any(_alias_in_text(item, q_norm) for item in {table_norm, singular, plural}):
            score += 20
        if col.table in ctx.primary_ids:
            score += 4
        if name_norm != "name":
            score += 3
        if score:
            candidates.append((score, col))
    if not candidates:
        return []
    candidates.sort(key=lambda item: item[0], reverse=True)
    col = candidates[0][1]
    if len(phrases) > 1 and re.search(r"\b(?:or|between)\b", question, re.I):
        return [_cmp({"col": col.ref}, "in", phrases)]
    return [_column_cmp(col, "=", phrases[0])]


def _phrase_is_quoted(text: str, phrase: str) -> bool:
    return re.search(rf"""["'][^"']*{re.escape(phrase)}[^"']*["']""", text, re.I) is not None


def _extract_relative_average_filter(
    question: str,
    ctx: EvidenceContext,
    scope_filters: Sequence[Dict[str, Any]],
    *,
    evidence: str = "",
) -> Optional[Dict[str, Any]]:
    match = re.search(
        r"(?P<prefix>.+?)\b(?P<pct>\d+(?:\.\d+)?)\s*%\s+"
        r"(?P<direction>higher|lower|above|below|more|less)\s+than\s+(?:the\s+)?average\b",
        question,
        re.I | re.S,
    )
    if not match:
        return None

    metric = _explicit_average_metric(evidence, ctx) or _resolve_metric_before_relative_average(match.group("prefix"), ctx)
    if metric is None or not _numeric_column(metric):
        return None

    pct = float(match.group("pct")) / 100.0
    direction = match.group("direction").lower()
    factor = 1.0 + pct if direction in {"higher", "above", "more"} else 1.0 - pct
    op = ">" if direction in {"higher", "above", "more"} else "<"
    metric_scope = [
        filt
        for filt in scope_filters
        if metric.ref not in _filter_refs([filt])
    ]
    subquery = _advanced(
        metric.table,
        [
            {
                "expr": _op("*", _func("avg", {"col": metric.ref}), {"lit": factor}),
                "alias": "value",
            }
        ],
        where=_and(metric_scope) if metric_scope else None,
        limit=1,
    )
    return _cmp({"col": metric.ref}, op, {"scalar_subquery": {"plan": subquery}})


def _explicit_average_metric(evidence: str, ctx: EvidenceContext) -> Optional[ColumnInfo]:
    for match in re.finditer(r"\b(?:AVG|AVERAGE)\s*\(\s*([^)]+?)\s*\)", evidence or "", re.I):
        col = _resolve_column_for_context(match.group(1), ctx, [], evidence)
        if col is not None and _numeric_column(col):
            return col
    return None


def _extract_above_below_average_filter(
    question: str,
    ctx: EvidenceContext,
    scope_filters: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    match = re.search(
        r"\b(?P<direction>above|below)[ -]average\s+(?P<metric>[A-Za-z][A-Za-z0-9_ -]{1,60}?)(?:\s+in\b|\s+for\b|[?.;,]|$)",
        question,
        re.I,
    )
    if match is None:
        return None
    metric = _resolve_column_for_context(match.group("metric"), ctx, scope_filters, question)
    if metric is None or not _numeric_column(metric):
        return None
    metric_scope = [filt for filt in scope_filters if metric.ref not in _filter_refs([filt])]
    subquery = _advanced(
        metric.table,
        [{"expr": _func("avg", {"col": metric.ref}), "alias": "value"}],
        where=_and(metric_scope) if metric_scope else None,
        limit=1,
    )
    return _cmp(
        {"col": metric.ref},
        ">" if match.group("direction").lower() == "above" else "<",
        {"scalar_subquery": {"plan": subquery}},
    )


def _resolve_metric_before_relative_average(prefix: str, ctx: EvidenceContext) -> Optional[ColumnInfo]:
    snippets: List[str] = []
    snippets.extend(reversed(re.findall(r"\(([^()]+)\)", prefix)))
    words = re.findall(r"[A-Za-z0-9_.+-]+", prefix)
    for width in range(min(8, len(words)), 0, -1):
        snippets.append(" ".join(words[-width:]))

    candidates: List[ColumnInfo] = []
    for snippet in snippets:
        col = _resolve_column_for_context(snippet, ctx, [], prefix)
        if col is not None and _numeric_column(col) and col not in candidates:
            candidates.append(col)
    return candidates[0] if candidates else None


def _extract_mapping_condition_filters(evidence: str, ctx: EvidenceContext) -> List[Dict[str, Any]]:
    filters: List[Dict[str, Any]] = []
    for _lhs, rhs in _mapping_clauses(evidence):
        if not _conditionish(rhs):
            continue
        cond_text = re.split(r"\bwhere\b", rhs, flags=re.I)[-1] if re.search(r"\bwhere\b", rhs, re.I) else rhs
        cond = _condition_tree(cond_text, ctx)
        if cond is not None:
            filters.append(cond)
    return _dedupe_bool(filters)


def _extract_contextual_mapping_where_filters(
    question: str,
    evidence: str,
    ctx: EvidenceContext,
) -> List[Dict[str, Any]]:
    """Resolve simple trailing WHERE clauses using the full request as context."""
    filters: List[Dict[str, Any]] = []
    context = f"{question}\n{evidence}"
    for clause in re.split(r";|\n", evidence or ""):
        match = re.search(r"\bwhere\b\s+(.+)$", clause, re.I | re.S)
        if match is None:
            continue
        for part in re.split(r"\s+\band\b\s+|\s+\bor\b\s+", match.group(1), flags=re.I):
            cmp_match = re.fullmatch(
                r"\s*([A-Za-z0-9_.\"` -]+?)\s*(=|>=|<=|>|<)\s*"
                r"('(?:[^']*)'|\"(?:[^\"]*)\"|[+-]?\d+(?:\.\d+)?)\s*",
                part,
                re.I,
            )
            if cmp_match is None:
                continue
            lhs, op, raw = cmp_match.groups()
            col = _resolve_column_for_context(lhs, ctx, filters, context)
            if col is None:
                continue
            filters.append(_column_cmp(col, op, _parse_literal_for_column(raw, col, op)))
    return _dedupe_bool(filters)


def _extract_standalone_condition_filters(evidence: str, ctx: EvidenceContext) -> List[Dict[str, Any]]:
    filters: List[Dict[str, Any]] = []
    for clause in re.split(r";|\n", evidence or ""):
        if re.search(r"\bcalculation\s*=", clause, re.I):
            continue
        if re.search(r"\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\(", clause, re.I):
            continue
        if not re.search(r"\b(?:SUBTRACT|year\s*\()", clause, re.I):
            continue
        if not _conditionish(clause):
            continue
        cond = _condition_tree(clause, ctx)
        if cond is not None:
            filters.append(cond)
    return _dedupe_bool(filters)


def _extract_is_a_filters(question: str, evidence: str, ctx: EvidenceContext) -> List[Dict[str, Any]]:
    filters: List[Dict[str, Any]] = []
    text = f"{question}\n{evidence}"
    for m in re.finditer(
        r"(?:'([^']+)'|\"([^\"]+)\"|([A-Z][A-Za-z0-9&/.'-]*(?:\s+[A-Z][A-Za-z0-9&/.'-]*){0,6}))\s+"
        r"(?:is|are)\s+(?:an?\s+|the\s+)?([A-Za-z][A-Za-z0-9_ -]{1,60}?)(?:[.;,\n]|$)",
        text,
    ):
        value = (m.group(1) or m.group(2) or m.group(3) or "").strip()
        phrase = (m.group(4) or "").strip()
        if not value or not phrase:
            continue
        if re.search(r"\bfull\s+name\b", phrase, re.I):
            continue
        col = _resolve_column(_clean_column_phrase(phrase), ctx)
        if col is None:
            continue
        filters.append(_column_cmp(col, "=", _parse_literal(value)))
    return _dedupe_bool(filters)


def _extract_full_name_filters(question: str, evidence: str, ctx: EvidenceContext) -> List[Dict[str, Any]]:
    mapped = _full_name_columns(evidence, ctx)
    if mapped is None:
        return []
    first_col, last_col = mapped
    names: List[Tuple[str, str]] = []
    text = f"{question}\n{evidence}"
    for m in re.finditer(
        r"([A-Z][A-Za-z.'-]+)\s+([A-Z][A-Za-z.'-]+)(?:'s)?\s+is\s+(?:the\s+)?full\s+name",
        text,
    ):
        names.append((m.group(1), m.group(2)))
    for m in re.finditer(r"\"([A-Z][A-Za-z.'-]+)\s+([A-Z][A-Za-z.'-]+)\"", question):
        names.append((m.group(1), m.group(2)))
    for m in re.finditer(r"\b([A-Z][A-Za-z.'-]+)\s+([A-Z][A-Za-z.'-]+)(?:'s)\b", question):
        names.append((m.group(1), m.group(2)))
    if not names:
        return []
    first, last = names[0]
    return [
        _column_cmp(first_col, "=", first.strip("'")),
        _column_cmp(last_col, "=", last.strip("'")),
    ]


def _extract_question_literal_filters(question: str, evidence: str, ctx: EvidenceContext) -> List[Dict[str, Any]]:
    filters: List[Dict[str, Any]] = []
    for m in re.finditer(r"\b([A-Za-z][A-Za-z0-9_ -]{1,40}?)\s+(?:number|no\.?|id|code)\s+\"([^\"]+)\"", question, re.I):
        col = _resolve_identifier_column(m.group(1), ctx)
        if col is not None:
            filters.append(_column_cmp(col, "=", _parse_literal(m.group(2))))

    for m in re.finditer(r"\b([A-Za-z][A-Za-z0-9_ -]{1,40}?)\s+(?:number|no\.?|id|code)\s+(?:No\.?\s*)?([A-Za-z0-9_-]+)", question, re.I):
        if m.group(2).lower() in {"of", "the", "a", "an", "and", "or", "with", "whose", "who", "that", "which"}:
            continue
        if _query_command_phrase(m.group(1)):
            continue
        col = _resolve_identifier_column(m.group(1), ctx)
        if col is not None:
            filters.append(_column_cmp(col, "=", _parse_literal(m.group(2))))

    for m in re.finditer(r"\b([A-Za-z][A-Za-z0-9_ -]{1,40}?)\s+id\s+No\.?\s*([A-Za-z0-9_-]+)", question, re.I):
        col = _resolve_identifier_column(m.group(1), ctx)
        if col is not None:
            filters.append(_column_cmp(col, "=", _parse_literal(m.group(2))))

    for m in re.finditer(r"\b(?:with|by|for)\s+(?:the\s+)?([A-Za-z][A-Za-z0-9_ -]{1,40}?)\s+\"([^\"]+)\"", question, re.I):
        col = _resolve_column(_clean_column_phrase(m.group(1)), ctx)
        if col is not None and _literal_fits_column(col, m.group(2)):
            filters.append(_column_cmp(col, "=", _parse_literal(m.group(2))))

    for m in re.finditer(r"\"([^\"]+)\"\s+([A-Za-z][A-Za-z0-9_ -]{1,40})", question, re.I):
        if re.match(r"\s*(?:against|versus|vs|overall|over)\b", m.group(2), re.I):
            continue
        col = _resolve_column(_clean_column_phrase(m.group(2)), ctx)
        if col is not None and _literal_fits_column(col, m.group(1)):
            filters.append(_column_cmp(col, "=", _parse_literal(m.group(1))))

    for m in re.finditer(r"\b([A-Za-z][A-Za-z0-9_ -]{1,30})\s+\"([^\"]+)\"", question, re.I):
        if _query_command_phrase(m.group(1)):
            continue
        phrase = _clean_column_phrase(m.group(1))
        col = _primary_id_for_table_phrase(phrase, ctx) or _resolve_column(f"{phrase} id", ctx) or _resolve_column(phrase, ctx)
        if col is not None and _literal_fits_column(col, m.group(2)):
            filters.append(_column_cmp(col, "=", _parse_literal(m.group(2))))

    for lhs, rhs in _mapping_clauses(evidence):
        if _conditionish(rhs) or not _mapping_lhs_requested(lhs, question):
            continue
        prep = re.search(r"\b(at|in|on|by|with|for)\s*$", lhs.strip(), re.I)
        if not prep:
            continue
        col = _resolve_column_for_context(_split_column_list(rhs)[0] if _split_column_list(rhs) else rhs, ctx, filters, f"{question}\n{evidence}")
        if col is None:
            continue
        pattern = rf"\b{re.escape(prep.group(1))}\s+([A-Z0-9][A-Za-z0-9&/.'_-]*(?:\s+[A-Z0-9][A-Za-z0-9&/.'_-]*){{0,4}})(?=\s+(?:and|with|where|that|which|who|as|in|on|at|by|for)\b|[?.;,]|$)"
        for value_match in re.finditer(pattern, question, re.I):
            value = value_match.group(1).strip()
            if value and value[0].islower():
                continue
            if value and _literal_fits_column(col, value):
                filters.append(_column_cmp(col, "=", _parse_literal(value)))

    for m in re.finditer(
        r"\bin\s+([A-Z][A-Za-z0-9&/.'-]*(?:\s+[A-Z][A-Za-z0-9&/.'-]*){0,4})\s+([A-Za-z][A-Za-z0-9_ -]{2,30})(?:[?.;,]|$)",
        question,
    ):
        value = m.group(1).strip()
        if m.group(2)[:1].isupper():
            continue
        phrase = _clean_column_phrase(m.group(2))
        col = _resolve_column_for_context(phrase, ctx, filters, question)
        if col is not None and _literal_fits_column(col, value):
            filters.append(_column_cmp(col, "=", _parse_literal(value)))

    return _dedupe_bool(filters)


def _extract_question_year_filters(
    question: str,
    ctx: EvidenceContext,
    existing_filters: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    existing_temporal_refs = {
        ref for ref in _filter_refs(existing_filters)
        if (col := _column_by_ref(ctx, ref)) is not None and _column_is_temporal(col)
    }
    if existing_temporal_refs:
        return []
    if _question_has_date_range(question):
        return []
    years = re.findall(r"\b(?:19|20)\d{2}\b", question or "")
    if len(years) != 1:
        return []
    if any(
        years[0] in str(value)
        for filt in existing_filters
        for value in [_single_filter_value(filt)]
        if value is not None
    ):
        return []
    if not re.search(r"\b(?:in|during|for|on)\s+(?:the\s+)?(?:year\s+)?(?:19|20)\d{2}\b", question, re.I):
        return []
    col = _temporal_column_for_context(ctx, question, existing_filters)
    if col is None:
        return []
    return [_cmp(_year_expr(col), "=", _year_comparison_value(years[0], col))]


def _extract_question_temporal_filters(
    question: str,
    evidence: str,
    ctx: EvidenceContext,
    existing_filters: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Extract explicit year/month ranges and bind them to the relevant date column."""
    text = f"{question}\n{evidence}"
    filters: List[Dict[str, Any]] = []
    col = _temporal_column_for_context(ctx, text, existing_filters)
    if col is None:
        return filters

    year_range = re.search(
        r"\b(?:between|from)\s+(?:the\s+year\s+)?((?:19|20)\d{2})\s+(?:and|to|through|-)\s+((?:19|20)\d{2})\b",
        question,
        re.I,
    )
    if year_range:
        lo = _year_comparison_value(year_range.group(1), col)
        hi = _year_comparison_value(year_range.group(2), col)
        filters.append(_cmp(_year_expr(col), "between", [lo, hi]))

    before = re.search(r"\b(?:born|dated|created|recorded)?\s*before\s+(?:the\s+year\s+)?((?:19|20)\d{2})\b", question, re.I)
    if before:
        filters.append(_cmp(_year_expr(col), "<", _year_comparison_value(before.group(1), col)))

    after = re.search(r"\b(?:born|dated|created|recorded)?\s*after\s+(?:the\s+year\s+)?((?:19|20)\d{2})\b", question, re.I)
    if after:
        filters.append(_cmp(_year_expr(col), ">", _year_comparison_value(after.group(1), col)))

    explicit_year = re.search(r"\b(?:birth\s*year|birthyear)\s+(?:of\s+)?((?:19|20)\d{2})\b", question, re.I)
    if explicit_year:
        filters.append(_cmp(_year_expr(col), "=", _year_comparison_value(explicit_year.group(1), col)))

    explicit_month = re.search(
        r"\b(?:birth\s*month|birthmonth)\s+(?:of\s+)?"
        r"(January|February|March|April|May|June|July|August|September|October|November|December|\d{1,2})\b",
        question,
        re.I,
    )
    if explicit_month:
        value = _month_value(explicit_month.group(1))
        filters.append(_cmp(_month_expr(col), "=", value))
    return _dedupe_bool(filters)


def _temporal_column_for_context(
    ctx: EvidenceContext,
    text: str,
    filters: Sequence[Dict[str, Any]],
) -> Optional[ColumnInfo]:
    question_text = text.split("\n", 1)[0]
    if re.search(r"\b(?:born|birth\w*)\b", question_text, re.I):
        birth_candidates = [
            col for col in ctx.columns
            if re.search(r"\bbirth", _norm(col.name)) and _column_is_temporal(col)
        ]
        if birth_candidates:
            return birth_candidates[0]
        birth = _age_source_date_column(ctx, text)
        if birth is not None:
            return birth

    candidates = [col for col in ctx.columns if _column_is_temporal(col)]
    if not candidates:
        return None
    text_norm = _norm(text)
    filter_tables = {ref.split(".", 1)[0] for ref in _filter_refs(filters) if "." in ref}
    metric = _best_numeric_measure_for_text(text, ctx)
    scored: List[Tuple[int, ColumnInfo]] = []
    for col in candidates:
        score = 0
        if any(_alias_in_text(alias, text_norm) for alias in col.aliases):
            score += 6
        if col.table in filter_tables:
            score += 8
        if metric is not None and col.table == metric.table:
            score += 16
        table_cols = [item for item in ctx.columns if item.table == col.table and item.ref != col.ref]
        score += 4 * sum(
            1 for item in table_cols
            if _column_mentioned_in_text(item, text_norm)
        )
        if ctx.primary_dates.get(col.table) == col.name:
            score += 2
        scored.append((score, col))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _column_mentioned_in_text(col: ColumnInfo, text: str) -> bool:
    text_norm = _norm(text)
    compact_text = _compact_identifier(text_norm)
    return any(
        len(alias) > 3
        and (
            _alias_in_text(alias, text_norm)
            or (_compact_identifier(alias) and _compact_identifier(alias) in compact_text)
        )
        for alias in col.aliases
    )


def _extract_percentage_question_filters(question: str, ctx: EvidenceContext) -> List[Dict[str, Any]]:
    if not re.search(r"\b(?:percent|percentage|ratio)\b", question, re.I):
        return []
    filters: List[Dict[str, Any]] = []

    target_values = []
    target_values.extend(re.findall(r"\bpercentage\s+of\s+\"([^\"]+)\"", question, flags=re.I))
    target_values.extend(re.findall(r"\bpercentage\s+of\s+'([^']+)'", question, flags=re.I))
    m = re.search(r"\bpercentage\s+of\s+(?:the\s+)?([A-Z][A-Z0-9_+-]{1,})\b", question)
    if m:
        target_values.append(m.group(1))
    m = re.search(r"\bused\s+([A-Z][A-Z0-9_+-]{1,})\b", question)
    if m:
        target_values.append(m.group(1))

    if not target_values:
        return []

    col = None
    m = re.search(r"\boverall\s+([A-Za-z_][A-Za-z0-9_-]*)", question, flags=re.I)
    if m:
        col = _resolve_ambiguous_overall_column(m.group(1), question, ctx)
        if col is None:
            col = _resolve_column(m.group(1), ctx)
    if col is None:
        # Last resort for cheap/no-model mode: if the question itself names a
        # schema column near the target literal, use that column.
        q_norm = _norm(question)
        candidates = [
            c for c in ctx.columns
            if any(_alias_in_text(alias, q_norm) for alias in c.aliases)
            and not _column_is_temporal(c)
        ]
        if len(candidates) == 1:
            col = candidates[0]
    if col is None:
        return []
    return [_column_cmp(col, "=", _parse_literal(target_values[0]))]


def _resolve_ambiguous_overall_column(phrase: str, question: str, ctx: EvidenceContext) -> Optional[ColumnInfo]:
    phrase_norm = _norm(phrase)
    candidates = [
        col for col in ctx.columns
        if phrase_norm and any(phrase_norm == alias or _alias_in_text(alias, phrase_norm) for alias in col.aliases)
    ]
    if len(candidates) <= 1:
        return candidates[0] if candidates else None
    q_norm = _norm(question)
    for candidate in candidates:
        table_cols = [c for c in ctx.columns if c.table == candidate.table]
        if any(any(_alias_in_text(alias, q_norm) for alias in c.aliases) for c in table_cols if c.name != candidate.name):
            return candidate
    return None


def _extract_primary_id_selects(
    question: str,
    ctx: EvidenceContext,
    filters: Sequence[Dict[str, Any]],
) -> List[Dict[str, str]]:
    q_norm = _norm(question)
    mentions_table = False
    for table in ctx.tables:
        table_norm = _norm(table)
        singular = table_norm[:-1] if table_norm.endswith("s") else table_norm
        plural = table_norm if table_norm.endswith("s") else f"{table_norm}s"
        if _alias_in_text(table_norm, q_norm) or _alias_in_text(singular, q_norm) or _alias_in_text(plural, q_norm):
            mentions_table = True
            break
    if not re.search(r"\b(?:ids?|records?)\b", question, re.I) and not (_selectish_question(question) and mentions_table):
        return []
    refs = list(_filter_refs(filters))
    scored: List[Tuple[int, ColumnInfo]] = []
    for table, pid in ctx.primary_ids.items():
        col = _column_by_table_name(ctx, table, pid)
        if col is None:
            continue
        score = 1
        if any(ref.startswith(f"{table}.") for ref in refs):
            score += 4
        table_norm = _norm(table)
        singular = table_norm[:-1] if table_norm.endswith("s") else table_norm
        plural = table_norm if table_norm.endswith("s") else f"{table_norm}s"
        if _alias_in_text(table_norm, q_norm) or _alias_in_text(singular, q_norm) or _alias_in_text(plural, q_norm):
            score += 3
        scored.append((score, col))
    if not scored:
        id_cols = [c for c in ctx.columns if c.name.lower() == "id"]
        scored = [(1, c) for c in id_cols]
    if not scored:
        return []
    scored.sort(key=lambda item: item[0], reverse=True)
    col = scored[0][1]
    return [{"ref": col.ref, "alias": _alias_for(col.name)}]


def _extract_boolean_answer(
    question: str,
    evidence: str,
    filters: Sequence[Dict[str, Any]],
    ctx: EvidenceContext,
) -> Optional[Dict[str, Any]]:
    if not re.match(r"\s*(?:did|was|were|is|are|does|do)\b", question, re.I):
        return None
    condition = None
    for lhs, rhs in _mapping_clauses(evidence):
        if not _conditionish(rhs):
            continue
        lhs_norm = _norm(lhs)
        if lhs_norm and lhs_norm not in _norm(question):
            continue
        condition = _condition_tree(rhs, ctx)
        if condition is not None:
            break
    if condition is not None:
        return {
            "select": {
                "expr": {"case": {"whens": [{"when": condition, "then": {"lit": True}}], "else": {"lit": False}}},
                "alias": "result",
            },
            "refs": list(_filter_refs([condition])),
            "filter_keys": _bool_keys(condition),
            "rowwise": True,
        }
    if filters:
        return {"select": {"expr": {"lit": "YES"}, "alias": "result"}, "refs": list(_filter_refs(filters)), "filter_keys": set()}
    return None


def _extract_per_unit_filters(question: str, evidence: str, ctx: EvidenceContext) -> List[Dict[str, Any]]:
    if not re.search(r"\bper\s+unit\b", f"{question}\n{evidence}", re.I):
        return []
    pair = _division_columns_from_text(evidence, ctx)
    if pair is None:
        return []
    numerator, denominator = pair
    m = re.search(r"\b(?:more\s+than|greater\s+than|over|above)\s+([+-]?\d+(?:\.\d+)?)\s+per\s+unit\b", question, re.I)
    op = ">"
    if not m:
        m = re.search(r"\b(?:less\s+than|under|below)\s+([+-]?\d+(?:\.\d+)?)\s+per\s+unit\b", question, re.I)
        op = "<"
    if not m:
        return []
    expr = _op("/", {"col": numerator.ref}, _func("nullif", {"col": denominator.ref}, {"lit": 0}))
    return [_cmp(expr, op, _parse_literal(m.group(1)))]


def _division_columns_from_text(text: str, ctx: EvidenceContext) -> Optional[Tuple[ColumnInfo, ColumnInfo]]:
    m = re.search(
        r"=\s*(?:total|sum)\s*\(\s*([A-Za-z0-9_.\"` -]+?)\s*\)\s*/\s*"
        r"(?:total|sum)\s*\(\s*([A-Za-z0-9_.\"` -]+?)\s*\)",
        text,
        re.I,
    )
    if m:
        left = _resolve_column(m.group(1), ctx)
        right = _resolve_column(m.group(2), ctx)
        if left is not None and right is not None:
            return left, right

    m = re.search(r"=\s*([A-Za-z0-9_.\"` -]+?)\s*/\s*([A-Za-z0-9_.\"` -]+?)(?:[.;\n]|$)", text, re.I)
    if not m:
        m = re.search(r"\bDIVIDE\s*\(\s*([A-Za-z0-9_.\"` -]+?)\s*,\s*([A-Za-z0-9_.\"` -]+?)\s*\)", text, re.I)
    if not m:
        return None
    left = _resolve_column(m.group(1), ctx)
    right = _resolve_column(m.group(2), ctx)
    if left is None or right is None:
        return None
    return left, right


def _resolve_identifier_column(phrase: str, ctx: EvidenceContext) -> Optional[ColumnInfo]:
    cleaned = _clean_column_phrase(phrase)
    candidates: List[str] = []
    parts = cleaned.split()
    if len(parts) > 1:
        tail = parts[-1]
        primary = _primary_id_for_table_phrase(tail, ctx)
        if primary is not None:
            return primary
        candidates.extend([tail, f"{tail} id", f"{tail} number", f"{tail} code"])
    primary = _primary_id_for_table_phrase(cleaned, ctx)
    if primary is not None:
        return primary
    candidates.extend([cleaned, f"{cleaned} id", f"{cleaned} number", f"{cleaned} code"])
    for candidate in candidates:
        col = _resolve_column(candidate, ctx)
        if col is not None:
            return col
    return None


def _extract_ranked_metric_values_plan(
    question: str,
    evidence: str,
    ctx: EvidenceContext,
    filters: Sequence[Dict[str, Any]],
    mapping_select: Sequence[Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    limit = _limit_for_question(question)
    if limit is None or limit <= 1:
        return None
    match = re.search(r"\b(MAX|MIN)\s*\(\s*([A-Za-z0-9_.\"` -]+?)\s*\)", evidence or "", re.I)
    if match is None:
        return None
    metric = _resolve_column_for_context(match.group(2), ctx, filters, f"{question}\n{evidence}")
    if metric is None or not _numeric_column(metric):
        return None
    requested_refs = {item.get("ref") for item in mapping_select}
    if metric.ref not in requested_refs and not _column_mentioned_in_text(metric, question):
        return None
    dataset = _choose_dataset(ctx, [metric.ref], filters, text=f"{question}\n{evidence}")
    if dataset is None:
        return None
    return _advanced(
        dataset,
        [{"expr": {"col": metric.ref}, "alias": _alias_for(metric.name)}],
        where=_and(filters) if filters else None,
        order_by=[{"by": {"col": metric.ref}, "dir": "asc" if match.group(1).lower() == "min" else "desc"}],
        limit=limit,
    )


def _extract_entity_extremum_plan(
    question: str,
    evidence: str,
    ctx: EvidenceContext,
    filters: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not re.match(r"\s*(?:who|which)\b", question, re.I):
        return None
    if re.search(r"\b(?:AVG|SUM|COUNT)\s*\(", evidence or "", re.I):
        return None
    match = re.search(r"\b(MAX|MIN)\s*\(\s*([A-Za-z0-9_.\"` -]+?)\s*\)", evidence or "", re.I)
    if match is None:
        return None
    metric = _resolve_column_for_context(match.group(2), ctx, filters, f"{question}\n{evidence}")
    if metric is None or not _numeric_column(metric):
        return None
    label = _implicit_entity_label_column(question, ctx, preferred_tables={metric.table})
    if label is None:
        return None
    dataset = _choose_dataset(ctx, [label.ref, metric.ref], filters, text=f"{question}\n{evidence}")
    if dataset is None:
        return None
    return _advanced(
        dataset,
        [{"expr": {"col": label.ref}, "alias": _alias_for(label.name)}],
        where=_and(filters) if filters else None,
        order_by=[{"by": {"col": metric.ref}, "dir": "asc" if match.group(1).lower() == "min" else "desc"}],
        limit=1,
    )


def _extract_extreme_group_comparison_plan(
    question: str,
    evidence: str,
    ctx: EvidenceContext,
    filters: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not re.search(r"\bbetween\b", question, re.I):
        return None
    nested = re.search(r"\b(?:MAX|MIN)\s*\(\s*AVG\s*\(\s*([A-Za-z0-9_.\"` -]+?)\s*\)\s*\)", evidence or "", re.I)
    extrema = re.findall(r"\b(MAX|MIN)\s*\(\s*([A-Za-z0-9_.\"` -]+?)\s*\)", evidence or "", re.I)
    if nested is None or len(extrema) < 2:
        return None
    metric = _resolve_column_for_context(nested.group(1), ctx, filters, f"{question}\n{evidence}")
    if metric is None or not _numeric_column(metric):
        return None
    dimension_candidates: List[Tuple[str, ColumnInfo]] = []
    for direction, raw in extrema:
        col = _resolve_column_for_context(raw, ctx, filters, f"{question}\n{evidence}")
        if col is None or col.ref == metric.ref or not _numeric_column(col):
            continue
        dimension_candidates.append((direction.lower(), col))
    by_direction = {direction: col for direction, col in dimension_candidates}
    if "max" not in by_direction or "min" not in by_direction:
        return None
    dimension = by_direction["max"]

    branches: List[Dict[str, Any]] = []
    for direction, label in (("max", "Max"), ("min", "Min")):
        subquery = _advanced(
            dimension.table,
            [{"expr": _func(direction, {"col": dimension.ref}), "alias": "value"}],
            limit=1,
        )
        branch_filters = list(filters) + [
            _cmp({"col": dimension.ref}, "=", {"scalar_subquery": {"plan": subquery}})
        ]
        dataset = _choose_dataset(ctx, [metric.ref, dimension.ref], branch_filters, text=f"{question}\n{evidence}")
        if dataset is None:
            return None
        branch = _advanced(
            dataset,
            [
                {"expr": _func("avg", {"col": metric.ref}), "alias": "result"},
                {"expr": {"lit": label}, "alias": "extreme_group"},
            ],
            where=_and(branch_filters),
            limit=None,
        )
        branches.append(_add_intent_aware_joins(branch, f"{question}\n{evidence}", ctx))

    union_plan = {
        "set_op": {"op": "union", "left": branches[0], "right": branches[1]},
        "order_by": [{"by": "result", "dir": "desc"}],
        "limit": 1,
        "offset": 0,
    }
    return {
        "version": "1.0",
        "with": [{"name": "extreme_groups", "plan": union_plan}],
        "dataset": "extreme_groups",
        "select": [{"expr": {"col": "extreme_group"}, "alias": "extreme_group"}],
        "limit": 1,
        "offset": 0,
    }


def _implicit_entity_label_column(
    question: str,
    ctx: EvidenceContext,
    *,
    preferred_tables: Optional[set[str]] = None,
) -> Optional[ColumnInfo]:
    preferred_tables = preferred_tables or set()
    q_norm = _norm(question)
    scored: List[Tuple[int, ColumnInfo]] = []
    for col in ctx.columns:
        name_norm = _norm(col.name)
        if name_norm != "name" and not name_norm.endswith(" name"):
            continue
        table_norm = _norm(col.table)
        singular = table_norm[:-1] if table_norm.endswith("s") else table_norm
        plural = table_norm if table_norm.endswith("s") else f"{table_norm}s"
        score = 0
        if col.table in preferred_tables:
            score += 10
        if any(_alias_in_text(item, q_norm) for item in {table_norm, singular, plural}):
            score += 20
        if name_norm != "name":
            score += 2
        if score:
            scored.append((score, col))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _question_requests_entity_label(question: str, ctx: EvidenceContext) -> bool:
    q_norm = _norm(question)
    for table in ctx.tables:
        table_norm = _norm(table)
        singular = table_norm[:-1] if table_norm.endswith("s") else table_norm
        plural = table_norm if table_norm.endswith("s") else f"{table_norm}s"
        entity = rf"(?:{re.escape(table_norm)}|{re.escape(singular)}|{re.escape(plural)})"
        if re.match(rf"\s*(?:who|which)\s+(?:[a-z]+\s+){{0,2}}{entity}\b", q_norm):
            return True
        if re.match(rf"\s*(?:list|name|identify)\b[^?.;,]{{0,40}}\b{entity}\b", q_norm):
            return True
    return False


def _extract_ranked_computed_report(
    question: str,
    evidence: str,
    ctx: EvidenceContext,
    filters: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    text = f"{question}\n{evidence}"
    if not re.search(r"\b(?:top|highest|largest|most|least|lowest|smallest)\b", text, re.I):
        return None
    if not re.search(r"\b(?:age|how\s+old)\b", question, re.I):
        return None
    extremum = re.search(r"\b(MAX|MIN)\s*\((.*?)\)", evidence or "", re.I)
    if not extremum:
        return None
    metric = _resolve_column_for_context(extremum.group(2), ctx, [], text)
    if metric is None:
        return None

    computed = _extract_computed_selects(question, evidence, ctx)
    if not computed:
        return None
    computed = _align_computed_to_metric_date(computed, metric, ctx)

    details = _extract_explicit_question_selects(question, ctx, filters)
    computed_refs = {ref for item in computed for ref in item.get("refs", []) if isinstance(ref, str)}
    details = [
        item
        for item in details
        if item.get("ref") not in computed_refs and item.get("ref") != metric.ref
    ]
    details = _prefer_details_from_refs(details, computed_refs, ctx)

    select_items = [{"expr": item["expr"], "alias": item["alias"]} for item in computed]
    select_items.extend({"expr": {"col": item["ref"]}, "alias": item["alias"]} for item in details)
    if len(select_items) <= len(computed):
        return None

    metric_filter = _cmp({"col": metric.ref}, "is_not_null", True)
    all_filters = _dedupe_bool(list(filters) + [metric_filter])
    refs = [metric.ref] + list(computed_refs) + [item["ref"] for item in details] + list(_filter_refs(all_filters))
    dataset = _choose_dataset(ctx, refs, all_filters, text=text)
    if dataset is None:
        return None
    direction = "asc" if extremum.group(1).lower() == "min" or re.search(r"\b(?:least|lowest|smallest)\b", question, re.I) else "desc"
    return _advanced(
        dataset,
        select_items,
        where=_and(all_filters),
        order_by=[{"by": {"col": metric.ref}, "dir": direction}],
        limit=1,
    )


def _align_computed_to_metric_date(
    computed: Sequence[Dict[str, Any]],
    metric: ColumnInfo,
    ctx: EvidenceContext,
) -> List[Dict[str, Any]]:
    metric_date = _date_column_for_table(ctx, metric.table)
    if metric_date is None:
        return [dict(item) for item in computed]

    def rewrite(expr: Any) -> Any:
        if isinstance(expr, list):
            return [rewrite(item) for item in expr]
        if not isinstance(expr, dict):
            return expr
        if (
            expr.get("op") == "-"
            and len(expr.get("args", [])) == 2
            and isinstance(expr["args"][0], dict)
            and expr["args"][0].get("func") == "date_part"
        ):
            return _op("-", _year_expr(metric_date), rewrite(expr["args"][1]))
        return {key: rewrite(value) for key, value in expr.items()}

    out: List[Dict[str, Any]] = []
    for item in computed:
        updated = dict(item)
        updated["expr"] = rewrite(item["expr"])
        refs = [ref for ref in item.get("refs", []) if isinstance(ref, str)]
        if refs:
            first_col = _column_by_ref(ctx, refs[0])
            if first_col is not None and _column_is_temporal(first_col):
                refs = refs[1:]
        refs.insert(0, metric_date.ref)
        updated["refs"] = _dedupe_strings(refs)
        out.append(updated)
    return out


def _prefer_details_from_refs(
    details: Sequence[Dict[str, str]],
    refs: set[str],
    ctx: EvidenceContext,
) -> List[Dict[str, str]]:
    preferred_tables = {ref.split(".", 1)[0] for ref in refs if "." in ref}
    by_alias: Dict[str, List[Dict[str, str]]] = {}
    for item in details:
        by_alias.setdefault(item.get("alias", item.get("ref", "")), []).append(item)
    out: List[Dict[str, str]] = []
    for items in by_alias.values():
        if len(items) == 1:
            out.extend(items)
            continue
        preferred = next(
            (
                item
                for item in items
                if isinstance(item.get("ref"), str) and item["ref"].split(".", 1)[0] in preferred_tables
            ),
            None,
        )
        out.append(preferred or items[0])
    return _dedupe_selects(out)


def _extract_ranked_entity_report(
    question: str,
    evidence: str,
    ctx: EvidenceContext,
    filters: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not re.search(r"\b(top|highest|largest|most|least|lowest|smallest)\b", question, re.I):
        return None
    if not re.search(r"\b(who|which|what)\b", question, re.I):
        return None
    if re.search(
        r"\bwhat\s+(?:is|was|are|were)\s+(?:the\s+)?(?:top|highest|largest|most|least|lowest|smallest)\s+"
        r"(?:amount|value|cost|price|total|average|number|count)\b",
        question,
        re.I,
    ):
        return None

    entity_col = _ranked_entity_column(question, ctx)
    if entity_col is None:
        return None

    metric = _ranked_report_metric(question, evidence, ctx)
    if metric is None:
        return None

    direction = "asc" if re.search(r"\b(least|lowest|smallest)\b", question, re.I) else "desc"
    metric_refs = list(metric["refs"])
    ranking_measure = _ranking_measure_for_text(
        f"{question}\n{evidence}",
        ctx,
        entity_col,
        exclude_refs=set(metric_refs),
    )
    if ranking_measure is None:
        return None
    ranking_entity_col = _entity_column_on_table(entity_col, ranking_measure.table, ctx)
    if ranking_entity_col is None:
        return None

    detail_selects = _ranked_detail_selects(question, ctx, filters, entity_col, metric_refs)
    select_items = [
        {"expr": {"col": entity_col.ref}, "alias": _alias_for(entity_col.name)},
        {"expr": metric["expr"], "alias": metric["alias"]},
    ]
    for detail in detail_selects:
        select_items.append({"expr": {"col": detail.ref}, "alias": _alias_for(detail.name)})

    group_by = [{"col": entity_col.ref}] + [{"col": detail.ref} for detail in detail_selects]
    subquery = {
        "version": "1.0",
        "dataset": ranking_measure.table,
        "select": [{"expr": {"col": ranking_entity_col.ref}, "alias": _alias_for(ranking_entity_col.name)}],
        "order_by": [{"by": {"col": ranking_measure.ref}, "dir": direction}],
        "limit": 1,
        "offset": 0,
    }
    ranked_filter = _cmp(
        {"col": entity_col.ref},
        "=",
        {"scalar_subquery": {"plan": subquery}},
    )
    all_filters = list(filters) + [ranked_filter]
    refs = [entity_col.ref] + metric_refs + [d.ref for d in detail_selects] + list(_filter_refs(all_filters))
    dataset = _choose_dataset(ctx, refs, all_filters, text=f"{question}\n{evidence}")
    if dataset is None:
        return None
    return _advanced(
        dataset,
        select_items,
        where=_and(all_filters),
        order_by=[],
        limit=1,
    ) | {"group_by": group_by}


def _extract_ranked_dimension_plan(
    question: str,
    evidence: str,
    ctx: EvidenceContext,
    filters: Sequence[Dict[str, Any]],
    mapping_select: Sequence[Dict[str, str]],
    metric_expr: Optional[Dict[str, Any]],
    metric_refs: Sequence[str],
) -> Optional[Dict[str, Any]]:
    if metric_expr is None:
        return None
    text = f"{question}\n{evidence}"
    if not re.search(r"\b(top|highest|largest|most|least|lowest|smallest|maximum|minimum)\b", text, re.I):
        return None
    if not re.search(r"\b(which|what|who|name|identify|indicate|state|list)\b", question, re.I):
        return None
    metric_ref_set = set(metric_refs)
    selects = list(mapping_select) + _extract_question_selects(question, ctx, filters)
    output: List[Dict[str, str]] = []
    for item in _dedupe_selects(selects):
        ref = item["ref"]
        if ref in metric_ref_set:
            continue
        col = _column_by_ref(ctx, ref)
        if col is not None and _numeric_column(col):
            continue
        output.append(item)
    if not output:
        if not re.match(r"\s*(?:who|which)\b", question, re.I):
            return None
        preferred_tables = {ref.split(".", 1)[0] for ref in metric_refs if "." in ref}
        label = _implicit_entity_label_column(question, ctx, preferred_tables=preferred_tables)
        if label is None:
            return None
        output.append({"ref": label.ref, "alias": _alias_for(label.name)})
    refs = [s["ref"] for s in output] + list(metric_refs) + list(_filter_refs(filters))
    dataset = _choose_dataset(ctx, refs, filters, text=text)
    if dataset is None:
        return None
    direction = "asc" if re.search(r"\b(least|lowest|smallest|minimum|MIN\s*\()\b", text, re.I) else "desc"
    plan = _advanced(
        dataset,
        [{"expr": {"col": s["ref"]}, "alias": s["alias"]} for s in output],
        where=_and(filters) if filters else None,
        order_by=[{"by": metric_expr, "dir": direction}],
        limit=_limit_for_question(question) or 1,
    )
    if _expr_has_aggregate(metric_expr):
        plan["group_by"] = _dedupe_exprs(
            [{"col": s["ref"]} for s in output] + _group_identity_refs(output, ctx)
        )
    return plan


def _extract_ranked_temporal_answer(
    question: str,
    evidence: str,
    ctx: EvidenceContext,
    filters: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not re.search(r"\b(?:youngest|oldest|earliest|latest|most recent)\b", question, re.I):
        return None
    if not re.search(r"\b(?:when|date|born|birthday|birth date)\b", question, re.I):
        return None
    if re.search(r"\b(?:age|how old)\b", question, re.I) or re.search(r"[,;]\s*and\b|\band\s+what\b", question, re.I):
        return None
    text = f"{question}\n{evidence}"
    col = _age_source_date_column(ctx, text) if re.search(r"\b(?:youngest|oldest|born|birthday|birth date)\b", question, re.I) else _date_column_for_text(ctx, text)
    if col is None:
        return None
    dataset = _choose_dataset(ctx, [col.ref], filters, text=text)
    if dataset is None:
        return None
    direction = "asc" if re.search(r"\b(?:oldest|earliest)\b", question, re.I) else "desc"
    return _advanced(
        dataset,
        [{"expr": {"col": col.ref}, "alias": _alias_for(col.name)}],
        where=_and(filters) if filters else None,
        order_by=[{"by": {"col": col.ref}, "dir": direction}],
        limit=1,
    )


def _extract_explicit_ranked_aggregate_plan(
    question: str,
    evidence: str,
    ctx: EvidenceContext,
    filters: Sequence[Dict[str, Any]],
    mapping_select: Sequence[Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    """Build a grouped top/bottom query from explicit nested aggregate evidence."""
    text = f"{question}\n{evidence}"
    match = re.search(r"\b(MAX|MIN)\s*\(\s*(SUM|COUNT)\s*\((.*?)\)\s*\)", evidence or "", re.I | re.S)
    if match is None or not re.search(r"\b(?:most|least|highest|lowest|largest|smallest|top|maximum|minimum)\b", text, re.I):
        return None

    outer, aggregate, inner = match.group(1).lower(), match.group(2).lower(), match.group(3).strip()
    metric_refs: List[str] = []
    if aggregate == "count":
        count_col = _resolve_count_aggregate_column(inner, ctx, text) if inner and inner != "*" else None
        if count_col is not None:
            metric_refs = [count_col.ref]
            metric_expr = _func("count", {"col": count_col.ref})
        else:
            metric_expr = _func("count")
    else:
        condition = _condition_tree(inner, ctx) if _conditionish(inner) else None
        if condition is not None:
            metric_refs = list(_filter_refs([condition]))
            metric_expr = _case_sum(condition, {"lit": 1})
        else:
            cols = [
                _resolve_column_for_context(part, ctx, filters, text)
                for part in _split_top_level_args(inner)
                if part.strip()
            ]
            cols = [col for col in cols if col is not None]
            if not cols:
                return None
            metric_refs = [col.ref for col in cols]
            row_expr: Dict[str, Any] = {"col": cols[0].ref}
            for col in cols[1:]:
                row_expr = _op("+", row_expr, {"col": col.ref})
            metric_expr = _func("sum", row_expr)

    mapped_refs = {item["ref"] for item in mapping_select if isinstance(item.get("ref"), str)}
    candidates = _dedupe_selects(list(mapping_select) + _extract_question_selects(question, ctx, filters))
    filter_refs = _filter_refs(filters)
    output: List[Dict[str, str]] = []
    for item in candidates:
        ref = item.get("ref")
        col = _column_by_ref(ctx, ref) if isinstance(ref, str) else None
        if col is None or ref in metric_refs:
            continue
        if _numeric_column(col) or _id_like_column(col):
            continue
        if ref in filter_refs and ref not in mapped_refs:
            continue
        output.append(item)
    if not output:
        return None

    refs = [item["ref"] for item in output] + metric_refs + list(filter_refs)
    dataset = _choose_dataset(ctx, refs, filters, text=text)
    if dataset is None:
        return None
    select_items = [{"expr": {"col": item["ref"]}, "alias": item["alias"]} for item in output]
    if aggregate == "count" and re.search(r"\b(?:how many|number of|count)\b", question, re.I):
        select_items.append({"expr": metric_expr, "alias": "count"})
    plan = _advanced(
        dataset,
        select_items,
        where=_and(filters) if filters else None,
        order_by=[{"by": metric_expr, "dir": "asc" if outer == "min" else "desc"}],
        limit=1,
    )
    plan["group_by"] = [{"col": item["ref"]} for item in output]
    return plan


def _resolve_count_aggregate_column(raw: str, ctx: EvidenceContext, text: str) -> Optional[ColumnInfo]:
    candidates = _resolve_column_candidates(raw, ctx)
    if not candidates:
        return _resolve_column_for_context(raw, ctx, [], text)
    scored: List[Tuple[int, ColumnInfo]] = []
    raw_norm = _norm(_clean_column_phrase(raw))
    text_norm = _norm(text)
    for col in candidates:
        score = 0
        if _norm(col.name) == raw_norm:
            score += 8
        if ctx.primary_ids.get(col.table) != col.name:
            score += 10
        if _alias_in_text(_norm(col.table), text_norm):
            score += 6
        score += min(5, sum(1 for item in ctx.columns if item.table == col.table and not _id_like_column(item)))
        scored.append((score, col))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _extract_most_common_dimension_plan(
    question: str,
    evidence: str,
    ctx: EvidenceContext,
    filters: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    text = f"{question}\n{evidence}"
    if not re.search(r"\bmost\s+common\b|\bhighest\s+count\b|\blargest\s+count\b", text, re.I):
        return None

    columns: List[ColumnInfo] = []
    for lhs, rhs in _mapping_clauses(evidence):
        if not _mapping_lhs_requested(lhs, question):
            continue
        m = re.search(r"(?:MAX\s*\(\s*)?COUNT\s*\(\s*([A-Za-z0-9_.\"` -]+?)\s*\)\s*\)?", rhs, re.I)
        if not m:
            continue
        col = _resolve_column_for_context(m.group(1), ctx, filters, text)
        if col is not None:
            columns.append(col)

    if not columns:
        for m in re.finditer(
            r"\bmost\s+common\s+([A-Za-z][A-Za-z0-9_ -]{1,60}?)(?:\s+they\b|\s+with\b|\s+for\b|\s+among\b|\s+in\b|[?.;,]|$)",
            question,
            re.I,
        ):
            col = _resolve_column_for_context(m.group(1), ctx, filters, text)
            if col is not None:
                columns.append(col)

    if not columns:
        return None

    col = columns[0]
    refs = [col.ref] + list(_filter_refs(filters))
    dataset = _choose_dataset(ctx, refs, filters, text=text)
    if dataset is None:
        return None
    plan = _advanced(
        dataset,
        [{"expr": {"col": col.ref}, "alias": _alias_for(col.name)}],
        where=_and(filters) if filters else None,
        order_by=[{"by": _func("count", {"col": col.ref}), "dir": "desc"}],
        limit=1,
    )
    plan["group_by"] = [{"col": col.ref}]
    return plan


def _extract_aggregate_dimension_plan(
    question: str,
    evidence: str,
    ctx: EvidenceContext,
    filters: Sequence[Dict[str, Any]],
    mapping_select: Sequence[Dict[str, str]],
    aggregate: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not re.search(r"\b(?:list|identify|show|provide|give|state|which|what)\b", question, re.I):
        return None
    metric_ref = aggregate.get("ref")
    if not isinstance(metric_ref, str):
        return None
    dimension_selects = list(mapping_select) + _extract_question_selects(question, ctx, filters)
    filter_refs = _filter_refs(filters)
    output: List[Dict[str, str]] = []
    for item in _dedupe_selects(dimension_selects):
        ref = item.get("ref")
        if not isinstance(ref, str) or ref == metric_ref or ref in filter_refs:
            continue
        col = _column_by_ref(ctx, ref)
        if col is None:
            continue
        if _numeric_column(col) and not _id_like_column(col):
            continue
        output.append(item)
    if not output:
        return None

    select_items = [{"expr": {"col": s["ref"]}, "alias": s["alias"]} for s in output]
    select_items.append(aggregate["select"])
    refs = [s["ref"] for s in output] + [metric_ref] + list(filter_refs)
    dataset = _choose_dataset(ctx, refs, filters, text=f"{question}\n{evidence}")
    if dataset is None:
        return None
    direction = _aggregate_order_direction(question, evidence)
    order_by = [{"by": aggregate["select"]["expr"], "dir": direction}] if direction else None
    plan = _advanced(
        dataset,
        select_items,
        where=_and(filters) if filters else None,
        order_by=order_by,
        limit=None,
    )
    plan["group_by"] = [{"col": s["ref"]} for s in output]
    return plan


def _aggregate_order_direction(question: str, evidence: str) -> Optional[str]:
    text = f"{question}\n{evidence}"
    if re.search(r"\b(?:ascending|lowest|smallest|least|minimum)\b", text, re.I):
        return "asc"
    if re.search(r"\b(?:descending|highest|largest|most|maximum|top)\b", text, re.I):
        return "desc"
    return None


def _entity_select_needs_distinct(
    selects: Sequence[Dict[str, str]],
    filters: Sequence[Dict[str, Any]],
    ctx: EvidenceContext,
) -> bool:
    select_tables = {
        item["ref"].split(".", 1)[0]
        for item in selects
        if isinstance(item.get("ref"), str) and "." in item["ref"]
    }
    if len(select_tables) != 1:
        return False
    table = next(iter(select_tables))
    if table not in ctx.primary_ids:
        return False
    selected_refs = {item["ref"] for item in selects if isinstance(item.get("ref"), str)}
    filter_tables = {
        ref.split(".", 1)[0]
        for ref in _filter_refs(filters)
        if "." in ref
    }
    if not (filter_tables - {table}):
        return False
    if _primary_id_ref(table, ctx) in selected_refs:
        return True
    selected_cols = [_column_by_ref(ctx, ref) for ref in selected_refs]
    return bool(selected_cols) and all(
        col is not None and not _numeric_column(col) and not _column_is_temporal(col)
        for col in selected_cols
    )


def _answer_not_null_filters(
    selects: Sequence[Dict[str, str]],
    filters: Sequence[Dict[str, Any]],
    ctx: EvidenceContext,
) -> List[Dict[str, Any]]:
    existing_refs = _filter_refs(filters)
    out: List[Dict[str, Any]] = []
    for item in selects:
        ref = item.get("ref")
        if not isinstance(ref, str) or ref in existing_refs:
            continue
        col = _column_by_ref(ctx, ref)
        if col is None or _id_like_column(col):
            continue
        out.append(_cmp({"col": ref}, "is_not_null", True))
    return out


def _ranked_entity_column(question: str, ctx: EvidenceContext) -> Optional[ColumnInfo]:
    q_norm = _norm(question)
    scored: List[Tuple[int, ColumnInfo]] = []
    for table, pid in ctx.primary_ids.items():
        col = _column_by_table_name(ctx, table, pid)
        if col is None:
            continue
        table_norm = _norm(table)
        singular = table_norm[:-1] if table_norm.endswith("s") else table_norm
        plural = table_norm if table_norm.endswith("s") else f"{table_norm}s"
        score = 0
        if _alias_in_text(plural, q_norm):
            score += 10
        elif _alias_in_text(table_norm, q_norm) or _alias_in_text(singular, q_norm):
            score += 8
        if any(_alias_in_text(alias, q_norm) for alias in col.aliases):
            score += 4
        if score:
            scored.append((score, col))
    if not scored:
        id_cols = [
            col for col in ctx.columns
            if col.name.lower().endswith("id") and any(_alias_in_text(alias, q_norm) for alias in col.aliases)
        ]
        scored = [(1, col) for col in id_cols]
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _ranked_report_metric(question: str, evidence: str, ctx: EvidenceContext) -> Optional[Dict[str, Any]]:
    text = f"{question}\n{evidence}"
    pair = _division_columns_from_text(text, ctx)
    if pair is not None and re.search(r"\bper\s+(?:single\s+)?(?:unit|item|record|entity)\b", text, re.I):
        numerator, denominator = pair
        row_ratio = _op("/", {"col": numerator.ref}, _func("nullif", {"col": denominator.ref}, {"lit": 0}))
        return {
            "expr": _func("sum", row_ratio),
            "alias": f"sum_{_alias_for(numerator.name)}_per_{_alias_for(denominator.name)}",
            "refs": [numerator.ref, denominator.ref],
        }

    aggregate = _extract_simple_aggregate(question, evidence, ctx)
    if aggregate is not None:
        return {
            "expr": aggregate["select"]["expr"],
            "alias": aggregate["select"]["alias"],
            "refs": [aggregate["ref"]],
        }
    return None


def _ranking_measure_for_text(
    text: str,
    ctx: EvidenceContext,
    entity_col: ColumnInfo,
    *,
    exclude_refs: set[str],
) -> Optional[ColumnInfo]:
    text_norm = _norm(text)
    spendish = re.search(r"\b(spend|spending|spent|paid|pay|usage|used|revenue|sales)\b", text_norm) is not None
    candidates = [
        col for col in ctx.columns
        if _numeric_column(col)
        and col.ref not in exclude_refs
        and not _id_like_column(col)
        and col.name not in ctx.primary_ids.values()
        and _entity_column_on_table(entity_col, col.table, ctx) is not None
    ]
    if not candidates:
        candidates = [
            col for col in ctx.columns
            if _numeric_column(col) and not _id_like_column(col) and col.name not in ctx.primary_ids.values()
        ]
    if not candidates:
        return None

    scored: List[Tuple[int, ColumnInfo]] = []
    for col in candidates:
        name_norm = _norm(col.name)
        score = 0
        if any(_alias_in_text(alias, text_norm) for alias in col.aliases):
            score += 6
        if spendish:
            if re.search(r"\b(usage|used|consumption)\b", name_norm):
                score += 12
            if re.search(r"\b(spend|spent|paid|revenue|sales|total|cost|price|amount)\b", name_norm):
                score += 7
        if col.table == entity_col.table:
            score += 1
        else:
            score += 2
        scored.append((score, col))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _entity_column_on_table(entity_col: ColumnInfo, table: str, ctx: EvidenceContext) -> Optional[ColumnInfo]:
    same_name = _column_by_table_name(ctx, table, entity_col.name)
    if same_name is not None:
        return same_name
    entity_aliases = set(entity_col.aliases)
    for col in ctx.columns:
        if col.table != table:
            continue
        if set(col.aliases) & entity_aliases:
            return col
    if table == entity_col.table:
        return entity_col
    return None


def _ranked_detail_selects(
    question: str,
    ctx: EvidenceContext,
    filters: Sequence[Dict[str, Any]],
    entity_col: ColumnInfo,
    metric_refs: Sequence[str],
) -> List[ColumnInfo]:
    ignored = {entity_col.ref, *metric_refs, *_filter_refs(filters)}
    details: List[ColumnInfo] = []
    for item in _extract_question_selects(question, ctx, filters):
        ref = item["ref"]
        if ref in ignored:
            continue
        col = _column_by_ref(ctx, ref)
        if col is None or _numeric_column(col):
            continue
        details.append(col)
    return _dedupe_columns(details)


def _dedupe_columns(columns: Sequence[ColumnInfo]) -> List[ColumnInfo]:
    seen: set[str] = set()
    out: List[ColumnInfo] = []
    for col in columns:
        if col.ref in seen:
            continue
        seen.add(col.ref)
        out.append(col)
    return out


def _primary_id_for_table_phrase(phrase: str, ctx: EvidenceContext) -> Optional[ColumnInfo]:
    phrase_norm = _norm(phrase)
    matches: List[ColumnInfo] = []
    for table, pid in ctx.primary_ids.items():
        table_norm = _norm(table)
        singular = table_norm[:-1] if table_norm.endswith("s") else table_norm
        if _alias_in_text(table_norm, phrase_norm) or _alias_in_text(singular, phrase_norm):
            col = _column_by_table_name(ctx, table, pid)
            if col is not None:
                matches.append(col)
    return matches[-1] if matches else None


def _full_name_columns(evidence: str, ctx: EvidenceContext) -> Optional[Tuple[ColumnInfo, ColumnInfo]]:
    for lhs, rhs in _mapping_clauses(evidence):
        if not re.search(r"\bfull\s+name\b", lhs, re.I):
            continue
        cols = []
        for part in _split_column_list(rhs):
            col = _resolve_column(part, ctx)
            if col is not None:
                cols.append(col)
        first = next((c for c in cols if "first" in c.name.lower()), None)
        last = next((c for c in cols if "last" in c.name.lower()), None)
        if first is not None and last is not None:
            return first, last

    first = _resolve_column("first name", ctx)
    last = _resolve_column("last name", ctx)
    if first is not None and last is not None and re.search(r"\bfull\s+name\b", evidence, re.I):
        return first, last
    return None


def _extract_represented_filters(question: str, evidence: str, ctx: EvidenceContext) -> List[Dict[str, Any]]:
    filters: List[Dict[str, Any]] = []
    all_text = f"{question}\n{evidence}"

    for m in re.finditer(
        r"([A-Za-z0-9_.\"` -]+?)\s+can\s+be\s+represented\s+as\s+the\s+([A-Za-z0-9_.\"` -]+?)\s+value(?:\s+in\s+the\s+([A-Za-z0-9_.\"` -]+?)\s+table)?\s+is\s+'([^']+)'",
        evidence,
        flags=re.I,
    ):
        col = _resolve_column(m.group(2), ctx, table_hint=m.group(3))
        if col is not None:
            filters.append(_cmp({"col": col.ref}, "=", _parse_literal(m.group(4))))

    for m in re.finditer(
        r"([A-Za-z0-9_.\"` -]+?)\s+can\s+be\s+represented\s+as\s+([A-Za-z0-9_.\"` -]+?)\s+BETWEEN\s+'([^']+)'\s+AND\s+'([^']+)'",
        evidence,
        flags=re.I,
    ):
        col = _resolve_column(m.group(2), ctx)
        if col is not None:
            filters.append(_cmp({"col": col.ref}, "between", [_parse_literal(m.group(3)), _parse_literal(m.group(4))]))

    for m in re.finditer(r"'([^']+)'\s+can\s+be\s+represented\s+by\s+'([^']+)'", evidence, flags=re.I):
        original, normalized = m.group(1), m.group(2)
        if original and original not in question and normalized not in question:
            continue
        literal = _parse_literal(normalized)
        if _looks_like_date(literal):
            col = _date_column_for_text(ctx, all_text)
        elif _looks_like_time(literal):
            col = _time_column_for_text(ctx, all_text)
        else:
            col = None
        if col is not None:
            filters.append(_cmp({"col": col.ref}, "=", literal))

    # Many benchmark/app questions put the literal time in the question and only
    # normalize the date in evidence. Bind those literals to obvious time/date
    # columns when the schema has them.
    time_range = _time_range(question)
    if time_range:
        col = _time_column_for_text(ctx, all_text)
        if col is not None:
            filters.append(_cmp({"col": col.ref}, "between", time_range))
    else:
        time_value = _time(question)
        if time_value:
            col = _time_column_for_text(ctx, all_text)
            if col is not None:
                filters.append(_cmp({"col": col.ref}, "=", time_value))

    date_value = _date_any_order(question)
    if (
        date_value
        and not _question_has_date_range(question)
        and not _question_has_relative_date_comparison(question)
        and not _date_value_explicitly_mapped(date_value, evidence)
        and not _question_date_explicitly_mapped(question, evidence)
    ):
        col = _date_column_for_text(ctx, all_text)
        if col is not None:
            filters.append(_cmp({"col": col.ref}, "=", date_value))

    return filters


def _question_date_explicitly_mapped(question: str, evidence: str) -> bool:
    for raw in re.findall(r"\b[0-9]{1,4}[/-][0-9]{1,2}[/-][0-9]{1,4}\b", question or ""):
        if _date_value_explicitly_mapped(_norm_date_text(raw), evidence):
            return True
    return False


def _date_value_explicitly_mapped(date_value: str, evidence: str) -> bool:
    normalized = _norm_date_text(date_value)
    for raw in re.findall(r"'([0-9]{1,4}[/-][0-9]{1,2}[/-][0-9]{1,4})'", evidence or ""):
        if _norm_date_text(raw) != normalized:
            continue
        if re.search(rf"{re.escape(raw)}\s+refers?\s*(?:to)?\s+[A-Za-z0-9_.\"` -]+\s*=", evidence or "", re.I):
            return True
    return False


def _norm_date_text(value: str) -> str:
    parsed = _date_any_order(str(value or ""))
    return parsed or str(value or "").strip()


def _extract_literal_value_filters(
    question: str,
    evidence: str,
    ctx: EvidenceContext,
    *,
    existing_filters: Sequence[Dict[str, Any]] = (),
) -> List[Dict[str, Any]]:
    filters: List[Dict[str, Any]] = []
    if not ctx.value_index:
        return filters

    # Evidence literals are already scoped by explicit conditions and mappings.
    # Only use the user's question for the broad value-index fallback.
    existing_text = question
    existing_values = {
        _norm(value)
        for filt in existing_filters
        for value in [_single_filter_value(filt)]
        if value is not None
    }
    values = _candidate_literal_values(existing_text)
    for value in values:
        parsed = _parse_literal(value)
        if _looks_like_date(parsed) or _looks_like_time(parsed) or _looks_like_compact_period(parsed):
            continue
        if len(_norm(value)) <= 1:
            continue
        if _resolve_column(value, ctx) is not None:
            continue
        if _norm(value) in existing_values:
            continue
        if any(part in existing_values for part in _norm(value).split()):
            continue
        for table, cols in ctx.value_index.items():
            for col_name, known_values in cols.items():
                match = _match_known_value(value, known_values)
                if match is None:
                    continue
                col = _column_by_table_name(ctx, table, col_name)
                if col is None:
                    continue
                if not _literal_type_fits_column(col, parsed):
                    continue
                filters.append(_cmp({"col": col.ref}, "=", match))
    return filters


def _extract_mapped_selects(
    question: str,
    evidence: str,
    ctx: EvidenceContext,
    filters: Sequence[Dict[str, Any]],
) -> List[Dict[str, str]]:
    selects: List[Dict[str, str]] = _extract_output_directive_selects(evidence, ctx)
    filter_refs = _filter_refs(filters)

    for lhs, rhs in _mapping_clauses(evidence):
        if not _mapping_lhs_requested(lhs, question):
            continue
        if _mapping_lhs_filter_context(lhs, question):
            continue
        select_rhs = re.split(r"\bwhere\b", rhs, maxsplit=1, flags=re.I)[0].strip()
        extremum = re.search(r"\b(MAX|MIN)\s*\((.*?)\)", select_rhs, re.I)
        if extremum:
            select_rhs = extremum.group(2)
        if any(op in select_rhs for op in ("=", ">", "<")) or re.search(r"\bbetween\b|\bcount\b|\bavg\b|\bsum\b", select_rhs, re.I):
            continue
        for part in _split_column_list(select_rhs):
            col = _resolve_column_for_context(part, ctx, filters, f"{question}\n{evidence}\n{lhs}\n{rhs}")
            if col is not None:
                if col.ref in filter_refs and not _mapping_lhs_names_output(lhs):
                    continue
                item = {"ref": col.ref, "alias": _alias_for(col.name)}
                lhs_pos = _norm(question).find(_norm(lhs))
                if lhs_pos >= 0:
                    item["pos"] = lhs_pos
                selects.append(item)

    return _dedupe_selects(selects)


def _extract_output_directive_selects(evidence: str, ctx: EvidenceContext) -> List[Dict[str, str]]:
    selects: List[Dict[str, str]] = []
    for clause in re.split(r";|\n", evidence or ""):
        match = re.search(
            r"\b(?:final\s+result|result|output|answer)\b[^;]*?\b(?:return|include|show|provide|give)\b\s+(.+)$",
            clause,
            re.I,
        )
        if match is None:
            continue
        for part in _split_column_list(match.group(1)):
            col = _resolve_column(part, ctx)
            if col is not None:
                selects.append({"ref": col.ref, "alias": _alias_for(col.name)})
    return _dedupe_selects(selects)


def _extract_question_selects(
    question: str,
    ctx: EvidenceContext,
    filters: Sequence[Dict[str, Any]],
) -> List[Dict[str, str]]:
    matches = _question_select_matches(question, ctx, filters)
    output_context = _output_context_columns(matches, question, filters, ctx)
    if output_context:
        return _dedupe_selects([{"ref": c.ref, "alias": _alias_for(c.name)} for c in output_context])
    if _question_requests_entity_label(question, ctx):
        label = _implicit_entity_label_column(question, ctx)
        if label is not None:
            return [{"ref": label.ref, "alias": _alias_for(label.name)}]

    # Avoid returning every generic column named "id" or "date" when the question
    # does not ask for a list/detail answer.
    if not _selectish_question(question):
        matches = [c for c in matches if len(c.name) > 3]

    matches = _disambiguate_columns_by_table_text(matches, question)
    return _dedupe_selects([{"ref": c.ref, "alias": _alias_for(c.name)} for c in matches])


def _extract_explicit_question_selects(
    question: str,
    ctx: EvidenceContext,
    filters: Sequence[Dict[str, Any]],
) -> List[Dict[str, str]]:
    matches = _question_select_matches(question, ctx, filters)
    output_context = _output_context_columns(matches, question, filters, ctx)
    return _dedupe_selects([{"ref": c.ref, "alias": _alias_for(c.name)} for c in output_context])


def _extract_requested_filter_selects(
    question: str,
    ctx: EvidenceContext,
    filters: Sequence[Dict[str, Any]],
) -> List[Dict[str, str]]:
    filter_refs = _filter_refs(filters)
    columns = [
        col
        for col in (_column_by_ref(ctx, ref) for ref in filter_refs)
        if col is not None
        and not _numeric_column(col)
        and not _column_is_temporal(col)
        and not _filter_column_is_named_constraint(question, col, filters)
    ]
    requested = _output_context_columns(columns, question, [], ctx)
    return _dedupe_selects([{"ref": col.ref, "alias": _alias_for(col.name)} for col in requested])


def _question_select_matches(
    question: str,
    ctx: EvidenceContext,
    filters: Sequence[Dict[str, Any]],
) -> List[ColumnInfo]:
    q_norm = _norm(question)
    filter_refs = _filter_refs(filters)
    matches: List[ColumnInfo] = []
    for col in ctx.columns:
        if col.ref in filter_refs:
            continue
        if any(_alias_in_question(alias, question, q_norm) for alias in col.aliases) or _general_synonym_hits(col, q_norm):
            matches.append(col)
    return matches


def _filter_column_is_named_constraint(
    question: str,
    col: ColumnInfo,
    filters: Sequence[Dict[str, Any]],
) -> bool:
    values = [
        _single_filter_value(filt)
        for filt in filters
        if _single_filter_ref(filt) == col.ref
    ]
    values = [value for value in values if value]
    if not values:
        return False
    aliases = [alias for alias in col.aliases if alias and len(alias) > 2]
    if not aliases:
        return False
    for alias in aliases:
        if re.search(rf"(?:\"[^\"]+\"|'[^']+')\s+{re.escape(alias)}\b", question, re.I):
            return True
        if re.search(rf"\b(?:of|in|from|for|with)\s+(?:the\s+)?(?:\"[^\"]+\"|'[^']+')[^?.;,]{{0,40}}\b{re.escape(alias)}\b", question, re.I):
            return True
    return False


def _alias_in_question(alias: str, question: str, q_norm: str) -> bool:
    found = _alias_in_text(alias, q_norm)
    if not found and " " not in alias:
        found = _token_variant_in_text(alias, q_norm)
    if not found:
        return False
    if _query_command_phrase(alias) and re.match(rf"\s*{re.escape(alias)}\b", question or "", re.I):
        return False
    return True


def _alias_variants(alias: str) -> List[str]:
    variants = [alias]
    if " " not in alias and len(alias) > 3:
        if alias.endswith("y"):
            variants.append(f"{alias[:-1]}ies")
        elif alias.endswith("s"):
            variants.append(alias[:-1])
        else:
            variants.append(f"{alias}s")
    return _dedupe_strings(variants)


def _extract_formula_select(
    question: str,
    evidence: str,
    ctx: EvidenceContext,
    filters: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    text = f"{question}\n{evidence}"
    formula_expr = _extract_formula_expression(text)
    if not formula_expr:
        return None

    parsed = _formula_expr(formula_expr, ctx, text)
    if parsed is None:
        return None
    expr = parsed["expr"]
    if re.search(r"\bpercentage\b|%\s*$|\*\s*100", text, re.I) and not parsed.get("scaled"):
        expr = _op("*", expr, {"lit": 100.0})
    return {
        "select": {"expr": expr, "alias": "value"},
        "refs": parsed["refs"] + list(_filter_refs(filters)),
        "filter_keys": parsed["filter_keys"],
        "condition_refs": parsed.get("condition_refs") or set(),
    }


def _extract_infix_aggregate_ratio_select(
    question: str,
    evidence: str,
    ctx: EvidenceContext,
    filters: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    match = re.search(
        r"\b((?:SUM|AVG|COUNT)\s*\([^()]+\))\s*/\s*((?:SUM|AVG|COUNT)\s*\([^()]+\))",
        evidence or "",
        re.I,
    )
    if match is None:
        return None
    left = _aggregate_or_condition(match.group(1), ctx, f"{question}\n{evidence}")
    right = _aggregate_or_condition(match.group(2), ctx, f"{question}\n{evidence}")
    if left is None or right is None:
        return None
    return {
        "select": {"expr": _ratio_expr(left["expr"], right["expr"]), "alias": "value"},
        "refs": left["refs"] + right["refs"] + list(_filter_refs(filters)),
        "filter_keys": set(left["filter_keys"]) | set(right["filter_keys"]),
        "condition_refs": set(left.get("condition_refs") or set()) | set(right.get("condition_refs") or set()),
    }


def _defer_scalar_formula(question: str, evidence: str, threshold: Optional[Dict[str, Any]]) -> bool:
    if re.search(r"\bhow\s+many\s+times\b", question, re.I) and _extract_formula_expression(evidence):
        return False
    if threshold is not None and _selectish_question(question):
        return True
    if _selectish_question(question) and re.search(
        r"(?:>=|<=|>|<)\s*(?:DIVIDE|DIVISION|SUBTRACT|MULTIPLY)\s*\(",
        evidence or "",
        re.I,
    ):
        return True
    if re.search(r"\b(?:percentage|ratio|calculation|difference)\s*=", evidence or "", re.I) and _extract_formula_expression(evidence):
        return False
    if _selectish_question(question) and re.search(
        r"\b(?:SUBTRACT|DIVIDE|DIVISION|MULTIPLY)\s*\([^;]+?\)\s*(?:=|>=|<=|>|<)",
        evidence or "",
        re.I | re.S,
    ):
        return True
    if _asks_for_count(question) and re.search(r"\bcalculation\s*=", evidence or "", re.I) and _extract_formula_expression(evidence):
        return False
    if _asks_for_count(question):
        return True
    if not re.search(r"\bage\b|\bhow\s+old\b", f"{question}\n{evidence}", re.I):
        return False
    if re.search(r"[,;]\s*and\b|\band\s+what\b|\band\s+(?:their|the)\s+(?:age|name|date)\b", question, re.I):
        return True
    return bool(re.search(r"\b(?:state|provide|list|name|identify)\b", question, re.I))


def _has_filter_only_aggregate(question: str, evidence: str) -> bool:
    text = f"{question}\n{evidence}"
    if not _selectish_question(question):
        return False
    return re.search(r"(?:=|>=|<=|>|<)\s*(?:AVG|SUM|MIN|MAX|COUNT)\s*\(", text, re.I) is not None


def _extract_computed_selects(
    question: str,
    evidence: str,
    ctx: EvidenceContext,
) -> List[Dict[str, Any]]:
    text = f"{question}\n{evidence}"
    if not re.search(r"\b(?:age|how old)\b", question, re.I):
        return []
    diff = _date_difference_expr(text, ctx)
    if diff is None:
        return []
    return [{"expr": diff["expr"], "alias": "age", "refs": diff["refs"]}]


def _extract_year_change_formula(
    question: str,
    evidence: str,
    ctx: EvidenceContext,
    filters: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not re.search(r"\b(?:decrease|increase|change)\s+rate\b", question, re.I):
        return None
    years = re.findall(r"\b(?:19|20)\d{2}\b", question)
    if len(years) < 2:
        return None
    old_year, new_year = years[0], years[-1]
    measure = _best_numeric_measure_for_text(f"{question}\n{evidence}", ctx)
    if measure is None:
        return None
    date_col = _date_column_for_table(ctx, measure.table) or _date_column_for_text(ctx, f"{question}\n{evidence}")
    if date_col is None:
        return None
    old_sum = _case_sum(_cmp(_year_expr(date_col), "=", old_year if _text_date_like(date_col) else int(old_year)), {"col": measure.ref})
    new_sum = _case_sum(_cmp(_year_expr(date_col), "=", new_year if _text_date_like(date_col) else int(new_year)), {"col": measure.ref})
    if re.search(r"\bincrease\b", question, re.I):
        diff = _op("-", new_sum, old_sum)
    else:
        diff = _op("-", old_sum, new_sum)
    return {
        "select": {"expr": _ratio_expr(diff, old_sum), "alias": "value"},
        "refs": [measure.ref, date_col.ref] + list(_filter_refs(filters)),
        "metric_refs": [measure.ref, date_col.ref],
        "filter_keys": set(),
    }


def _extract_year_difference_formula(
    question: str,
    evidence: str,
    ctx: EvidenceContext,
    filters: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    text = f"{question}\n{evidence}"
    if not re.search(r"\bdifference\b", question, re.I):
        return None
    if not re.search(r"\b(?:total|sum|amount|spent|cost|value)\b", question, re.I):
        return None
    years = re.findall(r"\b(?:19|20)\d{2}\b", question)
    if len(years) < 2:
        return None
    left_year, right_year = years[0], years[-1]
    measure = _money_measure_for_text(text, ctx)
    if measure is None:
        return None
    date_col = _date_column_for_table(ctx, measure.table) or _date_column_for_text(ctx, text)
    if date_col is None:
        return None
    left_value: Any = left_year if _text_date_like(date_col) else int(left_year)
    right_value: Any = right_year if _text_date_like(date_col) else int(right_year)
    left_sum = _case_sum(_cmp(_year_expr(date_col), "=", left_value), {"col": measure.ref})
    right_sum = _case_sum(_cmp(_year_expr(date_col), "=", right_value), {"col": measure.ref})
    return {
        "select": {"expr": _op("-", left_sum, right_sum), "alias": "value"},
        "refs": [measure.ref, date_col.ref] + list(_filter_refs(filters)),
        "metric_refs": [measure.ref, date_col.ref],
        "filter_keys": set(),
    }


def _extract_spent_period_formula(
    question: str,
    evidence: str,
    ctx: EvidenceContext,
    filters: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not re.search(r"\b(?:amount|total|how much)\b.*\bspent\b|\bspent\b.*\b(?:amount|total|how much)\b", question, re.I):
        return None
    period_filter = next((f for f in filters if _filter_has_compact_period(f)), None)
    if period_filter is None:
        return None
    measure = _money_measure_for_text(f"{question}\n{evidence}", ctx)
    if measure is None:
        return None
    total = _func("sum", {"col": measure.ref})
    period_cond = period_filter
    period = _case_sum(period_cond, {"col": measure.ref})
    return {
        "select": [
            {"expr": total, "alias": f"sum_{_alias_for(measure.name)}"},
            {"expr": period, "alias": f"period_sum_{_alias_for(measure.name)}"},
        ],
        "refs": [measure.ref] + list(_filter_refs(filters)),
        "filter_keys": {_bool_key(period_filter)},
    }


def _extract_percentage_from_filters(
    question: str,
    filters: Sequence[Dict[str, Any]],
    ctx: EvidenceContext,
) -> Optional[Dict[str, Any]]:
    if not re.search(r"\b(percent|percentage)\b", question, re.I) or not filters:
        return None

    target_terms = [_norm(v) for v in re.findall(r"\bpercentage\s+of\s+\"([^\"]+)\"", question, flags=re.I)]
    target_terms.extend(_norm(v) for v in re.findall(r"\bpercentage\s+of\s+'([^']+)'", question, flags=re.I))
    m = re.search(r"\bpercentage\s+of\s+([A-Za-z0-9_+-]+)", question, flags=re.I)
    if m:
        token = _norm(m.group(1))
        if token not in {"the", "a", "an", "all", "overall", "entities", "people", "records", "rows"}:
            target_terms.append(token)

    target_filters: List[Dict[str, Any]] = []
    base_filters: List[Dict[str, Any]] = []
    for filt in filters:
        col = _single_filter_column(filt, ctx)
        value = _single_filter_value(filt)
        value_norm = _norm(value) if value is not None else ""
        if target_terms and value_norm in target_terms:
            target_filters.append(filt)
            continue
        if col is not None and _column_is_temporal(col):
            base_filters.append(filt)
        elif not target_terms:
            target_filters.append(filt)
        else:
            base_filters.append(filt)

    if not target_filters:
        return None

    numerator_cond = _and(list(base_filters) + list(target_filters)) if base_filters else _and(target_filters)
    denominator = _func("count")
    return {
        "select": {"expr": _pct_expr(_case_sum(numerator_cond, {"lit": 1}), denominator), "alias": "pct"},
        "where": _and(base_filters) if base_filters else None,
    }


def _extract_count_threshold(evidence: str, ctx: EvidenceContext) -> Optional[Dict[str, Any]]:
    m = re.search(r"\bCOUNT\s*\(\s*([A-Za-z0-9_.\"` -]+?)\s*\)\s*(>=|<=|>|<|=)\s*([0-9]+)", evidence or "", re.I)
    if not m:
        return None
    col = _resolve_column(m.group(1), ctx)
    if col is None:
        return None
    op = m.group(2)
    value = int(m.group(3))
    if op == ">" and re.search(r"\bor\s+more\b|\bat\s+least\b", evidence or "", re.I):
        op = ">="
    return {"col": col, "op": op, "value": value}


def _threshold_count_ref(col: ColumnInfo, text: str, ctx: EvidenceContext) -> Optional[str]:
    if col.name.startswith("link_to_"):
        return col.ref
    text_norm = _norm(text)
    candidates: List[Tuple[int, ColumnInfo]] = []
    for link in ctx.schema.get("links", []) or []:
        if not isinstance(link, dict):
            continue
        for on in link.get("on", []) or []:
            left = on.get("left")
            right = on.get("right")
            if not isinstance(left, str) or not isinstance(right, str):
                continue
            pairs = [(left, right), (right, left)]
            for source_ref, target_ref in pairs:
                if target_ref != col.ref:
                    continue
                source_col = _column_by_ref(ctx, source_ref)
                if source_col is None:
                    continue
                score = 1
                if _alias_in_text(_norm(source_col.table), text_norm):
                    score += 6
                if any(_alias_in_text(alias, text_norm) for alias in source_col.aliases):
                    score += 3
                candidates.append((score, source_col))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1].ref


def _group_identity_refs(selects: Sequence[Dict[str, str]], ctx: EvidenceContext) -> List[Dict[str, Any]]:
    refs = {s["ref"] for s in selects if isinstance(s.get("ref"), str)}
    out: List[Dict[str, Any]] = []
    for ref in refs:
        table = ref.split(".", 1)[0] if "." in ref else ""
        pid = _primary_id_ref(table, ctx) if table else None
        if pid and pid not in refs:
            out.append({"col": pid})
    return out


def _dedupe_exprs(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for item in items:
        key = repr(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _expr_has_aggregate(expr: Any) -> bool:
    if isinstance(expr, list):
        return any(_expr_has_aggregate(item) for item in expr)
    if not isinstance(expr, dict):
        return False
    if str(expr.get("func") or "").lower() in {"count", "count_distinct", "sum", "avg", "min", "max"}:
        return True
    return any(_expr_has_aggregate(value) for value in expr.values())


def _aggregate_or_condition(text: str, ctx: EvidenceContext, context_text: str = "") -> Optional[Dict[str, Any]]:
    text = text.strip()
    average = re.search(r"\bAVG\s*\((.*?)\)\s*(?:where\s+(.+))?$", text, flags=re.I | re.S)
    if average:
        metric_text, cond_text = _split_metric_where(average.group(1))
        if average.group(2):
            cond_text = average.group(2)
        metric = _resolve_column(metric_text, ctx) or _resolve_column_for_context(metric_text, ctx, [], context_text or text)
        if metric is None or not _numeric_column(metric):
            return None
        cond = _condition_tree(cond_text, ctx) if cond_text else None
        cond = _align_condition_boundaries(cond, context_text or text, ctx)
        if cond is not None:
            refs = [metric.ref] + list(_filter_refs([cond]))
            return {
                "expr": _ratio_expr(
                    _case_sum(cond, {"col": metric.ref}),
                    _case_sum(cond, {"lit": 1}),
                ),
                "refs": refs,
                "filter_keys": _bool_keys(cond),
                "condition_refs": _filter_refs([cond]),
            }
        return {"expr": _func("avg", {"col": metric.ref}), "refs": [metric.ref], "filter_keys": set()}

    count = re.search(r"\bCOUNT\s*\((.*?)\)\s*(?:where\s+(.+))?$", text, flags=re.I | re.S)
    if count:
        inner = count.group(1)
        cond = _condition_tree(count.group(2) or inner, ctx) if count.group(2) else None
        cond = _align_condition_boundaries(cond, context_text or text, ctx)
        refs = list(_filter_refs([cond] if cond else []))
        if cond:
            return {
                "expr": _case_sum(cond, {"lit": 1}),
                "refs": refs,
                "filter_keys": _bool_keys(cond),
                "condition_refs": set(refs),
            }
        col = _resolve_column(inner, ctx)
        if col is not None:
            return {"expr": _func("count", {"col": col.ref}), "refs": [col.ref], "filter_keys": set()}
        return {"expr": _func("count"), "refs": [], "filter_keys": set()}

    sum_match = re.search(r"\bSUM\s*\((.*?)\)\s*(?:where\s+(.+))?$", text, flags=re.I | re.S)
    if sum_match:
        inner = sum_match.group(1)
        then_match = re.match(r"(.+?)\s+THEN\s+(.+)$", inner, flags=re.I | re.S)
        if then_match:
            cond_text = then_match.group(1).strip()
            metric_text = then_match.group(2).strip()
            metric = _resolve_column(metric_text, ctx) or _resolve_column_for_context(metric_text, ctx, [], context_text or text)
            cond = _condition_tree(cond_text, ctx)
            cond = _align_condition_boundaries(cond, context_text or text, ctx)
            if metric is not None and cond is not None:
                refs = [metric.ref] + list(_filter_refs([cond]))
                return {
                    "expr": _case_sum(cond, {"col": metric.ref}),
                    "refs": refs,
                    "filter_keys": _bool_keys(cond),
                    "condition_refs": _filter_refs([cond]),
                }
        metric_text, cond_text = _split_metric_where(inner)
        if sum_match.group(2):
            cond_text = sum_match.group(2)
        if not cond_text and _conditionish(metric_text):
            cond = _condition_tree(metric_text, ctx)
            cond = _align_condition_boundaries(cond, context_text or text, ctx)
            if cond:
                refs = list(_filter_refs([cond]))
                return {
                    "expr": _case_sum(cond, {"lit": 1}),
                    "refs": refs,
                    "filter_keys": _bool_keys(cond),
                    "condition_refs": set(refs),
                }
        nested = _formula_expr(metric_text, ctx, text) if re.search(r"\b(?:DIVIDE|DIVISION|SUBTRACT|MULTIPLY|YEAR)\s*\(", metric_text, re.I) else None
        if nested is not None and not cond_text:
            return {
                "expr": _func("sum", nested["expr"]),
                "refs": nested["refs"],
                "filter_keys": nested["filter_keys"],
                "condition_refs": nested.get("condition_refs") or set(),
            }
        metric = _resolve_column(metric_text, ctx) or _resolve_column_for_context(metric_text, ctx, [], context_text or text)
        if metric is None or not _numeric_column(metric):
            return None
        cond = _condition_tree(cond_text, ctx) if cond_text else None
        cond = _align_condition_boundaries(cond, context_text or text, ctx)
        if cond:
            refs = [metric.ref] + list(_filter_refs([cond]))
            return {
                "expr": _case_sum(cond, {"col": metric.ref}),
                "refs": refs,
                "filter_keys": _bool_keys(cond),
                "condition_refs": _filter_refs([cond]),
            }
        return {"expr": _func("sum", {"col": metric.ref}), "refs": [metric.ref], "filter_keys": set()}

    cond = _condition_tree(text, ctx)
    cond = _align_condition_boundaries(cond, context_text or text, ctx)
    if cond:
        refs = list(_filter_refs([cond]))
        return {
            "expr": _case_sum(cond, {"lit": 1}),
            "refs": refs,
            "filter_keys": _bool_keys(cond),
            "condition_refs": set(refs),
        }
    return None


def _extract_simple_aggregate(question: str, evidence: str, ctx: EvidenceContext) -> Optional[Dict[str, Any]]:
    text = f"{question}\n{evidence}"
    question_average = re.search(r"\b(?:average|avg)\b", question, re.I) is not None
    agg_match = None
    if question_average:
        agg_match = re.search(
            r"\baverage\s+(?:total\s+)?([A-Za-z_][A-Za-z0-9_ -]*?)(?:\s+that|\s+for|\s+of|\s+by|\s+in|[?.;]|$)",
            question,
            flags=re.I,
        )
        if not agg_match:
            agg_match = re.search(r"\baverage\b(?:\s+[A-Za-z0-9_-]+){0,5}?\s+([A-Za-z0-9_.\"` -]+)", evidence, flags=re.I)
        agg_name = "avg"
    else:
        agg_match = re.search(r"\b(AVG|AVERAGE|SUM|MIN|MAX)\s*\(\s*([A-Za-z0-9_.\"` -]+?)(?:\s+where\b[^)]*)?\)", text, flags=re.I)
    if not agg_match and re.search(r"\baverage\b", question, re.I):
        agg_name = "avg"
    elif not agg_match and re.search(r"\btotal\b", question, re.I):
        agg_match = re.search(
            r"\btotal\s+([A-Za-z_][A-Za-z0-9_ -]*?)(?:\s+that|\s+which|\s+for|\s+of|\s+by|\s+in|[?.;]|$)",
            question,
            flags=re.I,
        )
        agg_name = "sum"
    elif not question_average:
        agg_name = (agg_match.group(1).lower() if agg_match else "").replace("average", "avg")

    if not agg_match:
        return None

    col_text = agg_match.group(2 if not question_average and agg_match.lastindex and agg_match.lastindex >= 2 else 1)
    col = _resolve_column(col_text, ctx)
    if col is None:
        col = _resolve_column_for_context(col_text, ctx, [], text)
    if question_average and (col is None or not _numeric_column(col)):
        explicit = re.search(
            r"\b(?:AVG|AVERAGE)\s*\(\s*([A-Za-z0-9_.\"` -]+?)(?:\s+where\b[^)]*)?\)",
            evidence or "",
            re.I,
        )
        if explicit:
            col = _resolve_column_for_context(explicit.group(1), ctx, [], text)
    if col is None:
        return None
    agg = "avg" if agg_name == "average" else agg_name
    if agg not in {"avg", "sum", "min", "max"}:
        return None
    if agg in {"avg", "sum", "min", "max"} and not _numeric_column(col):
        return None
    return {
        "select": {"expr": _func(agg, {"col": col.ref}), "alias": f"{agg}_{_alias_for(col.name)}"},
        "ref": col.ref,
    }


def _extract_average_age_aggregate(question: str, evidence: str, ctx: EvidenceContext) -> Optional[Dict[str, Any]]:
    if not re.search(r"\b(?:average|avg)\s+age\b|\bage\b[^?.;]*\b(?:average|avg)\b", question, re.I):
        return None
    text = f"{question}\n{evidence}"
    date_col = _age_source_date_column(ctx, text)
    if date_col is None:
        return None
    expr = _op("-", _func("date_part", {"lit": "year"}, _func("now")), _year_expr(date_col))
    return {
        "select": {"expr": _func("avg", expr), "alias": "avg_age"},
        "ref": date_col.ref,
        "refs": [date_col.ref],
    }


def _age_source_date_column(ctx: EvidenceContext, text: str) -> Optional[ColumnInfo]:
    text_norm = _norm(text)
    scored: List[Tuple[int, ColumnInfo]] = []
    for table, date_name in ctx.primary_dates.items():
        col = _column_by_table_name(ctx, table, date_name)
        if col is None:
            continue
        score = 5
        table_norm = _norm(table)
        singular = table_norm[:-1] if table_norm.endswith("s") else table_norm
        plural = table_norm if table_norm.endswith("s") else f"{table_norm}s"
        if _alias_in_text(table_norm, text_norm) or _alias_in_text(singular, text_norm) or _alias_in_text(plural, text_norm):
            score += 5
        if table in ctx.primary_ids:
            score += 3
        scored.append((score, col))
    if not scored:
        return _date_column_for_text(ctx, text)
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _extract_formula_expression(text: str) -> Optional[str]:
    for name in ("DIVIDE", "DIVISION", "SUBTRACT", "MULTIPLY"):
        idx = re.search(rf"\b{name}\s*\(", text, re.I)
        if not idx:
            continue
        start = idx.start()
        expr = _balanced_call_from(text[start:])
        if not expr:
            tail = text[start:]
            end = re.search(r"(?:;|\n|$)", tail)
            raw = tail[: end.start()] if end else tail
            if raw.count("(") > raw.count(")"):
                raw = raw + (")" * (raw.count("(") - raw.count(")")))
            expr = raw.strip()
        elif _formula_arg_count(expr) < 2:
            tail = text[start + len(expr):]
            end = re.search(r"(?:;|\n|$)", tail)
            raw_tail = tail[: end.start()] if end else tail
            raw = expr[:-1] + raw_tail
            if raw.count("(") > raw.count(")"):
                raw = raw + (")" * (raw.count("(") - raw.count(")")))
            if _formula_arg_count(raw) >= 2:
                expr = raw.strip()
        if expr:
            suffix = text[start + len(expr): start + len(expr) + 20]
            if re.match(r"\s*\*\s*100", suffix):
                expr = f"MULTIPLY({expr}, 100)"
            return expr
    return None


def _formula_arg_count(expr: str) -> int:
    m = re.match(r"^[A-Za-z]+\s*\((.*)\)$", expr.strip(), re.S)
    if not m:
        return 0
    return len(_split_top_level_args(m.group(1)))


def _balanced_call_from(text: str) -> Optional[str]:
    m = re.match(r"\s*([A-Za-z]+)\s*\(", text)
    if not m:
        return None
    depth = 0
    for idx, ch in enumerate(text[m.start():], start=m.start()):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[: idx + 1].strip()
    return None


def _formula_expr(text: str, ctx: EvidenceContext, context_text: str = "") -> Optional[Dict[str, Any]]:
    text = _strip_wrapping_parentheses(text.strip().strip(".;"))
    literal = _literal_number_arg(text)
    if literal is not None:
        return {"expr": {"lit": literal}, "refs": [], "filter_keys": set()}
    call = re.match(r"^(DIVIDE|DIVISION|SUBTRACT|MULTIPLY)\s*\((.*)\)$", text, re.I | re.S)
    if call:
        name = call.group(1).lower()
        args = _split_top_level_args(_strip_wrapping_parentheses(call.group(2)))
        if len(args) < 2:
            return None
        if name == "subtract" and re.fullmatch(r"(?:DATETIME|NOW|CURRENT_TIMESTAMP)\s*\(\s*\)|CURRENT_TIMESTAMP", _strip_wrapping_parentheses(args[0]), re.I):
            date_col = _resolve_column_for_context(_strip_wrapping_parentheses(args[1]), ctx, [], context_text or text)
            if date_col is not None and (_column_is_temporal(date_col) or re.search(r"\bbirth", date_col.name, re.I)):
                return {
                    "expr": _op(
                        "-",
                        _func("date_part", {"lit": "year"}, _func("now")),
                        {"cast": {"expr": _year_expr(date_col), "type": "float"}},
                    ),
                    "refs": [date_col.ref],
                    "filter_keys": set(),
                }
        left = _formula_expr(args[0], ctx, context_text)
        right = _formula_expr(args[1], ctx, context_text)
        if left is None or right is None:
            return None
        refs = left["refs"] + right["refs"]
        keys = set(left["filter_keys"]) | set(right["filter_keys"])
        condition_refs = set(left.get("condition_refs") or set()) | set(right.get("condition_refs") or set())
        if name in {"divide", "division"}:
            return {
                "expr": _ratio_expr(left["expr"], right["expr"]),
                "refs": refs,
                "filter_keys": keys,
                "condition_refs": condition_refs,
                "scaled": bool(left.get("scaled")) or bool(right.get("scaled")),
            }
        if name == "subtract":
            return {
                "expr": _op("-", left["expr"], right["expr"]),
                "refs": refs,
                "filter_keys": keys,
                "condition_refs": condition_refs,
            }
        if name == "multiply":
            scaled = _literal_number_arg(args[1]) == 100 or bool(left.get("scaled")) or bool(right.get("scaled"))
            return {
                "expr": _op("*", left["expr"], right["expr"]),
                "refs": refs,
                "filter_keys": keys,
                "condition_refs": condition_refs,
                "scaled": scaled,
            }

    year_call = re.match(r"^YEAR\s*\((.*)\)$", text, re.I | re.S)
    if year_call:
        inner = year_call.group(1).strip()
        if re.fullmatch(r"(?:NOW|CURRENT_TIMESTAMP)\s*\(\s*\)|CURRENT_TIMESTAMP", inner, re.I):
            return {"expr": _func("date_part", {"lit": "year"}, _func("now")), "refs": [], "filter_keys": set()}
        col = _resolve_column(inner, ctx) or _resolve_column_for_context(inner, ctx, [], context_text)
        if col is not None:
            return {"expr": _year_expr(col), "refs": [col.ref], "filter_keys": set()}

    metric_text, cond_text = _split_metric_where(text)
    if cond_text and metric_text != text and not re.search(r"\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\(", metric_text, re.I):
        metric = _resolve_column_for_context(metric_text, ctx, [], context_text)
        cond = _condition_tree(cond_text, ctx)
        if metric is not None and cond is not None:
            return {
                "expr": _case_sum(cond, {"col": metric.ref}),
                "refs": [metric.ref] + list(_filter_refs([cond])),
                "filter_keys": _bool_keys(cond),
                "condition_refs": _filter_refs([cond]),
            }

    aggregate = _aggregate_or_condition(text, ctx, context_text)
    if aggregate is not None:
        return aggregate

    number = _literal_number_arg(text)
    if number is not None:
        return {"expr": {"lit": number}, "refs": [], "filter_keys": set()}

    col = _resolve_column(text, ctx)
    if col is None:
        col = _resolve_column_for_context(text, ctx, [], context_text)
    if col is not None:
        return {"expr": {"col": col.ref}, "refs": [col.ref], "filter_keys": set()}
    return None


def _strip_wrapping_parentheses(text: str) -> str:
    text = text.strip()
    while text.startswith("(") and text.endswith(")"):
        depth = 0
        wraps = True
        for idx, char in enumerate(text):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0 and idx != len(text) - 1:
                    wraps = False
                    break
        if not wraps or depth != 0:
            break
        text = text[1:-1].strip()
    return text


def _literal_number_arg(text: str) -> Optional[float]:
    value = text.strip().strip("'\"")
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value):
        return None
    parsed = float(value)
    return int(parsed) if parsed.is_integer() else parsed


def _split_metric_where(text: str) -> Tuple[str, str]:
    parts = re.split(r"\bwhere\b", text, maxsplit=1, flags=re.I)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    m = re.match(r"(.+?)\s+when\s+(.+)", text, flags=re.I | re.S)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return text.strip(), ""


def _extract_ordering(question: str, evidence: str, ctx: EvidenceContext) -> List[Dict[str, Any]]:
    text = f"{question}\n{evidence}"
    direction = None
    if re.search(r"\b(descending|highest|largest|youngest|younger|latest|most recent|top|maximum)\b", text, re.I):
        direction = "desc"
    elif re.search(r"\b(ascending|lowest|smallest|oldest|older|earliest|minimum)\b", text, re.I):
        direction = "asc"
    if direction is None:
        return []

    if re.search(r"\b(?:youngest|oldest|younger|older)\b", question, re.I) or re.search(r"\blarger\b.+\byounger\b", evidence or "", re.I):
        col = _age_source_date_column(ctx, text)
        if col is not None:
            if re.search(r"\b(?:youngest|younger)\b", question, re.I):
                direction = "desc"
            elif re.search(r"\b(?:oldest|older)\b", question, re.I):
                direction = "asc"
            return [{"by": {"col": col.ref}, "dir": direction}]

    for lhs, rhs in _mapping_clauses(evidence):
        if not _mapping_lhs_requested(lhs, question):
            continue
        extremum = re.search(r"\b(MAX|MIN)\s*\((.*?)\)", rhs, re.I)
        if not extremum:
            continue
        col = _resolve_column_for_context(extremum.group(2), ctx, [], text)
        if col is not None:
            mapped_dir = "asc" if extremum.group(1).lower() == "min" else "desc"
            return [{"by": {"col": col.ref}, "dir": mapped_dir}]

    candidates = _extract_question_selects(text, ctx, [])
    if not candidates:
        date_col = _best_date_column(ctx)
        if date_col is not None:
            candidates = [{"ref": date_col.ref, "alias": _alias_for(date_col.name)}]
    if not candidates:
        return []
    return [{"by": {"col": candidates[0]["ref"]}, "dir": direction}]


def _mapping_clauses(evidence: str) -> Iterable[Tuple[str, str]]:
    for clause in re.split(r";|\n", evidence):
        for part in re.split(r",\s*(?=[^,;]+?\s+(?:refers?\s*(?:to)?|means?)\s+)", clause):
            m = re.search(r"(.+?)\s+(?:refers?\s*(?:to)?|means?)\s+(.+)", part, flags=re.I)
            if m:
                yield m.group(1).strip(" ."), m.group(2).strip(" .")


def _rewrite_filter_value_aliases(
    filters: Sequence[Dict[str, Any]],
    evidence: str,
    ctx: EvidenceContext,
) -> List[Dict[str, Any]]:
    global_aliases, scoped_aliases = _evidence_value_aliases(evidence, ctx)
    if not global_aliases and not scoped_aliases:
        return list(filters)

    def rewrite(node: Any) -> Any:
        if isinstance(node, list):
            return [rewrite(item) for item in node]
        if not isinstance(node, dict):
            return node
        cmp_node = node.get("cmp")
        if isinstance(cmp_node, dict):
            updated = dict(cmp_node)
            left = updated.get("left")
            ref = left.get("col") if isinstance(left, dict) else None
            aliases = dict(global_aliases)
            if isinstance(ref, str):
                aliases.update(scoped_aliases.get(ref, {}))

            def rewrite_value(value: Any) -> Any:
                if isinstance(value, list):
                    return [rewrite_value(item) for item in value]
                if isinstance(value, (str, int, float)):
                    return aliases.get(_norm(value), value)
                return rewrite(value)

            updated["right"] = rewrite_value(updated.get("right"))
            return {"cmp": updated}
        return {key: rewrite(value) for key, value in node.items()}

    return _dedupe_bool([rewrite(filt) for filt in filters])


def _evidence_value_aliases(
    evidence: str,
    ctx: EvidenceContext,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    global_aliases: Dict[str, Any] = {}
    scoped_aliases: Dict[str, Dict[str, Any]] = {}
    quoted = r"(?:'([^']*)'|\"([^\"]*)\")"

    for lhs, rhs in _mapping_clauses(evidence):
        clean_lhs = re.sub(r"^(?:and|or)\s*", "", lhs.strip(), flags=re.I)
        lhs_literal = re.fullmatch(quoted, clean_lhs)
        rhs_literal = re.fullmatch(quoted, rhs.strip())
        if lhs_literal and rhs_literal:
            source = lhs_literal.group(1) if lhs_literal.group(1) is not None else lhs_literal.group(2)
            target = rhs_literal.group(1) if rhs_literal.group(1) is not None else rhs_literal.group(2)
            global_aliases[_norm(source)] = target
            continue

        scoped = re.fullmatch(rf"(.+?)\s*=\s*{quoted}", clean_lhs, re.I)
        if scoped and rhs_literal:
            col = _resolve_column(scoped.group(1), ctx)
            if col is None:
                continue
            stored = scoped.group(2) if scoped.group(2) is not None else scoped.group(3)
            source = rhs_literal.group(1) if rhs_literal.group(1) is not None else rhs_literal.group(2)
            scoped_aliases.setdefault(col.ref, {})[_norm(source)] = stored

    return global_aliases, scoped_aliases


def _mapping_lhs_requested(lhs: str, question: str) -> bool:
    lhs_norm = _norm(lhs)
    q_norm = _norm(question)
    if not lhs_norm:
        return False
    if re.search(r"\b(?:final\s+result|result|output|answer)\b.*\b(?:return|include|show|provide|give)\b", lhs_norm, re.I):
        return True
    if lhs_norm in q_norm:
        return True
    if lhs_norm == "full name" and re.search(r"\b(list|name|names|who|which|write|give|provide|mention)\b", question, re.I):
        return True
    stop = {
        "the", "a", "an", "of", "to", "in", "on", "for", "by", "with", "and",
        "or", "is", "are", "was", "were", "refers", "refer", "means", "mean",
        "all", "that", "which", "who", "whose", "value",
    }
    tokens = [tok for tok in lhs_norm.split() if len(tok) > 2 and tok not in stop]
    if not tokens:
        return False
    hits = sum(1 for tok in tokens if _token_variant_in_text(tok, q_norm))
    return hits >= min(2, len(tokens))


def _token_variant_in_text(token: str, text_norm: str) -> bool:
    variants = {token}
    if token.endswith("y") and len(token) > 3:
        variants.add(f"{token[:-1]}ies")
    elif token.endswith("s") and len(token) > 3:
        variants.add(token[:-1])
    else:
        variants.add(f"{token}s")
    return any(_alias_in_text(v, text_norm) for v in variants)


def _mapping_lhs_names_output(lhs: str) -> bool:
    return re.search(
        r"\b(name|date|amount|cost|status|type|category|number|source)\b",
        lhs or "",
        flags=re.I,
    ) is not None


def _mapping_lhs_filter_context(lhs: str, question: str) -> bool:
    lhs_norm = _norm(lhs)
    q_norm = _norm(question)
    if not lhs_norm or not _mapping_lhs_requested(lhs, question):
        return False
    if not re.search(r"\b(?:at|in|on|from|between|before|after|located|closed)\b", lhs_norm):
        return False
    direct_output = re.search(
        rf"\b(?:list|show|provide|give|state|write|name|identify|mention|include)\b[^?.;]*\b{re.escape(lhs_norm)}\b",
        q_norm,
    )
    return direct_output is None


def _split_column_list(text: str) -> List[str]:
    text = re.sub(r"\bwhere\b.+", "", text, flags=re.I)
    text = re.sub(r"\b(and|include|including)\b", ",", text, flags=re.I)
    return [p.strip(" .'\"`") for p in text.split(",") if p.strip(" .'\"`")]


def _resolve_column(text: str, ctx: EvidenceContext, table_hint: Optional[str] = None) -> Optional[ColumnInfo]:
    candidates = _resolve_column_candidates(text, ctx, table_hint=table_hint)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return _best_context_column(candidates, str(text or ""), ctx, [])


def _resolve_column_for_context(
    text: str,
    ctx: EvidenceContext,
    filters: Sequence[Dict[str, Any]],
    surrounding_text: str,
    table_hint: Optional[str] = None,
) -> Optional[ColumnInfo]:
    candidates = _resolve_column_candidates(text, ctx, table_hint=table_hint)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return _best_context_column(candidates, surrounding_text, ctx, filters)


def _resolve_column_candidates(text: str, ctx: EvidenceContext, table_hint: Optional[str] = None) -> List[ColumnInfo]:
    cleaned = _norm(_clean_column_phrase(text))
    if not cleaned:
        return []
    hint = _norm(table_hint or "")

    dotted = re.search(r"\b([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)\b", text)
    if dotted:
        table, col = dotted.group(1), dotted.group(2)
        for info in ctx.columns:
            if _norm(info.table) == _norm(table) and _norm(info.name) == _norm(col):
                return [info]

    matches: List[Tuple[int, ColumnInfo]] = []
    compact_cleaned = _compact_identifier(cleaned)
    for col in ctx.columns:
        if hint and not _table_matches_hint(col.table, hint):
            continue
        for alias in col.aliases:
            if not alias:
                continue
            if cleaned == alias:
                matches.append((1000 + len(alias), col))
            elif compact_cleaned and compact_cleaned == _compact_identifier(alias):
                matches.append((900 + len(alias), col))
            if cleaned.endswith(f" {alias}") or alias.endswith(f" {cleaned}"):
                matches.append((700 + len(alias), col))
            if _alias_in_text(alias, cleaned):
                matches.append((500 + len(alias), col))

    if not matches:
        return []

    matches.sort(key=lambda item: item[0], reverse=True)
    best_score = matches[0][0]
    best = [m[1] for m in matches if m[0] == best_score]
    return _dedupe_columns(best)


def _compact_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _norm(value))


def _best_context_column(
    candidates: Sequence[ColumnInfo],
    text: str,
    ctx: EvidenceContext,
    filters: Sequence[Dict[str, Any]],
) -> Optional[ColumnInfo]:
    if not candidates:
        return None
    scored = [(_column_context_score(col, text, ctx, filters), col) for col in candidates]
    scored.sort(key=lambda item: item[0], reverse=True)
    if len(scored) == 1:
        return scored[0][1]
    if scored[0][0] > scored[1][0]:
        return scored[0][1]
    return None


def _column_context_score(
    col: ColumnInfo,
    text: str,
    ctx: EvidenceContext,
    filters: Sequence[Dict[str, Any]],
) -> int:
    text_norm = _norm(text)
    score = 0
    table_norm = _norm(col.table)
    singular = table_norm[:-1] if table_norm.endswith("s") else table_norm
    plural = table_norm if table_norm.endswith("s") else f"{table_norm}s"
    if _alias_in_text(table_norm, text_norm) or _alias_in_text(singular, text_norm) or _alias_in_text(plural, text_norm):
        score += 8
    score += max((min(len(alias), 8) for alias in col.aliases if _alias_in_text(alias, text_norm)), default=0)

    filter_tables = {ref.split(".", 1)[0] for ref in _filter_refs(filters) if "." in ref}
    if col.table in filter_tables:
        score += 12
    elif filter_tables:
        distances = [_join_distance(ctx, col.table, table) for table in filter_tables]
        distances = [dist for dist in distances if dist is not None]
        if distances:
            score += max(0, 10 - min(distances) * 2)

    if col.table in ctx.primary_ids:
        score += 4
    if ctx.primary_ids.get(col.table) == col.name:
        score += 1
    if col.name.startswith("link_to_") and col.table not in ctx.primary_ids:
        score += 8
    if _id_like_column(col) and not _id_requested(text):
        score -= 6
    return score


def _id_requested(text: str) -> bool:
    return re.search(r"\b(?:id|identifier|number|code)\b", text or "", re.I) is not None


def _clean_column_phrase(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", str(text or ""))
    text = text.strip(" .'\"`;")
    text = re.sub(r"\b(?:the|a|an|of|for|by|in|on|with|whose|that|which|who|value|level|range)\b", " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def _comparison_lhs(text: str) -> str:
    parts = re.split(r"\bwhere\b", str(text or ""), flags=re.I)
    lhs = parts[-1].strip() if parts else str(text or "").strip()
    bits = re.split(r"\b(?:refers?\s*(?:to)?|means?)\b", lhs, flags=re.I)
    return bits[-1].strip() if bits else lhs


def _has_identifier_word_before(text: str, start: int) -> bool:
    prefix = text[:start].rstrip()
    if not prefix:
        return False
    m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)$", prefix)
    return m is not None


def _conditionish(text: str) -> bool:
    return bool(
        re.search(
            r"(?:=|>=|<=|>|<|\bbetween\b|\b(?:not\s+)?in\s*\(|"
            r"\bis\s+not\s+null\b|\bis\s+null\b|\bCOUNT\s*\(|\bAVG\s*\()",
            text,
            re.I,
        )
    )


def _normalize_condition_spacing(text: str) -> str:
    text = re.sub(r"([<>])\s+=", r"\1=", text or "")
    text = re.sub(r"=\s+([<>])", r"= \1", text)
    text = re.sub(r"'\s*(OR|AND)\b", r"' \1", text, flags=re.I)
    text = re.sub(r"\b(OR|AND)\s*'", r"\1 '", text, flags=re.I)
    return text


def _condition_tree(text: str, ctx: EvidenceContext) -> Optional[Dict[str, Any]]:
    text = _normalize_condition_spacing(text)
    text = re.sub(r"^[^:=]+?\s+(?:refers?\s*(?:to)?|means?)\s+", "", text, flags=re.I).strip()
    text = text.strip(" .;")
    if not text:
        return None

    or_parts = _split_bool(text, "or")
    if len(or_parts) > 1:
        children = [_condition_tree(part, ctx) for part in or_parts]
        children = [child for child in children if child]
        return {"or": children} if len(children) > 1 else (children[0] if children else None)

    and_parts = _split_bool(text, "and")
    if len(and_parts) > 1:
        children = [_condition_tree(part, ctx) for part in and_parts]
        children = [child for child in children if child]
        return {"and": children} if len(children) > 1 else (children[0] if children else None)

    return _single_condition(text, ctx)


def _align_condition_boundaries(
    cond: Optional[Dict[str, Any]],
    context_text: str,
    ctx: EvidenceContext,
) -> Optional[Dict[str, Any]]:
    if not isinstance(cond, dict):
        return cond
    if "and" in cond:
        return {"and": [_align_condition_boundaries(item, context_text, ctx) for item in cond["and"]]}
    if "or" in cond:
        return {"or": [_align_condition_boundaries(item, context_text, ctx) for item in cond["or"]]}
    if "not" in cond:
        return {"not": _align_condition_boundaries(cond["not"], context_text, ctx)}
    cmp_node = cond.get("cmp")
    if not isinstance(cmp_node, dict):
        return cond
    left = cmp_node.get("left")
    op = cmp_node.get("op")
    right = cmp_node.get("right")
    if not (isinstance(left, dict) and isinstance(left.get("col"), str) and op in {"<", ">"} and isinstance(right, (int, float))):
        return cond
    col = _column_by_ref(ctx, left["col"])
    if col is None:
        return cond
    names = {col.name, *col.aliases}
    for name in names:
        if not name or len(name) > 80:
            continue
        name_pattern = re.escape(name).replace(r"\ ", r"\s+").replace("_", r"[_\s]+")
        pattern = rf"\b{name_pattern}\s*{'<=' if op == '<' else '>='}\s*'?{re.escape(str(right))}'?"
        if re.search(pattern, context_text, re.I):
            updated = dict(cmp_node)
            updated["op"] = "<=" if op == "<" else ">="
            return {"cmp": updated}
    return cond


def _split_bool(text: str, op: str) -> List[str]:
    pattern = rf"\b{op}\b"
    parts = re.split(pattern, text, flags=re.I)
    if len(parts) <= 1:
        return [text]
    out: List[str] = []
    for part in parts:
        clean = part.strip(" ()")
        if clean:
            out.append(clean)
    return out or [text]


def _single_condition(text: str, ctx: EvidenceContext) -> Optional[Dict[str, Any]]:
    text = text.strip()

    wrapper = re.fullmatch(r"(?:MAX|MIN)\s*\((.*)\)", text, re.I | re.S)
    if wrapper and _conditionish(wrapper.group(1)):
        return _condition_tree(wrapper.group(1), ctx)

    m = re.search(r"([A-Za-z0-9_.\"` -]+?)\s*(=|>=|<=|>|<)\s*(AVG|SUM|MIN|MAX)\s*\(\s*([A-Za-z0-9_.\"` -]+?)\s*\)$", text, flags=re.I)
    if m:
        col = _resolve_column(m.group(1), ctx)
        metric = _resolve_column(m.group(4), ctx)
        if col is None or metric is None:
            return None
        dataset = _choose_dataset(ctx, [metric.ref], [], text=text) or metric.table
        subquery = _advanced(
            dataset,
            [{"expr": _func(m.group(3).lower(), {"col": metric.ref}), "alias": "value"}],
            limit=1,
        )
        return _cmp({"col": col.ref}, m.group(2), {"scalar_subquery": {"plan": subquery}})

    text = text.strip(" ()")

    m = re.search(r"(.+?)\s+(NOT\s+)?IN\s*\(([^)]*)\)\s*$", text, flags=re.I)
    if m:
        col = _resolve_column(m.group(1), ctx)
        if col is None:
            return None
        values = [
            _parse_literal_for_column(raw, col, "=")
            for raw in _split_top_level_args(m.group(3))
            if raw.strip()
        ]
        if not values:
            return None
        return _cmp({"col": col.ref}, "not_in" if m.group(2) else "in", values)

    m = re.search(
        r"(.+?)\s*=\s*((?:'[^']*'|\"[^\"]*\")\s*,\s*(?:'[^']*'|\"[^\"]*\")"
        r"(?:\s*,\s*(?:'[^']*'|\"[^\"]*\"))*)\s*$",
        text,
        re.I,
    )
    if m:
        col = _resolve_column(m.group(1), ctx)
        if col is None:
            return None
        values = [
            _parse_literal_for_column(raw, col, "=")
            for raw in _split_top_level_args(m.group(2))
            if raw.strip()
        ]
        return _cmp({"col": col.ref}, "in", values) if values else None

    m = re.search(r"(.+?)\s+is\s+not\s+null\b", text, flags=re.I)
    if m:
        col = _resolve_column(m.group(1), ctx)
        return _cmp({"col": col.ref}, "is_not_null", True) if col else None

    m = re.search(r"(.+?)\s+is\s+null\b", text, flags=re.I)
    if m:
        col = _resolve_column(m.group(1), ctx)
        return _cmp({"col": col.ref}, "is_null", True) if col else None

    m = re.search(r"(.+?)\s+not\s+between\s+'?([^']+?)'?\s+and\s+'?([^']+?)'?$", text, flags=re.I)
    if m:
        col = _resolve_column(m.group(1), ctx)
        if col is None:
            return None
        return {"not": _cmp({"col": col.ref}, "between", [_parse_literal(m.group(2)), _parse_literal(m.group(3))])}

    m = re.search(r"(.+?)\s+between\s+'?([^']+?)'?\s+and\s+'?([^']+?)'?$", text, flags=re.I)
    if m:
        left = m.group(1)
        if re.search(r"\byear\s*\(", left, re.I):
            ym = re.search(r"\byear\s*\(\s*([^)]+?)\s*\)", left, re.I)
            col = _resolve_column(ym.group(1), ctx) if ym else None
            if col is None:
                return None
            lo = _year_comparison_value(m.group(2), col)
            hi = _year_comparison_value(m.group(3), col)
            return _cmp(_year_expr(col), "between", [lo, hi])
        col = _resolve_column(left, ctx)
        if col is None:
            return None
        return _cmp({"col": col.ref}, "between", [_parse_literal(m.group(2)), _parse_literal(m.group(3))])

    m = re.search(r"\byear\s*\(\s*([^)]+?)\s*\)\s*(=|>=|<=|>|<)\s*'?([A-Za-z0-9./:-]+)'?", text, flags=re.I)
    if m:
        col = _resolve_column(m.group(1), ctx)
        if col is None:
            return None
        value = _year_comparison_value(m.group(3), col)
        age_cmp = _age_like_year_comparison(col, m.group(2), value, text, ctx)
        if age_cmp is not None:
            return age_cmp
        return _cmp(_year_expr(col), m.group(2), value)

    m = re.search(r"\bmonth\s*\(\s*([^)]+?)\s*\)\s*(=|>=|<=|>|<)\s*'?([A-Za-z0-9./:-]+)'?", text, flags=re.I)
    if m:
        col = _resolve_column(m.group(1), ctx)
        if col is None:
            return None
        value = _parse_comparison_value(m.group(3), force_year=False)
        if _text_date_like(col):
            value = _month_value(value)
        return _cmp(_month_expr(col), m.group(2), value)

    m = re.search(r"([A-Za-z0-9_.\"` -]+?)\s+like\s+'([^']+)'$", text, flags=re.I)
    if m:
        col = _resolve_column(m.group(1), ctx)
        if col is None:
            return None
        op, value = _like_to_op(m.group(2))
        return _cmp({"col": col.ref}, op, value)

    date_diff = _date_difference_condition(text, ctx)
    if date_diff is not None:
        return date_diff

    m = re.fullmatch(r"(.+?)\s*(=|>=|<=|>|<)\s*([A-Za-z_][A-Za-z0-9_.\"` -]*?)\s*", text, re.I | re.S)
    if m:
        left = _resolve_column_for_context(m.group(1), ctx, [], text)
        right = _resolve_column_for_context(m.group(3), ctx, [], text)
        if left is not None and right is not None and left.ref != right.ref:
            return _cmp({"col": left.ref}, m.group(2), {"col": right.ref})

    m = re.search(r"(.+?)\s*(=|>=|<=|>|<)\s*'?([+-]?\d+(?:\.\d+)?)'?\s*$", text, re.I | re.S)
    if m and re.search(r"\b(?:DIVIDE|DIVISION|SUBTRACT|MULTIPLY)\s*\(", m.group(1), re.I):
        parsed = _formula_expr(m.group(1), ctx, text)
        if parsed is not None:
            return _cmp(parsed["expr"], m.group(2), _parse_literal(m.group(3)))

    m = re.search(r"([A-Za-z0-9_.\"` -]+?)\s*(=|>=|<=|>|<)\s*(AVG|SUM|MIN|MAX)\s*\(\s*([A-Za-z0-9_.\"` -]+?)\s*\)$", text, flags=re.I)
    if m:
        col = _resolve_column(m.group(1), ctx)
        metric = _resolve_column(m.group(4), ctx)
        if col is None or metric is None:
            return None
        dataset = _choose_dataset(ctx, [metric.ref], [], text=text) or metric.table
        subquery = _advanced(
            dataset,
            [{"expr": _func(m.group(3).lower(), {"col": metric.ref}), "alias": "value"}],
            limit=1,
        )
        return _cmp({"col": col.ref}, m.group(2), {"scalar_subquery": {"plan": subquery}})

    m = re.search(r"^'?([+-]?\d+(?:\.\d+)?)'?\s*(<|<=|>|>=)\s*([A-Za-z0-9_.\"` -]+?)\s*(<|<=|>|>=)\s*'?([+-]?\d+(?:\.\d+)?)'?$", text)
    if m:
        col = _resolve_column(m.group(3), ctx)
        if col is None:
            return None
        left = _cmp({"col": col.ref}, _flip_op(m.group(2)), _parse_literal(m.group(1)))
        right = _cmp({"col": col.ref}, m.group(4), _parse_literal(m.group(5)))
        return {"and": [left, right]}

    m = re.search(r"([A-Za-z0-9_.\"` -]+?)\s*(=|>=|<=|>|<)\s*('(?:[^']*)'?|\"(?:[^\"]*)\"?|[+-]?\d+(?:\.\d+)?|[A-Z][A-Z0-9_+-]*)$", text, flags=re.I)
    if m:
        col = _resolve_column(m.group(1), ctx)
        if col is None:
            return None
        value = _parse_literal_for_column(m.group(3), col, m.group(2))
        if _looks_like_compact_period(value) and not _text_date_like(col):
            alt = _resolve_text_period_column(m.group(1), ctx)
            if alt is not None:
                col = alt
        return _column_cmp(col, m.group(2), value)

    return None


def _date_difference_condition(text: str, ctx: EvidenceContext) -> Optional[Dict[str, Any]]:
    if not re.search(r"\b(?:SUBTRACT|year\s*\()", text, re.I):
        return None
    expr = _date_difference_expr(text, ctx, prefer_context=True)
    if expr is None:
        return None
    m = re.search(r"\)\s*(=|>=|<=|>|<)\s*'?([+-]?\d+(?:\.\d+)?)'?\s*$", text)
    if not m:
        return None
    return _cmp(expr["expr"], m.group(1), _parse_literal(m.group(2)))


def _date_difference_expr(
    text: str,
    ctx: EvidenceContext,
    *,
    prefer_context: bool = False,
) -> Optional[Dict[str, Any]]:
    if not re.search(r"\b(?:SUBTRACT|year\s*\()", text, re.I):
        return None
    raw_parts = re.findall(r"\byear\s*\(\s*([^)]+?)\s*\)", text, flags=re.I)
    cols = [_resolve_column_for_context(part, ctx, [], text) for part in raw_parts]
    cols = [col for col in cols if col is not None]
    if not cols:
        return None
    left_col = _contextual_left_date(text, ctx, exclude_refs={cols[-1].ref}) if prefer_context and len(cols) == 1 else None
    if len(cols) > 1:
        left_col = cols[0]
    left = _year_expr(left_col) if left_col is not None else _func("date_part", {"lit": "year"}, _func("now"))
    right = _year_expr(cols[-1])
    refs = [col.ref for col in cols]
    if left_col is not None and left_col.ref not in refs:
        refs.insert(0, left_col.ref)
    return {"expr": _op("-", left, right), "refs": refs}


def _age_like_year_comparison(
    col: ColumnInfo,
    op: str,
    value: Any,
    text: str,
    ctx: EvidenceContext,
) -> Optional[Dict[str, Any]]:
    if not isinstance(value, (int, float)) or abs(float(value)) > 130:
        return None
    left_col = _contextual_left_date(text, ctx, exclude_refs={col.ref})
    if left_col is None:
        return None
    return _cmp(_op("-", _year_expr(left_col), _year_expr(col)), op, value)


def _contextual_left_date(
    text: str,
    ctx: EvidenceContext,
    *,
    exclude_refs: set[str],
) -> Optional[ColumnInfo]:
    candidates = []
    text_norm = _norm(text)
    for col in ctx.columns:
        if col.ref in exclude_refs or not _column_is_temporal(col):
            continue
        score = 0
        if any(len(alias) > 4 and _alias_in_text(alias, text_norm) for alias in col.aliases):
            score += 5
        if re.search(rf"\byear\s*\(\s*`?{re.escape(col.name)}`?\s*\)", text, re.I):
            score += 8
        if score and col.table in ctx.primary_dates and ctx.primary_dates[col.table] == col.name:
            score += 2
        if score:
            candidates.append((score, col))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _flip_op(op: str) -> str:
    return {"<": ">", "<=": ">=", ">": "<", ">=": "<="}.get(op, op)


def _resolve_text_period_column(text: str, ctx: EvidenceContext) -> Optional[ColumnInfo]:
    cleaned = _norm(text)
    candidates = [
        col for col in ctx.columns
        if _text_date_like(col) and any(cleaned == alias or _alias_in_text(alias, cleaned) for alias in col.aliases)
    ]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return candidates[0]
    return None


def _column_by_table_name(ctx: EvidenceContext, table: str, name: str) -> Optional[ColumnInfo]:
    for col in ctx.columns:
        if col.table == table and col.name == name:
            return col
    return None


def _table_matches_hint(table: str, hint: str) -> bool:
    table_norm = _norm(table)
    singular = table_norm[:-1] if table_norm.endswith("s") else table_norm
    return hint == table_norm or hint == singular or table_norm in hint or singular in hint


def _disambiguate_columns_by_table_text(columns: Sequence[ColumnInfo], text: str) -> List[ColumnInfo]:
    if len(columns) <= 1:
        return list(columns)
    text_norm = _norm(text)
    scored: List[Tuple[int, ColumnInfo]] = []
    for col in columns:
        score = 0
        table_norm = _norm(col.table)
        singular = table_norm[:-1] if table_norm.endswith("s") else table_norm
        plural = table_norm if table_norm.endswith("s") else f"{table_norm}s"
        if _alias_in_text(table_norm, text_norm) or _alias_in_text(singular, text_norm) or _alias_in_text(plural, text_norm):
            score += 5
        score += max((len(alias) for alias in col.aliases if _alias_in_text(alias, text_norm)), default=0)
        scored.append((score, col))
    best = max(score for score, _ in scored)
    if best <= 0:
        return list(columns)
    return [col for score, col in scored if score == best]


def _output_context_columns(
    columns: Sequence[ColumnInfo],
    question: str,
    filters: Sequence[Dict[str, Any]],
    ctx: EvidenceContext,
) -> List[ColumnInfo]:
    if not columns:
        return []
    q_norm = _norm(question)
    requested: List[ColumnInfo] = []
    for col in columns:
        for alias in col.aliases:
            if not alias or (len(alias) <= 2 and alias != "id"):
                continue
            matched = False
            for variant in _alias_variants(alias):
                if variant.endswith("ed") and re.search(rf"\b{re.escape(variant)}\s+for\b", q_norm):
                    continue
                if re.search(rf"\b(?:include|including|and|plus)\s+(?:the\s+|a\s+|an\s+|their\s+|its\s+)?(?:[a-z0-9_]+\s+){{0,2}}{re.escape(variant)}\b", q_norm):
                    requested.append(col)
                    matched = True
                    break
                if re.search(rf"\b(?:list|show|provide|give|state|write|name|identify|mention)\s+(?:all\s+|the\s+|a\s+|an\s+|their\s+|its\s+)?(?:[a-z0-9_]+\s+){{0,2}}{re.escape(variant)}\b", q_norm):
                    requested.append(col)
                    matched = True
                    break
                if re.search(rf"\bwhat\s+(?:are|is|was|were)\s+(?:the\s+)?(?:[a-z0-9_]+\s+){{0,2}}{re.escape(variant)}\b", q_norm):
                    requested.append(col)
                    matched = True
                    break
                if re.search(rf"[,;]\s*{re.escape(variant)}\b", q_norm):
                    requested.append(col)
                    matched = True
                    break
                if re.search(rf"\bby\s+(?:their\s+|the\s+)?{re.escape(variant)}\b", q_norm):
                    requested.append(col)
                    matched = True
                    break
            if matched:
                break
    if not requested:
        return []
    requested.sort(key=lambda col: _column_request_pos(col, q_norm))
    by_name: Dict[str, List[ColumnInfo]] = {}
    for col in requested:
        by_name.setdefault(_norm(col.name), []).append(col)
    out: List[ColumnInfo] = []
    filter_tables = {ref.split(".", 1)[0] for ref in _filter_refs(filters) if "." in ref}
    for group in by_name.values():
        if len(group) == 1:
            out.append(group[0])
            continue
        text_match = _disambiguate_columns_by_table_text(group, question)
        if len(text_match) < len(group):
            out.extend(text_match)
            continue
        same_filter_table = [col for col in group if col.table in filter_tables]
        out.extend(same_filter_table or _disambiguate_columns_by_table_text(group, question))
    return _dedupe_columns(out)


def _column_request_pos(col: ColumnInfo, q_norm: str) -> int:
    positions: List[int] = []
    if col.name in {"first_name", "last_name"}:
        idx = q_norm.find("full name")
        if idx >= 0:
            positions.append(idx)
    for alias in col.aliases:
        for variant in _alias_variants(alias):
            idx = q_norm.find(variant)
            if idx >= 0:
                positions.append(idx)
    return min(positions) if positions else 10**9


def _general_synonym_hits(col: ColumnInfo, text_norm: str) -> bool:
    groups = {
        "first_name": {"first name", "full name", "name"},
        "last_name": {"last name", "full name", "name"},
        "date": {"date", "day"},
        "time": {"time"},
    }
    terms = groups.get(col.name, set())
    if any(_alias_in_text(_norm(term), text_norm) for term in terms):
        return True
    if col.name.endswith("_name"):
        base = _norm(col.name[:-5])
        if base and _alias_in_text(base, text_norm):
            return True
    return False


def _choose_dataset(
    ctx: EvidenceContext,
    refs: Sequence[str],
    filters: Sequence[Dict[str, Any]],
    *,
    text: str = "",
) -> Optional[str]:
    scores = {table: 0 for table in ctx.tables}
    filter_refs = list(_filter_refs(filters))
    for ref in refs:
        table = ref.split(".", 1)[0] if "." in ref else ref
        if table in scores:
            scores[table] += 1
    for ref in filter_refs:
        table = ref.split(".", 1)[0] if "." in ref else ref
        if table in scores:
            scores[table] += 3
            col = _column_by_ref(ctx, ref)
            if col is not None and _column_is_temporal(col):
                scores[table] += 2
    text_norm = _norm(text)
    for table in ctx.tables:
        table_norm = _norm(table)
        singular = table_norm[:-1] if table_norm.endswith("s") else table_norm
        plural = table_norm if table_norm.endswith("s") else f"{table_norm}s"
        if _alias_in_text(table_norm, text_norm) or _alias_in_text(singular, text_norm) or _alias_in_text(plural, text_norm):
            scores[table] += 2

    target_tables = {ref.split(".", 1)[0] for ref in list(refs) + filter_refs if "." in ref}
    if len(target_tables) > 1:
        for table in ctx.tables:
            distances = [_join_distance(ctx, table, target) for target in target_tables]
            if all(d is not None for d in distances):
                scores[table] += max(0, 6 - sum(d or 0 for d in distances))
    if not scores:
        return None
    best_table, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score > 0:
        return best_table
    return ctx.tables[0] if ctx.tables else None


def _entity_scoped_formula_dataset(
    ctx: EvidenceContext,
    metric_refs: Sequence[str],
    filters: Sequence[Dict[str, Any]],
    *,
    text: str,
) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
    metric_tables = {ref.split(".", 1)[0] for ref in metric_refs if "." in ref}
    if len(metric_tables) != 1 or not filters:
        return None
    metric_table = next(iter(metric_tables))
    metric_filters: List[Dict[str, Any]] = []
    scope_filters: List[Dict[str, Any]] = []
    for filt in filters:
        tables = {ref.split(".", 1)[0] for ref in _filter_refs([filt]) if "." in ref}
        if tables and tables <= {metric_table}:
            metric_filters.append(filt)
        elif tables:
            scope_filters.append(filt)
        else:
            metric_filters.append(filt)
    if not scope_filters:
        return None

    target_cols = _entity_columns_on_table(metric_table, ctx)
    filter_tables = {
        ref.split(".", 1)[0]
        for ref in _filter_refs(scope_filters)
        if "." in ref and ref.split(".", 1)[0] != metric_table
    }
    for target_col in target_cols:
        for source_table in filter_tables:
            source_col = _column_by_table_name(ctx, source_table, target_col.name)
            if source_col is None:
                continue
            sub_dataset = _choose_dataset(ctx, [source_col.ref], scope_filters, text=text)
            if sub_dataset is None:
                continue
            subquery = {
                "version": "1.0",
                "dataset": sub_dataset,
                "select": [{"expr": {"col": source_col.ref}, "alias": _alias_for(source_col.name)}],
                "where": _and(scope_filters),
                "limit": 1,
                "offset": 0,
            }
            scoped_filter = _cmp(
                {"col": target_col.ref},
                "=",
                {"scalar_subquery": {"plan": subquery}},
            )
            return metric_table, metric_filters + [scoped_filter]
    return None


def _entity_columns_on_table(table: str, ctx: EvidenceContext) -> List[ColumnInfo]:
    cols: List[ColumnInfo] = []
    primary = ctx.primary_ids.get(table)
    if primary:
        col = _column_by_table_name(ctx, table, primary)
        if col is not None:
            cols.append(col)
    for col in ctx.columns:
        if col.table != table or col in cols:
            continue
        if col.name.lower().endswith("id"):
            cols.append(col)
    return cols


def _column_by_ref(ctx: EvidenceContext, ref: str) -> Optional[ColumnInfo]:
    table, _, name = ref.partition(".")
    if not table or not name:
        return None
    return _column_by_table_name(ctx, table, name)


def _join_distance(ctx: EvidenceContext, start: str, end: str) -> Optional[int]:
    if start == end:
        return 0
    graph: Dict[str, List[str]] = {t: [] for t in ctx.tables}
    for link in ctx.schema.get("links", []) or []:
        if not isinstance(link, dict):
            continue
        frm = link.get("from_table")
        to = link.get("to_table")
        if isinstance(frm, str) and isinstance(to, str):
            graph.setdefault(frm, []).append(to)
            graph.setdefault(to, []).append(frm)
    frontier = [(start, 0)]
    seen = {start}
    for node, dist in frontier:
        for nxt in graph.get(node, []):
            if nxt == end:
                return dist + 1
            if nxt in seen:
                continue
            seen.add(nxt)
            frontier.append((nxt, dist + 1))
    return None


def _add_intent_aware_joins(
    plan: Dict[str, Any],
    text: str,
    ctx: EvidenceContext,
) -> Dict[str, Any]:
    """Prefer a schema path whose intermediate relations are named by the request."""
    if plan.get("joins"):
        return plan
    dataset = plan.get("dataset")
    if not isinstance(dataset, str) or dataset not in ctx.tables:
        return plan

    targets = [table for table in _plan_referenced_tables(plan, ctx) if table != dataset]
    if not targets:
        return plan

    selected_paths: List[List[Dict[str, Any]]] = []
    needs_explicit_joins = False
    inner_links: set[str] = set()
    null_filter_tables = _null_filter_tables(plan, ctx)
    for target in targets:
        paths = _simple_join_paths(ctx, dataset, target, max_edges=6)
        if not paths:
            continue
        shortest_len = min(len(path) for path in paths)
        shortest_paths = [path for path in paths if len(path) == shortest_len]
        ranked = sorted(
            shortest_paths,
            key=lambda path: _join_path_text_score(path, text),
            reverse=True,
        )
        best = ranked[0]
        if len(ranked) > 1 and _join_path_text_score(best, text) > _join_path_text_score(ranked[1], text):
            needs_explicit_joins = True
        null_attribute_join = target in null_filter_tables or dataset in null_filter_tables
        missing_relation_named = _asks_for_missing_relation(text, target) or _asks_for_missing_relation(text, dataset)
        if null_attribute_join and not missing_relation_named:
            needs_explicit_joins = True
            inner_links.update(edge["link"]["name"] for edge in best)
        selected_paths.append(best)

    if not needs_explicit_joins:
        return plan

    joins: List[Dict[str, str]] = []
    for path in selected_paths:
        for edge in path:
            entry = {"link": edge["link"]["name"]}
            if edge["link"]["name"] in inner_links:
                entry["type"] = "inner"
            existing = next((item for item in joins if item.get("link") == entry["link"]), None)
            if existing is not None and entry.get("type") == "inner":
                existing["type"] = "inner"
            elif existing is None:
                joins.append(entry)
    if not joins:
        return plan
    updated = dict(plan)
    updated["joins"] = joins
    return updated


def _null_filter_tables(plan: Dict[str, Any], ctx: EvidenceContext) -> set[str]:
    tables: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            cmp_node = node.get("cmp")
            if isinstance(cmp_node, dict) and cmp_node.get("op") == "is_null":
                left = cmp_node.get("left")
                ref = left.get("col") if isinstance(left, dict) else None
                if isinstance(ref, str) and "." in ref:
                    table = ref.split(".", 1)[0]
                    if table in ctx.tables:
                        tables.add(table)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(plan.get("where"))
    return tables


def _asks_for_missing_relation(text: str, table: str) -> bool:
    table_norm = _norm(table)
    singular = table_norm[:-1] if table_norm.endswith("s") else table_norm
    plural = table_norm if table_norm.endswith("s") else f"{table_norm}s"
    return any(
        re.search(rf"\b(?:without|no|lacking|does not have|don't have)\s+(?:any\s+)?{re.escape(name)}\b", _norm(text))
        for name in {table_norm, singular, plural}
        if name
    )


def _plan_referenced_tables(plan: Dict[str, Any], ctx: EvidenceContext) -> List[str]:
    known = set(ctx.tables)
    out: List[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "scalar_subquery" in node or "exists" in node or "not_exists" in node:
                return
            ref = node.get("col")
            if isinstance(ref, str) and "." in ref:
                table = ref.split(".", 1)[0]
                if table in known and table not in out:
                    out.append(table)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(plan)
    return out


def _simple_join_paths(
    ctx: EvidenceContext,
    start: str,
    end: str,
    *,
    max_edges: int,
) -> List[List[Dict[str, Any]]]:
    graph: Dict[str, List[Dict[str, Any]]] = {table: [] for table in ctx.tables}
    for link in ctx.schema.get("links", []) or []:
        if not isinstance(link, dict) or not isinstance(link.get("name"), str):
            continue
        frm = link.get("from_table")
        to = link.get("to_table")
        if not isinstance(frm, str) or not isinstance(to, str):
            continue
        graph.setdefault(frm, []).append({"from": frm, "to": to, "link": link})
        graph.setdefault(to, []).append({"from": to, "to": frm, "link": link})

    paths: List[List[Dict[str, Any]]] = []
    frontier: List[Tuple[str, List[Dict[str, Any]], set[str]]] = [(start, [], {start})]
    for node, path, seen in frontier:
        if len(path) >= max_edges:
            continue
        for edge in graph.get(node, []):
            nxt = edge["to"]
            if nxt in seen:
                continue
            new_path = path + [edge]
            if nxt == end:
                paths.append(new_path)
                continue
            frontier.append((nxt, new_path, seen | {nxt}))
    return paths


def _join_path_text_score(path: Sequence[Dict[str, Any]], text: str) -> int:
    text_norm = _norm(text)
    score = -len(path)
    for edge in path:
        table = _norm(edge.get("to") or "")
        endpoint_tokens = set(_norm(edge.get("from") or "").split()) | set(table.split())
        singular = table[:-1] if table.endswith("s") else table
        if table and (_alias_in_text(table, text_norm) or _alias_in_text(singular, text_norm)):
            score += 20
        link = edge.get("link") or {}
        signals = [_norm(link.get("name") or "")]
        for on in link.get("on", []) or []:
            signals.extend([_norm(on.get("left") or ""), _norm(on.get("right") or "")])
        for signal in signals:
            tokens = [
                token
                for token in signal.split()
                if len(token) > 2 and token not in {"from", "table", "link"}
            ]
            score += 4 * sum(1 for token in tokens if _alias_in_text(token, text_norm))
            role_tokens = {
                token for token in tokens
                if token not in endpoint_tokens and token not in {"id", "api", "to"}
            }
            score -= 3 * sum(1 for token in role_tokens if not _alias_in_text(token, text_norm))
            content_tokens = [token for token in tokens if token not in {"id", "api"}]
            for width in range(min(4, len(content_tokens)), 1, -1):
                for start in range(len(content_tokens) - width + 1):
                    phrase = " ".join(content_tokens[start:start + width])
                    if _alias_in_text(phrase, text_norm):
                        score += 8 * width
    return score


def _primary_id_ref(dataset: str, ctx: EvidenceContext) -> Optional[str]:
    if dataset in ctx.primary_ids:
        return f"{dataset}.{ctx.primary_ids[dataset]}"
    for col in ctx.columns:
        if col.table == dataset and col.name == "id":
            return col.ref
    return None


def _count_ref_for_question(
    question: str,
    dataset: str,
    ctx: EvidenceContext,
    filters: Sequence[Dict[str, Any]],
) -> Optional[str]:
    q_norm = _norm(question)
    named = _count_named_column(question, ctx, filters)
    if named is not None:
        return named.ref
    scored: List[Tuple[int, ColumnInfo]] = []
    filter_refs = _filter_refs(filters)
    filter_cols = [_column_by_ref(ctx, ref) for ref in filter_refs]
    filter_cols = [col for col in filter_cols if col is not None]
    for table, pid in ctx.primary_ids.items():
        col = _column_by_table_name(ctx, table, pid)
        if col is None:
            continue
        table_norm = _norm(table)
        singular = table_norm[:-1] if table_norm.endswith("s") else table_norm
        plural = table_norm if table_norm.endswith("s") else f"{table_norm}s"
        score = 0
        if _alias_in_text(plural, q_norm):
            score += 10
        elif _alias_in_text(table_norm, q_norm) or _alias_in_text(singular, q_norm):
            score += 8
        head = table_norm.split()[0] if table_norm.split() else ""
        if head and len(head) > 3 and _alias_in_text(head, q_norm):
            score += 9
        explicit_names = {table_norm, singular, plural}
        if head and len(head) > 3:
            explicit_names.update({head, f"{head}s"})
        if any(re.search(rf"\b(?:how many|number of)\s+{re.escape(name)}\b", q_norm) for name in explicit_names if name):
            score += 20
        if any(
            re.search(rf"\b(?:among|of)\s+(?:the\s+)?{re.escape(name)}\b", q_norm)
            for name in explicit_names
            if name
        ) and re.search(r"\bhow many of (?:them|those)\b", q_norm):
            score += 30
        if _human_entity_phrase_matches_table(table_norm, q_norm):
            score += 6
        table_filter_cols = [fcol for fcol in filter_cols if fcol.table == table]
        if any(not _id_like_column(fcol) and not _column_is_temporal(fcol) for fcol in table_filter_cols):
            score += 8
        if table == dataset and table_filter_cols and re.search(r"\b(?:records?|rows?|entries|items|happen|happened|occurred)\b", q_norm):
            score += 12
        if _entity_detail_table(table, ctx) and re.search(
            r"\b(?:who|whose|them|people|persons?|users?|students?|customers?|patients?|"
            r"members?|attendees?|employees?|workers?)\b",
            q_norm,
        ):
            score += 20
        if col.ref in filter_refs:
            score -= 4
        if table == dataset:
            score += 1
        if score > 0:
            scored.append((score, col))
    if scored:
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1].ref
    return _primary_id_ref(dataset, ctx)


def _count_named_column(
    question: str,
    ctx: EvidenceContext,
    filters: Sequence[Dict[str, Any]],
) -> Optional[ColumnInfo]:
    q_norm = _norm(question)
    filter_refs = _filter_refs(filters)
    candidates: List[Tuple[int, ColumnInfo]] = []
    for col in ctx.columns:
        if col.ref in filter_refs or _id_like_column(col) or _column_is_temporal(col):
            continue
        score = 0
        for alias in col.aliases:
            if not alias or " " in alias:
                continue
            for variant in _alias_variants(alias):
                if re.search(rf"\b(?:how many|number of)\s+{re.escape(variant)}\b", q_norm):
                    score = max(score, 20 + len(variant))
        if score:
            candidates.append((score, col))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _entity_detail_table(table: str, ctx: EvidenceContext) -> bool:
    names = {col.name for col in ctx.columns if col.table == table}
    return bool({"first_name", "last_name", "email"} & names)


def _human_entity_phrase_matches_table(table_norm: str, question_norm: str) -> bool:
    human_terms = {"person", "people", "user", "users"}
    table_terms = set(table_norm.split())
    if table_norm.endswith("s"):
        table_terms.add(table_norm[:-1])
    if not table_terms & human_terms:
        return False
    return any(_alias_in_text(term, question_norm) for term in human_terms)


def _best_date_column(ctx: EvidenceContext) -> Optional[ColumnInfo]:
    for table, col in ctx.primary_dates.items():
        found = next((c for c in ctx.columns if c.table == table and c.name == col), None)
        if found is not None:
            return found
    return next((c for c in ctx.columns if _column_is_temporal(c)), None)


def _date_column_for_text(ctx: EvidenceContext, text: str) -> Optional[ColumnInfo]:
    text_norm = _norm(text)
    candidates = [
        c for c in ctx.columns
        if _column_is_temporal(c)
        and not (c.type == "time" or (c.name.lower() == "time" and "date" not in c.name.lower()))
    ]
    if not candidates:
        return None
    scored: List[Tuple[int, ColumnInfo]] = []
    for col in candidates:
        score = 0
        if _alias_in_text(_norm(col.name), text_norm):
            score += 4
        if _alias_in_text(_norm(col.table), text_norm):
            score += 2
        if col.name in ctx.primary_dates.values():
            score += 1
        scored.append((score, col))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _date_column_for_table(ctx: EvidenceContext, table: str) -> Optional[ColumnInfo]:
    primary = ctx.primary_dates.get(table)
    if primary:
        col = _column_by_table_name(ctx, table, primary)
        if col is not None:
            return col
    candidates = [c for c in ctx.columns if c.table == table and (c.type == "date" or "date" in c.name)]
    return candidates[0] if candidates else None


def _best_numeric_measure_for_text(text: str, ctx: EvidenceContext) -> Optional[ColumnInfo]:
    text_norm = _norm(text)
    question_norm = _norm(text.split("\n", 1)[0])
    candidates = [c for c in ctx.columns if _numeric_column(c) and not _id_like_column(c) and c.name not in ctx.primary_ids.values()]
    if not candidates:
        return None
    scored: List[Tuple[int, ColumnInfo]] = []
    for col in candidates:
        score = 0
        if _column_mentioned_in_text(col, text_norm):
            score += 5
        if _column_mentioned_in_text(col, question_norm):
            score += 6
        if _alias_in_text(_norm(col.table), text_norm):
            score += 1
        scored.append((score, col))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _money_measure_for_text(text: str, ctx: EvidenceContext) -> Optional[ColumnInfo]:
    text_norm = _norm(text)
    preferred = [
        c for c in ctx.columns
        if _numeric_column(c) and not _id_like_column(c) and re.search(r"\b(price|cost|spent|amount|total)\b", _norm(c.name))
    ]
    if not preferred:
        return _best_numeric_measure_for_text(text, ctx)
    scored: List[Tuple[int, ColumnInfo]] = []
    for col in preferred:
        score = 0
        if any(_alias_in_text(alias, text_norm) for alias in col.aliases):
            score += 5
        if "price" in _norm(col.name) and re.search(r"\bspent\b|\bpaid\b", text_norm):
            score += 8
        if "amount" in _norm(col.name) and re.search(r"\bspent\b|\bpaid\b", text_norm):
            score -= 2
        scored.append((score, col))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _filter_has_compact_period(filt: Dict[str, Any]) -> bool:
    value = _single_filter_value(filt)
    return value is not None and _looks_like_compact_period(value)


def _time_column_for_text(ctx: EvidenceContext, text: str) -> Optional[ColumnInfo]:
    text_norm = _norm(text)
    candidates = [c for c in ctx.columns if c.type == "time" or "time" in c.name]
    if not candidates:
        return None
    scored: List[Tuple[int, ColumnInfo]] = []
    for col in candidates:
        score = 0
        if _alias_in_text(_norm(col.name), text_norm):
            score += 4
        if _alias_in_text(_norm(col.table), text_norm):
            score += 2
        scored.append((score, col))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _candidate_literal_values(text: str) -> List[str]:
    values: List[str] = []
    skip = {"and", "or", "not"}
    for quoted in re.findall(r'"([^"]+)"|\'([^\']+)\'', text):
        value = quoted[0] or quoted[1]
        if value and value.lower() not in skip:
            values.append(value)
    for phrase in re.findall(r"\b[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z0-9][A-Za-z0-9_-]*){1,4}\b", text):
        if phrase.lower() not in skip:
            values.append(phrase)
    for token in re.findall(r"\b[A-Z][A-Z0-9_-]{1,}\b", text):
        if token.lower() not in skip:
            values.append(token)
    return _dedupe_strings(values)


def _match_known_value(value: str, known_values: Sequence[str]) -> Optional[str]:
    value_norm = _norm(value)
    for known in known_values:
        if _norm(known) == value_norm:
            return known
    symbolic_aliases = {
        "-": ("negative", "no", "false"),
        "+-": ("0", "neutral", "borderline", "indeterminate"),
        "+": ("positive", "yes", "true", "1"),
    }
    for alias in symbolic_aliases.get(value_norm, ()):
        for known in known_values:
            if _norm(known) == alias:
                return known
    return None


def _dedupe_strings(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out


def _filter_refs(filters: Sequence[Dict[str, Any]]) -> set[str]:
    refs: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            col = node.get("col")
            if isinstance(col, str):
                refs.add(col)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for filt in filters:
        walk(filt)
    return refs


def _rewrite_open_age_filters(
    filters: Sequence[Dict[str, Any]],
    ctx: EvidenceContext,
) -> List[Dict[str, Any]]:
    temporal_refs = [
        ref
        for ref in _filter_refs(filters)
        for col in [_column_by_ref(ctx, ref)]
        if col is not None and _column_is_temporal(col)
    ]
    temporal_refs = _dedupe_strings(temporal_refs)
    if not temporal_refs:
        return list(filters)

    def is_now_year(expr: Any) -> bool:
        return (
            isinstance(expr, dict)
            and expr.get("func") == "date_part"
            and len(expr.get("args", [])) == 2
            and expr["args"][0] == {"lit": "year"}
            and isinstance(expr["args"][1], dict)
            and expr["args"][1].get("func") == "now"
        )

    def rewrite(node: Any) -> Any:
        if isinstance(node, list):
            return [rewrite(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "cmp" in node:
            cmp_node = dict(node["cmp"])
            cmp_node["left"] = rewrite(cmp_node.get("left"))
            cmp_node["right"] = rewrite(cmp_node.get("right"))
            return {"cmp": cmp_node}
        if "and" in node:
            return {"and": [rewrite(item) for item in node["and"]]}
        if "or" in node:
            return {"or": [rewrite(item) for item in node["or"]]}
        if "not" in node:
            return {"not": rewrite(node["not"])}
        if node.get("op") == "-" and len(node.get("args", [])) == 2 and is_now_year(node["args"][0]):
            right_refs = _filter_refs([node["args"][1]])
            replacement = next((ref for ref in temporal_refs if ref not in right_refs), None)
            if replacement:
                return _op("-", {"func": "date_part", "args": [{"lit": "year"}, {"col": replacement}]}, rewrite(node["args"][1]))
        return {key: rewrite(value) for key, value in node.items()}

    return [rewrite(filt) for filt in filters]


def _prune_raw_age_year_filters(filters: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    age_right_refs: set[str] = set()

    def collect(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                collect(item)
            return
        if not isinstance(node, dict):
            return
        cmp_node = node.get("cmp")
        if isinstance(cmp_node, dict):
            left = cmp_node.get("left")
            if (
                isinstance(left, dict)
                and left.get("op") == "-"
                and len(left.get("args", [])) == 2
                and isinstance(left["args"][1], dict)
            ):
                age_right_refs.update(_filter_refs([left["args"][1]]))
        for value in node.values():
            collect(value)

    def is_raw_year_age(node: Dict[str, Any]) -> bool:
        cmp_node = node.get("cmp")
        if not isinstance(cmp_node, dict):
            return False
        left = cmp_node.get("left")
        right = cmp_node.get("right")
        if not (
            isinstance(left, dict)
            and left.get("func") == "date_part"
            and len(left.get("args", [])) == 2
            and left["args"][0] == {"lit": "year"}
            and isinstance(right, (int, float))
            and abs(float(right)) <= 130
        ):
            return False
        refs = _filter_refs([left])
        return bool(refs & age_right_refs)

    for filt in filters:
        collect(filt)
    if not age_right_refs:
        return list(filters)
    return [filt for filt in filters if not is_raw_year_age(filt)]


def _single_filter_column(filt: Dict[str, Any], ctx: EvidenceContext) -> Optional[ColumnInfo]:
    try:
        ref = filt["cmp"]["left"]["col"]
    except Exception:
        return None
    if not isinstance(ref, str):
        return None
    table, _, name = ref.partition(".")
    for col in ctx.columns:
        if col.table == table and col.name == name:
            return col
    return None


def _single_filter_value(filt: Dict[str, Any]) -> Optional[str]:
    try:
        value = filt["cmp"]["right"]
    except Exception:
        return None
    if isinstance(value, (str, int, float)):
        return str(value)
    return None


def _column_is_temporal(col: ColumnInfo) -> bool:
    name = col.name.lower()
    return (
        col.type in {"date", "time", "datetime", "timestamp"}
        or "date" in name
        or "time" in name
        or name in {"birthday", "birth_day", "birthdate", "birth_date"}
    )


def _id_like_column(col: ColumnInfo) -> bool:
    name = col.name.lower()
    return name.endswith("id") or name.startswith("link_to_") or name.endswith("_id")


def _literal_looks_identifier(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if re.search(r"\s|'", text):
        return False
    if re.fullmatch(r"\d{3}-\d{3}-\d{4}", text):
        return False
    if re.fullmatch(r"\d+", text):
        return True
    if re.fullmatch(r"[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,}", text):
        return True
    return bool(re.search(r"\d", text) and re.fullmatch(r"[A-Za-z0-9_-]+", text))


def _literal_fits_column(col: ColumnInfo, value: Any) -> bool:
    looks_identifier = _literal_looks_identifier(value)
    if _id_like_column(col):
        return looks_identifier
    if looks_identifier:
        return False
    return True


def _literal_type_fits_column(col: ColumnInfo, value: Any) -> bool:
    if _id_like_column(col):
        return False
    if _numeric_column(col):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if _column_is_temporal(col):
        return _looks_like_date(value) or _looks_like_time(value) or _looks_like_compact_period(value)
    return isinstance(value, str) and bool(value)


def _query_command_phrase(text: str) -> bool:
    return re.match(
        r"\s*(?:tell|list|show|give|state|write|provide|identify|indicate|mention|what|which|who|name)\b",
        text or "",
        flags=re.I,
    ) is not None


def _dedupe_bool(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for item in items:
        key = repr(item)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _dedupe_selects(items: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    seen: set[str] = set()
    out: List[Dict[str, str]] = []
    for item in items:
        ref = item["ref"]
        if ref not in seen:
            seen.add(ref)
            out.append(item)
    return out


def _prefer_mapped_selects(
    selects: Sequence[Dict[str, str]],
    mapped: Sequence[Dict[str, str]],
    ctx: EvidenceContext,
) -> List[Dict[str, str]]:
    mapped_refs = {item.get("ref") for item in mapped if isinstance(item.get("ref"), str)}
    if not mapped_refs:
        return list(selects)

    mapped_keys: set[str] = set()
    for item in mapped:
        ref = item.get("ref")
        if not isinstance(ref, str):
            continue
        col = _column_by_ref(ctx, ref)
        if col is not None:
            mapped_keys.add(_norm(col.name))
        alias = item.get("alias")
        if isinstance(alias, str):
            mapped_keys.add(_norm(alias))

    if not mapped_keys:
        return list(selects)

    mapped_has_specific_name = any(
        (col := _column_by_ref(ctx, item.get("ref", ""))) is not None
        and _norm(col.name).endswith(" name")
        and _norm(col.name) != "name"
        for item in mapped
    )
    out: List[Dict[str, str]] = []
    for item in selects:
        ref = item.get("ref")
        if not isinstance(ref, str):
            out.append(item)
            continue
        col = _column_by_ref(ctx, ref)
        alias = item.get("alias")
        keys = set()
        if col is not None:
            keys.add(_norm(col.name))
        if isinstance(alias, str):
            keys.add(_norm(alias))
        if ref not in mapped_refs and keys & mapped_keys:
            continue
        if ref not in mapped_refs and mapped_has_specific_name and col is not None and _norm(col.name) == "name":
            continue
        out.append(item)
    return _dedupe_selects(out)


def _valid_distinct_with_ordering(
    distinct: bool,
    selects: Sequence[Dict[str, str]],
    order_by: Sequence[Dict[str, Any]],
) -> bool:
    if not distinct or not order_by:
        return distinct
    selected_refs = {item.get("ref") for item in selects if isinstance(item.get("ref"), str)}
    ordered_refs = _filter_refs(order_by)
    if ordered_refs - selected_refs:
        return False
    return distinct


def _ordered_distinct_keys(
    distinct: bool,
    selects: Sequence[Dict[str, str]],
    order_by: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not distinct or not order_by:
        return []
    selected_refs = {item.get("ref") for item in selects if isinstance(item.get("ref"), str)}
    keys = [
        item["by"]
        for item in order_by
        if isinstance(item, dict)
        and isinstance(item.get("by"), dict)
        and isinstance(item["by"].get("col"), str)
        and item["by"]["col"] not in selected_refs
    ]
    return _dedupe_exprs(keys)


def _prefer_order_context_selects(
    selects: Sequence[Dict[str, str]],
    order_by: Sequence[Dict[str, Any]],
    ctx: EvidenceContext,
) -> List[Dict[str, str]]:
    ordered_tables = {
        ref.split(".", 1)[0]
        for ref in _filter_refs(order_by)
        if "." in ref
    }
    if not ordered_tables:
        return list(selects)

    out: List[Dict[str, str]] = []
    for item in selects:
        ref = item.get("ref")
        col = _column_by_ref(ctx, ref) if isinstance(ref, str) else None
        if col is None or col.table in ordered_tables:
            out.append(item)
            continue
        replacement = next(
            (
                candidate
                for candidate in ctx.columns
                if candidate.table in ordered_tables and _norm(candidate.name) == _norm(col.name)
            ),
            None,
        )
        if replacement is None:
            out.append(item)
        else:
            updated = dict(item)
            updated["ref"] = replacement.ref
            updated["alias"] = _alias_for(replacement.name)
            out.append(updated)
    return _dedupe_selects(out)


def _preserve_rowwise_selects(question: str) -> bool:
    if re.search(r"\b(?:was|were|is|are|did|does)\s+(?:each|every)\b", question, re.I):
        return True
    if re.search(r"\bincurred\b", question, re.I):
        return True
    return False


def _prefer_filter_backed_selects(
    selects: Sequence[Dict[str, str]],
    filters: Sequence[Dict[str, Any]],
    ctx: EvidenceContext,
) -> List[Dict[str, str]]:
    filter_refs = _filter_refs(filters)
    filter_tables = {ref.split(".", 1)[0] for ref in filter_refs if "." in ref}
    rebound: List[Dict[str, str]] = []
    for item in selects:
        ref = item.get("ref", "")
        col = _column_by_ref(ctx, ref)
        if col is None or col.name == "id" or not _id_like_column(col) or col.table in filter_tables:
            rebound.append(item)
            continue
        candidates = [
            candidate for candidate in ctx.columns
            if candidate.table in filter_tables and candidate.name == col.name
        ]
        if len(candidates) != 1:
            rebound.append(item)
            continue
        replacement = candidates[0]
        rebound.append({"ref": replacement.ref, "alias": _alias_for(replacement.name)})
    selects = _dedupe_selects(rebound)

    selected_filter_names = {
        _norm(col.name)
        for item in selects
        for col in [_column_by_ref(ctx, item.get("ref", ""))]
        if col is not None and item.get("ref") in filter_refs
    }
    if not selected_filter_names:
        return list(selects)

    out: List[Dict[str, str]] = []
    for item in selects:
        ref = item.get("ref", "")
        col = _column_by_ref(ctx, ref)
        if col is not None and _norm(col.name) in selected_filter_names and ref not in filter_refs:
            continue
        out.append(item)
    return _dedupe_selects(out)


def _identifier_values_need_distinct(
    question: str,
    selects: Sequence[Dict[str, str]],
    ctx: EvidenceContext,
) -> bool:
    if not _selectish_question(question):
        return False
    cols = [_column_by_ref(ctx, item.get("ref", "")) for item in selects]
    cols = [col for col in cols if col is not None]
    return bool(cols) and all(
        _id_like_column(col) and ctx.primary_ids.get(col.table) != col.name
        for col in cols
    )


def _expand_sibling_measure_selects(
    selects: Sequence[Dict[str, str]],
    question: str,
    ctx: EvidenceContext,
) -> List[Dict[str, str]]:
    if not re.search(r"\b(?:status|concentration|profile|panel)\b", question, re.I):
        return list(selects)
    additions: List[Dict[str, str]] = []
    selected_refs = {item.get("ref") for item in selects}
    for item in selects:
        ref = item.get("ref")
        col = _column_by_ref(ctx, ref) if isinstance(ref, str) else None
        if col is None or not _numeric_column(col):
            continue
        tokens = _norm(col.name).split()
        if len(tokens) < 3:
            continue
        prefix = tokens[:-1]
        siblings = [
            candidate
            for candidate in ctx.columns
            if candidate.table == col.table
            and candidate.ref not in selected_refs
            and _numeric_column(candidate)
            and _norm(candidate.name).split()[:-1] == prefix
        ]
        siblings.sort(key=lambda candidate: _norm(candidate.name).split()[-1:])
        for sibling in siblings:
            additions.append({"ref": sibling.ref, "alias": _alias_for(sibling.name)})
    if not additions:
        return list(selects)
    expanded = _dedupe_selects(list(selects) + additions)
    expanded.sort(key=lambda item: _norm(_column_by_ref(ctx, item["ref"]).name).split()[-1:] if _column_by_ref(ctx, item["ref"]) else [""])
    return expanded


def _drop_computed_source_selects(
    selects: Sequence[Dict[str, str]],
    computed_selects: Sequence[Dict[str, Any]],
    question: str,
    filters: Sequence[Dict[str, Any]],
    ctx: EvidenceContext,
    order_by: Sequence[Dict[str, Any]],
) -> List[Dict[str, str]]:
    source_refs = {
        ref
        for item in computed_selects
        for ref in item.get("refs", [])
        if isinstance(ref, str)
    }
    if not source_refs:
        return list(selects)

    out: List[Dict[str, str]] = []
    ordered_refs = _filter_refs(order_by)
    for item in selects:
        ref = item.get("ref", "")
        col = _column_by_ref(ctx, ref)
        if col is not None and ref in source_refs and ref in ordered_refs and re.search(r"\b(?:oldest|youngest)\b", question, re.I):
            out.append(item)
            continue
        if col is not None and ref in source_refs and not _output_context_columns([col], question, filters, ctx):
            continue
        out.append(item)
    return _dedupe_selects(out)


def _add_ordered_computed_sources(
    selects: Sequence[Dict[str, str]],
    computed_selects: Sequence[Dict[str, Any]],
    question: str,
    order_by: Sequence[Dict[str, Any]],
    ctx: EvidenceContext,
) -> List[Dict[str, str]]:
    if not re.search(r"\b(?:oldest|youngest)\b", question, re.I):
        return list(selects)
    ordered_refs = _filter_refs(order_by)
    source_refs = {
        ref
        for item in computed_selects
        for ref in item.get("refs", [])
        if isinstance(ref, str)
    }
    additions = []
    for ref in source_refs & ordered_refs:
        col = _column_by_ref(ctx, ref)
        if col is not None:
            additions.append({"ref": ref, "alias": _alias_for(col.name)})
    return _dedupe_selects(list(selects) + additions)


def _split_computed_source_selects(
    selects: Sequence[Dict[str, str]],
    computed_selects: Sequence[Dict[str, Any]],
    order_by: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    computed_refs = {
        ref
        for item in computed_selects
        for ref in item.get("refs", [])
        if isinstance(ref, str)
    }
    trailing_refs = computed_refs & _filter_refs(order_by)
    if not trailing_refs:
        return list(selects), []
    front: List[Dict[str, str]] = []
    tail: List[Dict[str, str]] = []
    for item in selects:
        if item.get("ref") in trailing_refs:
            tail.append(item)
        else:
            front.append(item)
    return front, tail


def _sort_selects_by_question(
    selects: Sequence[Dict[str, str]],
    question: str,
    ctx: EvidenceContext,
) -> List[Dict[str, str]]:
    q_norm = _norm(question)

    def key(item: Dict[str, str]) -> Tuple[int, int]:
        if isinstance(item.get("pos"), int):
            pos = item["pos"]
        else:
            pos = 10**9
        col = _column_by_ref(ctx, item.get("ref", ""))
        if col is not None:
            pos = min(pos, _column_request_pos(col, q_norm))
        try:
            original = list(selects).index(item)
        except ValueError:
            original = 0
        return pos, original

    return sorted(selects, key=key)


def _advanced(
    dataset: str,
    select: List[Dict[str, Any]],
    *,
    where: Optional[Dict[str, Any]] = None,
    distinct: bool = False,
    distinct_on: Optional[List[Dict[str, Any]]] = None,
    order_by: Optional[List[Dict[str, Any]]] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    plan: Dict[str, Any] = {
        "version": "1.0",
        "dataset": dataset,
        "select": select,
        "limit": limit,
        "offset": 0,
    }
    if where is not None:
        plan["where"] = where
    if distinct:
        plan["distinct"] = True
    if distinct_on:
        plan["distinct_on"] = distinct_on
    if order_by:
        plan["order_by"] = order_by
    return plan


def _cmp(left: Dict[str, Any], op: str, right: Any) -> Dict[str, Any]:
    return {"cmp": {"left": left, "op": op, "right": right}}


def _column_cmp(col: ColumnInfo, op: str, right: Any) -> Dict[str, Any]:
    op = op.lower()
    if op == "=" and _looks_like_compact_period(right) and _column_is_temporal(col):
        if _text_date_like(col):
            right = str(right)
        else:
            return _cmp({"col": col.ref}, "between", _compact_period_bounds(right))
    if op == "=" and _text_date_like(col) and _looks_like_date(right):
        return _cmp({"col": col.ref}, "starts_with", right)
    return _cmp({"col": col.ref}, op, right)


def _resolve_filter_values_with_index(
    filters: Sequence[Dict[str, Any]],
    ctx: EvidenceContext,
) -> List[Dict[str, Any]]:
    if not ctx.value_index:
        return list(filters)
    out: List[Dict[str, Any]] = []
    for filt in filters:
        col = _single_filter_column(filt, ctx)
        cmp_node = filt.get("cmp") if isinstance(filt, dict) else None
        value = cmp_node.get("right") if isinstance(cmp_node, dict) else None
        if col is None or value is None:
            out.append(filt)
            continue
        known = (ctx.value_index.get(col.table) or {}).get(col.name) or []
        if isinstance(value, list):
            resolved = [_match_known_value(str(item), known) or item for item in value]
            if resolved == value:
                out.append(filt)
                continue
            updated = dict(cmp_node)
            updated["right"] = resolved
            out.append({"cmp": updated})
            continue
        if not isinstance(value, (str, int, float)):
            out.append(filt)
            continue
        match = _match_known_value(str(value), known)
        if match is None or match == value:
            out.append(filt)
            continue
        out.append(_column_cmp(col, cmp_node["op"], match))
    return _dedupe_bool(out)


def _prune_conflicting_equalities(filters: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_col: Dict[str, List[Tuple[int, Dict[str, Any], Any]]] = {}
    values_by_col: Dict[str, set[str]] = {}
    for idx, filt in enumerate(filters):
        col_ref = _single_filter_ref(filt)
        value = _single_filter_value(filt)
        if col_ref is None or value is None or not _simple_equality_filter(filt):
            continue
        by_col.setdefault(col_ref, []).append((idx, filt, value))
        values_by_col.setdefault(col_ref, set()).add(_norm(value))

    other_values: Dict[str, set[str]] = {}
    for col_ref, values in values_by_col.items():
        merged: set[str] = set()
        for other_ref, other in values_by_col.items():
            if other_ref != col_ref:
                merged.update(other)
        other_values[col_ref] = merged

    drop: set[int] = set()
    for col_ref, group in by_col.items():
        unique_values = {_norm(value) for _idx, _filt, value in group}
        if len(unique_values) <= 1:
            continue
        keep = [
            item for item in group
            if _norm(item[2]) not in other_values.get(col_ref, set())
        ]
        if not keep:
            keep = sorted(group, key=lambda item: len(str(item[2])), reverse=True)[:1]
        keep_ids = {idx for idx, _filt, _value in keep}
        for idx, _filt, _value in group:
            if idx not in keep_ids:
                drop.add(idx)
    return [filt for idx, filt in enumerate(filters) if idx not in drop]


def _prune_equalities_covered_by_in(filters: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    covered: Dict[str, set[str]] = {}
    for filt in filters:
        cmp_node = filt.get("cmp") if isinstance(filt, dict) else None
        if not isinstance(cmp_node, dict) or cmp_node.get("op") != "in":
            continue
        left = cmp_node.get("left")
        values = cmp_node.get("right")
        ref = left.get("col") if isinstance(left, dict) else None
        if isinstance(ref, str) and isinstance(values, list) and len(values) > 1:
            covered.setdefault(ref, set()).update(_norm(str(value)) for value in values)

    out: List[Dict[str, Any]] = []
    for filt in filters:
        ref = _single_filter_ref(filt)
        value = _single_filter_value(filt)
        if ref in covered and value is not None and _simple_equality_filter(filt) and _norm(value) in covered[ref]:
            continue
        out.append(filt)
    return out


def _prune_type_incompatible_filters(
    filters: Sequence[Dict[str, Any]],
    ctx: EvidenceContext,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for filt in filters:
        cmp_node = filt.get("cmp") if isinstance(filt, dict) else None
        if not isinstance(cmp_node, dict):
            out.append(filt)
            continue
        left = cmp_node.get("left")
        right = cmp_node.get("right")
        if not isinstance(left, dict) or not isinstance(left.get("col"), str):
            out.append(filt)
            continue
        if isinstance(right, (dict, list)) or right is None:
            out.append(filt)
            continue
        col = _column_by_ref(ctx, left["col"])
        if col is None:
            out.append(filt)
            continue
        if _numeric_column(col) and isinstance(right, str) and not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", right.strip()):
            continue
        out.append(filt)
    return out


def _simple_equality_filter(filt: Dict[str, Any]) -> bool:
    cmp = filt.get("cmp") if isinstance(filt, dict) else None
    return isinstance(cmp, dict) and cmp.get("op") == "=" and isinstance(cmp.get("left"), dict) and "col" in cmp.get("left", {})


def _single_filter_ref(filt: Dict[str, Any]) -> Optional[str]:
    cmp = filt.get("cmp") if isinstance(filt, dict) else None
    if not isinstance(cmp, dict):
        return None
    left = cmp.get("left")
    if isinstance(left, dict) and isinstance(left.get("col"), str):
        return left["col"]
    return None


def _and(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    clean = [item for item in items if item]
    return clean[0] if len(clean) == 1 else {"and": clean}


def _func(name: str, *args: Dict[str, Any]) -> Dict[str, Any]:
    return {"func": name, "args": list(args)}


def _op(op: str, *args: Dict[str, Any]) -> Dict[str, Any]:
    return {"op": op, "args": list(args)}


def _case_sum(cond: Dict[str, Any], value: Dict[str, Any]) -> Dict[str, Any]:
    return _func("sum", {"case": {"whens": [{"when": cond, "then": value}], "else": {"lit": 0}}})


def _pct_expr(numerator: Dict[str, Any], denominator: Dict[str, Any]) -> Dict[str, Any]:
    return _op(
        "*",
        _op("/", _op("*", numerator, {"lit": 1.0}), _func("nullif", denominator, {"lit": 0})),
        {"lit": 100.0},
    )


def _ratio_expr(numerator: Dict[str, Any], denominator: Dict[str, Any]) -> Dict[str, Any]:
    return _op("/", _op("*", numerator, {"lit": 1.0}), _func("nullif", denominator, {"lit": 0}))


def _year_expr(col: ColumnInfo) -> Dict[str, Any]:
    if _text_date_like(col):
        return _func("substr", {"col": col.ref}, {"lit": 1}, {"lit": 4})
    return _func("date_part", {"lit": "year"}, {"col": col.ref})


def _month_expr(col: ColumnInfo) -> Dict[str, Any]:
    if _text_date_like(col):
        return {
            "case": {
                "whens": [
                    {
                        "when": {
                            "cmp": {
                                "left": _func("length", {"col": col.ref}),
                                "op": "=",
                                "right": 6,
                            }
                        },
                        "then": _func("substr", {"col": col.ref}, {"lit": 5}, {"lit": 2}),
                    }
                ],
                "else": _func("substr", {"col": col.ref}, {"lit": 6}, {"lit": 2}),
            }
        }
    return _func("date_part", {"lit": "month"}, {"col": col.ref})


def _text_date_like(col: ColumnInfo) -> bool:
    ctype = (col.type or "").lower()
    cname = col.name.lower()
    textish = any(tok in ctype for tok in ("char", "text", "string"))
    dateish = (
        cname == "date"
        or cname.endswith("_date")
        or cname.endswith("date")
        or cname.endswith("month")
        or cname in {"birthday", "birth_day"}
    )
    return textish and dateish


def _numeric_column(col: ColumnInfo) -> bool:
    ctype = (col.type or "").lower()
    return any(tok in ctype for tok in ("int", "numeric", "float", "double", "real", "decimal"))


def _month_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_month_value(v) for v in value]
    try:
        return f"{int(value):02d}"
    except Exception:
        text = str(value).strip()
        if text.lower() in MONTHS:
            return MONTHS[text.lower()]
        return text


def _bool_key(item: Dict[str, Any]) -> str:
    return repr(item)


def _bool_keys(node: Optional[Dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    if not isinstance(node, dict):
        return keys
    if "cmp" in node or "not" in node or "and" in node or "or" in node:
        keys.add(_bool_key(node))
    for key in ("and", "or"):
        items = node.get(key)
        if isinstance(items, list):
            for item in items:
                keys.update(_bool_keys(item))
    return keys


def _asks_for_count(question: str) -> bool:
    if re.search(r"\b(?:how many|count)\b", question, re.I):
        return True
    if re.search(r"\b(?:what|which)\s+number\s+of\b", question, re.I):
        return True
    return re.search(
        r"\b(?:(?:what|which)\s+(?:is|are|was|were)?\s+(?:the\s+)?|(?:find|give|provide|state|list|show)\s+(?:the\s+)?)number\s+of\b",
        question,
        re.I,
    ) is not None


def _asks_to_list_answers_with_count(question: str) -> bool:
    if not _asks_for_count(question):
        return False
    return re.search(
        r"\b(?:list|state|provide|give|show|name|identify)\b[^?.;]*(?:\bids?\b|\bthem\b|\bwhich\b|\bwho\b)",
        question,
        re.I,
    ) is not None


def _count_should_use_distinct(question: str, text: str) -> bool:
    if _asks_for_distinct(text):
        return True
    if re.search(r"\bCOUNT\s*\([^)]*\)\s*(?:>=|<=|>|<|=)", text, re.I):
        return True
    if re.search(r"\btimes\b|\boccurrences?\b|\brows?\b|\brecords?\b|\bentries\b", question, re.I):
        return False
    return True


def _asks_for_distinct(text: str) -> bool:
    return re.search(
        r"\b(distinct|unique|different|disparate|what kind|which kind|what type|which type|what category|which category|state the category|list the category|state the type|list the type)\b|\bwhat\s+(?:are|is)\s+(?:the\s+)?(?:[a-z0-9_]+\s+){0,5}(?:types?|categor(?:y|ies)|kinds?)\b|\blist\s+(?:the\s+)?names?\b|\bname\s+of\b",
        text,
        re.I,
    ) is not None


def _selectish_question(question: str) -> bool:
    return re.search(r"\b(list|state|provide|give|write|show|identify|name|include|tell|what is|which)\b", question, re.I) is not None


def _limit_for_question(question: str) -> Optional[int]:
    if re.search(r"\b(top|highest|lowest|oldest|older|youngest|younger|most common|least|tallest|shortest)\b", question, re.I):
        m = re.search(r"\btop\s+(\d+)\b", question, re.I)
        return int(m.group(1)) if m else 1
    return None


def _alias_for(name: str) -> str:
    alias = re.sub(r"\W+", "_", name.strip().lower()).strip("_")
    return alias or "value"


def _parse_comparison_value(value: str, *, force_year: bool = False) -> Any:
    if " and " in value.lower():
        parts = re.split(r"\s+and\s+", value, flags=re.I)
        return [_parse_comparison_value(p, force_year=force_year) for p in parts]
    parsed = _parse_literal(value)
    if force_year and isinstance(parsed, str) and re.fullmatch(r"\d{4}", parsed):
        return int(parsed)
    return parsed


def _year_comparison_value(value: str, col: ColumnInfo) -> Any:
    parsed = _parse_comparison_value(value, force_year=not _text_date_like(col))
    if _text_date_like(col) and isinstance(parsed, int):
        return str(parsed)
    if _text_date_like(col) and isinstance(parsed, list):
        return [str(item) if isinstance(item, int) else item for item in parsed]
    return parsed


def _parse_literal(value: str) -> Any:
    value = value.strip().strip("'\"`")
    date = _date_any_order(value)
    if date:
        return date
    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    if re.fullmatch(r"[+-]?\d+\.\d+", value):
        return float(value)
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    return value


def _parse_literal_for_column(raw: str, col: ColumnInfo, op: str) -> Any:
    text = str(raw).strip()
    quoted = (len(text) >= 2 and text[0] in {"'", '"'} and text[-1] == text[0])
    inner = text[1:-1] if quoted else text
    ctype = (col.type or "").lower()
    floatish = any(tok in ctype for tok in ("float", "real", "double"))
    date = _date_any_order(inner)
    if date is not None and _column_is_temporal(col):
        return date
    if quoted and not _numeric_column(col):
        return inner
    if quoted and op == "=" and floatish and re.fullmatch(r"[+-]?\d+\.\d+", inner):
        return inner
    return _parse_literal(text)


def _looks_like_date(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"(?:19|20)\d{2}-\d{2}-\d{2}", value) is not None


def _looks_like_time(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", value) is not None


def _looks_like_compact_period(value: Any) -> bool:
    return isinstance(value, (str, int)) and re.fullmatch(r"(?:19|20)\d{2}(?:0[1-9]|1[0-2])", str(value)) is not None


def _compact_period_bounds(value: Any) -> List[str]:
    text = str(value)
    year = int(text[:4])
    month = int(text[4:])
    days = [31, 29 if _leap_year(year) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return [f"{year:04d}-{month:02d}-01", f"{year:04d}-{month:02d}-{days[month - 1]:02d}"]


def _leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _question_has_date_range(text: str) -> bool:
    return re.search(r"\b(?:between|from)\b.+\b(?:and|to)\b", text or "", re.I | re.S) is not None


def _question_has_relative_date_comparison(text: str) -> bool:
    return re.search(r"\b(?:after|before|since|until|later than|earlier than|newer than|older than)\b", text or "", re.I) is not None


def _time(text: str) -> Optional[str]:
    m = re.search(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b", text)
    if not m:
        return None
    return f"{int(m.group(1)):02d}:{int(m.group(2)):02d}:{int(m.group(3) or 0):02d}"


def _time_range(text: str) -> Optional[List[str]]:
    m = re.search(r"\b(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*(\d{1,2}:\d{2}(?::\d{2})?)\b", text)
    if not m:
        return None
    return [_time(m.group(1)) or m.group(1), _time(m.group(2)) or m.group(2)]


def _like_to_op(pattern: str) -> Tuple[str, str]:
    if pattern.startswith("%") and pattern.endswith("%"):
        return "contains", pattern.strip("%")
    if pattern.endswith("%"):
        return "starts_with", pattern.rstrip("%")
    if pattern.startswith("%"):
        return "ends_with", pattern.lstrip("%")
    return "=", pattern


def _date_any_order(text: str) -> Optional[str]:
    text = text.strip().strip("'\"`")
    m = re.search(r"\b((?:19|20)\d{2})[/-](\d{1,2})[/-](\d{1,2})\b", text)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-]((?:19|20)\d{2})\b", text)
    if m:
        return f"{int(m.group(3)):04d}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    return None


def _split_top_level_args(text: str) -> List[str]:
    args: List[str] = []
    depth = 0
    start = 0
    for idx, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                args.append(text[start:idx].strip())
                return args
            depth -= 1
        elif ch == "," and depth == 0:
            args.append(text[start:idx].strip())
            start = idx + 1
    tail = text[start:].strip()
    if tail:
        args.append(tail)
    return args


def _alias_in_text(alias: str, text_norm: str) -> bool:
    if not alias:
        return False
    if len(alias) <= 2:
        return re.search(rf"\b{re.escape(alias)}\b", text_norm) is not None
    return re.search(rf"\b{re.escape(alias)}\b", text_norm) is not None


def _norm(text: str) -> str:
    text = _strip_db_identifier(str(text))
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.replace("_", " ")
    text = re.sub(r"[^a-zA-Z0-9+.-]+", " ", text).lower()
    return re.sub(r"\s+", " ", text).strip()
