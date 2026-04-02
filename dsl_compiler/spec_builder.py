"""
Programmatic spec builder.

Reads schema.yaml and generates a minimal queryplan_spec.yaml
containing only the tables, columns, operators, and auto-generated examples
relevant to that schema.

Usage (CLI):
    python -m dsl_compiler.spec_builder \
        --schema config/schema.yaml \
        --output config/queryplan_spec_generated.yaml

Usage (programmatic):
    from dsl_compiler.spec_builder import build_spec, write_spec
    spec = build_spec("config/schema.yaml")
    write_spec(spec, "config/queryplan_spec_generated.yaml")
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_spec(schema_path: str | Path) -> Dict[str, Any]:
    """
    Build a minimal queryplan spec dict from a schema.yaml file.
    Returns a dict suitable for yaml.dump().
    """
    schema = _load_schema(schema_path)
    tables = schema.get("tables", [])

    spec: Dict[str, Any] = {}

    spec["version"] = 1
    spec["name"] = "QCE QueryPlan Spec (auto-generated)"
    spec["description"] = (
        "Auto-generated from schema.yaml. "
        "Instructions for generating QueryPlan JSON for the QCE compiler. "
        "Output MUST be valid JSON only (no markdown, no prose, no SQL)."
    )

    spec["system_instructions"] = _build_system_instructions(tables)
    spec["defaults"] = {"limit": 100, "offset": 0, "max_limit": 1000}
    spec["operators_supported"] = _operators_block()
    spec["schema_summary"] = _build_schema_summary(tables)
    spec["structural_invariants"] = _structural_invariants()
    spec["plan_construction_procedure"] = _plan_construction_procedure()
    spec["semantics_rules"] = _semantics_rules(tables)
    spec["legacy_queryplan_format"] = _legacy_format_shape()
    spec["rollup"] = _rollup_block()
    spec["validation_checklist"] = _validation_checklist()
    spec["examples"] = _build_examples(tables)

    return spec


# Alias for backward compatibility — planner.py imports this name
build_minimal_queryplan_spec = build_spec


def write_spec(spec: Dict[str, Any], output_path: str | Path) -> None:
    """Write the spec dict to a YAML file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.dump(spec, sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    print(f"[spec_builder] Written → {out}  ({out.stat().st_size:,} bytes)")


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------

def _load_schema(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"schema.yaml not found: {p}")
    data = yaml.safe_load(p.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"schema.yaml must be a YAML mapping: {p}")
    return data


def _build_system_instructions(tables: list) -> str:
    table_names = [t["name"] for t in tables if "name" in t]
    names_str = ", ".join(f'"{n}"' for n in table_names)
    return (
        "You are a QueryPlan generator.\n"
        "Your job is to convert a natural-language question into a JSON QueryPlan object.\n"
        "You MUST output JSON only — no markdown, no prose, no SQL.\n\n"
        "RULES:\n"
        f"- dataset MUST be one of: {names_str}\n"
        "- Use ONLY the logical column names listed in schema_summary below.\n"
        "- Do NOT use physical DB column names or write SQL.\n"
        "- Always include limit and offset.\n"
        "- Give every dimension and metric an alias.\n"
        "- metric aliases must be unique within the plan.\n"
        "- Prefer consistent, deterministic plans: same question → same JSON.\n"
    )


def _build_schema_summary(tables: list) -> Dict[str, Any]:
    """
    Compact schema summary — only what the LLM needs to reference columns correctly.
    Excludes db_column (physical names) to avoid confusion.
    """
    summary: Dict[str, Any] = {}
    for t in tables:
        name = t.get("name")
        if not name:
            continue
        entry: Dict[str, Any] = {}
        if t.get("description"):
            entry["description"] = t["description"].strip()
        primary_id = t.get("primary_id")
        if primary_id:
            entry["primary_id"] = primary_id
            entry["primary_id_note"] = (
                f"Use count_distinct('{primary_id}') when counting distinct {name}."
            )
        cols = []
        for c in t.get("columns", []):
            col_entry: Dict[str, Any] = {
                "name": c["name"],
                "type": c.get("type", "varchar"),
            }
            if c.get("description"):
                # Keep description short — first sentence only
                desc = c["description"].strip()
                first_sentence = desc.split(".")[0].strip()
                if first_sentence:
                    col_entry["description"] = first_sentence
            cols.append(col_entry)
        entry["columns"] = cols
        summary[name] = entry
    return summary


def _operators_block() -> Dict[str, Any]:
    return {
        "comparison": ["=", "!=", ">", ">=", "<", "<="],
        "membership": ["in", "not_in"],
        "text": ["contains", "not_contains", "starts_with", "ends_with"],
        "null_checks": ["is_null", "is_not_null"],
    }


def _structural_invariants() -> str:
    return (
        "DATASET: must be a logical table name from schema_summary.\n\n"
        "DIMENSIONS: list of {field, alias} — alias is REQUIRED.\n\n"
        "METRICS: list of {agg, field, alias} — alias is REQUIRED and unique.\n"
        "  Allowed agg: count, count_distinct, sum, avg, min, max.\n"
        "  count: field may be '*' or a logical column.\n"
        "  count_distinct/sum/avg/min/max: field MUST be a logical column, NOT '*'.\n\n"
        "FILTERS: list of {field, op, value}.\n"
        "  'contains' = case-insensitive substring match (ILIKE %value%).\n\n"
        "ORDER_BY: must reference an existing dimension alias or metric alias.\n\n"
        "LIMIT/OFFSET: always present. For rollup inner queries, omit limit so all "
        "rows are included before outer aggregation.\n\n"
        "ROLLUP: if present, rollup.metrics[*].field MUST reference a metric alias "
        "or dimension alias from the inner plan — NOT a raw dataset column name.\n"
    )


def _plan_construction_procedure() -> str:
    return (
        "1) Choose dataset: the ONE table that contains the needed information.\n"
        "2) Extract filters: convert constraints to filter objects using logical field names.\n"
        "   Use op='contains' for fuzzy text matching (building names, keywords, etc.).\n"
        "3) Decide output shape:\n"
        "   - List of entities: dimensions + optional metrics.\n"
        "   - Single scalar: metrics only (no dimensions), or dimensions + rollup.\n"
        "4) Decide grouping: if question says 'by X' / 'per X' / 'for each X', "
        "put X in dimensions.\n"
        "5) Decide metrics: use exactly the aggregations requested. "
        "Always give each metric an alias.\n"
        "6) Two-step aggregation (e.g. 'average per building'):\n"
        "   - Inner: group by dimension, compute per-group metric with alias.\n"
        "   - Rollup: compute avg/stddev/etc over that INNER metric alias.\n"
        "   - OMIT limit from inner plan so all groups are included.\n"
        "   - rollup.limit=1 for scalar result.\n"
        "7) Order + limit: for 'top N' set order_by metric alias desc + limit=N.\n"
    )


def _semantics_rules(tables: list) -> str:
    lines = [
        "- Use 'contains' for fuzzy name/keyword matching (compiled as ILIKE %...%).",
        "- Use 'is_not_null' when excluding missing values is necessary for correctness.",
        "- Do NOT use avg(*) or avg(null) — avg/sum/min/max require a real column field.",
        "- For time-based filters use the $relative_date sentinel (never a SQL expression):",
        "    last 7 days:  {\"$relative_date\": {\"op\": \"now_minus_days\", \"days\": 7}}",
        "    last 24 hrs:  {\"$relative_date\": {\"op\": \"now_minus_hours\", \"hours\": 24}}",
        "    today:        {\"$relative_date\": {\"op\": \"today\"}}",
        "    last calendar year (two filters on the date column, AND):",
        "      >= {\"$relative_date\": {\"op\": \"calendar_year_start\", \"year_offset\": -1}}",
        "      <  {\"$relative_date\": {\"op\": \"calendar_year_start\", \"year_offset\": 0}}",
        "- Topic/trade keywords (plumbing, etc.): OR \"contains\" across all plausible string columns",
        "  for that table from schema_summary — do not use only one column unless the question names it.",
        "- Do NOT map trade names onto order_type/order_category unless the schema says so;",
        "  those columns are usually workflow (PLANNED, HOUSING, PREVENTIVE, …), not crafts.",
    ]
    # Per-table grain rules + optional keyword_search_or hints
    for t in tables:
        name = t.get("name")
        pid = t.get("primary_id")
        if pid and name:
            lines.append(
                f"- For '{name}': when counting distinct {name}, "
                f"use count_distinct('{pid}') — '{pid}' is the primary identifier."
            )
        kso = t.get("keyword_search_or")
        if name and isinstance(kso, list) and len(kso) >= 2:
            cols = ", ".join(str(c) for c in kso if isinstance(c, str))
            lines.append(
                f"- Table '{name}' declares keyword_search_or [{cols}]: use advanced `where.or` of "
                f"`contains` on those columns for keyword search; legacy filters are ANDed."
            )
    return "\n".join(lines)


def _legacy_format_shape() -> Dict[str, Any]:
    return {
        "description": (
            "Use this format for all standard analytics queries. "
            "Easier for LLMs and covers the vast majority of use cases."
        ),
        "shape": {
            "version": "1.0",
            "dataset": "<table name from schema_summary>",
            "dimensions": [{"field": "<logical_column>", "alias": "<string>"}],
            "metrics": [{"agg": "count|count_distinct|sum|avg|min|max", "field": "<logical_column_or_*>", "alias": "<string>"}],
            "filters": [{"field": "<logical_column>", "op": "<operator>", "value": "<any or null>"}],
            "order_by": [{"by": "<dimension_alias_or_metric_alias>", "dir": "asc|desc"}],
            "limit": "<int>",
            "offset": "<int>",
            "rollup": "<optional — see rollup block>",
        },
    }


def _rollup_block() -> Dict[str, Any]:
    return {
        "description": (
            "Use rollup for aggregate-of-aggregates: "
            "'average per X', 'stddev per X', 'total of totals', etc. "
            "The inner query groups and computes per-group values. "
            "The rollup outer query aggregates those values."
        ),
        "shape": {
            "metrics": [{"agg": "avg|sum|min|max|count|count_distinct|stddev|variance", "field": "<inner_alias>", "alias": "<string>"}],
            "dimensions": "<optional outer grouping>",
            "filters": "<optional filters on inner outputs>",
            "order_by": "<optional>",
            "limit": "<int — use 1 for scalar result>",
            "offset": "<int>",
        },
        "critical": (
            "rollup.metrics[*].field MUST be an alias from the inner plan "
            "(a metric alias or dimension alias), NOT a raw dataset column name."
        ),
    }


def _validation_checklist() -> str:
    return (
        "Before outputting JSON, verify:\n"
        "- dataset is a table name from schema_summary\n"
        "- every dimension has {field, alias}\n"
        "- every metric has {agg, field, alias} with unique aliases\n"
        "- no metric uses avg/sum/min/max with field='*' or field=null\n"
        "- order_by.by references a dimension alias or metric alias\n"
        "- limit and offset are present\n"
        "- if rollup: rollup.metrics is non-empty, "
        "rollup.metrics[*].field references an inner alias\n"
    )


def _build_examples(tables: list) -> List[Dict[str, Any]]:
    """
    Generate minimal but concrete examples from the actual schema tables.
    Covers: scalar count, list with filter, group-by, rollup.
    """
    examples = []
    if not tables:
        return examples

    # Pick the first table with a primary_id for count example
    count_table = next((t for t in tables if t.get("primary_id")), tables[0])
    count_table_name = count_table["name"]
    primary_id = count_table.get("primary_id") or (count_table.get("columns", [{}])[0].get("name", "id"))

    examples.append({
        "question": f"How many {count_table_name} are there in total?",
        "plan": {
            "version": "1.0",
            "dataset": count_table_name,
            "dimensions": [],
            "metrics": [{"agg": "count_distinct", "field": primary_id, "alias": f"total_{count_table_name}"}],
            "filters": [],
            "order_by": [],
            "limit": 1,
            "offset": 0,
        },
    })

    # Find a text column for filter example
    text_table = tables[0]
    text_col = next(
        (c for c in text_table.get("columns", []) if "name" in c and ("name" in c["name"] or "description" in c["name"] or "code" in c["name"])),
        text_table.get("columns", [{}])[0] if text_table.get("columns") else None,
    )

    if text_col:
        examples.append({
            "question": f"Show me {text_table['name']} where {text_col['name']} contains 'example'",
            "plan": {
                "version": "1.0",
                "dataset": text_table["name"],
                "dimensions": [],
                "metrics": [],
                "filters": [{"field": text_col["name"], "op": "contains", "value": "example"}],
                "order_by": [],
                "limit": 100,
                "offset": 0,
            },
        })

    # Group-by example: pick a table with a categorical column and a primary_id
    group_table = next(
        (t for t in tables if t.get("primary_id") and len(t.get("columns", [])) >= 2),
        None,
    )
    if group_table:
        pid = group_table["primary_id"]
        # Pick a categorical column (not the primary_id)
        cat_col = next(
            (c for c in group_table.get("columns", [])
             if c["name"] != pid and c.get("type", "varchar") in ("varchar", "text", "char")),
            None,
        )
        if cat_col:
            examples.append({
                "question": f"How many {group_table['name']} per {cat_col['name']}?",
                "plan": {
                    "version": "1.0",
                    "dataset": group_table["name"],
                    "dimensions": [{"field": cat_col["name"], "alias": cat_col["name"]}],
                    "metrics": [{"agg": "count_distinct", "field": pid, "alias": f"{group_table['name']}_count"}],
                    "filters": [],
                    "order_by": [{"by": f"{group_table['name']}_count", "dir": "desc"}],
                    "limit": 100,
                    "offset": 0,
                },
            })

    # Rollup example: pick a table with a primary_id and a grouping column
    rollup_table = next(
        (t for t in tables if t.get("primary_id") and len(t.get("columns", [])) >= 2),
        None,
    )
    if rollup_table:
        pid = rollup_table["primary_id"]
        cat_col = next(
            (c for c in rollup_table.get("columns", [])
             if c["name"] != pid and c.get("type", "varchar") in ("varchar", "text", "char")),
            None,
        )
        if cat_col:
            inner_alias = f"{rollup_table['name']}_per_{cat_col['name']}"
            examples.append({
                "question": f"What is the average number of {rollup_table['name']} per {cat_col['name']}?",
                "note": (
                    "Two-step rollup: inner groups by dimension, "
                    "outer computes avg over inner alias."
                ),
                "plan": {
                    "version": "1.0",
                    "dataset": rollup_table["name"],
                    "dimensions": [{"field": cat_col["name"], "alias": cat_col["name"]}],
                    "metrics": [{"agg": "count_distinct", "field": pid, "alias": inner_alias}],
                    "filters": [{"field": cat_col["name"], "op": "is_not_null", "value": None}],
                    "order_by": [],
                    "offset": 0,
                    # NOTE: limit intentionally omitted — rollup needs ALL rows
                    "rollup": {
                        "metrics": [{"agg": "avg", "field": inner_alias, "alias": f"avg_{inner_alias}"}],
                        "limit": 1,
                        "offset": 0,
                    },
                },
            })

    return examples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a minimal queryplan_spec.yaml from schema.yaml"
    )
    parser.add_argument(
        "--schema",
        default="config/schema.yaml",
        help="Path to schema.yaml (default: config/schema.yaml)",
    )
    parser.add_argument(
        "--output",
        default="config/queryplan_spec_generated.yaml",
        help="Output path (default: config/queryplan_spec_generated.yaml)",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_only",
        help="Print to stdout instead of writing a file",
    )
    args = parser.parse_args()

    spec = build_spec(args.schema)

    if args.print_only:
        print(yaml.dump(spec, sort_keys=False, allow_unicode=True, default_flow_style=False))
    else:
        write_spec(spec, args.output)
        # Print size comparison if original spec exists
        original = Path(args.schema).parent / "queryplan_spec.yaml"
        if original.exists():
            orig_size = original.stat().st_size
            new_size = Path(args.output).stat().st_size
            reduction = round((1 - new_size / orig_size) * 100)
            print(f"[spec_builder] Original spec: {orig_size:,} bytes")
            print(f"[spec_builder] Generated spec: {new_size:,} bytes")
            print(f"[spec_builder] Size reduction: {reduction}%")


if __name__ == "__main__":
    main()
