"""
schema_validator.py — Load-time validation for schema.yaml.

Validates the schema dict immediately after loading, before any compilation
or query execution. Raises SchemaError with a clear message on misconfiguration.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .exceptions import SchemaError


def validate_schema(schema: Dict[str, Any]) -> List[str]:
    """
    Validate a parsed schema dict.

    Returns a list of warning strings (non-fatal issues, e.g. primary_id
    pointing to an unknown column). Raises SchemaError on fatal issues.
    """
    warnings: List[str] = []

    tables = schema.get("tables")
    if not isinstance(tables, list) or not tables:
        raise SchemaError(
            "schema.yaml must contain a non-empty 'tables' list."
        )

    for i, t in enumerate(tables):
        loc = f"tables[{i}]"

        if not isinstance(t, dict):
            raise SchemaError(f"{loc}: each table must be an object.")

        name = t.get("name")
        if not isinstance(name, str) or not name:
            raise SchemaError(f"{loc}: 'name' is required and must be a non-empty string.")

        db_table = t.get("db_table")
        if not isinstance(db_table, str) or not db_table:
            raise SchemaError(
                f"{loc} ('{name}'): 'db_table' is required and must be a non-empty string. "
                f"Example: db_table: '\"MyTable\"'"
            )

        columns = t.get("columns")
        if not isinstance(columns, list) or not columns:
            raise SchemaError(
                f"{loc} ('{name}'): 'columns' is required and must be a non-empty list."
            )

        column_names = set()
        for j, c in enumerate(columns):
            cloc = f"{loc}.columns[{j}]"
            if not isinstance(c, dict):
                raise SchemaError(f"{cloc}: each column must be an object.")
            cname = c.get("name")
            if not isinstance(cname, str) or not cname:
                raise SchemaError(f"{cloc}: column 'name' is required.")
            db_column = c.get("db_column")
            if not isinstance(db_column, str) or not db_column:
                raise SchemaError(
                    f"{cloc} ('{cname}'): column 'db_column' is required."
                )
            column_names.add(cname)

        # Validate primary_id if declared — non-fatal warning if column not found
        primary_id = t.get("primary_id")
        if primary_id is not None:
            if not isinstance(primary_id, str) or not primary_id:
                raise SchemaError(
                    f"{loc} ('{name}'): 'primary_id' must be a non-empty string if declared."
                )
            if primary_id not in column_names:
                warnings.append(
                    f"Warning: table '{name}' declares primary_id='{primary_id}' "
                    f"but no column with that name exists in the columns list. "
                    f"Grain checks for this table will be silently skipped. "
                    f"Available columns: {sorted(column_names)}"
                )

        kso = t.get("keyword_search_or")
        if kso is not None:
            if not isinstance(kso, list) or len(kso) < 1:
                raise SchemaError(
                    f"{loc} ('{name}'): 'keyword_search_or' must be a non-empty list of column names."
                )
            for ki, col in enumerate(kso):
                if not isinstance(col, str) or not col:
                    raise SchemaError(
                        f"{loc}.keyword_search_or[{ki}]: must be a non-empty string."
                    )
                if col not in column_names:
                    raise SchemaError(
                        f"{loc} ('{name}'): keyword_search_or references unknown column {col!r}."
                    )

    # Validate links
    known_table_names = {t["name"] for t in tables if isinstance(t, dict) and t.get("name")}
    for i, link in enumerate(schema.get("links", []) or []):
        lloc = f"links[{i}]"
        if not isinstance(link, dict):
            raise SchemaError(f"{lloc}: each link must be an object.")

        lname = link.get("name")
        if not isinstance(lname, str) or not lname:
            raise SchemaError(f"{lloc}: link 'name' is required.")

        from_table = link.get("from_table")
        to_table = link.get("to_table")

        if from_table not in known_table_names:
            raise SchemaError(
                f"{lloc} ('{lname}'): from_table='{from_table}' is not a known table name."
            )
        if to_table not in known_table_names:
            raise SchemaError(
                f"{lloc} ('{lname}'): to_table='{to_table}' is not a known table name."
            )

        jt_raw = link.get("join_type", "left")
        if not isinstance(jt_raw, str) or not jt_raw:
            raise SchemaError(
                f"{lloc} ('{lname}'): 'join_type' must be a non-empty string "
                f"('left' or 'inner'), got {jt_raw!r}."
            )
        jt = jt_raw.lower()
        if jt not in {"left", "inner"}:
            raise SchemaError(
                f"{lloc} ('{lname}'): join_type must be 'left' or 'inner', got {jt_raw!r}."
            )

        on = link.get("on")
        # Guard against YAML parsing `on:` as boolean True (YAML 1.1 reserved word).
        # Always quote `"on":` in schema.yaml to avoid this.
        if on is True or on is False:
            raise SchemaError(
                f"{lloc} ('{lname}'): 'on' was parsed as a boolean by YAML — "
                f"this happens because 'on' is a reserved word in YAML 1.1. "
                f"Fix: quote it in schema.yaml as '\"on\":'  (with double-quotes)."
            )
        if not isinstance(on, list) or not on:
            raise SchemaError(
                f"{lloc} ('{lname}'): 'on' must be a non-empty list of join conditions."
            )

    return warnings
