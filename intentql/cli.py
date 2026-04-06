"""
cli.py — IntentQL command-line interface.

Commands:
    intentql init     — Introspect a Postgres database and generate schema.yaml
    intentql describe — Enrich schema.yaml with LLM-generated column descriptions
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from sqlalchemy import create_engine, inspect, text as sqla_text
from sqlalchemy.engine import Engine


# ── Type mapping ────────────────────────────────────────────────────────────

_PG_TYPE_MAP = {
    "integer": "integer",
    "bigint": "bigint",
    "smallint": "integer",
    "serial": "integer",
    "bigserial": "bigint",
    "real": "float",
    "double precision": "float",
    "numeric": "numeric",
    "decimal": "numeric",
    "boolean": "boolean",
    "date": "date",
    "timestamp without time zone": "timestamp",
    "timestamp with time zone": "timestamp",
    "character varying": "varchar",
    "character": "varchar",
    "text": "text",
    "uuid": "uuid",
    "json": "json",
    "jsonb": "jsonb",
}


def _map_pg_type(raw: str) -> str:
    raw_lower = raw.lower().split("(")[0].strip()
    return _PG_TYPE_MAP.get(raw_lower, "varchar")


# ── Introspection ───────────────────────────────────────────────────────────

def _needs_quoting(name: str) -> bool:
    return not name.islower() or not name.replace("_", "").isalnum()


def _quote_if_needed(name: str) -> str:
    if _needs_quoting(name):
        return f'"{name}"'
    return name


def _detect_primary_key(inspector: Any, table_name: str, schema: str) -> Optional[str]:
    try:
        pk = inspector.get_pk_constraint(table_name, schema=schema)
        cols = pk.get("constrained_columns", [])
        if len(cols) == 1:
            return cols[0]
    except Exception:
        pass
    return None


def _detect_primary_date(columns: List[Dict[str, Any]]) -> Optional[str]:
    """Heuristic: find the most likely date column for time-range filters."""
    date_cols = [c for c in columns if c["type"] in ("date", "timestamp")]
    if not date_cols:
        return None

    priority_names = [
        "created_at", "created_date", "entry_date", "date_created",
        "order_date", "event_date", "timestamp", "created",
    ]
    for pname in priority_names:
        for c in date_cols:
            if c["name"].lower() == pname or c["logical"].lower() == pname:
                return c["logical"]

    return date_cols[0]["logical"]


def _make_logical_name(db_name: str) -> str:
    """Convert a physical column/table name to a snake_case logical name."""
    import re
    name = db_name.strip('"')
    name = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)
    name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    return name.lower().replace(" ", "_").replace("-", "_")


def _detect_foreign_keys(inspector: Any, table_name: str, schema: str) -> List[Dict[str, Any]]:
    try:
        fks = inspector.get_foreign_keys(table_name, schema=schema)
        return fks
    except Exception:
        return []


def introspect_database(
    db_url: str,
    schema_name: str = "public",
    exclude_tables: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Connect to Postgres and build a schema dict from introspection."""
    engine = create_engine(db_url)
    inspector = inspect(engine)
    exclude = set(exclude_tables or [])

    table_names = inspector.get_table_names(schema=schema_name)
    tables: List[Dict[str, Any]] = []
    all_fks: List[Dict[str, Any]] = []

    for tname in sorted(table_names):
        if tname in exclude:
            continue

        logical_table = _make_logical_name(tname)
        db_table = _quote_if_needed(tname)

        raw_columns = inspector.get_columns(tname, schema=schema_name)
        columns: List[Dict[str, Any]] = []
        col_meta: List[Dict[str, Any]] = []

        for col in raw_columns:
            col_name = col["name"]
            col_type_str = str(col["type"])
            logical_col = _make_logical_name(col_name)
            mapped_type = _map_pg_type(col_type_str)
            db_col = _quote_if_needed(col_name)

            entry: Dict[str, Any] = {
                "name": logical_col,
                "db_column": db_col,
                "type": mapped_type,
            }
            columns.append(entry)
            col_meta.append({**entry, "logical": logical_col})

        table_entry: Dict[str, Any] = {
            "name": logical_table,
            "db_table": db_table,
            "columns": columns,
        }

        pk = _detect_primary_key(inspector, tname, schema_name)
        if pk:
            table_entry["primary_id"] = _make_logical_name(pk)

        pdate = _detect_primary_date(col_meta)
        if pdate:
            table_entry["primary_date"] = pdate

        text_cols = [c["name"] for c in columns if c["type"] in ("varchar", "text")]
        if len(text_cols) >= 2:
            desc_like = [c for c in text_cols if any(
                kw in c for kw in ("description", "desc", "keyword", "name", "title", "subject")
            )]
            if len(desc_like) >= 2:
                table_entry["keyword_search_or"] = desc_like[:4]

        tables.append(table_entry)

        fks = _detect_foreign_keys(inspector, tname, schema_name)
        for fk in fks:
            all_fks.append({
                "from_table": logical_table,
                "from_columns": [_make_logical_name(c) for c in fk["constrained_columns"]],
                "to_table": _make_logical_name(fk["referred_table"]),
                "to_columns": [_make_logical_name(c) for c in fk["referred_columns"]],
            })

    links: List[Dict[str, Any]] = []
    table_names_set = {t["name"] for t in tables}
    for fk in all_fks:
        if fk["to_table"] not in table_names_set:
            continue
        if len(fk["from_columns"]) != 1:
            continue
        link = {
            "name": f"{fk['from_table']}_to_{fk['to_table']}",
            "from_table": fk["from_table"],
            "to_table": fk["to_table"],
            "join_type": "left",
            "on": [{
                "left": f"{fk['from_table']}.{fk['from_columns'][0]}",
                "op": "=",
                "right": f"{fk['to_table']}.{fk['to_columns'][0]}",
            }],
        }
        links.append(link)

    result: Dict[str, Any] = {
        "version": 1,
        "dialect": "postgres",
        "tables": tables,
    }
    if links:
        result["links"] = links

    engine.dispose()
    return result


def _write_schema(schema: Dict[str, Any], output_path: str) -> None:
    """Write schema dict to YAML with proper quoting for 'on' keys."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    yaml_str = yaml.dump(
        schema,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    # YAML parses bare `on:` as boolean True; ensure it's quoted
    yaml_str = yaml_str.replace("\n    on:\n", '\n    "on":\n')

    out.write_text(yaml_str, encoding="utf-8")
    table_count = len(schema.get("tables", []))
    col_count = sum(len(t.get("columns", [])) for t in schema.get("tables", []))
    link_count = len(schema.get("links", []))
    print(f"Schema written to {out}")
    print(f"  {table_count} tables, {col_count} columns, {link_count} links")


# ── Describe (LLM enrichment) ──────────────────────────────────────────────

def _sample_values(engine: Engine, db_table: str, db_column: str, limit: int = 5) -> List[str]:
    try:
        sql = sqla_text(
            f"SELECT DISTINCT {db_column} FROM {db_table} "
            f"WHERE {db_column} IS NOT NULL "
            f"ORDER BY {db_column} LIMIT :lim"
        )
        with engine.connect() as conn:
            rows = conn.execute(sql, {"lim": limit}).fetchall()
        return [str(r[0]).strip() for r in rows if r[0] and str(r[0]).strip()]
    except Exception:
        return []


_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_MODEL = "gpt-4o-mini"


def _chat_completion(
    prompt: str,
    *,
    api_key: str,
    base_url: str = _DEFAULT_BASE_URL,
    model: str = _DEFAULT_MODEL,
) -> str:
    """Call any OpenAI-compatible chat completions endpoint via raw HTTP."""
    import urllib.request
    import json as _json

    url = f"{base_url.rstrip('/')}/chat/completions"
    body = _json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 1500,
    }).encode()

    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = _json.loads(resp.read())

    return data["choices"][0]["message"]["content"].strip()


def describe_schema(
    schema_path: str,
    db_url: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: str = _DEFAULT_BASE_URL,
    model: str = _DEFAULT_MODEL,
) -> None:
    """Enrich schema.yaml with LLM-generated descriptions using sample data."""
    api_key = api_key or os.getenv("LLM_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print(
            "Error: No API key found. Set LLM_API_KEY (or OPENAI_API_KEY) "
            "environment variable, or pass --api-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    schema = yaml.safe_load(Path(schema_path).read_text()) or {}

    engine = None
    if db_url:
        engine = create_engine(db_url)

    tables = schema.get("tables", [])
    for table in tables:
        tname = table.get("name", "")
        db_table = table.get("db_table", tname)
        columns = table.get("columns", [])

        col_info_lines = []
        for col in columns:
            cname = col.get("name", "")
            ctype = col.get("type", "")
            db_col = col.get("db_column", cname)

            samples = []
            if engine:
                samples = _sample_values(engine, db_table, db_col)

            sample_str = f" (sample values: {samples})" if samples else ""
            col_info_lines.append(f"  - {cname} ({ctype}){sample_str}")

        pk = table.get("primary_id", "")
        pdate = table.get("primary_date", "")
        kso = table.get("keyword_search_or", [])

        prompt = (
            f"You are documenting a database schema for an AI query engine.\n"
            f"Table: {tname}\n"
            f"  primary_id: {pk or 'none'}\n"
            f"  primary_date: {pdate or 'none'}\n"
            f"  keyword_search_or: {kso or 'none'}\n"
            f"Columns:\n" + "\n".join(col_info_lines) + "\n\n"
            f"Write a one-line description for the table and a one-line description for each column. "
            f"Be concise. Mention what the column represents and any useful query guidance "
            f"(e.g., 'use for time-range filters', 'values are in UPPER CASE').\n"
            f"Format your response as YAML:\n"
            f"table_description: \"...\"\n"
            f"columns:\n"
            f"  column_name: \"description\"\n"
            f"  column_name: \"description\"\n"
        )

        print(f"Describing table: {tname}...", file=sys.stderr)

        try:
            raw = _chat_completion(
                prompt, api_key=api_key, base_url=base_url, model=model,
            )
            raw = raw.replace("```yaml", "").replace("```", "").strip()
            descriptions = yaml.safe_load(raw) or {}
        except Exception as exc:
            print(f"  Failed: {exc}", file=sys.stderr)
            continue

        td = descriptions.get("table_description", "")
        if td:
            table["description"] = td

        col_descs = descriptions.get("columns", {})
        for col in columns:
            cname = col.get("name", "")
            if cname in col_descs and col_descs[cname]:
                col["description"] = col_descs[cname]

    if engine:
        engine.dispose()

    _write_schema(schema, schema_path)
    print("Descriptions added successfully.")


# ── CLI entry point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="intentql",
        description="IntentQL — guided Postgres SQL from natural language",
    )
    subparsers = parser.add_subparsers(dest="command")

    # intentql init
    init_parser = subparsers.add_parser(
        "init",
        help="Introspect a Postgres database and generate schema.yaml",
    )
    init_parser.add_argument(
        "--db",
        required=True,
        help="Database URL (e.g., postgresql://user:pass@host/db)",
    )
    init_parser.add_argument(
        "--output", "-o",
        default="config/schema.yaml",
        help="Output path for schema.yaml (default: config/schema.yaml)",
    )
    init_parser.add_argument(
        "--schema",
        default="public",
        help="Postgres schema to introspect (default: public)",
    )
    init_parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Table names to exclude",
    )

    # intentql describe
    desc_parser = subparsers.add_parser(
        "describe",
        help="Enrich schema.yaml with LLM-generated column descriptions",
    )
    desc_parser.add_argument(
        "--schema",
        default="config/schema.yaml",
        help="Path to schema.yaml (default: config/schema.yaml)",
    )
    desc_parser.add_argument(
        "--db",
        default=None,
        help="Database URL for sampling values (optional, improves descriptions)",
    )
    desc_parser.add_argument(
        "--api-key",
        default=None,
        help="LLM API key (default: reads LLM_API_KEY or OPENAI_API_KEY env var)",
    )
    desc_parser.add_argument(
        "--base-url",
        default=_DEFAULT_BASE_URL,
        help=f"OpenAI-compatible API base URL (default: {_DEFAULT_BASE_URL})",
    )
    desc_parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"Model name (default: {_DEFAULT_MODEL})",
    )

    args = parser.parse_args()

    if args.command == "init":
        print(f"Connecting to database...")
        schema = introspect_database(
            db_url=args.db,
            schema_name=args.schema,
            exclude_tables=args.exclude,
        )
        _write_schema(schema, args.output)

    elif args.command == "describe":
        if not Path(args.schema).exists():
            print(f"Error: {args.schema} not found. Run 'intentql init' first.", file=sys.stderr)
            sys.exit(1)
        describe_schema(
            schema_path=args.schema,
            db_url=args.db,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
