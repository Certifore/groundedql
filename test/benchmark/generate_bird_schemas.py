"""
Generate IntentQL schemas for BIRD Mini-Dev.

The PostgreSQL dump stores all BIRD Mini-Dev tables in one database, while the
benchmark examples are grouped by ``db_id``.  This script introspects the shared
Postgres database once and writes one filtered ``schema.yaml`` per BIRD db_id.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intentql.cli import _make_logical_name, _write_schema, introspect_database
from intentql.llm_adapters import env_value


def _candidate_dev_tables_files(root: Path) -> list[Path]:
    return [
        root / "dev_tables.json",
        root / "MINIDEV" / "dev_tables.json",
        root / "minidev" / "MINIDEV" / "dev_tables.json",
        root / "mini_dev_data" / "dev_tables.json",
    ]


def find_dev_tables_file(root: Path, explicit: str | None = None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"BIRD dev_tables file not found: {path}")
        return path

    for path in _candidate_dev_tables_files(root):
        if path.exists():
            return path

    tried = "\n  ".join(str(p) for p in _candidate_dev_tables_files(root))
    raise FileNotFoundError(f"No BIRD dev_tables.json file found. Tried:\n  {tried}")


def load_dev_tables(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list in {path}")
    return [item for item in raw if isinstance(item, dict) and item.get("db_id")]


def _column_ref(dev_meta: dict[str, Any], column_idx: int) -> tuple[str, str] | None:
    column_names = dev_meta.get("column_names_original") or []
    table_names = dev_meta.get("table_names_original") or []
    try:
        table_idx, column_name = column_names[column_idx]
        if table_idx < 0:
            return None
        return (
            _make_logical_name(str(table_names[table_idx])),
            _make_logical_name(str(column_name)),
        )
    except (IndexError, TypeError, ValueError):
        return None


def _collapsed(name: str) -> str:
    return name.replace("_", "").lower()


def _column_lookup(tables: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for table in tables:
        table_name = str(table.get("name"))
        table_lookup: dict[str, str] = {}
        for column in table.get("columns", []) or []:
            column_name = str(column.get("name"))
            table_lookup[column_name] = column_name
            table_lookup[_collapsed(column_name)] = column_name
        lookup[table_name] = table_lookup
    return lookup


def _resolve_column(column_lookup: dict[str, dict[str, str]], table: str, column: str) -> str:
    table_lookup = column_lookup.get(table, {})
    return table_lookup.get(column) or table_lookup.get(_collapsed(column)) or column


def _metadata_links(
    dev_meta: dict[str, Any],
    wanted_tables: set[str],
    column_lookup: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    links = []
    seen = set()

    for pair in dev_meta.get("foreign_keys") or []:
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        left_ref = _column_ref(dev_meta, pair[0])
        right_ref = _column_ref(dev_meta, pair[1])
        if left_ref is None or right_ref is None:
            continue

        left_table, left_column = left_ref
        right_table, right_column = right_ref
        if left_table not in wanted_tables or right_table not in wanted_tables:
            continue

        left_column = _resolve_column(column_lookup, left_table, left_column)
        right_column = _resolve_column(column_lookup, right_table, right_column)
        key = (left_table, left_column, right_table, right_column)
        if key in seen:
            continue
        seen.add(key)

        links.append({
            "name": f"{left_table}_to_{right_table}_{left_column}",
            "from_table": left_table,
            "to_table": right_table,
            "join_type": "left",
            "on": [{
                "left": f"{left_table}.{left_column}",
                "op": "=",
                "right": f"{right_table}.{right_column}",
            }],
        })

    return links


def _primary_id_links(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Infer obvious FK links by matching columns to another table's primary_id."""
    links = []
    seen = set()

    primary_ids = {
        str(table.get("name")): str(table.get("primary_id"))
        for table in tables
        if table.get("name") and table.get("primary_id")
    }

    for from_table in tables:
        from_name = str(from_table.get("name"))
        columns = {str(c.get("name")) for c in from_table.get("columns", []) or []}
        for to_name, to_pk in primary_ids.items():
            if from_name == to_name or not to_pk or to_pk == "id" or to_pk not in columns:
                continue
            key = (from_name, to_pk, to_name, to_pk)
            if key in seen:
                continue
            seen.add(key)
            links.append({
                "name": f"{from_name}_to_{to_name}_{to_pk}",
                "from_table": from_name,
                "to_table": to_name,
                "join_type": "left",
                "on": [{
                    "left": f"{from_name}.{to_pk}",
                    "op": "=",
                    "right": f"{to_name}.{to_pk}",
                }],
            })

    return links


def _merge_links(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = []
    seen = set()
    for group in groups:
        for link in group:
            on = link.get("on") or []
            if not on:
                continue
            first = on[0]
            key = (link.get("from_table"), first.get("left"), link.get("to_table"), first.get("right"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(link)
    return merged


def build_schema_for_db(full_schema: dict[str, Any], dev_meta: dict[str, Any]) -> dict[str, Any]:
    wanted_tables = {
        _make_logical_name(str(name))
        for name in dev_meta.get("table_names_original") or []
    }

    tables = [
        table
        for table in full_schema.get("tables", [])
        if table.get("name") in wanted_tables
    ]
    table_names = {table.get("name") for table in tables}

    introspected_links = [
        link
        for link in full_schema.get("links", []) or []
        if link.get("from_table") in table_names and link.get("to_table") in table_names
    ]

    schema: dict[str, Any] = {
        "version": full_schema.get("version", 1),
        "dialect": full_schema.get("dialect", "postgres"),
        "context": f"BIRD Mini-Dev database: {dev_meta['db_id']}",
        "tables": tables,
    }

    links = _merge_links(
        introspected_links,
        _metadata_links(dev_meta, table_names, _column_lookup(tables)),
        _primary_id_links(tables),
    )
    if links:
        schema["links"] = links

    return schema


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate IntentQL schemas for BIRD Mini-Dev.")
    parser.add_argument("--bird-root", default=env_value("BIRD_MINIDEV_ROOT") or "test/benchmark/bird_minidev")
    parser.add_argument("--dev-tables", default=None)
    parser.add_argument("--schema-dir", default="test/benchmark/schemas")
    parser.add_argument("--db-url", default=env_value("BIRD_DB_URL"))
    parser.add_argument("--postgres-schema", default="public")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.db_url:
        raise ValueError("Set BIRD_DB_URL in .env or pass --db-url.")

    dev_tables_file = find_dev_tables_file(Path(args.bird_root), args.dev_tables)
    dev_tables = load_dev_tables(dev_tables_file)
    full_schema = introspect_database(args.db_url, schema_name=args.postgres_schema)

    print(f"[bird] dev_tables={dev_tables_file}")
    print(f"[bird] full_schema_tables={len(full_schema.get('tables', []))}")

    schema_dir = Path(args.schema_dir)
    for dev_meta in dev_tables:
        db_id = str(dev_meta["db_id"])
        schema = build_schema_for_db(full_schema, dev_meta)
        output = schema_dir / db_id / "schema.yaml"
        _write_schema(schema, str(output))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
