"""
Rewrite table/column identifiers in generated SQL to exact spellings from schema.yaml.

Fixes LLM mistakes like ``\"finalworkorder\"`` vs ``\"finalWorkOrder\"`` (Postgres quoted
identifiers are case-sensitive). Data-driven — no per-error prompt edits.
"""

from __future__ import annotations

from .schema_catalog import SchemaCatalog, _norm_ident
from .sql_guard import (
    _build_alias_map,
    _column_key,
    _cte_names,
    _require_sqlglot,
)


def _postgres_needs_quotes(name: str) -> bool:
    """True if the identifier must be double-quoted in PostgreSQL."""
    if not name:
        return True
    if name != name.lower():
        return True
    if not name.replace("_", "").isalnum():
        return True
    if name[0] != "_" and not name[0].isalpha():
        return True
    return False


def canonicalize_sql(sql: str, catalog: SchemaCatalog) -> str:
    """
    Best-effort AST rewrite: resolve known tables/columns to ``exact_table`` /
    ``exact_column`` from :class:`SchemaCatalog`. On parse failure, returns ``sql`` unchanged.
    """
    _require_sqlglot()
    import sqlglot
    from sqlglot import exp

    text = (sql or "").strip()
    if not text:
        return sql

    try:
        parsed = sqlglot.parse_one(text, dialect="postgres")
    except Exception:
        return sql

    cte_names = _cte_names(parsed)

    for table in list(parsed.find_all(exp.Table)):
        n = _norm_ident(str(table.name))
        if n in cte_names:
            continue
        phys = catalog.logical_to_physical.get(n)
        if phys is None and n in catalog.physical_tables:
            phys = n
        if phys is None:
            continue
        exact = catalog.exact_table.get(phys)
        if not exact:
            continue
        table.set(
            "this",
            exp.Identifier(this=exact, quoted=_postgres_needs_quotes(exact)),
        )

    alias_to_physical = _build_alias_map(parsed, catalog, cte_names)

    for col in list(parsed.find_all(exp.Column)):
        if isinstance(col.this, exp.Star):
            continue
        key = _column_key(col)
        if not key:
            continue
        tbl = col.table
        if not tbl:
            continue
        tkey = _norm_ident(str(tbl))
        phys = alias_to_physical.get(tkey)
        if phys is None:
            continue
        cmap = catalog.exact_column.get(phys, {})
        exact = cmap.get(_norm_ident(key))
        if not exact:
            continue
        if isinstance(col.this, exp.Identifier):
            col.set(
                "this",
                exp.Identifier(this=exact, quoted=_postgres_needs_quotes(exact)),
            )

    try:
        return parsed.sql(dialect="postgres")
    except Exception:
        return sql
