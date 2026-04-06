"""
Validate LLM-generated Postgres SQL against :class:`schema_catalog.SchemaCatalog`.

Requires **sqlglot** (optional extra: ``pip install 'intentql[guided]'``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

from .schema_catalog import SchemaCatalog, _norm_ident


@dataclass
class SqlGuardResult:
    ok: bool
    message: str = ""
    sql: str = ""


def _require_sqlglot():
    try:
        import sqlglot  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "sql_guard requires sqlglot. Install with: pip install 'intentql[guided]' "
            "or pip install 'sqlglot>=23.0'"
        ) from e


def _forbidden_nodes(parsed) -> List[str]:
    from sqlglot import exp

    bad: List[str] = []
    for node in parsed.walk():
        if isinstance(
            node,
            (
                exp.Insert,
                exp.Update,
                exp.Delete,
                exp.Drop,
                exp.Create,
                exp.Alter,
                exp.TruncateTable,
                exp.Command,
                exp.Merge,
            ),
        ):
            bad.append(type(node).__name__)
    return bad


def _root_kind(parsed) -> str:
    from sqlglot import exp

    if isinstance(parsed, exp.With):
        inner = parsed.this
        if isinstance(inner, exp.Select):
            return "select"
        if isinstance(inner, exp.Union):
            return "union"
    if isinstance(parsed, exp.Select):
        return "select"
    if isinstance(parsed, exp.Union):
        return "union"
    return type(parsed).__name__


def _cte_names(parsed) -> Set[str]:
    from sqlglot import exp

    names: Set[str] = set()
    for cte in parsed.find_all(exp.CTE):
        al = cte.alias
        if al:
            names.add(_norm_ident(str(al)))
    return names


def _resolve_physical_table(
    table_name: str,
    catalog: SchemaCatalog,
    cte_names: Set[str],
):
    n = _norm_ident(table_name)
    if n in cte_names:
        return None
    if n in catalog.physical_tables:
        return n
    p = catalog.logical_to_physical.get(n)
    if p:
        return p
    return None


def _build_alias_map(parsed, catalog: SchemaCatalog, cte_names: Set[str]) -> dict[str, str]:
    from sqlglot import exp

    mapping: dict[str, str] = {}
    for table in parsed.find_all(exp.Table):
        raw_name = table.name
        n = _norm_ident(str(raw_name))
        phys = _resolve_physical_table(str(raw_name), catalog, cte_names)
        if phys is None:
            continue
        mapping[n] = phys
        al = table.alias
        if al:
            mapping[_norm_ident(str(al))] = phys
    return mapping


def _column_key(col) -> str:
    try:
        parts = col.parts
        if parts:
            last = parts[-1]
            if hasattr(last, "name") and last.name:
                return _norm_ident(str(last.name))
    except Exception:
        pass
    if col.name:
        return _norm_ident(str(col.name))
    return ""


def validate_sql(sql: str, catalog: SchemaCatalog) -> SqlGuardResult:
    _require_sqlglot()
    import sqlglot
    from sqlglot import exp

    text = (sql or "").strip()
    if not text:
        return SqlGuardResult(False, "Empty SQL")

    try:
        parsed = sqlglot.parse_one(text, dialect="postgres")
    except Exception as e:
        return SqlGuardResult(False, f"SQL parse error: {e}")

    stmts = sqlglot.parse(text, dialect="postgres")
    if len(stmts) != 1:
        return SqlGuardResult(False, "Only a single SELECT statement is allowed")

    bad = _forbidden_nodes(parsed)
    if bad:
        return SqlGuardResult(False, f"Forbidden statement type(s): {', '.join(sorted(set(bad)))}")

    rk = _root_kind(parsed)
    if rk not in ("select", "union"):
        return SqlGuardResult(False, f"Only read-only SELECT/WITH queries are allowed (got {rk})")

    cte_names = _cte_names(parsed)

    for table in parsed.find_all(exp.Table):
        n = _norm_ident(table.name)
        if n in cte_names:
            continue
        if n in catalog.allowed_table_tokens:
            continue
        return SqlGuardResult(False, f"Table not in schema allowlist: {table.name!r}")

    alias_to_physical = _build_alias_map(parsed, catalog, cte_names)
    all_cols = catalog.all_known_columns_lower()

    for col in parsed.find_all(exp.Column):
        if isinstance(col.this, exp.Star):
            continue
        key = _column_key(col)
        if not key or key == "*":
            continue

        tbl = col.table
        if not tbl:
            if key in all_cols:
                continue
            # ORDER BY / GROUP BY may reference SELECT list aliases (not physical columns).
            try:
                if col.find_ancestor(exp.Order) or col.find_ancestor(exp.Group):
                    continue
            except AttributeError:
                pass
            return SqlGuardResult(
                False,
                f"Unknown column {key!r} (not in schema column list)",
            )

        tkey = _norm_ident(str(tbl))
        phys = alias_to_physical.get(tkey)
        if phys is None:
            continue
        allowed = catalog.columns_by_physical.get(phys, set())
        if key not in allowed:
            return SqlGuardResult(
                False,
                f"Column {key!r} is not valid for table scope `{phys}`",
            )

    return SqlGuardResult(True, "", sql=text)


def apply_row_limit(sql: str, max_rows: int) -> str:
    _require_sqlglot()
    import sqlglot
    from sqlglot import exp

    s = sql.strip().rstrip(";")
    if max_rows <= 0:
        return s
    try:
        parsed = sqlglot.parse_one(s, dialect="postgres")
    except Exception:
        return f"{s} LIMIT {int(max_rows)}"
    if parsed.find(exp.Limit):
        return s
    return f"{s} LIMIT {int(max_rows)}"
