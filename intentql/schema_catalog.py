"""
Load schema.yaml into structures for guided-SQL prompts and sql_guard allowlists.

This is separate from :mod:`schema_validator` (load-time YAML checks): ``SchemaCatalog``
is built at runtime for LLM context and identifier allowlists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml


def _norm_ident(raw: str) -> str:
    s = (raw or "").strip().strip('"').strip("'")
    return s.lower()


def _exact_from_yaml(raw: Any) -> str:
    """Strip YAML/db_table/db_column quotes; keep inner spelling for Postgres."""
    if raw is None:
        return ""
    s = str(raw).strip()
    if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
        return s[1:-1]
    return s


@dataclass
class SchemaCatalog:
    """Allowlists and prompt text derived from schema.yaml."""

    dialect: str
    context: str
    physical_tables: Set[str] = field(default_factory=set)
    allowed_table_tokens: Set[str] = field(default_factory=set)
    columns_by_physical: Dict[str, Set[str]] = field(default_factory=dict)
    logical_to_physical: Dict[str, str] = field(default_factory=dict)
    # phys_lower -> exact table name spelling from schema (for identifier canonicalization)
    exact_table: Dict[str, str] = field(default_factory=dict)
    # phys_lower -> { col_norm -> exact column spelling from db_column / name }
    exact_column: Dict[str, Dict[str, str]] = field(default_factory=dict)
    schema_prompt_block: str = ""
    links_prompt_block: str = ""

    def all_known_columns_lower(self) -> Set[str]:
        out: Set[str] = set()
        for cols in self.columns_by_physical.values():
            out |= cols
        return out


def _parse_db_table(raw: Any) -> str:
    if raw is None:
        return ""
    return _norm_ident(str(raw).strip())


def _parse_db_column(raw: Any) -> str:
    if raw is None:
        return ""
    return _norm_ident(str(raw).strip())


def load_schema_catalog(schema_path: str | Path) -> SchemaCatalog:
    path = Path(schema_path)
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}

    dialect = str(doc.get("dialect") or "postgres")
    context = str(doc.get("context") or "").strip()

    physical_tables: Set[str] = set()
    allowed_table_tokens: Set[str] = set()
    columns_by_physical: Dict[str, Set[str]] = {}
    logical_to_physical: Dict[str, str] = {}
    exact_table: Dict[str, str] = {}
    exact_column: Dict[str, Dict[str, str]] = {}

    table_lines: List[str] = []

    for table in doc.get("tables", []) or []:
        logical_name = str(table.get("name") or "").strip()
        phys = _parse_db_table(table.get("db_table"))
        if not phys:
            continue

        physical_tables.add(phys)
        allowed_table_tokens.add(phys)
        if logical_name:
            ln = _norm_ident(logical_name)
            allowed_table_tokens.add(ln)
            logical_to_physical[ln] = phys

        et = _exact_from_yaml(table.get("db_table"))
        if et:
            exact_table[phys] = et

        tbl_desc = (table.get("description") or "").strip()
        extra: List[str] = []
        for key in ("primary_id", "primary_date", "keyword_search_or"):
            if table.get(key) is not None:
                extra.append(f"{key}: {table.get(key)}")
        header = f"### Table `{logical_name}` → physical `{phys}`"
        if tbl_desc:
            header += f"\n{tbl_desc}"
        if extra:
            header += "\n" + "\n".join(extra)

        col_lines: List[str] = []
        cols: Set[str] = set()
        col_exact: Dict[str, str] = exact_column.setdefault(phys, {})
        for col in table.get("columns", []) or []:
            cname = str(col.get("name") or "").strip()
            db_col = col.get("db_column")
            dbn = _parse_db_column(db_col) if db_col is not None else _norm_ident(cname)
            if cname:
                cols.add(_norm_ident(cname))
            if dbn:
                cols.add(dbn)
            exact_c = _exact_from_yaml(db_col) if db_col is not None else ""
            canonical = exact_c or cname
            if canonical:
                for nk in {_norm_ident(canonical), _norm_ident(cname)}:
                    if nk:
                        col_exact[nk] = canonical
            desc = (col.get("description") or "").strip()
            db_disp = db_col if db_col is not None else cname
            line = f"- **{cname}** (DB: `{db_disp}`)"
            if desc:
                line += f": {desc}"
            col_lines.append(line)

        columns_by_physical[phys] = cols
        table_lines.append(header + "\n" + "\n".join(col_lines))

    links_lines: List[str] = []
    for link in doc.get("links", []) or []:
        name = link.get("name", "")
        from_t = link.get("from_table", "")
        to_t = link.get("to_table", "")
        jt = link.get("join_type", "")
        on = link.get("on", [])
        links_lines.append(
            f"- **{name}**: `{from_t}` {jt or 'join'} `{to_t}` on {on!r}"
        )

    links_block = ""
    if links_lines:
        links_block = "### Suggested joins (soft; DB may not enforce FKs)\n" + "\n".join(links_lines)

    schema_block_parts = [
        f"Dialect: **{dialect}**.",
        "",
        "### Dataset context",
        context,
        "",
        "### Tables and columns (use only these identifiers; quote mixed-case Postgres names as needed)",
        "\n\n".join(table_lines),
    ]
    if links_block:
        schema_block_parts.extend(["", links_block])

    return SchemaCatalog(
        dialect=dialect,
        context=context,
        physical_tables=physical_tables,
        allowed_table_tokens=allowed_table_tokens,
        columns_by_physical=columns_by_physical,
        logical_to_physical=logical_to_physical,
        exact_table=exact_table,
        exact_column=exact_column,
        schema_prompt_block="\n".join(schema_block_parts),
        links_prompt_block=links_block,
    )
