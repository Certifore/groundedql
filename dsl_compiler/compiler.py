# dsl_compiler/compiler.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Tuple
import re

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from .exceptions import AmbiguousColumnError, QueryPlanError, SchemaError


def _require(cond: bool, code: str, message: str, path: str = "$", suggestion: Optional[str] = None):
    if not cond:
        raise QueryPlanError(message, code=code, path=path, suggestion=suggestion)


# -----------------------------
# Schema model
# -----------------------------
@dataclass(frozen=True)
class ColumnDef:
    logical: str
    db_column: str
    col_type: str
    description: str = ""


@dataclass(frozen=True)
class TableDef:
    logical: str
    db_table: str
    description: str
    columns: Dict[str, ColumnDef]  # logical -> ColumnDef


@dataclass(frozen=True)
class LinkDef:
    name: str
    from_table: str
    to_table: str
    join_type: str  # left|inner
    on: List[Dict[str, str]]  # list of {left, op, right}
    optional: bool = True


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _strip_outer_quotes(s: str) -> Tuple[str, bool]:
    s = (s or "").strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1], True
    return s, False


def _needs_quoting(ident: str) -> bool:
    return not re.fullmatch(r"[a-z_][a-z0-9_]*", ident or "")


def _split_table_column(ref: str, default_table: str, known_tables: FrozenSet[str]) -> Tuple[str, str]:
    """
    Split col ref into (logical_table, column).
    If ref contains multiple dots, prefer the longest prefix that matches a known logical table name
    so refs like 'a.b.col' work when 'a.b' is a declared table name.
    """
    if "." not in ref:
        return default_table, ref
    parts = ref.split(".")
    for n in range(len(parts) - 1, 0, -1):
        tkey = ".".join(parts[:n])
        if tkey in known_tables:
            return tkey, ".".join(parts[n:])
    return ref.split(".", 1)


def _map_sa_type(t: str) -> sa.types.TypeEngine:
    tt = (t or "").lower()
    if "smallint" in tt:
        return sa.SmallInteger()
    if "bigint" in tt:
        return sa.BigInteger()
    if "int" in tt:
        return sa.Integer()
    if "bool" in tt:
        return sa.Boolean()
    if "timestamp" in tt or "datetime" in tt:
        return sa.DateTime()
    if tt == "date":
        return sa.Date()
    if tt == "time" or (tt.startswith("time") and "stamp" not in tt):
        return sa.Time()
    if "interval" in tt:
        return sa.Interval()
    if "numeric" in tt or "decimal" in tt:
        return sa.Numeric()
    if "float" in tt or "double" in tt or "real" in tt:
        return sa.Float()
    if "json" in tt:
        return postgresql.JSONB()
    if "uuid" in tt:
        return postgresql.UUID(as_uuid=True)
    if tt == "text" or tt.endswith("text"):
        return sa.Text()
    return sa.String()


def _exported_column_count(stmt: sa.SelectBase) -> int:
    return len(stmt.exported_columns)


# -----------------------------
# Compiler
# -----------------------------
class Compiler:
    """
    Deterministic Postgres compiler:
      - YAML allowlist mapping logical->physical identifiers
      - parameterized values (bindparams)
      - link-based joins (optional)
      - legacy plans (dimensions/metrics/filters) lowered to select/where/group_by
      - rollup: outer aggregation over grouped inner query

    Execution note: ``scalar_subquery`` is not checked for single-row cardinality here;
    Postgres raises at runtime if the subquery returns more than one row.
    """

    def __init__(
        self,
        schema: dict,
        *,
        default_limit: int = 100,
        max_limit: int = 1000,
        max_joins: int = 8,
        max_select: int = 200,
        max_predicates: int = 200,
        dialect: Optional[sa.engine.Dialect] = None,
    ):
        self.default_limit = default_limit
        self.max_limit = max_limit
        self.max_joins = max_joins
        self.max_select = max_select
        self.max_predicates = max_predicates
        self.dialect = dialect or postgresql.dialect()

        self.tables: Dict[str, TableDef] = {}
        for t in schema.get("tables", []):
            cols: Dict[str, ColumnDef] = {}
            for c in t.get("columns", []):
                cols[c["name"]] = ColumnDef(
                    logical=c["name"],
                    db_column=c.get("db_column", c["name"]),
                    col_type=c.get("type", "varchar"),
                    description=c.get("description", "") or "",
                )
            self.tables[t["name"]] = TableDef(
                logical=t["name"],
                db_table=t["db_table"],
                description=t.get("description", "") or "",
                columns=cols,
            )

        self._known_logical_tables: FrozenSet[str] = frozenset(self.tables.keys())

        self.links: Dict[str, LinkDef] = {}
        for link_item in schema.get("links", []) or []:
            jt = (link_item.get("join_type") or "left").lower()
            if jt not in {"left", "inner"}:
                raise SchemaError(
                    f"link '{link_item.get('name')}': join_type must be 'left' or 'inner', "
                    f"got {link_item.get('join_type')!r}."
                )
            ld = LinkDef(
                name=link_item["name"],
                from_table=link_item["from_table"],
                to_table=link_item["to_table"],
                join_type=jt,
                on=link_item.get("on", []),
                optional=bool(link_item.get("optional", True)),
            )
            self.links[ld.name] = ld

        md = sa.MetaData()
        self._sa_tables: Dict[str, sa.Table] = {}

        for logical_name, tdef in self.tables.items():
            raw = tdef.db_table
            table_name, table_quoted = _strip_outer_quotes(raw)

            schema_name = None
            schema_quoted = False
            if "." in table_name:
                schema_name, table_name = table_name.split(".", 1)
                schema_name, schema_quoted = _strip_outer_quotes(schema_name)

            sa_cols = []
            for col in tdef.columns.values():
                col_name, col_quoted = _strip_outer_quotes(col.db_column)
                sa_cols.append(
                    sa.Column(
                        col_name,
                        _map_sa_type(col.col_type),
                        quote=col_quoted or _needs_quoting(col_name),
                    )
                )

            self._sa_tables[logical_name] = sa.Table(
                table_name,
                md,
                *sa_cols,
                schema=schema_name,
                quote=table_quoted or _needs_quoting(table_name),
                quote_schema=schema_quoted if schema_name else None,
            )

        self._param_counter = 0
        self._alias_counter = 0

    def _new_alias(self, logical_name: str) -> str:
        """Generate a unique table alias to prevent self-join collisions."""
        self._alias_counter += 1
        return f"{logical_name}_{self._alias_counter}"

    # ---------- Public API ----------
    def compile(self, plan: dict) -> Tuple[str, Dict[str, Any]]:
        _require(isinstance(plan, dict), "INVALID_PLAN", "QueryPlan must be an object.", "$")
        # Strip internal metadata before compilation
        plan = {k: v for k, v in plan.items() if k != "meta"}
        self._param_counter = 0  # reset per compile for cleaner param names
        self._alias_counter = 0  # reset per compile

        # Build a SQLAlchemy selectable (Select or CompoundSelect)
        selectable = self._build_selectable(plan, cte_map={}, path="$", outer_alias_map=None)

        compiled = selectable.compile(dialect=self.dialect, compile_kwargs={"render_postcompile": True})
        sql = str(compiled)
        params = dict(compiled.params)

        head_ok = re.match(r"^\s*\(*\s*(WITH|SELECT)\b", sql, flags=re.IGNORECASE) is not None
        _require(head_ok, "INTERNAL_ERROR", "Compiler produced non-SELECT SQL.", "$")

        return sql, params

    # ---------- Correlated subqueries (outer scope) ----------
    def _scope_for_nested_subquery(
        self,
        outer_alias_map: Optional[Dict[str, sa.FromClause]],
        inner_alias_map: Dict[str, sa.FromClause],
    ) -> Dict[str, sa.FromClause]:
        """
        Combine enclosing scopes for a nested EXISTS / scalar_subquery.
        Inner FROM-clause bindings win over outer when the same logical table name appears twice.
        """
        merged: Dict[str, sa.FromClause] = dict(outer_alias_map or {})
        merged.update(inner_alias_map)
        return merged

    # ---------- Core builder (handles legacy, CTE, set ops, rollup) ----------
    def _build_selectable(
        self,
        plan: dict,
        *,
        cte_map: Dict[str, sa.CTE],
        path: str,
        outer_alias_map: Optional[Dict[str, sa.FromClause]] = None,
    ) -> sa.SelectBase:
        _require(isinstance(plan, dict), "INVALID_PLAN", "Plan must be an object.", path)

        # Legacy lowering if needed
        if "dataset" in plan and any(k in plan for k in ("dimensions", "metrics", "filters")) and "select" not in plan and "set_op" not in plan:
            plan = self._legacy_to_select_plan(plan)

        # CTEs
        with_list = plan.get("with", []) or []
        _require(isinstance(with_list, list), "INVALID_PLAN", "with must be a list.", f"{path}.with")

        local_ctes = dict(cte_map)
        for i, cte_def in enumerate(with_list):
            p = f"{path}.with[{i}]"
            _require(isinstance(cte_def, dict), "INVALID_PLAN", "CTE must be an object.", p)
            name = cte_def.get("name")
            cte_plan = cte_def.get("plan")

            _require(isinstance(name, str) and _IDENT_RE.match(name), "INVALID_PLAN", "CTE name must be an identifier.", f"{p}.name")
            _require(name not in self.tables, "INVALID_PLAN", f"CTE name '{name}' conflicts with a schema table name.", f"{p}.name")
            _require(cte_plan is not None, "INVALID_PLAN", "CTE requires plan.", f"{p}.plan")

            selectable = self._build_selectable(
                cte_plan, cte_map=local_ctes, path=f"{p}.plan", outer_alias_map=outer_alias_map
            )
            cte_obj = selectable.cte(name)
            local_ctes[name] = cte_obj

        # Set operations
        if "set_op" in plan:
            return self._build_set_op(plan, cte_map=local_ctes, path=path, outer_alias_map=outer_alias_map)

        # Normal SELECT (+ optional rollup)
        return self._build_select_query(plan, cte_map=local_ctes, path=path, outer_alias_map=outer_alias_map)

    # ---------- Set operations ----------
    def _build_set_op(
        self,
        plan: dict,
        *,
        cte_map: Dict[str, sa.CTE],
        path: str,
        outer_alias_map: Optional[Dict[str, sa.FromClause]] = None,
    ) -> sa.SelectBase:
        p = f"{path}.set_op"
        sop = plan.get("set_op")
        _require(isinstance(sop, dict), "INVALID_PLAN", "set_op must be an object.", p)

        op = (sop.get("op") or "").lower()
        left_plan = sop.get("left")
        right_plan = sop.get("right")

        _require(op in {"union", "union_all", "intersect", "except"}, "INVALID_PLAN",
                 "set_op.op must be one of: union, union_all, intersect, except.", f"{p}.op")
        _require(isinstance(left_plan, dict), "INVALID_PLAN", "set_op.left must be an object.", f"{p}.left")
        _require(isinstance(right_plan, dict), "INVALID_PLAN", "set_op.right must be an object.", f"{p}.right")

        left = self._build_selectable(left_plan, cte_map=cte_map, path=f"{p}.left", outer_alias_map=outer_alias_map)
        right = self._build_selectable(right_plan, cte_map=cte_map, path=f"{p}.right", outer_alias_map=outer_alias_map)

        nl, nr = _exported_column_count(left), _exported_column_count(right)
        _require(
            nl == nr,
            "INVALID_PLAN",
            f"set_op branches must have the same number of select columns (left={nl}, right={nr}).",
            p,
        )

        if op == "union":
            comb = sa.union(left, right)
        elif op == "union_all":
            comb = sa.union_all(left, right)
        elif op == "intersect":
            comb = sa.intersect(left, right)
        else:
            comb = sa.except_(left, right)

        # Optional ORDER/LIMIT/OFFSET on the compound result
        order_by = plan.get("order_by", []) or []
        limit = plan.get("limit")
        offset = plan.get("offset", 0)

        if order_by:
            _require(isinstance(order_by, list), "INVALID_PLAN", "order_by must be a list.", f"{path}.order_by")
            ob_exprs = []
            for k, ob in enumerate(order_by):
                pp = f"{path}.order_by[{k}]"
                _require(isinstance(ob, dict), "INVALID_PLAN", "order_by items must be objects.", pp)
                by = ob.get("by")
                direction = (ob.get("dir") or "asc").lower()
                _require(isinstance(by, str) and _IDENT_RE.match(by), "INVALID_PLAN",
                         "For set_op, order_by.by must be a column/alias name (identifier).", f"{pp}.by")
                _require(direction in {"asc", "desc"}, "INVALID_PLAN", "order_by.dir must be asc|desc.", f"{pp}.dir")
                col = sa.column(by)
                ob_exprs.append(col.asc() if direction == "asc" else col.desc())
            comb = comb.order_by(*ob_exprs)

        if limit is not None:
            limit_i = self._clamp_int(limit, 1, self.max_limit, f"{path}.limit", default=self.default_limit)
            offset_i = self._clamp_int(offset, 0, 10_000_000, f"{path}.offset", default=0)
            comb = comb.limit(limit_i).offset(offset_i)

        return comb

    # ---------- Normal SELECT (+ rollup) ----------
    def _build_select_query(
        self,
        plan: dict,
        *,
        cte_map: Dict[str, sa.CTE],
        path: str,
        outer_alias_map: Optional[Dict[str, sa.FromClause]] = None,
    ) -> sa.SelectBase:
        dataset = plan.get("dataset")
        _require(isinstance(dataset, str), "INVALID_PLAN", "dataset is required.", f"{path}.dataset")

        rollup = plan.get("rollup")

        # If rollup is present, ignore inner order/limit/offset for correctness
        if rollup is None:
            limit = plan.get("limit")          # None = no LIMIT
            offset = plan.get("offset", 0)
            order_by = plan.get("order_by", []) or []
        else:
            limit = None                       # inner query is always unlimited when rollup is present
            offset = None
            order_by = []

        select_items = plan.get("select", [])
        joins = plan.get("joins", []) or []
        where = plan.get("where")
        group_by = plan.get("group_by", []) or []
        having = plan.get("having")
        distinct = bool(plan.get("distinct", False))

        _require(isinstance(select_items, list), "INVALID_PLAN", "select must be a list.", f"{path}.select")
        _require(len(select_items) >= 1, "INVALID_PLAN", "select must have at least 1 item.", f"{path}.select")
        _require(len(select_items) <= self.max_select, "QUERY_TOO_COMPLEX", "Too many select items.", f"{path}.select")
        _require(isinstance(joins, list), "INVALID_PLAN", "joins must be a list.", f"{path}.joins")
        _require(len(joins) <= self.max_joins, "QUERY_TOO_COMPLEX", "Too many joins.", f"{path}.joins")

        source = self._resolve_relation(dataset, cte_map, f"{path}.dataset")
        from_alias = self._new_alias(dataset)

        base = source.alias(from_alias)

        inner_alias_map: Dict[str, sa.FromClause] = {dataset: base}
        from_clause: sa.FromClause = base

        # Apply joins (inner FROM clause only — outer correlation is separate)
        for j_idx, j in enumerate(joins):
            p = f"{path}.joins[{j_idx}]"
            _require(isinstance(j, dict), "INVALID_PLAN", "Each join must be an object.", p)

            if "link" in j:
                link_name = j.get("link")
                _require(isinstance(link_name, str) and link_name in self.links,
                         "INVALID_PLAN", f"Unknown link '{link_name}'.", f"{p}.link")
                from_clause, inner_alias_map = self._apply_link_join(
                    from_clause,
                    inner_alias_map,
                    dataset,
                    link_name,
                    p,
                    cte_map=cte_map,
                    outer_alias_map=outer_alias_map,
                )
                continue

            j_ds = j.get("dataset")
            _require(isinstance(j_ds, str), "INVALID_PLAN", "join.dataset must be a string.", f"{p}.dataset")
            j_type = (j.get("type") or "left").lower()
            _require(j_type in {"left", "inner"}, "INVALID_PLAN", "join.type must be left|inner.", f"{p}.type")
            on_expr = j.get("on")
            _require(on_expr is not None, "INVALID_PLAN", "join.on is required for explicit joins.", f"{p}.on")

            j_as = j.get("as")
            if j_as is not None:
                _require(
                    isinstance(j_as, str) and _IDENT_RE.match(j_as),
                    "INVALID_PLAN",
                    "join.as must be a valid identifier when provided.",
                    f"{p}.as",
                )
                logical = j_as
            else:
                logical = j_ds

            _require(
                logical not in inner_alias_map,
                "INVALID_PLAN",
                f"Join alias '{logical}' is already in the FROM clause. "
                f"Use a different join.as when joining the same dataset again (e.g. self-join).",
                p,
                suggestion='Example: {"dataset": "orders", "as": "orders_2", "type": "inner", "on": ...}',
            )

            j_source = self._resolve_relation(j_ds, cte_map, f"{p}.dataset")
            inner_alias_map[logical] = j_source.alias(self._new_alias(j_ds))
            right = inner_alias_map[logical]
            cond = self._compile_bool_expr(
                on_expr, inner_alias_map, dataset, cte_map=cte_map, path=f"{p}.on", outer_alias_map=outer_alias_map
            )
            from_clause = from_clause.join(right, cond, isouter=(j_type == "left"))

        # SELECT
        compiled_select: List[sa.ColumnElement] = []
        alias_lookup: Dict[str, sa.ColumnElement] = {}

        for i, item in enumerate(select_items):
            p = f"{path}.select[{i}]"
            _require(isinstance(item, dict), "INVALID_PLAN", "select items must be objects.", p)
            expr = item.get("expr")
            alias = item.get("alias")
            _require(expr is not None, "INVALID_PLAN", "select item requires expr.", f"{p}.expr")

            col_expr = self._compile_expr(
                expr, inner_alias_map, dataset, cte_map=cte_map, path=f"{p}.expr", outer_alias_map=outer_alias_map
            )

            if alias is not None:
                _require(isinstance(alias, str) and _IDENT_RE.match(alias),
                         "INVALID_PLAN", "select.alias must be an identifier.", f"{p}.alias")
                col_expr = col_expr.label(alias)
                alias_lookup[alias] = col_expr

            compiled_select.append(col_expr)

        query = sa.select(*compiled_select).select_from(from_clause)

        if distinct:
            query = query.distinct()

        # WHERE
        if where is not None:
            where_expr = self._compile_bool_expr(
                where, inner_alias_map, dataset, cte_map=cte_map, path=f"{path}.where", outer_alias_map=outer_alias_map
            )
            query = query.where(where_expr)

        # GROUP BY
        if group_by:
            _require(isinstance(group_by, list), "INVALID_PLAN", "group_by must be a list.", f"{path}.group_by")
            gb_exprs = [
                self._compile_expr(
                    e, inner_alias_map, dataset, cte_map=cte_map, path=f"{path}.group_by[{k}]", outer_alias_map=outer_alias_map
                )
                for k, e in enumerate(group_by)
            ]
            query = query.group_by(*gb_exprs)

        # HAVING
        if having is not None:
            having_expr = self._compile_bool_expr(
                having, inner_alias_map, dataset, cte_map=cte_map, path=f"{path}.having", outer_alias_map=outer_alias_map
            )
            query = query.having(having_expr)

        # ORDER BY (only when no rollup)
        if rollup is None and order_by:
            _require(isinstance(order_by, list), "INVALID_PLAN", "order_by must be a list.", f"{path}.order_by")
            ob_exprs = []
            for k, ob in enumerate(order_by):
                pp = f"{path}.order_by[{k}]"
                _require(isinstance(ob, dict), "INVALID_PLAN", "order_by items must be objects.", pp)
                by = ob.get("by")
                direction = (ob.get("dir") or "asc").lower()
                _require(direction in {"asc", "desc"}, "INVALID_PLAN", "order_by.dir must be asc|desc.", f"{pp}.dir")

                if isinstance(by, str) and by in alias_lookup:
                    e = alias_lookup[by]
                else:
                    _require(by is not None, "INVALID_PLAN", "order_by.by required.", f"{pp}.by")
                    # Legacy order_by uses string column names; _compile_expr requires dict nodes.
                    if isinstance(by, str):
                        e = self._compile_expr(
                            {"col": by},
                            inner_alias_map,
                            dataset,
                            cte_map=cte_map,
                            path=f"{pp}.by",
                            outer_alias_map=outer_alias_map,
                        )
                    else:
                        e = self._compile_expr(
                            by, inner_alias_map, dataset, cte_map=cte_map, path=f"{pp}.by", outer_alias_map=outer_alias_map
                        )

                ob_exprs.append(e.asc() if direction == "asc" else e.desc())
            query = query.order_by(*ob_exprs)

        # LIMIT/OFFSET (only when no rollup)
        if rollup is None:
            if limit is not None:              # only emit LIMIT if explicitly set
                limit_i = self._clamp_int(limit, 1, self.max_limit, f"{path}.limit", default=self.default_limit)
                offset_i = self._clamp_int(offset, 0, 10_000_000, f"{path}.offset", default=0)
                query = query.limit(limit_i).offset(offset_i)
            return query

        # ---------- ROLLUP ----------
        _require(isinstance(rollup, dict), "INVALID_PLAN", "rollup must be an object.", f"{path}.rollup")
        roll_metrics = rollup.get("metrics", [])
        roll_dims = rollup.get("dimensions", []) or []
        roll_filters = rollup.get("filters", []) or []
        roll_order_by = rollup.get("order_by", []) or []
        roll_limit = rollup.get("limit")       # None = no LIMIT on rollup outer query
        roll_offset = rollup.get("offset", 0)

        _require(isinstance(roll_metrics, list) and len(roll_metrics) >= 1,
                 "INVALID_PLAN", "rollup.metrics must be a non-empty list.", f"{path}.rollup.metrics")
        _require(isinstance(roll_dims, list), "INVALID_PLAN", "rollup.dimensions must be a list.", f"{path}.rollup.dimensions")
        _require(isinstance(roll_filters, list), "INVALID_PLAN", "rollup.filters must be a list.", f"{path}.rollup.filters")
        _require(isinstance(roll_order_by, list), "INVALID_PLAN", "rollup.order_by must be a list.", f"{path}.rollup.order_by")

        inner_subq = query.subquery("inner_q")

        outer_select: List[sa.ColumnElement] = []
        outer_alias_lookup: Dict[str, sa.ColumnElement] = {}
        outer_group_by: List[sa.ColumnElement] = []

        # rollup dimensions (optional)
        for i, d in enumerate(roll_dims):
            pp = f"{path}.rollup.dimensions[{i}]"
            if isinstance(d, str):
                field = d
                alias = None
            else:
                _require(isinstance(d, dict), "INVALID_PLAN", "rollup dimension must be string or object.", pp)
                field = d.get("field")
                alias = d.get("alias")
            _require(isinstance(field, str) and field, "INVALID_PLAN", "rollup dimension.field required.", pp)
            _require(field in inner_subq.c, "INVALID_PLAN", f"rollup dimension '{field}' must reference an inner output alias/column.", pp)

            expr = inner_subq.c[field]
            if alias is not None:
                _require(isinstance(alias, str) and _IDENT_RE.match(alias),
                         "INVALID_PLAN", "rollup dimension.alias must be an identifier.", f"{pp}.alias")
                expr = expr.label(alias)
                outer_alias_lookup[alias] = expr
            else:
                outer_alias_lookup[field] = expr

            outer_select.append(expr)
            outer_group_by.append(inner_subq.c[field])

        # rollup metrics
        for i, m in enumerate(roll_metrics):
            pp = f"{path}.rollup.metrics[{i}]"
            _require(isinstance(m, dict), "INVALID_PLAN", "rollup metric must be an object.", pp)
            agg = (m.get("agg") or "").lower()
            field = m.get("field", "*")
            alias = m.get("alias")

            _require(isinstance(alias, str) and _IDENT_RE.match(alias),
                     "INVALID_PLAN", "rollup metric.alias must be an identifier.", f"{pp}.alias")
            _require(isinstance(agg, str) and agg, "INVALID_PLAN", "rollup metric.agg required.", f"{pp}.agg")

            if agg == "count" and field == "*":
                expr = sa.func.count()
            else:
                _require(isinstance(field, str) and field, "INVALID_PLAN", "rollup metric.field must be a string.", f"{pp}.field")
                _require(field in inner_subq.c, "INVALID_PLAN", f"rollup metric.field '{field}' must reference an inner output alias/column.", f"{pp}.field")
                col = inner_subq.c[field]
                if agg in {"count_distinct", "countdistinct"}:
                    expr = sa.func.count(sa.distinct(col))
                else:
                    expr = getattr(sa.func, agg)(col)

            expr = expr.label(alias)
            outer_select.append(expr)
            outer_alias_lookup[alias] = expr

        outer_query = sa.select(*outer_select).select_from(inner_subq)

        # rollup filters (WHERE on inner outputs)
        if roll_filters:
            where_exprs = [self._compile_rollup_filter(inner_subq, f, path=f"{path}.rollup.filters[{i}]")
                           for i, f in enumerate(roll_filters)]
            outer_query = outer_query.where(sa.and_(*where_exprs))

        if outer_group_by:
            outer_query = outer_query.group_by(*outer_group_by)

        # rollup order_by
        if roll_order_by:
            ob_exprs = []
            for k, ob in enumerate(roll_order_by):
                pp = f"{path}.rollup.order_by[{k}]"
                _require(isinstance(ob, dict), "INVALID_PLAN", "rollup.order_by items must be objects.", pp)
                by = ob.get("by")
                direction = (ob.get("dir") or "asc").lower()
                _require(direction in {"asc", "desc"}, "INVALID_PLAN", "rollup.order_by.dir must be asc|desc.", f"{pp}.dir")

                if isinstance(by, str) and by in outer_alias_lookup:
                    e = outer_alias_lookup[by]
                elif isinstance(by, str) and by in inner_subq.c:
                    e = inner_subq.c[by]
                else:
                    _require(False, "INVALID_PLAN", "rollup.order_by.by must be an alias or inner column name.", f"{pp}.by")

                ob_exprs.append(e.asc() if direction == "asc" else e.desc())
            outer_query = outer_query.order_by(*ob_exprs)

        if roll_limit is not None:             # only emit LIMIT on rollup if explicitly set
            limit_i = self._clamp_int(roll_limit, 1, self.max_limit, f"{path}.rollup.limit", default=self.default_limit)
            offset_i = self._clamp_int(roll_offset, 0, 10_000_000, f"{path}.rollup.offset", default=0)
            outer_query = outer_query.limit(limit_i).offset(offset_i)

        return outer_query

    # ---------- Legacy -> select-plan ----------
    def _legacy_to_select_plan(self, plan: dict) -> dict:
        dataset = plan["dataset"]
        dims = plan.get("dimensions", []) or []
        mets = plan.get("metrics", []) or []
        filts = plan.get("filters", []) or []
        joins = plan.get("joins", []) or []
        order_by = plan.get("order_by", []) or []
        having = plan.get("having")
        rollup = plan.get("rollup")
        distinct = bool(plan.get("distinct", False))

        limit = plan.get("limit", self.default_limit)
        offset = plan.get("offset", 0)

        select_items = []
        group_by = []

        for d in dims:
            field = d.get("field")
            alias = d.get("alias")
            _require(isinstance(field, str), "INVALID_PLAN", "dimension.field must be string.", "$.dimensions")
            select_items.append({"expr": {"col": field}, "alias": alias})
            group_by.append({"col": field})

        for m in mets:
            agg = (m.get("agg") or "").lower()
            field = m.get("field", "*")
            alias = m.get("alias")
            _require(isinstance(alias, str), "INVALID_PLAN", "metric.alias required.", "$.metrics")

            if agg == "count" and field == "*":
                expr = {"func": "count", "args": []}
            elif agg in {"count_distinct", "countdistinct"}:
                _require(isinstance(field, str) and field != "*", "INVALID_PLAN", "count_distinct requires a field.", "$.metrics")
                expr = {"func": "count_distinct", "args": [{"col": field}]}
            else:
                expr = {"func": agg, "args": [{"col": field}]}

            select_items.append({"expr": expr, "alias": alias})

        metric_aliases = {m.get("alias") for m in mets if isinstance(m.get("alias"), str)}

        # With GROUP BY, PostgreSQL requires ORDER BY expressions to appear in GROUP BY or be
        # aggregates. LLMs often set order_by to a time column (e.g. entry_date) without listing
        # it as a dimension — add those base columns to group_by so SQL is valid.
        if group_by and order_by:
            dim_alias_to_field = {
                d.get("alias"): d.get("field")
                for d in dims
                if isinstance(d.get("alias"), str) and isinstance(d.get("field"), str)
            }
            grouped: set[str] = set()
            for d in dims:
                f = d.get("field")
                if isinstance(f, str):
                    grouped.add(f)
                    grouped.add(f.split(".", 1)[-1])

            def _order_by_resolved_to_grouped(by_key: str) -> bool:
                if by_key in metric_aliases:
                    return True
                if by_key in dim_alias_to_field:
                    f = dim_alias_to_field[by_key]
                    return f in grouped or f.split(".", 1)[-1] in grouped
                if by_key in grouped:
                    return True
                if "." in by_key and by_key.split(".", 1)[-1] in grouped:
                    return True
                return False

            tbl = self.tables.get(dataset)
            if tbl is not None:
                logical_names = {c.logical for c in tbl.columns.values()}
            else:
                # Outer query over a CTE: `dataset` is not a schema table name, so the lookup
                # above is empty and ORDER BY columns (e.g. entry_date) were never appended to
                # GROUP BY, causing PostgreSQL GroupingError. Accept any logical column name
                # declared on schema tables (CTEs reuse the same logical names).
                logical_names = set()
                for t in self.tables.values():
                    logical_names.update(c.logical for c in t.columns.values())

            for ob in order_by:
                if not isinstance(ob, dict):
                    continue
                by = ob.get("by")
                if not isinstance(by, str):
                    continue
                if _order_by_resolved_to_grouped(by):
                    continue
                logical = by.split(".", 1)[-1] if "." in by else by
                if logical not in logical_names:
                    continue
                if logical in grouped:
                    continue
                group_by.append({"col": logical})
                grouped.add(logical)

        # Scalar aggregate (no dimensions → no GROUP BY): ORDER BY a raw column is invalid
        # (PostgreSQL GroupingError). LLMs often copy order_by from a detail query. Keep only
        # clauses that sort by a metric alias (present in the SELECT list).
        if not group_by and mets and order_by:
            order_by = [
                ob
                for ob in order_by
                if isinstance(ob, dict)
                and isinstance(ob.get("by"), str)
                and ob.get("by") in metric_aliases
            ]

        # If no dimensions and no metrics, select all columns from the schema table
        if not select_items:
            if dataset in self.tables:
                for col in self.tables[dataset].columns.values():
                    select_items.append({"expr": {"col": col.logical}, "alias": col.logical})
            else:
                # CTE or dynamic source — emit a literal star isn't safe; require at least 1 item
                _require(False, "INVALID_PLAN",
                         "Legacy plan with no dimensions and no metrics requires a known schema table.",
                         "$.select")

        where = None
        if filts:
            clauses = []
            for f in filts:
                field = f.get("field")
                op = f.get("op")
                value = f.get("value")
                _require(isinstance(field, str) and isinstance(op, str), "INVALID_PLAN", "Invalid filter.", "$.filters")
                clauses.append({"cmp": {"left": {"col": field}, "op": op, "right": value}})
            where = {"and": clauses}

        adv_where = plan.get("where")
        if adv_where is not None:
            if where is None:
                where = adv_where
            else:
                where = {"and": [where, adv_where]}

        out = {
            "dataset": dataset,
            "joins": joins,
            "select": select_items,
            "where": where,
            "group_by": group_by if group_by else [],
            "having": having,
            "order_by": order_by,
            "limit": limit,
            "offset": offset,
            "distinct": distinct,
        }
        if rollup is not None:
            out["rollup"] = rollup
        if "with" in plan:
            out["with"] = plan.get("with")
        return out

    # ---------- Relation resolver (schema tables or CTEs) ----------
    def _resolve_relation(self, name: str, cte_map: Dict[str, sa.CTE], path: str) -> sa.FromClause:
        if name in cte_map:
            return cte_map[name]
        _require(name in self._sa_tables, "UNKNOWN_DATASET", f"Unknown dataset/cte '{name}'.", path)
        return self._sa_tables[name]

    # ---------- Joins by link ----------
    def _apply_link_join(
        self,
        from_clause: sa.FromClause,
        alias_map: Dict[str, sa.FromClause],
        base_dataset: str,
        link_name: str,
        path: str,
        *,
        cte_map: Dict[str, sa.CTE],
        outer_alias_map: Optional[Dict[str, sa.FromClause]] = None,
    ) -> Tuple[sa.FromClause, Dict[str, sa.FromClause]]:
        link = self.links[link_name]

        _require(link.from_table in self.tables and link.to_table in self.tables,
                 "INVALID_PLAN", "Link references unknown tables.", path)

        if link.from_table not in alias_map:
            alias_map[link.from_table] = self._sa_tables[link.from_table].alias(self._new_alias(link.from_table))

        to_alias_name = self._new_alias(link.to_table)
        to_alias = self._sa_tables[link.to_table].alias(to_alias_name)

        # Register under the logical table name so qualified refs like
        # "assets.keyword_of_asset" resolve correctly.
        alias_map[link.to_table] = to_alias

        right_tbl = to_alias

        conds = []
        for i, on in enumerate(link.on):
            lp = f"{path}.on[{i}]"
            left_ref = on["left"]
            right_ref = on["right"]
            op = on.get("op", "=")
            _require(op == "=", "INVALID_PLAN", "Only '=' supported in link joins.", lp)

            l_expr = self._compile_expr(
                {"col": left_ref}, alias_map, base_dataset, cte_map=cte_map, path=f"{lp}.left", outer_alias_map=outer_alias_map
            )
            r_expr = self._compile_expr(
                {"col": right_ref}, alias_map, base_dataset, cte_map=cte_map, path=f"{lp}.right", outer_alias_map=outer_alias_map
            )
            conds.append(l_expr == r_expr)

        on_expr = sa.and_(*conds) if conds else sa.true()
        is_outer = (link.join_type == "left")
        from_clause = from_clause.join(right_tbl, on_expr, isouter=is_outer)
        return from_clause, alias_map

    # ---------- Expression compilation ----------
    def _new_param(self, prefix: str = "p") -> str:
        self._param_counter += 1
        return f"{prefix}_{self._param_counter}"

    def _clamp_int(self, value: Any, lo: int, hi: int, path: str, *, default: int) -> int:
        try:
            iv = int(value)
        except Exception:
            iv = default
        _require(lo <= iv <= hi, "INVALID_PLAN", f"Value must be between {lo} and {hi}.", path)
        return iv

    def _col(
        self,
        inner_alias_map: Dict[str, sa.FromClause],
        default_table: str,
        ref: str,
        path: str,
        *,
        outer_alias_map: Optional[Dict[str, sa.FromClause]] = None,
    ) -> sa.ColumnElement:
        _require(isinstance(ref, str), "INVALID_PLAN", "Column ref must be string.", path)
        t, c = _split_table_column(ref, default_table, self._known_logical_tables)

        def _resolve_column(alias_map: Dict[str, sa.FromClause], table_key: str) -> sa.ColumnElement:
            tbl = alias_map[table_key]
            if table_key in self.tables:
                tdef = self.tables[table_key]
                _require(c in tdef.columns, "UNKNOWN_COLUMN", f"Unknown column '{c}' in table '{table_key}'.", path)
                col_def = tdef.columns[c]
                db_col_name, _ = _strip_outer_quotes(col_def.db_column)
                return tbl.c[db_col_name]
            _require(c in tbl.c, "UNKNOWN_COLUMN", f"Unknown column '{c}' on dynamic source '{table_key}'.", path)
            return tbl.c[c]

        if "." not in ref:
            inner_matches = [tname for tname, tdef in self.tables.items() if tname in inner_alias_map and c in tdef.columns]
            if len(inner_matches) > 1:
                raise AmbiguousColumnError(c, inner_matches, path)
            if len(inner_matches) == 1:
                return _resolve_column(inner_alias_map, inner_matches[0])
            if outer_alias_map:
                outer_matches = [tname for tname, tdef in self.tables.items() if tname in outer_alias_map and c in tdef.columns]
                if len(outer_matches) > 1:
                    raise AmbiguousColumnError(c, outer_matches, path)
                if len(outer_matches) == 1:
                    return _resolve_column(outer_alias_map, outer_matches[0])
            # CTE / dynamic FROM (not in schema tables): unqualified refs must resolve to tbl.c
            dyn_matches = [
                tname
                for tname, tbl in inner_alias_map.items()
                if tname not in self.tables and c in tbl.c
            ]
            if len(dyn_matches) > 1:
                raise AmbiguousColumnError(c, dyn_matches, path)
            if len(dyn_matches) == 1:
                return _resolve_column(inner_alias_map, dyn_matches[0])
            if outer_alias_map:
                dyn_outer = [
                    tname
                    for tname, tbl in outer_alias_map.items()
                    if tname not in self.tables and c in tbl.c
                ]
                if len(dyn_outer) > 1:
                    raise AmbiguousColumnError(c, dyn_outer, path)
                if len(dyn_outer) == 1:
                    return _resolve_column(outer_alias_map, dyn_outer[0])
            raise QueryPlanError(
                f"Unknown unqualified column '{c}' for current scope.",
                code="UNKNOWN_COLUMN",
                path=path,
            )

        if t in inner_alias_map:
            return _resolve_column(inner_alias_map, t)
        if outer_alias_map and t in outer_alias_map:
            return _resolve_column(outer_alias_map, t)
        raise QueryPlanError(
            f"Table/alias '{t}' referenced but not in FROM/JOIN (or correlation scope).",
            code="INVALID_PLAN",
            path=path,
        )

    def _compile_expr(
        self,
        expr: Any,
        alias_map: Dict[str, sa.FromClause],
        default_table: str,
        *,
        cte_map: Dict[str, sa.CTE],
        path: str,
        outer_alias_map: Optional[Dict[str, sa.FromClause]] = None,
    ) -> sa.ColumnElement:
        _require(isinstance(expr, dict), "INVALID_PLAN", "Expression must be an object.", path)

        if "col" in expr:
            return self._col(alias_map, default_table, expr["col"], path=f"{path}.col", outer_alias_map=outer_alias_map)

        if "lit" in expr:
            pname = self._new_param("v")
            return sa.bindparam(pname, value=expr["lit"])

        if "distinct" in expr:
            inner = self._compile_expr(
                expr["distinct"], alias_map, default_table, cte_map=cte_map, path=f"{path}.distinct", outer_alias_map=outer_alias_map
            )
            return sa.distinct(inner)

        if "cast" in expr:
            node = expr["cast"]
            _require(isinstance(node, dict), "INVALID_PLAN", "cast must be an object.", f"{path}.cast")
            inner = self._compile_expr(
                node.get("expr"), alias_map, default_table, cte_map=cte_map, path=f"{path}.cast.expr", outer_alias_map=outer_alias_map
            )
            typ = node.get("type")
            _require(isinstance(typ, str) and typ, "INVALID_PLAN", "cast.type must be a string.", f"{path}.cast.type")
            return sa.cast(inner, _map_sa_type(typ))

        if "coalesce" in expr:
            args = expr["coalesce"]
            _require(isinstance(args, list) and args, "INVALID_PLAN", "coalesce must be a non-empty list.", f"{path}.coalesce")
            compiled = [
                self._compile_expr(a, alias_map, default_table, cte_map=cte_map, path=f"{path}.coalesce[{i}]", outer_alias_map=outer_alias_map)
                for i, a in enumerate(args)
            ]
            return sa.func.coalesce(*compiled)

        if "case" in expr:
            node = expr["case"]
            _require(isinstance(node, dict), "INVALID_PLAN", "case must be an object.", f"{path}.case")
            whens = node.get("whens", [])
            else_ = node.get("else")
            _require(isinstance(whens, list) and whens, "INVALID_PLAN", "case.whens must be non-empty list.", f"{path}.case.whens")
            compiled_whens = []
            for i, w in enumerate(whens):
                wp = f"{path}.case.whens[{i}]"
                _require(isinstance(w, dict), "INVALID_PLAN", "case.when item must be object.", wp)
                cond = self._compile_bool_expr(
                    w.get("when"), alias_map, default_table, cte_map=cte_map, path=f"{wp}.when", outer_alias_map=outer_alias_map
                )
                val = self._compile_expr(
                    w.get("then"), alias_map, default_table, cte_map=cte_map, path=f"{wp}.then", outer_alias_map=outer_alias_map
                )
                compiled_whens.append((cond, val))
            else_expr = None
            if else_ is not None:
                else_expr = self._compile_expr(
                    else_, alias_map, default_table, cte_map=cte_map, path=f"{path}.case.else", outer_alias_map=outer_alias_map
                )
            return sa.case(*compiled_whens, else_=else_expr)

        if "scalar_subquery" in expr:
            node = expr["scalar_subquery"]
            _require(isinstance(node, dict), "INVALID_PLAN", "scalar_subquery must be an object.", f"{path}.scalar_subquery")
            subplan = node.get("plan")
            _require(isinstance(subplan, dict), "INVALID_PLAN", "scalar_subquery.plan must be an object.", f"{path}.scalar_subquery.plan")
            sub_sel = self._ensure_subplan_has_select(subplan)
            nested_outer = self._scope_for_nested_subquery(outer_alias_map, alias_map)
            sub_query = self._build_selectable(
                sub_sel, cte_map=cte_map, path=f"{path}.scalar_subquery.plan", outer_alias_map=nested_outer
            )
            return sa.select(sub_query.scalar_subquery()).scalar_subquery()

        if "func" in expr:
            fn = expr.get("func")
            args = expr.get("args", [])
            _require(isinstance(fn, str), "INVALID_PLAN", "func name must be string.", f"{path}.func")
            _require(isinstance(args, list), "INVALID_PLAN", "func.args must be list.", f"{path}.args")
            compiled_args = [
                self._compile_expr(a, alias_map, default_table, cte_map=cte_map, path=f"{path}.args[{i}]", outer_alias_map=outer_alias_map)
                for i, a in enumerate(args)
            ]

            fn_lower = fn.lower()
            if fn_lower in {"count_distinct", "countdistinct"}:
                _require(len(compiled_args) == 1, "INVALID_PLAN", "count_distinct requires exactly one arg.", path)
                return sa.func.count(sa.distinct(compiled_args[0]))

            # sqlalchemy.func resolves unknown names to a generic generator; arity errors surface as TypeError.
            try:
                return getattr(sa.func, fn_lower)(*compiled_args)
            except TypeError as e:
                raise QueryPlanError(
                    f"Function '{fn}' does not accept the given arguments: {e}",
                    code="INVALID_PLAN",
                    path=f"{path}.func",
                ) from e

        if "over" in expr:
            node = expr["over"]
            _require(isinstance(node, dict), "INVALID_PLAN", "over must be an object.", f"{path}.over")
            base_expr = self._compile_expr(
                node.get("expr"), alias_map, default_table, cte_map=cte_map, path=f"{path}.over.expr", outer_alias_map=outer_alias_map
            )

            part = node.get("partition_by", []) or []
            ob = node.get("order_by", []) or []

            _require(isinstance(part, list), "INVALID_PLAN", "over.partition_by must be list.", f"{path}.over.partition_by")
            _require(isinstance(ob, list), "INVALID_PLAN", "over.order_by must be list.", f"{path}.over.order_by")

            part_exprs = [
                self._compile_expr(e, alias_map, default_table, cte_map=cte_map, path=f"{path}.over.partition_by[{i}]", outer_alias_map=outer_alias_map)
                for i, e in enumerate(part)
            ]

            ob_exprs = []
            for i, item in enumerate(ob):
                ip = f"{path}.over.order_by[{i}]"
                if isinstance(item, dict) and "expr" in item:
                    e = self._compile_expr(item["expr"], alias_map, default_table, cte_map=cte_map, path=f"{ip}.expr", outer_alias_map=outer_alias_map)
                    direction = (item.get("dir") or "asc").lower()
                    _require(direction in {"asc", "desc"}, "INVALID_PLAN", "order dir must be asc|desc.", f"{ip}.dir")
                    ob_exprs.append(e.asc() if direction == "asc" else e.desc())
                else:
                    if isinstance(item, str):
                        e = self._compile_expr(
                            {"col": item},
                            alias_map,
                            default_table,
                            cte_map=cte_map,
                            path=ip,
                            outer_alias_map=outer_alias_map,
                        )
                    else:
                        e = self._compile_expr(item, alias_map, default_table, cte_map=cte_map, path=ip, outer_alias_map=outer_alias_map)
                    ob_exprs.append(e.asc())

            # Most window funcs come from sa.func.*; they support .over()
            _require(hasattr(base_expr, "over"), "INVALID_PLAN", "over.expr must support windowing (function).", f"{path}.over.expr")
            return base_expr.over(partition_by=part_exprs or None, order_by=ob_exprs or None)

        if "op" in expr:
            op = expr.get("op")
            args = expr.get("args", [])
            _require(isinstance(op, str), "INVALID_PLAN", "op must be string.", f"{path}.op")
            _require(isinstance(args, list) and len(args) >= 1, "INVALID_PLAN", "op.args must be non-empty list.", f"{path}.args")
            compiled = [
                self._compile_expr(a, alias_map, default_table, cte_map=cte_map, path=f"{path}.args[{i}]", outer_alias_map=outer_alias_map)
                for i, a in enumerate(args)
            ]
            out = compiled[0]
            for nxt in compiled[1:]:
                out = out.op(op)(nxt)
            return out

        raise QueryPlanError("Unknown expression node.", code="INVALID_PLAN", path=path)

    def _compile_bool_expr(
        self,
        node: Any,
        alias_map: Dict[str, sa.FromClause],
        default_table: str,
        *,
        cte_map: Dict[str, sa.CTE],
        path: str,
        outer_alias_map: Optional[Dict[str, sa.FromClause]] = None,
    ) -> sa.ColumnElement:
        _require(isinstance(node, dict), "INVALID_PLAN", "Boolean expression must be an object.", path)

        if "and" in node:
            items = node["and"]
            _require(isinstance(items, list) and items, "INVALID_PLAN", "and must be non-empty list.", f"{path}.and")
            _require(len(items) <= self.max_predicates, "QUERY_TOO_COMPLEX", "Too many predicates.", f"{path}.and")
            parts = [
                self._compile_bool_expr(it, alias_map, default_table, cte_map=cte_map, path=f"{path}.and[{i}]", outer_alias_map=outer_alias_map)
                for i, it in enumerate(items)
            ]
            return sa.and_(*parts)

        if "or" in node:
            items = node["or"]
            _require(isinstance(items, list) and items, "INVALID_PLAN", "or must be non-empty list.", f"{path}.or")
            _require(len(items) <= self.max_predicates, "QUERY_TOO_COMPLEX", "Too many predicates.", f"{path}.or")
            parts = [
                self._compile_bool_expr(it, alias_map, default_table, cte_map=cte_map, path=f"{path}.or[{i}]", outer_alias_map=outer_alias_map)
                for i, it in enumerate(items)
            ]
            return sa.or_(*parts)

        if "not" in node:
            inner = node["not"]
            return sa.not_(
                self._compile_bool_expr(inner, alias_map, default_table, cte_map=cte_map, path=f"{path}.not", outer_alias_map=outer_alias_map)
            )

        if "exists" in node or "not_exists" in node:
            key = "exists" if "exists" in node else "not_exists"
            subnode = node[key]
            _require(isinstance(subnode, dict), "INVALID_PLAN", f"{key} must be an object.", f"{path}.{key}")
            subplan = subnode.get("plan")
            _require(isinstance(subplan, dict), "INVALID_PLAN", f"{key}.plan must be an object.", f"{path}.{key}.plan")
            subplan = self._ensure_subplan_has_select(subplan)
            nested_outer = self._scope_for_nested_subquery(outer_alias_map, alias_map)
            subq = self._build_selectable(subplan, cte_map=cte_map, path=f"{path}.{key}.plan", outer_alias_map=nested_outer)
            ex = sa.exists(subq)
            return sa.not_(ex) if key == "not_exists" else ex

        if "cmp" in node:
            cmpn = node["cmp"]
            _require(isinstance(cmpn, dict), "INVALID_PLAN", "cmp must be object.", f"{path}.cmp")

            left = self._compile_expr(
                cmpn.get("left"), alias_map, default_table, cte_map=cte_map, path=f"{path}.cmp.left", outer_alias_map=outer_alias_map
            )
            op = cmpn.get("op")
            right_node = cmpn.get("right")
            _require(isinstance(op, str), "INVALID_PLAN", "cmp.op must be string.", f"{path}.cmp.op")

            if op == "is_null":
                return left.is_(None)
            if op == "is_not_null":
                return left.is_not(None)

            if isinstance(right_node, dict):
                right = self._compile_expr(
                    right_node, alias_map, default_table, cte_map=cte_map, path=f"{path}.cmp.right", outer_alias_map=outer_alias_map
                )
            else:
                right = sa.bindparam(self._new_param("v"), value=right_node)

            if op == "=":
                return left == right
            if op == "!=":
                return left != right
            if op == ">":
                return left > right
            if op == ">=":
                return left >= right
            if op == "<":
                return left < right
            if op == "<=":
                return left <= right

            if op in {"in", "not_in"}:
                _require(not isinstance(right_node, dict), "INVALID_PLAN", "IN requires literal list.", f"{path}.cmp.right")
                _require(isinstance(right_node, list), "INVALID_PLAN", "IN requires list value.", f"{path}.cmp.right")
                _require(len(right_node) > 0, "INVALID_PLAN", "IN list must be non-empty.", f"{path}.cmp.right")
                bp = sa.bindparam(self._new_param("in"), value=right_node, expanding=True)
                expr = left.in_(bp)
                return sa.not_(expr) if op == "not_in" else expr

            if op in {"contains", "not_contains", "starts_with", "ends_with"}:
                _require(not isinstance(right_node, dict), "INVALID_PLAN", "LIKE ops require literal string.", f"{path}.cmp.right")
                _require(isinstance(right_node, str), "INVALID_PLAN", "LIKE ops require string value.", f"{path}.cmp.right")

                if op in {"contains", "not_contains"}:
                    pat = f"%{right_node}%"
                elif op == "starts_with":
                    pat = f"{right_node}%"
                else:
                    pat = f"%{right_node}"

                bp = sa.bindparam(self._new_param("s"), value=pat)
                like_expr = left.ilike(bp)
                return sa.not_(like_expr) if op == "not_contains" else like_expr

            raise QueryPlanError(
                f"Unsupported comparison op '{op}'.",
                code="UNSUPPORTED_OPERATION",
                path=f"{path}.cmp.op",
            )

        raise QueryPlanError("Unknown boolean node.", code="INVALID_PLAN", path=path)

    # ---------- Rollup filter helper ----------
    def _compile_rollup_filter(self, subq: sa.Subquery, filt: Any, path: str) -> sa.ColumnElement:
        _require(isinstance(filt, dict), "INVALID_PLAN", "rollup filter must be an object.", path)
        field = filt.get("field")
        op = filt.get("op")
        value = filt.get("value")

        _require(isinstance(field, str) and field, "INVALID_PLAN", "rollup filter.field required.", f"{path}.field")
        _require(field in subq.c, "INVALID_PLAN", f"rollup filter.field '{field}' not in inner output.", f"{path}.field")
        _require(isinstance(op, str) and op, "INVALID_PLAN", "rollup filter.op required.", f"{path}.op")

        col = subq.c[field]

        if op == "is_null":
            return col.is_(None)
        if op == "is_not_null":
            return col.is_not(None)

        if op in {"contains", "not_contains", "starts_with", "ends_with"}:
            _require(isinstance(value, str), "INVALID_PLAN", "LIKE ops require string value.", f"{path}.value")
            if op in {"contains", "not_contains"}:
                pat = f"%{value}%"
            elif op == "starts_with":
                pat = f"{value}%"
            else:
                pat = f"%{value}"
            bp = sa.bindparam(self._new_param("rs"), value=pat)
            like_expr = col.ilike(bp)
            return sa.not_(like_expr) if op == "not_contains" else like_expr

        if op in {"in", "not_in"}:
            _require(isinstance(value, list), "INVALID_PLAN", "IN ops require list value.", f"{path}.value")
            _require(len(value) > 0, "INVALID_PLAN", "IN list must be non-empty.", f"{path}.value")
            bp = sa.bindparam(self._new_param("rin"), value=value, expanding=True)
            expr = col.in_(bp)
            return sa.not_(expr) if op == "not_in" else expr

        bp = sa.bindparam(self._new_param("rv"), value=value)
        if op == "=":
            return col == bp
        if op == "!=":
            return col != bp
        if op == ">":
            return col > bp
        if op == ">=":
            return col >= bp
        if op == "<":
            return col < bp
        if op == "<=":
            return col <= bp

        raise QueryPlanError(
            f"Unsupported rollup filter op '{op}'.",
            code="UNSUPPORTED_OPERATION",
            path=f"{path}.op",
        )

    # ---------- Subplan select enforcement ----------
    def _ensure_subplan_has_select(self, subplan: dict) -> dict:
        # If a user gives {dataset, where,...} for EXISTS/scalar subquery, auto-add SELECT 1
        if "set_op" in subplan:
            return subplan
        if any(k in subplan for k in ("dimensions", "metrics", "filters")) and "select" not in subplan:
            subplan = self._legacy_to_select_plan(subplan)
        if "select" not in subplan or not subplan.get("select"):
            # minimal select for EXISTS/scalar subquery
            subplan = dict(subplan)
            subplan["select"] = [{"expr": {"lit": 1}, "alias": "one"}]
        return subplan