from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from .queryplan_models import QueryPlan

_MAX_CTE_DEPTH = 8


def _is_legacy_queryplan_shape(d: dict) -> bool:
    """Same heuristic as Compiler._build_selectable legacy branch."""
    return bool(
        d.get("dataset")
        and any(k in d for k in ("dimensions", "metrics", "filters"))
        and "select" not in d
        and "set_op" not in d
    )


def _prefix_cte_path(i: int, inner_path: str) -> str:
    base = f"$.with[{i}].plan"
    if inner_path == "$":
        return base
    if inner_path.startswith("$."):
        return base + inner_path[1:]
    return f"{base}.{inner_path}"


@dataclass
class ValidationErrorItem:
    path: str
    message: str


def load_schema_yaml(schema_path: str) -> Dict[str, Any]:
    with open(schema_path, "r") as f:
        return yaml.safe_load(f) or {}


def _schema_tables(schema: Dict[str, Any]) -> Dict[str, Set[str]]:
    """
    Returns:
      { "assets": {"asset_tag", ...}, "work_orders": {...} }
    """
    out: Dict[str, Set[str]] = {}
    for t in schema.get("tables", []) or []:
        tname = t.get("name")
        cols = set()
        for c in t.get("columns", []) or []:
            cname = c.get("name")
            if cname:
                cols.add(cname)
        if tname:
            out[tname] = cols
    return out


def validate_query_plan_dict(
    plan_dict: Dict[str, Any],
    schema_path: str,
    *,
    _cte_depth: int = 0,
    _visible_cte_names: Optional[Set[str]] = None,
) -> Tuple[Optional[QueryPlan], List[ValidationErrorItem]]:
    """
    Validates plan_dict:
      1) Pydantic (hard schema)
      2) Dataset/field allowlist from schema.yaml
      3) DSL semantic rules (rollup references aliases, etc.)

    The ``meta`` key (planner explainability: plan_hash, retries, etc.) is stripped
    before schema validation — same as ``execute_query_plan`` / ``validate_query_plan``.

    Nested ``with`` / CTE plans are validated recursively when they match the legacy
    QueryPlan shape (or full QueryPlan including nested ``with``).

    ``_visible_cte_names`` (internal): CTE names from earlier siblings in the same
    ``WITH`` list so ``dataset`` can reference ``WITH a AS (...), b AS (SELECT ... FROM a)``.

    Returns:
      (parsed_plan or None, errors)
    """
    errors: List[ValidationErrorItem] = []

    if _cte_depth > _MAX_CTE_DEPTH:
        return None, [ValidationErrorItem(path="$", message=f"CTE nesting exceeds {_MAX_CTE_DEPTH}.")]

    plan_body = {k: v for k, v in plan_dict.items() if k != "meta"}

    # 1) Hard schema validation
    try:
        plan = QueryPlan.model_validate(plan_body)
    except Exception as e:
        return None, [ValidationErrorItem(path="$", message=f"QueryPlan schema invalid: {e}")]

    # 2) Load schema.yaml allowlist
    schema = load_schema_yaml(schema_path)
    table_cols = _schema_tables(schema)

    local_cte_names = {c.name for c in (plan.ctes or []) if c.name}
    visible_cte_names = local_cte_names | (_visible_cte_names or set())

    if plan.dataset in table_cols:
        allowed_cols: Optional[Set[str]] = table_cols[plan.dataset]
    elif plan.dataset in visible_cte_names:
        # Query reads FROM a WITH subquery — column set depends on inner plan; compiler enforces.
        allowed_cols = None
    else:
        keys = sorted(table_cols.keys())
        if visible_cte_names:
            keys = sorted(set(keys) | visible_cte_names)
        errors.append(
            ValidationErrorItem(
                path="$.dataset",
                message=f"Unknown dataset '{plan.dataset}'. Must be a schema table or a CTE name from \"with\": {keys}",
            )
        )
        return plan, errors

    # helper
    def check_col(path: str, col: Optional[str]):
        if col is None:
            return
        if col == "*":
            return
        if allowed_cols is not None and col not in allowed_cols:
            errors.append(ValidationErrorItem(path=path, message=f"Unknown column '{col}' for dataset '{plan.dataset}'."))

    # filters
    for i, f in enumerate(plan.filters):
        check_col(f"$.filters[{i}].field", f.field)

    # dimensions
    for i, d in enumerate(plan.dimensions):
        check_col(f"$.dimensions[{i}].field", d.field)

    # metrics (inner)
    for i, m in enumerate(plan.metrics):
        # count(*) allowed, but avg(*) not allowed
        if (m.agg in {"avg", "sum", "min", "max", "count_distinct"}) and (m.field in (None, "*")):
            errors.append(ValidationErrorItem(
                path=f"$.metrics[{i}].field",
                message=f"Metric '{m.alias}' uses agg='{m.agg}' but field is missing or '*'. Provide a real column.",
            ))
        check_col(f"$.metrics[{i}].field", m.field)

    # alias uniqueness
    aliases = [m.alias for m in plan.metrics]
    if len(set(aliases)) != len(aliases):
        errors.append(ValidationErrorItem(path="$.metrics", message="Metric aliases must be unique."))

    # rollup semantics
    if plan.rollup is not None:
        # rollup.metrics[*].field must reference an INNER metric alias (not a raw column)
        inner_aliases = set(m.alias for m in plan.metrics)
        for i, rm in enumerate(plan.rollup.metrics):
            if rm.field not in inner_aliases:
                errors.append(ValidationErrorItem(
                    path=f"$.rollup.metrics[{i}].field",
                    message=f"Rollup field '{rm.field}' must reference an inner metric alias. Allowed: {sorted(inner_aliases)}",
                ))

            # rollup metrics must have real alias, and agg must be aggregation
            if not rm.alias:
                errors.append(ValidationErrorItem(path=f"$.rollup.metrics[{i}].alias", message="Rollup metric alias is required."))

        # rollup queries usually return 1 row; enforce sane defaults
        if plan.rollup.limit < 1:
            errors.append(ValidationErrorItem(path="$.rollup.limit", message="rollup.limit must be >= 1"))

    # Nested CTE plans (same column allowlist rules where applicable)
    if plan.ctes:
        for i, cte in enumerate(plan.ctes):
            sub = cte.plan
            if not isinstance(sub, dict):
                errors.append(ValidationErrorItem(path=f"$.with[{i}].plan", message="CTE plan must be an object."))
                continue
            merged = dict(sub)
            if _is_legacy_queryplan_shape(merged) and "version" not in merged:
                merged["version"] = "1.0"
            try:
                QueryPlan.model_validate(merged)
            except Exception:
                # Advanced select/set_op-only CTEs: compiler is authoritative; skip Pydantic subtree.
                continue
            prior_sibling_names = {
                plan.ctes[j].name for j in range(i) if plan.ctes[j].name
            }
            inherited_visible = prior_sibling_names | (_visible_cte_names or set())
            _, inner_errs = validate_query_plan_dict(
                merged,
                schema_path,
                _cte_depth=_cte_depth + 1,
                _visible_cte_names=inherited_visible,
            )
            for e in inner_errs:
                errors.append(ValidationErrorItem(path=_prefix_cte_path(i, e.path), message=e.message))

    return plan, errors