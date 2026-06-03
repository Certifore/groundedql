"""
join_planner.py — Automatic join-path resolution.

Builds a graph from the `links` section of schema.yaml and computes the
shortest join path between the primary dataset and every other referenced
logical table (BFS, multi-hop). Injects `{"link": "<name>"}` entries so the
compiler can connect any referenced schema tables that are linked in schema.yaml.

Recurses into WITH (CTE) bodies, set_op left/right branches, and nested
exists / scalar_subquery plan bodies so each sub-plan gets its own inference.

Returns a deep copy — never mutates the caller's plan dict.
"""
from __future__ import annotations

import copy
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple


def build_link_graph(schema: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build an adjacency graph from schema links.
    Each node is a logical table name.
    Each edge carries the link definition so we can reconstruct the join path.

    Returns: {table_name: [{"to": table, "link": link_def}, ...]}
    """
    graph: Dict[str, List[Dict[str, Any]]] = {}

    for link in schema.get("links", []) or []:
        frm = link.get("from_table")
        to = link.get("to_table")
        if not frm or not to:
            continue

        if frm not in graph:
            graph[frm] = []
        if to not in graph:
            graph[to] = []

        graph[frm].append({"to": to, "link": link})
        # Links are traversable in both directions
        graph[to].append({"to": frm, "link": link})

    return graph


def shortest_join_path(
    graph: Dict[str, List[Dict[str, Any]]],
    start: str,
    end: str,
) -> Optional[List[Dict[str, Any]]]:
    """
    BFS shortest path from `start` to `end` table.

    Returns list of link defs in traversal order, or None if no path exists.
    """
    if start == end:
        return []

    visited = {start}
    # Each queue entry: (current_table, path_of_link_defs_so_far)
    queue: deque[Tuple[str, List[Dict[str, Any]]]] = deque([(start, [])])

    while queue:
        current, path = queue.popleft()

        for edge in graph.get(current, []):
            neighbor = edge["to"]
            link = edge["link"]
            new_path = path + [link]

            if neighbor == end:
                return new_path

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, new_path))

    return None  # no path found


# Keys that contain nested plans — do NOT scan inside these for *this* plan's join targets
_SUBPLAN_KEYS = {"exists", "not_exists", "scalar_subquery", "with", "set_op"}
_BOOLEAN_SUBQUERY_KEYS = {"exists", "not_exists", "scalar_subquery"}


def _known_tables(schema: Dict[str, Any]) -> Set[str]:
    return {t["name"] for t in schema.get("tables", []) if isinstance(t.get("name"), str)}


def _extract_referenced_tables(plan: Dict[str, Any], schema: Dict[str, Any]) -> Set[str]:
    """
    Qualified col refs (table.column) anywhere in this plan node except inside
    exists / not_exists / scalar_subquery / with / set_op subtrees.
    """
    known_tables = _known_tables(schema)
    found: Set[str] = set()

    def _scan(obj: Any, depth: int = 0) -> None:
        if isinstance(obj, dict):
            if depth > 0 and any(k in obj for k in _SUBPLAN_KEYS):
                return
            if "col" in obj:
                ref = obj["col"]
                if isinstance(ref, str) and "." in ref:
                    table = ref.split(".", 1)[0]
                    if table in known_tables:
                        found.add(table)
            for k, v in obj.items():
                if k in _SUBPLAN_KEYS:
                    continue
                _scan(v, depth + 1)
        elif isinstance(obj, list):
            for item in obj:
                _scan(item, depth)

    _scan(plan)
    return found


def _extract_legacy_table_refs(plan: Dict[str, Any], schema: Dict[str, Any]) -> Set[str]:
    """
    Legacy dimensions / metrics / filters: unqualified fields belong to plan.dataset;
    qualified fields contribute their logical table.
    """
    known = _known_tables(schema)
    ds = plan.get("dataset")
    out: Set[str] = set()
    if isinstance(ds, str) and ds in known:
        out.add(ds)

    def add_field(field: Any) -> None:
        if not isinstance(field, str) or field == "*":
            return
        if "." in field:
            t = field.split(".", 1)[0]
            if t in known:
                out.add(t)
        elif isinstance(ds, str) and ds in known:
            out.add(ds)

    for d in plan.get("dimensions") or []:
        if isinstance(d, dict):
            add_field(d.get("field"))
    for m in plan.get("metrics") or []:
        if isinstance(m, dict):
            add_field(m.get("field"))
    for f in plan.get("filters") or []:
        if isinstance(f, dict):
            add_field(f.get("field"))
    return out


def _collect_tables_for_join_injection(plan: Dict[str, Any], schema: Dict[str, Any]) -> Set[str]:
    return _extract_referenced_tables(plan, schema) | _extract_legacy_table_refs(plan, schema)


def _eligible_for_link_inject(plan: Dict[str, Any]) -> bool:
    if not isinstance(plan, dict):
        return False
    if plan.get("joins"):
        return False
    if plan.get("set_op"):
        return False
    ds = plan.get("dataset")
    if not isinstance(ds, str) or not ds:
        return False
    if plan.get("select"):
        return True
    if plan.get("dimensions") or plan.get("metrics") or plan.get("filters"):
        return True
    return False


def _inject_joins_on_node(plan: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """
    Mutate plan in place: add joins list if inference succeeds.
    """
    if not _eligible_for_link_inject(plan):
        return

    primary_dataset = plan["dataset"]
    referenced = _collect_tables_for_join_injection(plan, schema)
    tables_to_join = [t for t in referenced if t != primary_dataset]
    if not tables_to_join:
        return

    graph = build_link_graph(schema)
    injected_joins: List[Dict[str, str]] = []

    for target in tables_to_join:
        path = shortest_join_path(graph, primary_dataset, target)
        if path is None:
            continue
        for link in path:
            join_entry = {"link": link["name"]}
            if join_entry not in injected_joins:
                injected_joins.append(join_entry)

    if injected_joins:
        plan["joins"] = injected_joins


def _auto_inject_joins_recursive(plan: Any, schema: Dict[str, Any]) -> None:
    """Walk plan tree (dict/list) and run inference on each selectable fragment."""
    if isinstance(plan, dict):
        for cte in plan.get("with") or []:
            if isinstance(cte, dict) and isinstance(cte.get("plan"), dict):
                _auto_inject_joins_recursive(cte["plan"], schema)

        sop = plan.get("set_op")
        if isinstance(sop, dict):
            if isinstance(sop.get("left"), dict):
                _auto_inject_joins_recursive(sop["left"], schema)
            if isinstance(sop.get("right"), dict):
                _auto_inject_joins_recursive(sop["right"], schema)

        _auto_inject_nested_subquery_plans(plan, schema)
        _inject_joins_on_node(plan, schema)
    elif isinstance(plan, list):
        for item in plan:
            _auto_inject_joins_recursive(item, schema)


def _auto_inject_nested_subquery_plans(obj: Any, schema: Dict[str, Any]) -> None:
    """Find nested {exists|not_exists|scalar_subquery: {plan: ...}} fragments."""
    if isinstance(obj, dict):
        for key in _BOOLEAN_SUBQUERY_KEYS:
            node = obj.get(key)
            if isinstance(node, dict) and isinstance(node.get("plan"), dict):
                _auto_inject_joins_recursive(node["plan"], schema)
        for value in obj.values():
            _auto_inject_nested_subquery_plans(value, schema)
    elif isinstance(obj, list):
        for item in obj:
            _auto_inject_nested_subquery_plans(item, schema)


def auto_inject_joins(plan: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Infer link-based joins for every selectable sub-plan (root, CTE bodies, set_op sides).

    - Multi-hop: shortest BFS path from primary `dataset` to each other referenced table.
    - Advanced (`select`) and legacy (`dimensions` / `metrics` / `filters`) shapes.
    - Deep-copies the plan first; the original dict is never modified.

    If the link graph has no path to a referenced table, that table is skipped
    (compiler may still error later). Nested subquery bodies get their own
    independent join inference.
    """
    cloned = copy.deepcopy(plan)
    _auto_inject_joins_recursive(cloned, schema)
    return cloned
