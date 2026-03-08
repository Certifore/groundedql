"""
join_planner.py — Automatic join-path resolution.

Builds a graph from the `links` section of schema.yaml and computes the
shortest join path between two tables using BFS. The planner uses this to
auto-inject join instructions into a QueryPlan when a query references
columns from multiple tables but declares no explicit joins.

This is purely additive — if the plan already has joins, this is a no-op.
If the plan only references one table, this is a no-op.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple


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


# Keys that contain nested plans — do NOT scan these for join injection
_SUBPLAN_KEYS = {"exists", "not_exists", "scalar_subquery", "with", "set_op"}


def _extract_referenced_tables(plan: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """
    Scan a plan's top-level select/where/group_by/order_by/having for qualified
    column refs (e.g. "work_orders.building_id") and return the set of tables
    referenced. Only returns tables in the schema.

    Deliberately does NOT recurse into subplan nodes (exists, scalar_subquery,
    with, set_op) to avoid injecting joins for tables that are only referenced
    inside a subquery scope.
    """
    known_tables = {t["name"] for t in schema.get("tables", [])}
    found = set()

    def _scan(obj: Any, depth: int = 0) -> None:
        if isinstance(obj, dict):
            # Stop recursing into subplan boundaries
            if depth > 0 and any(k in obj for k in _SUBPLAN_KEYS):
                return
            if "col" in obj:
                ref = obj["col"]
                if isinstance(ref, str) and "." in ref:
                    table = ref.split(".")[0]
                    if table in known_tables:
                        found.add(table)
            for k, v in obj.items():
                if k in _SUBPLAN_KEYS:
                    continue  # skip subplan branches entirely
                _scan(v, depth + 1)
        elif isinstance(obj, list):
            for item in obj:
                _scan(item, depth)

    _scan(plan)
    return list(found)


def auto_inject_joins(plan: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    If a plan references columns from multiple tables but has no explicit joins,
    compute the shortest join paths from the primary dataset to each referenced
    table and inject them as join instructions.

    Returns the (possibly modified) plan. Never modifies in-place.
    If no joins are needed or paths cannot be found, returns plan unchanged.
    """
    # Only applies to advanced format plans with a select list
    if "select" not in plan or "set_op" in plan or "with" in plan:
        return plan

    # If joins are already declared, do nothing
    if plan.get("joins"):
        return plan

    primary_dataset = plan.get("dataset")
    if not primary_dataset:
        return plan

    referenced = _extract_referenced_tables(plan, schema)

    # Tables to join: everything referenced except the primary dataset
    tables_to_join = [t for t in referenced if t != primary_dataset]
    if not tables_to_join:
        return plan

    graph = build_link_graph(schema)
    injected_joins = []

    for target in tables_to_join:
        path = shortest_join_path(graph, primary_dataset, target)
        if path is None:
            # No path found — leave plan unchanged and let compiler raise
            continue

        for link in path:
            join_entry = {
                "link": link["name"],
            }
            # Avoid duplicate joins
            if join_entry not in injected_joins:
                injected_joins.append(join_entry)

    if not injected_joins:
        return plan

    return {**plan, "joins": injected_joins}
