"""
Deterministic structural canonicalization of QueryPlan JSON.

Goals:
  - Same semantic plan (up to commutative reorderings we define) → same bytes for hashing.
  - Stable join / filter / dimension / metric ordering so auto_inject and LLM key order
    do not change plan_hash or compiled SQL shape.

Limits:
  - Does not prove semantic equivalence (different JSON can mean the same query).
  - Preserves order where it matters: order_by, case.whens, rollup.order_by, set_op sides.
  - NL → plan is still non-deterministic when the LLM varies; this normalizes structure only.
"""
from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Union

Json = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def _sort_dict_keys_deep(obj: Json) -> Json:
    if isinstance(obj, dict):
        return {k: _sort_dict_keys_deep(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        return [_sort_dict_keys_deep(x) for x in obj]
    return obj


def _dim_sort_key(d: Any) -> tuple:
    if isinstance(d, dict):
        return (d.get("alias") or "", d.get("field") or "", _stable_json(d))
    return ("", "", _stable_json(d))


def _metric_sort_key(m: Any) -> tuple:
    if isinstance(m, dict):
        return (m.get("alias") or "", m.get("agg") or "", m.get("field") or "", _stable_json(m))
    return ("", "", "", _stable_json(m))


def _filter_sort_key(f: Any) -> tuple:
    if isinstance(f, dict):
        return (
            str(f.get("field") or ""),
            str(f.get("op") or ""),
            _stable_json(f.get("value")),
            _stable_json(f),
        )
    return ("", "", "", _stable_json(f))


def _select_item_sort_key(item: Any) -> tuple:
    if isinstance(item, dict):
        return (item.get("alias") or "", _stable_json(item))
    return ("", _stable_json(item))


def _join_sort_key(j: Any) -> tuple:
    if isinstance(j, dict):
        if "link" in j:
            return (str(j.get("link") or ""), "", _stable_json(j))
        return (str(j.get("dataset") or ""), _stable_json(j.get("on")), _stable_json(j))
    return ("", "", _stable_json(j))


def _cte_sort_key(c: Any) -> tuple:
    if isinstance(c, dict):
        return (str(c.get("name") or ""), _stable_json(c))
    return ("", _stable_json(c))


def _rollup_filter_sort_key(f: Any) -> tuple:
    if isinstance(f, dict):
        return (str(f.get("field") or ""), str(f.get("op") or ""), _stable_json(f.get("value")))
    return ("", "", _stable_json(f))


def _canonicalize_commutative_lists(obj: Json) -> None:
    """In-place: reorder selected lists to a stable order."""
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if k == "filters" and isinstance(v, list):
                for x in v:
                    _canonicalize_commutative_lists(x)
                obj[k] = sorted(v, key=_filter_sort_key)
            elif k == "dimensions" and isinstance(v, list):
                for x in v:
                    _canonicalize_commutative_lists(x)
                obj[k] = sorted(v, key=_dim_sort_key)
            elif k == "metrics" and isinstance(v, list):
                for x in v:
                    _canonicalize_commutative_lists(x)
                obj[k] = sorted(v, key=_metric_sort_key)
            elif k == "select" and isinstance(v, list):
                for x in v:
                    _canonicalize_commutative_lists(x)
                obj[k] = sorted(v, key=_select_item_sort_key)
            elif k == "joins" and isinstance(v, list):
                for x in v:
                    _canonicalize_commutative_lists(x)
                obj[k] = sorted(v, key=_join_sort_key)
            elif k == "with" and isinstance(v, list):
                for x in v:
                    _canonicalize_commutative_lists(x)
                obj[k] = sorted(v, key=_cte_sort_key)
            elif k == "group_by" and isinstance(v, list):
                for x in v:
                    _canonicalize_commutative_lists(x)
                obj[k] = sorted(v, key=_stable_json)
            elif k == "rollup" and isinstance(v, dict):
                _canonicalize_commutative_lists(v)
                rf = v.get("filters")
                if isinstance(rf, list):
                    for x in rf:
                        _canonicalize_commutative_lists(x)
                    v["filters"] = sorted(rf, key=_rollup_filter_sort_key)
                rmd = v.get("dimensions")
                if isinstance(rmd, list):
                    for x in rmd:
                        _canonicalize_commutative_lists(x)
                    v["dimensions"] = sorted(rmd, key=_dim_sort_key)
                rmt = v.get("metrics")
                if isinstance(rmt, list):
                    for x in rmt:
                        _canonicalize_commutative_lists(x)
                    v["metrics"] = sorted(rmt, key=_metric_sort_key)
            elif k == "and" and isinstance(v, list):
                for x in v:
                    _canonicalize_commutative_lists(x)
                obj[k] = sorted(v, key=_stable_json)
            elif k == "or" and isinstance(v, list):
                for x in v:
                    _canonicalize_commutative_lists(x)
                obj[k] = sorted(v, key=_stable_json)
            elif k == "set_op" and isinstance(v, dict):
                _canonicalize_commutative_lists(v)
            elif k in ("exists", "not_exists") and isinstance(v, dict):
                p = v.get("plan")
                if isinstance(p, dict):
                    _canonicalize_commutative_lists(p)
            elif k == "scalar_subquery" and isinstance(v, dict):
                p = v.get("plan")
                if isinstance(p, dict):
                    _canonicalize_commutative_lists(p)
            else:
                _canonicalize_commutative_lists(v)
    elif isinstance(obj, list):
        for item in obj:
            _canonicalize_commutative_lists(item)


def canonicalize_query_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a deep-copied plan with commutative lists sorted and all object keys
    sorted recursively. Safe to call before compile and before plan_hash.
    """
    if not isinstance(plan, dict):
        raise TypeError("canonicalize_query_plan expects a dict")
    out = copy.deepcopy(plan)
    _canonicalize_commutative_lists(out)
    sorted_tree = _sort_dict_keys_deep(out)
    assert isinstance(sorted_tree, dict)
    return sorted_tree


def plan_fingerprint(plan: Dict[str, Any]) -> str:
    """Stable SHA-256 hex digest (64 chars) of the canonical JSON plan (no meta)."""
    import hashlib

    body = {k: v for k, v in plan.items() if k != "meta"}
    c = canonicalize_query_plan(body)
    payload = json.dumps(c, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
