"""
Summarize a BIRD Mini-Dev benchmark report.

This is intentionally heuristic: it helps find clusters worth debugging first,
while the JSON report remains the source of truth for exact examples.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _text(result: dict[str, Any]) -> str:
    parts = [
        str(result.get("question") or ""),
        str(result.get("evidence") or ""),
        str(result.get("gold_sql") or ""),
        str(result.get("sql") or ""),
        str(result.get("error") or ""),
    ]
    return "\n".join(parts).lower()


def tags_for(result: dict[str, Any]) -> set[str]:
    text = _text(result)
    tags = set()

    if "429" in text or "too many requests" in text:
        tags.add("rate_limit")
    if "validation" in text or "unknown" in text or "not found on table" in text:
        tags.add("schema_or_validation")
    if re.search(r"\bratio\b|\bpercentage\b|\bpercent\b|\bdifference\b|\bdeviation\b|\bincrease\b|\bdecrease\b", text):
        tags.add("arithmetic_ratio_delta")
    if re.search(r"\bjoin\b| exists ", text) or " in (select" in text or "not in (select" in text:
        tags.add("joins_or_subqueries")
    if re.search(r"\byear\b|\bmonth\b|\bdate\b|\btime\b|\bbetween\b|20\d\d|19\d\d", text):
        tags.add("date_time")
    if re.search(r"\bmost\b|\bleast\b|\btop\b|\bhighest\b|\blowest\b|\bmaximum\b|\bminimum\b", text) or "max(" in text or "min(" in text:
        tags.add("top_or_extreme")
    if re.search(r"\baverage\b|\bgroup by\b", text) or "avg(" in text or "sum(" in text or "count(" in text:
        tags.add("aggregation")
    if not tags:
        tags.add("other")

    return tags


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize BIRD Mini-Dev benchmark results.")
    parser.add_argument("report", nargs="?", default="test/benchmark/results/bird_minidev_full_mistral.json")
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    results = report.get("results", [])
    print({
        key: report.get(key)
        for key in ("planned_total", "total", "attempted", "correct", "wrong", "errors", "skipped", "execution_accuracy")
    })

    by_status = Counter(result.get("status") for result in results)
    print("\nstatus")
    for status, count in by_status.most_common():
        print(f"  {status}: {count}")

    print("\nby_db")
    by_db: dict[str, Counter[str]] = defaultdict(Counter)
    for result in results:
        by_db[str(result.get("db_id"))][str(result.get("status"))] += 1
    for db_id, counts in sorted(by_db.items()):
        print(f"  {db_id}: {dict(counts)}")

    print("\nfailure_tags")
    tag_counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        if result.get("status") == "correct":
            continue
        for tag in tags_for(result):
            tag_counts[tag] += 1
            if len(examples[tag]) < args.examples:
                examples[tag].append(result)

    for tag, count in tag_counts.most_common():
        print(f"  {tag}: {count}")
        for result in examples[tag]:
            print(f"    - #{result.get('id')} {result.get('db_id')}: {result.get('question')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
