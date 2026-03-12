"""
QCE Benchmark Runner

Usage:
    python3 benchmark/run_benchmarks.py

No DB connection required. No LLM required.
Results are printed to the terminal and saved to benchmark/results/.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the repo root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.bench_injection import run as run_injection
from benchmarks.bench_determinism import run as run_determinism
from benchmarks.bench_hallucination import run as run_hallucination

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _print_result(result: dict) -> None:
    name = result["benchmark"].replace("_", " ").title()
    passed = result["passed"]
    score = result["score"]
    status = "✅ PASS" if passed else "❌ FAIL"

    print(f"\n{'─' * 50}")
    print(f"  {name}")
    print(f"  Score : {score}")
    print(f"  Status: {status}")

    if not passed:
        key = "passed_through" if "passed_through" in result else "silent_failures" if "silent_failures" in result else "failures"
        failures = result.get(key, [])
        if failures:
            print(f"\n  Failed cases ({len(failures)}):")
            for f in failures[:5]:
                print(f"    [{f.get('id')}] {f.get('description') or f.get('question', '')}")
            if len(failures) > 5:
                print(f"    ... and {len(failures) - 5} more (see results file)")


def main() -> None:
    print("=" * 50)
    print("  QCE Benchmark Suite")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 50)

    results = []

    print("\nRunning Benchmark 1: Injection Resistance...")
    r1 = run_injection()
    results.append(r1)
    _print_result(r1)

    print("\nRunning Benchmark 2: Determinism...")
    r2 = run_determinism()
    results.append(r2)
    _print_result(r2)

    print("\nRunning Benchmark 3: Hallucination Rejection...")
    r3 = run_hallucination()
    results.append(r3)
    _print_result(r3)

    all_passed = all(r["passed"] for r in results)

    print(f"\n{'=' * 50}")
    print(f"  Overall: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")
    for r in results:
        name = r["benchmark"].replace("_", " ").title()
        print(f"    {name}: {r['score']} {'✅' if r['passed'] else '❌'}")
    print("=" * 50)

    # Save results
    output = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "all_passed": all_passed,
        "results": results,
    }
    out_path = RESULTS_DIR / "latest.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Results saved → {out_path}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
