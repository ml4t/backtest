"""Enforce release coverage thresholds from coverage.py JSON output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

GLOBAL_STATEMENT_MIN = 87.0
GLOBAL_BRANCH_MIN = 76.0
CRITICAL_THRESHOLDS = {
    "src/ml4t/backtest/accounting/policy.py": (89.0, 86.0),
    "src/ml4t/backtest/broker.py": (92.0, 80.0),
    "src/ml4t/backtest/config.py": (85.0, 64.0),
    "src/ml4t/backtest/core/execution_engine.py": (88.0, 81.0),
    "src/ml4t/backtest/core/fill_engine.py": (88.0, 79.0),
    "src/ml4t/backtest/core/order_book.py": (84.0, 71.0),
    "src/ml4t/backtest/core/risk_engine.py": (92.0, 77.0),
    "src/ml4t/backtest/datafeed.py": (90.0, 77.0),
    "src/ml4t/backtest/engine.py": (87.0, 84.0),
    "src/ml4t/backtest/execution/fill_executor.py": (98.0, 88.0),
    "src/ml4t/backtest/result.py": (93.0, 88.0),
    "src/ml4t/backtest/types.py": (95.0, 85.0),
}


def _percent(summary: dict[str, Any], kind: str) -> float:
    key = f"percent_{kind}_covered"
    value = summary.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"Coverage summary is missing numeric {key}")
    return float(value)


def coverage_failures(payload: dict[str, Any]) -> list[str]:
    meta = payload.get("meta", {})
    if meta.get("branch_coverage") is not True:
        return ["Coverage evidence was collected without branch coverage"]

    failures: list[str] = []
    totals = payload["totals"]
    global_statement = _percent(totals, "statements")
    global_branch = _percent(totals, "branches")
    if global_statement < GLOBAL_STATEMENT_MIN:
        failures.append(
            f"Global statement coverage {global_statement:.2f}% < {GLOBAL_STATEMENT_MIN:.2f}%"
        )
    if global_branch < GLOBAL_BRANCH_MIN:
        failures.append(f"Global branch coverage {global_branch:.2f}% < {GLOBAL_BRANCH_MIN:.2f}%")

    files = payload["files"]
    for filename, (statement_min, branch_min) in CRITICAL_THRESHOLDS.items():
        if filename not in files:
            failures.append(f"Critical module is absent from coverage evidence: {filename}")
            continue
        summary = files[filename]["summary"]
        statement = _percent(summary, "statements")
        branch = _percent(summary, "branches")
        if statement < statement_min:
            failures.append(
                f"{filename} statement coverage {statement:.2f}% < {statement_min:.2f}%"
            )
        if branch < branch_min:
            failures.append(f"{filename} branch coverage {branch:.2f}% < {branch_min:.2f}%")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("coverage_json", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.coverage_json.read_text(encoding="utf-8"))
    failures = coverage_failures(payload)
    if failures:
        for failure in failures:
            print(failure)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
