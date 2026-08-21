#!/usr/bin/env python3
"""Generate parity documentation from retained release evidence."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from common.correctness_evidence import correctness_report_failures
from common.framework_registry import load_framework_manifest
from large_scale_evidence import report_failures as large_scale_report_failures
from real_strategy_benchmark import report_failures as real_strategy_performance_failures
from real_strategy_evidence import report_failures as real_strategy_report_failures
from scenarios.definitions import SCENARIOS

PROJECT_ROOT = Path(__file__).parent.parent
CORRECTNESS_EVIDENCE = PROJECT_ROOT / "validation" / "CORRECTNESS_RESULTS.json"
LARGE_SCALE_EVIDENCE = PROJECT_ROOT / "validation" / "LARGE_SCALE_RESULTS.json"
REAL_STRATEGY_EVIDENCE = PROJECT_ROOT / "validation" / "REAL_STRATEGY_RESULTS.json"
REAL_STRATEGY_PERFORMANCE = PROJECT_ROOT / "validation" / "REAL_STRATEGY_PERFORMANCE.json"
TARGETS = (
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "docs" / "index.md",
    PROJECT_ROOT / "docs" / "user-guide" / "profiles.md",
    PROJECT_ROOT / "validation" / "README.md",
    PROJECT_ROOT / "validation" / "METHODOLOGY.md",
)
START_MARKER = "<!-- parity-claims:start -->"
END_MARKER = "<!-- parity-claims:end -->"
GITHUB_EVIDENCE_ROOT = "https://github.com/ml4t/backtest/blob/main/validation"
SCENARIO_FRAMEWORKS = load_framework_manifest().scenario_framework_ids


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Evidence must be a JSON object: {path}")
    return payload


def _validate_evidence(
    correctness: dict[str, Any],
    large_scale: dict[str, Any],
    real_strategy: dict[str, Any],
    real_performance: dict[str, Any] | None = None,
) -> None:
    failures = correctness_report_failures(correctness)
    if failures:
        raise ValueError("Accepted correctness evidence is invalid: " + "; ".join(failures))
    frameworks = correctness.get("frameworks")
    records = correctness.get("records")
    if not isinstance(frameworks, dict) or not isinstance(records, list):
        raise ValueError("Correctness evidence lacks framework pins or records")
    expected = set(SCENARIO_FRAMEWORKS)
    if set(frameworks) != expected:
        raise ValueError("Correctness evidence does not cover the required framework matrix")
    for framework in expected:
        framework_records = [record for record in records if record.get("framework") == framework]
        if len(framework_records) != len(SCENARIOS):
            raise ValueError(f"Expected {len(SCENARIOS)} retained records for {framework}")
    scale_failures = large_scale_report_failures(large_scale)
    if scale_failures:
        raise ValueError("Accepted large-scale evidence is invalid: " + "; ".join(scale_failures))
    real_failures = real_strategy_report_failures(real_strategy)
    if real_failures:
        raise ValueError("Accepted real-strategy evidence is invalid: " + "; ".join(real_failures))
    performance = (
        _load_json(REAL_STRATEGY_PERFORMANCE) if real_performance is None else real_performance
    )
    performance_failures = real_strategy_performance_failures(performance, real_strategy)
    if performance_failures:
        raise ValueError(
            "Accepted real-strategy performance evidence is invalid: "
            + "; ".join(performance_failures)
        )


def _comparison_checks(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    comparison = record.get("comparison")
    if not isinstance(comparison, dict) or not isinstance(comparison.get("checks"), list):
        raise ValueError(f"Large-scale comparison checks are missing: {record.get('framework')}")
    return {
        str(check["name"]): check
        for check in comparison["checks"]
        if isinstance(check, dict) and "name" in check
    }


def render_claims(
    correctness: dict[str, Any],
    large_scale: dict[str, Any],
    real_strategy: dict[str, Any],
    real_performance: dict[str, Any] | None = None,
) -> str:
    """Render the canonical documentation block."""
    if real_performance is None:
        real_performance = _load_json(REAL_STRATEGY_PERFORMANCE)
    _validate_evidence(correctness, large_scale, real_strategy, real_performance)
    frameworks = correctness["frameworks"]
    records = correctness["records"]
    correctness_url = f"{GITHUB_EVIDENCE_ROOT}/CORRECTNESS_RESULTS.json"
    large_scale_url = f"{GITHUB_EVIDENCE_ROOT}/LARGE_SCALE_RESULTS.json"
    real_strategy_url = f"{GITHUB_EVIDENCE_ROOT}/REAL_STRATEGY_RESULTS.json"
    real_performance_url = f"{GITHUB_EVIDENCE_ROOT}/REAL_STRATEGY_PERFORMANCE.json"

    rows = []
    for framework in SCENARIO_FRAMEWORKS:
        pin = frameworks[framework]
        framework_records = [record for record in records if record["framework"] == framework]
        required = [record for record in framework_records if record["required"]]
        passed = [record for record in required if record["status"] == "pass"]
        failures = [record for record in required if record["status"] != "pass"]
        if failures:
            failed_ids = ", ".join(record["scenario_id"] for record in failures)
            result = f"{len(passed)}/{len(required)} pass; blocked by scenario {failed_ids}"
        else:
            result = f"{len(passed)}/{len(required)} exact"
        pinned_framework = f"[{pin['display_name']} {pin['version']}]({pin['source']})"
        rows.append(
            f"| `{pin['profile']}` | {pinned_framework} | {result} | "
            f"[scenario evidence]({correctness_url}) |"
        )

    scale_rows = []
    for record in large_scale["frameworks"]:
        target = record["target"]
        checks = _comparison_checks(record)
        intents = checks["order_intents"]["expected_count"]
        fills = checks["fills"]["expected_count"]
        trades = checks["trades"]["expected_count"]
        terminal_value = checks["final_value"]["canonical_expected"]
        pinned_framework = f"[{target['display_name']} {target['version']}]({target['source']})"
        scale_rows.append(
            f"| `{target['profile']}` | {pinned_framework} | {intents:,} | {fills:,} | "
            f"{trades:,} | {terminal_value:,.6f} | "
            f"[scale evidence]({large_scale_url}) |"
        )

    workload = large_scale["workload"]
    recipe = workload["recipe"]
    real_rows = []
    required_real = [
        record for record in real_strategy["records"] if record["status"] != "unsupported"
    ]
    passed_real = sum(record["status"] == "pass" for record in required_real)
    real_targets = real_strategy["provenance"]["frameworks"]
    case_labels = {
        "etfs": "ETF allocation",
        "cme_futures": "CME futures",
        "crypto_perps_funding": "Crypto perpetual funding",
        "fx_pairs": "FX allocation (USD-quoted pairs)",
    }
    for record in required_real:
        target = real_targets[record["framework"]]
        pinned_framework = f"[{target['display_name']} {target['version']}]({target['source']})"
        fills = record["surfaces"]["fills"]
        equity = record["surfaces"]["equity"]
        terminal = record["surfaces"]["terminal"]
        if record["status"] == "pass":
            equity_raw = Decimal(equity["max_raw_difference"])
            terminal_raw = Decimal(terminal["max_raw_difference"])
            if equity_raw == 0 and terminal_raw == 0:
                result = (
                    f"fills exact at 1e-8; "
                    f"{equity['coverage']['shared_timestamps']:,} valuations and terminal "
                    "exact at 1e-8"
                )
            else:
                result = (
                    f"fills exact at 1e-8; "
                    f"{equity['coverage']['shared_timestamps']:,} valuations within $0.01 "
                    f"(max raw gap ${equity['max_raw_difference']}); terminal within $0.01 "
                    f"(raw gap ${terminal['max_raw_difference']})"
                )
        else:
            details = ["fills exact" if fills["passed"] else "fills differ"]
            coverage = equity["coverage"]
            if not equity["coverage_passed"]:
                details.append(
                    "valuation timestamps differ "
                    f"({coverage['framework_only_timestamps']} framework-only, "
                    f"{coverage['ml4t_only_timestamps']} ML4T-only)"
                )
            if equity["max_canonical_difference"] != "0.00000000":
                details.append(f"equity gap {equity['max_canonical_difference']}")
            if terminal["max_canonical_difference"] != "0.00000000":
                details.append(f"terminal gap {terminal['max_canonical_difference']}")
            result = "; ".join(details)
        real_rows.append(
            f"| {case_labels[record['case_study']]} | {pinned_framework} | {result} | "
            f"[real-strategy evidence]({real_strategy_url}) |"
        )

    return "\n".join(
        [
            START_MARKER,
            "<!-- Generated by validation/generate_parity_claims.py. Do not edit by hand. -->",
            "",
            "### Real-strategy audit",
            "",
            f"{passed_real}/{len(required_real)} required pairs pass. The audit uses four "
            "real-data strategy workloads with frozen historical market data and model-derived "
            "targets. A pass requires identical valuation timestamp coverage, complete fill "
            "streams equal at 1e-8, and account monetary values that round to the same cent. "
            "The FX workload uses the USD-quoted pairs in its frozen target stream so every "
            "required engine uses native USD valuation.",
            "",
            "The parity protocol disables transaction costs and position rules on both sides. It "
            "tests target sizing, order sequencing, fills, cash and margin behavior, funding where "
            "applicable, and valuation. It does not claim to reproduce each selected case-study "
            "production result with its original costs and risk overlays.",
            "",
            "| Real strategy | Pinned framework | Current result | Evidence |",
            "|---|---|---|---|",
            *real_rows,
            "",
            f"Engine-only timing samples for all {len(real_performance['records'])} passing pairs "
            "are retained in "
            f"[real-strategy performance evidence]({real_performance_url}). These measurements "
            "support only the named strategy, framework version, input bundle, and machine.",
            "",
            "### Synthetic diagnostic scenarios",
            "",
            'The scenario matrix contains synthetic conformance tests. "Exact" means terminal values, '
            "ordered closed trades, and ordered fills match after 1e-8 quantization. Each record "
            "declares whether a surface is native, reconstructed, aggregate-only, input-only, or "
            "unavailable. These results test isolated conventions, not realistic strategy "
            "equivalence.",
            "",
            "| Profile | Pinned framework | Required scenarios | Evidence |",
            "|---|---|---:|---|",
            *rows,
            "",
            f"The synthetic stress workload contains {recipe['assets']:,} assets and "
            f"{recipe['bars']:,} daily sessions ({workload['data_points']:,} bars). Every row "
            "has zero canonical gap for target intents, native fills, closed trades reconstructed "
            "from those fills, and terminal state reconstructed from the fill ledger and final "
            "marks.",
            "",
            "| Profile | Current framework | Target intents | Native fills | Fill-derived closed trades | Terminal value | Evidence |",
            "|---|---|---:|---:|---:|---:|---|",
            *scale_rows,
            END_MARKER,
        ]
    )


def replace_claims(document: str, claims: str, *, path: Path) -> str:
    """Replace one generated claims block in a document."""
    if document.count(START_MARKER) != 1 or document.count(END_MARKER) != 1:
        raise ValueError(f"Expected exactly one parity claims block in {path}")
    prefix, remainder = document.split(START_MARKER, 1)
    _, suffix = remainder.split(END_MARKER, 1)
    return f"{prefix}{claims}{suffix}"


def synchronize(*, check: bool) -> list[Path]:
    """Write generated blocks or return documents that differ from evidence."""
    claims = render_claims(
        _load_json(CORRECTNESS_EVIDENCE),
        _load_json(LARGE_SCALE_EVIDENCE),
        _load_json(REAL_STRATEGY_EVIDENCE),
        _load_json(REAL_STRATEGY_PERFORMANCE),
    )
    changed: list[Path] = []
    for path in TARGETS:
        current = path.read_text(encoding="utf-8")
        expected = replace_claims(current, claims, path=path)
        if current == expected:
            continue
        changed.append(path)
        if not check:
            path.write_text(expected, encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if generated claims are stale")
    args = parser.parse_args()
    changed = synchronize(check=args.check)
    if args.check and changed:
        print("Parity claims differ from retained evidence:")
        for path in changed:
            print(f"- {path.relative_to(PROJECT_ROOT)}")
        return 1
    if not args.check:
        for path in changed:
            print(f"Updated {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
